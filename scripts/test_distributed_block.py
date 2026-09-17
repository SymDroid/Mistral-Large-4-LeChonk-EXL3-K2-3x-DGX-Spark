#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import sys
import math
import argparse
import socket
import struct
import time
import torch

ROOT = Path("/workspace/mistral-large-3-flash")
EXL3 = ROOT / "refs/exllamav3"
RUNTIME_DIR = ROOT / "quant/mistral-large-3-flash-k2"

sys.path.insert(0, str(EXL3))

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import (
    RMSNorm,
    TransformerBlock,
    MLAttention,
    GatedMLP,
    BlockSparseMLP,
)
from exllamav3.util.rope import RopeStyle


parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--role", choices=["local", "send", "recv"], default="local")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--peer", default=None)
parser.add_argument("--port", type=int, default=35136)
args = parser.parse_args()

LAYER = args.layer

HIDDEN = 7168
NUM_Q_HEADS = 128

Q_LORA_RANK = 1536
KV_LORA_RANK = 512

QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128

EXPERT_HIDDEN = 4096
NUM_EXPERTS = 128
TOP_K = 4
NUM_SHARED_EXPERTS = 1

RMS_EPS = 1e-6
SM_SCALE = QK_HEAD_DIM ** -0.5

BASE = f"language_model.model.layers.{LAYER}"


class MinimalConfig:
    pass


def make_config():
    cfg = MinimalConfig()

    print("Opening SafetensorsCollection:")
    print(" ", RUNTIME_DIR)

    cfg.stc = SafetensorsCollection(str(RUNTIME_DIR))

    cfg.infer_params = SimpleNamespace(
        no_reconstruct=False,
    )

    cfg.arch = SimpleNamespace()

    # Architecture values used directly or indirectly by modules.
    cfg.hidden_size = HIDDEN
    cfg.num_q_heads = NUM_Q_HEADS

    cfg.q_lora_rank = Q_LORA_RANK
    cfg.kv_lora_rank = KV_LORA_RANK

    cfg.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    cfg.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    cfg.qk_head_dim = QK_HEAD_DIM
    cfg.v_head_dim = V_HEAD_DIM

    cfg.rms_norm_eps = RMS_EPS

    cfg.moe_intermediate_size = EXPERT_HIDDEN
    cfg.num_experts = NUM_EXPERTS
    cfg.num_experts_per_tok = TOP_K
    cfg.num_shared_experts = NUM_SHARED_EXPERTS

    cfg.sm_scale = SM_SCALE

    return cfg


def make_rope_settings():
    """
    Minimal RopeSettings-compatible object for Mistral Large 3.

    Mistral4 upstream:
      RopeStyle.GPTJ
      rope_theta = 10000
      original_max_position_embeddings = 8192
      yarn factor = 36
      llama_4_scaling beta = 0.1

    The MLA code consumes the settings object rather than the raw
    params.json dictionary.
    """

    # Import the real class if available. We deliberately inspect its
    # constructor here rather than reproducing RoPE math ourselves.
    from exllamav3.util import rope as rope_module
    import inspect

    RopeSettings = getattr(rope_module, "RopeSettings", None)

    if RopeSettings is None:
        raise RuntimeError(
            "Could not find exllamav3.util.rope.RopeSettings"
        )

    sig = inspect.signature(RopeSettings)

    print()
    print("RopeSettings constructor:")
    print(" ", sig)

    # Config.read_rope_settings_default normally creates this object.
    # Try the field names used by current ExLlamaV3.
    candidates = {
        "style": RopeStyle.GPTJ,
        "rope_style": RopeStyle.GPTJ,
        "base": 10000.0,
        "rope_theta": 10000.0,
        "head_dim": QK_ROPE_HEAD_DIM,
        "dim": QK_ROPE_HEAD_DIM,
        "max_position_embeddings": 294912,
        "original_max_position_embeddings": 8192,
        "factor": 36.0,
        "scaling_factor": 36.0,
        "llama_4_scaling_beta": 0.1,
        "beta": 0.1,
    }

    kwargs = {}

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif p.default is inspect.Parameter.empty:
            raise RuntimeError(
                "RopeSettings has required constructor argument "
                f"{name!r} which this test does not know how to populate."
            )

    print("RopeSettings kwargs:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    return RopeSettings(**kwargs)


def tensor_stats(label, x):
    xf = x.float()

    finite = torch.isfinite(xf).all().item()

    print(f"{label}:")
    print("  shape :", tuple(x.shape))
    print("  dtype :", x.dtype)
    print("  device:", x.device)
    print("  min   :", xf.min().item())
    print("  max   :", xf.max().item())
    print("  mean  :", xf.mean().item())
    print("  std   :", xf.std().item())
    print("  finite:", finite)

    if not finite:
        raise RuntimeError(f"{label} contains NaN or Inf")

    return x


def main():

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print(f"Layer {LAYER} distributed TransformerBlock CUDA runtime test")
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")

    cfg = make_config()
    rope_settings = make_rope_settings()

    print()
    print("Architecture:")
    print("  hidden_size      :", HIDDEN)
    print("  q_heads          :", NUM_Q_HEADS)
    print("  q_lora_rank      :", Q_LORA_RANK)
    print("  kv_lora_rank     :", KV_LORA_RANK)
    print("  qk_nope_head_dim :", QK_NOPE_HEAD_DIM)
    print("  qk_rope_head_dim :", QK_ROPE_HEAD_DIM)
    print("  qk_head_dim      :", QK_HEAD_DIM)
    print("  v_head_dim       :", V_HEAD_DIM)
    print("  sm_scale         :", SM_SCALE)
    print("  experts          :", NUM_EXPERTS)
    print("  experts/token    :", TOP_K)
    print("  interm_div       :", 128.0)

    # ------------------------------------------------------------
    # Norms
    # ------------------------------------------------------------

    print()
    print("Constructing RMSNorms...")

    attn_norm = RMSNorm(
        config=cfg,
        key=f"{BASE}.input_layernorm",
        rms_norm_eps=RMS_EPS,
    )

    mlp_norm = RMSNorm(
        config=cfg,
        key=f"{BASE}.post_attention_layernorm",
        rms_norm_eps=RMS_EPS,
    )

    # ------------------------------------------------------------
    # MLA
    # ------------------------------------------------------------

    print("Constructing MLAttention...")

    attn = MLAttention(
        config=cfg,
        key=f"{BASE}.self_attn",
        layer_idx=LAYER,
        hidden_size=HIDDEN,
        num_q_heads=NUM_Q_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        rope_settings=rope_settings,
        q_lora_rank=Q_LORA_RANK,
        sm_scale=SM_SCALE,
        rms_norm_eps=RMS_EPS,
        qmap="block.attn",
        out_dtype=torch.float,
        select_hq_bits=2,
    )

    # ------------------------------------------------------------
    # Shared expert
    # ------------------------------------------------------------

    print("Constructing shared expert...")

    shared = GatedMLP(
        config=cfg,
        key=f"{BASE}.mlp.shared_experts",
        hidden_size=HIDDEN,
        intermediate_size=EXPERT_HIDDEN,
        key_up="up_proj",
        key_gate="gate_proj",
        key_down="down_proj",
        qmap="block.mlp",
        interm_dtype=torch.half,
        out_dtype=torch.float,
        select_hq_bits=2,
        activation_fn="silu",
    )

    # ------------------------------------------------------------
    # Routed MoE
    # ------------------------------------------------------------

    print("Constructing BlockSparseMLP...")

    moe = BlockSparseMLP(
        config=cfg,
        key=f"{BASE}.mlp",
        hidden_size=HIDDEN,
        intermediate_size=EXPERT_HIDDEN,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,

        key_up="experts.{expert_idx}.up_proj",
        key_gate="experts.{expert_idx}.gate_proj",
        key_down="experts.{expert_idx}.down_proj",

        key_gate_up_split="experts.gate_up_proj",
        key_down_split="experts.down_proj",

        transpose_fused_weights=False,
        ftranspose_after_load=True,
        frange_dim=0,
        gate_up_interleaved=False,

        key_routing_gate="gate",
        key_e_score_bias=None,
        router_type="std",

        # Exact upstream Mistral4 value.
        interm_div=128.0,

        qmap="block.mlp",
        interm_dtype=torch.half,
        out_dtype=torch.float,

        shared_experts=shared,
        activation_fn="silu",
    )

    # ------------------------------------------------------------
    # TransformerBlock
    # ------------------------------------------------------------

    print("Constructing TransformerBlock...")

    block = TransformerBlock(
        config=cfg,
        key=BASE,
        layer_idx=LAYER,
        attn_norm=attn_norm,
        attn=attn,
        mlp_norm=mlp_norm,
        mlp=moe,
    )

    # ------------------------------------------------------------
    # Load complete block
    # ------------------------------------------------------------

    print()
    print(f"Loading complete layer {LAYER} onto CUDA...")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated(device)

    block.load(device)

    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated(device)

    print()
    print("Load complete.")
    print(
        "CUDA memory delta:",
        f"{(after - before) / 1024**3:.3f} GiB",
    )

    print()
    print("MoE state:")
    print("  is_quantized       :", moe.is_quantized)
    print("  support_quant_paths:", moe.support_quant_paths)
    print("  support_fused      :", moe.support_fused)

    if not moe.is_quantized:
        raise RuntimeError("MoE was not recognized as quantized")

    if any(
        str(m.quant_type) != "exl3"
        for m in moe.gates + moe.ups + moe.downs
    ):
        raise RuntimeError(
            "At least one routed projection is not EXL3"
        )

    print("  EXL3 projections   : 384/384 PASS")

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    torch.manual_seed(42)

    # Use more than one token. Attention over one token is a rather
    # boring attention test and does not exercise causal interaction.
    SEQLEN = 8

    x = torch.randn(
        (1, SEQLEN, HIDDEN),
        device=device,
        dtype=torch.half,
    )

    params = {
        "attn_mode": "flash_attn_nc",
        "cache": None,
    }

    print()
    tensor_stats("Input", x)

    # ------------------------------------------------------------
    # 1. Attention RMSNorm
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("1. RMSNorm")
    print("-" * 78)

    with torch.inference_mode():
        xn = attn_norm.forward(
            x,
            params,
            out_dtype=torch.half,
        )

    tensor_stats("Attention RMSNorm output", xn)

    print("RMSNorm: PASS")

    # ------------------------------------------------------------
    # 2. Q path
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("2. MLA query projection")
    print("-" * 78)

    with torch.inference_mode():
        q, q_resid = attn.project_q(
            xn,
            params,
            return_resid=True,
        )

    if q_resid is not None:
        tensor_stats("q_resid", q_resid)

    tensor_stats("Q projection", q)

    expected_q = NUM_Q_HEADS * QK_HEAD_DIM

    if q.shape[-1] != expected_q:
        raise RuntimeError(
            f"Unexpected Q width {q.shape[-1]}, "
            f"expected {expected_q}"
        )

    print("MLA q-path: PASS")

    # ------------------------------------------------------------
    # 3. KV path
    #
    # Do not guess project_kv()'s public signature here. The complete
    # MLA forward below exercises kv_a, kv_a_norm and kv_b through the
    # real upstream path.
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("3. MLA cache-less attention")
    print("-" * 78)

    torch.cuda.synchronize()

    with torch.inference_mode():
        attn_y = attn.forward(
            xn,
            params,
            out_dtype=torch.half,
        )

    torch.cuda.synchronize()

    tensor_stats("MLAttention output", attn_y)

    if tuple(attn_y.shape) != (1, SEQLEN, HIDDEN):
        raise RuntimeError(
            f"Unexpected attention output shape: {tuple(attn_y.shape)}"
        )

    print("MLAttention flash_attn_nc: PASS")

    # ------------------------------------------------------------
    # 4. Post-attention norm + MoE
    #
    # TransformerBlock can fuse the attention residual into this norm.
    # For this diagnostic pass we make the residual explicit.
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("4. Post-attention RMSNorm + K2 MoE")
    print("-" * 78)

    resid = x + attn_y

    tensor_stats("Attention residual", resid)

    with torch.inference_mode():
        mn = mlp_norm.forward(
            resid,
            params,
            out_dtype=torch.half,
        )

    tensor_stats("Post-attention RMSNorm", mn)

    with torch.inference_mode():
        moe_y = moe.forward(
            mn,
            params,
            out_dtype=torch.half,
        )

    tensor_stats("K2 MoE output", moe_y)

    if tuple(moe_y.shape) != (1, SEQLEN, HIDDEN):
        raise RuntimeError(
            f"Unexpected MoE output shape: {tuple(moe_y.shape)}"
        )

    print("K2 BlockSparseMLP interm_div=128: PASS")

    # ------------------------------------------------------------
    # 5. Complete TransformerBlock
    #
    # Fresh input because TransformerBlock performs in-place residual
    # operations.
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("5. Complete TransformerBlock")
    print("-" * 78)

    torch.manual_seed(42)

    xb = torch.randn(
        (1, SEQLEN, HIDDEN),
        device=device,
        dtype=torch.half,
    )

    block_params = {
        "attn_mode": "flash_attn_nc",
        "cache": None,
    }

    torch.cuda.synchronize()

    with torch.inference_mode():
        y = block.forward(
            xb,
            block_params,
            out_dtype=torch.half,
        )

    torch.cuda.synchronize()

    tensor_stats("TransformerBlock output", y)

    if tuple(y.shape) != (1, SEQLEN, HIDDEN):
        raise RuntimeError(
            f"Unexpected TransformerBlock output shape: "
            f"{tuple(y.shape)}"
        )

    # ------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    print()
    print("CUDA memory:")
    print("  allocated:", f"{allocated / 1024**3:.3f} GiB")
    print("  reserved :", f"{reserved / 1024**3:.3f} GiB")
    print("  peak     :", f"{peak / 1024**3:.3f} GiB")

    print()
    print("=" * 78)
    print("RMSNorm:                  PASS")
    print("MLA q-path:               PASS")
    print("MLAttention CUDA:         PASS")
    print("K2 MoE interm_div=128:    PASS")
    print("TransformerBlock CUDA:    PASS")
    print("=" * 78)
    print(
        "MISTRAL LARGE 3 FLASH K2 "
        f"LAYER {LAYER} TRANSFORMER BLOCK CUDA FORWARD: PASS"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

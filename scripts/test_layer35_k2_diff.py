#!/usr/bin/env python3

from pathlib import Path
import sys
import torch
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_layer35_moe_runtime_div128 as base

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP


SOURCE = (
    ROOT
    / "upstream-reference"
    / "consolidated-00122-of-00272.safetensors"
)

W_KEY = "layers.35.experts.0.w1.weight"
S_KEY = "layers.35.experts.0.w1.weight_scale"


def stats(name, x):
    x = x.float()

    print(
        f"{name:14s}"
        f" shape={tuple(x.shape)}"
        f" mean={x.mean().item():.6e}"
        f" std={x.std().item():.6e}"
        f" rms={x.square().mean().sqrt().item():.6e}"
        f" absmax={x.abs().max().item():.6e}"
        f" finite={torch.isfinite(x).float().mean().item()*100:.2f}%"
    )


def main():

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 Expert 0 gate_proj K2 differential")
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")

    # ----------------------------------------------------------
    # Original Mistral FP8 expert
    # ----------------------------------------------------------

    print("\nReading ORIGINAL FP8 expert...")

    with safe_open(
        SOURCE,
        framework="pt",
        device="cpu",
    ) as f:

        w_fp8 = f.get_tensor(W_KEY)
        scales = f.get_tensor(S_KEY)

    print("weight:", tuple(w_fp8.shape), w_fp8.dtype)
    print("scale :", tuple(scales.shape), scales.dtype)

    scale_full = (
        scales.float()
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
    )

    assert scale_full.shape == w_fp8.shape

    w_ref = w_fp8.float() * scale_full

    stats("Original W", w_ref)

    # ----------------------------------------------------------
    # Build EXACT runtime MoE
    # ----------------------------------------------------------

    print("\nConstructing runtime MoE...")

    cfg = base.make_config()

    shared = GatedMLP(
        config=cfg,
        key=f"{base.BASE}.mlp.shared_experts",
        hidden_size=base.HIDDEN,
        intermediate_size=base.EXPERT_HIDDEN,
        key_gate="gate_proj",
        key_up="up_proj",
        key_down="down_proj",
        activation_fn="silu",
        out_dtype=torch.float,
    )

    moe = BlockSparseMLP(
        config=cfg,
        key=f"{base.BASE}.mlp",
        hidden_size=base.HIDDEN,
        intermediate_size=base.EXPERT_HIDDEN,
        num_experts=base.NUM_EXPERTS,
        num_experts_per_tok=base.TOP_K,

        key_gate="experts.{expert_idx}.gate_proj",
        key_up="experts.{expert_idx}.up_proj",
        key_down="experts.{expert_idx}.down_proj",

        key_routing_gate="gate",

        shared_experts=shared,

        activation_fn="silu",

        interm_dtype=torch.half,
        interm_div=128.0,

        router_type="std",
        out_dtype=torch.float,
    )

    print("Loading layer 35 MoE...")

    torch.cuda.empty_cache()

    moe.load(device)

    torch.cuda.synchronize()

    gate = moe.gates[0]

    print("\nExpert runtime:")
    print(" projection :", type(gate))
    print(" quant_type :", getattr(gate, "quant_type", None))
    print(
        "weight_scale:",
        getattr(gate, "weight_scale", None),
    )

    # ----------------------------------------------------------
    # Deterministic test activation
    # ----------------------------------------------------------

    torch.manual_seed(42)

    x = torch.randn(
        (8, base.HIDDEN),
        dtype=torch.float16,
        device=device,
    )

    stats("Input", x)

    # Original FP8 reference
    print("\nRunning ORIGINAL FP8 reference...")

    y_ref = (
        x.float()
        @ w_ref.to(device).T
    )

    torch.cuda.synchronize()

    # EXL3 K2
    print("Running EXL3 K2 runtime...")

    y_k2 = gate.forward(x, {})

    torch.cuda.synchronize()

    y_ref = y_ref.float()
    y_k2 = y_k2.float()

    stats("Original out", y_ref)
    stats("K2 out", y_k2)

    if y_ref.shape != y_k2.shape:
        raise RuntimeError(
            f"Shape mismatch {y_ref.shape} vs {y_k2.shape}"
        )

    # ----------------------------------------------------------
    # Differential
    # ----------------------------------------------------------

    diff = y_k2 - y_ref

    rmse = diff.square().mean().sqrt()

    ref_rms = y_ref.square().mean().sqrt()

    rel_l2 = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(y_ref)
    )

    a = y_ref.flatten()
    b = y_k2.flatten()

    cosine = torch.dot(a, b) / (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    magnitude_ratio = (
        torch.linalg.vector_norm(b)
        / torch.linalg.vector_norm(a)
    )

    print("\n" + "=" * 78)
    print("K2 DIFFERENTIAL")
    print("=" * 78)

    print("RMSE           :", rmse.item())
    print("Reference RMS  :", ref_rms.item())
    print("Relative L2    :", rel_l2.item())
    print("Cosine         :", cosine.item())
    print("Magnitude ratio:", magnitude_ratio.item())
    print("Max abs diff   :", diff.abs().max().item())

    print()

    # Expected offline ballpark from bootstrap:
    #
    # cosine ~0.965
    # relL2  ~0.262
    #
    # Activation-space numbers will not be identical, but should
    # clearly remain correlated and have sane magnitude.

    if (
        torch.isfinite(y_k2).all()
        and cosine.item() > 0.90
        and 0.70 < magnitude_ratio.item() < 1.30
        and rel_l2.item() < 0.50
    ):
        print("K2 gate_proj runtime differential: PASS")
    else:
        print("K2 gate_proj runtime differential: FAIL")
        print(
            ">>> K2 pack/load/reconstruction is a prime suspect."
        )


if __name__ == "__main__":
    main()

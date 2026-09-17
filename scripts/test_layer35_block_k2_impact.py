#!/usr/bin/env python3

from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
REFDIR = ROOT / "upstream-reference"
INDEX = ROOT / "upstream-meta/consolidated.safetensors.index.json"

sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the exact already-working block construction
import test_layer35_block_runtime as blockbase


def load_tensor(key, weight_map):
    shard = weight_map[key]
    path = REFDIR / shard

    if not path.exists():
        raise FileNotFoundError(
            f"Missing reference shard for {key}:\n{path}"
        )

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def dequant_fp8(weight, scale, device):
    weight = weight.to(device)
    scale = scale.to(device)

    sf = (
        scale.float()
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
    )

    sf = sf[:weight.shape[0], :weight.shape[1]]

    return weight.float() * sf


def metrics(ref, test):
    a = ref.float().reshape(-1)
    b = test.float().reshape(-1)

    d = b - a

    nr = torch.linalg.vector_norm(a)
    nt = torch.linalg.vector_norm(b)

    return {
        "ref_rms": torch.sqrt(torch.mean(a * a)).item(),
        "test_rms": torch.sqrt(torch.mean(b * b)).item(),
        "rmse": torch.sqrt(torch.mean(d * d)).item(),
        "rel_l2": (torch.linalg.vector_norm(d) / nr).item(),
        "cosine": F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1,
        ).item(),
        "magnitude": (nt / nr).item(),
        "max_abs": d.abs().max().item(),
    }


def show(name, ref, test):
    m = metrics(ref, test)

    print("\n" + name)
    print("-" * 78)
    print(f"Reference RMS   : {m['ref_rms']:.9f}")
    print(f"K2 RMS          : {m['test_rms']:.9f}")
    print(f"RMSE            : {m['rmse']:.9f}")
    print(f"Relative L2     : {m['rel_l2']:.9f}")
    print(f"Cosine          : {m['cosine']:.9f}")
    print(f"Magnitude ratio : {m['magnitude']:.9f}")
    print(f"Max abs diff    : {m['max_abs']:.9f}")

    return m


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")

    with open(INDEX) as f:
        weight_map = json.load(f)["weight_map"]

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 full-block K2 impact differential")
    print("=" * 78)

    # ------------------------------------------------------------
    # Exact known-good runtime construction
    # ------------------------------------------------------------

    cfg = blockbase.make_config()
    rope_settings = blockbase.make_rope_settings()

    attn_norm = blockbase.RMSNorm(
        config=cfg,
        key=f"{blockbase.BASE}.input_layernorm",
        rms_norm_eps=blockbase.RMS_EPS,
    )

    mlp_norm = blockbase.RMSNorm(
        config=cfg,
        key=f"{blockbase.BASE}.post_attention_layernorm",
        rms_norm_eps=blockbase.RMS_EPS,
    )

    attn = blockbase.MLAttention(
        config=cfg,
        key=f"{blockbase.BASE}.self_attn",
        layer_idx=blockbase.LAYER,
        hidden_size=blockbase.HIDDEN,
        num_q_heads=blockbase.NUM_Q_HEADS,
        kv_lora_rank=blockbase.KV_LORA_RANK,
        qk_nope_head_dim=blockbase.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=blockbase.QK_ROPE_HEAD_DIM,
        v_head_dim=blockbase.V_HEAD_DIM,
        rope_settings=rope_settings,
        q_lora_rank=blockbase.Q_LORA_RANK,
        sm_scale=blockbase.SM_SCALE,
        rms_norm_eps=blockbase.RMS_EPS,
        qmap="block.attn",
        out_dtype=torch.float,
        select_hq_bits=2,
    )

    shared = blockbase.GatedMLP(
        config=cfg,
        key=f"{blockbase.BASE}.mlp.shared_experts",
        hidden_size=blockbase.HIDDEN,
        intermediate_size=blockbase.EXPERT_HIDDEN,
        key_up="up_proj",
        key_gate="gate_proj",
        key_down="down_proj",
        qmap="block.mlp",
        interm_dtype=torch.half,
        out_dtype=torch.float,
        select_hq_bits=2,
        activation_fn="silu",
    )

    moe = blockbase.BlockSparseMLP(
        config=cfg,
        key=f"{blockbase.BASE}.mlp",
        hidden_size=blockbase.HIDDEN,
        intermediate_size=blockbase.EXPERT_HIDDEN,
        num_experts=blockbase.NUM_EXPERTS,
        num_experts_per_tok=blockbase.TOP_K,

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

        interm_div=128.0,

        qmap="block.mlp",
        interm_dtype=torch.half,
        out_dtype=torch.float,

        shared_experts=shared,
        activation_fn="silu",
    )

    print("\nLoading layer 35...")
    torch.cuda.empty_cache()

    attn_norm.load(device)
    attn.load(device)
    mlp_norm.load(device)
    moe.load(device)

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Same input as existing block runtime test
    # ------------------------------------------------------------

    SEQLEN = 8

    torch.manual_seed(42)

    x = torch.randn(
        (1, SEQLEN, blockbase.HIDDEN),
        device=device,
        dtype=torch.half,
    )

    params = {
        "attn_mode": "flash_attn_nc",
        "cache": None,
    }

    # ------------------------------------------------------------
    # Attention half of block
    # ------------------------------------------------------------

    print("\nRunning common attention path...")

    with torch.inference_mode():

        xn = attn_norm.forward(
            x,
            params,
            out_dtype=torch.half,
        )

        attn_y = attn.forward(
            xn,
            params,
            out_dtype=torch.half,
        )

        resid = x + attn_y

        mn = mlp_norm.forward(
            resid,
            params,
            out_dtype=torch.half,
        )

    torch.cuda.synchronize()

    flat = mn.reshape(-1, blockbase.HIDDEN)

    # ------------------------------------------------------------
    # Routing on REAL block input
    # ------------------------------------------------------------

    selected, runtime_weights = moe.routing_fn(
        flat.shape[0],
        moe.routing_cfg,
        flat,
        {},
    )

    torch.cuda.synchronize()

    selected_cpu = selected.cpu()
    runtime_weights_f = runtime_weights.float()

    normalized_weights = (
        runtime_weights_f / float(moe.interm_div)
    )

    print("\nRouting on post-attention activations:")

    for t in range(SEQLEN):
        print(
            f"token {t}: "
            f"experts={selected_cpu[t].tolist()}  "
            f"sum={normalized_weights[t].sum().item():.6f}"
        )

    unique_experts = sorted(
        set(selected_cpu.flatten().tolist())
    )

    print("\nUnique routed experts:")
    print(unique_experts)
    print("count:", len(unique_experts))

    # Verify all required original shards exist BEFORE expensive work.

    missing = set()

    for e in unique_experts:
        for p in ("w1", "w2", "w3"):
            for suffix in ("weight", "weight_scale"):
                key = (
                    f"layers.{blockbase.LAYER}."
                    f"experts.{e}.{p}.{suffix}"
                )
                shard = weight_map[key]

                if not (REFDIR / shard).exists():
                    missing.add(shard)

    if missing:
        print("\nMISSING ORIGINAL SHARDS:")
        for s in sorted(missing):
            print(" ", s)
        raise RuntimeError(
            "Reference shards missing. No comparison performed."
        )

    print("\nAll required FP8 reference shards: PASS")

    # ------------------------------------------------------------
    # Actual K2 MoE
    # ------------------------------------------------------------

    print("\nRunning actual K2 BlockSparseMLP...")

    with torch.inference_mode():
        actual_k2_moe = moe.forward(
            mn,
            params,
            out_dtype=torch.half,
        )

    actual_k2_moe_f = actual_k2_moe.float().reshape(
        SEQLEN,
        blockbase.HIDDEN,
    )

    # ------------------------------------------------------------
    # Explicit K2 routed component
    #
    # This lets us subtract routed contribution from actual moe.forward
    # and recover everything common to both paths, notably shared expert.
    # ------------------------------------------------------------

    print("Computing explicit K2 routed component...")

    k2_routed = torch.zeros(
        (SEQLEN, blockbase.HIDDEN),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():

        for t in range(SEQLEN):

            xt = flat[t:t+1]

            for j in range(blockbase.TOP_K):

                e = int(selected_cpu[t, j])
                rw = runtime_weights_f[t, j]

                g = moe.gates[e].forward(xt, {})
                u = moe.ups[e].forward(xt, {})

                h = (F.silu(g) * u).half()

                out = moe.downs[e].forward(
                    h,
                    {},
                ).float()


                k2_routed[t] += rw * out.squeeze(0)

    torch.cuda.synchronize()

    # Algebraically isolate common contribution.
    common_component = actual_k2_moe_f - k2_routed

    # ------------------------------------------------------------
    # Original FP8 routed component
    #
    # Stream expert-by-expert. Do NOT retain all 30 dequantized experts.
    # ------------------------------------------------------------

    print("Computing original FP8 routed component...")

    fp8_routed = torch.zeros_like(k2_routed)

    for n, e in enumerate(unique_experts, 1):

        print(
            f"  [{n:02d}/{len(unique_experts):02d}] "
            f"expert {e}"
        )

        tensors = {}

        for p in ("w1", "w2", "w3"):

            wk = (
                f"layers.{blockbase.LAYER}."
                f"experts.{e}.{p}.weight"
            )

            sk = (
                f"layers.{blockbase.LAYER}."
                f"experts.{e}.{p}.weight_scale"
            )

            w = load_tensor(wk, weight_map)
            s = load_tensor(sk, weight_map)

            tensors[p] = dequant_fp8(
                w,
                s,
                device,
            )

        # Only process tokens routed to this expert.

        for t in range(SEQLEN):

            matches = (
                selected_cpu[t] == e
            ).nonzero(as_tuple=False)

            if matches.numel() == 0:
                continue

            j = int(matches[0, 0])

            xt = flat[t:t+1].float()

            gate = xt @ tensors["w1"].T
            up = xt @ tensors["w3"].T

            h = F.silu(gate) * up

            out = h @ tensors["w2"].T

            rw = normalized_weights[t, j]

            fp8_routed[t] += (
                rw * out.squeeze(0)
            )

        del tensors
        torch.cuda.empty_cache()

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Reconstruct reference MoE
    # ------------------------------------------------------------

    reference_moe = (
        common_component + fp8_routed
    )

    # Full post-MoE residual.
    #
    # Both branches share the exact same attention residual.
    # ------------------------------------------------------------

    resid_f = resid.float().reshape(
        SEQLEN,
        blockbase.HIDDEN,
    )

    reference_block = resid_f + reference_moe
    k2_block = resid_f + actual_k2_moe_f

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("DIFFERENTIAL RESULTS")
    print("=" * 78)

    routed_m = show(
        "1. ROUTED MoE ONLY",
        fp8_routed,
        k2_routed,
    )

    moe_m = show(
        "2. COMPLETE MoE INCLUDING COMMON/SHARED COMPONENT",
        reference_moe,
        actual_k2_moe_f,
    )

    block_m = show(
        "3. FULL BLOCK OUTPUT AFTER RESIDUAL",
        reference_block,
        k2_block,
    )

    print("\n" + "=" * 78)
    print("PER-TOKEN FULL BLOCK")
    print("=" * 78)

    for t in range(SEQLEN):

        m = metrics(
            reference_block[t:t+1],
            k2_block[t:t+1],
        )

        print(
            f"token {t}: "
            f"cos={m['cosine']:.8f}  "
            f"relL2={m['rel_l2']:.8f}  "
            f"mag={m['magnitude']:.8f}"
        )

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        f"Routed cosine : {routed_m['cosine']:.9f}"
    )
    print(
        f"MoE cosine    : {moe_m['cosine']:.9f}"
    )
    print(
        f"Block cosine  : {block_m['cosine']:.9f}"
    )

    print()

    if block_m["cosine"] >= 0.995:
        print(
            "RESULT: residual/shared path strongly suppresses "
            "the isolated K2 expert error."
        )
    elif block_m["cosine"] >= 0.98:
        print(
            "RESULT: K2 error remains measurable at block output."
        )
    else:
        print(
            "RESULT: severe K2 error survives the complete block."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

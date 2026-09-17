#!/usr/bin/env python3

from pathlib import Path
import sys
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
REFDIR = ROOT / "upstream-reference"
INDEX = ROOT / "upstream-meta/consolidated.safetensors.index.json"

sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_layer35_moe_runtime_div128 as base

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP


def load_tensor(key, weight_map):
    shard = weight_map[key]
    path = REFDIR / shard

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def dequant_fp8(weight, scale, device):
    """
    Native Mistral block-FP8:
      weight[out, in]
      scale[ceil(out/128), ceil(in/128)]
    """
    weight = weight.to(device)
    scale = scale.to(device)

    sf = (
        scale.float()
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
    )

    sf = sf[:weight.shape[0], :weight.shape[1]]

    return weight.float() * sf


def metrics(ref, test):
    ref = ref.float()
    test = test.float()

    diff = test - ref

    ref_rms = torch.sqrt(torch.mean(ref * ref)).item()
    test_rms = torch.sqrt(torch.mean(test * test)).item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()

    rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)

    cos = F.cosine_similarity(
        ref.reshape(1, -1),
        test.reshape(1, -1),
        dim=1,
    ).item()

    mag = (
        torch.linalg.vector_norm(test) /
        torch.linalg.vector_norm(ref)
    ).item()

    return {
        "ref_rms": ref_rms,
        "test_rms": test_rms,
        "rmse": rmse,
        "rel_l2": rel.item(),
        "cosine": cos,
        "magnitude": mag,
        "max_abs": diff.abs().max().item(),
    }


def print_metrics(name, m):
    print(f"\n{name}")
    print("-" * 72)
    print(f"Reference RMS   : {m['ref_rms']:.9f}")
    print(f"K2 RMS          : {m['test_rms']:.9f}")
    print(f"RMSE            : {m['rmse']:.9f}")
    print(f"Relative L2     : {m['rel_l2']:.9f}")
    print(f"Cosine          : {m['cosine']:.9f}")
    print(f"Magnitude ratio : {m['magnitude']:.9f}")
    print(f"Max abs diff    : {m['max_abs']:.9f}")


def main():
    device = torch.device("cuda:0")

    with open(INDEX) as f:
        weight_map = json.load(f)["weight_map"]

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 TOP-4 routed MoE differential")
    print("Original FP8 vs K2")
    print("=" * 78)

    cfg = base.make_config()

    # Construct same runtime module as our previous tests.
    # Shared expert exists because this mirrors the real architecture,
    # but we will NOT use moe.forward() for the routed-only differential.
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

    print("Loading K2 MoE...")
    moe.load(device)
    torch.cuda.synchronize()

    # Same deterministic input as previous probes
    torch.manual_seed(42)
    x = torch.randn(
        (8, base.HIDDEN),
        dtype=torch.float16,
        device=device,
    )

    # Exact runtime routing
    selected, runtime_weights = moe.routing_fn(
        x.shape[0],
        moe.routing_cfg,
        x,
        {},
    )

    torch.cuda.synchronize()

    selected_cpu = selected.cpu()
    runtime_weights_f = runtime_weights.float()

    # interm_div compensation:
    # routing_std gives weights summing to ~128.
    # Mathematical original MoE uses normalized weights summing to ~1.
    ref_weights = runtime_weights_f / float(moe.interm_div)

    print("\nRouting:")
    for t in range(x.shape[0]):
        print(
            f"token {t}: "
            f"experts={selected_cpu[t].tolist()} "
            f"runtime_sum={runtime_weights_f[t].sum().item():.6f} "
            f"normalized_sum={ref_weights[t].sum().item():.6f}"
        )

    unique_experts = sorted(set(selected_cpu.flatten().tolist()))

    print("\nUnique experts:", unique_experts)
    print("Count:", len(unique_experts))

    # ------------------------------------------------------------
    # Load original FP8 expert weights only for routed experts
    # ------------------------------------------------------------

    print("\nLoading original FP8 routed experts...")

    original = {}

    for n, e in enumerate(unique_experts, 1):
        print(f"  [{n:02d}/{len(unique_experts):02d}] expert {e}")

        ed = {}

        for proj in ("w1", "w2", "w3"):
            wk = f"layers.35.experts.{e}.{proj}.weight"
            sk = f"layers.35.experts.{e}.{proj}.weight_scale"

            w = load_tensor(wk, weight_map)
            s = load_tensor(sk, weight_map)

            ed[proj] = dequant_fp8(w, s, device)

        original[e] = ed

    # ------------------------------------------------------------
    # ORIGINAL FP8 routed top-4
    # ------------------------------------------------------------

    print("\nComputing original FP8 top-4 reference...")

    y_ref = torch.zeros(
        (x.shape[0], base.HIDDEN),
        dtype=torch.float32,
        device=device,
    )

    for t in range(x.shape[0]):
        xt = x[t:t+1].float()

        for j in range(base.TOP_K):
            e = int(selected_cpu[t, j])
            rw = ref_weights[t, j]

            w1 = original[e]["w1"]
            w2 = original[e]["w2"]
            w3 = original[e]["w3"]

            gate = xt @ w1.T
            up = xt @ w3.T

            hidden = F.silu(gate) * up

            out = hidden @ w2.T

            y_ref[t] += rw * out.squeeze(0)

    torch.cuda.synchronize()

    # Free original weights before K2 execution
    del original
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # K2 routed top-4
    #
    # Explicitly execute the selected K2 experts. This isolates the
    # routed component from shared expert/residual while preserving
    # the exact runtime routing choices.
    #
    # K2 expert modules themselves operate with interm_div scaling.
    # The runtime routing weights contain the inverse compensation.
    # ------------------------------------------------------------

    print("\nComputing K2 top-4...")

    y_k2 = torch.zeros_like(y_ref)

    for t in range(x.shape[0]):
        xt = x[t:t+1]

        for j in range(base.TOP_K):
            e = int(selected_cpu[t, j])
            rw = runtime_weights_f[t, j]

            gate_mod = moe.gates[e]
            up_mod = moe.ups[e]
            down_mod = moe.downs[e]

            g = gate_mod.forward(xt, {})
            u = up_mod.forward(xt, {})

            h = (F.silu(g) * u).half()

            out = down_mod.forward(h, {}).float()

            # Loose expert Linear modules have weight_scale=1.
            # Apply the expert-side interm_div here explicitly,
            # matching BlockSparseMLP's mathematical scaling.
            out = out / float(moe.interm_div)

            y_k2[t] += rw * out.squeeze(0)

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    all_m = metrics(y_ref, y_k2)
    print_metrics("ALL 8 TOKENS", all_m)

    print("\n" + "=" * 78)
    print("PER-TOKEN")
    print("=" * 78)

    cosines = []
    rels = []
    mags = []

    for t in range(x.shape[0]):
        m = metrics(y_ref[t:t+1], y_k2[t:t+1])

        cosines.append(m["cosine"])
        rels.append(m["rel_l2"])
        mags.append(m["magnitude"])

        print(
            f"token {t}: "
            f"cos={m['cosine']:.6f}  "
            f"relL2={m['rel_l2']:.6f}  "
            f"mag={m['magnitude']:.6f}  "
            f"rmse={m['rmse']:.6f}"
        )

    print("\nSummary:")
    print(f"mean cosine      : {sum(cosines)/len(cosines):.9f}")
    print(f"min cosine       : {min(cosines):.9f}")
    print(f"mean relL2       : {sum(rels)/len(rels):.9f}")
    print(f"max relL2        : {max(rels):.9f}")
    print(f"mean magnitude   : {sum(mags)/len(mags):.9f}")

    print("\n" + "=" * 78)

    if all_m["cosine"] >= 0.95 and all_m["rel_l2"] <= 0.35:
        print("RESULT: ROUTED TOP-4 LOOKS REASONABLE")
    elif all_m["cosine"] >= 0.90:
        print("RESULT: SIGNIFICANT K2 ROUTED-MOE DRIFT")
    else:
        print("RESULT: SEVERE K2 ROUTED-MOE DRIFT")

    print("=" * 78)


if __name__ == "__main__":
    main()

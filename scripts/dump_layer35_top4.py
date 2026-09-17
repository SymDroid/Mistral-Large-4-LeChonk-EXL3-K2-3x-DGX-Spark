#!/usr/bin/env python3

from pathlib import Path
import sys
import json
import torch

ROOT = Path("/workspace/mistral-large-3-flash")
sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_layer35_moe_runtime_div128 as base

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP

INDEX = ROOT / "upstream-meta/consolidated.safetensors.index.json"


def main():
    device = torch.device("cuda:0")

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 routing probe")
    print("=" * 78)

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

    print("Loading MoE...")
    moe.load(device)
    torch.cuda.synchronize()

    print("\nRouting configuration:")
    print("num_experts      :", moe.num_experts)
    print("top_k            :", moe.num_experts_per_tok)
    print("interm_div       :", moe.interm_div)

    pes = moe.routing_cfg.per_expert_scale
    if pes is None:
        print("per_expert_scale : None")
    else:
        print(
            "per_expert_scale :",
            pes.min().item(),
            "...",
            pes.max().item(),
        )

    # Same deterministic activations as full-expert differential
    torch.manual_seed(42)

    x = torch.randn(
        (8, base.HIDDEN),
        dtype=torch.float16,
        device=device,
    )

    selected, weights = moe.routing_fn(
        x.shape[0],
        moe.routing_cfg,
        x,
        {},
    )

    torch.cuda.synchronize()

    selected = selected.cpu()
    weights = weights.float().cpu()

    print("\n" + "=" * 78)
    print("TOP-4 ROUTING")
    print("=" * 78)

    for t in range(x.shape[0]):
        print(
            f"token {t}: "
            f"experts={selected[t].tolist()}  "
            f"weights={[round(v, 6) for v in weights[t].tolist()]}  "
            f"sum={weights[t].sum().item():.6f}"
        )

    unique = sorted(set(selected.flatten().tolist()))

    print("\nUnique routed experts:")
    print(unique)
    print("count:", len(unique))

    # ----------------------------------------------------------
    # Find original source shards required for these experts
    # ----------------------------------------------------------

    idx = json.load(open(INDEX))
    wm = idx["weight_map"]

    print("\n" + "=" * 78)
    print("ORIGINAL SHARDS REQUIRED")
    print("=" * 78)

    shards = set()

    for e in unique:
        print(f"\nExpert {e}:")

        for proj in ("w1", "w2", "w3"):
            for suffix in ("weight", "weight_scale"):
                key = f"layers.35.experts.{e}.{proj}.{suffix}"
                shard = wm.get(key)

                print(
                    f"  {proj}.{suffix:12s} -> {shard}"
                )

                if shard:
                    shards.add(shard)

    print("\nUnique shards:")
    for s in sorted(shards):
        print(" ", s)

    print("\nShard count:", len(shards))

    # Show which are already downloaded
    refdir = ROOT / "upstream-reference"

    print("\nDownload status:")
    missing = []

    for s in sorted(shards):
        exists = (refdir / s).exists()

        print(
            f"  {'HAVE' if exists else 'MISS'}  {s}"
        )

        if not exists:
            missing.append(s)

    print("\nMissing shards:")
    for s in missing:
        print(" ", s)

    if not missing:
        print("  none")

    print("=" * 78)


if __name__ == "__main__":
    main()

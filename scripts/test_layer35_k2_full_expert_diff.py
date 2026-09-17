#!/usr/bin/env python3

from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_layer35_moe_runtime_div128 as base

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP

SOURCE = ROOT / "upstream-reference" / \
    "consolidated-00122-of-00272.safetensors"


def dequant(f, name):
    w = f.get_tensor(name + ".weight")
    s = f.get_tensor(name + ".weight_scale")

    sf = (
        s.float()
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
    )

    assert sf.shape == w.shape

    return w.float() * sf


def stats(name, x):
    x = x.float()

    print(
        f"{name:16s}"
        f" mean={x.mean().item(): .6e}"
        f" std={x.std().item():.6e}"
        f" rms={x.square().mean().sqrt().item():.6e}"
        f" absmax={x.abs().max().item():.6e}"
        f" finite={torch.isfinite(x).float().mean().item()*100:.2f}%"
    )


def compare(name, ref, test):
    ref = ref.float()
    test = test.float()

    diff = test - ref

    rel = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(ref)
    )

    cosine = torch.dot(
        ref.flatten(),
        test.flatten(),
    ) / (
        torch.linalg.vector_norm(ref)
        * torch.linalg.vector_norm(test)
    )

    mag = (
        torch.linalg.vector_norm(test)
        / torch.linalg.vector_norm(ref)
    )

    print(f"\n{name}")
    print("  Relative L2    :", rel.item())
    print("  Cosine         :", cosine.item())
    print("  Magnitude ratio:", mag.item())
    print(
        "  RMSE           :",
        diff.square().mean().sqrt().item(),
    )
    print(
        "  Max abs diff   :",
        diff.abs().max().item(),
    )

    return rel.item(), cosine.item(), mag.item()


def main():

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 Expert 0 FULL EXPERT differential")
    print("=" * 78)

    device = torch.device("cuda:0")

    # ----------------------------------------------------------
    # Original FP8 expert
    # ----------------------------------------------------------

    print("\nReading original FP8 w1/w2/w3...")

    with safe_open(
        SOURCE,
        framework="pt",
        device="cpu",
    ) as f:

        w1 = dequant(
            f,
            "layers.35.experts.0.w1",
        ).to(device)

        w2 = dequant(
            f,
            "layers.35.experts.0.w2",
        ).to(device)

        w3 = dequant(
            f,
            "layers.35.experts.0.w3",
        ).to(device)

    print("w1:", tuple(w1.shape))
    print("w2:", tuple(w2.shape))
    print("w3:", tuple(w3.shape))

    # ----------------------------------------------------------
    # Runtime K2 MoE
    # ----------------------------------------------------------

    print("\nConstructing K2 runtime MoE...")

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

    moe.load(device)
    torch.cuda.synchronize()

    gate = moe.gates[0]
    up = moe.ups[0]
    down = moe.downs[0]

    print("gate:", gate.quant_type)
    print("up  :", up.quant_type)
    print("down:", down.quant_type)

    # ----------------------------------------------------------
    # Same deterministic activation
    # ----------------------------------------------------------

    torch.manual_seed(42)

    x = torch.randn(
        (8, base.HIDDEN),
        dtype=torch.float16,
        device=device,
    )

    stats("input", x)

    # ----------------------------------------------------------
    # Original expert
    #
    # Mistral:
    #   w2(silu(w1(x)) * w3(x))
    # ----------------------------------------------------------

    print("\nRunning ORIGINAL FP8 expert...")

    g_ref = x.float() @ w1.T
    u_ref = x.float() @ w3.T

    h_ref = F.silu(g_ref) * u_ref

    y_ref = h_ref @ w2.T

    torch.cuda.synchronize()

    # ----------------------------------------------------------
    # K2 expert, explicit projection path
    # ----------------------------------------------------------

    print("Running K2 expert...")

    g_k2 = gate.forward(x, {}).float()
    u_k2 = up.forward(x, {}).float()

    # Match runtime intermediate precision.
    #
    # BlockSparseMLP uses FP16 intermediate activations.
    h_k2 = (
        F.silu(g_k2) * u_k2
    ).half()

    y_k2 = down.forward(h_k2, {}).float()

    torch.cuda.synchronize()

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    print("\n" + "=" * 78)
    print("INTERMEDIATE DIFFERENTIALS")
    print("=" * 78)

    stats("gate original", g_ref)
    stats("gate K2", g_k2)
    compare("GATE", g_ref, g_k2)

    stats("up original", u_ref)
    stats("up K2", u_k2)
    compare("UP", u_ref, u_k2)

    stats("swiglu original", h_ref)
    stats("swiglu K2", h_k2)
    compare("SWIGLU", h_ref, h_k2)

    print("\n" + "=" * 78)
    print("FULL EXPERT OUTPUT")
    print("=" * 78)

    stats("expert original", y_ref)
    stats("expert K2", y_k2)

    rel, cos, mag = compare(
        "FULL EXPERT",
        y_ref,
        y_k2,
    )

    print("\n" + "=" * 78)

    if (
        torch.isfinite(y_k2).all()
        and cos > 0.90
        and rel < 0.50
        and 0.70 < mag < 1.30
    ):
        print("FULL K2 EXPERT differential: PASS")
    else:
        print("FULL K2 EXPERT differential: DEGRADED")

    print("=" * 78)


if __name__ == "__main__":
    main()

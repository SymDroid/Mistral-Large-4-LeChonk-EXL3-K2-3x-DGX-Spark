#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import sys
import torch

ROOT = Path("/workspace/mistral-large-3-flash")
EXL3 = ROOT / "refs/exllamav3"
RUNTIME_DIR = ROOT / "quant/runtime_test"

sys.path.insert(0, str(EXL3))

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.mlp import GatedMLP


LAYER = 35
HIDDEN = 7168
EXPERT_HIDDEN = 4096
NUM_EXPERTS = 128
TOP_K = 4

BASE = f"language_model.model.layers.{LAYER}"


class MinimalConfig:
    pass


def make_config():
    cfg = MinimalConfig()

    print("Opening SafetensorsCollection:")
    print(" ", RUNTIME_DIR)

    cfg.stc = SafetensorsCollection(
        str(RUNTIME_DIR)
    )

    # Fields used by Module / Linear / MoE runtime.
    cfg.infer_params = SimpleNamespace(
        no_reconstruct=False,
    )

    cfg.arch = SimpleNamespace()

    return cfg


def main():
    print("=" * 76)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 native BlockSparseMLP runtime test")
    print("=" * 76)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")

    cfg = make_config()

    # ------------------------------------------------------------
    # Shared expert
    #
    # Native:
    #   shared_experts.w1 -> gate_proj
    #   shared_experts.w3 -> up_proj
    #   shared_experts.w2 -> down_proj
    #
    # Large 3 has exactly one shared expert of width 4096.
    # ------------------------------------------------------------

    print()
    print("Constructing shared expert...")

    shared = GatedMLP(
        config=cfg,
        key=f"{BASE}.mlp.shared_experts",
        hidden_size=HIDDEN,
        intermediate_size=EXPERT_HIDDEN,
        key_gate="gate_proj",
        key_up="up_proj",
        key_down="down_proj",
        activation_fn="silu",
        out_dtype=torch.float,
    )

    # ------------------------------------------------------------
    # Routed MoE
    #
    # The important bit:
    #
    #   experts.{expert_idx}.gate_proj
    #   experts.{expert_idx}.up_proj
    #   experts.{expert_idx}.down_proj
    #
    # exactly matches our assembled EXL3 K2 groups.
    #
    # interm_div=128 is what ExLlama's Mistral4 architecture uses
    # for this MoE family.
    # ------------------------------------------------------------

    print("Constructing BlockSparseMLP...")

    moe = BlockSparseMLP(
        config=cfg,
        key=f"{BASE}.mlp",

        hidden_size=HIDDEN,
        intermediate_size=EXPERT_HIDDEN,

        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,

        key_gate="experts.{expert_idx}.gate_proj",
        key_up="experts.{expert_idx}.up_proj",
        key_down="experts.{expert_idx}.down_proj",

        key_routing_gate="gate",

        shared_experts=shared,

        activation_fn="silu",

        interm_dtype=torch.half,
        interm_div=1.0,

        router_type="std",

        out_dtype=torch.float,
    )

    print()
    print("Submodules constructed:")
    print("  gates :", len(moe.gates))
    print("  ups   :", len(moe.ups))
    print("  downs :", len(moe.downs))

    if len(moe.gates) != NUM_EXPERTS:
        raise RuntimeError(
            f"Expected {NUM_EXPERTS} gates, got {len(moe.gates)}"
        )

    if len(moe.ups) != NUM_EXPERTS:
        raise RuntimeError(
            f"Expected {NUM_EXPERTS} ups, got {len(moe.ups)}"
        )

    if len(moe.downs) != NUM_EXPERTS:
        raise RuntimeError(
            f"Expected {NUM_EXPERTS} downs, got {len(moe.downs)}"
        )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    print()
    print("Loading MoE onto CUDA...")
    print("This loads all 384 routed K2 matrices plus router/shared expert.")

    torch.cuda.empty_cache()

    before = torch.cuda.memory_allocated(device)

    moe.load(device)

    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated(device)

    print()
    print("Load complete.")
    print(
        "CUDA memory delta:",
        f"{(after - before) / 1024**3:.3f} GiB"
    )

    print()
    print("MoE runtime state:")
    print("  is_quantized       :", moe.is_quantized)
    print("  support_quant_paths:", moe.support_quant_paths)
    print("  support_fused      :", moe.support_fused)
    print("  device             :", moe.device)

    qtypes = {}

    for name, modules in (
        ("gate", moe.gates),
        ("up", moe.ups),
        ("down", moe.downs),
    ):
        values = {}

        for m in modules:
            qt = str(m.quant_type)
            values[qt] = values.get(qt, 0) + 1

        qtypes[name] = values

    print()
    print("Expert quantization:")
    for name, values in qtypes.items():
        print(f"  {name:4s}: {values}")

    if not moe.is_quantized:
        raise RuntimeError(
            "BlockSparseMLP did not recognize all experts as quantized"
        )

    if any(
        str(m.quant_type) != "exl3"
        for m in moe.gates + moe.ups + moe.downs
    ):
        raise RuntimeError(
            "At least one routed expert projection is not EXL3"
        )

    print()
    print("384/384 routed projections recognized as EXL3: PASS")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    print()
    print("Testing shared expert separately...")

    torch.manual_seed(42)
    x_probe = torch.randn(
        (1, HIDDEN),
        device=device,
        dtype=torch.half,
    )

    with torch.inference_mode():
        shared_y = shared.forward(
            x_probe.view(1, 1, HIDDEN),
            {},
            out_dtype=torch.half,
        )

    sy = shared_y.float()

    print()
    print("Shared expert projection diagnostics:")

    sx = x_probe

    with torch.inference_mode():
        sg = shared.gates[0].forward(sx, {})
        su = shared.ups[0].forward(sx, {})

    sgf = sg.float()
    suf = su.float()

    print("  gate_proj:")
    print("    quant :", shared.gates[0].quant_type)
    print("    dtype :", sg.dtype)
    print("    finite:", torch.isfinite(sgf).all().item())
    print("    range :", sgf.min().item(), sgf.max().item())

    print("  up_proj:")
    print("    quant :", shared.ups[0].quant_type)
    print("    dtype :", su.dtype)
    print("    finite:", torch.isfinite(suf).all().item())
    print("    range :", suf.min().item(), suf.max().item())

    sa = torch.nn.functional.silu(sgf) * suf

    print("  activation:")
    print("    finite:", torch.isfinite(sa).all().item())
    print("    range :", sa.min().item(), sa.max().item())

    with torch.inference_mode():
        sd = shared.downs[0].forward(
            sa.to(torch.half),
            {},
        )

    sdf = sd.float()

    print("  down_proj:")
    print("    quant :", shared.downs[0].quant_type)
    print("    dtype :", sd.dtype)
    print("    finite:", torch.isfinite(sdf).all().item())
    print("    range :", sdf.min().item(), sdf.max().item())

    print("Shared expert:")
    print("  shape :", tuple(shared_y.shape))
    print("  dtype :", shared_y.dtype)
    print("  min   :", sy.min().item())
    print("  max   :", sy.max().item())
    print("  mean  :", sy.mean().item())
    print("  std   :", sy.std().item())
    print("  finite:", torch.isfinite(sy).all().item())

    print()
    print("Router separately...")

    with torch.inference_mode():
        router_y = moe.routing_gate.forward(
            x_probe,
            {},
        )

    ry = router_y.float()

    print("Router:")
    print("  shape :", tuple(router_y.shape))
    print("  dtype :", router_y.dtype)
    print("  min   :", ry.min().item())
    print("  max   :", ry.max().item())
    print("  mean  :", ry.mean().item())
    print("  finite:", torch.isfinite(ry).all().item())

    print()
    print("Testing routed expert 90 separately...")

    e = 90

    with torch.inference_mode():
        g = moe.gates[e].forward(x_probe, {})
        u = moe.ups[e].forward(x_probe, {})

        print(
            "  gate finite:",
            torch.isfinite(g.float()).all().item(),
            "range:",
            g.float().min().item(),
            g.float().max().item(),
        )

        print(
            "  up finite  :",
            torch.isfinite(u.float()).all().item(),
            "range:",
            u.float().min().item(),
            u.float().max().item(),
        )

        a = torch.nn.functional.silu(g.float()) * u.float()
        a = a.to(torch.half)

        print(
            "  activation :",
            torch.isfinite(a.float()).all().item(),
            "range:",
            a.float().min().item(),
            a.float().max().item(),
        )

        d = moe.downs[e].forward(a, {})

    df = d.float()

    print("Expert 90 output:")
    print("  shape :", tuple(d.shape))
    print("  dtype :", d.dtype)
    print("  min   :", df.min().item())
    print("  max   :", df.max().item())
    print("  mean  :", df.mean().item())
    print("  std   :", df.std().item())
    print("  finite:", torch.isfinite(df).all().item())

    print()
    print("Running real MoE forward...")

    torch.manual_seed(42)

    # Use small amplitude, closer to normalized transformer states
    # than a huge arbitrary random vector.
    x = torch.randn(
        (1, 1, HIDDEN),
        device=device,
        dtype=torch.half,
    )

    params = {}

    torch.cuda.synchronize()

    with torch.inference_mode():
        y = moe.forward(
            x,
            params,
            out_dtype=torch.half,
        )

    torch.cuda.synchronize()

    print()
    print("Input :", tuple(x.shape), x.dtype, x.device)
    print("Output:", tuple(y.shape), y.dtype, y.device)

    yf = y.float()

    print()
    print("Output statistics:")
    print("  min :", yf.min().item())
    print("  max :", yf.max().item())
    print("  mean:", yf.mean().item())
    print("  std :", yf.std().item())

    finite = torch.isfinite(yf).all().item()

    print("  finite:", finite)

    if tuple(y.shape) != (1, 1, HIDDEN):
        raise RuntimeError(
            f"Unexpected output shape: {tuple(y.shape)}"
        )

    if not finite:
        raise RuntimeError(
            "MoE output contains NaN or Inf"
        )

    print()
    print("=" * 76)
    print("MISTRAL LARGE 3 FLASH K2 MoE CUDA FORWARD: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()

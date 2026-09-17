#!/usr/bin/env python3

from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
REF = ROOT / "upstream-reference/consolidated-00126-of-00272.safetensors"

sys.path.insert(0, str(ROOT / "refs/exllamav3"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_layer35_block_runtime as base


def dequant_fp8(w, s, device):
    w = w.to(device)
    s = s.to(device)

    sf = (
        s.float()
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)
    )

    sf = sf[:w.shape[0], :w.shape[1]]

    return w.float() * sf


def stats(name, x):
    xf = x.float()

    print(f"\n{name}")
    print("-" * 72)
    print("shape :", tuple(x.shape))
    print("RMS   :", torch.sqrt(torch.mean(xf * xf)).item())
    print("mean  :", xf.mean().item())
    print("std   :", xf.std().item())
    print("min   :", xf.min().item())
    print("max   :", xf.max().item())


def compare(name, ref, test):
    a = ref.float().reshape(-1)
    b = test.float().reshape(-1)

    d = b - a

    ref_rms = torch.sqrt(torch.mean(a * a)).item()
    test_rms = torch.sqrt(torch.mean(b * b)).item()
    rmse = torch.sqrt(torch.mean(d * d)).item()

    rel = (
        torch.linalg.vector_norm(d) /
        torch.linalg.vector_norm(a)
    ).item()

    cos = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=1,
    ).item()

    mag = (
        torch.linalg.vector_norm(b) /
        torch.linalg.vector_norm(a)
    ).item()

    print("\n" + name)
    print("=" * 72)
    print(f"Reference RMS   : {ref_rms:.9f}")
    print(f"Runtime RMS     : {test_rms:.9f}")
    print(f"RMSE            : {rmse:.9f}")
    print(f"Relative L2     : {rel:.9f}")
    print(f"Cosine          : {cos:.9f}")
    print(f"Magnitude ratio : {mag:.9f}")
    print(f"Max abs diff    : {d.abs().max().item():.9f}")

    return cos, rel, mag


def main():
    device = torch.device("cuda:0")

    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 shared expert FP8 differential")
    print("=" * 78)

    cfg = base.make_config()

    shared = base.GatedMLP(
        config=cfg,
        key=f"{base.BASE}.mlp.shared_experts",
        hidden_size=base.HIDDEN,
        intermediate_size=base.EXPERT_HIDDEN,

        key_up="up_proj",
        key_gate="gate_proj",
        key_down="down_proj",

        qmap="block.mlp",

        interm_dtype=torch.half,
        out_dtype=torch.float,

        select_hq_bits=2,
        activation_fn="silu",
    )

    print("\nLoading runtime shared expert...")
    shared.load(device)
    torch.cuda.synchronize()

    print("\nRuntime projection state:")

    for name, mod in [
        ("gate", shared.gates[0]),
        ("up", shared.ups[0]),
        ("down", shared.downs[0]),
    ]:
        print(
            f"{name:5s}: "
            f"quant_type={mod.quant_type} "
            f"weight_scale={getattr(mod, 'weight_scale', None)}"
        )

    # Same deterministic normalized-scale input family.
    torch.manual_seed(42)

    x = torch.randn(
        (8, base.HIDDEN),
        dtype=torch.float16,
        device=device,
    )

    stats("INPUT", x)

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------

    print("\nRunning runtime shared expert...")

    with torch.inference_mode():

        g_rt = shared.gates[0].forward(x, {})
        u_rt = shared.ups[0].forward(x, {})

        h_rt = (
            F.silu(g_rt) * u_rt
        ).half()

        y_rt_explicit = shared.downs[0].forward(
            h_rt,
            {},
        ).float()

        y_rt_forward = shared.forward(
            x.unsqueeze(0),
            {},
            out_dtype=torch.float,
        ).float().reshape(-1, 7168)

    torch.cuda.synchronize()

    stats("RUNTIME gate", g_rt)
    stats("RUNTIME up", u_rt)
    stats("RUNTIME SwiGLU", h_rt)
    stats("RUNTIME explicit output", y_rt_explicit)
    stats("RUNTIME shared.forward output", y_rt_forward)

    compare(
        "RUNTIME INTERNAL CONSISTENCY",
        y_rt_explicit,
        y_rt_forward,
    )

    # ------------------------------------------------------------
    # Original native FP8
    # ------------------------------------------------------------

    print("\nLoading original FP8 shared expert...")

    with safe_open(
        str(REF),
        framework="pt",
        device="cpu",
    ) as f:

        native = {}

        for p in ("w1", "w2", "w3"):

            wk = f"layers.35.shared_experts.{p}.weight"
            sk = f"layers.35.shared_experts.{p}.weight_scale"

            w = f.get_tensor(wk)
            s = f.get_tensor(sk)

            print(
                f"{p}: "
                f"weight={tuple(w.shape)} {w.dtype} "
                f"scale={tuple(s.shape)} {s.dtype} "
                f"scale[min,max]="
                f"{s.float().min().item():.9g}, "
                f"{s.float().max().item():.9g}"
            )

            native[p] = dequant_fp8(
                w,
                s,
                device,
            )

    xf = x.float()

    print("\nRunning original FP8 mathematics...")

    g_ref = xf @ native["w1"].T
    u_ref = xf @ native["w3"].T

    h_ref = F.silu(g_ref) * u_ref

    y_ref = h_ref @ native["w2"].T

    torch.cuda.synchronize()

    stats("REFERENCE gate", g_ref)
    stats("REFERENCE up", u_ref)
    stats("REFERENCE SwiGLU", h_ref)
    stats("REFERENCE output", y_ref)

    # ------------------------------------------------------------
    # Projection-by-projection
    # ------------------------------------------------------------

    compare(
        "GATE: ORIGINAL FP8 vs RUNTIME",
        g_ref,
        g_rt,
    )

    compare(
        "UP: ORIGINAL FP8 vs RUNTIME",
        u_ref,
        u_rt,
    )

    compare(
        "SWIGLU: ORIGINAL FP8 vs RUNTIME",
        h_ref,
        h_rt,
    )

    cos, rel, mag = compare(
        "FULL SHARED EXPERT: ORIGINAL FP8 vs RUNTIME",
        y_ref,
        y_rt_forward,
    )

    print("\n" + "=" * 78)

    if cos > 0.999 and rel < 0.01:
        print("RESULT: SHARED EXPERT FP8 RUNTIME MATCHES ORIGINAL")
    elif cos > 0.99:
        print("RESULT: SHARED EXPERT HAS MEASURABLE NUMERICAL DRIFT")
    else:
        print("RESULT: SHARED EXPERT SCALING/MAPPING IS SUSPECT")

    print("=" * 78)


if __name__ == "__main__":
    main()

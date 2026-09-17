import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "refs/exllamav3")

# Reuse the already proven construction helpers/constants.
import test_layer35_block_runtime as base

from exllamav3.modules import MLAttention


MODEL_DIR = Path("quant/mistral-large-3-flash-k2")
SHARD = MODEL_DIR / "model-layer-35-moe.safetensors"

KEY = (
    "language_model.model.layers.35."
    "self_attn.q_b_proj"
)


def stats(name, x):
    x = x.float()
    print(
        f"{name:12s}"
        f" shape={tuple(x.shape)}"
        f" mean={x.mean().item():.6e}"
        f" std={x.std().item():.6e}"
        f" absmax={x.abs().max().item():.6e}"
        f" finite={torch.isfinite(x).float().mean().item()*100:.2f}%"
    )


def main():
    print("=" * 78)
    print("MISTRAL LARGE 3 FLASH K2")
    print("Layer 35 q_b_proj FP8 differential")
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")

    cfg = base.make_config()
    rope_settings = base.make_rope_settings()

    print("\nConstructing MLAttention exactly as proven block test...")

    attn = MLAttention(
        config=cfg,
        key=f"{base.BASE}.self_attn",
        layer_idx=base.LAYER,
        hidden_size=base.HIDDEN,
        num_q_heads=base.NUM_Q_HEADS,
        kv_lora_rank=base.KV_LORA_RANK,
        qk_nope_head_dim=base.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=base.QK_ROPE_HEAD_DIM,
        v_head_dim=base.V_HEAD_DIM,
        rope_settings=rope_settings,
        q_lora_rank=base.Q_LORA_RANK,
        sm_scale=base.SM_SCALE,
        rms_norm_eps=base.RMS_EPS,
        qmap="block.attn",
        out_dtype=torch.float,
        select_hq_bits=2,
    )

    print("Loading attention...")
    attn.load(device)
    torch.cuda.synchronize()

    # Locate q_b projection without assuming more than necessary.
    candidates = [
        "q_b_proj",
        "q_b",
        "q_proj_b",
    ]

    q_b = None
    found_name = None

    for name in candidates:
        if hasattr(attn, name):
            q_b = getattr(attn, name)
            found_name = name
            break

    if q_b is None:
        print("\nMLAttention attributes containing 'q' or 'proj':")
        for name in dir(attn):
            if "q" in name.lower() or "proj" in name.lower():
                try:
                    obj = getattr(attn, name)
                    print(" ", name, type(obj))
                except Exception:
                    pass
        raise RuntimeError("Could not locate q_b projection")

    print("Runtime projection:", found_name, type(q_b))
    print("quant_type:", getattr(q_b, "quant_type", None))

    # ------------------------------------------------------------
    # Original assembled checkpoint FP8 data
    # ------------------------------------------------------------
    print("\nReading assembled FP8 tensor...")

    with safe_open(SHARD, framework="pt", device="cpu") as f:
        w_fp8 = f.get_tensor(KEY + ".weight")
        scale = f.get_tensor(KEY + ".weight_scale_inv")

    print("weight:", tuple(w_fp8.shape), w_fp8.dtype)
    print("scale :", tuple(scale.shape), scale.dtype)

    # Native Mistral block FP8 scale is one multiplicative scale
    # for every 128x128 weight block.
    scale_full = (
        scale.float()
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
    )

    assert scale_full.shape == w_fp8.shape, (
        scale_full.shape,
        w_fp8.shape,
    )

    w_ref = w_fp8.float() * scale_full

    stats("W reference", w_ref)

    # ------------------------------------------------------------
    # Deterministic input to q_b_proj
    #
    # Stored Linear weight is [out, in].
    # q_b input width = Q_LORA_RANK = 1536.
    # ------------------------------------------------------------
    torch.manual_seed(42)

    x = torch.randn(
        (8, base.Q_LORA_RANK),
        device=device,
        dtype=torch.float16,
    )

    # FP32 reference matmul:
    # x [N,in] @ W.T [in,out]
    y_ref = (
        x.float()
        @ w_ref.to(device=device).T
    )

    torch.cuda.synchronize()

    print("\nRunning ExLlama q_b_proj...")

    # ExLlama Linear modules normally expose forward().
    y_run = q_b.forward(x, {})

    torch.cuda.synchronize()

    y_ref_f = y_ref.float()
    y_run_f = y_run.float()

    stats("reference", y_ref_f)
    stats("runtime", y_run_f)

    if y_ref_f.shape != y_run_f.shape:
        raise RuntimeError(
            f"Shape mismatch: reference={y_ref_f.shape}, "
            f"runtime={y_run_f.shape}"
        )

    diff = y_run_f - y_ref_f

    rmse = diff.square().mean().sqrt()
    ref_rms = y_ref_f.square().mean().sqrt()
    rel_l2 = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(y_ref_f)
    )

    a = y_ref_f.flatten()
    b = y_run_f.flatten()

    cosine = torch.dot(a, b) / (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    print("\n" + "=" * 78)
    print("DIFFERENTIAL")
    print("=" * 78)

    print("RMSE        :", rmse.item())
    print("Reference RMS:", ref_rms.item())
    print("Relative L2 :", rel_l2.item())
    print("Cosine      :", cosine.item())
    print("Max abs diff:", diff.abs().max().item())

    # Deliberately generous. A correctly interpreted FP8 weight
    # should be vastly closer than this.
    if (
        torch.isfinite(diff).all()
        and cosine.item() > 0.999
        and rel_l2.item() < 0.02
    ):
        print("\nFP8 q_b_proj differential: PASS")
    else:
        print("\nFP8 q_b_proj differential: FAIL")
        print(">>> FP8 runtime scale semantics/path is now a prime suspect.")


if __name__ == "__main__":
    main()

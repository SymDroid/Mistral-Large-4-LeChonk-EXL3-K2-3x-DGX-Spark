import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


ROOT = Path("/workspace/ml4")
META = ROOT / "metadata"
WEIGHTS = ROOT / "weights"
OUTDIR = ROOT / "dense0-k4"
OUTFILE = OUTDIR / "dense0.safetensors"

REPO = "mistralai/Mistral-Large-4-1T-A52B"
DEVICE = torch.device("cuda:0")

OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Checkpoint index
# ---------------------------------------------------------

candidates = list(META.rglob("*.safetensors.index.json"))
if not candidates:
    raise SystemExit("Geen safetensors index gevonden")

INDEX = candidates[0]
WM = json.loads(INDEX.read_text())["weight_map"]


def shard_for(key):
    if key not in WM:
        raise KeyError(key)
    return WM[key]


def ensure_shard(shard):
    p = WEIGHTS / shard

    if p.is_file():
        return p

    print("download:", shard)

    downloaded = hf_hub_download(
        repo_id=REPO,
        filename=shard,
        local_dir=WEIGHTS,
    )

    return Path(downloaded)


handles = {}


def raw(key):
    shard = shard_for(key)
    path = ensure_shard(shard)

    if shard not in handles:
        handles[shard] = safe_open(
            path,
            framework="pt",
            device="cpu",
        )

    return handles[shard].get_tensor(key)


# ---------------------------------------------------------
# Exact ML4 FP8 block dequant
# ---------------------------------------------------------

def dequant_weight(base):
    key = base + ".weight"
    t = raw(key)

    scale_key = base + ".qscale_weight"

    if scale_key not in WM:
        return t.float().contiguous()

    scale = raw(scale_key).float()

    r, c = t.shape

    assert r % 128 == 0
    assert c % 128 == 0

    x = t.float().reshape(
        r // 128, 128,
        c // 128, 128,
    )

    x.mul_(scale[:, None, :, None])

    result = x.reshape(r, c).contiguous()

    del t, scale, x
    gc.collect()

    return result


# ---------------------------------------------------------
# Same blind EXL3 K4 path as attention
# ---------------------------------------------------------

def quantize_k4(W, prefix):
    in_features = W.shape[1]

    print()
    print("K4:", prefix)
    print("shape:", tuple(W.shape))
    print("input:", in_features)

    Wq = W.T.contiguous().to(DEVICE)

    H_data = {
        "H": torch.empty(
            (in_features, in_features),
            device="meta",
        ),
        "device": DEVICE,
    }

    qargs = {
        "K": 4,
        "devices": ["cuda:0"],
        "seed": 42,
        "apply_out_scales": False,
        "buf_size_k": 128,
        "sigma_reg": 0.025,
    }

    q, proxy, packed = quantize_exl3(
        Wq,
        H_data,
        qargs,
        True,
        progress_str=prefix,
        verbose=False,
    )

    result = {
        f"{prefix}.{field}":
            tensor.detach().cpu().contiguous()
        for field, tensor in packed.items()
    }

    print("proxy:", float(proxy))
    print("fallback:", qargs.get("q_fallback", False))
    print("packed:", sorted(result))

    del Wq, H_data, q, packed
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ---------------------------------------------------------
# Layer 0 dense MLP
# ---------------------------------------------------------

mapping = (
    (
        "layers.0.feed_forward.w1",
        "language_model.model.layers.0.mlp.gate_proj",
    ),
    (
        "layers.0.feed_forward.w3",
        "language_model.model.layers.0.mlp.up_proj",
    ),
    (
        "layers.0.feed_forward.w2",
        "language_model.model.layers.0.mlp.down_proj",
    ),
)

out = {}

for source, dest in mapping:
    W = dequant_weight(source)
    out.update(quantize_k4(W, dest))

    del W
    gc.collect()


# ---------------------------------------------------------
# Strict validation before write
# ---------------------------------------------------------

expected = {
    f"{dest}.{field}"
    for _, dest in mapping
    for field in ("suh", "svh", "trellis")
}

if set(out) != expected:
    raise SystemExit(
        f"Key mismatch before save\n"
        f"missing={sorted(expected - set(out))}\n"
        f"extra={sorted(set(out) - expected)}"
    )


save_file(
    out,
    OUTFILE,
)

with safe_open(
    OUTFILE,
    framework="pt",
    device="cpu",
) as f:
    actual = set(f.keys())

if actual != expected:
    raise SystemExit(
        f"VALIDATION FAILED\n"
        f"missing={sorted(expected - actual)}\n"
        f"extra={sorted(actual - expected)}"
    )

mib = OUTFILE.stat().st_size / 1024**2

print()
print("=" * 72)
print("DENSE LAYER 0 VALID")
print("keys:", len(actual))
print(f"size: {mib:.2f} MiB")
print("source shard retained for explicit cleanup")
print("=" * 72)

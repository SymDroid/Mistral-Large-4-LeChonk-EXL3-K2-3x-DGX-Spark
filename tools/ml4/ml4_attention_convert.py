import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

L = args.layer

if not 0 <= L <= 71:
    raise SystemExit("Layer moet tussen 0 en 71 liggen.")

ROOT = Path("/workspace/ml4")
META = ROOT / "metadata"
WEIGHTS = ROOT / "weights"
OUT = ROOT / "attention-k4" / f"layer-{L:03d}"

REPO = "mistralai/Mistral-Large-4-1T-A52B"
DEVICE = torch.device("cuda:0")

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Checkpoint index
# ---------------------------------------------------------

candidates = list(META.rglob("*.safetensors.index.json"))
if not candidates:
    raise SystemExit("Geen safetensors index gevonden")

INDEX = candidates[0]
data = json.loads(INDEX.read_text())
WM = data["weight_map"]


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
# FP8 block dequant
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

    return result


# ---------------------------------------------------------
# Blind EXL3 K4
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
# Layer 1
# ---------------------------------------------------------

native = f"layers.{L}.attention"
canon = f"language_model.model.layers.{L}.self_attn"

out = {}


# FP8 -> K4

for src, dst in (
    ("wq", "q_proj"),
    ("wo", "o_proj"),
    ("indexer.wq_b", "indexer.wq_b"),
):
    source_base = f"{native}.{src}"

    if source_base + ".weight" not in WM:
        print("skip absent K4:", source_base + ".weight")
        continue

    W = dequant_weight(source_base)

    out.update(
        quantize_k4(
            W,
            f"{canon}.{dst}",
        )
    )

    del W
    gc.collect()


# Raw BF16 state

raw_map = {
    "wkv_a_with_mqa.weight":
        "kv_a_proj_with_mqa.weight",

    "wkv_b.weight":
        "kv_b_proj.weight",

    "kv_a_norm.weight":
        "kv_a_layernorm.weight",

    "indexer.wk.weight":
        "indexer.wk.weight",

    "indexer.weights_proj.weight":
        "indexer.weights_proj.weight",

    "indexer.q_input_norm.weight":
        "indexer.q_input_norm.weight",

    "indexer.q_input_norm.bias":
        "indexer.q_input_norm.bias",

    "indexer.k_norm.weight":
        "indexer.k_norm.weight",

    "indexer.k_norm.bias":
        "indexer.k_norm.bias",

    "ssmax_scaling.s":
        "ssmax_scaling.s",

    "ssmax_scaling.b":
        "ssmax_scaling.b",

    "indexer.ssmax_scaling.s":
        "indexer.ssmax_scaling.s",

    "indexer.ssmax_scaling.b":
        "indexer.ssmax_scaling.b",
}

for src, dst in raw_map.items():
    key = f"{native}.{src}"

    if key not in WM:
        print("skip absent raw:", key)
        continue

    out[f"{canon}.{dst}"] = (
        raw(key)
        .to(torch.bfloat16)
        .contiguous()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

outfile = OUT / "attention.safetensors"

save_file(
    out,
    outfile,
    metadata={
        "source": "Mistral-Large-4-1T-A52B",
        "layer": str(L),
        "attention_quant": "EXL3-K4-blind",
    },
)

print()
print("SAVED:", outfile)
print("keys:", len(out))
print(
    "size:",
    f"{outfile.stat().st_size / 1024**2:.2f} MiB"
)

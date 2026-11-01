import gc
import json
import os
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path("/workspace/ml4")
FINAL = ROOT / "Mistral-Large-4-EXL3-K2K4"
META = ROOT / "metadata"
WEIGHTS = ROOT / "weights"

SHARD53 = WEIGHTS / "consolidated-00053-of-00054.safetensors"

if not SHARD53.is_file():
    raise SystemExit("Pinned shard 53 ontbreekt. Niets gewijzigd.")

FINAL.mkdir(parents=True, exist_ok=True)

# Refuse to mix with an unknown prior assembly.
existing = list(FINAL.iterdir())
if existing:
    raise SystemExit(
        f"Final directory is niet leeg: {FINAL}\n"
        "Verwijder/hernoem die eerst bewust."
    )


def hardlink(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(src)
    os.link(src, dst)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

shutil.copy2(META / "config.json", FINAL / "config.json")


# ---------------------------------------------------------
# K2 MoE: flatten 9159 files using unique filenames
# ---------------------------------------------------------

moe_files = sorted(
    p for p in ROOT.glob("k2-layer*-4096/**/*.safetensors")
    if p.is_file()
)

if len(moe_files) != 9159:
    raise SystemExit(
        f"Verwacht 9159 K2 files, gevonden {len(moe_files)}"
    )

for src in moe_files:
    m = re.search(r"k2-layer(\d+)-4096", str(src))
    if not m:
        raise SystemExit(f"Kan layer niet bepalen: {src}")

    layer = int(m.group(1))
    rel = src.relative_to(
        ROOT / f"k2-layer{layer}-4096"
    )

    # experts/expert-000.safetensors -> expert-000
    # shared-expert.safetensors etc. remain unique via layer prefix
    stem = "-".join(rel.with_suffix("").parts)

    dst = FINAL / f"moe-l{layer:03d}-{stem}.safetensors"
    hardlink(src, dst)


# ---------------------------------------------------------
# Attention K4: 72 hardlinks
# ---------------------------------------------------------

attn_files = sorted(
    ROOT.glob("attention-k4/layer-*/attention.safetensors")
)

if len(attn_files) != 72:
    raise SystemExit(
        f"Verwacht 72 attention files, gevonden {len(attn_files)}"
    )

for src in attn_files:
    layer = int(src.parent.name.split("-")[1])
    hardlink(
        src,
        FINAL / f"attn-l{layer:03d}.safetensors",
    )


# ---------------------------------------------------------
# Dense layer 0 K4
# ---------------------------------------------------------

dense0 = ROOT / "dense0-k4" / "dense0.safetensors"
hardlink(dense0, FINAL / "dense-l000.safetensors")


# ---------------------------------------------------------
# Raw BF16 text state from shard 53
# Split large tensors to reduce peak RAM
# ---------------------------------------------------------

with safe_open(SHARD53, framework="pt", device="cpu") as f:

    # Embedding
    emb = f.get_tensor("tok_embeddings.weight").contiguous()
    save_file(
        {
            "language_model.model.embed_tokens.weight": emb,
        },
        FINAL / "embedding.safetensors",
    )
    del emb
    gc.collect()

    # LM head
    head = f.get_tensor("output.weight").contiguous()
    save_file(
        {
            "language_model.lm_head.weight": head,
        },
        FINAL / "lm-head.safetensors",
    )
    del head
    gc.collect()

    # Small raw state
    small = {
        "language_model.model.norm.weight":
            f.get_tensor("norm.weight").contiguous(),

        # Intentionally exact top-level key for patched ML4 loader
        "lerope.alpha":
            f.get_tensor("lerope.alpha").contiguous(),
    }

    for layer in range(72):
        small[
            f"language_model.model.layers."
            f"{layer}.input_layernorm.weight"
        ] = f.get_tensor(
            f"layers.{layer}.attention_norm.weight"
        ).contiguous()

        small[
            f"language_model.model.layers."
            f"{layer}.post_attention_layernorm.weight"
        ] = f.get_tensor(
            f"layers.{layer}.ffn_norm.weight"
        ).contiguous()

        if layer >= 1:
            small[
                f"language_model.model.layers."
                f"{layer}.mlp.gate.weight"
            ] = f.get_tensor(
                f"layers.{layer}.gate.weight"
            ).contiguous()

save_file(
    small,
    FINAL / "text-state.safetensors",
)

del small
gc.collect()


# ---------------------------------------------------------
# Global tensor-key collision audit
# ---------------------------------------------------------

tensor_files = sorted(FINAL.glob("*.safetensors"))

seen = {}
duplicates = []

for path in tensor_files:
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key in seen:
                duplicates.append(
                    (key, seen[key], path.name)
                )
            else:
                seen[key] = path.name

if duplicates:
    print("DUPLICATE KEYS:")
    for key, a, b in duplicates[:50]:
        print(key, a, b)
    raise SystemExit(
        f"{len(duplicates)} duplicate tensor keys gevonden"
    )


# ---------------------------------------------------------
# Essential required-key audit
# ---------------------------------------------------------

required = {
    "language_model.model.embed_tokens.weight",
    "language_model.lm_head.weight",
    "language_model.model.norm.weight",
    "lerope.alpha",

    "language_model.model.layers.0.mlp.gate_proj.suh",
    "language_model.model.layers.0.mlp.up_proj.suh",
    "language_model.model.layers.0.mlp.down_proj.suh",

    "language_model.model.layers.0.self_attn.q_proj.suh",
    "language_model.model.layers.71.self_attn.q_proj.suh",

    "language_model.model.layers.1.mlp.gate.weight",
    "language_model.model.layers.71.mlp.gate.weight",

    "language_model.model.layers.1.mlp.experts.0.gate_proj.suh",
    "language_model.model.layers.71.mlp.experts.127.down_proj.suh",

    "language_model.model.layers.1.mlp.shared_experts.gate_proj.suh",
    "language_model.model.layers.71.mlp.shared_experts.down_proj.suh",
}

missing = sorted(required - set(seen))
if missing:
    print("MISSING ESSENTIAL KEYS:")
    for k in missing:
        print(" ", k)
    raise SystemExit("Essential-key audit FAILED")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

logical = sum(p.stat().st_size for p in FINAL.iterdir() if p.is_file())

print()
print("=" * 72)
print("FINAL ML4 EXL3 ASSEMBLY VALID")
print("directory:", FINAL)
print("safetensors files:", len(tensor_files))
print("tensor keys:", len(seen))
print(f"logical size: {logical / 1024**3:.3f} GiB")
print("duplicate keys: 0")
print("essential-key audit: PASS")
print("shard 53 retained:", SHARD53.is_file())
print("=" * 72)

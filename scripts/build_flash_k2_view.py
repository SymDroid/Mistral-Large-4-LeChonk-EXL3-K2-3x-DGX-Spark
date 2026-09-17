from pathlib import Path
import os
import shutil

ROOT = Path("/workspace/mistral-large-3-flash")
OUT = ROOT / "Mistral-Large-3-Flash-K2"

K2 = ROOT / "quant/bootstrap_k2"
PASS = ROOT / "quant/bootstrap_passthrough"

OUT.mkdir(parents=True, exist_ok=True)

# Clean only our generated links from a previous run.
for p in OUT.glob("*.safetensors"):
    if p.is_symlink():
        p.unlink()

count_k2 = 0
count_pass = 0

# Passthrough shards
for src in sorted(PASS.glob("*.safetensors")):
    dst = OUT / f"passthrough-{src.name}"
    dst.symlink_to(src.resolve())
    count_pass += 1

# K2 expert artifacts
for src in sorted(K2.glob("layer*/expert*-k2.safetensors")):
    layer = src.parent.name
    dst = OUT / f"k2-{layer}-{src.name}"
    dst.symlink_to(src.resolve())
    count_k2 += 1

# Copy metadata that already exists locally.
metadata_candidates = [
    "params.json",
    "README.md",
    "chat_template.jinja",
    "SYSTEM_PROMPT.txt",
    "tokenizer.model.v3",
    "tekken.json",
]

meta_dirs = [
    ROOT / "upstream-meta",
    ROOT,
]

copied = []

for name in metadata_candidates:
    for base in meta_dirs:
        src = base / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            copied.append(name)
            break

print("OUTPUT:", OUT)
print("K2 links:", count_k2)
print("passthrough links:", count_pass)
print("total safetensors:", count_k2 + count_pass)
print("metadata:", sorted(set(copied)))

assert count_k2 == 22272, count_k2
assert count_pass == 68, count_pass

print("VIEW COMPLETE")

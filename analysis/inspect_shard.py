from safetensors import safe_open
from pathlib import Path
import sys

path = Path(sys.argv[1])

with safe_open(path, framework="numpy") as f:
    keys = list(f.keys())

print(f"\nFile: {path}")
print(f"Tensors: {len(keys)}\n")

for k in keys:
    print(k)

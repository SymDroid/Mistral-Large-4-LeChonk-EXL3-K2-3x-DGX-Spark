from safetensors import safe_open
from pathlib import Path
import sys

path = Path(sys.argv[1])

print(f"\nShard: {path.name}\n")
print(f"{'Tensor':<100} {'Shape':<30} {'Dtype'}")
print("-" * 150)

with safe_open(path, framework="numpy") as f:
    for key in f.keys():
        sl = f.get_slice(key)
        shape = sl.get_shape()
        dtype = sl.get_dtype()

        print(f"{key:<100} {str(shape):<30} {dtype}")

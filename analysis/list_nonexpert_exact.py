from safetensors import safe_open
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])

routed_expert = re.compile(
    r"^layers\.\d+\.experts\.\d+\.(w1|w2|w3)\.(weight|weight_scale)$"
)

with safe_open(path, framework="numpy") as f:
    for key in f.keys():
        if not key.startswith("layers."):
            continue

        if routed_expert.match(key):
            continue

        print(key)

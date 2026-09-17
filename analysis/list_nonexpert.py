from safetensors import safe_open
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])

expert = re.compile(r"^layers\.\d+\.experts\.\d+\.")

with safe_open(path, framework="numpy") as f:
    for key in f.keys():
        if key.startswith("layers.") and not expert.match(key):
            print(key)

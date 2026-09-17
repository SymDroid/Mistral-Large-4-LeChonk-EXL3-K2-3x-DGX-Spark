from safetensors import safe_open
from pathlib import Path
import re
import sys
from collections import Counter

path = Path(sys.argv[1])

expert_re = re.compile(
    r"^layers\.(\d+)\.experts\.(\d+)\.(w1|w2|w3)\.(weight|weight_scale)$"
)

counts = Counter()

with safe_open(path, framework="numpy") as f:
    for key in f.keys():

        m = expert_re.match(key)

        if m:
            layer, expert, proj, kind = m.groups()

            if kind == "weight":
                counts["expert_weight"] += 1
            else:
                counts["expert_scale"] += 1

            continue

        if "attention" in key:
            counts["attention"] += 1
        elif "shared" in key:
            counts["shared_expert"] += 1
        elif "gate" in key or "router" in key:
            counts["router"] += 1
        elif "norm" in key:
            counts["norm"] += 1
        else:
            counts["other"] += 1

print()
print("Tensor classes")
print("==============")

for k, v in counts.items():
    print(f"{k:20} {v}")

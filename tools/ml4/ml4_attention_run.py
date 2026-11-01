import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


ROOT = Path("/workspace/ml4")
META = ROOT / "metadata"
WEIGHTS = ROOT / "weights"
OUTROOT = ROOT / "attention-k4"

CONVERTER = ROOT / "ml4_attention_convert.py"

PINNED = {
    "consolidated-00053-of-00054.safetensors",
}

# ---------------------------------------------------------
# Load source index
# ---------------------------------------------------------

idx = next(META.rglob("*.safetensors.index.json"))
wm = json.loads(idx.read_text())["weight_map"]


# ---------------------------------------------------------
# Canonical expected output
# ---------------------------------------------------------

RAW_MAP = {
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


def expected_keys(layer):
    native = f"layers.{layer}.attention"
    canon = (
        f"language_model.model.layers."
        f"{layer}.self_attn"
    )

    keys = set()

    for src, dst in (
        ("wq", "q_proj"),
        ("wo", "o_proj"),
        ("indexer.wq_b", "indexer.wq_b"),
    ):
        if f"{native}.{src}.weight" in wm:
            for field in ("suh", "svh", "trellis"):
                keys.add(
                    f"{canon}.{dst}.{field}"
                )

    for src, dst in RAW_MAP.items():
        if f"{native}.{src}" in wm:
            keys.add(f"{canon}.{dst}")

    return keys


def outfile(layer):
    return (
        OUTROOT
        / f"layer-{layer:03d}"
        / "attention.safetensors"
    )


def validate(layer, verbose=True):
    p = outfile(layer)

    if not p.is_file():
        if verbose:
            print(
                f"layer {layer:02d}: "
                "output ontbreekt"
            )
        return False

    try:
        with safe_open(
            p,
            framework="pt",
            device="cpu",
        ) as f:
            actual = set(f.keys())
    except Exception as exc:
        if verbose:
            print(
                f"layer {layer:02d}: "
                f"ongeldige safetensors: {exc}"
            )
        return False

    expected = expected_keys(layer)

    if actual != expected:
        if verbose:
            print(
                f"layer {layer:02d}: "
                "key mismatch"
            )

            missing = sorted(expected - actual)
            extra = sorted(actual - expected)

            if missing:
                print("  missing:")
                for k in missing:
                    print("   ", k)

            if extra:
                print("  extra:")
                for k in extra:
                    print("   ", k)

        return False

    if verbose:
        mib = p.stat().st_size / 1024**2
        print(
            f"layer {layer:02d}: VALID "
            f"({len(actual)} keys, "
            f"{mib:.2f} MiB)"
        )

    return True


# ---------------------------------------------------------
# Heavy attention shard mapping
# ---------------------------------------------------------

layers_by_shard = defaultdict(set)

for key, shard in wm.items():

    if shard in PINNED:
        continue

    if not key.startswith("layers."):
        continue

    if ".attention." not in key:
        continue

    try:
        layer = int(key.split(".")[1])
    except Exception:
        continue

    layers_by_shard[shard].add(layer)


def shard_number(name):
    m = re.search(
        r"consolidated-(\d+)-of-",
        name,
    )
    return int(m.group(1))


# Local heavy shards first, to free disk immediately.
shards = sorted(
    layers_by_shard,
    key=lambda s: (
        not (WEIGHTS / s).is_file(),
        shard_number(s),
    ),
)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

print("=== ML4 ATTENTION STREAM RUNNER ===")
print("converter :", CONVERTER)
print("shards    :", len(shards))
print("pinned    :", sorted(PINNED))
print()


for shard in shards:

    layers = sorted(layers_by_shard[shard])
    path = WEIGHTS / shard

    print()
    print("=" * 72)
    print("SHARD :", shard)
    print("LAYERS:", layers)
    print(
        "LOCAL :",
        "YES" if path.is_file() else "no",
    )
    print("=" * 72)

    # Convert/reuse every layer that depends on this heavy shard.
    for layer in layers:

        if validate(layer):
            print(
                f"layer {layer:02d}: "
                "resume skip"
            )
            continue

        print()
        print(
            f">>> converting layer {layer:02d}"
        )

        subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--layer",
                str(layer),
            ],
            check=True,
        )

        if not validate(layer):
            raise SystemExit(
                f"VALIDATION FAILED layer {layer}"
            )

    # Absolutely verify every layer again before deleting source.
    failed = [
        layer
        for layer in layers
        if not validate(layer, verbose=False)
    ]

    if failed:
        raise SystemExit(
            f"Shard {shard} NIET verwijderd; "
            f"ongeldige layers: {failed}"
        )

    if path.is_file():
        gib = path.stat().st_size / 1024**3

        print()
        print(
            f"DELETE validated source shard: "
            f"{shard} ({gib:.2f} GiB)"
        )

        path.unlink()

    else:
        print(
            "source shard niet lokaal na afloop "
            "(onverwacht maar outputs zijn geldig)"
        )

    print(
        "SHARD COMPLETE:",
        shard,
    )


# ---------------------------------------------------------
# Final audit
# ---------------------------------------------------------

print()
print("=" * 72)
print("FINAL 72-LAYER AUDIT")
print("=" * 72)

bad = []

for layer in range(72):
    if not validate(layer):
        bad.append(layer)

if bad:
    raise SystemExit(
        f"Attention run incompleet: {bad}"
    )

print()
print("ALL 72 ATTENTION LAYERS VALID")
print("Pinned shard 53 retained:", (WEIGHTS / "consolidated-00053-of-00054.safetensors").is_file())

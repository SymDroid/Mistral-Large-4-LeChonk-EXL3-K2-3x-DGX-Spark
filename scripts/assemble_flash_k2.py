#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


ROOT = Path("/workspace/mistral-large-3-flash")
K2_ROOT = ROOT / "quant/bootstrap_k2"
TEST_ROOT = ROOT / "quant/assembly_test"

PATTERN = re.compile(
    r"expert(?P<expert>\d+)-(?P<proj>w[123])-k2\.safetensors$"
)


def assemble_layer(layer: int):
    src_dir = K2_ROOT / f"layer{layer:02d}"

    if not src_dir.exists():
        raise RuntimeError(f"Missing source directory: {src_dir}")

    files = sorted(src_dir.glob("expert*-k2.safetensors"))

    print(f"Layer       : {layer}")
    print(f"Source dir  : {src_dir}")
    print(f"Source files: {len(files)}")

    if len(files) != 128 * 3:
        raise RuntimeError(
            f"Expected 384 K2 files for layer {layer}, found {len(files)}"
        )

    TEST_ROOT.mkdir(parents=True, exist_ok=True)

    dst = TEST_ROOT / f"layer{layer:02d}-experts-k2.safetensors"
    tmp = dst.with_suffix(".safetensors.tmp")

    out = {}

    seen = set()

    for i, src in enumerate(files, 1):
        m = PATTERN.match(src.name)

        if not m:
            raise RuntimeError(f"Unexpected filename: {src.name}")

        expert = int(m.group("expert"))
        proj = m.group("proj")

        pair = (expert, proj)

        if pair in seen:
            raise RuntimeError(f"Duplicate expert/projection: {pair}")

        seen.add(pair)

        tensors = load_file(str(src), device="cpu")

        expected = {"trellis", "suh", "svh"}

        if set(tensors.keys()) != expected:
            raise RuntimeError(
                f"{src}: expected {expected}, got {set(tensors.keys())}"
            )

        prefix = f"layers.{layer}.experts.{expert}.{proj}"

        for suffix in ("trellis", "suh", "svh"):
            key = f"{prefix}.{suffix}"

            if key in out:
                raise RuntimeError(f"Duplicate output key: {key}")

            out[key] = tensors[suffix]

        if i % 32 == 0 or i == len(files):
            print(
                f"\rProcessed {i:3d}/{len(files)} matrices "
                f"| output tensors {len(out):4d}",
                end="",
                flush=True,
            )

    print()

    expected_pairs = {
        (expert, proj)
        for expert in range(128)
        for proj in ("w1", "w2", "w3")
    }

    missing = expected_pairs - seen
    extra = seen - expected_pairs

    if missing:
        raise RuntimeError(f"Missing expert/projection pairs: {sorted(missing)[:20]}")

    if extra:
        raise RuntimeError(f"Unexpected expert/projection pairs: {sorted(extra)[:20]}")

    if len(out) != 128 * 3 * 3:
        raise RuntimeError(
            f"Expected 1152 output tensors, got {len(out)}"
        )

    print(f"Writing      : {dst}")

    save_file(out, str(tmp))

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("Temporary output file was not written correctly")

    tmp.replace(dst)

    size_gib = dst.stat().st_size / 1024**3

    print(f"Size         : {size_gib:.3f} GiB")
    print("Assembly     : OK")

    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    assemble_layer(args.layer)


if __name__ == "__main__":
    main()

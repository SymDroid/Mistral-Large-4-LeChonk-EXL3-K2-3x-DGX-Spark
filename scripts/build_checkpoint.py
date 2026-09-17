#!/usr/bin/env python3

import argparse
import gc
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from audit_mapping import (
    ROOT,
    INDEX,
    LM,
    map_native_key,
    exl3_prefix,
)

PASSTHROUGH = ROOT / "quant/bootstrap_passthrough"
K2_ROOT = ROOT / "quant/bootstrap_k2"
DEFAULT_OUTPUT = ROOT / "quant/mistral-large-3-flash-k2"

K2_RE = re.compile(
    r"layer(\d+)/expert(\d+)-(w[123])-k2\.safetensors$"
)

EXL3_COMPONENTS = ("trellis", "suh", "svh")


def parse_size(value: str) -> int:
    s = value.strip().upper()

    units = {
        "KIB": 1024,
        "MIB": 1024 ** 2,
        "GIB": 1024 ** 3,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
    }

    for suffix, mul in units.items():
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mul)

    return int(s)


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class ShardWriter:
    def __init__(self, output_dir: Path, max_bytes: int):
        self.output_dir = output_dir
        self.max_bytes = max_bytes

        self.pending = {}
        self.pending_bytes = 0

        self.shards = []
        self.weight_map = {}
        self.total_size = 0

        self.all_keys = set()

    def add(self, key: str, tensor: torch.Tensor):
        if key in self.all_keys:
            raise RuntimeError(f"duplicate destination key: {key}")

        tensor = tensor.detach().cpu().contiguous()
        size = tensor_nbytes(tensor)

        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()

        self.pending[key] = tensor
        self.pending_bytes += size
        self.total_size += size
        self.all_keys.add(key)

    def flush(self):
        if not self.pending:
            return

        shard_num = len(self.shards) + 1
        provisional = f"model-{shard_num:05d}.safetensors"

        tmp = self.output_dir / f".{provisional}.tmp"
        final = self.output_dir / provisional

        print(
            f"[WRITE] {provisional}: "
            f"{len(self.pending):,} tensors, "
            f"{self.pending_bytes / 1024**3:.3f} GiB"
        )

        save_file(self.pending, str(tmp))

        # Immediate physical validation before atomic rename
        with safe_open(tmp, framework="pt", device="cpu") as f:
            written_keys = set(f.keys())

        expected_keys = set(self.pending)

        if written_keys != expected_keys:
            raise RuntimeError(
                f"verification failed for {tmp}: "
                f"expected {len(expected_keys)}, "
                f"found {len(written_keys)}"
            )

        os.replace(tmp, final)

        for key in self.pending:
            self.weight_map[key] = provisional

        self.shards.append(provisional)
        self.pending = {}
        self.pending_bytes = 0

        gc.collect()

    def finalize(self):
        self.flush()

        count = len(self.shards)

        # Rename model-00001.safetensors ->
        # model-00001-of-00045.safetensors
        rename_map = {}

        for i, old_name in enumerate(self.shards, start=1):
            new_name = f"model-{i:05d}-of-{count:05d}.safetensors"

            old = self.output_dir / old_name
            new = self.output_dir / new_name

            os.replace(old, new)
            rename_map[old_name] = new_name

        self.shards = [
            rename_map[name]
            for name in self.shards
        ]

        self.weight_map = {
            key: rename_map[name]
            for key, name in self.weight_map.items()
        }

        index = {
            "metadata": {
                "total_size": self.total_size,
            },
            "weight_map": dict(sorted(self.weight_map.items())),
        }

        tmp_index = self.output_dir / ".model.safetensors.index.json.tmp"
        final_index = self.output_dir / "model.safetensors.index.json"

        with tmp_index.open("w") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        os.replace(tmp_index, final_index)

        return index


def iter_passthrough():
    files = sorted(PASSTHROUGH.glob("*.safetensors"))

    if len(files) != 68:
        raise RuntimeError(
            f"expected 68 passthrough shards, found {len(files)}"
        )

    for path in files:
        print(f"[PASS] {path.name}")

        with safe_open(path, framework="pt", device="cpu") as f:
            for src in f.keys():
                dst = map_native_key(src)

                if dst is None:
                    raise RuntimeError(
                        f"unmapped passthrough tensor: {src}"
                    )

                if dst == "__ROUTED_REPLACED__":
                    raise RuntimeError(
                        f"routed source leaked into passthrough: {src}"
                    )

                yield dst, f.get_tensor(src)


def iter_k2():
    files = sorted(
        K2_ROOT.glob("layer*/expert*-w*-k2.safetensors")
    )

    if len(files) != 22272:
        raise RuntimeError(
            f"expected 22,272 K2 files, found {len(files):,}"
        )

    for n, path in enumerate(files, start=1):
        rel = path.relative_to(K2_ROOT).as_posix()
        m = K2_RE.fullmatch(rel)

        if not m:
            raise RuntimeError(f"bad K2 filename: {rel}")

        layer = int(m.group(1))
        expert = int(m.group(2))
        projection = m.group(3)

        prefix = exl3_prefix(layer, expert, projection)

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            if keys != set(EXL3_COMPONENTS):
                raise RuntimeError(
                    f"{rel}: expected {EXL3_COMPONENTS}, "
                    f"found {sorted(keys)}"
                )

            for component in EXL3_COMPONENTS:
                yield (
                    f"{prefix}.{component}",
                    f.get_tensor(component),
                )

        if n % 500 == 0 or n == len(files):
            print(
                f"[K2] {n:,}/{len(files):,} "
                f"({n / len(files) * 100:.2f}%)"
            )


def validate_final(output_dir: Path, index: dict):
    print()
    print("===== FINAL PHYSICAL AUDIT =====")

    weight_map = index["weight_map"]
    expected_keys = set(weight_map)

    print(f"Index tensors : {len(expected_keys):,}")
    print(
        f"Index size    : "
        f"{index['metadata']['total_size'] / 1024**3:.3f} GiB"
    )

    shard_names = sorted(set(weight_map.values()))
    print(f"Shards        : {len(shard_names):,}")

    found_keys = set()
    physical_size = 0

    for i, name in enumerate(shard_names, start=1):
        path = output_dir / name

        if not path.exists():
            raise RuntimeError(f"missing output shard: {name}")

        physical_size += path.stat().st_size

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            for key in keys:
                if key in found_keys:
                    raise RuntimeError(
                        f"duplicate physical tensor: {key}"
                    )
                found_keys.add(key)

        if i % 10 == 0 or i == len(shard_names):
            print(
                f"[VERIFY] {i}/{len(shard_names)} shards"
            )

    missing = expected_keys - found_keys
    unexpected = found_keys - expected_keys

    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors missing from physical shards"
        )

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected physical tensors"
        )

    # Routed EXL3 final topology
    routed_re = re.compile(
        rf"^{re.escape(LM)}\.layers\.(\d+)\.mlp\.experts\."
        r"(\d+)\.(gate_proj|up_proj|down_proj)\."
        r"(trellis|suh|svh)$"
    )

    routed = {
        key
        for key in found_keys
        if routed_re.fullmatch(key)
    }

    expected_routed = 22272 * 3

    if len(routed) != expected_routed:
        raise RuntimeError(
            f"expected {expected_routed:,} EXL3 tensors, "
            f"found {len(routed):,}"
        )

    native_leaks = [
        key for key in found_keys
        if key.startswith("layers.")
        or key.startswith("vision_encoder.")
        or key.startswith("vision_language_adapter.")
        or key.startswith("patch_merger.")
        or key.startswith("pre_mm_projector_norm.")
        or key in {"tok_embeddings.weight", "output.weight", "norm.weight"}
    ]

    if native_leaks:
        raise RuntimeError(
            f"native namespace leaked into final model: "
            f"{native_leaks[:20]}"
        )

    tmp_files = list(output_dir.glob("*.tmp"))
    tmp_files += list(output_dir.glob(".*.tmp"))

    if tmp_files:
        raise RuntimeError(
            f"temporary files remain: {tmp_files}"
        )

    print(f"Physical keys : {len(found_keys):,}")
    print(f"EXL3 tensors  : {len(routed):,}")
    print(
        f"Disk bytes    : "
        f"{physical_size / 1024**3:.3f} GiB"
    )
    print("Native leaks  : 0")
    print("Missing       : 0")
    print("Unexpected    : 0")
    print()
    print("PHYSICAL CHECKPOINT: PASS")


def copy_metadata(output_dir: Path):
    candidates = [
        ROOT / "upstream-meta/README.md",
        ROOT / "upstream-meta/chat_template.jinja",
        ROOT / "upstream-meta/SYSTEM_PROMPT.txt",
        ROOT / "upstream-meta/params.json",
    ]

    for src in candidates:
        if src.exists():
            shutil.copy2(src, output_dir / src.name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-shard-size",
        default="4GiB",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output directory before assembly",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    output_dir = args.output
    max_bytes = parse_size(args.max_shard_size)

    if args.verify_only:
        index_path = output_dir / "model.safetensors.index.json"

        with index_path.open() as f:
            index = json.load(f)

        validate_final(output_dir, index)
        return

    if output_dir.exists():
        existing = list(output_dir.iterdir())

        if existing and not args.force:
            raise RuntimeError(
                f"{output_dir} is not empty. "
                f"Use --force to rebuild."
            )

        if args.force:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(
        output_dir=output_dir,
        max_bytes=max_bytes,
    )

    print("===== MISTRAL LARGE 3 FLASH K2 ASSEMBLY =====")
    print(f"Output         : {output_dir}")
    print(
        f"Max shard size : "
        f"{max_bytes / 1024**3:.3f} GiB"
    )
    print()

    print("===== PASSTHROUGH =====")

    passthrough_count = 0

    for key, tensor in iter_passthrough():
        writer.add(key, tensor)
        passthrough_count += 1

    if passthrough_count != 1597:
        raise RuntimeError(
            f"expected 1,597 passthrough tensors, "
            f"found {passthrough_count:,}"
        )

    print()
    print(
        f"Passthrough complete: "
        f"{passthrough_count:,} tensors"
    )

    print()
    print("===== EXL3 K2 =====")

    k2_tensor_count = 0

    for key, tensor in iter_k2():
        writer.add(key, tensor)
        k2_tensor_count += 1

    expected_k2_tensors = 22272 * 3

    if k2_tensor_count != expected_k2_tensors:
        raise RuntimeError(
            f"expected {expected_k2_tensors:,} K2 tensors, "
            f"found {k2_tensor_count:,}"
        )

    print()
    print(
        f"K2 complete: "
        f"{k2_tensor_count:,} tensors"
    )

    index = writer.finalize()

    copy_metadata(output_dir)

    print()
    print("===== ASSEMBLY COMPLETE =====")
    print(
        f"Total tensors : "
        f"{len(index['weight_map']):,}"
    )
    print(
        f"Tensor bytes  : "
        f"{index['metadata']['total_size'] / 1024**3:.3f} GiB"
    )
    print(
        f"Shards        : "
        f"{len(set(index['weight_map'].values())):,}"
    )

    validate_final(output_dir, index)


if __name__ == "__main__":
    main()

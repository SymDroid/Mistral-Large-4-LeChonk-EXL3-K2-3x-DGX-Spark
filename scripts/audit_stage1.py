#!/usr/bin/env python3

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]

K2_DIR = ROOT / "quant" / "bootstrap_k2"
PASSTHROUGH_DIR = ROOT / "quant" / "bootstrap_passthrough"

TENSOR_MANIFEST = K2_DIR / "manifest.csv"
SHARD_MANIFEST = K2_DIR / "shard_manifest.csv"

REPORT_JSON = K2_DIR / "STAGE1_AUDIT.json"

EXPECTED_LAYERS = 58
MOE_LAYER_START = 3
MOE_LAYER_END = 60
MOE_LAYERS = range(MOE_LAYER_START, MOE_LAYER_END + 1)
EXPECTED_EXPERTS_PER_LAYER = 128
PROJECTIONS = ("w1", "w2", "w3")

EXPECTED_ROUTED = (
    EXPECTED_LAYERS
    * EXPECTED_EXPERTS_PER_LAYER
    * len(PROJECTIONS)
)

FILE_RE = re.compile(
    r"^expert(?P<expert>\d+)-(?P<projection>w[123])-k(?P<k>\d+)\.safetensors$"
)

LAYER_RE = re.compile(r"^layer(?P<layer>\d+)$")


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.3f} {unit}"
        n /= 1024.0


def fail(msg):
    print(f"[FAIL] {msg}")


def ok(msg):
    print(f"[ OK ] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def read_csv(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def inspect_safetensor(path):
    """
    Actually opens the safetensors header and enumerates every tensor.

    This catches malformed/truncated safetensors metadata without loading
    the full tensor payload into RAM/VRAM.
    """
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        info = {}
        for key in keys:
            sl = f.get_slice(key)
            info[key] = {
                "shape": list(sl.get_shape()),
            }

        metadata = f.metadata()

    return {
        "keys": keys,
        "tensor_count": len(keys),
        "tensors": info,
        "metadata": metadata,
    }


def main():
    print("=" * 72)
    print("Mistral Large 3 Flash K2 - Stage 1 Integrity Audit")
    print("=" * 72)
    print(f"Root               : {ROOT}")
    print(f"K2 directory       : {K2_DIR}")
    print(f"Expected routed    : {EXPECTED_ROUTED}")
    print()

    errors = []
    warnings = []

    # ------------------------------------------------------------
    # 1. Discover all K2 output files
    # ------------------------------------------------------------

    files = sorted(K2_DIR.glob("layer*/expert*-k2.safetensors"))

    print(f"Discovered K2 files: {len(files)}")

    if len(files) == EXPECTED_ROUTED:
        ok(f"Found exactly {EXPECTED_ROUTED} routed K2 files")
    else:
        msg = (
            f"Expected {EXPECTED_ROUTED} routed K2 files, "
            f"found {len(files)}"
        )
        fail(msg)
        errors.append(msg)

    # ------------------------------------------------------------
    # 2. Validate filename topology
    # ------------------------------------------------------------

    seen = Counter()
    layer_counts = Counter()
    expert_proj = defaultdict(set)

    malformed = []

    for path in files:
        lm = LAYER_RE.match(path.parent.name)
        fm = FILE_RE.match(path.name)

        if not lm or not fm:
            malformed.append(str(path))
            continue

        layer = int(lm.group("layer"))
        expert = int(fm.group("expert"))
        projection = fm.group("projection")
        k = int(fm.group("k"))

        key = (layer, expert, projection)

        seen[key] += 1
        layer_counts[layer] += 1
        expert_proj[(layer, expert)].add(projection)

        if k != 2:
            msg = f"Unexpected K={k}: {path}"
            errors.append(msg)

    if malformed:
        msg = f"{len(malformed)} malformed filenames"
        fail(msg)
        errors.append(msg)
    else:
        ok("All filenames match expected layer/expert/projection topology")

    duplicates = [k for k, count in seen.items() if count != 1]

    expected_keys = {
        (layer, expert, projection)
        for layer in MOE_LAYERS
        for expert in range(EXPECTED_EXPERTS_PER_LAYER)
        for projection in PROJECTIONS
    }

    actual_keys = set(seen)

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)

    if not duplicates:
        ok("No duplicate routed tensor coordinates")
    else:
        msg = f"{len(duplicates)} duplicate coordinates"
        fail(msg)
        errors.append(msg)

    if not missing:
        ok("No missing routed tensor coordinates")
    else:
        msg = f"{len(missing)} routed tensor coordinates missing"
        fail(msg)
        errors.append(msg)

    if not unexpected:
        ok("No unexpected routed tensor coordinates")
    else:
        msg = f"{len(unexpected)} unexpected routed coordinates"
        fail(msg)
        errors.append(msg)

    # Layer topology
    expected_per_layer = EXPECTED_EXPERTS_PER_LAYER * len(PROJECTIONS)

    bad_layers = {
        layer: count
        for layer, count in layer_counts.items()
        if count != expected_per_layer
    }

    if (
        len(layer_counts) == EXPECTED_LAYERS
        and not bad_layers
        and set(layer_counts) == set(MOE_LAYERS)
    ):
        ok(
            f"All {EXPECTED_LAYERS} layers contain "
            f"{expected_per_layer} routed tensors"
        )
    else:
        msg = "Layer topology is incomplete or inconsistent"
        fail(msg)
        errors.append(msg)

    bad_experts = []

    for layer in MOE_LAYERS:
        for expert in range(EXPECTED_EXPERTS_PER_LAYER):
            got = expert_proj.get((layer, expert), set())
            if got != set(PROJECTIONS):
                bad_experts.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "found": sorted(got),
                    }
                )

    if not bad_experts:
        ok("Every expert contains exactly w1, w2 and w3")
    else:
        msg = f"{len(bad_experts)} experts have incomplete projections"
        fail(msg)
        errors.append(msg)

    # ------------------------------------------------------------
    # 3. Actually open every safetensors file
    # ------------------------------------------------------------

    print()
    print("Opening all routed safetensors...")
    print("This may take a little while.")

    corrupt = []
    empty_files = []
    empty_tensor_sets = []
    tensor_key_histogram = Counter()

    total_bytes = 0

    for i, path in enumerate(files, 1):
        try:
            size = path.stat().st_size
            total_bytes += size

            if size <= 0:
                empty_files.append(str(path))
                continue

            info = inspect_safetensor(path)

            if info["tensor_count"] == 0:
                empty_tensor_sets.append(str(path))

            tensor_key_histogram.update(info["keys"])

        except Exception as exc:
            corrupt.append(
                {
                    "file": str(path),
                    "error": repr(exc),
                }
            )

        if i % 1000 == 0 or i == len(files):
            print(
                f"  checked {i:5d} / {len(files)}",
                flush=True,
            )

    if not corrupt:
        ok(f"All {len(files)} routed safetensors open successfully")
    else:
        msg = f"{len(corrupt)} routed safetensors failed to open"
        fail(msg)
        errors.append(msg)

    if not empty_files:
        ok("No zero-byte routed safetensors")
    else:
        msg = f"{len(empty_files)} zero-byte routed files"
        fail(msg)
        errors.append(msg)

    if not empty_tensor_sets:
        ok("Every routed safetensors file contains tensor data")
    else:
        msg = f"{len(empty_tensor_sets)} safetensors files contain no tensors"
        fail(msg)
        errors.append(msg)

    print()
    print(f"Exact routed bytes  : {total_bytes:,}")
    print(f"Routed size         : {human_bytes(total_bytes)}")

    # ------------------------------------------------------------
    # 4. Tensor manifest
    # ------------------------------------------------------------

    manifest_rows = read_csv(TENSOR_MANIFEST)

    print()
    print(f"Tensor manifest rows: {len(manifest_rows)}")

    manifest_done = [
        row for row in manifest_rows
        if row.get("status", "").strip().lower() == "done"
    ]

    if len(manifest_done) == EXPECTED_ROUTED:
        ok(f"Tensor manifest contains {EXPECTED_ROUTED} done rows")
    else:
        msg = (
            f"Tensor manifest has {len(manifest_done)} done rows; "
            f"expected {EXPECTED_ROUTED}"
        )
        warn(msg)
        warnings.append(msg)

    # Effective BPW summary
    bpw_values = []

    for row in manifest_done:
        try:
            bpw_values.append(float(row["effective_bpw"]))
        except (KeyError, TypeError, ValueError):
            pass

    bpw_summary = None

    if bpw_values:
        bpw_summary = {
            "count": len(bpw_values),
            "min": min(bpw_values),
            "max": max(bpw_values),
            "mean": sum(bpw_values) / len(bpw_values),
        }

        print(
            "Effective BPW       : "
            f"mean={bpw_summary['mean']:.6f}, "
            f"min={bpw_summary['min']:.6f}, "
            f"max={bpw_summary['max']:.6f}"
        )

    # ------------------------------------------------------------
    # 5. Shard manifest
    # ------------------------------------------------------------

    shard_rows = read_csv(SHARD_MANIFEST)

    shard_latest = {}

    # Upsert manifests should normally already contain one row/shard,
    # but latest-by-name makes audit robust if historical duplicates exist.
    for row in shard_rows:
        shard = row.get("shard", "").strip()
        if shard:
            shard_latest[shard] = row

    done_shards = {
        shard: row
        for shard, row in shard_latest.items()
        if row.get("status", "").strip().lower() == "done"
    }

    print()
    print(f"Shard manifest      : {len(done_shards)} done / 272 expected")

    if len(done_shards) == 272:
        ok("All 272 upstream shards are marked done")
    else:
        msg = f"Only {len(done_shards)} / 272 upstream shards are done"
        fail(msg)
        errors.append(msg)

    expected_shard_names = {
        f"consolidated-{i:05d}-of-00272.safetensors"
        for i in range(1, 273)
    }

    missing_shards = sorted(expected_shard_names - set(done_shards))
    unexpected_shards = sorted(set(done_shards) - expected_shard_names)

    if not missing_shards:
        ok("No upstream shards missing from done manifest")
    else:
        msg = f"{len(missing_shards)} expected shards missing"
        fail(msg)
        errors.append(msg)

    if unexpected_shards:
        msg = f"{len(unexpected_shards)} unexpected shard names"
        warn(msg)
        warnings.append(msg)

    # ------------------------------------------------------------
    # 6. Passthrough validation
    # ------------------------------------------------------------

    passthrough_expected = {}

    for shard, row in done_shards.items():
        try:
            count = int(row.get("passthrough_tensors", "0") or 0)
        except ValueError:
            count = -1

        if count > 0:
            passthrough_expected[shard] = {
                "count": count,
                "path": row.get("passthrough_file", "").strip(),
            }

    passthrough_bad = []
    passthrough_total_tensors = 0
    passthrough_total_bytes = 0

    print()
    print(
        f"Passthrough files expected: {len(passthrough_expected)}"
    )

    for shard, spec in sorted(passthrough_expected.items()):
        raw_path = spec["path"]

        if raw_path:
            path = Path(raw_path)

            # Manifest was written in container namespace.
            # Translate /workspace/... to host project root if necessary.
            if not path.exists():
                prefix = "/workspace/mistral-large-3-flash/"
                if raw_path.startswith(prefix):
                    path = ROOT / raw_path[len(prefix):]
        else:
            path = PASSTHROUGH_DIR / shard

        if not path.exists():
            passthrough_bad.append(
                {
                    "shard": shard,
                    "error": "missing passthrough file",
                    "path": str(path),
                }
            )
            continue

        try:
            passthrough_total_bytes += path.stat().st_size
            info = inspect_safetensor(path)
            actual_count = info["tensor_count"]

            passthrough_total_tensors += actual_count

            if actual_count != spec["count"]:
                passthrough_bad.append(
                    {
                        "shard": shard,
                        "error": "tensor count mismatch",
                        "expected": spec["count"],
                        "actual": actual_count,
                        "path": str(path),
                    }
                )

        except Exception as exc:
            passthrough_bad.append(
                {
                    "shard": shard,
                    "error": repr(exc),
                    "path": str(path),
                }
            )

    if not passthrough_bad:
        ok("All passthrough safetensors validate successfully")
    else:
        msg = f"{len(passthrough_bad)} passthrough files failed validation"
        fail(msg)
        errors.append(msg)

    print(
        f"Passthrough tensors : {passthrough_total_tensors:,}"
    )
    print(
        f"Passthrough size    : {human_bytes(passthrough_total_bytes)}"
    )

    # ------------------------------------------------------------
    # 7. Temporary/incomplete files
    # ------------------------------------------------------------

    tmp_files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and (
            p.name.endswith(".tmp")
            or p.name.endswith(".partial")
            or p.name.endswith(".incomplete")
        )
        and (
            K2_DIR in p.parents
            or PASSTHROUGH_DIR in p.parents
        )
    )

    if not tmp_files:
        ok("No temporary/incomplete Stage 1 files found")
    else:
        msg = f"{len(tmp_files)} temporary/incomplete files remain"
        warn(msg)
        warnings.append(msg)

    # ------------------------------------------------------------
    # 8. Final report
    # ------------------------------------------------------------

    status = "PASS" if not errors else "FAIL"

    report = {
        "stage": "bootstrap_k2",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "expected": {
            "layers": EXPECTED_LAYERS,
            "experts_per_layer": EXPECTED_EXPERTS_PER_LAYER,
            "projections": list(PROJECTIONS),
            "routed_tensor_files": EXPECTED_ROUTED,
            "upstream_shards": 272,
            "K": 2,
        },
        "actual": {
            "routed_tensor_files": len(files),
            "routed_total_bytes": total_bytes,
            "routed_total_human": human_bytes(total_bytes),
            "unique_coordinates": len(actual_keys),
            "missing_coordinates": len(missing),
            "duplicate_coordinates": len(duplicates),
            "unexpected_coordinates": len(unexpected),
            "corrupt_routed_files": len(corrupt),
            "empty_routed_files": len(empty_files),
            "done_shards": len(done_shards),
            "passthrough_files_expected": len(passthrough_expected),
            "passthrough_tensors": passthrough_total_tensors,
            "passthrough_total_bytes": passthrough_total_bytes,
            "passthrough_total_human": human_bytes(
                passthrough_total_bytes
            ),
            "temporary_files": len(tmp_files),
        },
        "effective_bpw": bpw_summary,
        "tensor_key_histogram": dict(
            sorted(tensor_key_histogram.items())
        ),
        "errors": errors,
        "warnings": warnings,
        "details": {
            "missing_coordinates": [
                list(x) for x in missing[:100]
            ],
            "duplicates": [
                list(x) for x in duplicates[:100]
            ],
            "unexpected_coordinates": [
                list(x) for x in unexpected[:100]
            ],
            "corrupt_files": corrupt[:100],
            "bad_experts": bad_experts[:100],
            "passthrough_bad": passthrough_bad[:100],
            "temporary_files": [
                str(x) for x in tmp_files[:100]
            ],
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print()
    print("=" * 72)
    print(f"STAGE 1 STATUS      : {status}")
    print(f"Audit report        : {REPORT_JSON}")
    print("=" * 72)

    if warnings:
        print(f"Warnings            : {len(warnings)}")

    if errors:
        print(f"Errors              : {len(errors)}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

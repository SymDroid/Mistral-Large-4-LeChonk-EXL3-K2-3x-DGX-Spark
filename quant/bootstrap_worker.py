#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(".").resolve()

INDEX = (
    ROOT /
    "upstream-meta/consolidated.safetensors.index.json"
)

SOURCE_DIR = ROOT / "upstream-stream"

PASSTHROUGH_DIR = (
    ROOT /
    "quant/bootstrap_passthrough"
)

SHARD_MANIFEST = (
    ROOT /
    "quant/bootstrap_k2/shard_manifest.csv"
)

REPO_ID = (
    "mistralai/"
    "Mistral-Large-3-675B-Instruct-2512"
)


#
# Load our proven tensor converter
#

sys.path.insert(0, str(ROOT))

from quant import bootstrap_k2 as bk


PROJECTIONS = (
    "w1",
    "w2",
    "w3",
)


def load_weight_map():

    with open(INDEX) as f:
        data = json.load(f)

    return data["weight_map"]


def shard_number(name):

    # consolidated-00020-of-00272.safetensors

    try:
        return int(
            name.split("-")[1]
        )

    except Exception:
        return 999999


def all_shards(weight_map):

    shards = sorted(
        set(weight_map.values()),
        key=shard_number,
    )

    return shards


def is_routed_expert_tensor(key):

    if ".experts." not in key:
        return False

    parts = key.split(".")

    if len(parts) < 6:
        return False

    proj = parts[4]

    if proj not in PROJECTIONS:
        return False

    if key.endswith(".weight"):
        return True

    if key.endswith(".weight_scale"):
        return True

    return False


def tensors_for_shard(
    shard_name,
    weight_map,
):

    return sorted(
        key
        for key, shard in weight_map.items()
        if shard == shard_name
    )


def passthrough_keys(
    shard_name,
    weight_map,
):

    keys = tensors_for_shard(
        shard_name,
        weight_map,
    )

    return [
        key
        for key in keys
        if not is_routed_expert_tensor(key)
    ]


def find_existing_source(
    shard_name,
):

    matches = list(
        ROOT.rglob(shard_name)
    )

    #
    # Never consider our preserved passthrough
    # file to be an upstream source shard.
    #

    matches = [
        p
        for p in matches
        if PASSTHROUGH_DIR
        not in p.parents
    ]

    if not matches:
        return None

    #
    # Prefer upstream-stream first.
    #

    for path in matches:

        if SOURCE_DIR in path.parents:
            return path

    return matches[0]


def ensure_source(
    shard_name,
):

    existing = find_existing_source(
        shard_name,
    )

    if existing is not None:

        print(
            f"SOURCE existing: {existing}"
        )

        return existing, False

    SOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"DOWNLOAD {shard_name}"
    )

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=shard_name,
        local_dir=str(SOURCE_DIR),
    )

    path = Path(path).resolve()

    print(
        f"SOURCE downloaded: {path}"
    )

    return path, True


def preserve_passthrough(
    source,
    shard_name,
    weight_map,
):

    keys = passthrough_keys(
        shard_name,
        weight_map,
    )

    PASSTHROUGH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        PASSTHROUGH_DIR /
        shard_name
    )

    #
    # No passthrough tensors in this shard.
    #

    if not keys:

        print(
            "PASSTHROUGH none"
        )

        return None, 0

    #
    # Already preserved.
    #

    if (
        outfile.exists()
        and outfile.stat().st_size > 0
    ):

        print(
            f"PASSTHROUGH existing: "
            f"{outfile}"
        )

        return outfile, len(keys)

    print()
    print(
        f"PASSTHROUGH preserving "
        f"{len(keys)} tensors"
    )

    tensors = {}

    with safe_open(
        source,
        framework="pt",
        device="cpu",
    ) as f:

        available = set(
            f.keys()
        )

        for key in keys:

            if key not in available:

                raise RuntimeError(
                    "Tensor listed in index "
                    "missing from shard: "
                    f"{key}"
                )

            tensors[key] = (
                f.get_tensor(key)
                .contiguous()
            )

    tmp = outfile.with_suffix(
        outfile.suffix + ".tmp"
    )

    save_file(
        tensors,
        tmp,
    )

    tmp.replace(outfile)

    total_bytes = sum(
        tensor.numel()
        * tensor.element_size()
        for tensor in tensors.values()
    )

    print(
        f"PASSTHROUGH saved: "
        f"{outfile}"
    )

    print(
        "PASSTHROUGH size : "
        f"{total_bytes / 2**20:.2f} MiB"
    )

    del tensors

    gc.collect()

    return outfile, len(keys)


def expected_output(
    layer,
    expert,
    proj,
    K,
):

    return (
        ROOT /
        "quant/bootstrap_k2" /
        f"layer{layer:02d}" /
        (
            f"expert{expert:03d}-"
            f"{proj}-k{K}.safetensors"
        )
    )


def verify_jobs(
    jobs,
    K,
):

    missing = []

    empty = []

    for layer, expert, proj in jobs:

        outfile = expected_output(
            layer,
            expert,
            proj,
            K,
        )

        if not outfile.exists():

            missing.append(
                str(outfile)
            )

            continue

        if outfile.stat().st_size == 0:

            empty.append(
                str(outfile)
            )

    if missing:

        print()
        print(
            "VERIFY FAILED: "
            f"{len(missing)} missing"
        )

        for path in missing[:20]:
            print(
                f"  MISSING {path}"
            )

        return False

    if empty:

        print()
        print(
            "VERIFY FAILED: "
            f"{len(empty)} empty"
        )

        for path in empty[:20]:
            print(
                f"  EMPTY {path}"
            )

        return False

    print(
        "VERIFY routed tensors: "
        f"{len(jobs)}/{len(jobs)}"
    )

    return True


def verify_passthrough(
    passthrough_file,
    expected_keys,
):

    if expected_keys == 0:

        return True

    if passthrough_file is None:

        return False

    if not passthrough_file.exists():

        return False

    if passthrough_file.stat().st_size == 0:

        return False

    with safe_open(
        passthrough_file,
        framework="pt",
        device="cpu",
    ) as f:

        actual = len(
            list(f.keys())
        )

    print(
        "VERIFY passthrough   : "
        f"{actual}/{expected_keys}"
    )

    return actual == expected_keys


def read_shard_manifest():

    if not SHARD_MANIFEST.exists():
        return {}

    rows = {}

    with open(
        SHARD_MANIFEST,
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows[
                row["shard"]
            ] = row

    return rows


def write_shard_status(
    shard_name,
    **values,
):

    fields = [
        "shard",
        "status",
        "routed_jobs",
        "passthrough_tensors",
        "source_deleted",
        "passthrough_file",
        "timestamp",
    ]

    rows = read_shard_manifest()

    previous = rows.get(
        shard_name,
        {},
    )

    row = {
        field:
        previous.get(field, "")
        for field in fields
    }

    row["shard"] = shard_name

    for key, value in values.items():

        if key in row:

            row[key] = str(value)

    row["timestamp"] = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    rows[shard_name] = row

    SHARD_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = SHARD_MANIFEST.with_suffix(
        ".csv.tmp"
    )

    with open(
        tmp,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for name in sorted(
            rows,
            key=shard_number,
        ):

            writer.writerow(
                rows[name]
            )

    tmp.replace(
        SHARD_MANIFEST
    )


def safe_delete_source(
    source,
):

    source = source.resolve()

    stream = SOURCE_DIR.resolve()

    #
    # Critical safeguard:
    #
    # only delete files that are actually
    # inside upstream-stream.
    #

    if stream not in source.parents:

        print(
            "SOURCE retained "
            "(not worker-downloaded): "
            f"{source}"
        )

        return False

    print(
        f"DELETE source: {source}"
    )

    source.unlink()

    return True


def process_shard(
    shard_name,
    weight_map,
    K,
    device,
    keep_source,
    dry_run,
):

    print()
    print(
        "#" * 78
    )

    print(
        f"SHARD {shard_name}"
    )

    print(
        "#" * 78
    )

    jobs = bk.jobs_for_shard(
        shard_name,
        weight_map,
    )

    pass_keys = passthrough_keys(
        shard_name,
        weight_map,
    )

    print(
        f"routed jobs         : "
        f"{len(jobs)}"
    )

    print(
        f"passthrough tensors : "
        f"{len(pass_keys)}"
    )

    if dry_run:

        for layer, expert, proj in jobs:

            print(
                f"JOB layer={layer:02d} "
                f"expert={expert:03d} "
                f"proj={proj}"
            )

        return

    write_shard_status(
        shard_name,
        status="processing",
        routed_jobs=len(jobs),
        passthrough_tensors=len(
            pass_keys
        ),
        source_deleted=False,
    )

    source, downloaded = (
        ensure_source(
            shard_name,
        )
    )

    #
    # Preserve all tensors that are not
    # routed w1/w2/w3 + scales.
    #

    passthrough_file, pass_count = (
        preserve_passthrough(
            source,
            shard_name,
            weight_map,
        )
    )

    #
    # Quantize routed expert tensors.
    #

    for (
        layer,
        expert,
        proj,
    ) in jobs:

        bk.quantize_tensor(
            layer,
            expert,
            proj,
            K,
            device,
            weight_map,
        )

    #
    # Verification gate.
    #

    routed_ok = verify_jobs(
        jobs,
        K,
    )

    passthrough_ok = (
        verify_passthrough(
            passthrough_file,
            pass_count,
        )
    )

    if not routed_ok:

        write_shard_status(
            shard_name,
            status="verify_failed",
            source_deleted=False,
        )

        raise RuntimeError(
            "Routed tensor verification "
            f"failed for {shard_name}"
        )

    if not passthrough_ok:

        write_shard_status(
            shard_name,
            status="verify_failed",
            source_deleted=False,
        )

        raise RuntimeError(
            "Passthrough verification "
            f"failed for {shard_name}"
        )

    #
    # Only now can the source be deleted.
    #

    source_deleted = False

    if keep_source:

        print(
            "SOURCE retained "
            "(--keep-source)"
        )

    else:

        source_deleted = (
            safe_delete_source(
                source
            )
        )

    write_shard_status(
        shard_name,
        status="done",
        routed_jobs=len(jobs),
        passthrough_tensors=pass_count,
        source_deleted=source_deleted,
        passthrough_file=(
            str(
                passthrough_file
            )
            if passthrough_file
            else ""
        ),
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"SHARD COMPLETE: "
        f"{shard_name}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Streaming Mistral Large 3 "
            "FP8 -> EXL3 K2 bootstrap worker"
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "First shard number"
        ),
    )

    parser.add_argument(
        "--end",
        type=int,
        default=272,
        help=(
            "Last shard number"
        ),
    )

    parser.add_argument(
        "--K",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--keep-source",
        action="store_true",
        help=(
            "Do not delete worker-downloaded "
            "source shards"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--include-done",
        action="store_true",
        help=(
            "Revisit shards already marked done"
        ),
    )

    args = parser.parse_args()

    weight_map = load_weight_map()

    shards = all_shards(
        weight_map
    )

    manifest = (
        read_shard_manifest()
    )

    selected = []

    for shard in shards:

        n = shard_number(
            shard
        )

        if n < args.start:
            continue

        if n > args.end:
            continue

        if (
            not args.include_done
            and manifest.get(
                shard,
                {},
            ).get(
                "status"
            ) == "done"
        ):
            continue

        selected.append(
            shard
        )

    print(
        "Mistral Large 3 Flash "
        "streaming bootstrap worker"
    )

    print(
        f"repo        : {REPO_ID}"
    )

    print(
        f"K           : {args.K}"
    )

    print(
        f"device      : {args.device}"
    )

    print(
        f"range       : "
        f"{args.start}..{args.end}"
    )

    print(
        f"shards      : "
        f"{len(selected)}"
    )

    print(
        f"keep source : "
        f"{args.keep_source}"
    )

    print(
        f"dry run     : "
        f"{args.dry_run}"
    )

    for shard in selected:

        process_shard(
            shard,
            weight_map,
            args.K,
            args.device,
            args.keep_source,
            args.dry_run,
        )

    print()
    print(
        "=" * 78
    )

    print(
        "STREAMING BOOTSTRAP "
        "PASS COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from exllamav3.modules.quant.exl3_lib import quantize_exl3


ROOT = Path(".")
INDEX = ROOT / "upstream-meta/consolidated.safetensors.index.json"
OUT_ROOT = ROOT / "quant/bootstrap_k2"
MANIFEST = OUT_ROOT / "manifest.csv"

PROJECTIONS = ("w1", "w2", "w3")


def load_index():
    with open(INDEX) as f:
        return json.load(f)["weight_map"]


def append_manifest(row):
    new = not MANIFEST.exists()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "layer",
        "expert",
        "projection",
        "K",
        "status",
        "source_shard",
        "output_file",
        "params",
        "packed_bytes",
        "effective_bpw",
        "proxy_error",
        "runtime_s",
    ]

    with open(MANIFEST, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if new:
            writer.writeheader()

        writer.writerow(row)


def already_done(outfile):
    return outfile.exists() and outfile.stat().st_size > 0


def find_local_shard(shard_name):
    matches = list(ROOT.rglob(shard_name))

    if not matches:
        raise FileNotFoundError(
            f"{shard_name} is required but not present locally"
        )

    return matches[0]


def dequant_fp8(qweight, scales):
    """
    Mistral Large 3 FP8 block dequantization.

    Input:
        qweight [out_features, in_features]
        scales  128x128 block scales

    Output:
        EXL3 row-major [in_features, out_features] float32
    """

    rows, cols = qweight.shape

    scale_full = (
        scales.float()
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
    )

    scale_full = scale_full[:rows, :cols]

    weight_native = qweight.float() * scale_full

    return weight_native.T.contiguous()


def make_fallback_H(k, device):
    return {
        "H": torch.empty(
            (k, k),
            device="meta",
        ),
        "first_key": "bootstrap",
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(
            2,
            dtype=torch.long,
            device=device,
        ),
        "device": torch.device(device),
    }


def jobs_for_shard(shard_name, weight_map):
    jobs = []

    for key, source in weight_map.items():

        if source != shard_name:
            continue

        if ".experts." not in key:
            continue

        if not key.endswith(".weight"):
            continue

        parts = key.split(".")

        # Expected:
        # layers.14.experts.100.w1.weight

        if len(parts) < 6:
            continue

        if parts[0] != "layers":
            continue

        if parts[2] != "experts":
            continue

        try:
            layer = int(parts[1])
            expert = int(parts[3])
        except ValueError:
            continue

        proj = parts[4]

        if proj not in PROJECTIONS:
            continue

        jobs.append(
            (layer, expert, proj)
        )

    return sorted(jobs)


def quantize_tensor(
    layer,
    expert,
    proj,
    K,
    device,
    weight_map,
):
    tensor = (
        f"layers.{layer}.experts."
        f"{expert}.{proj}.weight"
    )

    scale = f"{tensor}_scale"

    if tensor not in weight_map:
        raise KeyError(
            f"Missing tensor in index: {tensor}"
        )

    if scale not in weight_map:
        raise KeyError(
            f"Missing scale in index: {scale}"
        )

    shard_name = weight_map[tensor]

    if weight_map[scale] != shard_name:
        raise RuntimeError(
            "Weight and scale live in different shards: "
            f"{tensor}"
        )

    shard = find_local_shard(shard_name)

    outdir = (
        OUT_ROOT /
        f"layer{layer:02d}"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        f"expert{expert:03d}-{proj}-k{K}.safetensors"
    )

    if already_done(outfile):
        print(f"SKIP  {tensor}")
        return

    print("=" * 78)
    print(f"QUANT {tensor}")
    print(f"shard : {shard}")
    print(f"out   : {outfile}")

    with safe_open(
        shard,
        framework="pt",
        device="cpu",
    ) as f:

        qweight = f.get_tensor(tensor)
        scales = f.get_tensor(scale)

    print(
        f"FP8   : {tuple(qweight.shape)} "
        f"{qweight.dtype}"
    )

    print(
        f"scale : {tuple(scales.shape)} "
        f"{scales.dtype}"
    )

    weight = dequant_fp8(
        qweight,
        scales,
    )

    del qweight
    del scales

    gc.collect()

    params = weight.numel()

    print(
        f"EXL3  : {tuple(weight.shape)} "
        f"{weight.dtype} "
        f"params={params:,}"
    )

    H_data = make_fallback_H(
        weight.shape[0],
        device,
    )

    quant_args = {
        "K": K,
        "devices": [device],
        "seed": 42,
        "apply_out_scales": False,
        "buf_size_k": 128,
        "sigma_reg": 0.025,
    }

    torch.cuda.synchronize()

    start = time.perf_counter()

    weight_q, proxy_error, out_tensors = quantize_exl3(
        weight,
        H_data,
        quant_args,
        return_weight_q=True,
        verbose=False,
    )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    packed = {}
    packed_bytes = 0

    for name, tensor_out in out_tensors.items():

        if tensor_out is None:
            continue

        t = (
            tensor_out
            .detach()
            .cpu()
            .contiguous()
        )

        packed[name] = t

        packed_bytes += (
            t.numel() *
            t.element_size()
        )

    save_file(
        packed,
        outfile,
    )

    effective_bpw = (
        packed_bytes *
        8 /
        params
    )

    print(
        f"proxy : {proxy_error}"
    )

    print(
        f"time  : {elapsed:.2f}s"
    )

    print(
        f"packed: "
        f"{packed_bytes / 2**20:.3f} MiB"
    )

    print(
        f"bpw   : {effective_bpw:.6f}"
    )

    append_manifest({
        "layer": layer,
        "expert": expert,
        "projection": proj,
        "K": K,
        "status": "done",
        "source_shard": shard.name,
        "output_file": str(outfile),
        "params": params,
        "packed_bytes": packed_bytes,
        "effective_bpw": effective_bpw,
        "proxy_error": proxy_error,
        "runtime_s": elapsed,
    })

    del weight_q
    del out_tensors
    del packed
    del weight
    del H_data

    gc.collect()
    torch.cuda.empty_cache()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mistral Large 3 Flash "
            "EXL3 bootstrap converter"
        )
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--expert-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--expert-end",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--projection",
        choices=[
            "w1",
            "w2",
            "w3",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help=(
            "Quantize all routed expert "
            "weights contained in one shard"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
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

    args = parser.parse_args()

    if args.shard is None and args.layer is None:
        parser.error(
            "--layer is required "
            "unless --shard is used"
        )

    weight_map = load_index()

    print(
        "Mistral Large 3 Flash "
        "bootstrap converter"
    )

    print(f"K           : {args.K}")
    print(f"device      : {args.device}")

    #
    # SHARD MODE
    #

    if args.shard:

        jobs = jobs_for_shard(
            args.shard,
            weight_map,
        )

        if not jobs:
            raise RuntimeError(
                "No routed expert tensors "
                f"found in {args.shard}"
            )

        print(f"shard       : {args.shard}")
        print(f"tensor jobs : {len(jobs)}")

        for layer, expert, proj in jobs:

            if args.dry_run:

                print(
                    f"JOB layer={layer:02d} "
                    f"expert={expert:03d} "
                    f"proj={proj}"
                )

                continue

            quantize_tensor(
                layer,
                expert,
                proj,
                args.K,
                args.device,
                weight_map,
            )

    #
    # EXPERT MODE
    #

    else:

        projections = (
            PROJECTIONS
            if args.projection == "all"
            else (args.projection,)
        )

        print(f"layer       : {args.layer}")

        print(
            "experts     : "
            f"{args.expert_start}"
            ".."
            f"{args.expert_end}"
        )

        print(
            "projections : "
            + ",".join(projections)
        )

        for expert in range(
            args.expert_start,
            args.expert_end + 1,
        ):

            for proj in projections:

                if args.dry_run:

                    print(
                        f"JOB layer={args.layer:02d} "
                        f"expert={expert:03d} "
                        f"proj={proj}"
                    )

                    continue

                quantize_tensor(
                    args.layer,
                    expert,
                    proj,
                    args.K,
                    args.device,
                    weight_map,
                )

    print()
    print("BOOTSTRAP PASS COMPLETE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path("/workspace/mistral-large-3-flash")
PASS_DIR = ROOT / "quant/bootstrap_passthrough"
K2_ROOT = ROOT / "quant/bootstrap_k2"

OUT_DIR = ROOT / "quant/mistral-large-3-flash-k2"

N_LAYERS = 61
FIRST_K_DENSE = 3
N_EXPERTS = 128

LM = "language_model.model"

PROJ_MAP = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def runtime_suffix(suffix):
    return (
        "weight_scale_inv"
        if suffix == "weight_scale"
        else suffix
    )


def map_layer_key(layer, rest):
    base = f"{LM}.layers.{layer}"

    if rest == "attention_norm.weight":
        return f"{base}.input_layernorm.weight"

    if rest == "ffn_norm.weight":
        return f"{base}.post_attention_layernorm.weight"

    attention_map = {
        "attention.wq_a.weight":
            "self_attn.q_a_proj.weight",

        "attention.wq_a.weight_scale":
            "self_attn.q_a_proj.weight_scale_inv",

        "attention.q_a_norm.weight":
            "self_attn.q_a_layernorm.weight",

        "attention.wq_b.weight":
            "self_attn.q_b_proj.weight",

        "attention.wq_b.weight_scale":
            "self_attn.q_b_proj.weight_scale_inv",

        "attention.wkv_a_with_mqa.weight":
            "self_attn.kv_a_proj_with_mqa.weight",

        "attention.wkv_a_with_mqa.weight_scale":
            "self_attn.kv_a_proj_with_mqa.weight_scale_inv",

        "attention.kv_a_norm.weight":
            "self_attn.kv_a_layernorm.weight",

        "attention.wkv_b.weight":
            "self_attn.kv_b_proj.weight",

        "attention.wkv_b.weight_scale":
            "self_attn.kv_b_proj.weight_scale_inv",

        "attention.wo.weight":
            "self_attn.o_proj.weight",

        "attention.wo.weight_scale":
            "self_attn.o_proj.weight_scale_inv",
    }

    if rest in attention_map:
        return f"{base}.{attention_map[rest]}"

    # Dense layers 0..2
    m = re.fullmatch(
        r"feed_forward\.(w[123])\.(weight|weight_scale)",
        rest,
    )

    if m:
        proj, suffix = m.groups()

        return (
            f"{base}.mlp.{PROJ_MAP[proj]}."
            f"{runtime_suffix(suffix)}"
        )

    # MoE router
    if rest == "gate.weight":
        return f"{base}.mlp.gate.weight"

    # Shared expert
    m = re.fullmatch(
        r"shared_experts\.(w[123])\.(weight|weight_scale)",
        rest,
    )

    if m:
        proj, suffix = m.groups()

        return (
            f"{base}.mlp.shared_experts."
            f"{PROJ_MAP[proj]}."
            f"{runtime_suffix(suffix)}"
        )

    if rest.startswith("experts."):
        return None

    raise RuntimeError(
        f"Unmapped layer tensor: layers.{layer}.{rest}"
    )


def map_global_key(key):
    if key == "tok_embeddings.weight":
        return "language_model.model.embed_tokens.weight"

    if key == "norm.weight":
        return "language_model.model.norm.weight"

    if key == "output.weight":
        return "language_model.lm_head.weight"

    return None


def build_passthrough_locator():
    print("Indexing passthrough tensors...")

    locator = {}
    files = sorted(PASS_DIR.glob("*.safetensors"))

    for i, p in enumerate(files, 1):
        with safe_open(
            p,
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                if key in locator:
                    raise RuntimeError(
                        f"Duplicate native passthrough tensor: {key}"
                    )
                locator[key] = p

        if i % 10 == 0 or i == len(files):
            print(
                f"  indexed {i:2d}/{len(files)} files"
            )

    print(
        f"Passthrough tensor locator: {len(locator)} tensors"
    )

    return locator


def get_tensor(locator, key):
    p = locator.get(key)

    if p is None:
        raise RuntimeError(
            f"Native tensor not found: {key}"
        )

    with safe_open(
        p,
        framework="pt",
        device="cpu",
    ) as f:
        return f.get_tensor(key)


def collect_layer_native_keys(locator, layer):
    prefix = f"layers.{layer}."

    return sorted(
        key
        for key in locator
        if key.startswith(prefix)
    )


def build_layer_tensors(locator, layer):
    output = {}

    prefix = f"layers.{layer}."

    native_keys = collect_layer_native_keys(
        locator,
        layer,
    )

    for key in native_keys:
        rest = key[len(prefix):]

        dst = map_layer_key(layer, rest)

        if dst is None:
            continue

        if dst in output:
            raise RuntimeError(
                f"Duplicate runtime key: {dst}"
            )

        output[dst] = get_tensor(
            locator,
            key,
        ).contiguous()

    # Routed K2 experts for MoE layers.
    if layer >= FIRST_K_DENSE:
        k2_dir = K2_ROOT / f"layer{layer:02d}"

        files = sorted(
            k2_dir.glob("expert*-w*-k2.safetensors")
        )

        if len(files) != N_EXPERTS * 3:
            raise RuntimeError(
                f"Layer {layer}: expected 384 K2 files, "
                f"found {len(files)}"
            )

        pattern = re.compile(
            r"expert(?P<expert>\d+)-"
            r"(?P<proj>w[123])-k2\.safetensors$"
        )

        seen = set()

        for p in files:
            m = pattern.fullmatch(p.name)

            if not m:
                raise RuntimeError(
                    f"Unexpected K2 filename: {p}"
                )

            expert = int(m.group("expert"))
            proj = m.group("proj")

            pair = (expert, proj)

            if pair in seen:
                raise RuntimeError(
                    f"Duplicate K2 group layer {layer}: {pair}"
                )

            seen.add(pair)

            with safe_open(
                p,
                framework="pt",
                device="cpu",
            ) as f:
                keys = set(f.keys())

                expected = {
                    "trellis",
                    "suh",
                    "svh",
                }

                if keys != expected:
                    raise RuntimeError(
                        f"Bad K2 artifact {p}: {sorted(keys)}"
                    )

                runtime_base = (
                    f"{LM}.layers.{layer}.mlp."
                    f"experts.{expert}."
                    f"{PROJ_MAP[proj]}"
                )

                for suffix in (
                    "trellis",
                    "suh",
                    "svh",
                ):
                    dst = (
                        f"{runtime_base}.{suffix}"
                    )

                    if dst in output:
                        raise RuntimeError(
                            f"Duplicate runtime key: {dst}"
                        )

                    output[dst] = (
                        f.get_tensor(suffix)
                        .contiguous()
                    )

        if len(seen) != N_EXPERTS * 3:
            raise RuntimeError(
                f"Layer {layer}: incomplete K2 coverage"
            )

    return output


def build_globals(locator):
    output = {}

    for native in (
        "tok_embeddings.weight",
        "norm.weight",
        "output.weight",
    ):
        dst = map_global_key(native)

        output[dst] = get_tensor(
            locator,
            native,
        ).contiguous()

    return output


def tensor_bytes(t):
    return t.numel() * t.element_size()


def verify_file(path, expected_keys):
    with safe_open(
        path,
        framework="pt",
        device="cpu",
    ) as f:
        actual = set(f.keys())

    expected = set(expected_keys)

    if actual != expected:
        missing = expected - actual
        extra = actual - expected

        raise RuntimeError(
            f"Shard verification failed: {path}\n"
            f"missing={list(missing)[:10]}\n"
            f"extra={list(extra)[:10]}"
        )


def write_atomic(path, tensors):
    tmp = Path(str(path) + ".tmp")

    if tmp.exists():
        tmp.unlink()

    save_file(
        tensors,
        tmp,
        metadata={
            "format": "pt",
            "quantization": "EXL3-K2+FP8",
            "model": "Mistral-Large-3-Flash-K2",
        },
    )

    if tmp.stat().st_size <= 0:
        raise RuntimeError(
            f"Empty temporary shard: {tmp}"
        )

    verify_file(
        tmp,
        tensors.keys(),
    )

    os.replace(
        tmp,
        path,
    )


def valid_existing(path, expected_keys):
    if not path.exists():
        return False

    if path.stat().st_size <= 0:
        return False

    try:
        verify_file(
            path,
            expected_keys,
        )
        return True
    except Exception:
        return False


def add_to_index(index, shard_name, tensors):
    for key in tensors:
        if key in index:
            raise RuntimeError(
                f"Duplicate tensor in index: {key}"
            )

        index[key] = shard_name


def human_gib(n):
    return f"{n / 1024**3:.3f} GiB"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("MISTRAL LARGE 3 FLASH K2")
    print("TEXT-ONLY RUNTIME CHECKPOINT ASSEMBLER")
    print("=" * 76)
    print()
    print("Output:")
    print(" ", out_dir)
    print()

    locator = build_passthrough_locator()

    weight_map = {}
    total_size = 0
    shard_records = []

    # ---------------------------------------------------------
    # Globals shard
    # ---------------------------------------------------------

    print()
    print("Building globals...")

    globals_tensors = build_globals(locator)

    global_name = "model-globals.safetensors"
    global_path = out_dir / global_name

    if (
        not args.force
        and valid_existing(
            global_path,
            globals_tensors.keys(),
        )
    ):
        print(
            f"  SKIP valid existing {global_name}"
        )
    else:
        print(
            f"  WRITE {global_name}"
        )
        write_atomic(
            global_path,
            globals_tensors,
        )

    add_to_index(
        weight_map,
        global_name,
        globals_tensors,
    )

    global_size = global_path.stat().st_size
    total_size += global_size

    shard_records.append(
        (global_name, global_size, len(globals_tensors))
    )

    del globals_tensors

    # ---------------------------------------------------------
    # One shard per transformer layer
    # ---------------------------------------------------------

    for layer in range(N_LAYERS):
        print()
        print(
            f"[layer {layer:02d}/{N_LAYERS - 1:02d}] "
            f"assembling..."
        )

        tensors = build_layer_tensors(
            locator,
            layer,
        )

        kind = (
            "dense"
            if layer < FIRST_K_DENSE
            else "moe"
        )

        shard_name = (
            f"model-layer-{layer:02d}-{kind}.safetensors"
        )

        shard_path = out_dir / shard_name

        expected_mem = sum(
            tensor_bytes(t)
            for t in tensors.values()
        )

        print(
            f"  tensors : {len(tensors)}"
        )
        print(
            f"  payload : {human_gib(expected_mem)}"
        )

        if (
            not args.force
            and valid_existing(
                shard_path,
                tensors.keys(),
            )
        ):
            print(
                f"  SKIP valid existing {shard_name}"
            )
        else:
            print(
                f"  WRITE {shard_name}"
            )

            write_atomic(
                shard_path,
                tensors,
            )

        add_to_index(
            weight_map,
            shard_name,
            tensors,
        )

        shard_size = shard_path.stat().st_size
        total_size += shard_size

        shard_records.append(
            (
                shard_name,
                shard_size,
                len(tensors),
            )
        )

        print(
            f"  disk    : {human_gib(shard_size)}"
        )

        del tensors

    # ---------------------------------------------------------
    # Final index
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print("WRITING INDEX")
    print("=" * 76)

    index = {
        "metadata": {
            "total_size": total_size,
            "format": "pt",
            "quantization": "EXL3-K2+FP8",
            "model": "Mistral-Large-3-Flash-K2",
            "text_only": True,
        },
        "weight_map": weight_map,
    }

    index_tmp = (
        out_dir / "model.safetensors.index.json.tmp"
    )

    index_final = (
        out_dir / "model.safetensors.index.json"
    )

    with open(
        index_tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            index,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    os.replace(
        index_tmp,
        index_final,
    )

    # ---------------------------------------------------------
    # Copy metadata useful for next runtime stage
    # ---------------------------------------------------------

    meta_src = ROOT / "upstream-meta"

    for name in (
        "params.json",
        "README.md",
        "chat_template.jinja",
        "SYSTEM_PROMPT.txt",
    ):
        src = meta_src / name

        if src.exists():
            shutil.copy2(
                src,
                out_dir / name,
            )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print("FINAL CHECK")
    print("=" * 76)

    all_keys = set()
    disk_total = 0

    for shard_name, _, _ in shard_records:
        p = out_dir / shard_name

        if not p.exists():
            raise RuntimeError(
                f"Missing final shard: {p}"
            )

        disk_total += p.stat().st_size

        with safe_open(
            p,
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                if key in all_keys:
                    raise RuntimeError(
                        f"Duplicate final tensor: {key}"
                    )
                all_keys.add(key)

    if all_keys != set(weight_map):
        raise RuntimeError(
            "Final tensor inventory != index inventory"
        )

    print(
        f"Final shards       : {len(shard_records)}"
    )
    print(
        f"Final tensors      : {len(all_keys)}"
    )
    print(
        f"Indexed tensors    : {len(weight_map)}"
    )
    print(
        f"Checkpoint size    : {human_gib(disk_total)}"
    )

    print()
    print("Shard inventory:")

    for name, size, count in shard_records:
        print(
            f"  {name:38s} "
            f"{human_gib(size):>12s} "
            f"{count:5d} tensors"
        )

    print()
    print("=" * 76)
    print("MISTRAL LARGE 3 FLASH K2 TEXT CHECKPOINT: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()

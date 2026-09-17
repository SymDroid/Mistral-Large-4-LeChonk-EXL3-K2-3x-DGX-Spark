#!/usr/bin/env python3

from pathlib import Path
import re
import os
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path("/workspace/mistral-large-3-flash")

LAYER = 35

PASS_DIR = ROOT / "quant/bootstrap_passthrough"
K2_DIR = ROOT / f"quant/bootstrap_k2/layer{LAYER:02d}"
OUT_DIR = ROOT / "quant/runtime_test"
OUT_FILE = OUT_DIR / f"layer{LAYER:02d}-mistral4-k2.safetensors"
TMP_FILE = OUT_FILE.with_suffix(".safetensors.tmp")

# We deliberately use the old-style ExLlama Mistral3/4 namespace:
#
#   language_model.model.layers.N...
#
# This corresponds to config.new_key_style == False.
#
BASE = f"language_model.model.layers.{LAYER}"


def map_native_key(key: str):
    """
    Map native Mistral Large 3 tensor names to the namespace expected by
    ExLlamaV3's Mistral4TextModel / MLAttention / BlockSparseMLP.

    Returns:
        runtime_key or None
    """

    prefix = f"layers.{LAYER}."

    if not key.startswith(prefix):
        return None

    rest = key[len(prefix):]

    # ---------------------------------------------------------
    # Norms
    # ---------------------------------------------------------

    if rest == "attention_norm.weight":
        return f"{BASE}.input_layernorm.weight"

    if rest == "ffn_norm.weight":
        return f"{BASE}.post_attention_layernorm.weight"

    # ---------------------------------------------------------
    # MLA attention
    # ---------------------------------------------------------

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

        # SPECIAL CASE:
        #
        # MLAttention reads kv_b directly rather than through Linear.
        # Its FP8 dequant path explicitly looks for weight_scale_inv.
        #
        # Mistral's weight_scale values are already the multiplicative
        # block dequant scales. We rename them but DO NOT numerically
        # invert them.
        "attention.wkv_b.weight_scale":
            "self_attn.kv_b_proj.weight_scale_inv",

        "attention.wo.weight":
            "self_attn.o_proj.weight",

        "attention.wo.weight_scale":
            "self_attn.o_proj.weight_scale_inv",
    }

    if rest in attention_map:
        return f"{BASE}.{attention_map[rest]}"

    # ---------------------------------------------------------
    # Router
    # ---------------------------------------------------------

    if rest == "gate.weight":
        return f"{BASE}.mlp.gate.weight"

    # ---------------------------------------------------------
    # Shared expert
    # ---------------------------------------------------------

    shared_match = re.fullmatch(
        r"shared_experts\.(w[123])\.(weight|weight_scale)",
        rest,
    )

    if shared_match:
        proj, suffix = shared_match.groups()

        proj_map = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }

        runtime_suffix = (
            "weight_scale_inv"
            if suffix == "weight_scale"
            else suffix
        )
        return (
            f"{BASE}.mlp.shared_experts."
            f"{proj_map[proj]}.{runtime_suffix}"
        )

    # ---------------------------------------------------------
    # Routed experts are NOT taken from passthrough.
    # They come from EXL3 K2 artifacts below.
    # ---------------------------------------------------------

    if rest.startswith("experts."):
        return None

    # Something layer-local exists that we have not mapped.
    return "__UNMAPPED__:" + key


def add_tensor(output, key, tensor, source):
    if key in output:
        raise RuntimeError(
            f"Duplicate output key:\n"
            f"  key    : {key}\n"
            f"  source : {source}"
        )

    output[key] = tensor.contiguous()


def load_layer_passthrough():
    output = {}
    native_seen = 0
    mapped = 0
    unmapped = []

    files = sorted(PASS_DIR.glob("*.safetensors"))

    print(f"Scanning passthrough files: {len(files)}")

    for p in files:
        with safe_open(p, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            for key in keys:
                if not key.startswith(f"layers.{LAYER}."):
                    continue

                native_seen += 1

                mapped_key = map_native_key(key)

                if mapped_key is None:
                    continue

                if mapped_key.startswith("__UNMAPPED__:"):
                    unmapped.append(key)
                    continue

                tensor = f.get_tensor(key)

                add_tensor(
                    output,
                    mapped_key,
                    tensor,
                    str(p),
                )

                mapped += 1

    print()
    print("Layer passthrough:")
    print("  native tensors seen :", native_seen)
    print("  mapped tensors      :", mapped)
    print("  unmapped tensors    :", len(unmapped))

    if unmapped:
        print()
        print("UNMAPPED LAYER TENSORS:")
        for k in unmapped:
            print(" ", k)

        raise RuntimeError(
            "Layer contains unmapped passthrough tensors. "
            "Refusing to build incomplete runtime layer."
        )

    return output


def add_k2_experts(output):
    files = sorted(K2_DIR.glob("expert*-w*-k2.safetensors"))

    print()
    print("K2 expert source files:", len(files))

    if len(files) != 128 * 3:
        raise RuntimeError(
            f"Expected 384 K2 files, found {len(files)}"
        )

    pattern = re.compile(
        r"expert(?P<expert>\d+)-"
        r"(?P<proj>w[123])-k2\.safetensors$"
    )

    proj_map = {
        "w1": "gate_proj",
        "w3": "up_proj",
        "w2": "down_proj",
    }

    pairs = set()
    groups = 0
    tensors_added = 0

    for i, p in enumerate(files, 1):
        m = pattern.fullmatch(p.name)

        if not m:
            raise RuntimeError(f"Unexpected filename: {p}")

        expert = int(m.group("expert"))
        proj = m.group("proj")

        if not 0 <= expert < 128:
            raise RuntimeError(
                f"Invalid expert index {expert}: {p}"
            )

        pair = (expert, proj)

        if pair in pairs:
            raise RuntimeError(
                f"Duplicate expert/projection: {pair}"
            )

        pairs.add(pair)

        tensors = load_file(p, device="cpu")

        required = {"trellis", "suh", "svh"}

        if set(tensors.keys()) != required:
            raise RuntimeError(
                f"Unexpected EXL3 tensors in {p}\n"
                f"Expected: {sorted(required)}\n"
                f"Found   : {sorted(tensors.keys())}"
            )

        runtime_base = (
            f"{BASE}.mlp.experts.{expert}."
            f"{proj_map[proj]}"
        )

        for suffix in ("trellis", "suh", "svh"):
            add_tensor(
                output,
                f"{runtime_base}.{suffix}",
                tensors[suffix],
                str(p),
            )
            tensors_added += 1

        groups += 1

        if i % 64 == 0 or i == len(files):
            print(
                f"  processed {i:3d}/384 matrices"
            )

    expected_pairs = {
        (e, p)
        for e in range(128)
        for p in ("w1", "w2", "w3")
    }

    missing = expected_pairs - pairs
    extra = pairs - expected_pairs

    if missing or extra:
        raise RuntimeError(
            f"K2 expert coverage error\n"
            f"missing={sorted(missing)[:10]}\n"
            f"extra={sorted(extra)[:10]}"
        )

    if groups != 384:
        raise RuntimeError(
            f"Expected 384 EXL3 groups, got {groups}"
        )

    if tensors_added != 1152:
        raise RuntimeError(
            f"Expected 1152 EXL3 tensors, got {tensors_added}"
        )

    print()
    print("K2 experts:")
    print("  groups         :", groups)
    print("  tensors added  :", tensors_added)


def validate_inventory(output):
    print()
    print("Runtime inventory checks")

    required = [
        f"{BASE}.input_layernorm.weight",
        f"{BASE}.post_attention_layernorm.weight",

        f"{BASE}.self_attn.q_a_proj.weight",
        f"{BASE}.self_attn.q_a_layernorm.weight",
        f"{BASE}.self_attn.q_b_proj.weight",

        f"{BASE}.self_attn.kv_a_proj_with_mqa.weight",
        f"{BASE}.self_attn.kv_a_layernorm.weight",

        f"{BASE}.self_attn.kv_b_proj.weight",
        f"{BASE}.self_attn.kv_b_proj.weight_scale_inv",

        f"{BASE}.self_attn.o_proj.weight",

        f"{BASE}.mlp.gate.weight",

        f"{BASE}.mlp.shared_experts.gate_proj.weight",
        f"{BASE}.mlp.shared_experts.up_proj.weight",
        f"{BASE}.mlp.shared_experts.down_proj.weight",
    ]

    missing = [k for k in required if k not in output]

    if missing:
        print()
        print("MISSING REQUIRED KEYS:")
        for k in missing:
            print(" ", k)

        raise RuntimeError(
            "Runtime layer is missing required tensors."
        )

    # Validate kv_b FP8 geometry
    kvb = output[
        f"{BASE}.self_attn.kv_b_proj.weight"
    ]

    kvb_scale = output[
        f"{BASE}.self_attn.kv_b_proj.weight_scale_inv"
    ]

    print("  kv_b weight :", tuple(kvb.shape), kvb.dtype)
    print("  kv_b scale  :", tuple(kvb_scale.shape), kvb_scale.dtype)

    if tuple(kvb.shape) != (32768, 512):
        raise RuntimeError(
            f"Unexpected kv_b shape: {tuple(kvb.shape)}"
        )

    if tuple(kvb_scale.shape) != (256, 4):
        raise RuntimeError(
            f"Unexpected kv_b scale grid: {tuple(kvb_scale.shape)}"
        )

    if kvb.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise RuntimeError(
            f"kv_b is not FP8: {kvb.dtype}"
        )

    # Validate all EXL3 groups structurally
    bad = []

    for expert in range(128):
        for proj in (
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            base = (
                f"{BASE}.mlp.experts."
                f"{expert}.{proj}"
            )

            expected = [
                base + ".trellis",
                base + ".suh",
                base + ".svh",
            ]

            if not all(k in output for k in expected):
                bad.append(base)

    if bad:
        raise RuntimeError(
            f"Invalid EXL3 groups: {bad[:10]}"
        )

    print("  EXL3 groups : 384/384")
    print("  required    : PASS")


def main():
    print("=" * 72)
    print("MISTRAL LARGE 3 FLASH K2")
    print(f"Runtime layer assembly test: layer {LAYER}")
    print("=" * 72)
    print()

    if not PASS_DIR.exists():
        raise RuntimeError(
            f"Missing passthrough directory: {PASS_DIR}"
        )

    if not K2_DIR.exists():
        raise RuntimeError(
            f"Missing K2 directory: {K2_DIR}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output = load_layer_passthrough()

    add_k2_experts(output)

    validate_inventory(output)

    print()
    print("Total output tensors:", len(output))

    print()
    print("Writing:")
    print(" ", OUT_FILE)

    if TMP_FILE.exists():
        TMP_FILE.unlink()

    save_file(output, TMP_FILE)

    size = TMP_FILE.stat().st_size

    if size <= 0:
        raise RuntimeError(
            "Temporary output file is empty."
        )

    os.replace(TMP_FILE, OUT_FILE)

    print(
        f"Size: {size / 1024**3:.3f} GiB"
    )

    # Reopen final file and verify key count.
    with safe_open(
        OUT_FILE,
        framework="pt",
        device="cpu",
    ) as f:
        final_keys = list(f.keys())

    if len(final_keys) != len(output):
        raise RuntimeError(
            f"Final key count mismatch: "
            f"{len(final_keys)} != {len(output)}"
        )

    print()
    print("=" * 72)
    print("LAYER 35 FLASH K2 ASSEMBLY: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()

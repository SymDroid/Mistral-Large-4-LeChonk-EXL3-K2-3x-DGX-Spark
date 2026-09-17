#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspace/mistral-large-3-flash")
INDEX = ROOT / "upstream-meta/consolidated.safetensors.index.json"

LM = "language_model.model"
HEAD = "language_model.lm_head"

EXPECTED_MOE_LAYERS = range(3, 61)
EXPECTED_EXPERTS = range(128)
EXPECTED_PROJECTIONS = ("w1", "w2", "w3")


def map_projection(name):
    return {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    }[name]


def map_scale_suffix(src, dst_prefix):
    """
    Native Mistral FP8 weight_scale is a multiplicative dequant scale.

    Ordinary ExLlama Linear supports .weight_scale directly.

    MLA kv_b is special: it is loaded raw by MLAttention rather than
    through Linear, and MLAttention calls its multiplicative scale
    .weight_scale_inv.
    """
    if src == "wkv_b":
        return f"{dst_prefix}.weight_scale_inv"

    return f"{dst_prefix}.weight_scale"


def map_native_key(key):
    # ================================================================
    # TEXT GLOBALS
    # ================================================================

    if key == "tok_embeddings.weight":
        return f"{LM}.embed_tokens.weight"

    if key == "output.weight":
        return f"{HEAD}.weight"

    if key == "norm.weight":
        return f"{LM}.norm.weight"

    # ================================================================
    # TEXT NORMS
    # ================================================================

    m = re.fullmatch(r"layers\.(\d+)\.attention_norm\.weight", key)
    if m:
        l = int(m.group(1))
        return f"{LM}.layers.{l}.input_layernorm.weight"

    m = re.fullmatch(r"layers\.(\d+)\.ffn_norm\.weight", key)
    if m:
        l = int(m.group(1))
        return f"{LM}.layers.{l}.post_attention_layernorm.weight"

    # ================================================================
    # MLA ATTENTION
    # ================================================================

    mla = {
        "wq_a": "q_a_proj",
        "q_a_norm": "q_a_layernorm",
        "wq_b": "q_b_proj",
        "wkv_a_with_mqa": "kv_a_proj_with_mqa",
        "kv_a_norm": "kv_a_layernorm",
        "wkv_b": "kv_b_proj",
        "wo": "o_proj",
    }

    m = re.fullmatch(
        r"layers\.(\d+)\.attention\.([^.]+)\.(weight|weight_scale)",
        key,
    )
    if m:
        l = int(m.group(1))
        src = m.group(2)
        suffix = m.group(3)

        if src not in mla:
            return None

        dst = mla[src]
        prefix = f"{LM}.layers.{l}.self_attn.{dst}"

        if suffix == "weight":
            return f"{prefix}.weight"

        return map_scale_suffix(src, prefix)

    # ================================================================
    # DENSE MLP, layers 0..2
    # ================================================================

    m = re.fullmatch(
        r"layers\.(\d+)\.feed_forward\.(w[123])\.(weight|weight_scale)",
        key,
    )
    if m:
        l = int(m.group(1))
        proj = map_projection(m.group(2))
        suffix = m.group(3)

        prefix = f"{LM}.layers.{l}.mlp.{proj}"

        if suffix == "weight":
            return f"{prefix}.weight"

        return f"{prefix}.weight_scale"

    # ================================================================
    # ROUTED EXPERTS
    #
    # Native FP8 weight + weight_scale are replaced by EXL3
    # trellis + suh + svh.
    # ================================================================

    m = re.fullmatch(
        r"layers\.(\d+)\.experts\.(\d+)\.(w[123])\."
        r"(weight|weight_scale)",
        key,
    )
    if m:
        return "__ROUTED_REPLACED__"

    # ================================================================
    # ROUTER
    # ================================================================

    m = re.fullmatch(
        r"layers\.(\d+)\.(?:gate|router)\.(weight|weight_scale)",
        key,
    )
    if m:
        l = int(m.group(1))
        suffix = m.group(2)

        prefix = f"{LM}.layers.{l}.mlp.gate"

        if suffix == "weight":
            return f"{prefix}.weight"

        return f"{prefix}.weight_scale"

    # ================================================================
    # SHARED EXPERT
    # ================================================================

    m = re.fullmatch(
        r"layers\.(\d+)\.(?:shared_experts|shared_expert)\."
        r"(w[123])\.(weight|weight_scale)",
        key,
    )
    if m:
        l = int(m.group(1))
        proj = map_projection(m.group(2))
        suffix = m.group(3)

        prefix = f"{LM}.layers.{l}.mlp.shared_experts.{proj}"

        if suffix == "weight":
            return f"{prefix}.weight"

        return f"{prefix}.weight_scale"

    # ================================================================
    # VISION GLOBALS
    # ================================================================

    if key == "vision_encoder.patch_conv.weight":
        return "vision_tower.patch_conv.weight"

    if key == "vision_encoder.ln_pre.weight":
        return "vision_tower.ln_pre.weight"

    if key == "pre_mm_projector_norm.weight":
        return "multi_modal_projector.norm.weight"

    if key == "patch_merger.merging_layer.weight":
        return "multi_modal_projector.patch_merger.merging_layer.weight"

    if key == "vision_language_adapter.w_in.weight":
        return "multi_modal_projector.linear_1.weight"

    if key == "vision_language_adapter.w_out.weight":
        return "multi_modal_projector.linear_2.weight"

    # ================================================================
    # VISION TRANSFORMER NORMS
    # ================================================================

    m = re.fullmatch(
        r"vision_encoder\.transformer\.layers\.(\d+)\."
        r"(attention_norm|ffn_norm)\.weight",
        key,
    )
    if m:
        l = int(m.group(1))
        norm = m.group(2)

        return (
            f"vision_tower.transformer.layers.{l}."
            f"{norm}.weight"
        )

    # ================================================================
    # VISION ATTENTION
    # ================================================================

    vision_attn = {
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
    }

    m = re.fullmatch(
        r"vision_encoder\.transformer\.layers\.(\d+)\."
        r"attention\.(wq|wk|wv|wo)\.weight",
        key,
    )
    if m:
        l = int(m.group(1))
        proj = vision_attn[m.group(2)]

        return (
            f"vision_tower.transformer.layers.{l}."
            f"attention.{proj}.weight"
        )

    # ================================================================
    # VISION MLP
    # ================================================================

    m = re.fullmatch(
        r"vision_encoder\.transformer\.layers\.(\d+)\."
        r"feed_forward\.(w[123])\.weight",
        key,
    )
    if m:
        l = int(m.group(1))
        proj = map_projection(m.group(2))

        return (
            f"vision_tower.transformer.layers.{l}."
            f"feed_forward.{proj}.weight"
        )

    return None


def exl3_prefix(layer, expert, projection):
    proj = map_projection(projection)

    return (
        f"{LM}.layers.{layer}.mlp."
        f"experts.{expert}.{proj}"
    )


def audit_exl3_files():
    base = ROOT / "quant/bootstrap_k2"

    pattern = re.compile(
        r"expert(\d+)-(w[123])-k2\.safetensors$"
    )

    seen = set()
    topology = set()
    errors = []

    for layer_dir in sorted(base.glob("layer*")):
        m_layer = re.fullmatch(r"layer(\d+)", layer_dir.name)

        if not m_layer:
            continue

        layer = int(m_layer.group(1))

        for path in sorted(
            layer_dir.glob("expert*-w*-k2.safetensors")
        ):
            m = pattern.fullmatch(path.name)

            if not m:
                errors.append(f"bad filename: {path}")
                continue

            expert = int(m.group(1))
            projection = m.group(2)

            item = (layer, expert, projection)
            dst = exl3_prefix(layer, expert, projection)

            if item in topology:
                errors.append(
                    f"duplicate EXL3 topology item: {item}"
                )

            if dst in seen:
                errors.append(
                    f"duplicate EXL3 destination: {dst}"
                )

            topology.add(item)
            seen.add(dst)

    expected = {
        (layer, expert, projection)
        for layer in EXPECTED_MOE_LAYERS
        for expert in EXPECTED_EXPERTS
        for projection in EXPECTED_PROJECTIONS
    }

    missing = sorted(expected - topology)
    unexpected = sorted(topology - expected)

    return seen, errors, missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-unknown", type=int, default=200)
    args = parser.parse_args()

    with INDEX.open() as f:
        index = json.load(f)

    keys = list(index["weight_map"])

    counts = Counter()
    unknown = []
    destinations = defaultdict(list)

    for key in keys:
        mapped = map_native_key(key)

        if mapped is None:
            counts["unknown"] += 1
            unknown.append(key)
            continue

        if mapped == "__ROUTED_REPLACED__":
            counts["routed_source"] += 1
            continue

        counts["mapped"] += 1
        destinations[mapped].append(key)

    duplicates = {
        dst: src
        for dst, src in destinations.items()
        if len(src) > 1
    }

    exl3, exl3_errors, exl3_missing, exl3_unexpected = (
        audit_exl3_files()
    )

    print("=== Mistral Large 3 -> EXL3 assembly audit v2 ===")
    print()

    print(f"Original index keys     : {len(keys):,}")
    print(f"Mapped passthrough      : {counts['mapped']:,}")
    print(f"Routed source keys      : {counts['routed_source']:,}")
    print(f"Unknown                 : {counts['unknown']:,}")
    print()

    print(f"EXL3 linears found      : {len(exl3):,}")
    print(f"EXL3 tensors target     : {len(exl3) * 3:,}")
    print(f"Duplicate mappings      : {len(duplicates):,}")
    print(f"EXL3 filename errors    : {len(exl3_errors):,}")
    print(f"EXL3 topology missing   : {len(exl3_missing):,}")
    print(f"EXL3 topology unexpected: {len(exl3_unexpected):,}")
    print()

    print(
        f"Accounting              : "
        f"{counts['mapped'] + counts['routed_source']:,}"
        f" / {len(keys):,}"
    )

    if duplicates:
        print("\n=== DUPLICATE DESTINATIONS ===")
        for dst, src in list(duplicates.items())[:50]:
            print(dst)
            for s in src:
                print("   <-", s)

    if unknown:
        print("\n=== UNKNOWN KEYS ===")
        for key in unknown[:args.show_unknown]:
            print(key)

    if exl3_errors:
        print("\n=== EXL3 ERRORS ===")
        for e in exl3_errors[:100]:
            print(e)

    if exl3_missing:
        print("\n=== EXL3 MISSING ===")
        for e in exl3_missing[:100]:
            print(e)

    if exl3_unexpected:
        print("\n=== EXL3 UNEXPECTED ===")
        for e in exl3_unexpected[:100]:
            print(e)

    print()

    clean = (
        counts["unknown"] == 0
        and not duplicates
        and not exl3_errors
        and not exl3_missing
        and not exl3_unexpected
        and len(exl3) == 58 * 128 * 3
        and counts["mapped"] + counts["routed_source"] == len(keys)
    )

    if clean:
        print("TOPOLOGY: CLEAN")
        print("READY FOR PHYSICAL ASSEMBLY")
    else:
        print("TOPOLOGY: NEEDS MAPPING WORK")


if __name__ == "__main__":
    main()

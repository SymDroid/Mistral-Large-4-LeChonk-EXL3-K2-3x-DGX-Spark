#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import re
import torch
from safetensors import safe_open

ROOT = Path("/workspace/mistral-large-3-flash")
PASS_DIR = ROOT / "quant/bootstrap_passthrough"
K2_ROOT = ROOT / "quant/bootstrap_k2"

N_LAYERS = 61
FIRST_K_DENSE = 3
N_EXPERTS = 128
PROJS = ("w1", "w2", "w3")

LM = "language_model.model"

FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}

stats = Counter()
destinations = {}
unmapped = []
non_text = []
fp8_weights = {}
fp8_scales = {}


def runtime_suffix(suffix):
    return "weight_scale_inv" if suffix == "weight_scale" else suffix


def map_layer_key(layer, rest):
    base = f"{LM}.layers.{layer}"

    # Norms
    if rest == "attention_norm.weight":
        return f"{base}.input_layernorm.weight"

    if rest == "ffn_norm.weight":
        return f"{base}.post_attention_layernorm.weight"

    # MLA
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

    # Dense MLP, layers 0..2
    m = re.fullmatch(
        r"feed_forward\.(w[123])\.(weight|weight_scale)",
        rest,
    )

    if m:
        proj, suffix = m.groups()

        proj_map = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }

        return (
            f"{base}.mlp.{proj_map[proj]}."
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

        proj_map = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }

        return (
            f"{base}.mlp.shared_experts."
            f"{proj_map[proj]}.{runtime_suffix(suffix)}"
        )

    # Routed experts deliberately absent from passthrough.
    if rest.startswith("experts."):
        return None

    return "__UNMAPPED__"


def register_destination(dst, src):
    if dst in destinations:
        raise RuntimeError(
            "Duplicate runtime destination:\n"
            f"  {dst}\n"
            f"  first : {destinations[dst]}\n"
            f"  second: {src}"
        )

    destinations[dst] = src


def scan_passthrough():
    files = sorted(PASS_DIR.glob("*.safetensors"))

    if not files:
        raise RuntimeError(
            f"No passthrough files found in {PASS_DIR}"
        )

    print(f"Passthrough files: {len(files)}")

    layer_counts = defaultdict(Counter)

    for fi, p in enumerate(files, 1):
        with safe_open(
            p,
            framework="pt",
            device="cpu",
        ) as f:

            for key in f.keys():
                stats["passthrough_tensors"] += 1

                m = re.match(
                    r"layers\.(\d+)\.(.+)$",
                    key,
                )

                if not m:
                    non_text.append((key, p.name))
                    stats["non_text"] += 1
                    continue

                layer = int(m.group(1))
                rest = m.group(2)

                if not 0 <= layer < N_LAYERS:
                    unmapped.append((key, p.name))
                    stats["unmapped"] += 1
                    continue

                dst = map_layer_key(layer, rest)

                if dst is None:
                    stats["ignored_routed_native"] += 1
                    continue

                if dst == "__UNMAPPED__":
                    unmapped.append((key, p.name))
                    layer_counts[layer]["unmapped"] += 1
                    stats["unmapped"] += 1
                    continue

                register_destination(
                    dst,
                    f"{p.name}:{key}",
                )

                layer_counts[layer]["mapped"] += 1
                stats["mapped_text"] += 1

                sl = f.get_slice(key)
                shape = tuple(sl.get_shape())
                dtype = sl.get_dtype()

                if dst.endswith(".weight"):
                    fp8_weights[dst] = (
                        shape,
                        dtype,
                        key,
                    )

                elif dst.endswith(".weight_scale_inv"):
                    weight_dst = dst[
                        :-len(".weight_scale_inv")
                    ] + ".weight"

                    fp8_scales[weight_dst] = (
                        shape,
                        dtype,
                        key,
                    )

        if fi % 10 == 0 or fi == len(files):
            print(
                f"  scanned {fi:2d}/{len(files)} files"
            )

    print()
    print("Per-layer passthrough mapping:")

    for layer in range(N_LAYERS):
        c = layer_counts[layer]
        kind = (
            "dense"
            if layer < FIRST_K_DENSE
            else "MoE"
        )

        print(
            f"  layer {layer:02d} "
            f"{kind:5s} "
            f"mapped={c['mapped']:2d} "
            f"unmapped={c['unmapped']:2d}"
        )

    return layer_counts


def validate_text_layers(layer_counts):
    print()
    print("=" * 76)
    print("TEXT LAYER VALIDATION")
    print("=" * 76)

    problems = []

    for layer in range(N_LAYERS):
        base = f"{LM}.layers.{layer}"

        required = [
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",

            f"{base}.self_attn.q_a_proj.weight",
            f"{base}.self_attn.q_a_layernorm.weight",
            f"{base}.self_attn.q_b_proj.weight",

            f"{base}.self_attn.kv_a_proj_with_mqa.weight",
            f"{base}.self_attn.kv_a_layernorm.weight",

            f"{base}.self_attn.kv_b_proj.weight",
            f"{base}.self_attn.kv_b_proj.weight_scale_inv",

            f"{base}.self_attn.o_proj.weight",
        ]

        if layer < FIRST_K_DENSE:
            required += [
                f"{base}.mlp.gate_proj.weight",
                f"{base}.mlp.up_proj.weight",
                f"{base}.mlp.down_proj.weight",
            ]
        else:
            required += [
                f"{base}.mlp.gate.weight",

                f"{base}.mlp.shared_experts.gate_proj.weight",
                f"{base}.mlp.shared_experts.up_proj.weight",
                f"{base}.mlp.shared_experts.down_proj.weight",
            ]

        missing = [
            k for k in required
            if k not in destinations
        ]

        if missing:
            problems.append(
                (layer, missing)
            )

    if problems:
        print("FAILED layers:")

        for layer, missing in problems:
            print(f"\nLayer {layer}:")
            for k in missing:
                print("  MISSING", k)

        raise RuntimeError(
            f"{len(problems)} text layers are incomplete"
        )

    print("61/61 text layers contain required native tensors: PASS")


def validate_fp8_geometry():
    print()
    print("=" * 76)
    print("FP8 BLOCK GEOMETRY")
    print("=" * 76)

    checked = 0
    bad = []

    for weight_dst, (
        scale_shape,
        scale_dtype,
        native_scale,
    ) in sorted(fp8_scales.items()):

        if weight_dst not in fp8_weights:
            bad.append(
                (
                    weight_dst,
                    "scale exists but weight missing",
                )
            )
            continue

        weight_shape, weight_dtype, native_weight = (
            fp8_weights[weight_dst]
        )

        if len(weight_shape) != 2 or len(scale_shape) != 2:
            bad.append(
                (
                    weight_dst,
                    f"non-2D geometry "
                    f"{weight_shape} / {scale_shape}",
                )
            )
            continue

        expected = (
            scale_shape[0] * 128,
            scale_shape[1] * 128,
        )

        if weight_shape != expected:
            bad.append(
                (
                    weight_dst,
                    f"weight={weight_shape} "
                    f"scale={scale_shape} "
                    f"expected={expected}",
                )
            )
            continue

        checked += 1

    print("FP8 weight/scale pairs checked:", checked)

    if bad:
        print()
        print("BAD FP8 GEOMETRY:")

        for k, reason in bad[:30]:
            print(" ", k)
            print("    ", reason)

        raise RuntimeError(
            f"{len(bad)} FP8 geometry failures"
        )

    print("FP8 block geometry: PASS")


def scan_k2():
    print()
    print("=" * 76)
    print("EXL3 K2 INVENTORY")
    print("=" * 76)

    pattern = re.compile(
        r"expert(?P<expert>\d+)-"
        r"(?P<proj>w[123])-k2\.safetensors$"
    )

    total_files = 0
    total_groups = 0
    total_tensors = 0
    total_bytes = 0

    problems = []

    for layer in range(FIRST_K_DENSE, N_LAYERS):
        d = K2_ROOT / f"layer{layer:02d}"

        files = sorted(
            d.glob("expert*-w*-k2.safetensors")
        )

        expected = N_EXPERTS * len(PROJS)

        if len(files) != expected:
            problems.append(
                f"layer {layer}: "
                f"{len(files)} files, expected {expected}"
            )
            continue

        seen = set()

        for p in files:
            m = pattern.fullmatch(p.name)

            if not m:
                problems.append(
                    f"Unexpected filename: {p}"
                )
                continue

            expert = int(m.group("expert"))
            proj = m.group("proj")

            pair = (expert, proj)

            if pair in seen:
                problems.append(
                    f"Duplicate layer {layer}: {pair}"
                )
                continue

            seen.add(pair)

            with safe_open(
                p,
                framework="pt",
                device="cpu",
            ) as f:

                keys = set(f.keys())

                if keys != {
                    "trellis",
                    "suh",
                    "svh",
                }:
                    problems.append(
                        f"{p}: unexpected tensors "
                        f"{sorted(keys)}"
                    )
                    continue

            runtime_proj = {
                "w1": "gate_proj",
                "w3": "up_proj",
                "w2": "down_proj",
            }[proj]

            runtime_base = (
                f"{LM}.layers.{layer}.mlp."
                f"experts.{expert}.{runtime_proj}"
            )

            for suffix in (
                "trellis",
                "suh",
                "svh",
            ):
                register_destination(
                    f"{runtime_base}.{suffix}",
                    str(p),
                )

            total_files += 1
            total_groups += 1
            total_tensors += 3
            total_bytes += p.stat().st_size

        expected_pairs = {
            (e, p)
            for e in range(N_EXPERTS)
            for p in PROJS
        }

        if seen != expected_pairs:
            problems.append(
                f"layer {layer}: expert coverage mismatch"
            )

        print(
            f"  layer {layer:02d}: "
            f"{len(files):3d}/384 matrices"
        )

    expected_groups = (
        (N_LAYERS - FIRST_K_DENSE)
        * N_EXPERTS
        * 3
    )

    print()
    print("K2 matrices/groups :", total_groups)
    print("Expected           :", expected_groups)
    print("EXL3 tensors       :", total_tensors)
    print(
        "K2 bytes           :",
        f"{total_bytes / 1024**3:.3f} GiB",
    )

    if total_groups != expected_groups:
        problems.append(
            f"Total groups {total_groups}, "
            f"expected {expected_groups}"
        )

    if problems:
        print()
        print("K2 PROBLEMS:")

        for p in problems[:50]:
            print(" ", p)

        raise RuntimeError(
            f"{len(problems)} K2 inventory problems"
        )

    print("K2 routed expert inventory: PASS")

    return total_bytes


def show_non_text_inventory():
    print()
    print("=" * 76)
    print("GLOBAL / VISION / NON-LAYER INVENTORY")
    print("=" * 76)

    groups = Counter()

    for key, source in non_text:
        if key.startswith("vision"):
            group = "vision"
        elif key.startswith("tok_embeddings"):
            group = "tok_embeddings"
        elif key.startswith("output"):
            group = "output"
        elif key.startswith("norm"):
            group = "norm"
        else:
            group = key.split(".", 1)[0]

        groups[group] += 1

    for group, count in sorted(groups.items()):
        print(
            f"  {group:30s} {count:5d}"
        )

    print()
    print("Exact non-layer keys:")

    for key, source in sorted(non_text):
        print(f"  {key}")

    print()
    print(
        "NOTE: these tensors are inventory-only in this dry run."
    )
    print(
        "We will map globals + vision only after their exact "
        "native namespaces are confirmed above."
    )


def main():
    print("=" * 76)
    print("MISTRAL LARGE 3 FLASH K2")
    print("FULL MODEL ASSEMBLY DRY-RUN")
    print("=" * 76)
    print()

    if not PASS_DIR.exists():
        raise RuntimeError(
            f"Missing {PASS_DIR}"
        )

    if not K2_ROOT.exists():
        raise RuntimeError(
            f"Missing {K2_ROOT}"
        )

    layer_counts = scan_passthrough()

    validate_text_layers(layer_counts)
    validate_fp8_geometry()

    k2_bytes = scan_k2()

    show_non_text_inventory()

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    print(
        "Passthrough tensors :",
        stats["passthrough_tensors"],
    )
    print(
        "Mapped text tensors :",
        stats["mapped_text"],
    )
    print(
        "Non-layer tensors   :",
        len(non_text),
    )
    print(
        "Unmapped text       :",
        len(unmapped),
    )
    print(
        "Runtime destinations:",
        len(destinations),
    )
    print(
        "K2 routed size      :",
        f"{k2_bytes / 1024**3:.3f} GiB",
    )

    if unmapped:
        print()
        print("UNMAPPED TEXT TENSORS:")

        for key, source in unmapped:
            print(
                f"  {key}  [{source}]"
            )

        raise RuntimeError(
            "Unmapped text tensors remain"
        )

    print()
    print("=" * 76)
    print("TEXT STACK + K2 DRY-RUN: PASS")
    print("No output model files were written.")
    print("=" * 76)


if __name__ == "__main__":
    main()

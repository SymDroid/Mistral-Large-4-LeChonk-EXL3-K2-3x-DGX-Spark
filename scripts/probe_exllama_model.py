#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path("/workspace/mistral-large-3-flash")
EXL = ROOT / "refs" / "exllamav3"
MODEL_DIR = ROOT / "runtime-stage"

sys.path.insert(0, str(EXL))

from exllamav3.model.config import Config
from exllamav3.architecture.architectures import get_architectures

print("===== LOAD CONFIG =====")

cfg = Config.from_directory(str(MODEL_DIR))
arch_entry = get_architectures()[cfg.architecture]
ModelClass = arch_entry["model_class"]

print("config class :", type(cfg).__name__)
print("model class  :", ModelClass.__name__)

print()
print("===== CONSTRUCT MODEL =====")

model = ModelClass.from_config(cfg)

print("model class  :", type(model).__name__)
print("model name   :", model.get_name())

print()
print("===== COMPATIBILITY =====")

compat = model.check_compat()
print("check_compat :", compat)

print()
print("===== MODULE ACCOUNTING =====")

print("unmapped     :", model.num_unmapped_layers())

cache_layers = model.get_cache_layers()
print("cache layers :", len(cache_layers))

print()
print("===== CRITICAL MODULE RESOLUTION =====")

keys = [
    "language_model.model.embed_tokens",
    "language_model.model.layers.0.self_attn",
    "language_model.model.layers.0.mlp.gate_proj",
    "language_model.model.layers.2.mlp.down_proj",

    "language_model.model.layers.3.mlp",
    "language_model.model.layers.3.mlp.gate",

    "language_model.model.layers.14.mlp",
    "language_model.model.layers.14.mlp.experts.0.gate_proj",
    "language_model.model.layers.14.mlp.experts.0.up_proj",
    "language_model.model.layers.14.mlp.experts.0.down_proj",

    "language_model.model.layers.60.mlp.experts.127.gate_proj",

    "language_model.model.norm",
    "language_model.lm_head",
]

resolved = {}

for key in keys:
    try:
        module = model.find_module(key)
    except Exception as e:
        module = None
        print(f"FAIL {key}")
        print("    exception:", repr(e))
    else:
        print(
            f"{chr(80)+chr(65)+chr(83)+chr(83) if module is not None else chr(77)+chr(73)+chr(83)+chr(83)} "
            f"{key}"
            + (
                f" -> {type(module).__name__}"
                if module is not None else ""
            )
        )

    resolved[key] = module

print()
print("===== PHYSICAL EXL3 STORAGE =====")

prefixes = [
    "language_model.model.layers.3.mlp.experts.0.gate_proj",
    "language_model.model.layers.14.mlp.experts.0.gate_proj",
    "language_model.model.layers.14.mlp.experts.0.up_proj",
    "language_model.model.layers.14.mlp.experts.0.down_proj",
    "language_model.model.layers.60.mlp.experts.127.gate_proj",
]

for prefix in prefixes:
    states = []

    for component in ("trellis", "suh", "svh"):
        key = f"{prefix}.{component}"
        ok = cfg.stc.has_tensor(key)
        states.append(ok)
        print(("PASS" if ok else "MISS"), key)

    assert all(states), f"Incomplete EXL3 storage for {prefix}"

print()
print("===== HARD ASSERTS =====")

critical = [
    "language_model.model.embed_tokens",
    "language_model.model.layers.0.self_attn",
    "language_model.model.layers.0.mlp.gate_proj",
    "language_model.model.layers.3.mlp",
    "language_model.model.layers.14.mlp.experts.0.gate_proj",
    "language_model.model.layers.14.mlp.experts.0.up_proj",
    "language_model.model.layers.14.mlp.experts.0.down_proj",
    "language_model.model.layers.60.mlp.experts.127.gate_proj",
    "language_model.model.norm",
    "language_model.lm_head",
]

missing = [k for k in critical if resolved.get(k) is None]

if missing:
    print("Missing model modules:")
    for k in missing:
        print(" ", k)
    raise SystemExit(1)

print("model construction     : PASS")
print("critical modules       : PASS")
print("EXL3 physical storage  : PASS")

print()
print("EXLLAMAV3 MODEL PROBE: PASS")

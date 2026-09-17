#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path("/workspace/mistral-large-3-flash")
EXL = ROOT / "refs" / "exllamav3"
MODEL = ROOT / "runtime-stage"

sys.path.insert(0, str(EXL))

print("===== EXLLAMAV3 CONFIG PROBE =====")
print("Model:", MODEL)
print()

from exllamav3.model.config import Config

cfg = Config.from_directory(str(MODEL))

print()
print("===== CONFIG CREATED =====")
print("class                 :", type(cfg).__name__)
print("architecture          :", cfg.architecture)
print("arch string           :", cfg.arch_string)

print()
print("===== TEXT MODEL =====")

attrs = (
    "hidden_size",
    "num_hidden_layers",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "first_k_dense_replace",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "vocab_size",
    "max_position_embeddings",
)

for attr in attrs:
    if hasattr(cfg, attr):
        print(f"{attr:24}: {getattr(cfg, attr)}")

print()
print("===== KEY STYLE =====")

print("new_key_style         :", getattr(cfg, "new_key_style", "<missing>"))

old_key = "language_model.model.embed_tokens.weight"
new_key = "model.language_model.embed_tokens.weight"

print("old-style embedding   :", cfg.stc.has_tensor(old_key))
print("new-style embedding   :", cfg.stc.has_tensor(new_key))

assert cfg.stc.has_tensor(old_key), \
    "Old-style embedding key missing"

assert not cfg.stc.has_tensor(new_key), \
    "Unexpected new-style embedding key present"

assert getattr(cfg, "new_key_style", False) is False, \
    "Expected new_key_style=False"

print()
print("===== ROPE =====")

rs = getattr(cfg, "rope_settings", None)

if rs is None:
    print("rope_settings         : <missing>")
else:
    for attr in (
        "head_dim",
        "rope_theta",
        "max_position_embeddings",
        "original_max_position_embeddings",
        "rope_scaling",
    ):
        if hasattr(rs, attr):
            print(f"{attr:24}: {getattr(rs, attr)}")

print()
print("===== VISION =====")

vision_attrs = (
    "vision_hidden_size",
    "vision_intermediate_size",
    "vision_num_hidden_layers",
    "vision_num_attention_heads",
    "vision_head_dim",
    "vision_patch_size",
    "vision_image_size",
    "spatial_merge_size",
)

for attr in vision_attrs:
    if hasattr(cfg, attr):
        print(f"{attr:24}: {getattr(cfg, attr)}")

print()
print("===== HARD ASSERTS =====")

assert cfg.architecture == "Mistral3ForConditionalGeneration"
assert type(cfg).__name__ == "Mistral3Config"

assert cfg.num_hidden_layers == 61
assert cfg.vocab_size == 131072
assert cfg.max_position_embeddings == 294912

if hasattr(cfg, "hidden_size"):
    assert cfg.hidden_size == 7168

if hasattr(cfg, "first_k_dense_replace"):
    assert cfg.first_k_dense_replace == 3

if hasattr(cfg, "n_routed_experts"):
    assert cfg.n_routed_experts == 128

if hasattr(cfg, "num_experts_per_tok"):
    assert cfg.num_experts_per_tok == 4

print("architecture          : PASS")
print("storage namespace     : PASS")
print("key style             : PASS")
print("core topology         : PASS")

print()
print("EXLLAMAV3 CONFIG PROBE: PASS")

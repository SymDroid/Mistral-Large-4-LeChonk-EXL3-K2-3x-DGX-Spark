#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path("/workspace/mistral-large-3-flash")
PARAMS = ROOT / "upstream-meta" / "params.json"
PROCESSOR = ROOT / "upstream-runtime" / "processor_config.json"
OUT = ROOT / "runtime-stage"

OUT.mkdir(parents=True, exist_ok=True)

with PARAMS.open() as f:
    p = json.load(f)

with PROCESSOR.open() as f:
    processor = json.load(f)

moe = p["moe"]
vision = p["vision_encoder"]

# Deterministic translation:
# official Mistral params.json -> ExLlamaV3/HF-style config.json

text_config = {
    "model_type": "mistral4",

    "hidden_size": p["dim"],
    "intermediate_size": p["hidden_dim"],

    "num_hidden_layers": p["n_layers"],
    "num_attention_heads": p["n_heads"],
    "num_key_value_heads": p["n_kv_heads"],

    "q_lora_rank": p["q_lora_rank"],
    "kv_lora_rank": p["kv_lora_rank"],
    "qk_nope_head_dim": p["qk_nope_head_dim"],
    "qk_rope_head_dim": p["qk_rope_head_dim"],
    "v_head_dim": p["v_head_dim"],

    "hidden_act": "silu",
    "rms_norm_eps": p["norm_eps"],

    "moe_intermediate_size": moe["expert_hidden_dim"],
    "n_shared_experts": moe["num_shared_experts"],
    "n_routed_experts": moe["num_experts"],
    "num_experts_per_tok": moe["num_experts_per_tok"],
    "first_k_dense_replace": moe["first_k_dense_replace"],

    "norm_topk_prob": True,
    "n_group": moe["num_expert_groups"],
    "topk_group": moe["num_expert_groups_per_tok"],
    "routed_scaling_factor": moe["routed_scale"],

    "rope_interleave": True,
    "rope_theta": p["rope_theta"],
    "max_position_embeddings": p["max_position_embeddings"],

    "vocab_size": p["vocab_size"],
    "tie_word_embeddings": p["tied_embeddings"],

    # Preserve the official long-context scaling information.
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": p["yarn"]["factor"],
        "original_max_position_embeddings":
            p["yarn"]["original_max_position_embeddings"],
        "beta_fast": p["yarn"]["beta"],
        "beta_slow": p["yarn"]["alpha"],
        "llama_4_scaling_beta": p["llama_4_scaling"]["beta"],
    },
}

vision_config = {
    "model_type": "pixtral",
    "hidden_size": vision["hidden_size"],
    "intermediate_size": vision["intermediate_size"],
    "num_hidden_layers": vision["num_hidden_layers"],
    "num_attention_heads": vision["num_attention_heads"],
    "num_key_value_heads": vision["num_attention_heads"],
    "head_dim": vision["hidden_size"] // vision["num_attention_heads"],
    "num_channels": vision["num_channels"],
    "image_size": vision["image_size"],
    "patch_size": vision["patch_size"],
    "rope_theta": vision["rope_theta"],
}

config = {
    "architectures": ["Mistral3ForConditionalGeneration"],
    "model_type": "mistral3",

    "text_config": text_config,
    "vision_config": vision_config,

    "vocab_size": p["vocab_size"],
    "max_position_embeddings": p["max_position_embeddings"],

    "vision_feature_layer": -1,
    "multimodal_projector_bias": vision["adapter_bias"],
    "spatial_merge_size": vision["spatial_merge_size"],

    "image_token_id": vision["image_token_id"],
    "image_break_token_id": vision["image_break_token_id"],
    "image_end_token_id": vision["image_end_token_id"],
}

# -------- self-audit --------

assert config["architectures"] == ["Mistral3ForConditionalGeneration"]
assert text_config["model_type"] == "mistral4"

assert text_config["hidden_size"] == 7168
assert text_config["num_hidden_layers"] == 61
assert text_config["num_attention_heads"] == 128

assert text_config["q_lora_rank"] == 1536
assert text_config["kv_lora_rank"] == 512
assert text_config["qk_nope_head_dim"] == 128
assert text_config["qk_rope_head_dim"] == 64
assert text_config["v_head_dim"] == 128

assert text_config["n_routed_experts"] == 128
assert text_config["num_experts_per_tok"] == 4
assert text_config["n_shared_experts"] == 1
assert text_config["first_k_dense_replace"] == 3
assert text_config["moe_intermediate_size"] == 4096

assert text_config["norm_topk_prob"] is True
assert text_config["n_group"] == 1
assert text_config["topk_group"] == 1
assert text_config["routed_scaling_factor"] == 1.0
assert text_config["rope_interleave"] is True

assert text_config["vocab_size"] == 131072
assert text_config["max_position_embeddings"] == 294912

assert vision_config["hidden_size"] == 1664
assert vision_config["intermediate_size"] == 8192
assert vision_config["num_hidden_layers"] == 48
assert vision_config["num_attention_heads"] == 16
assert vision_config["head_dim"] == 104
assert vision_config["patch_size"] == 14
assert vision_config["image_size"] == 1540

assert config["spatial_merge_size"] == 2

ip = processor["image_processor"]
assert ip["image_processor_type"] in (
    "PixtralImageProcessor",
    "PixtralImageProcessorFast",
)
assert ip["patch_size"] == vision_config["patch_size"]

out = OUT / "config.json"
with out.open("w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

print("RUNTIME CONFIG: PASS")
print()
print("Architecture :", config["architectures"][0])
print("Text model   :", text_config["model_type"])
print("Layers       :", text_config["num_hidden_layers"])
print("Dense first  :", text_config["first_k_dense_replace"])
print("MoE layers   :", text_config["num_hidden_layers"] - text_config["first_k_dense_replace"])
print("Experts      :", text_config["n_routed_experts"])
print("Experts/tok  :", text_config["num_experts_per_tok"])
print("Hidden       :", text_config["hidden_size"])
print("Context      :", text_config["max_position_embeddings"])
print("Vision       :", vision_config["num_hidden_layers"], "layers")
print("Vision patch :", vision_config["patch_size"])
print("Spatial merge:", config["spatial_merge_size"])
print("Output       :", out)

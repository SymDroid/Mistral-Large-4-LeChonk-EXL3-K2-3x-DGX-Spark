from pathlib import Path
from safetensors import safe_open
import torch

ROOT = Path("/workspace/mistral-large-3-flash")
REF = ROOT / "upstream-reference/consolidated-00001-of-00272.safetensors"
QDIR = ROOT / "quant/mistral-large-3-flash-k2"

def load_runtime():
    out = {}
    for p in sorted(QDIR.glob("*.safetensors")):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("language_model.model.layers.0.") or \
                   k.startswith("language_model.model.layers.1."):
                    out[k] = f.get_tensor(k)
    return out

def map_key(k):
    # layers.N.*
    p = k.split(".")
    L = p[1]
    rest = ".".join(p[2:])

    prefix = f"language_model.model.layers.{L}."

    replacements = {
        "attention_norm.weight":
            "input_layernorm.weight",

        "ffn_norm.weight":
            "post_attention_layernorm.weight",

        "attention.wq_a.weight":
            "self_attn.q_a_proj.weight",

        "attention.q_a_norm.weight":
            "self_attn.q_a_layernorm.weight",

        "attention.wq_b.weight":
            "self_attn.q_b_proj.weight",

        "attention.wq_b.weight_scale":
            "self_attn.q_b_proj.weight_scale_inv",

        "attention.wkv_a_with_mqa.weight":
            "self_attn.kv_a_proj_with_mqa.weight",

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

        "feed_forward.w1.weight":
            "mlp.gate_proj.weight",

        "feed_forward.w1.weight_scale":
            "mlp.gate_proj.weight_scale_inv",

        "feed_forward.w2.weight":
            "mlp.down_proj.weight",

        "feed_forward.w2.weight_scale":
            "mlp.down_proj.weight_scale_inv",

        "feed_forward.w3.weight":
            "mlp.up_proj.weight",

        "feed_forward.w3.weight_scale":
            "mlp.up_proj.weight_scale_inv",
    }

    if rest not in replacements:
        raise KeyError(rest)

    return prefix + replacements[rest]

runtime = load_runtime()

print("=" * 90)
print("L0/L1 ASSEMBLY DIFFERENTIAL")
print("=" * 90)

bad = 0
checked = 0

with safe_open(str(REF), framework="pt", device="cpu") as f:
    for L in (0, 1):
        print(f"\n=== LAYER {L} ===")

        keys = sorted(k for k in f.keys() if k.startswith(f"layers.{L}."))
        assert len(keys) == 18, f"L{L}: expected 18 tensors, got {len(keys)}"

        for src_key in keys:
            dst_key = map_key(src_key)

            if dst_key not in runtime:
                print("MISSING:", dst_key)
                bad += 1
                continue

            a = f.get_tensor(src_key)
            b = runtime[dst_key]

            checked += 1

            shape_ok = tuple(a.shape) == tuple(b.shape)
            dtype_ok = a.dtype == b.dtype

            if shape_ok:
                if a.dtype == torch.float8_e4m3fn:
                    # Compare raw FP8 values after widening.
                    af = a.float()
                    bf = b.float()
                    exact = torch.equal(af, bf)
                    maxdiff = (af - bf).abs().max().item()
                else:
                    exact = torch.equal(a, b)
                    maxdiff = (a.float() - b.float()).abs().max().item()
            else:
                exact = False
                maxdiff = float("nan")

            status = "PASS" if shape_ok and dtype_ok and exact else "FAIL"

            print(
                f"{status:4}  "
                f"{src_key:<48} -> "
                f"{dst_key.split(f'layers.{L}.',1)[-1]:<38} "
                f"shape={tuple(a.shape)} "
                f"dtype={a.dtype} "
                f"maxdiff={maxdiff:g}"
            )

            if status != "PASS":
                if not shape_ok:
                    print("      runtime shape:", tuple(b.shape))
                if not dtype_ok:
                    print("      runtime dtype:", b.dtype)
                bad += 1

print("\n" + "=" * 90)
print(f"CHECKED: {checked}")
print(f"FAILURES: {bad}")

if bad == 0:
    print("RESULT: EXACT ASSEMBLY MATCH")
else:
    print("RESULT: ASSEMBLY MISMATCH DETECTED")
    raise SystemExit(1)

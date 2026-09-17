#!/usr/bin/env python3

import sys
import gc
from pathlib import Path

import torch

ROOT = Path("/workspace/mistral-large-3-flash")
EXL = ROOT / "refs" / "exllamav3"
MODEL_DIR = ROOT / "runtime-stage"

sys.path.insert(0, str(EXL))

from exllamav3.model.config import Config
from exllamav3.architecture.architectures import get_architectures

KEY = "language_model.model.layers.14.mlp.experts.0.gate_proj"
DEVICE = torch.device("cuda:0")

def gib(x):
    return x / 1024**3

def cuda_stats(label):
    torch.cuda.synchronize()
    print(
        f"{label:24} "
        f"allocated={gib(torch.cuda.memory_allocated()):.4f} GiB  "
        f"reserved={gib(torch.cuda.memory_reserved()):.4f} GiB"
    )

print("===== CUDA =====")
print("device       :", torch.cuda.get_device_name(0))
print("capability   :", torch.cuda.get_device_capability(0))
print()

print("===== CONFIG + MODEL =====")

cfg = Config.from_directory(str(MODEL_DIR))
ModelClass = get_architectures()[cfg.architecture]["model_class"]
model = ModelClass.from_config(cfg)

linear = model.find_module(KEY)

assert linear is not None

print("module       :", KEY)
print("class        :", type(linear).__name__)
print("in_features  :", linear.in_features)
print("out_features :", linear.out_features)

print()
print("===== STORAGE DETECTION =====")

assert linear.is_exl3_storage(KEY)

for suffix in ("trellis", "suh", "svh"):
    full = f"{KEY}.{suffix}"
    assert cfg.stc.has_tensor(full)
    print("PASS", full)

print()
print("===== MEMORY BEFORE LOAD =====")

gc.collect()
torch.cuda.empty_cache()
cuda_stats("before load")

print()
print("===== LOAD ONE EXL3 LINEAR =====")

linear.load(DEVICE)

torch.cuda.synchronize()

print("quant_type   :", linear.quant_type)
print("inner class  :", type(linear.inner).__name__)
print("device       :", linear.device)

assert linear.quant_type == "exl3"
assert type(linear.inner).__name__ == "LinearEXL3"

cuda_stats("after load")

print()
print("===== RUNTIME OBJECT =====")

inner = linear.inner

for attr in (
    "in_features",
    "out_features",
    "out_dtype",
):
    if hasattr(inner, attr):
        print(f"{attr:14}:", getattr(inner, attr))

print()
print("===== SINGLE-LINEAR FORWARD =====")

# Small but nontrivial token batch.
# Use FP16 first since EXL3 kernels normally operate on FP16/BF16 activations.
x = torch.randn(
    (4, linear.in_features),
    device=DEVICE,
    dtype=torch.float16,
)

print("input shape  :", tuple(x.shape))
print("input dtype  :", x.dtype)

with torch.inference_mode():
    y = linear.forward(x, {})

torch.cuda.synchronize()

print("output shape :", tuple(y.shape))
print("output dtype :", y.dtype)
print("finite       :", bool(torch.isfinite(y).all().item()))
print("mean abs     :", float(y.abs().mean().item()))
print("max abs      :", float(y.abs().max().item()))

assert tuple(y.shape) == (4, linear.out_features)
assert torch.isfinite(y).all()

cuda_stats("after forward")

print()
print("===== UNLOAD =====")

del y
del x

linear.unload()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("inner        :", linear.inner)
print("device       :", linear.device)

assert linear.inner is None
assert linear.device is None

cuda_stats("after unload")

print()
print("EXL3 CUDA LINEAR LOAD : PASS")
print("EXL3 CUDA FORWARD     : PASS")

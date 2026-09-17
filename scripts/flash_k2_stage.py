#!/usr/bin/env python3

import argparse
import gc
from pathlib import Path

import torch

from exllamav3.model.config import Config
from exllamav3.architecture.mistral4 import Mistral4TextModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "quant" / "mistral-large-3-flash-k2"


def gib(n):
    return n / 1024**3


def memory(tag):
    gc.collect()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()

    print(f"\n=== {tag} ===")
    print(f"CUDA allocated : {gib(torch.cuda.memory_allocated()):.3f} GiB")
    print(f"CUDA reserved  : {gib(torch.cuda.memory_reserved()):.3f} GiB")
    print(f"CUDA free      : {gib(free):.3f} GiB")
    print(f"CUDA total     : {gib(total):.3f} GiB")


parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["A", "B"], required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

device = torch.device(args.device)

print("=" * 72)
print("MISTRAL LARGE 3 FLASH K2 - 2x DGX SPARK STAGE LOADER")
print("=" * 72)
print("Model :", MODEL_DIR)
print("Stage :", args.stage)
print("GPU   :", torch.cuda.get_device_name(device))

memory("BEFORE CONFIG")

config = Config.from_directory(str(MODEL_DIR))

model = Mistral4TextModel(config)

all_modules = model.modules

assert len(all_modules) == 64, f"Expected 64 text modules, got {len(all_modules)}"

if args.stage == "A":
    # embedding + transformer layers 0..31
    start = 0
    stop = 33
else:
    # transformer layers 32..60 + final norm + lm_head
    start = 33
    stop = 64

stage_modules = all_modules[start:stop]

print()
print(f"Full module count : {len(all_modules)}")
print(f"Stage module range: [{start}:{stop}]")
print(f"Stage module count: {len(stage_modules)}")
print()
print("First stage module:", stage_modules[0].key)
print("Last stage module :", stage_modules[-1].key)

print("\nLoading stage...")

model._load_single(
    progressbar=True,
    device=device,
    config=config,
    modules=stage_modules,
    verbose=True,
)

# Build execution map while preserving ORIGINAL/global module indices.
model.fwd_modules = [
    (module, 0, global_idx)
    for global_idx, module in enumerate(all_modules)
    if start <= global_idx < stop
]

model.active_devices = [device.index or 0]
model.output_device = stage_modules[-1].device

memory("AFTER STAGE LOAD")

print()
print("=" * 72)
print(f"FLASH K2 STAGE {args.stage}: RESIDENT")
print("=" * 72)

if args.stage == "A":
    print("Embedding + layers 0..31 loaded.")
else:
    print("Layers 32..60 + final norm + LM head loaded.")

print("Process will remain alive so weights stay resident.")
print("Press Ctrl-C to unload.")

try:
    while True:
        import time
        time.sleep(3600)
except KeyboardInterrupt:
    print("\nStopping.")

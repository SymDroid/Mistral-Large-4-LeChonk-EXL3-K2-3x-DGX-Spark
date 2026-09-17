#!/usr/bin/env python3

import sys
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
EXL3_ROOT = ROOT / "refs" / "exllamav3"
sys.path.insert(0, str(EXL3_ROOT))

from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


parser = argparse.ArgumentParser(
    description="Mistral Large 3 EXL3 quantization sweep"
)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--expert", type=int, required=True)
parser.add_argument("--proj", choices=["w1", "w2", "w3"], required=True)
parser.add_argument("--K", type=int, default=2)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

TENSOR = f"layers.{args.layer}.experts.{args.expert}.{args.proj}.weight"
SCALE = TENSOR.replace(".weight", ".weight_scale")
DEVICE = args.device
K = args.K

def find_tensor():
    """
    Find the local safetensors shard containing TENSOR.
    """
    candidates = list(ROOT.rglob("*.safetensors"))

    for path in candidates:
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                if TENSOR in f.keys():
                    return path
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find {TENSOR} in any local .safetensors file"
    )


def tensor_bytes(t):
    return t.numel() * t.element_size()


print("=" * 72)
print(f"Mistral Large 3 -> EXL3 K{K} sweep test")
print("=" * 72)

shard = find_tensor()
print(f"\nSource shard: {shard}")
print(f"Tensor:       {TENSOR}")
print(f"Scale:        {SCALE}")


# ----------------------------------------------------------------------
# Load native Mistral FP8 tensor
# ----------------------------------------------------------------------

with safe_open(shard, framework="pt", device="cpu") as f:
    qweight = f.get_tensor(TENSOR)
    scales = f.get_tensor(SCALE)

print("\nNative checkpoint:")
print(f"  weight shape : {tuple(qweight.shape)}")
print(f"  weight dtype : {qweight.dtype}")
print(f"  scale shape  : {tuple(scales.shape)}")
print(f"  scale dtype  : {scales.dtype}")


# ----------------------------------------------------------------------
# Dequantize Mistral block-FP8
#
# Native:
#
#   weight [4096,7168]
#   scale  [32,56]
#
# Each scale covers one 128x128 block.
# ----------------------------------------------------------------------

rows, cols = qweight.shape

assert rows % 128 == 0
assert cols % 128 == 0

expected_scale_shape = (rows // 128, cols // 128)

assert tuple(scales.shape) == expected_scale_shape, (
    f"Unexpected scale shape {tuple(scales.shape)}, "
    f"expected {expected_scale_shape}"
)

print("\nDequantizing FP8 128x128 blocks...")

scale_full = (
    scales.float()
    .repeat_interleave(128, dim=0)
    .repeat_interleave(128, dim=1)
)

weight_native = qweight.float() * scale_full

del qweight, scales, scale_full


# ----------------------------------------------------------------------
# IMPORTANT:
#
# Safetensors / Mistral:
#
#       [out_features, in_features]
#
# quantize_exl3():
#
#       [in_features, out_features]
# ----------------------------------------------------------------------

weight = weight_native.T.contiguous()

del weight_native

print("\nEXL3 input:")
print(f"  shape : {tuple(weight.shape)}")
print(f"  dtype : {weight.dtype}")
print(f"  rms   : {weight.square().mean().sqrt().item():.8f}")

assert weight.dtype == torch.float32
assert weight.shape[0] % 16 == 0
assert weight.shape[1] % 128 == 0


# ----------------------------------------------------------------------
# Trigger ExLlamaV3's explicit uncalibrated fallback path.
#
# finalize_capture_H() contains:
#
#     if H_data['H'].is_meta:
#         ...
#         q_fallback = True
#
# Therefore no enormous real Hessian is allocated.
# ----------------------------------------------------------------------

k = weight.shape[0]

H_data = {
    "H": torch.empty((k, k), device="meta"),
    "device": torch.device(DEVICE),
}


# ----------------------------------------------------------------------
# First experiment:
#
# Plain EXL3 K2.
#
# No MCG/MUL1 experimentation yet. First establish that native Large 3
# expert weights can make the complete EXL3 round trip.
# ----------------------------------------------------------------------

quant_args = {
    "K": K,
    "devices": [DEVICE],
    "seed": 42,
    "apply_out_scales": False,
    "buf_size_k": 128,
    "sigma_reg": 0.025,
}


print("\nQuantizing...")
print(f"  K          : {K}")
print(f"  device     : {DEVICE}")
print(f"  fallback H : yes")
print()


t0 = time.perf_counter()

weight_q, proxy_err, packed = quantize_exl3(
    weight=weight,
    H_data=H_data,
    quant_args=quant_args,
    return_weight_q=True,
    progress_str="<step> Large3 expert",
    verbose=True,
)


torch.cuda.synchronize()
elapsed = time.perf_counter() - t0


# ----------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------

reference = weight.to(DEVICE)
reconstructed = weight_q.to(DEVICE)

diff = reconstructed - reference

mse = diff.square().mean()
rmse = mse.sqrt()

signal_mse = reference.square().mean()
nmse = mse / signal_mse

relative_l2 = (
    torch.linalg.vector_norm(diff)
    / torch.linalg.vector_norm(reference)
)

cosine = F.cosine_similarity(
    reference.flatten(),
    reconstructed.flatten(),
    dim=0,
)

max_abs = diff.abs().max()


# ----------------------------------------------------------------------
# Packed size / effective bitrate
# ----------------------------------------------------------------------

packed_bytes = sum(tensor_bytes(t) for t in packed.values())
num_weights = reference.numel()

effective_bpw = packed_bytes * 8.0 / num_weights


print("\n" + "=" * 72)
print("RESULT")
print("=" * 72)

print(f"weights          : {num_weights:,}")
print(f"proxy error      : {float(proxy_err):.10g}")
print(f"MSE              : {mse.item():.10g}")
print(f"RMSE             : {rmse.item():.10g}")
print(f"NMSE             : {nmse.item():.10g}")
print(f"relative L2      : {relative_l2.item():.10g}")
print(f"cosine similarity: {cosine.item():.10g}")
print(f"max abs error    : {max_abs.item():.10g}")

print("\nPacked tensors:")

for name, tensor in packed.items():
    b = tensor_bytes(tensor)
    print(
        f"  {name:10s} "
        f"{str(tuple(tensor.shape)):24s} "
        f"{str(tensor.dtype):16s} "
        f"{b / 1024 / 1024:9.3f} MiB"
    )

print(f"\nPacked total     : {packed_bytes / 1024 / 1024:.3f} MiB")
print(f"Effective bitrate: {effective_bpw:.6f} bpw")


# ----------------------------------------------------------------------
# Save actual EXL3 payload
# ----------------------------------------------------------------------

outdir = ROOT / "quant" / "output"
outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / f"layer{args.layer}-expert{args.expert}-{args.proj}-k{K}.safetensors"

save_tensors = {
    f"{TENSOR}.{name}": tensor.detach().cpu().contiguous()
    for name, tensor in packed.items()
}

save_file(save_tensors, outfile)

print(f"\nSaved: {outfile}")

print(f"Runtime          : {elapsed:.2f} s")

csv_path = outdir / "sweep_results.csv"
write_header = not csv_path.exists()

with csv_path.open("a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "layer",
            "expert",
            "projection",
            "K",
            "params",
            "effective_bpw",
            "proxy_error",
            "mse",
            "rmse",
            "nmse",
            "relative_l2",
            "cosine",
            "max_abs_error",
            "packed_mib",
            "runtime_s",
            "source_shard",
        ])

    writer.writerow([
        args.layer,
        args.expert,
        args.proj,
        K,
        num_weights,
        effective_bpw,
        float(proxy_err),
        mse.item(),
        rmse.item(),
        nmse.item(),
        relative_l2.item(),
        cosine.item(),
        max_abs.item(),
        packed_bytes / 1024 / 1024,
        elapsed,
        str(shard),
    ])

print(f"Sweep CSV       : {csv_path}")


# ----------------------------------------------------------------------
# Estimate routed-expert storage from measured effective bpw
# ----------------------------------------------------------------------

ROUTED_PARAMS = 653.91e9

estimated_bytes = ROUTED_PARAMS * effective_bpw / 8
estimated_gib = estimated_bytes / (1024 ** 3)

print("\nLarge 3 routed-expert extrapolation:")
print(f"  effective bpw : {effective_bpw:.6f}")
print(f"  ~653.91B params -> {estimated_gib:.2f} GiB")

print("\n✅ K2 experiment completed.")

N_LAYERS = 61
DENSE_LAYERS = 3
MOE_LAYERS = N_LAYERS - DENSE_LAYERS

DIM = 7168
EXPERT_DIM = 4096
N_EXPERTS = 128

# SwiGLU expert: gate_proj + up_proj + down_proj
PARAMS_PER_EXPERT = 3 * DIM * EXPERT_DIM
ROUTED_EXPERT_PARAMS = PARAMS_PER_EXPERT * N_EXPERTS * MOE_LAYERS

print(f"MoE layers: {MOE_LAYERS}")
print(f"Params/expert: {PARAMS_PER_EXPERT / 1e6:.2f} M")
print(f"Routed expert params: {ROUTED_EXPERT_PARAMS / 1e9:.2f} B")

for bpw in [4, 3, 2.75, 2.5, 2.25, 2.0, 1.75]:
    gb = ROUTED_EXPERT_PARAMS * bpw / 8 / 1e9
    gib = ROUTED_EXPERT_PARAMS * bpw / 8 / (1024**3)
    print(f"{bpw:>4} bpw : {gb:7.1f} GB / {gib:7.1f} GiB")

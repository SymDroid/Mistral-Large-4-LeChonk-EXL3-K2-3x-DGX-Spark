Mistral Large 4 LeChonk EXL3 K2/K4 on 3× DGX Spark

Experimental community project by SymDroid for preparing and running Mistral Large 4 1T-A52B on NVIDIA DGX Spark systems using a mixed EXL3 quantization layout and explicit pipeline parallelism.

«Current status: the Mistral Large 4 text model has been successfully converted, assembled and structurally validated with ExLlamaV3. The 3-node pipeline runtime is the next implementation stage.»

This project is nicknamed LeChonk.

Current validated build

The current text-only assembly has been validated with the following properties:

Component| Representation
Routed MoE experts, layers 1–71| EXL3 K2
Shared experts, layers 1–71| EXL3 K2
Attention, layers 0–71| EXL3 K4
Dense MLP, layer 0| EXL3 K4
Routers| BF16
Layer norms| BF16
Token embeddings| BF16
LM head| BF16
LeRoPE alpha| BF16

Final assembled text-model size:

252.441 GiB

Assembly validation:

- 9,235 safetensors files
- 83,917 tensor keys
- 0 duplicate tensor keys
- essential-key audit passed
- all 72 attention layers validated
- ExLlamaV3 model constructor passed

Model architecture

Reference model:

Mistral Large 4 1T-A52B

Relevant text architecture:

- approximately 1.06T total source parameters
- 72 transformer layers
- hidden size 6144
- 128 routed experts
- top-4 expert routing
- 1 shared expert
- layer 0 dense
- layers 1–71 MoE
- MLA attention
- latent KV rank 512
- 32 index heads
- configurable context up to 1,048,576 tokens in the source configuration

Mistral Large 4 uses a substantially different attention architecture from Mistral Large 3. The LeChonk runtime therefore includes Mistral 4-specific ExLlamaV3 work for MLA, shared KV sources, learned SSMax and LeRoPE.

Quantization layout

MoE core

The 71 sparse MoE layers use EXL3 K2 for the routed and shared expert matrices.

Measured output:

- routed experts: approximately 240.260 GiB
- shared experts: approximately 1.877 GiB
- total MoE EXL3 core: approximately 242.1 GiB

Each expert retains separate EXL3 representations for:

- "gate_proj"
- "up_proj"
- "down_proj"

Attention

All 72 attention layers are converted to EXL3 K4 where applicable.

Measured attention output:

5.4 GiB

The converter handles the different Mistral Large 4 attention layouts, including:

- full indexer + local KV
- shared indexer + local KV
- full indexer + shared KV

All 72 layers passed exact key validation.

Dense layer 0

Mistral Large 4 uses one dense transformer MLP before the MoE stack.

Layer 0 is converted to EXL3 K4:

- "gate_proj"
- "up_proj"
- "down_proj"

Measured size:

216.18 MiB

FP8 source conversion

The upstream checkpoint uses block-scaled FP8 weights.

LeChonk performs direct 128×128 block dequantization before EXL3 conversion:

w = wf8.float().reshape(r // 128, 128, c // 128, 128)
w.mul_(scale[:, None, :, None])
w = w.reshape(r, c)

No additional "/128" correction is applied.

Mistral Large 4 ExLlamaV3 support

The repository includes a patch for the Mistral Large 4 text architecture:

tools/ml4/patches/exllamav3-mistral-large4.patch

The patch adds or extends support for:

- "MistralLarge4ForConditionalGeneration"
- "mistral_large4_text"
- Mistral 4 MLA attention
- direct-hidden DSA indexer
- learned SSMax
- shared KV source layers
- conditional RoPE / no-RoPE layers
- LeRoPE alpha scaling
- Mistral Large 4 key layouts
- serialization of learned SSMax state

LeRoPE uses the checkpoint tensor:

lerope.alpha

and applies:

scaled_inv_freq = inv_freq * exp(alpha)

The implementation also avoids applying the older Llama-4 beta scaling on top of learned SSMax.

Build tools

The Mistral Large 4 conversion tools are under:

tools/ml4/

Attention conversion

ml4_attention_convert.py

Converts one Mistral Large 4 attention layer to the mixed EXL3 K4/raw representation.

Streaming attention runner

ml4_attention_run.py

Processes all 72 layers, validates each output and removes large source shards only after every dependent layer has passed validation.

Final result:

ALL 72 ATTENTION LAYERS VALID
Pinned shard 53 retained: True

Dense layer 0

ml4_dense0_convert.py

Converts the dense layer-0 MLP to EXL3 K4.

Final assembly

ml4_assemble_final.py

Builds the final flat ExLlamaV3 model directory.

Large quantized files are assembled with hardlinks so the approximately 248 GiB K2/K4 payload does not need to be duplicated on disk.

The assembler also:

- canonicalizes raw tensor names
- preserves "lerope.alpha"
- validates required tensors
- checks global tensor-key uniqueness
- keeps the original pinned source shard until runtime validation is complete

Validated ExLlamaV3 model construction

The assembled model successfully initializes as:

architecture: MistralLarge4ForConditionalGeneration
new_key_style: False
layers: 72
hidden: 6144
experts: 128
first dense layers: 1
index heads: 32
kv rank: 512

MODEL CONSTRUCTOR OK
modules: 75
first block: 1
last kv module: 72

For the current text-only validation, the Pixtral vision/preprocessor initialization is intentionally bypassed.

Target deployment: 3× DGX Spark

The intended LeChonk runtime uses pipeline parallelism, not conventional cross-node tensor parallelism.

Target topology:

DGX Spark / symChonk-a
Embedding + early transformer layers
        |
        | hidden-state transfer
        v
DGX Spark / symChonk-b
Middle transformer layers
        |
        | hidden-state transfer
        v
DGX Spark / symChonk-c
Late transformer layers + final norm + LM head

The exact layer boundaries will be selected according to measured resident bytes rather than simply splitting 72 layers into equal thirds.

The goal is to leave sufficient memory on each node for:

- CUDA/runtime allocations
- activations
- MoE working buffers
- MLA/DSA state
- KV cache
- API/runtime overhead

Why three DGX Sparks?

The validated text weights alone occupy:

252.441 GiB

Two DGX Sparks provide 256 GB total unified memory, leaving essentially no practical runtime headroom.

Three DGX Sparks provide substantially more room for pipeline execution and KV cache while retaining the aggressive K2 expert representation.

A two-Spark short-context experiment may still be investigated, but it is not the primary deployment target for this repository.

Current project status

Completed:

- Mistral Large 4 checkpoint inspection
- FP8 block dequantization
- 71-layer MoE EXL3 K2 conversion
- shared-expert conversion
- 72-layer attention EXL3 K4 conversion
- dense layer-0 EXL3 K4 conversion
- Mistral Large 4 ExLlamaV3 architecture patch
- learned SSMax support
- LeRoPE support
- shared-KV architecture support
- final 252.441 GiB assembly
- global key validation
- ExLlamaV3 model-constructor smoke test

In progress:

- 3-node pipeline placement
- "symChonk-a"
- "symChonk-b"
- "symChonk-c"
- pipeline transport
- full runtime loading
- KV-cache validation
- generation smoke test
- OpenAI-compatible serving layer
- performance measurements

No generation-speed figures are published yet because the complete 3-node runtime has not yet been validated.

Repository structure

tools/ml4/
    ml4_attention_convert.py
    ml4_attention_run.py
    ml4_dense0_convert.py
    ml4_assemble_final.py
    patches/
        exllamav3-mistral-large4.patch

refs/exllamav3/
    pinned ExLlamaV3 source tree

quant/
    earlier quantization and reference work

analysis/
    model inspection and experimental analysis

scripts/
    runtime and deployment utilities

Large model weights, generated checkpoints and quantization output are intentionally excluded from Git.

Hardware

Development and conversion have been performed on NVIDIA DGX Spark / GB10 ARM64 systems.

The target runtime is:

3× NVIDIA DGX Spark
384 GB combined unified memory
direct high-speed interconnect
pipeline-parallel execution

Experimental status

LeChonk is currently an experimental reference implementation.

The model conversion and structural assembly are validated, but the complete three-node inference runtime is still under active development.

Do not interpret the current repository as a finished turnkey inference server.

Disclaimer

This is an independent community project by SymDroid.

It is not affiliated with or endorsed by Mistral AI, NVIDIA, ExLlamaV3, Hugging Face or Onyx.

Model licensing and third-party components remain subject to their respective upstream licenses.

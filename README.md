Mistral Large 3 675B EXL3 K2 on 2× DGX Spark






Community deployment recipe by SymDroid for running Mistral Large 3 675B across two NVIDIA DGX Spark systems using EXL3 K2, a custom two-node pipeline runtime, and an OpenAI-compatible API.

Unlike the companion Mistral Medium 3.5 recipe, this deployment does not use vLLM tensor parallelism. Flash K2 explicitly splits the model across two DGX Spark nodes and transfers hidden states between them.

Tested configuration

2× DGX Spark

256 GB unified memory total

Mistral Large 3 675B Instruct

NVIDIA GB10 / ARM64

CUDA 13.2

PyTorch + ExLlamaV3

EXL3 K2 routed experts

FP8/BF16 non-expert components

explicit two-stage pipeline execution

direct ConnectX / QSFP link

OpenAI-compatible API

streaming SSE responses

official Mistral tokenizer through mistral_common

native Mistral tool calling

multiple tool calls per assistant turn

tested end-to-end with Onyx

Observed on the reference setup:

Metric

Result

Total unified memory

256 GB

Direct pipeline decode

~4.9 tok/s

OpenAI API sustained decode

~3.9–4.0 tok/s

TTFT, 4,249-token prompt

~20.5 s

TTFT, 5,431-token prompt

~23.8 s

Stage B prefill on larger prompts

~9–11 s

Current concurrency

C1

API port

8018

These are measurements from one setup, not guaranteed performance figures.

Topology

DGX Sparkx / Stage A
192.168.100.11
Embedding + layers 0..31
        |
        | hidden-state transfer
        | direct high-speed link
        v
192.168.100.10
DGX Spark / Stage B
Layers 32..60 + final norm/head
        |
        | token
        v
DGX Sparkx / Stage A
OpenAI API :8018

Flash K2 deliberately avoids conventional cross-node tensor parallelism and NCCL collectives for model execution. Each node owns a contiguous portion of the transformer.

Model architecture

Reference model: Mistral Large 3 675B Instruct

Approximate architecture:

675B total parameters

~41B active parameters

61 transformer layers

hidden size 7168

128 routed experts

top-4 expert routing

one shared expert

first three layers dense

58 MoE layers

The routed expert population accounts for the overwhelming majority of model parameters.

EXL3 K2 layout

Flash K2 uses a mixed representation:

Routed MoE experts              -> EXL3 K2
Attention/shared/dense tensors  -> FP8 / BF16
Routers/norms/sensitive tensors -> BF16

The routed experts alone represent roughly 654B parameters. Compressing that population to K2 is what makes this deployment practical across the combined 256 GB unified memory of two DGX Spark systems.

Pipeline split

The model is divided into two explicit stages:

Stage A
Embedding
Layers 0..31
Internal modules [0:33]

        |
        | FP32 hidden state
        v

Stage B
Layers 32..60
Final norm
LM head
Internal modules [33:64]

The hidden size is 7168, so the pipeline transfers 28,672 bytes per token when using FP32 hidden-state transfer.

Inference stack

Onyx / OpenAI client
        |
        v
FastAPI
        |
        v
flash_k2_server.py
        |
        v
PyTorch
        |
        v
ExLlamaV3 + Flash K2 patches
        |
        v
CUDA
        |
        v
2× NVIDIA DGX Spark

This is not a vLLM runtime.

ExLlamaV3

ExLlamaV3 is included as a pinned Git submodule under:

refs/exllamav3

Pinned upstream commit:

02aef45cd681b960a00afcd0749a4ab99e6c1bfe

Flash K2-specific changes are maintained separately:

patches/exllamav3/flash-k2.patch

The patch contains DGX Spark / ARM64 compatibility work and EXL3 runtime fixes required for the Mistral MoE checkpoint, including correct EXL3 weight_scale handling.

1. Clone the repository

git clone --recurse-submodules <repository-url>
cd Mistral-Large-3-EXL3-K2-2x-DGX-Spark

If cloned without submodules:

git submodule update --init --recursive

Apply the Flash K2 ExLlamaV3 patch:

cd refs/exllamav3
git apply ../../patches/exllamav3/flash-k2.patch
cd ../..

2. Model weights

Model weights are intentionally not included in this repository.

The reference deployment mounts the prepared checkpoint at:

/models/symistral

Stage A contains the first portion of the model. Stage B only requires the tensors used by its portion of the pipeline.

3. Start Stage B

Stage B must be available before Stage A starts serving requests.

export PYTHONPATH=/workspace/mistral-large-3-flash/refs/exllamav3
python scripts/flash_k2_server.py --stage B

The reference Stage B worker listens on:

0.0.0.0:35137

4. Start Stage A

export PYTHONPATH=/workspace/mistral-large-3-flash/refs/exllamav3
python scripts/flash_k2_server.py --stage A

The OpenAI-compatible API is then available on port:

8018

5. Check the API

Available endpoints:

GET  /health
GET  /v1/models
POST /v1/chat/completions

The served model name is:

symistral

Example:

curl http://localhost:8018/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "symistral",
    "messages": [
      {
        "role": "user",
        "content": "Explain why the sky is blue in three sentences."
      }
    ],
    "max_tokens": 128
  }'

Native Mistral tool calling

Flash K2 uses the official Mistral tokenizer and translates Mistral's native tool-call representation to the OpenAI tool-calling schema.

The current API supports:

OpenAI tools

tool_choice

assistant tool_calls

tool-result messages

streaming delta.tool_calls

multiple tool calls in one assistant turn

Native tool execution has been tested end-to-end with Onyx, including multi-tool assistant turns.

Why a custom pipeline?

Mistral Large 3 is a sparse MoE model with a very large routed-expert population. Even though only a fraction of the parameters are active for each token, the expert weights still need to reside somewhere.

Flash K2 combines aggressive expert quantization with an explicit layer split. Instead of presenting both DGX Sparks as one tensor-parallel model, Stage A computes the first part of the transformer and sends the resulting hidden state to Stage B.

Stage B completes the forward pass and returns the sampled token.

This also provides a path around cross-node tensor-parallel/NCCL constraints that can appear on experimental DGX Spark software stacks.

Concurrency

The v0.1 server currently uses a shared generation cache and a generation lock.

Current concurrency:

C1

Future work includes independent request caches, batching, higher concurrency, prefix caching and further decode optimization.

Repository structure

analysis/          tensor inspection and memory analysis
golden/            known-good components and debug references
patches/           external dependency patches
quant/             K2 quantization and validation tooling
refs/              upstream source dependencies
runtime-stage/     runtime model metadata
scripts/           assembly, validation, pipeline and API server
upstream-meta/     source-model metadata
upstream-runtime/  tokenizer/runtime metadata

Large generated checkpoints, model weights, quantization output and CUDA build artifacts are excluded from Git.

Notes

The current release is an experimental reference implementation rather than a turnkey production inference server.

The normal startup order is:

start Stage B

wait for Stage B to finish loading

start Stage A

wait for the OpenAI API to become available

connect Onyx or another OpenAI-compatible client

The current implementation prioritizes making Mistral Large 3 run correctly across two DGX Sparks. Performance, concurrency and deployment ergonomics can be improved further.

Disclaimer

This is an independent community project and is not affiliated with or endorsed by Mistral AI, NVIDIA, ExLlamaV3 or Onyx.

Model licensing and third-party components remain subject to their respective upstream licenses.

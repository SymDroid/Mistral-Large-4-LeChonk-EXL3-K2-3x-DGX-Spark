import argparse
import asyncio
import json
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from exllamav3.model.config import Config
from exllamav3.architecture.mistral4 import Mistral4TextModel
from exllamav3.cache import Cache
from exllamav3.modules.attn import prepare_for_attn

ROOT = Path("/workspace/mistral-large-3-flash")
MODEL_DIR = Path("/models/symistral")

MODEL_NAME = "symistral"

HOST_B = "192.168.100.10"
PORT_B = 35137

API_HOST = "0.0.0.0"
API_PORT = 8018

CACHE_TOKENS = 32768
DEVICE = torch.device("cuda:0")

BOS_ID = 1
EOS_ID = 2

# C1 for the first server version.
INFERENCE_LOCK = threading.Lock()


def mem(label):
    free, total = torch.cuda.mem_get_info()
    print(
        f"{label}: allocated={torch.cuda.memory_allocated()/2**30:.3f} GiB  "
        f"reserved={torch.cuda.memory_reserved()/2**30:.3f} GiB  "
        f"free={free/2**30:.3f} GiB"
    )


def recv_exact(sock, n):
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("socket closed")

        data.extend(chunk)

    return bytes(data)


def send_tensor(sock, tensor):
    t = tensor.detach().contiguous().cpu()

    if t.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected stage-boundary dtype: {t.dtype}"
        )

    shape = list(t.shape)

    header = struct.pack("!I", len(shape))
    header += struct.pack(
        "!" + "Q" * len(shape),
        *shape,
    )

    payload = t.numpy().tobytes()

    sock.sendall(header)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_tensor(sock):
    ndim = struct.unpack(
        "!I",
        recv_exact(sock, 4),
    )[0]

    shape = struct.unpack(
        "!" + "Q" * ndim,
        recv_exact(sock, 8 * ndim),
    )

    nbytes = struct.unpack(
        "!Q",
        recv_exact(sock, 8),
    )[0]

    raw = recv_exact(sock, nbytes)

    buf = bytearray(raw)

    x = torch.frombuffer(
        buf,
        dtype=torch.float32,
    ).clone()

    return x.reshape(shape).to(DEVICE)


def send_int(sock, value):
    sock.sendall(
        struct.pack("!q", int(value))
    )


def recv_int(sock):
    return struct.unpack(
        "!q",
        recv_exact(sock, 8),
    )[0]


def build(stage):
    print("=" * 72)
    print(f"FLASH K2 SERVER - STAGE {stage}")
    print("=" * 72)

    config = Config.from_directory(
        str(MODEL_DIR)
    )

    model = Mistral4TextModel(config)

    # Temporary cache required before weight loading.
    bootstrap_cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        max_batch_size=1,
    )

    modules = list(model.modules)

    assert len(modules) == 64, len(modules)

    if stage == "A":
        start, stop = 0, 33
    else:
        start, stop = 33, 64

    stage_modules = modules[start:stop]

    print(
        f"Loading modules [{start}:{stop}]..."
    )

    mem("before load")

    model._load_single(
        progressbar=True,
        device=DEVICE,
        config=config,
        modules=stage_modules,
        verbose=True,
    )

    model.fwd_modules = [
        (m, 0, i)
        for i, m in enumerate(modules)
        if start <= i < stop
    ]

    model.active_devices = [0]
    model.output_device = stage_modules[-1].device

    mem("after load")

    return model, bootstrap_cache


def make_params(cache, past_len):
    return {
        "cache": cache,
        "attn_mode": "flash_attn",
        "batch_shape": (1, CACHE_TOKENS),
        "past_len": past_len,
    }


# ---------------------------------------------------------------------
# STAGE B
# ---------------------------------------------------------------------

def handle_stage_b_connection(conn, addr, model, cache):
    print(f"[B] Stage A connected: {addr}")

    past_len = 0

    try:
        while True:
            seq_len = recv_int(conn)

            if seq_len == -1:
                print("[B] Request complete")
                break

            hidden = recv_tensor(conn)

            dummy_ids = torch.zeros(
                (hidden.shape[0], seq_len),
                dtype=torch.long,
                device=DEVICE,
            )

            params = make_params(
                cache,
                past_len,
            )

            prepare_for_attn(
                dummy_ids,
                params,
            )

            params["last_tokens_only"] = 1

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model.forward_ls(
                hidden,
                params,
            )

            torch.cuda.synchronize()

            dt = time.perf_counter() - t0

            # v0.1: greedy decoding.
            token = int(
                torch.argmax(
                    logits[0, -1]
                ).item()
            )

            send_int(conn, token)

            conn.sendall(
                struct.pack("!d", dt)
            )

            past_len += seq_len

    except (
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
    ) as e:
        print(f"[B] Connection ended: {e}")

    except Exception as e:
        print(f"[B] ERROR: {type(e).__name__}: {e}")
        raise

    finally:
        try:
            conn.close()
        except Exception:
            pass

        del cache

        torch.cuda.empty_cache()

        print("[B] Ready for next request")


def stage_b():
    model, cache = build("B")

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    server.bind(
        ("0.0.0.0", PORT_B)
    )

    server.listen(4)

    print()
    print(
        f"[B] Persistent worker listening "
        f"on 0.0.0.0:{PORT_B}"
    )

    while True:
        conn, addr = server.accept()

        # C1 worker. Process one request at a time.
        handle_stage_b_connection(
            conn,
            addr,
            model,
            cache,
        )


# ---------------------------------------------------------------------
# STAGE A ENGINE
# ---------------------------------------------------------------------

class FlashK2Engine:

    def __init__(self):
        from mistral_common.tokens.tokenizers.mistral import (
            MistralTokenizer,
        )

        self.model, self.cache = build("A")

        self.mtok = MistralTokenizer.from_file(
            str(MODEL_DIR / "tekken.json")
        )

        self.raw = (
            self.mtok
            .instruct_tokenizer
            .tokenizer
        )

        print()
        print(
            f"[A] Tokenizer ready "
            f"BOS={self.raw.bos_id} "
            f"EOS={self.raw.eos_id}"
        )

    def encode_messages(
        self,
        messages,
        tools=None,
        tool_choice="auto",
    ):
        from mistral_common.protocol.instruct.messages import (
            UserMessage,
            AssistantMessage,
            SystemMessage,
            ToolMessage,
        )

        from mistral_common.protocol.instruct.tool_calls import (
            ToolCall,
            FunctionCall,
        )

        from mistral_common.protocol.instruct.request import (
            ChatCompletionRequest,
            ToolChoiceEnum,
            convert_openai_tools,
        )

        mistral_messages = []

        def normalize_content(content):
            if content is None:
                return None

            if isinstance(content, list):
                parts = []

                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                    ):
                        parts.append(
                            part.get("text", "")
                        )

                return "\n".join(parts)

            if isinstance(content, str):
                return content

            return str(content)

        for msg in messages:
            role = msg.get("role")
            content = normalize_content(
                msg.get("content")
            )

            if role == "system":
                mistral_messages.append(
                    SystemMessage(
                        content=content or ""
                    )
                )

            elif role == "user":
                mistral_messages.append(
                    UserMessage(
                        content=content or ""
                    )
                )

            elif role == "assistant":
                openai_tool_calls = msg.get(
                    "tool_calls"
                )

                mistral_tool_calls = None

                if openai_tool_calls:
                    mistral_tool_calls = []

                    for tc in openai_tool_calls:
                        fn = tc.get(
                            "function",
                            {},
                        )

                        arguments = fn.get(
                            "arguments",
                            "{}",
                        )

                        if not isinstance(
                            arguments,
                            str,
                        ):
                            import json
                            arguments = json.dumps(
                                arguments
                            )

                        mistral_tool_calls.append(
                            ToolCall(
                                id=tc.get(
                                    "id",
                                    "null",
                                ),
                                function=FunctionCall(
                                    name=fn.get(
                                        "name",
                                        "",
                                    ),
                                    arguments=arguments,
                                ),
                            )
                        )

                mistral_messages.append(
                    AssistantMessage(
                        content=content,
                        tool_calls=mistral_tool_calls,
                    )
                )

            elif role == "tool":
                if content is None:
                    content = ""

                mistral_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get(
                            "tool_call_id"
                        ),
                        name=msg.get("name"),
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported role: {role}"
                )

        if not mistral_messages:
            raise ValueError(
                "messages must not be empty"
            )

        mistral_tools = None

        if tools:
            mistral_tools = convert_openai_tools(
                tools
            )

        if tool_choice is None:
            tool_choice = "auto"

        if isinstance(tool_choice, str):
            choice_map = {
                "auto": ToolChoiceEnum.auto,
                "none": ToolChoiceEnum.none,
                "required": ToolChoiceEnum.required,
                "any": ToolChoiceEnum.any,
            }

            if tool_choice not in choice_map:
                raise ValueError(
                    "Unsupported tool_choice: "
                    f"{tool_choice}"
                )

            mistral_tool_choice = choice_map[
                tool_choice
            ]

        else:
            # Named OpenAI tool_choice is not needed
            # for the first implementation. Refuse it
            # rather than silently changing semantics.
            raise ValueError(
                "Named tool_choice is not yet supported"
            )

        request = ChatCompletionRequest(
            messages=mistral_messages,
            tools=mistral_tools,
            tool_choice=mistral_tool_choice,
        )

        encoded = (
            self.mtok
            .encode_chat_completion(request)
        )

        return encoded.tokens

    def connect_b(self):
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        sock.settimeout(30)

        sock.connect(
            (HOST_B, PORT_B)
        )

        sock.settimeout(None)

        return sock

    def generate(
        self,
        messages,
        max_tokens=4096,
        on_token=None,
        tools=None,
        tool_choice="auto",
    ):
        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be >= 1"
            )

        prompt_ids = self.encode_messages(
            messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        if len(prompt_ids) >= CACHE_TOKENS:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens; "
                f"current cache limit is {CACHE_TOKENS}"
            )

        max_tokens = min(
            max_tokens,
            CACHE_TOKENS - len(prompt_ids),
        )

        cache = self.cache

        sock = None

        generated = []

        # Mistral/Tekken native control tokens, verified
        # against the tokenizer shipped with this model.
        TOOL_CALLS_ID = 9
        ARGS_ID = 32

        tool_mode = False

        try:
            sock = self.connect_b()

            ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=DEVICE,
            )

            params = make_params(
                cache,
                0,
            )

            prepare_for_attn(
                ids,
                params,
            )

            torch.cuda.synchronize()

            request_start = time.perf_counter()

            hidden = self.model.forward_ls(
                ids,
                params,
            )

            torch.cuda.synchronize()

            send_int(
                sock,
                ids.shape[1],
            )

            send_tensor(
                sock,
                hidden,
            )

            token = recv_int(sock)

            b_dt = struct.unpack(
                "!d",
                recv_exact(sock, 8),
            )[0]

            ttft = (
                time.perf_counter()
                - request_start
            )

            past_len = ids.shape[1]

            finish_reason = "length"

            if token == EOS_ID:
                finish_reason = "stop"
            else:
                generated.append(token)

                if token == TOOL_CALLS_ID:
                    tool_mode = True

                if on_token and not tool_mode:
                    on_token(
                        token,
                        self.raw.decode(
                            generated
                        ),
                    )

            decode_start = time.perf_counter()

            while (
                finish_reason != "stop"
                and len(generated) < max_tokens
            ):
                token_in = generated[-1]

                x = torch.tensor(
                    [[token_in]],
                    dtype=torch.long,
                    device=DEVICE,
                )

                params = make_params(
                    cache,
                    past_len,
                )

                prepare_for_attn(
                    x,
                    params,
                )

                hidden = self.model.forward_ls(
                    x,
                    params,
                )

                send_int(sock, 1)
                send_tensor(sock, hidden)

                token = recv_int(sock)

                _b_dt = struct.unpack(
                    "!d",
                    recv_exact(sock, 8),
                )[0]

                past_len += 1

                if token == EOS_ID:
                    finish_reason = "stop"
                    break

                generated.append(token)

                if token == TOOL_CALLS_ID:
                    tool_mode = True

                if on_token and not tool_mode:
                    on_token(
                        token,
                        self.raw.decode(
                            generated
                        ),
                    )

            decode_dt = (
                time.perf_counter()
                - decode_start
            )

            total_dt = (
                time.perf_counter()
                - request_start
            )

            text = self.raw.decode(
                generated
            )

            completion_tokens = len(generated)

            tool_calls = None

            if tool_mode:
                try:
                    marker_pos = generated.index(
                        TOOL_CALLS_ID
                    )

                    # Native Mistral format:
                    #
                    # [TOOL_CALLS]
                    #   tool_1 [ARGS] {...}
                    #   tool_2 [ARGS] {...}
                    #   ...
                    #
                    # ARGS_ID therefore marks each individual
                    # tool call. Parse the calls at token level.
                    args_positions = [
                        i
                        for i in range(
                            marker_pos + 1,
                            len(generated),
                        )
                        if generated[i] == ARGS_ID
                    ]

                    if not args_positions:
                        raise ValueError(
                            "Tool call contains no ARGS token"
                        )

                    tool_calls = []

                    name_start = marker_pos + 1

                    for call_index, args_pos in enumerate(
                        args_positions
                    ):
                        name_tokens = generated[
                            name_start:args_pos
                        ]

                        # Arguments continue until the start of
                        # the next tool name. We cannot simply
                        # decode everything after ARGS because
                        # Mistral may emit multiple tool calls.
                        if call_index + 1 < len(args_positions):
                            next_args_pos = args_positions[
                                call_index + 1
                            ]

                            # Find exactly where the JSON object
                            # ends by incrementally decoding and
                            # using JSONDecoder.raw_decode().
                            candidate_tokens = generated[
                                args_pos + 1:next_args_pos
                            ]
                        else:
                            candidate_tokens = generated[
                                args_pos + 1:
                            ]

                        tool_name = self.raw.decode(
                            name_tokens
                        ).strip()

                        if not tool_name:
                            raise ValueError(
                                f"Empty tool name at call "
                                f"{call_index}"
                            )

                        raw_candidate = self.raw.decode(
                            candidate_tokens
                        ).strip()

                        decoder = json.JSONDecoder()
                        parsed_args, json_end = (
                            decoder.raw_decode(raw_candidate)
                        )

                        tool_arguments = raw_candidate[
                            :json_end
                        ]

                        # Preserve valid JSON as a string for
                        # OpenAI tool_calls.
                        json.loads(tool_arguments)

                        tool_calls.append(
                            {
                                "id": uuid.uuid4().hex[:9],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_arguments,
                                },
                            }
                        )

                        print(
                            "[A] Native tool call | "
                            f"name={tool_name} | "
                            f"arguments={tool_arguments}",
                            flush=True,
                        )

                        # Anything between this JSON object's end
                        # and the next ARGS token is the next
                        # tool's name.
                        if call_index + 1 < len(args_positions):
                            prefix_text = self.raw.decode(
                                candidate_tokens
                            )

                            consumed_text = prefix_text[
                                :json_end
                            ]

                            # Determine consumed token count by
                            # finding the shortest token prefix
                            # whose decoded text contains the
                            # complete JSON object.
                            consumed = 0
                            for n in range(
                                1,
                                len(candidate_tokens) + 1,
                            ):
                                decoded_prefix = self.raw.decode(
                                    candidate_tokens[:n]
                                )
                                if len(decoded_prefix) >= len(
                                    consumed_text
                                ):
                                    consumed = n
                                    break

                            name_start = (
                                args_pos + 1 + consumed
                            )

                    text = None
                    finish_reason = "tool_calls"

                except Exception as e:
                    raise ValueError(
                        "Failed to parse native "
                        f"Mistral tool call: {e}"
                    ) from e

            # TTFT already includes generation of token 1.
            # decode_dt measures the subsequent decode loop, so the
            # comparable throughput count is completion_tokens - 1.
            decode_tokens = max(
                completion_tokens - 1,
                0,
            )

            decode_tps = (
                decode_tokens / decode_dt
                if decode_dt > 0 and decode_tokens > 0
                else 0.0
            )

            print(
                "[A] Generation complete | "
                f"prompt={len(prompt_ids)} tok | "
                f"completion={completion_tokens} tok | "
                f"TTFT={ttft:.3f}s | "
                f"decode={decode_tps:.3f} tok/s | "
                f"decode_time={decode_dt:.3f}s | "
                f"B_prefill={b_dt:.3f}s | "
                f"total={total_dt:.3f}s | "
                f"finish={finish_reason}",
                flush=True,
            )

            return {
                "text": text,
                "tool_calls": tool_calls,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "ttft": ttft,
                "decode_time": decode_dt,
                "decode_tokens_per_second": decode_tps,
                "total_time": total_dt,
                "stage_b_prefill_time": b_dt,
            }

        finally:
            if sock is not None:
                try:
                    send_int(sock, -1)
                except Exception:
                    pass

                try:
                    sock.close()
                except Exception:
                    pass

            del cache
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# OPENAI API
# ---------------------------------------------------------------------

def run_api():
    import uvicorn

    from fastapi import (
        FastAPI,
        HTTPException,
    )

    from fastapi.responses import (
        StreamingResponse,
    )

    engine = FlashK2Engine()

    app = FastAPI(
        title="Mistral Large 3 Flash K2",
        version="0.1.0",
    )

    created_at = int(time.time())

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "stage_b": f"{HOST_B}:{PORT_B}",
            "cache_tokens": CACHE_TOKENS,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "symdroid",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]):
        requested_model = body.get(
            "model",
            MODEL_NAME,
        )

        if requested_model != MODEL_NAME:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{requested_model}' "
                    f"not found"
                ),
            )

        messages = body.get("messages")

        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400,
                detail="messages must be a list",
            )

        max_tokens = int(
            body.get(
                "max_tokens",
                body.get("max_completion_tokens", 4096),
            )
        )

        stream = bool(
            body.get("stream", False)
        )

        tools = body.get("tools")
        tool_choice = body.get(
            "tool_choice",
            "auto",
        )

        if (
            tools is not None
            and not isinstance(tools, list)
        ):
            raise HTTPException(
                status_code=400,
                detail="tools must be a list",
            )

        request_id = (
            "chatcmpl-"
            + uuid.uuid4().hex
        )

        created = int(time.time())

        if not stream:

            def do_generate():
                with INFERENCE_LOCK:
                    return engine.generate(
                        messages=messages,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice,
                    )

            try:
                result = await asyncio.to_thread(
                    do_generate
                )

            except Exception as e:
                print(
                    f"[A] Generation error: "
                    f"{type(e).__name__}: {e}"
                )

                raise HTTPException(
                    status_code=500,
                    detail=str(e),
                )

            assistant_message = {
                "role": "assistant",
                "content": result["text"],
            }

            if result["tool_calls"]:
                assistant_message[
                    "tool_calls"
                ] = result["tool_calls"]

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": result[
                            "finish_reason"
                        ],
                    }
                ],
                "usage": {
                    "prompt_tokens": result[
                        "prompt_tokens"
                    ],
                    "completion_tokens": result[
                        "completion_tokens"
                    ],
                    "total_tokens": (
                        result["prompt_tokens"]
                        + result["completion_tokens"]
                    ),
                },
            }

        async def event_stream():

            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_token(
                token_id,
                full_text,
            ):
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    (
                        "token",
                        token_id,
                        full_text,
                    ),
                )

            def worker():
                try:
                    with INFERENCE_LOCK:
                        result = engine.generate(
                            messages=messages,
                            max_tokens=max_tokens,
                            on_token=on_token,
                            tools=tools,
                            tool_choice=tool_choice,
                        )

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("done", result),
                    )

                except Exception as e:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("error", e),
                    )

            threading.Thread(
                target=worker,
                daemon=True,
            ).start()

            previous_text = ""

            first_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(first_chunk)
                + "\n\n"
            )

            while True:
                item = await queue.get()

                if item[0] == "token":
                    _, token_id, full_text = item

                    # A trailing replacement character can be a
                    # temporary decode artifact while a multi-byte
                    # character is waiting for the next token.
                    safe_text = full_text
                    while safe_text.endswith("\ufffd"):
                        safe_text = safe_text[:-1]

                    if safe_text.startswith(previous_text):
                        delta = safe_text[
                            len(previous_text):
                        ]
                    elif previous_text.startswith(safe_text):
                        # Decoder temporarily shortened/revised its tail.
                        # Hold output until it becomes stable again.
                        delta = ""
                    else:
                        # Find the longest common prefix. Never resend
                        # already-streamed text to the client.
                        common = 0
                        limit = min(
                            len(previous_text),
                            len(safe_text),
                        )

                        while (
                            common < limit
                            and previous_text[common]
                            == safe_text[common]
                        ):
                            common += 1

                        if common < len(previous_text):
                            # Already emitted text cannot be retracted in
                            # OpenAI SSE. Wait for a stable decode.
                            delta = ""
                        else:
                            delta = safe_text[common:]

                    if delta:
                        previous_text += delta

                    if not delta:
                        continue

                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": delta
                                },
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield (
                        "data: "
                        + json.dumps(chunk)
                        + "\n\n"
                    )

                elif item[0] == "done":
                    result = item[1]

                    if result["tool_calls"]:
                        tool_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": []
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                        for index, tc in enumerate(
                            result["tool_calls"]
                        ):
                            tool_chunk["choices"][0][
                                "delta"
                            ]["tool_calls"].append(
                                {
                                    "index": index,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc[
                                            "function"
                                        ]["name"],
                                        "arguments": tc[
                                            "function"
                                        ]["arguments"],
                                    },
                                }
                            )

                        yield (
                            "data: "
                            + json.dumps(tool_chunk)
                            + "\n\n"
                        )

                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": result[
                                    "finish_reason"
                                ],
                            }
                        ],
                    }

                    yield (
                        "data: "
                        + json.dumps(final_chunk)
                        + "\n\n"
                    )

                    yield "data: [DONE]\n\n"
                    break

                else:
                    error = item[1]

                    import traceback
                    print("[A] Stream error traceback:")
                    traceback.print_exception(
                        type(error),
                        error,
                        error.__traceback__
                    )
                    print(
                        f"[A] Stream error: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    error_chunk = {
                        "error": {
                            "message": str(error),
                            "type": "server_error",
                        }
                    }

                    yield (
                        "data: "
                        + json.dumps(error_chunk)
                        + "\n\n"
                    )

                    yield "data: [DONE]\n\n"
                    break

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    print()
    print("=" * 72)
    print("FLASH K2 OPENAI API")
    print("=" * 72)
    print(f"Model: {MODEL_NAME}")
    print(f"API:   http://{API_HOST}:{API_PORT}")
    print(f"B:     {HOST_B}:{PORT_B}")
    print("=" * 72)

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stage",
        choices=["A", "B"],
        required=True,
    )

    args = ap.parse_args()

    if args.stage == "B":
        stage_b()
    else:
        run_api()


if __name__ == "__main__":
    main()

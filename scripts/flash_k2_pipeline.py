import argparse
import socket
import struct
import time
from pathlib import Path

import torch

from exllamav3.model.config import Config
from exllamav3.architecture.mistral4 import Mistral4TextModel
from exllamav3.cache import Cache
from exllamav3.modules.attn import prepare_for_attn

ROOT = Path("/workspace/mistral-large-3-flash")
MODEL_DIR = ROOT / "quant/mistral-large-3-flash-k2"

HOST_B = "192.168.100.10"
PORT = 35137
CACHE_TOKENS = 8192
DEVICE = torch.device("cuda:0")


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
    shape = list(t.shape)

    # Mistral4 stage boundary currently emits FP32.
    assert t.dtype == torch.float32, f"unexpected hidden dtype: {t.dtype}"

    header = struct.pack("!I", len(shape))
    header += struct.pack("!" + "Q" * len(shape), *shape)
    payload = t.numpy().tobytes()

    sock.sendall(header)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_tensor(sock):
    ndim = struct.unpack("!I", recv_exact(sock, 4))[0]
    shape = struct.unpack("!" + "Q" * ndim, recv_exact(sock, 8 * ndim))
    nbytes = struct.unpack("!Q", recv_exact(sock, 8))[0]

    raw = recv_exact(sock, nbytes)

    # bytearray gives writable storage and avoids PyTorch warning
    buf = bytearray(raw)
    x = torch.frombuffer(buf, dtype=torch.float32).clone()
    return x.reshape(shape).to(DEVICE)


def send_int(sock, value):
    sock.sendall(struct.pack("!q", int(value)))


def recv_int(sock):
    return struct.unpack("!q", recv_exact(sock, 8))[0]


def build(stage):
    print("=" * 72)
    print(f"FLASH K2 PIPELINE - STAGE {stage}")
    print("=" * 72)

    config = Config.from_directory(str(MODEL_DIR))
    model = Mistral4TextModel(config)

    # IMPORTANT: cache must exist before weights are loaded.
    cache = Cache(
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

    print(f"Loading modules [{start}:{stop}]...")
    mem("before load")

    model._load_single(
        progressbar=True,
        device=DEVICE,
        config=config,
        modules=stage_modules,
        verbose=True,
    )

    # Preserve GLOBAL module indices.
    model.fwd_modules = [
        (m, 0, i)
        for i, m in enumerate(modules)
        if start <= i < stop
    ]

    model.active_devices = [0]
    model.output_device = stage_modules[-1].device

    mem("after load")

    return model, cache


def make_params(cache, past_len):
    return {
        "cache": cache,
        "attn_mode": "flash_attn",
        "batch_shape": (1, CACHE_TOKENS),
        "past_len": past_len,
    }


def stage_b():
    model, cache = build("B")

    print(f"\nListening on 0.0.0.0:{PORT}")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(1)

    conn, addr = server.accept()
    print("Stage A connected:", addr)

    past_len = 0

    while True:
        seq_len = recv_int(conn)
        if seq_len < 0:
            break

        hidden = recv_tensor(conn)

        # Stage B needs token shape/positions for attention metadata,
        # but token values are irrelevant because embedding is on A.
        dummy_ids = torch.zeros(
            (hidden.shape[0], seq_len),
            dtype=torch.long,
            device=DEVICE,
        )

        params = make_params(cache, past_len)
        prepare_for_attn(dummy_ids, params)

        # Only keep final token at logits-producing head
        params["last_tokens_only"] = 1

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model.forward_ls(hidden, params)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Greedy first boot
        token = int(torch.argmax(logits[0, -1]).item())

        send_int(conn, token)
        conn.sendall(struct.pack("!d", dt))

        past_len += seq_len

        print(
            f"B: seq={seq_len} past={past_len} "
            f"compute={dt:.3f}s token={token}"
        )

    conn.close()
    server.close()


def stage_a(prompt, max_new_tokens):
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    model, cache = build("A")

    mtok = MistralTokenizer.from_file(str(MODEL_DIR / "tekken.json"))
    raw = mtok.instruct_tokenizer.tokenizer

    # First boot uses raw Tekken + BOS.
    # Once execution is proven we'll switch to the full instruct request encoder.
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[UserMessage(content=prompt)]
    )
    prompt_ids = mtok.encode_chat_completion(request).tokens

    print("\nPROMPT:", prompt)
    print("Prompt tokens:", len(prompt_ids))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to Stage B {HOST_B}:{PORT}...")
    sock.connect((HOST_B, PORT))
    print("Connected.")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    params = make_params(cache, 0)
    prepare_for_attn(ids, params)

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    hidden = model.forward_ls(ids, params)

    torch.cuda.synchronize()

    send_int(sock, ids.shape[1])
    send_tensor(sock, hidden)

    first_token = recv_int(sock)
    b_dt = struct.unpack("!d", recv_exact(sock, 8))[0]

    torch.cuda.synchronize()
    ttft = time.perf_counter() - total_start

    generated = [first_token]
    print("\nFIRST TOKEN:", first_token)
    print("TEXT:", raw.decode(generated))
    print(f"TTFT: {ttft:.3f}s  (B compute {b_dt:.3f}s)")

    past_len = ids.shape[1]

    decode_start = time.perf_counter()

    for n in range(1, max_new_tokens):
        token_in = generated[-1]

        x = torch.tensor([[token_in]], dtype=torch.long, device=DEVICE)

        params = make_params(cache, past_len)
        prepare_for_attn(x, params)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        hidden = model.forward_ls(x, params)

        torch.cuda.synchronize()
        a_dt = time.perf_counter() - t0

        send_int(sock, 1)
        send_tensor(sock, hidden)

        token = recv_int(sock)
        b_dt = struct.unpack("!d", recv_exact(sock, 8))[0]

        generated.append(token)
        past_len += 1

        text = raw.decode(generated)
        print(
            f"\r[{n+1}/{max_new_tokens}] "
            f"A={a_dt:.3f}s B={b_dt:.3f}s | {text}",
            end="",
            flush=True,
        )

    print()

    decode_dt = time.perf_counter() - decode_start
    decode_tokens = max(0, len(generated) - 1)

    if decode_tokens:
        print(f"\nDecode: {decode_tokens/decode_dt:.3f} tok/s")
        print(f"Decode time: {decode_dt:.3f}s")

    print(f"TTFT: {ttft:.3f}s")
    print("\nFINAL:")
    print(raw.decode(generated))

    send_int(sock, -1)
    sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument(
        "--prompt",
        default="Explain why the sky is blue in three sentences."
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    if args.stage == "B":
        stage_b()
    else:
        stage_a(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()

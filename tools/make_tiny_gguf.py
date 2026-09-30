"""Create a tiny random-weight llama-architecture GGUF model for integration tests.

The model is meaningless (random weights), but it is a real GGUF file that the
real ``llama-server`` loads and serves. With JSON-schema constrained output,
llama.cpp's grammar guarantees schema-valid JSON even from random weights, so
it exercises the full runtime path: process start, model load, health check,
chat completion, schema-constrained decoding and shutdown.

    pip install gguf numpy
    python tools/make_tiny_gguf.py tests-tiny.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def build(path: Path, n_embd: int = 32, n_layer: int = 1, n_head: int = 2, n_ff: int = 64, seed: int = 0,
          structured: bool = True) -> Path:
    import gguf

    rng = np.random.default_rng(seed)
    tokens = ["<unk>", "<s>", "</s>"]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]
    tokens += [f"<0x{b:02X}>" for b in range(256)]
    types += [gguf.TokenType.BYTE] * 256
    printable = [chr(c) for c in range(33, 127)] + ["▁"]  # '▁' is the SentencePiece space
    for piece in printable + ["▁" + w for w in ("true", "false", "null")] + ['"', '":', '",', "[]", "{}"]:
        if piece not in tokens:
            tokens.append(piece)
            types.append(gguf.TokenType.NORMAL)
    scores = [0.0] * len(tokens)
    vocab = len(tokens)

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("css-tiny-test")
    writer.add_context_length(8192)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_chat_template(CHAT_TEMPLATE)

    def t(name, *shape):
        data = np.zeros(shape, np.float32) if structured else (rng.standard_normal(shape) * 0.02).astype(np.float32)
        writer.add_tensor(name, data)

    if structured:
        # Every token embeds to the same vector and the blocks are zero, so the final hidden state is
        # constant; output rows then act as fixed per-token preferences. Under a JSON grammar with greedy
        # decoding this closes strings/arrays/objects as early as allowed -> short, complete JSON.
        direction = rng.standard_normal(n_embd).astype(np.float32)
        writer.add_tensor("token_embd.weight", np.tile(direction, (vocab, 1)))
        hidden = direction / np.sqrt(np.mean(direction ** 2) + 1e-5)
        preference = {'"': 10.0, "]": 9.5, "}": 9.0, "5": 8.0, "0": 7.5, "\u2581false": 7.0, "\u2581null": 6.5, ",": 6.0}
        output = np.zeros((vocab, n_embd), np.float32)
        for piece, score in preference.items():
            if piece in tokens:
                output[tokens.index(piece)] = hidden * (score / float(hidden @ hidden))
        writer.add_tensor("output.weight", output)
    else:
        t("token_embd.weight", vocab, n_embd)
        t("output.weight", vocab, n_embd)
    writer.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    for i in range(n_layer):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        for proj in ("attn_q", "attn_k", "attn_v", "attn_output"):
            t(f"blk.{i}.{proj}.weight", n_embd, n_embd)
        t(f"blk.{i}.ffn_gate.weight", n_ff, n_embd)
        t(f"blk.{i}.ffn_up.weight", n_ff, n_embd)
        t(f"blk.{i}.ffn_down.weight", n_embd, n_ff)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny-test.gguf")
    build(target)
    print(f"{target}: {target.stat().st_size} bytes")

# Tools

## Where to get a model small enough to actually fit

You will not find a pretrained "tiny LLaMA" that matches this engine's
weight layout off the shelf — you need to **train your own tiny transformer**
with a training script of your choosing (PyTorch is the path of least
resistance), aiming for:

- `dim` around 64-288
- `n_layers` around 4-8
- `hidden_dim` ~2-3x `dim`
- a **small, closed vocabulary** (hundreds to a couple thousand tokens —
  word-level or even character-level, not a 32K-token BPE vocab) since the
  tokenizer here is greedy longest-match, not real BPE
- `seq_len` (context length) kept short — 64-256 — since the KV cache
  scales directly with it

A character-level or small-word-level model trained on a narrow domain
(e.g. a fixed set of short stories, or your own robot's command grammar)
is the realistic target — think "toy demo," not "general chatbot."

## `quantize_and_convert.py`

Takes your trained weights (as plain `.npy`/`.npz` arrays — see the
`EXPECTED_ARRAYS` list at the top of the script for exact names/shapes)
and a vocab list (one token per line, in id order) and produces:

- `model.bin` — the config header + int8-quantized tensors, in the exact
  order `llm_engine.h`'s `load()` expects (see the format spec below)
- `vocab_data.h` — a plain C header with your vocab as a `const char*`
  array, used regardless of whether the model itself is loaded from a
  file or baked in

```
python quantize_and_convert.py \
  --weights my_model_weights.npz \
  --vocab my_vocab.txt \
  --out-bin model.bin \
  --out-vocab vocab_data.h
```

## `make_c_array.py`

Optional second step — if you'd rather bake `model.bin` directly into
flash instead of reading it from LittleFS/SD at runtime:

```
python make_c_array.py model.bin model_data.h
```

Produces `model_data.h` with `const uint8_t esp32_llm_model[] PROGMEM = {...}`
and `esp32_llm_model_len` — `#include` it in `lua_engine.cpp` and pass
`esp32_llm_model, esp32_llm_model_len` to `Engine::load()` instead of using
`loadFromFile()`.

## Binary format (`model.bin`)

```
offset 0:  LLMConfig header (7 x int32, little-endian):
             dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len

then, in order:
  token_embedding:  int8[vocab_size * dim]           + float32 scale

  for each of n_layers layers:
    rms_att_weight: float32[dim]
    wq:             int8[dim * (n_heads*head_size)]   + float32 scale
    wk:             int8[dim * (n_kv_heads*head_size)]+ float32 scale
    wv:             int8[dim * (n_kv_heads*head_size)]+ float32 scale
    wo:             int8[(n_heads*head_size) * dim]   + float32 scale
    rms_ffn_weight: float32[dim]
    w1:             int8[dim * hidden_dim]            + float32 scale
    w2:             int8[hidden_dim * dim]            + float32 scale
    w3:             int8[dim * hidden_dim]            + float32 scale

  rms_final_weight: float32[dim]
  wcls:             int8[dim * vocab_size]            + float32 scale
```

`head_size = dim / n_heads`. Quantization is a single symmetric scale per
tensor: `scale = max(abs(weights)) / 127`, `int8_value = round(weight / scale)`.

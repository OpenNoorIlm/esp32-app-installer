# esp32.cpp — a llama.cpp-style transformer runtime, scaled down for ESP32

This is **not** a way to run real LLaMA / GPT-style production models on an
ESP32 — those are gigabytes of weights and no microcontroller in existence
will run them. What this *is*: a small, from-scratch, int8-quantized
transformer inference engine (embedding → attention w/ RoPE → SwiGLU FFN →
softmax) written in plain C++ for the Arduino/ESP32 toolchain, sized for
**tiny transformer models you train yourself** — typically in the
hundreds-of-thousands to ~15M parameter range.

Think: a toy character-level story generator, a tiny intent classifier
phrased as next-token prediction, a small command-completion model — not
a chatbot.

## Why so small?

| Board                     | Usable RAM      | Realistic model size |
|----------------------------|-----------------|-----------------------|
| Plain ESP32 (no PSRAM)     | ~320 KB          | a few hundred K params, short context |
| ESP32-S3 + 2-8 MB PSRAM    | 2-8 MB           | up to ~15M params, still short context |

Weight tensors are stored **int8** and kept in flash (not copied into RAM)
whenever they're compiled in as a C array — only activations and the KV
cache live in RAM, same trick TFLite Micro and llama.cpp's mmap both rely
on in spirit.

## What's in this zip

```
esp32.cpp.app/
  config.json                    -- app-installer metadata
  main.lua                       -- the installable "app": prompts the model, prints output
  firmware/
    llm_config.h                 -- model hyperparameter struct
    llm_engine.h                 -- the actual transformer forward pass + sampler
    llm_tokenizer.h              -- minimal greedy-longest-match tokenizer
    llm_lua_bindings.snippet.cpp -- Lua bindings to splice into lua_engine.cpp
    INTEGRATION.md               -- step-by-step wiring instructions
  tools/
    quantize_and_convert.py      -- turns raw float32 weights into this engine's .bin format
    make_c_array.py              -- turns that .bin into a flash-resident C array (bin2c)
    README.md                    -- the .bin file format spec + how to train a tiny model
  models/
    README.md                    -- where to get / how to train something small enough
```

## Quick start

1. Train (or find) a tiny transformer checkpoint — see `tools/README.md`.
2. Run `tools/quantize_and_convert.py` to get an int8 `.bin`, then either:
   - copy it to `/models/model.bin` on the ESP32's LittleFS/SD, **or**
   - run `tools/make_c_array.py` to bake it into the firmware as a C array.
3. Follow `firmware/INTEGRATION.md` to compile the engine into your existing
   `esp32.ino` sketch (this is a firmware change — it can't be delivered
   purely through `app-installer`, same constraint as the earlier TFLite
   discussion).
4. Once flashed, `main.lua` is the actual installable "app" — drop it in
   `/apps/esp32-cpp/main.lua` (or publish it through your `app-installer`
   repo the normal way) and run it with:
   ```
   lua load(esp32.fs.cat("/apps/esp32-cpp/main.lua"))()
   ```

## Honest limitations

- No real BPE/SentencePiece tokenizer — a greedy longest-match tokenizer
  over a fixed vocab list. Fine for a closed, small vocabulary; not a
  general-purpose tokenizer.
- Single-tensor-scale int8 quantization (not per-channel) — simplest
  possible scheme, trades some accuracy for code size.
- No multi-threading, no SIMD intrinsics — this is a reference
  implementation to build on, not a speed-optimized kernel library.
- Generation will be **slow** (seconds per token is plausible on plain
  ESP32) — this is fundamentally a proof-of-concept scaffold.

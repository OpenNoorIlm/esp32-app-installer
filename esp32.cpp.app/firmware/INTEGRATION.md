# Wiring esp32.cpp into your existing firmware

This engine is a native C++ addition to your sketch, not something
`app-installer` can deliver on its own (same constraint that applies to any
native inference engine — same as the ESP-SR/ESP-DL/TFLite discussion
earlier). Only the thin `main.lua` reaction layer installs normally once
this part is flashed.

## Steps

1. **Copy files into your sketch.**
   Copy `llm_config.h`, `llm_engine.h`, and `llm_tokenizer.h` into your
   `esp32/` folder, next to `lua_engine.h`/`lua_engine.cpp`.

2. **Get a model + vocab.**
   Run `tools/quantize_and_convert.py` on your trained weights to produce:
   - `model.bin` (if you'll load it from LittleFS/SD at runtime), **or**
   - run `tools/make_c_array.py` afterward to also get `model_data.h`
     (if you'd rather bake the model straight into flash)
   Either way you'll also get `vocab_data.h` — copy whichever of these you
   use into `esp32/` as well.

3. **Splice the Lua bindings in.**
   Open `firmware/llm_lua_bindings.snippet.cpp` — it's commented with
   exactly which four edits to make inside `esp32/lua_engine.cpp`:
   - add the `#include`s
   - add the global `Engine` instance
   - add the two `l_llm_*` functions + the `llmFuncs` table
   - add one line inside `begin()` to expose it as `esp32.llm.*`

4. **Memory planning.**
   - If you're on a plain ESP32 (no PSRAM), keep models *very* small — the
     KV cache (`n_layers * seq_len * kv_dim * 4 bytes * 2`) adds up fast on
     top of the ~320KB you have.
   - If you're on an ESP32-S3 with PSRAM, `llm_engine.h` already tries
     `heap_caps_malloc(..., MALLOC_CAP_SPIRAM)` for the KV cache and file
     buffer automatically, falling back to regular heap if PSRAM isn't
     found — check `sysinfo` (from `capability.h`) confirms PSRAM is
     actually detected on your board before assuming this path is taken.

5. **Storage, if loading from a file instead of a C array.**
   Put `model.bin` at `/models/model.bin` via the shell's `mkdir`/your
   normal file-transfer path (e.g. `curl -o /models/model.bin <url>` now
   that `curl` supports `-o`, if you're hosting it somewhere reachable).

6. **Flash, then install/run the Lua half.**
   Once compiled and flashed, drop `main.lua` at `/apps/esp32-cpp/main.lua`
   (or publish it in your `app-installer` repo folder the normal way) and
   run it with:
   ```
   lua load(esp32.fs.cat("/apps/esp32-cpp/main.lua"))()
   ```

## Debugging tips

- If `esp32.llm.load()` returns `false`: check the file actually exists at
  that path (`cat` it, or `ls` the directory) and that its size is
  non-zero — `loadFromFile()` fails silently on a missing/empty file.
- If generation output looks like garbage: double-check `vocab_data.h`
  actually matches the model you converted — mismatched vocab/model pairs
  will "work" (no crash) but produce nonsense, since token ids just won't
  line up with what the model was trained against.
- If it crashes/reboots during `forward()`: almost always a RAM issue —
  either the KV cache didn't get PSRAM (fell back to internal RAM and
  it doesn't fit), or `seq_len`/`n_layers`/`dim` in your model are bigger
  than what your board can hold. Shrink the model or the context length.

-- esp32.cpp.app -- Lua front-end for the native tiny-transformer engine.
--
-- This script alone does nothing: it calls esp32.llm.load()/esp32.llm.generate(),
-- which only exist once firmware/llm_engine.h + the bindings in
-- firmware/llm_lua_bindings.snippet.cpp have been compiled into the sketch.
-- See firmware/INTEGRATION.md.
--
-- Run with:
--   lua load(esp32.fs.cat("/apps/esp32-cpp/main.lua"))()

local MODEL_PATH   = "/models/model.bin" -- ignored if a C-array model was baked in instead
local PROMPT       = "Once upon a time"
local MAX_NEW_TOKS = 64

print("esp32.cpp -- tiny transformer runtime")
print("Loading model: " .. MODEL_PATH)

local ok = esp32.llm.load(MODEL_PATH)
if not ok then
  print("error: could not load a model.")
  print("  - if using LittleFS/SD: check " .. MODEL_PATH .. " exists")
  print("  - if using a baked-in C array: pass \"\" instead of a path")
  return
end

print("Prompt: " .. PROMPT)
print("Generating (this will be slow) ...")

local output = esp32.llm.generate(PROMPT, MAX_NEW_TOKS)
print("---")
print(output)

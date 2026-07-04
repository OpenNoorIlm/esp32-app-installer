#pragma once
#include <stdint.h>

// Model hyperparameters. This exact 7-int32 layout is the fixed header at
// the front of every .bin file produced by tools/quantize_and_convert.py
// (and the first bytes of a baked-in C array produced by
// tools/make_c_array.py) -- LLMEngine::load() reads it directly off the
// front of whatever buffer it's given.
struct LLMConfig {
  int32_t dim;         // residual stream / embedding width
  int32_t hidden_dim;  // FFN inner width
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;  // <= n_heads; lets you use grouped-query attention
  int32_t vocab_size;
  int32_t seq_len;     // max context length this model was exported with
};

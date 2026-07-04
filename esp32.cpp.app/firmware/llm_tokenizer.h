#pragma once
#include <Arduino.h>
#include <string.h>

// Minimal greedy-longest-match tokenizer. This is deliberately much simpler
// than a real BPE/SentencePiece tokenizer -- it just repeatedly consumes the
// longest vocab entry that matches the remaining input, falling back to
// skipping one byte if nothing matches (so it degrades gracefully instead of
// erroring on unseen text). Fine for a small, closed vocabulary trained
// alongside the model; not a general-purpose tokenizer.
//
// Vocab layout expected: a flat array of C strings, index == token id,
// produced by tools/quantize_and_convert.py from your training vocab file.
namespace LLMTokenizer {

struct Vocab {
  const char* const* tokens; // tokens[id] -> null-terminated string
  int32_t size;
};

inline int encode(const Vocab& v, const String& text, int32_t* outIds, int maxIds) {
  int n = 0;
  size_t pos = 0;
  while (pos < text.length() && n < maxIds) {
    int bestId = -1;
    size_t bestLen = 0;
    for (int32_t id = 0; id < v.size; id++) {
      size_t len = strlen(v.tokens[id]);
      if (len > bestLen && pos + len <= text.length() &&
          text.substring(pos, pos + len) == v.tokens[id]) {
        bestId = id;
        bestLen = len;
      }
    }
    if (bestId == -1) { pos++; continue; } // unseen byte -- skip it
    outIds[n++] = bestId;
    pos += bestLen;
  }
  return n;
}

inline String decode(const Vocab& v, const int32_t* ids, int n) {
  String out;
  for (int i = 0; i < n; i++) {
    if (ids[i] >= 0 && ids[i] < v.size) out += v.tokens[ids[i]];
  }
  return out;
}

} // namespace LLMTokenizer

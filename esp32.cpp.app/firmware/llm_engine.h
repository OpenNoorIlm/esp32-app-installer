#pragma once
#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <FS.h>
#include "llm_config.h"
#include "llm_tokenizer.h"

#if defined(ESP32)
#include <esp_heap_caps.h>
#endif

// A from-scratch, int8-quantized transformer forward pass -- the same
// architecture family as llama.cpp/llama2.c (embedding -> N x [RMSNorm ->
// multi-head attention w/ RoPE -> RMSNorm -> SwiGLU FFN] -> RMSNorm ->
// output projection), written independently and scaled down for ESP32:
//   - weight tensors are int8 with one float scale per tensor (simplest
//     possible quantization scheme -- not per-channel)
//   - weights are read straight out of flash (a `const` C array or an
//     mmap'd/loaded file buffer) and never copied, so only activations and
//     the KV cache use RAM
//   - activations stay float32 for simplicity/correctness; a
//     speed-optimized version would quantize activations too and use
//     integer SIMD, which this reference implementation deliberately
//     doesn't attempt
//
// See tools/README.md for the exact on-disk/on-flash layout this expects.
namespace LLMEngine {

struct QMatrix {
  const int8_t* data = nullptr; // row-major, dOut x dIn
  float scale = 1.0f;
};

struct LayerWeights {
  const float* rms_att_weight = nullptr; // [dim]
  QMatrix wq, wk, wv, wo;
  const float* rms_ffn_weight = nullptr; // [dim]
  QMatrix w1, w2, w3;
};

struct ModelWeights {
  QMatrix token_embedding;         // [vocab_size x dim]
  LayerWeights* layers = nullptr;  // [n_layers]
  const float* rms_final_weight = nullptr; // [dim]
  QMatrix wcls;                    // [dim x vocab_size]
};

struct RunState {
  float* x = nullptr;      // [dim]      current residual stream
  float* xb = nullptr;     // [dim]      normed scratch
  float* xb2 = nullptr;    // [dim]      attention output before residual add
  float* hb = nullptr;     // [hidden_dim]
  float* hb2 = nullptr;    // [hidden_dim]
  float* q = nullptr;      // [dim]
  float* att = nullptr;    // [n_heads * seq_len]
  float* logits = nullptr; // [vocab_size]
  float* key_cache = nullptr;   // [n_layers * seq_len * kv_dim]
  float* value_cache = nullptr; // [n_layers * seq_len * kv_dim]
};

// ---- small math helpers -----------------------------------------------

inline void rmsnorm(float* out, const float* x, const float* weight, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; i++) ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / size + 1e-5f);
  for (int i = 0; i < size; i++) out[i] = weight[i] * (x[i] * ss);
}

inline void softmax(float* x, int size) {
  float maxv = x[0];
  for (int i = 1; i < size; i++) if (x[i] > maxv) maxv = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// out[d] = scale * sum_n(row[n] * x[n]) for each of the d rows in w.
// w.data is int8, row-major (d x n); x stays float for simplicity (see
// header comment -- a speed-optimized version would quantize x too).
inline void qmatmul(float* out, const float* x, const QMatrix& w, int n, int d) {
  for (int i = 0; i < d; i++) {
    const int8_t* row = w.data + (size_t)i * n;
    int32_t acc = 0;
    for (int j = 0; j < n; j++) acc += (int32_t)row[j] * (int32_t)lroundf(x[j] / w.scale);
    out[i] = acc * w.scale * w.scale;
  }
}

inline float silu(float x) { return x / (1.0f + expf(-x)); }

// -------------------------------------------------------------------------

class Engine {
public:
  LLMConfig cfg{};
  ModelWeights w;
  RunState s;
  LLMTokenizer::Vocab vocab{};
  bool ready = false;

  // Parses a raw buffer (config header + quantized tensors, see
  // tools/README.md) into `w`, using pointers straight into `buf` -- the
  // caller must keep `buf` alive for the engine's whole lifetime (that's
  // exactly what a `const uint8_t model[] = {...};` C array gives you for
  // free, since it lives in flash for the life of the program).
  bool load(const uint8_t* buf, size_t len, const LLMTokenizer::Vocab& vocabIn) {
    if (len < sizeof(LLMConfig)) return false;
    memcpy(&cfg, buf, sizeof(LLMConfig));
    vocab = vocabIn;

    const uint8_t* p = buf + sizeof(LLMConfig);
    int head_size = cfg.dim / cfg.n_heads;
    int kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;

    auto readQ = [&](QMatrix& m, size_t n, size_t d) {
      m.data = reinterpret_cast<const int8_t*>(p);
      p += n * d;
      memcpy(&m.scale, p, sizeof(float));
      p += sizeof(float);
    };
    auto readF = [&](const float*& fp, size_t n) {
      fp = reinterpret_cast<const float*>(p);
      p += n * sizeof(float);
    };

    readQ(w.token_embedding, cfg.dim, cfg.vocab_size);

    w.layers = new LayerWeights[cfg.n_layers];
    for (int l = 0; l < cfg.n_layers; l++) {
      LayerWeights& lw = w.layers[l];
      readF(lw.rms_att_weight, cfg.dim);
      readQ(lw.wq, cfg.dim, cfg.n_heads * head_size);
      readQ(lw.wk, cfg.dim, cfg.n_kv_heads * head_size);
      readQ(lw.wv, cfg.dim, cfg.n_kv_heads * head_size);
      readQ(lw.wo, cfg.n_heads * head_size, cfg.dim);
      readF(lw.rms_ffn_weight, cfg.dim);
      readQ(lw.w1, cfg.dim, cfg.hidden_dim);
      readQ(lw.w2, cfg.hidden_dim, cfg.dim);
      readQ(lw.w3, cfg.dim, cfg.hidden_dim);
    }

    readF(w.rms_final_weight, cfg.dim);
    readQ(w.wcls, cfg.dim, cfg.vocab_size);

    if (!allocState(kv_dim)) return false;
    ready = true;
    return true;
  }

  // Reads an entire model file (LittleFS or SD, via the fs::FS handed in)
  // into a heap buffer and parses it the same way as load() above. Only
  // sane for models small enough to fully fit in RAM once -- fine for the
  // model sizes this project targets; streaming/mmap-from-flash-partition
  // is the obvious next step for anything bigger.
  bool loadFromFile(fs::FS& filesystem, const String& path, const LLMTokenizer::Vocab& vocabIn) {
    File f = filesystem.open(path, "r");
    if (!f) return false;
    size_t len = f.size();
    uint8_t* buf = (uint8_t*)
#if defined(ESP32)
      heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) buf = (uint8_t*)malloc(len); // fall back to regular heap if no PSRAM
#else
      malloc(len);
#endif
    if (!buf) { f.close(); return false; }
    f.read(buf, len);
    f.close();
    fileBuf_ = buf; // keep alive for the engine's lifetime; freed in ~Engine()
    return load(buf, len, vocabIn);
  }

  ~Engine() { if (fileBuf_) free(fileBuf_); delete[] w.layers; freeState(); }

  // Runs one forward step for `token` at position `pos` (0-indexed into the
  // sequence), returns a pointer to logits[vocab_size]. Caller is
  // responsible for calling this with pos = 0, 1, 2, ... in order, since
  // the KV cache is written incrementally.
  float* forward(int token, int pos) {
    int dim = cfg.dim, head_size = dim / cfg.n_heads;
    int kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    int kv_mul = cfg.n_heads / cfg.n_kv_heads;

    const int8_t* embRow = w.token_embedding.data + (size_t)token * dim;
    for (int i = 0; i < dim; i++) s.x[i] = embRow[i] * w.token_embedding.scale;

    for (int l = 0; l < cfg.n_layers; l++) {
      LayerWeights& lw = w.layers[l];
      rmsnorm(s.xb, s.x, lw.rms_att_weight, dim);

      float* kCache = s.key_cache + ((size_t)l * cfg.seq_len + pos) * kv_dim;
      float* vCache = s.value_cache + ((size_t)l * cfg.seq_len + pos) * kv_dim;
      qmatmul(s.q, s.xb, lw.wq, dim, dim);
      qmatmul(kCache, s.xb, lw.wk, dim, kv_dim);
      qmatmul(vCache, s.xb, lw.wv, dim, kv_dim);

      // RoPE: rotate each adjacent pair within every head, by an angle that
      // depends on position `pos` and the pair's index within the head.
      for (int i = 0; i < dim; i += 2) {
        int headDimIdx = i % head_size;
        float freq = 1.0f / powf(10000.0f, (float)headDimIdx / head_size);
        float angle = pos * freq;
        float cosv = cosf(angle), sinv = sinf(angle);
        float q0 = s.q[i], q1 = s.q[i + 1];
        s.q[i] = q0 * cosv - q1 * sinv;
        s.q[i + 1] = q0 * sinv + q1 * cosv;
        if (i < kv_dim) {
          float k0 = kCache[i], k1 = kCache[i + 1];
          kCache[i] = k0 * cosv - k1 * sinv;
          kCache[i + 1] = k0 * sinv + k1 * cosv;
        }
      }

      // Multi-head attention against everything cached so far (0..pos).
      for (int h = 0; h < cfg.n_heads; h++) {
        float* qh = s.q + h * head_size;
        float* atth = s.att + h * cfg.seq_len;
        int kvh = h / kv_mul;
        for (int t = 0; t <= pos; t++) {
          float* kt = s.key_cache + ((size_t)l * cfg.seq_len + t) * kv_dim + kvh * head_size;
          float score = 0.0f;
          for (int i = 0; i < head_size; i++) score += qh[i] * kt[i];
          atth[t] = score / sqrtf((float)head_size);
        }
        softmax(atth, pos + 1);
        float* xbh = s.xb2 + h * head_size;
        for (int i = 0; i < head_size; i++) xbh[i] = 0.0f;
        for (int t = 0; t <= pos; t++) {
          float* vt = s.value_cache + ((size_t)l * cfg.seq_len + t) * kv_dim + kvh * head_size;
          for (int i = 0; i < head_size; i++) xbh[i] += atth[t] * vt[i];
        }
      }

      qmatmul(s.xb, s.xb2, lw.wo, dim, dim);
      for (int i = 0; i < dim; i++) s.x[i] += s.xb[i];

      // SwiGLU FFN.
      rmsnorm(s.xb, s.x, lw.rms_ffn_weight, dim);
      qmatmul(s.hb, s.xb, lw.w1, dim, cfg.hidden_dim);
      qmatmul(s.hb2, s.xb, lw.w3, dim, cfg.hidden_dim);
      for (int i = 0; i < cfg.hidden_dim; i++) s.hb[i] = silu(s.hb[i]) * s.hb2[i];
      qmatmul(s.xb, s.hb, lw.w2, cfg.hidden_dim, dim);
      for (int i = 0; i < dim; i++) s.x[i] += s.xb[i];
    }

    rmsnorm(s.x, s.x, w.rms_final_weight, dim);
    qmatmul(s.logits, s.x, w.wcls, dim, cfg.vocab_size);
    return s.logits;
  }

  // Greedy sampling (argmax). temperature > 0 switches to weighted random
  // sampling over the softmax'd logits instead, for less repetitive output.
  int sample(float* logits, float temperature) {
    if (temperature <= 0.0f) {
      int best = 0;
      for (int i = 1; i < cfg.vocab_size; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    for (int i = 0; i < cfg.vocab_size; i++) logits[i] /= temperature;
    softmax(logits, cfg.vocab_size);
    float r = (float)random(0, 10000) / 10000.0f;
    float cum = 0.0f;
    for (int i = 0; i < cfg.vocab_size; i++) {
      cum += logits[i];
      if (r < cum) return i;
    }
    return cfg.vocab_size - 1;
  }

  // Tokenizes `prompt`, runs the prompt through forward() to prime the KV
  // cache, then autoregressively samples up to maxNewTokens more, decoding
  // the whole thing back to text.
  String generate(const String& prompt, int maxNewTokens, float temperature = 0.8f) {
    if (!ready) return "error: model not loaded";
    static const int MAX_TOKENS = 256;
    int32_t ids[MAX_TOKENS];
    int nPrompt = LLMTokenizer::encode(vocab, prompt, ids, MAX_TOKENS);
    if (nPrompt == 0) return "error: prompt encoded to zero tokens";

    int32_t allIds[MAX_TOKENS];
    memcpy(allIds, ids, nPrompt * sizeof(int32_t));
    int total = nPrompt;
    int pos = 0;
    int token = allIds[0];

    while (pos < cfg.seq_len - 1 && total < MAX_TOKENS) {
      float* logits = forward(token, pos);
      pos++;
      if (pos < nPrompt) {
        token = allIds[pos]; // still feeding the prompt
      } else {
        int next = sample(logits, temperature);
        allIds[total++] = next;
        token = next;
        if (total - nPrompt >= maxNewTokens) break;
      }
    }
    return LLMTokenizer::decode(vocab, allIds, total);
  }

private:
  uint8_t* fileBuf_ = nullptr;

  bool allocState(int kv_dim) {
    s.x = (float*)calloc(cfg.dim, sizeof(float));
    s.xb = (float*)calloc(cfg.dim, sizeof(float));
    s.xb2 = (float*)calloc(cfg.dim, sizeof(float));
    s.hb = (float*)calloc(cfg.hidden_dim, sizeof(float));
    s.hb2 = (float*)calloc(cfg.hidden_dim, sizeof(float));
    s.q = (float*)calloc(cfg.dim, sizeof(float));
    s.att = (float*)calloc((size_t)cfg.n_heads * cfg.seq_len, sizeof(float));
    s.logits = (float*)calloc(cfg.vocab_size, sizeof(float));

    size_t cacheSize = (size_t)cfg.n_layers * cfg.seq_len * kv_dim;
#if defined(ESP32)
    s.key_cache = (float*)heap_caps_calloc(cacheSize, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s.value_cache = (float*)heap_caps_calloc(cacheSize, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s.key_cache) s.key_cache = (float*)calloc(cacheSize, sizeof(float));
    if (!s.value_cache) s.value_cache = (float*)calloc(cacheSize, sizeof(float));
#else
    s.key_cache = (float*)calloc(cacheSize, sizeof(float));
    s.value_cache = (float*)calloc(cacheSize, sizeof(float));
#endif
    return s.x && s.xb && s.xb2 && s.hb && s.hb2 && s.q && s.att && s.logits &&
           s.key_cache && s.value_cache;
  }

  void freeState() {
    free(s.x); free(s.xb); free(s.xb2); free(s.hb); free(s.hb2);
    free(s.q); free(s.att); free(s.logits);
    free(s.key_cache); free(s.value_cache);
  }
};

} // namespace LLMEngine

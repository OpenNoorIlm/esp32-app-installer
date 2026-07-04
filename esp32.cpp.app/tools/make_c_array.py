#!/usr/bin/env python3
"""
make_c_array.py -- bin2c for model.bin, so it can be baked directly into
flash as a `const` C array instead of loaded from LittleFS/SD at runtime.

Usage:
  python make_c_array.py model.bin model_data.h
"""
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_c_array.py <input.bin> <output.h>")

    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        data = f.read()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated from {in_path} by tools/make_c_array.py -- do not hand-edit.\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write(f"static const uint32_t esp32_llm_model_len = {len(data)};\n")
        f.write("static const uint8_t esp32_llm_model[] = {\n")
        for i in range(0, len(data), 20):
            chunk = data[i:i + 20]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")

    print(f"wrote {out_path} ({len(data)} bytes -> {len(data)} flash bytes as a C array)")


if __name__ == "__main__":
    main()

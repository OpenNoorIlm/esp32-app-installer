# models/

No model weights are included in this zip — there's no pretrained model
that already matches this engine's exact layout (see `tools/README.md`
for the binary format spec), and any real LLaMA/GPT-scale checkpoint is
many orders of magnitude too big for an ESP32 regardless.

To get something usable here:

1. Train your own tiny transformer (see the sizing guidance in
   `tools/README.md`) — a few hundred K to ~15M parameters, small closed
   vocabulary, short context length.
2. Run it through `tools/quantize_and_convert.py` to get `model.bin` +
   `vocab_data.h`.
3. Either drop `model.bin` here and copy it to `/models/model.bin` on the
   ESP32's filesystem, or run `tools/make_c_array.py` on it to bake it
   into the firmware instead.

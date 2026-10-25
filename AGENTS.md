# AGENTS.md

Voice-controlled elevator project: train tiny MLPs on WAV features in Python, export weights as C float arrays, run inference on an ESP32. No git, no tests, no requirements.txt, no packaging — scripts are run directly from the repo root (they use relative paths like `samples`, `features.npy`).

## Two independent models

- **Digit recognition** — `with_ui.py`: 10 classes, input 84 = 28 frames × 3 features, hidden 48 → 24.
- **Keyword detection** — `keyword_with_ui.py`: 2 classes, input 237 = 79 frames × 3 features, hidden 128 → 64.

Both are tkinter GUIs with the same flow: Select Samples Folder → Load/Split → Train → Test WAV → Export Weights (needs a display). Feature extraction is the same VAD + energy/mean-abs/ZCR pipeline, **copy-pasted into every script** — there is no shared module, so pipeline changes must be made in each file individually or they will drift.

## Data conventions

- `samples/` holds 102 WAVs: 100 numbered digit samples (1.wav…100.wav) plus duplicates `7(1).wav` and `80(1).wav`. Internal labels are `file_index // 10`; the GUI displays `prediction + 1`.
- The keyword GUI labels the first 10 WAVs as keyword (class 1), the rest as not (class 0), and hardcodes split: train = indices {0-6, 10-16}, test = {7-9, 17-19}. It expects ≥ 20 files.
- Every loader sorts files by `int(os.path.splitext(x)[0])` — `samples/` contains `7(1).wav` and `80(1).wav` which crash that sort. Rename/remove them before running any loader.

## Gotchas

- `target_frames` differs per script and must match the model: 28 in `with_ui.py` (→84), 79 in `keyword_with_ui.py` (→237). `extract_all_features.py` uses 32 (→96) and `feature_visulization.py` matches that, despite comments claiming 84/28 — trust the code, not the comments.
- `vad_signal.py` and `get_ave_frames.py` reference `keyword-samples/`, which does not exist (only `mic-test/`, which is empty). They crash as-is; `vad_signal.py` also uses VAD threshold 0.03 vs 0.05 everywhere else.
- The exported scaler is applied in C as `(x - mean) * scale` with `scale = 1/std`. The keyword export uses `scaler.scale_` directly; the digit export computes `1 / sqrt(var_)`. The ESP32 sketch depends on this exact form.
- Weight export writes `const float network_0_weight[...]` etc. from `nn.Sequential` indices (0, 3, 6). If the model architecture changes, the exported array names, dims, and the ESP32 forward pass must all change together.

## ESP32 test

- `esp_tests/Eelevator mk I.ino` does `#include "weights.h"`, which is **not in the repo** — it's the exported file from `keyword_with_ui.py` saved as `weights.h` next to the sketch. `weights/Key_Word_Weights.txt` is the same export already saved.
- The sketch hardcodes a 237-float `test_sample`; it comes from the print output of `keyword_with_ui.py`'s "Select WAV File to Test" (paste the printed values).
- Manual forward pass: 237→128→64→2, ReLU on layers 1–2, raw logits on output, winner takes all. Test LED on GPIO 15.

## Dependencies

- Python: torch, numpy, scipy, scikit-learn, matplotlib, tkinter. Headless environments can still run `extract_all_features.py` → `feature_visulization.py` (batch feature extraction → `features.npy`/`labels.npy` + plots).

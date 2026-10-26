# AGENTS.md

Voice-controlled elevator project: train tiny MLPs on WAV features in Python, export weights as C float arrays, run inference on an ESP32.no tests, no requirements.txt, no packaging — scripts are run directly from the repo root (they use relative paths like `samples`, `features.npy`).
This project is an academic embedded speech-recognition system built around an ESP32-S2 Mini and an INMP441 MEMS I2S microphone.
The overall goal is to implement a small speech-recognition pipeline on the ESP32 using manually trained MLP neural networks whose weights are exported from Python/PyTorch and then executed directly on the microcontroller.

## Project Goal
The system is intended to behave approximately as follows:

-  listen in time intervals through the INMP441 microphone.
- Detect speech (wake word detection) using the trained wake word detection model weights and the small inference algorithm on the esp32 s2 mini.
- Extract simple time-domain speech features for detections and predictions in the model.
- Convert those features into the exact input representation expected by the trained MLP.
- Run the exported MLP weights directly on the ESP32-S2 using and inference.
- Determine whether the spoken word is the wake word.
- If the wake word is detected, the system will subsequently, turn on the integrated led on the board for user intraction and listen for a spoken number.
- A second MLP, already trained separately for numbers 1–10, will classify the number print the predicted number on the serial console and also turn on the predicted led on the board for 2 seconds.
- The number-recognition stage will eventually control the elevator prototype.

** the training models are already implemented with python and should be the main inspiration for other sections. the ml models both train on the sample folder (for numbers now) and export their weights in a .txt file wich can be stored in the esp32 s2 mini and then later used while predicting the number or deciding the wake word has been called.**

## Extraction Model
here is how the voice signal processing and extraction model works:
WAV -> Stereo → Mono -> VAD -> Speech segment -> Energy / Average amplitude / ZCR -> Normalization -> Resize to 79 or 28 (different for wake word and numbers classifire) frames -> Flatten -> StandardScaler -> 237-dimensional vector -> MLP

## Current Feature Extraction
The project deliberately uses simple time-domain features rather than MFCCs or a CNN.
Current parameters:

### Wake Word Classifier
frame_size = 1024
target_frames = 79
vad_threshold = 0.05
hop_size = frame_size // 2

Therefore:

frame size = 1024 samples
hop size   = 512 samples

Three features are calculated for every frame:

- Energy

energy = np.sum(frame ** 2)

- Average absolute amplitude

avg = np.mean(np.abs(frame))

- Zero Crossing Rate

The number of sign changes in the frame is counted.
The three features are stored as:

[energy, average, zcr]

for each frame.

** the full feature extraction process can be found in the wiht_ui.py and used anywhere else.**

## Original Training Data
The original training recordings were made using a laptop and Audacity.
The recordings were:
WAV
44.1 kHz
16-bit PCM
stereo originally

converted to mono during feature extraction, The Python feature extractor performs:

`sample_rate, data = wav.read(path)

if len(data.shape) == 2:
    data = np.mean(data, axis=1)
`
Therefore the actual features are extracted from mono audio.
The original system successfully trained and tested on these WAV files.
Do not assume that the existing MLP is mathematically incorrect simply because the recordings originated from a laptop.

- VAD Implementation
The current Python implementation computes frame energy:
num_frames = (len(data) - frame_size) // (frame_size // 2) + 1

for i in range(num_frames):
    start = i * (frame_size // 2)
    end = start + frame_size
    frame = data[start:end]
    enrg = np.sum(frame ** 2)

The energies are normalized against the maximum frame energy:

nrom_enrg = vad_energies / max(vad_energies)

A frame is considered active when:

nrom_enrg[frame] >= 0.05

The first and last active frames define the speech region.
This is currently a simple energy-based VAD, not a sophisticated speech detector.
*** Any future ESP implementation should reproduce the mathematical behavior of this preprocessing as closely as practical.***

- Reszising
Speech duration is not guaranteed to produce exactly the same number of frames.
The project therefore resizes the VAD-extracted feature sequence to desired values using linear interpolation.

Each frame contains:

3 features

Therefore for example:

79 × 3 = 237

The MLP input size is consequently:

INPUT_SIZE = 237

The feature ordering is:

[e0, a0, z0,
 e1, a1, z1,
 e2, a2, z2,
 ...
 e78, a78, z78]

This ordering is extremely important. The ESP32 implementation must produce the exact same ordering.

## Dataset Split
The current wake-word dataset uses:

7 keyword samples     → training
3 keyword samples     → testing

7 non-keyword samples → training
3 non-keyword samples → testing

 
## ESP S2 MINI implementation
here is the instructions for esp32 s2 mini implementation:

- eps32 implementation codes should be written in the esp_tests folder, later used in arduino ide manually by the developer to export the code inside the board itself.
- weights are also exported manually by developer into the esp32 using the arduino IDE. 
- the esp32 inference should be light and not computationally heavy, since **esp32 s2 mini boards are not computational heavy boards and the amount of computational power and limited memory should be noted allways while making or editing the code. 
- esp code is currently in arduino code and is (.ino) for arduino IDE.
- make sure to always add serial prints of different stages the program is and prediction for developer ease. 


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

## Dependencies

- Python: torch, numpy, scipy, scikit-learn, matplotlib, tkinter. Headless environments can still run `extract_all_features.py` → `feature_visulization.py` (batch feature extraction → `features.npy`/`labels.npy` + plots).

## Rules
- never run git commands or shell commands from outside of the project directory.
- never rename or change names and typos as drive-by clean up.
- never assume values specially ESP32 S2 mini and gpio pins.
- with_ui.py is the whole source of the project (number-classifier), and can be used in anywhere else
- never add changes to the with_ui.py or keyword_with_ui.py without asking me first.

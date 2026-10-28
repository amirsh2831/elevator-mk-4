#include <Arduino.h>
#include <math.h>
#include "driver/i2s.h"
#include "keyword_weights.h"
#include "digit_weights.h"

// ============================================================
// Settings
// ============================================================
constexpr i2s_port_t kPort = I2S_NUM_0;
constexpr int kPinBclk = 5, kPinWs = 7, kPinData = 9;
constexpr int kLedWake = 16, kLedNumber = 15;

constexpr uint32_t kSampleRate = 44100;        // must match the training WAVs
constexpr int kFrameSize = 1024;               // Python: frame_size
constexpr int kHop = 512;                      // Python: frame_size // 2
constexpr float kVadThreshold = 0.05f;         // Python: vad_threshold
constexpr uint32_t kClipSamples = kSampleRate * 2;               // 2 s clip
constexpr int kMaxFrames = (kClipSamples - kFrameSize) / kHop + 1;

constexpr uint32_t kSettleMs = 300;            // mic start-up, discarded
constexpr float kGain = 1.0f / 4.0f;           // use the same gain as your working recorder
constexpr float kDcPole = 0.9986f;             // DC blocker, about 10 Hz at 44.1 kHz
constexpr float kMinPeakRms = 150.0f;          // silence gate; tune it from the printed peakRms

// ============================================================
// Types (all defined before any function, so Arduino's auto-prototypes work)
// ============================================================
struct Layer {
  const float* w;   // [out][in], row-major (PyTorch layout)
  const float* b;
  int in, out;
};

struct Model {
  const char* name;
  const Layer* layers;     // always 3 layers: Linear-ReLU-Linear-ReLU-Linear
  const float* mean;       // StandardScaler.mean_
  const float* scale;      // StandardScaler.scale_ (std). Applied as (x - mean) / scale
  int targetFrames;        // Python: target_frames
  bool normalizeByMax;     // Python: the "x / max" block at the end of extract()
};

#define CHECK_SIZE(arr, count) \
  static_assert(sizeof(arr) == sizeof(float) * (count), #arr " has an unexpected size")
CHECK_SIZE(keyword_network_0_weight, 128 * 237);
CHECK_SIZE(keyword_network_3_weight, 64 * 128);
CHECK_SIZE(keyword_network_6_weight, 2 * 64);
CHECK_SIZE(keyword_scaler_mean, 237);
CHECK_SIZE(keyword_scaler_scale, 237);
CHECK_SIZE(digit_network_0_weight, 48 * 84);
CHECK_SIZE(digit_network_3_weight, 24 * 48);
CHECK_SIZE(digit_network_6_weight, 10 * 24);
CHECK_SIZE(digit_scaler_mean, 84);
CHECK_SIZE(digit_scaler_scale, 84);

static const Layer kKeywordLayers[3] = {
  {(const float*)keyword_network_0_weight, (const float*)keyword_network_0_bias, 237, 128},
  {(const float*)keyword_network_3_weight, (const float*)keyword_network_3_bias, 128, 64},
  {(const float*)keyword_network_6_weight, (const float*)keyword_network_6_bias, 64, 2},
};
static const Layer kDigitLayers[3] = {
  {(const float*)digit_network_0_weight, (const float*)digit_network_0_bias, 84, 48},
  {(const float*)digit_network_3_weight, (const float*)digit_network_3_bias, 48, 24},
  {(const float*)digit_network_6_weight, (const float*)digit_network_6_bias, 24, 10},
};

// CHECK: normalizeByMax for the digit model must match the digit model's own Python extract().
static const Model kKeywordModel = {"keyword", kKeywordLayers,
    (const float*)keyword_scaler_mean, (const float*)keyword_scaler_scale, 79, true};
static const Model kDigitModel = {"digit", kDigitLayers,
    (const float*)digit_scaler_mean, (const float*)digit_scaler_scale, 28, false};

// ============================================================
// Buffers
// ============================================================
static int16_t* gAudio = nullptr;              // 2 s of 16-bit samples
static float gFeat[kMaxFrames][3];             // per frame: energy, mean |x|, zero crossings
static float gInput[79 * 3];                   // resized, flattened, scaled model input

// ============================================================
// 1) Capture: mono 32-bit I2S -> 16-bit samples, DC removed
//    A fresh driver per clip avoids the overflow noise we fixed earlier.
// ============================================================
static bool captureClip(int ledPin) {
  const i2s_config_t cfg = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kSampleRate,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // mic L/R pin on GND
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };
  const i2s_pin_config_t pins = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = kPinBclk,
      .ws_io_num = kPinWs,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = kPinData,
  };
  if (i2s_driver_install(kPort, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(kPort, &pins) != ESP_OK) { i2s_driver_uninstall(kPort); return false; }

  const uint32_t settle = kSampleRate * kSettleMs / 1000;
  const uint32_t total = settle + kClipSamples;
  int32_t raw[256];
  float x1 = 0, y1 = 0;
  uint32_t n = 0;
  bool ok = true, ledOn = false;

  while (n < total) {
    size_t bytes = 0;
    if (i2s_read(kPort, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(500)) != ESP_OK ||
        bytes == 0 || bytes % 4 != 0) { ok = false; break; }
    for (size_t i = 0; i < bytes / 4 && n < total; i++, n++) {
      if (!ledOn && n >= settle) { digitalWrite(ledPin, HIGH); ledOn = true; }
      const float x = (float)(raw[i] >> 8);          // 24-bit sample
      const float y = x - x1 + kDcPole * y1;         // remove DC offset
      x1 = x; y1 = y;
      if (n >= settle) {
        long v = lroundf(y * kGain);
        gAudio[n - settle] = (int16_t)constrain(v, -32768L, 32767L);
      }
    }
  }
  digitalWrite(ledPin, LOW);
  i2s_driver_uninstall(kPort);
  return ok;
}

// ============================================================
// 2) Features. Mirrors Python extract(): VAD on frame energy, then
//    energy / mean|x| / zero-crossings per frame, optionally divided by the clip max.
//    Samples stay in raw int16 units, like scipy's wav.read.
// ============================================================
static int64_t frameEnergy(const int16_t* f) {
  int64_t e = 0;
  for (int j = 0; j < kFrameSize; j++) e += (int32_t)f[j] * f[j];
  return e;
}

// Returns the number of feature frames (Python: length = w_end - w_start), or 0 if silent.
static int extractFeatures(const Model& m, bool verbose) {
  // --- VAD (Python: vad_energies / nrom_enrg) ---
  static int64_t energy[kMaxFrames];
  int64_t maxE = 0;
  for (int n = 0; n < kMaxFrames; n++) {
    energy[n] = frameEnergy(gAudio + (size_t)n * kHop);
    if (energy[n] > maxE) maxE = energy[n];
  }
  const float peakRms = sqrtf((float)maxE / kFrameSize);
  if (verbose) Serial.printf("[%s] peakRms=%.1f (gate %.1f)\n", m.name, peakRms, kMinPeakRms);
  if (peakRms < kMinPeakRms) return 0;               // silence: Python would divide by ~0 here

  int wStart = 0, wEnd = 0;
  for (int n = 0; n < kMaxFrames; n++)
    if ((float)energy[n] / (float)maxE >= kVadThreshold) { wStart = n; break; }
  for (int n = kMaxFrames - 1; n >= 0; n--)
    if ((float)energy[n] / (float)maxE >= kVadThreshold) { wEnd = n; break; }
  const int length = wEnd - wStart;                  // last active frame excluded, like Python
  if (verbose) Serial.printf("[%s] length=%d w_start=%d w_end=%d\n", m.name, length, wStart, wEnd);
  if (length <= 0) return 0;

  // --- Per-frame features ---
  for (int n = 0; n < length; n++) {
    const int16_t* f = gAudio + (size_t)(wStart + n) * kHop;
    int64_t e = 0;
    int32_t absSum = 0;
    int zc = 0;
    for (int a = 0; a < kFrameSize; a++) {
      e += (int32_t)f[a] * f[a];
      absSum += abs((int)f[a]);
      if (a >= 1) {
        const int32_t p = (int32_t)f[a] * f[a - 1];
        if (p < 0) {
          zc++;
        } else if (p == 0) {
          // Python: frame[a] * frame[a-2]. For a == 1 Python's index -2+1 wraps to frame[-1].
          const int16_t before = f[a >= 2 ? a - 2 : kFrameSize - 1];
          if ((int32_t)f[a] * before < 0) zc++;
        }
      }
    }
    gFeat[n][0] = (float)e;
    gFeat[n][1] = (float)absSum / kFrameSize;        // np.mean(np.abs(frame))
    gFeat[n][2] = (float)zc;
  }

  // --- Divide each feature by its max over the clip (Python: max_energy, max_avg, max_zcr) ---
  if (m.normalizeByMax) {
    for (int k = 0; k < 3; k++) {
      float mx = 0;
      for (int n = 0; n < length; n++) if (gFeat[n][k] > mx) mx = gFeat[n][k];
      if (mx <= 0) mx = 1.0f;
      for (int n = 0; n < length; n++) gFeat[n][k] /= mx;
    }
  }
  return length;
}

// Python resize(): linear interpolation along time, then flatten() (frame-major).
static void resizeAndFlatten(int original, int target) {
  const double scale = original > 1 ? (double)(original - 1) / (target - 1) : 0.0;
  for (int i = 0; i < target - 1; i++) {
    const double index = i * scale;
    const int j = (int)index;
    const float wx = (float)(index - j);
    for (int k = 0; k < 3; k++) {
      gInput[i * 3 + k] = (j + 1 < original)
          ? gFeat[j][k] * (1 - wx) + gFeat[j + 1][k] * wx
          : gFeat[j][k];
    }
  }
  for (int k = 0; k < 3; k++) gInput[(target - 1) * 3 + k] = gFeat[original - 1][k];
}

// ============================================================
// 3) Inference. Mirrors scaler.transform() + the MLP in eval mode (dropout off).
// ============================================================
static void dense(const Layer& L, const float* x, float* y, bool relu) {
  for (int o = 0; o < L.out; o++) {
    const float* row = L.w + (size_t)o * L.in;
    float s = L.b[o];
    for (int i = 0; i < L.in; i++) s += row[i] * x[i];
    y[o] = (relu && s < 0) ? 0.0f : s;
  }
}

// Returns the predicted class (argmax of the logits), or -1 if there was no speech.
static int runModel(const Model& m, float* probOut) {
  const int length = extractFeatures(m, true);
  if (length == 0) { Serial.println("No speech detected."); return -1; }

  resizeAndFlatten(length, m.targetFrames);

  const int inputs = m.targetFrames * 3;
  float sumSq = 0;
  for (int i = 0; i < inputs; i++) {
    gInput[i] = (gInput[i] - m.mean[i]) / m.scale[i];   // StandardScaler.transform
    sumSq += gInput[i] * gInput[i];
  }

  static float a[128], b[128], logits[10];
  dense(m.layers[0], gInput, a, true);
  dense(m.layers[1], a, b, true);
  dense(m.layers[2], b, logits, false);

  const int n = m.layers[2].out;
  int best = 0;
  for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
  float sum = 0;
  for (int i = 0; i < n; i++) sum += expf(logits[i] - logits[best]);
  *probOut = 1.0f / sum;

  // inputRms near 1 means the audio looks like the training data. 5+ means it is far outside it.
  Serial.printf("[%s] inputRms=%.2f logits:", m.name, sqrtf(sumSq / inputs));
  for (int i = 0; i < n; i++) Serial.printf(" %.2f", logits[i]);
  Serial.println();
  return best;
}

// ============================================================
// Main
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(kLedWake, OUTPUT);
  pinMode(kLedNumber, OUTPUT);

  gAudio = (int16_t*)(psramFound() ? ps_malloc(kClipSamples * sizeof(int16_t))
                                   : malloc(kClipSamples * sizeof(int16_t)));
  if (!gAudio) {
    Serial.println("ERR: could not allocate the audio buffer");
    while (true) delay(1000);
  }
  Serial.println("Ready.");
}

void loop() {
  float prob = 0;

  Serial.println("\nListening for the wake word (speak while the LED is on)...");
  if (!captureClip(kLedWake)) { Serial.println("ERR: I2S capture failed"); delay(500); return; }
  const int key = runModel(kKeywordModel, &prob);
  Serial.printf("Keyword class: %d (p=%.2f)\n", key, prob);
  if (key != 1) return;                              // class 1 = keyword, as in training

  Serial.println("Wake word detected. Say a number (speak while the LED is on)...");
  if (!captureClip(kLedNumber)) { Serial.println("ERR: I2S capture failed"); delay(500); return; }
  const int digit = runModel(kDigitModel, &prob);
  if (digit >= 0) Serial.printf("Final number: %d (p=%.2f)\n", digit + 1, prob);
}
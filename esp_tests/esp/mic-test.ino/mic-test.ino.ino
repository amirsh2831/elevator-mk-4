/*
 * INMP441 (I2S MEMS mic) live test for ESP32-S2 Mini
 * ---------------------------------------------------------------
 * Requires: "esp32" by Espressif, Arduino core 3.0.0 or newer.
 *
 * Arduino IDE settings (Tools menu):
 *   Board ............ "LOLIN S2 Mini"  (or "ESP32S2 Dev Module")
 *   USB CDC On Boot .. "Enabled"   <-- needed for Serial over the USB port
 *   Serial Monitor ... 115200 baud, line ending: "No line ending" or "Newline"
 *
 * Wiring (INMP441 -> ESP32-S2 Mini). Pins are changeable below.
 *   VDD  -> 3V3        (3.3 V ONLY, never 5 V)
 *   GND  -> GND
 *   SCK  -> GPIO 7     (I2S bit clock, BCLK)
 *   WS   -> GPIO 9     (I2S word select, LRCLK)
 *   SD   -> GPIO 11    (I2S data out of mic -> data in of ESP)
 *   L/R  -> GND        (left channel; connect to 3V3 for right and set MIC_USE_RIGHT 1)
 *
 * Serial commands:
 *   m  level meter (default)        p  level history for Serial Plotter
 *   w  raw waveform for Plotter     t  re-run self-test
 *   +/-  waveform display gain      u/d  LED threshold up/down (3 dB steps)
 *   h  help
 */

#include <Arduino.h>
#include <math.h>
#include "driver/i2s_std.h"

#if ESP_ARDUINO_VERSION_MAJOR < 3
#error "Needs esp32 Arduino core 3.0.0+ (Boards Manager -> esp32 by Espressif Systems)."
#endif

// ----------------------------- Configuration -----------------------------
constexpr int      PIN_I2S_BCLK   = 5;
constexpr int      PIN_I2S_WS     = 7;
constexpr int      PIN_I2S_DIN    = 9;
constexpr int      PIN_LED        = 15;      // onboard LED on the S2 Mini
constexpr bool     MIC_USE_RIGHT  = false;   // true if L/R pin is tied to 3V3

constexpr uint32_t SAMPLE_RATE    = 44200;
constexpr size_t   BLOCK_SAMPLES  = 256;     // 16 ms per block at 16 kHz
constexpr uint32_t REPORT_MS      = 100;     // meter refresh period
constexpr int      WAVE_DECIMATE  = 8;       // print every Nth sample in wave mode (2 kHz)

constexpr float    FULL_SCALE     = 8388608.0f;  // 2^23, INMP441 is 24-bit signed
constexpr float    SPL_OFFSET_DB  = 120.0f;      // sensitivity -26 dBFS @ 94 dB SPL -> ~dBFS + 120
constexpr float    MIN_DB         = -120.0f;

// ------------------------------- State -----------------------------------
enum class Mode : uint8_t { Meter, Plot, Wave };

static i2s_chan_handle_t rxChan = nullptr;
static int32_t  block[BLOCK_SAMPLES];
static Mode     mode          = Mode::Meter;
static float    ledThreshDb   = -40.0f;
static int      waveGain      = 8;           // multiplier for wave display
static float    dcEstimate    = 0.0f;        // slow DC tracker for wave display
static int      decimateCount = 0;

struct Window {                               // accumulators for one report period
  double   sum    = 0;
  double   sumSq  = 0;
  uint32_t count  = 0;
  int32_t  peak   = 0;
  void reset() { *this = Window(); }
};
static Window win;

// ------------------------------ Helpers ----------------------------------
static inline float toDbFs(float amplitude) {
  if (amplitude <= 0.0f) return MIN_DB;
  const float db = 20.0f * log10f(amplitude / FULL_SCALE);
  return db < MIN_DB ? MIN_DB : db;
}

static void printHelp() {
  Serial.println(F("\nCommands: m=meter  p=plot history  w=plot waveform  t=self-test"));
  Serial.println(F("          +/-=wave gain  u/d=LED threshold  h=help"));
}

// ------------------------------- I2S -------------------------------------
static bool micBegin() {
  i2s_chan_config_t chanCfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chanCfg.dma_desc_num  = 6;
  chanCfg.dma_frame_num = BLOCK_SAMPLES;

  esp_err_t err = i2s_new_channel(&chanCfg, nullptr, &rxChan);
  if (err != ESP_OK) {
    Serial.printf("i2s_new_channel failed: %s\n", esp_err_to_name(err));
    return false;
  }

  // INMP441: Philips I2S, 24-bit data left-aligned in a 32-bit slot.
  i2s_std_config_t stdCfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = static_cast<gpio_num_t>(PIN_I2S_BCLK),
      .ws   = static_cast<gpio_num_t>(PIN_I2S_WS),
      .dout = I2S_GPIO_UNUSED,
      .din  = static_cast<gpio_num_t>(PIN_I2S_DIN),
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  stdCfg.slot_cfg.slot_mask = MIC_USE_RIGHT ? I2S_STD_SLOT_RIGHT : I2S_STD_SLOT_LEFT;

  err = i2s_channel_init_std_mode(rxChan, &stdCfg);
  if (err != ESP_OK) {
    Serial.printf("i2s_channel_init_std_mode failed: %s\n", esp_err_to_name(err));
    return false;
  }
  err = i2s_channel_enable(rxChan);
  if (err != ESP_OK) {
    Serial.printf("i2s_channel_enable failed: %s\n", esp_err_to_name(err));
    return false;
  }
  return true;
}

// Reads up to n samples; returns how many were actually read.
static size_t readBlock(int32_t *dst, size_t n, uint32_t timeoutMs = 200) {
  size_t bytesRead = 0;
  const esp_err_t err = i2s_channel_read(rxChan, dst, n * sizeof(int32_t), &bytesRead, timeoutMs);
  if (err != ESP_OK && err != ESP_ERR_TIMEOUT) {
    Serial.printf("i2s read error: %s\n", esp_err_to_name(err));
  }
  return bytesRead / sizeof(int32_t);
}

// ----------------------------- Self-test ---------------------------------
static void runSelfTest() {
  Serial.println(F("\n=== INMP441 self-test (~1.5 s, keep the room quiet) ==="));

  // Mic needs ~85 ms to start up; discard 300 ms to be safe.
  const uint32_t warmupEnd = millis() + 300;
  while (millis() < warmupEnd) readBlock(block, BLOCK_SAMPLES);

  int32_t  minV = INT32_MAX, maxV = INT32_MIN;
  uint32_t zeros = 0, total = 0;
  double   sum = 0, sumSq = 0;

  const uint32_t endMs = millis() + 1000;
  while (millis() < endMs) {
    const size_t n = readBlock(block, BLOCK_SAMPLES);
    for (size_t i = 0; i < n; ++i) {
      const int32_t s = block[i] >> 8;              // 24-bit signed value
      if (s == 0) ++zeros;
      if (s < minV) minV = s;
      if (s > maxV) maxV = s;
      sum   += s;
      sumSq += static_cast<double>(s) * s;
      ++total;
    }
  }

  if (total == 0) {
    Serial.println(F("[FAIL] No samples received. Check SCK/WS wiring and that the pins match the sketch."));
    return;
  }

  const double mean  = sum / total;
  const double var   = sumSq / total - mean * mean;
  const float  rms   = sqrtf(var > 0 ? static_cast<float>(var) : 0.0f);
  const float  noise = toDbFs(rms);

  Serial.printf("samples=%u  min=%ld  max=%ld  DC=%.0f  noise=%.1f dBFS\n",
                static_cast<unsigned>(total), static_cast<long>(minV), static_cast<long>(maxV), mean, noise);

  bool ok = true;
  if (zeros == total) {
    Serial.println(F("[FAIL] All samples are 0. Check SD pin, 3V3/GND power, and the L/R pin (GND=left)."));
    ok = false;
  } else if (minV == maxV) {
    Serial.println(F("[FAIL] Data line is stuck at a constant value (floating or shorted SD?)."));
    ok = false;
  } else if (rms < 2.0f) {
    Serial.println(F("[WARN] Almost no signal variation. Mic may be on the other channel (L/R pin)."));
    ok = false;
  }
  if (fabs(mean) > FULL_SCALE * 0.25) {
    Serial.println(F("[WARN] Very large DC offset; data may be misaligned or noisy wiring."));
    ok = false;
  }
  if (noise > -50.0f && ok) {
    Serial.println(F("[WARN] Noise floor is high for a quiet room. Loud environment, or long/loose wires?"));
    ok = false;
  }
  if (maxV >= 8388607 || minV <= -8388608) {
    Serial.println(F("[WARN] Samples hit full scale (clipping)."));
  }
  Serial.println(ok ? F("[PASS] Mic looks alive. Make some noise in meter mode.")
                    : F("[DONE] See warnings above."));
  Serial.println(F("========================================================="));
  win.reset();
}

// ---------------------------- Reporting ----------------------------------
static void reportMeter(float rmsDb, float peakDb, float dc) {
  constexpr int BAR_LEN = 30;
  int filled = static_cast<int>((rmsDb + 90.0f) / 90.0f * BAR_LEN);
  filled = constrain(filled, 0, BAR_LEN);

  char bar[BAR_LEN + 1];
  for (int i = 0; i < BAR_LEN; ++i) bar[i] = (i < filled) ? '#' : '.';
  bar[BAR_LEN] = '\0';

  Serial.printf("RMS %6.1f dBFS | ~%5.1f dB SPL | peak %6.1f dBFS | DC %7.0f | [%s]\n",
                rmsDb, rmsDb + SPL_OFFSET_DB, peakDb, dc, bar);
}

static void processBlock(size_t n) {
  double bSum = 0, bSumSq = 0;

  for (size_t i = 0; i < n; ++i) {
    const int32_t s = block[i] >> 8;
    bSum   += s;
    bSumSq += static_cast<double>(s) * s;

    const int32_t a = s < 0 ? -s : s;
    if (a > win.peak) win.peak = a;

    if (mode == Mode::Wave) {
      dcEstimate += (s - dcEstimate) * 0.001f;
      if (++decimateCount >= WAVE_DECIMATE) {
        decimateCount = 0;
        const long v = constrain(static_cast<long>((s - dcEstimate) * waveGain / 256.0f), -32768L, 32767L);
        // Fixed top/bottom series keep the plotter axis from rescaling constantly.
        Serial.printf("wave:%ld,top:32767,bottom:-32768\n", v);
      }
    }
  }

  win.sum   += bSum;
  win.sumSq += bSumSq;
  win.count += n;

  // LED follows the AC level of this block.
  const double bMean = bSum / n;
  const double bVar  = bSumSq / n - bMean * bMean;
  const float  bDb   = toDbFs(sqrtf(bVar > 0 ? static_cast<float>(bVar) : 0.0f));
  digitalWrite(PIN_LED, bDb > ledThreshDb ? HIGH : LOW);
}

static void maybeReport() {
  static uint32_t lastMs = 0;
  const uint32_t now = millis();
  if (now - lastMs < REPORT_MS || win.count == 0) return;
  lastMs = now;

  const double mean = win.sum / win.count;
  const double var  = win.sumSq / win.count - mean * mean;
  const float  rmsDb  = toDbFs(sqrtf(var > 0 ? static_cast<float>(var) : 0.0f));
  const float  peakDb = toDbFs(static_cast<float>(win.peak));

  if (mode == Mode::Meter) {
    reportMeter(rmsDb, peakDb, static_cast<float>(mean));
  } else if (mode == Mode::Plot) {
    Serial.printf("rms_dBFS:%.1f,peak_dBFS:%.1f,led_thresh:%.1f\n", rmsDb, peakDb, ledThreshDb);
  }
  win.reset();
}

// ------------------------------ Commands ---------------------------------
static void handleSerial() {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    switch (c) {
      case 'm': mode = Mode::Meter; Serial.println(F("Mode: meter")); break;
      case 'p': mode = Mode::Plot;  Serial.println(F("Mode: plot history (open Serial Plotter)")); break;
      case 'w': mode = Mode::Wave;  Serial.println(F("Mode: waveform (open Serial Plotter)")); break;
      case 't': runSelfTest(); break;
      case '+': waveGain = min(waveGain * 2, 4096); Serial.printf("Wave gain x%d\n", waveGain); break;
      case '-': waveGain = max(waveGain / 2, 1);    Serial.printf("Wave gain x%d\n", waveGain); break;
      case 'u': ledThreshDb = min(ledThreshDb + 3.0f, -3.0f);   Serial.printf("LED threshold %.0f dBFS\n", ledThreshDb); break;
      case 'd': ledThreshDb = max(ledThreshDb - 3.0f, -100.0f); Serial.printf("LED threshold %.0f dBFS\n", ledThreshDb); break;
      case 'h': printHelp(); break;
      default: break;   // ignore newlines etc.
    }
  }
}

// ------------------------------ Arduino ----------------------------------
void setup() {
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);   // wait for USB CDC, but don't hang forever

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  Serial.println(F("\nINMP441 test on ESP32-S2 Mini"));
  Serial.printf("Pins: BCLK=%d WS=%d DIN=%d | %lu Hz | %s channel\n",
                PIN_I2S_BCLK, PIN_I2S_WS, PIN_I2S_DIN,
                static_cast<unsigned long>(SAMPLE_RATE), MIC_USE_RIGHT ? "right" : "left");

  if (!micBegin()) {
    Serial.println(F("I2S init failed. Halting."));
    while (true) delay(1000);
  }

  runSelfTest();
  printHelp();
}

void loop() {
  handleSerial();

  const size_t n = readBlock(block, BLOCK_SAMPLES);
  if (n == 0) {
    static uint32_t lastWarn = 0;
    if (millis() - lastWarn > 2000) {
      lastWarn = millis();
      Serial.println(F("No I2S data (timeout). Check SCK/WS wiring."));
    }
    return;
  }

  processBlock(n);
  maybeReport();
}

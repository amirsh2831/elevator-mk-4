#include <Arduino.h>
#include <math.h>
#include "driver/i2s.h"

// ---------- Hardware ----------
constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr int PIN_SCK = 5;
constexpr int PIN_WS  = 7;
constexpr int PIN_SD  = 9;

// ---------- Audio ----------
constexpr uint32_t SAMPLE_RATE     = 16000;
constexpr uint32_t MIN_DURATION_MS = 100;
constexpr uint32_t MAX_DURATION_MS = 5000;   // 5 s = 160 KB RAM
constexpr uint32_t SETTLE_MS       = 300;    // mic wake-up + DC filter warm-up
constexpr int      MIC_SLOT        = 0;      // 0 or 1; confirm with DIAG
// 24-bit -> 16-bit. 1/256 = true scale, 1/64 = +12 dB. Tune using peaks.
constexpr float    S24_TO_S16      = 1.0f / 4.0f;
constexpr float    DC_POLE         = 0.995f; // DC blocker
constexpr size_t   FRAMES_PER_READ = 128;

constexpr uint8_t MAGIC[4] = {'A', 'U', 'D', '1'};

static bool setupI2S() {
    const i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // mono: L/R pin on GND
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
        .bck_io_num = PIN_SCK,
        .ws_io_num = PIN_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = PIN_SD,
    };
    if (i2s_driver_install(I2S_PORT, &cfg, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) {
        i2s_driver_uninstall(I2S_PORT);
        return false;
    }
    return true;
}

static void teardownI2S() { i2s_driver_uninstall(I2S_PORT); }


static bool capture(int16_t* out, uint32_t wantedFrames) {
    if (!setupI2S()) return false;

    const uint32_t settleFrames = SAMPLE_RATE * SETTLE_MS / 1000;
    const uint32_t totalFrames  = settleFrames + wantedFrames;

    int32_t raw[FRAMES_PER_READ];
    float x1 = 0, y1 = 0;
    uint32_t frame = 0;
    bool ok = true;

    while (frame < totalFrames) {
        size_t bytes = 0;
        if (i2s_read(I2S_PORT, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(500)) != ESP_OK ||
            bytes == 0 || bytes % 4 != 0) {
            ok = false;
            break;
        }
        const size_t frames = bytes / 4;
        for (size_t f = 0; f < frames && frame < totalFrames; f++, frame++) {
            const float x = (float)(raw[f] >> 8);      // 24-bit value
            const float y = x - x1 + DC_POLE * y1;     // DC blocker
            x1 = x;
            y1 = y;
            if (frame >= settleFrames) {
                long v = lroundf(y * S24_TO_S16);
                out[frame - settleFrames] = (int16_t)constrain(v, -32768L, 32767L);
            }
        }
    }
    teardownI2S();
    return ok;
}
// ---------- Transport ----------
static uint32_t crc32(const uint8_t* p, size_t n) {   // same as zlib.crc32
    uint32_t crc = 0xFFFFFFFFu;
    while (n--) {
        crc ^= *p++;
        for (int i = 0; i < 8; i++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static bool writeAll(const uint8_t* p, size_t n) {
    uint32_t lastProgress = millis();
    while (n > 0) {
        const size_t w = Serial.write(p, n < 512 ? n : 512);
        if (w > 0) { p += w; n -= w; lastProgress = millis(); }
        else if (millis() - lastProgress > 2000) return false;
        else delay(1);
    }
    return true;
}

static bool sendRecording(const int16_t* pcm, uint32_t frames) {
    const uint32_t rate = SAMPLE_RATE;
    const uint8_t* bytes = (const uint8_t*)pcm;
    const size_t nbytes = (size_t)frames * sizeof(int16_t);
    const uint32_t crc = crc32(bytes, nbytes);
    return writeAll(MAGIC, 4) &&
           writeAll((const uint8_t*)&rate, 4) &&
           writeAll((const uint8_t*)&frames, 4) &&
           writeAll(bytes, nbytes) &&
           writeAll((const uint8_t*)&crc, 4);
}

// ---------- Commands ----------
static void handleRec(uint32_t ms) {
    ms = constrain(ms, MIN_DURATION_MS, MAX_DURATION_MS);
    const uint32_t frames = SAMPLE_RATE * ms / 1000;

    int16_t* pcm = (int16_t*)malloc(frames * sizeof(int16_t));
    if (!pcm) { Serial.println("ERR:NO_MEMORY"); return; }

    if (!capture(pcm, frames))          Serial.println("ERR:I2S_READ");
    else if (!sendRecording(pcm, frames)) { /* host stopped reading; nothing to do */ }

    free(pcm);
}

// Per-slot stats in 24-bit units: the slot with the mic should show a small
// stable DC and a modest RMS; the empty slot ~0 or constant.
static void handleDiag() {
    if (!setupI2S()) { Serial.println("ERR:I2S_INIT"); return; }
    const uint32_t settle = SAMPLE_RATE * SETTLE_MS / 1000;
    const uint32_t total = settle + SAMPLE_RATE;
    int32_t raw[FRAMES_PER_READ];
    double sum = 0, sq = 0;
    int32_t mn = INT32_MAX, mx = INT32_MIN;
    uint32_t frame = 0, counted = 0;
    bool ok = true;

    while (frame < total) {
        size_t bytes = 0;
        if (i2s_read(I2S_PORT, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(500)) != ESP_OK || bytes % 4) {
            ok = false;
            break;
        }
        for (size_t f = 0; f < bytes / 4 && frame < total; f++, frame++) {
            if (frame < settle) continue;
            const int32_t v = raw[f] >> 8;
            counted++;
            sum += v;
            sq += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    teardownI2S();
    if (!ok || counted == 0) { Serial.println("ERR:I2S_READ"); return; }
    const double mean = sum / counted;
    Serial.printf("LEFT min=%ld max=%ld dc=%.0f rms=%.0f\n",
                  (long)mn, (long)mx, mean, sqrt(sq / counted - mean * mean));
}

static bool readLine(char* buf, size_t cap) {
    static size_t len = 0;
    while (Serial.available()) {
        const char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') { buf[len] = 0; len = 0; return true; }
        if (len < cap - 1) buf[len++] = c;
    }
    return false;
}

void setup() {
    Serial.begin(115200);
    delay(500);
}

void loop() {
    char line[32];
    if (!readLine(line, sizeof(line))) return;

    if (strncmp(line, "REC", 3) == 0) handleRec(strtoul(line + 3, nullptr, 10));
    else if (strcmp(line, "DIAG") == 0) handleDiag();
    else if (strcmp(line, "PING") == 0) Serial.println("PONG");
    else Serial.println("ERR:UNKNOWN_COMMAND");
}
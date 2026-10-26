#include <Arduino.h>
#include "driver/i2s.h"

// ============================================================
// Hardware
// ============================================================

#define I2S_PORT I2S_NUM_0

#define I2S_WS   7
#define I2S_SCK  5
#define I2S_SD   9

// Must match your current training pipeline for now.
#define SAMPLE_RATE 16000

// We output mono 16-bit PCM.
#define OUTPUT_BITS 16

// I2S reads 32-bit containers from the INMP441.
#define I2S_READ_SAMPLES 128

// ============================================================
// I2S configuration
// ============================================================

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        ),

        .sample_rate = SAMPLE_RATE,

        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,

        .communication_format = I2S_COMM_FORMAT_I2S_MSB,

        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 6,
        .dma_buf_len = 512,

        .use_apll = true,

        .tx_desc_auto_clear = false,

        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(
        I2S_PORT,
        &i2s_config,
        0,
        nullptr
    );

    i2s_set_pin(I2S_PORT, &pin_config);

    i2s_zero_dma_buffer(I2S_PORT);
}

// ============================================================
// Capture audio
// ============================================================

void recordAudio(uint32_t durationMs) {
    const uint32_t totalSamples =
        ((uint64_t)SAMPLE_RATE * durationMs) / 1000;

    Serial.printf(
        "BEGIN %lu %lu\n",
        totalSamples,
        (uint32_t)SAMPLE_RATE
    );

    // Give the PC a little time to switch from text parsing
    // to binary reading.
    delay(50);

    int32_t i2sBuffer[I2S_READ_SAMPLES];

    uint32_t samplesRecorded = 0;

    while (samplesRecorded < totalSamples) {
        size_t bytesRead = 0;

        esp_err_t result = i2s_read(
            I2S_PORT,
            i2sBuffer,
            sizeof(i2sBuffer),
            &bytesRead,
            portMAX_DELAY
        );

        if (result != ESP_OK) {
            Serial.println("\nERROR:I2S_READ");
            return;
        }

        uint32_t receivedSamples =
            bytesRead / sizeof(int32_t);

        /*
         * I2S is configured as RIGHT_LEFT, so the buffer is:

             [LEFT, RIGHT, LEFT, RIGHT, ...]

         * The INMP441 only drives one of these slots depending
         * on its L/R pin configuration.
         *
         * We currently select the RIGHT slot.
         *
         * If your microphone's L/R pin is configured for LEFT,
         * change:
         *
         *     i += 2
         *
         * to:
         *
         *     i = 0
         */
        for (
    uint32_t i = 0;
    i < receivedSamples && samplesRecorded < totalSamples;
    i += 2
) {
            // INMP441 gives 24-bit data inside a 32-bit I2S word.
            // Keep the same conversion currently used by the project.
            int16_t sample =
                (int16_t)(i2sBuffer[i] >> 16);

            Serial.write(
                (uint8_t*)&sample,
                sizeof(sample)
            );

            samplesRecorded++;
        }
    }

    // Make sure all binary data has physically left the serial buffer.
    Serial.flush();

    delay(20);

    Serial.println();
    Serial.println("END");
}

// ============================================================
// Arduino setup
// ============================================================

void setup() {
    Serial.begin(921600);

    delay(1000);

    setupI2S();

    Serial.println();
    Serial.println("ESP32 AUDIO RECORDER");
    Serial.println("READY");
    Serial.printf("SAMPLE_RATE=%d\n", SAMPLE_RATE);
}

// ============================================================
// Main loop
// ============================================================

void loop() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');

        command.trim();

        if (command.startsWith("REC")) {
            /*
             * Expected:
             *
             * REC 2000
             *
             * where 2000 = milliseconds
             */

            int spaceIndex = command.indexOf(' ');

            uint32_t durationMs = 2000;

            if (spaceIndex > 0) {
                durationMs =
                    command.substring(spaceIndex + 1).toInt();
            }

            // Reasonable safety limits.
            if (durationMs < 100) {
                durationMs = 100;
            }

            if (durationMs > 10000) {
                durationMs = 10000;
            }

            recordAudio(durationMs);
        }

        else if (command == "PING") {
            Serial.println("PONG");
        }

        else {
            Serial.println("ERROR:UNKNOWN_COMMAND");
        }
    }
}
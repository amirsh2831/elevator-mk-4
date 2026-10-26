import argparse
import struct
import wave
from pathlib import Path

import serial


BAUD_RATE = 921600
BITS_PER_SAMPLE = 16
CHANNELS = 1


def read_line(ser):
    line = ser.readline()

    if not line:
        raise RuntimeError("ESP32 stopped responding.")

    return line.decode("ascii", errors="replace").strip()


def record(port, output, duration_ms):
    output = Path(output)

    print(f"Opening {port}...")

    with serial.Serial(
        port=port,
        baudrate=BAUD_RATE,
        timeout=5,
    ) as ser:

        # Give USB serial time to settle.
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("Checking ESP32...")

        ser.write(b"PING\n")
        ser.flush()

        response = read_line(ser)

        if response != "PONG":
            raise RuntimeError(
                f"Unexpected ESP32 response: {response!r}"
            )

        print(f"Recording {duration_ms} ms...")
        print("Speak now.")

        ser.reset_input_buffer()

        ser.write(f"REC {duration_ms}\n".encode())
        ser.flush()

        # --------------------------------------------------
        # Read recording header
        # --------------------------------------------------

        header = read_line(ser)

        print(f"ESP32: {header}")

        parts = header.split()

        if len(parts) != 3 or parts[0] != "BEGIN":
            raise RuntimeError(
                f"Invalid recording header: {header!r}"
            )

        sample_count = int(parts[1])
        sample_rate = int(parts[2])

        expected_bytes = sample_count * 2

        print(
            f"Samples:     {sample_count:,}\n"
            f"Sample rate: {sample_rate:,} Hz\n"
            f"PCM bytes:   {expected_bytes:,}"
        )

        # --------------------------------------------------
        # Read binary PCM
        # --------------------------------------------------

        pcm = bytearray()

        while len(pcm) < expected_bytes:
            chunk = ser.read(
                min(
                    4096,
                    expected_bytes - len(pcm)
                )
            )

            if not chunk:
                raise RuntimeError(
                    "Timeout while receiving audio."
                )

            pcm.extend(chunk)

        print(f"Received {len(pcm):,} bytes.")

        # --------------------------------------------------
        # Read END marker
        # --------------------------------------------------

        # end_line = read_line(ser)
        #
        # if end_line != "END":
        #     raise RuntimeError(
        #         f"Expected END marker, got: {end_line!r}"
        #     )

        # --------------------------------------------------
        # Save WAV
        # --------------------------------------------------

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(BITS_PER_SAMPLE // 8)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)

        print()
        print("Recording saved:")
        print(output.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Record audio from ESP32 + INMP441"
    )

    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="ESP32 serial port"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=2000,
        help="Recording duration in milliseconds"
    )

    parser.add_argument(
        "--output",
        default="recordings/test.wav",
        help="Output WAV file"
    )

    args = parser.parse_args()

    record(
        port=args.port,
        output=args.output,
        duration_ms=args.duration,
    )


if __name__ == "__main__":
    main()

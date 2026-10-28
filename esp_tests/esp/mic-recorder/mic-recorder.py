import argparse
import struct
import time
import wave
import zlib
from pathlib import Path

import serial

MAGIC = b"AUD1"
HEADER = struct.Struct("<II")        # sample_rate, sample_count
MAX_SAMPLES = 16000 * 10


def read_exact(ser, n, deadline):
    buf = bytearray()
    while len(buf) < n:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out: got {len(buf)}/{n} bytes")
        buf.extend(ser.read(min(4096, n - len(buf))))
    return bytes(buf)


def sync_to_magic(ser, deadline):
    window = bytearray()
    while not window.endswith(MAGIC):
        if time.monotonic() > deadline:
            raise TimeoutError(f"No frame header. Last bytes: {bytes(window[-32:])!r}")
        b = ser.read(1)
        if b:
            window += b
            del window[:-64]


def analyze(pcm):
    s = memoryview(pcm).cast("h")
    n = len(s)
    mean = sum(s) / n
    rms = (sum(v * v for v in s) / n) ** 0.5
    peak = max(max(s), -min(s))
    clipped = sum(1 for v in s if v in (32767, -32768))
    print(f"peak={peak}  rms={rms:.0f}  dc={mean:.0f}  clipped={clipped} ({100 * clipped / n:.2f}%)")
    if clipped:
        print("WARNING: clipping. Lower the gain (S24_TO_S16) or check wiring.")
    if peak < 300:
        print("WARNING: very quiet. Raise the gain or check the mic.")


def record(port, output, duration_ms):
    with serial.Serial(port, baudrate=115200, timeout=0.2) as ser:
        ser.reset_input_buffer()
        ser.write(f"REC {duration_ms}\n".encode())
        ser.flush()

        deadline = time.monotonic() + duration_ms / 1000 + 5
        sync_to_magic(ser, deadline)

        rate, count = HEADER.unpack(read_exact(ser, HEADER.size, deadline))
        if not (0 < count <= MAX_SAMPLES) or not (8000 <= rate <= 48000):
            raise RuntimeError(f"Implausible header: rate={rate} count={count}")

        pcm = read_exact(ser, count * 2, deadline)
        (crc,) = struct.unpack("<I", read_exact(ser, 4, deadline))

    if zlib.crc32(pcm) != crc:
        raise RuntimeError("CRC mismatch: audio corrupted in transit. Not saving.")

    analyze(pcm)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    print(f"Saved {output.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Record audio from ESP32 + INMP441")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--duration", type=int, default=2000, help="milliseconds")
    p.add_argument("--output", type=Path, default=Path("recordings/test.wav"))
    a = p.parse_args()
    record(a.port, a.output, a.duration)


if __name__ == "__main__":
    main()

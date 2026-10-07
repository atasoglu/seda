"""Transcribe a 16-bit PCM WAV file in streaming fashion.

Usage: uv run python examples/transcribe.py audio.wav
"""

import sys

from sedalib import Recognizer, read_wav

CHUNK_SECONDS = 0.64


def main() -> None:
    samples, sample_rate = read_wav(sys.argv[1])
    recognizer = Recognizer.from_pretrained()
    stream = recognizer.stream()

    step = int(sample_rate * CHUNK_SECONDS)
    for i in range(0, len(samples), step):
        partial = stream.accept(samples[i : i + step], sample_rate)
        print(f"\r{partial}", end="", flush=True)
    print(f"\r{stream.finish()}")


if __name__ == "__main__":
    main()

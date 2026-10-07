"""Transcribe speech from the default microphone in real time.

Usage: uv run --group examples python examples/streaming.py
"""

from array import array

import sounddevice as sd

from sedalib import SAMPLE_RATE, Recognizer

BLOCK_SECONDS = 0.1


def main() -> None:
    recognizer = Recognizer.from_pretrained()
    stream = recognizer.stream()

    print("Listening... press Ctrl+C to stop.")
    block = int(SAMPLE_RATE * BLOCK_SECONDS)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as mic:
        try:
            while True:
                data, _ = mic.read(block)
                partial = stream.accept(array("f", bytes(data)), SAMPLE_RATE)
                print(f"\r{partial}", end="", flush=True)
                if stream.is_endpoint:
                    if partial:
                        print()
                    stream.reset()
        except KeyboardInterrupt:
            print(f"\r{stream.finish()}")


if __name__ == "__main__":
    main()

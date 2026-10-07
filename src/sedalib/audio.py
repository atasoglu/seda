import wave
from array import array
from pathlib import Path


def read_wav(path: str | Path) -> tuple[list[float], int]:
    """Read a 16-bit PCM WAV file and return mono float samples in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        channels = f.getnchannels()
        sample_rate = f.getframerate()
        pcm = array("h", f.readframes(f.getnframes()))

    samples = [s / 32768 for s in pcm]
    if channels > 1:
        samples = [
            sum(samples[i : i + channels]) / channels
            for i in range(0, len(samples), channels)
        ]
    return samples, sample_rate

import wave
from array import array
from collections.abc import Sequence
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


def write_wav(path: str | Path, samples: Sequence[float], sample_rate: int) -> None:
    """Write mono float samples in [-1, 1] as a 16-bit PCM WAV file."""
    pcm = array("h", (round(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())

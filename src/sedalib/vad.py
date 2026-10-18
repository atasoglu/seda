from __future__ import annotations

from array import array
from collections.abc import Sequence

import sherpa_onnx

VAD_SAMPLE_RATE = 16000


class EndOfSpeech:
    """Detects the end of an utterance with Silero VAD, independently of the recognizer's chunk size.

    Every ~36 ms window of audio is classified as speech or not. An utterance starts after
    ``min_speech`` seconds of speech windows and ends after ``silence`` seconds of non-speech windows.
    """

    def __init__(
        self,
        model: str,
        *,
        silence: float,
        min_speech: float = 0.25,
        threshold: float = 0.5,
    ):
        self._model = sherpa_onnx.VadModel.create(
            sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=model, threshold=threshold
                ),
                sample_rate=VAD_SAMPLE_RATE,
                num_threads=1,
            )
        )
        self._window = self._model.window_size()
        seconds = self._window / VAD_SAMPLE_RATE
        self._need_speech = max(1, round(min_speech / seconds))
        self._need_silence = max(1, round(silence / seconds))
        self.reset()

    def reset(self) -> None:
        self._model.reset()
        self._buffer = array("f")
        self._speech = self._quiet = 0
        self._active = False
        self.ended = False
        self._resample = _Resampler()

    def accept(self, samples: Sequence[float], sample_rate: int) -> None:
        self._buffer.extend(self._resample(samples, sample_rate))
        w = self._window
        consumed = 0
        while len(self._buffer) - consumed >= w:
            window = self._buffer[consumed : consumed + w].tolist()
            consumed += w
            if self._model.is_speech(window):
                self._speech, self._quiet = self._speech + 1, 0
                self._active = self._active or self._speech >= self._need_speech
            else:
                self._speech, self._quiet = 0, self._quiet + 1
                if self._active and self._quiet >= self._need_silence:
                    self.ended = True
                    self._active = False
        del self._buffer[:consumed]


class _Resampler:
    """To 16 kHz by linear interpolation, across calls: enough for a VAD, and no numpy needed."""

    def __init__(self):
        # Input rate, position of the next output sample, last input sample.
        self._rate = VAD_SAMPLE_RATE
        self._position = 0.0
        self._previous: list[float] = []

    def __call__(self, samples: Sequence[float], sample_rate: int) -> array:
        if sample_rate == VAD_SAMPLE_RATE:
            return array("f", samples)
        if sample_rate != self._rate:
            self._rate, self._position, self._previous = sample_rate, 0.0, []
        x = self._previous + list(samples)
        if len(x) < 2:
            self._previous = x
            return array("f")
        step = sample_rate / VAD_SAMPLE_RATE
        out = array("f")
        t, last = self._position, len(x) - 1
        while t < last:
            i = int(t)
            out.append(x[i] + (x[i + 1] - x[i]) * (t - i))
            t += step
        self._position = t - last
        self._previous = [x[-1]]
        return out

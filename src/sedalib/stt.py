from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import sherpa_onnx

from .hub import download

DEFAULT_REPO_ID = "atasoglu/seda-v0.1"
SAMPLE_RATE = 16000
_ACOUSTIC_FILES = [
    "encoder.int8.onnx",
    "decoder.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
]
_LM_FILES = ["lm/rnnlm.int8.onnx", "lm/2gram.fst"]
_TAIL_PADDING_SECONDS = 0.66


class Stream:
    """A single streaming recognition session."""

    def __init__(self, recognizer: sherpa_onnx.OnlineRecognizer):
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()
        self._sample_rate = SAMPLE_RATE

    def accept(self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE) -> str:
        """Feed mono float32 samples in [-1, 1] and return the current partial text."""
        self._sample_rate = sample_rate
        self._stream.accept_waveform(sample_rate, samples)
        self._decode()
        return self.text

    def finish(self) -> str:
        """Flush remaining audio and return the final text."""
        padding = [0.0] * int(self._sample_rate * _TAIL_PADDING_SECONDS)
        self._stream.accept_waveform(self._sample_rate, padding)
        self._stream.input_finished()
        self._decode()
        return self.text

    def reset(self) -> None:
        """Clear the decoder state, e.g. after an endpoint."""
        self._recognizer.reset(self._stream)

    @property
    def text(self) -> str:
        return self._recognizer.get_result(self._stream).strip()

    @property
    def is_endpoint(self) -> bool:
        return self._recognizer.is_endpoint(self._stream)

    def _decode(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)


class Recognizer:
    """Streaming Turkish speech recognizer."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        use_lm: bool = True,
        num_threads: int = 1,
    ):
        d = Path(model_dir)
        lm_args = (
            {
                "lm": str(d / "lm/rnnlm.int8.onnx"),
                "lm_scale": 0.6,
                "lodr_fst": str(d / "lm/2gram.fst"),
                "lodr_scale": -0.5,
            }
            if use_lm
            else {}
        )
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            **lm_args,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        revision: str | None = None,
        use_lm: bool = True,
        num_threads: int = 1,
    ) -> Recognizer:
        """Download the model from the Hugging Face Hub (cached) and load it."""
        files = _ACOUSTIC_FILES + (_LM_FILES if use_lm else [])
        model_dir = download(repo_id, revision=revision, files=files)
        return cls(model_dir, use_lm=use_lm, num_threads=num_threads)

    def stream(self) -> Stream:
        return Stream(self._recognizer)

    def transcribe(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE
    ) -> str:
        """Transcribe a complete mono float32 waveform in [-1, 1]."""
        stream = self.stream()
        stream.accept(samples, sample_rate)
        return stream.finish()

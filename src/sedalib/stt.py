from __future__ import annotations

import math
import queue
import threading
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sherpa_onnx

from .hub import download
from .vad import EndOfSpeech

DEFAULT_REPO_ID = "atasoglu/seda-v0.2"
SAMPLE_RATE = 16000
# Presets of (use_lm, two_pass, chunk).
#   fast:     greedy search on 0.64 s chunks.
#   balanced: beam search with the language model on 1.28 s chunks, the more accurate streaming model.
#   accurate: live partials from 0.64 s chunks; each finished utterance is decoded again with full
#             context, so the final text does not depend on the chunk size.
MODES = {
    "fast": (False, False, 32),
    "balanced": (True, False, 64),
    "accurate": (True, True, 32),
}
# Streaming exports of the same weights: chunk frames (10 ms each, before 2x subsampling) -> subdir.
# The 1.28 s model sits at the repo root, where sedalib 0.2 looks for it. 0.16 and 0.32 s react fastest
# and suit slow devices with greedy search; not every model ships them.
CHUNKS = {64: "", 32: "c32", 16: "c16", 8: "c8"}
_STREAMING_FILES = ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"]
_LM_FILES = ["lm/rnnlm.int8.onnx", "lm/2gram.fst"]
_FINAL_FILES = [
    "final/encoder.int8.onnx",
    "final/decoder.onnx",
    "final/joiner.int8.onnx",
]
_FORMATTER_FILES = ["formatter/formatter.int8.onnx", "formatter/tokenizer.json"]
# Silero VAD v6 (16 kHz, MIT), optional: models that ship it detect the end of speech with it.
_VAD_FILE = "vad/silero_vad.onnx"
# Silence appended to flush the end of an utterance (finish(), end_utterance()): a chunk is decoded
# only once it is full, so the padding covers one chunk plus the encoder's right context (~0.14 s).
_TAIL_PADDING_SECONDS = {64: 1.5, 32: 0.9, 16: 0.6, 8: 0.5}
# Pause after speech, in seconds, that makes is_endpoint true: with the VAD, which decides every
# ~36 ms; without it, sherpa-onnx's rule 2, which is checked once per decoded chunk.
_ENDPOINT_SILENCE = {True: 0.25, False: 1.2}
# The second-pass model reads 5.12 s chunks, each with 20.48 s of left context. It is fed while the
# utterance is spoken, so at its end only the last chunk is left to decode. A chunk is decoded once it
# is full: the end is padded with silence up to the chunk boundary plus the encoder's right context.
_FINAL_CHUNK_SECONDS = 5.12
_FINAL_MARGIN_SECONDS = 0.5
_BEAM = 4
_FINAL_BEAM = 8
_LM_SCALE = 0.6
_LODR_SCALE = -0.5
# One encoder output frame: 10 ms features, subsampled 4x. A token's timestamp is the frame it was
# emitted at, so a word ends about one frame after its last token.
_FRAME_SECONDS = 0.04
# Optional: logistic regressions from word features to the probability that the word is right, fitted
# per decoding setup on held-out speech. Models without it, or setups it lacks, give no confidence.
_CALIBRATION_FILE = "calibration.json"
_FEATURE_VERSION = 1


@dataclass(frozen=True)
class Word:
    """A recognized word, in spoken form, with its time in seconds from the start of the audio.

    ``score`` is the geometric mean of the model's probabilities for the word's tokens. It ranks words
    (higher is more reliable) but is not the probability that the word is right. ``confidence`` is
    that probability, calibrated on held-out read and broadcast speech; it is None when the model has
    no calibration for the decoding setup in use.
    """

    text: str
    start: float
    end: float
    score: float
    confidence: float | None = None


def _features(log_probs: list[float], index: int, count: int) -> list[float]:
    """Word features of calibration feature version 1: mean and minimum token log probability, the
    minimum squared (with the LM, a token the acoustic model found unlikely is often right), the
    number of tokens, and whether the word is the first or the last of the utterance."""
    low = min(log_probs)
    return [
        sum(log_probs) / len(log_probs),
        low,
        low * low,
        float(len(log_probs)),
        float(index == 0),
        float(index == count - 1),
    ]


def _calibration_key(chunk: int, use_lm: bool, two_pass: bool) -> str:
    # The final text of the two-pass setup comes from the second pass, whatever the streaming chunk.
    return f"final-lm{int(use_lm)}" if two_pass else f"chunk{chunk}-lm{int(use_lm)}"


def _load_calibration(
    path: Path, chunk: int, use_lm: bool, two_pass: bool
) -> tuple[list[float], float] | None:
    if not path.is_file():
        return None
    import json

    cal = json.loads(path.read_text(encoding="utf-8"))
    if cal.get("feature_version") != _FEATURE_VERSION:
        return None
    c = cal["configs"].get(_calibration_key(chunk, use_lm, two_pass))
    return (c["weights"], c["bias"]) if c else None


def _confidence(
    calibration: tuple[list[float], float] | None,
    log_probs: list[float],
    index: int,
    count: int,
) -> float | None:
    if calibration is None:
        return None
    weights, bias = calibration
    z = bias + sum(w * f for w, f in zip(weights, _features(log_probs, index, count)))
    return 1 / (1 + math.exp(-z))


@dataclass(frozen=True)
class _Tokens:
    """The tokens of a decoding result: text pieces (a leading space starts a word), times relative
    to the start of the stream, and log probabilities."""

    tokens: tuple[str, ...] = ()
    times: tuple[float, ...] = ()
    log_probs: tuple[float, ...] = ()

    @classmethod
    def of(cls, recognizer: sherpa_onnx.OnlineRecognizer, stream) -> _Tokens:
        r = recognizer.get_result_all(stream)
        # After a reset, timestamps count from the reset (start_time).
        return cls(
            tuple(r.tokens),
            tuple(r.start_time + t for t in r.timestamps),
            tuple(r.ys_probs),
        )

    def words(
        self,
        offset: float = 0.0,
        calibration: tuple[list[float], float] | None = None,
    ) -> list[Word]:
        groups: list[tuple[str, list[float], list[float]]] = []
        for tok, t, lp in zip(self.tokens, self.times, self.log_probs):
            if tok.startswith(" ") or not groups:
                groups.append((tok.strip(), [t], [lp]))
            else:
                text, times, lps = groups[-1]
                groups[-1] = (text + tok, times + [t], lps + [lp])
        groups = [g for g in groups if g[0]]
        return [
            Word(
                text,
                round(offset + times[0], 2),
                round(offset + times[-1] + _FRAME_SECONDS, 2),
                math.exp(sum(lps) / len(lps)),
                _confidence(calibration, lps, i, len(groups)),
            )
            for i, (text, times, lps) in enumerate(groups)
        ]


def _resolve(
    mode: str, use_lm: bool | None, two_pass: bool | None, chunk: int | None
) -> tuple[bool, bool, int]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    if chunk is not None and chunk not in CHUNKS:
        raise ValueError(
            f"chunk must be one of {', '.join(map(str, CHUNKS))}, got {chunk!r}"
        )
    lm, tp, ch = MODES[mode]
    return (
        lm if use_lm is None else use_lm,
        tp if two_pass is None else two_pass,
        ch if chunk is None else chunk,
    )


def _streaming_file(chunk: int, name: str) -> str:
    return f"{CHUNKS[chunk]}/{name}" if CHUNKS[chunk] else name


def _required_files(
    use_lm: bool, two_pass: bool, chunk: int, punctuate: bool
) -> list[str]:
    return (
        [_streaming_file(chunk, f) for f in _STREAMING_FILES]
        + ["tokens.txt"]
        + (_LM_FILES if use_lm else [])
        + (_FINAL_FILES if two_pass else [])
        + (_FORMATTER_FILES if punctuate else [])
    )


def _decode(recognizer: sherpa_onnx.OnlineRecognizer, stream) -> None:
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)


class _FinalPass:
    """The second pass over one utterance, fed in a background thread while the utterance is spoken."""

    def __init__(self, recognizer: sherpa_onnx.OnlineRecognizer):
        self._recognizer = recognizer
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._done = threading.Event()
        self._text = ""
        self._tokens = _Tokens()
        threading.Thread(target=self._run, daemon=True).start()

    def accept(self, sample_rate: int, samples: Sequence[float]) -> None:
        self._queue.put((sample_rate, array("f", samples)))

    def finish(self) -> None:
        """No more audio: decode the rest. Safe to call more than once."""
        self._queue.put(None)

    def result(self) -> str:
        self.finish()
        self._done.wait()
        return self._text

    def tokens(self) -> _Tokens:
        self.result()
        return self._tokens

    def _run(self) -> None:
        stream = self._recognizer.create_stream()  # used by this thread only
        sample_rate, seconds = SAMPLE_RATE, 0.0
        while (item := self._queue.get()) is not None:
            sample_rate, samples = item
            seconds += len(samples) / sample_rate
            stream.accept_waveform(sample_rate, samples)
            _decode(self._recognizer, stream)
        if seconds:
            end = math.ceil((seconds + _FINAL_MARGIN_SECONDS) / _FINAL_CHUNK_SECONDS)
            pad = end * _FINAL_CHUNK_SECONDS - seconds + _FINAL_MARGIN_SECONDS
            stream.accept_waveform(
                sample_rate, array("f", bytes(4 * int(pad * sample_rate)))
            )
            stream.input_finished()
            _decode(self._recognizer, stream)
            self._text = self._recognizer.get_result(stream).strip()
            self._tokens = _Tokens.of(self._recognizer, stream)
        self._done.set()


class Stream:
    """A single streaming recognition session.

    Its methods may be called from different threads (e.g. a web server's audio and stop handlers);
    they run one at a time.
    """

    def __init__(self, recognizer: Recognizer):
        # sherpa-onnx crashes if two threads use one stream at once.
        self._lock = threading.RLock()
        self._owner = recognizer
        self._recognizer = recognizer._streaming
        self._stream = self._recognizer.create_stream()
        self._sample_rate = SAMPLE_RATE
        self._padding_seconds = _TAIL_PADDING_SECONDS[recognizer.chunk]
        self._vad = recognizer._end_of_speech()
        self._final = recognizer._final_pass()
        # Seconds of audio accepted, and where the current utterance starts: word times count from
        # the start of the stream.
        self._seconds = 0.0
        self._start = 0.0

    def accept(self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE) -> str:
        """Feed mono float32 samples in [-1, 1] and return the current partial text."""
        with self._lock:
            self._sample_rate = sample_rate
            self._seconds += len(samples) / sample_rate
            if self._final:
                self._final.accept(sample_rate, samples)
            if self._vad:
                self._vad.accept(samples, sample_rate)
            self._stream.accept_waveform(sample_rate, samples)
            _decode(self._recognizer, self._stream)
            return self.text

    def finish(self) -> str:
        """Flush remaining audio and return the final text."""
        return self._finish().finalize()

    def _finish(self) -> Utterance:
        with self._lock:
            self._flush()
            self._stream.input_finished()
            _decode(self._recognizer, self._stream)
            return Utterance(self._owner, self._stream, self._final, self._start)

    def cut(self) -> Utterance:
        """End the current utterance and start a new one right away, e.g. at an endpoint or when the
        application knows that the speaker stopped (push-to-talk).

        The audio accepted so far is flushed first, so the last word is never cut off. The returned
        utterance's final text (second pass, formatter) can then be computed in another thread while
        this stream keeps accepting audio.
        """
        with self._lock:
            if not self._final:  # the second pass has the audio already
                self._flush()
            utterance = Utterance(self._owner, self._stream, self._final, self._start)
            if self._final:
                self._final.finish()
            self._stream = self._recognizer.create_stream()
            self._final = self._owner._final_pass()
            self._start = self._seconds
            if self._vad:
                self._vad.reset()
            return utterance

    def end_utterance(self) -> str:
        """The final text of the current utterance, and start a new one: ``cut().finalize()``."""
        return self.cut().finalize()

    def reset(self) -> None:
        """Drop the current utterance, including audio that is not decoded yet."""
        with self._lock:
            self._stream = self._recognizer.create_stream()
            if self._final:
                self._final.finish()
                self._final = self._owner._final_pass()
            self._start = self._seconds
            if self._vad:
                self._vad.reset()

    @property
    def text(self) -> str:
        """Partial text of the current utterance from the streaming pass, in spoken form."""
        with self._lock:
            return self._recognizer.get_result(self._stream).strip()

    @property
    def is_endpoint(self) -> bool:
        """True once the speaker paused for ``endpoint_silence`` seconds after speaking."""
        with self._lock:
            if self._vad:
                return self._vad.ended
            if not self._recognizer.is_endpoint(self._stream):
                return False
            if self.text:
                return True
            # A long silence before any speech (sherpa-onnx's rule 1) is not an utterance. Restart the
            # endpoint counters, so idle time does not add up to the maximum utterance length (rule 3);
            # this keeps audio that is not decoded yet, which may hold the start of the next words.
            self._recognizer.reset(self._stream)
            return False

    def __del__(self):
        final = getattr(self, "_final", None)
        if final:  # let its thread end
            final.finish()

    def _flush(self) -> None:
        padding = array("f", bytes(4 * int(self._sample_rate * self._padding_seconds)))
        self._stream.accept_waveform(self._sample_rate, padding)
        _decode(self._recognizer, self._stream)


class Utterance:
    """A finished utterance, cut from a :class:`Stream`."""

    def __init__(
        self,
        recognizer: Recognizer,
        stream,
        final: _FinalPass | None,
        offset: float = 0.0,
    ):
        self._owner = recognizer
        self._stream = stream
        self._final = final
        self._offset = offset

    @property
    def text(self) -> str:
        """Its text from the streaming pass, in spoken form."""
        return self._owner._streaming.get_result(self._stream).strip()

    @property
    def words(self) -> list[Word]:
        """The words of the final text, in spoken form (the formatter does not apply), with times
        from the start of the stream. Waits for the second pass, if it is enabled."""
        tokens = (
            self._final.tokens()
            if self._final
            else _Tokens.of(self._owner._streaming, self._stream)
        )
        return tokens.words(self._offset, self._owner._calibration)

    def finalize(self) -> str:
        """The final text: the second pass and the formatter run here, if they are enabled."""
        text = self._final.result() if self._final else self.text
        return self._owner._format(text)


class Recognizer:
    """Streaming Turkish speech recognizer.

    ``mode`` picks a preset: "fast", "balanced" (default) or "accurate". ``use_lm``, ``two_pass`` and
    ``chunk`` (8, 16, 32 or 64 frames: 0.16, 0.32, 0.64 or 1.28 s of audio per streaming step) override the preset's
    choice, and ``punctuate`` adds digits, punctuation and casing to final texts (needs the ``format``
    extra).

    ``vad`` detects the end of speech with the model's Silero VAD (default: if the model has one),
    and ``endpoint_silence`` is the pause, in seconds, after which ``Stream.is_endpoint`` reports it
    (default 0.25 s with the VAD, 1.2 s without).
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        mode: str = "balanced",
        use_lm: bool | None = None,
        two_pass: bool | None = None,
        chunk: int | None = None,
        punctuate: bool = False,
        num_threads: int = 1,
        vad: bool | None = None,
        endpoint_silence: float | None = None,
    ):
        d = Path(model_dir)
        self.vad = (d / _VAD_FILE).is_file() if vad is None else vad
        self.endpoint_silence = (
            _ENDPOINT_SILENCE[self.vad]
            if endpoint_silence is None
            else endpoint_silence
        )
        self._vad_model = str(d / _VAD_FILE)
        self.use_lm, self.two_pass, self.chunk = _resolve(mode, use_lm, two_pass, chunk)
        self.punctuate = punctuate
        missing = [
            f
            for f in _required_files(self.use_lm, self.two_pass, self.chunk, punctuate)
            + ([_VAD_FILE] if self.vad else [])
            if not (d / f).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{d} lacks {', '.join(missing)}; this model may not support the chosen options"
            )
        self._streaming = self._load(
            d,
            CHUNKS[self.chunk],
            beam=_BEAM if self.use_lm else 1,
            num_threads=num_threads,
            endpoint=not self.vad,
            endpoint_silence=self.endpoint_silence,
        )
        self._final = (
            self._load(
                d, "final", beam=_FINAL_BEAM, num_threads=num_threads, endpoint=False
            )
            if self.two_pass
            else None
        )
        self._calibration = _load_calibration(
            d / _CALIBRATION_FILE, self.chunk, self.use_lm, self.two_pass
        )
        self._formatter = None
        if punctuate:
            from .formatter import Formatter

            self._formatter = Formatter(d / "formatter", num_threads=num_threads)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        revision: str | None = None,
        mode: str = "balanced",
        use_lm: bool | None = None,
        two_pass: bool | None = None,
        chunk: int | None = None,
        punctuate: bool = False,
        num_threads: int = 1,
        vad: bool | None = None,
        endpoint_silence: float | None = None,
    ) -> Recognizer:
        """Download the model from the Hugging Face Hub (cached) and load it.

        Only the files needed for the chosen options are downloaded.
        """
        lm, tp, ch = _resolve(mode, use_lm, two_pass, chunk)
        files = (
            _required_files(lm, tp, ch, punctuate)
            + ([] if vad is False else [_VAD_FILE])
            + [_CALIBRATION_FILE]
        )
        model_dir = download(repo_id, revision=revision, files=files)
        return cls(
            model_dir,
            use_lm=lm,
            two_pass=tp,
            chunk=ch,
            punctuate=punctuate,
            num_threads=num_threads,
            vad=vad,
            endpoint_silence=endpoint_silence,
        )

    def stream(self) -> Stream:
        return Stream(self)

    def transcribe(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE
    ) -> str:
        """Transcribe a complete mono float32 waveform in [-1, 1]."""
        if self.two_pass:  # partials are not needed, so only the second pass runs
            final = self._final_pass()
            final.accept(sample_rate, samples)
            return self._format(final.result())
        stream = self.stream()
        stream.accept(samples, sample_rate)
        return stream.finish()

    def transcribe_words(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE
    ) -> list[Word]:
        """The words of a complete mono float32 waveform in [-1, 1], in spoken form, with times."""
        if self.two_pass:
            final = self._final_pass()
            final.accept(sample_rate, samples)
            return final.tokens().words(calibration=self._calibration)
        stream = self.stream()
        stream.accept(samples, sample_rate)
        return stream._finish().words

    def _load(
        self,
        d: Path,
        sub: str,
        *,
        beam: int,
        num_threads: int,
        endpoint: bool,
        endpoint_silence: float = _ENDPOINT_SILENCE[False],
    ) -> sherpa_onnx.OnlineRecognizer:
        m = d / sub
        lm_args = (
            {
                "lm": str(d / "lm/rnnlm.int8.onnx"),
                "lm_scale": _LM_SCALE,
                "lodr_fst": str(d / "lm/2gram.fst"),
                "lodr_scale": _LODR_SCALE,
            }
            if self.use_lm
            else {}
        )
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(m / "encoder.int8.onnx"),
            decoder=str(m / "decoder.onnx"),
            joiner=str(m / "joiner.int8.onnx"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=endpoint,
            rule2_min_trailing_silence=endpoint_silence,
            decoding_method="modified_beam_search" if beam > 1 else "greedy_search",
            max_active_paths=beam,
            **lm_args,
        )

    def _end_of_speech(self) -> EndOfSpeech | None:
        if not self.vad:
            return None
        return EndOfSpeech(self._vad_model, silence=self.endpoint_silence)

    def _final_pass(self) -> _FinalPass | None:
        return _FinalPass(self._final) if self.two_pass else None

    def _format(self, text: str) -> str:
        return self._formatter(text) if self._formatter and text else text

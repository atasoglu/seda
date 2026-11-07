"""Keyword spotting on the recognizer's final words, with calibrated confidence.

    spotter = KeywordSpotter(recognizer, ["seda", "kaydı durdur"])
    spotter.find(samples, sample_rate)   # -> [Detection(keyword="seda", text="sedanın", ...)]

A keyword is found when the recognizer's final text contains its words in a row. Keywords are matched
in spoken form, as the model writes: lowercase, numbers spelled out, no apostrophes ("Ankara'da" is
"ankarada"). With ``suffixes`` (default), the last word may carry Turkish suffixes from the closed
list in :mod:`sedalib.itn`: "kamp" matches "kampa" and "kampın", not "kampanya". Stem changes are not
handled: list "kitabı" yourself if "kitap" should match it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .itn import SUFFIXES, number_to_words
from .stt import SAMPLE_RATE, Recognizer, Stream, Word

# Pieces fed to the stream by find(): the end of speech is checked once per piece.
_FIND_PIECE_SECONDS = 0.1
# Suffixes a keyword's last word may carry: the case, plural and possessive endings of sedalib.itn,
# plus the copula ("sahiptir", "evdedir").
_SUFFIXES = SUFFIXES | {
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
    "dedir",
    "dadır",
    "tedir",
    "tadır",
    "lerdir",
    "lardır",
}


@dataclass(frozen=True)
class Detection:
    """A keyword heard in the audio.

    ``text`` is what was recognized (e.g. "kampa" for the keyword "kamp"), ``start`` and ``end`` are
    in seconds from the start of the audio. ``confidence`` is the probability that the words were
    recognized right: the product of the words' calibrated confidences, None if the model has no
    calibration for the decoding setup. ``score`` is the same product of the raw word scores.
    """

    keyword: str
    text: str
    start: float
    end: float
    score: float
    confidence: float | None


def _spoken(text: str) -> list[str]:
    text = (
        text.replace("İ", "i")
        .replace("I", "ı")
        .lower()
        .replace("'", "")
        .replace("’", "")
    )
    text = re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)
    return re.sub(r"[^\w\s]|_|\d", " ", text).split()


class KeywordSpotter:
    """Finds the given keywords (words or phrases) in a waveform or a live stream.

    ``min_confidence`` drops detections the recognizer is less sure of (with models that have no
    calibration, nothing is dropped). ``suffixes`` lets the last word of a keyword carry Turkish
    suffixes. Detections come from final texts, so in a live stream they arrive at the end of each
    utterance (the VAD's pause), not while the word is still being spoken.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        keywords: Iterable[str],
        *,
        min_confidence: float = 0.5,
        suffixes: bool = True,
    ):
        self.recognizer = recognizer
        self.min_confidence = min_confidence
        self.suffixes = suffixes
        self._keywords: list[tuple[str, list[str]]] = []
        for k in keywords:
            words = _spoken(k)
            if not words:
                raise ValueError(f"keyword {k!r} has no words")
            self._keywords.append((k, words))

    def find(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE
    ) -> list[Detection]:
        """The keywords in a complete mono float32 waveform in [-1, 1], in order of time."""
        stream = self.stream()
        step = int(_FIND_PIECE_SECONDS * sample_rate)
        found = []
        for i in range(0, len(samples), step):
            found += stream.accept(samples[i : i + step], sample_rate)
        return found + stream.finish()

    def stream(self) -> KeywordStream:
        return KeywordStream(self)

    def match(self, words: Sequence[Word]) -> list[Detection]:
        """The keywords in a list of recognized words, e.g. ``Utterance.words``."""
        found = []
        for keyword, kw in self._keywords:
            n = len(kw)
            for i in range(len(words) - n + 1):
                span = words[i : i + n]
                if not all(w.text == k for w, k in zip(span[:-1], kw[:-1])):
                    continue
                last = span[-1].text
                if last != kw[-1] and not (
                    self.suffixes
                    and last.startswith(kw[-1])
                    and last[len(kw[-1]) :] in _SUFFIXES
                ):
                    continue
                d = _detection(keyword, span)
                if d.confidence is None or d.confidence >= self.min_confidence:
                    found.append(d)
        return sorted(found, key=lambda d: (d.start, d.keyword))


class KeywordStream:
    """Keyword spotting on live audio. Feed it pieces with :meth:`accept`; it returns the detections
    of each utterance as soon as the utterance ends."""

    def __init__(self, spotter: KeywordSpotter):
        self._spotter = spotter
        self._stream: Stream = spotter.recognizer.stream()

    def accept(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE
    ) -> list[Detection]:
        """Feed mono float32 samples in [-1, 1]; returns the detections of an utterance that just
        ended, else an empty list."""
        self._stream.accept(samples, sample_rate)
        if not self._stream.is_endpoint:
            return []
        return self._spotter.match(self._stream.cut().words)

    def finish(self) -> list[Detection]:
        """End of the audio: the detections in the last utterance."""
        return self._spotter.match(self._stream._finish().words)


def _detection(keyword: str, words: Sequence[Word]) -> Detection:
    score, confidence = 1.0, 1.0
    for w in words:
        score *= w.score
        confidence = (
            None
            if confidence is None or w.confidence is None
            else confidence * w.confidence
        )
    return Detection(
        keyword,
        " ".join(w.text for w in words),
        words[0].start,
        words[-1].end,
        score,
        confidence,
    )

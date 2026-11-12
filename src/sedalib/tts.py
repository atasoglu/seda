from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

# Text is normalized exactly as in training, so the symbol table must match the model's.
_EXTRA_HINT = 'speech synthesis needs the "tts" extra: pip install "sedalib[tts]"'
_MAX_BERT_TOKENS = 256
_MAX_PIECE_CHARS = 120  # the longest texts the model was trained on are 117 to 127 characters
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
# A piece starts and ends with about 0.13 s of its own silence; real pauses between sentences are about 0.24 s.
_SENTENCE_PAUSE_S = 0.12
DEFAULT_TTS_REPO_ID = "atasoglu/seda-tts-v0.1"
_MODEL_FILES = ("config.json", "tokenizer.json", "bert.onnx", "encoder.onnx", "decoder.onnx")


def normalize(text: str, symbols: set[str]) -> str:
    """Turkish lower-casing; characters the model has no symbol for become spaces.

    Numbers, dates, currency and abbreviations should be spoken out first (see ``Synthesizer(expand=True)``);
    whatever is left of them here is dropped.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("̇", "").replace("İ", "i").replace("I", "ı").lower()
    text = "".join(c if c in symbols and c != "_" else " " for c in text)
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """Group whole sentences into pieces of at most 120 characters, the longest the model was trained on.

    A third of the training clips hold two or more sentences, so within a piece the model pauses between
    them as a speaker would. A sentence that is longer on its own is split between words.
    """
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text.strip()):
        if len(sentence) <= _MAX_PIECE_CHARS:
            if current and len(current) + 1 + len(sentence) <= _MAX_PIECE_CHARS:
                current = f"{current} {sentence}"
            else:
                if current:
                    pieces.append(current)
                current = sentence
            continue
        if current:
            pieces.append(current)
            current = ""
        for w in sentence.split():
            if current and len(current) + 1 + len(w) > _MAX_PIECE_CHARS:
                pieces.append(current)
                current = w
            else:
                current = f"{current} {w}" if current else w
        pieces.append(current)
        current = ""
    if current:
        pieces.append(current)
    return pieces


class Synthesizer:
    """Turkish text to speech with a seda VITS model.

    ``Synthesizer(dir).synthesize("merhaba dünya")`` returns mono float32 samples at ``sample_rate``.
    ``stream`` yields them piece by piece, so playback can start before the sentence is finished.
    The text encoder is a BERT, so the first piece costs one BERT pass; later pieces reuse nothing.

    The model only knows letters and ``. , ! ? ; : ' -``. With ``expand=True`` (the default) numbers, dates,
    times, currency and common abbreviations are first written out in words by
    `normalizer-tr <https://pypi.org/project/normalizer-tr/>`_ (``"24 derece"`` -> ``"yirmi dört derece"``).
    It needs Python 3.11 or newer; on 3.10 pass ``expand=False`` and spell numbers out yourself.

    A model's ``config.json`` may name the voices and styles that are released (``speakers``, ``styles``);
    the others are not offered. Each style has its own emotions (``emotions_for``) and settings
    (``defaults_for``): a whisper needs more variation than normal speech to sound breathy, not buzzing.
    """

    def __init__(
        self, model_dir: str | Path, *, num_threads: int = 1, expand: bool = True
    ):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from normalizer_tr import NormalizationError, Normalizer
        except ImportError as e:
            raise ImportError(_EXTRA_HINT) from e
        import json

        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self._np = np
        self._expand = Normalizer() if expand else None
        self._norm_error = NormalizationError
        self.sample_rate: int = cfg["sample_rate"]
        self._hop: int = cfg["hop_length"]
        self._symbols: list[str] = cfg["symbols"]
        self._ids = {s: i for i, s in enumerate(self._symbols)}
        self._symbol_set = set(self._symbols)
        # the model's ids come from the full lists; a release may offer only some of them
        self._speaker_ids = {s: i for i, s in enumerate(cfg["speakers"])}
        self._style_ids = {s: i for i, s in enumerate(cfg["styles"])}
        release = cfg.get("release", {})
        self.speakers: list[str] = release.get("speakers", cfg["speakers"])
        self.emotions: list[str] = cfg["emotions"]
        self._style_emotions: dict[str, list[str] | None] = release.get(
            "styles", dict.fromkeys(cfg["styles"])
        )
        self.styles: list[str] = list(self._style_emotions)
        self.default_speaker: str = release.get("default_speaker", self.speakers[0])
        self._auto_emotion: int = cfg["auto_emotion_id"]
        self.defaults: dict[str, float] = {
            k: v
            for k, v in cfg["inference_defaults"].items()
            if isinstance(v, (int, float))
        }
        self._style_defaults: dict[str, dict[str, float]] = release.get(
            "style_defaults", {}
        )

        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(_MAX_BERT_TOKENS)
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1

        def load(name: str):
            return ort.InferenceSession(
                str(d / name), options, providers=["CPUExecutionProvider"]
            )

        self._bert = load("bert.onnx")
        self._encoder = load("encoder.onnx")
        self._decoder = load("decoder.onnx")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_TTS_REPO_ID,
        *,
        revision: str | None = None,
        **kwargs,
    ) -> Synthesizer:
        """Download a model from the Hugging Face Hub (cached) and load it; ``kwargs`` go to the constructor.

        Only the model files are downloaded, not the samples or the model card.
        """
        from .hub import download

        return cls(download(repo_id, revision=revision, files=list(_MODEL_FILES)), **kwargs)

    def defaults_for(self, style: str) -> dict[str, float]:
        """``noise_scale``, ``noise_scale_w`` and ``length_scale`` the model sounds best with in ``style``."""
        self._check(self.styles, style, "style")
        return {**self.defaults, **self._style_defaults.get(style, {})}

    def emotions_for(self, style: str) -> list[str]:
        """The emotions ``style`` was trained with."""
        self._check(self.styles, style, "style")
        return self._style_emotions[style] or self.emotions

    def synthesize(self, text: str, **kwargs) -> "np.ndarray":  # noqa: F821
        """All the samples of ``text``; takes the arguments of :meth:`stream`."""
        np = self._np
        pieces = list(self.stream(text, **kwargs))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def stream(
        self,
        text: str,
        *,
        speaker: str | None = None,
        emotion: str | None = None,
        style: str = "normal",
        noise_scale: float | None = None,
        noise_scale_w: float | None = None,
        length_scale: float | None = None,
        chunk_frames: int = 48,
        padding_frames: int = 24,
        seed: int | None = None,
    ) -> Iterator["np.ndarray"]:  # noqa: F821
        """Yield float32 sample chunks in [-1, 1].

        ``speaker=None`` is ``default_speaker``; ``emotion=None`` lets the model infer it from the text.
        ``noise_scale`` and ``noise_scale_w`` set how varied the voice and the rhythm are; ``length_scale`` > 1
        slows speech down; ``None`` takes the style's own value (``defaults_for``). ``chunk_frames`` is the
        decoder step in frames of ``hop_length`` samples, ``padding_frames`` the context around it (24 is
        what reproduces the one-piece output).
        """
        np = self._np
        speaker = speaker or self.default_speaker
        self._check(self.speakers, speaker, "speaker")
        self._check(self.styles, style, "style")
        if emotion is not None:
            self._check(self.emotions_for(style), emotion, f"emotion for {style!r}")
        cond = {
            "sid": np.array([self._speaker_ids[speaker]]),
            "eid": np.array(
                [self._auto_emotion if emotion is None else self.emotions.index(emotion)]
            ),
            "stid": np.array([self._style_ids[style]]),
        }
        defaults = self.defaults_for(style)
        noise_scale = defaults["noise_scale"] if noise_scale is None else noise_scale
        noise_scale_w = defaults["noise_scale_w"] if noise_scale_w is None else noise_scale_w
        length_scale = defaults["length_scale"] if length_scale is None else length_scale
        rng = np.random.default_rng(seed)
        text = " ".join(text.split())  # the normalizer rejects empty text and control characters
        if self._expand and text:
            try:
                # "fallback" reads an ambiguous number ("3.") as a cardinal instead of leaving digits behind
                text = self._expand.normalize(
                    text, ambiguity_policy="fallback"
                ).normalized_text
            except self._norm_error:  # e.g. invalid Unicode: speak what the model knows of it
                pass
        pause = False  # the previous piece ended a sentence
        for piece in split_sentences(text):
            norm = normalize(piece, self._symbol_set)
            if not any(c.isalpha() for c in norm):
                continue
            x, bert = self._text_inputs(norm)
            m_p, logs_p, g = self._acoustic(
                x, bert, cond, rng.standard_normal((1, 2, x.shape[1])) * noise_scale_w, length_scale
            )
            z_p = m_p + rng.standard_normal(m_p.shape) * np.exp(logs_p) * noise_scale
            if pause:
                yield np.zeros(round(_SENTENCE_PAUSE_S * self.sample_rate), dtype=np.float32)
            yield from self._vocode(z_p.astype(np.float32), g, chunk_frames, padding_frames)
            pause = piece[-1] in ".!?;:"

    @staticmethod
    def _check(names: list[str], name: str, kind: str) -> None:
        if name not in names:
            raise ValueError(f"unknown {kind} {name!r}, choose from {names}")

    def _text_inputs(self, norm: str):
        """Symbol ids with a blank between all symbols, and BERT features aligned to them."""
        np = self._np
        enc = self._tokenizer.encode(norm)
        out = self._bert.run(
            None,
            {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
            },
        )[0][0]
        # the features were stored as fp16 during training
        vec = out.astype(np.float16).astype(np.float32)

        char_to_token = np.zeros(len(norm), dtype=np.int64)
        for i, (a, b) in enumerate(enc.offsets):
            if b > a:
                char_to_token[a:b] = i
        for i in range(1, len(norm)):
            if norm[i] == " ":
                char_to_token[i] = char_to_token[i - 1]

        n = len(norm)
        ids = np.zeros(2 * n + 1, dtype=np.int64)
        ids[1::2] = [self._ids[c] for c in norm]
        feats = np.zeros((2 * n + 1, vec.shape[1]), dtype=np.float32)
        feats[1::2] = vec[char_to_token]
        feats[0:-1:2] = vec[char_to_token]  # a blank takes the features of the next symbol
        feats[-1] = vec[char_to_token[-1]]
        return ids[None], feats[None]

    def _acoustic(self, x, bert, cond, sdp_noise, length_scale):
        np = self._np
        return self._encoder.run(
            None,
            {
                "x": x,
                "x_lengths": np.array([x.shape[1]], dtype=np.int64),
                "bert": bert,
                "sdp_noise": sdp_noise.astype(np.float32),
                "length_scale": np.array([length_scale], dtype=np.float32),
                **cond,
            },
        )

    def _vocode(self, z_p, g, chunk: int, pad: int) -> Iterator["np.ndarray"]:  # noqa: F821
        """Decode in chunks of ``chunk`` frames, each with ``pad`` frames of context on both sides.

        The first chunk is half as long, so the first samples arrive sooner. Less than 24 frames of context
        leaves audible steps at the chunk borders (flow and vocoder see about that far).
        """
        frames = z_p.shape[2]
        start = 0
        while start < frames:
            end = min(start + (chunk // 2 if start == 0 else chunk), frames)
            lo, hi = max(0, start - pad), min(frames, end + pad)
            wav = self._decoder.run(None, {"z_p": z_p[:, :, lo:hi], "g": g})[0][0, 0]
            yield wav[(start - lo) * self._hop : (end - lo) * self._hop]
            start = end

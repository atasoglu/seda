from __future__ import annotations

from pathlib import Path

from .itn import itn

# Labels of the three token-classification heads, in the model's output order.
PUNCT = ("", ",", ".", "?", "!", ":", ";")
CASE = ("lower", "title", "upper")
_WINDOW_WORDS = 48  # the model was trained on windows of 12-48 words
_MAX_TOKENS = 128
_EXTRA_HINT = 'the formatter needs the "format" extra: pip install "sedalib[format]"'


def _tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


class Formatter:
    """Turns spoken-form text into written form: digits, punctuation, casing and apostrophes.

    ``Formatter(dir)("bugün ankarada hava yirmi dört derece")`` -> ``"Bugün Ankara'da hava 24 derece"``.
    Numbers go through :func:`sedalib.itn.itn`; the rest comes from a small BERT token classifier.
    """

    def __init__(self, model_dir: str | Path, *, num_threads: int = 1):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(_EXTRA_HINT) from e
        d = Path(model_dir)
        self._np = np
        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(_MAX_TOKENS)
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(d / "formatter.int8.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, text: str) -> str:
        words = itn(text).split()
        out, sentence_start = [], True
        for w, (p, c, a) in zip(words, self._predict(words)):
            # The heads are independent, so the word after a predicted . ? ! starts a sentence.
            if sentence_start and CASE[c] == "lower":
                c = CASE.index("title")
            sentence_start = PUNCT[p] in (".", "?", "!")
            # itn already wrote digits and their suffixes
            if not any(ch.isdigit() for ch in w):
                if 0 < a < len(w):
                    w = w[:-a] + "'" + w[-a:]
                if CASE[c] == "title":
                    w = _tr_upper(w[0]) + w[1:]
                elif CASE[c] == "upper":
                    stem, _, suffix = w.partition("'")
                    w = _tr_upper(stem) + ("'" + suffix if suffix else "")
            out.append(w + PUNCT[p])
        return " ".join(out)

    def _predict(self, words: list[str]) -> list[tuple[int, int, int]]:
        """(punctuation, case, apostrophe suffix length) label ids per word."""
        np, labels = self._np, []
        for start in range(0, len(words), _WINDOW_WORDS):
            window = words[start : start + _WINDOW_WORDS]
            enc = self._tokenizer.encode(window, is_pretokenized=True)
            feeds = {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            }
            punct, case, apostrophe = self._session.run(None, feeds)
            first = {}
            for j, wi in enumerate(enc.word_ids):
                if wi is not None and wi not in first:
                    first[wi] = j
            for i in range(len(window)):
                j = first.get(i)  # None if the window was truncated before this word
                labels.append(
                    (0, 0, 0)
                    if j is None
                    else (
                        int(punct[0, j].argmax()),
                        int(case[0, j].argmax()),
                        int(apostrophe[0, j].argmax()),
                    )
                )
        return labels

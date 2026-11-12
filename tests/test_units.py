"""Tests that need no model."""

import builtins
import math

import pytest
from conftest import DATA

from sedalib import MODES, KeywordSpotter, Recognizer, itn, keywords, read_wav, stt
from sedalib.formatter import Formatter
from sedalib.vad import _Resampler


def test_presets():
    assert stt._resolve("fast", None, None, None) == (False, False, 32)
    assert stt._resolve("balanced", None, None, None) == (True, False, 64)
    assert stt._resolve("accurate", None, None, None) == (True, True, 32)
    assert stt._resolve("balanced", False, True, 32) == (False, True, 32)
    assert set(MODES) == {"fast", "balanced", "accurate"}


@pytest.mark.parametrize("kwargs", [{"mode": "slow"}, {"chunk": 48}])
def test_invalid_options(kwargs):
    args = {
        "mode": "balanced",
        "use_lm": None,
        "two_pass": None,
        "chunk": None,
        **kwargs,
    }
    with pytest.raises(ValueError):
        stt._resolve(args["mode"], args["use_lm"], args["two_pass"], args["chunk"])


def test_required_files():
    fast = stt._required_files(False, False, 32, False)
    assert "c32/encoder.int8.onnx" in fast and "tokens.txt" in fast
    assert not any(f.startswith(("lm/", "final/", "formatter/")) for f in fast)
    balanced = stt._required_files(True, False, 64, False)
    assert "encoder.int8.onnx" in balanced and "lm/rnnlm.int8.onnx" in balanced
    accurate = stt._required_files(True, True, 32, True)
    assert {"final/encoder.int8.onnx", "formatter/tokenizer.json"} <= set(accurate)


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="encoder.int8.onnx"):
        Recognizer(tmp_path)


@pytest.mark.parametrize(
    "spoken, written",
    [
        ("bin dokuz yüz doksan dokuz depreminde", "1999 depreminde"),
        ("yüzde kırk artmış", "%40 artmış"),
        ("yirmi milyon kişi", "20 milyon kişi"),
        ("on beşine kadar", "15'ine kadar"),
        ("üç yüz elli bin lira", "350.000 lira"),
        ("bir gün gelecek", "bir gün gelecek"),
        ("saat dokuzda gel", "saat dokuzda gel"),
        ("", ""),
    ],
)
def test_itn(spoken, written):
    assert itn(spoken) == written


def test_formatter_without_extra(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def no_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_onnxruntime)
    with pytest.raises(ImportError, match=r"sedalib\[format\]"):
        Formatter(tmp_path)


def test_resampler_passthrough():
    assert list(_Resampler()([0.5, -0.5], 16000)) == [0.5, -0.5]


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_resampler_is_continuous_across_calls(rate):
    n = rate  # one second of a 440 Hz tone
    x = [math.sin(2 * math.pi * 440 * i / rate) for i in range(n)]
    whole = _Resampler()(x, rate)
    r = _Resampler()
    pieces = []
    for i in range(0, n, 1234):
        pieces.extend(r(x[i : i + 1234], rate))
    # the last input sample is kept for the next call, so up to 16000 / rate outputs are still to come
    assert 0 <= 16000 - len(whole) <= 16000 / rate + 1
    assert len(pieces) == len(whole)
    assert max(abs(a - b) for a, b in zip(pieces, whole)) < 1e-6


def test_read_wav():
    samples, rate = read_wav(DATA / "speech.wav")
    assert rate == 16000
    assert 5 < len(samples) / rate < 8
    assert all(-1 <= s <= 1 for s in samples)


def test_tokens_to_words():
    tokens = stt._Tokens(
        (" kam", "p", " ", "a", "la", "nı"),
        (0.4, 0.44, 0.8, 0.84, 0.88, 0.92),
        (
            math.log(0.9),
            math.log(0.4),
            0.0,
            math.log(0.5),
            math.log(0.5),
            math.log(0.5),
        ),
    )
    a, b = tokens.words(offset=10.0)
    assert (a.text, a.start, a.end) == ("kamp", 10.4, 10.48)
    assert a.score == pytest.approx(0.6)  # geometric mean of 0.9 and 0.4
    assert (b.text, b.start, b.end) == (
        "alanı",
        10.8,
        10.96,
    )  # a lone word-boundary token
    assert a.confidence is None and b.confidence is None


def test_word_confidence(tmp_path):
    path = tmp_path / "calibration.json"
    assert (
        stt._load_calibration(path, 32, False, False) is None
    )  # no file: no confidence
    path.write_text(
        '{"feature_version": 1, "configs": {"chunk32-lm0": '
        '{"weights": [1.0, 0.0, 0.5, 0.0, -1.0, 2.0], "bias": 0.5}, '
        '"final-lm1": {"weights": [0, 0, 0, 0, 0, 0], "bias": 0.0}}}'
    )
    cal = stt._load_calibration(path, 32, False, False)
    assert stt._load_calibration(path, 64, True, False) is None  # a setup it lacks
    # the second pass does not depend on the streaming chunk
    assert stt._load_calibration(path, 16, True, True) == ([0] * 6, 0.0)
    (word,) = stt._Tokens((" ev",), (0.0,), (math.log(0.5),)).words(calibration=cal)
    lp = math.log(0.5)
    z = (
        0.5 + lp + 0.5 * lp * lp - 1.0 + 2.0
    )  # the first and the last word of the utterance
    assert word.confidence == pytest.approx(1 / (1 + math.exp(-z)))
    path.write_text('{"feature_version": 99, "configs": {}}')
    assert (
        stt._load_calibration(path, 32, False, False) is None
    )  # unknown features: ignored


def _words(*items):
    """Word(text, start=i, end=i+1, score, confidence) from (text, confidence) pairs."""
    return [stt.Word(t, float(i), i + 1.0, 0.9, c) for i, (t, c) in enumerate(items)]


def test_keyword_normalization():
    assert keywords._spoken("Ankara'da İSTANBUL, 2 kişi!") == [
        "ankarada",
        "istanbul",
        "iki",
        "kişi",
    ]
    with pytest.raises(ValueError):
        KeywordSpotter(None, ["?!"])


def test_keyword_match():
    words = _words(
        ("özel", 0.9),
        ("kampa", 0.8),
        ("kampanya", 0.9),
        ("sahiptir", 0.4),
        ("ev", None),
    )
    spotter = KeywordSpotter(None, ["Kamp", "özel kamp", "sahip", "ev"])
    found = [(d.keyword, d.text, d.start, d.end) for d in spotter.match(words)]
    assert found == [
        ("özel kamp", "özel kampa", 0.0, 2.0),
        (
            "Kamp",
            "kampa",
            1.0,
            2.0,
        ),  # a suffix from the list; "kampanya" is another word
        ("ev", "ev", 4.0, 5.0),  # no calibration: kept
    ]  # "sahiptir" is below min_confidence
    (d,) = [d for d in spotter.match(words) if d.keyword == "özel kamp"]
    assert d.confidence == pytest.approx(0.72) and d.score == pytest.approx(0.81)
    strict = KeywordSpotter(None, ["kamp", "sahip"], suffixes=False, min_confidence=0.0)
    assert strict.match(words) == []
    assert [
        d.text for d in KeywordSpotter(None, ["sahip"], min_confidence=0.3).match(words)
    ] == ["sahiptir"]


def test_tts_normalize_and_split():
    from sedalib import tts

    symbols = set("_ .,!?;:'-abcçdefgğhıijklmnoöprsştuüvyz")
    assert tts.normalize("İSTANBUL ışık, ÇOK 12!", symbols) == "istanbul ışık, çok !"
    # short sentences stay together, as in a third of the training clips
    assert tts.split_sentences("Merhaba. Nasılsın? İyiyim") == [
        "Merhaba. Nasılsın? İyiyim"
    ]
    sentence = "Bugün hava çok güzel, akşamüstü sahilde uzun bir yürüyüşe çıkalım mı?"  # 70 characters
    assert tts.split_sentences(f"{sentence} {sentence}") == [sentence, sentence]
    # a sentence longer than the model was trained on is split between words, its neighbours kept whole
    long = " ".join(["kelime"] * 100) + "."
    pieces = tts.split_sentences(f"Kısa. {long} Son.")
    assert pieces[0] == "Kısa." and pieces[-1] == "Son."
    assert all(len(p) <= 120 for p in pieces)
    assert " ".join(pieces) == f"Kısa. {long} Son."


def test_write_wav_roundtrip(tmp_path):
    from sedalib import write_wav

    write_wav(tmp_path / "a.wav", [0.0, 0.5, -0.5, 2.0], 22050)
    samples, rate = read_wav(tmp_path / "a.wav")
    assert rate == 22050 and len(samples) == 4
    assert samples[1] == pytest.approx(0.5, abs=1e-3) and samples[3] < 1.0

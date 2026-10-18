"""Tests that need no model."""

import builtins
import math

import pytest
from conftest import DATA

from sedalib import MODES, Recognizer, itn, read_wav, stt
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

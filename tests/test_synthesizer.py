"""Tests for speech synthesis. They need a TTS model in ``SEDA_TEST_TTS_MODEL``: a local directory or a repo id."""

import os
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("normalizer_tr")

from sedalib import Synthesizer

MODEL = os.environ.get("SEDA_TEST_TTS_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="SEDA_TEST_TTS_MODEL is not set")


def _model_dir():
    if Path(MODEL).is_dir():
        return Path(MODEL)
    from sedalib.hub import download

    return download(MODEL)


@pytest.fixture(scope="module")
def synth():
    return Synthesizer(_model_dir())


def test_from_pretrained_downloads_model_files_only(monkeypatch):
    import sedalib.hub

    asked = {}

    def fake_download(repo_id, *, revision=None, files=None):
        asked.update(repo_id=repo_id, revision=revision, files=files)
        return model_dir

    model_dir = _model_dir()
    monkeypatch.setattr(sedalib.hub, "download", fake_download)
    synth = Synthesizer.from_pretrained("atasoglu/x", revision="r", num_threads=2)
    assert asked["repo_id"] == "atasoglu/x" and asked["revision"] == "r"
    assert sorted(asked["files"]) == sorted(
        ["config.json", "tokenizer.json", "bert.onnx", "encoder.onnx", "decoder.onnx"]
    )
    assert synth.sample_rate > 0


def test_synthesize(synth):
    wav = synth.synthesize("Merhaba, bugün hava 24 derece.", seed=0)
    assert wav.dtype.name == "float32" and 1.5 < len(wav) / synth.sample_rate < 6
    assert 0.05 < abs(wav).max() <= 1.0


def test_chunks_match_one_piece(synth):
    text = "Bugün hava çok güzel, yürüyüşe çıkalım mı?"
    chunked = synth.synthesize(text, seed=1)
    whole = synth.synthesize(text, seed=1, chunk_frames=10**6, padding_frames=0)
    assert len(chunked) == len(whole)
    assert abs(chunked - whole).mean() < 1e-4


def test_empty_and_unknown(synth):
    for text in ["", "  ", "???", "12 \ud800"]:
        assert len(synth.synthesize(text)) == 0
    with pytest.raises(ValueError):
        synth.synthesize("merhaba", speaker="yok")


def test_styles_and_defaults(synth):
    for style in synth.styles:
        assert {"noise_scale", "noise_scale_w", "length_scale"} <= set(
            synth.defaults_for(style)
        )
        assert synth.emotions_for(style)
    assert synth.default_speaker in synth.speakers
    with pytest.raises(ValueError):
        synth.synthesize("merhaba", style="yok")
    untrained = set(synth.emotions) - set(synth.emotions_for(synth.styles[-1]))
    if untrained:  # an emotion the style was not trained with
        with pytest.raises(ValueError):
            synth.synthesize("merhaba", style=synth.styles[-1], emotion=min(untrained))


def test_pause_between_pieces(synth):
    sentence = "Bugün hava çok güzel, akşamüstü sahilde uzun bir yürüyüşe çıkalım mı?"
    chunks = list(synth.stream(f"{sentence} {sentence}", seed=0))
    pause = round(0.12 * synth.sample_rate)
    assert any(len(c) == pause and not c.any() for c in chunks)

"""Tests that run a model; see conftest.py for SEDA_TEST_MODEL."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from sedalib import MODES, Recognizer

from conftest import SPEECH_TEXT, supports

PIECE = 1600  # 100 ms at 16 kHz, like a microphone callback


def feed(stream, samples, rate=16000):
    for i in range(0, len(samples), PIECE):
        stream.accept(samples[i : i + PIECE], rate)


def recognizer(model_dir, mode, **kwargs):
    if not supports(model_dir, mode, kwargs.get("punctuate", False)):
        pytest.skip(f"the model has no files for mode={mode!r}")
    return Recognizer(model_dir, mode=mode, **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_transcribe(model_dir, speech, mode):
    assert recognizer(model_dir, mode).transcribe(*speech) == SPEECH_TEXT


@pytest.mark.parametrize("mode", MODES)
def test_streaming_matches_transcribe(model_dir, speech, mode):
    rec = recognizer(model_dir, mode)
    stream = rec.stream()
    feed(stream, speech[0])
    assert stream.text  # live partials while the audio arrives
    assert stream.end_utterance() == rec.transcribe(*speech)


def test_finish(model_dir, speech):
    stream = recognizer(model_dir, "balanced").stream()
    feed(stream, speech[0])
    assert stream.finish() == SPEECH_TEXT


@pytest.mark.parametrize("mode", MODES)
def test_two_utterances_split_at_endpoint(model_dir, speech, mode):
    """Speech, a pause, speech: one endpoint between them, and both texts are complete."""
    stream = recognizer(model_dir, mode).stream()
    samples = speech[0] + [0.0] * 16000 * 3 + speech[0]
    texts = []
    for i in range(0, len(samples), PIECE):
        stream.accept(samples[i : i + PIECE])
        if stream.is_endpoint:
            texts.append(stream.end_utterance())
    texts.append(stream.end_utterance())
    assert texts == [SPEECH_TEXT, SPEECH_TEXT]


def test_cut_and_finalize_in_another_thread(model_dir, speech):
    rec = recognizer(model_dir, "accurate")
    stream = rec.stream()
    feed(stream, speech[0])
    utterance = stream.cut()
    with ThreadPoolExecutor(1) as pool:
        final = pool.submit(utterance.finalize)
        feed(stream, speech[0])  # the stream goes on with the next utterance meanwhile
        assert final.result() == SPEECH_TEXT
    assert stream.end_utterance() == SPEECH_TEXT


def test_reset_drops_the_utterance(model_dir, speech):
    stream = recognizer(model_dir, "balanced").stream()
    feed(stream, speech[0][: len(speech[0]) // 2])
    stream.reset()
    assert stream.text == ""
    feed(stream, speech[0])
    assert stream.end_utterance() == SPEECH_TEXT


def test_other_sample_rates(model_dir, speech):
    samples = speech[0][::2]  # 8 kHz, naive decimation is enough here
    stream = recognizer(model_dir, "balanced").stream()
    feed(stream, samples, rate=8000)
    assert stream.end_utterance()


@pytest.mark.parametrize("mode", MODES)
def test_concurrent_streams(model_dir, speech, mode):
    """One Recognizer serves several streams in parallel threads."""
    rec = recognizer(model_dir, mode)

    def run(_):
        stream = rec.stream()
        feed(stream, speech[0])
        return stream.end_utterance()

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(run, range(4))) == [SPEECH_TEXT] * 4


def test_one_stream_from_two_threads(model_dir, speech):
    """A web server may feed audio and poll a stream from different threads; this must not crash."""
    stream = recognizer(model_dir, "balanced").stream()
    done = threading.Event()

    def poll():
        while not done.is_set():
            _ = stream.text, stream.is_endpoint

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        feed(stream, speech[0])
    finally:
        done.set()
        poller.join()
    assert stream.end_utterance() == SPEECH_TEXT


def test_punctuate(model_dir, speech):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    text = recognizer(model_dir, "balanced", punctuate=True).transcribe(*speech)
    assert text[0].isupper() and text.endswith(".")
    assert "".join(c for c in text.lower() if c not in ",.?!") == SPEECH_TEXT

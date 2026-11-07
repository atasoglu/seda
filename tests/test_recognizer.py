"""Tests that run a model; see conftest.py for SEDA_TEST_MODEL."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import SPEECH_TEXT, supports

from sedalib import MODES, KeywordSpotter, Recognizer

PIECE = 1600  # 100 ms at 16 kHz, like a microphone callback


def feed(stream, samples, rate=16000):
    for i in range(0, len(samples), PIECE):
        stream.accept(samples[i : i + PIECE], rate)


def recognizer(model_dir, mode, **kwargs):
    if not supports(model_dir, mode, kwargs.get("punctuate", False)):
        pytest.skip(f"the model has no files for mode={mode!r}")
    return Recognizer(model_dir, mode=mode, **kwargs)


def word_errors(text: str, reference: str = SPEECH_TEXT) -> int:
    """Word-level edit distance (the small models may miss a word in greedy search)."""
    hyp, ref = text.split(), reference.split()
    row = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        prev, row[0] = row[0], i
        for j, r in enumerate(ref, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (h != r))
    return row[-1]


@pytest.mark.parametrize("mode", MODES)
def test_transcribe(model_dir, speech, mode):
    assert word_errors(recognizer(model_dir, mode).transcribe(*speech)) <= 1


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
    rec = recognizer(model_dir, mode)
    stream = rec.stream()
    samples = speech[0] + [0.0] * 16000 * 3 + speech[0]
    texts = []
    for i in range(0, len(samples), PIECE):
        stream.accept(samples[i : i + PIECE])
        if stream.is_endpoint:
            texts.append(stream.end_utterance())
    texts.append(stream.end_utterance())
    assert texts == [rec.transcribe(*speech)] * 2


@pytest.mark.parametrize("vad", [True, False])
def test_long_silence_before_speech(model_dir, speech, vad):
    """An idle microphone must not end, or split, the utterance that follows."""
    if vad and not (model_dir / "vad/silero_vad.onnx").is_file():
        pytest.skip("the model has no VAD")
    stream = recognizer(model_dir, "balanced", vad=vad).stream()
    samples = [0.0] * 16000 * 25 + speech[0] + [0.0] * 16000 * 3
    texts = []
    for i in range(0, len(samples), PIECE):
        stream.accept(samples[i : i + PIECE])
        if stream.is_endpoint:
            texts.append(stream.end_utterance())
    assert texts == [SPEECH_TEXT]


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
        assert list(pool.map(run, range(4))) == [rec.transcribe(*speech)] * 4


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


@pytest.mark.parametrize("mode", MODES)
def test_words_match_text(model_dir, speech, mode):
    rec = recognizer(model_dir, mode)
    words = rec.transcribe_words(*speech)
    assert " ".join(w.text for w in words) == rec.transcribe(*speech)
    assert all(0 < w.score <= 1 for w in words)
    starts = [w.start for w in words]
    assert starts == sorted(starts) and all(w.end > w.start for w in words)


def test_word_times_count_from_stream_start(model_dir, speech):
    rec = recognizer(model_dir, "fast")
    stream = rec.stream()
    feed(stream, speech[0])
    first = stream.cut().words
    feed(stream, speech[0])
    second = stream.cut().words
    duration = len(speech[0]) / speech[1]
    assert second[-1].start > duration > first[-1].start


@pytest.mark.parametrize("mode", MODES)
def test_keywords_in_file_and_stream(model_dir, speech, mode):
    spotter = KeywordSpotter(
        recognizer(model_dir, mode), ["çocuklu aileler", "kampanya", "festival"]
    )
    found = spotter.find(*speech)
    assert {d.keyword for d in found} <= {"çocuklu aileler", "festival"}
    assert "çocuklu aileler" in {d.keyword for d in found}
    # live: the same speech twice with a pause; detections arrive per utterance, timed from the start
    samples, rate = speech
    stream = spotter.stream()
    live = []
    for part in (samples, [0.0] * rate, samples, [0.0] * rate):
        for i in range(0, len(part), PIECE):
            live += stream.accept(part[i : i + PIECE], rate)
    live += stream.finish()
    starts = [d.start for d in live if d.keyword == "çocuklu aileler"]
    assert len(starts) == 2 and starts[1] > len(samples) / rate + 1 > starts[0]

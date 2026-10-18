# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added

- Support for the new [seda-v0.2](https://huggingface.co/atasoglu/seda-v0.2) model, now the default: 6.38 WER on FLEURS-TR test in the `accurate` mode, 7.26 in `balanced` (seda-v0.1: 11.60).
- `mode` presets: `fast`, `balanced` (default) and `accurate`, plus `use_lm`, `two_pass` and `chunk` to override them.
- Second pass (`two_pass`, part of `accurate`): each utterance is decoded again with 20 s of left context and beam search + LM. It runs in a background thread while the utterance is spoken, so the final text arrives about 0.4 s after the speech.
- End of speech by Silero VAD v6 (`vad`, on by default when the model ships it), after a pause of `endpoint_silence` seconds (default 0.25 s).
- `Stream.end_utterance()` returns the final text and starts the next utterance. `Stream.cut()` returns an `Utterance` whose `finalize()` can run in another thread while the stream keeps accepting audio.
- Optional formatter (`punctuate=True`, needs the `format` extra): digits, punctuation, casing and apostrophes. `sedalib.itn` converts spoken numbers to digits without the extra.
- `Stream` methods may be called from different threads.
- The demo has a control panel for all options with an Apply button that loads and warms up the models. It shows live formatted text and how long the final text took after the speech ended.
- Tests (`uv run pytest`), run in CI after lint against seda-v0.2 and seda-v0.1.

### Changed

- `use_lm=False` now uses greedy search instead of beam search without the LM, which is faster.
- `Stream.reset()` also drops audio that was accepted but not decoded yet. It used to leak into the next utterance.
- The silence appended at the end of an utterance depends on the chunk size (0.9 s for 0.64 s chunks, 1.5 s for 1.28 s chunks).

### Fixed

- Without the VAD, a long silence before any speech no longer counts as an endpoint, and it no longer splits the utterance that follows.

### Compatibility

- seda-v0.1 still works in the `balanced` mode, with or without `use_lm`. It lacks the files for `fast`, `accurate`, the VAD and the formatter.
- sedalib 0.1 and 0.2 work with seda-v0.2 as before: they load the 1.28 s streaming model at the repo root. sedalib 0.1.0 pads too little silence for 1.28 s chunks and may drop the last word, so upgrade.

## [0.2.1] - 2026-10-16

### Changed

- The default model is back to [seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1), because seda-v0.2 is not public yet. 0.2.0 pointed to seda-v0.2 by default and fails to download the model; upgrade to 0.2.1.

## [0.2.0] - 2026-10-16

### Changed

- The default model was switched to seda-v0.2, which is not public yet.

### Fixed

- `Stream.finish()` (and `Recognizer.transcribe`) now appends 1.5 s of silence instead of 0.66 s, so the last word is not dropped when the audio ends right after the speech.

## [0.1.0] - 2026-10-07

### Added

- Streaming Turkish speech recognition with `Recognizer` and `Stream`, built on sherpa-onnx.
- Support for the [seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1) model, downloaded and cached from the Hugging Face Hub, with an optional language model.
- `read_wav` helper for loading 16-bit PCM WAV files.
- Microphone streaming example (`examples/streaming.py`).
- Gradio demo with live microphone streaming, file transcription and performance stats, available via the `demo` extra.

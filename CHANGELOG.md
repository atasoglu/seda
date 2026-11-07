# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-11-07

### Added

- `Stream.endpoint_silence`: the pause that ends an utterance can be read and changed while the stream runs, for example from a setting in the application. It needs the VAD.
- Keyword spotting demo: a Pause slider (0.05 to 1.5 s, default 0.25 s) that works while recording; the real-time factor (RTF) of each match and of the session; the legend, the model link and the recording clock moved or restyled; the page is laid out around a centered main column. `SEDA_PACKET_SECONDS` sets how often the browser sends audio (default 0.05 s).

### Changed

- The default `endpoint_silence` with the VAD is 0.75 s, up from 0.25 s, because of the fix below: the pause that was effective in 1.2.0 was about 0.75 s (the hangover plus 0.25 s), so by default nothing changes. Measured on FLEURS dev, with the old and the new default, the VAD reports the end of speech 1.00 s and 0.98 s after it (median of 100 recordings) and single-word keyword recall of the fast and balanced modes is the same (0.796 and 0.859). Shorter pauses are now really shorter: 0.6 s reports the end after 0.83 s, 0.25 s after 0.48 s, at the price of recall: 0.780 and 0.847 at 0.6 s, 0.754 and 0.812 at 0.25 s. Pass `endpoint_silence` or set `Stream.endpoint_silence` to choose.

### Fixed

- `endpoint_silence` was not the pause it said: sherpa-onnx's Silero model keeps its own 0.5 s hangover (`min_silence_duration`), which was added to it. Now `endpoint_silence` is the whole pause.

## [1.2.0] - 2026-11-07

### Added

- Keyword spotting: `KeywordSpotter(recognizer, keywords).find(samples)` for files and `.stream()` for live audio return `Detection(keyword, text, start, end, score, confidence)`. Matching is on spoken-form words; the last word may carry Turkish suffixes ("kamp" finds "kampa", not "kampanya"). `min_confidence` (default 0.5) drops unsure detections.
- Word timestamps and confidence: `Recognizer.transcribe_words()` and `Utterance.words` return `Word(text, start, end, score, confidence)` in spoken form. Times count from the start of the stream across `cut()`. `confidence` is the probability that the word is right, calibrated per decoding setup on held-out speech; it is `None` for models without a `calibration.json` or for a setup it does not cover. seda-v0.2 and seda-v0.2-tiny cover the `fast`, `balanced` and `accurate` modes and the 0.16 and 0.32 s chunks (`chunk=8` and `16`), with and without the LM.
- Keyword spotting demo (`python -m sedalib.demo.keywords`): keywords as tags, live microphone with a waveform and a recording clock, and matches in a scrolling panel with their confidence. Two modes on seda-v0.2-tiny with 0.16 s steps: Low latency (greedy) gives a keyword's confidence as soon as the next word starts, since a word greedy search has written does not change; High accuracy (beam search + LM) gives it at the pause. A keyword shows up greyed out as soon as it is heard, and each match shows how soon after it was shown and final. Keywords and the minimum confidence can be changed while recording; a keyword counts from the moment it is added. `SEDA_MODEL`, `SEDA_CHUNK` and `SEDA_ENDPOINT_SILENCE` change the defaults.

## [1.1.0] - 2026-10-25

### Added

- Support for the new [seda-v0.2-tiny](https://huggingface.co/atasoglu/seda-v0.2-tiny) model: 4.5M parameters (seda-v0.2: 66M), distilled from seda-v0.2 for slow CPUs, phones and the browser. 10.51 WER on FLEURS-TR test in the `accurate` mode, 11.24 in `balanced` (Whisper small: 11.87). All modes work with it.
- `chunk=8` and `chunk=16`: 0.16 and 0.32 s streaming steps, for the lowest latency. seda-v0.2 and seda-v0.2-tiny ship them.
- The demo can switch between seda-v0.2 and seda-v0.2-tiny in the options panel. Options a model does not support are reported instead of applied.

### Changed

- Model tests allow one word error in the transcript (the tiny model's greedy search), and compare the other checks with the model's own transcript.

## [1.0.0] - 2026-10-18

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

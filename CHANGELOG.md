# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

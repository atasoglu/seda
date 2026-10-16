# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-16

### Changed

- The default model is now [seda-v0.2](https://huggingface.co/atasoglu/seda-v0.2): 8.89 WER on FLEURS (seda-v0.1: 11.52) and much more robust to noise, reverb and telephone audio. It uses 1.28 s chunks. Pass `"atasoglu/seda-v0.1"` to `Recognizer.from_pretrained` to keep the old model.

### Fixed

- `Stream.finish()` (and `Recognizer.transcribe`) now appends 1.5 s of silence instead of 0.66 s, so the last word is not dropped when the audio ends right after the speech.

## [0.1.0] - 2026-10-07

### Added

- Streaming Turkish speech recognition with `Recognizer` and `Stream`, built on sherpa-onnx.
- Support for the [seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1) model, downloaded and cached from the Hugging Face Hub, with an optional language model.
- `read_wav` helper for loading 16-bit PCM WAV files.
- Microphone streaming example (`examples/streaming.py`).
- Gradio demo with live microphone streaming, file transcription and performance stats, available via the `demo` extra.

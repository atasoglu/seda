# seda

[![PyPI](https://img.shields.io/pypi/v/sedalib)](https://pypi.org/project/sedalib/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/atasoglu/seda)](LICENSE)
[![ONNX](https://img.shields.io/badge/runtime-ONNX-005CED?logo=onnx&logoColor=white)](https://github.com/k2-fsa/sherpa-onnx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E)](https://huggingface.co/atasoglu/seda-v0.1)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> *Bâki kalan bu kubbede bir hoş sadâ imiş...*
>
> — Bâkî

Lightweight, low-latency Turkish speech models for real-time applications.

Seda aims to provide models that are small enough to run on a CPU, fast enough for live audio, accurate enough for production, and available for commercial use. `sedalib` is the Python package for running them. It currently ships streaming speech recognition (STT) on top of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

> [!WARNING]
> This library and its models are under active development. APIs and model releases may change.

## Models

| Model | Task | Architecture | Params | Eval |
| --- | --- | --- | --- | --- |
| [seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1) | Streaming STT | Zipformer2 transducer | 66.1M | 11.52 WER (FLEURS) |

WER is measured with beam search (4 paths) and the language model enabled. The model processes audio in 0.64-second chunks, so partial results arrive in real time. See the model card for more benchmarks.

## Installation

```bash
pip install sedalib
```

Python 3.10 or newer is required. The only runtime dependencies are `sherpa-onnx` and `huggingface-hub`.

## Quick start

Transcribe a complete waveform. `read_wav` loads a 16-bit PCM WAV file as mono float samples in [-1, 1] together with its sample rate:

```python
from sedalib import Recognizer, read_wav

samples, sample_rate = read_wav("audio.wav")

recognizer = Recognizer.from_pretrained()  # downloads the model once and caches it
text = recognizer.transcribe(samples, sample_rate)
print(text)
```

Audio at other sample rates is resampled automatically.

## Streaming

For live audio, open a stream and feed it chunks as they arrive. `accept` returns the current partial transcript, and `is_endpoint` tells you when the speaker has finished an utterance.

```python
stream = recognizer.stream()

for chunk in microphone_chunks():  # e.g. 0.64 s of audio at a time
    partial = stream.accept(chunk, sample_rate=16000)
    print(partial)

    if stream.is_endpoint:
        print("final:", stream.text)
        stream.reset()  # start a new utterance
```

Call `stream.finish()` at the end of the audio to flush the remaining frames and get the final text. A single `Recognizer` can serve many independent streams.

## Options

```python
Recognizer.from_pretrained(
    "atasoglu/seda-v0.1",  # Hugging Face repo id
    revision=None,  # pin a branch, tag or commit
    use_lm=True,  # rescore with the language model
    num_threads=1,  # CPU threads used for inference
)
```

The language model improves accuracy at the cost of some extra compute. Pass `use_lm=False` for the lightest setup. To load a model you have already downloaded, use `Recognizer("path/to/model_dir")`.

## Demo

A Gradio demo with live microphone streaming and file transcription is available as an extra:

```bash
pip install "sedalib[demo]"
python -m sedalib.demo.stt
```

## Development

```bash
uv sync
uv run prek install  # ruff lint and format hooks
```

## License

The code is released under Apache-2.0. The acoustic model is Apache-2.0 as well. The optional language model is CC BY-SA 4.0 because it was trained on Wikipedia data, so pass `use_lm=False` if that is a concern for your use case.

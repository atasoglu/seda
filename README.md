![seda](https://raw.githubusercontent.com/atasoglu/seda/main/assets/banner.jpeg)

[![PyPI](https://img.shields.io/pypi/v/sedalib)](https://pypi.org/project/sedalib/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/atasoglu/seda)](LICENSE)
[![ONNX](https://img.shields.io/badge/runtime-ONNX-005CED?logo=onnx&logoColor=white)](https://github.com/k2-fsa/sherpa-onnx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E)](https://huggingface.co/atasoglu/seda-v0.2)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Lightweight, low-latency Turkish speech models for real-time applications.

Seda aims to provide models that are small enough to run on a CPU, fast enough for live audio, accurate enough for production, and available for commercial use. `sedalib` is the Python package for running them. It currently ships streaming speech recognition (STT) on top of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Models

| Model | Task | Architecture | Params | FLEURS-TR test WER |
| --- | --- | --- | --- | --- |
| [seda-v0.2](https://huggingface.co/atasoglu/seda-v0.2) | Streaming STT | Zipformer2 transducer + second pass | 66.1M | 6.38 (`accurate`), 7.26 (`balanced`) |
| [seda-v0.2-tiny](https://huggingface.co/atasoglu/seda-v0.2-tiny) | Streaming STT | Zipformer2 transducer + second pass | 4.5M | 10.51 (`accurate`), 11.24 (`balanced`) |
| [seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1) | Streaming STT | Zipformer2 transducer | 66.1M | 11.60 |

For comparison, Whisper small scores 11.87, Whisper medium 6.62 and Whisper large-v3 4.62 on the same set. See the [seda-v0.2 card](https://huggingface.co/atasoglu/seda-v0.2) for noisy, telephone and broadcast audio. seda-v0.2-tiny is distilled from seda-v0.2 for slow CPUs, phones and the browser: its acoustic model is 7 MB, and it runs every mode with the same options (`Recognizer.from_pretrained("atasoglu/seda-v0.2-tiny")`).

## Installation

```bash
pip install sedalib
pip install "sedalib[format]"  # adds digits, punctuation and casing to the output
```

Python 3.10 or newer is required. The core package depends only on `sherpa-onnx` and `huggingface-hub`.

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

For live audio, open a stream and feed it small pieces as they arrive. `accept` returns the current partial transcript, and `is_endpoint` turns true when the speaker pauses. `end_utterance` returns the final text and starts the next utterance.

```python
stream = recognizer.stream()

for chunk in microphone_chunks():  # e.g. 100 ms of audio at a time
    partial = stream.accept(chunk, sample_rate=16000)
    print(partial)

    if stream.is_endpoint:
        print("final:", stream.end_utterance())
```

The end of speech is detected by Silero VAD after a 0.25 s pause (`endpoint_silence`). If your application already knows that the speaker stopped, for example with a push-to-talk button, call `end_utterance()` right away.

To keep accepting audio while the final text is computed, split the two steps: `stream.cut()` ends the utterance and returns it, and `utterance.finalize()` can run in another thread.

A single `Recognizer` can serve many streams at once, from different threads. Call `stream.finish()` at the end of the audio to flush the remaining frames and get the final text.

## Modes

| Mode | Live text | Final text | WER | Final text after a pause | RTF |
| --- | --- | --- | --- | --- | --- |
| `fast` | 0.64 s chunks, greedy | the same | 9.38 | 0.32 s | 0.042 |
| `balanced` (default) | 1.28 s chunks, beam search + LM | the same | 7.26 | 0.37 s | 0.064 |
| `accurate` | 0.64 s chunks, greedy | second pass with full context, beam search + LM | 6.38 | 0.77 s | 0.137 |

- **WER:** FLEURS-TR test.
- **Final text after a pause:** median time from the end of speech, including the 0.25 s pause the VAD waits for.
- **Speed:** RTF on one thread of an Intel Core 5 210H CPU, including the second pass.

The second pass of `accurate` runs in the background while the utterance is spoken, so most of its work is done when the speaker stops.

## Options

```python
Recognizer.from_pretrained(
    "atasoglu/seda-v0.2",  # Hugging Face repo id
    revision=None,  # pin a branch, tag or commit
    mode="balanced",  # "fast", "balanced" or "accurate"
    use_lm=None,  # override the mode: rescore with the language model
    two_pass=None,  # override the mode: decode each utterance again for the final text
    chunk=None,  # override the mode: 8, 16, 32 or 64 frames (0.16 to 1.28 s per streaming step)
    punctuate=False,  # digits, punctuation, casing and apostrophes in final texts (needs [format])
    num_threads=1,  # CPU threads per model
    vad=None,  # Silero VAD for the end of speech (default: if the model ships it)
    endpoint_silence=None,  # pause that ends an utterance, in s (default 0.25 with the VAD, 1.2 without)
)
```

`from_pretrained` downloads only the files the chosen options need. To load a model you have already downloaded, use `Recognizer("path/to/model_dir")`.

- **Faster end of speech:** a shorter pause gives the final text sooner, but splits sentences more often at short hesitations. With 0.15 s, the final text arrives after about 0.22 s (`fast`) and 0.27 s (`balanced`).
- **Threads:** `num_threads=2` makes `fast` and `balanced` about 15% faster. More threads do not help a single stream. To serve many users, give each stream one thread and run them in parallel: one recognizer scales to about 4× the throughput on 8 cores.
- **Formatter:** `punctuate=True` writes "Bugün Ankara'da hava 24 derece" instead of "bugün ankarada hava yirmi dört derece". `sedalib.itn` converts only the numbers and needs no extra.
- **Short chunks:** `chunk=8` or `16` react sooner (0.16 or 0.32 s steps) but cost more CPU per second of audio and are a little less accurate. On slow devices, prefer `chunk=16` with greedy search or the default 0.64 s.
- **seda-v0.1:** works with the `balanced` mode, with or without `use_lm`, and ends utterances with the 1.2 s silence rule. It does not have the files for `fast`, `accurate`, the short chunks, the VAD or the formatter.

## Demo

A Gradio demo with live microphone streaming, file transcription, a control panel for all options and latency stats is available as an extra:

```bash
pip install "sedalib[demo]"
python -m sedalib.demo.stt
```

## Development

```bash
uv sync --all-extras
uv run prek install  # ruff lint and format hooks
uv run pytest  # tests that need a model are skipped
SEDA_TEST_MODEL=atasoglu/seda-v0.2,atasoglu/seda-v0.2-tiny,atasoglu/seda-v0.1 uv run pytest  # with models
```

## License

The code is released under Apache-2.0. The seda-v0.2 and seda-v0.2-tiny model files are CC BY-SA 4.0, because parts of their training data are share-alike: Khan Academy for the acoustic model, and Wikipedia for the language model and the formatter. The Silero VAD file shipped with the model is MIT. Commercial use is allowed under these terms; see the model cards for details.

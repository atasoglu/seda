"""Gradio demo for streaming Turkish speech recognition.

Usage: uv run python -m sedalib.demo.stt
"""

import time

import gradio as gr
import numpy as np

from sedalib import DEFAULT_REPO_ID, Recognizer

MODEL_ID = DEFAULT_REPO_ID
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"


def rss_mb() -> float | None:
    """Current resident memory of this process in MB (Linux), peak elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, ValueError):
        return None


_rss_before = rss_mb()
_t0 = time.perf_counter()
recognizer = Recognizer.from_pretrained()
LOAD_SECONDS = time.perf_counter() - _t0
_rss_after = rss_mb()
MODEL_MB = (
    _rss_after - _rss_before
    if _rss_before is not None and _rss_after is not None
    else None
)


def to_mono_float(audio: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    sample_rate, data = audio
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sample_rate


def fmt_rtf(infer: float, audio: float) -> str:
    rtf = infer / audio if audio else 0.0
    return f"{rtf:.3f} (~{1 / rtf:.0f}x real time)" if rtf else "-"


def fmt_rss() -> str:
    rss = rss_mb()
    return f"{rss:.0f} MB" if rss is not None else "n/a"


def new_state() -> dict:
    return {
        "stream": recognizer.stream(),
        "done": [],
        "chunks": 0,
        "audio": 0.0,
        "infer": 0.0,
        "last": 0.0,
        "worst": 0.0,
    }


def stream_stats(state: dict) -> str:
    n = state["chunks"]
    chunk = (
        f"{state['last'] * 1000:.0f} ms (mean {state['infer'] / n * 1000:.0f}, "
        f"max {state['worst'] * 1000:.0f})"
        if n
        else "-"
    )
    return (
        f"**RTF** {fmt_rtf(state['infer'], state['audio'])} · "
        f"**Inference per chunk** {chunk} · "
        f"**Audio** {state['audio']:.1f} s · **Memory** {fmt_rss()}"
    )


def stream_mic(state: dict | None, audio: tuple[int, np.ndarray] | None):
    if state is None:
        state = new_state()
    if audio is None:
        return state, " ".join(state["done"]), stream_stats(state)

    stream = state["stream"]
    samples, sample_rate = to_mono_float(audio)
    start = time.perf_counter()
    partial = stream.accept(samples, sample_rate)
    elapsed = time.perf_counter() - start

    state["chunks"] += 1
    state["audio"] += len(samples) / sample_rate
    state["infer"] += elapsed
    state["last"] = elapsed
    state["worst"] = max(state["worst"], elapsed)

    if stream.is_endpoint:
        if partial:
            state["done"].append(partial)
        stream.reset()
        partial = ""
    text = " ".join([*state["done"], partial]).strip()
    return state, text, stream_stats(state)


def clear():
    return None, "", stream_stats({"chunks": 0, "audio": 0.0, "infer": 0.0})


def transcribe(audio: tuple[int, np.ndarray] | None):
    if audio is None:
        return "", ""
    samples, sample_rate = to_mono_float(audio)
    duration = len(samples) / sample_rate
    start = time.perf_counter()
    text = recognizer.transcribe(samples, sample_rate)
    elapsed = time.perf_counter() - start
    stats = (
        f"**Audio** {duration:.1f} s · **Inference** {elapsed * 1000:.0f} ms · "
        f"**RTF** {fmt_rtf(elapsed, duration)} · **Memory** {fmt_rss()}"
    )
    return text, stats


def build() -> gr.Blocks:
    model = f"{MODEL_MB:.0f} MB" if MODEL_MB is not None else "n/a"
    with gr.Blocks(title="Seda STT") as demo:
        gr.Markdown(
            "# Seda STT\n"
            f"Streaming Turkish speech recognition on CPU with [{MODEL_ID}]({MODEL_URL}). "
            f"Loaded in {LOAD_SECONDS:.1f} s using {model} of RAM."
        )

        with gr.Tab("Streaming"):
            state = gr.State()
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            live = gr.Textbox(label="Transcript", lines=5, buttons=["copy"])
            stats = gr.Markdown()
            clear_btn = gr.Button("Clear")
            outputs = [state, live, stats]
            mic.stream(
                stream_mic,
                [state, mic],
                outputs,
                stream_every=0.64,
                show_progress="hidden",
            )
            mic.start_recording(clear, None, outputs, show_progress="hidden")
            clear_btn.click(clear, None, outputs, show_progress="hidden")

        with gr.Tab("File"):
            audio = gr.Audio(sources=["upload", "microphone"], label="Audio")
            out = gr.Textbox(label="Transcript", lines=5, buttons=["copy"])
            info = gr.Markdown()
            gr.Button("Transcribe", variant="primary").click(
                transcribe, audio, [out, info]
            )

    return demo


if __name__ == "__main__":
    build().launch()

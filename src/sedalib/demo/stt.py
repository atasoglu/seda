"""Gradio demo for streaming Turkish speech recognition, with a panel for the inference options.

Usage: uv run python -m sedalib.demo.stt
       SEDA_MODEL=path/to/model_dir uv run python -m sedalib.demo.stt   # a local copy or another repo id

Options take effect when Apply is pressed: the model is loaded and warmed up then, so the timings shown
for recordings and files never include loading or first-run costs.
"""

import ctypes
import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import numpy as np

from sedalib import CHUNKS, DEFAULT_REPO_ID, MODES, Formatter, Recognizer, download

MODEL_ID = os.environ.get("SEDA_MODEL", DEFAULT_REPO_ID)
LOCAL = Path(MODEL_ID).is_dir()
MODEL_URL = None if LOCAL else f"https://huggingface.co/{MODEL_ID}"
MAX_THREADS = max(1, min(8, os.cpu_count() or 1))
DEFAULT = {
    "use_lm": True,
    "two_pass": False,
    "chunk": 64,
    "punctuate": False,
    "threads": 1,
    "vad": True,
    "pause": 0.25,
}
# Audio is sent by the browser in pieces of this many seconds.
PACKET_SECONDS = 0.2
# 2 s of faint noise: runs every model once, so the first real request does not pay for it.
_WARMUP_AUDIO = (np.random.default_rng(0).standard_normal(32000) * 0.01).astype(
    np.float32
)

# Only the recognizer of the applied options stays loaded, so the memory figure is honest.
_recognizers: dict[tuple, Recognizer] = {}
_formatter: Formatter | None = None
# Loading is serialized: two requests must not load the same model in parallel.
_load_lock = threading.RLock()
# Finished utterances are finalized (second pass, formatter) here, so live audio keeps flowing.
_finalizer = ThreadPoolExecutor(max_workers=2)


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


def release_memory() -> None:
    """Hand memory of dropped models back to the OS (glibc keeps freed memory otherwise)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def load_recognizer(
    use_lm: bool, two_pass: bool, chunk: int, threads: int, vad: bool, pause: float
) -> Recognizer:
    options = {
        "use_lm": use_lm,
        "two_pass": two_pass,
        "chunk": int(chunk),
        "num_threads": int(threads),
        "vad": vad,
        "endpoint_silence": pause,
    }
    if LOCAL:
        return Recognizer(MODEL_ID, **options)
    return Recognizer.from_pretrained(MODEL_ID, **options)


def get_recognizer(o: dict) -> Recognizer:
    """The recognizer of the applied options o, loading it if needed."""
    key = (
        o["use_lm"],
        o["two_pass"],
        int(o["chunk"]),
        int(o["threads"]),
        o["vad"],
        o["pause"],
    )
    with _load_lock:
        if key not in _recognizers:
            if _recognizers:
                _recognizers.clear()  # running streams keep their own reference
                release_memory()
            _recognizers[key] = load_recognizer(*key)
        return _recognizers[key]


def get_formatter() -> Formatter:
    global _formatter
    with _load_lock:
        if _formatter is None:
            root = (
                Path(MODEL_ID) if LOCAL else download(MODEL_ID, files=["formatter/*"])
            )
            _formatter = Formatter(root / "formatter")
        return _formatter


def warm_up(recognizer: Recognizer, punctuate: bool) -> None:
    """Run the streaming pass, the second pass (if any) and the formatter once."""
    stream = recognizer.stream()
    for i in range(0, len(_WARMUP_AUDIO), 1600):
        stream.accept(_WARMUP_AUDIO[i : i + 1600])
    stream.finish()
    if punctuate:
        get_formatter()("bugün hava çok güzel yarın ne olacak")


def finalize(text: str, punctuate: bool) -> str:
    return get_formatter()(text) if punctuate and text else text


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


def describe(o: dict) -> str:
    threads = int(o["threads"])
    parts = [
        "beam search + LM" if o["use_lm"] else "greedy search",
        f"{int(o['chunk']) * 0.02:.2f} s chunks",
        "two-pass" if o["two_pass"] else "single pass",
        "formatted" if o["punctuate"] else "spoken form",
        f"{threads} thread{'s' if threads > 1 else ''}",
        f"end after a {o['pause']:.2f} s pause ({'VAD' if o['vad'] else 'decoder rule'})",
    ]
    return " · ".join(parts)


def as_options(use_lm, two_pass, chunk, punctuate, threads, vad, pause) -> dict:
    return {
        "use_lm": bool(use_lm),
        "two_pass": bool(two_pass),
        "chunk": int(chunk),
        "punctuate": bool(punctuate),
        "threads": int(threads),
        "vad": bool(vad),
        "pause": round(float(pause), 2),
    }


# Options panel


def apply_preset(mode: str):
    return MODES[mode]


def mark_changed(applied: dict, *values):
    """An option changed: Apply is needed (or not, if the values are back to the applied ones)."""
    if as_options(*values) == applied:
        return gr.update(interactive=False), applied_status(applied)
    return gr.update(interactive=True), (
        "**Options changed.** Press **Apply** to load and warm up the model; until then "
        f"the applied options stay in use: {describe(applied)}."
    )


def applied_status(
    o: dict, load: float | None = None, warm: float | None = None
) -> str:
    timing = (
        f" · loaded in {load:.1f} s, warmed up in {warm:.1f} s"
        if load is not None
        else ""
    )
    return f"**Applied:** {describe(o)}{timing} · process memory {fmt_rss()}"


def apply_options(*values):
    """Load the recognizer (and formatter) for the options and warm them up."""
    o = as_options(*values)
    yield gr.skip(), "**Loading and warming up…**", gr.update(interactive=False)
    start = time.perf_counter()
    recognizer = get_recognizer(o)
    if o["punctuate"]:
        get_formatter()
    load = time.perf_counter() - start
    start = time.perf_counter()
    warm_up(recognizer, o["punctuate"])
    warm = time.perf_counter() - start
    yield o, applied_status(o, load, warm), gr.update(interactive=False)


# Streaming


def new_state(o: dict) -> dict:
    recognizer = get_recognizer(o)
    return {
        "stream": recognizer.stream(),
        "options": o,
        "punctuate": o["punctuate"],
        "pending": [],
        "done": [],
        "audio": 0.0,
        "infer": 0.0,
        "chunks": 0,
        "last": 0.0,
        "worst": 0.0,
        "finals": [],
        "shown": None,
        "closed": False,
        "finished": False,
    }


def stream_stats(state: dict | None) -> str:
    if not state or not state["audio"]:
        return ""
    n = state["chunks"]
    chunk = (
        f"{state['last'] * 1000:.0f} ms (mean {state['infer'] / n * 1000:.0f}, "
        f"max {state['worst'] * 1000:.0f})"
    )
    finals, o = state["finals"], state["options"]
    final = ""
    if finals:
        # The pause rule is part of the wait: the end is only known once the pause is over.
        wait = o["pause"] * 1000 + finals[-1] * 1000
        final = (
            f" · **Final text** ~{wait:.0f} ms after you stopped speaking ({o['pause'] * 1000:.0f} "
            f"ms pause + {finals[-1] * 1000:.0f} ms to finalize"
            f"{'' if o['vad'] else ', the decoder rule adds up to a chunk'})"
        )
    return (
        f"**RTF** {fmt_rtf(state['infer'], state['audio'])} · "
        f"**Inference per chunk** {chunk}{final} · "
        f"**Audio** {state['audio']:.1f} s · **Memory** {fmt_rss()}"
    )


def transcript(state: dict, partial: str = "") -> str:
    """Finished utterances (their live text until the final one is ready) and the current one."""
    done = [u["final"] if u["final"] is not None else u["live"] for u in state["done"]]
    return " ".join(t for t in [*done, partial] if t)


def live_text(state: dict, text: str) -> str:
    """Partial text, formatted on the fly when the formatter is on (it takes a few ms)."""
    return finalize(text, state["punctuate"])


def end_utterance(state: dict, finish: bool = False) -> None:
    """Finalize the current utterance in the background (second pass, formatter); the stream goes on
    with the next utterance right away."""
    stream = state["stream"]
    start = time.perf_counter()
    if finish:
        cut, live = None, live_text(state, stream.text)
    else:
        cut = stream.cut()  # flushes the streaming pass
        live = live_text(state, cut.text)
    utterance = {"live": live, "final": None}
    state["done"].append(utterance)

    def work():
        text = stream.finish() if cut is None else cut.finalize()
        utterance["final"] = finalize(text, state["punctuate"])
        state["finals"].append(time.perf_counter() - start)

    state["pending"].append(_finalizer.submit(work))


def start_stream(state: dict | None, applied: dict):
    """A new recording starts with an empty transcript and locked options. The previous session is
    only marked closed (in place): the next audio chunk opens a new one, because Gradio may deliver
    that chunk before this handler's outputs land."""
    if state is not None:
        state["closed"] = True
    return "", "", f"Recording with: {describe(applied)}"


def stream_mic(state: dict | None, audio, applied: dict):
    if state is None or state["closed"]:
        state = new_state(applied)
    if state["finished"] or audio is None:  # e.g. a chunk after the recording stopped
        return state, gr.skip(), gr.skip()

    stream = state["stream"]
    samples, sample_rate = to_mono_float(audio)
    start = time.perf_counter()
    partial = stream.accept(samples, sample_rate)
    elapsed = time.perf_counter() - start
    state["infer"] += elapsed
    state["audio"] += len(samples) / sample_rate
    state["chunks"] += 1
    state["last"] = elapsed
    state["worst"] = max(state["worst"], elapsed)

    if stream.is_endpoint:
        end_utterance(state)
        partial = ""
    text = transcript(state, live_text(state, partial) if partial else "")
    if text == state["shown"]:  # nothing new (e.g. silence): leave the page as it is
        return state, gr.skip(), gr.skip()
    state["shown"] = text
    return state, text, stream_stats(state)


def stop_stream(state: dict | None):
    if state is None or state["finished"]:
        return state, gr.skip(), gr.skip()
    end_utterance(state, finish=True)
    state["finished"] = True
    for future in state["pending"]:
        future.result()
    return state, transcript(state), stream_stats(state)


def clear_stream(state: dict | None):
    """Empty the transcript; a running recording continues with a fresh utterance."""
    if state is not None:
        state["stream"].reset()
        state["done"].clear()
        state["finals"].clear()
        state["audio"] = state["infer"] = state["last"] = state["worst"] = 0.0
        state["chunks"] = 0
        state["shown"] = ""
    return state, "", ""


# Files


def transcribe(audio, applied: dict):
    if audio is None:
        return "", ""
    samples, sample_rate = to_mono_float(audio)
    recognizer = get_recognizer(applied)
    duration = len(samples) / sample_rate
    start = time.perf_counter()
    raw = recognizer.transcribe(samples, sample_rate)
    asr = time.perf_counter() - start
    text = finalize(raw, applied["punctuate"])
    total = time.perf_counter() - start
    stats = (
        f"{describe(applied)}  \n"
        f"**Audio** {duration:.1f} s · **Inference** {total * 1000:.0f} ms "
        f"(formatter {(total - asr) * 1000:.0f} ms) · **RTF** {fmt_rtf(total, duration)} · "
        f"**Memory** {fmt_rss()}"
    )
    return text, stats


def compare(audio, applied: dict):
    """Run every preset on the same audio, each loaded and warmed up before it is timed. The
    applied recognizer stays loaded; the others are dropped afterwards."""
    if audio is None:
        yield ""
        return
    samples, sample_rate = to_mono_float(audio)
    duration = len(samples) / sample_rate
    threads = int(applied["threads"])
    form = "formatted" if applied["punctuate"] else "spoken form"
    note = (
        f"Audio {duration:.1f} s · {threads} thread(s) · {form} · "
        "every preset is warmed up before it is timed."
    )
    head = [
        note,
        "",
        "| Mode | Inference | RTF | Transcript |",
        "| --- | --- | --- | --- |",
    ]
    rows = []
    for mode, (use_lm, two_pass, chunk) in MODES.items():
        yield "\n".join([*head, *rows, f"| {mode} | loading… | | |"])
        with _load_lock:
            recognizer = load_recognizer(
                use_lm, two_pass, chunk, threads, applied["vad"], applied["pause"]
            )
        warm_up(recognizer, applied["punctuate"])
        start = time.perf_counter()
        text = finalize(
            recognizer.transcribe(samples, sample_rate), applied["punctuate"]
        )
        took = time.perf_counter() - start
        del recognizer
        release_memory()
        text = text.replace("|", "\\|") or "-"
        rows.append(
            f"| {mode} | {took * 1000:.0f} ms | {took / duration:.3f} | {text} |"
        )
    yield "\n".join([*head, *rows])


def build() -> gr.Blocks:
    t0 = time.perf_counter()
    warm_up(get_recognizer(DEFAULT), DEFAULT["punctuate"])
    startup = time.perf_counter() - t0
    source = f"`{MODEL_ID}`" if LOCAL else f"[{MODEL_ID}]({MODEL_URL})"
    with gr.Blocks(title="Seda STT") as demo:
        gr.Markdown(
            f"# Seda STT\nStreaming Turkish speech recognition on CPU with {source}."
        )
        applied = gr.State(dict(DEFAULT))

        with gr.Sidebar(open=True):
            gr.Markdown("## Inference options")
            preset = gr.Radio(
                [
                    ("fast: greedy search, 0.64 s chunks", "fast"),
                    ("balanced: beam search + LM, 1.28 s chunks", "balanced"),
                    (
                        "accurate: beam search + LM, 0.64 s chunks + second pass",
                        "accurate",
                    ),
                ],
                value="balanced",
                label="Preset",
            )
            use_lm = gr.Checkbox(
                value=DEFAULT["use_lm"],
                label="Language model",
                info="Beam search with the RNN language model. Off: greedy search.",
            )
            two_pass = gr.Checkbox(
                value=DEFAULT["two_pass"],
                label="Two-pass decoding",
                info="Re-decode each finished utterance with full context. Most accurate; "
                "the final text arrives after the second pass.",
            )
            chunk = gr.Radio(
                [(f"{c * 0.02:.2f} s", c) for c in sorted(CHUNKS)],
                value=DEFAULT["chunk"],
                label="Streaming chunk",
                info="Shorter: partial text updates twice as often, slightly less accurate.",
            )
            punctuate = gr.Checkbox(
                value=DEFAULT["punctuate"],
                label="Formatter",
                info="Digits, punctuation, casing and apostrophes on final texts.",
            )
            threads = gr.Slider(
                1,
                MAX_THREADS,
                value=DEFAULT["threads"],
                step=1,
                label="CPU threads",
                info="Threads per recognizer.",
            )
            vad = gr.Checkbox(
                value=DEFAULT["vad"],
                label="Voice activity detection (Silero VAD)",
                info="Detects the end of speech every ~36 ms. Off: the decoder's own rule, "
                "checked once per chunk (slower to react).",
            )
            pause = gr.Slider(
                0.1,
                1.5,
                value=DEFAULT["pause"],
                step=0.05,
                label="End of utterance after a pause of (s)",
                info="Shorter: the final text comes sooner, but a pause inside a sentence may "
                "split it.",
            )
            apply_btn = gr.Button("Apply", variant="primary", interactive=False)
            status = gr.Markdown(
                applied_status(DEFAULT)
                + f" · loaded and warmed up at startup in {startup:.1f} s"
            )
            gr.Markdown(
                "Apply loads the model for the chosen options and runs it once on a short "
                "sample (warm-up), so recordings and files are timed without loading costs."
            )
        options = [use_lm, two_pass, chunk, punctuate, threads, vad, pause]
        preset.change(apply_preset, preset, [use_lm, two_pass, chunk])
        for option in options:
            option.change(
                mark_changed, [applied, *options], [apply_btn, status], queue=False
            )
        apply_btn.click(
            apply_options, options, [applied, status, apply_btn], concurrency_limit=1
        )

        with gr.Tab("Streaming"):
            state = gr.State()
            settings = gr.Markdown()
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            live = gr.Textbox(label="Transcript", lines=6, buttons=["copy"])
            clear_live = gr.Button("Clear", size="sm")
            stats = gr.Markdown()
            gr.Markdown(
                "Options are locked while recording. With the formatter on, the live text is "
                "formatted as you speak. After the pause set in the options the utterance is "
                "finalized in the background (second pass, formatter) and its live text replaced; "
                "the browser sends audio every 0.2 s, which adds up to 0.2 s on screen."
            )
            mic.start_recording(
                start_stream,
                [state, applied],
                [live, stats, settings],
                show_progress="hidden",
            )
            mic.stream(
                stream_mic,
                [state, mic, applied],
                [state, live, stats],
                stream_every=PACKET_SECONDS,
                show_progress="hidden",
            )
            mic.stop_recording(
                stop_stream, state, [state, live, stats], show_progress="hidden"
            )
            # Options and Apply are locked while recording.
            controls = [preset, *options]
            mic.start_recording(
                lambda: [gr.update(interactive=False)] * (len(controls) + 1),
                None,
                [*controls, apply_btn],
            )
            mic.stop_recording(
                lambda a, *v: [
                    *[gr.update(interactive=True)] * len(controls),
                    gr.update(interactive=as_options(*v) != a),
                ],
                [applied, *options],
                [*controls, apply_btn],
            )
            clear_live.click(
                clear_stream, state, [state, live, stats], show_progress="hidden"
            )

        with gr.Tab("File"):
            audio = gr.Audio(sources=["upload", "microphone"], label="Audio")
            out = gr.Textbox(label="Transcript", lines=6, buttons=["copy"])
            info = gr.Markdown()
            with gr.Row():
                gr.Button("Transcribe", variant="primary").click(
                    transcribe, [audio, applied], [out, info]
                )
                compare_btn = gr.Button("Compare presets")
                clear_file = gr.Button("Clear")
            table = gr.Markdown()
            compare_btn.click(compare, [audio, applied], table)
            clear_file.click(lambda: ("", "", ""), None, [out, info, table])

    return demo


if __name__ == "__main__":
    build().launch()

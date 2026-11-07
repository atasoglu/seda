"""Gradio demo for keyword spotting: type keywords, record, and see each match with its confidence.

Usage: uv run python -m sedalib.demo.keywords
       SEDA_MODEL=atasoglu/seda-v0.2 SEDA_CHUNK=16 uv run python -m sedalib.demo.keywords

Two modes, both with short streaming steps (SEDA_CHUNK frames, default 8: 0.16 s):

- Low latency: greedy search. A word it has written never changes, so a keyword is final with its
  confidence as soon as the next word starts, or at the pause.
- High accuracy: beam search + LM, far fewer recognition errors. Words may still change until the
  utterance ends, so a keyword is final at the pause.

A keyword shows up as soon as the streaming pass hears it, greyed out and without a confidence; when
it is final, it gets its calibrated confidence.
Each match shows how long the keyword was spoken, and how much audio after its end it took to show up
and to become final (audio time on the server: the browser sends audio every PACKET_SECONDS on top).
Keywords and the minimum confidence can be changed at any time, also while recording. A keyword counts
from the moment it is added: words said before that are not matched.
"""

import html
import os
from pathlib import Path

import gradio as gr
import numpy as np

from sedalib import KeywordSpotter, Recognizer
from sedalib.keywords import _spoken
from sedalib.stt import _Tokens

# The tiny model: 7 MB, decodes a live stream with little CPU. Calibrated like seda-v0.2.
MODEL_ID = os.environ.get("SEDA_MODEL", "atasoglu/seda-v0.2-tiny")
# Streaming step in 20 ms frames: 8 (0.16 s) hears a word soonest; 16, 32 or 64 are more accurate.
CHUNK = int(os.environ.get("SEDA_CHUNK", "8"))
# Demo mode -> sedalib mode, whose chunk is overridden with CHUNK.
MODES = {"Low latency": "fast", "High accuracy": "balanced"}
# Pause that ends an utterance and gives the final match. Shorter than the library default (0.25 s):
# splitting a sentence at a hesitation costs little here.
ENDPOINT_SILENCE = float(os.environ.get("SEDA_ENDPOINT_SILENCE", "0.15"))
# Audio is sent by the browser in pieces of this many seconds.
PACKET_SECONDS = 0.1
EXAMPLE_KEYWORDS = ["deneme", "bir", "iki", "üç"]

CSS = """
/* Centered page, the mode on the left and everything else on the right; stacked on narrow screens. */
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
#kw-settings { gap: 0.75rem; }
#kw-settings .kw-panel-title, #kw-main .kw-panel-title { font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--body-text-color-subdued); }
#title h1 { font-size: 1.6rem; font-weight: 600; margin: 0; }
.kw-model { font-size: 0.85rem; color: var(--body-text-color-subdued); }
#title p { color: var(--body-text-color-subdued); margin: 0.25rem 0 0; }
.kw-live { display: flex; gap: 0.75rem; align-items: baseline; min-height: 1.5rem; }
.kw-clock { font-variant-numeric: tabular-nums; font-weight: 600; flex: none; }
.kw-clock.on::before { content: "●"; color: #e5484d; margin-right: 0.35rem; font-size: 0.8em; }
.kw-partial { color: var(--body-text-color-subdued); font-style: italic; overflow-wrap: anywhere; }
.kw-pending { opacity: 0.5; border-style: dashed; }
.kw-empty { color: var(--body-text-color-subdued); padding: 1.5rem 0; text-align: center; }
/* A fixed panel: new matches go on top and older ones scroll inside it, the page does not grow. */
#kw-results { height: 22rem !important; overflow-y: auto !important; overscroll-behavior: contain; padding: 0.5rem;
  border: 1px solid var(--border-color-primary); border-radius: 0.75rem; }
.kw-list { display: flex; flex-direction: column; gap: 0.5rem; }
.kw-row { display: grid; grid-template-columns: minmax(0, 1fr) 7rem 3.2rem; align-items: center;
  gap: 0.75rem; padding: 0.6rem 0.85rem; border: 1px solid var(--border-color-primary);
  border-radius: 0.6rem; background: var(--background-fill-primary); }
.kw-word { font-weight: 600; overflow-wrap: anywhere; }
.kw-heard { color: var(--body-text-color-subdued); font-weight: 400; margin-left: 0.4rem; }
.kw-time { color: var(--body-text-color-subdued); font-size: 0.85rem; margin-left: 0.4rem;
  font-variant-numeric: tabular-nums; }
.kw-legend { display: flex; flex-direction: column; gap: 0.1rem; font-size: 0.8rem;
  color: var(--body-text-color-subdued); margin-bottom: 0.25rem; }
.kw-legend b { color: var(--body-text-color); font-weight: 600; margin-right: 0.25rem; }
.kw-meta span[title] { cursor: help; }
.kw-meta { color: var(--body-text-color-subdued); font-size: 0.8rem; margin-top: 0.15rem;
  font-variant-numeric: tabular-nums; }
.kw-bar { height: 0.4rem; border-radius: 1rem; background: var(--border-color-primary); overflow: hidden; }
.kw-bar > div { height: 100%; border-radius: 1rem; background: var(--color-accent); }
.kw-conf { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
@media (max-width: 480px) { .kw-row { grid-template-columns: minmax(0, 1fr) 3.2rem; }
  .kw-bar { display: none; } }
"""


def load_recognizer(mode: str) -> Recognizer:
    options = {"mode": mode, "chunk": CHUNK, "endpoint_silence": ENDPOINT_SILENCE}
    if Path(MODEL_ID).is_dir():
        return Recognizer(MODEL_ID, **options)
    return Recognizer.from_pretrained(MODEL_ID, **options)


# Both are loaded at start, so switching between recordings is instant.
RECOGNIZERS = {name: load_recognizer(mode) for name, mode in MODES.items()}
DEFAULT_MODE = next(iter(MODES))


def early_final(r: Recognizer) -> bool:
    """With greedy search and no second pass, a word the streaming pass has written never changes:
    once the next word has started, its confidence is the one it gets at the end of the utterance,
    and a keyword is final without waiting for the speaker to pause."""
    return not r.use_lm and not r.two_pass and r._calibration is not None


def model_info(name: str) -> str:
    """The model and decoding setup of a demo mode, for the page header."""
    r = RECOGNIZERS[name]
    search = "beam search + LM" if r.use_lm else "greedy"
    final = ", second pass" if r.two_pass else ""
    when_final = "next word or pause" if early_final(r) else "at the pause"
    confidence = (
        f"confidence final: {when_final}"
        if r._calibration
        else "no calibration for this setup: confidence –"
    )
    return (
        f"<div class='kw-model'>Model <b>{html.escape(Path(MODEL_ID).name)}</b> · "
        f"{r.chunk * 0.02:.2f} s steps, {search}{final} · end of speech after "
        f"{r.endpoint_silence:.2f} s · {confidence}</div>"
    )


def to_mono_float(audio) -> tuple[np.ndarray, int]:
    sample_rate, data = audio
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sample_rate


def valid(keywords: list[str] | None) -> list[str]:
    return [k for k in keywords or [] if any(c.isalnum() for c in k)]


def clock(seconds: float, recording: bool, text: str = "") -> str:
    m, s = divmod(seconds, 60)
    on = " on" if recording else ""
    return (
        f'<div class="kw-live"><span class="kw-clock{on}">{int(m)}:{s:04.1f}</span>'
        f'<span class="kw-partial">{html.escape(text)}</span></div>'
    )


def heard_so_far(stream) -> list:
    """The words of the utterance being spoken, from the streaming pass. Only the confidences of the
    words before the last one are final, and only with early_final()."""
    r = stream._owner
    with stream._lock:
        tokens = _Tokens.of(r._streaming, stream._stream)
        return tokens.words(stream._start, r._calibration if early_final(r) else None)


def when(state: dict, name: str, d) -> float | None:
    """Seconds of audio accepted when the keyword was first shown (name "seen") or became final
    before the end of its utterance ("final")."""
    for keyword, start, seconds in state.get(name, []):
        if keyword == d.keyword and abs(start - d.start) < 0.3:
            return seconds
    return None


def first_seen(state: dict, d) -> float | None:
    return when(state, "seen", d)


# The metrics under each match, with their explanation (shown in the legend and on hover).
METRICS = {
    "long": "how long the keyword was spoken",
    "shown": "seconds after the keyword ended that it appeared, greyed out",
    "final": "seconds after the keyword ended that it got its confidence: in Low latency when the"
    " next word starts or when you pause, whichever comes first; in High accuracy when you pause",
}
LEGEND = (
    '<div class="kw-legend">'
    + "".join(f"<span><b>{name}</b> {text}</span>" for name, text in METRICS.items())
    + "<span>Times count audio received by the server; the 0:12.3 next to a keyword is when it "
    "was said in the recording.</span></div>"
)


def metrics(d, seen: float | None, final: float | None) -> str:
    parts = [("long", f"{d.end - d.start:.2f} s long")]
    if seen is not None:
        parts.append(("shown", f"shown {seen - d.end:+.2f} s"))
    if final is not None:
        parts.append(("final", f"final {final - d.end:+.2f} s"))
    spans = (f'<span title="{METRICS[name]}">{text}</span>' for name, text in parts)
    return f'<div class="kw-meta">{" · ".join(spans)}</div>'


def row(
    d, pending: bool = False, seen: float | None = None, final: float | None = None
) -> str:
    heard = (
        f'<span class="kw-heard">“{html.escape(d.text)}”</span>'
        if d.text != " ".join(_spoken(d.keyword))
        else ""
    )
    m, s = divmod(d.start, 60)
    conf = 0.0 if pending else d.confidence or 0.0
    if pending:
        label = "…"
    else:  # None: the model has no calibration for this setup
        label = f"{conf * 100:.0f}%" if d.confidence is not None else "–"
    css = "kw-row kw-pending" if pending else "kw-row"
    return (
        f'<div class="{css}"><div><span class="kw-word">{html.escape(d.keyword)}</span>'
        f'{heard}<span class="kw-time">{int(m)}:{s:04.1f}</span>'
        f"{metrics(d, seen, final)}</div>"
        f'<div class="kw-bar"><div style="width:{conf * 100:.0f}%"></div></div>'
        f'<div class="kw-conf">{label}</div></div>'
    )


def render(
    state: dict | None, keywords: list[str] | None, min_confidence: float
) -> str:
    keywords = valid(keywords)
    if not keywords:
        return '<div class="kw-empty">Add keywords above, then start recording.</div>'
    # Matching does not use the recognizer: any one will do.
    spotter = KeywordSpotter(
        RECOGNIZERS[DEFAULT_MODE], keywords, min_confidence=min_confidence
    )
    state = state or {}
    added = state.get("added", {})

    def counts(d) -> bool:  # said after the keyword was added
        return d.start >= added.get(d.keyword, 0.0)

    # (detection, seconds of audio when its utterance ended)
    found = [
        (d, cut)
        for words, cut in state.get("utterances", [])
        for d in spotter.match(words)
        if counts(d)
    ]
    # In the utterance being spoken: final ones (see early_final) are kept by their confidence, the
    # others are shown greyed out.
    rows = []
    heard = KeywordSpotter(RECOGNIZERS[DEFAULT_MODE], keywords, min_confidence=0)
    for d in reversed(heard.match(state.get("pending", []))):
        final = when(state, "final", d)
        if not counts(d) or (final is not None and d.confidence < min_confidence):
            continue
        rows.append(
            row(d, pending=final is None, seen=first_seen(state, d), final=final)
        )
    for d, cut in reversed(found):
        final = min(cut, when(state, "final", d) or cut)
        rows.append(row(d, seen=first_seen(state, d), final=final))
    if not rows:
        listening = state and not state["stopped"]
        return f'<div class="kw-empty">{"Listening…" if listening else "No matches yet."}</div>'
    # Newest first.
    return f'<div class="kw-list">{"".join(rows)}</div>'


def on_keywords(state, keywords, min_confidence):
    """Keywords added now count from the current point of the recording, not retroactively."""
    if state is not None:
        now = state["stream"]._seconds
        current = valid(keywords)
        state["added"] = {k: state["added"].get(k, now) for k in current}
    return state, render(state, keywords, min_confidence)


def start(mode, keywords, min_confidence):
    state = {
        "stream": RECOGNIZERS[mode].stream(),
        "utterances": [],
        "pending": [],
        "seen": [],  # (keyword, start, seconds of audio when first shown)
        "final": [],  # (keyword, start, seconds of audio when final before the utterance ended)
        # Seconds of audio when each keyword was added; the ones there at the start count from 0.
        "added": {k: 0.0 for k in valid(keywords)},
        "stopped": False,
        "html": "",
    }
    state["html"] = render(state, keywords, min_confidence)
    # The mode is fixed while recording.
    return state, state["html"], clock(0.0, True), gr.update(interactive=False)


def on_audio(state, audio, keywords, min_confidence):
    if state is None or state["stopped"] or audio is None:
        return state, gr.skip(), gr.skip()
    stream = state["stream"]
    samples, sample_rate = to_mono_float(audio)
    partial = stream.accept(samples, sample_rate)
    if stream.is_endpoint:
        state["utterances"].append((stream.cut().words, stream._seconds))
        state["pending"], partial = [], ""
    else:
        state["pending"] = heard_so_far(stream)
        if valid(keywords):
            spotter = KeywordSpotter(stream._owner, valid(keywords), min_confidence=0)
            last = state["pending"][-1] if state["pending"] else None
            early = early_final(stream._owner)
            for d in spotter.match(state["pending"]):
                if first_seen(state, d) is None:
                    state["seen"].append((d.keyword, d.start, stream._seconds))
                # The next word has started: the keyword's words are final.
                if early and d.end <= last.start and when(state, "final", d) is None:
                    state["final"].append((d.keyword, d.start, stream._seconds))
    results = render(state, keywords, min_confidence)
    changed = results != state["html"]
    state["html"] = results
    return (
        state,
        results if changed else gr.skip(),
        clock(stream._seconds, True, partial),
    )


def stop(state, keywords, min_confidence):
    if state is None or state["stopped"]:
        return state, gr.skip(), gr.skip(), gr.update(interactive=True)
    stream = state["stream"]
    state["utterances"].append((stream.cut().words, stream._seconds))
    state["pending"], state["stopped"] = [], True
    state["html"] = render(state, keywords, min_confidence)
    return (
        state,
        state["html"],
        clock(stream._seconds, False),
        gr.update(interactive=True),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="Seda keyword spotting") as demo:
        gr.Markdown(
            "# Keyword spotting\nSay the words you are listening for. Matches appear below with "
            "the probability that they were heard right.",
            elem_id="title",
        )
        with gr.Row(equal_height=False):
            # Left: only the mode and the model it runs.
            with (
                gr.Column(scale=1, min_width=260, elem_id="kw-settings"),
                gr.Group(),
            ):
                mode = gr.Radio(
                    list(MODES),
                    value=DEFAULT_MODE,
                    label="Mode",
                    info="Low latency: confidence as soon as the next word starts. High "
                    "accuracy: fewer recognition errors, confidence at the pause.",
                )
                model_line = gr.HTML(model_info(DEFAULT_MODE), padding=True)
            with gr.Column(scale=3, min_width=320, elem_id="kw-main"):
                keywords = gr.Dropdown(
                    choices=EXAMPLE_KEYWORDS,
                    value=EXAMPLE_KEYWORDS,
                    multiselect=True,
                    allow_custom_value=True,
                    label="Keywords",
                    info="Type a word or phrase and press Enter. Suffixes are matched too: "
                    "“kamp” finds “kampa”.",
                )
                min_confidence = gr.Slider(
                    0, 1, value=0.5, step=0.05, label="Minimum confidence"
                )
                mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    show_label=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                partial = gr.HTML(clock(0.0, False))
                gr.HTML(LEGEND)
                results = gr.HTML(
                    render(None, EXAMPLE_KEYWORDS, 0.5), elem_id="kw-results"
                )
        state = gr.State()

        mode.change(model_info, mode, model_line, show_progress="hidden")
        mic.start_recording(
            start,
            [mode, keywords, min_confidence],
            [state, results, partial, mode],
            show_progress="hidden",
        )
        mic.stream(
            on_audio,
            [state, mic, keywords, min_confidence],
            [state, results, partial],
            stream_every=PACKET_SECONDS,
            show_progress="hidden",
        )
        mic.stop_recording(
            stop,
            [state, keywords, min_confidence],
            [state, results, partial, mode],
            show_progress="hidden",
        )
        keywords.change(
            on_keywords,
            [state, keywords, min_confidence],
            [state, results],
            show_progress="hidden",
        )
        min_confidence.change(
            render,
            [state, keywords, min_confidence],
            results,
            show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    build().launch(theme=gr.themes.Base(primary_hue="slate", radius_size="lg"), css=CSS)

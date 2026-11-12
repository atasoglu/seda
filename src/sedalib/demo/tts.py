"""Gradio demo for Turkish speech synthesis: type text, pick a style, hear it.

Usage: uv run python -m sedalib.demo.tts

The model is downloaded from the Hugging Face Hub. ``SEDA_TTS_MODEL`` picks another one: a local directory
or a repo id.

The audio plays in one waveform player. Under it are the time to the first chunk (what an application
that plays the chunks of ``Synthesizer.stream`` waits before playback) and the real-time factor (RTF). The voices are synthetic: label audio made with them as such.
"""

import os
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np

from sedalib import Synthesizer
from sedalib.tts import DEFAULT_TTS_REPO_ID

MODEL = os.environ.get("SEDA_TTS_MODEL", DEFAULT_TTS_REPO_ID)
# text, style, emotion: a sentence that carries its emotion, so the voice and the words agree
EXAMPLES = [
    [
        "Sınavı geçtim! İnanamıyorum, bütün o geceler sonunda işe yaradı. Hadi bu akşam kutlayalım!",
        "normal",
        "happy",
    ],
    [
        "Bütün yaz bu geziyi bekledim ama son anda iptal edildi. Şimdi hiçbir şey yapmak istemiyorum.",
        "normal",
        "sad",
    ],
    [
        "Yeter artık! Kaç kere söyleyeceğim sana? Arabamı bir daha sormadan alırsan çok fena olur!",
        "normal",
        "angry",
    ],
    [
        "Biri içeride! Kapının arkasında bir gölge var, çabuk polisi ara! Lütfen acele et!",
        "normal",
        "fearful",
    ],
    [
        "Ne? Gerçekten mi? Piyangodan büyük ikramiye bize mi çıktı? Bu imkânsız, inanamıyorum!",
        "normal",
        "surprised",
    ],
    [
        "Toplantı 12 Mayıs'ta saat 14:30'da başlayacak; katılım ücreti 250 TL.",
        "normal",
        "neutral",
    ],
    ["Uyu artık, her şey yolunda. Ben hemen yanındayım.", "whisper", "calm"],
    ["Sokak çok karanlık. Biri bizi takip ediyor, korkuyorum.", "whisper", "fearful"],
    [
        "Konser biletleri tükenmiş, bu akşam gidemiyoruz. Çok üzüldüm ama ona belli etme.",
        "whisper",
        "sad",
    ],
    [
        "Anahtar masanın sağ çekmecesinde, mavi defterin arasında. Kimseye söyleme.",
        "whisper",
        "neutral",
    ],
]
AUTO = "auto (from the text)"

_synth: Synthesizer | None = None
# one request decodes at a time: a single thread is the low-latency setup
_lock = threading.Lock()


def _load() -> Synthesizer:
    global _synth
    if _synth is None:
        if Path(MODEL).is_dir():
            _synth = Synthesizer(MODEL)
        else:
            _synth = Synthesizer.from_pretrained(MODEL)
        _synth.synthesize("ısınma")  # first run pays for graph optimization
    return _synth


def speak(text, speaker, emotion, style, noise_scale, noise_scale_w, length_scale):
    """The whole audio for one waveform player.

    Gradio buffers a streamed output for about a second before it plays and can only show it in the
    browser's plain player, so the demo plays the finished audio and reports the latency of chunked playback.
    """
    if not text.strip():
        raise gr.Error("Type some text.")
    synth = _load()
    if emotion != AUTO and emotion not in synth.emotions_for(style):
        raise gr.Error(f"The {style} style has no {emotion} emotion.")
    chunks = []
    with _lock:
        start = time.perf_counter()
        first = None
        for chunk in synth.stream(
            text,
            speaker=speaker,
            emotion=None if emotion == AUTO else emotion,
            style=style,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        ):
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk)
        total = time.perf_counter() - start
    if not chunks:
        raise gr.Error("Nothing to say: the text has no letters.")
    wav = np.concatenate(chunks)
    seconds = len(wav) / synth.sample_rate
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    return (synth.sample_rate, pcm), (
        f"{seconds:.2f} s of audio · first chunk after {first * 1000:.0f} ms · "
        f"RTF {total / seconds:.3f}"
    )


def style_changed(style, emotion):
    """A style has its own emotions and settings: a whisper at the normal settings sounds buzzing."""
    synth = _load()
    d = synth.defaults_for(style)
    choices = [AUTO, *synth.emotions_for(style)]
    return (
        gr.update(choices=choices, value=emotion if emotion in choices else AUTO),
        d["noise_scale"],
        d["noise_scale_w"],
        d["length_scale"],
    )


def build() -> gr.Blocks:
    synth = _load()
    d = synth.defaults_for("normal")
    with gr.Blocks(title="Seda TTS") as demo:
        gr.Markdown("# Seda TTS\nTurkish speech synthesis. The voices are synthetic.")
        waveform = gr.Audio(
            label="Audio",
            interactive=False,
            editable=False,
            autoplay=True,
            buttons=["download"],
        )
        stats = gr.Markdown()

        text = gr.Textbox(
            label="Text", lines=3, value=EXAMPLES[0][0], placeholder="Metni yazın"
        )
        button = gr.Button("Speak", variant="primary")
        with gr.Row():
            speaker = gr.Dropdown(
                synth.speakers,
                value=synth.default_speaker,
                label="Speaker",
                visible=len(synth.speakers) > 1,
            )
            style = gr.Dropdown(synth.styles, value="normal", label="Style")
            emotion = gr.Dropdown(
                [AUTO, *synth.emotions_for("normal")],
                value=EXAMPLES[0][2],
                label="Emotion",
                # an example sets style and emotion together, before the style narrows the choices
                allow_custom_value=True,
            )
        gr.Examples(
            EXAMPLES,
            inputs=[text, style, emotion],
            label="Examples",
            examples_per_page=len(EXAMPLES),
        )
        with gr.Accordion("Advanced", open=False), gr.Row():
            noise_scale = gr.Slider(
                0.0, 1.0, d["noise_scale"], step=0.05, label="Voice variation"
            )
            noise_scale_w = gr.Slider(
                0.0, 1.0, d["noise_scale_w"], step=0.05, label="Rhythm variation"
            )
            length_scale = gr.Slider(
                0.5, 2.0, d["length_scale"], step=0.05, label="Duration (×)"
            )

        style.change(
            style_changed,
            [style, emotion],
            [emotion, noise_scale, noise_scale_w, length_scale],
        )
        inputs = [
            text,
            speaker,
            emotion,
            style,
            noise_scale,
            noise_scale_w,
            length_scale,
        ]
        gr.on(
            [button.click, text.submit],
            speak,
            inputs,
            [waveform, stats],
        )
    return demo


if __name__ == "__main__":
    build().launch()

# CLAUDE.md

## Purpose

Seda builds and publishes Turkish speech models (STT, TTS, etc.). `sedalib` is the Python package that runs them. It currently wraps sherpa-onnx for streaming speech recognition with the [atasoglu/seda-v0.1](https://huggingface.co/atasoglu/seda-v0.1) model.

## Philosophy

- **Low latency:** suitable for real-time, streaming use.
- **Lightweight:** low RAM and CPU usage. Add a runtime dependency only when it is essential. Optional features go into extras, such as `demo`.
- **Production quality:** code and models should be reliable enough to ship.
- **Free for commercial use:** every dependency and model must allow it.
- **Simple and modular:** small, focused modules. Model-specific constants and logic stay in their task module (`stt.py`), and shared helpers (`hub.py`, `audio.py`) stay generic.

## Commands

Always use `uv`. Never use bare `pip` or `python`.

```bash
uv sync --all-extras                                   # set up the environment
uv run prek install                                    # install git hooks
uv run prek run --all-files                            # lint and format, same as CI
uv run python -m sedalib.demo.stt                      # Gradio demo
uv run --group examples python examples/streaming.py   # microphone example
```

## Development notes

- Python >= 3.10, `src/` layout.
- The version comes from git tags (hatch-vcs). Never hardcode it. Pushing a `v*` tag publishes to PyPI, and the release notes come from `CHANGELOG.md`.
- Update `CHANGELOG.md` (Keep a Changelog) for user-facing changes.
- No tests yet.
- The README is in English and stays concise: no file-structure section and no `---` dividers.
- Ask the maintainer to review before committing.

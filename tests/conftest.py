"""Shared fixtures.

Tests that need a model run once per entry of ``SEDA_TEST_MODEL``: a comma-separated list of local model
directories or Hugging Face repo ids, e.g. ``SEDA_TEST_MODEL=atasoglu/seda-v0.2,atasoglu/seda-v0.1``.
Without it they are skipped.
"""

import os
from pathlib import Path

import pytest

from sedalib import download, read_wav
from sedalib import stt

DATA = Path(__file__).parent / "data"
# tests/data/speech.wav, FLEURS (CC BY 4.0), Turkish dev split.
SPEECH_TEXT = "bazı festivaller küçük çocuklu aileler için özel kamp alanlarına sahiptir"
_ALL_FILES = (
    [stt._streaming_file(c, f) for c in stt.CHUNKS for f in stt._STREAMING_FILES]
    + ["tokens.txt", stt._VAD_FILE]
    + stt._LM_FILES
    + stt._FINAL_FILES
    + stt._FORMATTER_FILES
)


def _models() -> list[str]:
    return [m.strip() for m in os.environ.get("SEDA_TEST_MODEL", "").split(",") if m.strip()]


def pytest_generate_tests(metafunc):
    if "model_dir" in metafunc.fixturenames:
        models = _models()
        metafunc.parametrize("model_dir", models or [None], ids=models or ["no-model"], indirect=True)


@pytest.fixture(scope="session")
def _downloads() -> dict[str, Path]:
    return {}


@pytest.fixture
def model_dir(request, _downloads) -> Path:
    if request.param is None:
        pytest.skip("set SEDA_TEST_MODEL to run tests that need a model")
    name = request.param
    if name not in _downloads:
        path = Path(name)
        # Only the files sedalib uses; patterns a model lacks are skipped.
        _downloads[name] = path if path.is_dir() else download(name, files=_ALL_FILES)
    return _downloads[name]


@pytest.fixture(scope="session")
def speech() -> tuple[list[float], int]:
    return read_wav(DATA / "speech.wav")


def supports(model_dir: Path, mode: str = "balanced", punctuate: bool = False) -> bool:
    """Whether the model ships the files for a mode (seda-v0.1 has only the root model and the LM)."""
    lm, two_pass, chunk = stt._resolve(mode, None, None, None)
    return all((model_dir / f).is_file() for f in stt._required_files(lm, two_pass, chunk, punctuate))

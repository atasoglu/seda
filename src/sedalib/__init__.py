from importlib.metadata import version

from .audio import read_wav
from .formatter import Formatter
from .hub import download
from .itn import itn
from .stt import (
    CHUNKS,
    DEFAULT_REPO_ID,
    MODES,
    SAMPLE_RATE,
    Recognizer,
    Stream,
    Utterance,
)

__version__ = version("sedalib")

__all__ = [
    "CHUNKS",
    "DEFAULT_REPO_ID",
    "MODES",
    "SAMPLE_RATE",
    "Formatter",
    "Recognizer",
    "Stream",
    "Utterance",
    "__version__",
    "download",
    "itn",
    "read_wav",
]

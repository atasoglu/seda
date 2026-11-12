from importlib.metadata import version

from .audio import read_wav, write_wav
from .formatter import Formatter
from .hub import download
from .itn import itn
from .keywords import Detection, KeywordSpotter, KeywordStream
from .stt import (
    CHUNKS,
    DEFAULT_REPO_ID,
    MODES,
    SAMPLE_RATE,
    Recognizer,
    Stream,
    Utterance,
    Word,
)
from .tts import Synthesizer

__version__ = version("sedalib")

__all__ = [
    "CHUNKS",
    "DEFAULT_REPO_ID",
    "MODES",
    "SAMPLE_RATE",
    "Detection",
    "Formatter",
    "KeywordSpotter",
    "KeywordStream",
    "Recognizer",
    "Stream",
    "Synthesizer",
    "Utterance",
    "Word",
    "__version__",
    "download",
    "itn",
    "read_wav",
    "write_wav",
]

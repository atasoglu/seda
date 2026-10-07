from importlib.metadata import version

from .audio import read_wav
from .hub import download
from .stt import DEFAULT_REPO_ID, SAMPLE_RATE, Recognizer, Stream

__version__ = version("sedalib")

__all__ = [
    "DEFAULT_REPO_ID",
    "SAMPLE_RATE",
    "Recognizer",
    "Stream",
    "__version__",
    "download",
    "read_wav",
]

from pathlib import Path

from huggingface_hub import snapshot_download


def download(
    repo_id: str,
    *,
    revision: str | None = None,
    files: list[str] | None = None,
) -> Path:
    """Download a model repo (or only the given files) from the Hugging Face Hub and return the local directory."""
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=files))

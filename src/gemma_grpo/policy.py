from __future__ import annotations

import hashlib
from pathlib import Path


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_fingerprint(path: str | Path) -> str:
    directory = Path(path)
    digest = hashlib.sha256()
    for name in ADAPTER_FILES:
        file_path = directory / name
        if not file_path.is_file():
            raise ValueError(f"Adapter {directory} is missing {name}")
        digest.update(name.encode())
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"lora-sha256:{digest.hexdigest()}"


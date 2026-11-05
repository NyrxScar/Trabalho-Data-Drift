"""Helpers for persisted artefacts: content hashing and strict JSON I/O."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HASH_CHUNK_BYTES: int = 1 << 20


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (used for data lineage)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    """Recursively convert values into strict-JSON-compatible Python objects.

    NaN and +/-inf become ``None`` (``null``); numpy scalars become Python
    scalars; tuples become lists; mapping keys become strings.
    """
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` as indented UTF-8 JSON, rejecting NaN/inf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write {path}.")
    return path


def _reject_constant(constant: str) -> None:
    raise ValueError(f"Invalid JSON constant: {constant}")


def read_json(path: Path) -> Any:
    """Read a strict JSON file (``NaN``/``Infinity`` literals are rejected).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the content is not valid strict JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)

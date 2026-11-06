"""Durable filesystem primitives shared by the tools and the product store.

The filesystem is the platform's source of truth, so these helpers exist to make
one guarantee: a crash at any instant leaves either the old complete file or the
new complete file, never a torn mixture of the two.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str | None:
    """Return the file's SHA-256, or None when it does not exist."""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path: Path | str) -> None:
    """Flush a directory entry so a rename survives power loss."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path | str, data: bytes, mode: int | None = None) -> None:
    """Replace *path* with *data* atomically (temp file, fsync, rename, dir fsync)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        mode = target.stat().st_mode & 0o777 if target.exists() else 0o644

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
        fsync_dir(target.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path | str, text: str, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def append_line(path: Path | str, line: str) -> None:
    """Durably append one line, healing a torn tail left by an earlier crash.

    A crash mid-append can leave a partial final line.  Terminating it first
    keeps every later record on a line of its own, so readers lose at most the
    one record that was being written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = line.rstrip("\n").encode("utf-8") + b"\n"

    needs_newline = False
    if target.exists() and target.stat().st_size > 0:
        with open(target, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"

    fd = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (b"\n" if needs_newline else b"") + payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path | str) -> tuple[list[dict[str, Any]], int]:
    """Return (records, skipped_line_count), tolerating a torn line from a crash."""
    target = Path(path)
    if not target.exists():
        return [], 0
    records: list[dict[str, Any]] = []
    skipped = 0
    with open(target, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                skipped += 1
    return records, skipped

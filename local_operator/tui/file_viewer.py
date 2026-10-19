"""Bounded local document reads shared by the slash and callout routes.

The affordance is authority, not the suffix router. Revalidate the original
spelling and the descriptor identity: resolving a symlink once is not a licence
to read its replacement. No parser, extraction, network request or write belongs
on this path. The async worker yields between bounded local reads and owns its
file descriptor directly, so cancellation cannot strand a ``to_thread`` reader.
"""
from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from local_operator.tui.link_targets import file_affordance

MAX_BYTES = 1024 * 1024
MAX_LINES = 20_000
CHUNK_BYTES = 64 * 1024
Identity = tuple[int, int, int, int]

_CODE = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript",
    ".tsx": "tsx", ".json": "json", ".jsonl": "json", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".xml": "xml", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".java": "java", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".rb": "ruby", ".swift": "swift", ".kt": "kotlin",
}
TEXT_SUFFIXES = frozenset({".txt", ".text", ".log", ".rst", ".csv", ".tsv"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
VIEWER_SUFFIXES = frozenset(_CODE) | TEXT_SUFFIXES | MARKDOWN_SUFFIXES


class ViewerError(ValueError):
    """A catalogue key suffix, never an OS error that might expose extra paths."""


def viewer_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _CODE:
        return "code"
    return "text" if suffix in TEXT_SUFFIXES else None


def lexer_for(path: Path) -> str:
    return _CODE.get(path.suffix.lower(), "text")


def file_identity(path: Path) -> Identity:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class Document:
    path: Path
    text: str
    kind: str
    size: int
    truncated: bool
    identity: Identity


def _read_steps(
    raw: str, *, cwd: Path, expected: Identity | None = None
) -> Generator[None, None, Document]:
    # Absolute /view arguments are converted to the affordance's session-relative
    # spelling, not trusted through a separate absolute-path bypass.
    spelling = os.path.relpath(raw, cwd) if Path(raw).is_absolute() else raw
    uri = file_affordance(spelling, cwd=cwd)
    if uri is None:
        raise ViewerError("unreadable")
    path = Path(unquote(urlparse(uri).path))
    kind = viewer_kind(path)
    if kind is None:
        raise ViewerError("unsupported")
    try:
        before = file_identity(path)
        if expected is not None and expected != before:
            raise ViewerError("changed")
        # O_NOFOLLOW closes the final-component replacement race after the
        # affordance resolved it. fstat also catches a changed parent/symlink.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ViewerError("unreadable")
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if identity != before:
                raise ViewerError("changed")
            data = bytearray()
            lines = 0
            truncated = False
            while len(data) <= MAX_BYTES:
                chunk = source.read(min(CHUNK_BYTES, MAX_BYTES + 1 - len(data)))
                if not chunk:
                    break
                if b"\x00" in chunk or any(c < 9 or 13 < c < 32 for c in chunk):
                    raise ViewerError("binary")
                data.extend(chunk)
                lines += chunk.count(b"\n")
                if lines > MAX_LINES or len(data) > MAX_BYTES:
                    truncated = True
                    break
                yield None
            if file_identity(path) != identity or file_affordance(spelling, cwd=cwd) != uri:
                raise ViewerError("changed")
    except OSError as exc:
        raise ViewerError("unreadable") from exc
    bounded = bytes(data[:MAX_BYTES])
    # Slice bytes/lines BEFORE decoding: invalid UTF-8 replacement must not
    # change where the resource bound is enforced.
    rows = bounded.splitlines(keepends=True)
    if len(rows) > MAX_LINES:
        truncated = True
        bounded = b"".join(rows[:MAX_LINES])
    return Document(path, bounded.decode("utf-8", errors="replace"), kind, info.st_size,
                    truncated, identity)


def read_document(raw: str, *, cwd: Path, expected: Identity | None = None) -> Document:
    steps = _read_steps(raw, cwd=cwd, expected=expected)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


async def read_document_async(
    raw: str, *, cwd: Path, expected: Identity | None = None
) -> Document:
    steps = _read_steps(raw, cwd=cwd, expected=expected)
    try:
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value
            await asyncio.sleep(0)
    finally:
        # Only affordance-checked regular files reach this worker. Chunk and
        # total caps bound each read before decode. No thread means no orphan:
        # Esc/reload cancellation closes the generator and descriptor here.
        steps.close()

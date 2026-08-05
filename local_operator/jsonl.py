"""JSON Lines (newline-delimited JSON) reading and writing.

Agent state — conversation history, execution history, learnings, schedules —
is persisted one JSON object per line so that a long history streams in and
out without holding a second parsed copy of the whole file, and so a partially
written file is still readable up to the last complete line.

The format is one line of ``json.dumps`` output per record, UTF-8 encoded,
each terminated by ``\\n``. That is small enough to own outright rather than
carry a dependency (and its transitive ``attrs``) for, and owning it means the
exact on-disk encoding is pinned here instead of inherited from a library
default that could shift under us.

Encoding contract (must not change without migrating existing state files):

* ``ensure_ascii=False`` — non-ASCII characters are written literally as UTF-8
  rather than ``\\uXXXX`` escapes. Conversation text is mostly prose, so this
  keeps files legible and materially smaller.
* Default separators (``", "`` / ``": "``). Not compact, but this is what
  existing state files on disk already use.
* Records must be JSON-native. Values such as ``datetime`` raise
  :class:`TypeError` from :func:`json.dumps`; callers that hold rich types are
  expected to dump them to JSON-safe primitives first (for pydantic models,
  ``model_dump(mode="json")``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, IO, Union

__all__ = ["InvalidLineError", "dump_jsonl", "read_jsonl", "write_jsonl"]

StrOrPath = Union[str, Path]


class InvalidLineError(ValueError):
    """A line in a JSON Lines file could not be decoded.

    Carries the 1-based line number so a corrupted state file can be pointed
    at directly instead of reported as an opaque parse failure.
    """

    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(f"invalid JSON on line {line_number}: {reason}")
        self.line_number = line_number


def read_jsonl(path: StrOrPath) -> Iterator[Any]:
    """Yield each record from the JSON Lines file at ``path``.

    Reading is lazy: the file handle stays open until the iterator is
    exhausted (or garbage collected), so callers that stop early should
    ``close()`` the generator or simply consume it fully.

    Every line must decode, including blank ones. Silently skipping
    undecodable lines would let a truncated write turn into quietly missing
    history, which is far worse than a loud failure the caller can log and
    recover from.

    Raises:
        InvalidLineError: If any line is not a complete JSON value.
        OSError: If the file cannot be read.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise InvalidLineError(line_number, str(exc)) from exc


def dump_jsonl(handle: IO[str], records: Iterable[Any]) -> None:
    """Write ``records`` to an already-open text handle, one JSON per line."""
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def write_jsonl(path: StrOrPath, records: Iterable[Any]) -> None:
    """Write ``records`` to ``path`` as JSON Lines, replacing any existing file.

    Raises:
        TypeError: If a record contains a value :mod:`json` cannot encode. The
            file is left truncated at that point — callers treat a failed save
            as a failed save rather than trying to recover a partial one.
        OSError: If the file cannot be written.
    """
    with open(path, "w", encoding="utf-8") as handle:
        dump_jsonl(handle, records)

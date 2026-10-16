"""Disk publication boundaries for the REAL peer-session test fixture reader.

A peer can append unrelated journal entries while a viewer assertion reads the
file. The last LF publishes a complete JSONL record; it is not an fsync/durability
claim. These tests pause one real open writer without sockets, Session/runtime
boot, sleeps, retries or an AST-extracted/replacement baseline parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.network.test_session_plane import _RealServed


def _reader(path: Path) -> _RealServed:
    # Run the actual constructor. Its other capabilities are deliberately
    # unavailable: this partial fixture exercises ONLY disk transcript reading,
    # not on_session_loop/runtime disposal, and needs no loop or thread boot.
    unused_runtime_hook: Any = None
    return _RealServed(
        SimpleNamespace(transcript=SimpleNamespace(path=path)),
        None,
        None,
        unused_runtime_hook,
        unused_runtime_hook,
    )


def _encode(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("pause", ["partial_json", "partial_utf8", "valid_without_lf"])
def test_transcript_reader_waits_for_record_publication(tmp_path: Path, pause: str) -> None:
    path = tmp_path / "transcript.jsonl"
    served = _reader(path)
    first = {"id": "first", "payload": {"text": "already published"}}
    second = {"id": "second", "payload": {"text": "\u20ac second"}}
    raw = _encode(second)
    cut = {
        "partial_json": 5,
        "partial_utf8": raw.index("\u20ac".encode("utf-8")) + 1,
        "valid_without_lf": len(raw),
    }[pause]
    with path.open("wb") as writer:
        writer.write(_encode(first) + b"\n" + raw[:cut])
        writer.flush()
        assert served.transcript_entries() == [first], "unfinished tail was published"
        # Same descriptor, exact suffix: after LF both baseline and repaired
        # readers must agree on the original records, not synthesized state.
        writer.write(raw[cut:] + b"\n")
        writer.flush()
        assert served.transcript_entries() == [first, second]


@pytest.mark.parametrize(
    "corrupt,error",
    [(b'{"broken":}\n', json.JSONDecodeError), (b'{"text":"\xff"}\n', UnicodeDecodeError)],
)
def test_transcript_reader_still_rejects_complete_corrupt_records(
    tmp_path: Path, corrupt: bytes, error: type[Exception]
) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(_encode({"id": "prior"}) + b"\n" + corrupt)
    with pytest.raises(error):
        _reader(path).transcript_entries()


def test_transcript_reader_keeps_crlf_and_blank_record_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    entry = {"id": "crlf"}
    path.write_bytes(b"\n\r\n \t\r\n" + _encode(entry) + b"\r\n\n")
    assert _reader(path).transcript_entries() == [entry]


def test_transcript_reader_does_not_split_literal_unicode_line_separator(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    entry = {"payload": {"text": "before\u2028after"}}
    raw = _encode(entry)
    assert b"\xe2\x80\xa8" in raw, "fixture escaped away the literal separator"
    path.write_bytes(raw + b"\n")
    assert _reader(path).transcript_entries() == [entry]


@pytest.mark.parametrize("exists", [False, True])
def test_transcript_reader_missing_or_empty_file(tmp_path: Path, exists: bool) -> None:
    path = tmp_path / "transcript.jsonl"
    if exists:
        path.write_bytes(b"")
    assert _reader(path).transcript_entries() == []

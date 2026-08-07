"""The JSON Lines on-disk contract.

`local_operator/jsonl.py` replaced the `jsonlines` dependency, and its module
docstring calls the exact on-disk encoding a contract that "must not change
without migrating existing state files". It then shipped with ZERO tests — and a
subsequent change to its newline handling went in unexercised. These are the
cases that make that contract checkable: the line terminator, non-ASCII, embedded
newlines, and every shape of malformed input the reader can meet.
"""

from __future__ import annotations

import json

import pytest

from local_operator.jsonl import InvalidLineError, read_jsonl, write_jsonl


def _records(path):
    """Drain read_jsonl, which is a lazy Iterator rather than a list.

    Laziness is the right shape for a state file that can be large, but it means
    a malformed line raises when it is REACHED, not when the file is opened —
    every error case below has to iterate to observe it.
    """
    return list(read_jsonl(path))


def test_writes_lf_terminators_on_every_platform(tmp_path) -> None:
    """The docstring promises `\\n`. Without `newline="\\n"` Python translates to
    `os.linesep`, so this file would carry CRLF on Windows and the format would
    be platform-dependent — exactly what owning the module was supposed to fix."""
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw == b'{"a": 1}\n{"b": 2}\n'


def test_non_ascii_is_written_as_utf8_not_escaped(tmp_path) -> None:
    """`ensure_ascii=False` is part of the pinned format: the file stays
    human-readable and byte-compact instead of exploding into \\uXXXX."""
    path = tmp_path / "b.jsonl"
    record = {"text": "héllo 世界 🎉"}
    write_jsonl(path, [record])
    raw = path.read_bytes()
    assert "héllo 世界 🎉".encode("utf-8") in raw
    assert b"\\u" not in raw
    assert _records(path) == [record]


def test_embedded_newlines_never_split_a_record(tmp_path) -> None:
    """A newline inside a string value is escaped by json.dumps, so one record
    stays one line. If it ever split, every later record would shift."""
    path = tmp_path / "c.jsonl"
    records = [{"text": "line one\nline two\r\nline three\rtail"}, {"n": 2}]
    write_jsonl(path, records)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert _records(path) == records


def test_round_trip_preserves_types_and_order(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    records = [
        {"i": 1, "f": 1.5, "b": True, "n": None, "l": [1, 2], "d": {"k": "v"}},
        {"i": 2},
        {},
    ]
    write_jsonl(path, records)
    assert _records(path) == records


def test_missing_trailing_newline_still_reads(tmp_path) -> None:
    """Hand-written and signal-truncated files lack the final terminator; losing
    the last record silently would be the worst possible failure here."""
    path = tmp_path / "e.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert _records(path) == [{"a": 1}, {"a": 2}]


def test_crlf_separated_file_reads(tmp_path) -> None:
    """`\\r` is JSON whitespace, so a CRLF file written by another tool (or by an
    older build of this module on Windows) must still parse."""
    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a": 1}\r\n{"a": 2}\r\n')
    assert _records(path) == [{"a": 1}, {"a": 2}]


def test_blank_lines_are_a_hard_error_not_a_skip(tmp_path) -> None:
    """Deliberate strictness, pinned because it is a decision rather than an
    oversight: write_jsonl never emits a blank line, so one means the file was
    truncated or hand-edited. Skipping it would turn a partial write into
    quietly missing history, which is worse than a loud failure the caller can
    log and recover from.
    """
    path = tmp_path / "g.jsonl"
    path.write_bytes(b'{"a": 1}\n\n{"a": 2}\n')
    with pytest.raises(InvalidLineError) as excinfo:
        _records(path)
    assert "2" in str(excinfo.value)


def test_malformed_line_names_its_line_number(tmp_path) -> None:
    """A corrupt state file must say WHERE, or the operator has to bisect it."""
    path = tmp_path / "h.jsonl"
    path.write_bytes(b'{"a": 1}\nnot json at all\n{"a": 3}\n')
    with pytest.raises(InvalidLineError) as excinfo:
        _records(path)
    assert "2" in str(excinfo.value)


def test_truncated_final_record_raises_rather_than_silently_dropping(tmp_path) -> None:
    """A half-written last line is corruption, not an empty file: reading it as
    "everything before it" without complaint would hide data loss."""
    path = tmp_path / "i.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2')
    with pytest.raises(InvalidLineError):
        _records(path)


def test_empty_and_absent_files(tmp_path) -> None:
    empty = tmp_path / "j.jsonl"
    empty.write_bytes(b"")
    assert _records(empty) == []
    write_jsonl(tmp_path / "k.jsonl", [])
    assert _records(tmp_path / "k.jsonl") == []


def test_separators_are_compact_and_stable(tmp_path) -> None:
    """The pinned separators are json.dumps defaults (', ' and ': '). Pinning
    them means a future default change cannot silently rewrite every state file."""
    path = tmp_path / "l.jsonl"
    write_jsonl(path, [{"a": 1, "b": 2}])
    assert path.read_text(encoding="utf-8").strip() == json.dumps({"a": 1, "b": 2})

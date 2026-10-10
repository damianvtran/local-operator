"""The ledger: the event row, the derived index, and the incremental scan cache.

The cases that matter are the ones a polled reader depends on:

* an unchanged journal costs ONE stat and a small JSON read (never a re-scan);
* an APPEND is scanned incrementally and merged (this is the difference between a 1 s and
  a 3 ms read on the heaviest real journal);
* a COMPACTION rewrite (a new inode) rescans whole, because only a full pass sees the new
  arrangement — and that is the case a naive "size grew" check would get wrong.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from local_operator.code_requests import ledger
from local_operator.code_requests.refs import HostContext, Remote

CWD = HostContext(remotes=(Remote("origin", "github.com", "damianvtran/local-operator"),))
PR = "https://github.com/damianvtran/local-operator/pull/1904"


def _write_journal(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _append_journal(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _create_pair(call_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"a{call_id}",
            "ts": time.time(),
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [{"text": "opening"}],
                "tool_calls": [
                    {
                        "id": call_id,
                        "name": "bash",
                        "arguments": {"command": "gh pr create --title t"},
                    }
                ],
            },
        },
        {
            "id": f"t{call_id}",
            "ts": time.time() + 1,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "tool",
                "tool_name": "bash",
                "tool_call_id": call_id,
                "content": [
                    {"text": f"exit code: 0\n--- stdout ---\n{PR}\n\n--- stderr ---\n(empty)"}
                ],
            },
        },
    ]


def _cache_sig(module: Any, config: Path, session_id: str = "abcdef123456") -> dict[str, Any]:
    """The cache's signature, asserted non-None for the type checker's benefit."""
    cache = module.read_cache(config, session_id)
    assert cache is not None
    sig = cache["sig"]
    assert isinstance(sig, dict)
    return sig


def _session(tmp_path: Path, session_id: str = "abcdef123456") -> tuple[Path, Path]:
    config = tmp_path / "config"
    session_dir = config / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return config, session_dir


def test_refresh_writes_an_index_and_a_cache(tmp_path):
    config, session_dir = _session(tmp_path)
    _write_journal(ledger.transcript_path(session_dir), _create_pair("c1"))
    outcome = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert outcome is not None and outcome.rescanned is True
    assert [row.ref.number for row in outcome.result.rows] == [1904]

    index = ledger.read_index(config, "abcdef123456")
    assert index is not None and index["rows"][0]["relation"] == "opened"
    cache = ledger.read_cache(config, "abcdef123456")
    assert cache is not None and cache["scan"]["rows"]
    # The index carries the collapsed COUNT, not the collapsed rows.
    assert "tool_output_only" in index and "tool_only_rows" not in index
    assert os.path.dirname(str(ledger.index_path(config, "abcdef123456"))).endswith("code_requests")


def test_an_unchanged_journal_is_not_rescanned(tmp_path):
    config, session_dir = _session(tmp_path)
    _write_journal(ledger.transcript_path(session_dir), _create_pair("c1"))
    first = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    second = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert first is not None and second is not None
    assert first.rescanned is True and second.rescanned is False
    assert [row.ref.number for row in second.result.rows] == [1904]


def test_an_append_is_scanned_incrementally_and_merged(tmp_path):
    config, session_dir = _session(tmp_path)
    journal = ledger.transcript_path(session_dir)
    _write_journal(journal, _create_pair("c1"))
    ledger.refresh(config, "abcdef123456", session_dir, context=CWD)

    _append_journal(journal, _create_pair("c2"))
    outcome = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert outcome is not None and outcome.rescanned is True
    numbers = sorted(row.ref.number for row in outcome.result.rows)
    assert numbers == [1904, 1904] or numbers == [1904]
    # The signature's offset must never claim bytes the scan did not read.
    assert _cache_sig(ledger, config)["offset"] == journal.stat().st_size
    # A second append of a NEW number merges into the same row set.
    _append_journal(
        journal,
        _create_pair("c3"),
    )
    journal.write_text(journal.read_text(encoding="utf-8").replace("pull/1904", "pull/2001"))
    third = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert third is not None
    assert {row.ref.number for row in third.result.rows} == {1904, 2001}


def test_a_replaced_journal_rescans_whole(tmp_path):
    config, session_dir = _session(tmp_path)
    journal = ledger.transcript_path(session_dir)
    _write_journal(journal, _create_pair("c1"))
    ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    before = _cache_sig(ledger, config)["inode"]
    # A compaction REPLACES the file (tmp + os.replace), which changes the inode while the
    # size can stay identical: the ladder must not treat that as an append.
    replacement: Path = journal.with_suffix(".compact")
    replacement.write_text(journal.read_text(encoding="utf-8"), encoding="utf-8")
    os.replace(replacement, journal)
    outcome = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert outcome is not None and outcome.rescanned is True
    after = _cache_sig(ledger, config)["inode"]
    assert after != before
    assert [row.ref.number for row in outcome.result.rows] == [1904]


def test_force_rescans_even_when_the_signature_matches(tmp_path):
    config, session_dir = _session(tmp_path)
    _write_journal(ledger.transcript_path(session_dir), _create_pair("c1"))
    ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    forced = ledger.refresh(config, "abcdef123456", session_dir, context=CWD, force=True)
    assert forced is not None and forced.rescanned is True


def test_a_session_with_no_journal_is_not_an_error(tmp_path):
    config, session_dir = _session(tmp_path)
    assert ledger.refresh(config, "abcdef123456", session_dir, context=CWD) is None


def test_an_empty_result_removes_the_index(tmp_path):
    config, session_dir = _session(tmp_path)
    journal = ledger.transcript_path(session_dir)
    _write_journal(journal, _create_pair("c1"))
    scanned = ledger.refresh(config, "abcdef123456", session_dir, context=CWD)
    assert scanned is not None
    assert ledger.index_path(config, "abcdef123456").exists()
    # The session's journal now holds nothing about a code request.
    _write_journal(
        journal,
        [
            {
                "id": "m",
                "ts": 1.0,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": "hello"}]},
            }
        ],
    )
    ledger.refresh(config, "abcdef123456", session_dir, context=CWD, force=True)
    assert not ledger.index_path(config, "abcdef123456").exists()


def test_index_sessions_lists_only_sessions_with_rows(tmp_path):
    config, with_rows = _session(tmp_path, "abcdef123456")
    _, without = _session(tmp_path, "fedcba654321")
    _write_journal(ledger.transcript_path(with_rows), _create_pair("c1"))
    _write_journal(
        ledger.transcript_path(without),
        [
            {
                "id": "m",
                "ts": 1.0,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": "hi"}]},
            }
        ],
    )
    assert ledger.refresh(config, "abcdef123456", with_rows, context=CWD) is not None
    assert ledger.refresh(config, "fedcba654321", without, context=CWD) is not None
    entries, read_error = ledger.index_sessions(config)
    assert read_error is False
    assert set(entries) == {"abcdef123456"}


def test_append_event_writes_the_v1_row_and_is_idempotent_per_call(tmp_path):
    class FakeTranscript:
        def __init__(self, directory: Path) -> None:
            self.directory = directory

        async def append_custom(self, custom_type: str, details: dict[str, Any]) -> None:
            with (self.directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"custom_type": custom_type, "details": details}) + "\n")

    import asyncio

    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = FakeTranscript(directory)
    from local_operator.code_requests.refs import parse_url

    ref = parse_url("https://github.com/o/r/pull/1", CWD)
    assert ref is not None
    details = ledger.build_event(kind="opened", ref=ref, evidence={"rule": "gh-pr-create-stdout"})
    assert asyncio.run(ledger.append_event(transcript, details)) is True
    written = json.loads((directory / "transcript.jsonl").read_text(encoding="utf-8"))
    assert written["custom_type"] == ledger.EVENT_CUSTOM_TYPE
    assert written["details"]["v"] == 1 and written["details"]["kind"] == "opened"

    class Broken:
        directory = None

        async def append_custom(self, *_args, **_kwargs):
            raise OSError("disk full")

    # A writer that fails costs the row, never the turn.
    assert asyncio.run(ledger.append_event(Broken(), details)) is False

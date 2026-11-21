"""A GROWN journal's cold-tail proof scans only the appended span.

WHAT THIS PINS. ``build_cold_tail_metadata`` proves facts the tail did not meet
by scanning every byte below the tail for their spellings. The cold metadata
record is keyed on the exact journal version, so an ACTIVE session misses it on
every open, and before this change every open re-scanned the whole journal.
These tests hold the three properties the incremental proof must keep:

* after growth, the scan reads O(appended bytes), not O(journal);
* a rewrite (new inode), a truncation below the boundary or a changed boundary
  row falls back to the full scan;
* the record filed is byte-equal to the one a cold full proof files.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session import replay_cache
from local_operator.session import transcript as transcript_module
from local_operator.session.attention import ATTENTION_CUSTOM_TYPE
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.transcript import Transcript

WANTED = (FRONTEND_CHECKPOINT_CUSTOM_TYPE, SESSION_SPEND_CUSTOM_TYPE)


async def _journal(directory: Path, turns: int) -> None:
    """Messages only: no checkpoint, spend or usage row anywhere, so the tail
    must prove all three absent below itself — the whole-journal scan."""
    transcript = Transcript(directory)
    for turn in range(turns):
        await transcript.append_message(
            Message(id=f"u{turn}", role="user", content=[TextContent(text="filler " * 40)])
        )
        await transcript.append_message(
            Message(id=f"a{turn}", role="assistant", content=[TextContent(text="ok " * 40)])
        )
    transcript.flush()


async def _grow(directory: Path, tag: str) -> int:
    """Append what a live turn writes: visible messages plus bookkeeping rows.
    Returns the bytes appended."""
    journal = directory / "transcript.jsonl"
    before = journal.stat().st_size
    transcript = Transcript(directory)
    await transcript.append_message(
        Message(id=f"gu{tag}", role="user", content=[TextContent(text=f"grow {tag}")])
    )
    await transcript.append_message(
        Message(id=f"ga{tag}", role="assistant", content=[TextContent(text=f"answer {tag}")])
    )
    await transcript.append_custom(ATTENTION_CUSTOM_TYPE, {"tag": tag})
    await transcript.append_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"checkpoint_id": tag})
    transcript.flush()
    return journal.stat().st_size - before


def _key(directory: Path) -> replay_cache.ReplayKey:
    key = replay_cache.replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    assert key is not None
    return key


@pytest.fixture
def scans(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every ``(start, end)`` range the absence scan was asked to read."""
    seen: list[tuple[int, int]] = []
    real = transcript_module._spellings_absent

    def counting(handle: Any, start: int, end: int, needles: Any, **kw: Any) -> bool:
        seen.append((start, end))
        return real(handle, start, end, needles, **kw)

    monkeypatch.setattr(transcript_module, "_spellings_absent", counting)
    return seen


def _build(directory: Path, record: Path, stats: dict[str, Any] | None = None) -> Any:
    result = replay_cache.build_cold_tail_metadata(_key(directory), record, stats)
    assert result is not None, "this journal is one the tail can serve"
    return result


def _cold_full_payload(directory: Path, scratch: Path) -> dict[str, Any]:
    """The record a FIRST-EVER proof of these bytes files (no sidecar exists)."""
    return _build(directory, scratch / "cold" / "record.json").to_payload()


@pytest.mark.asyncio
async def test_a_grown_journal_scans_only_the_appended_bytes(
    tmp_path: Path, scans: list[tuple[int, int]]
) -> None:
    directory = tmp_path / "session"
    await _journal(directory, turns=600)
    record = tmp_path / "cache" / "record.json"
    size = (directory / "transcript.jsonl").stat().st_size

    first: dict[str, Any] = {}
    _build(directory, record, first)
    assert first["proof_mode"] == "full"
    full_bytes = sum(end - start for start, end in scans)
    assert scans[0][0] == 0 and full_bytes > size // 2, "the first proof is O(journal)"
    assert replay_cache.absence_proof_path(record).exists()

    sidecar = replay_cache.absence_proof_path(record)
    for tag in ("one", "two", "three"):
        old_boundary = json.loads(sidecar.read_text())["boundary"]
        appended = await _grow(directory, tag)
        scans.clear()
        stats: dict[str, Any] = {}
        grown = _build(directory, record, stats)
        assert stats["proof_mode"] == "incremental"
        read = sum(end - start for start, end in scans)
        new_boundary = json.loads(sidecar.read_text())["boundary"]
        # The bounded tail window slides up by as many rows as were appended, so
        # the only newly uncovered span is the rows it slid past, plus one needle
        # overlap on each side of the old boundary.
        assert read <= (new_boundary - old_boundary) + 2 * 64, (read, appended)
        assert new_boundary - old_boundary < 4 * appended
        assert read < full_bytes // 20, "O(tail), not O(journal)"
        assert stats["absence_scan_bytes"] == read
        # Byte-equal to a cold full proof of the same bytes.
        assert grown.to_payload() == _cold_full_payload(directory, tmp_path / tag)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["rewrite", "truncate", "boundary_row"])
async def test_anything_but_growth_falls_back_to_the_full_proof(
    tmp_path: Path, scans: list[tuple[int, int]], change: str
) -> None:
    directory = tmp_path / "session"
    await _journal(directory, turns=600)
    journal = directory / "transcript.jsonl"
    record = tmp_path / "cache" / "record.json"
    _build(directory, record)
    proof = json.loads(replay_cache.absence_proof_path(record).read_text())
    boundary = proof["boundary"]

    if change == "rewrite":
        # What compaction does: the same bytes plus more, swapped in under a NEW
        # inode with ``os.replace``.
        staged = directory / "staged.jsonl"
        shutil.copyfile(journal, staged)
        os.replace(staged, journal)
        await _grow(directory, "after-rewrite")
    elif change == "truncate":
        # A cut below the boundary, at a row end so no torn row is left behind.
        cut = journal.read_bytes().rfind(b"\n", 0, boundary // 2) + 1
        with journal.open("r+b") as handle:
            handle.truncate(cut)
        await _grow(directory, "after-truncate")
    else:
        # A same-inode, same-length in-place change of the row ending at the
        # boundary: the digest refuses it.
        with journal.open("r+b") as handle:
            handle.seek(boundary - 15)
            original = handle.read(1)
            handle.seek(boundary - 15)
            handle.write(bytes([original[0] ^ 1]))
        await _grow(directory, "after-edit")

    scans.clear()
    stats: dict[str, Any] = {}
    record_now = replay_cache.build_cold_tail_metadata(_key(directory), record, stats)
    assert stats["proof_mode"] == "full", change
    assert scans and scans[0][0] == 0, "the scan restarts at byte 0"
    assert record_now is not None
    assert record_now.to_payload() == _cold_full_payload(directory, tmp_path / "cold")


@pytest.mark.asyncio
async def test_a_prior_proof_missing_a_needed_spelling_is_not_used(
    tmp_path: Path, scans: list[tuple[int, int]]
) -> None:
    """A proof only covers the spellings it scanned for. When a later read needs
    one it never proved absent, the whole range is scanned again."""
    directory = tmp_path / "session"
    await _journal(directory, turns=600)
    record = tmp_path / "cache" / "record.json"
    _build(directory, record)
    sidecar = replay_cache.absence_proof_path(record)
    proof = json.loads(sidecar.read_text())
    proof["needles"] = [n for n in proof["needles"] if "usage" not in n]
    sidecar.write_text(json.dumps(proof))
    await _grow(directory, "more")
    scans.clear()
    stats: dict[str, Any] = {}
    _build(directory, record, stats)
    assert stats["proof_mode"] == "full"
    assert scans[0][0] == 0

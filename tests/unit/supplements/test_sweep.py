"""The supplement sweep: what it reclaims, and every guard that stops it.

The safety rules are pinned BY NAME (the digest is in ``kept``, the pass is
``skipped``) rather than by "nothing broke", because the failure classes here
are about what was NOT deleted: an in-flight job's blob, a referenced blob, a
user artifact, or anything a pass could not verify.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import TranscriptEntry
from local_operator.supplements import sweep

from .conftest import load

pytestmark = pytest.mark.asyncio

#: Backdate used for "clearly old": far past any grace window this module ships.
OLD_AGE_S = 10 * sweep.SWEEP_GRACE_S


def _blob(root: Path, raw: bytes, mime: str = sweep.COMPONENT_MIME, *, age_s: float = 0.0) -> str:
    """Write a real store blob through the store's own writer; return its digest."""
    store = AttachmentStore(root=root / "attachments")
    ref = store.put_bytes(raw, mime)
    assert ref is not None
    if age_s:
        stale = os.stat(store.root / f"{ref.digest}.bin").st_mtime - age_s
        os.utime(store.root / f"{ref.digest}.bin", (stale, stale))
        os.utime(store.root / f"{ref.digest}.json", (stale, stale))
    return ref.digest


def _journal(root: Path, session_id: str, lines: list[dict[str, Any]]) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    return directory


def _row(anchor: str, digest: str, *, journal_id: str = "0" * 32) -> dict[str, Any]:
    """One journal line referencing ``digest`` (the sync-regex-visible spelling)."""
    return {
        "id": journal_id,
        "ts": 1790000000.0,
        "type": "custom",
        "payload": {
            "custom_type": "supplement_v1",
            "details": {
                "anchor": anchor,
                "job": "3f9c1a7e5b20",
                "version": 1,
                "state": "done",
                "files": [],
                "components": [
                    {
                        "attachment": digest,
                        "title": "t",
                        "source": "s",
                        "mime": "text/html",
                        "height_hint": 320,
                    }
                ],
                "at": 1790000000.0,
            },
        },
    }


async def test_a_blob_a_live_job_just_stored_survives(tmp_path: Path) -> None:
    """THE in-flight guard: a blob inside the grace window is not even a candidate.

    This is the exact window between ``_store`` (the blob lands) and the terminal
    row append: no journal row references the digest yet, so only the age gate
    protects it. A job's whole lifetime is bounded by ``timeoutS`` (90 s default)
    plus its append; the window is 1 hour, 40x that.
    """
    digest = _blob(tmp_path, b"<p>fresh</p>", age_s=0.0)
    report = sweep.store_sweep(tmp_path)
    assert report.candidates == 0 and report.deleted == 0
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_an_orphaned_component_blob_is_reclaimed_once_it_is_old(tmp_path: Path) -> None:
    digest = _blob(tmp_path, b"<p>orphan</p>", age_s=OLD_AGE_S)
    report = sweep.store_sweep(tmp_path)
    assert report.deleted == 1
    assert report.bytes_freed == len(b"<p>orphan</p>")
    assert not (tmp_path / "attachments" / f"{digest}.bin").exists()
    assert not (tmp_path / "attachments" / f"{digest}.json").exists()


async def test_a_referenced_component_blob_survives_however_old(tmp_path: Path) -> None:
    digest = _blob(tmp_path, b"<p>kept</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "s1", [_row("a" * 32, digest)])
    report = sweep.store_sweep(tmp_path)
    assert report.deleted == 0
    assert digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_media_blobs_and_sidecarless_content_are_never_candidates(tmp_path: Path) -> None:
    """The store's never-delete contract for everything that is not a component."""
    image = _blob(tmp_path, b"\x89PNG-pixels", mime="image/png", age_s=OLD_AGE_S)
    # Content without a sidecar: the store's documented "merely unused disk" state;
    # the sweep must not guess a mime for it.
    lonely = tmp_path / "attachments" / ("f" * 32 + ".bin")
    lonely.write_bytes(b"<p>no sidecar</p>")
    stale = os.stat(lonely).st_mtime - OLD_AGE_S
    os.utime(lonely, (stale, stale))
    report = sweep.store_sweep(tmp_path)
    assert report.candidates == 0 and report.deleted == 0
    assert (tmp_path / "attachments" / f"{image}.bin").exists()
    assert lonely.exists()


async def test_a_transcript_with_a_pinned_mtime_still_protects_its_blob(tmp_path: Path) -> None:
    """R1 regression: the order production actually writes.

    ``persistence.append_row`` appends with ``preserve_mtime=True``, and the restore
    pins the transcript's mtime back to before the append -- which necessarily
    precedes the blob the row names, because the runner stores the blob first
    (``runtime/supplements._store``) and commits the row after. So no mtime
    relationship in this file proves it holds no reference: a scan that narrows by
    transcript mtime skips this transcript and deletes a REFERENCED blob (silent
    data loss -- the row's digest then dangles and the frame can never render).
    """
    digest = _blob(tmp_path, b"<p>pinned</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "s1", [_row("a" * 32, digest)])
    transcript = tmp_path / "sessions" / "s1" / "transcript.jsonl"
    blob_mtime = os.stat(tmp_path / "attachments" / f"{digest}.bin").st_mtime
    pinned = blob_mtime - 120.0
    os.utime(transcript, (pinned, pinned))
    report = sweep.store_sweep(tmp_path)
    assert report.scanned_sessions == 1, "the referencing transcript must be scanned"
    assert report.deleted == 0 and digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_the_reference_scan_is_never_narrowed_by_transcript_mtime(tmp_path: Path) -> None:
    """EVERY transcript is scanned: a file's age proves nothing about its references.

    The "old" transcript here names no candidate, and it is still scanned -- the
    mtime is pinned back by every ``preserve_mtime`` append, so an arbitrarily old
    file may name an arbitrarily new blob (R1). The pass stays bounded by the scan
    budget instead, which fails closed (the budget test below).
    """
    digest = _blob(tmp_path, b"<p>narrow</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "old", [_row("b" * 32, "1" * 32)])
    old_transcript = tmp_path / "sessions" / "old" / "transcript.jsonl"
    long_ago = os.stat(old_transcript).st_mtime - 100 * sweep.SWEEP_GRACE_S
    os.utime(old_transcript, (long_ago, long_ago))
    _journal(tmp_path, "new", [_row("c" * 32, digest)])
    report = sweep.store_sweep(tmp_path)
    assert report.scanned_sessions == 2  # both transcripts; nothing is skipped by age
    assert report.deleted == 0 and digest in report.kept


async def test_an_unreadable_transcript_blocks_deletion(tmp_path: Path) -> None:
    """Fail-closed: an unseen reference must not become a deletion."""
    if os.geteuid() == 0:  # root reads mode-000 files; the hazard the test needs cannot exist
        pytest.skip("running as root: cannot make a file unreadable via mode bits")
    digest = _blob(tmp_path, b"<p>blk</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "s1", [])
    os.chmod(tmp_path / "sessions" / "s1" / "transcript.jsonl", 0o000)
    report = sweep.store_sweep(tmp_path)
    assert report.skipped.startswith("scan:")
    assert report.deleted == 0 and digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_a_session_dir_without_traverse_permission_blocks_deletion(tmp_path: Path) -> None:
    """R2-1 regression: a session dir without ``+x`` must fail strictly closed.

    The directory cannot be traversed, so the transcript inside it is invisible
    to discovery (``transcript.is_file()`` raises EACCES) and the scan reads as
    "no references"; a pass that skips it silently then reclaims the blob the
    transcript still names -- the reviewer's reproduction: mode 0600,
    transcript intact, ``skipped=''``, blob deleted and unrecoverable. The
    discovery failure must be counted and abort the deletion phase instead.
    """
    if os.geteuid() == 0:  # root traverses any mode; the hazard the test needs cannot exist
        pytest.skip("running as root: cannot make a directory untraversable via mode bits")
    digest = _blob(tmp_path, b"<p>invisible</p>", age_s=OLD_AGE_S)
    session_dir = _journal(tmp_path, "s1", [_row("a" * 32, digest)])
    os.chmod(session_dir, 0o600)
    try:
        report = sweep.store_sweep(tmp_path)
    finally:
        os.chmod(session_dir, 0o700)  # let pytest's own cleanup descend again
    assert report.skipped.startswith("scan:")
    assert report.deleted == 0 and digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_an_unlistable_sessions_dir_blocks_deletion(tmp_path: Path) -> None:
    """The other half of R2-1: a discovery source that cannot be listed at all.

    With ``sessions/`` unlistable no reference is readable; before the fix this
    returned ``[]`` with no problem recorded, so the delete phase ran on the
    empty set and reclaimed referenced blobs silently.
    """
    if os.geteuid() == 0:  # root lists any mode; the hazard the test needs cannot exist
        pytest.skip("running as root: cannot make a directory unlistable via mode bits")
    digest = _blob(tmp_path, b"<p>unlistable</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "s1", [_row("a" * 32, digest)])
    sessions = tmp_path / "sessions"
    os.chmod(sessions, 0o000)
    try:
        report = sweep.store_sweep(tmp_path)
    finally:
        os.chmod(sessions, 0o700)  # let pytest's own cleanup descend again
    assert report.skipped.startswith("scan:")
    assert report.deleted == 0 and digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_a_sessions_dir_without_traverse_permission_blocks_deletion(tmp_path: Path) -> None:
    """A sessions dir that lists but cannot be traversed is still a discovery fault.

    Mode 0400 lets ``iterdir()`` read the names but ``child.is_dir()`` cannot
    stat through to any entry (EACCES): the per-session scan cannot be trusted,
    so the pass must delete nothing. The failure rides the same problems
    channel as the two tests above.
    """
    if os.geteuid() == 0:  # root traverses any mode; the hazard the test needs cannot exist
        pytest.skip("running as root: cannot make a directory untraversable via mode bits")
    digest = _blob(tmp_path, b"<p>blind</p>", age_s=OLD_AGE_S)
    _journal(tmp_path, "s1", [_row("a" * 32, digest)])
    sessions = tmp_path / "sessions"
    os.chmod(sessions, 0o400)
    try:
        report = sweep.store_sweep(tmp_path)
    finally:
        os.chmod(sessions, 0o700)  # let pytest's own cleanup descend again
    assert report.skipped.startswith("scan:")
    assert report.deleted == 0 and digest in report.kept
    assert (tmp_path / "attachments" / f"{digest}.bin").exists()


async def test_deletion_is_bounded_per_pass_and_converges(tmp_path: Path) -> None:
    digests = [_blob(tmp_path, f"<p>{i}</p>".encode(), age_s=OLD_AGE_S) for i in range(3)]
    first = sweep.store_sweep(tmp_path, max_deletes=2, max_candidates=2)
    assert first.deleted == 2 and first.candidates == 2
    second = sweep.store_sweep(tmp_path, max_deletes=2, max_candidates=2)
    assert second.deleted == 1
    assert not any((tmp_path / "attachments" / f"{d}.bin").exists() for d in digests)


async def test_a_scan_budget_overrun_skips_the_pass(tmp_path: Path) -> None:
    digest = _blob(tmp_path, b"<p>budget</p>", age_s=OLD_AGE_S)
    # A non-empty transcript: the budget check fires before its bytes are read.
    _journal(tmp_path, "s1", [{"id": "u1", "ts": 1.0, "type": "message", "payload": {}}])
    report = sweep.store_sweep(tmp_path, max_scan_bytes=1)
    assert report.skipped == "scan:budget"
    assert report.deleted == 0 and digest in report.kept


async def test_the_interval_gate_claims_the_interval(tmp_path: Path) -> None:
    moment = 1_800_000_000.0
    assert sweep.sweep_due(tmp_path, now=moment) is True
    assert sweep.sweep_due(tmp_path, now=moment + 60) is False
    assert sweep.sweep_due(tmp_path, now=moment + sweep.SWEEP_INTERVAL_S + 1) is True


# ---------------------------------------------------------------------------
# Stale rows
# ---------------------------------------------------------------------------


def _entry(details: dict[str, Any], *, entry_id: str = "e") -> TranscriptEntry:
    return TranscriptEntry(
        id=entry_id,
        ts=1.0,
        type="custom",
        payload={"custom_type": "supplement_v1", "details": details},
    )


def _details(anchor: str, state: str, version: int = 1, job: str = "j1") -> dict[str, Any]:
    return {
        "anchor": anchor,
        "job": job,
        "version": version,
        "state": state,
        "files": [],
        "components": [],
        "at": 1.0,
    }


async def test_stale_rows_finds_newest_non_terminal_rows_only() -> None:
    entries = [
        _entry(_details("a" * 32, "decided")),
        _entry(_details("b" * 32, "done")),
        _entry(_details("c" * 32, "queued", version=1)),
        _entry(_details("c" * 32, "cancelled", version=2)),
        _entry(_details("d" * 32, "skipped")),
        TranscriptEntry(id="m", ts=1.0, type="message", payload={}),
        TranscriptEntry(
            id="x", ts=1.0, type="custom", payload={"custom_type": "other_v1", "details": {}}
        ),
    ]
    rows = sweep.stale_rows(entries)
    assert [row["anchor"] for row in rows] == ["a" * 32]  # b/d terminal; c newest is terminal


async def test_the_runner_settles_a_stale_row_end_to_end(tmp_path: Path) -> None:
    """A journal carrying the C0 stale fixture settles to a terminal row, once."""
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        seed_transcript,
        text_turn,
    )

    directory = tmp_path / "sessions" / "stale-e2e"
    await seed_transcript(directory, [Message.user("hi")])
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(load("rows/queued_stale.json")) + "\n")
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(
        session,
        asyncio.get_running_loop(),
        install_gates=False,
        cwd=str(tmp_path),
        config_dir=tmp_path,
    )
    runner = handle._supplements
    try:
        # The handle's constructor scheduled its own pass; cancel it first so this
        # test owns the interval stamp and the ordering is deterministic.
        scheduled = runner._sweep_task
        if scheduled is not None:
            scheduled.cancel()
            await asyncio.gather(scheduled, return_exceptions=True)
            runner._sweep_task = None
        await runner._sweep()
        rows = [
            entry.payload["details"]
            for entry in session.transcript.entries()
            if entry.payload.get("custom_type") == "supplement_v1"
        ]
        assert [row["state"] for row in rows] == ["queued", "cancelled"]
        assert rows[-1]["version"] == 2 and rows[-1]["error"] == sweep.DISPOSED_ERROR
        # Files still shown by the reader are restated on the settled row (next_version).
        assert [f["path"] for f in rows[-1]["files"]] == ["reports/latency.md", "data/bench.csv"]
        # A second pass inside the interval is gated out; even past it, terminal rows
        # are invisible to stale_rows, so nothing more is written.
        assert sweep.sweep_due(tmp_path, now=time.time() + 1) is False
        await runner._settle_stale_rows()
        assert (
            len(
                [
                    e
                    for e in session.transcript.entries()
                    if e.payload.get("custom_type") == "supplement_v1"
                ]
            )
            == 2
        )
    finally:
        await handle.dispose()
        await session.dispose()


async def test_scheduling_requires_an_explicit_root_and_reaps(tmp_path: Path) -> None:
    """A handle with no declared root never schedules (it would hit the real store)."""
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        seed_transcript,
        text_turn,
    )

    directory = tmp_path / "sessions" / "sched"
    await seed_transcript(directory, [])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    loop = asyncio.get_running_loop()
    bare = ServingSessionHandle(session, loop, install_gates=False, cwd=str(tmp_path))
    try:
        assert bare._supplements._sweep_task is None
        bare._supplements.schedule_sweep(loop)
        assert bare._supplements._sweep_task is None
    finally:
        await bare.dispose()
        await session.dispose()

    session2 = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    rooted = ServingSessionHandle(
        session2, loop, install_gates=False, cwd=str(tmp_path), config_dir=tmp_path
    )
    # Read the slot BEFORE the try so the finally cannot see an unbound name
    # (constructing the handle is what schedules the pass).
    awaited = rooted._supplements._sweep_task
    try:
        assert awaited is not None and not awaited.done()
    finally:
        # Disposal reaps the sweep task (cancelled if still pending, joined either
        # way) and clears the slot -- the await must not hang or leak.
        await rooted.dispose()
        assert awaited is not None and awaited.done()
        assert awaited.cancelled() or awaited.exception() is None
        assert rooted._supplements._sweep_task is None
        await session2.dispose()

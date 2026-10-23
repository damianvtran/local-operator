"""The stage-C consumer matrix: every surface that asks about a transcript.

A compressed session is not an empty one, and this file is the list of places
that used to think otherwise. Each cell builds a REAL compressed session through
``compression.compress_session`` (not a hand-written archive) and drives the
production entry point that used to stat the plain name:

* the two failed-create removals (aida, librarian) — never remove an archived one;
* the two wake-index reapers — never drop an archived session's cadence;
* the supervisor's ghost guard and the multiplexer's resumability probe;
* the trigger reconciler's ghost drop;
* project staleness — the age must not move;
* the supplements sweep — a compressed transcript is still scanned for refs;
* the session tools (``_peek_*``) and the marker extras;
* the TUI's resume hint and the cleanup primitive's identity proof;
* sync: a compressed OWNER serves a plan and a compressed REPLICA accepts an append;
* the picker preview and the stored model selection.

A cell that only asserted "no exception" would not have caught the original
defect, so every one asserts the OBSERVABLE answer a user or a caller gets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import compression, cleanup


def _journal(rows: int = 6) -> bytes:
    out = []
    for index in range(rows):
        out.append(
            json.dumps(
                {
                    "id": f"row{index:03d}",
                    "ts": 1000.0 + index,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user" if index % 2 == 0 else "assistant",
                        # Block form, as every real journal writes it: the
                        # preview's own reader folds these blocks, so a bare
                        # string here would test a shape no writer produces.
                        "content": [{"text": f"turn {index} about the harbour"}],
                    },
                }
            )
        )
    return b"".join(row.encode() + b"\n" for row in out)


def _store(tmp_path: Path, name: str = "config") -> Path:
    config = tmp_path / name
    sessions = config / "sessions"
    sessions.mkdir(parents=True)
    (sessions / cleanup.STORE_MARKER_NAME).write_text("{}", encoding="utf-8")
    return config


def _session(config: Path, session_id: str, *, body: bytes | None = None) -> Path:
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / compression.PLAIN_FILENAME).write_bytes(_journal() if body is None else body)
    return directory


def _compress(directory: Path) -> None:
    outcome = compression.compress_session(directory)
    assert outcome.status == "compressed", outcome
    assert compression.archive_path(directory) is not None
    assert not (directory / compression.PLAIN_FILENAME).exists()


@pytest.fixture()
def archived(tmp_path: Path) -> tuple[Path, Path]:
    """``(config, session directory)`` for one session already compressed."""
    config = _store(tmp_path)
    directory = _session(config, "archived01")
    _compress(directory)
    return config, directory


# ---------------------------------------------------------------------------
# The two failed-create removals
# ---------------------------------------------------------------------------


def test_aida_never_removes_a_failed_create_that_holds_an_archive(
    archived: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.aida import bootstrap

    config, directory = archived
    calls: list[Path] = []
    monkeypatch.setattr(
        "local_operator.session.cleanup.remove_fresh_session_dir",
        lambda target, **_kwargs: calls.append(target) or True,
    )
    bootstrap._discard_failed_create(config, directory.name)
    assert calls == []
    assert directory.is_dir()


def test_librarian_never_removes_a_failed_create_that_holds_an_archive(
    archived: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.librarian import bootstrap

    config, directory = archived
    calls: list[Path] = []
    monkeypatch.setattr(
        "local_operator.session.cleanup.remove_fresh_session_dir",
        lambda target, **_kwargs: calls.append(target) or True,
    )
    bootstrap._discard_failed_create(config, directory.name)
    assert calls == []
    assert directory.is_dir()


# ---------------------------------------------------------------------------
# The wake-index reapers
# ---------------------------------------------------------------------------


def test_aida_keeps_its_rows_for_a_session_whose_transcript_is_an_archive(
    archived: tuple[Path, Path],
) -> None:
    from local_operator.aida import proactive
    from local_operator.wakes import store as wake_store

    config, directory = archived
    wake_store.write_entry(
        config,
        directory.name,
        cwd=str(config),
        schedules=[{"id": "aida-cadence", "every_s": 3600, "next_at": 0}],
    )
    assert proactive.reap_orphan_rows(config, keep="someone-else") == ()
    entry = wake_store.read_index(config)[directory.name]
    assert [row["id"] for row in entry["schedules"]] == ["aida-cadence"]


def test_librarian_keeps_its_rows_for_a_session_whose_transcript_is_an_archive(
    archived: tuple[Path, Path],
) -> None:
    from local_operator.librarian import proactive
    from local_operator.wakes import store as wake_store

    config, directory = archived
    wake_store.write_entry(
        config,
        directory.name,
        cwd=str(config),
        schedules=[{"id": "librarian-pass", "every_s": 3600, "next_at": 0}],
    )
    assert proactive.reap_orphan_rows(config, keep="someone-else") == ()
    entry = wake_store.read_index(config)[directory.name]
    assert [row["id"] for row in entry["schedules"]] == ["librarian-pass"]


# ---------------------------------------------------------------------------
# Resumability: the supervisor's ghost guard, the multiplexer's probe, the TUI hint
# ---------------------------------------------------------------------------


def test_the_supervisor_finds_a_compressed_session_resumable(
    archived: tuple[Path, Path],
) -> None:
    from local_operator.wakes.supervisor import _session_exists

    config, directory = archived
    assert _session_exists(config, directory.name) is True
    assert _session_exists(config, "does-not-exist") is False


def test_the_multiplexer_binds_a_compressed_session(
    archived: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.multiplexer import broadcast

    config, directory = archived
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    assert broadcast.is_resumable_session(directory.name) is True
    assert broadcast.is_resumable_session("does-not-exist") is False


def test_the_tui_resume_hint_survives_compression(
    archived: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.tui.app import OperatorApp

    config, directory = archived
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    class _Stub:
        _session = type("S", (), {"session_id": directory.name})()
        _stopped_session_id = ""

    assert OperatorApp._resumable_session_id(_Stub()) == directory.name


def test_the_cleanup_primitive_refuses_an_archived_session(
    archived: tuple[Path, Path],
) -> None:
    config, directory = archived
    assert (
        cleanup.remove_fresh_session_dir(
            directory,
            config_dir=config,
            policy="failed-create",
            reason="test: an archived session is never a failed create",
            actor="test",
        )
        is False
    )
    assert directory.is_dir()
    assert compression.has_transcript(directory)


def test_an_unreadable_session_is_not_an_absent_one(tmp_path: Path) -> None:
    """The tri-state, which every remover's permission rests on.

    ``has_transcript`` answers ``False`` for a directory this process cannot
    read — fine for a listing row, fatal if a removal treats it as "nothing
    durable here". ``transcript_state`` keeps the two apart, and the audited
    removal primitive refuses on the third answer.
    """
    import os

    config = _store(tmp_path, "unreadable")
    absent = config / "sessions" / "absent01"
    absent.mkdir()
    assert compression.transcript_state(absent) == compression.TRANSCRIPT_ABSENT

    directory = _session(config, "locked01")
    assert compression.transcript_state(directory) == compression.TRANSCRIPT_PRESENT

    os.chmod(directory, 0o000)
    try:
        assert compression.transcript_state(directory) == compression.TRANSCRIPT_UNREADABLE
        assert compression.has_transcript(directory) is False
        assert (
            cleanup.remove_fresh_session_dir(
                directory,
                config_dir=config,
                policy="failed-create",
                reason="test: an unreadable directory is never a fresh one",
                actor="test",
            )
            is False
        )
    finally:
        os.chmod(directory, 0o700)
    assert directory.is_dir()


# ---------------------------------------------------------------------------
# Staleness + the trigger reconciler
# ---------------------------------------------------------------------------


def test_staleness_does_not_move_when_a_session_is_compressed(tmp_path: Path) -> None:
    import os
    import time

    from local_operator.wakes.triggers.sources import project_staleness

    config = _store(tmp_path, "stale")
    directory = _session(config, "stale01")
    stamp = time.time() - 10 * 86400
    os.utime(directory / compression.PLAIN_FILENAME, (stamp, stamp))
    before = project_staleness._last_activity_age_s(directory)
    _compress(directory)
    after = project_staleness._last_activity_age_s(directory)
    assert before is not None and after is not None
    assert abs(after - before) <= 2


def test_the_trigger_reconciler_keeps_a_compressed_target_and_drops_a_ghost(
    archived: tuple[Path, Path],
) -> None:
    from local_operator.wakes import triggers

    config, directory = archived
    pending: dict[str, dict[str, Any]] = {
        directory.name: {"updated_at_ms": triggers._now_ms(), "text": "check in"},
        "deadbeef": {"updated_at_ms": triggers._now_ms(), "text": "ghost"},
    }
    triggers.reconcile(config, pending)
    assert directory.name in pending
    assert "deadbeef" not in pending


# ---------------------------------------------------------------------------
# The indexed-read path: an archive is UNAVAILABLE, and the page reader degrades
# ---------------------------------------------------------------------------


def test_an_indexed_raw_read_answers_unavailable_and_the_page_keeps_its_rows(
    tmp_path: Path,
) -> None:
    """The stage-B note, verified where a user can meet it.

    ``read_indexed_raw_rows`` cannot witness an archive (its spans are byte
    offsets into the plain journal, and its proof is that file's inode), so it
    answers ``None`` — "source unavailable", never an invented absence. The
    caller that reaches a user is the page builder, whose contract is that an
    unavailable proof leaves the ORIGINAL ROWS VISIBLE. What this cell pins is
    that the conversation does not turn into "no record": the rows survive and no
    exception escapes.
    """
    from types import SimpleNamespace

    from local_operator.harness import peer_cycle_display
    from local_operator.session import transcript, transcript_index

    config = _store(tmp_path, "indexed")
    directory = _session(config, "indexed01")
    index = transcript_index.refresh_index(config, directory.name)
    assert index is not None and index.journal_rows
    ids = ["row000", "row001"]
    assert transcript.read_indexed_raw_rows(directory, index, ids) is not None

    _compress(directory)

    assert transcript.read_indexed_raw_rows(directory, index, ids) is None

    messages = [
        SimpleNamespace(id="row000", custom_type=None, payload={}),
        SimpleNamespace(id="row001", custom_type=None, payload={}),
    ]

    class _Window:
        status = "ok"
        history_generation = 0
        conversation_id = directory.name

        def __init__(self) -> None:
            self.messages = messages

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"messages": [{"id": message.id} for message in messages]}

    window = _Window()
    # A Transcript-shaped reader object: the page builder asks it for its
    # generation and its directory, and constructing a REAL Transcript here would
    # restore the archive and destroy the case under test.
    reader = SimpleNamespace(directory=directory, _history_generation=0)
    returned = peer_cycle_display.attach_page_records(window, reader, max_wire_bytes=1 << 20)
    assert returned is window
    assert returned.messages == messages
    # No fabricated "missing" record either: either no envelope at all, or an
    # envelope naming the proof unavailable — never a record claiming absence.
    records = getattr(returned, "peer_cycle_display_records", None)
    if records is not None:
        assert records["records"] == []


# ---------------------------------------------------------------------------
# The supplements sweep: an archive is still scanned for attachment references
# ---------------------------------------------------------------------------


def test_the_sweep_scans_a_compressed_transcript(tmp_path: Path) -> None:
    from local_operator.supplements import sweep

    config = _store(tmp_path, "sweep")
    digest = "a" * 32
    body = _journal() + (
        json.dumps(
            {"id": "ref1", "ts": 1.0, "type": "custom", "payload": {"attachment": digest}}
        )
        + "\n"
    ).encode()
    directory = _session(config, "sweep01", body=body)
    _compress(directory)

    digests, scanned, _bytes, problem = sweep.referenced_digests(config / "sessions")
    assert problem == ""
    assert scanned == 1
    assert digest in digests


# ---------------------------------------------------------------------------
# The session tools
# ---------------------------------------------------------------------------


def test_peek_reads_an_archived_transcript(archived: tuple[Path, Path]) -> None:
    from local_operator.tools import builtin

    _config, directory = archived
    head = builtin._peek_head(directory, need=2)
    assert isinstance(head, builtin._PeekWindow)
    assert len(head.steps) == 2
    assert any("harbour" in step.body for step in head.steps)
    assert head.has_newer is True

    found = builtin._peek_search(directory, needle="harbour", pattern=None, need=1)
    assert isinstance(found, builtin._PeekWindow)
    assert found.steps


def test_the_marker_extras_report_the_archives_path(
    archived: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.tools import builtin

    _config, directory = archived
    monkeypatch.setattr(builtin, "config_dir", lambda: directory.parent.parent)
    extras = builtin._sessions_marker_extras(directory.name)
    assert extras["transcript_path"] == str(compression.archive_path(directory))


# ---------------------------------------------------------------------------
# The picker preview and the stored model selection
# ---------------------------------------------------------------------------


def test_the_picker_preview_reads_a_compressed_session(archived: tuple[Path, Path]) -> None:
    from local_operator.session.preview import SessionPreviews

    _config, directory = archived
    previews = SessionPreviews(directory.parent)
    turns = previews.condensed(directory.name)
    assert turns, "a compressed session's preview must not read as empty"
    assert any("harbour" in str(turn.text) for turn in turns)
    assert previews.unreadable(directory.name) is False


def test_the_stored_model_selection_is_read_from_the_archive(tmp_path: Path) -> None:
    from local_operator.session import model_selection

    config = _store(tmp_path, "model")
    row = {
        "id": "sel1",
        "ts": 2000.0,
        "type": "custom",
        "payload": {
            "custom_type": model_selection.SELECTED_MODEL_CUSTOM_TYPE,
            "details": {
                "version": model_selection.SELECTION_VERSION,
                "selector": "anthropic/claude-x",
            },
        },
    }
    directory = _session(config, "model01", body=_journal() + (json.dumps(row) + "\n").encode())
    _compress(directory)

    selection = model_selection.read_model_selection(directory)
    assert selection is not None
    assert selection.provider == "anthropic"


# ---------------------------------------------------------------------------
# Sync: the owner restores before serving, the destination before appending
# ---------------------------------------------------------------------------


def test_sync_treats_a_compressed_replica_as_held(archived: tuple[Path, Path]) -> None:
    from local_operator.network import sync

    _config, directory = archived
    assert sync.transcript_exists(directory) is True


def test_the_owner_serves_a_plan_for_a_compressed_session(
    archived: tuple[Path, Path],
) -> None:
    from local_operator.network import sync

    config, directory = archived
    plan = sync.build_manifest(config, directory.name)
    assert plan["transcript"]["total_bytes"] == len(_journal())
    # The owner restored in place before serving: the wire still speaks plain bytes.
    assert (directory / compression.PLAIN_FILENAME).exists()


def test_a_compressed_replica_is_restored_before_an_append(
    archived: tuple[Path, Path], tmp_path: Path
) -> None:
    from local_operator.network import sync

    config, directory = archived
    dest = _session(tmp_path, "replica01", body=_journal(rows=3))
    _compress(dest)

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame["session_id"])
        if phase == "plan":
            return sync.build_manifest(
                config, session_id, have=frame.get("have") if isinstance(frame, dict) else {}
            )
        if phase == "fetch":
            return sync.serve_fetch(
                config,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                offset=int(frame.get("offset") or 0),
                limit=int(frame.get("limit") or sync.SYNC_CHUNK_BYTES),
            )
        if phase == "verify":
            return sync.serve_verify(
                config,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                prefix_bytes=int(frame.get("prefix_bytes") or 0),
                prefix_digest=str(frame.get("prefix_digest") or ""),
            )
        raise AssertionError(phase)

    result = sync.pull(
        tmp_path,
        directory.name,
        ask=ask,
        dest_dir=dest,
        attachments_root=destination_attachments(dest),
        purpose="replica",
    )
    assert result.transcript_mode == "replace", result
    assert compression.has_transcript(dest)
    assert (dest / compression.PLAIN_FILENAME).read_bytes() == _journal()


def destination_attachments(dest: Path) -> Path:
    """The replica's own store root: a replica keeps its blobs inside itself."""
    return dest

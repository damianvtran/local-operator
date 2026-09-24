"""The sync primitive: the copy set, the incremental cursor, and the replica.

NO RELAYS HERE, on purpose. ``network/sync.py`` takes its transport as a callable
(``ask``), so the whole copy path — plan, chunked fetch, verification, resumption,
the cursor — is exercised against a second config root directly. What that buys is
that these are UNIT tests of the protocol's byte-level promises: an append that is
only the append region, a compaction that forces a full replace, a torn chunk that
is refused, a resume that is verified rather than hopeful. The relay's own use of
this code is tested in ``test_mobility.py``, which is where the link is real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import sync


def _ask(root: Path) -> Any:
    """A transport that serves ``root`` in-process — the owner's half, no link."""

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if phase == "plan":
            return sync.build_manifest(
                root, session_id, have=frame.get("have") if isinstance(frame, dict) else {}
            )
        if phase == "fetch":
            return sync.serve_fetch(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                offset=int(frame.get("offset") or 0),
                limit=int(frame.get("limit") or sync.SYNC_CHUNK_BYTES),
            )
        if phase == "verify":
            return sync.serve_verify(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                prefix_bytes=int(frame.get("prefix_bytes") or 0),
                prefix_digest=str(frame.get("prefix_digest") or ""),
            )
        raise AssertionError(f"unexpected phase {phase!r}")

    return ask


def _row(index: int, text: str) -> str:
    return json.dumps({"id": f"e{index}", "type": "user", "content": text}) + "\n"


def seed(root: Path, session_id: str, *, rows: int = 3) -> Path:
    """One session with the WHOLE copy set, so a test can assert it all travels."""
    from local_operator.fork import FORK_BOUNDARY_NAME, FORK_BOUNDARY_VERSION
    from local_operator.resume import (
        ATTACHMENT_SIDECAR_NAME,
        ORIGIN_NAME,
        TITLE_SIDECAR_NAME,
    )
    from local_operator.session.retention import DESKTOP_MARKER_NAME
    from local_operator.session.runtime.inbox import INBOX_NAME
    from local_operator.session.runtime.registry import (
        STOP_MARKER_NAME,
        TURN_JOURNAL_NAME,
    )

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(_row(index, f"line {index}") for index in range(rows)), encoding="utf-8"
    )
    (directory / TITLE_SIDECAR_NAME).write_text(
        json.dumps({"title": "mesh design"}), encoding="utf-8"
    )
    (directory / ATTACHMENT_SIDECAR_NAME).write_text(
        json.dumps({"team": "lopdev"}), encoding="utf-8"
    )
    (directory / ORIGIN_NAME).write_text(json.dumps({"origin": "fork"}), encoding="utf-8")
    (directory / TURN_JOURNAL_NAME).write_text(json.dumps({"turn": 2}), encoding="utf-8")
    (directory / STOP_MARKER_NAME).write_text(json.dumps({"rung": "sigterm"}), encoding="utf-8")
    (directory / INBOX_NAME).write_text(json.dumps({"id": "m1"}) + "\n", encoding="utf-8")
    (directory / FORK_BOUNDARY_NAME).write_text(
        json.dumps({"version": FORK_BOUNDARY_VERSION}), encoding="utf-8"
    )
    (directory / DESKTOP_MARKER_NAME).write_text(json.dumps({"cwd": "/tmp"}), encoding="utf-8")
    # A file that must NEVER travel: the liveness marker names a pid.
    (directory / ".session.pid").write_text("4242", encoding="utf-8")
    (directory / "subagent-roster.v1.json").write_text("[]", encoding="utf-8")
    # An attachment the transcript references, in the shared content-addressed store.
    blob = b"\x89PNG-ish bytes nobody can decode"
    ref = "0" * 32
    (root / "attachments").mkdir(parents=True, exist_ok=True)
    (root / "attachments" / f"{ref}.bin").write_bytes(blob)
    (root / "attachments" / f"{ref}.json").write_text(
        json.dumps({"mime_type": "image/png"}), encoding="utf-8"
    )
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"id": "e9", "type": "image", "attachment": ref, "mime_type": "image/png"})
            + "\n"
        )
    return directory


# ---------------------------------------------------------------------------
# The copy set
# ---------------------------------------------------------------------------


def test_the_copy_set_names_match_the_modules_that_own_them() -> None:
    """The literals in ``sync.COPY_SET_NAMES`` are pinned against their owners.

    The list is spelled there rather than imported because the relay imports this
    module at CONSTRUCTION and those are session-engine modules. A duplicated
    literal is only safe with a test that fails when the two come apart — which is
    this one, and it is the reason the duplication is allowed.
    """
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.resume import (
        ATTACHMENT_SIDECAR_NAME,
        ORIGIN_NAME,
        TITLE_SIDECAR_NAME,
    )
    from local_operator.session.retention import DESKTOP_MARKER_NAME
    from local_operator.session.runtime.inbox import INBOX_NAME
    from local_operator.session.runtime.registry import (
        STOP_MARKER_NAME,
        TURN_JOURNAL_NAME,
    )

    assert set(sync.COPY_SET_NAMES) == {
        "transcript.jsonl",
        TITLE_SIDECAR_NAME,
        ATTACHMENT_SIDECAR_NAME,
        ORIGIN_NAME,
        TURN_JOURNAL_NAME,
        STOP_MARKER_NAME,
        INBOX_NAME,
        FORK_BOUNDARY_NAME,
        DESKTOP_MARKER_NAME,
    }
    # And the deny-list is a real deny-list: every name on it is a file a session
    # directory actually holds, so the copy's allow-list is the only thing keeping
    # them out.
    assert ".session.pid" in sync.NEVER_COPIED
    assert "subagent-roster.v1.json" in sync.NEVER_COPIED
    assert not (set(sync.NEVER_COPIED) & set(sync.COPY_SET_NAMES))


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


def test_a_copy_transfers_the_whole_set_byte_for_byte_and_verifies_it(tmp_path: Path) -> None:
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "9f3ac1e0b7d2")
    dest = tmp_path / "holder"

    result = sync.sync_from(source, "9f3ac1e0b7d2", ask=_ask(source), into=dest)

    assert result["ok"] is True
    assert result["mode"] == "replace"
    assert result["bytes"] > 0
    transcript = (directory / "transcript.jsonl").read_bytes()
    assert (dest / "transcript.jsonl").read_bytes() == transcript
    for name in sync.COPY_SET_NAMES:
        if name == "transcript.jsonl":
            continue
        assert (dest / name).is_file(), name
    # THE FILES THAT MUST NOT TRAVEL. A copied `.session.pid` would name the
    # SOURCE's process as the owner of the destination's copy, which is how a
    # viewer refuses to open a session that is not running there.
    assert not (dest / ".session.pid").exists()
    assert not (dest / "subagent-roster.v1.json").exists()
    # The referenced blob came with it, into the replica's own store.
    assert (dest / "attachments" / f"{'0' * 32}.bin").is_file()


def test_the_cursor_records_the_verified_prefix_not_a_hopeful_offset(tmp_path: Path) -> None:
    source = tmp_path / "owner"
    source.mkdir()
    seed(source, "abc123")
    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)

    cursor = sync.read_replica_cursor(source, "abc123")["cursor"]
    payload = (source / "sessions" / "abc123" / "transcript.jsonl").read_bytes()
    assert cursor["prefix_bytes"] == len(payload)
    assert cursor["prefix_digest"] == sync.sha256_bytes(payload)
    assert cursor["frontier"]


def test_the_second_sync_sends_only_the_append_region(tmp_path: Path) -> None:
    """R22's whole point: a follow-up sync costs the delta, not the transcript."""
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    before = (directory / "transcript.jsonl").stat().st_size

    appended = _row(42, "the turn that just finished") + _row(43, "and its answer")
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(appended)

    plan = sync.build_manifest(
        source, "abc123", have={"cursor": sync.read_replica_cursor(source, "abc123")["cursor"]}
    )
    assert plan["transcript"]["mode"] == "append"
    assert plan["transcript"]["region_bytes"] == len(appended.encode())

    second = sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert second["mode"] == "append"
    # ONLY the append region crossed: the delta is the new rows' bytes, and the
    # destination's file is the source's file.
    assert second["bytes"] == len(appended.encode())
    assert (dest / "transcript.jsonl").read_bytes() == (directory / "transcript.jsonl").read_bytes()
    assert (directory / "transcript.jsonl").stat().st_size == before + len(appended.encode())


def test_a_compaction_forces_a_full_replace(tmp_path: Path) -> None:
    """``Transcript.compact_file`` rewrites the file, so the prefix digest moves.

    This is what makes the cursor safe to keep: an append is offered ONLY when the
    owner's own first ``prefix_bytes`` still hash to what the holder recorded, and a
    compaction is exactly the operation that breaks that.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    cursor = sync.read_replica_cursor(source, "abc123")["cursor"]

    # A compaction: same rows, one of them rewritten (the prune journal folded in),
    # replacing the file exactly as ``compact_file`` does.
    payload = (directory / "transcript.jsonl").read_bytes()
    compacted = payload.replace(b"line 1", b"line one")
    (directory / "transcript.jsonl").write_bytes(compacted)

    plan = sync.build_manifest(source, "abc123", have={"cursor": cursor})
    assert plan["transcript"]["mode"] == "replace"
    assert plan["transcript"]["region_bytes"] == len(compacted)

    second = sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert second["mode"] == "replace"
    assert (dest / "transcript.jsonl").read_bytes() == compacted


def test_a_torn_tail_row_is_not_served(tmp_path: Path) -> None:
    """A half-written last line is dropped; the next sync carries it whole.

    ``--keep`` copies a session that is still being written, so the boundary both
    sides can agree on without parsing is the last newline.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    complete = (directory / "transcript.jsonl").read_bytes()
    torn = b'{"id": "e77", "type": "assist'
    with (directory / "transcript.jsonl").open("ab") as handle:
        handle.write(torn)

    plan = sync.build_manifest(source, "abc123")
    assert plan["transcript"]["total_bytes"] == len(complete)
    assert plan["transcript"]["source_bytes"] == len(complete) + len(torn)
    # And once the runtime finishes the row, the next plan carries it.
    with (directory / "transcript.jsonl").open("ab") as handle:
        handle.write(b'ant", "content": "done"}\n')
    grown = sync.build_manifest(source, "abc123")
    assert grown["transcript"]["total_bytes"] > len(complete)


def test_an_interrupted_copy_resumes_from_a_verified_prefix(tmp_path: Path) -> None:
    """The design's "resumable by construction", made checkable.

    The interrupted attempt is modelled the way a dropped link models it: the
    transport stops answering mid-file. What must NOT happen is the retry splicing
    unverified bytes — so the resume asks the owner to verify the prefix it finds
    on disk, and continues only when the owner agrees.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    # A transcript big enough to need several chunks, so "interrupted" means a
    # PARTIAL file rather than no file: one chunk that lands whole would test the
    # success path with an exception bolted on.
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        for index in range(200, 2600):
            handle.write(_row(index, "x" * 400))
    dest = tmp_path / "holder"
    real = _ask(source)
    delivered = {"chunks": 0}

    def flaky(frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("phase") == "fetch":
            delivered["chunks"] += 1
            if delivered["chunks"] > 2:
                raise ConnectionError("the link dropped mid-copy")
        return real(frame)

    with pytest.raises(ConnectionError):
        sync.sync_from(source, "abc123", ask=flaky, into=dest)
    partial = (dest / "transcript.jsonl").stat().st_size
    assert 0 < partial < (directory / "transcript.jsonl").stat().st_size

    resumes = {"asked": 0}
    verifying = _ask(source)

    def watching(frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("phase") == "verify":
            resumes["asked"] += 1
        return verifying(frame)

    result = sync.sync_from(source, "abc123", ask=watching, into=dest)
    assert result["ok"] is True
    assert resumes["asked"] == 1, "a partial file must be verified before it is resumed"
    assert result["resumed_bytes"] == partial
    assert (dest / "transcript.jsonl").read_bytes() == (directory / "transcript.jsonl").read_bytes()


def test_a_resume_never_takes_the_destination_s_word_for_it(tmp_path: Path) -> None:
    """A partial file that is NOT the source's prefix is discarded, not appended to.

    THIS IS THE TEST THAT MAKES THE RESUME SAFE RATHER THAN HOPEFUL. ``_fetch_item``
    will continue from bytes already on disk only when the owner confirms its own
    first N bytes hash the same, so a destination whose copy has diverged (a failed
    earlier attempt, a hand-edited file, a source that was compacted in between)
    gets a fresh copy instead of a splice. Without the check the file below keeps
    its extra row and the two devices disagree about the conversation with nothing
    downstream able to tell.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    # The destination's copy diverges AND is SHORTER than the source, which is the
    # shape that reaches the resume path: a file at least as long as the source is
    # simply replaced, so only a short one can tempt an append.
    whole = (directory / "transcript.jsonl").read_bytes()
    divergent = whole[: len(whole) // 2].replace(b"line 1", b"LINE 1")
    assert len(divergent) < len(whole) and divergent != whole[: len(divergent)]
    (dest / "transcript.jsonl").write_bytes(divergent)

    result = sync.sync_from(source, "abc123", ask=_ask(source), into=dest)

    assert result["ok"] is True
    assert result["resumed_bytes"] == 0, "divergent bytes must not be treated as a prefix"
    assert b"LINE 1" not in (dest / "transcript.jsonl").read_bytes()
    assert (dest / "transcript.jsonl").read_bytes() == whole
    assert (dest / "transcript.jsonl").read_bytes() == (directory / "transcript.jsonl").read_bytes()
    assert b"invented" not in (dest / "transcript.jsonl").read_bytes()


def test_a_torn_chunk_is_refused_and_never_reaches_the_copy(tmp_path: Path) -> None:
    source = tmp_path / "owner"
    source.mkdir()
    seed(source, "abc123")
    dest = tmp_path / "holder"
    real = _ask(source)

    def corrupting(frame: dict[str, Any]) -> dict[str, Any]:
        answer = real(frame)
        if frame.get("phase") == "fetch" and answer.get("bytes"):
            raw = bytearray(__import__("base64").b64decode(answer["data"]))
            raw[0] ^= 0xFF
            answer = {**answer, "data": __import__("base64").b64encode(bytes(raw)).decode()}
        return answer

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.sync_from(source, "abc123", ask=corrupting, into=dest)
    assert refusal.value.code == "digest_mismatch"


def test_a_stale_plan_is_refused_rather_than_spliced(tmp_path: Path) -> None:
    """A chunk asked for against a plan that no longer describes the source."""
    source = tmp_path / "owner"
    source.mkdir()
    directory = seed(source, "abc123")
    plan = sync.build_manifest(source, "abc123")
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_row(50, "a turn landed between the plan and the fetch"))

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.serve_fetch(
            source,
            "abc123",
            plan=str(plan["plan_id"]),
            name="transcript.jsonl",
            offset=0,
        )
    assert refusal.value.code == "stale_plan"


def test_a_fetch_cannot_reach_outside_the_copy_set(tmp_path: Path) -> None:
    """The deny-list is enforced on the SERVE side, not just by the planner."""
    source = tmp_path / "owner"
    source.mkdir()
    seed(source, "abc123")
    plan = sync.build_manifest(source, "abc123")
    for name in (".session.pid", "subagent-roster.v1.json", "../transcript.jsonl"):
        with pytest.raises(sync.SyncRefused) as refusal:
            sync.serve_fetch(source, "abc123", plan=str(plan["plan_id"]), name=name, offset=0)
        assert refusal.value.code == "unknown_item"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_a_replica_is_recovered_as_a_new_id_fork(tmp_path: Path) -> None:
    """INV-1's deliberate deviation: the copy NEVER comes back under the old id.

    A peer that spins down can come back. Promoting the replica under the original
    id is then two devices writing one conversation, and nothing downstream can
    tell which is the real one.
    """
    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    # The replica holds the copy; the session it came from is NOT on this device.
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")

    promoted = sync.promote_replica(root, "9f3ac1e0b7d2")

    new_id = promoted["session_id"]
    assert new_id != "9f3ac1e0b7d2"
    target = root / "sessions" / new_id
    assert (target / "transcript.jsonl").is_file()
    assert not (root / "sessions" / "9f3ac1e0b7d2").exists()
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.session.placement import read_stamp

    assert (target / FORK_BOUNDARY_NAME).is_file()
    stamp = read_stamp(root, new_id)
    assert stamp is not None
    assert stamp.origin["kind"] == "fork"
    assert stamp.origin["source_session_id"] == "9f3ac1e0b7d2"
    assert stamp.origin["source_device"] == "d_owner"
    assert stamp.replicas == []
    # The blobs reached the SHARED store, or the recovered transcript's images
    # would render broken.
    assert (root / "attachments" / f"{'0' * 32}.bin").is_file()
    # No liveness claim was left behind: it would name the promoting process.
    assert not (target / ".session.pid").exists()


def test_recovery_refuses_when_there_is_nothing_synced(tmp_path: Path) -> None:
    with pytest.raises(sync.SyncRefused) as refusal:
        sync.promote_replica(tmp_path, "never-synced")
    assert refusal.value.code == "no_replica"


# ---------------------------------------------------------------------------
# The owner's watcher
# ---------------------------------------------------------------------------


class _FakeServer:
    """The watcher's whole surface: a root, a dial, and a request number."""

    def __init__(self, root: Path, pushed: list[tuple[str, str, str]]) -> None:
        self.root = root
        self.identity = type("I", (), {"device_id": "d_me"})()
        self._pushed = pushed
        self._req = 100

    def _next_relay_req(self) -> int:
        self._req += 1
        return self._req

    def _member_name(self, device_id: str) -> str:
        return device_id

    def _ensure_link(self, device_id: str) -> Any:
        holder = self

        class _Link:
            def request(self, frame: dict[str, Any], timeout: float | None = None) -> Any:
                holder._pushed.append(
                    (device_id, str(frame.get("session_id") or ""), str(frame.get("phase") or ""))
                )
                return {"op": "ack", "req": frame.get("req"), "detail": {"acknowledged": True}}

        return _Link()


def _publish_live_record(root: Path, session_id: str) -> None:
    """A REAL registry record naming THIS process, so a session reads as live.

    The watcher's second trigger is the runtime's record DISAPPEARING, and a fake
    ``live`` set would test the fake. Publishing a record whose pid is this process
    makes ``registry.scan`` classify it live for the ordinary reason, and
    ``registry.unpublish`` then models the idle exit exactly.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            session_id=session_id,
            kind="tui",
            cwd=str(root),
            conversation_name="",
            model_label="",
            control_port=0,
            control_key="",
        ),
        root,
    )


def test_the_watcher_debounces_then_flushes_on_the_idle_exit(tmp_path: Path) -> None:
    """Cadence: one push per quiet period, and one immediately when the runtime goes.

    Driven through ``tick(now=…)`` rather than by sleeping: the debounce window is
    the class's whole behaviour, and a test that waits 30 real seconds for it is a
    test nobody runs twice.
    """
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )
    from local_operator.session.runtime import registry

    push: list[tuple[str, str, str]] = []
    server = _FakeServer(tmp_path, push)
    seed(tmp_path, "abc123")
    write_stamp(
        tmp_path,
        MeshStamp(
            session_id="abc123",
            network_id="n1",
            home_device="d_me",
            placement=SessionPlacement(mode="peer", network_id="n1", home_device="d_me"),
            replicas=["d_holder"],
        ),
    )
    _publish_live_record(tmp_path, "abc123")
    # The watcher's whole surface is four methods on the relay, so a stand-in IS the
    # point of this test: the stand-in records the pushes and answers the dial.
    settings = sync.SyncSettings(debounce_s=30.0, tick_s=15.0)
    watcher = sync.SyncWatcher(server, settings)  # type: ignore[arg-type]
    transcript = tmp_path / "sessions" / "abc123" / "transcript.jsonl"

    assert watcher.tick(now=1000.0) == []  # first sight: nothing has changed yet

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_row(60, "a turn landed"))
    assert watcher.tick(now=1010.0) == []  # changed, but not quiet yet
    pushed = watcher.tick(now=1045.0)  # 35 s of quiet
    assert pushed == [("abc123", "quiet")]
    assert push and push[-1][2] == "available"

    # No new change: no second push, however many ticks pass.
    first = len(push)
    assert watcher.tick(now=1100.0) == []
    assert watcher.tick(now=1200.0) == []
    assert len(push) == first

    # THE IDLE EXIT: the runtime's record went away, so the last turn's bytes are
    # flushed NOW rather than after another debounce. This is the design's whole
    # answer to "the final message before idle", and it needs no runtime change.
    registry.unpublish(os.getpid(), tmp_path)
    flushed = watcher.tick(now=1201.0)
    assert flushed == [("abc123", "idle-exit")]
    assert len(push) == first + 1


def test_the_watcher_says_nothing_about_a_session_without_replicas(tmp_path: Path) -> None:
    """A stamp with no holders means no watcher work: this is the 0-peer case."""
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    push: list[tuple[str, str, str]] = []
    server = _FakeServer(tmp_path, push)
    seed(tmp_path, "abc123")
    write_stamp(
        tmp_path,
        MeshStamp(
            session_id="abc123",
            network_id="n1",
            home_device="d_me",
            placement=SessionPlacement(mode="peer", network_id="n1", home_device="d_me"),
            replicas=[],
        ),
    )
    settings = sync.SyncSettings(debounce_s=30.0, tick_s=15.0)
    watcher = sync.SyncWatcher(server, settings)  # type: ignore[arg-type]
    assert watcher.tick(now=1000.0) == []
    assert watcher.tick(now=5000.0) == []
    assert push == []

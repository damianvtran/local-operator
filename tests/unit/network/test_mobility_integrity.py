"""The round-1 integrity findings, each as the property the design promises.

WHY THIS FILE EXISTS. Review round 1 of slice M measured two BLOCKERs and seven
MAJORs against ``ff04d03f1`` with a probe file (``test_review_m_probes.py``, the
reviewer's scratchpad). Every test here is the permanent form of one of those
probes, kept next to the code it protects rather than in a scratchpad, and each one
fails on the pre-fix code for the reason its docstring names.

WHAT IT COVERS, in the reviewer's own numbering:

* **B-M2 / p2c** — a move carries ``scratchpad/`` and ``created_at.json``; and a
  move REFUSES a session holding an entry the copy set does not carry, because the
  commit deletes the source directory.
* **M-1 / p2f** — a moved conversation's attachments land where the store reads
  them, with their ``.json`` sidecars.
* **M-2 / p2a, p2b** — the owner commits only on a content digest it can re-derive
  from its own bytes, and the destination re-verifies its staging before adopting.
* **M-3 / p6a, p1c** — recovery runs itself (a relay start, an engage), and an
  engage refusal names the device the conversation is coming FROM.
* **M-4 / p5b** — a push makes the holder PULL: the replica comes to hold the last
  assistant message over a real link, not by counting calls on a fake server.
* **MINOR 1 / p3** — a refused move does not rewrite the source's stamp, and a
  move's copy does not register its destination as a replica holder.

The relay-level helpers (``pair``, ``_owned_session``, ``_move``) come from
``test_mobility.py`` so these cells run the same real two-relay mesh as the rest of
the slice's tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, relay, sync
from local_operator.network.projection import read_tombstones
from local_operator.session.placement import (
    HANDOFF_PHASE_HANDING_OFF,
    read_handoff_journal,
    read_stamp,
    write_handoff_entry,
)
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures
    SESSION,
    Devices,
    _move,
    _owned_session,
    _transcript,
    devices,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair


def _transcript_with_scratchpad(
    server: relay.RelayServer, session_id: str = SESSION
) -> tuple[Path, str, bytes]:
    """A session with a scratchpad TREE and a birth-time sidecar, as the product has.

    Built by hand rather than by ``_owned_session`` because these are the two
    entries whose absence from the copy set was the data loss: 3,017 scratchpads and
    10,840 ``created_at.json`` sidecars exist in the operator's real store, so the
    fixture has to hold them (B-M2).
    """
    directory = _owned_session(server, session_id)
    birth = "1758230400.5"
    (directory / "created_at.json").write_text(birth, encoding="utf-8")
    notes = directory / "scratchpad" / "notes"
    notes.mkdir(parents=True)
    (notes / "plan.md").write_text("the operator's plan\n", encoding="utf-8")
    (directory / "scratchpad" / "run.sh").write_text("echo hello\n", encoding="utf-8")
    return directory, birth, (notes / "plan.md").read_bytes()


# ---------------------------------------------------------------------------
# B-M2: the copy set covers what the source will delete
# ---------------------------------------------------------------------------


def test_a_move_carries_the_scratchpad_and_the_birth_time(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """``scratchpad/`` and ``created_at.json`` travel, and the source is deleted.

    Measured before the fix (``p2c``): ``{"ok": true, "src_exists": false,
    "dest_scratchpad": false, "dest_created_at": false}`` — the scratch files existed
    on NEITHER device afterwards, and the moved conversation's creation date reset to
    the destination's own ``st_birthtime``. Both are unrecoverable once the source
    directory is gone, which is what made this a blocker.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source, birth, plan_bytes = _transcript_with_scratchpad(server_a)
    before = _transcript(server_a.root, SESSION)

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    destination = server_b.root / "sessions" / SESSION
    assert not source.exists(), "the source directory is retired by a move"
    assert _transcript(server_b.root, SESSION) == before
    assert (destination / "scratchpad" / "notes" / "plan.md").read_bytes() == plan_bytes
    assert (destination / "scratchpad" / "run.sh").read_text(encoding="utf-8") == "echo hello\n"
    assert (destination / "created_at.json").read_text(encoding="utf-8") == birth


def test_an_entry_that_appears_after_prepare_refuses_the_commit(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The commit re-checks, because a file can appear between prepare and commit.

    THE RACE THE EARLY CHECK CANNOT COVER, and the reason both exist. ``prepare``
    lists the directory once; anything the user (or an agent) writes into it while the
    copy is in flight — a scratch download, an editor's swap file, a database — is
    invisible to that listing and would be DELETED by the commit. So the same rule runs
    again immediately before the only delete, next to the lease re-check, and it fails
    closed with the source directory untouched and no tombstone written.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    real = mobility.LinkTransport.ask
    late = source / "written-while-the-copy-ran.dat"

    def meddling(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        answer = real(self, frame)
        if frame.get("phase") == "prepare":
            late.write_bytes(b"the operator's other work")
        return answer

    monkeypatch.setattr(mobility.LinkTransport, "ask", meddling)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert "written-while-the-copy-ran.dat" in str(refused["message"]), refused
    assert source.is_dir() and late.read_bytes() == b"the operator's other work"
    assert _transcript(server_a.root, SESSION) == before
    assert read_tombstones(server_a.root) == {}, "the id was tombstoned while refusing"
    assert read_handoff_journal(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_the_unlisted_entry_refusal_happens_before_a_single_byte_is_copied(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The entry is refused BEFORE the retire and before the copy, not after it.

    TWO CHECKS, ONE RULE, and this is why both exist: the plan-time refusal (here)
    costs nothing, and the commit-time one (the previous test's subject) is what
    cannot be raced. The source is retired and the whole conversation is copied on the
    way to a commit-time-only refusal, so a session with an unclassified entry would
    spend a real transfer and a stopped runtime to reach the same sentence. Counted
    on the wire: B's side asks for no chunks at all.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    (source / "something-nobody-classified.dat").write_bytes(b"\x00\x01\x02")
    fetched: list[str] = []
    real = mobility.LinkTransport.ask

    def counting(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("phase") == "fetch":
            fetched.append(str(frame.get("name") or ""))
        return real(self, frame)

    monkeypatch.setattr(mobility.LinkTransport, "ask", counting)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert "something-nobody-classified.dat" in str(refused["message"]), refused
    assert fetched == [], f"the refusal came after copying {fetched}"
    assert source.is_dir()
    assert not sync.staging_dir(server_b.root, SESSION).exists()


def test_a_move_refuses_a_session_holding_an_entry_the_copy_set_does_not_carry(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """FAIL CLOSED: an unclassified entry refuses the move rather than being deleted.

    THE GUARD THAT MAKES THE NEXT MISS SURVIVABLE. A copy set written as a list of
    files cannot notice a file type nobody told it about, and the fix for B-M2 that
    only added two names would leave the next one to be destroyed the same way. The
    refusal happens BEFORE the retire, so the conversation is not even stopped, and
    the same rule is re-checked at the commit for an entry that appears in between.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    (source / "something-nobody-classified.dat").write_bytes(b"\x00\x01\x02")
    before = _transcript(server_a.root, SESSION)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert "something-nobody-classified.dat" in str(refused["message"]), refused
    # NOTHING CHANGED ON EITHER SIDE, and the source is still whole.
    assert source.is_dir() and _transcript(server_a.root, SESSION) == before
    assert (source / "something-nobody-classified.dat").read_bytes() == b"\x00\x01\x02"
    assert read_handoff_journal(server_a.root) == {}
    assert read_tombstones(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not sync.staging_dir(server_b.root, SESSION).exists()


# ---------------------------------------------------------------------------
# M-1: attachments land where the store reads them
# ---------------------------------------------------------------------------


def test_a_moved_conversation_s_attachments_land_where_the_store_reads_them(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A real ``AttachmentStore.put`` on the source resolves on the destination.

    Measured before the fix (``p2f``): ``blob_paths_on_b:
    ["attachments/attachments/c202….bin"], store_resolves: false`` — the blob landed
    one directory too deep (the store root was handed to a lookup that appends the
    store-relative name again), so EVERY image in a moved conversation rendered
    broken, and the ``.json`` sidecar that carries the mime type was not copied at
    all.
    """
    import base64

    from local_operator.session.attachments import AttachmentStore

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    reference = AttachmentStore(server_a.root / "attachments").put(
        base64.b64encode(b"\x89PNG fake image bytes").decode(), "image/png"
    )
    assert reference is not None
    digest = getattr(reference, "digest", None) or str(reference)
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "e7", "type": "image", "attachment": digest}) + "\n")

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    destination = AttachmentStore(server_b.root / "attachments")
    resolved = destination.get(digest)
    assert (
        resolved is not None
    ), "the moved conversation's image does not resolve on the destination"
    payload, mime = resolved
    assert mime == "image/png", resolved
    assert base64.b64decode(payload) == b"\x89PNG fake image bytes"
    # The path the reviewer measured, named: the blob is in the install's shared
    # store exactly once, not nested under another ``attachments/``.
    assert (server_b.root / "attachments" / f"{digest}.bin").is_file()
    assert not (server_b.root / "attachments" / "attachments").exists()


# ---------------------------------------------------------------------------
# M-2: the commit decides on content it can re-derive
# ---------------------------------------------------------------------------


def test_a_commit_refuses_a_destination_digest_that_does_not_match_its_bytes(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A zeroed digest commits nothing: the owner compares its OWN content.

    Measured before the fix (``p2a``): rewriting the ``ready`` frame's digest to
    ``sha256:000…`` still produced ``{"ok": true, "src_exists": false}``, because the
    owner's only check was that the field was non-empty — and the field was the
    owner's own manifest digest echoed back, so comparing it could not have proved
    anything about the destination's bytes anyway. What it compares now is a digest
    each end derives from its own directory with ``sync.copy_content_digest``.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    real = mobility.LinkTransport.ask
    seen: dict[str, Any] = {}

    def zeroed(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        if frame.get("phase") == "ready":
            seen["sent"] = frame.get("content_digest")
            frame = {**frame, "content_digest": "sha256:" + "0" * 64}
        return real(self, frame)

    monkeypatch.setattr(mobility.LinkTransport, "ask", zeroed)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert seen.get("sent"), "the destination must report a digest of its own bytes"
    assert seen["sent"] != "sha256:" + "0" * 64
    assert refused["ok"] is False, refused
    assert refused["code"] == "digest_mismatch", refused
    # A ROLLBACK, NOT A LOSS: the conversation is still here, byte for byte.
    assert source.is_dir() and _transcript(server_a.root, SESSION) == before
    assert read_handoff_journal(server_a.root) == {}
    assert read_tombstones(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_a_truncated_staging_copy_is_never_committed(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A copy truncated after verification is caught by the digest, not trusted.

    Measured before the fix (``p2b``): truncating the staged transcript to 100 bytes
    of 2,580 still committed — ``{"ok": true, "src_bytes": 2580, "dest_bytes": 100,
    "src_exists": false}`` — so a truncated conversation was kept and the whole one
    deleted. The destination re-derives the digest of what it ACTUALLY holds at the
    moment it declares the copy ready, so the truncation is a mismatch.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a, rows=50)
    before = _transcript(server_a.root, SESSION)
    real_sync_from = sync.sync_from

    def truncating(*args: Any, **kwargs: Any) -> Any:
        out = real_sync_from(*args, **kwargs)
        staged = Path(kwargs["into"]) / "transcript.jsonl"
        staged.write_bytes(staged.read_bytes()[:100])
        return out

    monkeypatch.setattr(sync, "sync_from", truncating)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "digest_mismatch", refused
    assert source.is_dir() and _transcript(server_a.root, SESSION) == before
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert read_tombstones(server_a.root) == {}


def test_damage_after_the_ready_handshake_is_never_promoted(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The destination re-hashes its staging before the one ``os.replace``.

    THE WINDOW THE OWNER CANNOT COVER: by the time this device promotes, the owner
    may already have deleted its copy, so the staged bytes are the last ones. A
    partial or edited copy must not be presented as a session — and the bytes are
    KEPT (not swept) so a human can still recover the conversation by hand, which the
    refusal sentence says.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a, rows=20)
    real = mobility.LinkTransport.ask
    staged_path = sync.staging_dir(server_b.root, SESSION) / "transcript.jsonl"

    def damaging(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        answer = real(self, frame)
        # The owner has committed by the time this frame is answered; damage the
        # staged copy in the instant before the destination adopts it.
        if frame.get("phase") == "ready" and staged_path.is_file():
            staged_path.write_bytes(staged_path.read_bytes()[:64])
        return answer

    monkeypatch.setattr(mobility.LinkTransport, "ask", damaging)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "digest_mismatch", refused
    assert not (
        server_b.root / "sessions" / SESSION
    ).exists(), "a damaged copy was promoted as a session"
    assert staged_path.is_file(), "the only remaining copy was swept instead of kept"
    assert not source.exists(), "the owner had already committed, so this is the last copy"


# ---------------------------------------------------------------------------
# M-3: recovery runs itself, and names the right device
# ---------------------------------------------------------------------------


def test_a_relay_start_recovers_a_stale_handoff_and_frees_the_owner_s_session(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A fresh relay on a root a dead one left an entry in settles it by itself.

    Measured before the fix (``p6a``): after a fresh ``RelayServer`` started on the
    same root, the ``prepared`` entry from instance ``i_dead`` was still there and
    ``engage_runtime`` answered "This conversation is being handed to build-box" —
    the owner's OWN conversation, unusable until somebody happened to run a move of
    the same id again. Nothing in the product called ``reconcile``.
    """
    from local_operator.session.runtime.launch import (
        RuntimeStartupError,
        WarmErrand,
        engage_runtime,
    )

    server_a, _server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": "d_gone",
            "to_name": "build-box",
            "instance_id": "i_dead",
            "at": 1.0,
        },
    )

    fresh = relay.RelayServer(
        root=server_a.root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    fresh.bind_control()
    fresh.start()
    try:
        deadline = time.monotonic() + 10
        while SESSION in read_handoff_journal(server_a.root) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert SESSION not in read_handoff_journal(
            server_a.root
        ), "a relay start left a stale handoff entry in place"
    finally:
        fresh.stop()

    # And the guard no longer speaks for it: the engage gets as far as its own
    # runtime work rather than being refused by a handoff that is not happening.
    import asyncio

    refused = ""
    try:
        asyncio.run(
            engage_runtime(
                SESSION, str(server_a.root), WarmErrand(), config_dir=server_a.root, deadline_s=0.0
            )
        )
    except RuntimeStartupError as exc:
        refused = str(exc)
    except Exception as exc:  # noqa: BLE001 — a timeout is a fine answer here
        refused = f"other:{type(exc).__name__}"
    assert "being handed to" not in refused, refused


def test_an_engage_clears_a_stale_entry_left_by_a_dead_relay(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The engage PATH runs the recovery, not just the helper.

    ``recover_stale_handoff`` is correct and useless unless the engage calls it: this
    drives ``engage_runtime`` — the single entry point every engage path uses (a
    local viewer's first message, the phone daemon, a scheduled wake, ``lop exec``,
    the relay's own ``net_session_engage``) — and asserts the stale entry is gone
    afterwards, so an engage cannot be answered by a move that is not happening.
    With the recovery removed from that path, the guard speaks for the dead relay and
    this test fails on its sentence.
    """
    import asyncio

    from local_operator.session.runtime.launch import (
        RuntimeStartupError,
        WarmErrand,
        engage_runtime,
    )

    server_a, _server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    # No relay on this root: an entry here is a leftover by definition.
    server_a.stop()
    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": "d_gone",
            "to_name": "build-box",
            "instance_id": "i_dead",
            "at": 1.0,
        },
    )

    refused = ""
    try:
        asyncio.run(
            engage_runtime(
                SESSION, str(server_a.root), WarmErrand(), config_dir=server_a.root, deadline_s=0.0
            )
        )
    except RuntimeStartupError as exc:
        refused = str(exc)
    except Exception as exc:  # noqa: BLE001 — a timeout is a fine answer here
        refused = f"other:{type(exc).__name__}"

    assert "being handed to" not in refused, refused
    assert SESSION not in read_handoff_journal(
        server_a.root
    ), "the engage path left a dead relay's handoff entry in place"


def test_the_destination_s_engage_refusal_names_the_source_device(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The sentence names the device the conversation comes FROM, not us.

    An entry records BOTH ends, and the old guard used ``to_name`` — which for a
    destination entry is this device — so an engage was refused with "This
    conversation is being received from <this device's own id>": a sentence about
    nothing, naming the wrong end (M-3, NIT 5).
    """
    from local_operator.session.placement import handoff_guard_refusal

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_b)
    write_handoff_entry(
        server_b.root,
        SESSION,
        {
            "role": "destination",
            "phase": HANDOFF_PHASE_HANDING_OFF,
            "from_device": server_a.identity.device_id,
            "from_name": "build-box",
            "to_device": server_b.identity.device_id,
            "instance_id": "i_live",
            "at": 1.0,
        },
    )

    sentence = handoff_guard_refusal(server_b.root, SESSION)

    assert "being received from build-box" in sentence, sentence
    assert server_b.identity.device_id not in sentence, sentence


def test_an_engage_recovers_a_stale_entry_when_no_relay_is_running(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The engage-time trigger: a leftover from a DEAD relay is rolled back.

    With no live relay there can be no live move — every phase of one is driven from
    a relay — so an entry is a leftover and the recovery applies. With a live relay
    on the root the entry is left alone unless it names a different instance, which is
    what keeps this from rolling a real handoff back.
    """
    from local_operator.session.runtime.launch import recover_stale_handoff

    server_a, _server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    directory = _owned_session(server_a)
    server_a.stop()
    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": "d_gone",
            "to_name": "pixel",
            "instance_id": "i_dead",
            "at": 1.0,
        },
    )

    report = recover_stale_handoff(server_a.root, SESSION)

    assert [row["action"] for row in report] == ["rolled_back"], report
    assert SESSION not in read_handoff_journal(server_a.root)
    assert directory.is_dir(), "rolling a prepared move back must not touch the session"


# ---------------------------------------------------------------------------
# M-4: a push makes the holder pull
# ---------------------------------------------------------------------------


def test_a_push_makes_the_holder_pull_the_last_message_over_a_real_link(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """End to end: B syncs, A appends the final answer and pushes, B's copy catches up.

    Measured before the fix (``p5b``): the push was acknowledged and NOTHING was
    pulled — the replica still lacked "the last answer" two seconds later — because
    the ``available`` branch's ack WAS the whole handler and no code anywhere called
    ``sync_from`` except the CLI and a move. The only test that looked like coverage
    counted calls on a fake server; this one reads the bytes off B's disk after a pull
    over the real link.
    """
    import os

    from local_operator.session.runtime import registry
    from tests.unit.network.test_sync import _publish_live_record

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    first = sync.request_sync(SESSION, root=server_b.root)
    assert first["ok"] is True, first
    replica = sync.replica_dir(server_b.root, SESSION) / "transcript.jsonl"
    assert b"the last answer" not in replica.read_bytes()

    # A publishes a live record, its watcher takes a FIRST sight of the session, then
    # the last turn lands and the runtime EXITS: the design's idle-exit trigger, which
    # is what carries the final assistant message. The first tick is not decoration —
    # the trigger is the record DISAPPEARING, so the watcher has to have seen it
    # present.
    _publish_live_record(server_a.root, SESSION)
    watcher = sync.SyncWatcher(server_a, sync.SyncSettings(debounce_s=30.0, tick_s=15.0))
    assert watcher.tick(now=1000.0) == []
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"id": "e50", "type": "assistant", "content": "the last answer"}) + "\n"
        )
    registry.unpublish(os.getpid(), server_a.root)
    pushed = watcher.tick(now=1001.0)
    assert (SESSION, "idle-exit") in pushed, pushed

    # THE HOLDER'S OWN REFRESHER does the pull, over the link the push came in on,
    # and it is the one THIS RELAY registered — not a stand-in built by the test —
    # so what is proven is the wiring the product has: the push marks a replica,
    # ``ensure_refresher`` owns the thread that pulls it, and one tick is the
    # cadence's unit. (A push for a session this device holds no verified replica of
    # is refused by the chokepoint; the ack above is the carve-out working.)
    refresher = sync.ensure_refresher(server_b)
    assert refresher.mark  # the push marked this refresher, not another one
    deadline = time.monotonic() + 20
    while b"the last answer" not in replica.read_bytes() and time.monotonic() < deadline:
        refresher.tick()
        time.sleep(0.05)

    assert (
        b"the last answer" in replica.read_bytes()
    ), "the push was acknowledged and nothing pulled"
    assert replica.read_bytes() == (source / "transcript.jsonl").read_bytes()


def test_a_relay_built_the_way_network_serve_builds_it_admits_the_owner_s_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME property, on the construction the product actually uses (Q-XH-1).

    WHY THIS TEST EXISTS. Every other push test in this package builds its relay as
    ``RelayServer(root=…)``, and ``lop network serve`` builds ``RelayServer(settings=…)``
    with NO root (``network/cli.py::_cmd_serve``). That difference was the whole bug:
    ``__init__`` handed the raw ``root`` argument to the authoriser's ``StoreView``, so
    on the real path ``_root`` was ``None``, ``replica_owner`` answered ``""`` for every
    id, ``Authorizer._replica_scope`` never admitted the owner's ``net_sync available``
    push, and every replica across a REAL host boundary stayed frozen for 150 s beyond
    the debounce — refused ``authorisation_refused … capability_denied`` at every push —
    while a manual ``lop sessions sync`` worked (cross-host QA, Q-XH-1). Passing an
    explicit root is not coverage for the shape the product supplies, so this cell builds
    BOTH relays the way ``serve`` does, with the ambient config dir naming each one's
    root, and drives the push over a real link with the holder's own refresher pulling.
    """
    from local_operator.network import identity

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    identity_a = identity.mint(root_a, name="serve-a")
    identity_b = identity.mint(root_b, name="serve-b")

    def _serve_shaped(root: Path, device_identity: Any) -> relay.RelayServer:
        # EXACTLY `lop network serve`'s call: no ``root``, settings only. The env tells
        # this relay whose install it is, which is why it is set per server.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        server = relay.RelayServer(
            settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
            identity=device_identity,
        )
        assert server.root == root, "a serve-shaped relay did not resolve the ambient root"
        return server

    server_a = _serve_shaped(root_a, identity_a)
    server_b = _serve_shaped(root_b, identity_b)
    host, port = server_a.bind()
    server_a.bind_control()
    server_a.start()
    # Named `serve_pair`, not `pair`: this module imports a FIXTURE called `pair` and a
    # local of that name shadows it (flake8 F811), which pytest would then hand to any
    # test that asked for it.
    serve_pair: Devices = (server_a, server_b, host, port)
    _pair(serve_pair, monkeypatch, role="admin")
    # B HAS TO ANSWER ITS OWN CONTROL SOCKET for the enrolment below, which is what
    # `lop sessions sync` does on the holder — the same reason test_mobility's `pair`
    # fixture starts B rather than leaving it dial-only.
    server_b.bind_control()
    server_b.start()
    source = _owned_session(server_a)

    # The holder enrols by CONTACT (the CLI's `sessions sync`), as the real holder does.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    first = sync.request_sync(SESSION, root=server_b.root)
    assert first["ok"] is True, first
    # THE AUTHORISER'S OWN QUESTION, answered off the store the relay built. Pre-fix this
    # was "" on a serve-shaped relay, which is the refusal that hid the bug.
    assert (
        server_b.store_view.replica_owner(SESSION) == server_a.identity.device_id
    ), "the holder's store does not know which device its replica came from"

    # Now the owner pushes over the REAL link, and the reply is the assertion: a refusal
    # here is `authorisation_refused … capability_denied`, the exact record both real
    # peers' audits carried across the internet.
    link = server_a._ensure_link(server_b.identity.device_id)  # noqa: SLF001 — the dial seam
    assert link is not None, "the owner could not reach the holder"
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "e77", "type": "assistant", "content": "xh1"}) + "\n")
    pushed = link.request(
        {
            "op": "net_sync",
            "req": 9001,
            "phase": "available",
            "session_id": SESSION,
            "reason": "test",
        }
    )
    assert pushed is not None and pushed.get("op") == "ack", pushed

    # …and the pull follows with NO manual command: the ack marked the holder's OWN
    # refresher, and a tick of it brings the owner's new bytes across.
    refresher = sync.ensure_refresher(server_b)
    assert refresher.mark, "the admitted push did not mark the holder's refresher"
    replica = sync.replica_dir(server_b.root, SESSION) / "transcript.jsonl"
    deadline = time.monotonic() + 20
    while (
        replica.read_bytes() != (source / "transcript.jsonl").read_bytes()
        and time.monotonic() < deadline
    ):
        refresher.tick()
        time.sleep(0.05)
    assert (
        replica.read_bytes() == (source / "transcript.jsonl").read_bytes()
    ), "the replica did not advance by itself after the push"


# ---------------------------------------------------------------------------
# MINOR 1: a refused move leaves the source alone
# ---------------------------------------------------------------------------


def test_the_move_s_plan_does_not_register_a_replica_holder(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """``net_sync plan`` records a REPLICA holder; a move is not one, and says so.

    Measured before the fix (``p3``): the copy rides ``net_sync plan``, and that
    handler recorded the destination in the source's ``replicas`` and started the
    owner's watcher — so the owner pushed copies of the session to the device it had
    just handed the id to, and even a REFUSED move left the source's stamp changed.
    The two calls below are the same handler with one field different, which is what
    makes this a test of the reason rather than of the mechanism: only the replica
    request may register a holder.
    """
    from local_operator.network import sync

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    stamp_before = (source / "mesh.json").read_bytes()
    link = type(
        "Link",
        (),
        {
            "device_id": server_b.identity.device_id,
            "network_id": server_b.identity.device_id,
            "context": None,
        },
    )()
    handler = sync.make_handler(server_a)

    # A MOVE: the plan is served, and the source's stamp is untouched.
    plan = handler(
        link, {"op": "net_sync", "phase": "plan", "session_id": SESSION, "purpose": "move"}
    )
    assert plan["session_id"] == SESSION
    assert (
        source / "mesh.json"
    ).read_bytes() == stamp_before, "a move's plan registered its destination as a replica holder"
    stamp = read_stamp(server_a.root, SESSION)
    assert stamp is not None
    assert stamp.replicas == []
    assert id(server_a) not in sync._watchers  # noqa: SLF001 — nothing to push to

    # A REPLICA request is the thing that records one, so the assertion above is a
    # difference this handler makes rather than a field nobody sets.
    handler(
        link,
        {"op": "net_sync", "phase": "plan", "session_id": SESSION, "purpose": "replica"},
    )
    stamp = read_stamp(server_a.root, SESSION)
    assert stamp is not None and stamp.replicas == [server_b.identity.device_id]


def test_a_refused_move_does_not_rewrite_the_source_stamp(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A refusal mutates NOTHING on the source, down to the byte.

    The same defect from the other side: the copy rides ``net_sync plan``, and the
    plan handler used to record the destination as a replica holder before anything
    was verified — so even a move that ended in a digest mismatch left the source's
    stamp changed (``p3``). Property 3 of the design says a refusal changes nothing.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    stamp_before = (source / "mesh.json").read_bytes()
    listing_before = sorted(child.name for child in source.iterdir())
    real = mobility.LinkTransport.ask

    def meddling(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        answer = real(self, frame)
        if frame.get("phase") == "prepare":
            # The source changes under the copy: the commit must roll back.
            with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "e99", "type": "user", "content": "late"}) + "\n")
        return answer

    monkeypatch.setattr(mobility.LinkTransport, "ask", meddling)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert (
        source / "mesh.json"
    ).read_bytes() == stamp_before, "a refused move rewrote the source's mesh.json"
    assert sorted(child.name for child in source.iterdir()) == listing_before
    assert read_handoff_journal(server_a.root) == {}
    assert read_tombstones(server_a.root) == {}

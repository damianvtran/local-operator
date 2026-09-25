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
import shutil
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
from tests.unit.network.test_relay_e2e import _pair, serve_shaped_relay


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

    # THE CADENCE COMES FROM THE CONFIG, written before ANY push can start the refresher
    # (it reads its settings once, at construction, from this device's store). What the
    # cell then proves is that the RELAY'S OWN THREAD pulls — not that a call to ``tick()``
    # does what ``tick()`` says, which is what caught this round: the earlier version drove
    # the tick by hand, so the round-1 finding stayed reproducible while the test was green
    # (review round 2, MINOR 5). ``tick_s`` bottoms out at 1.0 s in the registry — a real
    # cadence rather than a busy loop, and short enough for a cell that waits on the thread.
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_writer = ConfigManager(server_b.root)
    for settings_key, settings_value in (
        ("network.sync.tick_s", 1.0),
        ("network.sync.debounce_s", 1.0),
    ):
        settings_io.write_setting(settings_writer, settings_io.BY_KEY[settings_key], settings_value)
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
    assert refresher._settings.tick_s == 1.0, "the refresher reads the config, not a constant"
    deadline = time.monotonic() + 20
    while b"the last answer" not in replica.read_bytes() and time.monotonic() < deadline:
        # NO tick() HERE: the thread this relay registered is the thing under test.
        time.sleep(0.05)

    assert (
        b"the last answer" in replica.read_bytes()
    ), "the push was acknowledged and the refresher's own cadence pulled nothing"
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
    explicit root is not coverage for the shape the product supplies, so both relays here
    are built through ``test_relay_e2e.serve_shaped_relay`` — the product's own shape, no
    ``root=`` — and the push is driven over a real link with the holder's own refresher
    pulling. That helper is now what the shared ``devices`` fixture builds too, so this
    module and the pairing/pilot matrix answer the authoriser's questions off the same
    construction the product uses (agent review round 1, MAJOR 3).
    """
    from local_operator.network import identity

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    identity_a = identity.mint(root_a, name="serve-a")
    identity_b = identity.mint(root_b, name="serve-b")

    server_a = serve_shaped_relay(root_a, monkeypatch, identity=identity_a)
    server_b = serve_shaped_relay(root_b, monkeypatch, identity=identity_b)
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


# ---------------------------------------------------------------------------
# Round 2: what the real store has, and the promote that landed and then failed
# ---------------------------------------------------------------------------


def test_a_move_carries_a_torn_tail_whole(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A partial last row travels BYTE FOR BYTE when the source is being deleted.

    THE ROUND-1 FINDING THAT SURVIVED THE ROUND-2 REMEDIATION (``p2d``): source 203 bytes,
    destination 150, and the source DELETED. ``_safe_region``'s cut exists for a session a
    runtime is still writing to — the next sync carries the tail — and a DELETING move has
    neither a writer (its runtime was retired first) nor a next sync. So the cut had to go
    for moves only, and the failure it caused was invisible to every digest: both ends
    hashed through the same cut, so the two agreed on the truncated bytes.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id": "e9", "type": "assistant", "content": "partial')
    before = _transcript(server_a.root, SESSION)
    assert not before.endswith(b"\n"), "the fixture must end mid-row for this to mean anything"

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    assert not source.exists(), "a move retires the source"
    assert (
        _transcript(server_b.root, SESSION) == before
    ), "the torn tail was dropped by the copy while the source was deleted"


def test_a_move_carries_a_judged_goal(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """``goal.json`` travels, so the objective and its history survive a handover.

    MEASURED BEFORE THE FIX (review round 2, MAJOR 1): a session with a judged goal could
    not be moved AT ALL (``unlisted_content``), and both ``--keep`` and replica recovery
    dropped the record silently. The judged-goal record is work in progress — an objective,
    its status and its settled history — and it is not derivable from the transcript.
    """
    from local_operator.resume import GOAL_SIDECAR_NAME

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    record = json.dumps({"goal": "ship the move", "status": "active", "judge": {"run": 1}})
    (source / GOAL_SIDECAR_NAME).write_text(record, encoding="utf-8")

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    landing = server_b.root / "sessions" / SESSION / GOAL_SIDECAR_NAME
    assert landing.is_file(), "the judged goal did not travel with the session"
    assert landing.read_text(encoding="utf-8") == record


def test_a_promote_that_landed_and_then_failed_can_still_be_opened(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The ``p1c`` window: ``os.replace`` succeeded and the process died before cleanup.

    The owner has COMMITTED and the session directory is on this device, so the move is
    finished — but the destination's handoff entry still says "being received", and every
    engage was refused with ``This conversation is being received from device-a; it will be
    available when the move finishes`` until this device's relay restarted. A live relay's
    entry is deliberately skipped by every recovery path (it may be driving a handoff this
    second), so the fix is a fact instead of an inference: a destination entry whose
    session directory exists WITH a transcript describes a handoff that already landed.
    """
    import os
    from pathlib import Path
    from typing import Any

    from local_operator.session.placement import handoff_guard_refusal

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    real_promote = mobility._promote

    class Died(BaseException):
        """The promote lands and the process stops before anything after it."""

    def promote_then_die(server: Any, staging: Path, target_id: str) -> bool:
        target = Path(server.root) / "sessions" / target_id
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging), str(target))
        raise Died()

    monkeypatch.setattr(mobility, "_promote", promote_then_die)
    # The move reports a failure — the promote raised out of its own call — and the point
    # of the cell is what state that leaves behind, not the sentence it produced.
    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)
    monkeypatch.setattr(mobility, "_promote", real_promote)
    assert failed.get("ok") is not True, failed

    assert (server_b.root / "sessions" / SESSION / "transcript.jsonl").is_file()
    assert SESSION not in read_handoff_journal(
        server_b.root
    ), "an entry for a handoff that already landed still blocks the conversation"
    assert handoff_guard_refusal(server_b.root, SESSION) == ""
    # The source went with it, and nothing was lost on this side.
    assert not (server_a.root / "sessions" / SESSION).exists()
    assert read_tombstones(server_a.root)


def test_a_move_of_a_session_whose_scratchpad_is_a_link_is_refused(
    pair: Devices, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — fixture
) -> None:
    """THE MAJOR, through the real two-relay mesh: the source SURVIVES with the link intact.

    The plan-level cell beside this one asserts the sentence and the plan; this one drives
    the thing that DELETES — ``_move`` retires the source runtime, verifies the copy and
    removes the directory. Before the fix it answered ``ok: true`` with ``trees: []`` and
    ``trees_skipped: []`` and removed the session directory with the link inside it (the
    reviewer's own ``test_r3_11`` reproduced exactly that).
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    directory = _owned_session(server_a)
    elsewhere = tmp_path / "other-volume"
    (elsewhere / "scratchpad").mkdir(parents=True)
    (elsewhere / "scratchpad" / "notes.md").write_text("the operator's notes\n", encoding="utf-8")
    shutil.rmtree(directory / "scratchpad", ignore_errors=True)
    (directory / "scratchpad").symlink_to(elsewhere / "scratchpad")

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result.get("ok") is not True, result
    assert result.get("code") == "unlisted_content", result
    assert "scratchpad" in str(result.get("message")), result
    assert directory.is_dir(), "the source survived the refusal"
    assert (directory / "scratchpad").is_symlink(), "and so did its link"
    assert (elsewhere / "scratchpad" / "notes.md").is_file()


def test_a_replica_with_a_corrupted_sidecar_is_not_promoted(tmp_path: Path) -> None:
    """A recovery verifies the WHOLE copy set, not only the transcript.

    MEASURED BEFORE THE FIX (review round 2, MINOR 3): replacing ``title.json`` in a
    verified replica let the promote succeed, and the recovered session came back wearing
    ``{"title": "WRONG"}`` — a title its owner never wrote. The cursor held no digest for
    the other members, so nothing had verified them; it records the whole attested set now,
    and the refusal names the member.
    """
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "owner"
    root.mkdir()
    directory = seed(root, "abc123", rows=50)
    assert (directory / "title.json").is_file()
    sync.sync_from(
        root,
        "abc123",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "abc123"),
    )
    replica = sync.replica_dir(root, "abc123")
    (replica / "title.json").write_text(json.dumps({"title": "WRONG"}), encoding="utf-8")
    before = (replica / "transcript.jsonl").read_bytes()

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.promote_replica(root, "abc123")

    assert refusal.value.code == "incomplete_replica", refusal.value
    assert "title.json" in refusal.value.message, "the refusal must name the member"
    # NOTHING WAS PROMOTED: the fixture's own session is the only directory in the store.
    assert [p.name for p in sorted((root / "sessions").iterdir())] == ["abc123"]
    assert (replica / "transcript.jsonl").read_bytes() == before, "the replica's bytes were kept"


def test_a_replica_missing_an_attested_member_is_refused_and_named(tmp_path: Path) -> None:
    """A member the cursor attested and that is GONE is named, not only digested.

    MEASURED BEFORE THE FIX (review round 3, MINOR 3): deleting ``turn-journal.json`` from a
    verified replica refused the recovery with ``incomplete_replica`` — the right answer,
    reached through the whole-set digest, and an ANONYMOUS one, while a REWRITTEN
    ``title.json`` named its file. The missing case is also the one the old comment reasoned
    about ("its absence is the honest state of this replica"): a member the cursor RECORDS
    was on disk when the cursor was written (the attested set is built from this device's own
    report), so its absence now is a loss, not a state.
    """
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "owner"
    root.mkdir()
    seed(root, "abc123", rows=50)
    sync.sync_from(
        root,
        "abc123",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "abc123"),
    )
    replica = sync.replica_dir(root, "abc123")
    assert (replica / "turn-journal.json").is_file()
    (replica / "turn-journal.json").unlink()
    before = (replica / "transcript.jsonl").read_bytes()

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.promote_replica(root, "abc123")

    assert refusal.value.code == "incomplete_replica", refusal.value
    assert (
        "turn-journal.json" in refusal.value.message
    ), "a member the sync verified and recovery refuses to promote must be NAMED"
    # NOTHING WAS PROMOTED, and the replica kept what it still has.
    assert [p.name for p in sorted((root / "sessions").iterdir())] == ["abc123"]
    assert (replica / "transcript.jsonl").read_bytes() == before


def test_the_source_is_hashed_before_its_runtime_is_retired(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """The digest pass runs FIRST; the conversation is stopped for the copy only.

    THE COST FIX'S OTHER HALF (review round 2, MAJOR 2: "keep the session's source runtime
    stopped only as long as it must be"). The plan and the content digest are one pass over
    every member — for the 1 GB scratchpad in the real store, 0.55 s of CPU and 5.8 s of
    wall on this host — and they used to run AFTER ``_retire_local_runtime``, so the
    conversation was shut for that pass and then again for the copy. Measured here by the
    call order, which is the property; a stopwatch would only say how fast this host is.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)

    order: list[str] = []
    real_stamps = sync.member_stamps
    real_retire = mobility._retire_local_runtime

    def stamps(*args: Any, **kwargs: Any) -> Any:
        order.append("hash")
        return real_stamps(*args, **kwargs)

    def retire(*args: Any, **kwargs: Any) -> Any:
        order.append("retire")
        return real_retire(*args, **kwargs)

    monkeypatch.setattr(sync, "member_stamps", stamps)
    monkeypatch.setattr(mobility, "_retire_local_runtime", retire)

    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    assert order[:2] == ["hash", "retire"], f"the source is stopped before it is digested: {order}"


def test_a_source_that_changes_while_it_is_being_prepared_is_refused(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """A write between the digest and the retire refuses, and nothing is changed.

    THE RACE THE REORDERING OPENS, closed rather than accepted: the digest is taken from a
    session whose runtime is still alive, so a turn can land in between. Nothing has been
    journaled and nothing deleted at that point, so the answer is a refusal the caller can
    simply repeat (``--wait`` already re-probes), and ``stamps_valid`` is what detects it —
    one ``stat`` per member against the sizes and ``mtime_ns`` the digest was taken from.
    The alternative, re-digesting inside the stopped window, doubles it for a race the
    caller can retry; accepting it would commit a copy of bytes the source has moved past.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    real_retire = mobility._retire_local_runtime

    def retire_after_a_write(*args: Any, **kwargs: Any) -> Any:
        # A turn landing in exactly the window the reordering introduced.
        with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "e77", "type": "user", "content": "a new turn"}) + "\n")
        return real_retire(*args, **kwargs)

    monkeypatch.setattr(mobility, "_retire_local_runtime", retire_after_a_write)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "busy", refused
    assert "changed while this device was preparing" in str(refused["message"]), refused
    # NOTHING WAS MOVED AND NOTHING WAS LOST, on either side.
    assert source.is_dir()
    assert (source / "transcript.jsonl").read_bytes() != before, "the test's own write landed"
    assert read_handoff_journal(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not sync.staging_dir(server_b.root, SESSION).exists()


def test_a_destination_entry_with_bytes_still_staged_is_never_settled(
    tmp_path: Path,
) -> None:
    """THE DISCRIMINATOR BETWEEN "IT ARRIVED" AND "IT IS STILL BEING RECEIVED".

    MEASURED AS A RED CELL (review round 4's cross-check with ``session_factory``).
    ``settle_promoted_handoff`` used to decide on a session directory plus a transcript,
    and ``session_factory``'s opener calls ``recover_stale_handoff`` and then the handoff
    guard — so a recovery that settled the entry on that evidence disarmed the refusal it
    runs immediately before: a destination entry whose VERIFIED COPY IS STILL STAGED (the
    owner may already have deleted its own bytes, so that copy is the only one left) was
    cleared instead of refused, and ``test_session_factory``'s
    ``test_the_factory_refuses_a_session_whose_move_is_in_flight`` went red on this branch
    while green at the base and at the tip. ``_promote`` is ONE ``os.replace``, so a
    promote that landed CONSUMED this device's staging directory; a staging directory that
    is still here means the handoff has not finished, whatever else is on disk.

    Both directions are asserted, because the fail-safe reading would be useless if it also
    blocked the ``p1c`` case it exists beside: with the bytes staged, nothing is cleared and
    the guard still refuses; once the promote has consumed them, the entry is settled and
    the conversation opens.
    """
    from local_operator.session.placement import (
        handoff_guard_refusal,
        handoff_in_flight,
        read_handoff_journal,
        write_handoff_entry,
    )

    root = tmp_path / "store"
    session_id = "abc123"
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps({"id": "e1", "type": "user", "content": "hello"}) + "\n", encoding="utf-8"
    )
    write_handoff_entry(
        root,
        session_id,
        {
            "role": "destination",
            "phase": "handing-off",
            "from_device": "d_" + "a" * 32,
            "from_name": "build-box",
            "to_device": "d_" + "b" * 32,
            "instance_id": "i_live",
            "at": 1.0,
        },
    )
    # THE UNPROMOTED DESTINATION: its verified bytes are here, not in ``sessions/``.
    staging = sync.staging_dir(root, session_id)
    staging.mkdir(parents=True)
    ready = staging / "ready.json"
    ready.write_text(
        json.dumps({"version": 1, "content_digest": "sha256:" + "a" * 64}), encoding="utf-8"
    )

    assert (
        mobility.settle_promoted_handoff(root, session_id) is False
    ), "a destination entry with its verified copy still staged is a handoff in flight"
    assert handoff_in_flight(root, session_id) is not None, "the entry must survive"
    assert (
        handoff_guard_refusal(root, session_id) != ""
    ), "the guard must still refuse: this is the refusal the settle call was disarming"
    assert ready.is_file(), "the staged bytes are untouched"

    # WHAT THE PROMOTE DOES: one ``os.replace``, which takes the whole directory with it.
    shutil.rmtree(staging)

    assert mobility.settle_promoted_handoff(root, session_id) is True
    assert read_handoff_journal(root).get(session_id) is None
    assert handoff_guard_refusal(root, session_id) == ""

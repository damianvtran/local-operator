"""A failed copy says it failed — it never kills the relay (live defect, 2026-10-21).

Measured on the operator's mesh: a recall of a session on ``cloud-node-1`` hit an
unreachable peer during the COPY phase. ``sync_from``'s ``ask`` raised
``Moved('unreachable')``, nothing in ``_destination_move`` / ``_recall`` / the relay's
``mobility._handle`` caught it, and the ``mesh-control-conn`` thread died with the
uncaught exception — the CLI then read a "wedged relay" (``relay_unavailable``) twice
while the source was left holding a ``prepared`` handoff journal that refused every
retry. The session itself survived on the source; only the report and the retry path
were broken.

Each cell pins one half of the repair:

* the copy-phase ``Moved`` is rolled back on the destination — including a
  destination handoff entry that already exists — and reported as a refusal naming
  the destination and the source holder
  (``test_a_copy_phase_link_failure_refuses_and_sweeps_the_destination``);
* the move handler turns a ``Moved`` from ``_recall``/``_offload`` into a refusal
  instead of dropping the control connection (``test_the_move_handler_...``);
* after such a failure the same destination's retry is refused ``in_progress``
  naming the source, and only a source relay restart (a fresh reconcile instance)
  clears the prepared entry before a retry prepares
  (``test_a_retry_after_a_failed_copy_...``);
* a prepared entry for another destination refuses the same way
  (``test_a_prepared_handoff_for_another_destination_...``);
* a ``--keep`` failure says the copy was discarded without claiming the source
  paused (U8), and identical device names get role qualifiers (m1).

The rig is ``test_mobility``'s two real relays, driven the way the CLI drives them,
for the same reason that module gives: a second implementation of the move would not
be the thing under test.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator.network import mobility, sync
from local_operator.session.placement import (
    HANDOFF_PHASE_HANDING_OFF,
    HANDOFF_PHASE_PREPARED,
    read_handoff_journal,
    write_handoff_entry,
)
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _move,
    _owned_session,
    _transcript,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """Two relays with BOTH ends able to serve a control socket (as ``test_mobility``)."""
    value: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


class _Link:
    """The one attribute ``_source_prepare`` reads from a peer link."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id


def _prepare_frame(to_device: str) -> dict[str, Any]:
    return {
        "op": "net_session_move",
        "action": "prepare",
        "session_id": SESSION,
        "to_device": to_device,
        "mode": "move",
        "have": {},
    }


def _lose_the_second_fetch(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], Any]:
    """Make the destination's copy lose its link on the SECOND file's first chunk.

    The plan and the transcript's chunks are real, so the staging directory holds
    partial bytes when the failure lands — the shape the live defect had. Returns
    the counter for the cell's assertion and the real ``ask`` so a retry can restore
    the transport.
    """
    real_ask = mobility.LinkTransport.ask
    fetches = {"n": 0}

    def losing_copy_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("op") or "") == "net_sync" and str(frame.get("phase") or "") == "fetch":
            fetches["n"] += 1
            if fetches["n"] == 2:
                # NOT DELEGATED: the source stopped answering between chunks, after
                # the transcript had been written into the staging directory.
                raise mobility.Moved(
                    "unreachable", "cloud-node-1 stopped answering during the copy"
                )
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", losing_copy_ask)
    return fetches, real_ask


# ---------------------------------------------------------------------------
# 1 — the copy-phase failure is a refusal and the destination is swept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", [HANDOFF_PHASE_PREPARED, HANDOFF_PHASE_HANDING_OFF])
def test_a_copy_phase_link_failure_refuses_and_sweeps_the_destination(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """The live defect, driven through the real transport.

    The failure is injected at the SECOND ``net_sync`` ``fetch`` — the transcript's
    own chunks have already been written into the staging directory by then, so this
    is a partial copy being rolled back, not a failure before any byte moved. The
    first ``net_sync`` (the plan) and the first file's chunks are the real calls.

    A destination handoff entry is planted FIRST, at BOTH phases the destination
    can hold while a copy is in flight (R1-2, m2): ``prepared`` is what the invite
    route pre-writes before any byte moves, ``handing-off`` is the post-copy shape.
    The live instance's entry is skipped by reconcile, so it survives to the
    rollback, and the rollback has to clear it rather than leave a claim on an id
    nobody committed.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)
    write_handoff_entry(
        server_b.root,
        SESSION,
        {
            "role": "destination",
            "phase": phase,
            "from_device": server_a.identity.device_id,
            "instance_id": str(server_b.instance_id),
        },
    )
    assert read_handoff_journal(server_b.root)[SESSION]["phase"] == phase

    fetches, _ = _lose_the_second_fetch(monkeypatch)

    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert fetches["n"] == 2, fetches
    # A REFUSAL, NOT AN ESCAPE: the code, the phase and the "nothing changed" flag
    # are what a front end renders.
    assert failed["ok"] is False, failed
    assert failed["code"] == "unreachable", failed
    assert failed["changed"] is False, failed
    assert failed["phase_reached"] == "prepared", failed
    # THE SENTENCE, pinned exactly: the cause, WHO discarded what, the holder, the
    # paused state and ONE action. R1-2's actor confusion is why the destination is
    # named by label rather than "this device".
    message = str(failed["message"])
    dest = str(server_b._own_label())  # noqa: SLF001
    holder = str(server_a._own_label())  # noqa: SLF001
    assert message == (
        f"cloud-node-1 stopped answering during the copy. {dest} discarded its "
        f"partial copy; {holder} still has the conversation, paused for this move. "
        f"On {holder}, restart the relay (`lop network restart`), then run the move "
        "again."
    ), message
    assert "this device" not in message, message
    assert "nothing was lost" not in message, message
    assert "nothing was changed" not in message, message
    # THE DESTINATION SWEPT EVERYTHING IT STAGED AND RELEASED ITS OWN CLAIM, so a
    # retry starts clean.
    assert not staging.exists(), "the partial copy was left staged"
    assert read_handoff_journal(server_b.root) == {}, "the destination kept a journal entry"
    assert not (server_b.root / "sessions" / SESSION).exists()
    # THE SOURCE STILL HOLDS THE CONVERSATION, byte for byte, with its journal at the
    # phase the failure left it in (the destination could not reach it to clear it).
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    source_entry = read_handoff_journal(server_a.root)[SESSION]
    assert source_entry["role"] == "source", source_entry
    assert source_entry["phase"] == "prepared", source_entry


def test_a_keep_copy_failure_leaves_the_source_running(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U8: a copy leaves the source running and journals nothing.

    So a copy failure may not claim the source "paused" or send the reader to
    restart a relay that has nothing to unpick — the remedy is just to copy again.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    retired: list[str] = []
    real_retire = mobility._retire_local_runtime

    def watching_retire(root: Any, session_id: str, *, deadline_s: float = 0) -> Any:
        retired.append(session_id)
        return real_retire(root, session_id, deadline_s=deadline_s)

    monkeypatch.setattr(mobility, "_retire_local_runtime", watching_retire)
    fetches, _ = _lose_the_second_fetch(monkeypatch)

    failed = _move(server_b, SESSION, keep=True, monkeypatch=monkeypatch)

    assert fetches["n"] == 2, fetches
    assert failed["ok"] is False and failed["code"] == "unreachable", failed
    assert failed["changed"] is False, failed
    dest = str(server_b._own_label())  # noqa: SLF001
    holder = str(server_a._own_label())  # noqa: SLF001
    message = str(failed["message"])
    assert message == (
        f"cloud-node-1 stopped answering during the copy. {dest} discarded its "
        f"partial copy; {holder} still has the conversation and was not paused "
        "(a copy leaves it running). Run the copy again."
    ), message
    assert "restart" not in message, message
    # THE SOURCE ENGINE STILL RUNS: ``--keep`` never retires it, and no journal
    # entry was written for a copy (the launch guard reads that file).
    assert retired == [], retired
    assert read_handoff_journal(server_a.root) == {}
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before


def test_the_copy_failure_sentence_disambiguates_identical_names(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1: member names are not unique, so a shared name gets a role qualifier.

    Both ends carry the same label in this cell; without the qualifier the sentence
    reads as one device talking about itself, which is the state U6 exists to stop.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    real_resolve = mobility.resolve_remote_owner
    same_name = str(server_b._own_label())  # noqa: SLF001

    def resolving_to_the_same_name(server: Any, session_id: str) -> tuple[str, str]:
        device_id, _name = real_resolve(server, session_id)
        return device_id, same_name

    monkeypatch.setattr(mobility, "resolve_remote_owner", resolving_to_the_same_name)
    _fetches, _ = _lose_the_second_fetch(monkeypatch)

    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert failed["ok"] is False and failed["code"] == "unreachable", failed
    message = str(failed["message"])
    dest = f"{same_name} (this move's destination)"
    holder = f"{same_name} (the source)"
    assert message == (
        f"cloud-node-1 stopped answering during the copy. {dest} discarded its "
        f"partial copy; {holder} still has the conversation, paused for this move. "
        f"On {holder}, restart the relay (`lop network restart`), then run the move "
        "again."
    ), message


# ---------------------------------------------------------------------------
# 2 — the handler's last arm: no movement failure may kill the control connection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "to"),
    [("recall", "local"), ("offload", "peer")],
)
def test_the_move_handler_turns_a_peer_refusal_into_a_refusal(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    to: str,
) -> None:
    """``_handle`` returned a refusal instead of raising, for BOTH verbs.

    ``_recall`` and ``_offload`` already catch most peer answers; this arm exists for
    the one that got past them in the field. It is driven at the handler because the
    defect was the handler letting the exception unwind the relay's thread.
    """
    server_a, server_b, _host, _port = pair
    target = "_recall" if action == "recall" else "_offload"
    called: list[str] = []

    def raising(*args: Any, **kwargs: Any) -> dict[str, Any]:
        called.append(action)
        raise mobility.Moved("unreachable", f"the peer link died during the {action}")

    monkeypatch.setattr(mobility, target, raising)
    handler = mobility.local_move_handler(server_a)
    resolved = "local" if to == "local" else server_b.identity.device_id

    result = handler({"action": action, "session_id": SESSION, "to": resolved})

    assert called == [action], "the handler never reached the patched verb"
    assert isinstance(result, dict), result
    assert result["ok"] is False, result
    assert result["code"] == "unreachable", result
    assert result["changed"] is False, result
    assert result["message"] == f"the peer link died during the {action}", result


# ---------------------------------------------------------------------------
# 3 — a retry is refused until the SOURCE relay restarts
# ---------------------------------------------------------------------------


def test_a_retry_after_a_failed_copy_is_refused_until_the_source_relay_restarts(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1: ``prepared`` is NOT a licence for the same destination to re-prepare.

    ``prepared`` is precisely the phase in which the SAME destination's copy may
    still be live: the destination writes its own role entry only after the copy,
    so two prepares from one device can share the staging directory, and clearing
    on the second lets the loser's rollback delete the winner's verified copy after
    the source committed — the only copy. So the retry is refused with the source's
    sentence, and only a source relay restart (a fresh reconcile instance, which
    makes the entry stale) clears it before a retry prepares.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    fetches, real_ask = _lose_the_second_fetch(monkeypatch)

    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert failed["ok"] is False and failed["code"] == "unreachable", failed
    assert fetches["n"] == 2, fetches
    monkeypatch.setattr(mobility.LinkTransport, "ask", real_ask)

    # THE SAME DESTINATION ASKING AGAIN IS REFUSED, and the sentence names the
    # SOURCE whose relay has to be restarted — not the destination that read it.
    retried = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert retried["ok"] is False, retried
    assert retried["code"] == "in_progress", retried
    holder = str(server_a._own_label())  # noqa: SLF001
    message = str(retried["message"])
    assert f"already in progress on {holder}" in message, message
    assert (
        f"so nothing was changed. On {holder}, restart the relay "
        "(`lop network restart`), then try the move again."
    ) in message, message
    assert "this device" not in message, message
    assert "other device" not in message, message
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before
    assert read_handoff_journal(server_a.root)[SESSION]["phase"] == "prepared"

    # THE SOURCE RELAY RESTART: a fresh instance makes the live writer stale, which
    # is the one recovery reconcile applies to a prepared entry.
    report = mobility.reconcile(server_a.root, server=server_a, own_instance="restarted-a")
    assert [row["action"] for row in report] == ["rolled_back"], report
    assert read_handoff_journal(server_a.root) == {}

    retried = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert retried["ok"] is True, retried
    assert retried["state"] == "completed", retried
    assert _transcript(server_b.root, SESSION) == before
    assert not source.exists()
    assert read_handoff_journal(server_a.root) == {}


def test_a_prepared_handoff_for_another_destination_still_refuses_and_names_the_source(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different device asking is a SECOND mover: ``in_progress``, and the
    sentence names the device whose relay holds the handoff.

    The reader is the DESTINATION, so "restart this device's relay" pointed at the
    wrong machine: restarting it cleared nothing on the source. The journal is
    device-local, so the holder is this relay's own device.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    destination = server_b.identity.device_id

    first = mobility._source_prepare(
        server_a, cast(Any, _Link(destination)), _prepare_frame(destination)
    )  # noqa: SLF001
    assert first["result"] == "prepared", first
    journal_before = read_handoff_journal(server_a.root)[SESSION]

    refused = mobility._source_prepare(  # noqa: SLF001
        server_a, cast(Any, _Link("device-c")), _prepare_frame("device-c")
    )

    assert refused["result"] == "refused", refused
    assert refused["code"] == "in_progress", refused
    message = str(refused["message"])
    assert str(server_a._own_label()) in message, message  # noqa: SLF001
    assert "this device" not in message, message
    assert "lop network restart" in message, message
    # NOTHING CHANGED: the entry still names the first destination, and the session
    # is still on the source.
    assert read_handoff_journal(server_a.root)[SESSION] == journal_before
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before


def test_a_handing_off_entry_is_never_cleared_by_a_retry(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ``prepared`` rolls back freely; ``handing-off`` is monotone.

    Once the destination has answered ``ready`` the source's commit is the only
    outcome, so a same-destination ask must keep today's refusal rather than clear
    an entry whose bytes may already be owned.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    destination = server_b.identity.device_id
    from local_operator.session.placement import (
        HANDOFF_PHASE_HANDING_OFF,
        write_handoff_entry,
    )

    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": HANDOFF_PHASE_HANDING_OFF,
            "to_device": destination,
            "mode": "move",
            "instance_id": str(server_a.instance_id),
        },
    )

    refused = mobility._source_prepare(  # noqa: SLF001
        server_a, cast(Any, _Link(destination)), _prepare_frame(destination)
    )

    assert refused["result"] == "refused", refused
    assert refused["code"] == "in_progress", refused
    assert str(server_a._own_label()) in str(refused["message"])  # noqa: SLF001
    still = read_handoff_journal(server_a.root)[SESSION]
    assert still["phase"] == HANDOFF_PHASE_HANDING_OFF, still
    assert still["to_device"] == destination, still

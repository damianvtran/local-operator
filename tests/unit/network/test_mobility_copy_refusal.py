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

* the copy-phase ``Moved`` is rolled back on the destination and reported as a
  refusal that says the source still holds the conversation, and the same
  destination's retry then lands (``test_a_copy_phase_link_failure_...``);
* the move handler turns a ``Moved`` from ``_recall``/``_offload`` into a refusal
  instead of dropping the control connection (``test_the_move_handler_...``);
* a ``prepared`` entry left by a failed copy is re-prepareable by the SAME
  destination (a prepared handoff never committed), while a different destination
  still gets ``in_progress`` — naming the source device, not "this device"
  (``test_a_prepared_handoff_from_the_same_destination_...`` /
  ``test_a_prepared_handoff_for_another_destination_...``).

The rig is ``test_mobility``'s two real relays, driven the way the CLI drives them,
for the same reason that module gives: a second implementation of the move would not
be the thing under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import mobility, sync
from local_operator.session.placement import read_handoff_journal
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


# ---------------------------------------------------------------------------
# 1 — the copy-phase failure is a refusal, the destination is swept, a retry lands
# ---------------------------------------------------------------------------


def test_a_copy_phase_link_failure_refuses_and_sweeps_the_destination(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live defect, driven through the real transport.

    The failure is injected at the SECOND ``net_sync`` ``fetch`` — the transcript's
    own chunks have already been written into the staging directory by then, so this
    is a partial copy being rolled back, not a failure before any byte moved. The
    first ``net_sync`` (the plan) and the first file's chunks are the real calls.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    staging = sync.staging_dir(server_b.root, SESSION)

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

    failed = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert fetches["n"] == 2, fetches
    # A REFUSAL, NOT AN ESCAPE: the code, the phase and the "nothing changed" flag
    # are what a front end renders.
    assert failed["ok"] is False, failed
    assert failed["code"] == "unreachable", failed
    assert failed["changed"] is False, failed
    assert failed["phase_reached"] == "prepared", failed
    # THE SENTENCE: the cause, the intact source, and the retry's one precondition.
    message = str(failed["message"])
    assert "cloud-node-1 stopped answering during the copy" in message, message
    assert "still holds the conversation" in message, message
    assert "prepared handoff" in message, message
    assert "lop network restart" in message, message
    # THE DESTINATION SWEPT EVERYTHING IT STAGED, so a retry starts clean.
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

    # THE RETRY IS THE OPERATOR'S NEXT ACTION, and it must not need the source's
    # relay restarted: the same destination asking again clears the prepared entry
    # (cell 3) and the move completes.
    monkeypatch.setattr(mobility.LinkTransport, "ask", real_ask)
    retried = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert retried["ok"] is True, retried
    assert retried["state"] == "completed", retried
    assert _transcript(server_b.root, SESSION) == before
    assert not source.exists()
    assert read_handoff_journal(server_a.root) == {}
    assert read_handoff_journal(server_b.root) == {}


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
# 3 — a prepared handoff from the SAME destination is a retry, not a second mover
# ---------------------------------------------------------------------------


def test_a_prepared_handoff_from_the_same_destination_re_prepares(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepare, simulate the failed copy's leftover, prepare again: it proceeds.

    The entry a copy-phase link failure leaves is written by the source's LIVE relay
    instance, and the reconcile at the top of ``_source_prepare`` skips its own
    instance's entries — so before this rule the only way out was restarting the
    source's relay. ``prepared`` is the licence: the source stopped its runtime and
    wrote a plan, no byte was committed, and the SAME destination asking again is
    the retry of exactly that attempt.
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
    first_entry = read_handoff_journal(server_a.root)[SESSION]
    assert first_entry["phase"] == "prepared", first_entry
    assert first_entry["to_device"] == destination, first_entry

    second = mobility._source_prepare(
        server_a, cast(Any, _Link(destination)), _prepare_frame(destination)
    )  # noqa: SLF001

    assert second["result"] == "prepared", second
    second_entry = read_handoff_journal(server_a.root)[SESSION]
    assert second_entry["phase"] == "prepared", second_entry
    assert second_entry["to_device"] == destination, second_entry
    # RE-JOURNALED, not the same entry carried forward: a new attempt has a new
    # lease, so the destination's ready can commit against the plan it was given.
    assert second_entry["lease_epoch"] != first_entry["lease_epoch"], (
        first_entry,
        second_entry,
    )
    # THE PREPARE NEVER DELETES: the conversation is still on the source.
    assert source.is_dir()
    assert _transcript(server_a.root, SESSION) == before


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


def test_the_destination_holds_no_journal_entry_after_a_refused_copy(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollback also releases the destination's own claim on the id, which a
    retry would otherwise read as a handoff in progress HERE."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    real_ask = mobility.LinkTransport.ask

    def refusing_ask(self: Any, frame: dict[str, Any], *, timeout: float | None = None):
        if str(frame.get("op") or "") == "net_sync" and str(frame.get("phase") or "") == "fetch":
            raise mobility.Moved("unreachable", "the source vanished mid-copy")
        return real_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", refusing_ask)

    refused = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert refused["ok"] is False and refused["code"] == "unreachable", refused
    # The destination's journal named the id while it was receiving; nothing
    # committed, so the sweep must leave it empty.
    assert read_handoff_journal(server_b.root) == {}
    assert Path(server_b.root, "sessions", SESSION).exists() is False

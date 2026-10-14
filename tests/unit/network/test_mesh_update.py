"""S1 of the mesh rolling updates: the ``update`` capability and ONE peer moved.

One cell per row of the refusal map this slice freezes (RU §2's register + §3's
vocabulary), plus the positive cell — an idle member moves one version — and the
never-force headline: a synthetic busy session on the member answers ``busy``,
the build stamp is unchanged and the transcript is intact.

THE RIG IS THE MESH SUITE'S OWN LOOPBACK PATTERN (``tests/unit/network/
test_relay_e2e.py``'s ``devices`` fixture: two relays, two config roots, real
TCP): the MEMBER is the inviter's relay (``server_a``), the ORIGIN is the joiner
(``server_b``), and the grant under test is written into the MEMBER's own record
— its row for the origin — exactly as ``lop network member grant`` writes it.

TEETH (fail-first). Each cell that asserts a guard held also shows what happens
with the guard removed, where that is a one-line neutralisation: the busy cell
re-asks with ``busy_sessions`` neutralised and the install recorder then FIRES;
the ungranted cell re-asks after the grant and the ask then REACHES the handler.
A cell that cannot be shown red by removing its guard says so instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import authorizer as authorizer_mod
from local_operator.network import meshupdate, store, types, wire
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SessionRecord
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

REPO = Path(__file__).resolve().parents[3]

Devices = tuple[Any, Any, str, int]

MEMBER_SAYS_NO_GRANT = "does not hold the 'update' capability"


def _set_peer_endpoints(server: Any, record: Any, device_id: str, endpoints: list[str]) -> None:
    """Point ``server``'s record at a live address for ``device_id``.

    The ``devices`` fixture binds ``server_a`` but writes no endpoints into the
    joiner's record, and ``update_peer`` dials through ``_ensure_link`` — which
    reads the member's recorded endpoints. This is the same helper shape
    ``test_mcpdefs_link`` uses for its push direction.
    """
    with store.mutate(record.network_id, server.root) as copy:
        row = copy.member(device_id)
        assert row is not None
        row.endpoints = list(endpoints)
        store.save(copy, server.root)


def _rig(
    devices: Devices,  # noqa: F811 — the parameter shadows the fixture import (house pattern)
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any]:
    """Pair, point the ORIGIN at the member, and pin the origin's target.

    ``origin_target`` is pinned to a fixed version in every cell that asks the
    member something: the alternative is the running test process's own version,
    which is stable but makes a cell's arithmetic depend on the checkout's
    ``pyproject.toml``.
    """
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    _set_peer_endpoints(server_b, record, server_a.identity.device_id, [f"{host}:{port}"])
    monkeypatch.setattr(
        meshupdate, "origin_target", lambda: {"version": "0.99.0", "source_ref": ""}
    )
    return server_a, server_b, record


def _grant(server_a: Any, record: Any, requester: str) -> None:
    """The member grants ``update`` to the origin — its OWN row, its own decision."""
    reply = server_a.control_dispatch(
        "net_member_caps",
        {"req": 1, "network": record.network_id, "device_id": requester, "grant": ["update"]},
    )
    assert reply["op"] == "ack", reply
    assert reply["detail"]["added"] == ["update"], reply


def _record_installs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the install with a recorder; returns the list it appends to."""
    calls: list[str] = []

    def _install(version: str, *, runner: Any = None) -> str:
        calls.append(version)
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    return calls


def _concrete_member(monkeypatch: pytest.MonkeyPatch, *, current: str) -> dict[str, str]:
    """A member that is not an editable checkout and is on ``current``."""
    from local_operator.update import InstallKind

    state = {"current": current}
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: InstallKind.UV_TOOL)
    monkeypatch.setattr(meshupdate, "generation_layout_supported", lambda: True)
    monkeypatch.setattr(meshupdate, "current_member_version", lambda: state["current"])
    return state


def _member_audit_rows(server_a: Any, event: str) -> list[dict[str, Any]]:
    return [row for row in server_a.audit.tail(200) if str(row.get("event")) == event]


class _Outcome:
    """A stand-in for ``control.RefreshOutcome`` (the fields the pass reads)."""

    def __init__(self, method: str, session_id: str = "") -> None:
        self.method = method
        self.session_id = session_id
        self.pid = 0
        self.name = ""
        self.line = ""


def _stub_pass(
    monkeypatch: pytest.MonkeyPatch, *, outcomes: tuple[Any, ...] = ()
) -> dict[str, Any]:
    """Neutralise the pass tail: ``outcomes`` as given, the recorder for the roll.

    A cell that only needs the install path must not dial this host's real
    runtimes, spawn successors, or roll the operator's daemons — the pass's
    tail (sweep, re-engage, services roll) is neutralised here and each piece
    has its own cell below. Returns the roll recorder.
    """
    calls: dict[str, Any] = {"rolls": []}
    monkeypatch.setattr(
        meshupdate,
        "refresh_runtimes",
        lambda root, *, budget=None: (list(outcomes), ""),
    )
    monkeypatch.setattr(
        meshupdate,
        "reengage_displaced",
        lambda root, records, found, *, budget=None: (0, ""),
    )
    monkeypatch.setattr(meshupdate, "roll_services_after_reply", lambda: calls["rolls"].append(1))
    return calls


# ---------------------------------------------------------------------------
# Tables and registration
# ---------------------------------------------------------------------------


def test_the_capability_and_its_ops_are_in_the_closed_tables() -> None:
    """``update`` is a capability of its own, per member, and its ops are total.

    The words name the BOUND (RU §2), not just the act; no NON-ADMIN role
    carries it (read/drive gaining it would silently widen every existing member
    of those roles — the blanket ``admin`` role holds the full vocabulary for
    every capability by construction, which is why the exception is asserted
    below rather than denied); it is self-decided (the grant governs THIS
    device's own install); and the two op halves sit in the tables their
    totality rules demand.
    """
    assert "update" in types.CAPABILITIES
    assert "update" in types.GRANTABLE_CAPABILITIES
    assert "update" in types.SELF_DECIDED_SCOPES
    assert types.CAPABILITY_WORDS["update"] == (
        "install a newer build here when this peer asks, and only from this device's "
        "own update channel"
    )
    for role in ("read", "drive"):
        assert "update" not in types.capabilities_for_role(
            role
        ), "a non-admin role that carried `update` would widen every existing member"
    # ADMIN IS THE EXCEPTION, BY CONSTRUCTION (agent review round 1, m2):
    # ``ROLE_CAPABILITIES["admin"] = CAPABILITIES``, so every capability is in
    # it. Pinned so the claim in the docstring cannot rot in either direction.
    assert "update" in types.ROLE_CAPABILITIES["admin"]
    assert "net_update" in types.NET_OPS
    assert types.OP_CAPABILITY["net_update"] == "update"
    assert "peer_update" in types.LOCAL_OPS
    assert wire.MESH_UPDATE_V1 in wire.LINK_CAPABILITIES
    # The totality rule the chokepoint exists for, over every table at once.
    assert authorizer_mod.op_tables_are_total() == []


def test_the_totality_check_fails_by_name_when_the_update_row_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PROVE THE TEST CAN FAIL: drop the row and the guard names it."""
    monkeypatch.delitem(types.OP_CAPABILITY, "net_update")
    assert "net_update" in authorizer_mod.op_tables_are_total()


def test_both_ends_serve_the_op_and_it_holds_a_slow_slot(request: pytest.FixtureRequest) -> None:
    """Registered on both relays, and SLOW — an install may run inside the call."""
    both: Devices = request.getfixturevalue("devices")
    for server in both[:2]:
        assert "net_update" in server._handlers  # noqa: SLF001 — the slice's seat
        assert "peer_update" in server._local_slice_handlers  # noqa: SLF001
        assert server.slow_op_deadline("net_update") == meshupdate.UPDATE_OP_DEADLINE_S
    # The peer op is one a slice may register at all (the closed table's rule).
    from local_operator.network import relay

    assert "net_update" in relay.SLICE_PEER_OPS
    assert "peer_update" in relay.SLICE_LOCAL_OPS


# ---------------------------------------------------------------------------
# The refusals, one cell per row
# ---------------------------------------------------------------------------


def test_an_ungranted_origin_is_refused_by_the_members_own_chokepoint(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: no grant. The member's row for the origin decides; nothing is touched.

    First half: without the grant the authoriser refuses before the handler
    reads a field — no ``update_requested`` row exists, so nothing on the member
    was touched. Second half (TEETH): after the member grants ``update`` to the
    origin, THE SAME ASK reaches the handler — the capability, not anything
    else, was what refused it.
    """
    from local_operator.update import InstallKind

    server_a, server_b, record = _rig(devices, monkeypatch)
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: InstallKind.EDITABLE)
    calls = _record_installs(monkeypatch)

    first = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert first["ok"] is False
    assert first["code"] == "refused", first
    # The member's own sentence, verbatim: it names the device and the capability.
    assert MEMBER_SAYS_NO_GRANT in first["message"], first
    assert calls == []
    assert _member_audit_rows(server_a, "update_requested") == []

    _grant(server_a, record, server_b.identity.device_id)
    second = meshupdate.update_peer(server_b, server_a.identity.device_id)
    # The ask now reaches the handler, which refuses the editable member by
    # name — so the grant (and only the grant) opened the door.
    assert second["state"] == "refused", second
    assert second["code"] == "editable_install", second
    assert len(_member_audit_rows(server_a, "update_requested")) == 1


def test_a_peer_without_the_feature_string_is_never_asked(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: old peer. The origin's pre-check means no request is ever sent.

    TEETH: the same link WITH the feature string sends the request (that is the
    granted cell above); here the stripped link must produce zero member-side
    rows — a check that fires is only proven by the thing it prevented.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    link, reason = server_b.dial(
        record.network_id, host=f"{devices[2]}:{devices[3]}", epoch=record.epoch
    )
    assert link is not None, reason
    try:
        link.capabilities = frozenset(
            name for name in link.capabilities if name != wire.MESH_UPDATE_V1
        )
        detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    finally:
        link.close("test")
    assert detail["code"] == "predates_rolling_updates", detail
    assert _member_audit_rows(server_a, "update_requested") == []


def test_a_busy_member_answers_busy_and_nothing_is_touched(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: busy — the never-force headline, cell for cell.

    A synthetic live session with ``busy`` set is the member's own authority on
    its busyness (RU §4.3). The answer is ``busy``, the install recorder stays
    empty, and the session's record — the transcript's live half — is intact and
    still busy. TEETH: with ``busy_sessions`` neutralised the same ask attempts
    the install, so the busy gate is what held it.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    # The channel check runs BEFORE the idle gate (RU §3's order), so this cell
    # must get past it without touching the network to reach the busy answer.
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    calls = _record_installs(monkeypatch)
    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="daemon",
            session_id="bu5y-cell-0001",
            conversation_name="a deliberately busy session",
            cwd=str(server_a.root),
            model_label="test/mock",
            control_port=1,
            control_key="k" * 32,
            busy=True,
            started=True,
        ),
        server_a.root,
    )

    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "busy", detail
    assert "nothing has been touched" in detail["message"], detail
    assert calls == [], "a busy member must never be forced"
    rows = registry.scan(server_a.root, reap=False, check_zombie=False)
    live = [record_ for record_, state in rows if record_.session_id == "bu5y-cell-0001"]
    assert live and live[0].busy is True, rows
    assert live[0].conversation_name == "a deliberately busy session"

    # TEETH: remove the gate and the very same ask tries the install.
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert calls == ["0.99.0"], "the recorder did not fire with the gate removed"


def test_an_editable_member_refuses_the_standing_path(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: editable/source install — refused BY NAME, never attempted."""
    from local_operator.update import InstallKind

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: InstallKind.EDITABLE)
    calls = _record_installs(monkeypatch)

    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "refused", detail
    assert detail["code"] == "editable_install", detail
    assert "development tree" in detail["message"], detail
    assert calls == []


def test_a_target_not_newer_than_the_member_is_a_skip_not_an_install(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: downgrade. ``ahead_of_target`` skips; ``already_on_target`` no-ops.

    Two answers, one rule: the standing path never moves backwards, and a member
    already on the target costs nothing. Neither takes the update lock and
    neither touches the install seam.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _record_installs(monkeypatch)

    ahead = _concrete_member(monkeypatch, current="9.9.9")
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "ahead_of_target", detail
    assert "never moves backwards" in detail["message"], detail
    assert calls == []
    # m4: the skip IS a row — when it is a rule outcome, not the idempotent
    # no-op; the design's audit table lists ahead beside the other rule refusals.
    refused_rows = _member_audit_rows(server_a, "update_refused")
    assert len(refused_rows) == 1, refused_rows
    assert refused_rows[0]["detail"]["code"] == "ahead_of_target", refused_rows[0]

    ahead["current"] = "0.99.0"
    same = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert same["state"] == "already_on_target", same
    assert calls == []
    assert (
        len(_member_audit_rows(server_a, "update_refused")) == 1
    ), "already_on_target is the idempotent no-op: it writes no row"


def test_a_second_update_while_one_is_in_flight_is_refused_by_name(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: update already in flight — ``update_in_progress``, no queueing.

    The lock is held by the TEST process on the member's own root, which is
    exactly what an in-flight install looks like from the file lock's side
    (``flock`` conflicts between two open file descriptions, same process or
    not). Nothing is queued behind it and the install seam never fires.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    calls = _record_installs(monkeypatch)

    with meshupdate.update_lock(server_a.root, meshupdate.UpdateTarget(version="9.9.9")):
        detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "refused", detail
    assert detail["code"] == "update_in_progress", detail
    assert calls == []


def test_the_reply_vocabulary_is_the_one_the_design_freezes(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The reply shape every producer must agree on (RU §3's frame).

    One shape, every state: a reader can branch on ``state``/``code`` without
    reading a sentence, and ``sessions`` counts what moved and what was kept.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.99.0")  # already on target: no seams needed
    member_reply = meshupdate.member_execute(
        server_a,
        None,
        {"target": {"version": "0.99.0", "source_ref": ""}, "rollout": "ro_vocab"},
    )
    for key in ("state", "code", "reason", "method", "version", "sessions", "updated_at"):
        assert key in member_reply, (key, member_reply)
    assert member_reply["state"] == "already_on_target", member_reply
    assert member_reply["sessions"] == {"moved": 0, "kept": 0}
    assert isinstance(member_reply["version"], str)
    # The origin's payload is its own shape (ok/message/name for the CLI), and it
    # carries the member's state through by name.
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    for key in ("ok", "state", "code", "message", "method", "version", "sessions"):
        assert key in detail, (key, detail)
    assert detail["state"] == "already_on_target", detail
    assert detail["ok"] is True
    # THE TAIL IS A PASS-THROUGH (round 3, M2): the member's own receipt
    # sentence is reported verbatim — authored prose must survive the report,
    # and this pins that it does for the common case.
    assert detail["message"] == "this device already runs 0.99.0", detail


# ---------------------------------------------------------------------------
# The positive cell
# ---------------------------------------------------------------------------


def test_an_idle_member_moves_one_version(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """POSITIVE CELL: an idle, granted member installs the target it was given.

    The install seam is the slice's own (``meshupdate.install_build``) because
    the real one shells out to uv; everything else is the production path — the
    chokepoint, the grant, the idle probe, the lock, the version verification
    after the install, and the receipt. The verification's second read of
    ``current_member_version`` is why the seam updates the member's version too:
    an install that did NOT move the version must answer ``failed`` (that is the
    whole point of the check), so a cell that faked only the install would be
    pinning the wrong half.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    _stub_pass(monkeypatch)  # the tail dials real runtimes; its own cells cover it
    calls: list[str] = []

    def _install(version: str, *, runner: Any = None) -> str:
        calls.append(version)
        state["current"] = version  # the install MOVED the member's build
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    # Outside the lock left by the seam, nothing else touches it: proves the
    # receipt came from a settled install, not a queued ask.
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["ok"] is True, detail
    assert detail["state"] == "done", detail
    assert detail["version"] == "0.99.0", detail
    assert detail["method"] == meshupdate.METHOD_GENERATION, detail
    assert calls == ["0.99.0"]
    assert len(_member_audit_rows(server_a, "update_started")) == 1
    assert len(_member_audit_rows(server_a, "update_completed")) == 1


def test_an_install_that_does_not_move_the_version_answers_failed(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The verification half of the positive cell, on its own.

    A seam that returns success without moving the member's build must produce
    ``failed`` — otherwise "done" would be a claim about the installer's exit
    code rather than about the build this device now runs.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    calls = _record_installs(monkeypatch)  # records but never moves the version

    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["ok"] is False, detail
    assert detail["state"] == "failed", detail
    assert calls == ["0.99.0"]


# ---------------------------------------------------------------------------
# The member pass (S2): wait → move → roll → re-engage, and its disclosures
# ---------------------------------------------------------------------------


def test_the_pass_waits_for_the_turn_then_updates_and_the_transcript_survives(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """S2's headline: the wait, the move, and the never-force artifact.

    The member is busy: the ask answers ``busy`` and touches nothing. The turn
    ends and the SAME ask path re-asks (a fresh probe, the origin's cadence) —
    the pass installs and answers ``done``, the sweep's outcome rides the
    receipt, and the session's own files are BYTE-IDENTICAL across both asks:
    the drain waited for the turn; it never cut one.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    calls: list[str] = []

    def _install(version: str, *, runner: Any = None) -> str:
        calls.append(version)
        state["current"] = version  # the install MOVED the member's build
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    sid = "w4it-cell-0001"
    pass_calls = _stub_pass(monkeypatch, outcomes=(_Outcome("moved", sid),))
    transcript = server_a.root / "sessions" / sid / "transcript.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"role": "user", "text": "a long turn"}\n', encoding="utf-8")
    before = transcript.read_bytes()
    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="daemon",
            session_id=sid,
            conversation_name="the long turn",
            cwd=str(server_a.root),
            model_label="test/mock",
            control_port=1,
            control_key="k" * 32,
            busy=True,
            started=True,
        ),
        server_a.root,
    )

    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "busy", detail
    assert calls == [], "the busy ask must not install"
    assert transcript.read_bytes() == before, "the busy ask touched the session's files"

    # THE TURN ENDS (the record's own busy bit clears); the origin re-asks.
    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="daemon",
            session_id=sid,
            conversation_name="the long turn",
            cwd=str(server_a.root),
            model_label="test/mock",
            control_port=1,
            control_key="k" * 32,
            busy=False,
            started=True,
        ),
        server_a.root,
    )
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert calls == ["0.99.0"]
    assert detail["sessions"] == {"moved": 1, "kept": 0}, detail
    assert "1 session(s) retired for the new build" in detail["message"], detail
    assert transcript.read_bytes() == before, "the transcript must survive byte for byte"
    rows = registry.scan(server_a.root, reap=False, check_zombie=False)
    live = [record_ for record_, _state in rows if record_.session_id == sid]
    assert live and live[0].busy is False, rows
    assert pass_calls["rolls"] == [1], "the services roll is scheduled only after the reply"


def test_the_classic_install_is_disclosed_when_the_generation_layout_cannot_run(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """§8.1's fallback: a uv-tool member without the layout takes the in-place arm, disclosed.

    The argv is the classic pinned reinstall; the receipt's ``method`` says
    ``uv-tool-classic``; the attestation pair ``perform_upgrade`` stages for its
    own in-place branch is staged here too (and NOT withdrawn on success).
    """
    from local_operator import update as update_mod

    server_a, server_b, record = _rig(devices, monkeypatch)
    state = {"current": "0.0.1"}
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: update_mod.InstallKind.UV_TOOL)
    monkeypatch.setattr(meshupdate, "generation_layout_supported", lambda: False)
    monkeypatch.setattr(meshupdate, "current_member_version", lambda: state["current"])
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    _stub_pass(monkeypatch)
    stubs: dict[str, list[Any]] = {"attested": [], "withdrawn": []}
    monkeypatch.setattr(
        update_mod, "note_doomed_runtimes", lambda root, **kw: stubs["attested"].append(root) or []
    )
    monkeypatch.setattr(
        update_mod,
        "withdraw_involuntary_stops",
        lambda attested, **kw: stubs["withdrawn"].append(1),
    )
    monkeypatch.setattr(update_mod, "write_source_marker", lambda root, **kw: True)
    seen: list[list[str]] = []

    def _runner(argv: list[str], env: dict[str, str]) -> int:
        seen.append(list(argv))
        state["current"] = "0.99.0"
        return 0

    reply = meshupdate.member_execute(
        server_a,
        None,
        {"target": {"version": "0.99.0", "source_ref": ""}},
        runner=_runner,
    )
    assert reply["state"] == "done", reply
    assert reply["method"] == "uv-tool-classic", reply
    assert seen == [
        [
            "uv",
            "tool",
            "install",
            "--force",
            "--refresh-package",
            "local-operator",
            "local-operator==0.99.0",
        ]
    ], seen
    assert stubs["attested"] and not stubs["withdrawn"], stubs


def test_a_non_uv_tool_member_without_the_layout_is_refused(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The carve-out is exactly the uv-tool member: other shapes keep S1's refusal."""
    from local_operator import update as update_mod

    server_a, server_b, record = _rig(devices, monkeypatch)
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: update_mod.InstallKind.PIPX)
    monkeypatch.setattr(meshupdate, "generation_layout_supported", lambda: False)
    monkeypatch.setattr(meshupdate, "current_member_version", lambda: "0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    reply = meshupdate.member_execute(
        server_a, None, {"target": {"version": "0.99.0", "source_ref": ""}}
    )
    assert reply["state"] == "refused", reply
    assert reply["code"] == "editable_install", reply


def test_the_generation_install_prunes_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """M1: the direct generation call runs the retention tail, best-effort.

    ``install_into_generation``'s retention is the CALLER'S (its own docstring),
    every other generation install prunes immediately after, and a rolled
    member is one that gets re-rolled — so this cell pins the caller calling it
    (``perform_upgrade``'s own shape) and that a failed prune does not fail a
    settled install. Sequenced with #2181.
    """
    from local_operator import update as update_mod

    log: list[Any] = []
    monkeypatch.setattr(meshupdate, "generation_layout_supported", lambda: True)
    monkeypatch.setattr(
        update_mod,
        "install_into_generation",
        lambda source=None, version=None, runner=None: log.append(("install", version)),
    )
    monkeypatch.setattr(update_mod, "referenced_install_roots", lambda: "REF")
    monkeypatch.setattr(
        update_mod,
        "prune_generations",
        lambda *, referenced, actor: log.append(("prune", referenced, actor)) or "PLAN",
    )
    monkeypatch.setattr(update_mod, "prune_notice_lines", lambda plan: [f"notice:{plan}"])
    assert meshupdate.install_build("0.99.0") == meshupdate.METHOD_GENERATION
    assert log[0] == ("install", "0.99.0"), log
    assert log[1] == ("prune", "REF", update_mod.ACTOR_UPGRADE), log

    def _prune_boom(*, referenced: Any, actor: Any) -> Any:
        raise OSError("a generation is in use")

    monkeypatch.setattr(update_mod, "prune_generations", _prune_boom)
    # A failed prune is NOT a failed install: it logs and the pass proceeds.
    assert meshupdate.install_build("0.99.0") == meshupdate.METHOD_GENERATION


def test_a_failed_classic_install_is_labelled_with_the_shape_that_ran(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """m2/Q-1: the failed receipt and its audit row name the classic arm it ran."""
    from local_operator import update as update_mod

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    monkeypatch.setattr(meshupdate, "detect_install_kind", lambda: update_mod.InstallKind.UV_TOOL)
    monkeypatch.setattr(meshupdate, "generation_layout_supported", lambda: False)
    monkeypatch.setattr(meshupdate, "current_member_version", lambda: "0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(update_mod, "note_doomed_runtimes", lambda root, **kw: [])
    monkeypatch.setattr(update_mod, "withdraw_involuntary_stops", lambda attested, **kw: None)

    def _runner(argv: list[str], env: dict[str, str]) -> int:
        return 3

    reply = meshupdate.member_execute(
        server_a, None, {"target": {"version": "0.99.0", "source_ref": ""}}, runner=_runner
    )
    assert reply["state"] == "failed", reply
    assert reply["method"] == meshupdate.METHOD_CLASSIC, reply
    rows = _member_audit_rows(server_a, "update_failed")
    assert rows and rows[-1]["detail"]["method"] == meshupdate.METHOD_CLASSIC, rows


def test_a_failed_install_carries_the_installers_own_output(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Q-3: the failure tail carries uv's words, not just the updater's exit line.

    The generation route runs through ``_CapturedInstall`` when no test runner
    is injected; the classic arm's own capture is covered by its argv cell and
    the label cell above. Pinned at the shared owned-runner seam, never a real uv.
    """
    from local_operator import update as update_mod

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))

    from local_operator import install_transaction

    def _captured_failure(argv: Any, *, on_output: Any, **kwargs: Any) -> int:
        on_output("error: No space left on device (os error 28)")
        return 1

    monkeypatch.setattr(install_transaction, "run_installer", _captured_failure)

    def _fake_install(source: Any = None, version: str = "", runner: Any = None) -> None:
        assert runner is not None
        runner(["uv", "tool", "install"], {})  # the capture seam actually runs
        raise update_mod.UpdateError("installer exited 1")

    monkeypatch.setattr(update_mod, "install_into_generation", _fake_install)
    reply = meshupdate.member_execute(
        server_a, None, {"target": {"version": "0.99.0", "source_ref": ""}}
    )
    assert reply["state"] == "failed", reply
    assert "installer exited 1" in reply["reason"], reply
    assert "No space left on device" in reply["reason"], reply


def test_the_receipt_counts_kept_only_working_runtimes(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """m1: unreachable/unsupported are NOT kept — the note carries them."""
    outcomes = [_Outcome("moved", "s1"), _Outcome("unreachable", "s2"), _Outcome("busy", "s3")]
    assert meshupdate.count_pass_outcomes(outcomes) == (1, 1)

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))

    def _install(version: str, *, runner: Any = None) -> str:
        state["current"] = version
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    _stub_pass(monkeypatch, outcomes=tuple(outcomes))
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert detail["sessions"] == {"moved": 1, "kept": 1}, detail
    assert "1 not answerable" in detail["message"], detail


def test_the_sweep_passes_outcomes_through_and_degrades_to_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep that cannot run is a note on the receipt, never a failed pass.

    F1: the ask set is the sweep's OWN read and ``only_pids`` is ALWAYS passed
    — pinned here, because the dialled set must not exceed the sized set.
    """
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from local_operator.session.runtime import control

    outcomes = [_Outcome("moved"), _Outcome("busy")]
    seen: list[dict[str, Any]] = []
    reads: list[Any] = []

    async def _fake(*, timeout_s: float, own_pid: int, only_pids: Any, _root: Any) -> list[Any]:
        seen.append({"timeout_s": timeout_s, "only_pids": only_pids})
        return outcomes

    monkeypatch.setattr(control, "refresh_all", _fake)
    targets = [SimpleNamespace(pid=11), SimpleNamespace(pid=12)]
    monkeypatch.setattr(
        meshupdate, "live_session_targets", lambda root: reads.append(root) or targets
    )
    got, note = meshupdate.refresh_runtimes(_Path("/nonexistent"))
    assert got == outcomes and note == ""
    assert reads, "the sweep must read the live set it sizes on"
    # Both targets fit the budget — and they are still NAMED as the ask set, so
    # ``refresh_all`` cannot dial anything this call did not size for (F1).
    assert seen[0]["only_pids"] == frozenset({11, 12}), seen

    async def _boom(*, timeout_s: float, own_pid: int, only_pids: Any, _root: Any) -> list[Any]:
        raise OSError("registry unreadable")

    monkeypatch.setattr(control, "refresh_all", _boom)
    got, note = meshupdate.refresh_runtimes(_Path("/nonexistent"))
    assert got == [] and "could not run" in note, note


def test_the_sweep_returns_early_when_no_time_remains(monkeypatch: pytest.MonkeyPatch) -> None:
    """F3a: the zero-budget early return — observed, not stubbed.

    Deleting the guard in ``refresh_runtimes`` left the whole file green before
    this cell existed (the reviewer's mutation probe); it asserts the REAL
    function returns the degrade note WITHOUT reading the registry or dialling
    anything, so the guard's removal goes red — the path would then read the
    live set and dial at the floor.
    """
    from pathlib import Path as _Path

    from local_operator.session.runtime import control

    called: list[str] = []

    async def _boom(*args: Any, **kwargs: Any) -> list[Any]:
        called.append("refresh_all")
        raise AssertionError("refresh_all must not be called with no budget left")

    monkeypatch.setattr(control, "refresh_all", _boom)
    monkeypatch.setattr(
        meshupdate, "live_session_targets", lambda root: called.append("read") or []
    )
    got, note = meshupdate.refresh_runtimes(_Path("/x"), budget=0.0)
    assert got == [] and note == "no time remained for the runtime sweep", (got, note)
    assert called == [], called


def test_the_sweep_chunks_a_live_set_past_its_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """QA Q-2's arithmetic: the sequential total never exceeds the budget.

    ``REFRESH_MIN_PER_CALL_S`` is a floor, so ``2 s × live`` used to exceed the
    nominal 60 s past 30 sessions. The sweep now asks only the targets the
    budget can hold at the per-call size and names the rest in the note, so
    the sum is bounded for ANY live count.
    """
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from local_operator.session.runtime import control

    seen: list[dict[str, Any]] = []

    async def _fake(*, timeout_s: float, own_pid: int, only_pids: Any, _root: Any) -> list[Any]:
        seen.append({"timeout_s": timeout_s, "only_pids": only_pids})
        return []

    monkeypatch.setattr(control, "refresh_all", _fake)
    monkeypatch.setattr(
        meshupdate,
        "live_session_targets",
        lambda root: [SimpleNamespace(pid=i + 1) for i in range(200)],
    )
    got, note = meshupdate.refresh_runtimes(_Path("/x"), budget=60.0)
    assert got == []
    assert seen[0]["timeout_s"] == meshupdate.REFRESH_MIN_PER_CALL_S, seen
    # 60 s of budget at a 0.5 s floor = 120 dials; the other 80 are not asked.
    assert seen[0]["only_pids"] == frozenset(range(1, 121)), seen
    assert "the live set (200) exceeds the sweep budget; 80 not asked" in note, note


def test_a_failed_classic_install_carries_the_installers_own_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F3b: the classic arm's capture branch (no runner → subprocess), observed.

    Both other classic cells inject a runner — and the generation-route capture
    cell patches only its own route — so this branch shipped unexecuted; it
    patches the subprocess seam itself and asserts uv's own words ride the
    raised sentence, with the attestation pair withdrawn on the failure.
    """
    from local_operator import update as update_mod

    class _Proc:
        returncode = 3
        stdout = ""
        stderr = "error: Permission denied (os error 13)"

    monkeypatch.setattr(meshupdate.subprocess, "run", lambda *a, **kw: _Proc())
    stubs: dict[str, list[Any]] = {"attested": [], "withdrawn": [], "marker": []}
    monkeypatch.setattr(
        update_mod, "note_doomed_runtimes", lambda root, **kw: stubs["attested"].append(root) or []
    )

    def _withdraw(attested: Any, **kw: Any) -> None:
        stubs["withdrawn"].append(1)

    monkeypatch.setattr(update_mod, "withdraw_involuntary_stops", _withdraw)
    monkeypatch.setattr(
        update_mod, "write_source_marker", lambda root, **kw: stubs["marker"].append(1) or True
    )

    with pytest.raises(update_mod.UpdateError) as caught:
        meshupdate._install_classic("0.99.0", runner=None, update_mod=update_mod)
    message = str(caught.value)
    assert "installer exited 3" in message, message
    assert "Permission denied" in message, message
    assert stubs["withdrawn"] == [1] and stubs["marker"] == [], stubs


def test_the_reengage_total_is_capped_at_the_nominal_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2: the 120 s nominal is a CAP under a fat slow-op remainder.

    A fake clock advances 90 s per engage with a 500 s budget: with the cap,
    total = min(120, 500) so engages 1 (left 120) and 2 (left 30) run and 3 is
    skipped; the old ``max(0, budget)`` (total 500) would run all three — the
    cell goes red on a revert.
    """
    from pathlib import Path as _Path
    from types import SimpleNamespace

    records = [
        SimpleNamespace(session_id="s1"),
        SimpleNamespace(session_id="s2"),
        SimpleNamespace(session_id="s3"),
    ]
    outcomes = [
        _Outcome("moved", "s1"),
        _Outcome("unreachable", "s2"),
        _Outcome("draining", "s3"),
    ]
    clock = {"t": 1000.0}
    monkeypatch.setattr(meshupdate.time, "monotonic", lambda: clock["t"])

    def _engage(record: Any, *, root: Any, deadline_s: float) -> str:
        clock["t"] += 90.0
        return ""

    monkeypatch.setattr(meshupdate, "_engage_successor", _engage)
    engaged, note = meshupdate.reengage_displaced(_Path("/x"), records, outcomes, budget=500.0)
    assert engaged == 2, engaged
    assert "1 not re-engaged inside the budget" in note, note


def test_reengage_targets_only_the_displaced_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``busy``/``current`` keep a working runtime; retiring and gone sessions get a successor."""
    from pathlib import Path as _Path
    from types import SimpleNamespace

    records = [SimpleNamespace(session_id="s1"), SimpleNamespace(session_id="s3")]
    calls: list[str] = []

    def _engage(record: Any, *, root: Any, deadline_s: float) -> str:
        calls.append(record.session_id)
        return "no runtime accepted the engage" if record.session_id == "s3" else ""

    monkeypatch.setattr(meshupdate, "_engage_successor", _engage)
    outcomes = [
        _Outcome("moved", "s1"),
        _Outcome("busy", "s2"),
        _Outcome("unreachable", "s3"),
        _Outcome("current", "s4"),
    ]
    engaged, note = meshupdate.reengage_displaced(_Path("/x"), records, outcomes)
    assert calls == ["s1", "s3"], calls
    assert engaged == 1, engaged
    assert "1 session(s) could not be re-engaged" in note, note

    # A budget already spent skips every displacee, named, never engaged (M2).
    engaged, note = meshupdate.reengage_displaced(_Path("/x"), records, outcomes, budget=0.0)
    assert calls == ["s1", "s3"], calls
    assert engaged == 0, engaged
    assert "2 not re-engaged inside the budget" in note, note


def test_the_successor_engage_uses_the_warm_errand_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The engage is the relay's own speculative shape — start the runtime, deliver nothing."""
    from types import SimpleNamespace

    from local_operator.session.runtime import launch

    seen: list[tuple[Any, ...]] = []

    async def _engage(
        session_id: str, cwd: str, errand: Any, *, config_dir: Any, deadline_s: float
    ) -> None:
        seen.append((session_id, cwd, errand, config_dir, deadline_s))

    monkeypatch.setattr(launch, "engage_runtime", _engage)
    root = tmp_path / "member"
    (root / "sessions" / "s9").mkdir(parents=True)
    record = SimpleNamespace(session_id="s9", cwd="/tmp/x")
    assert (
        meshupdate._engage_successor(record, root=root, deadline_s=meshupdate.REENGAGE_DEADLINE_S)
        == ""
    )
    session_id, cwd, errand, config_dir, deadline = seen[0]
    assert session_id == "s9" and cwd == "/tmp/x"
    assert isinstance(errand, launch.WarmErrand), errand
    assert config_dir == root and deadline == meshupdate.REENGAGE_DEADLINE_S

    # An id with no directory on THIS device is a raced registry read, not a
    # session to spawn for — the relay's own pre-check (n3).
    ghost = SimpleNamespace(session_id="gone", cwd="/tmp/x")
    assert "does not hold a session gone" in meshupdate._engage_successor(
        ghost, root=root, deadline_s=5.0
    )

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("could not start a runtime for session s9: no credential")

    monkeypatch.setattr(launch, "engage_runtime", _boom)
    assert "no credential" in meshupdate._engage_successor(
        record, root=root, deadline_s=meshupdate.REENGAGE_DEADLINE_S
    )


def test_the_services_roll_is_scheduled_after_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage runs on a daemon thread: the call returns while the stage blocks.

    WHAT THIS PROVES: the roll is started FIRE-AND-FORGET — a synchronous stage
    (the shape the note's Q4 wording invited) would make the call below block
    for the stage's whole duration, so the elapse assert goes red. It does NOT
    prove the wire ordering of the ``done`` frame against the kickstart: the
    frame's write belongs to the caller, and the thread's startup is the
    margin (m3).
    """
    import threading
    import time

    from local_operator import update as update_mod

    entered = threading.Event()
    released = threading.Event()

    def _stage(**kw: Any) -> None:
        entered.set()
        released.wait(5.0)

    monkeypatch.setattr(update_mod, "refresh_service_daemons_after_upgrade", _stage)
    started = time.monotonic()
    meshupdate.roll_services_after_reply()
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"the roll blocked for {elapsed:.2f}s — a synchronous stage"
    assert entered.wait(5.0), "the roll thread did not start the stage"
    released.set()


def test_a_linux_member_reaches_the_relay_refresh_through_the_roll_hand_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drill's seam end to end, in-process: roll → stage → child → relay step.

    The member pass's post-receipt roll (``roll_services_after_reply``) hands off
    to ``update.refresh_service_daemons_after_upgrade``, and on a Linux member
    whose relay unit is installed that must reach the relay's own Linux refresh
    (``refresh_unit_if_stale`` — the stale-relay memo's slice, 2026-10-11).
    Before it, this chain ended in the A24 sentence: the install moved and the
    relay kept advertising the previous build (the drill's receipt vs peer-row
    divergence).

    The child is run IN-PROCESS (the real ``daemons_refresh_command`` executing
    the real platform table, its relay repair replaced by a recorder) so the cell
    exercises the hand-off without an interpreter spawn; the subprocess seam
    itself is pinned by the update cells.

    THE CELL WAITS FOR THE CHILD TO RETURN, not for the repair alone. The first
    revision waited on an event set by the repair — which fires MID-CHILD — and
    then read ``seen``, filled by ``_run`` only after ``daemons_refresh_command``
    returns; on a loaded CI worker the test thread won that race and read an
    empty dict (``test (3.12, 3)`` on run 38191435186, ``assert None == 0``).
    ``finished`` is set in ``_run``'s ``finally``, so a child that raised still
    wakes the waiter and the ``seen["error"]`` assertion reports it.
    """
    import io
    import threading
    from contextlib import redirect_stderr, redirect_stdout

    from local_operator import launchd
    from local_operator import update as update_mod
    from local_operator.network import relay

    finished = threading.Event()
    repairs: list[str] = []
    seen: dict[str, Any] = {}

    def _repair() -> Any:
        repairs.append("network relay")
        return launchd.PlistRefresh(name="network relay", kind="current")

    monkeypatch.setattr(relay, "refresh_unit_if_stale", _repair)
    monkeypatch.setattr(update_mod, "_DAEMONS_ARE_LAUNCHD_AGENTS", False)
    monkeypatch.setattr(update_mod, "_installed_daemon_plists", lambda: [])
    monkeypatch.setattr(update_mod, "_linux_relay_unit_installed", lambda: True)
    # The child's own identity guard: this venv is an editable install, which the
    # child would refuse before it reached any step (its subject is elsewhere).
    monkeypatch.setattr(update_mod, "_repair_refusal", lambda: None)

    def _run(argv: list[str], **kwargs: Any) -> Any:
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = update_mod.daemons_refresh_command()
            seen["argv"] = list(argv)
            seen["code"] = code
            return subprocess.CompletedProcess(
                argv, code, stdout=out.getvalue(), stderr=err.getvalue()
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the asserting thread
            seen["error"] = repr(exc)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(subprocess, "run", _run)

    meshupdate.roll_services_after_reply()

    assert finished.wait(10.0), f"the roll never ran the child to completion: {seen}"
    assert "error" not in seen, seen.get("error")
    assert repairs == ["network relay"]
    assert seen.get("code") == 0
    assert "--services-only" in seen.get("argv", [])


def test_the_roll_starts_only_after_the_install_sweep_and_reengage(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Execution order on the done path: no tail effect precedes the install.

    Discriminates the reorderings the receipt depends on (m3): a roll started
    before the install or between the tail's steps shows up here as an
    out-of-order log. The frame's wire ordering itself is the caller's, not
    this module's — see the services cell above.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    log: list[str] = []

    def _install(version: str, *, runner: Any = None) -> str:
        log.append("install")
        state["current"] = version
        return meshupdate.METHOD_GENERATION

    def _sweep(root: Any, *, budget: Any = None) -> tuple[list[Any], str]:
        log.append("sweep")
        return [], ""

    def _engage(root: Any, records: Any, found: Any, *, budget: Any = None) -> tuple[int, str]:
        log.append("reengage")
        return 0, ""

    monkeypatch.setattr(meshupdate, "install_build", _install)
    monkeypatch.setattr(meshupdate, "refresh_runtimes", _sweep)
    monkeypatch.setattr(meshupdate, "reengage_displaced", _engage)
    monkeypatch.setattr(meshupdate, "roll_services_after_reply", lambda: log.append("roll"))
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert log == ["install", "sweep", "reengage", "roll"], log


def test_the_tail_releases_the_shared_install_transaction(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Service convergence must not monopolize root while newer builds prepare.

    Candidate registration/promotion own the common root transaction; the mesh
    tail is convergence, not another install, and runs after release.
    """
    import fcntl

    from local_operator import update

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))

    def _install(version: str, *, runner: Any = None) -> str:
        state["current"] = version
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    lock_path = update.stable_root() / "install.lock"
    held: list[bool] = []

    def _sweep(root: Any, *, budget: Any = None) -> tuple[list[Any], str]:
        fd = open(lock_path, "a+")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(False)
        except OSError:
            held.append(True)
        finally:
            fd.close()
        return [], ""

    monkeypatch.setattr(meshupdate, "refresh_runtimes", _sweep)
    monkeypatch.setattr(
        meshupdate, "reengage_displaced", lambda root, records, found, *, budget=None: (0, "")
    )
    monkeypatch.setattr(meshupdate, "roll_services_after_reply", lambda: None)
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert held == [False], held


def test_the_tail_degrades_when_the_slow_budget_is_nearly_spent(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """M2: the tail is clamped by what the requester is still listening for.

    ``relay.slow_op_remaining_s()`` is None outside a slow op — the nominal
    budgets apply. Inside one with less left than :data:`TAIL_RESERVE_S`, the
    tail must degrade to notes: ``_dispatch_slow`` DROPS a reply that lands
    after the deadline, and half a receipt is worse than a whole one.
    """
    from local_operator.network import relay

    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))

    def _install(version: str, *, runner: Any = None) -> str:
        state["current"] = version
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)
    budgets: list[Any] = []

    def _sweep(root: Any, *, budget: Any = None) -> tuple[list[Any], str]:
        budgets.append(budget)
        return [], "no time remained for the runtime sweep"

    monkeypatch.setattr(meshupdate, "refresh_runtimes", _sweep)
    monkeypatch.setattr(
        meshupdate, "reengage_displaced", lambda root, records, found, *, budget=None: (0, "")
    )
    monkeypatch.setattr(meshupdate, "roll_services_after_reply", lambda: None)

    monkeypatch.setattr(relay, "slow_op_remaining_s", lambda: 10.0)  # < TAIL_RESERVE_S
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert budgets == [0.0], budgets
    assert "no time remained for the runtime sweep" in detail["message"], detail

    # With budget left, the clamp is the reserve-adjusted remainder capped by
    # the nominal sweep bound: 90 - 30 = 60.
    state["current"] = "0.0.1"
    monkeypatch.setattr(relay, "slow_op_remaining_s", lambda: 90.0)
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "done", detail
    assert budgets[-1] == 60.0, budgets


# ---------------------------------------------------------------------------
# The probes themselves
# ---------------------------------------------------------------------------


def test_an_unreadable_registry_refuses_rather_than_installing(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """A failed read is "cannot prove idle", not "idle" — never-force, not best-effort."""
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    calls = _record_installs(monkeypatch)

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("registry unreadable")

    monkeypatch.setattr(registry, "scan", _explode)
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "busy", detail
    assert "session registry" in detail["message"], detail
    assert calls == []


def test_the_origin_reports_the_members_own_words_and_never_guesses_the_grant() -> None:
    """The origin-side classifier, unit-level (no rig).

    A CODED refusal passes through by name; a CODELESS one (the authoriser's,
    whose guard is deliberately withheld) is reported as ``refused`` with the
    member's sentence, never guessed into ``no_grant`` — the open question this
    slice records rather than papers over. A coded ``not_authorised`` DOES map
    to the grant's absence, and its remedy names the member and the command that
    runs THERE.
    """
    assert meshupdate.classify_member_answer(None) == (
        "no_answer",
        "the member did not answer before the bound",
    )
    assert meshupdate.classify_member_answer(
        {"op": "error", "code": "update_in_progress", "message": "an update is already running"}
    ) == ("update_in_progress", "an update is already running")
    assert meshupdate.classify_member_answer(
        {"op": "error", "message": "device-b may not do that on this device"}
    ) == ("refused", "device-b may not do that on this device")
    assert meshupdate.classify_member_answer({"op": "ack", "detail": {"state": "busy"}}) == (
        "busy",
        "",
    )

    remedy = meshupdate.no_grant_remedy(
        label="cloud-node-1", network="damian-mesh", requester="d_" + "a" * 32
    )
    assert "cloud-node-1 has not granted `update` to this device" in remedy
    assert "lop network member grant damian-mesh d_" + "a" * 32 + " update" in remedy


def test_a_target_the_members_channel_cannot_serve_is_refused_by_name(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """ROW: ``target_not_published`` — refused before the idle gate, nothing touched.

    The member's channel POSITIVELY said the version is absent (see
    :func:`test_the_channel_check_refuses_only_on_a_confirmed_absence`), so the
    answer is a terminal refusal rather than an install attempt — and it is
    decided without consulting the session registry at all, which is RU §3's
    "the no-action and refusal outcomes need no idle".
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    calls = _record_installs(monkeypatch)
    monkeypatch.setattr(
        meshupdate,
        "check_published",
        lambda version: (False, "the member's channel does not serve this version"),
    )
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "refused", detail
    assert detail["code"] == "target_not_published", detail
    assert calls == []
    refused_rows = _member_audit_rows(server_a, "update_refused")
    assert refused_rows and refused_rows[-1]["detail"]["code"] == "target_not_published"


def test_the_channel_check_precedes_the_idle_gate(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """m6: the ORDER is load-bearing — a non-servable target refuses before busy.

    The member is busy AND the target cannot be served. Under the design's
    order (RU §3) the first ask gets the TERMINAL refusal and the busy probe is
    never consulted (``busy_calls == []``); with the check moved back after the
    idle gate the same ask answers ``busy`` and this cell goes red — the two
    asserts below are what make that revert fail (agent review round 2: the
    reorder left 18/18 green before this pin existed).

    CONTROL: with the target servable the same busy member answers ``busy``,
    so the cell pins the ORDER and not merely the refusal.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    calls = _record_installs(monkeypatch)
    busy_calls: list[int] = []

    def _busy(root: Any) -> tuple[int, str]:
        busy_calls.append(1)
        return 1, "1 live session (a deliberately busy session)"

    monkeypatch.setattr(meshupdate, "busy_sessions", _busy)
    monkeypatch.setattr(
        meshupdate,
        "check_published",
        lambda version: (False, "the member's channel does not serve this version"),
    )

    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "refused", detail
    assert detail["code"] == "target_not_published", detail
    assert busy_calls == [], "the idle gate must not run before a terminal refusal"
    assert calls == []

    # CONTROL: servable target, still busy → the busy answer, probe consulted.
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "busy", detail
    assert busy_calls == [1], busy_calls
    assert calls == []


def test_the_channel_check_refuses_only_on_a_confirmed_absence() -> None:
    """m3: the asymmetry that keeps the op from depending on a weaker read.

    A 404 refuses; a 5xx, an auth error and a transport failure all mean
    "attempt it" — the installer is the final authority and its own tail is what
    the failed receipt carries. Driven through the client seam, so no network.
    """

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class _Client:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def get(self, url: str, timeout: float) -> _Response:
            return _Response(self.status_code)

    class _Broken:
        def get(self, url: str, timeout: float) -> Any:
            raise OSError("no route to the index")

    assert meshupdate.check_published("9.9.9", client=_Client(404)) == (
        False,
        "the member's channel does not serve this version",
    )
    for status in (200, 403, 500, 503):
        assert meshupdate.check_published("9.9.9", client=_Client(status)) == (True, ""), status
    assert meshupdate.check_published("9.9.9", client=_Broken()) == (True, "")


def test_the_unreachable_sentence_carries_the_dial_seams_reason(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """m5 + Q-R3-1: the dial fact is said in words — and only when it exists.

    ``no_endpoint`` / ``not_a_member`` must not collapse into one vague sentence,
    and the token must not reach the line raw: the parenthetical goes through
    ``resume.peer_reason_words``, the mesh's one spelling for a reason on a human
    line. Q-R3-1: an EMPTY dial reason stays a MISSING parenthetical — the CI
    round made it unconditional, which would have rendered the gloss's
    reachability default ("it did not answer") onto a line that had no reason to
    say it. (The tail's read in the same function is the member's own prose and
    is passed through VERBATIM, declared in the guardian — round 3, M2; this
    cell is the dial half only.)
    """
    server_a, server_b, record = _rig(devices, monkeypatch)

    # NO reason: no parenthetical at all (Q-R3-1).
    monkeypatch.setattr(server_b, "_ensure_link_with_reason", lambda device_id, **kw: (None, ""))
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["code"] == "unreachable", detail
    assert "it did not answer" not in detail["message"], detail
    assert detail["message"].endswith("update"), detail

    # A reason: glossed, never the token.
    def _no_link(device_id: str, **kwargs: Any) -> tuple[None, str]:
        return None, "no_endpoint"

    monkeypatch.setattr(server_b, "_ensure_link_with_reason", _no_link)
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["code"] == "unreachable", detail
    assert "no address published for it" in detail["message"], detail
    assert "no_endpoint" not in detail["message"], "the token is the --json register"


def test_the_members_receipt_sentence_is_reported_verbatim(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """M2/Q-R3-2: the member's own reason is authored prose; the tail repeats it.

    The member's updater writes this sentence for a reader, and the design's
    contract is "report the member's own words" (RU §3) — so the origin's
    ``message`` hands it over UNTOUCHED. The shapes below are the ones the
    CI-round gloss mangled: the leading clause dropped (``uv tool install
    failed: disk full`` -> ``disk full``), a colon-tail without a space falling
    to a false "it did not answer" (``…: E2BIG``), the realistic ``UpdateError``
    text ``[Errno 13] Permission denied: '<path>'`` reading as silence, and a
    token-shaped reason reported as authored. The member composes each through
    its real failed path (the installer raises); only the installer seam is
    monkeypatched. The guardian's DECLARED read is the static half of this pin.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    shapes = (
        "uv tool install failed: disk full",
        "uv tool install failed: E2BIG",
        "[Errno 13] Permission denied: '/Users/x/.local/share/lop'",
        "connect_failed:TimeoutError",  # a token shape, reported as authored
    )

    for shape in shapes:

        def _fail(version: str, *, runner: Any = None, _shape: str = shape) -> str:
            raise RuntimeError(_shape)

        monkeypatch.setattr(meshupdate, "install_build", _fail)
        detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
        assert detail["state"] == "failed", (shape, detail)
        assert detail["message"] == shape, (shape, detail)


def test_a_missing_member_reason_stays_a_missing_message(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """n4: ``""``/absent member reason -> ``""``, never a default sentence.

    The emptiness guard's pass-through half: an ack whose ``reason`` is empty or
    absent must produce an empty ``message``. Driven through a fake link so the
    reply's detail is exact; nothing else about the ask is stubbed away.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)

    class _FakeLink:
        capabilities = frozenset({wire.MESH_UPDATE_V1})
        detail: dict[str, Any] = {"state": "already_on_target", "code": "", "reason": ""}

        def request(self, frame: Any, *, timeout: float) -> Any:
            return {"op": "ack", "detail": dict(self.detail)}

    monkeypatch.setattr(
        server_b, "_ensure_link_with_reason", lambda device_id, **kw: (_FakeLink(), "")
    )
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["message"] == "", detail

    class _NoReason(_FakeLink):
        detail = {"state": "already_on_target", "code": ""}

    monkeypatch.setattr(
        server_b, "_ensure_link_with_reason", lambda device_id, **kw: (_NoReason(), "")
    )
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["message"] == "", detail


_PROBE = """
import pathlib, sys, tempfile
sys.modules["fcntl"] = None  # any later `import fcntl` raises ImportError
from local_operator.network import meshupdate, types
try:
    with meshupdate.update_lock(
        pathlib.Path(tempfile.mkdtemp()), meshupdate.UpdateTarget(version="0.0.1")
    ):
        print("NO-REFUSAL")
except types.MeshRefusal as refusal:
    print("ok", refusal.code)
"""


def test_the_module_imports_and_refuses_by_name_without_fcntl() -> None:
    """B1/Q1: the POSIX import is function-local, and its absence refuses by name.

    A FRESH interpreter with ``sys.modules['fcntl'] = None`` (importing a module
    mapped to ``None`` raises ImportError), so a module-scope ``import fcntl``
    would fail the import itself — which ``_install_slices`` swallows, making the
    whole op silently absent on Windows. The repo's own subprocess-probe pattern
    (``test_structure``): pytest has already imported half the tree, so an
    in-process check could pass on a real regression.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "ok update_lock_unsupported", proc.stdout

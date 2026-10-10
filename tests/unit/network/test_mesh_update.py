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

NOT RUN YET: this file was drafted under a fleet disk hold (see the PR body);
"what I ran" there names exactly which commands have and have not executed.
"""

from __future__ import annotations

import os
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


def _rig(devices: Devices, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
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


# ---------------------------------------------------------------------------
# Tables and registration
# ---------------------------------------------------------------------------


def test_the_capability_and_its_ops_are_in_the_closed_tables() -> None:
    """``update`` is a capability of its own, per member, and its ops are total.

    The words name the BOUND (RU §2), not just the act; no role carries it (a
    role that did would silently widen every existing member); it is
    self-decided (the grant governs THIS device's own install); and the two op
    halves sit in the tables their totality rules demand.
    """
    assert "update" in types.CAPABILITIES
    assert "update" in types.GRANTABLE_CAPABILITIES
    assert "update" in types.SELF_DECIDED_SCOPES
    assert types.CAPABILITY_WORDS["update"] == (
        "install a newer build here when this peer asks, and only from this device's "
        "own update channel"
    )
    for role in ("read", "drive"):
        assert "update" not in types.capabilities_for_role(role), (
            "a role that carried `update` would widen every existing member"
        )
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
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROW: old peer. The origin's pre-check means no request is ever sent.

    TEETH: the same link WITH the feature string sends the request (that is the
    granted cell above); here the stripped link must produce zero member-side
    rows — a check that fires is only proven by the thing it prevented.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    link = server_b.dial(record.network_id, host=f"{devices[2]}:{devices[3]}", epoch=record.epoch)
    assert link is not None
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
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert calls == ["0.99.0"], "the recorder did not fire with the gate removed"


def test_an_editable_member_refuses_the_standing_path(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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

    ahead["current"] = "0.99.0"
    same = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert same["state"] == "already_on_target", same
    assert calls == []


def test_a_second_update_while_one_is_in_flight_is_refused_by_name(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
    calls = _record_installs(monkeypatch)

    with meshupdate.update_lock(server_a.root, meshupdate.UpdateTarget(version="9.9.9")):
        detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    assert detail["state"] == "refused", detail
    assert detail["code"] == "update_in_progress", detail
    assert calls == []


def test_the_reply_vocabulary_is_the_one_the_design_freezes(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reply shape every producer must agree on (RU §3's frame).

    One shape, every state: a reader can branch on ``state``/``code`` without
    reading a sentence, and ``sessions`` counts what moved and what was kept.
    """
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.99.0")  # already on target: no seams needed
    detail = meshupdate.update_peer(server_b, server_a.identity.device_id)
    for key in ("state", "code", "reason", "method", "version", "sessions"):
        assert key in detail, (key, detail)
    assert detail["sessions"] == {"moved": 0, "kept": 0}
    assert isinstance(detail["version"], str)
    assert detail["state"] in {
        "busy",
        "done",
        "already_on_target",
        "ahead_of_target",
        "refused",
        "failed",
    }


# ---------------------------------------------------------------------------
# The positive cell
# ---------------------------------------------------------------------------


def test_an_idle_member_moves_one_version(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
    devices: Devices, monkeypatch: pytest.MonkeyPatch
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
# The probes themselves
# ---------------------------------------------------------------------------


def test_an_unreadable_registry_refuses_rather_than_installing(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed read is "cannot prove idle", not "idle" — never-force, not best-effort."""
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    _concrete_member(monkeypatch, current="0.0.1")
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
    assert meshupdate.classify_member_answer(
        {"op": "ack", "detail": {"state": "busy"}}
    ) == ("busy", "")

    remedy = meshupdate.no_grant_remedy(
        label="cloud-node-1", network="damian-mesh", requester="d_" + "a" * 32
    )
    assert "cloud-node-1 has not granted `update` to this device" in remedy
    assert "lop network member grant damian-mesh d_" + "a" * 32 + " update" in remedy

"""Ask ONE peer to restart its own relay, then prove the process changed.

The origin never restarts a member directly. ``net_restart`` is authorised by the
member's self-decided ``update`` grant, the same lifecycle grant that already
allows a peer to disrupt this device's own build and that no read/drive role
carries. The member refuses while its registry has a busy session, naming that
work, and a second probe immediately before the supervisor call closes the
acceptance-to-restart race.

A response sent by the old process cannot itself prove a restart. The member
therefore answers ``accepted`` and waits for that reply's link to close before
calling only ``relay.service_action('restart')``. The origin closes the link,
reconnects, and asks the read-only ``evidence`` phase. Success requires a new
handshake instance plus a changed pid or a reset start time; acceptance, busy,
errors, silence and unconfirmed reconnects are distinct non-success receipts.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from local_operator.network import meshupdate, wire
from local_operator.session.runtime import restart_fence

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

# The control call must outlive reconnect confirmation; the peer op itself is
# quick because the supervisor runs only after the accepted reply's link closes.
RESTART_CONTROL_TIMEOUT_S = 25.0
RESTART_CONFIRM_WINDOW_S = 15.0
RESTART_REQUEST_TIMEOUT_S = 5.0
RESTART_EVIDENCE_TIMEOUT_S = 2.0
RESTART_REPLY_GRACE_S = 2.0
RESTART_ADMISSION_WAIT_S = 1.0
RESTART_POLL_S = 0.1

# A failed supervisor call or the second busy probe happens after ``accepted``;
# retain that outcome in the OLD process so its evidence phase can explain why
# no new instance will arrive. A real restart drops this state with the process,
# which is fine: the new handshake and fresh evidence are the success proof.
_OUTCOMES: weakref.WeakKeyDictionary[RelayServer, dict[str, str]] = weakref.WeakKeyDictionary()
_OUTCOMES_LOCK = threading.Lock()


def _put_outcome(server: RelayServer, state: str, reason: str = "", code: str = "") -> None:
    with _OUTCOMES_LOCK:
        _OUTCOMES[server] = {"state": state, "reason": reason, "code": code}


def _read_outcome(server: RelayServer) -> dict[str, str]:
    with _OUTCOMES_LOCK:
        return dict(_OUTCOMES.get(server) or {})


def _label(server: RelayServer, device_id: str) -> str:
    return server._member_name(device_id) or device_id  # noqa: SLF001


def _evidence(server: RelayServer) -> dict[str, Any]:
    """Process facts reported by the answering relay, never inferred by the origin."""
    return {
        "pid": os.getpid(),
        "instance_id": server.instance_id,
        "started_at": server.started_at,
        "version": server.build.get("version", ""),
        "uptime_s": round(max(0.0, time.time() - server.started_at), 1),
    }


def _ack(frame: Mapping[str, Any], detail: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": "ack", "req": frame.get("req"), "detail": dict(detail)}


def _member_busy(server: RelayServer) -> dict[str, Any] | None:
    count, detail = meshupdate.busy_sessions(server.root, include_names=True)
    if count == 0:
        return None
    code = "busy" if count > 0 else "busy_unknown"
    return {
        "state": "busy",
        "code": code,
        "reason": f"{detail}; restart waits for idle so the running work is not interrupted",
    }


def restart_after_reply(server: RelayServer, link: PeerLink) -> None:
    """Restart only this relay after the accepted response has left the link.

    There is no response-after-exit: a supervised restart kills the process that
    would send it. Waiting for the requester to close the accepted link avoids
    the fixed-timer race that could otherwise turn a successful restart into a
    lost receipt. The grace bound still lets recovery proceed if that close is
    lost; the origin reports an unconfirmed outcome rather than inventing one.
    """

    def _run() -> None:
        from local_operator.network import relay

        deadline = time.monotonic() + RESTART_REPLY_GRACE_S
        while link.alive and time.monotonic() < deadline:
            time.sleep(0.02)

        # A turn may have started while the accepted reply drained. The exclusive
        # cross-process fence makes the final idle read and supervisor call ONE
        # admission ordering: a turn either publishes busy first (and is refused)
        # or proceeds only after this restart decision has completed.
        try:
            with restart_fence.exclusive(server.root, timeout_s=RESTART_ADMISSION_WAIT_S):
                busy = _member_busy(server)
                if busy is not None:
                    _put_outcome(
                        server,
                        "busy",
                        str(busy["reason"]),
                        str(busy["code"]),
                    )
                    return
                result = relay.service_action("restart")
        except restart_fence.AdmissionConflict:
            busy = _member_busy(server)
            reason = (
                str(busy["reason"])
                if busy is not None
                else "a turn admission is still settling; restart waits for idle"
            )
            _put_outcome(server, "busy", reason, "busy")
            return
        except Exception as exc:  # noqa: BLE001 — the old process must report the failure
            _put_outcome(server, "failed", f"relay restart failed: {exc}", "restart_failed")
            return
        if not result.get("ok"):
            reason = str(result.get("error") or "the supervisor refused the restart")
            _put_outcome(server, "failed", f"relay restart failed: {reason}", "restart_failed")
        # On success the supervisor normally kills this process. If it returns
        # without doing so, leave ``accepted`` in place: only fresh process
        # evidence may complete the origin's receipt.

    threading.Thread(
        target=_run,
        name=f"mesh-peer-restart-{server.instance_id[:8]}",
        daemon=True,
    ).start()


def member_execute(server: RelayServer, link: PeerLink, frame: Mapping[str, Any]) -> dict[str, Any]:
    """The member side: answer evidence, refuse busy, or accept one own-relay restart."""
    phase = str(frame.get("phase") or "request")
    if phase == "evidence":
        return _ack(
            frame,
            {
                "state": "evidence",
                "evidence": _evidence(server),
                "outcome": _read_outcome(server),
            },
        )
    if phase != "request":
        return _ack(
            frame,
            {
                "state": "failed",
                "code": "bad_request",
                "reason": f"unknown peer-restart phase {phase!r}",
            },
        )
    try:
        busy = _member_busy(server)
        if busy is not None:
            return _ack(frame, busy)
        before = _evidence(server)
        _put_outcome(server, "accepted")
        restart_after_reply(server, link)
    except Exception as exc:  # noqa: BLE001 — a peer op answers failure; it must not drop the link
        return _ack(
            frame,
            {"state": "failed", "code": "restart_failed", "reason": f"restart failed: {exc}"},
        )
    return _ack(frame, {"state": "accepted", "evidence": before})


def _result(
    server: RelayServer,
    device_id: str,
    *,
    ok: bool,
    state: str,
    code: str,
    message: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "device_id": device_id,
        "name": _label(server, device_id),
        "state": state,
        "code": code,
        "message": message,
        **fields,
    }


def _no_answer(server: RelayServer, device_id: str, label: str) -> dict[str, Any]:
    return _result(
        server,
        device_id,
        ok=False,
        state="no_answer",
        code="no_answer",
        message=(
            f"{label} did not answer the peer-restart request; restart is unconfirmed "
            "and is not claimed"
        ),
    )


def _reply_error(server: RelayServer, device_id: str, reply: Mapping[str, Any]) -> dict[str, Any]:
    code = str(reply.get("code") or "refused")
    message = str(reply.get("message") or "the peer refused the restart")
    if code == "not_authorised":
        network = meshupdate._shared_network_name(server, device_id)  # noqa: SLF001
        message = meshupdate.no_grant_remedy(
            label=_label(server, device_id),
            network=network,
            requester=server.identity.device_id,
        )
    elif code == "unknown_op":
        message = (
            f"{_label(server, device_id)} does not recognize peer restart "
            "(unknown op); run `lop network restart` on that device, or update it first"
        )
    return _result(
        server,
        device_id,
        ok=False,
        state="refused",
        code=code,
        message=message,
    )


def _evidence_from_reply(
    reply: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    if reply is None or reply.get("op") == "error":
        return None
    detail = reply.get("detail")
    if not isinstance(detail, Mapping) or detail.get("state") != "evidence":
        return None
    evidence = detail.get("evidence")
    outcome = detail.get("outcome")
    if not isinstance(evidence, Mapping):
        return None
    return dict(evidence), dict(outcome) if isinstance(outcome, Mapping) else {}


def _confirmed(before: Mapping[str, Any], after: Mapping[str, Any], link_instance: str) -> bool:
    """A new handshake alone lacks process evidence; pid or uptime must reset too."""
    old_instance = str(before.get("instance_id") or "")
    new_instance = str(after.get("instance_id") or "")
    if not old_instance or not new_instance or new_instance == old_instance:
        return False
    if new_instance != link_instance:
        return False
    old_pid = before.get("pid")
    new_pid = after.get("pid")
    pid_changed = (
        isinstance(old_pid, int)
        and not isinstance(old_pid, bool)
        and old_pid > 0
        and isinstance(new_pid, int)
        and not isinstance(new_pid, bool)
        and new_pid > 0
        and old_pid != new_pid
    )
    old_started = before.get("started_at")
    new_started = after.get("started_at")
    start_reset = (
        isinstance(old_started, (int, float))
        and not isinstance(old_started, bool)
        and isinstance(new_started, (int, float))
        and not isinstance(new_started, bool)
        and new_started > old_started
    )
    return pid_changed or start_reset


def _success(
    server: RelayServer,
    device_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    label = _label(server, device_id)
    evidence = [f"instance {before.get('instance_id')} → {after.get('instance_id')}"]
    if before.get("pid") != after.get("pid"):
        evidence.insert(0, f"pid {before.get('pid')} → {after.get('pid')}")
    if after.get("started_at") != before.get("started_at"):
        evidence.insert(-1, f"uptime reset to {after.get('uptime_s', '?')}s")
    return _result(
        server,
        device_id,
        ok=True,
        state="restarted",
        code="restarted",
        message=f"{label} restarted its relay ({'; '.join(evidence)})",
        before=dict(before),
        after=dict(after),
    )


def _confirm_restart(
    server: RelayServer,
    device_id: str,
    before: Mapping[str, Any],
) -> dict[str, Any]:
    label = _label(server, device_id)
    deadline = time.monotonic() + RESTART_CONFIRM_WINDOW_S
    last_reason = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            suffix = f": {last_reason}" if last_reason else ""
            return _result(
                server,
                device_id,
                ok=False,
                state="unconfirmed",
                code="restart_unconfirmed",
                message=(
                    f"{label} accepted the restart, but a new relay instance with pid or "
                    f"uptime evidence was not confirmed within {RESTART_CONFIRM_WINDOW_S:g}s"
                    f"{suffix}; restart state is unknown"
                ),
                before=dict(before),
            )

        link, reason = server._ensure_link_with_reason(  # noqa: SLF001 — relay dial helper
            device_id, probe_timeout_s=min(1.0, remaining)
        )
        last_reason = reason
        if link is None:
            time.sleep(min(RESTART_POLL_S, max(0.0, deadline - time.monotonic())))
            continue
        try:
            if wire.MESH_RESTART_V1 not in link.capabilities:
                return _result(
                    server,
                    device_id,
                    ok=False,
                    state="unconfirmed",
                    code="restart_unconfirmed",
                    message=(
                        f"{label} accepted the restart, but the answering peer no longer "
                        "advertises peer-restart evidence; restart state is unknown"
                    ),
                    before=dict(before),
                )
            try:
                reply = link.request(
                    {
                        "op": "net_restart",
                        "req": server._next_relay_req(),  # noqa: SLF001
                        "locality": "remote",
                        "phase": "evidence",
                    },
                    timeout=min(RESTART_EVIDENCE_TIMEOUT_S, remaining),
                )
            except (OSError, TimeoutError):
                reply = None
            parsed = _evidence_from_reply(reply)
            if parsed is None:
                if isinstance(reply, Mapping) and reply.get("op") == "error":
                    last_reason = str(reply.get("message") or "evidence request refused")
                else:
                    last_reason = "the peer did not answer the evidence request"
            else:
                after, outcome = parsed
                if str(after.get("instance_id") or "") == str(before.get("instance_id") or ""):
                    # The old process is still answering. A delayed busy refusal or
                    # supervisor error is definitive; an accepted marker is not.
                    if outcome.get("state") in ("busy", "failed"):
                        return _result(
                            server,
                            device_id,
                            ok=False,
                            state=str(outcome["state"]),
                            code=str(outcome.get("code") or outcome["state"]),
                            message=str(outcome.get("reason") or "the peer did not restart"),
                            before=dict(before),
                            after=after,
                        )
                elif _confirmed(before, after, link.instance_id):
                    return _success(server, device_id, before, after)
                else:
                    last_reason = "the new instance did not supply matching pid or uptime evidence"
        finally:
            # Never let the next probe reuse a cached link to the old instance.
            link.close("peer-restart-evidence-complete")
        time.sleep(min(RESTART_POLL_S, max(0.0, deadline - time.monotonic())))


def restart_peer(server: RelayServer, device_id: str) -> dict[str, Any]:
    """Ask one member to restart its relay; report success only after fresh evidence."""
    label = _label(server, device_id)
    link, reason = server._ensure_link_with_reason(device_id)  # noqa: SLF001
    if link is None:
        suffix = f": {reason}" if reason else ""
        return _result(
            server,
            device_id,
            ok=False,
            state="unreachable",
            code="unreachable",
            message=f"{label} is not answering right now, so it was not asked to restart{suffix}",
        )
    try:
        if wire.MESH_RESTART_V1 not in link.capabilities:
            return _result(
                server,
                device_id,
                ok=False,
                state="refused",
                code="predates_peer_restart",
                message=(
                    f"{label} does not support peer restart; run `lop network restart` "
                    "on that device, or update it first"
                ),
            )
        try:
            reply = link.request(
                {
                    "op": "net_restart",
                    "req": server._next_relay_req(),  # noqa: SLF001
                    "locality": "remote",
                    "phase": "request",
                },
                timeout=RESTART_REQUEST_TIMEOUT_S,
            )
        except (OSError, TimeoutError):
            return _no_answer(server, device_id, label)
        if reply is None:
            return _no_answer(server, device_id, label)
        if reply.get("op") == "error":
            return _reply_error(server, device_id, reply)
        detail = reply.get("detail")
        if not isinstance(detail, Mapping):
            return _no_answer(server, device_id, label)
        state = str(detail.get("state") or "")
        if state == "busy":
            return _result(
                server,
                device_id,
                ok=False,
                state="busy",
                code=str(detail.get("code") or "busy"),
                message=str(detail.get("reason") or "the peer is busy; restart refused"),
            )
        before = detail.get("evidence")
        if state != "accepted" or not isinstance(before, Mapping):
            return _result(
                server,
                device_id,
                ok=False,
                state="failed",
                code=str(detail.get("code") or "bad_response"),
                message=str(detail.get("reason") or "the peer did not accept the restart"),
            )
        before = dict(before)
    finally:
        # Acceptance is the member's cue to run its supervisor; closing here is
        # also what guarantees the accepted response was received before restart.
        link.close("peer-restart-request-complete")
    return _confirm_restart(server, device_id, before)


def make_handler(server: RelayServer) -> Any:
    def _handle(link: PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        return member_execute(server, link, frame)

    return _handle


def local_restart_handler(server: RelayServer) -> Any:
    def _handle(payload: dict[str, Any]) -> dict[str, Any]:
        device_id = server._resolve_peer(str(payload.get("peer") or ""))  # noqa: SLF001
        return restart_peer(server, device_id)

    return _handle


def install(server: RelayServer) -> None:
    """Register the closed-table peer op and its local client half."""
    server.register_ops(
        {"net_restart": make_handler(server)},
        local_handlers={"peer_restart": local_restart_handler(server)},
    )

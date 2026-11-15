"""The session-side client for the desktop app's browser host.

The transport is NOT re-implemented here: :class:`UiHostClient` is
:class:`~local_operator.browser_bridge.backend.HostClient` pointed at this
package's discovery record. That inheritance is the whole design of this module —
the RPC shape, the 401 handling, the error mapping and the client timeout
derivation are already identical for both hosts, and they are the parts most
likely to be fixed once and forgotten in the other copy.

What IS host-specific, and therefore lives here:

* which record to read (this package's ``state``),
* what a healthy ``/health`` says (the UI host's answer carries no
  ``extension_connected`` — it has no third party to be detached from, so the
  probe instead asks "is the pid that answered the same pid the file named?"),
* and the four availability answers every browsable host must supply, because
  ``builtin`` branches on them.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from local_operator.browser_bridge.backend import (
    HEALTH_PROBE_TIMEOUT_S,
    HOST_UI,
    BridgeUnreachable,
    HostClient,
    HostPreDeliveryFailure,
)
from local_operator.ui_browser import state as state_store

logger = logging.getLogger(__name__)

#: MEASURED BEFORE CAPPED (2026-11-15, 42 ``ui_host_health_probe_failed`` lines in
#: ``~/.local-operator/logs/exec-*.log``, Nov 10-15, machine load 45-48):
#:
#: * 18 had the CHECKER starved: loop_lag 648-14129 ms, 17 of 18 at or under
#:   5.24 s;
#: * 16 had the checker on time (loop_lag 2-228 ms) yet BOTH 1.5 s attempts
#:   expired (elapsed 3.05-3.10 s) — the request itself, not the loop, was
#:   starved (thread/socket scheduling at load ~47);
#: * 4 were reachable after one attempt, 4 were connection refused.
#:
#: Meanwhile the host side (the desktop app's synchronous ``/health``) answered
#: in 0-1 ms during all three 2026-11-15 incidents (its main-loop delay: n=4777,
#: median 1 ms, max 1125 ms). So almost every failure condemned a healthy host
#: because a second 1.5 s window was no longer than the first. The first attempt
#: stays at ``HEALTH_PROBE_TIMEOUT_S`` (the healthy path is unchanged and a
#: refused port still fails at once); the ONE retry gets this longer budget,
#: which covers 17/18 measured stalls, still clamped to the caller's deadline.
HEALTH_RETRY_TIMEOUT_S = 5.0

#: Pause between the two attempts of a probe whose first attempt TIMED OUT.
#: Long enough for a descheduled checker (and the thread pool / socket it
#: needs) to get a turn before the retry's clock starts, instead of starting
#: that clock inside the same stall; short against the 5 s retry budget. The
#: worst case a hung host costs is ``first + backoff + retry`` (~7 s), never
#: past the caller's deadline.
HEALTH_RETRY_BACKOFF_S = 0.5

#: Durable record of probe failures, in :func:`local_operator.paths.log_dir`.
#: WHY a file of its own: the session runtimes log to the shared rotating
#: ``runtime.log``, which ~25 concurrent sessions roll through all five
#: generations in about two minutes (measured 2026-11-15: ``runtime.log`` ..
#: ``runtime.log.4`` spanned 17:19:17-17:21:12). The WARNING was written and
#: gone before anyone looked, so the three 2026-11-15 incidents had no
#: ``loop_lag_ms``. Failures are rare, so this file holds weeks.
HEALTH_FAILURE_LOG_NAME = "browser-health.log"
#: Size at which the file is rolled to one ``.1`` backup (~1,000 lines each).
HEALTH_FAILURE_LOG_MAX_BYTES = 256 * 1024

#: Attempts a TIMED-OUT probe gets. One retry, never more: a host that is truly
#: hung must still be reported within ``first + backoff + HEALTH_RETRY_TIMEOUT_S``
#: (~7 s, deadline-clamped), and anything that is not a
#: timeout (refused, mismatch, bad body) is a definite answer that a second GET
#: cannot change.
HEALTH_MAX_ATTEMPTS = 2

#: Longest slice of host-controlled text (a ``/health`` field) allowed into the log
#: line and the agent-facing sentence. A foreign service on a recycled port
#: controls these bytes; bounding them keeps a hostile or buggy listener from
#: flooding ``runtime.log`` or the model's context.
_DETAIL_MAX_CHARS = 80

#: Minimum seconds between two WARNINGs for the same (pid, port, outcome) from the
#: REACHABILITY site. ``ui_browser_reachable`` runs on the action path, so a silent
#: host would otherwise log once per action (twice for an unpinned ``open``). The
#: preflight site is never limited: it is once per dispatched command and carries
#: the user-visible failure.
_REACHABLE_LOG_INTERVAL_S = 30.0
_reachable_last_logged: dict[tuple[Any, Any, str], float] = {}

#: Sampling period of the loop-lag canary (see ``_LoopLagCanary``).
_LAG_TICK_S = 0.05


class UiHostClient(HostClient):
    """One authenticated call against the app's loopback host, or a typed error."""

    host = HOST_UI

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(state_store, root)

    def request_headers(self, current: Any) -> dict[str, str]:
        headers = super().request_headers(current)
        if current.instance_id is not None:
            headers["X-UI-Browser-Instance"] = current.instance_id
        return headers

    async def preflight(self, current: Any, *, deadline: float | None = None) -> None:
        # A fresh heartbeat plus a live/reused PID is not process identity. Only
        # modern records buy this bounded check; legacy UI contracts stay intact.
        remaining = HEALTH_PROBE_TIMEOUT_S
        if deadline is not None:
            remaining = min(remaining, deadline - asyncio.get_running_loop().time())
            if remaining <= 0:
                raise TimeoutError("browser operation deadline exhausted before health")
        if current.instance_id is None:
            return

        def record_rotated() -> bool:
            # A rotated record means `call` will rediscover and probe the NEW
            # generation; spending a second full budget on the old one only delays
            # that. Unreadable discovery counts as "unchanged": decide on the probe.
            try:
                latest = self._read()
            except Exception:  # noqa: BLE001 - discovery may never raise at a call site
                return False
            return latest is not None and self._record_identity(latest) != self._record_identity(
                current
            )

        probe = await probe_health(
            current,
            first_budget=remaining,
            deadline=deadline,
            site="preflight",
            abandon_retry_if=record_rotated,
        )
        if not probe.ok:
            raise HostPreDeliveryFailure(_failure_copy(probe, current, self.copy_for()))

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        current = self._read()
        if current is None:
            raise BridgeUnreachable(self.copy_for().no_state)
        budget = self.timeout_for(method, params)
        deadline = asyncio.get_running_loop().time() + budget
        try:
            # HTTPX's timeout bounds each I/O phase, not the sum of phases. The
            # outer clock also fences health/rediscovery and an ambiguous expiry
            # after send can never enter the pre-delivery retry branch.
            async with asyncio.timeout(budget):
                try:
                    return await self.call_once(current, method, params, deadline=deadline)
                except HostPreDeliveryFailure:
                    replacement = self._read()
                    # An ABSENT record means "nothing new to try": re-raise the
                    # typed pre-delivery failure rather than dispatch with None
                    # (which would raise AttributeError in preflight). Only a
                    # present record that CHANGED identity is worth a retry.
                    changed = replacement is not None and (
                        self._record_identity(replacement) != self._record_identity(current)
                    )
                    if not changed:
                        raise
                    return await self.call_once(replacement, method, params, deadline=deadline)
        except TimeoutError as exc:
            raise BridgeUnreachable(
                self.copy_for().timeout.format(method=method, timeout=budget)
            ) from exc

    @staticmethod
    def _record_identity(current: state_store.UiHostState) -> tuple[int, int, str, str | None]:
        # Includes key rotation inside the same process, but is never logged: the
        # session key remains secret and identity is not an ownership proof.
        return (current.port, current.pid, current.session_key, current.instance_id)


def ui_browser_available(root: Path | None = None) -> bool:
    """File-only "known-good right now" probe: no socket, never raises.

    File-only for the same reason the bridge's is: this runs while constructing
    every session, and a socket round-trip there would tax startup for every
    session on the machine.
    """
    try:
        return state_store.available(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_browser_advertisable(root: Path | None = None) -> bool:
    """File-only gate for whether the `browser` TOOL is offered at all.

    Accepts a STALE-but-alive heartbeat, like the bridge's gate: advertising is a
    weaker commitment than executing, and hiding the tool from a host whose
    heartbeat writer stopped is how the agent ends up with no browser at all and
    no explanation for a host that is running.
    """
    try:
        return state_store.advertisable(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_liveness(
    root: Path | None = None,
) -> tuple[state_store.Liveness, state_store.UiHostState | None]:
    """Classify the host from the file, never raising at a diagnostic site."""
    try:
        return state_store.liveness(root)
    except Exception:  # noqa: BLE001 - a diagnostic may never raise
        return state_store.Liveness.ABSENT, None


async def ui_browser_reachable(
    root: Path | None = None,
    *,
    classified: tuple[state_store.Liveness, state_store.UiHostState | None] | None = None,
) -> bool:
    """Availability for the BROWSER PATH: file first, socket only to acquit.

    The same contract as the bridge's, and it exists for the same reason: the
    heartbeat is a proxy that lies in both directions, so ``FRESH`` answers yes
    without a probe, ``ABSENT`` answers no without a probe, and only ``STALE``
    buys ONE bounded ``/health`` request before the host is condemned. Modern
    instance-labelled records are also probed when fresh: a heartbeat/PID cannot
    identify a recycled process. File-only tool advertisement stays inexpensive.
    """
    try:
        status, current = classified if classified is not None else state_store.liveness(root)
    except Exception:  # noqa: BLE001 - discovery must never raise at a call site
        return False
    if current is None or status not in (state_store.Liveness.FRESH, state_store.Liveness.STALE):
        return False
    if status is state_store.Liveness.FRESH and current.instance_id is None:
        return True
    # ONE attempt: this runs on the action path with no caller deadline, ahead of
    # `preflight`, which owns the starved-checker retry. Retrying here as well
    # doubled the worst case of a stale, silent host (measured 6.2-6.4 s vs 3.3-3.9 s
    # on main) for no gain: a false "unreachable" here only re-routes this action,
    # while the command itself is still protected by preflight's retry.
    probe = await probe_health(current, site="reachable", max_attempts=1)
    return probe.ok


class HealthVerdict(enum.Enum):
    """Why a ``/health`` probe did or did not acquit the published host.

    One boolean used to stand for all of these, so a host that was merely slow to
    be *checked* read exactly like one that had died. The distinction drives both
    the retry policy (only ``TIMEOUT`` is retried) and the sentence the agent
    sees (``_failure_copy``).
    """

    OK = "ok"
    #: Nothing accepted the connection / the transport broke: the process is gone
    #: or its socket is. A second GET cannot change that, so it is never retried.
    REFUSED = "refused"
    #: No complete answer within the budget on EVERY attempt. The only verdict
    #: that does not say the host is bad — it may be fine and the checker late.
    TIMEOUT = "timeout"
    #: Answered, but as a different process/generation than the record names
    #: (recycled pid or port, restarted app, another service).
    MISMATCH = "mismatch"
    #: Answered, but not with a healthy host body (non-200, malformed, not ready).
    BAD_RESPONSE = "bad_response"


@dataclass(frozen=True)
class HealthProbe:
    """The measured outcome of one (possibly retried) bounded ``/health`` probe.

    Carries what an operator needs to tell "host late" from "checker late" after
    the fact. Deliberately contains no session key: it is rendered into the log.
    """

    verdict: HealthVerdict
    #: Wall time across all attempts including the backoff.
    elapsed_s: float
    attempts: int
    #: The per-attempt budget the FIRST attempt was given (deadline-clamped).
    budget_s: float
    #: Exception class name that ended the last attempt, "" when none was raised.
    error: str = ""
    #: Worst lateness of the checker's own event loop while it was probing, in ms.
    loop_lag_ms: float = 0.0
    #: Short human reason for MISMATCH/BAD_RESPONSE, clipped to _DETAIL_MAX_CHARS.
    detail: str = ""
    #: True when the caller's deadline shortened any attempt below the standard
    #: budget, so "within <budget>s" would overstate what was actually allowed.
    clamped: bool = False
    #: The budget the retry was given (deadline-clamped); 0.0 when none ran.
    retry_budget_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict is HealthVerdict.OK


class _LoopLagCanary:
    """Measures how late the CHECKER's event loop runs its own timers.

    HTTPX's timeout is wall-clock: if the checker process is blocked or
    descheduled for longer than the budget while the host answers instantly, the
    timer fires first and a healthy host is condemned. A self-rescheduling
    ``call_later`` tick records how far past its due time each tick ran, so the
    failure log can say which side was late. A ``call_later`` chain rather than a
    task keeps the happy path to a couple of timer handles.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._handle: asyncio.TimerHandle | None = None
        self._due = 0.0
        self.max_lag_s = 0.0

    def start(self) -> None:
        self._due = time.monotonic() + _LAG_TICK_S
        self._handle = self._loop.call_later(_LAG_TICK_S, self._tick)

    def _tick(self) -> None:
        now = time.monotonic()
        self.max_lag_s = max(self.max_lag_s, now - self._due)
        self._due = now + _LAG_TICK_S
        self._handle = self._loop.call_later(_LAG_TICK_S, self._tick)

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        # A loop blocked at the very end of an attempt never ran its tick; the
        # time it overran is still real lateness, so fold it in here.
        self.max_lag_s = max(self.max_lag_s, time.monotonic() - self._due)


def _clip(text: object) -> str:
    """Bound host-controlled text; an ellipsis marks that something was cut."""
    value = str(text)
    return value if len(value) <= _DETAIL_MAX_CHARS else value[:_DETAIL_MAX_CHARS] + "…"


async def _probe_once(current: Any, budget: float) -> tuple[HealthVerdict, str, str]:
    """One ``/health`` GET bounded by ``budget``: (verdict, exception class, detail).

    The probe requires the answering process to be the pid the FILE named, not
    merely HTTP 200: a stale record whose port has been recycled by another
    process would otherwise be acquitted as this host. That is the UI-host
    equivalent of the bridge's ``extension_connected`` requirement in
    :func:`~local_operator.browser_bridge.backend._health_ok`.

    A modern record additionally pins ``instance_id``: a recycled PID/port or a
    same-version UI restart must not pass as the old process. Legacy records omit
    it and retain the original PID/host contract.

    ``proto`` is reported but deliberately not required to equal
    ``PROTO_VERSION``: a version-skewed host is a REAL host that can explain
    itself, so it stays reachable and the per-action path returns the typed
    ``proto_mismatch`` whose copy names the remedy.

    The outer ``asyncio.timeout`` makes ``budget`` a TOTAL bound for the attempt;
    HTTPX's own timeout bounds each I/O phase separately and would let connect
    plus read exceed it.
    """
    try:
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(timeout=budget) as client:
                response = await client.get(f"http://127.0.0.1:{current.port}/health")
    except (httpx.TimeoutException, TimeoutError) as exc:
        return HealthVerdict.TIMEOUT, type(exc).__name__, ""
    except httpx.TransportError as exc:
        # ConnectError (refused), reset, protocol error: the socket is not serving.
        return HealthVerdict.REFUSED, type(exc).__name__, ""
    except Exception as exc:  # noqa: BLE001 - any other failure is "not a healthy answer"
        return HealthVerdict.BAD_RESPONSE, type(exc).__name__, ""
    if response.status_code != 200:
        return HealthVerdict.BAD_RESPONSE, "", _clip(f"http {response.status_code}")
    try:
        body = response.json()
        answered_pid = int(body.get("pid", -1)) if isinstance(body, dict) else -1
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the body is host-controlled: NOTHING it holds may raise
        # `pid` can be inf (``1e999`` -> OverflowError), NaN (ValueError), a list or
        # dict (TypeError), a non-numeric string, or a literal past int()'s digit
        # limit; the body itself can be invalid JSON/UTF-8. All are one verdict, and
        # the old bool helper's blanket `except` guaranteed the same.
        return HealthVerdict.BAD_RESPONSE, type(exc).__name__, "malformed body"
    if not isinstance(body, dict):
        return HealthVerdict.BAD_RESPONSE, "", "malformed body"
    if body.get("host") != state_store.HOST:
        return HealthVerdict.MISMATCH, "", _clip(f"host {body.get('host')!r}")
    if answered_pid != current.pid:
        return HealthVerdict.MISMATCH, "", _clip(f"answering pid {answered_pid}")
    if current.instance_id is not None and body.get("instance_id") != current.instance_id:
        return HealthVerdict.MISMATCH, "", "instance_id differs from the published generation"
    if body.get("ready") is False:
        return HealthVerdict.BAD_RESPONSE, "", "host reports not ready"
    return HealthVerdict.OK, "", ""


async def probe_health(
    current: Any,
    *,
    first_budget: float | None = None,
    deadline: float | None = None,
    site: str,
    max_attempts: int = HEALTH_MAX_ATTEMPTS,
    abandon_retry_if: Callable[[], bool] | None = None,
) -> HealthProbe:
    """Probe the published host's ``/health``, surviving a starved CHECKER, bounded.

    WHY the retry exists: measured host latency is single-digit milliseconds even
    under heavy load, so a 1.5 s miss is almost never the host being slow. The
    clock that expired is the checker's — its event loop was blocked or
    descheduled past the budget while the answer sat ready — and one fresh
    attempt, run once the loop is moving again, acquits a healthy host.

    Policy (the bound a hung host can cost is ``budget + backoff +
    HEALTH_RETRY_TIMEOUT_S``, and never past the caller's ``deadline``):

    * the retry's budget is ``min(HEALTH_RETRY_TIMEOUT_S, time left before
      deadline)``, longer than the first: a retry no longer than the attempt
      that just starved mostly starved too (see the measurements beside the
      constant);
    * only ``TIMEOUT`` is retried. A starved attempt (loop lag past its budget)
      always surfaces as ``TIMEOUT`` — the outer ``asyncio.timeout`` fires as
      soon as the loop runs — so no separate "starved" trigger is needed;
      refused / mismatch / bad body are definite and
      fail on the first attempt, so a dead host is reported as fast as before;
    * a timeout whose pid is no longer alive is also definite (no retry);
    * the happy path is exactly one GET;
    * ``max_attempts=1`` disables the retry (the reachability gate, which has no
      caller deadline and must not double a silent host's cost);
    * ``abandon_retry_if`` is consulted before the retry: when it says the record
      this probe is checking was replaced, the retry is skipped so the caller's
      rediscovery can probe the new generation instead.

    A failure is logged once as a single ``key=value`` line (nothing on success).
    ``deadline`` is on ``asyncio``'s loop clock, like ``preflight``'s.
    """
    loop = asyncio.get_running_loop()
    canary = _LoopLagCanary(loop)
    started = time.monotonic()
    # Resolved per call, not at import, so the budget is one tunable constant.
    first_budget = HEALTH_PROBE_TIMEOUT_S if first_budget is None else first_budget
    budget = first_budget
    attempts = 0
    clamped = first_budget < HEALTH_PROBE_TIMEOUT_S
    verdict, error, detail = HealthVerdict.TIMEOUT, "", ""
    retry_budget = 0.0
    while True:
        attempts += 1
        canary.start()
        try:
            verdict, error, detail = await _probe_once(current, budget)
        finally:
            canary.stop()
        if verdict is not HealthVerdict.TIMEOUT or attempts >= max_attempts:
            break
        if not state_store.pid_alive(current.pid):
            break
        if abandon_retry_if is not None and abandon_retry_if():
            break
        # The pause is clamped too: a caller's deadline is a hard ceiling, and the
        # backoff is part of what a retry costs.
        backoff = HEALTH_RETRY_BACKOFF_S
        budget = HEALTH_RETRY_TIMEOUT_S
        if deadline is not None:
            backoff = min(backoff, max(0.0, deadline - loop.time()))
        await asyncio.sleep(backoff)
        if deadline is not None:
            budget = min(budget, deadline - loop.time())
            if budget <= 0:
                # The deadline, not the host, denied the retry: "within <budget>s
                # over 1 attempt" would hide that a second attempt was owed.
                clamped = True
                break
            clamped = clamped or budget < HEALTH_RETRY_TIMEOUT_S
        retry_budget = budget
    probe = HealthProbe(
        verdict=verdict,
        elapsed_s=time.monotonic() - started,
        attempts=attempts,
        budget_s=first_budget,
        error=error,
        loop_lag_ms=canary.max_lag_s * 1000.0,
        detail=detail,
        clamped=clamped,
        retry_budget_s=retry_budget,
    )
    if not probe.ok:
        _log_probe_failure(current, probe, site)
    return probe


def _log_probe_failure(current: Any, probe: HealthProbe, site: str) -> None:
    """One greppable line per failed probe; never the session key.

    ``loop_lag_ms`` is the evidence for "which side was late": near zero means the
    checker's loop ran on time, so a timeout is the host's; hundreds of ms or more
    means the checker itself was starved. It is reported, not turned into a
    verdict, because on a loaded machine both can be true at once. ``generation``
    is the host's ``started_at``, which separates "the same process, repeatedly"
    from "a recycled one".
    """
    if site == "reachable":
        key = (current.pid, current.port, probe.verdict.value)
        now = time.monotonic()
        last = _reachable_last_logged.get(key)
        if last is not None and now - last < _REACHABLE_LOG_INTERVAL_S:
            return
        _reachable_last_logged[key] = now
    line = (
        "ui_host_health_probe_failed site=%s outcome=%s pid=%s port=%s instance_id=%s "
        "generation=%s budget_s=%.3f retry_budget_s=%.3f elapsed_s=%.3f attempts=%d "
        "loop_lag_ms=%.1f error=%s detail=%r pid_alive=%s"
    ) % (
        site,
        probe.verdict.value,
        current.pid,
        current.port,
        current.instance_id,
        getattr(current, "started_at", None),
        probe.budget_s,
        probe.retry_budget_s,
        probe.elapsed_s,
        probe.attempts,
        probe.loop_lag_ms,
        probe.error or "none",
        probe.detail,
        state_store.pid_alive(current.pid),
    )
    logger.warning("%s", line)
    _persist_probe_failure(line)


def _persist_probe_failure(line: str) -> None:
    """Append one failure line to :data:`HEALTH_FAILURE_LOG_NAME`; never raises.

    A plain ``O_APPEND`` write rather than a logging handler: several session
    runtimes write this file at once, and a single short ``write`` on an
    append-mode descriptor lands whole, without the cross-process rename races
    a ``RotatingFileHandler`` per process would add. Rolling is a size check
    and one rename; two processes rolling together at worst lose one backup of
    a diagnostic, which is acceptable. The line carries no session key (see
    :class:`HealthProbe`), so the file is safe to hand to support; it is 0600
    anyway like every other log here.
    """
    try:
        from local_operator.paths import ensure_log_dir

        directory = ensure_log_dir()
        if directory is None:
            return
        path = directory / HEALTH_FAILURE_LOG_NAME
        try:
            if path.stat().st_size >= HEALTH_FAILURE_LOG_MAX_BYTES:
                os.replace(path, path.with_name(path.name + ".1"))
        except FileNotFoundError:
            pass
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record = f"{stamp} pid={os.getpid()} {line}\n".encode("utf-8", "replace")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, record)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - a diagnostic write may never fail the action
        logger.debug("ui_host_health_failure_log_unwritable", exc_info=True)


def _failure_copy(probe: HealthProbe, current: Any, copy: Any) -> str:
    """The agent-facing sentence for a failed probe; each class says its own thing.

    Every one ends "no command was sent" because preflight runs before dispatch,
    which is also what keeps these retriable as :class:`HostPreDeliveryFailure`.
    The old single sentence called all of them "unavailable", which for a merely
    late check sent the agent off to restart a healthy app.
    """
    if probe.verdict is HealthVerdict.TIMEOUT and state_store.pid_alive(current.pid):
        plural = "s" if probe.attempts != 1 else ""
        # When the caller's deadline shortened an attempt, "within 1.5s" would claim
        # more patience than was possible; say what was true instead.
        window = (
            "within the time allowed "
            f"({probe.elapsed_s:.1f}s over {probe.attempts} attempt{plural})"
            if probe.clamped
            else f"within {probe.budget_s:.1f}s over {probe.attempts} attempt{plural}"
        )
        return (
            f"the desktop browser host is published (pid {current.pid}, port {current.port}) "
            f"but did not answer the bounded health check {window}; the host may be "
            "fine and this machine overloaded — retry the same command; no command was sent"
        )
    if probe.verdict is HealthVerdict.MISMATCH:
        return (
            f"the process answering at 127.0.0.1:{current.port} is not the published desktop "
            f"browser host generation (pid {current.pid}; {probe.detail}); the app may have "
            "restarted — retry the same command; no command was sent"
        )
    if probe.verdict is HealthVerdict.BAD_RESPONSE:
        return (
            f"the desktop browser host at 127.0.0.1:{current.port} answered the health check "
            f"with an unhealthy response ({probe.detail or probe.error}); no command was sent"
        )
    # REFUSED, or a timeout whose process is gone: the existing "no longer
    # answering" voice, which tells the user to re-open the app.
    return f"{copy.not_answering.format(port=current.port)} No command was sent."

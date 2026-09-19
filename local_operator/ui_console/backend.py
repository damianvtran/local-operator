"""The session-side client for the desktop app's console surfaces.

The transport is NOT re-implemented here: :class:`ConsoleHostClient` is
:class:`~local_operator.browser_bridge.backend.HostClient` pointed at the app
host's discovery record, with two things overridden and nothing else:

* **The failure copy.** The app host's own table entry names its *browser*
  ("open a browser tab in it, then retry"), which is the wrong remedy for a
  session that asked the console to read a surface even though the process is the
  same. The console's sentences live here, next to the console.
* **The timeout table.** The shared arithmetic is `base +
  ORIGIN_PROMPT_WINDOW_S + margin` because a browser command can sit on a human's
  site-approval popup. No console method waits on a prompt — the app answers or
  refuses within a second or two — so inheriting that arithmetic would give an
  unknown console method a 190-second ceiling and turn a wedged app into a tool
  call that hangs for three minutes. That is precisely the outcome the console's
  absence copy exists to replace with an honest answer (design §15), so the
  console sizes its own calls.

What is deliberately NOT here: the console's method names are absent from
:data:`local_operator.browser_bridge.protocol.METHODS`. That tuple is the
BRIDGE/extension wire's closed list — ``gen_ts`` renders it into the extension's
TypeScript union, and a test pins ``set(METHODS) == set(COMMAND_TIMEOUTS)`` — and
the console's ten methods are served by the app's RPC host, not by the extension,
which has no handler for any of them. Adding them there would advertise ten
extension methods that answer with a bare ``internal``. The vocabulary is frozen
by the UI side of the split (design §17.1 row A); this module carries the Python
half's copy of it so a caller can validate a name before it reaches the wire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from local_operator.browser_bridge.backend import (
    HEALTH_PROBE_TIMEOUT_S,
    HOST_UI,
    BridgeError,
    HostClient,
    HostCopy,
)
from local_operator.browser_bridge.protocol import ErrorCode
from local_operator.ui_console import state as state_store

#: The ten console methods (design §10.2), in the app's own order. A tuple so the
#: order is stable and a caller can render the accepted set in a refusal.
CONSOLE_METHODS: tuple[str, ...] = (
    "console_list",
    "console_create",
    "console_status",
    "console_read",
    "console_screenshot",
    "console_input",
    "console_keys",
    "console_resize",
    "console_secure",
    "console_close",
)

#: Per-method HTTP budget, in seconds. Sized to the work, not to a prompt: a
#: screenshot is the slowest (the app may have to build its one-at-a-time capture
#: view and retry a blank first frame, design §13.2/§13.3), a create forks a pty,
#: and everything else is a record read or a write into a queue. Anything that
#: exceeds these is a wedged app, and saying so in 25 seconds is worth more to the
#: model than waiting three minutes to say the same thing.
CONSOLE_TIMEOUTS: dict[str, float] = {
    "console_list": 15.0,
    "console_create": 30.0,
    "console_status": 15.0,
    "console_read": 30.0,
    "console_screenshot": 60.0,
    "console_input": 15.0,
    "console_keys": 15.0,
    "console_resize": 15.0,
    "console_secure": 15.0,
    "console_close": 20.0,
}

#: Slack over the app's own budget for scheduling jitter and transport setup,
#: the same role `_CLIENT_TIMEOUT_MARGIN_S` plays for the browser's arithmetic.
CONSOLE_TIMEOUT_MARGIN_S = 5.0

#: The console's sentences for the transport failures `HostClient.call` raises.
#: Same five slots as every host's record, for the reason `HostCopy` gives: each
#: is a claim about WHICH THING the reader should go and look at.
CONSOLE_COPY = HostCopy(
    label="Local Operator desktop app's console",
    no_state=(
        "the Local Operator desktop app is not running, so its console surfaces are "
        "gone with it — a surface lives inside the app's process and cannot outlive "
        "it. Ask the user to open the app, then create a new surface; anything the "
        "old one printed is not reachable from here."
    ),
    not_answering=(
        "the Local Operator desktop app is no longer answering at 127.0.0.1:{port}. "
        "If the user quit it, ask them to re-open it: the surfaces ended with the "
        "app and a new one has to be created."
    ),
    timeout=(
        "the Local Operator desktop app accepted '{method}' but did not answer within "
        "{timeout:.0f}s, so the app is not draining its console work — do not retry "
        "in a loop. Ask the user to check the app (its Console pane and its log), and "
        "report this as the app being unresponsive rather than as a tool failure."
    ),
    rejected_key=(
        "the Local Operator desktop app rejected the state-file key, so this session "
        "cannot reach its console. Ask the user to restart the app."
    ),
    invalid_response=(
        "the Local Operator desktop app's console returned an invalid response " "(HTTP {status})."
    ),
)

#: What the model reads when the app is not there at all. Deliberately cheaper
#: than a socket attempt: the record is file-only and synchronous, so a session
#: that has watched the app quit gets the honest answer immediately instead of
#: after a connect timeout. The surfaces are named as ENDED, because that is the
#: fact the agent needs (design §15, last row).
CONSOLE_ABSENT_COPY = (
    "No Local Operator desktop app is running, so there is no console to drive and "
    "any surface an earlier turn created has ended with it. Tell the user the surfaces "
    "are gone; do not retry, and do not try to reach a terminal by another route — "
    "another emulator's window is not this console."
)

#: What the model reads when the app is up but its console feature is off. The
#: three possible reasons are named because the app's record cannot distinguish
#: them for us and the remedies differ (a settings toggle, a launch flag, a broken
#: native module worth reporting).
CONSOLE_FEATURE_OFF_COPY = (
    "The Local Operator desktop app is running but reports its console feature as "
    "unavailable, so no surface can be created. That is one of: the console disabled "
    "in the app's Settings, the app launched with LOCAL_OPERATOR_UI_CONSOLE_HOST=0, or "
    "the terminal component failing to load (a packaging fault the app's log names, "
    "and one worth reporting). Tell the user which to check rather than retrying."
)


class ConsoleHostClient(HostClient):
    """One authenticated console call against the app's loopback host."""

    # The app IS a UI host — the record, the key and the safety rules are the
    # same ones the browser path uses — so `host` stays `ui`; only the sentences
    # and the budget differ, and they are overridden below.
    host = HOST_UI
    failure_copy = CONSOLE_COPY

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(state_store, root)

    def timeout_for(self, method: str, params: dict[str, Any]) -> float:
        """The console's own budget: no method waits on a human prompt."""
        del params
        base = CONSOLE_TIMEOUTS.get(method, max(CONSOLE_TIMEOUTS.values()))
        return base + CONSOLE_TIMEOUT_MARGIN_S


def console_timeout(method: str) -> float:
    """The client budget for ``method``, without constructing a client.

    Exported for the tests and for the tool's own diagnostics: the number is part
    of the contract a reader can check against the app's behaviour, and a caller
    should not have to build an HTTP client to read it.
    """
    return ConsoleHostClient().timeout_for(method, {})


def ui_console_available(root: Path | None = None) -> bool:
    """File-only "known-good right now" probe: no socket, never raises.

    File-only for the same reason the browser's is: this runs while constructing
    every session, and a socket round-trip there would tax startup for every
    session on the machine.
    """
    try:
        return state_store.available(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_console_advertisable(root: Path | None = None) -> bool:
    """File-only gate for whether the `console` TOOL is offered at all.

    One predicate on the ONE `createIf` entry, the same shape as
    `ui_browser_advertisable`: file-only, synchronous, never raising, and
    accepting a STALE-but-alive heartbeat so an app whose heartbeat writer stopped
    is still reachable. The capability bit is required in addition, which is where
    this differs — see the `state` module's docstring.
    """
    try:
        return state_store.advertisable(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_console_liveness(
    root: Path | None = None,
) -> tuple[state_store.Liveness, state_store.ConsoleHostState | None]:
    """Classify the app host from the file, never raising at a diagnostic site."""
    try:
        return state_store.liveness(root)
    except Exception:  # noqa: BLE001 - a diagnostic may never raise
        return state_store.Liveness.ABSENT, None


async def ui_console_reachable(
    root: Path | None = None,
    *,
    classified: tuple[state_store.Liveness, state_store.ConsoleHostState | None] | None = None,
) -> bool:
    """Availability for the CONSOLE PATH: file first, one socket probe to acquit.

    The same contract as the browser path's, for the same reason: the heartbeat is
    a proxy that lies in both directions, so ``FRESH`` answers yes without a probe
    and only ``STALE`` buys ONE bounded ``/health`` request before the host is
    condemned. The capability bit is checked first — probing a host that has told
    us its console is off would only ever confirm the refusal.
    """
    try:
        status, current = classified if classified is not None else state_store.liveness(root)
    except Exception:  # noqa: BLE001 - discovery must never raise at a call site
        return False
    if current is None or not current.console:
        return False
    if status is state_store.Liveness.FRESH:
        return True
    if status is not state_store.Liveness.STALE:
        return False
    return await _health_ok(current.port, current.pid)


async def _health_ok(port: int, pid: int) -> bool:
    """One bounded loopback /health probe; any failure means "not reachable".

    Requires the answering process to be the pid the FILE named, like the
    browser's probe: a stale record whose port has been recycled by another
    process must not be acquitted as this host. The console capability in the
    body is NOT required again here — the file check above already decided it, and
    the two answers disagreeing (record says on, health says off) means the app is
    mid-teardown, where letting the call through produces the typed refusal rather
    than a fabricated unreachability.
    """
    try:
        async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_S) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
        if response.status_code != 200:
            return False
        body = response.json()
        return (
            isinstance(body, dict)
            and body.get("host") == state_store.HOST
            and int(body.get("pid", -1)) == pid
        )
    except Exception:  # noqa: BLE001 - unreachable, malformed, or timed out
        return False


def console_error_text(error: BridgeError) -> str:
    """The model-facing sentence for one typed console refusal (design §15).

    Typed, never substring-matched: the caller reads ``error.code`` and the copy
    comes from here, so a reworded host message cannot change how a session
    behaves. What the host already knows is carried through where the doc says it
    should be — the exit code, the accepted byte count, the clamp — because the
    remedy differs by value (an exited program needs a `create`, a full input queue
    needs a `read` first).

    An unrecognised code falls through to the host's own message with the code
    named, which is the honest answer for a refusal this version of `lop` has
    never heard of: a NEW app can refuse a method with a code an older session
    does not model, and inventing a sentence for it would be guessing.
    """
    code = error.code
    data = error.data or {}
    message = (error.message or "").strip()
    if code == ErrorCode.UNSUPPORTED_METHOD:
        return (
            "This app version has no console. Update Local Operator (the app must be "
            "rebuilt with the console feature; the `lop` side is ready for it)."
        )
    if code == ErrorCode.SURFACE_UNAVAILABLE:
        handle = str(data.get("handle") or data.get("surface") or "").strip()
        known = data.get("count")
        tail = (
            f" {known} surface(s) exist on this host."
            if isinstance(known, int)
            else " Call `console` with method='list' to see the live handles."
        )
        return (
            f"No console surface named {handle or 'that handle'} on this host — it may "
            "have been closed or the handle may be stale." + tail
        )
    if code == ErrorCode.SURFACE_NOT_OWNED:
        handle = str(data.get("handle") or data.get("surface") or "").strip()
        return (
            f"Surface {handle or 'that handle'} belongs to another session, and a "
            "session can only read its own surfaces. `list` shows the ones that are "
            "yours; the user's other conversations keep theirs."
        )
    if code == ErrorCode.PROCESS_EXITED:
        exit_code = data.get("exit_code")
        where = f" (exit code {exit_code})" if exit_code is not None else ""
        retained = data.get("retain")
        tail = (
            "its output is still retained and readable with method='read', " "mode='scrollback'"
            if retained is not False
            else "it was not retained, so its output is gone"
        )
        return f"The program in that surface has exited{where}: {tail}."
    if code == ErrorCode.INPUT_QUEUE_FULL:
        accepted = data.get("bytes")
        count = (
            f"{accepted} bytes were accepted before" if accepted is not None else "the queue filled"
        )
        return (
            f"The app refused the input because the surface is not draining it: {count} "
            "it stopped accepting. Read the surface to see why it is blocked, then "
            "retry the remainder."
        )
    if code == ErrorCode.UNKNOWN_KEY:
        # The offending name comes from the DATA, never from the host's free-text
        # message: this module's whole contract is that the model reads harness
        # copy, so that a reworded app cannot change a session's behaviour (and so
        # that a host's prose cannot smuggle anything into the transcript).
        accepted = data.get("accepted") or data.get("keys")
        names = (
            ", ".join(str(name) for name in accepted)
            if isinstance(accepted, (list, tuple)) and accepted
            else "the named set in `guide://console`"
        )
        sent = str(data.get("key") or data.get("name") or "").strip()
        return (
            f"Unknown key name{': ' + sent if sent else ''}. The accepted names are "
            f"{names}. Send a sequence of accepted names, not raw escape bytes."
        )
    if code == ErrorCode.SECURE_INPUT_ACTIVE:
        return (
            "The user has secure input ON for this surface, so the app refuses to read "
            "or capture it — nothing is wrong with the surface. Wait, or ask the user "
            "to toggle the lock off in the Console pane when they are done typing; "
            "do not work around it by reading the surface another way."
        )
    if code == ErrorCode.CONSOLE_UNAVAILABLE:
        return (
            "The app reports its console feature as unavailable and refused the call. "
            "That is a settings toggle, the launch flag LOCAL_OPERATOR_UI_CONSOLE_HOST=0, "
            "or the terminal component failing to load; tell the user rather than retrying."
        )
    if code == ErrorCode.INVALID_GRID:
        clamp = data.get("cols"), data.get("rows")
        applied = (
            f" It applied {clamp[0]}x{clamp[1]} instead."
            if all(isinstance(value, int) for value in clamp)
            else ""
        )
        return f"That grid is outside what the app will honour.{applied}"
    if code == ErrorCode.CONSOLE_CAPTURE_FULL:
        return (
            "The app's offscreen capture view is busy with another surface's frame (it "
            "is deliberately one at a time). Retry this screenshot in a moment, or read "
            "the surface as text."
        )
    if code == ErrorCode.PROTO_MISMATCH:
        return (
            f"The app and this session speak different protocol versions ({message}). "
            "Update Local Operator, then retry."
        )
    return (
        f"The app refused '{code.value}': {message or 'no detail given'}"
        if message
        else f"The app refused the call with '{code.value}'."
    )

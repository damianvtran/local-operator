"""Session-side client for the browser bridge's authenticated HTTP leg."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    COMMAND_TIMEOUTS,
    ORIGIN_PROMPT_WINDOW_S,
    PROTO_VERSION,
    ErrorCode,
    ErrorDetail,
    Request,
    Response,
)

#: Slack on top of the daemon's worst-case budget so scheduling jitter and the
#: daemon's own response serialization never push a legitimate typed answer
#: past the client's deadline.
_CLIENT_TIMEOUT_MARGIN_S = 5.0

#: Model-facing copy for the codes whose remedy is EXTENSION-specific. Kept as
#: the public ``ERROR_MESSAGES`` name because every existing consumer (and the
#: tool's `_bridge_absent_result`) means "the extension host's table", and the
#: UI host's table below is a partial OVERRIDE rather than a parallel universe:
#: the shared codes (`tab_closed`, `nav_failed`, `busy`, `tab_limit`, …) keep
#: one spelling for every host, because two spellings of the same fact is how
#: copy drifts.
_EXTENSION_ERROR_MESSAGES = {
    ErrorCode.EXTENSION_DISCONNECTED: (
        "browser extension not connected: the bridge daemon is running but no browser is "
        "attached. Ask the user to open their browser (the extension reconnects automatically), "
        "or check the extension is enabled."
    ),
    ErrorCode.NOT_PAIRED: (
        "browser bridge not paired: run 'lop browser pair' and enter the code in the "
        "extension popup, then retry."
    ),
    ErrorCode.DEBUGGER_CONFLICT: (
        "cannot drive the tab: DevTools (or another debugger) is attached to it. Ask the "
        "user to close DevTools on that tab."
    ),
    ErrorCode.BUSY: "the browser bridge is busy with another command; retry this action once.",
    ErrorCode.EXTENSION_UNRESPONSIVE: (
        # Every clause names an action that measurably works. The remedy it leads
        # to is the one with a measurement behind it: toggling the extension OFF
        # then ON in `chrome://extensions` reloads the wedged worker and
        # PRESERVES pairing (verified on real hardware — GUIDE, "failure UX").
        #
        # The previous text sent the reader to "retry in a few seconds", which
        # landed on EXTENSION_DISCONNECTED — "no browser is attached… ask the
        # user to open their browser" — for a browser that is open, i.e. the
        # exact misdirection class this code exists to remove (design D1). The
        # daemon no longer answers with that string during the cooling-off
        # window either (see `_drop_unproven_link` / `LINK_DROP_TTL_S`), so a
        # retry now lands here or on a live link.
        #
        # "toggle … OFF then ON" rather than "reload": chrome://extensions also
        # offers Update and Remove right there, and only the toggle has a
        # measurement attached (design D7).
        #
        # Deliberately NOT reusing EXTENSION_DISCONNECTED's copy.
        "the browser extension is attached and paired but has stopped answering, so the "
        "browser cannot be driven. Retry once in a few seconds. If the same action fails "
        "again the worker is wedged and only a reload clears it: ask the user to toggle "
        "the Local Operator extension OFF then ON in chrome://extensions — pairing is "
        "preserved, but open tab handles, snapshot refs and pending site decisions are "
        "lost, so re-'open' and re-'snapshot' afterwards. A daemon restart does not help."
    ),
    ErrorCode.PROTO_MISMATCH: (
        "browser bridge protocol mismatch: update Local Operator and the browser extension, "
        "then restart the bridge daemon."
    ),
}

#: The extension host's table, under its historical public name.
ERROR_MESSAGES = _EXTENSION_ERROR_MESSAGES

#: Codes whose remedy differs on the UI host. ADDITIVE: every code absent here
#: falls through to the shared table above. The ones overridden are exactly the
#: codes that name a remedy only the extension has (open your browser, toggle
#: the extension in `chrome://extensions`, `lop browser install`) — telling the
#: user of a running desktop app to install or restart a daemon they do not have
#: is the misdirection class this table exists to remove.
_UI_ERROR_MESSAGES = {
    ErrorCode.EXTENSION_DISCONNECTED: (
        "the Local Operator desktop app's browser host has no browser tab attached. "
        "Ask the user to open a browser tab in the app, then retry."
    ),
    ErrorCode.NOT_PAIRED: (
        "the Local Operator desktop app's browser host is not authenticated for this "
        "session. Ask the user to restart the desktop app, then retry."
    ),
    ErrorCode.EXTENSION_UNRESPONSIVE: (
        "the Local Operator desktop app's browser host is running but has stopped "
        "answering, so the browser cannot be driven. Retry once; if the same action "
        "fails again, ask the user to restart the desktop app — tab handles, snapshot "
        "refs and pending site decisions are lost with it, so re-'open' and "
        "re-'snapshot' afterwards."
    ),
    ErrorCode.PROTO_MISMATCH: (
        "the Local Operator desktop app's browser host speaks a different bridge "
        "protocol than this Local Operator. Update the desktop app or Local Operator, "
        "then retry."
    ),
}


def error_messages(host: str) -> dict[ErrorCode, str]:
    """The message table for one host: shared codes plus that host's overrides."""
    if host == HOST_UI:
        return {**_EXTENSION_ERROR_MESSAGES, **_UI_ERROR_MESSAGES}
    return _EXTENSION_ERROR_MESSAGES


#: The one `extension_disconnected` shape that is NOT "no browser is attached":
#: the daemon's wire fence detected that the extension REPLACED its connection
#: while a command was in flight (see `daemon.py`'s `_admit`/`_complete` send and
#: response fences). The browser is open and the worker has already re-dialled,
#: so "ask the user to open their browser" names an action that does not apply
#: and `lop browser install` is not even in scope — it resolves itself in about a
#: second (design D3-3). Selected by the `phase` the daemon carries, because the
#: code alone cannot tell this apart from a genuinely absent browser.
REPLACED_PHASE = "replaced"
REPLACED_PHASE_MESSAGE = (
    "the browser extension replaced its connection while this command was in flight, so the "
    "command was never answered. The browser is open and reconnected — retry the action; no "
    "user action is needed."
)


class BridgeError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class BridgeUnreachable(RuntimeError):
    pass


#: The two host identifiers the copy and the client seam are parameterised by.
#: Strings rather than an enum because the same spelling travels outside Python —
#: in the `ui:`/`bridge:` surface prefixes and in the resource record's `host`
#: field — so there is one spelling shared by the tool, the record and the tests.
HOST_EXTENSION = "extension"
HOST_UI = "ui"


@dataclass(frozen=True)
class HostCopy:
    """One host's sentences for the transport failures `HostClient.call` raises.

    A per-host record rather than an `if host == …` spray inside `call`, because
    each of these five sentences is a claim about WHICH PROCESS the reader should
    go and look at, and a reader told to restart the wrong one is worse off than
    one told nothing. The extension strings are byte-identical to what the
    client raised before this seam existed.
    """

    #: What to call the thing that did not answer, in prose.
    label: str
    #: There is no live discovery record at all.
    no_state: str
    #: The record exists but its socket is not answering (refused/died/reset).
    not_answering: str
    #: Connected, accepted the command, never answered within the budget.
    timeout: str
    #: The record's key was rejected (HTTP 401).
    rejected_key: str
    #: A 2xx whose body is not a valid Response envelope.
    invalid_response: str


HOST_COPY: dict[str, HostCopy] = {
    HOST_EXTENSION: HostCopy(
        label="browser bridge",
        no_state=(
            "browser bridge unreachable: no live daemon state. Run 'lop browser status'; "
            "'lop browser install' starts it."
        ),
        not_answering=(
            "browser bridge unreachable: the daemon at 127.0.0.1:{port} is not "
            "answering. Run 'lop browser status'; 'lop browser install' starts it."
        ),
        timeout=(
            "the browser bridge accepted '{method}' but did not answer within "
            "{timeout:.0f}s. The command may be stuck in the browser — e.g. waiting "
            "on a site-permission decision in the extension popup. Ask the user to "
            "check the extension popup before restarting anything."
        ),
        rejected_key=(
            "browser bridge rejected its state-file key; restart with 'lop browser restart'."
        ),
        invalid_response="browser bridge returned an invalid response (HTTP {status}).",
    ),
    HOST_UI: HostCopy(
        # Deliberately never says "run 'lop browser install'": the bridge daemon is
        # not what this host is, and installing it changes nothing about an app
        # that is closed (design, failure-UX table).
        label="Local Operator desktop app's browser host",
        no_state=(
            "the Local Operator desktop app's browser host is not running: no live host "
            "state. Open the desktop app (and a browser tab in it), then retry."
        ),
        not_answering=(
            "the Local Operator desktop app is no longer answering at 127.0.0.1:{port}; "
            "if you quit it, re-open it and retry."
        ),
        timeout=(
            "the Local Operator desktop app's browser host accepted '{method}' but did "
            "not answer within {timeout:.0f}s. The command may be waiting on a site "
            "decision in the app. Ask the user to check the app's browser tab before "
            "restarting anything."
        ),
        rejected_key=(
            "the Local Operator desktop app rejected the state-file key; ask the user to "
            "restart the app."
        ),
        invalid_response=(
            "the Local Operator desktop app's browser host returned an invalid response "
            "(HTTP {status})."
        ),
    ),
}


#: Ceiling on the confirmation probe below. It runs at most once per browser
#: action and ONLY when the cheap file check was about to condemn a daemon
#: whose pid is still alive, so it is never on the common path. Short because
#: a loopback /health answers in single-digit milliseconds; anything slower is
#: a daemon that is genuinely not serving.
HEALTH_PROBE_TIMEOUT_S = 1.5


def bridge_browser_available(root: Path | None = None) -> bool:
    """File-only "known-good right now" probe: no socket, no subprocess, never raises.

    Stays file-only on purpose: it runs while constructing EVERY session, where
    a socket round-trip would tax startup for every session on the machine. The
    stale-but-alive rescue lives in :func:`bridge_browser_reachable`, on the
    browser path.

    Use :func:`bridge_browser_advertisable` for tool GATING, which is a weaker
    commitment and must not hide a stale-but-alive daemon.
    """
    try:
        return state_store.available(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def bridge_browser_advertisable(root: Path | None = None) -> bool:
    """File-only gate for whether the `browser` TOOL is offered at all.

    Same cost and the same no-socket/no-subprocess contract as
    :func:`bridge_browser_available`, but it also accepts a STALE heartbeat
    whose pid is alive, so the RC2 rescue in ``execute_browser`` can actually
    be reached on a host with no cmux. See
    :func:`local_operator.browser_bridge.state.advertisable` for the full
    reasoning and the hot-path constraint it preserves.
    """
    try:
        return state_store.advertisable(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


async def bridge_browser_reachable(
    root: Path | None = None,
    *,
    classified: tuple[state_store.Liveness, state_store.BridgeState | None] | None = None,
) -> bool:
    """Availability for the BROWSER PATH: file first, socket only to acquit.

    The file heartbeat is a proxy that lies in both directions, and when it
    lied the failure was silent and total: a daemon whose heartbeat writer had
    died kept serving ``/health`` while every session read the file, concluded
    the extension was gone, and fell back to cmux — disagreeing with
    ``lop browser status``, which reads the live socket. Nothing reconciled
    them, so both the agent and the user concluded a phantom tab held a lock.

    The contract that removes the contradiction, without slowing the common
    case:

    - ``FRESH``  → available. No probe (the overwhelmingly common path).
    - ``ABSENT`` → unavailable. No probe; there is nothing to acquit.
    - ``STALE``  → the file cannot tell, and the pid is alive, so spend ONE
      bounded ``/health`` request before condemning the bridge.

    A daemon that answers is available regardless of what the file says.

    ``classified`` lets a caller that has ALREADY classified the daemon pass
    its answer in, so one browser action performs one file read instead of
    several and the demotion diagnostic cannot describe a different reading
    than the decision it explains.
    """
    try:
        status, current = classified if classified is not None else state_store.liveness(root)
    except Exception:  # noqa: BLE001 - discovery must never raise at a call site
        return False
    if status is state_store.Liveness.FRESH:
        return True
    if status is not state_store.Liveness.STALE or current is None:
        return False
    return await _health_ok(current.port)


async def _health_ok(port: int) -> bool:
    """One bounded loopback /health probe; any failure means "not reachable"."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_S) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
        return response.status_code == 200 and bool(
            response.json().get("extension_connected", False)
        )
    except Exception:  # noqa: BLE001 - unreachable, malformed, or timed out
        return False


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else value


#: Where an approval decision lands, per host, for the sites that name it.
_APPROVAL_SURFACE = {
    HOST_EXTENSION: "the Local Operator extension popup",
    HOST_UI: "the Local Operator desktop app's browser tab",
}

#: `origin_denied` / `origin_prompt_pending` are the two codes whose remedy is
#: "go look at the consent UI", said in one sentence per host. Templates rather
#: than a fragment, because each is a whole claim about WHERE the user looks and
#: splicing a noun into one of them reads as machine copy.
_ORIGIN_DENIED_COPY = {
    HOST_EXTENSION: (
        " Do not retry the same origin; ask the user to allow it from the extension "
        "popup if it is needed."
    ),
    HOST_UI: (
        " Do not retry the same origin; ask the user to allow it in the Local Operator "
        "desktop app's browser tab if it is needed."
    ),
}
_ORIGIN_PROMPT_PENDING_COPY = {
    HOST_EXTENSION: "the extension is waiting for the user to approve this site in its popup.",
    HOST_UI: (
        "the Local Operator desktop app is waiting for the user to approve this site in "
        "its browser tab."
    ),
}

#: The one remedy that measurably clears a wedged browser side, per host. These
#: differ because the PROCESS differs: the extension is reloaded in
#: `chrome://extensions`, the app is restarted. Naming the other host's remedy is
#: not a cosmetic error — it sends the user to a page that cannot fix anything.
_WEDGE_REMEDY = {
    HOST_EXTENSION: (
        "ask the user to toggle the Local Operator extension OFF then ON in "
        "chrome://extensions (pairing is preserved)"
    ),
    HOST_UI: "ask the user to restart the Local Operator desktop app",
}

#: What the sentence's SUBJECT is called in prose: the thing that received a
#: command and did not answer it. Distinct from `HostCopy.label`, which names the
#: TRANSPORT ("browser bridge") for the fallback error line.
_HOST_SUBJECT = {
    HOST_EXTENSION: "the browser extension",
    HOST_UI: "the Local Operator desktop app's browser host",
}


def format_error(
    error: BridgeError, *, action: str = "", surface: str = "", host: str = HOST_EXTENSION
) -> str:
    """Map every wire error to one actionable model-facing diagnostic.

    ``host`` selects the host-specific copy only. Both hosts speak the same wire
    and therefore raise the same codes; what differs is the remedy, because the
    process the reader has to go and look at differs. Every branch below is
    shared unless the sentence names chrome (`chrome://extensions`, the extension
    popup, `lop browser install`) — see `_APPROVAL_SURFACE`/`_WEDGE_REMEDY`.
    """
    messages = error_messages(host)
    label = HOST_COPY.get(host, HOST_COPY[HOST_EXTENSION]).label
    subject = _HOST_SUBJECT.get(host, _HOST_SUBJECT[HOST_EXTENSION])
    approval = _APPROVAL_SURFACE.get(host, _APPROVAL_SURFACE[HOST_EXTENSION])
    denied_copy = _ORIGIN_DENIED_COPY.get(host, _ORIGIN_DENIED_COPY[HOST_EXTENSION])
    pending_copy = _ORIGIN_PROMPT_PENDING_COPY.get(
        host, _ORIGIN_PROMPT_PENDING_COPY[HOST_EXTENSION]
    )
    remedy = _WEDGE_REMEDY.get(host, _WEDGE_REMEDY[HOST_EXTENSION])
    # Checked BEFORE the table: this code's own copy is the "no browser is
    # attached" one, and it is exactly wrong for a replaced wire — the browser is
    # open, mid-reconnect, and the command may simply be retried (design D3-3).
    if error.code == ErrorCode.EXTENSION_DISCONNECTED and error.data.get("phase") == REPLACED_PHASE:
        return REPLACED_PHASE_MESSAGE
    # Also before the table, and for the same reason: `proto_mismatch` on the UI
    # host is only actionable with BOTH numbers, and the table is a static string.
    # The wire key is `proto`, the name every other frame in this protocol uses
    # for it. No producer in THIS repo emits this code yet (the extension leg
    # validates at the socket instead), so the table's sentence is what usually
    # renders; this branch is what renders once a host reports its revision.
    if error.code == ErrorCode.PROTO_MISMATCH and host == HOST_UI:
        peer = error.data.get("proto")
        if peer is not None:
            return (
                "the Local Operator desktop app's browser host speaks bridge protocol "
                f"{peer}; this Local Operator speaks {PROTO_VERSION}. Update the "
                "desktop app or Local Operator, then retry."
            )
    if error.code in messages:
        return messages[error.code]
    if error.code == ErrorCode.TAB_CLOSED:
        return (
            f"browser tab {surface or '(unknown)'} is gone; dropped the handle. "
            "Use 'open' with a URL to get a new tab."
        )
    # Proto mismatch on the UI host names BOTH numbers when the peer reported
    # one, because "update the app or Local Operator" is not actionable without
    # knowing which side is behind. Implemented above the table lookup.
    if error.code == ErrorCode.ORIGIN_DENIED:
        origin = str(error.data.get("origin") or _origin(str(error.data.get("url", ""))))
        return (
            f"navigation to {origin or '(unknown origin)'} was denied by the user (or the "
            "permission prompt went unanswered)." + denied_copy
        )
    if error.code == ErrorCode.NAV_TIMEOUT:
        return f"navigation did not complete: {error.message}"
    if error.code == ErrorCode.NAV_FAILED:
        return f"navigation failed: {error.message}"
    if error.code == ErrorCode.ELEMENT_NOT_FOUND:
        return (
            f"element not found: {error.message}. Take a new snapshot and retry with a fresh ref."
        )
    if error.code == ErrorCode.ORIGIN_PROMPT_PENDING:
        return pending_copy
    if error.code == ErrorCode.TAB_LIMIT:
        # The extension's message already names the cap and the remedy; append
        # the discovery verb. A session that owns none of the capped tabs
        # cannot close one (handles in the listing are redacted, deliberately);
        # its remedy is asking the other sessions — or the user — to close.
        return (
            f"{error.message}. Use 'tabs' to see what is open; close only YOUR tab "
            "if one is marked '(yours)'. If none is yours, another session (or the user) "
            "must close one."
        )
    if error.code == ErrorCode.TAB_AMBIGUOUS:
        # Under-specified close, not a fault: relay the extension's message,
        # which already names the (redacted) live handles.
        return error.message
    if error.code == ErrorCode.ORIGIN_NOT_ALLOWED:
        # The teaching error of the approval flow: it must name the exact next
        # actions, because the failure it replaces (blocking the navigation on
        # a popup prompt) had agents misread the bridge as broken while the
        # prompt expired unseen. The url is echoed back so the agent can paste
        # it into the follow-up calls without re-deriving it. The agent is the
        # PRIMARY notification channel — Chrome's own banner is best-effort
        # (macOS frequently suppresses it without Notification Center
        # authorization), so the instruction to message the user is load-
        # bearing, not politeness.
        origin = str(error.data.get("origin") or _origin(str(error.data.get("url", ""))))
        url = str(error.data.get("url") or origin)
        return (
            f"site {origin or '(unknown origin)'} is not allowed yet. Call browser "
            f"action='request_access' url={url} to raise the approval prompt, then NOTIFY "
            f"THE USER (via the ask tool or a message) to approve it in {approval} — the "
            "popup badge alone is not reliably seen — and only then "
            f"action='await_access' url={url} to wait for the decision."
        )
    if error.code == ErrorCode.INTERNAL and error.data.get("tab_crashed"):
        return (
            f"the browser tab crashed while {action or 'the action'} was running. "
            "'open' the URL again to recover."
        )
    if error.code == ErrorCode.INTERNAL and error.data.get("stalled"):
        # An extension-side per-call deadline fired: the op SETTLED rather than
        # hung, which is what drains the extension's serialized chains and lets
        # the next command run. Retry is genuinely correct here, unlike for
        # EXTENSION_UNRESPONSIVE.
        return (
            f"{subject} stalled on {error.data['stalled']} and gave up on this "
            f"command. Retry; if it repeats, {remedy}."
        )
    if error.code == ErrorCode.INTERNAL and error.data.get("undrivable_tab"):
        # Chrome refuses to attach the debugger to another extension's page
        # ("Cannot access a chrome-extension:// URL of different extension").
        # The tab is alive but can never be driven, and the extension has
        # already pruned the surface, so the session's only move is a new tab.
        #
        # No tab is named: the only handle available here is the session's own
        # opaque `bridge:<tabId>:<nonce>` capability string, and interpolating
        # it (or a literal `(unknown)` when there is none) puts a useless token
        # in the middle of prose (design D6).
        return (
            "that tab cannot be driven: it is another extension's page, so Chrome refuses "
            "the debugger attachment. The handle was dropped; use 'open' with a URL to get "
            "a new tab."
        )
    if error.code == ErrorCode.INTERNAL and "timeout_s" in error.data:
        # The daemon's own budget expired with no typed answer from the
        # extension. P3 correctly stopped this reading as a version mismatch,
        # but left it on the generic fallback: a raw internal code and the
        # daemon's internal verb name for `owner_recover`, with NO remedy, on
        # the one command whose entire job is recovery (design D5). Name the
        # action the model took and give the remedy that measurably clears a
        # wedged worker; the code and the budget stay in details, out of the
        # sentence.
        seconds = error.data.get("timeout_s")
        budget = f" within {seconds:g}s" if isinstance(seconds, (int, float)) else ""
        return (
            f"{subject} received {action or 'the command'} but did not answer"
            f"{budget}. Retry once; if it repeats, the browser side is unhealthy rather "
            f"than slow — {remedy}."
        )
    return f"{label} error ({error.code.value}): {error.message}"


def client_timeout(method: str) -> float:
    """HTTP budget for one RPC: the daemon's worst case, plus margin.

    The timeout chain (finding A3) is extension deny 60 s < daemon prompt
    window 65 s < this. The daemon deliberately holds a command open for
    base + ORIGIN_PROMPT_WINDOW_S while the extension shows its approval
    popup, so the client must outlive that whole budget or it fabricates an
    "unreachable" failure mid-prompt while the daemon is healthy and about to
    deliver a typed origin_denied/result (the flat 35 s timeout this replaces
    did exactly that; QA transcript 0ee4974ba84a). Unknown methods get the
    most conservative budget — the daemon rejects them quickly anyway.

    Known residual gap: a redirect chain can pause on SEVERAL origins in one
    command (origins.ts keys prompts by origin for this reason) and the daemon
    re-extends its deadline per pause, so two human-prompted hops can hold a
    legitimate wait past this budget. We deliberately do NOT size for N
    prompts: the chain depth is unbounded so any N is arbitrary, and every
    second added here delays reporting a genuinely hung daemon. Instead the
    timeout error's message points at the extension popup, which is the right
    advice in exactly that overrun.
    """
    base = COMMAND_TIMEOUTS.get(method, max(COMMAND_TIMEOUTS.values()))
    return base + ORIGIN_PROMPT_WINDOW_S + _CLIENT_TIMEOUT_MARGIN_S


class HostClient:
    """One host's authenticated loopback leg, over that host's discovery file.

    The transport is ~65 lines and it is the same transport for every host: one
    JSON `Request` POSTed to `http://127.0.0.1:<port>/rpc` with the session key
    from the host's own 0600 discovery file, mapped back to a result or a typed
    error. What differs between hosts is WHICH record to read (`store`) and WHICH
    PROCESS the failure copy should name (`host`).

    A subclass rather than a parameter so the extension's call sites keep their
    `BridgeClient()` spelling and therefore keep working unchanged, while the UI
    host gets the same code with a different record and a different voice —
    duplicating this method would mean two places to fix the next transport bug.
    """

    #: Selects both the copy (`HOST_COPY`) and, on the UI side, the message
    #: overrides. Subclasses set it; it is not derived from the store, because a
    #: test's fake store must not be able to change the copy silently.
    host: str = HOST_EXTENSION

    def __init__(self, store: Any = state_store, root: Path | None = None) -> None:
        # The namespace is the STORE's business, never a client parameter: every
        # host's state module fixes its own directory and filename, so a client
        # that could also pass them would be a second place to get them wrong.
        self.store = store
        self.root = root

    def _read(self) -> Any:
        return self.store.read(self.root)

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        copy = HOST_COPY.get(self.host, HOST_COPY[HOST_EXTENSION])
        current = self._read()
        if current is None:
            raise BridgeUnreachable(copy.no_state)
        request_id = f"r-{secrets.token_hex(6)}"
        request = Request(id=request_id, method=method, params=params)
        timeout = client_timeout(method)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                http_response = await client.post(
                    f"http://127.0.0.1:{current.port}/rpc",
                    headers={"X-Bridge-Key": current.session_key},
                    json=request.model_dump(mode="json"),
                )
        except httpx.ConnectTimeout as exc:
            # Timeout WHILE connecting: nothing was ever reached, so this
            # belongs with the unreachable branch below, not the popup one —
            # httpx.ConnectTimeout subclasses TimeoutException, not
            # ConnectError, so ordering matters here (review finding m1).
            raise BridgeUnreachable(copy.not_answering.format(port=current.port)) from exc
        except httpx.TimeoutException as exc:
            # Read/write/pool timeout AFTER connecting: the host accepted the
            # command and never answered within a budget that already covers
            # every legitimate wait (base + prompt window + margin). Calling
            # this "unreachable" sent a QA session (transcript 0ee4974ba84a)
            # into an hour of restarting a healthy daemon while the extension
            # popup sat waiting on the human, so name the likely cause. We only
            # KNOW the connection was accepted — say that, not "running".
            raise BridgeUnreachable(copy.timeout.format(method=method, timeout=timeout)) from exc
        except httpx.RequestError as exc:
            # Everything else — ConnectError (refused), ReadError /
            # RemoteProtocolError (daemon died mid-request), reset — means the
            # daemon is gone or dying, so restart advice is honest here
            # (review finding m2: these must NOT get the popup message).
            raise BridgeUnreachable(copy.not_answering.format(port=current.port)) from exc
        if http_response.status_code == 401:
            raise BridgeUnreachable(copy.rejected_key)
        try:
            response = Response.model_validate(http_response.json())
        except (ValueError, json.JSONDecodeError) as exc:
            raise BridgeUnreachable(
                copy.invalid_response.format(status=http_response.status_code)
            ) from exc
        if not response.ok:
            detail = response.error or ErrorDetail(
                code=ErrorCode.INTERNAL, message="unknown failure"
            )
            raise BridgeError(detail.code, detail.message, detail.data)
        return response.result or {}


class BridgeClient(HostClient):
    """The extension bridge's client. Constructor signature unchanged."""

    host = HOST_EXTENSION

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(state_store, root)

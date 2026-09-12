"""Loopback-only browser bridge daemon.

The extension is the WebSocket client because an MV3 worker cannot listen.
Local Operator sessions remain stateless HTTP callers; the daemon owns the
single extension connection and bounds every command so a dead worker can
never hang a tool call.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    COMMAND_TIMEOUTS,
    ORIGIN_PROMPT_WINDOW_S,
    PROTO_VERSION,
    ErrorCode,
    ErrorDetail,
    Hello,
    HelloAck,
    PairRequest,
    PairResult,
    Request,
    Response,
)
from local_operator.paths import config_dir

logger = logging.getLogger(__name__)
DEFAULT_PORT = 4099
PING_INTERVAL_S = 20.0
#: How long the daemon tolerates TOTAL silence — no frame of any kind — from a
#: TCP-connected extension before it declares the link unproven. Two missed
#: pings plus one interval of slack: the ping loop is `await sleep(20)` then
#: send, so tick spacing drifts with loop load, and the supervisor can add its
#: first 1 s of backoff after a failure.
#:
#: A healthy worker pongs in microseconds from the socket's own onmessage
#: handler (`worker.ts`, `frame.event === "ping"`) — NOT behind any of the
#: extension's serialized queues — and emits nothing else while idle, so its
#: longest legitimate silence is one ping interval. That is the bound that sets
#: this number, and 50 s leaves 2.5x headroom against it.
#:
#: Cross-checked against the client's patience so the session always gets this
#: typed error rather than an HTTP timeout: the tightest session budget is the
#: base command timeout + ORIGIN_PROMPT_WINDOW_S + margin (90 s for `read`).
LINK_SILENCE_TIMEOUT_S = 50.0
#: How long the daemon keeps reporting a link it severed as *unresponsive*
#: rather than absent, once the socket is gone.
#:
#: The teardown ANSWERS with the honest state and nulls the socket in the same
#: breath, so without a latched reason the two facts a session most needs to
#: tell apart — "the browser is open but mute" and "no browser is attached" —
#: become byte-identical one command later, and the second one tells the
#: operator to open a browser that is already open. That is the misdiagnosis
#: this change exists to remove (design D2/R1-5, QA Q1). The latch is what lets
#: the CLI, `/health`, the popup and the next RPC answer honestly for as long as
#: the observation is still true.
#:
#: Sized to the slowest real recovery so it outlives the state it describes:
#: after a signal-4000 close the worker re-dials on its ~1 s fast path, but a
#: worker that suspends inside that window returns on the alarm floor —
#: measured at 58.21 s on real hardware (§5.2). 60 s covers it. A genuinely
#: wedged worker that never re-dials stays honest only for this window, then
#: correctly reads as absent (by then nothing is attached, and it is not).
LINK_DROP_TTL_S = 60.0
#: Ceiling on one write to the extension socket. A loopback `send_json` of a
#: small frame is sub-millisecond; seconds here means a peer that has stopped
#: draining its TCP receive buffer, which is already fatal. Generous enough
#: that a GC pause or a momentarily full buffer cannot trip it, and short
#: enough to be invisible inside a 20-30 s command budget.
#:
#: This wraps `ExtensionLink.send` INCLUDING its lock acquisition, which is what
#: makes a *stuck lock holder* bounded rather than merely a stuck `send_json`.
#: Note `wait_for` cancels the waiter, not the holder — the teardown that
#: follows is what deals with the holder.
LINK_SEND_TIMEOUT_S = 5.0
#: How often the daemon re-reads the pairing file to notice an out-of-process
#: revoke. Short enough that "Unpair" feels immediate, cheap enough to poll.
REVOKE_WATCH_S = 3.0
PAIR_TTL_S = 120.0
PAIR_MAX_ATTEMPTS = 5
PAIRING_FILENAME = "browser/pairing.json"
PENDING_FILENAME = "run/browser/pairing-pending.json"
#: Ceiling on the supervisor's per-failure backoff. The delay is the number of
#: CONSECUTIVE failures in seconds (1s, 2s, 3s …), clamped here: linear rather
#: than exponential on purpose, because these loops recover the moment the
#: cause clears and an exponential delay would keep the bridge unavailable long
#: after the disk drained. A loop whose iteration keeps raising still must not
#: spin a core or flood the log, which is what the clamp guarantees — it is
#: reached after this many consecutive failures, not after a few.
SUPERVISOR_BACKOFF_CAP_S = 30.0
#: Poll granularity for the extendable command wait. A pending future is
#: normally resolved by the receive loop the instant the response lands; this
#: only bounds how quickly a deadline EXTENSION (awaiting_origin) is noticed.
_WAIT_TICK_S = 0.5


def _private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _pairing_path(root: Path | None = None) -> Path:
    return (root or config_dir()) / PAIRING_FILENAME


def _pending_path(root: Path | None = None) -> Path:
    return (root or config_dir()) / PENDING_FILENAME


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def pairing_status(root: Path | None = None) -> dict[str, Any]:
    """Return only display-safe pairing metadata; token hashes stay private."""
    saved = _read_json(_pairing_path(root))
    pending = _read_json(_pending_path(root))
    now = time.time()
    return {
        "paired": saved is not None,
        "extension_id": str(saved.get("extension_id", "")) if saved else "",
        "pending_code": (
            str(pending.get("code", ""))
            if pending and float(pending.get("expires_at", 0)) > now
            else ""
        ),
        "pending_expires_at": float(pending.get("expires_at", 0)) if pending else 0.0,
    }


def reset_pairing(root: Path | None = None) -> None:
    for path in (_pairing_path(root), _pending_path(root)):
        with suppress(OSError):
            path.unlink()


#: Key for driven-tab records that arrive without a surface handle (an older
#: extension build). One reserved key keeps a mixed-version pair reporting a
#: single driven tab rather than one phantom per navigation.
_UNKEYED_TAB = ""

#: Marks a REDACTED surface handle (`state.ts:redactToken` truncates the nonce
#: and appends this). The extension redacts every handle that leaves it other
#: than the caller's own `open` response, so a redacted token can arrive as a
#: command result and must never be treated as a real handle: see
#: :func:`_is_real_handle`.
_REDACTION_MARK = "\u2026"


def _handle_matches_listed(handle: str, listed: str) -> bool:
    """Whether our full ``handle`` names the surface a listing entry describes.

    The `tabs` listing redacts handles (the full token IS the drive capability,
    so listing it would hand every session control of every tab), which still
    leaves enough nonce to recognise one's OWN handle by prefix. This is the
    daemon-side twin of `state.ts:ownsRedacted`; keep the two in step.
    """
    if listed.endswith(_REDACTION_MARK):
        return handle.startswith(listed[: -len(_REDACTION_MARK)])
    return handle == listed


def _is_real_handle(tab: str) -> bool:
    """Whether ``tab`` is a full surface handle that may KEY a driven record.

    A handle-less `status` returns a redacted token (`bridge:7:abcdef\u2026`)
    because an unproven caller must not receive the drive capability. That is
    still a non-empty string, so it was accepted as a handle and became a
    SECOND key for a tab already tracked under its full token — the driven
    count over-reported, and after the tab genuinely closed `note_closed(full)`
    dropped only the full key, leaving the redacted entry advertising a dead
    URL forever. That is exactly the phantom this change exists to remove,
    reintroduced one layer down.

    A redacted handle proves nothing about which tab it names, so it is treated
    as ABSENT: the update refreshes the most recent record instead of forking a
    new one, which is the same safe reading a handle-less update already gets.
    """
    return bool(tab) and not tab.endswith(_REDACTION_MARK)


@dataclass
class DrivenTab:
    """One tab the extension is currently driving, as last reported."""

    url: str
    title: str
    updated_at: float


class ExtensionLink:
    """The one connected extension plus in-flight request correlation."""

    def __init__(self) -> None:
        self.websocket: WebSocket | None = None
        self.extension_id = ""
        self.browser = ""
        self.paired = False
        self.pending: dict[str, asyncio.Future[Response]] = {}
        # Request ids the extension has told us are blocked on a human origin
        # decision, with the origin being asked about. The RPC wait consults
        # this to extend the deadline past the base command timeout (A3), and
        # the popup/status surfaces read it so a pending approval is visible.
        self.awaiting_origin: dict[str, str] = {}
        # Last known URL/title PER DRIVEN TAB, keyed by the extension's surface
        # handle, pushed by the extension so the Connected popup and `status`
        # can show the human WHAT is being driven — the tab is inactive, so the
        # debugger infobar alone is not a signal the user sees (finding U3).
        #
        # Per-tab rather than one global slot: sessions get a tab each (up to
        # MAX_SURFACES), so a single last-writer-wins field showed whichever
        # tab was touched most recently as though it were THE bound tab. When
        # that tab then closed without anything clearing the field, `status`
        # advertised a URL whose tab — and whose server — were long gone, which
        # read to both the user and the agent as a system-wide lock held by a
        # phantom. A dict makes "how many tabs are driven" answerable, and
        # makes closing one tab clear exactly that tab.
        self.driven: dict[str, DrivenTab] = {}
        self.send_lock = asyncio.Lock()
        # Monotonic timestamp of the last frame received from the extension —
        # ANY frame, before any dispatch on its type, so an older extension's
        # frames count too. Set when the socket is accepted (the peer has just
        # spoken: it sent `hello`) and on every frame of the receive loop; reset
        # by `disconnect`. This is the only thing that can tell a healthy idle
        # extension from a mute one, and without it `status` advertised a wedged
        # bridge as connected while every session hung.
        self.last_frame_at = 0.0
        # Why the daemon last severed an attached link, and the silence it had
        # measured when it did. Read as a LATCHED fact that outlives the socket
        # for LINK_DROP_TTL_S, because `disconnect()` (the same teardown used
        # for an ordinary peer close) nulls the socket as part of ANSWERING
        # with the unresponsive state. See LINK_DROP_TTL_S for the whole
        # reasoning; `silent_drop_*` is set only by `_drop_unproven_link`, so a
        # browser that was simply closed never reads as "attached but mute".
        self.silent_drop_at = 0.0
        self.silent_drop_silence_s = 0.0

    @property
    def proven(self) -> bool:
        """Whether the connected socket has recently proven it is listening.

        ``websocket is not None`` is true whenever TCP is up and the peer has
        not closed, so a peer that completes `hello`, pairs, and then never
        speaks again is indistinguishable from a healthy idle one. Never trust
        the bare socket object: a silent link must fail sessions fast and be
        dropped, not be advertised as connected.
        """
        if self.websocket is None or self.last_frame_at <= 0.0:
            return False
        return (time.monotonic() - self.last_frame_at) <= LINK_SILENCE_TIMEOUT_S

    def silent_for(self) -> float:
        """Seconds since the extension last said anything; 0.0 if it never has.

        Read by the promotion rule in ``_dispatch_locked``: a command that
        consumed its whole budget while the link was ALSO silent for 1.5 ping
        intervals is corroborated as a dead peer rather than a slow page.
        """
        if self.last_frame_at <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - self.last_frame_at)

    def note_unproven_drop(self, silence_s: float) -> None:
        """Latch WHY the link was severed, for as long as that stays true."""
        self.silent_drop_at = time.monotonic()
        self.silent_drop_silence_s = silence_s

    def clear_unproven_drop(self) -> None:
        """Forget the latched reason: a link this daemon did not sever.

        Called when the socket ends for any ordinary reason (the worker died,
        the browser closed, the peer was evicted) and when a NEW socket becomes
        authoritative, so the latch can never make a freshly connected bridge
        look mute.
        """
        self.silent_drop_at = 0.0
        self.silent_drop_silence_s = 0.0

    def recent_drop_silence(self) -> float:
        """The silence measured at the last unproven drop, while the latch is live.

        0.0 when no such drop happened or the window has expired. It can also be
        0.0 while the latch IS live (a `link.send` deadline can fire with the peer
        still ponging), so "is it latched" is a separate question — see
        `dropped_unproven`.
        """
        return self.silent_drop_silence_s if self.dropped_unproven() else 0.0

    def dropped_unproven(self) -> bool:
        """Whether "this daemon severed an attached link" is still the truth.

        The discriminator that keeps "attached but not answering" from
        outliving the state it describes: it expires after `LINK_DROP_TTL_S`,
        and every path that ends the link for another reason clears it.
        """
        if self.silent_drop_at <= 0.0:
            return False
        return (time.monotonic() - self.silent_drop_at) <= LINK_DROP_TTL_S

    @property
    def current_url(self) -> str:
        """Most recently driven live tab's URL, or "" when none is driven.

        Kept as a property because /health, the popup, and `status` are an
        established contract; "" now genuinely means "nothing is driven"
        rather than "nobody has updated this field yet".
        """
        latest = self._latest_driven()
        return latest.url if latest else ""

    @property
    def current_title(self) -> str:
        latest = self._latest_driven()
        return latest.title if latest else ""

    def _latest_driven(self) -> DrivenTab | None:
        return max(self.driven.values(), key=lambda tab: tab.updated_at, default=None)

    def note_driven(self, tab: str, url: str, title: str) -> None:
        """Record/refresh one driven tab.

        The handle is absent in two cases, and conflating them would invent
        phantoms — the exact class of bug this change removes:

        - An OLDER extension that only ever sent a bare ``tab_update``. There
          is no handle to be had, so those collapse onto one reserved key and a
          mixed-version pair reports a single driven tab, not one entry per
          navigation.
        - A handle-less command RESULT (``goto`` returns url/title but no
          ``tab``) arriving just after the worker's keyed ``tab_update`` for
          the same navigation. Creating an unkeyed entry there would double-
          count one tab. So when tabs are already tracked, a handle-less update
          REFRESHES the most recent one instead of adding to the map.

        A REDACTED handle counts as absent for both purposes; see
        :func:`_is_real_handle`. The daemon enforces this even though the
        current extension no longer sends one, because the two run independent
        release cycles and an old or third-party build must not be able to
        plant a phantom.
        """
        keyed = _is_real_handle(tab)
        key = tab if keyed else _UNKEYED_TAB
        if not keyed and self.driven:
            # A REDACTED handle may not KEY a record, but it still carries
            # enough nonce to RECOGNISE the one it belongs to, so match it
            # before falling back to recency. The fallback is only safe when
            # nothing names the tab: the two sides count "most recent" on
            # DIFFERENT clocks — `nav.ts` resolves a handle-less command
            # against the most recently USED surface (bumped by every
            # tab-scoped command), while `_latest_driven` sees the most
            # recently UPDATED record — so a handle-less `status` describing
            # tab A could refresh tab B's record with A's URL. Cosmetic (it
            # heals on B's next keyed update and forks no phantom), but the
            # daemon already owns the exact matcher `repair()` uses for this
            # question, so recency is the wrong answer when a handle is here.
            # Only a REDACTED token can name a tab; a truly handle-less update
            # (an old build) stays on the documented recency path, where an
            # existing unkeyed record would otherwise self-match on "".
            matched = (
                next(
                    (key_ for key_ in self.driven if _handle_matches_listed(key_, tab)),
                    None,
                )
                if tab
                else None
            )
            if matched is not None:
                key = matched
            else:
                latest = self._latest_driven()
                key = next((k for k, v in self.driven.items() if v is latest), _UNKEYED_TAB)
        self.driven[key] = DrivenTab(url=url, title=title, updated_at=time.time())

    def note_closed(self, tab: str) -> None:
        """Drop one closed tab, or every tab when the handle is unknown.

        A handle-carrying event drops ONLY that tab, so one session closing its
        tab never blanks another's. It deliberately does NOT also drop the
        unkeyed record: that record belongs to a DIFFERENT, older peer (it only
        exists in a mixed-version pair), and dropping it made a new session
        closing its own tab blank an old extension's still-live entry — the
        docstring promised isolation the code did not deliver. The unkeyed
        record is cleared by its own handle-less close, by `disconnect`, or by
        `repair`; a stale one is self-correcting on the peer's next update.

        A handle-less (or redacted, which proves nothing — see
        :func:`_is_real_handle`) ``tab_closed`` cannot name what went away, so
        it blanks everything: the alternative, keeping entries alive, is
        exactly the phantom this fixes. Callers that CAN name the surface must
        do so — `worker.ts` resolves the sole-surface `close` shape to its real
        handle before announcing, so this clear-all stays what it is documented
        to be: the last resort for a peer that genuinely cannot say.
        """
        if _is_real_handle(tab):
            self.driven.pop(tab, None)
        else:
            self.driven.clear()

    async def send(self, payload: dict[str, Any]) -> None:
        websocket = self.websocket
        if websocket is None:
            raise RuntimeError("extension disconnected")
        async with self.send_lock:
            await websocket.send_json(payload)

    def disconnect(self) -> None:
        self.websocket = None
        self.paired = False
        self.last_frame_at = 0.0
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError("extension disconnected"))
        self.pending.clear()
        self.awaiting_origin.clear()
        # Nothing is driven once the browser is gone: the surfaces live in the
        # extension's session storage and do not outlive the connection.
        self.driven.clear()


class BridgeService:
    def __init__(self, port: int = DEFAULT_PORT, root: Path | None = None) -> None:
        self.port = port
        self.root = root
        self.link = ExtensionLink()
        self.started_at = time.time()
        self.state = state_store.BridgeState(
            pid=os.getpid(),
            port=port,
            session_key=secrets.token_urlsafe(32),
            proto=PROTO_VERSION,
            started_at=self.started_at,
        )
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._revoke_task: asyncio.Task[None] | None = None
        # Consecutive failed discovery-file writes, so recovery can be logged
        # once rather than on every tick (see publish_safely).
        self._publish_failures = 0
        # Per-tab command serialization. v1 is explicitly a SINGLE active
        # browser surface (one extension, one dedicated tab): the design's
        # "session->tab table" is deferred, and instead of silently
        # interleaving two sessions' commands against the one tab (finding A4)
        # the daemon serializes them behind a per-tab lock so each command runs
        # to completion before the next starts. Concurrent sessions therefore
        # SHARE the tab safely rather than clobbering each other's navigation.
        self._tab_locks: dict[str, asyncio.Lock] = {}

    def publish(self) -> None:
        # `proven`, never `websocket is not None` (see ExtensionLink.proven).
        # Every downstream consumer reads this ONE bit: state.liveness() turns
        # it into ABSENT, which makes available()/advertisable() false, which is
        # what stops sessions hanging on a mute peer and falls them back to cmux
        # (or to a typed diagnostic) instead.
        self.state.extension_connected = self.link.proven
        self.state.paired = self.link.paired
        self.state.extension_id = self.link.extension_id
        self.state.browser_name = self.link.browser
        state_store.publish(self.state, self.root)

    def publish_safely(self) -> bool:
        """Publish discovery state, absorbing a failed write instead of raising.

        Every event-driven caller (pairing, connect, disconnect, tab updates)
        used to publish inline and unguarded. A single failed write there took
        down whichever coroutine happened to be running, and on the heartbeat
        path it killed the only task that refreshes the file — after which
        ``state.available()`` was false for EVERY session on the machine, for
        the rest of the daemon's life, while ``/health`` kept answering 200.
        That contradiction is the whole incident (a full disk raised ENOSPC out
        of ``tempfile.mkstemp``; nothing restarted the writer or logged that it
        had gone).

        Publishing is a best-effort CACHE refresh, never a correctness
        requirement: the daemon's authoritative state lives in memory and is
        served by ``/health``. So a failed write is logged and swallowed, and
        the next heartbeat tick retries — which is what makes recovery
        automatic once the disk drains.
        """
        try:
            self.publish()
            if self._publish_failures:
                logger.warning(
                    "browser bridge state file writable again after %d failed attempt(s)",
                    self._publish_failures,
                )
                self._publish_failures = 0
            return True
        except OSError as error:
            self._publish_failures += 1
            # ENOSPC is the one a user can actually act on, and it is what
            # bit this machine, so it gets its own actionable line rather than
            # being buried in a generic write failure.
            if error.errno == errno.ENOSPC:
                logger.error(
                    "browser bridge cannot write %s: the disk is full. Sessions will fall "
                    "back to cmux until space is freed; the daemon keeps serving /health "
                    "and recovers on its own once the write succeeds.",
                    state_store.state_path(self.root),
                )
            else:
                logger.warning(
                    "browser bridge state publish failed (attempt %d); retrying next tick",
                    self._publish_failures,
                    exc_info=True,
                )
        except Exception:  # noqa: BLE001 - a cache refresh may never kill a loop
            self._publish_failures += 1
            logger.warning(
                "browser bridge state publish failed unexpectedly (attempt %d)",
                self._publish_failures,
                exc_info=True,
            )
        return False

    async def _supervise(self, name: str, body: Callable[[], Awaitable[None]]) -> None:
        """Run one iteration-based background loop forever, come what may.

        The three background loops here are LIVENESS infrastructure: if one
        exits, the daemon does not crash and nothing notices — it just quietly
        stops doing its job, which is strictly worse than a crash because the
        process keeps answering /health as though it were healthy. That is how
        a full disk turned into "every session falls back to cmux forever while
        status says the extension is connected".

        So no per-iteration exception may end a loop. Cancellation still ends
        it promptly (shutdown depends on that), and a genuinely persistent
        failure is rate-limited in the log rather than spun on — a tight retry
        loop against a broken syscall would burn a core and flood the log.
        """
        failures = 0
        while True:
            try:
                await body()
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a supervisory loop must not die
                failures += 1
                logger.warning(
                    "browser bridge %s loop iteration failed (%d consecutive); continuing",
                    name,
                    failures,
                    exc_info=True,
                )
                # Back off one second per consecutive failure so a synchronous
                # failure cannot become a busy loop, clamped at
                # SUPERVISOR_BACKOFF_CAP_S. Linear, so recovery stays prompt.
                await asyncio.sleep(min(SUPERVISOR_BACKOFF_CAP_S, float(failures)))

    async def _heartbeat_tick(self) -> None:
        self.publish_safely()
        await asyncio.sleep(state_store.HEARTBEAT_INTERVAL_S)

    async def _ping_tick(self) -> None:
        await asyncio.sleep(PING_INTERVAL_S)
        if self.link.websocket is None:
            return
        if not self.link.proven:
            # Total silence for two ping intervals: not a slow peer, a dead
            # one. The check lives HERE because this loop already ticks every
            # PING_INTERVAL_S whether or not a session is active, so an idle
            # bridge still notices instead of waiting for the next command.
            await self._drop_unproven_link(
                f"no frame for {self.link.silent_for():.0f}s "
                f"(deadline {LINK_SILENCE_TIMEOUT_S:.0f}s)"
            )
            return
        try:
            # Bounded for the same reason as every other send: a ping that
            # parks silently disarms the very detector this design rests on.
            await asyncio.wait_for(self.link.send({"event": "ping"}), timeout=LINK_SEND_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self._drop_unproven_link("ping send exceeded its deadline")
        except Exception:  # noqa: BLE001 - receive loop owns teardown
            logger.debug("browser extension ping failed", exc_info=True)

    async def _drop_unproven_link(self, reason: str) -> None:
        """Sever a link the daemon no longer trusts, and say why ONCE.

        Deliberately the SAME teardown the ordinary disconnect path uses
        (``link.disconnect()``), with a new trigger rather than a new mechanism:
        every pending future is failed with ``extension disconnected`` (which
        the dispatch site turns into a typed EXTENSION_DISCONNECTED response),
        and ``awaiting_origin``/``driven`` are cleared — so every waiting
        session gets a typed answer now instead of timing out.

        Close code 4000 is the whole wire-compatibility story: an
        already-installed worker handles 4000 explicitly, so it resets its
        state and arms its ~1 s fast-path reconnect rather than reading the
        close as a lost pairing. A code the old worker does not understand
        would be treated as an ordinary close (which is why 4000, already
        meaningful to every released build, is the right one).
        """
        websocket = self.link.websocket
        logger.warning("browser bridge dropped an unresponsive extension: %s", reason)
        # Latch WHY, and how long the peer had been quiet, BEFORE the teardown
        # nulls the socket: this is the answer `_drop_unproven_link` is about to
        # give, and without the latch it would be gone by the time anyone (the
        # next RPC, `/health`, the CLI or the popup) could read it.
        self.link.note_unproven_drop(self.link.silent_for())
        # Clear the link BEFORE the close, not after. `disconnect()` is what
        # fails every pending future, i.e. what turns three waiting sessions into
        # three typed answers; `websocket.close()` is a SEND, so it can block on
        # a peer that has stopped draining — the same hazard as any other
        # unbounded write on this socket. Ordering the close first would make the
        # teardown itself the next thing that can hang, which is the class of bug
        # this whole change exists to remove. The close still goes out, and a
        # peer that never sees it is already the disconnected path.
        self.link.disconnect()
        self.publish_safely()
        if websocket is not None:
            with suppress(Exception):
                await websocket.close(code=4000)

    async def _heartbeat(self) -> None:
        await self._supervise("heartbeat", self._heartbeat_tick)

    async def _ping(self) -> None:
        await self._supervise("ping", self._ping_tick)

    def _live_pairing_matches(self) -> bool:
        """Whether the ON-DISK pairing still authorizes the connected extension.

        Read from disk, never from ``self.link.paired`` alone, because
        ``lop browser pair --reset`` runs in a SEPARATE process and can only
        touch the file (findings A5/U1). A revoke there must take authority
        away from an already-connected socket immediately, not merely at the
        next reconnect, so the gate and the watcher both consult the file.
        """
        saved = _read_json(_pairing_path(self.root))
        return bool(saved and saved.get("extension_id") == self.link.extension_id)

    async def revoke(self) -> None:
        """Drop the pairing AND cut the live connection.

        Flipping ``paired`` false is not enough on its own: an open socket the
        extension already holds would keep delivering RPCs until it happened to
        disconnect. So this closes the socket too, which is what makes the
        popup's \"take this back any time\" and the CLI's \"revoked\" promise real.
        """
        reset_pairing(self.root)
        self.link.paired = False
        # An unpair is not a wedge: forget any latched unresponsive reason, or a
        # deliberate revoke would keep reading as "attached but not answering".
        self.link.clear_unproven_drop()
        websocket = self.link.websocket
        if websocket is not None:
            with suppress(Exception):
                # 4003 = unpaired, the same code the handshake uses so the
                # popup renders \"waiting to pair\" rather than a mystery drop.
                await websocket.close(code=4003)
        self.link.disconnect()
        self.publish_safely()

    async def _revocation_tick(self) -> None:
        await asyncio.sleep(REVOKE_WATCH_S)
        if (
            self.link.websocket is not None
            and self.link.paired
            and not self._live_pairing_matches()
        ):
            logger.info("pairing revoked on disk; closing the live extension socket")
            await self.revoke()

    async def _watch_revocation(self) -> None:
        """Poll the pairing file so an out-of-process revoke severs a live link.

        Cheap (one stat-and-parse every few seconds) and only acts on the
        transition from paired-with-file to paired-without-file, so it never
        fights the handshake that is mid-flight. Supervised: a transient read
        error must not silently disarm revocation for the daemon's lifetime,
        which would leave a revoked browser able to drive until it reconnected.
        """
        await self._supervise("revocation-watch", self._revocation_tick)

    async def startup(self) -> None:
        # Startup publishes through the guarded path too: a daemon that cannot
        # write its discovery file on a full disk must still boot and serve
        # /health, so that `lop browser status` and the tool's socket probe can
        # both still reach it and report the truth.
        self.publish_safely()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        self._ping_task = asyncio.create_task(self._ping())
        self._revoke_task = asyncio.create_task(self._watch_revocation())

    async def shutdown(self) -> None:
        for task in (self._heartbeat_task, self._ping_task, self._revoke_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self.link.disconnect()
        state_store.remove(self.root)

    def _origin_extension_id(self, websocket: WebSocket) -> str:
        origin = websocket.headers.get("origin", "")
        prefix = "chrome-extension://"
        if not origin.startswith(prefix):
            return ""
        extension_id = origin[len(prefix) :]
        # Chromium IDs are 32 lowercase a-p characters. A strict check stops a
        # web origin from smuggling slashes or suffixes into the pinned value.
        if len(extension_id) != 32 or any(char not in "abcdefghijklmnop" for char in extension_id):
            return ""
        return extension_id

    def _rotate_pending(self, extension_id: str) -> None:
        """Mint a brand-new code, unconditionally invalidating any prior one.

        Used when the attempt cap is reached or the code expires: rotating is
        what turns the documented 5-guess limit into a real lockout — the
        exhausted code stops working the instant a fresh one is issued, so a
        local brute-force process cannot keep guessing the same secret until
        the TTL lapses (finding A1).
        """
        _private_write(
            _pending_path(self.root),
            {
                "extension_id": extension_id,
                "code": f"{secrets.randbelow(1_000_000):06d}",
                "expires_at": time.time() + PAIR_TTL_S,
                "attempts": 0,
            },
        )

    def _ensure_pending(self, extension_id: str) -> None:
        """Guarantee a live pending code exists, reusing a valid one.

        A code is reused only while it is unexpired AND still under the attempt
        cap; anything else rotates. The cap check here is the second half of
        the A1 fix: even if a caller forgets to rotate on cap, an exhausted
        code is never handed back out as "still live".
        """
        pending = _read_json(_pending_path(self.root))
        if (
            pending
            and pending.get("extension_id") == extension_id
            and float(pending.get("expires_at", 0)) > time.time()
            and int(pending.get("attempts", 0)) < PAIR_MAX_ATTEMPTS
        ):
            return
        self._rotate_pending(extension_id)

    def _valid_saved_token(self, extension_id: str, token: str) -> bool:
        saved = _read_json(_pairing_path(self.root))
        if not saved or saved.get("extension_id") != extension_id or not token:
            return False
        digest = hashlib.sha256(token.encode()).hexdigest()
        return secrets.compare_digest(str(saved.get("token_sha256", "")), digest)

    async def _try_pair(self, request: PairRequest) -> PairResult:
        pending = _read_json(_pending_path(self.root))
        if not pending or pending.get("extension_id") != self.link.extension_id:
            self._ensure_pending(self.link.extension_id)
            return PairResult(ok=False, message="No live pairing code. Run lop browser pair again.")
        attempts = int(pending.get("attempts", 0)) + 1
        expired = float(pending.get("expires_at", 0)) <= time.time()
        matches = secrets.compare_digest(str(pending.get("code", "")), request.code)
        if expired or attempts >= PAIR_MAX_ATTEMPTS or not matches:
            # Reaching the cap (or expiry) rotates to a fresh code, so the
            # guessed-at code is dead the moment this branch runs — the
            # lockout the design promised. A wrong guess still under the cap
            # persists the incremented counter so the cap is actually reached
            # (the previous code saturated the stored count at 4 and never
            # rotated — finding A1). ``attempts >= cap`` on this, the cap-th
            # failure, is deliberate: the cap-th wrong guess is the last one.
            if attempts >= PAIR_MAX_ATTEMPTS or expired:
                self._rotate_pending(self.link.extension_id)
                message = (
                    "Too many attempts. That code is now dead — run 'lop browser "
                    "pair' for a fresh one."
                    if not expired
                    else "That code expired. Run 'lop browser pair' for a fresh one."
                )
            else:
                pending["attempts"] = attempts
                _private_write(_pending_path(self.root), pending)
                # Two lines at the popup's 300px width, not three. The popup
                # reserves a fixed slot for this message so a failed attempt
                # cannot resize the window (extension/src/popup/popup.css), and
                # the reservation is sized to the LONGEST string that can land
                # in it — so every word here costs vertical space on a card that
                # is showing no error at all. Kept byte-identical to the
                # extension's own PAIR_MISMATCH_MESSAGE fallback, which is the
                # string a user sees when the daemon sends none.
                message = "That code didn't match. Codes expire after two minutes — check the app."
            return PairResult(ok=False, message=message)
        token = secrets.token_urlsafe(32)
        _private_write(
            _pairing_path(self.root),
            {
                "extension_id": self.link.extension_id,
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "paired_at": time.time(),
            },
        )
        with suppress(OSError):
            _pending_path(self.root).unlink()
        self.link.paired = True
        self.publish_safely()
        return PairResult(ok=True, token=token)

    async def extension(self, websocket: WebSocket) -> None:
        extension_id = self._origin_extension_id(websocket)
        if not extension_id:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        try:
            raw = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            hello = Hello.model_validate(raw)
        except (asyncio.TimeoutError, ValidationError, ValueError):
            await websocket.close(code=4001)
            return
        if hello.proto != PROTO_VERSION:
            await websocket.close(code=4001)
            return
        saved = _read_json(_pairing_path(self.root))
        if saved and saved.get("extension_id") != extension_id:
            await websocket.close(code=4004)
            return

        # A later extension wins. This prevents two browser profiles from both
        # receiving commands while preserving reconnect after worker death.
        if self.link.websocket is not None:
            with suppress(Exception):
                await self.link.websocket.close(code=4000)
            self.link.disconnect()
        self.link.websocket = websocket
        self.link.extension_id = extension_id
        self.link.browser = hello.browser
        # A fresh authoritative socket supersedes any latched drop reason from
        # the link it just replaced (including the "later connection wins"
        # eviction above), so a reconnect cannot inherit a mute label.
        self.link.clear_unproven_drop()
        # The peer has just proven it is listening by sending `hello`; stamp it
        # so `proven` is true from the first instant of the connection instead
        # of waiting for the first pong.
        self.link.last_frame_at = time.monotonic()
        self.link.paired = self._valid_saved_token(extension_id, hello.token)
        if not self.link.paired:
            self._ensure_pending(extension_id)
        self.publish_safely()
        await self.link.send(HelloAck(paired=self.link.paired).model_dump(mode="json"))
        try:
            while True:
                frame = await websocket.receive_json()
                # ANY frame, recorded BEFORE any dispatch on its type: this is
                # the liveness signal `proven` reads, and taking it here (rather
                # than inside each event branch) is what makes an older
                # extension that only pongs count as alive.
                self.link.last_frame_at = time.monotonic()
                if frame.get("event") == "pair":
                    try:
                        pair = PairRequest.model_validate(frame)
                    except ValidationError:
                        continue
                    result = await self._try_pair(pair)
                    await self.link.send(result.model_dump(mode="json"))
                    continue
                if frame.get("event") == "awaiting_origin":
                    # The extension paused this request on a human origin
                    # decision. Record it so the RPC wait extends its deadline
                    # (A3) and the popup/status can show what is pending (U2).
                    request_id = str(frame.get("id", ""))
                    if request_id:
                        self.link.awaiting_origin[request_id] = str(frame.get("origin", ""))
                        self.publish_safely()
                    continue
                if frame.get("event") == "awaiting_origin_cleared":
                    # The extension's queue entry for this command is gone
                    # (decided, cancelled, or expired) without a response the
                    # daemon will see. Drop the record so /health stops echoing
                    # a prompt the popup can no longer resolve — the stale echo
                    # is what looped the approval popup on "Request changed."
                    request_id = str(frame.get("id", ""))
                    if request_id and request_id in self.link.awaiting_origin:
                        self.link.awaiting_origin.pop(request_id, None)
                        self.publish_safely()
                    continue
                if frame.get("event") == "unpair":
                    # The options page "Unpair this browser" reaches the daemon
                    # here so revocation severs THIS live socket, mirroring the
                    # CLI --reset path (findings A5/U1).
                    await self.revoke()
                    return
                if frame.get("event") == "tab_update":
                    # Pushed by the extension on navigation so the popup reflects
                    # the driven site promptly even between commands (U3).
                    self.link.note_driven(
                        str(frame.get("tab", "")),
                        str(frame.get("url", "")),
                        str(frame.get("title", "")),
                    )
                    self.publish_safely()
                    continue
                if frame.get("event") == "tab_closed":
                    # The extension reports the CLOSED SURFACE by handle, so
                    # only that tab is dropped: with several sessions driving a
                    # tab each, blanking everything on one close (as this did)
                    # would have reported the survivors as gone.
                    self.link.note_closed(str(frame.get("tab", "")))
                    self.publish_safely()
                    continue
                if frame.get("event") in ("pong", "origin_decision"):
                    continue
                try:
                    response = Response.model_validate(frame)
                except ValidationError:
                    continue
                # Any successful command carrying a live url/title updates the
                # cached driven-page shown in the popup (U3).
                if response.ok and response.result:
                    url = response.result.get("url")
                    if isinstance(url, str) and url:
                        title = response.result.get("title")
                        handle = response.result.get("tab")
                        self.link.note_driven(
                            handle if isinstance(handle, str) else "",
                            url,
                            title if isinstance(title, str) else "",
                        )
                self.link.awaiting_origin.pop(response.id, None)
                future = self.link.pending.pop(response.id, None)
                if future is not None and not future.done():
                    future.set_result(response)
        except WebSocketDisconnect:
            pass
        finally:
            if self.link.websocket is websocket:
                # The peer ended this link itself (worker died, browser closed,
                # tab torn down). That is NOT the unresponsive path — the daemon
                # did not sever it — so drop any latched unresponsive reason and
                # let the honest "not currently attached" state stand.
                self.link.clear_unproven_drop()
                self.link.disconnect()
                self.publish_safely()

    async def rpc(self, http_request: HttpRequest) -> JSONResponse:
        supplied = http_request.headers.get("x-bridge-key", "")
        if not secrets.compare_digest(supplied, self.state.session_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            request = Request.model_validate(await http_request.json())
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=422)
        if request.method == "ping":
            return JSONResponse({"id": request.id, "ok": True, "result": {"pong": True}})
        if self.link.websocket is None:
            # A link THIS daemon severed for silence must not answer with
            # EXTENSION_DISCONNECTED's "no browser is attached; ask the user to
            # open their browser" for the LINK_DROP_TTL_S cooling-off window.
            # The browser IS attached (and re-dialling); telling the operator to
            # open it re-creates, one step later, exactly the misdirection this
            # change exists to remove (design D1/D2). A peer that REALLY closed
            # never latches the reason (`clear_unproven_drop` runs on every
            # ordinary socket end), so it still gets the honest absent copy.
            dropped_for = self.link.recent_drop_silence()
            if self.link.dropped_unproven():
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_UNRESPONSIVE,
                    "the browser extension is attached but has not re-dialled since the "
                    "bridge dropped its unresponsive link",
                    {"phase": "dropped", "link_silent_s": dropped_for},
                )
            return self._error_response(
                request.id, ErrorCode.EXTENSION_DISCONNECTED, "extension not connected"
            )
        if not self.link.proven:
            # A socket that is TCP-open but has said nothing for
            # LINK_SILENCE_TIMEOUT_S. Refuse in milliseconds rather than let
            # the command burn its whole budget on a peer nobody is home at,
            # AND drop the link so the extension re-dials instead of staying
            # wedged. This is the gate that turns "everything hangs" into
            # "typed error immediately" even before a ping tick has noticed.
            silent = self.link.silent_for()
            await self._drop_unproven_link(f"command refused: link silent for {silent:.0f}s")
            return self._error_response(
                request.id,
                ErrorCode.EXTENSION_UNRESPONSIVE,
                "the browser extension stopped answering",
                {"phase": "gate", "link_silent_s": silent},
            )
        # Re-validate against the on-disk record, not just the in-memory flag:
        # a separate-process ``pair --reset`` must fail in-flight and subsequent
        # RPCs immediately, even before the revocation watcher's next tick
        # (findings A5/U1). If the file is gone, sever the socket now too.
        if not self.link.paired or not self._live_pairing_matches():
            if self.link.paired:
                await self.revoke()
            return self._error_response(request.id, ErrorCode.NOT_PAIRED, "extension is not paired")
        if request.method not in COMMAND_TIMEOUTS:
            return self._error_response(
                request.id, ErrorCode.INTERNAL, f"unknown method: {request.method}"
            )
        if request.id in self.link.pending:
            return self._error_response(request.id, ErrorCode.BUSY, "request id already in flight")
        # Serialize per tab so concurrent sessions cannot interleave commands on
        # the same surface (finding A4); commands on DIFFERENT tabs run in
        # parallel, which is what lets each session drive its own surface.
        # Commands that name no tab (open, status, tabs) serialize on a shared
        # key so a fresh open cannot race another open past the surface cap, and
        # a listing cannot interleave with an open's map write.
        #
        # The access-flow methods never join the global key: they touch no tab
        # and no surface map, and on the global key an await_access slice (up
        # to 20 s of polling a human's decision) would block every session's
        # open behind a wait on a human.
        #
        # await_access takes NO shared lock at all (a per-request key that
        # nothing else uses): its extension side only READS state, and every
        # extension-side mutation is serialized by the worker's own session-
        # mutation queue (state.ts withSessionMutation), so daemon-side
        # serialization adds nothing. Sharing __access__ with request_access
        # was round-2 M3: one waiting session queued every other session's
        # request/replace behind its 20 s slice, defeating the supersession
        # design and stacking waiters toward the HTTP timeout.
        #
        # request_access keeps a shared short key: raise/replace is a
        # read-modify-write of the single prompt slot, and two concurrent
        # raises interleaving daemon-side would make the supersession receipts
        # nondeterministic. It never waits on a human, so the hold is ms.
        return await self._dispatch_serialized(request)

    @staticmethod
    def lock_key_for(request: Request) -> str:
        """The serialization key one RPC dispatches under (see the comment
        above; a separate method so the lock topology is directly testable)."""
        if request.method == "await_access":
            return f"__await__:{request.id}"
        if request.method == "request_access":
            return "__access__"
        return str(request.params.get("tab") or "__global__")

    async def _dispatch_serialized(self, request: Request) -> JSONResponse:
        tab_key = self.lock_key_for(request)
        lock = self._tab_locks.setdefault(tab_key, asyncio.Lock())
        # A per-request await key is used exactly once (request ids are
        # unique) and MUST be evicted on every exit path — success, typed
        # error, timeout, AND cancellation (client disconnect surfaces as
        # CancelledError inside the HTTP handler). Evicting only after a
        # normal return was round-3 M1: a cancelled parked await propagated
        # past the eviction line and permanently retained its unique key, so
        # repeated disconnects grew the lock map without bound. The `finally`
        # block covers all exits; the pop is unconditional because no other
        # request can ever share a per-request key.
        if request.method == "await_access":
            try:
                async with lock:
                    return await self._dispatch_locked(request)
            finally:
                self._tab_locks.pop(tab_key, None)
        async with lock:
            response = await self._dispatch_locked(request)
        # Per-tab keys used to be near-singleton; with every opened tab minting
        # a token the map would now grow for the daemon's lifetime. Evict the
        # key once its tab is closed and nothing is waiting on the lock —
        # unlocked-and-unwaited means a later command for the same (now dead)
        # handle can safely mint a fresh Lock.
        if request.method == "close" and tab_key != "__global__" and not lock.locked():
            self._tab_locks.pop(tab_key, None)
        return response

    async def _dispatch_locked(self, request: Request) -> JSONResponse:
        future: asyncio.Future[Response] = asyncio.get_running_loop().create_future()
        self.link.pending[request.id] = future
        try:
            # Bounded INCLUDING the ``send_lock`` acquisition inside
            # ``link.send``: a stuck lock holder used to hang the RPC past every
            # deadline in COMMAND_TIMEOUTS while holding this tab's lock key,
            # queueing every other command for the same key behind it.
            await asyncio.wait_for(
                self.link.send(request.model_dump(mode="json")), timeout=LINK_SEND_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            # Never left the daemon. Distinct from "delivered, no answer" and
            # reported as such, so the diagnostic distinguishes "could not
            # deliver" from "no reply".
            #
            # Drop OUR future before the teardown: the teardown fails every
            # pending future with RuntimeError, and this one is never awaited
            # (we return a typed error instead), so leaving it registered would
            # produce an unretrieved-exception warning at collection time.
            self.link.pending.pop(request.id, None)
            silent = self.link.silent_for()
            await self._drop_unproven_link(
                f"send of {request.method} exceeded {LINK_SEND_TIMEOUT_S:.0f}s"
            )
            return self._error_response(
                request.id,
                ErrorCode.EXTENSION_UNRESPONSIVE,
                f"{request.method} could not be delivered to the browser extension",
                {"phase": "send", "link_silent_s": silent},
            )
        except Exception as exc:  # noqa: BLE001 - transport failure becomes typed wire error
            self.link.pending.pop(request.id, None)
            return self._error_response(request.id, ErrorCode.EXTENSION_DISCONNECTED, str(exc))
        try:
            response = await self._await_response(
                request.id, future, COMMAND_TIMEOUTS[request.method]
            )
            return JSONResponse(response.model_dump(mode="json", exclude_none=True))
        except asyncio.TimeoutError:
            silent = self.link.silent_for()
            # Promotion rule: a command that consumed its whole budget while the
            # link was ALSO silent for 1.5 ping intervals is a dead peer, not a
            # slow page. The slack is the record's own derivation for the
            # sibling threshold — one tick can be dropped or delayed, tick
            # spacing drifts with loop load, and `_supervise` inserts up to
            # SUPERVISOR_BACKOFF_CAP_S of backoff after a failed tick, all of
            # which make a HEALTHY peer's silence a sawtooth that can exceed a
            # bare PING_INTERVAL_S (review R1-2 reproduced a healthy link torn
            # down, close 4000, every session's pending futures failed). Zero
            # slack here was strictly more aggressive than the 2.5× silence
            # threshold it corroborates with, which is backwards.
            #
            # 1.5× covers a dropped tick plus the first backoff step (~21 s of
            # spacing). Beyond that the DAEMON's own ping loop is the delayed
            # party, and a genuine wedge has the rpc gate's 50 s detector and
            # the full candidate length to trip on. A slow SITE still cannot
            # trip this: the extension is ponging throughout, so silent_for()
            # stays ~0.
            if silent > PING_INTERVAL_S * 1.5:
                await self._drop_unproven_link(
                    f"{request.method} unanswered with the link silent for {silent:.0f}s"
                )
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_UNRESPONSIVE,
                    f"{request.method} was delivered but the extension stopped answering",
                    {"phase": "response", "link_silent_s": silent},
                )
            code = (
                ErrorCode.NAV_TIMEOUT if request.method in ("open", "goto") else ErrorCode.INTERNAL
            )
            return self._error_response(
                request.id,
                code,
                f"{request.method} timed out",
                {"timeout_s": COMMAND_TIMEOUTS[request.method]},
            )
        except Exception as exc:  # noqa: BLE001 - transport failure becomes typed wire error
            return self._error_response(request.id, ErrorCode.EXTENSION_DISCONNECTED, str(exc))
        finally:
            self.link.pending.pop(request.id, None)
            self.link.awaiting_origin.pop(request.id, None)

    async def _await_response(
        self, request_id: str, future: asyncio.Future[Response], base_timeout: float
    ) -> Response:
        """Wait for a command's response with a deadline that EXTENDS while the
        extension is blocked on a human origin decision.

        Without the extension, a first-visit navigation could sit on the
        approval popup for up to 60 s while the base 25–30 s command timeout
        fired underneath it — reporting failure to the session while the tab
        went on to navigate once the user finally clicked Allow (finding A3).
        The extension announces the block with an ``awaiting_origin`` event;
        while that is set for this id, the deadline rides the longer prompt
        window so the human's answer, not a stopwatch, decides the outcome.
        The extension still enforces its own 60 s deny, so this never waits
        forever — that deny arrives as a typed ORIGIN_DENIED response.
        """
        deadline = time.monotonic() + base_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if request_id in self.link.awaiting_origin:
                    # Blocked on a human: push the deadline out to the prompt
                    # window rather than failing a decision in progress.
                    deadline = time.monotonic() + ORIGIN_PROMPT_WINDOW_S
                    continue
                raise asyncio.TimeoutError
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future), timeout=min(remaining, _WAIT_TICK_S)
                )
            except asyncio.TimeoutError:
                if future.done():
                    return future.result()
                continue

    @staticmethod
    def _error_response(
        request_id: str,
        code: ErrorCode,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> JSONResponse:
        response = Response(
            id=request_id,
            ok=False,
            error=ErrorDetail(code=code, message=message, data=data or {}),
        )
        return JSONResponse(response.model_dump(mode="json", exclude_none=True))

    async def health(self, _request: HttpRequest) -> JSONResponse:
        # current_url/pending_origin let the popup render the Connected site
        # (U3) and any in-flight approval (U2) without a separate RPC.
        pending = sorted(set(self.link.awaiting_origin.values()))
        connected = self.link.proven
        return JSONResponse(
            {
                "status": "ok",
                "proto": PROTO_VERSION,
                # `proven`, never `websocket is not None` (see
                # ExtensionLink.proven): a mute peer is NOT connected, and this
                # one bit is what the popup, `lop browser status` and
                # backend._health_ok all read.
                "extension_connected": connected,
                "paired": self.link.paired,
                "browser": self.link.browser,
                "current_url": self.link.current_url,
                "current_title": self.link.current_title,
                # Additive OPTIONAL fields (HTTP, not the WS protocol, so an old
                # client that does not know them simply ignores them). They let
                # the CLI AND the popup tell "no browser" from "browser present
                # but mute" directly instead of inferring it from one boolean.
                #
                # `extension_unresponsive` covers BOTH halves of that state:
                # a socket still attached and mute, and the LINK_DROP_TTL_S
                # window after the daemon severed it for silence. Without the
                # second half the honest line survives only until the teardown
                # that ANSWERS with it, i.e. exactly one command — and the user
                # who runs `lop browser status` afterwards, as the guide tells
                # them to, reads "browser not currently attached" about a browser
                # that is open (design D2/R1-5, QA Q1).
                "extension_unresponsive": (self.link.websocket is not None and not connected)
                or self.link.dropped_unproven(),
                # Whether a link is attached RIGHT NOW, which is not the same as
                # healthy (`extension_connected` owns that). It exists so the
                # status line can word the same observation truthfully in both
                # halves: the bridge WILL drop a mute-but-attached link, and HAS
                # dropped one that is only latched (design D3).
                "link_attached": self.link.websocket is not None,
                # Live silence while a socket is up; otherwise the silence
                # measured at the drop that latched the state, so the number
                # does not collapse to 0 the instant the daemon acts on it.
                "link_silent_s": (
                    self.link.silent_for()
                    if self.link.websocket is not None
                    else self.link.recent_drop_silence()
                ),
                # How many tabs are driven, and their URLs. `current_url` alone
                # framed a multi-tab world as one binding, so a stale value
                # read as a system-wide lock; the count lets `status` say "no
                # tabs driven" (or name all of them) truthfully.
                "driven_tabs": [
                    {"url": tab.url, "title": tab.title}
                    for tab in sorted(
                        self.link.driven.values(), key=lambda t: t.updated_at, reverse=True
                    )
                ],
                "pending_origin": pending[0] if pending else "",
            }
        )

    async def repair(self, _request: HttpRequest) -> JSONResponse:
        """Reconcile advertised state against reality, and report what changed.

        The incident left the user with no way to say "clear whatever you think
        you are holding": the daemon advertised a driven tab that no longer
        existed and a heartbeat that had stopped, and the only remedy anyone
        could think of was killing a healthy daemon.

        Safe while sessions are live, by construction: it asks the EXTENSION
        which surfaces really exist and drops only the records that reality
        does not back. It never closes a tab, never touches pairing, and never
        cancels an in-flight command — so a session driving a live tab keeps
        driving it. Unauthenticated like /health because it is loopback-only
        and confers no drive capability.
        """
        cleaned: list[str] = []
        before = dict(self.link.driven)
        if self.link.proven and before:
            # The extension's own live-surface listing is the ground truth;
            # anything we advertise that it does not list is a ghost. `tabs`
            # prunes dead surfaces extension-side as it lists, so this also
            # reclaims handles leaked by a session that died without `close`.
            try:
                request = Request(id=f"repair-{secrets.token_hex(4)}", method="tabs", params={})
                raw = await self._dispatch_serialized(request)
                payload: Any = json.loads(bytes(raw.body).decode("utf-8"))
                listed = (payload.get("result") or {}).get("tabs", []) if payload.get("ok") else []
                entries = [entry for entry in listed if isinstance(entry, dict)]
                live_handles = [str(entry.get("tab", "")) for entry in entries]
                live_urls = {str(entry.get("url", "")) for entry in entries}
                for key, tab in before.items():
                    # Match on the SURFACE HANDLE, which is stable, rather than
                    # on the URL, which is not: the listing reads chrome.tabs at
                    # call time, so a tab that navigated after its last
                    # tab_update (or is still resolving a redirect) presents a
                    # different URL and was dropped as a phantom while genuinely
                    # live. Harmless-but-wrong is still wrong for a verb
                    # advertised as safe to run while sessions are driving.
                    if _is_real_handle(key):
                        alive = any(
                            _handle_matches_listed(key, listed_handle)
                            for listed_handle in live_handles
                        )
                    else:
                        # The unkeyed record (an older extension) has no handle
                        # to match on, so the URL remains the only signal.
                        alive = not tab.url or tab.url in live_urls
                    if not alive:
                        self.link.driven.pop(key, None)
                        cleaned.append(tab.url)
            except Exception:  # noqa: BLE001 - repair must never fail loudly
                logger.warning("browser bridge repair could not list tabs", exc_info=True)
        elif before:
            # No browser attached, OR one attached that has stopped answering:
            # nothing can be driven either way, so every record is a ghost by
            # definition. Reading `proven` here (not the bare socket) is what
            # keeps repair's answer truthful about a mute peer instead of
            # dispatching a `tabs` RPC that now fails fast and clearing nothing.
            cleaned = [tab.url for tab in before.values() if tab.url]
            self.link.driven.clear()
        republished = self.publish_safely()
        return JSONResponse(
            {
                "status": "ok",
                "cleared_tabs": cleaned,
                "driven_tabs": len(self.link.driven),
                "heartbeat_republished": republished,
                "extension_connected": self.link.proven,
            }
        )


def create_app(port: int = DEFAULT_PORT, root: Path | None = None) -> Starlette:
    service = BridgeService(port, root)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await service.startup()
        try:
            yield
        finally:
            await service.shutdown()

    app = Starlette(
        routes=[
            Route("/health", service.health, methods=["GET"]),
            Route("/repair", service.repair, methods=["POST"]),
            Route("/rpc", service.rpc, methods=["POST"]),
            WebSocketRoute("/extension", service.extension),
        ],
        lifespan=lifespan,
    )
    app.state.bridge = service
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Operator browser bridge daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.port), host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

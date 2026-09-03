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
        self.state.extension_connected = self.link.websocket is not None
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
        if self.link.websocket is not None:
            try:
                await self.link.send({"event": "ping"})
            except Exception:  # noqa: BLE001 - receive loop owns teardown
                logger.debug("browser extension ping failed", exc_info=True)

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
                message = (
                    "That code did not match. Codes expire after two minutes; "
                    "check the app for a fresh one."
                )
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
        self.link.paired = self._valid_saved_token(extension_id, hello.token)
        if not self.link.paired:
            self._ensure_pending(extension_id)
        self.publish_safely()
        await self.link.send(HelloAck(paired=self.link.paired).model_dump(mode="json"))
        try:
            while True:
                frame = await websocket.receive_json()
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
            return self._error_response(
                request.id, ErrorCode.EXTENSION_DISCONNECTED, "extension not connected"
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
            await self.link.send(request.model_dump(mode="json"))
            response = await self._await_response(
                request.id, future, COMMAND_TIMEOUTS[request.method]
            )
            return JSONResponse(response.model_dump(mode="json", exclude_none=True))
        except asyncio.TimeoutError:
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
        return JSONResponse(
            {
                "status": "ok",
                "proto": PROTO_VERSION,
                "extension_connected": self.link.websocket is not None,
                "paired": self.link.paired,
                "browser": self.link.browser,
                "current_url": self.link.current_url,
                "current_title": self.link.current_title,
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
        if self.link.websocket is not None and before:
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
            # No browser attached: nothing can be driven, so every record is a
            # ghost by definition.
            cleaned = [tab.url for tab in before.values() if tab.url]
            self.link.driven.clear()
        republished = self.publish_safely()
        return JSONResponse(
            {
                "status": "ok",
                "cleared_tabs": cleaned,
                "driven_tabs": len(self.link.driven),
                "heartbeat_republished": republished,
                "extension_connected": self.link.websocket is not None,
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

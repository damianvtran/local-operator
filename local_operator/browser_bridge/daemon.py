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
#: Ceiling on one `websocket.close()` during a TEARDOWN. A close is a send, so
#: it carries the send's hazard: a peer that has stopped draining can leave it
#: pending forever, and every teardown path here runs INSIDE the recovery the
#: teardown exists to perform. Bounded with the send's own reasoning — a
#: loopback close frame is sub-millisecond and seconds mean a dead peer — and
#: kept separate from `LINK_SEND_TIMEOUT_S` only so the two call sites can be
#: shortened independently in tests.
LINK_CLOSE_TIMEOUT_S = LINK_SEND_TIMEOUT_S
#: How long a direct liveness PROBE waits for the peer to say anything back.
#:
#: The promotion rule used to infer liveness from elapsed silence alone, which
#: is why it needed a slack above one ping interval — the daemon is the only
#: party that solicits speech, so daemon-side tick delay is indistinguishable
#: from peer death (review R1-2). That slack (1.5 x 20 s = 30 s) then exceeded
#: the 20 s budget of `read`/`snapshot`/`screenshot`, making the rule inert for
#: exactly the methods the record added it for (review R2-3 / QA Q2-4). A probe
#: asks the peer directly instead of inferring: a healthy worker answers a ping
#: within one event-loop turn whatever it is doing (§2.2), and the daemon is
#: demonstrably live at the moment it probes — it is running the timeout path.
#: So a silence is no longer ambiguous by construction, and the window only has
#: to cover a pathologically slow loop turn, which is the same generosity
#: argument `LINK_SEND_TIMEOUT_S` already makes.
PING_PROBE_TIMEOUT_S = 5.0
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


def _identities_from(saved: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every authorised identity in one pairing record, schema 1 or 2.

    ONE reader, so the schema-1 upgrade path exists in exactly one place: a
    legacy ``{extension_id, token_sha256, paired_at}`` record IS the sole
    identity, which is what makes an already-paired operator never have to
    re-pair when this build lands.

    Deliberately a PURE read: it never rewrites the file into schema 2. A write
    on a read path is the defect class ``state.py`` documents (a full disk made
    a read fail as a write), and the pairing file is read on every handshake,
    every revocation tick and every ``lop browser pair``.
    """
    if not saved:
        return []
    listed = saved.get("identities")
    if isinstance(listed, list) and listed:
        return [entry for entry in listed if isinstance(entry, dict)]
    return [saved]


def _identities(root: Path | None = None) -> list[dict[str, Any]]:
    return _identities_from(_read_json(_pairing_path(root)))


def _identity_ids(root: Path | None = None) -> set[str]:
    return {str(entry.get("extension_id", "")) for entry in _identities(root)} - {""}


def _write_pairing(
    root: Path | None, identities: list[dict[str, Any]], *, driver_id: str = ""
) -> None:
    """Write the allow-list, keeping the legacy trio as the DRIVER's record.

    The top-level ``extension_id``/``token_sha256``/``paired_at`` keys are the
    DOWNGRADE CONTRACT, not redundancy: an older daemon (or an older `lop`)
    reading this file still finds whichever identity is currently driving, so a
    rollback degrades to single-identity instead of breaking pairing.

    ``driver_id`` falls back to the most recently paired entry because a caller
    that revokes the identity which happened to be driving must still leave a
    coherent trio behind rather than an empty one.
    """
    chosen: dict[str, Any] | None = None
    if driver_id:
        chosen = next(
            (entry for entry in identities if entry.get("extension_id") == driver_id), None
        )
    if chosen is None and identities:
        chosen = max(identities, key=lambda entry: float(entry.get("paired_at", 0) or 0))
    payload: dict[str, Any] = {}
    if chosen is not None:
        payload["extension_id"] = chosen.get("extension_id", "")
        payload["token_sha256"] = chosen.get("token_sha256", "")
        payload["paired_at"] = chosen.get("paired_at", 0.0)
    payload["schema"] = 2
    payload["identities"] = identities
    _private_write(_pairing_path(root), payload)


def add_identity(
    root: Path | None,
    extension_id: str,
    token_sha256: str,
    *,
    label: str = "",
) -> None:
    """Authorise one extension identity, replacing any existing entry for it.

    Re-pairing the same identity (a wiped token, a new browser profile that
    happens to derive the same id) must ROTATE its credential rather than add a
    second entry, or the old hash would stay live alongside the new one and a
    stale token could still authenticate.

    One token per identity, never one shared token: revocation has to be a fact
    about the file rather than a hope about a spoofable Origin header
    (design decision 2), and per-identity hashes cost one dict lookup.
    """
    identities = [entry for entry in _identities(root) if entry.get("extension_id") != extension_id]
    identities.append(
        {
            "extension_id": extension_id,
            "token_sha256": token_sha256,
            "paired_at": time.time(),
            "label": label,
            # Seeded from the pairing time: the daemon reports a LIVE last-seen
            # over /health while it is running, so the file only ever needs the
            # value it last had a reason to write. Refreshing this on every
            # reconnect would put an atomic write on the handshake path for a
            # purely cosmetic field.
            "last_seen_at": time.time(),
        }
    )
    _write_pairing(root, identities, driver_id=extension_id)


def revoke_identity(root: Path | None, extension_id: str) -> None:
    """Remove ONE identity, and only that one, from the allow-list.

    The remaining identities keep their own hashes, so a revoke is a real
    revocation for the caller and a no-op for everybody else — which is what
    makes "revoke the dev build" possible without disturbing the store build.
    Emptying the list removes the file entirely, so "nothing is authorised" has
    one representation rather than two.
    """
    identities = [
        entry for entry in _identities(root) if str(entry.get("extension_id", "")) != extension_id
    ]
    if not identities:
        with suppress(OSError):
            _pairing_path(root).unlink()
        return
    _write_pairing(root, identities)


def revoke_all(root: Path | None = None) -> None:
    """Revoke EVERY identity and drop any waiting code (``pair --reset``)."""
    for path in (_pairing_path(root), _pending_path(root)):
        with suppress(OSError):
            path.unlink()


def reset_pairing(root: Path | None = None) -> None:
    revoke_all(root)


def _pending_entries(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Live pairing codes keyed by extension ID, schema 1 tolerated.

    Per-ID is mandatory rather than tidy: a single-slot record let a second
    identity's dial rotate the FIRST identity's live code away, after which the
    first failed with "That code didn't match" for a code the user had read
    correctly. That is unreachable while only one identity may connect at all,
    and becomes a live footgun the moment the allow-list lands (design §1.3).
    """
    saved = _read_json(_pending_path(root))
    if not saved:
        return {}
    entries = saved.get("pending")
    if isinstance(entries, dict):
        return {
            str(key): value for key, value in entries.items() if isinstance(value, dict) and key
        }
    # Schema 1: the record was the single waiting identity's entry.
    extension_id = str(saved.get("extension_id", ""))
    return {extension_id: saved} if extension_id else {}


def _write_pending(root: Path | None, entries: dict[str, dict[str, Any]]) -> None:
    if not entries:
        with suppress(OSError):
            _pending_path(root).unlink()
        return
    _private_write(_pending_path(root), {"pending": entries})


def _browser_label(user_agent: str, extension_version: str) -> str:
    """A short, human-recognisable name for one install's pairing entry.

    Derived from what the peer tells us at ``hello`` (it never sends a friendly
    name) and deliberately includes the extension version: the whole point of
    the label is to let an operator tell two installs of the SAME browser apart,
    and differing extension builds are exactly how the reported case presents
    (a store build and a locally-loaded one).
    """
    order = (
        ("Edg/", "Edge"),
        ("OPR/", "Opera"),
        ("Brave", "Brave"),
        ("Arc", "Arc"),
        ("Chromium", "Chromium"),
        ("Chrome/", "Chrome"),
    )
    agent = user_agent or ""
    name = next((label for token, label in order if token in agent), "")
    if not name:
        return f"extension {extension_version}" if extension_version else "extension"
    return f"{name} {extension_version}" if extension_version else name


def pairing_status(root: Path | None = None) -> dict[str, Any]:
    """Return only display-safe pairing metadata; token hashes stay private.

    ``extension_id`` and ``pending_code`` are kept verbatim for the callers
    that predate the allow-list (``install.py`` prints the code at the end of
    its run, and every released CLI reads the id), and the list-shaped
    additions are what the multi-identity surfaces read.
    """
    saved = _read_json(_pairing_path(root))
    identities = _identities_from(saved)
    now = time.time()
    live = [
        (extension_id, entry)
        for extension_id, entry in _pending_entries(root).items()
        if float(entry.get("expires_at", 0)) > now
    ]
    # Deterministic primary: the oldest unexpired code (a constant TTL means
    # that is the smallest expiry), so the legacy single-code field cannot
    # flap between two waiting identities on successive calls.
    live.sort(key=lambda item: float(item[1].get("expires_at", 0)))
    primary = live[0] if live else None
    return {
        "paired": bool(identities),
        "extension_id": str(saved.get("extension_id", "")) if saved else "",
        "identities": [
            {
                "extension_id": str(entry.get("extension_id", "")),
                "label": str(entry.get("label", "")),
                "paired_at": float(entry.get("paired_at", 0) or 0),
                "last_seen_at": float(entry.get("last_seen_at", 0) or 0),
                "driving": bool(saved)
                and str(entry.get("extension_id", "")) == str(saved.get("extension_id", "")),
            }
            for entry in identities
        ],
        "pending_code": str(primary[1].get("code", "")) if primary else "",
        "pending_expires_at": float(primary[1].get("expires_at", 0)) if primary else 0.0,
        "pending": [
            {
                "extension_id": extension_id,
                "code": str(entry.get("code", "")),
                "expires_at": float(entry.get("expires_at", 0)),
                "label": str(entry.get("label", "")),
            }
            for extension_id, entry in live
        ],
    }


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
        # Monotonic id of the authoritative socket, bumped by every accept.
        #
        # Socket IDENTITY alone cannot fence a decision that spans an await: a
        # caller captures `self.websocket`, awaits, and a replacement connection
        # can arrive inside that window — at which point the captured socket is
        # a closed object and the CURRENT one is a fresh healthy peer. Pairing
        # the id with the object lets every check say "the link I decided about
        # is still the link" in one expression, and keeps saying it when the
        # socket has since been nulled (which identity-with-None cannot: a
        # never-connected daemon and a dropped link would look alike).
        self.generation = 0
        # Monotonic time this socket was installed. The failover rule promotes
        # the LONGEST-ATTACHED standby, which needs an order that a re-dial
        # cannot fake — generation is monotonic too, but it is bumped by every
        # accept including the promoted link's own past lives, so it says
        # nothing about how long THIS socket has been up.
        self.attached_at = 0.0
        # "driver" or "standby". Set by the handshake and by promotion; read by
        # `/health`, the CLI and the hello ack. Only the driver's link object is
        # ever returned by `BridgeService.link`, so this is a statement about
        # the role of a socket, not a second source of truth for who drives.
        self.role = "driver"
        # Set by the receive loop on EVERY frame, so an await can wait for the
        # peer to speak instead of polling `last_frame_at`. Used only by the
        # liveness probe (`_peer_answers_a_solicited_ping`).
        self.frame_event = asyncio.Event()
        self.extension_id = ""
        self.browser = ""
        # Reported in `hello` and kept so the pairing entry can be labelled with
        # the build it belongs to. The label is what lets an operator tell two
        # installs of the same browser apart in `lop browser pair --list`, which
        # is the whole point of naming them.
        self.extension_version = ""
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

    def attach(self, websocket: WebSocket, generation: int) -> int:
        """Install ``websocket`` as this link's socket at ``generation``.

        The generation is passed IN rather than bumped here because it must stay
        globally monotonic across every socket this daemon has ever installed:
        per-link counters would collide, and `is_authoritative`'s "the link I
        decided about is still the link" is only a total order while the ids do
        not repeat. #996 threaded that fence through this path; it is not
        rewritten, only fed from one counter on the service.
        """
        self.generation = generation
        self.websocket = websocket
        self.attached_at = time.monotonic()
        return generation

    def is_authoritative(self, websocket: WebSocket | None, generation: int) -> bool:
        """Whether ``(websocket, generation)`` is still the live link.

        The one predicate every deferred decision and every teardown re-checks
        across its awaits (audit A1). Deliberately an identity check on the
        socket OBJECT rather than a comparison of close codes or states: a
        later connection replaces the object, so identity is the only thing
        that cannot be spoofed by a peer that reconnects quickly.
        """
        return self.websocket is websocket and self.generation == generation

    async def send(self, payload: dict[str, Any], *, wire: WebSocket | None = None) -> None:
        """Write one frame to ``wire``, defaulting to the current link.

        A caller that captured its socket before an await must PASS it here.
        Defaulting to ``self.websocket`` at call time is how a command from a
        superseded connection was delivered to its replacement — a request the
        new peer never received, for a tab it does not own (audit A1).
        """
        websocket = wire if wire is not None else self.websocket
        if websocket is None:
            raise RuntimeError("extension disconnected")
        async with self.send_lock:
            await websocket.send_json(payload)

    def disconnect(self) -> None:
        self.websocket = None
        self.forget_link_state()

    def forget_link_state(self) -> None:
        """Drop everything scoped to the CURRENT link, keeping the socket field.

        A replacement connection needs exactly this and not `disconnect()`: the
        superseded link's pending futures can never be answered by its
        replacement, so they must fail NOW rather than at their own deadlines —
        but nulling `websocket` here would then wipe the socket that is about to
        be installed (audit A1).
        """
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
        # Every socket this daemon has ever installed, keyed by its global
        # generation, plus the IDLE link at generation 0 — the object
        # `self.link` resolves to when NOTHING drives.
        #
        # The idle link is a real member of the map rather than a
        # separately-returned stand-in so that "the link" is ONE notion
        # everywhere: severing, retiring, publishing and the wire fences all
        # take a link object, and the empty state has to be expressible as one
        # of those objects or every one of them grows a "nothing attached"
        # special case. Generation 0 is never handed to a real link
        # (`next_generation` increments first), so it cannot collide.
        self._idle_link = ExtensionLink()
        self._idle_link.generation = 0
        self.links: dict[int, ExtensionLink] = {0: self._idle_link}
        self.driver_generation = 0
        # Global, monotonic across every accept on this daemon: see
        # `ExtensionLink.attach` for why it cannot be per link.
        self._generation = 0
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
        # In-flight callers per admission-only lock key (a per-OWNER key or
        # `__global__`), so an eviction can distinguish "nobody holds it" from
        # "nobody holds it yet, a queued caller is about to" — `asyncio.Lock`
        # reports the former during the hand-off window that IS the latter. See
        # `_release_key` (audit A4).
        # Set only by `_drop_unproven_link`, and MIRRORED onto the service.
        #
        # The link the latch belongs to is retired in the same breath as it is
        # latched, so a reader that resolves `self.link` afterwards finds a
        # fresh idle link and would report `extension_unresponsive: false` — the
        # #996 wedge copy going false at the exact moment the daemon acted on
        # it, which is the misdiagnosis that copy exists to prevent. The mirror
        # carries the reason across the retire; `clear_drop_latch` is called
        # wherever the link-level latch used to be cleared, so the lifetime is
        # unchanged.
        self._drop_latch_at = 0.0
        self._drop_latch_silence_s = 0.0
        # Strong references to in-flight role frames; see `_announce_role`.
        self._role_tasks: set[asyncio.Task[None]] = set()
        self._key_callers: dict[str, int] = {}
        self._tab_locks: dict[str, asyncio.Lock] = {}

    @property
    def link(self) -> ExtensionLink:
        """The DRIVING link, or an idle link when nothing drives.

        Kept as a property rather than renamed to `links` everywhere because
        every one of the ~106 existing `self.link.<attr>` call sites is about the
        link that serves COMMANDS: `rpc`, `_admit`, `_complete`, `/health`,
        `repair`, `publish` and the whole lock topology stay correct by
        construction while standby links are served only by the handshake and
        the receive loop. Rewriting those sites into an explicit parameter would
        be a far larger diff across exactly the code #996 just fenced.

        The generation fence is preserved, not weakened: a standby's
        ``(socket, generation)`` can never equal the driver's, so
        `is_authoritative` keeps failing for it by construction.
        """
        link = self.links.get(self.driver_generation)
        return link if link is not None else self._idle_link

    def latch_drop(self, link: ExtensionLink, silence_s: float) -> None:
        """Record that the daemon severed this link for silence, and why."""
        link.note_unproven_drop(silence_s)
        self._drop_latch_at = time.monotonic()
        self._drop_latch_silence_s = silence_s

    def clear_drop_latch(self) -> None:
        """Forget the unresponsive reason: the state it described is over."""
        self._drop_latch_at = 0.0
        self._drop_latch_silence_s = 0.0

    def drop_latched(self) -> bool:
        """Whether the daemon has latched "I severed the driver for silence".

        The single read for `extension_unresponsive` and its two tenses.
        """
        if self.link.dropped_unproven():
            return True
        return (
            self._drop_latch_at > 0.0
            and (time.monotonic() - self._drop_latch_at) <= LINK_DROP_TTL_S
        )

    def drop_silence_value(self) -> float:
        """Seconds the peer had been quiet when it was severed (0.0 if never)."""
        latched = self.link.recent_drop_silence()
        return latched if latched > 0.0 else self._drop_latch_silence_s

    def next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def standby_links(self) -> list[ExtensionLink]:
        """Attached links that are not the driver, oldest attachment first."""
        return [
            link
            for generation, link in sorted(self.links.items())
            if generation != self.driver_generation and link.websocket is not None
        ]

    def _promote_standby(self) -> ExtensionLink | None:
        """Make the longest-attached surviving standby the driver, or give up.

        "Longest-attached" and not "most recently seen": during a driver
        disappearance every standby is equally idle, so recency is noise, while
        attachment age is a fact the operator can reason about ("the one that
        was already there keeps the wheel").

        "Surviving" means PROVEN **and PAIRED**. Promoting a mute standby
        would hand the wheel to a peer the daemon already believes is
        unresponsive, and promoting an UNPAIRED one hands it to a peer that
        cannot serve a single command — every session then reads `not_paired`
        for as long as it drives. Both were measured on the real rig; neither
        is a hypothetical. Leaving such a link as a standby costs nothing: a
        re-dial from its identity finds the wheel free and takes it, and a
        standby that PAIRS while the wheel is idle takes it at that moment (see
        `_take_free_wheel`).

        Decides and publishes with NO await in between (audit A1's discipline):
        the caller may only await the role frame AFTER this returns.
        """
        candidates = [link for link in self.standby_links() if link.proven and link.paired]
        if not candidates:
            return None
        promoted = max(candidates, key=lambda link: link.attached_at)
        self.driver_generation = promoted.generation
        promoted.role = "driver"
        for link in self.standby_links():
            link.role = "standby"
        self.publish_safely()
        return promoted

    async def _tell_role(self, link: ExtensionLink) -> None:
        """Inform a link it has been promoted or demoted, best effort.

        Additive on the wire and therefore safe for the released store build,
        which parses frames without a schema and ignores an event it does not
        know (`worker.ts`). A standby that misses this frame still learns its
        role on its next hello ack, and a promoted one that misses it keeps
        serving commands regardless — the daemon's own state is what gates
        commands, never the extension's belief.
        """
        payload = {"event": "role", "role": link.role}
        with suppress(Exception):
            await link.send(payload)

    def _take_free_wheel(self, link: ExtensionLink) -> ExtensionLink | None:
        """Give a newly PAIRED link the wheel when nothing else holds it.

        A link can become paired while standing by — the second install pairing
        through its own socket, or re-pairing after a revoke — and the wheel can
        be idle at that moment, because the driver went away and the only
        surviving standby was unpaired (which `_promote_standby` deliberately
        will not promote). Without this the install would sit paired and
        stationary while the daemon answered `extension_disconnected` to every
        session: reachable by doing exactly what the popup tells the user to do.
        """
        if self.link.websocket is not None:
            return None
        link.role = "driver"
        self.driver_generation = link.generation
        return link

    def _retire_link(self, link: ExtensionLink) -> ExtensionLink | None:
        """Retire a link whose socket has ended; promote a standby if it drove.

        Returns the promoted link so the caller can await its role frame — the
        decision itself must happen with no await in between (audit A1).

        A driver change follows the socket actually being gone, not a timer: the
        demoted install's surfaces keep their debugger attachments until its own
        worker learns it is a standby, and a daemon that decided earlier would
        have published a handover that had not happened.
        """
        if link is self._idle_link:
            # The link `self.link` resolves to when NOTHING is attached. Retiring
            # it is the empty transition, and it has to be expressible: it is a
            # real member of the map, so a caller that reaches it while severing
            # must clear its claim and republish (failing anything parked on it)
            # rather than fall through the "already replaced" guard and leave a
            # half-cleared state behind.
            link.disconnect()
            self.publish_safely()
            return None
        if self.links.get(link.generation) is not link:
            return None
        was_driver = link.generation == self.driver_generation
        self.links.pop(link.generation, None)
        link.disconnect()
        promoted = self._promote_standby() if was_driver else None
        if was_driver and promoted is None:
            # Nothing drives now. Point the wheel at the idle link so every
            # `self.link` read says "no driver" from ONE place, rather than
            # from whatever object the retired generation used to name.
            self.driver_generation = self._idle_link.generation
        self.publish_safely()
        return promoted

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
        # The latched "attached but stopped answering" verdict, so the discovery
        # file can tell a reader what /health would say. It matters because the
        # demotion guard in `tools/builtin.py` decides from the FILE (a probe on
        # the ABSENT side is forbidden by `bridge_browser_reachable`'s contract)
        # and a drop writes `extension_connected=false` — without this the file
        # is indistinguishable from a host with no bridge, which is how a paired
        # running bridge got told to run `lop browser install` (design D3-2).
        self.state.extension_unresponsive = self.drop_latched()
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
        # EVERY attached link, not just the driver's. A standby receives no
        # commands, so a ping is the ONLY traffic it ever sees; without one its
        # liveness would decay past LINK_SILENCE_TIMEOUT_S within the first
        # promotion window, and the failover rule — which promotes only a
        # SURVIVING standby — could never promote anybody. Keeping standbys
        # proven is what makes `_promote_standby`'s proof test meaningful
        # instead of vacuously false.
        for standby in self.standby_links():
            if standby.websocket is None:
                continue
            if not standby.proven:
                await self._drop_unproven_standby(
                    standby,
                    f"no frame for {standby.silent_for():.0f}s "
                    f"(deadline {LINK_SILENCE_TIMEOUT_S:.0f}s)",
                )
                continue
            try:
                await asyncio.wait_for(
                    standby.send({"event": "ping"}, wire=standby.websocket),
                    timeout=LINK_SEND_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await self._drop_unproven_standby(standby, "ping send exceeded its deadline")
            except Exception:  # noqa: BLE001 - the receive loop owns teardown
                logger.debug("browser extension standby ping failed", exc_info=True)
        if self.link.websocket is None:
            return
        # Capture the wire this tick is ABOUT before its first await. The ping
        # below can be suspended across a reconnect, and the teardown it then
        # triggers must sever the socket it was pinging — never the healthy
        # replacement that arrived in the meantime, which would fail every new
        # session's futures for a socket that is answering perfectly (audit A1).
        wire = (self.link.websocket, self.link.generation)
        if not self.link.proven:
            # Total silence for two ping intervals: not a slow peer, a dead
            # one. The check lives HERE because this loop already ticks every
            # PING_INTERVAL_S whether or not a session is active, so an idle
            # bridge still notices instead of waiting for the next command.
            await self._drop_unproven_link(
                f"no frame for {self.link.silent_for():.0f}s "
                f"(deadline {LINK_SILENCE_TIMEOUT_S:.0f}s)",
                expected=wire,
            )
            return
        try:
            # Bounded for the same reason as every other send: a ping that
            # parks silently disarms the very detector this design rests on.
            await asyncio.wait_for(
                self.link.send({"event": "ping"}, wire=wire[0]), timeout=LINK_SEND_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            await self._drop_unproven_link("ping send exceeded its deadline", expected=wire)
        except Exception:  # noqa: BLE001 - receive loop owns teardown
            logger.debug("browser extension ping failed", exc_info=True)

    async def _drop_unproven_standby(self, link: ExtensionLink, reason: str) -> bool:
        """Sever a STANDBY link the daemon no longer trusts.

        A standby serves no commands, so the driver's wider teardown (whose
        purpose is to answer waiting sessions with a typed refusal) has nothing
        to do here. What matters is that a mute standby is not left standing
        between the driver and a healthy one: `_promote_standby` takes only
        PROVEN links, and re-dialling re-enters as a fresh, probe-able socket.
        """
        if self.links.get(link.generation) is not link:
            return False
        websocket = link.websocket
        if websocket is None:
            return False
        logger.warning("browser bridge dropped an unresponsive standby: %s", reason)
        self._retire_link(link)
        with suppress(Exception):
            # Bounded and scoped to the captured socket (audit A2), like every
            # other close on this wire: 4000 means "reconnect" to every
            # released worker, so a standby simply re-dials and re-registers.
            await asyncio.wait_for(websocket.close(code=4000), timeout=LINK_CLOSE_TIMEOUT_S)
        return True

    async def _peer_answers_a_solicited_ping(self) -> bool:
        """Ask the peer directly whether it is still listening.

        The promotion rule used to decide this from ELAPSED SILENCE alone, which
        forced a threshold above the healthy sawtooth (1.5 x PING_INTERVAL_S) —
        and that exceeded the 20 s budget of `read`/`snapshot`/`screenshot`, so
        the rule could not fire for precisely the methods the record added it
        for (review R2-3 / QA Q2-4). Lowering the number instead would re-open
        review R1-2 (a delayed ping tick torn down as a dead peer), because the
        DAEMON is the only party that solicits speech.

        A probe removes the ambiguity rather than re-tuning it: the daemon is
        demonstrably alive at the moment it asks (it is running the timeout
        path), and a healthy worker answers a ping within one event-loop turn
        no matter what a command is doing (record §2.2). An unanswered
        solicitation is therefore direct evidence rather than an inference, and
        ANY frame counts as the answer — the same "any frame" rule the silence
        detector uses. It also fits inside every method's budget by
        construction: it is bounded by PING_PROBE_TIMEOUT_S, not by a multiple
        of PING_INTERVAL_S.

        Deliberately bounded TWICE: the solicitation itself can hang, and the
        answer can fail to come. Returns False when either does, which the
        caller treats exactly as it treats the elapsed-silence corroboration.
        """
        websocket = self.link.websocket
        if websocket is None:
            return False
        self.link.frame_event.clear()
        try:
            await asyncio.wait_for(
                self.link.send({"event": "ping"}, wire=websocket), timeout=LINK_SEND_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001 - an undeliverable solicitation is not an answer
            return False
        try:
            await asyncio.wait_for(self.link.frame_event.wait(), PING_PROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return False
        return True

    def _wire_loss(self, expected: tuple[WebSocket | None, int]) -> str:
        """Classify why the wire a command captured is no longer the live one.

        Three answers, because the three need different remedies and the fence's
        single refusal used to collapse them into one (review R3-4):

        - ``"replaced"``: a handshake installed a NEW socket, which advances the
          generation. The command died with its own connection, the replacement
          is answering, so retrying is the whole remedy.
        - ``"severed"``: THIS daemon severed the link for silence — the latch
          `_drop_unproven_link` sets is live — and nothing has replaced it. The
          browser is OPEN and the worker is mute, which is the wedge this PR
          exists for, and "no browser is attached, ask the user to open it" is
          then exactly the misdirection design D1 removed. The common
          multi-session case (two commands timing out on ONE frozen worker)
          lands here.
        - ``"gone"``: the socket ended for any other reason — the peer closed it
          or the pairing was revoked, both of which clear the latch. "No browser
          is attached" is the honest answer there.

        Compared on (socket, generation) together, the pair the fence itself
        uses: only a replacement advances the generation, and only a replacement
        puts a different live socket in place.
        """
        if self.link.generation != expected[1] or (
            self.link.websocket is not None and self.link.websocket is not expected[0]
        ):
            return "replaced"
        if self.link.websocket is None and self.drop_latched():
            return "severed"
        return "gone"

    def _drop_silence(self, measured: float) -> float:
        """The silence to report for a link this daemon severed.

        The caller measured it on the live link moments earlier; a sibling's
        drop also LATCHES the figure taken at its own drop, and that one is the
        truth about why the link went away, so prefer it while the latch is
        live (review R3-4's sibling case).
        """
        latched = self.drop_silence_value()
        return latched if self.drop_latched() and latched > 0.0 else measured

    async def _drop_unproven_link(
        self,
        reason: str,
        *,
        expected: tuple[WebSocket | None, int] | None = None,
    ) -> bool:
        """Sever a link the daemon no longer trusts, and say why ONCE.

        Returns whether the teardown was performed: False means the fence refused
        it because the wire this decision was about is no longer authoritative.
        Callers that answer the session must distinguish the two — "the extension
        stopped answering" and "the connection this command was on was replaced"
        are different facts, and the fence is exactly what tells them apart.

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

        ``expected`` is the wire the CALLER decided about, captured before the
        caller's own await. Teardown is fenced to it (audit A1): a decision that
        spans an await can otherwise land after a replacement connection is
        installed, at which point ``self.link.websocket`` is a fresh healthy peer
        and this method would close it with 4000 and fail every new session's
        futures. The close below also goes to the CAPTURED object, never to
        whatever is current, so even a replacement arriving during the close
        cannot be touched.
        """
        current = (self.link.websocket, self.link.generation)
        if expected is not None and current != expected:
            # Nothing of OURS is left to sever: the socket this decision was
            # about has been superseded (or already torn down). The latch is
            # left alone, because the live replacement owns the link's state.
            logger.debug("browser bridge skipped a stale teardown: %s", reason)
            return False
        websocket = current[0]
        logger.warning("browser bridge dropped an unresponsive extension: %s", reason)
        # Latch WHY, and how long the peer had been quiet, BEFORE the teardown
        # nulls the socket: this is the answer `_drop_unproven_link` is about to
        # give, and without the latch it would be gone by the time anyone (the
        # next RPC, `/health`, the CLI or the popup) could read it.
        link = self.link
        self.latch_drop(link, link.silent_for())
        # Clear the link BEFORE the close, not after. `disconnect()` is what
        # fails every pending future, i.e. what turns three waiting sessions into
        # three typed answers; `websocket.close()` is a SEND, so it can block on
        # a peer that has stopped draining — the same hazard as any other
        # unbounded write on this socket. Ordering the close first would make the
        # teardown itself the next thing that can hang, which is the class of bug
        # this whole change exists to remove. The close still goes out, and a
        # peer that never sees it is already the disconnected path.
        #
        # `_retire_link` also hands the wheel to the longest-attached surviving
        # standby: a driver the daemon just severed is gone by every definition,
        # and leaving the daemon with no driver until a standby happens to
        # re-dial would fail the next session's RPC instead of serving it.
        promoted = self._retire_link(link)
        if promoted is not None:
            await self._tell_role(promoted)
        if websocket is not None:
            # BOUNDED, and scoped to the captured socket (audit A2). The caller
            # is the RPC gate, the command-timeout path or the ping supervisor,
            # so an unbounded close here parks the recovery that this teardown
            # IS — a send-stalled peer would hang the initiating RPC and stall
            # the liveness loop inside its own recovery. The state above is
            # already cleared and every sibling future already failed, so the
            # only thing this deadline bounds is the courtesy frame.
            with suppress(Exception):
                await asyncio.wait_for(websocket.close(code=4000), timeout=LINK_CLOSE_TIMEOUT_S)
        return True

    async def _heartbeat(self) -> None:
        await self._supervise("heartbeat", self._heartbeat_tick)

    async def _ping(self) -> None:
        await self._supervise("ping", self._ping_tick)

    def _identity_listed(self, extension_id: str) -> bool:
        """Whether the ON-DISK allow-list still authorises this identity.

        Read from disk, never from ``link.paired`` alone, because
        ``lop browser pair --reset`` / ``--revoke`` runs in a SEPARATE process
        and can only touch the file (findings A5/U1). A revoke there must take
        authority away from an already-connected socket immediately, not merely
        at the next reconnect, so the gate and the watcher both consult the
        file — now per identity, since one identity's revocation must leave
        every other identity's authority untouched.
        """
        return bool(extension_id) and extension_id in _identity_ids(self.root)

    def _live_pairing_matches(self) -> bool:
        """Whether the on-disk pairing still authorises the DRIVING link."""
        return self._identity_listed(self.link.extension_id)

    async def _sever_identity(
        self, extension_id: str, *, link: ExtensionLink | None = None
    ) -> None:
        """Remove ONE identity from the allow-list and sever only ITS sockets.

        Flipping ``paired`` false is not enough on its own: an open socket the
        extension already holds would keep delivering RPCs until it happened to
        disconnect. So the sockets are closed too, which is what makes the
        popup's take-this-back-any-time affordance and the CLI's revoked promise
        real.

        Per IDENTITY and not "all": with an allow-list, revoking the dev build
        must leave the store build driving, and the file is emptied only when
        nothing is left authorised (design §3.5.2).
        """
        revoke_identity(self.root, extension_id)
        targets = [entry for entry in self.links.values() if entry.extension_id == extension_id]
        if link is not None and link in targets:
            # The sender first: it is the socket whose unpair the user is
            # watching, and its close is the one the popup renders.
            targets = [link] + [entry for entry in targets if entry is not link]
        for target in targets:
            target.paired = False
            # An unpair is not a wedge: forget any latched unresponsive reason,
            # or a deliberate revoke would keep reading as "attached but not
            # answering".
            target.clear_unproven_drop()
            if target.generation == self.driver_generation:
                # Revoking the driver ends the state the latch describes, so the
                # service mirror goes too: "attached but not answering" over a
                # deliberately unpaired bridge is a claim about a pairing that
                # no longer exists (review R2-1).
                self.clear_drop_latch()
            websocket = target.websocket
            # The socket AND its generation, captured together before the close
            # below yields: the guard after it asks whether this revoke is still
            # looking at the live link, not merely whether some socket exists
            # (audit A1, the revoke/replacement crossing).
            generation = target.generation
            if websocket is not None:
                with suppress(Exception):
                    # 4003 = unpaired, the same code the handshake uses so the
                    # popup renders "waiting to pair" rather than a mystery drop.
                    await asyncio.wait_for(websocket.close(code=4003), timeout=LINK_CLOSE_TIMEOUT_S)
                if not target.is_authoritative(websocket, generation):
                    # A handshake installed itself while that close was in flight.
                    # The revoke must NOT clear the link it did not close: doing
                    # so tore down the socket that replaced the revoked one and
                    # forgot its state, so the replacement's own re-dial was
                    # reported as gone.
                    #
                    # Preserving that connection is not the same as authorizing
                    # it. The new handshake computed its own `paired` from the
                    # pairing file this revoke had already rewritten, so it
                    # answers through the same `not_paired` gate as every other
                    # unpaired peer — nothing here revives a revoked token.
                    continue
            # Retires the link and hands the wheel to a standby if this was the
            # driver (the ONE retire path, so a revoke cannot diverge from a
            # driver loss).
            promoted = self._retire_link(target)
            if promoted is not None:
                await self._tell_role(promoted)
        self.publish_safely()

    async def _revocation_tick(self) -> None:
        await asyncio.sleep(REVOKE_WATCH_S)
        if not self.links:
            return
        listed = _identity_ids(self.root)
        for target in list(self.links.values()):
            if not target.extension_id or target.extension_id in listed:
                continue
            # The pairing is gone ON DISK. Clear the latched drop reason HERE,
            # before any socket test, because the severing below is unreachable
            # once a drop has happened: `disconnect()` nulls the socket AND
            # `paired`, so the guard is false forever exactly while the latch is
            # live (review R2-1, confirming QA Q2-1). Clearing it only in the
            # revoke path therefore left a deliberately unpaired bridge
            # answering "attached and paired … pairing is preserved" for the rest
            # of LINK_DROP_TTL_S — a claim about a pairing that no longer exists.
            #
            # The bound is one watch period: a revoke is reflected within
            # REVOKE_WATCH_S of the file changing, the same order as the
            # `/health` and RPC answers that read it.
            target.clear_unproven_drop()
            if target.generation == self.driver_generation:
                self.clear_drop_latch()
            if not target.paired:
                # An UNPAIRED link holds no authority to take away, so there is
                # nothing to sever: this is an install sitting on the pairing
                # form (or with a stale token) and its popup must stay up while
                # the user types the code. Severing it would close the very
                # socket the code is submitted on — which is why the guard is
                # "is it paired", not "is it listed".
                continue
            logger.info("pairing revoked on disk; closing the live extension socket")
            await self._sever_identity(target.extension_id, link=target)

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

    def _rotate_pending(self, extension_id: str, label: str = "") -> None:
        """Mint a brand-new code for ONE identity, invalidating only its prior one.

        Used when the attempt cap is reached or the code expires: rotating is
        what turns the documented 5-guess limit into a real lockout — the
        exhausted code stops working the instant a fresh one is issued, so a
        local brute-force process cannot keep guessing the same secret until
        the TTL lapses (finding A1).

        Keyed by extension id, because a single slot let a SECOND identity's
        dial rotate the first identity's live code away — after which the first
        failed with "That code didn't match" for a code the user had read
        correctly (design §1.3).
        """
        entries = _pending_entries(self.root)
        entries[extension_id] = {
            "code": f"{secrets.randbelow(1_000_000):06d}",
            "expires_at": time.time() + PAIR_TTL_S,
            "attempts": 0,
            # The label lives on the CODE, not only on the pairing record: with
            # two installs waiting, `lop browser pair` has to say which popup
            # each code belongs to, and at that moment neither install is paired
            # yet — so there is no pairing entry to name them from.
            "label": label,
        }
        _write_pending(self.root, entries)

    def _ensure_pending(self, extension_id: str, label: str = "") -> None:
        """Guarantee a live pending code exists for this identity, reusing one.

        A code is reused only while it is unexpired AND still under the attempt
        cap; anything else rotates. The cap check here is the second half of
        the A1 fix: even if a caller forgets to rotate on cap, an exhausted
        code is never handed back out as "still live".
        """
        entry = _pending_entries(self.root).get(extension_id)
        if (
            entry is not None
            and float(entry.get("expires_at", 0)) > time.time()
            and int(entry.get("attempts", 0)) < PAIR_MAX_ATTEMPTS
        ):
            return
        self._rotate_pending(extension_id, label)

    def _valid_saved_token(self, extension_id: str, token: str) -> bool:
        """Whether ``token`` is the secret stored for THIS identity, and no other.

        One token per identity, selected by id: a shared token would make
        "revoke the dev build" a claim about the Origin header — which any local
        process can forge and which the contract's own threat model does not
        treat as a boundary — instead of a fact about the file.
        """
        if not token:
            return False
        digest = hashlib.sha256(token.encode()).hexdigest()
        for entry in _identities(self.root):
            if str(entry.get("extension_id", "")) != extension_id:
                continue
            return secrets.compare_digest(str(entry.get("token_sha256", "")), digest)
        return False

    async def _try_pair(self, request: PairRequest, link: ExtensionLink) -> PairResult:
        """Submit a code for ONE link's identity.

        ``link`` is passed in rather than read from ``self.link`` because this
        is served on a STANDBY link too: the popup opens its own socket to
        submit the code, and with an allow-list that socket is a standby
        whenever another identity is already driving. Requiring driver status
        here would make pairing a second install deadlock on itself (§3.5).
        """
        extension_id = link.extension_id
        entry = _pending_entries(self.root).get(extension_id)
        if entry is None:
            self._ensure_pending(extension_id, _browser_label(link.browser, link.extension_version))
            return PairResult(ok=False, message="No live pairing code. Run lop browser pair again.")
        attempts = int(entry.get("attempts", 0)) + 1
        expired = float(entry.get("expires_at", 0)) <= time.time()
        matches = secrets.compare_digest(str(entry.get("code", "")), request.code)
        if expired or attempts >= PAIR_MAX_ATTEMPTS or not matches:
            # Reaching the cap (or expiry) rotates to a fresh code, so the
            # guessed-at code is dead the moment this branch runs — the
            # lockout the design promised. A wrong guess still under the cap
            # persists the incremented counter so the cap is actually reached
            # (the previous code saturated the stored count at 4 and never
            # rotated — finding A1). ``attempts >= cap`` on this, the cap-th
            # failure, is deliberate: the cap-th wrong guess is the last one.
            if attempts >= PAIR_MAX_ATTEMPTS or expired:
                # The label rides along: it is the only thing that names this
                # install in `lop browser pair`, and dropping it here would
                # leave a waiting install anonymous exactly after a lockout,
                # when the user most needs to know which popup to re-open.
                self._rotate_pending(extension_id, str(entry.get("label", "")))
                message = (
                    "Too many attempts. That code is now dead — run 'lop browser "
                    "pair' for a fresh one."
                    if not expired
                    else "That code expired. Run 'lop browser pair' for a fresh one."
                )
            else:
                self._set_pending_attempts(extension_id, attempts)
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
        # Rotation, not insertion: re-pairing the same identity must invalidate
        # the hash it held before, or a stale token would stay live beside the
        # new one. The label is derived from what the peer reported at `hello`
        # and kept verbatim if this daemon has no fresh value for it.
        add_identity(
            self.root,
            extension_id,
            hashlib.sha256(token.encode()).hexdigest(),
            label=_browser_label(link.browser, link.extension_version),
        )
        self._drop_pending(extension_id)
        link.paired = True
        promoted = self._take_free_wheel(link)
        self.publish_safely()
        if promoted is not None:
            await self._tell_role(promoted)
        return PairResult(ok=True, token=token)

    def _set_pending_attempts(self, extension_id: str, attempts: int) -> None:
        """Persist the incremented guess counter for one identity's code."""
        entries = _pending_entries(self.root)
        entry = entries.get(extension_id)
        if not entry:
            return
        entry["attempts"] = attempts
        _write_pending(self.root, entries)

    def _drop_pending(self, extension_id: str) -> None:
        """Retire one identity's pending code, leaving every other one live."""
        entries = _pending_entries(self.root)
        if entries.pop(extension_id, None) is not None:
            _write_pending(self.root, entries)

    async def extension(self, websocket: WebSocket) -> None:
        # The four rejections below (and the ORIGIN one above) close WITHOUT a
        # deadline, and that is correct rather than an oversight: every one of
        # them returns BEFORE `attach()`, so no link state exists yet and no
        # recovery is in flight — there is nothing for a stalled close to park.
        # Every close AFTER the install IS bounded (`LINK_CLOSE_TIMEOUT_S`),
        # because there the peer may be mid-teardown for a live link. That is the
        # whole of the A2 rule; a close site added below this comment must
        # re-check which side of `attach()` it sits on.
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
        listed = _identity_ids(self.root)
        if listed and extension_id not in listed and hello.token:
            # 4004 still refuses BEFORE `attach()`, so the unbounded-close rule
            # above is preserved verbatim — what changed is WHO it refuses.
            #
            # A peer that presents a TOKEN is claiming a pairing it does not
            # have (a revoked identity coming back with its old secret, or an
            # install pointed at the wrong daemon), and refusing it is the whole
            # point of the gate. A peer that presents NO token is not claiming
            # anything: it is asking to pair, which is exactly what the first
            # install did, and it must stay admissible or a SECOND install could
            # never be added once the first is authorised — the operator's own
            # case (a store build already paired, then a locally loaded build),
            # and the case this change exists for. Refusing it here produced a
            # dial-refuse-redial loop against the running daemon and left the
            # second install with no code to enter at all (see PR evidence).
            await websocket.close(code=4004)
            return

        # A later extension wins WITHIN an identity; incumbency holds ACROSS
        # identities. Two profiles can therefore both stay connected instead of
        # evicting each other forever (a mutual-eviction war at the alarm period:
        # each side's `onopen` resets its reconnect attempt counter, so a 4000
        # eviction buys it a 1 s re-dial — see worker.ts), while reconnect after
        # worker death keeps working because a same-identity dial still replaces
        # the incumbent.
        #
        # A distinct link object per SOCKET, not per daemon. Every frame, latch
        # and pending future below belongs to the socket draining it, so a
        # superseded socket cannot write into the live connection's state at all
        # — #996's fences, made structural rather than re-derived.
        link = ExtensionLink()
        link.extension_id = extension_id
        link.browser = hello.browser
        link.extension_version = hello.extension_version
        previous = next(
            (entry for entry in self.links.values() if entry.extension_id == extension_id), None
        )
        previous_wire: WebSocket | None = None
        if previous is not None:
            # The superseded link's work is abandoned NOW: its futures can never
            # be answered by the connection replacing it, so the waiters get the
            # typed disconnect instead of burning their budgets. Its OWN socket
            # is kept for the bounded close below — `disconnect` clears the link's
            # claim to it, which is what makes its still-running receive loop
            # stand down on its very next frame instead of stamping liveness.
            previous_wire = previous.websocket
            previous.disconnect()
            self.links.pop(previous.generation, None)
        generation = self.next_generation()
        self.links[generation] = link
        # ── THE AUTHORITATIVE INSTALL IS ONE UNINTERRUPTED BLOCK ────────────
        # Nothing between the line above and `publish_safely` below may await,
        # and every write below is a SHARED-state write (identity, pairing,
        # liveness, latch). The install used to sit after the superseded
        # socket's bounded close, and `close` is an await: a third handshake
        # could install itself in that window, after which the suspended one
        # stamped THIS handshake's verdict — pairing included — onto the newer
        # link. An unauthenticated peer whose token the daemon had just
        # rejected was thereby authorized and drove RPCs (audit A1). Ordering
        # is the whole guard here: a handshake that no longer owns the link has
        # nothing left to write, because all of its writes already happened.
        link.attach(websocket, generation)
        # A fresh authoritative socket supersedes any latched drop reason from
        # the link it just replaced (including the "later connection wins"
        # eviction above), so a reconnect cannot inherit a mute label.
        link.clear_unproven_drop()
        self.clear_drop_latch()
        # The peer has just proven it is listening by sending `hello`; stamp it
        # so `proven` is true from the first instant of the connection instead
        # of waiting for the first pong.
        link.last_frame_at = time.monotonic()
        link.paired = self._valid_saved_token(extension_id, hello.token)
        # Cold-start tie-break AND within-identity later-wins, in one test: the
        # wheel is free whenever nothing holds it — the previous same-identity
        # link is already gone from `links`, so if it was driving, the wheel is
        # idle and this socket takes it. A different identity's live driver is
        # still holding the wheel, so this one becomes a standby and receives no
        # commands. Deliberately not a configured priority: a preferred install
        # that reclaimed the wheel on EVERY reconnect would reproduce the
        # eviction war at the alarm period instead of at 1 Hz.
        if self.link.websocket is None:
            self.driver_generation = generation
        link.role = "driver" if self.driver_generation == generation else "standby"
        if not link.paired:
            self._ensure_pending(extension_id, _browser_label(link.browser, link.extension_version))
        self.publish_safely()
        # Only now, with this handshake fully installed, is the socket it
        # replaced closed: a bounded courtesy to a peer that is already
        # non-authoritative, whose outcome this handshake's authority must not
        # depend on (audit A2).
        if previous_wire is not None:
            with suppress(Exception):
                await asyncio.wait_for(previous_wire.close(code=4000), timeout=LINK_CLOSE_TIMEOUT_S)
        if not link.is_authoritative(websocket, generation):
            # A newer handshake installed itself while the superseded socket was
            # being closed, so this one is now the superseded side. It must not
            # speak: an ack on a wire that is no longer current tells a peer it
            # is paired with a daemon that has already moved on — the ack half
            # of the same window as the metadata install above (audit A1).
            with suppress(Exception):
                await asyncio.wait_for(websocket.close(code=4000), timeout=LINK_CLOSE_TIMEOUT_S)
            return
        try:
            # Bounded and wire-scoped like every other send on this socket: the
            # handshake answer is the first thing a superseded write would
            # misdeliver, and a peer that stopped draining must not park the
            # accept path (audit A1/A2).
            #
            # `role` and `authorized_count` are ADDITIVE. The released store
            # build parses frames without a schema and ignores what it does not
            # know, and an older DAEMON simply never sent them — which is why the
            # extension must read an absent `role` as "driver" (design §7.2).
            # `Hello` itself gains nothing: a new field THERE would be closed 4001
            # by every already-released daemon (`extra="forbid"`).
            await asyncio.wait_for(
                link.send(
                    HelloAck(
                        paired=link.paired,
                        role=link.role,
                        authorized_count=len(_identity_ids(self.root)),
                    ).model_dump(mode="json"),
                    wire=websocket,
                ),
                timeout=LINK_SEND_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - an undeliverable handshake is a dead dial
            with suppress(Exception):
                await asyncio.wait_for(websocket.close(code=4000), timeout=LINK_CLOSE_TIMEOUT_S)
            promoted = self._retire_link(link)
            if promoted is not None:
                await self._tell_role(promoted)
            return
        try:
            while True:
                frame = await websocket.receive_json()
                if not link.is_authoritative(websocket, generation):
                    # A replacement connection is authoritative now and this
                    # socket is a superseded one still draining buffered frames.
                    # Its events must not stamp liveness, publish into the driven
                    # record, drop an approval, or resolve a future — every one of
                    # which is the LIVE link's state (audit A1, receive-side half).
                    logger.debug("browser bridge stopped reading a superseded socket")
                    break
                # ANY frame, recorded BEFORE any dispatch on its type: this is
                # the liveness signal `proven` reads, and taking it here (rather
                # than inside each event branch) is what makes an older
                # extension that only pongs count as alive. The probe waits on
                # this event, so it is set on the same line as the stamp: a
                # frame that counts as liveness but not as an answer would leave
                # the probe blind to a peer that is plainly talking.
                #
                # Stamped on THIS link, never on `self.link`: a standby's frame
                # must not read as the driver being alive, and a superseded one
                # cannot reach here at all.
                link.last_frame_at = time.monotonic()
                link.frame_event.set()
                if frame.get("event") == "pair":
                    try:
                        pair = PairRequest.model_validate(frame)
                    except ValidationError:
                        continue
                    # Served on a STANDBY link too (§3.5.1): the popup submits its
                    # code over its own socket, and with an allow-list that socket
                    # is a standby whenever another identity drives. Requiring
                    # driver status here would make pairing a second install
                    # deadlock on itself.
                    result = await self._try_pair(pair, link)
                    with suppress(Exception):
                        await asyncio.wait_for(
                            link.send(result.model_dump(mode="json"), wire=websocket),
                            timeout=LINK_SEND_TIMEOUT_S,
                        )
                    continue
                if frame.get("event") == "awaiting_origin":
                    # The extension paused this request on a human origin
                    # decision. Record it so the RPC wait extends its deadline
                    # (A3) and the popup/status can show what is pending (U2).
                    request_id = str(frame.get("id", ""))
                    if request_id:
                        link.awaiting_origin[request_id] = str(frame.get("origin", ""))
                        self.publish_safely()
                    continue
                if frame.get("event") == "awaiting_origin_cleared":
                    # The extension's queue entry for this command is gone
                    # (decided, cancelled, or expired) without a response the
                    # daemon will see. Drop the record so /health stops echoing
                    # a prompt the popup can no longer resolve — the stale echo
                    # is what looped the approval popup on "Request changed."
                    request_id = str(frame.get("id", ""))
                    if request_id and request_id in link.awaiting_origin:
                        link.awaiting_origin.pop(request_id, None)
                        self.publish_safely()
                    continue
                if frame.get("event") == "unpair":
                    # The options page "Unpair this browser" reaches the daemon
                    # here so revocation severs THIS live socket, mirroring the
                    # CLI --revoke path (findings A5/U1). Per identity: the
                    # install that asked to unpair must be the only one dropped,
                    # which is what makes the two-install case usable instead of
                    # a race to re-pair both.
                    await self._sever_identity(link.extension_id, link=link)
                    return
                if frame.get("event") == "tab_update":
                    # Pushed by the extension on navigation so the popup reflects
                    # the driven site promptly even between commands (U3).
                    link.note_driven(
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
                    link.note_closed(str(frame.get("tab", "")))
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
                        link.note_driven(
                            handle if isinstance(handle, str) else "",
                            url,
                            title if isinstance(title, str) else "",
                        )
                link.awaiting_origin.pop(response.id, None)
                future = link.pending.pop(response.id, None)
                if future is not None and not future.done():
                    future.set_result(response)
        except WebSocketDisconnect:
            pass
        finally:
            if link.websocket is websocket:
                # The peer ended this link itself (worker died, browser closed,
                # tab torn down). That is NOT the unresponsive path — the daemon
                # did not sever it — so drop any latched unresponsive reason and
                # let the honest "not currently attached" state stand.
                link.clear_unproven_drop()
                if link.generation == self.driver_generation:
                    self.clear_drop_latch()
                # If this was the driver, the wheel passes to the
                # longest-attached surviving standby HERE: a driver change must
                # follow the socket actually being gone, not a timer, or the
                # demoted install's tabs would keep their debugger attachments
                # while nothing could reach them.
                promoted = self._retire_link(link)
                if promoted is not None:
                    self._announce_role(promoted)

    async def driver(self, http_request: HttpRequest) -> JSONResponse:
        """Pin the driving extension explicitly: ``lop browser drive <id-or-label>``.

        The escape hatch from the incumbency rule. It exists because that rule is
        deliberately self-STABLE rather than clever: with two installs up, the
        one already driving keeps the wheel, and reconnecting the other cannot
        take it (a same-identity dial replaces only that identity's own link).
        Without this the operator's only lever is quitting a browser — which is
        the situation this whole change exists to remove.

        Authenticated with the session key exactly as ``/rpc`` is: choosing which
        extension drives the user's browser is the same authority the session leg
        already carries, and this is HTTP rather than the WS protocol so a
        released extension never has to know it exists.
        """
        supplied = http_request.headers.get("x-bridge-key", "")
        if not secrets.compare_digest(supplied, self.state.session_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            payload = json.loads(await http_request.body() or b"{}")
        except ValueError:
            return JSONResponse({"error": "invalid_request"}, status_code=422)
        target = str(payload.get("target", "") if isinstance(payload, dict) else "")
        link = self._resolve_extension(target)
        if link is None:
            return JSONResponse(
                {
                    "error": "unknown_extension",
                    "authorized_extension_ids": sorted(_identity_ids(self.root)),
                    "standby_extension_ids": [entry.extension_id for entry in self.standby_links()],
                },
                status_code=404,
            )
        if not link.paired:
            # Checked BEFORE the already-driving shortcut, because an unpaired
            # link CAN be the driver: the handshake gives the wheel to whoever
            # dials when nothing holds it. Same rule the automatic promotion path
            # applies — the wheel only goes to a link that can serve a command.
            return JSONResponse(
                {
                    "error": "not_paired",
                    "message": (
                        "that install is connected but not paired yet; run "
                        "'lop browser pair' and enter its code first"
                    ),
                },
                status_code=409,
            )
        if link.generation == self.driver_generation and link.role == "driver":
            # Already driving: report success rather than churning the link, so
            # `drive` is idempotent and safe to script.
            return JSONResponse({"ok": True, "driver_extension_id": link.extension_id})
        previous = self.link
        if previous.websocket is not None and previous.role == "driver":
            previous.role = "standby"
        # Decide and publish together, then tell the peers (audit A1): the
        # demoted install learns it holds nothing of its own from the role
        # frame, and until it does, the daemon is already refusing it commands.
        self.driver_generation = link.generation
        link.role = "driver"
        self.publish_safely()
        for entry in [link, previous]:
            if entry.websocket is not None:
                await self._tell_role(entry)
        return JSONResponse({"ok": True, "driver_extension_id": link.extension_id})

    def _announce_role(self, link: ExtensionLink) -> None:
        """Deliver a role frame as its own task rather than awaiting it here.

        Used from the receive loop's `finally`, which also runs when the task is
        being CANCELLED (connection teardown, server shutdown). An `await` there
        raises `CancelledError` before the frame leaves the process, so a
        promoted install would go on believing it is a standby — serving
        commands while its popup says otherwise — until something made it
        re-dial. The daemon's own state already gates commands, so the frame is
        the courtesy half, but a courtesy that silently never arrives on exactly
        the path that just moved the wheel is how a mystery gets planted.
        """
        task = asyncio.get_running_loop().create_task(self._tell_role(link))
        # asyncio keeps only a WEAK reference to a running task, so a frame in
        # flight while nothing else is scheduled can be collected before it is
        # sent. The set is the strong reference; the callback drops it.
        self._role_tasks.add(task)
        task.add_done_callback(self._role_tasks.discard)

    def _resolve_extension(self, target: str) -> ExtensionLink | None:
        """Resolve ``<id-or-label>`` to an attached link, or None.

        Accepts an exact id, an unambiguous id PREFIX (the operator copies 32
        opaque characters; matching the first few is the difference between one
        command and a copy-paste exercise), or a case-insensitive label
        SUBSTRING. An ambiguous target resolves to nothing rather than to a
        guess: silently moving the wheel to the wrong browser is worse than
        asking for one more character.
        """
        wanted = target.strip().lower()
        if not wanted:
            return None
        candidates = [entry for entry in self.links.values() if entry.websocket is not None]
        exact = [entry for entry in candidates if entry.extension_id.lower() == wanted]
        if exact:
            return exact[0]
        labels = _identities(self.root)
        by_prefix = [entry for entry in candidates if entry.extension_id.lower().startswith(wanted)]
        if len(by_prefix) == 1:
            return by_prefix[0]
        by_label = []
        for entry in candidates:
            label = next(
                (
                    str(saved.get("label", ""))
                    for saved in labels
                    if str(saved.get("extension_id", "")) == entry.extension_id
                ),
                "",
            )
            if label and wanted in label.lower():
                by_label.append(entry)
        unique = {entry.generation: entry for entry in by_label}
        if len(unique) == 1:
            return next(iter(unique.values()))
        return None

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
            # Read the latched silence only AFTER the guard that decides whether
            # it means anything: the value is 0.0 both for "never dropped" and
            # for "dropped at time zero", so trusting it before the check invites
            # reading an ambiguous number as a fact (review R2-6).
            if self.drop_latched():
                dropped_for = self.drop_silence_value()
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
            await self._drop_unproven_link(
                f"command refused: link silent for {silent:.0f}s",
                expected=(self.link.websocket, self.link.generation),
            )
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
                # Per identity: the on-disk record still authorises OTHER
                # installs, so severing only this one leaves a standby driving
                # rather than dropping every socket because one token went
                # stale.
                await self._sever_identity(self.link.extension_id, link=self.link)
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
        above; a separate method so the lock topology is directly testable).

        Three shapes, and the difference between them is WHOSE budget a stuck
        command may spend (audit A4, scoping answer D2):

        - a per-REQUEST key for `await_access` (a human wait must queue behind
          nothing);
        - the short `__access__` key for `request_access`;
        - a per-TAB key whenever the request names a tab — and here the lock
          spans the WHOLE command on purpose, because that span is what stops
          two commands interleaving into one tab's CDP session;
        - otherwise a per-OWNER key (`__owner__:<proof>`) when the request
          carries one, and `__global__` only for the proof-less remainder
          (`tabs`, a handle-less `status`, legacy capability clients).

        Why the owner key replaced `__global__` for owned no-tab commands: they
        carry no `tab` (`resources.py` builds owner params without one), so one
        owner's `open`/`owner_recover`/`owner_finish` used to serialize with
        EVERY other owner's — across a full navigation (30 s) or an approval
        wait (+65 s). A parked `open` therefore blocked a different owner's
        `open`, and the `owner_recover` that exists to recover from exactly that
        (audit A4). The cap race `__global__` was a PROXY guard for is not the
        daemon's to enforce any more: `MAX_SURFACES` is a property of the
        extension's surfaces map, and every mutation of it is a read-modify-write
        inside that worker's `storeQueue`. The extension owns it; the extension
        now guards it (`withAdmission`, `state.ts`). See the addendum's D2/D3.
        """
        if request.method == "await_access":
            return f"__await__:{request.id}"
        if request.method == "request_access":
            return "__access__"
        tab = request.params.get("tab")
        if tab:
            return str(tab)
        proof = request.params.get("owner_proof")
        if isinstance(proof, str) and proof:
            return f"__owner__:{proof}"
        return "__global__"

    @staticmethod
    def _admission_only(tab_key: str) -> bool:
        """Whether this key's lock may be released before the response.

        True for exactly the two keys the scoping change is about — a per-OWNER
        key and the proof-less `__global__` one — which exist to order the frame
        onto the wire, not to hold a navigation or a human wait.

        Everything else keeps the whole-command span: a per-TAB key IS the
        CDP-interleave guard (and starves nobody, being per tab), `await_access`
        has its own per-request key so its span cannot queue behind anything, and
        `__access__` is the short shared key `request_access` has always used.
        Derived from the KEY rather than re-reading the request, so the decision
        cannot drift from `lock_key_for`.
        """
        return tab_key == "__global__" or tab_key.startswith("__owner__:")

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
        if not self._admission_only(tab_key):
            # The per-TAB (and `__access__`) arm, counted the SAME way as the
            # admission-only arm below: the count spans key-capture → release, so
            # `_release_key` can tell "nobody holds it" from "nobody holds it
            # yet, someone is queued".
            #
            # Why the count is required HERE and not merely tidy: per-tab keys
            # used to be near-singleton, but with every opened tab minting a
            # token the map would grow for the daemon's lifetime, so a `close`
            # evicts the key once its tab is gone. The eviction used to test
            # `lock.locked()` alone, which is False the instant `release()` hands
            # the lock to its FIRST WAITER — so a `close` answering while another
            # command was already queued on that tab dropped the key out from
            # under the waiter. The waiter then ran under a Lock object the map no
            # longer contained while the next request for the same tab minted a
            # SECOND one and interleaved with it, losing the per-tab mutual
            # exclusion that IS the reason this key shape holds its lock for the
            # whole command (review R4-1, the same defect round 3 fixed for the
            # admission-only arm and left behind here).
            #
            # The `finally` is what keeps a CANCELLED `close` from leaking its
            # count: a leaked count pins the key forever, which is the unbounded
            # growth the eviction exists to prevent. Only `close` releases the
            # key; every other method returns the count to zero without evicting,
            # because a per-tab key is deliberately retained between commands.
            self._key_callers[tab_key] = self._key_callers.get(tab_key, 0) + 1
            try:
                async with lock:
                    response = await self._dispatch_locked(request)
            finally:
                if request.method == "close":
                    self._release_key(tab_key, lock)
                else:
                    self._uncount_key(tab_key)
            return response
        # ADMISSION ONLY (`_admission_only`'s docstring): register the pending
        # future and get the frame onto the wire under the key, then release it
        # BEFORE waiting for the answer. Nothing in the response phase reads
        # shared daemon state under a key — `_await_response` resolves one future
        # from `link.pending` and the timeout/teardown paths key off
        # `request.id` — so holding the lock across the answer bought nothing and
        # cost every other owner its budget.
        future = self._register_pending(request)
        # Counted BEFORE the lock so `_release_key` can tell "nobody holds it"
        # from "nobody holds it yet, someone is queued" (see its docstring).
        self._key_callers[tab_key] = self._key_callers.get(tab_key, 0) + 1
        try:
            wire = self._wire()
            async with lock:
                refused = await self._admit(request, future, wire)
            if refused is not None:
                return refused
            return await self._complete(request, future, wire)
        finally:
            # CANCELLATION-SAFE CLEANUP for the registration → lock → send →
            # response lifetime (audit A4). Registering the future BEFORE the
            # lock is what this key shape needs — the answer may arrive while
            # the caller is still queued — but it also means a caller cancelled
            # while QUEUED (the ordinary client-disconnect case) never reaches
            # `_complete`'s finally. Pre-fix it left a future in `link.pending`
            # for a frame that was never sent and no response can ever answer,
            # holding its request id until link teardown. The `finally` covers
            # every exit: admitted, refused, timed out, cancelled.
            self._forget_pending(request.id, future)
            self._release_key(tab_key, lock)

    async def _dispatch_locked(self, request: Request) -> JSONResponse:
        """Dispatch one command with the caller's key held for its WHOLE life.

        The per-TAB path: for those keys the lock span IS the guard that stops two
        commands interleaving into one tab's CDP session, and it starves nobody
        because it is per tab. Keys that may be released after admission go
        through `_admit`/`_complete` directly from `_dispatch_serialized` — see
        `_admission_only`.
        """
        future = self._register_pending(request)
        # The wire THIS command is being sent on, captured before the first await
        # on it. Every later decision about the link — the send deadline's
        # teardown, the promotion rule — is fenced to it, so a command that was
        # suspended across a reconnect can never sever (or answer on behalf of)
        # the healthy replacement that arrived in the meantime (audit A1).
        wire = self._wire()
        try:
            refused = await self._admit(request, future, wire)
            if refused is not None:
                return refused
            return await self._complete(request, future, wire)
        finally:
            # The same cancellation-safe exit as the admission-only path: a
            # caller that goes away while its frame is being written (or while
            # the answer is awaited) must not leave its future registered for a
            # response nobody is left to receive (audit A4).
            self._forget_pending(request.id, future)

    def _wire(self) -> tuple[WebSocket | None, int]:
        """The current link as a fenceable pair (socket identity + generation)."""
        return (self.link.websocket, self.link.generation)

    def _register_pending(self, request: Request) -> asyncio.Future[Response]:
        """Register this command's future BEFORE its frame can be answered."""
        future: asyncio.Future[Response] = asyncio.get_running_loop().create_future()
        self.link.pending[request.id] = future
        return future

    async def _admit(
        self, request: Request, future: asyncio.Future[Response], wire: tuple[WebSocket | None, int]
    ) -> JSONResponse | None:
        """Get one command onto the wire. None means it was delivered.

        The ADMISSION phase, split out of `_dispatch_locked` so a caller can hold
        its lock for exactly this much and release it before the answer (see
        `_admission_only`). Nothing here waits on the peer — a bounded write and
        nothing else — so every result it can return is final.
        """
        try:
            # Bounded INCLUDING the ``send_lock`` acquisition inside
            # ``link.send``: a stuck lock holder used to hang the RPC past every
            # deadline in COMMAND_TIMEOUTS while holding this tab's lock key,
            # queueing every other command for the same key behind it.
            await asyncio.wait_for(
                self.link.send(request.model_dump(mode="json"), wire=wire[0]),
                timeout=LINK_SEND_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            # Never left the daemon. Distinct from "delivered, no answer" and
            # reported as such, so the diagnostic distinguishes "could not
            # deliver" from "no reply".
            #
            # Drop OUR future before the teardown: the teardown fails every
            # pending future with RuntimeError, and this one is never awaited
            # (we return a typed error instead), so leaving it registered would
            # produce an unretrieved-exception warning at collection time. By
            # identity, not by id — `pending` is id-keyed and a teardown clears
            # it, so an id-only pop can take a LATER request's future (audit A4).
            self._forget_pending(request.id, future)
            silent = self.link.silent_for()
            dropped = await self._drop_unproven_link(
                f"send of {request.method} exceeded {LINK_SEND_TIMEOUT_S:.0f}s",
                expected=wire,
            )
            if not dropped and self._wire_loss(wire) == "replaced":
                # The wire this command was being written to was superseded while
                # the write was suspended, so the command was never delivered and
                # there is no unresponsive peer to report — the replacement is
                # answering. Say what actually happened (audit A1): the session's
                # connection went away mid-send.
                #
                # `phase` is what makes that sayable to a reader: the client's
                # table lookup returns one copy per CODE, and
                # `extension_disconnected`'s copy is "no browser is attached, ask
                # the user to open it" — false here, where the browser is open and
                # the worker has just re-dialled. The two phase-carrying siblings
                # below already route this way (design D3-3).
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_DISCONNECTED,
                    f"{request.method} was not delivered: the extension replaced its "
                    "connection while the command was being written",
                    {"phase": "replaced"},
                )
            # Either this caller severed the link, or a sibling already had: both
            # mean the same wedged peer, so this is the unresponsive answer, not
            # the replaced one (review R3-4).
            return self._error_response(
                request.id,
                ErrorCode.EXTENSION_UNRESPONSIVE,
                f"{request.method} could not be delivered to the browser extension",
                {"phase": "send", "link_silent_s": self._drop_silence(silent)},
            )
        except Exception as exc:  # noqa: BLE001 - transport failure becomes typed wire error
            # A future failed by a SIBLING's severance lands here — that
            # teardown fails every pending future — and `extension_disconnected`
            # renders "no browser is attached" for an open, wedged browser: the
            # same misdirection the fence arm above answers, on the other path
            # into it (review R3-4).
            loss = self._wire_loss(wire)
            if loss == "replaced":
                # The same sibling case one step further out (review R4-3): a
                # REPLACEMENT handshake fails every pending future too
                # (`forget_link_state`), and it does so IMMEDIATELY rather than at
                # a deadline — so on the path a worker re-dial actually takes,
                # this arm, not the send-timeout fence above, is the one that
                # answers. Without the test the code alone rendered "no browser is
                # attached… ask the user to open their browser" for a browser that
                # is open and already reconnected; `phase` is what routes it to the
                # honest copy (design D3-3).
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_DISCONNECTED,
                    f"{request.method} was not delivered: the extension replaced its connection",
                    {"phase": "replaced"},
                )
            if loss == "severed":
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_UNRESPONSIVE,
                    f"{request.method} could not be delivered to the browser extension",
                    {"phase": "send", "link_silent_s": self._drop_silence(0.0)},
                )
            return self._error_response(request.id, ErrorCode.EXTENSION_DISCONNECTED, str(exc))
        return None

    async def _complete(
        self, request: Request, future: asyncio.Future[Response], wire: tuple[WebSocket | None, int]
    ) -> JSONResponse:
        """Wait for one admitted command's answer and render it.

        The RESPONSE phase, and deliberately key-independent: it resolves one
        future out of ``link.pending`` and keys every teardown off ``request.id``
        (plus A1's wire fence), so no caller needs to hold a lock across it. An
        abandoned command — the daemon gave up, the extension answers later — is
        covered by the ``pending`` eviction below (the id is gone, so a late frame
        finds no future) together with that fence.
        """
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
            #
            # …but elapsed silence ALONE cannot reach 30 s inside a 20 s budget,
            # so on its own that rule was inert for exactly the methods the
            # record added it for — `read`/`snapshot`/`screenshot` all time out
            # at 20 s (review R2-3 / QA Q2-4 measured the change: the frozen
            # worker returned `internal {timeout_s: 20.0}` where round 1 returned
            # `extension_unresponsive`). The OR below is the fix for that, and it
            # is deliberately NOT a lower threshold: a threshold below the
            # healthy sawtooth is what review R1-2 reproduced, because the daemon
            # is the only party that solicits speech. Asking the peer directly
            # instead of inferring from a clock gives a definitive answer inside
            # whatever budget the method has, so the rule fires for the tight
            # methods without re-opening the false positive. The already-answered
            # case short-circuits on the OR before probing.
            if silent > PING_INTERVAL_S * 1.5 or not await self._peer_answers_a_solicited_ping():
                dropped = await self._drop_unproven_link(
                    f"{request.method} unanswered with the link silent for {silent:.0f}s",
                    expected=wire,
                )
                if not dropped and self._wire_loss(wire) == "replaced":
                    # Same fence as the send deadline: nothing of ours is left to
                    # sever, so the honest answer is the one the failed future
                    # already carries. `phase` for the same reason as its sibling
                    # in `_admit`: the code alone would render "no browser is
                    # attached" (design D3-3).
                    return self._error_response(
                        request.id,
                        ErrorCode.EXTENSION_DISCONNECTED,
                        f"{request.method} was delivered on a connection the extension "
                        "has since replaced",
                        {"phase": "replaced"},
                    )
                # This caller severed the link, or a sibling already had — the
                # same wedged peer either way (review R3-4).
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_UNRESPONSIVE,
                    f"{request.method} was delivered but the extension stopped answering",
                    {"phase": "response", "link_silent_s": self._drop_silence(silent)},
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
            # Same sibling case as the send arm: a severed link makes this a
            # wedged browser, not an absent one (review R3-4) — and a REPLACED
            # one makes it an open browser that needs no user action at all
            # (review R4-3). The replacement arrives as a failed future, so this
            # arm is the one a real re-dial reaches; the two timeout fences above
            # cover only the narrower case where the replacement lands inside a
            # timeout window.
            loss = self._wire_loss(wire)
            if loss == "replaced":
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_DISCONNECTED,
                    f"{request.method} was delivered on a connection the extension "
                    "has since replaced",
                    {"phase": "replaced"},
                )
            if loss == "severed":
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_UNRESPONSIVE,
                    f"{request.method} was delivered but the extension stopped answering",
                    {"phase": "response", "link_silent_s": self._drop_silence(0.0)},
                )
            return self._error_response(request.id, ErrorCode.EXTENSION_DISCONNECTED, str(exc))
        finally:
            self._forget_pending(request.id, future)

    def _forget_pending(self, request_id: str, future: asyncio.Future[Response]) -> None:
        """Drop THIS request's pending record — by identity, never by id alone.

        `link.pending` is keyed by request id, and a teardown (`forget_link_state`)
        clears it, so a LATER request may re-use an id an earlier one had. A
        cleanup that popped the id unconditionally would then strand that
        unrelated caller's future: its answer would find no waiter and hang
        until its own timeout (audit A4's identity requirement).

        The awaiting-origin record is dropped only under the SAME identity proof,
        because the two halves are removed for one reason and by one owner, and
        the identity of the future is what proves the ownership. Popping the
        marker by id alone re-opened the very window the future's guard exists
        to close (review R4-2): `_drop_unproven_link` clears BOTH maps and then
        parks up to `LINK_CLOSE_TIMEOUT_S` in its bounded `websocket.close()` on
        a wedged peer, so a worker can genuinely re-dial, re-pair and file a NEW
        request — passing the busy guard, because that guard reads the map the
        teardown just emptied — while this older task is still unwinding. The
        extension then announces a real human approval for that new request, and
        an unguarded pop here deleted its marker, so the request lost its
        deadline extension and failed at its BASE timeout while the user was
        still looking at the prompt. Guarding only the future left exactly that
        loss, with `NEW_FUTURE_PRESERVED: true, NEW_APPROVAL_MARKER_PRESERVED:
        false` as the signature.

        Correct on every path, because the only caller that may legitimately
        clear another request's marker is the teardown itself, and it clears the
        whole map. A request whose own future the teardown removed therefore
        finds no identity match and touches nothing — which is what it must do,
        since by then the marker belongs to whoever holds that id now.
        """
        if self.link.pending.get(request_id) is future:
            del self.link.pending[request_id]
            self.link.awaiting_origin.pop(request_id, None)

    def _uncount_key(self, tab_key: str) -> None:
        """Return one caller's registration on a key, WITHOUT deciding eviction.

        The counting half of `_release_key`, split out for the callers that must
        bring the count back to zero but must NOT evict: a per-TAB key is
        deliberately retained between commands (evicting it between them is what
        let two sessions interleave into one tab's CDP session), so only a
        `close` ever reaches the eviction decision below.
        """
        remaining = self._key_callers.get(tab_key, 0) - 1
        if remaining > 0:
            self._key_callers[tab_key] = remaining
        else:
            self._key_callers.pop(tab_key, None)

    def _release_key(self, tab_key: str, lock: asyncio.Lock) -> None:
        """Drop one caller's hold on a key, evicting it if the key is idle.

        A per-OWNER key is minted per session-resource, so without eviction
        `_tab_locks` grows for the daemon's lifetime — the same unbounded-growth
        defect round-3 M1 fixed for the `__await__` keys, which is why the
        eviction exists at all even though the lock held no navigation. The
        per-TAB arm reaches here from `close` for the same reason.

        Eviction is safe only when ALL THREE hold: no caller is queued on the
        key, nobody holds it, and the map still holds THIS Lock object (an
        earlier caller may already have evicted it while a later request minted
        a replacement).

        Why the queued half needs its own counter: `asyncio.Lock.locked()` goes
        False the instant `release()` hands the lock to the first waiter, so a
        caller checking only `locked()` evicted a lock another request was
        ALREADY waiting on. That waiter then ran under a Lock object the map no
        longer contained, while the next request for the key minted a second
        one and interleaved with it — mutual exclusion lost for one owner's
        commands (audit A4), and for one TAB's CDP session on the per-tab arm
        (review R4-1). `_key_callers` counts every caller from the moment it
        captures the key to the moment it releases, so "idle" here really means
        idle.

        Residual, benign and deliberate (unchanged from the scoping change): two
        commands of ONE owner can still be admitted concurrently in the window
        where the key was evicted between them — admission is a frame write, and
        the extension's per-proof lane re-serializes on arrival, so the worst
        case stays "two frames admitted in an unspecified order".
        """
        self._uncount_key(tab_key)
        if (
            not self._key_callers.get(tab_key)
            and self._tab_locks.get(tab_key) is lock
            and not lock.locked()
        ):
            self._tab_locks.pop(tab_key, None)

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
        driver_id = self.link.extension_id
        driver_label = next(
            (
                str(entry.get("label", ""))
                for entry in _identities(self.root)
                if str(entry.get("extension_id", "")) == driver_id and driver_id
            ),
            "",
        )
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
                or self.drop_latched(),
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
                    else self.drop_silence_value()
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
                # ── Multi-identity: additive, optional, driver-scoped ────────
                #
                # Every field ABOVE keeps describing the DRIVER, deliberately:
                # the popup's wedge card, `lop browser status` and
                # `backend._health_ok` are all asking about the link that serves
                # commands, and re-pointing them at "some link" would have made
                # a standby-only bridge look connected. What is new is only the
                # answer to "who else is here".
                #
                # `extension_connected`/`paired`/`link_attached`/`link_silent_s`
                # therefore mean what they always meant. The READER that needs
                # more asks for it by name.
                "driver_extension_id": driver_id,
                "standby_extension_ids": [link.extension_id for link in self.standby_links()],
                "authorized_extension_ids": sorted(_identity_ids(self.root)),
                # The driver's human label, so the standby popup can NAME the
                # install holding the wheel rather than saying "another one".
                # Empty when the file has no label for it (an entry paired
                # before labels existed), which the card words accordingly.
                "driver_label": driver_label,
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
            Route("/driver", service.driver, methods=["POST"]),
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

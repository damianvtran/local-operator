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

from local_operator import browser_files
from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    EXPECTED_EXTENSION_VERSION,
    MIN_SUPPORTED_PROTO,
    ORIGIN_PROMPT_WINDOW_S,
    PROTO_VERSION,
    Capabilities,
    ErrorCode,
    ErrorDetail,
    Hello,
    HelloAck,
    PairRequest,
    PairResult,
    Request,
    Response,
    extension_older,
    proto_supported,
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
#: Consecutive unanswered liveness PROBES on one link before the daemon severs it.
#:
#: ONE unanswered question is not evidence of a dead peer. The probe rides the
#: same event loop and the same socket as the command that timed out, so a miss
#: is also the shape a GC pause, a slow renderer or a starved loop produces —
#: and severing on the first miss failed every OTHER pending future on that
#: worker, i.e. one slow `read` destroyed three sessions' in-flight work and
#: logged it as "silent for 1s" one second after the peer had spoken. Two
#: consecutive misses on a link already proven, with the peer's last frame at
#: least one ping interval back, is the bar. Not a tuned number: 2 is the
#: smallest count that distinguishes "did not answer once" from "is not
#: answering", and the strike counter resets on any frame the peer sends, so a
#: live link can never accumulate its way to a teardown.
#:
#: "Consecutive" here means consecutive OBSERVATIONS, not consecutive probes: a
#: second miss only counts when its probe began after the window of the previous
#: one closed (`note_probe_strike`). Two sessions whose commands time out at the
#: same moment are the shape this rule is FOR, and they produce two misses
#: inside one window — counting those as two would spend the whole bar on one
#: silent instant and sever, which is the single-miss teardown again (review R1:
#: misses 0.6 ms apart, `close 4000`).
PROBE_STRIKES_BEFORE_SEVER = 2
#: How late one ping-tick cycle may be relative to `PING_INTERVAL_S` before the
#: daemon says so at WARNING. A late tick is the daemon's own event loop failing
#: to run, which produces the SAME silence measurement a mute peer does — the
#: ambiguity that made the original incident take hours to localise — so it has
#: to be visible in the log the operator actually gets (`~/*/log/browser-bridge.log`
#: holds WARNING and above: no code path here configures the root logger below
#: that, so an INFO line about the lag would be written nowhere). One second is
#: well above a healthy tick's jitter and far below the multi-second starvation
#: that produces a failed probe.
PING_TICK_LAG_WARN_S = 1.0
#: How often the daemon re-reads the pairing file to notice an out-of-process
#: revoke. Short enough that "Unpair" feels immediate, cheap enough to poll.
REVOKE_WATCH_S = 3.0
PAIR_TTL_S = 120.0
PAIR_MAX_ATTEMPTS = 5

#: WebSocket close codes that mean "the SERVER is going down", as opposed to
#: The ONE WebSocket close code that means "the SERVER is going down", as opposed
#: to "this peer went away". uvicorn's websocket protocols put 1012 ("service
#: restart") on every live connection while the server shuts down, and it is
#: delivered before the lifespan shutdown event — which is how a daemon served
#: straight from `create_app()` learns it is leaving in time (QA round 3, Q3-1;
#: see `_daemon_leaving`).
#:
#: 1012 ALONE, deliberately (review round 4, finding 1). The first version also
#: accepted 1001 ("going away") on the theory that other ASGI servers use it — but
#: in this position a disconnect only carries a code the PEER sent it: uvicorn's
#: `asgi_receive` returns its own code solely from `self.close_code`, i.e. the
#: client's frame, while its shutdown path is 1012. 1001 is therefore a browser
#: navigating away or tearing a socket down, and latching the daemon's permanent
#: "we are leaving" flag on it froze the durable driver record and disabled
#: failover for the life of the process — the R2-1 harm, silently and forever,
#: from one client close. Two further facts make the single code sufficient:
#: a close frame with NO status code (what a browser's `close()` sends) arrives as
#: 1005, not 1000, on the installed uvicorn (`websockets_impl.py:377`, `:386`), and
#: the extension's own four `wire.close()` calls send no code either — so neither
#: can be confused with 1012, and no runner needs to be trusted for it.
#:
#: WHAT THIS DOES NOT GUARANTEE (review round 5, finding 1). A browser cannot send
#: 1012 (`WebSocket.close()` rejects codes outside 1000 and 3000-4999), but the
#: Origin header is shape validation, not a boundary, so a LOCAL process can dial
#: `chrome-extension://` shaped and put 1012 on the wire itself. There is no
#: corroboration available at this point to tell that frame from uvicorn's own
#: teardown: the server's shutdown path puts 1012 on the socket BEFORE the lifespan
#: event that sets `_shutting_down`, and `asgi_receive` returns one code for both
#: cases, so "the server is leaving" cannot be cross-checked here without a signal
#: the app does not get before the close. The mitigation is therefore a narrowing,
#: not a proof: the latch fires only for a link that holds PAIRING AUTHORITY or the
#: WHEEL (see the read site), which is exactly the set of links whose ending can
#: move the durable record — so a stranger's forged 1012 is inert, and reaching the
#: freeze now requires a local process that either pairs (i.e. obtained the terminal
#: code) or first took the free wheel.
#:
#: The RESIDUAL RISK, stated rather than implied: a local process that takes the
#: free wheel with no install attached and then closes with 1012 can still latch
#: this flag for the life of the process, which freezes the durable record and
#: disables failover (a denial of service against the bridge, not a disclosure).
#: Closing it would need the daemon to corroborate "the server is leaving" from a
#: trusted signal — the production entrypoint has one (`watch_server_exit` attaches
#: the server object and `_daemon_leaving()` reads `server.should_exit` lazily when
#: asked, which is why that path is tested separately), an ASGI app with no server
#: object does not. (Review round 6, NIT 2: this used to say the watcher "polls"
#: the flag, which sizes the mechanism the wrong way — there is no poll loop to
#: look for, and no interval at which the signal is sampled.)
SERVER_GOING_DOWN_CLOSE_CODE = 1012
PAIRING_FILENAME = "browser/pairing.json"
PENDING_FILENAME = "run/browser/pairing-pending.json"

#: How many links with NO pairing may hold a socket at once before the oldest is
#: retired. Every token-less dial is admitted, because that is how a second
#: install asks to pair, and each one holds a socket, a label, a link entry and a
#: pending-code record — all of it grown by a single unauthenticated frame (review
#: round 1, m1). Four is generous for the operator's own case (two installed
#: builds, plus headroom for a reinstall) while keeping the map bounded, and the
#: OLDEST stranger is the one retired, so a legitimate new install is never
#: refused because somebody else's stale socket got there first.
MAX_UNLISTED_LINKS = 4

#: How long a pairing record survives after its code expires. Kept past the
#: expiry so the honest "that code expired" answer still has a record to read
#: (`_try_pair` distinguishes expired from wrong), then dropped: the A1 lockout is
#: the ROTATION itself, not the record, so forgetting a long-dead entry cannot
#: hand anybody authority — a fresh dial from an unpaired identity mints a fresh
#: code it must still read off the terminal (review round 1, m1).
PENDING_RETENTION_S = 2 * PAIR_TTL_S


def _prune_pending(entries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Drop pending records whose code expired more than one TTL ago."""
    now = time.time()
    return {
        extension_id: entry
        for extension_id, entry in entries.items()
        if now - float(entry.get("expires_at", 0) or 0) < PENDING_RETENTION_S
    }


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


def _previous_record_id(root: Path | None) -> str:
    """The identity the legacy trio names RIGHT NOW, or "" when there is none."""
    return str((_read_json(_pairing_path(root)) or {}).get("extension_id", ""))


def _driver_record_id(identities: list[dict[str, Any]], *, driver_id: str, previous: str) -> str:
    """Which identity the legacy trio should name, in the documented order.

    ONE function so every writer and the change-detection in ``_record_driver``
    cannot drift apart:

    1. ``driver_id`` — the caller knows who holds the wheel;
    2. else the identity the trio already names, when it is still listed — the
       trio is the DRIVER's record, so pairing or revoking a standby must not
       move it (round 1, M2/Q2);
    3. else the most recently paired survivor.

    Returns "" only when there is nobody to name.
    """
    listed = {str(entry.get("extension_id", "")) for entry in identities}
    if driver_id and driver_id in listed:
        return driver_id
    if previous and previous in listed:
        return previous
    if not identities:
        return ""
    newest = max(identities, key=lambda entry: float(entry.get("paired_at", 0) or 0))
    return str(newest.get("extension_id", ""))


def normalise_target(target: str) -> str:
    """A user-supplied target with the display ellipsis stripped.

    `status` and `pair --list` print handles as `ohcmfhja…`, and copying what the
    screen shows is the single most likely user action — so the printed form has
    to resolve as printed (UX round 2, U8 / copy review C8). Only a TRAILING
    ellipsis is removed: no extension id contains one, so a target ending in one is
    either a pasted handle or a typo, and a prefix search is the right reading in
    both cases. Lives here rather than in `cli.py` because the daemon resolves
    targets too (`POST /driver`), and the two must not drift.
    """
    return target.strip().rstrip("\u2026").strip()


def _write_pairing(
    root: Path | None, identities: list[dict[str, Any]], *, driver_id: str = ""
) -> None:
    """Write the allow-list, keeping the legacy trio as the DRIVER's record.

    The top-level ``extension_id``/``token_sha256``/``paired_at`` keys are the
    DOWNGRADE CONTRACT, not redundancy: an older daemon (or an older `lop`)
    reading this file finds exactly ONE identity — the one it will authorise and
    let drive — so a rollback degrades to single-identity instead of breaking
    pairing.

    Which one, in order (review round 1, M2/Q2):

    1. ``driver_id``, when the caller names a listed identity — pairing or
       driving has just told us who holds the wheel;
    2. otherwise the identity the trio ALREADY names, if it is still listed.
       This is the case that was wrong: pairing a second install, or revoking a
       standby, must not move the record onto the newcomer, because an older
       daemon reading a NEWER file would then authorise the standby and refuse
       the install the operator is actually using — a forced re-pair, on the one
       path decision 1 promises never needs one;
    3. otherwise the most recently paired survivor. Only reachable when the
       identity the trio named has just been revoked and nothing has taken the
       wheel yet, where a coherent trio naming somebody is better than none.

    Every path that MOVES the wheel rewrites the file through
    ``_record_driver`` (promotion on a driver loss, ``drive``, pairing), so
    (2) is a *stable* record rather than a stale one.
    """
    chosen_id = _driver_record_id(
        identities, driver_id=driver_id, previous=_previous_record_id(root)
    )
    chosen = next((entry for entry in identities if entry.get("extension_id") == chosen_id), None)
    if chosen is None and identities:  # pragma: no cover - chosen_id comes from `identities`
        chosen = identities[0]
    payload: dict[str, Any] = {}
    if chosen is not None:
        payload["extension_id"] = chosen.get("extension_id", "")
        payload["token_sha256"] = chosen.get("token_sha256", "")
        payload["paired_at"] = chosen.get("paired_at", 0.0)
    payload["schema"] = 2
    payload["identities"] = identities
    _private_write(_pairing_path(root), payload)


#: How stale a recorded ``last_seen_at`` may get before a handshake refreshes it.
#: The field is what `lop browser pair --list` prints as "last seen", i.e. the one
#: line that tells an operator which of two identically-labelled installs is
#: actually in use — so it has to be a liveness signal rather than a copy of
#: `paired_at` (review round 4, finding 2). Refreshing on EVERY dial would put a
#: synchronous rewrite of the pairing record on the handshake path that every
#: install's every reconnect pays, for a line nobody reads at that rate; one
#: write per identity per minute is the compromise, and it is bounded by this
#: constant rather than by traffic.
LAST_SEEN_REFRESH_S = 60.0


def note_identity_seen(root: Path | None, extension_id: str) -> None:
    """Refresh one identity's ``last_seen_at`` once it has gone stale.

    Called from the handshake. What it does in each case, stated because the first
    version of this function got the third one wrong and broke the zero-migration
    promise (QA round 5, Q5-3):

    * **no entry for this id** → return; a dial cannot create pairing authority.
    * **entry with NO ``last_seen_at``** (a schema-1 record, i.e. every file written
      before this field existed) → return. Absent is *unknown*, never "infinitely
      stale": the staleness test read the missing stamp as ``0.0``, so the first
      handshake of a legacy install rewrote the operator's pairing file 163 B →
      392 B and schema 1 → 2. Decision 1 promises zero migration in BOTH
      directions, and a silent rewrite of the user's own file on connect is exactly
      the user-state mutation the design exists to avoid.
    * **entry whose stamp is malformed** → return; a broken value is not evidence
      of staleness either.
    * **stamp is fresh** (within ``LAST_SEEN_REFRESH_S``) → return, no read-modify-
      write at all beyond the parse. This is the ordinary reconnect.
    * **stamp is stale** → one guarded write, which is the only case that touches
      the file:

      - the identities are re-read immediately before the write, and the write is
        skipped if this id is gone from that fresh read — ``lop browser pair
        --revoke`` runs in a SEPARATE process and writes this same file, and the
        failure direction of clobbering it is authority-critical (a revoke that
        reports success and does not stick). The race window is the span between
        that read and ``os.replace``, measured on a loaded host (QA round 6) at
        **261 µs min / 351 µs median / 108.8 ms max** over 50 refreshes — the
        worst case is two orders of magnitude above the typical one, because the
        process can be suspended anywhere in that span. It cannot be closed from
        here because the file has no lock and the other writer is another process,
        so the window is NARROWED, not closed, which is why it is also stated in
        ``_write_pairing``'s contract rather than implied to be zero — do not read
        the typical figure as a guarantee under contention.
      - ``OSError`` is absorbed, because this runs on the handshake path where a
        config root that is full or read-only must not fail the connection — this
        CALL, specifically. ``_record_driver()`` runs two lines earlier on the same
        path and is deliberately NOT guarded (a wheel move that cannot be written
        must fail loudly rather than serve a route the record does not name), so a
        read-only root still raises out of the handshake whenever the trio has to
        move. Absorbing here only keeps a cosmetic timestamp from doing the same.
        Same reason ``publish_safely`` exists next door. A refresh that cannot be
        written is a stale timestamp, not a broken bridge.

    Deliberately routed through ``_write_pairing``, the single writer, so the legacy
    trio keeps naming the DRIVER (rule 2 of ``_driver_record_id``) rather than the
    install that just dialled: this file is also the rollback contract, and a
    cosmetic refresh must not move it.
    """
    entry = next(
        (item for item in _identities(root) if str(item.get("extension_id", "")) == extension_id),
        None,
    )
    if entry is None:
        return
    stamp = entry.get("last_seen_at")
    if stamp is None:
        return
    try:
        age = time.time() - float(stamp)
    except (TypeError, ValueError):
        return
    if age < LAST_SEEN_REFRESH_S:
        return
    # Re-read: the revoke watcher's caller is another process writing this file, and
    # losing a revoke to a cosmetic timestamp is the one failure direction this
    # function may not have.
    current = _identities(root)
    if not any(str(item.get("extension_id", "")) == extension_id for item in current):
        return
    refreshed = [
        (
            {**item, "last_seen_at": time.time()}
            if str(item.get("extension_id", "")) == extension_id
            else item
        )
        for item in current
    ]
    with suppress(OSError):
        _write_pairing(root, refreshed)


def add_identity(
    root: Path | None,
    extension_id: str,
    token_sha256: str,
    *,
    label: str = "",
    driver_id: str = "",
) -> None:
    """Authorise one extension identity, replacing any existing entry for it.

    Re-pairing the same identity (a wiped token, a new browser profile that
    happens to derive the same id) must ROTATE its credential rather than add a
    second entry, or the old hash would stay live alongside the new one and a
    stale token could still authenticate.

    One token per identity, never one shared token: revocation has to be a fact
    about the file rather than a hope about a spoofable Origin header
    (design decision 2), and per-identity hashes cost one dict lookup.

    ``driver_id`` names the identity that is driving AT THIS MOMENT, which is
    usually NOT ``extension_id``: the second install pairs while the first
    holds the wheel, and the legacy trio must keep naming the first (M2/Q2).
    Callers pass ``""`` when nothing is driving, and the trio then names the
    identity just paired — it took the wheel (``_take_free_wheel``).
    """
    identities = [entry for entry in _identities(root) if entry.get("extension_id") != extension_id]
    identities.append(
        {
            "extension_id": extension_id,
            "token_sha256": token_sha256,
            "paired_at": time.time(),
            "label": label,
            # Seeded at pairing time and refreshed by `note_identity_seen` on a
            # handshake once it is more than LAST_SEEN_REFRESH_S stale. It is NOT
            # read from /health: this field lives in the file, and the CLI reads
            # the file, so a claim that the daemon reports a live value over
            # HTTP was simply wrong (review round 4, finding 2).
            "last_seen_at": time.time(),
        }
    )
    _write_pairing(root, identities, driver_id=driver_id or extension_id)


def revoke_identity(root: Path | None, extension_id: str, *, driver_id: str = "") -> None:
    """Remove ONE identity, and only that one, from the allow-list.

    The remaining identities keep their own hashes, so a revoke is a real
    revocation for the caller and a no-op for everybody else — which is what
    makes "revoke the dev build" possible without disturbing the store build.
    Emptying the list removes the file entirely, so "nothing is authorised" has
    one representation rather than two.

    ``driver_id`` is the SURVIVING driver when the caller knows one (a revoke
    that promotes a standby writes the new driver into the legacy trio);
    otherwise the existing record is preserved where it is still listed, which
    is what keeps "revoke the dev build" from renaming the store build.
    """
    identities = [
        entry for entry in _identities(root) if str(entry.get("extension_id", "")) != extension_id
    ]
    if not identities:
        with suppress(OSError):
            _pairing_path(root).unlink()
        return
    _write_pairing(root, identities, driver_id=driver_id)


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
        return _prune_pending(
            {str(key): value for key, value in entries.items() if isinstance(value, dict) and key}
        )
    # Schema 1: the record was the single waiting identity's entry.
    extension_id = str(saved.get("extension_id", ""))
    return {extension_id: saved} if extension_id else {}


def _write_pending(root: Path | None, entries: dict[str, dict[str, Any]]) -> None:
    if not entries:
        with suppress(OSError):
            _pending_path(root).unlink()
        return
    _private_write(_pending_path(root), {"pending": entries})


def _resolve_authorised_target(
    target: str, identities: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve a target against the FILE's identities, for messages only.

    Mirrors `lop browser`'s own rules (exact id, then id prefix, then label
    substring) so a target the CLI printed resolves the same way in both places,
    and returns None on ambiguity: guessing which of two identical labels the
    operator meant is not this daemon's decision to make. Deliberately used only
    to choose BETWEEN error messages — authorisation never flows through a
    string a user typed.
    """
    wanted = target.strip().lower()
    if not wanted:
        return None
    for entry in identities:
        if str(entry.get("extension_id", "")).lower() == wanted:
            return entry
    prefixed = [
        entry
        for entry in identities
        if str(entry.get("extension_id", "")).lower().startswith(wanted)
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    labelled = [entry for entry in identities if wanted in str(entry.get("label", "")).lower()]
    return labelled[0] if len(labelled) == 1 else None


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
    # "Chrome extension 0.1.13", never "Chrome 0.1.13" (copy review C3): Chrome
    # itself is at 153 in the reported case, so the bare form reads as an ancient
    # or broken BROWSER, and the string's whole job is to say which LOCAL OPERATOR
    # build in which browser holds the wheel. The word costs one line and removes
    # the misreading; the version still does the disambiguating work.
    if not name:
        return f"extension {extension_version}" if extension_version else "extension"
    return f"{name} extension {extension_version}" if extension_version else name


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
                # "This entry is the DRIVER'S RECORD" — the legacy top-level
                # trio, which every path that moves the wheel now rewrites
                # (`_record_driver`), so this agrees with /health's
                # `driver_extension_id` whenever a driver is attached. It was
                # wrong before that fix: the file named the last-PAIRED install
                # (QA round 1, Q3), so this flag called a standby the driver.
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
        # Unanswered liveness probes on THIS socket that each observed a
        # DISTINCT window; any frame the peer sends clears both fields.
        # Link-scoped rather than service-scoped on purpose: a replacement
        # socket is a new object, so a reconnect can never inherit the strikes
        # of the mute link it replaced. See `PROBE_STRIKES_BEFORE_SEVER` for why
        # one miss is not enough to act on and `note_probe_strike` for why the
        # count is of windows rather than of probes.
        self.probe_strikes = 0
        # Monotonic instant at which the window that produced the latest counted
        # strike closed; 0.0 before any. A probe that began BEFORE this is
        # overlap of that same observation and is not counted again — the rule
        # that keeps N commands timing out together from spending the two-strike
        # budget in one silent window (review R1).
        self.probe_window_closed_at = 0.0
        self.extension_id = ""
        self.browser = ""
        # The peer's own protocol version and reported extension version, both
        # stamped from its `hello` — here, on the link being CREATED, and before
        # that link is registered or can be driving: a standby dial must stamp
        # its own socket's identity, never the incumbent driver's (`self.link` is
        # the DRIVING link since #1038's multi-identity handshake). Three
        # consumers depend on them:
        #
        #   * `peer_proto` is what any FUTURE daemon->extension frame must be
        #     gated on — "can this peer understand it" is a question about the
        #     peer, never about this daemon's own PROTO_VERSION.
        #   * `extension_version` is what tells a pre-ownership extension from a
        #     current-but-wedged one when `owner_recover` fails identically on
        #     both (see `OWNERSHIP_MIN_EXTENSION_VERSION`).
        #   * …and it labels the pairing entry with the build it belongs to, which
        #     is what lets an operator tell two installs of the same browser apart
        #     in `lop browser pair --list`.
        #
        # `peer_proto`'s default matters: it is this daemon's own version, so a
        # never-stamped link reports a value the daemon itself would speak. That
        # is only a safety net for a link that has not handshaken; `health`
        # reports both fields as "unknown" unless the link is PROVEN, since a
        # stale stamp from a dead socket must not drive a decision.
        self.peer_proto: int = PROTO_VERSION
        self.extension_version: str = ""
        # What this peer ADVERTISED it serves, learned from its `capabilities`
        # event and kept per-socket like every other stamp here: a stale list
        # would authorise a method the peer that replaced it does not have. An
        # older extension never sends the event, so this stays empty and the
        # daemon refuses `download`/`upload` for it with a typed
        # `capability_unsupported` rather than burning a 120 s budget on a
        # method the worker answers with a bare `internal`.
        self.capabilities: list[str] = []
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

    def note_probe_strike(self, observed_from: float) -> tuple[int, bool]:
        """Count an unanswered liveness probe that observed a window of its own.

        ``observed_from`` is when the probe BEGAN. A miss is only counted as a
        new strike when that instant is at or after the close of the window that
        produced the previous one; otherwise this probe overlapped the earlier
        observation and cannot establish anything it did not.

        Without that rule the counter measures probes, not windows, and probes
        overlap by construction under exactly the traffic the rule exists for:
        N sessions whose commands time out at the same moment run N probes
        inside ONE ``PING_PROBE_TIMEOUT_S`` window, so N=2 met the two-strike bar
        on a single silent window and severed the link — the pre-fix behaviour,
        reproduced by review R1 with two misses 0.6 ms apart (`close 4000`,
        latched, every pending future failed). Folding them into one strike is
        the fix; the second call returns the same total and reports that it
        opened no window, so the caller can say so in its log line instead of
        silently re-counting.

        The boundary is the window's own close, stamped here AFTER the wait,
        rather than the reviewer's suggested ``now - last_strike_at >=
        PING_PROBE_TIMEOUT_S``. Both stop the concurrent burst; the elapsed gap
        between two misses is a proxy for overlap and is wrong in the other
        direction too — two probes that began a full interval apart but resolved
        close together did NOT share a window, and a solicitation that could not
        be delivered at all fails in microseconds, which an interval-long
        required gap would read as the same window forever.

        Returns ``(running total, opened_a_window)``.
        """
        if observed_from < self.probe_window_closed_at:
            return self.probe_strikes, False
        self.probe_strikes += 1
        self.probe_window_closed_at = time.monotonic()
        return self.probe_strikes, True

    def clear_probe_strikes(self) -> None:
        """The peer spoke, so nothing is left to corroborate.

        Called when a probe is ANSWERED (the receive loop sets `frame_event` on
        every frame, not only on a pong), which is what keeps the strike count a
        measure of CONSECUTIVE misses rather than a tally: a link that answers
        once, ever, is back to zero — and so is the window boundary, so the next
        miss after a frame opens a fresh window instead of being folded into the
        one that preceded it.
        """
        self.probe_strikes = 0
        self.probe_window_closed_at = 0.0

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
        # Link-scoped like everything else here: the extension that reported
        # them is gone, so the reported version must not outlive its socket (a
        # stale version would drive the update advisory and the ownership
        # split against a peer that is no longer talking). `peer_proto` returns
        # to its declared default rather than 0 — there is no peer to describe,
        # and the field's own contract is "the proto the peer would speak".
        self.peer_proto = PROTO_VERSION
        self.extension_version = ""
        # Capabilities are per-socket too, for the same reason the version is:
        # a list outliving its socket would let a method be sent to a peer that
        # never advertised it.
        self.capabilities = []
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
        #: Set once the daemon has been asked to stop (see `_daemon_leaving`).
        self._shutting_down = False
        #: The uvicorn server object, when a runner provides one. Its
        #: `should_exit` is set before sockets are closed, which is the earliest
        #: in-process signal that a socket ending is teardown, not a driver loss.
        self._server: Any = None

        self._ping_task: asyncio.Task[None] | None = None
        self._revoke_task: asyncio.Task[None] | None = None
        # Ping-tick cadence, for the loop-lag record: when the previous tick
        # landed and what lag it measured. Kept on the SERVICE rather than on a
        # link because the loop is the daemon's, not a socket's — a starvation
        # spans reconnects, which is exactly the case it exists to describe
        # (`_note_ping_tick_lag`).
        self._ping_tick_at = 0.0
        self._ping_tick_lag_s = 0.0
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
        """Attached links that are not the driver, oldest attachment first.

        PAIRED ones only (review round 1, m2). An unlisted, token-less dial is
        admitted so it can PAIR, and it is not a standby in the sense this role
        means: it holds no authority, receives no commands, and `_promote_standby`
        will never promote it. Listing it as one told the operator — in the popup
        and in `lop browser status` — that an install for a peer with no pairing
        at all "is standing by", which is a false statement about a real thing.
        The count of attached-and-mute strangers stays observable through
        `unlisted_links()`.
        """
        return [link for link in self.other_links() if link.paired]

    def other_links(self) -> list[ExtensionLink]:
        """Every attached link that is not the driver, paired or not.

        The role-neutral enumeration. Anything USER-FACING wants
        ``standby_links`` (a paired install standing by is a fact the operator
        can act on); role bookkeeping and the bound on unlisted dials want this
        one, because an admitted-but-unpaired dial is still a link holding a
        socket that must be tracked and bounded.
        """
        return [
            link
            for generation, link in sorted(self.links.items())
            if generation != self.driver_generation and link.websocket is not None
        ]

    def unlisted_links(self) -> list[ExtensionLink]:
        """Attached links with no pairing at all: dials asking to be let in."""
        return [link for link in self.other_links() if not link.paired]

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
        candidates = [link for link in self.standby_links() if link.proven]
        if not candidates:
            return None
        promoted = max(candidates, key=lambda link: link.attached_at)
        self.driver_generation = promoted.generation
        promoted.role = "driver"
        for link in self.other_links():
            link.role = "standby"
        # The wheel moved, so the downgrade record moves with it (M2/Q2). Inside
        # the no-await block: it is a synchronous file write, like `publish`.
        self._record_driver()
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

    def _live_driver_id(self) -> str:
        """The extension id holding the wheel, or "" when nothing is attached.

        The ONE place the driver's identity becomes an id for the pairing file.
        ``self.link`` already resolves to the driver (#996's property), and the
        idle link carries no id, so an idle wheel reports "" rather than a stale
        connection's id.
        """
        driver = self.link
        if driver.websocket is None:
            return ""
        return str(driver.extension_id or "")

    def _record_driver(self) -> None:
        """Keep the legacy trio naming the LIVE driver — the DURABLE INVARIANT.

        The invariant this maintains, stated precisely so it cannot be read as
        more than it is: **the trio names a LISTED identity, and whenever a
        paired driver is attached it names that driver** — so the file agrees
        with `/health.driver_extension_id` in every state an operator can be
        served from. The one state where the two differ is a token-less dial
        TEMPORARILY holding the wheel (admitted so it can pair): it is not a
        driver that can serve, it has no token hash to record, and the file
        therefore keeps naming the last listed driver — which is precisely what
        the note's "the file must not name a non-driver" asks for. That state
        ends at its next dial or at its pairing, both of which come back through
        here.

        It is what the downgrade contract means (decision 1): a rollback, or an
        older `lop`, reads exactly one identity out of this file, and it must be
        the install the operator is actually using — not a standby it will
        refuse with 4004.

        Called from EVERY path that moves or settles the wheel: both handshake
        branches (round 2, R2-1 — the cold-start dial after a restart was the
        hole), a promotion on a driver loss, `_take_free_wheel`, `POST /driver`,
        and a revoke that promotes. Cheap and synchronous because callers run it
        inside the no-await decision blocks (audit A1); it writes only when the
        name would CHANGE, so a reconnect that leaves the wheel where it was
        costs three file reads and no atomic write — `_pairing_path().exists()`,
        `_identities()` and two `_previous_record_id()` calls for the comparison
        (review round 3, N3: this comment said ONE read, which is what a future
        reader would size this path by). The reads are cheap and the write is the
        expensive part, so the comparison is deliberately not hoisted yet: a
        cached previous id would have to be invalidated on every write path for
        an amount of work measured in microseconds.

        Declines to write while the daemon is going down (round 2, R2-1, QA's
        half): a promotion during teardown is an artifact of us closing sockets,
        not a fact about which install the operator is using, and persisting it
        made a rollback bind to the standby — flakily, which is worse (measured
        at 2/6 SIGTERM runs). The record then keeps naming the last install that
        was driving, which is the install that will drive again after the
        restart, and the next handshake reconciles it if not.
        """
        if self._daemon_leaving():
            return
        if not _pairing_path(self.root).exists():
            return
        identities = _identities(self.root)
        if not identities:
            return
        wanted = _driver_record_id(
            identities, driver_id=self._live_driver_id(), previous=_previous_record_id(self.root)
        )
        if not wanted or wanted == _previous_record_id(self.root):
            return
        _write_pairing(self.root, identities, driver_id=self._live_driver_id())

    def _daemon_leaving(self) -> bool:
        """True once this daemon has been asked to stop, from ANY of three sources.

        Three, because the first two do not cover every way the service is run —
        and the third is the one that makes the property hold for a daemon served
        straight from `create_app()` by a runner that wires nothing for us:

        1. `begin_shutdown()`, set by the lifespan `shutdown()` handler;
        2. the HTTP server's own `should_exit`, when a runner called
           `watch_server_exit()` (uvicorn sets it before it closes listeners or
           connection sockets, see `Server.shutdown`);
        3. `begin_shutdown()` again, this time from the RECEIVE LOOP: uvicorn's
           websocket protocols deliver a `websocket.disconnect` with code 1012
           to every live connection when the server is going down, and they do
           it BEFORE the lifespan shutdown event (QA round 3, Q3-1: with sources
           1 and 2 alone, a `uvicorn.run(create_app(...))` daemon promoted on the
           way out in 9 of 12 runs and wrote the standby into the downgrade
           record, which is how a rollback ends up refusing the install in use).

        Sources 1 and 2 are still worth keeping: 1 covers a runner that never
        touches the socket (a test client), 2 covers a runner that programmatically
        stops without a signal reaching Python's socket layer.
        """
        if self._shutting_down:
            return True
        return bool(getattr(self._server, "should_exit", False))

    def begin_shutdown(self) -> None:
        """Mark the daemon as going down; called by the lifespan, a runner, or the
        receive loop (a server-initiated close — see `_daemon_leaving`).

        Public because a runner that owns the server object may know earlier than
        the lifespan does (uvicorn fires the lifespan shutdown only AFTER it has
        finished tearing connections down), and because the receive loop learns it
        from the close itself even when nobody wired the server for us.
        """
        self._shutting_down = True

    def watch_server_exit(self, server: Any) -> None:
        """Attach the HTTP server this service is served by (see `_daemon_leaving`)."""
        self._server = server

    def _reconcile_driver_record(self) -> None:
        """Make the legacy trio name a LISTED identity at daemon start.

        Startup is the one moment the file is authoritative and nothing is
        attached, so it is where a stale record self-heals instead of being
        assumed correct: after an out-of-process `pair --revoke` of the identity
        the trio names, the file would otherwise point at somebody who is no
        longer authorised, and a rollback would authorise nobody. Repair uses the
        same order as every other write (the surviving entries, newest first);
        when the trio already names a listed identity it is left exactly as it
        is, so an ordinary restart does not touch the file.
        """
        if not _pairing_path(self.root).exists():
            return
        identities = _identities(self.root)
        if not identities:
            return
        previous = _previous_record_id(self.root)
        listed = {str(entry.get("extension_id", "")) for entry in identities}
        if previous and previous in listed:
            return
        _write_pairing(self.root, identities, driver_id="")

    def _take_free_wheel(self, link: ExtensionLink) -> list[ExtensionLink]:
        """Give a newly PAIRED link the wheel when nothing that can SERVE holds it.

        A link can become paired while standing by — the second install pairing
        through its own socket, or re-pairing after a revoke — and the wheel can
        be idle at that moment, because the driver went away and the only
        surviving standby was unpaired (which `_promote_standby` deliberately
        will not promote). Without this the install would sit paired and
        stationary while the daemon answered `extension_disconnected` to every
        session: reachable by doing exactly what the popup tells the user to do.

        The same "paired outranks unpaired" rule the handshake and the promotion
        path apply (review round 3, R3-3), and the reason this is not just an
        idle check: an UNPAIRED dial can hold the wheel — the handshake gives it
        to whoever dials when nothing else does — and it cannot answer a single
        command, so the install that just proved it can (the code was entered)
        takes the wheel from it here rather than waiting for that dial's own next
        handshake to demote it. Bounded either way, but "until the next dial" can
        be the ~1 minute alarm floor, which is a user watching a dead agent.

        Returns every link that must be TOLD about the move (the new driver, and
        the demoted one when there was one): a decision made here is published in
        the same no-await block by the caller, and the role frames follow after.
        """
        holder = self.link
        if holder.websocket is not None and holder.paired:
            return []
        demoted = holder if holder.websocket is not None else None
        link.role = "driver"
        self.driver_generation = link.generation
        if demoted is not None:
            demoted.role = "standby"
        self._record_driver()
        return [link, demoted] if demoted is not None else [link]

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
        # No handover while the daemon is going down (round 2, R2-1, QA's half).
        # Promoting then would move the wheel for nobody: the process is
        # exiting, the promoted install will take the wheel on its own next dial
        # after the restart, and `_record_driver()` would persist a driver the
        # operator never used — which is how a graceful stop wrote the STANDBY
        # into the downgrade record and made a rollback refuse the install in use
        # (flaky: 2/6 SIGTERM runs). The wheel is left pointing at the idle link
        # by the branch below, so every `self.link` read still says "no driver".
        promoted = self._promote_standby() if was_driver and not self._daemon_leaving() else None
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
        # file carries the same WHY as `/health`. Note the two are no longer the
        # same VALUE and must not be compared (review round 5, NIT 4): this field
        # is `drop_latched()` ungated, while `/health`'s `extension_unresponsive`
        # also asks whether a proven link is serving, so for the TTL window after a
        # promotion the file says "a drop is latched" and `/health` says "the
        # driver is answering". Both are true of different questions; the file is
        # read by the demotion guard, which wants the latched fact.
        # It matters because the
        # demotion guard in `tools/builtin.py` decides from the FILE (a probe on
        # the ABSENT side is forbidden by `bridge_browser_reachable`'s contract)
        # and a drop writes `extension_connected=false` — without this the file
        # is indistinguishable from a host with no bridge, which is how a paired
        # running bridge got told to run `lop browser install` (design D3-2).
        self.state.extension_unresponsive = self.drop_latched()
        # The extension's identity, published for consumers that CANNOT open a
        # socket: `BrowserResource` decides whether it may use the ownership
        # lifecycle from these fields (see `resources.py`), and the session-side
        # browser tool must not pay a round-trip on a decision the daemon has
        # already made about the link it owns.
        #
        # Blanked when the link is not proven, deliberately. A version stamp
        # outliving its socket would drive both the update advisory and the
        # ownership split against a peer that is no longer talking — which is
        # exactly the stale-lie class the discovery file already exists to stop
        # (see `extension_unresponsive`'s own comment).
        if self.link.proven:
            self.state.extension_version = self.link.extension_version
            self.state.extension_proto = self.link.peer_proto
            # Sorted, because the consumer compares membership and the file is
            # read by a human in `lop browser status`; blanked with the version
            # above so a proven-only fact stays proven.
            self.state.capabilities = sorted(self.link.capabilities)
            # The ONE advisory predicate, shared with `/health`: a KNOWN version
            # strictly below the one this runtime ships with. Unparseable is not
            # "older" and an extension AHEAD is not behind, so neither nags.
            self.state.extension_update_available = extension_older(
                self.link.extension_version, EXPECTED_EXTENSION_VERSION
            )
        else:
            self.state.extension_version = ""
            self.state.extension_proto = 0
            # Nothing is served by a link that is not talking: the empty list is
            # what makes the session-side capability check refuse rather than
            # send into a socket nobody is reading.
            self.state.capabilities = []
            self.state.extension_update_available = False
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

    def _note_ping_tick_lag(self) -> None:
        """Record how late this ping tick is, and say so when it matters.

        A teardown for silence and a daemon whose own event loop was starved
        produce the SAME evidence — nothing heard from the peer — and, before
        this, the same log line. The difference is whether the daemon actually
        got to ASK: a lagging tick means the question itself was late. Publishing
        the tick's own lateness beside the peer's silence is what makes the two
        tellable apart, which is why the original incident took hours to localise
        (every log line said the browser had gone quiet; none said the daemon had
        stopped running).

        The lag is `observed cycle − PING_INTERVAL_S`, i.e. this tick's body plus
        whatever the loop and scheduler added on either side. A tick that FAILED
        (the supervisor's backoff before it re-runs) shows up here as lag too,
        and that is the right reading rather than a false positive: during the
        backoff this loop was not evaluating liveness either, which is the exact
        distinction the line exists to publish. LEVELS ARE
        DELIBERATE: the daemon runs at WARNING in production (`browser serve`
        configures no logging, and the launchd/supervisor log it writes to is
        WARNING-and-above), so a per-tick INFO line would be written nowhere. The
        per-tick record is therefore DEBUG for anyone who enables it, the lag is
        announced at WARNING when it crosses `PING_TICK_LAG_WARN_S`, and it also
        rides on every unproven-drop warning (`_drop_unproven_link`), which is
        the line a reader actually meets during an incident.
        """
        now = time.monotonic()
        previous = self._ping_tick_at
        self._ping_tick_at = now
        if previous <= 0.0:
            # First tick: there is no interval to be late for yet.
            return
        lag = (now - previous) - PING_INTERVAL_S
        self._ping_tick_lag_s = max(0.0, lag)
        silent = self.link.silent_for()
        logger.debug(
            "browser bridge ping tick: peer silent %.1fs, daemon loop lag %.2fs",
            silent,
            lag,
        )
        if lag > PING_TICK_LAG_WARN_S:
            logger.warning(
                "browser bridge ping tick ran %.1fs late (loop lag); peer silent %.1fs",
                lag,
                silent,
            )

    async def _ping_tick(self) -> None:
        await asyncio.sleep(PING_INTERVAL_S)
        self._note_ping_tick_lag()
        # EVERY attached link, not just the driver's. A standby receives no
        # commands, so a ping is the ONLY traffic it ever sees; without one its
        # liveness would decay past LINK_SILENCE_TIMEOUT_S within the first
        # promotion window, and the failover rule — which promotes only a
        # SURVIVING standby — could never promote anybody. Keeping standbys
        # proven is what makes `_promote_standby`'s proof test meaningful
        # instead of vacuously false.
        for standby in self.other_links():
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
        # Answered: forget any earlier miss, so a link that recovers never walks
        # its way to the sever threshold on misses accumulated minutes apart.
        self.link.clear_probe_strikes()
        return True

    async def _probe_verdict(self, method: str) -> str | None:
        """Whether one failed liveness probe justifies severing, and the reason to.

        Returns ``None`` when the evidence is NOT enough — the caller then answers
        its OWN command with its own timeout and leaves the link, every sibling
        future, and every other session's in-flight command untouched. That
        asymmetry is the whole point of this method: a timed-out command is the
        only caller the daemon has a positive result for, and failing the fleet is
        not a remedy for one slow page.

        Two guards, both from a live multi-session incident:

        * **Recent speech.** An unanswered probe on a link whose last frame is
          younger than ``PING_INTERVAL_S`` establishes that one question went
          unanswered, nothing more. The silence a teardown reports is measured at
          the COMMAND's timeout, so severing on the probe alone announced
          "the link silent for 1s" — a drop one second after the peer had spoken,
          which no reader can reconcile with a browser that is plainly alive.
          The guard reads ``silent_for()`` AFTER the wait, so its reach is one
          full ping interval of silence AT THE MOMENT OF THE VERDICT — about 15 s
          measured at probe start, since the ≤ ``PING_PROBE_TIMEOUT_S`` probe sits
          between the two (QA Q2). That ordering is the correct one and is kept:
          a guard measured at probe start would allow severing a peer that spoke
          15 s before the verdict, which is the same defect with a bigger number,
          while measuring at the verdict means the silence the reader is shown is
          the silence the decision was made on. The effective window is stated
          here rather than in a claim about a bare ``PING_INTERVAL_S``.
        * **Strikes, from distinct windows.** ``PROBE_STRIKES_BEFORE_SEVER``
          consecutive misses are required, and "consecutive" is counted over
          observation WINDOWS, not probes: a miss is only a strike when its probe
          began after the previous window closed (``note_probe_strike``). Link-
          scoped and cleared by any frame the peer sends, so a link that answers
          even once is back to zero and a replacement socket starts clean (see
          ``clear_probe_strikes``).

        The silence in the returned reason is measured HERE, after the probe, so
        the number the reader is shown is the number the decision was made on —
        including the ``PING_INTERVAL_S`` guard above, which reads the same value.
        """
        # The instant this observation BEGAN, captured before the probe so the
        # strike can be folded into the window it actually belongs to: the answer
        # to "is this a new observation?" is a fact about when the question was
        # asked, not about when it gave up.
        probed_from = time.monotonic()
        answered = await self._peer_answers_a_solicited_ping()
        silent = self.link.silent_for()
        if answered:
            return None
        strikes, fresh = self.link.note_probe_strike(probed_from)
        if not fresh:
            # A miss from inside the window that already struck: the peer has
            # now been asked twice inside one silent instant, which is what
            # CONCURRENT commands produce (review R1). It corroborates nothing
            # the first probe did not, so the count does not move and nothing is
            # severed — the next strike needs a window of its own.
            logger.warning(
                "browser bridge probe unanswered inside the window of strike %d/%d "
                "for %s (concurrent command, not a new observation)",
                strikes,
                PROBE_STRIKES_BEFORE_SEVER,
                method,
            )
            return None
        if silent < PING_INTERVAL_S:
            logger.warning(
                "browser bridge probe unanswered but the peer spoke %.1fs ago "
                "(strike %d for %s, not severing)",
                silent,
                strikes,
                method,
            )
            return None
        if strikes < PROBE_STRIKES_BEFORE_SEVER:
            logger.warning(
                "browser bridge probe %d/%d unanswered with the link silent for "
                "%.0fs (not severing yet, %s)",
                strikes,
                PROBE_STRIKES_BEFORE_SEVER,
                silent,
                method,
            )
            return None
        # `strikes` counts observation WINDOWS, not probes (see the docstring's
        # "Strikes, from distinct windows"), so the reason must not report it as
        # a probe count: concurrent command timeouts inside one window strike
        # once, and a reader told "3 probes" would look for three probes.
        return (
            f"{method} unanswered with the link silent for {silent:.0f}s "
            f"({strikes} silent windows)"
        )

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
        logger.warning(
            "browser bridge dropped an unresponsive extension: %s (daemon loop lag %.2fs)",
            reason,
            self._ping_tick_lag_s,
        )
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
        # A revoke CAN move the wheel (revoking the driver promotes a standby), so
        # the record is refreshed from the surviving driver once every socket has
        # been dealt with. `revoke_identity` above deliberately preserved the
        # existing record rather than renaming it, which is right for the usual
        # case — revoking a standby must not rename the store build.
        self._record_driver()
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
        # Before serving anybody: if the file the daemon just read names an
        # identity that is no longer authorised (an out-of-process revoke while
        # this daemon was down), repair it from the surviving entries — the
        # downgrade contract must never point at nobody (round 2, R2-1).
        self._reconcile_driver_record()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        self._ping_task = asyncio.create_task(self._ping())
        self._revoke_task = asyncio.create_task(self._watch_revocation())

    async def shutdown(self) -> None:
        # FIRST, before anything closes a socket: a socket ending from here on is
        # teardown, not a driver loss (see `_record_driver` and `_retire_link`).
        self.begin_shutdown()
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
            # Who holds the wheel RIGHT NOW. Pairing a second install must leave
            # the legacy trio naming the first, or a rollback would authorise the
            # newcomer and refuse the install in use (M2/Q2). "" when the wheel is
            # idle, and then the identity just paired takes both (it is about to,
            # via _take_free_wheel).
            driver_id=self._live_driver_id(),
        )
        self._drop_pending(extension_id)
        link.paired = True
        # Decision and publish stay in one no-await block (audit A1); the role
        # frames are awaited only after the wheel state is visible.
        told = self._take_free_wheel(link)
        self.publish_safely()
        for entry in told:
            await self._tell_role(entry)
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
        # The three rejections below (and the ORIGIN one above) close WITHOUT a
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
        if not proto_supported(hello.proto, low=MIN_SUPPORTED_PROTO, high=PROTO_VERSION):
            # A WINDOW, not an equality (see MIN_SUPPORTED_PROTO). The two
            # release lines move independently and the extension's half sits in
            # store review, so requiring an exact match means a daemon release
            # refuses every installed browser until Google approves the
            # matching extension — with a failure that tells the user to update
            # an extension the store will not serve yet.
            #
            # 4001 is KEPT rather than replaced with a new code: `extension/
            # src/worker.ts` maps it to the popup's "incompatible" card, and a
            # new code would be interpreted only by a future extension while
            # everything unknown renders as a plain disconnect. The reason
            # string is a HINT for a future peer (today's extension does not
            # read it — see docs/design/browser-extension.md 4.2); the two
            # spellings exist so a reader of a debug log can tell which side of
            # the window the peer fell off.
            reason = "proto_too_old" if hello.proto < MIN_SUPPORTED_PROTO else "proto_too_new"
            await websocket.close(code=4001, reason=reason)
            return
        # An UNLISTED identity is admitted whether or not it presents a token,
        # and no token it carries is ever consulted: `link.paired` below is
        # `_valid_saved_token`, which selects the entry for THIS id and can only
        # answer for a listed one. So an unlisted dial is an ASKER in every case
        # — and still needs the terminal code, which is the property decision 3
        # keeps for every identity.
        #
        # What an unlisted dial CANNOT do, stated precisely rather than as
        # shorthand (review round 4, finding 3 — this comment used to say it
        # "cannot drive", which is not what the code does): it takes the driver
        # role on a FREE wheel like any other dial, and `/health` may name it as
        # `driver_extension_id` while nothing else is attached. What it cannot do
        # is anything that requires a pairing — it cannot serve a single RPC
        # (`_dispatch_serialized` refuses an unpaired link), it cannot be pinned
        # (`POST /driver` answers 409 `not_paired`), it cannot be promoted
        # (`standby_links()` is paired-only, so `_promote_standby` never sees a
        # candidate in it), and it cannot hold the wheel over a paired install
        # (the M1 branch below demotes it on that install's dial, and
        # `_take_free_wheel` takes it as soon as one pairs). It also never
        # appears in `authorized_extension_ids`, which is the file. So the worst
        # it achieves is a transient driver LABEL while nothing else is attached,
        # and the sentence here is the security property a future auditor will
        # trust — which is why it must not overstate it.
        #
        # A pre-attach 4004 for the token-bearing case was here, and it was a
        # dead end rather than a defence (design D1 / UX U3): a revoked install
        # keeps the token it was issued (`worker.ts` does not clear it on the
        # 4003 close), so it re-dialled with it, was closed BEFORE `attach()`,
        # and therefore never reached `_ensure_pending` — no code was ever
        # minted for it, `lop browser pair` answered "already paired … use
        # --reset", and the form its popup showed could not be completed. The
        # only escapes were `--reset` (which revokes the working install too) or
        # Settings → unpair. Since being unlisted already means "cannot be
        # authorised", the refusal bought nothing the admission does not:
        # refusing a token that could never match is not an authorization
        # boundary, it is a locked door in front of an empty room.

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
        # Stamped HERE, with the rest of the identity and before the link is
        # registered, for the A1 ordering argument: a later handshake cannot leave
        # a superseded one's proto behind for a future frame gate to read.
        link.peer_proto = hello.proto
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
        demoted: ExtensionLink | None = None
        if self.link.websocket is None:
            self.driver_generation = generation
        elif link.paired and not self.link.paired:
            # A PAIRED install outranks an incumbent that cannot serve a single
            # command (review round 1, M1). The free-wheel rule above is right for
            # a cold start — but it is also reachable with an UNPAIRED dial, since
            # a token-less peer is admitted so it can pair: whoever completes
            # `hello` first takes the wheel, and if that is a freshly loaded,
            # not-yet-paired install, the authorised+paired install that dials
            # next is told `standby` and every session reads `not_paired` while a
            # perfectly good browser sits idle. `_promote_standby` and
            # `POST /driver` both already refuse to hand the wheel to a link that
            # cannot serve a command; this is the third place that hands it out.
            #
            # The reverse — an unpaired dial taking the wheel FROM a paired
            # incumbent — stays impossible: that case falls to `standby` below,
            # which is the whole point of decision 4.
            demoted = self.link
            self.driver_generation = generation
        link.role = "driver" if self.driver_generation == generation else "standby"
        # The wheel moved (or settled) on THIS socket, so the durable record
        # follows it here too (round 2, R2-1). Both branches above move it: the
        # free-wheel branch is a cold start — after a restart the first dial
        # takes the idle wheel, and the file still named whoever drove BEFORE
        # the restart, so a rollback would authorise that install and refuse the
        # live one — and the paired-demotes-unpaired branch is the M1 rule. The
        # call is a no-op when the name would not change, so an ordinary
        # reconnect that leaves the wheel where it was writes nothing.
        self._record_driver()
        # `last_seen_at` is refreshed here, inside the same no-await block, for the
        # same reason `_record_driver` is: both are synchronous file writes that
        # describe this handshake, and a caller that awaits before them could
        # write a later handshake's truth. Bounded inside the helper (one write
        # per identity per LAST_SEEN_REFRESH_S), so the hot path stays a read on
        # every ordinary reconnect.
        note_identity_seen(self.root, extension_id)
        if demoted is not None:
            # The demoted incumbent holds a live socket and still believes it is
            # driving. The daemon already refuses it commands (it is `self.link`
            # no longer), and the honest role frame is sent below, outside the
            # no-await block, exactly as `drive` and `_promote_standby` do.
            demoted.role = "standby"
        # Bound the UNPAIRED population (review round 1, m1). Collected here —
        # no awaits — because the decision belongs inside the install block; the
        # closes that carry it out happen below, with the superseded peer's.
        surplus: list[tuple[ExtensionLink, WebSocket]] = []
        if not link.paired:
            strangers = sorted(
                (
                    entry
                    for entry in self.links.values()
                    if entry is not link and not entry.paired and entry.websocket is not None
                ),
                key=lambda entry: entry.attached_at,
            )
            while len(strangers) >= MAX_UNLISTED_LINKS:
                victim = strangers.pop(0)
                wire = victim.websocket
                if wire is None:  # pragma: no cover - the filter above excludes it
                    continue
                surplus.append((victim, wire))
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
        for victim, wire in surplus:
            # 4004, the same answer the pre-change daemon gave a second identity:
            # this peer holds no pairing and the bridge already has as many
            # unpaired dials as it will track. Retiring first drops its link
            # entry and any parked futures, so the map is bounded either way the
            # close lands.
            self._retire_link(victim)
            with suppress(Exception):
                await asyncio.wait_for(wire.close(code=4004), timeout=LINK_CLOSE_TIMEOUT_S)
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
                        # The NEGOTIATED value, not this daemon's own ceiling: a peer
                        # inside the window is told what the pair will actually
                        # speak, which is the hook a future extension needs before it
                        # gates anything on the daemon's proto. Nothing reads this
                        # today, so it is free.
                        proto=min(hello.proto, PROTO_VERSION),
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
        if demoted is not None:
            # AFTER the ack: a handshake that turned out to be superseded returns
            # above without speaking, and this frame is a statement about the
            # wheel that only the current handshake may make.
            await self._tell_role(demoted)
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
                # The same frame proves there is nothing left to corroborate, so
                # the two-strike probe's counter resets here rather than only on
                # a pong: a peer that is talking is listening, whatever it is
                # doing about the command that timed out (see
                # `PROBE_STRIKES_BEFORE_SEVER`).
                link.clear_probe_strikes()
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
                if frame.get("event") == "capabilities":
                    # Which methods this build serves (design §6.3). Validated
                    # rather than trusted: a malformed frame is dropped, not
                    # allowed to blank a working advertisement — the same
                    # discipline the pairing and response branches use.
                    #
                    # ADDITIVE by construction: an already-released daemon
                    # reaches the Response.model_validate below, fails, and
                    # drops the frame (protocol.py's rule for what keeps
                    # PROTO_VERSION where it is).
                    try:
                        advertised = Capabilities.model_validate(frame)
                    except ValidationError:
                        continue
                    link.capabilities = [str(name) for name in advertised.methods]
                    # Published straight away rather than waiting for the next
                    # heartbeat: the session-side check reads the FILE, so a
                    # capability that only reached it 30 s later would show the
                    # new action as unavailable on a host that serves it.
                    self.publish_safely()
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
        except WebSocketDisconnect as exc:
            # A SERVER-initiated close is not a driver loss, and the finally
            # below must not read it as one. uvicorn's websocket protocols send
            # 1012 ("service restart") on every live connection while the server
            # is going down, and they send it BEFORE the lifespan shutdown event
            # that sets `_shutting_down` via `shutdown()` — see `_daemon_leaving`
            # for the measurement and for why detecting it here is what makes the
            # no-handover-during-teardown rule hold for embedders too.
            #
            # 1012 is the single discriminator; see
            # `SERVER_GOING_DOWN_CLOSE_CODE` for why 1001 is deliberately NOT
            # accepted here (a peer's "going away" frame would latch this flag
            # permanently, freezing the record and disabling failover).
            #
            # Scoped to a link that holds pairing authority or the WHEEL (review
            # round 5, finding 1). Those are precisely the links whose ending can
            # move the durable record, or hand the wheel to a standby on the way
            # out — the harm the guard exists for. A stranger that dials with a
            # forged origin and sends 1012 holds neither, so its frame is inert
            # instead of latching the daemon's permanent flag. See the constant for
            # the residual this does NOT close.
            #
            # ``link.paired`` covers the PAIRED NON-DRIVER, and it is the half a
            # reader cannot size from the code alone (review round 6, MINOR 1), so
            # both the harm and its present status:
            #
            # * The harm it covers: a paired link is one whose identity is in the
            #   durable record and which the daemon may promote — a retiring link
            #   can hand the wheel to a standby and move the record on its way out,
            #   which is the same class of state change this flag must not
            #   mis-attribute to the server leaving.
            # * Its status TODAY: a non-driver retirement can neither promote nor
            #   persist (the promotion and record paths both read the link's own
            #   generation against the wheel, and a standby's close leaves the
            #   driver untouched), so this half is currently defence in depth rather
            #   than load-bearing — deleting it changes no reachable behaviour. That
            #   is exactly why it is pinned by a row
            #   (``test_r5_7_a_paired_standbys_1012_latches``) instead of being
            #   assumed to be required: a future edit that makes a retiring link
            #   able to move the record must not find this predicate already
            #   narrowed away, silently.
            if exc.code == SERVER_GOING_DOWN_CLOSE_CODE and (
                link.paired or link.generation == self.driver_generation
            ):
                self.begin_shutdown()
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
            # An identity that is AUTHORISED but has no live link is a different
            # answer from an unknown target, and conflating them is what made
            # `drive <id>` contradict `status` (UX round 3, U4): the listing said
            # "(cmadnonj…) paired, not connected" while the command said nothing
            # matched — reading as a typo'd id, in the state a handover leaves
            # behind, which is exactly when the operator reaches for this
            # command. Resolution against live links first is unchanged; this
            # only names the reason when the target IS known here.
            authorised = _resolve_authorised_target(target, _identities(self.root))
            if authorised is not None:
                return JSONResponse(
                    {
                        "error": "not_connected",
                        "message": (
                            "that install is authorised but not connected right now. "
                            "Open its browser, then retry."
                        ),
                        "extension_id": str(authorised.get("extension_id", "")),
                    },
                    status_code=409,
                )
            return JSONResponse(
                {
                    "error": "unknown_extension",
                    # How many attached installs the target matched, so the caller
                    # can word "nothing matched" and "several matched"
                    # differently (copy review C8) instead of listing every
                    # authorised install under the word "matches".
                    "matches": len(self._matching_extensions(target)),
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
        # The pin is a wheel move like any other, so the downgrade record
        # follows it (M2/Q2) — before the awaits below, per the audit-A1 rule.
        self._record_driver()
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

    def _matching_extensions(self, target: str) -> list[ExtensionLink]:
        """Every attached link the target could mean, most specific rule first.

        Split out of `_resolve_extension` so a refusal can say HOW MANY matched:
        the CLI words "no connected extension matches" and "no single connected
        extension matches" differently, and the same 404 shape used to carry both
        readings, which is how an unknown id came to be listed under the word
        "matches" with two unrelated installs beneath it (copy review C8).

        Exact id wins outright; otherwise an id-prefix search; otherwise a
        case-insensitive label substring. Every candidate is an ATTACHED link:
        resolving against the file would let a target name an install that cannot
        answer.
        """
        wanted = normalise_target(target).lower()
        if not wanted:
            return []
        candidates = [entry for entry in self.links.values() if entry.websocket is not None]
        exact = [entry for entry in candidates if entry.extension_id.lower() == wanted]
        if exact:
            return exact
        prefixed = [entry for entry in candidates if entry.extension_id.lower().startswith(wanted)]
        if prefixed:
            return prefixed
        labels = _identities(self.root)
        matched: list[ExtensionLink] = []
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
                matched.append(entry)
        return matched

    def _resolve_extension(self, target: str) -> ExtensionLink | None:
        """Resolve ``<id-or-label>`` to an attached link, or None.

        Accepts an exact id, an unambiguous id PREFIX (the operator copies 32
        opaque characters, and now also the printed `ohcmfhja…` form — see
        `normalise_target`), or a case-insensitive label SUBSTRING. An ambiguous
        target resolves to nothing rather than to a guess: silently moving the
        wheel to the wrong browser is worse than asking for one more character.
        """
        matches = self._matching_extensions(target)
        return matches[0] if len(matches) == 1 else None

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
        # REFUSE TO SEND what the attached host did not advertise, and do it here
        # rather than letting the peer fail it: the worker answers an unknown
        # method with a bare `internal`, so an ungated `download` sent to a
        # pre-feature extension would spend the whole command budget and then
        # report nothing the caller can act on (design §6.3). The refusal names
        # the method, the host and the host's own reported version — the two
        # remedies it separates ("this build predates the feature" versus "this
        # build is current and stopped answering") are otherwise
        # indistinguishable, and sending the reader to the wrong one is the
        # misdiagnosis `OWNERSHIP_MIN_EXTENSION_VERSION` exists to prevent.
        #
        # Only methods that ARE capability-gated are checked: every other method
        # predates the advertisement, and refusing them on a pre-feature peer
        # would break the whole tool for a host that works today.
        if (
            request.method in CAPABILITY_GATED_METHODS
            and request.method not in self.link.capabilities
        ):
            return self._error_response(
                request.id,
                ErrorCode.CAPABILITY_UNSUPPORTED,
                f"the attached host does not serve {request.method}",
                {
                    "method": request.method,
                    "advertised": sorted(self.link.capabilities),
                    "extension_version": self.link.extension_version,
                },
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
            response = await self._await_response(request.id, future, self._command_budget(request))
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
            # `extension_unresponsive`). Probing instead of inferring is the fix
            # for that, and it is deliberately NOT a lower threshold: a threshold
            # below the healthy sawtooth is what review R1-2 reproduced, because
            # the daemon is the only party that solicits speech. Asking the peer
            # directly gives a definitive answer inside whatever budget the
            # method has, so the rule fires for the tight methods without
            # re-opening the false positive.
            #
            # The PROBE arm is two-strike over DISTINCT observation windows and
            # refuses to sever at all while the peer has spoken recently — see
            # `_probe_verdict` for both guards and the incident that produced
            # them. The short version: one unanswered
            # question used to sever the link and fail EVERY pending future on
            # that worker, so a single slow `read` answered three sessions with
            # "the extension stopped answering" and destroyed their in-flight
            # work. What is NOT relaxed: the 1.5× clock arm above still severs on
            # its own evidence, and a link whose silence the daemon can
            # corroborate still goes `close 4000` → refuse-fast → alarm re-dial.
            #
            # Either arm produces the reason the teardown is announced with; the
            # one that did NOT fire contributes nothing, so `reason` being None
            # means this command answers ALONE and every sibling future is left on
            # the link, intact, to finish or time out on its own budget. The
            # replaced-wire test below is deliberately outside the arm: a reply
            # that never arrived on a connection the extension has since replaced
            # is a fact about THIS command, and stays sayable whether or not the
            # arm decided to sever.
            reason = (
                f"{request.method} unanswered with the link silent for {silent:.0f}s"
                if silent > PING_INTERVAL_S * 1.5
                else await self._probe_verdict(request.method)
            )
            if reason is not None:
                dropped = await self._drop_unproven_link(reason, expected=wire)
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
            if self._wire_loss(wire) == "replaced":
                # A reply that never arrived on a connection the extension has
                # since replaced must still NAME the replacement, whether or not
                # the probe arm decided to sever anything (it no longer does on a
                # single miss, or while the replacement is still talking).
                # "read timed out" would hide a fact the reader can act on — the
                # browser is open and reconnected — behind a symptom.
                return self._error_response(
                    request.id,
                    ErrorCode.EXTENSION_DISCONNECTED,
                    f"{request.method} was delivered on a connection the extension "
                    "has since replaced",
                    {"phase": "replaced"},
                )
            code = (
                ErrorCode.NAV_TIMEOUT if request.method in ("open", "goto") else ErrorCode.INTERNAL
            )
            return self._error_response(
                request.id,
                code,
                f"{request.method} timed out",
                {"timeout_s": self._command_budget(request)},
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

    def _command_budget(self, request: Request) -> float:
        """The wall-clock budget for one admitted command (design 6.1).

        Only `download` may ask for more than its table value: it waits on a PAGE
        (a 200 MB file on a slow link legitimately takes minutes), and the caller's
        `timeout_s` is clamped to the shared ceiling by the TOOL before it reaches
        the wire. Every other method takes its table value, so no method can
        extend its own budget by inventing a parameter. The session-side client
        reads the same wire key (`client_timeout`), which is what keeps the two
        ends of the timeout chain from disagreeing.
        """
        base = COMMAND_TIMEOUTS[request.method]
        if request.method != "download":
            return base
        raw = request.params.get("timeout_s")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
            return base
        return max(base, min(float(raw), browser_files.DOWNLOAD_TIMEOUT_MAX_S))

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
                # ADDITIVE, and a STRING only: the quarantine root a `download`
                # writes into (design §4.4). The daemon does not stat it — this
                # payload is polled by the popup, and a directory walk on a polled
                # path is I/O for a number nobody reads there. The CLI, which runs
                # on this machine already, prints the size.
                "downloads_dir": str(browser_files.downloads_root()),
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
                #
                # The MIRROR half is reported only while NO link is serving
                # (review round 4 / UX U1: `and not connected`). The latch is a fact
                # about the link that was severed, and the TTL window exists for the
                # case where nothing has taken over — which is the case the paragraph
                # above is about, and where `self.link` is the idle link and
                # `connected` is false. Once a different, PROVEN link holds the wheel,
                # reporting it made this payload contradict itself in adjacent fields
                # (measured: `extension_connected: true` with `link_silent_s: 0.0016`
                # next to `extension_unresponsive: true`), and the popup painted its
                # red wedge card on the install that was answering commands in 7 ms —
                # with a Reload button that would have reloaded the serving install.
                # Scoping the REPORT, not clearing the state, leaves the latch
                # available for the window it was written for and touches none of the
                # drop/promotion machinery #996 fenced.
                #
                # The CURRENT link's own memory of a drop is a separate clause, and
                # deliberately NOT gated: `dropped_unproven()` records "this link was
                # measured silent for longer than the deadline and severed", which
                # outlives the socket it describes by design — including in the window
                # where the same peer has re-dialled and looks proven again
                # (`test_a_peer_that_closes_its_own_socket_is_absent_not_unresponsive`
                # pins exactly that reading).
                "extension_unresponsive": (self.link.websocket is not None and not connected)
                or self.link.dropped_unproven()
                or (self.drop_latched() and not connected),
                # Additive OPTIONAL fields (HTTP, so an old client ignores them)
                # reporting the EXTENSION's identity as this daemon last saw it
                # in `hello`. `proto` above is deliberately NOT reused: that is
                # the daemon's OWN version, and conflating the two is what made
                # a skew undiagnosable.
                #
                # Empty/0 while the link is not proven, so a stale stamp from a
                # dead socket cannot be read as a live one.
                "extension_version": self.link.extension_version if connected else "",
                "extension_proto": self.link.peer_proto if connected else 0,
                # The version this runtime ships with, so a reader never has to
                # know the constant to render the advisory.
                "extension_expected_version": EXPECTED_EXTENSION_VERSION,
                # The ONE update predicate: proven AND a KNOWN version strictly
                # below what this runtime expects. Unparseable is not "older",
                # and an extension ahead is not behind, so neither produces a
                # nag. The advisory this drives never blocks anything — it is a
                # hint the store cannot be asked about, not a requirement.
                "extension_update_available": connected
                and extension_older(self.link.extension_version, EXPECTED_EXTENSION_VERSION),
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
                # How long until the daemon acts on a mute wheel-holder, or null
                # when nothing is pending (UX round 3, U9). Present ONLY while the
                # wheel-holding link is attached and not answering — the state a
                # standby install sees for ~a minute before the daemon severs it
                # and promotes somebody. Without a number here the standby card can
                # only either promise an immediate takeover (false for this event:
                # measured 61.2 s on a real rig, 60.1-60.2 s on the isolated one)
                # or say nothing about the event it is actually in.
                #
                # Derived, not authoritative: `LINK_SILENCE_TIMEOUT_S` is the
                # deadline `proven` measures and `PING_INTERVAL_S` is the tick that
                # notices it, so this is the upper bound for an ordinary event loop
                # and a delayed tick pushes it later. It answers "which of the two
                # cases is the user in" — a value means silence (takeover pending),
                # null means either a healthy driver or a wheel nobody holds.
                "takeover_within_s": (
                    round(
                        max(
                            0.0,
                            LINK_SILENCE_TIMEOUT_S + PING_INTERVAL_S - self.link.silent_for(),
                        ),
                        1,
                    )
                    if self.link.websocket is not None and not connected
                    else None
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
                # The standby LABELS, in the same order as the ids above, so a
                # reader can act on a property of the install rather than asking
                # the operator to evaluate it (copy review C4 / UX U7: the
                # pre-0.1.13 note used to print for ANY standby, including an
                # all-current pair, because the only version data on hand was the
                # driver's). Live link values, not the file's recorded label, so
                # an install that updated since pairing reports the build it is
                # actually running.
                "standby_labels": [
                    _browser_label(link.browser, link.extension_version)
                    for link in self.standby_links()
                ],
                # Admitted dials with NO pairing. Not standbys — they hold no
                # authority and can never be promoted — but the operator (and the
                # bound in MAX_UNLISTED_LINKS) needs them countable rather than
                # invisible in the one place a reader already looks.
                "unlisted_extension_count": len(self.unlisted_links()),
                "authorized_extension_ids": sorted(_identity_ids(self.root)),
                # The driver's human label, so the standby popup can NAME the
                # install holding the wheel rather than saying "another one".
                # Empty when the file has no label for it (an entry paired
                # before labels existed), which the card words accordingly.
                "driver_label": driver_label,
                # The driver's SHORT id. The label alone stops answering "so
                # which one is it?" when both installs run the same build — the
                # collision is routine for two profiles of one unpacked build, or
                # any two builds at one version — and the id prefix is the one
                # token that always resolves in `lop browser drive` (UX U2 /
                # design D3). Sent whole rather than truncated so the reader
                # decides how much of it to show.
                "driver_short_id": driver_id[:8] if driver_id else "",
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
    app = create_app(args.port)
    # Built explicitly rather than via `uvicorn.run` so the service can see the
    # server going down. uvicorn sets `Server.should_exit` when a stop signal
    # arrives and only afterwards closes listeners and connection sockets; the
    # lifespan shutdown event fires LAST. Without that reference the service
    # cannot tell "a driver's socket ended because we are stopping" from "a
    # driver's socket ended and a standby should take over", and the wrong answer
    # writes the standby into the durable downgrade record (round 2, R2-1).
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info"))
    app.state.bridge.watch_server_exit(server)
    server.run()
    return 0


if __name__ == "__main__":
    # Linux comm axis: this daemon's launchd/systemd unit names the IMAGE on
    # macOS, and where there is no such image the process names itself (see
    # :func:`procname.brand_this_process`; a no-op on macOS).
    from local_operator import procname

    procname.brand_this_process()
    raise SystemExit(main())

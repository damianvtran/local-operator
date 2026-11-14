"""One answer to "is remote access working here", shared by every surface.

`lop tunnel status`, the desktop route and (through the park file) the TUI all
describe the same machine, and the incident this module exists for was a status
command that answered the question from the CLOUD's cached view while the local
connector was dead. So the assembly lives here, once: these are the shapes a
caller renders or forwards, and a second implementation is how two surfaces come
to disagree about what "parked" means — with the one nobody was looking at
being the wrong one.

Two rules the code below keeps, both of them lessons from that incident:

* The connector's own state is read on THIS device. A park file outranks a
  health probe, because a parked connector is not running and the probe can
  only ever answer "there is no gateway here".
* A check that cannot run reports that it could not run. Neither a lost network
  nor an unreadable store is allowed to read as "your login is dead", which is
  what sent an offline machine to a login it did not need.
* A state the store is WAITING OUT is reported as that wait. A refresh the store
  deferred (its token's last exchange is unsettled) is neither a dead login nor
  an unanswerable check, so `login_verdict` gives it its own state — `deferred` —
  read from the row and never memoised, and every surface prints it without a
  command: it clears by itself, and a sign-in offered for it is advice to fix
  something that is not broken.
* A configuration that names a login this device no longer has is reported as
  that — `owner_missing` — and never as `login_required`: a fresh sign-in makes
  a NEW row and nothing here re-points the configuration, so the remedy a
  refused grant earns would loop the operator forever. Like `deferred` it is a
  fact read from the store ahead of the memo and never memoised (a row can be
  deleted at any moment, and the memo's key cannot watch a row that is gone),
  and its remedy is the command that re-points the device.
* A check that costs a network call is BOUNDED, and a negative answer is reused
  for the window in which re-asking cannot learn anything new. The login verdict
  is the one check here that reaches the network at all (see REFRESH_WAIT_S and
  VERDICT_TTL_S): it is read by routes the desktop polls on open, so an unbounded
  call there is a stalled poll rather than a slow answer. Bounded, NOT CANCELLED:
  the exchange it waits for is handed to the store's supervisor, so giving up on
  the ANSWER cannot strand the write-ahead marker only that answer can resolve.
"""

from __future__ import annotations

import asyncio
import datetime
import time
from contextlib import closing
from typing import Any

import httpx

from local_operator.i18n.keys import tunnel_status
from local_operator.i18n.text import tr
from local_operator.providers.auth_store import (
    DEFAULT_BLOCK_MS,
    AuthStore,
    AuthStoreError,
    CredentialInvalidError,
    RefreshUnconfirmedError,
)
from local_operator.tunnels import config, gateway, state

_t = tr("tunnel.status")

#: How long the login verdict may wait for a token endpoint that is not answering.
#:
#: `AuthStore`'s refresh POST carries a 30-second client timeout, which is right
#: for a model request — a slow identity provider must not fail a turn — and wrong
#: here, where the caller is a route the desktop polls on open. Measured on this
#: branch with the token endpoint blackholed: **30.9 s** for one
#: `GET /v1/desktop/tunnel` (review round 1, M1), the whole poll spent on a call
#: whose own answer is "I could not check". Past the bound the verdict is
#: `unknown`, which is already the honest one for a check that did not finish.
REFRESH_WAIT_S = 2.0

#: How long a NON-`ok` verdict is reused instead of being recomputed.
#:
#: The same shape merged #1340 uses for the desktop's side of this question
#: (`DIAGNOSIS_TTL_S` / `_diagnosis_key` in `server/routes/desktop_radient.py`),
#: for the same class of cost: a verdict that cost a token-endpoint POST is not
#: re-asked on every poll while it still holds — there, measured as seven failed
#: refreshes for six refusals. `DEFAULT_BLOCK_MS` is the window the store's own
#: cascade keeps a credential with a failed refresh out of rotation for, so it is
#: exactly the window in which re-asking can learn nothing new.
#:
#: `ok` is never reused: it is the answer that is usually FREE (a token inside its
#: refresh skew returns without a POST) and the one whose staleness would matter
#: most. A row that CHANGED — a re-login, or a refresh a peer landed — is decided
#: again whatever the clock says, because the row's `updated_at` is in the key.
VERDICT_TTL_S = DEFAULT_BLOCK_MS / 1000

#: The last non-`ok` login verdict, keyed by the row it was made about.
#:
#: Module state, and deliberately: it memoises THIS device's store reading, is
#: keyed by that store's own database and the row's own write stamp (see
#: :func:`_verdict_key`), and expires on its own within :data:`VERDICT_TTL_S`, so
#: it caches no fact this process is not allowed to know. Purging is opportunistic
#: on write.
#:
#: It holds verdicts the CHECK produced, and never the deferral
#: (`login_verdict`'s `deferred`): that one is read from the row ahead of the memo,
#: so it cannot be masked by an entry composed before it became true, and it needs
#: no remembering because it costs nothing to re-decide. The memo's whole job is
#: the verdicts that spent a token-endpoint POST.
_VERDICTS: dict[tuple[str, int, int], tuple[dict[str, Any], float]] = {}


def _verdict_key(store: AuthStore, credential_id: int, updated_at: int) -> tuple[str, int, int]:
    """What a remembered verdict is ABOUT: one store's row, as it stands now.

    The store's database path is in the key because this memo outlives one call
    while two daemons can share this process, and credential ids restart at 1 in
    every config dir. ``updated_at`` is what the store writes on every change to a
    row, so a re-login — or a refresh another process landed — composes a different
    key and the verdict is decided again; a FAILED refresh writes nothing, which is
    what lets a verdict hold across the very failures it describes.

    One thing ``updated_at`` deliberately does NOT move for: a send-marker write
    (``_update_payload(..., moves_write_stamp=False)``), because the marker is the
    store's own bookkeeping about a request rather than a change to the credential.
    So the state that marker describes — the deferral — is read from the row BEFORE
    this key is consulted instead of being memoised (``login_verdict``): a memo
    composed while the marker was absent would otherwise answer for the whole
    :data:`VERDICT_TTL_S` after it appeared.
    """
    return (str(store.db_path), credential_id, updated_at)


def _remembered_verdict(key: tuple[str, int, int]) -> dict[str, Any] | None:
    """The verdict still in force for this row, or ``None`` to decide again."""
    entry = _VERDICTS.get(key)
    if entry is None:
        return None
    verdict, expires = entry
    if time.monotonic() >= expires:
        _VERDICTS.pop(key, None)
        return None
    return dict(verdict)


def _remember_verdict(key: tuple[str, int, int], verdict: dict[str, Any]) -> dict[str, Any]:
    """Reuse ``verdict`` for this row for the store's own block window."""
    now = time.monotonic()
    # Purge first: an entry is dead within VERDICT_TTL_S and there is never a
    # reason to keep one, so a row that keeps changing cannot grow this dict
    # without bound.
    for stale, (_verdict, expires) in list(_VERDICTS.items()):
        if expires <= now:
            _VERDICTS.pop(stale, None)
    _VERDICTS[key] = (dict(verdict), now + VERDICT_TTL_S)
    return verdict


async def probe(value: dict[str, Any]) -> dict[str, Any]:
    """Ask the local gateway what it is doing, for a connector with no park."""
    healthy = False
    connected = False
    served = False
    reason = ""
    detail = ""
    transport = ""
    cloudflared = False
    listening = False
    status_code: int | None = None
    last: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            reply = await client.get(
                f"http://127.0.0.1:{value['gateway_port']}/_lop_tunnel/health", timeout=2
            )
            status_code = reply.status_code
            served = status_code == 200
            payload = reply.json() if served else {}
            if not isinstance(payload, dict):
                # A stale or foreign listener on this port can answer 200 with
                # any JSON at all. Anything but an object is not a health
                # payload, and a status surface must not raise over it.
                payload = {}
            healthy = served and payload.get("ok") is True
            connected = healthy and payload.get("connected") is True
            # Older gateways (and foreign listeners) send no transport; those
            # only ever ran cloudflared, and "" keeps the status line silent
            # rather than guessing.
            if payload.get("transport") in ("relay", "cloudflare"):
                transport = str(payload["transport"])
                cloudflared = payload.get("cloudflared") is True
            if not healthy:
                # The gateway names why it is refusing relayed requests, and
                # this is the surface where a command can be offered at all.
                # A reason this build does not know falls back to the relay's
                # own sentence rather than saying nothing.
                reason = str(payload.get("reason") or "")
                detail = gateway.terminal_detail(reason, str(payload.get("detail") or ""))
    except httpx.ConnectError:
        # REFUSED: nothing is listening on the port. Only here can a missing or
        # dead unit be the explanation — but a connector in its retry loop also
        # leaves the port closed between attempts (it exits and the supervisor
        # restarts it 10 s later), so a CURRENT retrying exit wins over the
        # install advice (field report: a Radient 503 read here as "run lop
        # tunnel install" while the real cause sat in service.log). An old exit
        # does not: the log outlives the outage, so a failure from last week
        # beside a closed port means the unit is gone, and the install advice
        # is the right answer (review round 1, R1-1). It is still returned as
        # `last_error`, which the CLI prints as history.
        last = last_connector_failure()
        if last is not None and last["current"]:
            reason = last["reason"]
            detail = gateway.terminal_detail(reason, "")
        else:
            detail = (
                "The local relay gateway is not answering on "
                f"127.0.0.1:{value['gateway_port']}; run lop tunnel install to restore it"
            )
    except (httpx.HTTPError, ValueError) as failure:
        # The port ACCEPTED the connection, so the unit is there: install advice
        # would send the operator to repair something that is running. Say what
        # was actually observed (timed out, or an answer that is not a health
        # payload) — the exception's own text is not used, because httpx echoes
        # the request URL.
        listening = True
        port = str(value["gateway_port"])
        if isinstance(failure, httpx.TimeoutException):
            detail = _t(tunnel_status.gateway_listening_timed_out(port=port))
        else:
            detail = _t(tunnel_status.gateway_listening_unreadable(port=port))
    if not served and status_code is not None:
        # The port answered HTTP, but not the 200 every lop gateway gives its
        # health route: a foreign listener holds the port, or the gateway is
        # failing. Name the status rather than printing a bare "stopped"
        # (QA round 1, Q2); install advice would not free the port either.
        listening = True
        detail = _t(
            tunnel_status.gateway_listening_http_status(
                port=str(value["gateway_port"]), status=str(status_code)
            )
        )
    if listening:
        last = last_connector_failure()
        if last is not None and last["current"]:
            # Only a CURRENT exit explains a listener that is not serving; an
            # old one is history and is printed as such by the CLI (R1-1).
            reason = last["reason"]
            detail += " " + gateway.terminal_detail(reason, "")
    if connected:
        word = "connected"
    elif healthy:
        word = "connecting"
    elif served or listening:
        # The gateway answered and is refusing to serve. That is not a stopped
        # connector — cloudflared may still hold the edge connection — and
        # "stopped" beside a sentence promising it clears itself would
        # contradict the payload this just read.
        word = "not serving"
    else:
        word = "stopped"
    result: dict[str, Any] = {
        "state": word,
        "reason": reason,
        "detail": detail,
        "since": None,
        "remedy": None,
    }
    if transport:
        result["transport"] = transport
        result["cloudflared"] = cloudflared
    if last is not None:
        # Additive: the code and when it was written, so `--json` readers and
        # the CLI can show the connector's own last exit beside the state.
        result["last_error"] = last
    return result


#: How much of `service.log`'s TAIL is read for the connector's last exit line.
#: Bounded because the launchd log is only truncated by `lop tunnel install`, so
#: on a long-lived machine it can be large, and `status` must stay a cheap read.
#: A retry line is ~200 bytes and the connector writes at most one per restart
#: (the supervisor's 10 s floor), so 64 KiB holds the last ~300 lines — far more
#: than the one line this needs, and still ~0.1 ms to read.
LOG_TAIL_BYTES = 64 * 1024

#: The ONE line shape this reads: `service.main`'s retrying exit, which is this
#: package's own literal (`connector stopped reason=<code>…`), never an upstream
#: body. Parks are not read here — they live in `state.json`.
_SERVICE_PREFIX = " local_operator.tunnels.service: "
_RETRY_MARKER = _SERVICE_PREFIX + "connector stopped reason="

#: How old a retrying exit may be and still be the CURRENT cause.
#:
#: Derived from the retry cadence, not chosen for taste: a retrying connector
#: exits 1 and its supervisor restarts it no sooner than 10 s later
#: (`ThrottleInterval: 10` in `tunnels/install.py`'s launchd plan and
#: `restart_sec=10` in its systemd unit), and each failed start writes a fresh line. So while a
#: fault persists the newest line is at most ~10 s plus one start attempt old
#: (the control-plane calls are themselves bounded well under 30 s). 60 s is
#: several missed cycles of slack; past it nothing is restarting the connector,
#: and the line is history (review round 1, R1-1: a two-month-old line beside a
#: closed port had replaced the install advice).
CURRENT_FAILURE_S = 60


def last_connector_failure(*, now: float | None = None) -> dict[str, Any] | None:
    """The connector's most recent retrying exit from `service.log`, if any.

    `state.json` deliberately records only PARKS (see `tunnels/state.py`), so a
    connector cycling on a transient fault — the field report's Radient 503 —
    leaves its cause only in the log its supervisor captures. This reads that
    cause back as a reason CODE (this package's vocabulary) plus the line's own
    timestamp; the free text after the code is not returned, so nothing beyond
    the fixed vocabulary reaches a status surface.

    ``current`` says whether the exit can explain what is happening NOW: it must
    be the LAST line the service wrote (a later line — a relay start, a park
    withdrawn — means the connector moved on) and no older than
    :data:`CURRENT_FAILURE_S`. `service.log` is truncated only by `lop tunnel
    install`, so without both checks one transient blip would explain every
    later outage forever.
    """
    log = config.directory() / "service.log"
    try:
        with log.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            tail = handle.read(LOG_TAIL_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    superseded = False
    for line in reversed(tail.splitlines()):
        stamp, marker, rest = line.partition(_RETRY_MARKER)
        if not marker:
            if _SERVICE_PREFIX in line:
                superseded = True
            continue
        code = rest.split(":", 1)[0].split(";", 1)[0].strip()
        if not code or not all(c.isalnum() or c == "_" for c in code):
            return None
        at: int | None
        try:
            at = int(datetime.datetime.fromisoformat(stamp.strip()).timestamp())
        except ValueError:
            at = None
        age = (time.time() if now is None else now) - at if at is not None else None
        # Both directions bounded: a stamp from the future (a clock moved back)
        # is no evidence of a retry happening now either.
        current = not superseded and age is not None and abs(age) <= CURRENT_FAILURE_S
        return {"reason": code, "at": at, "current": current}
    return None


async def connector_state(value: dict[str, Any], *, reachable: bool = True) -> dict[str, Any]:
    """The connector's real state, and why it is in it.

    The park file outranks the health probe, and it has to: a parked connector
    is not running, so the probe can only ever answer "there is no gateway
    here" — reporting that as the state is the lie this surface is being fixed
    for. A deliberate local stop outranks both: a tunnel the operator stopped is
    not waiting on anyone.

    ``reachable=False`` skips the loopback probe for callers that must not
    block on a socket they do not need — `GET /v1/auth/status` is polled beside
    an interactive login form, where a 2-second connect timeout would be felt.
    Those callers get the parked/stopped facts, which is the whole of what they
    are asking about, and ``unknown`` otherwise.
    """
    if value.get("stopped"):
        return {"state": "stopped", "reason": "", "detail": "", "since": None, "remedy": None}
    parked = state.parked()
    if parked is not None:
        return {
            "state": "parked",
            "reason": str(parked.get("reason") or ""),
            "detail": str(parked.get("detail") or ""),
            "since": parked.get("first_at"),
            "remedy": parked.get("remedy") if isinstance(parked.get("remedy"), dict) else None,
        }
    if not reachable:
        return {"state": "unknown", "reason": "", "detail": "", "since": None, "remedy": None}
    return await probe(value)


async def login_verdict(value: dict[str, Any]) -> dict[str, Any]:
    """The pinned login's verdict, plus a provable same-account successor when it is dead.

    See :func:`_pinned_login_verdict` for the verdict itself.

    WHEN THE SAME-ACCOUNT HEAL COULD NOT RUN (the tunnel was stopped at the
    time, a build before it wrote the login), a dead or missing pin with a
    provable same-account successor carries ``successor_credential_id``. The
    STATE is unchanged — it is still the truth about the pinned row — but
    ``remedy()`` then names the re-point instead of a sign-in that has already
    happened. The successor is
    added here, outside it, because the verdict may come from the memo (whose
    dict must never be mutated) and because whether a successor exists moves
    with writes the memo's key cannot see — a sign-in writes a DIFFERENT row.
    """
    verdict = await _pinned_login_verdict(value)
    if verdict.get("state") not in (gateway.LOGIN_REQUIRED, gateway.OWNER_MISSING):
        return verdict
    from local_operator.tunnels import api

    record = value.get("record")
    try:
        successor = api.same_account_successor(
            value.get("credential_id"),
            owner_account=record.get("owner_account_id") if isinstance(record, dict) else None,
        )
    except (AuthStoreError, OSError, ValueError):
        # The verdict stands on its own; a store that cannot answer this second
        # question must not turn a correct `login_required` into an error.
        return verdict
    if successor is None:
        return verdict
    return {**verdict, "successor_credential_id": successor}


async def _pinned_login_verdict(value: dict[str, Any]) -> dict[str, Any]:
    """Whether the login that owns this tunnel still works, decided locally.

    Decided from this device's own credential store rather than from the cloud
    read, because when the login is dead the cloud read is precisely what cannot
    answer. A refresh that could not REACH the token endpoint reports `unknown`
    rather than `login_required`: sending an offline machine to a login it does
    not need is the misdirection this surface exists to remove, and
    `CredentialInvalidError` is the one signal that separates the two.

    BOUNDED, and memoised, because the check is a network call after all
    (:data:`REFRESH_WAIT_S`, :data:`VERDICT_TTL_S`). It fires exactly when the
    stored access token is outside its refresh skew — which is the condition this
    surface exists to describe, not an edge case — and it reached the Radient
    refresh POST with its 30-second client timeout on the server's event loop, so
    a partitioned network (the incident's OWN condition) made every poll of
    `GET /v1/desktop/tunnel` and `GET /v1/auth/status` wait it out. The two bounds
    turn that into one bounded wait per window, and `unknown` is what the wait
    resolves to when it expires: the check did not finish, which is a fact about
    this machine and not a verdict about the login.

    BOUNDED, NOT CANCELLED (the defect this shape exists for). `wait_for` around
    the exchange itself cancels it at the bound, and a cancelled exchange reaches
    none of the store's handlers: its write-ahead send marker stays armed with
    nothing left to resolve it, which on a due bearer is a window of refusal the
    phone renders as an expired login. So the exchange is handed to the store's
    supervisor (``AuthStore.detached_refresh``, which owns it and the store it
    runs on) and only the WAITING is bounded — with ``asyncio.wait``, not with a
    shielded ``wait_for`` (see the comment at the call for the measured reason).

    WHAT THAT GUARANTEE HINGES ON, stated exactly (agent review round 1, R1): the
    exchange survives THIS caller walking away **for as long as the loop it runs on
    lives**. A long-lived caller — the desktop runtime, the connector, any process
    that keeps serving — is where it pays in full: the exchange lands its rotation,
    clears the marker, and the bound only cost the caller its own answer. A
    SHORT-LIVED caller (``lop tunnel status`` under ``asyncio.run``, which tears the
    loop down when the command returns) cancels the still-pending exchange at
    process exit, so its request dies mid-wire and the marker stays armed as
    ``unknown`` — an accepted, bounded consequence rather than a bug: nothing can
    complete a POST whose process is gone, and design §6.1 rules out draining the
    task here, since draining would put the measured 30.9 s poll latency back. The
    cost is the derived :data:`UNCONFIRMED_SEND_TTL_S` window instead of the hour
    it used to be, and it is bounded on the other side too: an invocation after the
    marker lapses can arm a FRESH window, so a repeatedly-invoked CLI against a
    stalling endpoint can renew the deferral — which is why the long-lived callers
    are the recovery mechanism, and why the marker is never held beyond its bound.
    ``tests/unit/test_tunnels.py`` pins this path's outcome (marker armed and
    complete, ≤ bound, lease released, row consistent, no closed-database error).

    THE LEASE STAYS WITH THE EXCHANGE, and the reason the caller cannot help with
    it is worth stating: the lease is taken on the supervisor's store and
    ``_release_refresh_lease`` is holder-scoped, so a release from this store would
    be a no-op rather than a hazard — which is exactly why the call that used to be
    here is deleted instead of left as reassurance. The hazard is real if anything
    ever frees that lease while the exchange is alive: a peer could then take it and
    re-present the same token concurrently, the reuse-detection POST that revokes
    the whole family. The exchange's own ``finally`` is the only release.

    THE DEFERRAL IS A STATE, AND IT IS READ FROM THE ROW — ``deferred``, below —
    because the check it describes cannot produce it: a refresh whose token is in
    doubt returns (or raises) immediately, so it costs nothing to re-decide and
    it is the one verdict here whose truth moves on its own, inside the window
    the memo would otherwise hold it for. It is therefore taken BEFORE the memo and
    never remembered, and that ordering is the whole of the protection: a memo
    written just before a marker appeared (a timeout, a transport failure that left
    the marker behind) keeps the key it was stored under — a marker write does not
    move `updated_at`, which is what the key is built from — so a memo-first order
    would answer `unknown` for the rest of :data:`VERDICT_TTL_S` while the row was
    in fact deferred. Reading the row first also means the reverse is true: once the
    marker clears, the next poll recomputes rather than holding a stale `deferred`
    for the remainder of the window.

    AND THE LOGIN ITSELF CAN BE GONE FROM THIS DEVICE — `owner_missing`, not
    `login_required`. The two used to be one verdict, and the cost was an operator
    sent to a sign-in that could never clear what they were looking at: a new
    sign-in produces a NEW row and nothing re-points ``tunnel/config.json``, so
    `lop login radient` was advice to repeat forever (issue #1711). It is read
    from the store LIKE the deferral — ahead of the memo, never remembered — for
    the same reason the row-first ordering exists: the fact moves with no write
    this key can watch (deletion removes the row the key is built from), and
    re-deciding it costs one store read, so a held answer would be staleness with
    nothing bought. The two arms below are ONE fact: a `credential_id` that cannot
    name a row (absent here, or not an int at all) and one naming a row this
    device does not have are both "the configuration points at a login that is not
    here", and neither is cleared by signing in.

    WHY THE CREDENTIAL-WRITE HOOK HEALS ONLY THE SAME-ACCOUNT CASE (where
    ``install.rearm_if_parked`` already runs): a GENERAL re-point — to whatever
    login is on this device — needs a CLOUD read to learn who owns the tunnel,
    and that hook is deliberately cheap by construction (stats and small file
    reads), so the read would turn every login into a network call; and surfaces
    must be honest anyway for the cases no heal can fix (a tunnel owned by
    another account; an unrecoverable account). That stays a deliberate decision
    (issue #1711). The ONE re-point the hook does make,
    ``install.heal_pin_after_login``, needs no cloud read because it is proven
    locally (``api.same_account_successor``): a verified sign-in to the SAME
    account id, on the tunnel's own hub, replacing a pin that is dead or gone —
    the owner cannot change, only the row carrying its bearer (2026-11-06: a
    legacy-keyed pin with a refused grant, and every sign-in landing on a new
    row beside it).

    When that heal could not run, :func:`login_verdict` (the wrapper) adds the
    successor to this verdict; this function never returns it.
    """
    selected = value.get("credential_id")
    dead: dict[str, Any] = {"credential_id": selected, "state": "login_required"}
    if not isinstance(selected, int) or isinstance(selected, bool):
        # No usable id at all is the same operator-facing fact as an id with no
        # row (see the docstring): the configuration names no login that exists
        # here, and a sign-in cannot re-point it. `dead` stays for the one arm
        # below that a sign-in DOES clear — the store's own refused-grant verdict.
        return {"credential_id": selected, "state": gateway.OWNER_MISSING}
    with closing(AuthStore()) as store:
        row = store.get_credential(selected)
        if row is None or row.provider != "radient" or row.credential_type != "oauth":
            # Read from the store BEFORE the memo below, and never remembered:
            # the row can be deleted (logout) at any moment, with no write this
            # memo's key can watch, and one store read is all re-deciding costs.
            return {"credential_id": selected, "state": gateway.OWNER_MISSING}
        if store.refresh_deferred(selected):
            # The row itself says an exchange's outcome is unsettled and the
            # bearer it holds is spent, so the store will not present that token
            # again until the marker expires or an exchange resolves it. That is
            # a state, not a failure, and it CLEARS BY ITSELF: the surfaces print
            # it without a command and without "sign-in expired", which is what
            # this case used to be reported as (`unknown`, rendered as "could not
            # be checked", wrong twice — the check did run, and no network had
            # anything to do with it). "Clears by itself" is about ONE WINDOW and
            # not about the state (agent review round 2, R6): a stalling endpoint
            # re-arms the marker, so the copy carries the sign-in escalation for
            # the case it persists rather than an absolute.
            return {"credential_id": selected, "state": "deferred"}
        key = _verdict_key(store, selected, row.updated_at)
        remembered = _remembered_verdict(key)
        if remembered is not None:
            return remembered
        task = store.detached_refresh(selected)
        # Bound the WAIT, never the work — and not with `wait_for(shield(task))`,
        # which is the same intent one stdlib layer up: on Python 3.14 `shield`
        # installs a `_log_on_exception` callback on the inner task when the outer
        # is cancelled, so every abandoned exchange that later failed (a 5xx, a
        # stalled read, an answer that never arrived) logged a full
        # "AuthStoreError exception in shielded future" traceback through the
        # loop's exception handler — measured here, one per occurrence, on the
        # ORDINARY path of this change. `asyncio.wait` has no such callback and
        # the same guarantee (the task is not cancelled when the wait times out),
        # so the exchange's outcome is reported by the exchange and by nothing
        # else. Measured: shield ⇒ 1 loop-level error; asyncio.wait ⇒ 0.
        done, _pending = await asyncio.wait({task}, timeout=REFRESH_WAIT_S)
        if not done:
            # The caller stops WAITING. While the process's loop lives, the exchange
            # does not stop: it is still running on its own store and its own lease,
            # and it — not this caller — owns resolving the marker (a landed
            # rotation, a pre-send failure and an answered refusal all resolve it
            # there, and an answer that never arrives leaves it armed until it
            # expires). When the LOOP does not outlive the call — `lop tunnel status`
            # under `asyncio.run` — the pending exchange is cancelled at process
            # exit: the marker stays armed `unknown` for the derived bound and the
            # next long-lived caller resolves it. Both halves are pinned by tests;
            # the docstring above states why the second is accepted rather than
            # engineered around.
            #
            # The `store._release_refresh_lease(selected)` call that used to sit
            # here is DELETED, and the precise reason matters because the obvious
            # one is wrong: `_release_refresh_lease` is holder-scoped, and the lease
            # now belongs to the supervisor's own `AuthStore`, so that call against
            # THIS store is a no-op — a line that reads like a safety mechanism and
            # does nothing (measured: re-adding it changes no assertion in the suite).
            # What must not happen is the thing it looks like it does: freeing the
            # lease would let a peer take it and re-present the token while this
            # POST is on the wire — the reuse-detection POST that revokes the whole
            # family — so the release stays where it is meaningful, in the
            # exchange's own `finally`.
            return _remember_verdict(key, {"credential_id": selected, "state": "unknown"})
        try:
            await task
        except CredentialInvalidError:
            return _remember_verdict(key, dead)
        except RefreshUnconfirmedError:
            # The marker went live between the row read above and the exchange
            # taking the lease — a peer, or an earlier exchange of our own — so
            # this arm is the same state as the early return, reached the other
            # way. Caught BEFORE the `AuthStoreError` arm it subclasses, because
            # an unexpected-order regression here would report a self-clearing
            # deferral as "could not be checked" again.
            return {"credential_id": selected, "state": "deferred"}
        except AuthStoreError:
            # Reachable row, unusable answer: the network, not the login.
            return _remember_verdict(key, {"credential_id": selected, "state": "unknown"})
    return {"credential_id": selected, "state": "ok"}


def remedy(
    value: dict[str, Any], connector: dict[str, Any], login: dict[str, Any]
) -> dict[str, Any] | None:
    """The one command that clears what was just found, when one exists.

    Every remedy also carries ``kind`` — the button-shaped twin of ``command``
    from `gateway.REMEDY_KIND`, the vocabulary the desktop setup route speaks —
    so a graphical surface can offer the ACTION (reconnect, accept the price, add
    credit, open the console) instead of printing a terminal command a
    non-technical user cannot run. A connector that is RETRYING on a named cause
    (no park) now gets a remedy too, because "the tunnel isn't active" with no
    action is exactly the dead end the field report hit; the waiting kinds
    (`wait_network`, `wait_radient`) carry no command, since they clear by
    themselves.

    The connector's own park carries its remedy (only it knows whether the fix is
    a login, an install or a re-enrolment); nothing else has one to offer, and
    `null` says so rather than inventing a command.

    ONE PARK YIELDS TO THE LOGIN VERDICT: a `login_required` park whose login is
    `owner_missing` (the stored row is GONE from this device, below). The park's
    own command there is provably unable to clear anything — a fresh sign-in is a
    new row and nothing re-points the configuration — so the verdict's re-point
    command stands in. The same holds for a `login_required` park whose verdict
    carries ``successor_credential_id``: a newer sign-in to the same account is
    already here, so the park's sign-in command cannot clear it either. Every
    other park, including a genuine refused grant with no successor, keeps its
    own remedy.

    A deliberately stopped tunnel has NO remedy even when the credential behind
    it is dead: `stopped` means the operator is not using the tunnel, and
    handing them a sign-in command for remote access they turned off is the nag
    the design rules out. The state word cannot carry this — a gateway that is
    simply not answering also reports `stopped` — so the decision is taken from
    the configuration, which is where the operator's intent lives.
    """
    if value.get("stopped"):
        return None
    parked = connector.get("remedy")
    if isinstance(parked, dict) and parked.get("command"):
        # The ONE park the login verdict can contradict (issue #1711): a
        # connector parked for a login it could not use, on a device whose
        # stored row for that login is GONE (`owner_missing` below). The park's
        # own command cannot clear that state — a fresh sign-in is a NEW row and
        # nothing re-points the configuration at it — while the verdict's command
        # does, so the verdict's stands in. Every other park keeps its own
        # remedy: the connector knows better than this module what its own fix is.
        #
        # The same holds for a park whose dead login has a same-account
        # successor already on this device (`successor_credential_id`): the
        # park's sign-in has, in effect, already been done, and repeating it
        # writes the same new row again.
        if not (
            connector.get("reason") == gateway.LOGIN_REQUIRED
            and (
                login.get("state") == gateway.OWNER_MISSING
                or login.get("successor_credential_id") is not None
            )
        ):
            return {
                "kind": gateway.remedy_kind(str(connector.get("reason") or "")) or "run",
                "command": str(parked["command"]),
                "url": str(parked.get("url") or gateway.CONSOLE_URL),
            }
    if login.get("successor_credential_id") is not None and login.get("state") in (
        gateway.LOGIN_REQUIRED,
        gateway.OWNER_MISSING,
    ):
        # Ahead of both arms below: a newer sign-in to the same account is
        # already here, so the fix is re-pointing at it, never another sign-in.
        return {
            "kind": gateway.REMEDY_KIND[gateway.LOGIN_SUPERSEDED],
            "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_SUPERSEDED],
            "url": gateway.CONSOLE_URL,
        }
    if login.get("state") == gateway.OWNER_MISSING:
        # The re-point command, never a sign-in: the login this tunnel was
        # enrolled with is not on this device, and only re-pointing the
        # configuration (or signing in to the account that owns the tunnel, which
        # the command's own refusal explains) brings remote access back. The
        # command is pinned in `gateway`'s table so every surface renders one
        # spelling of it.
        return {
            "kind": gateway.REMEDY_KIND[gateway.OWNER_MISSING],
            "command": gateway.TERMINAL_REMEDY[gateway.OWNER_MISSING],
            "url": gateway.CONSOLE_URL,
        }
    if login.get("state") == "login_required":
        return {
            "kind": gateway.REMEDY_KIND[gateway.LOGIN_REQUIRED],
            "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED],
            "url": gateway.CONSOLE_URL,
        }
    return _retrying_remedy(connector)


def _retrying_remedy(connector: dict[str, Any]) -> dict[str, Any] | None:
    """The remedy for a connector that is not parked but names why it is failing.

    The reason arrives from the gateway's health answer or the connector's last
    CURRENT retrying exit (`probe`), both already this package's vocabulary.
    Only causes with an ACTION or a reassuring wait are surfaced: `check_status`
    kinds (a lease about to renew, a briefly busy store) stay ``None``, because
    a callout for a state that clears in seconds is noise, and an unknown code
    from another build is never handed a guessed button.
    """
    reason = str(connector.get("reason") or "")
    kind = gateway.remedy_kind(reason)
    if kind is None or kind == "check_status":
        return None
    waiting = kind in gateway.WAITING_KINDS
    return {
        "kind": kind,
        # A waiting kind has nothing to run; naming `lop tunnel status` there
        # would send a reader to "fix" a state that is recovering by itself.
        "command": None if waiting else gateway.TERMINAL_REMEDY.get(reason),
        "url": None if waiting else gateway.CONSOLE_URL,
    }


def unconfigured_remedy() -> dict[str, Any]:
    """What a machine with NO tunnel is offered: set phone access up, or sign in.

    `run` when a Radient login is already on this device (the desktop renders its
    "Set up phone access" card from it and posts the setup route), `sign_in` when
    there is none yet — setup cannot start without a login, and offering `run`
    would put a button in front of a refusal. Read-only: a metadata read of the
    credential store, never a write, so the polled routes stay side-effect free.
    """
    try:
        with closing(AuthStore(read_only=True)) as store:
            signed_in = any(
                row.credential_type == "oauth" for row in store.list_credentials("radient")
            )
    except (AuthStoreError, OSError):
        signed_in = False
    if signed_in:
        return {"kind": "run", "command": "lop tunnel setup", "url": None}
    return {
        "kind": gateway.REMEDY_KIND[gateway.LOGIN_REQUIRED],
        "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED],
        "url": None,
    }


def payload(
    value: dict[str, Any],
    record: dict[str, Any],
    *,
    source: str,
    cloud_reason: str,
    connector: dict[str, Any],
    login: dict[str, Any],
) -> dict[str, Any]:
    """The machine-readable state: the shape every consumer reads.

    READ-ONLY BY CONSTRUCTION: there is no field here a caller could use to
    change anything, which is what lets the desktop route expose it without
    inventing an operation. ``cloud.source`` is the provenance of the tunnel
    record — `live` when the cloud answered this request, `cached` when the copy
    stored at the last `create`/`connect`/`configure` had to stand in — and it
    exists because reading `status: active` off a cached copy is exactly how a
    withdrawn tunnel looked healthy.
    """
    return {
        "tunnel_id": record.get("id"),
        "cloud": {
            "status": record.get("status", "configured"),
            "source": source,
            "reason": cloud_reason,
        },
        "connector": connector,
        "login": login,
        "remedy": remedy(value, connector, login),
    }


async def local_payload(*, reachable: bool = True) -> dict[str, Any]:
    """The payload for a surface with no cloud read of its own.

    ``cloud.source`` is honestly `cached`: this reports what is stored on this
    device plus what the connector is doing, which is the part a cloud read
    cannot answer when the connection or the login is what is broken.
    """
    try:
        value = config.load()
    except ValueError:
        return {
            "configured": False,
            "tunnel_id": None,
            "cloud": {"status": "not configured", "source": "cached", "reason": ""},
            # Not "stopped": nothing was stopped here, and a surface that said so
            # would have an operator looking for a connector this machine has
            # never enrolled.
            "connector": {
                "state": "not configured",
                "reason": "",
                "detail": "",
                "since": None,
                "remedy": None,
            },
            "login": {"credential_id": None, "state": "unknown"},
            # Not null: a machine with no tunnel has one obvious next step, and
            # "nothing to do" is how a teammate ended up at a phone that could
            # not connect with no idea why.
            "remedy": unconfigured_remedy(),
        }
    stored = value.get("record")
    assembled = payload(
        value,
        stored if isinstance(stored, dict) else {},
        source="cached",
        cloud_reason="",
        connector=await connector_state(value, reachable=reachable),
        login=await login_verdict(value),
    )
    assembled["configured"] = True
    # The post-sign-in no-spend run, while it holds `setup.lock` (U10): said
    # outright, so a desktop reading this mid-reconnect shows "reconnecting"
    # rather than the stale cause that run is busy clearing. `autosetup`, not
    # `setup`: this polled read must not load the CLI and installers.
    from local_operator.tunnels import autosetup

    reconnecting = await asyncio.to_thread(autosetup.auto_reconnecting)
    assembled["reconnecting"] = reconnecting
    if reconnecting and not value.get("stopped"):
        assembled["remedy"] = {"kind": "wait_reconnect", "command": None, "url": None}
    return assembled

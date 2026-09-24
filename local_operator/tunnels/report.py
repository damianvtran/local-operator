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
import time
from contextlib import closing
from typing import Any

import httpx

from local_operator.providers.auth_store import (
    DEFAULT_BLOCK_MS,
    AuthStore,
    AuthStoreError,
    CredentialInvalidError,
    RefreshUnconfirmedError,
)
from local_operator.tunnels import config, gateway, state

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
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            reply = await client.get(
                f"http://127.0.0.1:{value['gateway_port']}/_lop_tunnel/health", timeout=2
            )
            served = reply.status_code == 200
            payload = reply.json() if served else {}
            if not isinstance(payload, dict):
                # A stale or foreign listener on this port can answer 200 with
                # any JSON at all. Anything but an object is not a health
                # payload, and a status surface must not raise over it.
                payload = {}
            healthy = served and payload.get("ok") is True
            connected = healthy and payload.get("connected") is True
            if not healthy:
                # The gateway names why it is refusing relayed requests, and
                # this is the surface where a command can be offered at all.
                # A reason this build does not know falls back to the relay's
                # own sentence rather than saying nothing.
                reason = str(payload.get("reason") or "")
                detail = gateway.terminal_detail(reason, str(payload.get("detail") or ""))
    except (httpx.HTTPError, ValueError):
        # A stopped connector and a gateway that is not there are also
        # different jobs: the first is this process, the second is the unit.
        detail = (
            "The local relay gateway is not answering on "
            f"127.0.0.1:{value['gateway_port']}; run lop tunnel install to restore it"
        )
    if connected:
        word = "connected"
    elif healthy:
        word = "connecting"
    elif served:
        # The gateway answered and is refusing to serve. That is not a stopped
        # connector — cloudflared may still hold the edge connection — and
        # "stopped" beside a sentence promising it clears itself would
        # contradict the payload this just read.
        word = "not serving"
    else:
        word = "stopped"
    return {"state": word, "reason": reason, "detail": detail, "since": None, "remedy": None}


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
    """
    selected = value.get("credential_id")
    dead: dict[str, Any] = {"credential_id": selected, "state": "login_required"}
    if not isinstance(selected, int) or isinstance(selected, bool):
        return dead
    with closing(AuthStore()) as store:
        row = store.get_credential(selected)
        if row is None or row.provider != "radient" or row.credential_type != "oauth":
            return dead
        if store.refresh_deferred(selected):
            # The row itself says an exchange's outcome is unsettled and the
            # bearer it holds is spent, so the store will not present that token
            # again until the marker expires or an exchange resolves it. That is
            # a state, not a failure, and it CLEARS BY ITSELF: the surfaces print
            # it without a command and without "sign-in expired", which is what
            # this case used to be reported as (`unknown`, rendered as "could not
            # be checked", wrong twice — the check did run, and no network had
            # anything to do with it).
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
            # The caller stops WAITING; the exchange does not stop. It is still
            # running on its own store and its own lease, and it — not this
            # caller — owns resolving the marker: a landed rotation, a pre-send
            # failure and an answered refusal all resolve it there, and an answer
            # that never arrives leaves it armed until it expires.
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
) -> dict[str, str] | None:
    """The one command that clears what was just found, when one exists.

    The connector's own park carries its remedy (only it knows whether the fix is
    a login, an install or a re-enrolment); nothing else has one to offer, and
    `null` says so rather than inventing a command.

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
        return {
            "command": str(parked["command"]),
            "url": str(parked.get("url") or gateway.CONSOLE_URL),
        }
    if login.get("state") == "login_required":
        return {
            "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED],
            "url": gateway.CONSOLE_URL,
        }
    return None


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
            "remedy": None,
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
    return assembled

"""Owner-pinned Radient API calls. Never use the model-routing key cascade."""

from __future__ import annotations

import asyncio
from contextlib import closing
from typing import Any

import httpx

from local_operator.providers.account_successor import (
    same_account_successor as _successor,
)
from local_operator.providers.auth_store import (
    AuthStore,
    AuthStoreBusyError,
    AuthStoreError,
    CredentialInvalidError,
)
from local_operator.tunnels.config import DEFAULT_API_URL
from local_operator.tunnels.errors import LoginRequired, RecordNotFound

#: Pauses before re-opening the store after `AuthStoreBusyError`. Why retry at
#: all: the bootstrap guard is held only while another `lop` process opens the
#: same database (milliseconds when the machine is idle), so the contention is
#: over long before the tunnel supervisor's 10 s restart floor would bring the
#: unit back — on 2026-11-11 one such timeout at cold start cost a 15 s outage
#: (one observation, n=1; no distribution exists yet). Why so short and so few:
#: each attempt already waits the store's own 5 s guard deadline, so two extra
#: tries bound the added wait at ~1.5 s of sleep plus 10 s of guard, still
#: inside the poller's 10 s cadence plus one attempt, and a store that stays
#: busy that long surfaces as "local credential store busy" rather than hanging.
STORE_BUSY_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0)


async def _open_store() -> AuthStore:
    """`AuthStore()`, retried briefly while another process holds its bootstrap guard.

    Opened on a worker thread: the guard wait is a blocking sleep of up to its 5 s
    deadline, and on the event loop it froze the relay connector's pings and
    stream pipes for that long (measured: one contended open held the loop
    5.0 s). The store's connection is `check_same_thread=False` behind its own
    mutex, so using it back on the loop thread is safe.
    """
    for delay in STORE_BUSY_RETRY_DELAYS:
        try:
            return await asyncio.to_thread(AuthStore)
        except AuthStoreBusyError:
            await asyncio.sleep(delay)
    try:
        return await asyncio.to_thread(AuthStore)
    except AuthStoreBusyError as failure:
        # A ValueError, chained, like every other failure `request` reports: the
        # poller and `service.main` catch ValueError, and a bare AuthStoreError
        # used to escape both — killing the poller task (a unit restart) or
        # crashing a cold start with a traceback (the 2026-11-11 16:58 outage).
        # `service.authorization_failure_reason` reads the busy type off the chain.
        raise ValueError("This computer's credential store was busy.") from failure


def credential_id(selected: int | None = None) -> int:
    with closing(AuthStore()) as store:
        rows = [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
        if selected is not None:
            rows = [row for row in rows if row.id == selected]
        if len(rows) != 1:
            # The shell spelling: this travels out of `lop tunnel status`/`install`
            # through the `except ValueError → return str(exc)` path, so `/login
            # radient` here was a TUI command handed to a shell (review round 2,
            # m5 — the same rule as `config.load`'s message one module over).
            raise ValueError(
                "Log in with lop login radient first. If multiple accounts are signed in, "
                "select one with --credential-id from lop login-status."
            )
        return rows[0].id


#: The hub every tunnel request goes to (``RadientTunnels.request`` below posts to
#: ``DEFAULT_API_URL + "/v1/tunnels"``), in the spelling a Radient sign-in stamps
#: into a row's ``api_endpoint`` (``https://api.radienthq.com/v1``). A successor
#: row must be bound to THIS hub: an account label on another hub is a different
#: binding (``auth_store._radient_account_identity``), and re-pointing the tunnel
#: at it would send this tunnel's requests with a bearer minted elsewhere.
TUNNEL_HUB = DEFAULT_API_URL + "/v1"


def same_account_successor(
    pin: object,
    *,
    owner_account: object = None,
    fresh_id: int | None = None,
    read_only: bool = False,
) -> int | None:
    """The live row that should own this tunnel instead of a dead ``pin``, if PROVABLE.

    The proof (and why re-pointing on it is safe) is
    :func:`local_operator.providers.account_successor.same_account_successor`,
    shared with the Team Bridge enrollment heal so the two owners cannot drift.
    This wrapper supplies what is the TUNNEL's: the ambient credential store and
    :data:`TUNNEL_HUB`, the only hub a tunnel request ever goes to.

    ``read_only`` exists for the desktop's setup POLL (issue #3520). That poll is
    metadata-only by contract: it may not create the auth database, its WAL or
    its schema while merely opening the affordance, which the writable
    ``AuthStore()`` constructor does. The successor proof is nothing but store
    reads, so it runs unchanged over a ``mode=ro`` handle; an absent database
    reads there as an empty store and answers ``None``.
    """
    if not isinstance(pin, int) or isinstance(pin, bool):
        return None  # before the open, as it always was: no pin, no store read
    with closing(AuthStore(read_only=read_only)) as store:
        return _successor(
            pin, store=store, hub=TUNNEL_HUB, owner_account=owner_account, fresh_id=fresh_id
        )


def usable_credential_id(selected: object, *, owner_account: object = None) -> int | None:
    """``selected`` when it names a Radient oauth row this device still has.

    The predicate the tunnel commands' DEFAULT selection needs before trusting a
    pinned id: `credential_id(selected)` filters the store by an exact id and
    fails when nothing matches, so handing it a pin whose row is gone asks the
    operator to "log in first" while a current login sits right there — the
    loop issue #1711 describes (nothing re-points the configuration at the new
    row). Callers use this to drop a DEAD pin and fall back to the current login
    (``credential_id(None)``'s own rule); an EXPLICIT ``--credential-id`` is
    deliberately not routed through it, because selecting an id that names
    nothing is a typo or a stale note, and it must keep failing loudly.

    A PIN THAT IS PRESENT BUT DEAD, or gone, yields to its provable same-account
    successor (:func:`same_account_successor`) first: a dead-grant pin used to be
    returned as-is, so bare `lop tunnel configure` re-proposed the very row the
    IdP refused, and with two Radient logins on the device ``credential_id(None)``
    could not pick one either. ``owner_account`` is the cached record's
    ``owner_account_id``, the only same-account evidence left for a gone pin.
    """
    if not isinstance(selected, int) or isinstance(selected, bool):
        return None
    successor = same_account_successor(selected, owner_account=owner_account)
    if successor is not None:
        return successor
    with closing(AuthStore()) as store:
        row = store.get_credential(selected)
    if row is None or row.provider != "radient" or row.credential_type != "oauth":
        return None
    return selected


class RadientTunnels:
    """A connector stays bound to the credential chosen at enrollment.

    The inference resolver deliberately rotates credentials and may fall back
    to environment API keys. That policy is unsuitable for owning a tunnel:
    neither a quota event nor an expired login may switch the owning account.
    """

    def __init__(self, selected: int, client: httpx.AsyncClient) -> None:
        self.selected = selected
        self.client = client

    async def request(
        self,
        method: str,
        path: str = "",
        *,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        with closing(await _open_store()) as store:
            row = store.get_credential(self.selected)
            if row is None or row.provider != "radient" or row.credential_type != "oauth":
                # LoginRequired, not ValueError: no retry can conjure a credential
                # row back, and the connector's supervisor must be told to stop
                # retrying rather than to keep trying every 10 seconds forever.
                raise LoginRequired("The tunnel's Radient login is unavailable; log in again.")
            try:
                credentials = await store.ensure_oauth_fresh_or_raise(self.selected)
            except AuthStoreError as failure:
                # Chained, never flattened. A refresh that could not REACH the
                # token endpoint and a grant the IdP rejected both arrive here,
                # and only a caller that reads the chain can tell an operator
                # whether the fault is their network or their login: reporting
                # the first as an expired login is what sent a lost network to
                # /login. `authorization_failure_reason` reads `__cause__`.
                #
                # The store names a dead grant with its own type (the store-level
                # face of a token-endpoint refusal the classifier could name), and
                # that distinction is carried in the CLASS the supervisor sees, so
                # the one terminal case is never inferred from this sentence --
                # which is deliberately the same for both, because the redaction
                # rule below forbids echoing what the provider actually said.
                #
                # A DEFERRED refresh (`RefreshUnconfirmedError`) gets the same
                # treatment, and deliberately: this sentence is not where the
                # distinction lives, the chain is, and it survives there because
                # this raise uses `from failure`. `service.authorization_failure_reason`
                # and `service.classify_failure` read the chain rather than the prose,
                # so the relay can say "the refresh is deferred and is being retried"
                # instead of blaming a login nothing has happened to.
                if isinstance(failure, CredentialInvalidError):
                    raise LoginRequired(
                        "The tunnel's Radient login could not be refreshed."
                    ) from failure
                raise ValueError("The tunnel's Radient login could not be refreshed.") from failure
        token = credentials.get("access") if credentials else None
        if not isinstance(token, str) or not token:
            raise LoginRequired("The tunnel's Radient login expired; log in again.")
        headers = {"Authorization": f"Bearer {token}"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = await self.client.request(
            method,
            DEFAULT_API_URL + "/v1/tunnels" + path,
            json=body,
            headers=headers,
            timeout=30,
            follow_redirects=False,
        )
        # Never surface provider error bodies: they may echo a bearer, client
        # secret, or connector token. The status is enough for a useful retry.
        if response.status_code >= 400:
            # 404 gets a TYPE, not just the generic sentence: it is the one status
            # the tunnel commands translate, because under an owner-pinned
            # request it usually means the selected login does not own this
            # tunnel (the message is composed where the local record and the
            # selected login meet — see `errors.RecordNotFound`).
            if response.status_code == 404:
                raise RecordNotFound("Radient has no record of this tunnel for the selected login.")
            raise ValueError(f"Radient tunnel request failed (HTTP {response.status_code}).")
        envelope = response.json()
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise ValueError("Radient returned an invalid tunnel response.")
        return envelope["result"]

    async def account_id(self) -> str | None:
        """The selected login's Radient account id, read live from `GET /v1/me`.

        Used only to explain a 404 on this tunnel's record: `result.account.id`
        is the same identifier the cloud writes into a tunnel record's
        `owner_account_id` (with `result.identity.account_id` carrying it for an
        account object without an id). Every failure on the way — a store read,
        a refresh, transport, a non-200, an unparseable body, an absent field —
        answers `None` on purpose: a comparison this module cannot make must
        degrade to the sentence that names BOTH possibilities, never claim one.
        Never raises, and never echoes a provider body (see `request` above for
        the redaction rule this shares).
        """
        with closing(AuthStore()) as store:
            row = store.get_credential(self.selected)
            if row is None or row.provider != "radient" or row.credential_type != "oauth":
                return None
            try:
                credentials = await store.ensure_oauth_fresh_or_raise(self.selected)
            except AuthStoreError:
                return None
        token = credentials.get("access") if credentials else None
        if not isinstance(token, str) or not token:
            return None
        try:
            response = await self.client.get(
                DEFAULT_API_URL + "/v1/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            return None
        for source in (result.get("account"), result.get("identity")):
            if not isinstance(source, dict):
                continue
            identifier = source.get("id") or source.get("account_id")
            if isinstance(identifier, str) and identifier:
                return identifier
        return None

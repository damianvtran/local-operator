"""``MeshAwareAuthStore``: ``AuthStore`` with brokering as the LAST rung.

WHY A WRAPPER AND NOT A NEW CASCADE (design §0, property 2). Everything that
already resolves credentials on this device keeps working untouched: the local
tiers run first and in their existing order, and the mesh is consulted only when
they produced NOTHING. A device with no borrowable placement never gets this class
at all (:func:`build_auth_store`), so the 0-peer topology runs the plain
``AuthStore`` byte for byte.

THE SURFACE IS ABOUT TWENTY METHODS, NOT THREE. The design's §3.6a listed the
three protocols the failover driver consumes, but the real call sites are wider
(build plan §0 finding 5): ``model/configure.py`` alone reaches ``list_credentials``,
``is_blocked_for_model``, ``get_credential``, ``block_credential``,
``session_credential_id``, ``release_session_credential``, ``pin_session_credential``,
``list_oauth_accesses``, ``ensure_oauth_fresh``, ``deprioritize_credential`` and
``clear_blocks_for_model``, many of them keyed by ``credential_id`` — and MCP calls
``upsert_credential``/``delete_credential`` on the same object
(``session_factory.wire_mcp_into_session``). A wrapper that covered three would
have raised ``AttributeError`` on the first real turn.

THE SYNTHETIC ID IS HOW THAT SURFACE WORKS. A borrowed credential has no row here,
so it needs an id: ``types.synthetic_credential_id`` returns a NEGATIVE integer,
which SQLite's ``INTEGER PRIMARY KEY`` (assigned from 1 upward) can never produce.
Every method that takes a ``credential_id`` checks
:func:`types.is_synthetic_credential_id` and answers for the borrow — never by
writing to the local database, because the row it would write is the OWNER's, on
another machine, and this device has no authority over it.

THE ONE THING THIS CLASS MUST NOT DO is spend the owner's refresh token. It has
none: ``get_api_key`` and ``get_oauth_access`` ask the broker for a bearer, and the
borrower's own refresh path (``ensure_oauth_fresh``, ``_ensure_oauth_fresh``) is
UNREACHABLE for a brokered credential because there is no row to refresh. That is
why a brokered failure is reported to the owner rather than repaired here, and why
``rotate_sibling`` never reaches ``disable_credential`` from this side.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Collection, Protocol

from local_operator.network.credentials.client import MeshCredentialClient, key_for
from local_operator.network.credentials.placement import placement_for_store
from local_operator.network.credentials.types import (
    BrokerError,
    Grant,
    is_mcp_key,
    is_synthetic_credential_id,
    synthetic_credential_id,
)


class CredentialSource(Protocol):
    """The broker seam this store consumes, stated as a Protocol.

    WHY A PROTOCOL AND NOT THE CLASS. Two reasons, and both are the repo's own
    pattern (``mcp.auth.StructuralAuthStore`` is the precedent): the store must be
    testable WITHOUT a wire — a fake that satisfies this names exactly what the
    broker rung uses and nothing more, so a test cannot pass by accident — and MCP
    reads the same seam through ``mesh_client`` (``mcp/manager.py``), which should
    depend on the shape it uses rather than on the whole client.
    """

    #: This device's id, as the placement document spells it.
    self_device: str
    #: The in-memory grants. Never persisted; see ``client.GrantCache``.
    grants: Any
    placement: Any

    def should_borrow(self, key: str) -> bool: ...

    def owner_of(self, key: str) -> str: ...
    def owner_label(self, key: str) -> str: ...
    def owner_last_seen_s(self, device: str) -> float | None: ...

    async def grant_async(
        self,
        key: str,
        *,
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        provider: str = "",
    ) -> Any: ...

    def report_sync(
        self,
        key: str,
        *,
        kind: str,
        session_id: str = "",
        model_id: str = "",
        retry_after_ms: int = 0,
        block_scope: str = "",
    ) -> None: ...

    def close(self) -> None: ...


def build_auth_store(config_dir: Path | None = None) -> Any:
    """The store a session should use: plain, or mesh-aware when this device borrows.

    THE ONE CONSTRUCTION SITE (``session_factory``), and the only place the 0-peer
    guarantee is enforced. It returns ``AuthStore`` — the class the call site used
    before this slice existed — unless there is a placement entry naming a
    credential that ANOTHER device owns and this one is a holder for. A device that
    owns everything it knows about, or holds nothing, therefore takes the old path
    with the old object, and ``type(build_auth_store(cfg)) is AuthStore`` is the
    test that pins it.
    """
    from local_operator.providers.auth_store import AuthStore

    found = placement_for_store(config_dir)
    if found is None:
        return AuthStore(config_dir=config_dir)
    _network_id, document = found
    client = MeshCredentialClient.for_this_device(config_dir)
    if client is None:  # pragma: no cover - placement_for_store is the same predicate
        return AuthStore(config_dir=config_dir)
    return MeshAwareAuthStore(AuthStore(config_dir=config_dir), mesh=client, config_dir=config_dir)


class MeshAwareAuthStore:
    """An ``AuthStore`` with one added rung, consulted last."""

    def __init__(
        self,
        local: Any,
        *,
        mesh: CredentialSource | None,
        config_dir: Path | None = None,
    ) -> None:
        self._local = local
        self._mesh = mesh
        self.config_dir = config_dir
        #: One future per ``(provider, session_id)`` so several provider calls in one
        #: turn share ONE borrow (design §3.5). Without it a turn issues one request
        #: per provider call, and each is a round trip to a device that may be
        #: refreshing.
        self._grants: dict[tuple[str, str], asyncio.Future[Any]] = {}
        #: The synthetic id of the credential last borrowed for ``(provider,
        #: session_id)``, so a later failure can be attributed to the BORROW rather
        #: than to a local row with a coincidentally similar id.
        self._brokered: dict[tuple[str, str], int] = {}

    # -- the two resolution entry points the providers use ------------------

    async def get_api_key(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
        model_id: str = "",
        exclude_keys: Collection[str] | None = None,
        exclude_credential_ids: Collection[int] | None = None,
    ) -> str | None:
        """LOCAL FIRST, then the broker. Returns ``None`` exactly as before."""
        key = await self._local.get_api_key(
            provider,
            session_id,
            force_refresh=force_refresh,
            read_only=read_only,
            model_id=model_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        if key:
            # LOCAL-FIRST IS NOT NEGOTIABLE, and it is also what keeps a device that
            # has its own login from quietly spending someone else's account.
            return key
        grant = await self._borrow(
            key_for(provider=provider),
            provider=provider,
            session_id=session_id or "",
            model_id=model_id,
            force_refresh=force_refresh,
        )
        return grant.access_token if isinstance(grant, Grant) else None

    async def get_oauth_access(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
        model_id: str = "",
        exclude_keys: Collection[str] | None = None,
        exclude_credential_ids: Collection[int] | None = None,
    ) -> Any:
        """Local access, else a borrowed one as an ``OAuthAccess``."""
        access = await self._local.get_oauth_access(
            provider,
            session_id,
            force_refresh=force_refresh,
            read_only=read_only,
            model_id=model_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        if access is not None:
            return access
        grant = await self._borrow(
            key_for(provider=provider),
            provider=provider,
            session_id=session_id or "",
            model_id=model_id,
            force_refresh=force_refresh,
        )
        if not isinstance(grant, Grant):
            return None
        return self._access_from_grant(grant)

    # -- the broker rung ----------------------------------------------------

    async def _borrow(
        self,
        key: str,
        *,
        provider: str,
        session_id: str,
        model_id: str,
        force_refresh: bool,
    ) -> Grant | BrokerError | None:
        """One borrow, shared with the other provider calls in this turn.

        THE FUTURE IS THE COALESCER: the first caller creates it, later callers in
        the same turn await the same one. The event loop is this session's own, so
        the future belongs to it; a borrow that is already finished is not cached
        here — the CLIENT caches grants (15 min, in memory), which is where the TTL
        belongs because it is a fact about the token rather than about the turn.
        """
        if self._mesh is None or not self._mesh.should_borrow(key):
            return None
        pair = (key, session_id)
        existing = self._grants.get(pair)
        if existing is not None and not existing.done():
            return await asyncio.shield(existing)
        loop = asyncio.get_running_loop()
        slot: asyncio.Future[Any] = loop.create_future()
        self._grants[pair] = slot
        try:
            grant = await self._mesh.grant_async(
                key,
                session_id=session_id,
                model_id=model_id,
                force_refresh=force_refresh,
                provider=provider,
            )
        except BaseException as exc:
            slot.set_exception(exc)
            raise
        if isinstance(grant, Grant):
            self._brokered[pair] = self._credential_id(key, grant)
        slot.set_result(grant)
        return grant

    def _credential_id(self, key: str, grant: Grant) -> int:
        return synthetic_credential_id(key, grant.credential_ref.owner_device)

    def _access_from_grant(self, grant: Grant) -> Any:
        """A borrowed bearer shaped as the ``OAuthAccess`` wire clients consume.

        ``raw`` is deliberately ``None``. That field exists so a USAGE fetch can read
        the provider's own token out of the row, and a broker grant has no row here:
        handing a usage fetcher a borrowed bearer would spend the owner's quota from
        this device a second time, for a number the owner already reports on its own
        panel. So the borrowed credential has a bearer and no raw row, which is
        exactly the truth.
        """
        from local_operator.providers.auth_store import OAuthAccess

        credential_kind = "api_key" if grant.credential_ref.kind == "api_key" else "oauth"
        return OAuthAccess(
            access_token=grant.access_token,
            credential_id=self._credential_id(
                key_for(provider=grant.credential_ref.provider), grant
            ),
            account_id=grant.identity.get("account_id"),
            email=grant.identity.get("email"),
            org_id=grant.identity.get("org_id"),
            kind=credential_kind,
            raw=None,
        )

    def _synthetic_for_provider(self, provider: str) -> int | None:
        if self._mesh is None:
            return None
        key = key_for(provider=provider)
        owner = self._mesh.owner_of(key)
        if not owner or not self._mesh.should_borrow(key):
            return None
        return synthetic_credential_id(key, owner)

    def _brokered_pair(self, credential_id: int) -> tuple[str, str] | None:
        """The ``(key, session_id)`` a synthetic id belongs to, if any."""
        for pair, value in self._brokered.items():
            if value == credential_id:
                return pair
        return None

    # -- the failure-accounting rung ---------------------------------------

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: BaseException,
        api_key: str | None = None,
        block_ms: int = 60_000,
        *,
        model_id: str = "",
    ) -> bool:
        """Account for a failure WHERE THE ROW LIVES — which is the owner, for a borrow.

        ``AuthStore.rotate_sibling`` soft-deletes an invalidated credential
        (``disable_credential(cause="invalidated-token")``). Running it here against
        a borrowed credential would write that verdict into THIS device's database
        for a row it does not have — and, worse, the design's original sketch had the
        owner run it on a peer's word, which would let one bad 401 on one peer log
        the operator out of every device (finding 8, cut line unsafe item 2).

        So: a local failure delegates unchanged, and a brokered one becomes a
        REPORT. The owner decides, and this device never touches the owner's row.
        The owner also BOUNDS what a report can do (``owner.py``'s report arm): a
        quota report writes at most a short, family-scoped block, never an
        account-wide one.
        ``False`` means "no sibling of this type here", which lets the failover
        driver move on to another provider — the honest answer, because there is no
        local sibling to rotate to.
        """
        pair = self._brokered_pair_for(provider, session_id, api_key)
        if pair is None:
            return bool(
                self._local.rotate_sibling(
                    provider, session_id, error, api_key, block_ms, model_id=model_id
                )
            )
        self._report_failure(pair[0], error=error, session_id=session_id or "", model_id=model_id)
        return False

    def block_credential(
        self,
        credential_id: int,
        provider: str,
        block_scope: str = "",
        block_ms: int = 0,
    ) -> None:
        """A synthetic id is reported, never blocked locally.

        ``AuthStore.block_credential`` writes an ``auth_credential_blocks`` row keyed
        by ``credential_id``. For a borrowed credential the row that verdict belongs
        to is on the OWNER, so a local write would either create a block against an id
        that does not exist — or, if ids ever collided, against an unrelated local
        login. The owner's own ``block_credential`` is reached through the report
        instead (``owner.py``'s quota arm).

        ``block_scope`` TRAVELS WITH THE REPORT. ``configure._write_quota_block``
        writes ``model:fable`` for a Fable-only cap, and dropping it here made the
        owner receive an unscoped report — which the first version turned into an
        ACCOUNT-WIDE block (review round 1, F3). An unscoped report is now written
        nowhere on the owner, so losing the scope would lose the verdict; carrying
        it keeps the family block the borrower actually observed.
        """
        if is_synthetic_credential_id(credential_id):
            pair = self._brokered_pair(credential_id)
            if pair is not None and self._mesh is not None:
                self._mesh.report_sync(
                    pair[0],
                    kind="quota",
                    session_id=pair[1],
                    retry_after_ms=int(block_ms or 0),
                    block_scope=block_scope,
                )
            return
        kwargs: dict[str, Any] = {}
        if block_ms:
            kwargs["block_ms"] = block_ms
        self._local.block_credential(credential_id, provider, block_scope=block_scope, **kwargs)

    def _brokered_pair_for(
        self, provider: str, session_id: str | None, api_key: str | None
    ) -> tuple[str, str] | None:
        """Which brokered credential a failure belongs to, or ``None`` for a local one.

        Three ways in, and each exists for a real caller: the recorded synthetic id
        for this ``(provider, session)`; a bearer this device is holding a grant for;
        and the provider naming a remote owner with no local row at all. The last one
        matters because a 401 can arrive before any grant was recorded — the provider
        answered the owner's bearer, and this device has no local credential to blame.
        """
        key = key_for(provider=provider)
        if self._mesh is None:
            return None
        recorded = self._brokered.get((key, session_id or ""))
        if recorded is not None and is_synthetic_credential_id(recorded):
            return (key, session_id or "")
        if api_key:
            found = self._mesh.grants.find_by_token(key, api_key)
            if found is not None:
                return found
        if self._local_has_rows(provider):
            return None
        if self._mesh.should_borrow(key):
            return (key, session_id or "")
        return None

    def _local_has_rows(self, provider: str) -> bool:
        try:
            return bool(self._local.list_credentials(provider))
        except Exception:  # noqa: BLE001 — an unreadable store is not "has rows"
            return True

    def _report_failure(
        self, key: str, *, error: BaseException, session_id: str, model_id: str
    ) -> None:
        """Classify the provider's own error and report it, using the driver's helpers.

        The classification is the failover driver's, not this module's: the same
        ``is_invalidated_credential_error`` / ``retry_after_ms_from_error`` the local
        rotation uses, so a 401 is a 401 on both sides. Inventing a second classifier
        here would let the two disagree about what an error means.
        """
        if self._mesh is None:
            return
        from local_operator.providers.failover import retry_after_ms_from_error

        self._mesh.grants.drop(key, session_id)
        kind = report_kind_for(error)
        self._mesh.report_sync(
            key,
            kind=kind,
            session_id=session_id,
            model_id=model_id,
            retry_after_ms=int(retry_after_ms_from_error(error) or 0),
        )

    # -- the read-through surface keyed by credential_id --------------------

    def get_credential(self, credential_id: int) -> Any:
        """A synthetic descriptor for a brokered id, the real row otherwise.

        The descriptor is what ``configure.py`` needs in order to name the
        credential it resolved: an id, a provider and a type. It carries NO ``data``
        — the payload is the owner's, and a borrower that materialised it here would
        be holding the row it was refused.
        """
        if not is_synthetic_credential_id(credential_id):
            return self._local.get_credential(credential_id)
        pair = self._brokered_pair(credential_id)
        if pair is None:
            return None
        from local_operator.providers.auth_store import StoredCredential

        key = pair[0]
        provider = mcp_provider_label(key) if is_mcp_key(key) else key
        return StoredCredential(
            id=credential_id,
            provider=provider,
            credential_type="oauth",
            data={},
            disabled_cause=None,
            identity_key=None,
        )

    def is_blocked(self, credential_id: int, provider: str) -> bool:
        if is_synthetic_credential_id(credential_id):
            return False
        return bool(self._local.is_blocked(credential_id, provider))

    def is_blocked_for_model(self, credential_id: int, provider: str, model_id: str) -> bool:
        """A borrow is never blocked here: the owner's own row carries the verdict.

        Answering ``True`` from a stale local cache would take a working credential
        out of THIS device's cascade for a block the owner may already have cleared;
        the owner refuses per request anyway, and its refusal is authoritative.
        """
        if is_synthetic_credential_id(credential_id):
            return False
        return bool(self._local.is_blocked_for_model(credential_id, provider, model_id))

    def clear_blocks_for_model(self, credential_id: int, provider: str, model_id: str) -> None:
        if is_synthetic_credential_id(credential_id):
            return
        self._local.clear_blocks_for_model(credential_id, provider, model_id)

    def deprioritize_credential(self, provider: str, credential_id: int) -> None:
        if is_synthetic_credential_id(credential_id):
            return
        self._local.deprioritize_credential(provider, credential_id)

    def session_credential_id(self, provider: str, session_id: str | None) -> int | None:
        local = self._local.session_credential_id(provider, session_id)
        if local is not None:
            return local
        recorded = self._brokered.get((key_for(provider=provider), session_id or ""))
        if recorded is not None:
            return recorded
        return self._synthetic_for_provider(provider)

    def pin_session_credential(
        self, provider: str, session_id: str | None, credential_id: int
    ) -> None:
        """Pin a brokered credential for the life of THIS PROCESS only.

        ``AuthStore.pin_session_credential`` writes an IN-MEMORY sticky pointer, so a
        synthetic id can live there harmlessly — and it must, because the session path
        uses sticky resolution to keep a turn on the account it is already
        transacting with. There is nothing durable to write and nothing to un-write: the
        mapping is (provider, session) → synthetic id, and the synthetic id is
        recomputed from the placement on the next turn.
        """
        if is_synthetic_credential_id(credential_id):
            pair = self._brokered_pair(credential_id)
            if pair is not None:
                self._brokered[(key_for(provider=provider), session_id or "")] = credential_id
            return
        self._local.pin_session_credential(provider, session_id, credential_id)

    def release_session_credential(self, provider: str, session_id: str | None) -> None:
        self._local.release_session_credential(provider, session_id)
        self._brokered.pop((key_for(provider=provider), session_id or ""), None)

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[Any]:
        """Local rows, PLUS a descriptor for each key this device may borrow.

        The synthetic rows are what make the "why is there no bearer" surface able to
        answer at all: without them a borrowed provider looks like a provider the
        device has never heard of, which is the state this slice exists to remove.
        They carry ``identity_key=None`` and no ``data`` (see ``get_credential``).
        """
        rows = list(self._local.list_credentials(provider, include_disabled))
        if self._mesh is None or self._mesh.placement is None:
            return rows
        from local_operator.providers.auth_store import StoredCredential

        for key in self._mesh.placement.borrowable_keys(self._mesh.self_device):
            entry = self._mesh.placement.entry(key)
            if entry is None:
                continue
            label = mcp_provider_label(key) if is_mcp_key(key) else key
            if provider is not None and label != provider:
                continue
            rows.append(
                StoredCredential(
                    id=synthetic_credential_id(key, entry.owner_device),
                    provider=label,
                    credential_type="oauth",
                    data={},
                    disabled_cause=None,
                    identity_key=None,
                )
            )
        return rows

    async def list_oauth_accesses(self, provider: str) -> list[Any]:
        """LOCAL accesses only, deliberately.

        A borrowed bearer is not a local access: ``list_oauth_accesses`` feeds the
        usage panel, and a usage fetch is a request the OWNER already makes and
        reports. Listing the borrow here would make this device spend the owner's
        quota a second time to learn a number the owner publishes, and it would
        report a token with no local row behind it as though one existed.
        """
        return list(await self._local.list_oauth_accesses(provider))

    async def ensure_oauth_fresh(self, credential_id: int) -> dict[str, Any] | None:
        """Never refreshes a borrow, because a borrow has no refresh token.

        ``None`` is the same answer the local store gives for a row it does not have,
        and it is the truthful one: this device cannot make a borrowed bearer
        fresher. The owner is asked again on the next borrow instead, and its own
        grant TTL bounds how stale this device's copy can be.
        """
        if is_synthetic_credential_id(credential_id):
            return None
        return await self._local.ensure_oauth_fresh(credential_id)

    async def ensure_oauth_fresh_or_raise(self, credential_id: int) -> dict[str, Any] | None:
        if is_synthetic_credential_id(credential_id):
            return None
        return await self._local.ensure_oauth_fresh_or_raise(credential_id)

    def grant_is_dead(self, credential_id: int) -> bool:
        if is_synthetic_credential_id(credential_id):
            return False
        return bool(self._local.grant_is_dead(credential_id))

    def send_unconfirmed(self, credential_id: int, refresh_token: str | None = None) -> bool:
        if is_synthetic_credential_id(credential_id):
            return False
        return bool(self._local.send_unconfirmed(credential_id, refresh_token))

    def delete_credentials_for_provider(
        self, provider: str, disabled_cause: str = "logged-out"
    ) -> int:
        """Local rows only, and that is the point.

        ``/logout`` on this device must not be able to delete the OWNER's row. A
        shared credential is un-shared by its owner (``credential revoke``), and a
        borrower's logout drops its cached bearer without touching the login it was
        borrowing.

        THE SIGNATURE IS ``AuthStore``'s, keyword included: both callers
        (``ProviderController.logout`` and ``auth_cli``) pass ``disabled_cause``, and
        the first version's narrower signature raised ``TypeError`` there (review
        round 1, F5).
        """
        self._drop_borrowed(provider)
        return int(
            self._local.delete_credentials_for_provider(provider, disabled_cause=disabled_cause)
        )

    def _drop_borrowed(self, provider: str) -> None:
        """Drop this device's cached borrows for ``provider``.

        ``self._brokered`` is keyed by ``(key, session_id)`` PAIRS, not by key: the
        first version of this loop iterated the dict and called ``startswith`` on what
        it got, which is a ``tuple`` — an ``AttributeError`` on a cold path nobody
        exercises until a logout, which is the kind of defect a type checker finds and
        a happy-path test does not. So the key is taken from the pair.
        """
        if self._mesh is None:
            return
        wanted = key_for(provider=provider)
        keys = {wanted}
        keys.update(pair[0] for pair in self._brokered if pair[0] == wanted)
        for key in keys:
            self._mesh.grants.drop(key)
        for pair in [p for p in self._brokered if p[0] == wanted]:
            self._brokered.pop(pair, None)

    # -- pass-through -------------------------------------------------------

    @property
    def mesh_client(self) -> CredentialSource | None:
        """The broker client when this device borrows, else ``None``.

        The seam MCP reads (``mcp/manager.py``'s ``_build_oauth_auth``): the presence
        of this client IS the statement that the device may broker, so a manager that
        finds it ``None`` takes exactly the path it took before this slice existed.
        Public rather than a private reach-through because it is the one thing that
        decides which of the two MCP auth shapes is built.
        """
        return self._mesh

    @property
    def db_path(self) -> Path:
        return self._local.db_path

    @property
    def local(self) -> Any:
        """The wrapped store, for a caller that must know which side answered."""
        return self._local

    def close(self) -> None:
        self._mesh_close()
        self._local.close()

    def _mesh_close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()

    def __getattr__(self, name: str) -> Any:
        """Everything else goes to the local store.

        DOCUMENTED RATHER THAN IMPLICIT, because it is load-bearing: it is what makes
        the wrapper a structural superset of ``AuthStore`` for code written before or
        after this slice, so a method added to the store cannot make the mesh-aware
        path raise ``AttributeError`` on the first turn that needs it. The methods
        with brokered behaviour are all defined ABOVE, so a call only reaches here
        when the answer is genuinely the local one.
        """
        local = self.__dict__.get("_local")
        if local is None:  # pragma: no cover - during unpickling/copy, never in a turn
            raise AttributeError(name)
        return getattr(local, name)


def report_kind_for(error: BaseException) -> str:
    """What a borrower tells the owner a provider said. The FAILOVER DRIVER's classes.

    The same predicates ``AuthStore.rotate_sibling`` branches on, in the same order,
    so the owner hears what its own rotation would have concluded. The first version
    sent ``quota`` for EVERY non-invalidation error, so a provider 529 overload
    arrived as quota exhaustion and blocked the owner's login for a fault no
    credential caused (review round 1, F3).

    * ``invalid`` — an explicit revocation (``invalid_grant``, ``token_revoked``);
    * ``unavailable`` — the provider failed (5xx/529, timeout): never a block;
    * ``quota`` — a usage limit (429 / an exhausted-quota body);
    * ``unauthorized`` — any other auth refusal: the owner may refresh, once;
    * ``failed`` — anything else: audited on the owner and changes nothing.
    """
    from local_operator.providers.failover import (
        classify_provider_error,
        is_invalidated_credential_error,
        is_server_side_failure,
        is_usage_limit_error,
    )

    if is_invalidated_credential_error(error):
        return "invalid"
    if is_server_side_failure(error):
        return "unavailable"
    if is_usage_limit_error(error):
        return "quota"
    if classify_provider_error(error) == "auth":
        return "unauthorized"
    return "failed"


def mcp_provider_label(key: str) -> str:
    """The provider string a synthetic ``mcp:<url>`` credential reports.

    ``mcp-oauth`` is the provider the real MCP rows use
    (``mcp/auth.py MCP_OAUTH_PROVIDER``), so a borrowed MCP credential reports the
    same provider as a local one and the session path's provider matching keeps
    working.
    """
    return "mcp-oauth"

"""The owner's half: refresh here, hand back a bearer, and nothing else.

THE SAFETY ARGUMENT IS STRUCTURAL (design §0, R14): the refresh never moves.
:class:`MeshCredentialBroker` calls the owner's own
``AuthStore.get_oauth_access(..., read_only=True)``, whose chain is ``_resolve`` →
``_ensure_oauth_fresh`` — so every defence that already protects the owner's own
sessions applies unchanged:

* the per-process ``asyncio.Lock`` (``_refresh_lock_for``),
* the cross-process SQLite lease (``_try_refresh_lease`` / ``AUTH_REFRESH_LEASE_MS``),
* the write-ahead send marker (``send_unconfirmed``),
* the dead-grant tombstone (``CredentialInvalidError``).

That is why ONE long-lived event loop is not a nicety. ``asyncio.Lock`` binds to
the loop that first uses it, so a handler that called ``asyncio.run`` per request
(build plan §7 unsafe item 5, and the shape ``relay.py`` itself uses elsewhere for
short work) would hand the store a DIFFERENT lock on the second request: two
peers asking at once would then each take their own lock, both would POST the same
rotating refresh token, and the loser's token is dead. The loop is created once
here and every request is submitted to it with
:meth:`_BrokerLoop.submit`.

WHO IS ASKING IS THE TRANSPORT'S ANSWER, NEVER THE FRAME'S (review round 1, F1).
A ``PeerLink`` exists only after a handshake that proved the far end holds the
private key its member row names, so ``link.device_id`` is authenticated; every
field inside a frame is whatever the sender chose to write. The first version took
``frame["from_device"]`` as the caller, and a stranger that named the owner's own
id was served, allowed a forced refresh, and rewrote the owner's holder list on
disk. So :meth:`MeshCredentialBroker._caller` reads the link, treats
``from_device`` as an ASSERTION that must agree with it, and refuses a mismatch as
``identity_mismatch``. The same rule covers the credential itself: the login
served is the one the OWNER'S placement entry names, never the frame's
``provider`` field, or a holder of one key could be served another.

THE SHARING LIST IS READ FROM DISK ON EVERY REQUEST (F2). The CLI's ``revoke`` runs
in another process, so a copy loaded at relay start kept serving a revoked device
until the relay restarted, and saving that copy later wrote the revoked device
back. The broker holds no document; :meth:`MeshCredentialBroker._document` loads
one per request, and a peer's document reaches disk only through
``placement.merge_from_peer`` (load, merge and save under the lock). Revocation
latency is therefore the life of a grant ALREADY lent, bounded by ``grant_ttl_s``.

WHAT A REPORT FROM A PEER MAY DO — and what it must never do (finding 8). A peer's
401 arrives as ``credential_report``, and the tempting implementation is to run
the owner's ``rotate_sibling`` on it. That would be a catastrophic default:
``rotate_sibling`` calls ``disable_credential(failing.id, cause="invalidated-token")``
on an invalidated-token error, so ONE bad 401 observed by ONE borrower would
soft-delete the operator's login on EVERY device — exactly the failure the
requirement exists to prevent. So a report may, at most:

* record a MODEL-FAMILY-scoped quota block for a ``quota`` failure, through the
  owner's own ``block_credential`` — never account-wide, never longer than
  ``REMOTE_QUOTA_BLOCK_MAX_MS`` whatever the peer claims, and at most once per
  holder per credential per ``REPORT_BLOCK_MIN_INTERVAL_S`` (F3: one account-wide
  report with a huge retry time locked the owner out of its own login for an hour);
* provoke AT MOST ONE coalesced owner-side refresh per credential per
  ``REPORT_REFRESH_MIN_INTERVAL_S``, through ``_ensure_oauth_fresh``, so a stale
  borrower can ask the owner to try — and the owner decides;
* and nothing else. ``rotate_sibling``, ``disable_credential`` and
  ``delete_credential`` are never reachable from this module, and
  ``tests/unit/network/test_credentials_owner.py`` proves it with a fixture in which
  a ``rotate_sibling`` WOULD find and disable the row.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from pathlib import Path
from typing import Any

from local_operator.network.credentials import github as github_app
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.messages import render_repair_notice
from local_operator.network.credentials.placement import PlacementDocument
from local_operator.network.credentials.types import (
    DEVICE_BOUND_PROVIDERS,
    BrokerError,
    CredentialPlacementEntry,
    CredentialRef,
    Grant,
    GrantScope,
    device_bound_refusal,
    is_mcp_key,
    mcp_url_from_key,
    peer_int,
    synthetic_credential_id,
)

#: How long a second asker joins an in-flight resolve for the same key. Short on
#: purpose: it exists to catch the CONCURRENT case (several provider calls in one
#: turn, or two peers asking at the same instant, arriving within a scheduler
#: tick), not to make a caller wait for a slow one. A joiner whose window expires
#: resolves itself, and the store's own per-credential lock is what then keeps the
#: two from POSTing twice.
JOIN_WINDOW_S = 0.25

#: How long the handler waits for the loop before answering "did not finish".
#: BELOW the deadline registered for ``net_broker`` in
#: ``network/credentials/__init__.py`` (``BROKER_OP_DEADLINE_S``, 75 s): the relay's
#: own deadline would otherwise fire first and the peer would get the relay's
#: generic sentence instead of the broker's specific one.
HANDLER_WAIT_MARGIN_S = 20.0

#: The floor between two owner-side refreshes provoked by a PEER's report, per
#: credential. The design's number (§2.3: "at most one coalesced owner-side refresh
#: per credential per 5 min"). A report is the cheapest way for a borrower to make
#: the owner spend a POST, so it is the one path that needs its own rate limit
#: rather than relying on the token's freshness.
REPORT_REFRESH_MIN_INTERVAL_S = 300.0

#: The longest block a PEER'S quota report may write on the owner's row, whatever
#: ``retry_after_ms`` it claims. The owner's own default backoff for a rate limit
#: (``auth_store.DEFAULT_BLOCK_MS``, restated so this module stays light): a remote
#: report is second-hand evidence, so it buys the shortest block the owner would
#: have written on its own evidence, and the owner's own usage probe decides anything
#: longer. Pinned against the store's constant by a test.
REMOTE_QUOTA_BLOCK_MAX_MS = 60_000

#: At most one block per holder per credential per this window. With the cap above
#: this bounds what a hostile or broken holder can do to one model family on the
#: owner to 60 s in every 300 s, rather than "blocked for as long as it keeps
#: reporting".
REPORT_BLOCK_MIN_INTERVAL_S = 300.0

#: The number of worker threads the broker's loop uses for blocking sections.
#: Small: the only blocking work is the file-lock acquire inside the MCP refresh.
LOOP_EXECUTOR_THREADS = 2


class _BrokerLoop:
    """ONE long-lived event loop and its thread, for every brokered request.

    See this module's docstring for why this is a correctness requirement rather
    than an optimisation. The loop is created on first use (a relay that never
    brokers anything pays nothing) and is a DAEMON thread, so a process that exits
    never waits on it.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop
            ready = threading.Event()

            def run() -> None:
                import concurrent.futures

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # The store's refresh path is `await`-only, but the MCP half takes a
                # FILE lock, and a file lock acquire inside the loop thread would
                # stall every other broker request behind it. The executor is how
                # that blocking section leaves the loop without a second loop.
                loop.set_default_executor(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=LOOP_EXECUTOR_THREADS, thread_name_prefix="mesh-broker-io"
                    )
                )
                self._loop = loop
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(
                target=run, name=f"mesh-broker-loop-{self.name}", daemon=True
            )
            self._thread.start()
            ready.wait(10.0)
            if self._loop is None:  # pragma: no cover - a thread that never started
                raise RuntimeError("the credential broker's event loop did not start")
            return self._loop

    def submit(self, coro: Any, *, timeout: float) -> Any:
        """Run ``coro`` on the loop and block the CALLER's thread for its result.

        Blocking here is deliberate and bounded: the caller is a relay slow-op
        worker (never a link reader — ``install`` registers ``net_broker`` as SLOW),
        so waiting occupies one bounded slot rather than the link.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop())
        try:
            return future.result(timeout)
        except FutureTimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        with self._lock:
            loop = self._loop
            self._loop = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)


class MeshCredentialBroker:
    """Serves ``net_broker`` on the owning device."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        self_device: str = "",
        self_device_name: str = "",
        network_id: str = "",
        audit: Any = None,
        auth_store: Any = None,
        client: MeshCredentialClient | None = None,
        identity: Any = None,
        github_minter: Any = None,
    ) -> None:
        # NO ``placement`` ATTRIBUTE, on purpose (F2): a document held here is a copy
        # that a revoke in the CLI process never reaches. See :meth:`_document`.
        self.root = root
        self.self_device = self_device
        self.self_device_name = self_device_name
        self.network_id = network_id
        self.audit = audit
        self.identity = identity
        self._auth_store = auth_store
        self._client = client
        #: The github adapter's minter seam (a test injects a fake transport here)
        #: and its one lender, built on the first mint — a relay that never lends
        #: github pays nothing (see :meth:`_lender`).
        self._github_minter = github_minter
        self._github_lender: github_app.GithubLender | None = None
        self._loop = _BrokerLoop(self_device[-8:] or "owner")
        self._inflight: dict[tuple[str, str, bool], asyncio.Future[Any]] = {}
        self._report_refreshed: dict[int, float] = {}
        self._report_blocked: dict[tuple[str, int], float] = {}
        #: ONE lock around each report arm's read-then-stamp (review round 4, R4-n2):
        #: two reports racing the same window both read "not stamped", both answered
        #: ``refreshed`` and both audited a refresh, while the store coalesced them
        #: into ONE token POST — an acknowledgement and a record the owner did not earn.
        self._report_lock = threading.Lock()
        self._op_deadline_s = _broker_deadline_s()

    # -- construction -------------------------------------------------------

    @classmethod
    def relay_handler(cls, server: Any) -> Any:
        """The ``net_broker`` handler a relay registers: builds the broker ON DEMAND.

        WHY NOT BUILD IT AT RELAY START, which the first version did: a relay started
        before this device's first ``credential share`` found nothing to lend, got no
        broker, and refused every borrow until it was restarted (review round 1, F2).
        So the decision "do I own anything to lend" is taken PER REQUEST, from the
        document on disk; while the answer is no, the refusal is the same by-name
        ``not_implemented`` sentence a relay with no broker always gave, so the 0-peer
        behaviour is unchanged. The broker (and its one event loop) is built the first
        time the answer is yes, and kept.
        """
        from local_operator.network.relay import not_implemented_peer_op

        refuse = not_implemented_peer_op("net_broker")

        def _handle(link: Any, frame: dict[str, Any]) -> dict[str, Any] | None:
            # ONE broker per relay, shared with the ``credential_revoke`` local op:
            # the revoke op must reach the same mint registry this handler mints
            # into (``broker_for_relay``).
            broker = broker_for_relay(server)
            if broker is None:
                return refuse(link, frame)
            return broker.on_broker(link, frame)

        return _handle

    @classmethod
    def for_relay(cls, server: Any) -> MeshCredentialBroker | None:
        """The broker a relay serves, or ``None`` when it owns nothing to lend.

        ``None`` is the whole of the 0-peer guarantee on the owning side as well:
        a device that owns no credential has nothing to lend, so it registers no
        handler — and its ``net_broker`` then answers with the same by-name refusal
        it did before this slice existed.
        """
        root = getattr(server, "root", None)
        identity = getattr(server, "identity", None)
        self_device = str(getattr(identity, "device_id", ""))
        placement = PlacementDocument.resolve(root, self_device=self_device)
        if placement is None:
            return None
        owned = placement.keys_owned_by(self_device)
        if not owned:
            return None
        return cls(
            root=root,
            self_device=self_device,
            self_device_name=str(getattr(identity, "name", "")),
            network_id=placement.network_id,
            audit=getattr(server, "audit", None),
            identity=identity,
            client=MeshCredentialClient.for_relay(server),
        )

    # -- the relay's peer handler ------------------------------------------

    def on_broker(self, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        """One ``net_broker`` request. Runs on a slow-op worker, never a reader.

        REGISTERED AS SLOW so the link keeps reading while a provider refresh runs
        (P0's off-reader dispatch, build plan §0 finding 4). This handler issues NO
        request over the link it is serving — it answers — so the own-link refusal
        P0 added is unreachable from here by construction.
        """
        req = frame.get("req")
        if req is None:
            # A FRAME WITH NO ``req`` CAN NEVER BE ANSWERED, so it is refused BEFORE
            # any work (QA round 1, Q2). The sender's ``PeerLink.request`` registers no
            # waiter for such a frame, and the reply is dropped as a stray on arrival.
            # Serving it anyway is what made every failed borrow on the first build
            # cost the owner a real token POST — refreshed, audited as a grant, and
            # thrown away. Refused by name through the transport's own refusal, never
            # an ack: there is nobody to read either, and an ack would look served.
            from local_operator.network.types import MeshRefusal

            raise MeshRefusal(
                "protocol_error",
                "a net_broker request must carry a req so its answer can be matched; "
                "nothing was refreshed or lent",
            )
        kind = str(frame.get("kind") or "")
        if kind == "grant":
            detail = self.grant(link, frame)
        elif kind == "report":
            detail = self.report(link, frame)
        elif kind == "placement":
            detail = self.placement_frame(link, frame)
        else:
            detail = BrokerError(
                code="internal",
                message=f"the broker does not know the kind {kind!r}",
            ).to_detail()
        return {"op": "ack", "req": req, "detail": detail}

    # -- grant --------------------------------------------------------------

    def grant(self, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        """Authorise, coalesce, resolve, audit, reply — in that order (§3.6c)."""
        key = str(frame.get("key") or "")
        # A LABEL until the entry is read: the credential served is the one the
        # owner's own placement entry names (step 1), never this field.
        provider = str(frame.get("provider") or key)
        for_session = str(frame.get("for_session") or "")
        model_id = str(frame.get("model_id") or "")
        force = bool(frame.get("force_refresh"))
        started = time.monotonic()
        by, refused = self._caller(link, frame, key, provider)
        if refused is not None:
            return refused

        # 0. A device-bound provider is refused BY NAME before anything else: the
        #    answer does not depend on placement, and refusing it late would mean a
        #    document written before this rule could still broker it. (Checked again
        #    against the entry's own provider below, which is the one served.)
        if provider in DEVICE_BOUND_PROVIDERS or key in DEVICE_BOUND_PROVIDERS:
            return self._refuse(
                link, key, provider, by, "device_bound", device_bound_refusal(provider)
            )

        # 1. The half of authorisation the transport does not know about. The
        #    network half — membership, epoch, locality and the `broker_credential`
        #    capability — was already decided at the chokepoint (types.OP_CAPABILITY
        #    maps net_broker to broker_credential, and P0's dispatcher is the only
        #    caller). Re-implementing it here would be a second authoriser.
        entry = self._entry(key)
        if entry is None or entry.owner_device != self.self_device:
            return self._refuse(
                link,
                key,
                provider,
                by,
                "not_owner",
                f"{self.self_device_name or self.self_device} does not hold {provider!r}; "
                "ask the device that signed in to it",
            )
        # The OWNER'S word for which login this key is. Resolving the frame's
        # ``provider`` instead let a holder of one key name another of the owner's
        # logins and be served it (found auditing F1's class).
        provider = entry.provider or provider
        if provider in DEVICE_BOUND_PROVIDERS:
            return self._refuse(
                link, key, provider, by, "device_bound", device_bound_refusal(provider)
            )
        if not entry.is_holder(by):
            return self._refuse(
                link,
                key,
                provider,
                by,
                "not_a_holder",
                f"{self.self_device_name or self.self_device} does not share {provider!r} "
                f"with {by!r}",
            )

        # 2. Forced refreshes are refused for anyone but an admin of THIS network.
        #    They are the cheap way to provoke an IdP's refresh-token reuse
        #    detection, which revokes the whole family — the exact loss the design
        #    exists to prevent (cut line unsafe item 3).
        holder = entry.holder(by)
        if force and not self._is_admin(by):
            return self._refuse(
                link,
                key,
                provider,
                by,
                "not_authorised",
                "a forced credential refresh may only be asked for by an admin device",
            )

        # 2b. GITHUB IS DEVICE-SCOPED BY CONSTRUCTION (design F3). A row that says
        #     ``session`` for this key enforces nothing (the borrower has no
        #     rail-authenticated session identity) and would read as a bound that
        #     does not exist — so it is refused BY NAME before any mint work,
        #     whatever wrote it. The share verb refuses to write such a row at all.
        if github_app.is_github_key(key) and (holder is None or holder.scope != "device"):
            return self._refuse(
                link,
                key,
                provider,
                by,
                github_app.CODE_DEVICE_SCOPE,
                "this device lends GitHub to the DEVICE, not to a session; nothing was lent",
            )

        # 3-5. Coalesce, resolve, audit. All on the owner's ONE loop.
        try:
            outcome = self._loop.submit(
                self._coalesced_resolve(
                    key=key,
                    provider=provider,
                    for_session=for_session,
                    model_id=model_id,
                    force=force,
                    holder_scope=holder.scope if holder is not None else "session",
                    by=by,
                ),
                timeout=max(1.0, self._op_deadline_s - HANDLER_WAIT_MARGIN_S),
            )
        except (FutureTimeoutError, TimeoutError):
            return self._refuse(
                link,
                key,
                provider,
                by,
                "rate_limited",
                "this device did not finish refreshing the credential in time; nothing was lent",
                retry_after_ms=30_000,
            )
        except Exception as exc:  # noqa: BLE001 — a broker bug must answer, not close the link
            return self._refuse(link, key, provider, by, "internal", exc.__class__.__name__)
        if isinstance(outcome, BrokerError):
            return self._refuse(
                link,
                key,
                provider,
                by,
                outcome.code,
                outcome.message,
                retry_after_ms=outcome.retry_after_ms,
            )
        grant: Grant = outcome
        grant.latency_ms = int((time.monotonic() - started) * 1000)
        self._audit(
            "credential.grant",
            actor=self.self_device,
            subject=by,
            session_id=for_session,
            detail={
                "credential_key": key,
                "act": self.self_device,
                "sub": by,
                "grant_id": grant.grant_id,
                "credential_kind": grant.credential_ref.kind,
                "scope": grant.scope.kind,
                "refreshed": grant.refreshed,
                "latency_ms": grant.latency_ms,
            },
        )
        return grant.to_detail()

    async def _coalesced_resolve(self, **fields: Any) -> Any:
        """Share one resolve between askers that arrive together (§3.4).

        The LEADER runs it; a joiner waits ``JOIN_WINDOW_S`` for the leader's
        answer and resolves itself if that expires. Identity of the shared ask is
        ``(key, for_session, force)``: a joiner must never receive an answer to a
        different question, and two sessions asking are asking for two different
        stickiness decisions.
        """
        ckey = (fields["key"], fields["for_session"], bool(fields["force"]))
        loop = asyncio.get_running_loop()
        slot = self._inflight.get(ckey)
        if slot is not None:
            try:
                return await asyncio.wait_for(asyncio.shield(slot), JOIN_WINDOW_S)
            except Exception:  # noqa: BLE001 — the leader's answer is not mine to adopt
                return await self._resolve_once(**fields)
        slot = loop.create_future()
        self._inflight[ckey] = slot
        try:
            value = await self._resolve_once(**fields)
            with suppress(Exception):
                slot.set_result(value)
            return value
        except BaseException as exc:
            with suppress(Exception):
                slot.set_exception(exc)
            raise
        finally:
            if self._inflight.get(ckey) is slot:
                del self._inflight[ckey]
        # The slot is dropped once the answer is published: a joiner that read it
        # before then still receives that answer, and one that arrives afterwards
        # becomes the next leader and resolves for itself. The dict therefore cannot
        # grow with the number of sessions a long-lived relay has ever seen.

    async def _resolve_once(
        self,
        *,
        key: str,
        provider: str,
        for_session: str,
        model_id: str,
        force: bool,
        holder_scope: str,
        by: str,
    ) -> Any:
        """The owner's own read-only resolve. Returns a :class:`Grant` or a refusal."""
        if github_app.is_github_key(key):
            return await self._resolve_github(
                key=key,
                provider=provider,
                for_session=for_session,
                holder_scope=holder_scope,
                by=by,
            )
        if is_mcp_key(key):
            return await self._resolve_mcp(
                key=key,
                provider=provider,
                for_session=for_session,
                force=force,
                holder_scope=holder_scope,
                by=by,
            )
        store = self._auth_store_instance()
        before = self._row_stamps(provider)
        try:
            access = await store.get_oauth_access(
                provider,
                for_session or None,
                force_refresh=force,
                # READ-ONLY ON THE OWNER (§2.1). A peer's request never writes
                # stickiness and never blocks one of the owner's rows: a lender that
                # let a borrower re-point the owner's own account would be a
                # delegation that is not narrower than the delegator's, which is the
                # rule §3.3 states. A successful refresh still persists the rotated
                # token — that is the same account's own bookkeeping, and dropping it
                # would throw away a single-use refresh token.
                read_only=True,
                model_id=model_id,
            )
        except Exception as exc:  # noqa: BLE001 — classified below, never propagated
            return self._classify_refresh_failure(exc, key)
        if access is None:
            # THE EMPTY ANSWER IS SPLIT BY ITS CAUSE (audit Q5 #9). ``quota_blocked``
            # had no producer: the owner's cascade excludes blocked rows
            # (``is_blocked_for_model``), so a peer asking for a family whose cap is
            # spent received ``no_local_credential`` — "nothing usable is signed in
            # here", cached for 300 s with a re-login remedy — for a rate limit that
            # heals on its own. When EVERY row this provider has is blocked for the
            # requested model, that block is the reason, and the remainder travels
            # with the code so the borrower waits the real time instead of guessing.
            blocked_ms = self._quota_remaining_ms(provider, model_id)
            if blocked_ms:
                return BrokerError(
                    code="quota_blocked",
                    key=key,
                    owner_device=self.self_device,
                    owner_device_name=self.self_device_name,
                    retry_after_ms=blocked_ms,
                    message=(
                        f"every login for {provider!r} is rate-limited on this device "
                        f"for another {max(1, blocked_ms // 1000)} s"
                    ),
                )
            return BrokerError(
                code="no_local_credential",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=f"nothing usable is signed in for {provider!r} on this device",
            )
        if not getattr(access, "access_token", None):
            return BrokerError(
                code="no_local_credential",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=f"nothing usable is signed in for {provider!r} on this device",
            )
        raw = access.raw if isinstance(getattr(access, "raw", None), dict) else {}
        token_exp_ms = int(raw.get("expires") or 0)
        # ``refreshed`` AN OBSERVATION, NOT AN ATTRIBUTION: the owner's row for the
        # winning credential was rewritten while THIS request was being served, which
        # means the bearer being handed back was minted during the request rather than
        # read from storage. The store stamps ``updated_at`` on every write, and a
        # successful refresh persists the rotated token, so a moved stamp is exactly
        # that fact. Read before and after on purpose — the alternative, asking the
        # store "did YOU refresh", is not a question it answers, and a caller that
        # guesses would report a refresh the owner never did.
        #
        # Under CONCURRENCY two askers that overlapped the same exchange both see the
        # moved stamp, and both are right: each is holding the freshly minted bearer.
        # That is why no caller may treat this flag as "I caused a POST".
        refreshed = self._row_stamps(provider).get(int(access.credential_id), 0) > before.get(
            int(access.credential_id), 0
        )
        return self._mint_grant(
            key=key,
            provider=provider,
            token=str(access.access_token),
            credential_id=int(access.credential_id),
            kind=str(getattr(access, "kind", "oauth") or "oauth"),
            token_exp_ms=token_exp_ms,
            refreshed=refreshed,
            holder_scope=holder_scope,
            for_session=for_session,
            identity={
                name: str(value)
                for name, value in (
                    ("account_id", getattr(access, "account_id", None)),
                    ("email", getattr(access, "email", None)),
                    ("org_id", getattr(access, "org_id", None)),
                )
                if value
            },
        )

    async def _resolve_mcp(
        self,
        *,
        key: str,
        provider: str,
        for_session: str,
        force: bool,
        holder_scope: str,
        by: str,
    ) -> Any:
        """Serve an MCP server's ACCESS TOKEN, refreshed by the owner's own path.

        MCP is class 5 in the design's inventory: the ACCESS token may be brokered,
        the GRANT may not, because a new grant can only be created by an interactive
        loopback callback on this device. The refresh is the owner's own
        ``ensure_mcp_oauth_fresh`` — running here, on the owner — which takes the
        existing cross-process file lock (``_oauth_refresh_lock``) and short-circuits
        before any POST on a grant the server already rejected
        (``GRANT_DEAD_AT_KEY``). The borrower never builds an ``OAuthClientProvider``
        and never binds the callback port.
        """
        del force  # a forced MCP refresh is not offered in v1; the deadline bounds it
        url = mcp_url_from_key(key)
        try:
            from local_operator.mcp.auth import (
                MCP_OAUTH_PROVIDER,
                McpTokenStorage,
                ensure_mcp_oauth_fresh,
            )
            from local_operator.mcp.config import MCPHttpServerConfig
        except Exception as exc:  # noqa: BLE001 — reported as a refusal, not a crash
            return BrokerError(
                code="internal",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=f"the MCP store is unavailable here ({exc.__class__.__name__})",
            )
        store = self._auth_store_instance()
        storage = McpTokenStorage(url, store)
        # The ROW's integer id. ``storage.credential_id`` is NOT one: it is the logical
        # string ``mcp_oauth:<url>``, and ``int()`` of it made every MCP grant answer
        # ``internal ValueError`` (review round 1, F4).
        row_id = self._mcp_row_id(url)
        if row_id is None:
            return BrokerError(
                code="no_local_credential",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=f"nothing is signed in to {url!r} on this device",
            )
        # ``refreshed`` IS OBSERVED, NOT ASSUMED (audit Q5 #8). The provider path
        # watches the row's ``updated_at`` move across the resolve, and MCP rows are
        # written through the same ``upsert_credential``; this call was a hard-coded
        # ``False``, so every borrower and every audit record was told an MCP grant
        # never refreshed, whatever ``ensure_mcp_oauth_fresh`` had just done.
        #
        # The same read the provider path makes, read before the refresh runs.
        before = self._row_stamps(MCP_OAUTH_PROVIDER)
        try:
            await ensure_mcp_oauth_fresh(url, MCPHttpServerConfig(url=url), store)
        except Exception as exc:  # noqa: BLE001 — a refresh failure is a refusal
            return self._classify_refresh_failure(exc, key)

        refreshed = self._row_stamps(MCP_OAUTH_PROVIDER).get(row_id, 0) > before.get(row_id, 0)
        try:
            tokens = await storage.get_tokens()
        except Exception as exc:  # noqa: BLE001
            return BrokerError(
                code="internal",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=exc.__class__.__name__,
            )
        token = str(getattr(tokens, "access_token", "") or "")
        expiry = storage.stored_token_expiry()
        if not token or _already_expired(expiry):
            # The row exists but holds no USABLE access token: either it has none, or
            # the token it has died and the refresh above could not replace it (no
            # reachable authorization server, a grant the server already rejected).
            # Only an interactive login can produce a live one, and that login can
            # only happen HERE. The peer is told `interactive_required`; the OWNER
            # gets no pushed notice — the mechanism is the `credential.report` row
            # written below, which `credentials/repair.py` derives into the repair
            # row that `lop network doctor` and the `/network` panel show here.
            # Never the other way round, because the borrower has no browser flow to
            # offer.
            #
            # WHY AN EXPIRED TOKEN IS A REFUSAL AND NOT A GRANT (audit round 2): the
            # owner can see the expiry it is about to lend, and serving it would hand
            # the peer a bearer its own record calls dead — a 401 at the MCP server
            # with no repair path for the operator, which is exactly the state §4.7's
            # `interactive_required` row exists to prevent. Measured on the rig with a
            # genuinely expired row (`tokens_obtained_at` two hours back, `expires_in`
            # 3600): before this, the peer received the dead bearer and read a bare 401.
            self._audit(
                "credential.report",
                actor=self.self_device,
                subject=by,
                session_id=for_session,
                detail={
                    "credential_key": key,
                    "act": self.self_device,
                    "sub": by,
                    "failure": "interactive_required",
                },
            )
            return BrokerError(
                code="interactive_required",
                key=key,
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                message=render_repair_notice(self.self_device_name or self.self_device, key),
            )
        return self._mint_grant(
            key=key,
            provider=provider,
            token=token,
            credential_id=row_id,
            kind="mcp-oauth",
            token_exp_ms=int(expiry * 1000) if expiry else 0,
            refreshed=refreshed,
            holder_scope=holder_scope,
            for_session=for_session,
            identity={},
        )

    def _mint_grant(
        self,
        *,
        key: str,
        provider: str,
        token: str,
        credential_id: int,
        kind: str,
        token_exp_ms: int,
        refreshed: bool,
        holder_scope: str,
        for_session: str,
        identity: dict[str, str],
    ) -> Grant:
        """Build the grant, applying §3.3's narrowing rule to the expiry.

        ``min(token_expiry, now + grant_ttl_s)``. The borrower's copy never outlives
        the owner's token, and never outlives the configured TTL — which is what
        bounds how long a peer removed from ``holders`` still holds a live bearer
        (design §3.7: revocation latency is bounded, not zero, and the guide says so
        rather than glossing it).
        """
        now_ms = int(time.time() * 1000)
        ttl_ms = int(_grant_ttl_s(self.root) * 1000)
        ceiling = now_ms + ttl_ms
        grant_exp = min(token_exp_ms, ceiling) if token_exp_ms else ceiling
        return Grant(
            access_token=token,
            kind="bearer" if kind in ("oauth", "mcp-oauth") else "api_key",
            token_expires_at_ms=token_exp_ms,
            grant_expires_at_ms=grant_exp,
            credential_ref=CredentialRef(
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                provider=provider,
                kind=kind,
                credential_id=credential_id,
            ),
            served_by=self.self_device,
            refreshed=refreshed,
            scope=GrantScope(
                kind="device" if holder_scope == "device" else "session",
                session_id="" if holder_scope == "device" else for_session,
            ),
            identity=identity,
            grant_id=f"g_{os.urandom(8).hex()}",
        )

    # -- github: App installation tokens ------------------------------------

    def _lender(self) -> github_app.GithubLender:
        """The broker's ONE github lender (minter + revoke registry), built lazily.

        Built on the first mint so a relay that never lends github pays nothing;
        the event loop it is handed is the broker's own — the revoker schedules
        there and its DELETE calls run in the loop's executor, off the loop.
        """
        if self._github_lender is None:
            minter = self._github_minter
            if minter is None:
                minter = github_app.GithubMinter()
            self._github_lender = github_app.GithubLender(minter=minter)
            self._github_lender.attach_loop(self._loop.loop())
            self._github_lender.attach_audit(self._github_audit_sink)
        return self._github_lender

    def _github_audit_sink(
        self, cause: str, key: str, holder: str, revoked: int, deferred: int
    ) -> None:
        """One ``credential.revoke`` row per revoke batch, act/sub included.

        ``cause`` separates the three paths an incident reader must tell apart:
        ``grant_expired`` (the scheduled window-end revoke), ``revoked`` (the
        operator's immediate revoke), and the retry that follows a failed call.
        """
        self._audit(
            "credential.revoke",
            actor=self.self_device,
            subject=holder,
            detail={
                "credential_key": key,
                "act": self.self_device,
                "sub": holder,
                "cause": cause,
                "revoked": revoked,
                "deferred": deferred,
            },
        )

    def revoke_outstanding(self, key: str, holder: str, *, cause: str = "revoked") -> int:
        """DELETE every outstanding token for ``(key, holder)`` now (the revoke op).

        Zero without a lender: nothing was ever minted on this relay, so there is
        nothing to revoke. The lender bounds this call's own wait (``REVOKE_SYNC_MAX``)
        and schedules the rest; a DELETE that fails is retried while the relay
        lives, and the token's own hour is the hard bound underneath.
        """
        lender = self._github_lender
        if lender is None or not github_app.is_github_key(key):
            return 0
        return lender.revoke_holder(holder, key=key, cause=cause)

    async def _resolve_github(
        self,
        *,
        key: str,
        provider: str,
        for_session: str,
        holder_scope: str,
        by: str,
    ) -> Any:
        """Serve one github bearer from the owner's source ladder (§3.2).

        The arm is resolved fresh at EVERY serve, so a re-login or a PAT
        rotation on the owner is picked up by the node's next command with no
        gesture on the node (§3.3's refresh path). Arms 2/3 return the token
        itself with ``refreshed: false`` and NO lender registration — nothing
        was minted here, so there is no window-end DELETE to schedule, and the
        revocation receipt says why. A configured-but-broken arm refuses BY
        NAME rather than falling through: falling through would silently lend a
        wider credential than the one the operator set up.
        """
        common: dict[str, Any] = {
            "key": key,
            "owner_device": self.self_device,
            "owner_device_name": self.self_device_name,
        }
        if holder_scope != "device":
            # ``grant()`` refuses this before any work; this is the resolver's own
            # belt — nothing is minted for a row that claims a bound that exists.
            return BrokerError(
                code=github_app.CODE_DEVICE_SCOPE,
                message="this key is device-scoped; the placement row that asked is not",
                **common,
            )
        source = github_app.resolve_source(self.root)
        if source == github_app.SOURCE_APP:
            return await self._serve_github_app(key=key, provider=provider, by=by, common=common)
        if source == github_app.SOURCE_UNREADABLE:
            # THE LADDER STOPS HERE (M1): the store exists but cannot be read, so
            # a narrower arm may be hiding in it — serving anything wider (the gh
            # login) would be the silent downgrade this policy forbids.
            return BrokerError(
                code=github_app.CODE_STORE_UNREADABLE,
                message=(
                    "the secret store on this device exists but could not be read, so the "
                    "GitHub source ladder cannot be resolved — nothing was lent, and no "
                    "wider source is substituted while it is unreadable (repair or restore "
                    "the store on this device; the network guide has the ladder)"
                ),
                **common,
            )
        if source in (github_app.SOURCE_TOKEN, github_app.SOURCE_GH):
            try:
                if source == github_app.SOURCE_TOKEN:
                    token = github_app.read_token_secret(self.root)
                else:
                    # OWNER-OFF-LOOP: asking gh may spawn gh, so it runs in the
                    # broker's executor like the mint — a hung gh must not stall
                    # the relay (and never runs on the borrower).
                    token = await asyncio.get_running_loop().run_in_executor(
                        None, github_app.read_gh_token
                    )
            except (github_app.GithubTokenError, github_app.GithubGhError) as exc:
                if exc.kind == "absent":
                    # A race with the presence probe (the secret or hosts file was
                    # removed between describe and read): the honest answer is the
                    # one an empty ladder gives.
                    return BrokerError(
                        code="no_local_credential",
                        message=github_app.no_source_message(),
                        **common,
                    )
                code = (
                    github_app.CODE_TOKEN_UNUSABLE
                    if source == github_app.SOURCE_TOKEN
                    else github_app.CODE_GH_UNUSABLE
                )
                return BrokerError(code=code, message=str(exc), **common)
            return self._grant_from_owner_token(
                token=token, key=key, provider=provider, common=common
            )
        return BrokerError(
            code="no_local_credential", message=github_app.no_source_message(), **common
        )

    async def _serve_github_app(
        self, *, key: str, provider: str, by: str, common: dict[str, Any]
    ) -> Any:
        """Mint one installation token from the owner's GitHub App (§D3/D4).

        The ladder's strongest arm, unchanged from before the ladder existed:
        unlike every provider path it touches no ``auth.db`` row — the bearer
        is minted on the wire, and it is registered with the revoker BEFORE it
        is returned, so a token whose reply never reaches the borrower is still
        revoked at its window end. The mint is blocking HTTP and runs off this
        loop through the broker's executor — a stuck GitHub must not stall the
        relay (the same reason the MCP refresh leaves the loop for its lock).
        """
        try:
            app = github_app.read_app_key(self.root)
        except github_app.GithubAppKeyError as exc:
            if exc.kind == "absent":
                return BrokerError(
                    code="no_local_credential",
                    message=github_app.no_source_message(),
                    **common,
                )
            return BrokerError(code=github_app.CODE_APP_UNUSABLE, message=str(exc), **common)
        repositories = github_app.repositories_for(self.root)
        if not repositories:
            return BrokerError(
                code=github_app.CODE_REPOSITORIES_UNSET,
                message=(
                    "no repositories are designated for GitHub brokering on this device; "
                    "set network.credentials.github.repositories (see the network "
                    "guide) — nothing was lent"
                ),
                **common,
            )
        lender = self._lender()
        try:
            minted = await asyncio.get_running_loop().run_in_executor(
                None, lender.mint, app, repositories
            )
        except github_app.GithubApiError as exc:
            if exc.kind == "auth":
                return BrokerError(code=github_app.CODE_APP_UNUSABLE, message=str(exc), **common)
            if exc.kind == "coverage":
                return BrokerError(code=github_app.CODE_REPO_REFUSED, message=str(exc), **common)
            return BrokerError(
                code="refresh_failed", retry_after_ms=30_000, message=str(exc), **common
            )
        now_ms = int(time.time() * 1000)
        ttl_ms = int(_grant_ttl_s(self.root) * 1000)
        ceiling = now_ms + ttl_ms
        grant_exp = min(minted.expires_at_ms, ceiling) if minted.expires_at_ms else ceiling
        grant = Grant(
            access_token=minted.token,
            kind="bearer",
            token_expires_at_ms=minted.expires_at_ms,
            grant_expires_at_ms=grant_exp,
            credential_ref=CredentialRef(
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                provider=provider or github_app.GITHUB_KEY,
                # The wire kind the revocation receipt and the borrower branch on.
                kind=github_app.GITHUB_KIND,
                # No row anywhere, so the credential's id is the synthetic one the
                # borrower path already uses for "not a local row".
                credential_id=synthetic_credential_id(key, self.self_device),
            ),
            served_by=self.self_device,
            # Minted during THIS request by construction — the flag is an
            # observation on the provider paths and a fact here.
            refreshed=True,
            scope=GrantScope(kind="device", session_id=""),
            identity={},
            grant_id=f"g_{os.urandom(8).hex()}",
        )
        lender.register(
            key=key,
            holder=by,
            grant_id=grant.grant_id,
            token=minted.token,
            grant_exp_ms=grant_exp,
            token_exp_ms=minted.expires_at_ms,
        )
        return grant

    def _grant_from_owner_token(
        self, *, token: str, key: str, provider: str, common: dict[str, Any]
    ) -> Any:
        """The Grant arms 2/3 serve: NO registration and no server-side revoke.

        ``refreshed: false`` is a fact here, not an observation — nothing was
        minted or rotated by this serve; the value is whatever the owner's
        source holds at this instant, re-read on every serve (§3.3).
        ``token_expires_at_ms = 0`` is the store's own encoding for "no known
        expiry"; a PAT can still expire at the forge, and a user token lives
        until it is revoked there, so the revoke receipt names that end instead
        of inventing a ceiling (``cli.py``'s per-source copy). The grant window
        below is what lop promises about ITS OWN hand-off, not what the token
        promises.
        """
        now_ms = int(time.time() * 1000)
        ttl_ms = int(_grant_ttl_s(self.root) * 1000)
        return Grant(
            access_token=token,
            kind="bearer",
            token_expires_at_ms=0,
            grant_expires_at_ms=now_ms + ttl_ms,
            credential_ref=CredentialRef(
                owner_device=self.self_device,
                owner_device_name=self.self_device_name,
                provider=provider or github_app.GITHUB_KEY,
                kind=github_app.GITHUB_KIND,
                # No row anywhere, so the credential's id is the synthetic one
                # the borrower path already uses for "not a local row".
                credential_id=synthetic_credential_id(key, self.self_device),
            ),
            served_by=self.self_device,
            refreshed=False,
            scope=GrantScope(kind="device", session_id=""),
            identity={},
            grant_id=f"g_{os.urandom(8).hex()}",
        )

    # -- report -------------------------------------------------------------

    def report(self, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        """Attribute a failure a borrower observed, WITHOUT touching the login.

        The three arms and their bounds are this module's docstring. The one thing
        worth repeating in the code: there is no ``rotate_sibling`` here, and adding
        one is the single change that would let a peer's bad 401 log the operator
        out of every device.

        ``credential_id`` IS A CLAIM, CHECKED AGAINST THIS DEVICE'S OWN ID SPACE
        (audit Q5 #4). The arms act on the named row only when it is one this device
        really has for the key; an absent field keeps the pre-existing first-row
        pick, so an older peer behaves exactly as it did.
        """
        key = str(frame.get("key") or "")
        failure = str(frame.get("failure") or "")
        reported_id = peer_int(frame.get("credential_id"))
        by, refused = self._caller(link, frame, key, key)
        if refused is not None:
            return refused
        self._audit(
            "credential.report",
            actor=self.self_device,
            subject=by,
            session_id=str(frame.get("for_session") or ""),
            detail={
                "credential_key": key,
                "act": self.self_device,
                "sub": by,
                "failure": failure,
            },
        )
        entry = self._entry(key)
        if entry is None or entry.owner_device != self.self_device or not entry.is_holder(by):
            return {"kind": "error", "code": "not_a_holder", "key": key}
        if failure == "quota":
            return self._remote_quota_block(key, entry, by, frame, reported_id=reported_id)
        if failure in ("invalid", "unauthorized"):
            return self._one_owner_refresh(key, entry, by, reason=failure, reported_id=reported_id)
        # Anything else — ``unavailable`` (a provider 5xx/529 overload), ``failed`` —
        # is audited above and changes nothing: a provider-side fault is not evidence
        # against the credential, and the owner's own rotation only DEPRIORITISES on
        # it, which is routing state a peer must not move (§2.1).
        return {"kind": "ack", "key": key, "action": "noted"}

    def _remote_quota_block(
        self,
        key: str,
        entry: CredentialPlacementEntry,
        by: str,
        frame: dict[str, Any],
        *,
        reported_id: int = 0,
    ) -> dict[str, Any]:
        """A peer's 429 as AT MOST a short, family-scoped, rate-limited block (F3).

        Each bound closes a reproduced lock-out:

        * **Never account-wide.** An account-wide block is what took the owner's OWN
          session off its login for an hour on one report with no model. A report
          whose scope cannot be carried faithfully — no model, or a family this
          device's registry does not know — is NOTED, never widened: the block READ
          matches a scope by substring (``is_blocked_for_model``), so an unknown or
          short slug could stop far more than the family it names.
        * **The owner's duration, not the peer's.** The block is ALWAYS
          :data:`REMOTE_QUOTA_BLOCK_MAX_MS`, and ``retry_after_ms`` is not read at all.
          The owner's own path writes ``max(60 s, retry_after)``, so 60 s is the
          shortest block it would write on its own evidence; honouring a smaller claim
          went below that, and a non-numeric one crashed this arm after it had spent
          the holder's slot (review round 2, m1).
        * **Never over a live block.** A family already out of rotation — on the
          owner's own evidence or an earlier report — is left exactly as it is. The
          store's write is an unconditional upsert, so writing here rewrote a 45-minute
          block the owner had measured down to the peer's 1 s (review round 2, m2). A
          second-hand report may start a short block; it never shortens or extends one.
        * **Once per holder per window.** :data:`REPORT_BLOCK_MIN_INTERVAL_S`.
        """
        if is_mcp_key(key):
            return {"kind": "ack", "key": key, "action": "noted", "reason": "unscoped"}
        scope = remote_block_scope(
            str(frame.get("block_scope") or ""), str(frame.get("model_id") or "")
        )
        if not scope:
            return {"kind": "ack", "key": key, "action": "noted", "reason": "unscoped"}
        credential_id = self._credential_id_for(key, reported_id)
        if credential_id is None:
            return {"kind": "ack", "key": key, "action": "noted"}
        now = time.monotonic()
        slot = (by, credential_id)
        with self._report_lock:
            last = self._report_blocked.get(slot)
            if last is not None and now - last < REPORT_BLOCK_MIN_INTERVAL_S:
                return {"kind": "ack", "key": key, "action": "coalesced"}
            self._report_blocked[slot] = now
        block_ms = REMOTE_QUOTA_BLOCK_MAX_MS
        store = self._auth_store_instance()
        try:
            # ``is_blocked_for_model`` matches a scoped block by its slug appearing in
            # the model id, so asking with the bare slug is exactly "is a block for
            # THIS family (or the whole account) live now" — the read the owner's own
            # routing uses, so the two cannot disagree about what counts as blocked.
            if store.is_blocked_for_model(
                credential_id, entry.provider, scope.removeprefix("model:")
            ):
                return {"kind": "ack", "key": key, "action": "noted", "reason": "already_blocked"}
            store.block_credential(
                credential_id, entry.provider, block_scope=scope, block_ms=block_ms
            )
        except Exception:  # noqa: BLE001 — a failed block is not a failed turn
            return {"kind": "ack", "key": key, "action": "noted"}
        return {
            "kind": "ack",
            "key": key,
            "action": "blocked",
            "scope": scope,
            "block_ms": block_ms,
        }

    def _one_owner_refresh(
        self,
        key: str,
        entry: CredentialPlacementEntry,
        by: str,
        *,
        reason: str,
        reported_id: int = 0,
    ) -> dict[str, Any]:
        """AT MOST ONE owner-side refresh per credential per report window.

        The owner decides what a 401 means for its own row. If the IdP really has
        refused the grant, the owner's OWN resolve raises ``CredentialInvalidError``
        and the tombstone is written by the path that already writes it — not by a
        peer's observation.
        """
        credential_id = self._credential_id_for(key, reported_id)
        if credential_id is None:
            return {"kind": "ack", "key": key, "action": "noted"}
        now = time.monotonic()
        # ``None`` MEANS "NEVER REFRESHED", and it is checked as such (review round 3,
        # F1). A ``0.0`` default read as "refreshed at clock zero", and ``monotonic``
        # counts from boot: on any host up for less than the window — a fresh CI
        # runner, a laptop just after a reboot — the FIRST report was coalesced and
        # nothing was refreshed. The same ``is not None`` guard the block arm uses.
        with self._report_lock:
            last = self._report_refreshed.get(credential_id)
            if last is not None and now - last < REPORT_REFRESH_MIN_INTERVAL_S:
                return {"kind": "ack", "key": key, "action": "coalesced"}
            self._report_refreshed[credential_id] = now
        try:
            self._loop.submit(
                self._owner_refresh(key, credential_id),
                timeout=max(1.0, self._op_deadline_s - HANDLER_WAIT_MARGIN_S),
            )
        except Exception:  # noqa: BLE001 — a probe is a read; its failure is not an event
            pass
        self._audit(
            "credential.refresh",
            actor=self.self_device,
            subject=by,
            detail={
                "credential_key": key,
                "act": self.self_device,
                "sub": by,
                "cause": reason,
            },
        )
        return {"kind": "ack", "key": key, "action": "refreshed"}

    async def _owner_refresh(self, key: str, credential_id: int) -> Any:
        """The owner's OWN refresh for ``key``: the provider store's, or MCP's.

        MCP rows are not provider rows — ``AuthStore.ensure_oauth_fresh`` has no
        provider definition for ``mcp-oauth`` — so an MCP report runs the same
        ``ensure_mcp_oauth_fresh`` a grant does, under its cross-process lock.
        """
        if is_mcp_key(key):
            from local_operator.mcp.auth import ensure_mcp_oauth_fresh
            from local_operator.mcp.config import MCPHttpServerConfig

            url = mcp_url_from_key(key)
            return await ensure_mcp_oauth_fresh(
                url, MCPHttpServerConfig(url=url), self._auth_store_instance()
            )
        return await self._auth_store_instance().ensure_oauth_fresh(credential_id)

    # -- placement ----------------------------------------------------------

    def placement_frame(self, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        """Push or pull the placement document on ONE op, ``kind: placement``.

        A ``document`` in the frame is merged INTO THE FILE under its lock
        (``placement.merge_from_peer``), attributed to the device the TRANSPORT
        authenticated — a push that names another device in ``from_device`` is
        refused before the merge (F1), and ``merge`` itself refuses anything that
        claims to come from this device. ``want: "pull"`` answers with THIS device's
        document as it is on disk now. The pull direction is what makes a share
        reach a running peer without a proactive fan-out the relay would have to
        own, and it costs one bounded round trip on a path that was about to refuse
        anyway.
        """
        by, refused = self._caller(link, frame, "", "the placement document")
        if refused is not None:
            return refused
        document = frame.get("document")
        if isinstance(document, dict):
            with suppress(OSError):
                placement_mod.merge_from_peer(
                    self.network_id,
                    document,
                    from_device=by,
                    self_device=self.self_device,
                    root=self.root,
                )
        if str(frame.get("want") or "") != "pull":
            return {"kind": "ack", "key": ""}
        return {"kind": "placement", "document": self._document().to_json()}

    # -- helpers ------------------------------------------------------------

    def _caller(
        self, link: Any, frame: dict[str, Any], key: str, label: str
    ) -> tuple[str, dict[str, Any] | None]:
        """``(device, refusal)``: WHO is asking, from the transport, never the frame.

        ``link.device_id`` is the device whose key the handshake verified; the
        frame's ``from_device`` is only what the sender wrote. It is kept as an
        ASSERTION — an honest client always sends its own id, so a mismatch is a
        forgery or a build bug and is refused as ``identity_mismatch`` with nothing
        lent or changed. A link with no device id at all is refused the same way:
        there is no caller to authorise.
        """
        authenticated = str(getattr(link, "device_id", "") or "")
        claimed = str(frame.get("from_device") or "")
        if not authenticated:
            return "", self._refuse(
                link,
                key,
                label,
                claimed,
                "identity_mismatch",
                "this request arrived on a link with no authenticated device; nothing was "
                "lent or changed",
            )
        if claimed and claimed != authenticated:
            return authenticated, self._refuse(
                link,
                key,
                label,
                authenticated,
                "identity_mismatch",
                f"this request named {claimed} as its sender but arrived from "
                f"{authenticated}; nothing was lent or changed",
            )
        return authenticated, None

    def _document(self) -> PlacementDocument:
        """The sharing list AS IT IS ON DISK NOW. Read per request, never cached (F2).

        One small JSON read per broker request. A cached copy is what let a revoke
        made by the CLI — another process — go unnoticed until the relay restarted.
        """
        return PlacementDocument.load(self.network_id, self.root, self_device=self.self_device)

    def _entry(self, key: str) -> CredentialPlacementEntry | None:
        return self._document().entry(key)

    def _mcp_row_id(self, url: str) -> int | None:
        """The owner's ``mcp-oauth`` ROW id for ``url``, the same row ``McpTokenStorage``
        reads (matched on ``identity_key == url``), or ``None``."""
        from local_operator.mcp.auth import MCP_OAUTH_PROVIDER

        try:
            rows = self._auth_store_instance().list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:  # noqa: BLE001 — an unreadable store holds nothing to lend
            return None
        for row in rows:
            if row.identity_key == url:
                return int(row.id)
        return None

    def _auth_store_instance(self) -> Any:
        """The owner's own ``AuthStore``, built on first use.

        Built with ``config_dir=root``, EXACTLY as ``session_factory`` builds the
        one a session uses: the broker must read the same ``auth.db`` the operator's
        sessions do, and a second store with its own path would hold its own locks —
        the cross-process lease is the defence, and two leases over two files
        enforce nothing.
        """
        if self._auth_store is None:
            from local_operator.providers.auth_store import AuthStore

            self._auth_store = AuthStore(config_dir=self.root)
        return self._auth_store

    def _credential_id_for(self, key: str, reported_id: int = 0) -> int | None:
        """The owner's row id for ``key``, for the report arms. Prefers the lent row.

        THE REPORT'S TARGET MUST BE THE ROW THAT WAS LENT (audit Q5 #4): on a
        multi-row owner, resolving a peer's report to the provider's FIRST row made a
        quota block — or a provoked refresh — land on a login the borrower never
        touched. A report naming the granted ``credential_ref.credential_id`` is
        believed only when that id exists in the id space the arms can act on — this
        device's own rows for the entry's provider — because the id is a peer's
        claim. A name this device cannot match is NOT actioned (the arms answer
        ``noted``) rather than redirected to a row the peer never mentioned; an
        ABSENT field keeps the first-row pick exactly as it was, which is what an
        older peer's frame carries.

        MCP rows are matched by the server URL, which is their identity: one row per
        URL, and the id a peer saw is only a spelling of that same row, so the
        reported id adds nothing there and is ignored.
        """
        if is_mcp_key(key):
            return self._mcp_row_id(mcp_url_from_key(key))
        entry = self._entry(key)
        if entry is None:
            return None
        try:
            rows = self._auth_store_instance().list_credentials(entry.provider)
        except Exception:  # noqa: BLE001
            return None
        ids = [int(row.id) for row in rows]
        if reported_id:
            return reported_id if reported_id in ids else None
        return ids[0] if ids else None

    def _quota_remaining_ms(self, provider: str, model_id: str) -> int:
        """How long every row for ``provider`` is quota-blocked for ``model_id``, ms.

        The producer ``quota_blocked`` never had (audit Q5 #9), read only when the
        owner's own cascade came back EMPTY and only when EVERY row the provider has
        is blocked for the requested model — the exact reason ``_usable_key_rows``
        could not offer one. A row that is still servable means the empty answer has
        another cause, and the caller keeps ``no_local_credential``. The earliest
        unblock is the wait worth reporting: it is when something can be served
        again, whatever happened to the other rows.
        """
        try:
            store = self._auth_store_instance()
            rows = store.list_credentials(provider)
            waits = [int(store.blocked_remaining_ms(row.id, provider, model_id)) for row in rows]
        except Exception:  # noqa: BLE001 — an unreadable store explains nothing
            return 0
        if not waits:
            return 0
        return min(waits) if all(wait > 0 for wait in waits) else 0

    def _row_stamps(self, provider: str) -> dict[int, int]:
        """``{credential_id: updated_at}`` for a provider, to observe a refresh.

        The store stamps ``updated_at`` on every write, so a refresh is visible as a
        moved stamp on the row that served the grant. Read BEFORE and AFTER the
        resolve, which is what makes ``refreshed`` on the wire an observation rather
        than a guess.
        """
        try:
            rows = self._auth_store_instance().list_credentials(provider)
        except Exception:  # noqa: BLE001
            return {}
        return {int(row.id): int(row.updated_at or 0) for row in rows}

    def _classify_refresh_failure(self, exc: BaseException, key: str) -> BrokerError:
        """Turn the store's failure classes into the design's refusal codes.

        The distinction the codes exist for: ``grant_invalid`` means the IdP has
        refused the grant and only a re-login helps (the tombstone, written by the
        store's own path), while ``refresh_failed`` is a transient outage that a
        retry can fix. Collapsing them is how an operator is told to sign in again
        for a network blip.
        """
        from local_operator.providers.auth_store import CredentialInvalidError

        # ``Any`` because the dict is unpacked with ``**``: typed as ``dict[str, str]``
        # pyright reads every key as a ``str``, including ``retry_after_ms``, and
        # reports the explicit int argument as a conflict. The same spelling this
        # package's CLI already uses for its ``**fields``.
        common: dict[str, Any] = {
            "key": key,
            "owner_device": self.self_device,
            "owner_device_name": self.self_device_name,
        }
        if isinstance(exc, CredentialInvalidError):
            return BrokerError(code="grant_invalid", message=str(exc), **common)
        return BrokerError(
            code="refresh_failed",
            retry_after_ms=30_000,
            message=f"{exc.__class__.__name__}",
            **common,
        )

    def _is_admin(self, device: str) -> bool:
        """Whether ``device`` holds the ``admin`` ROLE in this network.

        Read from the member record rather than from a capability: the design's
        capability vocabulary has no ``trust``/``admin`` member, and role is the
        thing an admin invite grants.
        """
        if not device:
            return False
        if device == self.self_device:
            return True
        try:
            from local_operator.network import store

            for record in store.list_networks(self.root):
                member = record.member(device)
                if member is not None:
                    return str(member.role) == "admin"
        except Exception:  # noqa: BLE001
            return False
        return False

    def _refuse(
        self,
        link: Any,
        key: str,
        provider: str,
        by: str,
        code: str,
        message: str,
        *,
        retry_after_ms: int = 0,
    ) -> dict[str, Any]:
        """One refusal, audited on the owner and shaped as §3.2's failure reply."""
        error = BrokerError(
            code=code,
            key=key,
            owner_device=self.self_device,
            owner_device_name=self.self_device_name,
            retry_after_ms=retry_after_ms,
            message="",
        )
        # The peer gets the CODE and the owner's own words; the SENTENCE the operator
        # reads is rendered where the operator is (``client.py`` →
        # ``messages.py``), because only that side knows the last-seen time and
        # whether the operator can run the remedy locally.
        self._audit(
            "credential.grant_refused",
            actor=self.self_device,
            subject=by,
            detail={
                "credential_key": key,
                "act": self.self_device,
                "sub": by,
                "code": code,
                "capability": "broker_credential",
            },
        )
        error.message = message or f"{provider!r} was refused ({code})"
        return error.to_detail()

    def _audit(self, event: str, **fields: Any) -> None:
        """One audit record, best effort. The token is never a parameter."""
        if self.audit is None:
            return
        try:
            from local_operator.network.audit import AuditEvent

            self.audit.record(
                AuditEvent(
                    event=event,
                    network_id=self.network_id,
                    epoch=self._epoch(),
                    actor_name=self.self_device_name,
                    actor_kind="device",
                    **fields,
                )
            )
        except Exception:  # noqa: BLE001 — a log that cannot be written is not a refusal
            pass

    def _epoch(self) -> int | None:
        try:
            from local_operator.network import store

            return int(store.load(self.network_id, self.root).epoch)
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        """Stop the broker's loop. For tests and for a relay shutting down."""
        if self._github_lender is not None:
            self._github_lender.close()
        self._loop.close()


#: The private attribute a relay server carries its one broker under.
#:
#: A STASH ON THE SERVER, not a side table: a ``WeakKeyDictionary`` cannot key
#: every object a caller hands in (a test's ``SimpleNamespace`` is not weakly
#: referenceable, and the per-relay keeper is part of the tested behaviour), and
#: an ``id()``-keyed table can alias after a freed server's id is reused — a
#: stale broker served for a different relay. The server object IS the identity;
#: the broker lives on it and dies with it. Built under a lock because the
#: relay's peer handler and the ``credential_revoke`` local op can race for the
#: first build on two different threads.
_BROKER_ATTR = "_mesh_credential_broker"
_BROKER_LOCK = threading.Lock()


def broker_for_relay(server: Any) -> MeshCredentialBroker | None:
    """The broker for ``server``, built on first need and kept — or ``None``.

    ONE accessor for BOTH callers (the ``net_broker`` peer handler and the
    ``credential_revoke`` local op) because there must be exactly one broker per
    relay: the revoke op must reach the same in-memory mint registry the peer
    handler mints into, and a second broker object would revoke nothing.
    ``None`` is deliberately NOT cached — a share that arrives after the relay
    started must be served without a restart (``for_relay``'s docstring).
    """
    with _BROKER_LOCK:
        broker = getattr(server, _BROKER_ATTR, None)
        if broker is not None:
            return broker
        broker = MeshCredentialBroker.for_relay(server)
        if broker is not None:
            try:
                setattr(server, _BROKER_ATTR, broker)
            except Exception:  # noqa: BLE001 — a server that refuses attributes
                # keeps no keeper; it pays a rebuild per request rather than
                # growing a module-level table that outlives it.
                pass
        return broker


def _already_expired(expiry: float | None, *, now: float | None = None) -> bool:
    """Whether a stored token's own recorded expiry is in the past.

    ``None`` means "no opinion" (no row, no token, or a server that quoted no
    lifetime) and must NEVER read as expired: forcing a re-authorisation on a
    provider that issues non-expiring tokens is the failure
    ``McpTokenStorage.stored_token_expiry`` documents at its own definition.
    """
    if expiry is None:
        return False
    return float(expiry) <= (now if now is not None else time.time())


def remote_block_scope(claimed: str, model_id: str) -> str:
    """The ``model:<family>`` scope a peer's quota report may write, or ``""``.

    ``""`` means "this report cannot be scoped faithfully", and the caller then
    writes NOTHING rather than an account-wide block. Only a family this device's
    own registry parses out of a known model is accepted, because the block READ
    (``AuthStore.is_blocked_for_model``) matches by substring: a slug like ``a`` or
    ``""`` would stop every model on the account, which is the over-block a scoped
    report exists to avoid. The borrower's own scope (``configure._write_quota_block``
    writes ``model:fable``) wins when it is one of those families; otherwise the
    family is derived from the model the borrower ran, as ``rotate_sibling`` does.
    """
    families = _known_model_families()
    if claimed:
        slug = claimed.removeprefix("model:") if claimed.startswith("model:") else ""
        return f"model:{slug}" if slug in families else ""
    if not model_id:
        return ""
    from local_operator.model.registry import model_family

    family = model_family(model_id)
    return f"model:{family}" if family in families else ""


def _known_model_families() -> frozenset[str]:
    """Every non-empty family the shipped registry's Anthropic models parse to."""
    from local_operator.model.registry import anthropic_models, model_family

    return frozenset(filter(None, (model_family(model_id) for model_id in anthropic_models)))


def _broker_deadline_s() -> float:
    """The registered owner-side deadline for ``net_broker``, or a safe default."""
    try:
        from local_operator.network.credentials import BROKER_OP_DEADLINE_S

        return float(BROKER_OP_DEADLINE_S)
    except Exception:  # noqa: BLE001
        return 75.0


def _grant_ttl_s(root: Path | None) -> float:
    """``network.credentials.grant_ttl_s`` through the package's ONE config reader."""
    from local_operator.network.credentials import grant_ttl_s

    return float(grant_ttl_s(root))

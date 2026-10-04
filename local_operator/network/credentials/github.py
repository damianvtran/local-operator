"""The ``github`` credential: a GitHub App installation token, minted by its owner.

WHAT THIS CREDENTIAL IS, AND WHY IT IS NOT A PROVIDER LOGIN. Every other key in
``placement`` names a login this device holds in its own ``auth.db`` and serves
through ``AuthStore``. ``github`` has no row anywhere: the OWNER mints a
short-lived GitHub **App installation token** on demand (GitHub caps these at
one hour), hands it to a borrower over the existing broker wire, and the
borrower uses it for ``git`` and ``gh`` — never for provider calls. So this
module is the adapter the broker dispatches to, plus the borrower-side delivery
(env construction, the git credential helper, and the self-revoke belt).

THE APP KEY LIVES IN THE OWNER'S ENCRYPTED SECRET STORE (``GITHUB_APP``), one
JSON blob holding ``app_id``, ``installation_id`` and the PEM private key. The
one-time setup checklist is in the network guide; until that secret exists the
adapter refuses ``no_local_credential`` in the reader's own terms (push and
PR-write are unavailable; public clones and non-GitHub work are unaffected).
Nothing here fails closed on the missing secret in a way that touches any other
credential class.

DEVICE-SCOPED BY CONSTRUCTION (design F3). The borrower side has no
rail-authenticated session identity — a same-uid process can claim any
``for_session`` — so a session-scoped grant would enforce nothing. A session id
therefore travels ONLY as attribution (audit rows, cache keying), and both
directions refuse a session-scoped row by name: the share verb at the document
(``placement.py``) and the owner at serve time (:meth:`MeshCredentialBroker`
dispatch in ``owner.py``). The disclosure that follows — any process or session
on the borrower node can borrow while the share stands — is stated on the share
receipt and in the network guide (the mandated T7(b) copy).

THE F1 CLOSE, WHICH IS WHY THE ENV IS SAFE. A node's global
``credential.helper=store`` would otherwise receive the brokered token on
``approve`` and write it to ``~/.git-credentials``. The delivery therefore resets
the helper list FOR GITHUB.COM ONLY before adding the brokered helper
(``GIT_CONFIG_COUNT=3`` + the empty first entry) — spike-proven: with the reset,
github.com resolves to the brokered helper alone while other hosts keep their
configured helpers; without it the store receives and persists the token. The
helper itself re-checks ``protocol``/``host`` and requires a well-formed ``path``
in the configured allow-list, so a lookalike host or a naive
``git credential fill`` extraction is refused. `git` rides the helper;
``gh`` rides ``GH_TOKEN``/``GITHUB_TOKEN`` (gh has no credential-helper
protocol, so the env is the only carrier for that leg).

MINT-REVOKE (design F4). The owner keeps a bounded in-memory registry of
minted-but-unrevoked tokens per ``(key, holder)`` and calls
``DELETE /installation/token`` at each token's window end on the broker's own
loop; ``credential revoke`` (and only that path today) revokes immediately via
the ``credential_revoke`` local op. Revoke is idempotent: a token that no longer
authenticates IS the goal state. A failed call is recorded and retried while
this process lives; if no revoke can ever be delivered, the token falls back to
its own 60-minute ceiling. The borrower holds a best-effort belt: it self-revokes
the tokens it fetched at their window end.

Stdlib only at import: the relay imports the credentials package at
construction, so ``jwt`` and ``httpx`` are imported inside the functions that
reach the wire.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The placement key AND the provider spelling: this credential is not a
#: provider login, so the two are the same string on purpose (a second spelling
#: is how a share and a borrow come to disagree about which key they mean).
GITHUB_KEY = "github"

#: The placement row's kind. ``CredentialRef.kind`` on the wire carries it too,
#: and the revocation receipt branches on it for the github-app wording.
GITHUB_KIND = "github-app"

#: The owner's secret-store entry. ONE JSON blob: ``app_id``, ``installation_id``,
#: ``private_key`` (PEM). One name, so the checklist is one command; see
#: ``local_operator/guides/network/GUIDE.md``.
APP_SECRET_NAME = "GITHUB_APP"

#: Where the designated repositories live. The SAME key is read by the owner (the
#: mint narrows to it — and refuses when it is empty rather than minting a token
#: that covers the whole installation) and by the borrower's helper as its
#: use-time backstop.
REPOSITORIES_PATH: tuple[str, ...] = ("network", "credentials", "github", "repositories")

#: The default: none. An empty list is a REFUSAL at mint time, never "no
#: narrowing" — ``repositories`` absent on the wire would grant the token access
#: to everything the installation covers.
DEFAULT_REPOSITORIES: tuple[str, ...] = ()

#: The API surface. ``2022-11-28`` is the version the shipped endpoint shapes
#: were verified against (mint: ``repositories`` + ``permissions``; revoke:
#: ``DELETE /installation/token`` -> 204, idempotent).
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

#: The permissions the mint asks for, fixed rather than configurable: the two a
#: git push and a pull request need. A narrower request would refuse at the
#: first ``git push``; a wider one would be a capability nobody asked for.
GITHUB_PERMISSIONS: dict[str, str] = {"contents": "write", "pull_requests": "write"}

#: The refusal codes this adapter adds to the broker's closed set (registered in
#: ``types.BROKER_ERROR_TTL_MS``; sentences in ``messages.py``).
CODE_DEVICE_SCOPE = "device_scope_required"
CODE_APP_UNUSABLE = "github_app_unusable"
CODE_REPOSITORIES_UNSET = "github_repositories_unset"
CODE_REPO_REFUSED = "github_repo_refused"

#: How long a fetch may block on one HTTP round trip. Sequential and bounded: a
#: mint sits inside one slow-op worker, and the CLI's revoke op waits for its
#: DELETE calls.
HTTP_TIMEOUT_S = 10.0

#: How often a failed revoke is retried while this process lives.
REVOKE_RETRY_S = 60.0

#: The most tokens tracked per ``(key, holder)``. Each grant is one window
#: (<= ``grant_ttl_s``), so 16 is far above any honest overlap; past it the
#: OLDEST outstanding token is revoked immediately rather than forgotten — a
#: registry that drops records would silently lose revokability.
MAX_TRACKED_PER_HOLDER = 16

#: The most tokens one synchronous ``revoke_all`` will DELETE before scheduling
#: the rest. Bounds the control op's wait: each call is bounded by
#: ``HTTP_TIMEOUT_S``, and a holder's outstanding set is one window deep in any
#: honest workflow.
REVOKE_SYNC_MAX = 4


def is_github_key(key: str) -> bool:
    """Whether ``key`` names this adapter's credential. The one spelling rule."""
    return str(key or "") == GITHUB_KEY


# ---------------------------------------------------------------------------
# The configured allow-list
# ---------------------------------------------------------------------------


def normalise_repository(entry: Any) -> str:
    """``owner/repo`` from a config entry, or ``""`` when unusable.

    Tolerates a trailing ``.git`` (git writes ``path=owner/repo.git`` in the
    helper input, so an operator copying that spelling should not get a silent
    non-match). Case is preserved: GitHub treats owner/repo case-insensitively
    but the allow-list is compared exactly, and normalising case would make the
    receipt less copy-pasteable.
    """
    text = str(entry or "").strip()
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = text.split("/")
    if len(parts) != 2:
        return ""
    owner, name = parts[0].strip(), parts[1].strip()
    if not owner or not name or owner in (".", "..") or name in (".", ".."):
        return ""
    return f"{owner}/{name}"


def repositories_for(root: Path | None = None) -> tuple[str, ...]:
    """The designated repositories, normalised, from this device's config.

    Read through ``network.store.read_config`` (the package's ONE reader), and
    re-read per call: the config can change while a relay runs, and the mint is
    the moment the list has to be true.
    """
    from local_operator.network import store

    raw = store.read_config(REPOSITORIES_PATH, DEFAULT_REPOSITORIES, root)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return DEFAULT_REPOSITORIES
    normalised = [normalise_repository(entry) for entry in raw]
    return tuple(entry for entry in normalised if entry)


def mint_repository_names(repositories: Sequence[str]) -> list[str]:
    """The wire form of the allow-list: BARE repository names.

    ``POST /app/installations/{id}/access_tokens`` documents ``repositories`` as
    "list of repository names", and the official ``actions/create-github-app-token``
    action splits its ``owner/repo`` inputs down to the bare name before
    sending (``lib/main.js``: ``repositories.map(parseRepositoryInput)`` then
    ``repositoryNames``). The owner portion is fixed by the installation, so
    sending it is at best ignored and at worst a 422; the config keeps
    ``owner/repo`` because the helper's path check compares against git's own
    ``owner/repo`` spelling.
    """
    names: list[str] = []
    for entry in repositories:
        parts = str(entry).split("/", 1)
        if len(parts) == 2 and parts[1] and parts[1] not in names:
            names.append(parts[1])
    return names


# ---------------------------------------------------------------------------
# The App key (owner side)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GithubAppKey:
    """The three fields the checklist stores, parsed."""

    app_id: str
    installation_id: str
    private_key: str


class GithubAppKeyError(Exception):
    """``GITHUB_APP`` is absent or unusable. ``kind``: ``absent`` | ``unusable``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def app_secret_present(root: Path | None = None) -> bool:
    """Whether a ``GITHUB_APP`` secret exists, WITHOUT reading its value.

    The metadata read (``describe``), not a retrieval: the ledger and the share
    verb must answer "is a GitHub App configured here" without pulling key
    material into a process that only needs to say yes or no. An unreadable
    store answers ``False`` — the closed direction at a grant seam, the same
    rule ``offers.credential_here`` states for provider rows.
    """
    try:
        from local_operator.secrets import access
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path

        if not store_path(root).exists():
            # A store that was never created is a definite "not there": opening
            # one with ``create=True`` would make a read a writer.
            return False
        store = access.open_store(root)
        try:
            store.describe(APP_SECRET_NAME)
            return True
        except SecretStoreError:
            return False
    except Exception:  # noqa: BLE001 — an unreadable store is "not known"
        return False


def read_app_key(root: Path | None = None) -> GithubAppKey:
    """The App key from the owner's secret store. Raises :class:`GithubAppKeyError`.

    The VALUE is read through ``access.retrieve_secret`` — the one announced
    path (design §6) — rather than a bare ``open_store().get``, so the read is
    attributed exactly like every other value retrieval in this tree. The key
    itself never reaches a model-visible channel; only the tokens minted FROM it
    do (and those are exact-value registered on the borrower).
    """
    from local_operator.secrets import access
    from local_operator.secrets.errors import SecretStoreError
    from local_operator.secrets.keys import store_path

    if not store_path(root).exists():
        raise GithubAppKeyError(
            "absent",
            f"no secret store exists here, so no {APP_SECRET_NAME} is configured",
        )
    try:
        raw = access.retrieve_secret(APP_SECRET_NAME, root)
    except SecretStoreError as exc:
        raise GithubAppKeyError("absent", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — a locked/hardened store is a store that cannot serve
        raise GithubAppKeyError(
            "unusable", f"{APP_SECRET_NAME} could not be read ({exc.__class__.__name__})"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — malformed is its own answer
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is not a JSON object")
    app_id = str(payload.get("app_id") or "").strip()
    installation_id = str(payload.get("installation_id") or "").strip()
    private_key = str(payload.get("private_key") or "")
    missing = [
        name
        for name, value in (
            ("app_id", app_id),
            ("installation_id", installation_id),
            ("private_key", private_key),
        )
        if not value
    ]
    if missing:
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is missing {', '.join(missing)}")
    return GithubAppKey(app_id=app_id, installation_id=installation_id, private_key=private_key)


# ---------------------------------------------------------------------------
# The wire calls (mint + revoke)
# ---------------------------------------------------------------------------


class GithubApiError(Exception):
    """One failed API call. ``kind``: ``auth`` | ``coverage`` | ``offline`` | ``server``.

    The kinds are the owner's mapping table to refusal codes; the HTTP status is
    kept for the diagnostic (the borrower never sees it — messages are rendered
    from codes, and a status is not a sentence).
    """

    def __init__(self, kind: str, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclasses.dataclass(frozen=True)
class MintedToken:
    """One fresh installation token. The token is material; nothing logs it."""

    token: str
    expires_at_ms: int


@dataclasses.dataclass(frozen=True)
class _Reply:
    status: int
    body: dict[str, Any]


class _HttpxTransport:
    """The production transport: one ``httpx`` client per call.

    ``httpx`` is imported HERE (see the module docstring's stdlib-at-import
    rule). ``follow_redirects=False``: api.github.com does not redirect the two
    endpoints this module calls, and following one would re-send the
    Authorization header to wherever it pointed.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
    ) -> _Reply:
        import httpx

        with httpx.Client(timeout=HTTP_TIMEOUT_S, follow_redirects=False) as client:
            response = client.request(method, url, headers=headers, json=json_body)
        body: dict[str, Any] = {}
        try:
            decoded = response.json()
            if isinstance(decoded, dict):
                body = decoded
        except Exception:  # noqa: BLE001 — a non-JSON body is empty; the status is the answer
            body = {}
        return _Reply(status=response.status_code, body=body)


def _app_jwt(app: GithubAppKey, now: float | None = None) -> str:
    """The App JWT: ``iss`` = app id, ``iat`` 60 s back (clock skew), 10-min cap.

    GitHub's documented bounds: ``iat`` no more than 60 s in the past, ``exp`` no
    more than 10 minutes ahead. ``jwt`` (pyjwt[crypto]) is a direct dependency
    and is imported here, not at module scope.
    """
    import jwt

    moment = int(now if now is not None else time.time())
    return jwt.encode(
        {"iat": moment - 60, "exp": moment + 600, "iss": app.app_id},
        app.private_key,
        algorithm="RS256",
    )


def _expires_at_ms(expires_at: Any, now: float) -> int:
    """GitHub's ISO-8601 ``expires_at`` as epoch ms, or ``now + 1 h`` when unreadable.

    One hour is the documented lifetime of an installation token; a reply whose
    timestamp this build cannot parse still gets a correct-enough window rather
    than an epoch-zero expiry that would look already-dead.
    """
    try:
        parsed = datetime.datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except Exception:  # noqa: BLE001 — see above
        return int(now * 1000) + 3600_000


class GithubMinter:
    """The two API calls: mint one narrow token, revoke one token."""

    def __init__(self, *, base_url: str = GITHUB_API_BASE, transport: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport if transport is not None else _HttpxTransport()

    def mint(self, app: GithubAppKey, repositories: Sequence[str]) -> MintedToken:
        """Mint a token narrowed to ``repositories`` (bare names on the wire).

        THE MINT IS THE ALLOW-LIST CHECK (§D4): ``repositories`` names the
        covered set, ``permissions`` narrows the authority, and GitHub refuses
        (422) a repository the installation does not cover — "no silent
        widening, no separate convention". Callers let :class:`GithubApiError`
        carry the kind; nothing here retries (a retry loop lives in the caller
        only where the caller owns a schedule).
        """
        now = time.time()
        names = mint_repository_names(repositories)
        if not names:
            # THE BELT UNDER THE OWNER'S CHECK: minting without ``repositories``
            # would hand the token access to EVERYTHING the installation covers
            # (docs: absent means the installation's full set), so an empty
            # allow-list is refused here rather than widened silently.
            raise GithubApiError("coverage", "no repositories are designated")
        url = f"{self.base_url}/app/installations/{app.installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {_app_jwt(app, now)}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        try:
            reply = self._transport.request(
                "POST",
                url,
                headers=headers,
                json_body={"repositories": names, "permissions": dict(GITHUB_PERMISSIONS)},
            )
        except Exception as exc:  # noqa: BLE001 — any transport failure is "offline"
            raise GithubApiError("offline", f"{exc.__class__.__name__}") from exc
        if reply.status == 201:
            token = str(reply.body.get("token") or "")
            if not token:
                raise GithubApiError("server", "the mint reply carried no token", status=201)
            return MintedToken(
                token=token, expires_at_ms=_expires_at_ms(reply.body.get("expires_at"), now)
            )
        message = str(reply.body.get("message") or f"HTTP {reply.status}")
        if reply.status in (401, 403, 404):
            # The App's own authentication was refused, or the installation id
            # does not exist. The remedy is on the owner's setup, not a retry.
            raise GithubApiError("auth", message, status=reply.status)
        if reply.status == 422:
            # Under-covered repositories (or a malformed request). The remedy is
            # the installation/allow-list, and the sentence names the repos.
            raise GithubApiError("coverage", message, status=reply.status)
        raise GithubApiError("server", message, status=reply.status)

    def revoke(self, token: str) -> bool:
        """``DELETE /installation/token``. Idempotent: "already dead" is success.

        204 is the documented success. 401/403/404 are ALSO success: the only
        way this call can be refused authentication is that the token itself no
        longer authenticates — which is the goal state, because the call
        authenticates WITH the token it revokes. Anything else is a failure the
        caller records and retries while it lives.
        """
        url = f"{self.base_url}/installation/token"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        try:
            reply = self._transport.request("DELETE", url, headers=headers)
        except Exception:  # noqa: BLE001 — a failed call is retried by the lender
            return False
        return reply.status in (204, 401, 403, 404)


# ---------------------------------------------------------------------------
# The owner's mint-revoke registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TrackedToken:
    """One minted-but-unrevoked token, kept ONLY to be revocable."""

    key: str
    holder: str
    grant_id: str
    token: str
    grant_exp_ms: int
    token_exp_ms: int
    attempts: int = 0
    last_error: str = ""


class GithubLender:
    """Mints on demand and revokes at each token's window end (design F4).

    ONE instance per broker process, owned by :class:`MeshCredentialBroker`. The
    registry is IN MEMORY and intentionally so: a token's life is one window
    (<= ``grant_ttl_s``) and the hard fallback is the token's own one-hour
    ceiling, so persistence would buy a durability claim this design does not
    make. What it buys instead is the scheduled DELETE on the broker's own loop
    (:meth:`schedule`) and the immediate one on ``credential revoke``
    (:meth:`revoke_holder`).
    """

    def __init__(
        self,
        *,
        minter: GithubMinter | None = None,
        clock: Callable[[], float] = time.time,
        retry_s: float = REVOKE_RETRY_S,
    ) -> None:
        self._minter = minter if minter is not None else GithubMinter()
        self._clock = clock
        self._retry_s = retry_s
        self._lock = threading.Lock()
        #: ``(key, holder) -> {grant_id: _TrackedToken}``, insertion-ordered.
        self._tracked: dict[tuple[str, str], dict[str, _TrackedToken]] = {}
        self._loop: Any = None
        self._tasks: set[Any] = set()
        #: Set by the broker so scheduled fires can report without re-importing.
        self._audit_sink: Any = None

    # -- construction -------------------------------------------------------

    def attach_loop(self, loop: Any) -> None:
        """Bind the broker's one long-lived loop; call once, before any mint."""
        self._loop = loop

    def attach_audit(self, sink: "Callable[[str, str, str, int, int], None]") -> None:
        """``sink(cause, key, holder, revoked, deferred)`` — the broker's audit hook.

        A callable rather than the broker itself on purpose: this module must
        not import ``owner.py`` (the relay's construction path imports BOTH,
        and a cycle would be resolved by an import-time failure elsewhere).
        """
        self._audit_sink = sink

    # -- mint ---------------------------------------------------------------

    def mint(self, app: GithubAppKey, repositories: Sequence[str]) -> MintedToken:
        """Pass-through to the minter; kept on the lender so tests inject ONE seam."""
        return self._minter.mint(app, repositories)

    # -- registry -----------------------------------------------------------

    def register(
        self,
        *,
        key: str,
        holder: str,
        grant_id: str,
        token: str,
        grant_exp_ms: int,
        token_exp_ms: int,
    ) -> None:
        """Track a fresh token and schedule its window-end revoke."""
        record = _TrackedToken(
            key=key,
            holder=holder,
            grant_id=grant_id,
            token=token,
            grant_exp_ms=grant_exp_ms,
            token_exp_ms=token_exp_ms,
        )
        overflow: list[_TrackedToken] = []
        with self._lock:
            slot = self._tracked.setdefault((key, holder), {})
            slot[grant_id] = record
            while len(slot) > MAX_TRACKED_PER_HOLDER:
                _oldest_id, oldest = next(iter(slot.items()))
                del slot[_oldest_id]
                overflow.append(oldest)
        # Overflow is revoked EARLY, never forgotten: the registry's bound must
        # not be a way to lose revokability (see MAX_TRACKED_PER_HOLDER).
        for spilled in overflow:
            self._schedule(spilled, delay_s=0.0)
        self._schedule(record)

    def outstanding(self, *, key: str = GITHUB_KEY, holder: str = "") -> int:
        with self._lock:
            if holder:
                return len(self._tracked.get((key, holder), {}))
            return sum(
                len(slot) for (slot_key, _holder), slot in self._tracked.items() if slot_key == key
            )

    def revoke_holder(self, holder: str, *, key: str = GITHUB_KEY, cause: str = "revoked") -> int:
        """Revoke every outstanding token for ``(key, holder)`` NOW. Returns how many died.

        Blocking with a bounded shape (see REVOKE_SYNC_MAX): the caller is a
        control-op worker answering ``credential revoke``, and the operator is
        owed a count. Tokens past the sync budget are revoked via the loop (or
        synchronously when no loop is attached, which is the CLI-only case).
        """
        with self._lock:
            slot = self._tracked.get((key, holder), {})
            records = list(slot.values())
        revoked = 0
        deferred = 0
        for index, record in enumerate(records):
            if index < REVOKE_SYNC_MAX or self._loop is None:
                ok = self._revoke_now(record)
                if not ok and self._loop is not None:
                    # The retry contract covers the OPERATOR's immediate revoke
                    # too: a DELETE that failed must come back while the relay
                    # lives, not merely be counted as missing.
                    self._schedule(record, delay_s=self._retry_s)
            else:
                ok = False
                deferred += 1
                self._schedule(record, delay_s=0.0)
            if ok:
                revoked += 1
        self._record(cause=cause, key=key, holder=holder, revoked=revoked, deferred=deferred)
        return revoked

    def revoke_all_holders(self, *, key: str = GITHUB_KEY, cause: str = "revoked") -> int:
        """Every outstanding token for ``key``, whatever the holder. Returns the count."""
        with self._lock:
            holders = sorted({holder for (slot_key, holder) in self._tracked if slot_key == key})
        return sum(self.revoke_holder(holder, key=key, cause=cause) for holder in holders)

    # -- internals ----------------------------------------------------------

    def _revoke_now(self, record: _TrackedToken) -> bool:
        """One immediate DELETE. True when nothing is left to do for the token.

        True covers both success and "past the token's own ceiling": a token
        past its 1-hour expiry is dead by GitHub's clock, and a DELETE cannot
        authenticate with a dead token, so retrying it would be a loop with no
        reachable success. False means the caller should retry later.
        """
        ok = self._minter.revoke(record.token)
        with self._lock:
            slot = self._tracked.get((record.key, record.holder), {})
            if ok or self._clock() * 1000 >= record.token_exp_ms:
                slot.pop(record.grant_id, None)
                if not slot:
                    self._tracked.pop((record.key, record.holder), None)
                return True
            record.attempts += 1
            record.last_error = "revoke call failed"
            return False

    def _schedule(self, record: _TrackedToken, *, delay_s: float | None = None) -> None:
        """Queue one revoke on the broker's loop (thread-safe), or run it inline."""
        if delay_s is None:
            delay_s = max(0.05, record.grant_exp_ms / 1000.0 - self._clock())
        loop = self._loop
        if loop is None:
            # No loop (a unit-test lender, or a process that never bound one):
            # the token's own ceiling remains, stated rather than hidden.
            return
        loop.call_soon_threadsafe(self._fire_on_loop, record, max(0.05, delay_s))

    def _fire_on_loop(self, record: _TrackedToken, delay_s: float) -> None:
        """Runs on the loop thread: schedule the DELETE off the loop."""
        loop = self._loop
        if loop is None:  # pragma: no cover — close() raced the fire
            return
        loop.call_later(delay_s, self._start_reap, record)

    def _start_reap(self, record: _TrackedToken) -> None:
        """Create the reap task and RETAIN it: ``create_task`` holds a weak
        reference, so a task nobody keeps can be collected mid-run."""
        import asyncio

        task = asyncio.get_running_loop().create_task(self._reap(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _reap(self, record: _TrackedToken) -> None:
        """Off-loop DELETE; on failure retry while this process lives."""
        import asyncio

        with self._lock:
            slot = self._tracked.get((record.key, record.holder), {})
            if record.grant_id not in slot:
                return  # already revoked through another path; nothing to report
        loop = asyncio.get_running_loop()
        retry = record.attempts > 0
        ok = await loop.run_in_executor(None, self._revoke_now, record)
        if not ok:
            self._schedule(record, delay_s=self._retry_s)
        # EVERY fire reports, so the window-end DELETE and its retries are visible
        # to an incident reader — not only the operator-issued ones. The cause
        # tells them apart (see ``_github_audit_sink``).
        self._record(
            cause="revoke_retry" if retry else "grant_expired",
            key=record.key,
            holder=record.holder,
            revoked=1 if ok else 0,
            deferred=0,
        )

    def _record(self, *, cause: str, key: str, holder: str, revoked: int, deferred: int) -> None:
        sink = self._audit_sink
        if sink is None:
            return
        try:
            sink(cause, key, holder, revoked, deferred)
        except Exception:  # noqa: BLE001 — a log that cannot be written is not a refusal
            pass

    def close(self) -> None:
        """Cancel any pending fires; the loop itself is the broker's to stop."""
        loop = self._loop
        if loop is None:
            return
        for task in list(self._tasks):
            try:
                loop.call_soon_threadsafe(task.cancel)
            except Exception:  # noqa: BLE001 — a closed loop is already stopped
                pass
        self._tasks.clear()


# ---------------------------------------------------------------------------
# The borrower's delivery
# ---------------------------------------------------------------------------


def helper_command() -> str:
    """The ``GIT_CONFIG`` helper value: ``!`` + THIS BUILD's CLI, shell-quoted.

    Spelled with ``python_argv`` rather than a bare ``lop``: the helper MUST be
    the build that constructed the env (its host/protocol semantics are what the
    injected reset pair is paired with), and a ``lop`` resolved from the child's
    PATH may be a different generation entirely — the same reasoning
    ``builtin._sessions_launch`` states for spawning this build's CLI. The
    subcommand still ships as ``lop credential git-helper`` for a person to run
    and for tests to invoke.
    """
    from local_operator.interpreter import python_argv

    return "!" + shlex.join(python_argv("-m", "local_operator.cli", "credential", "git-helper"))


def git_env_for_token(token: str) -> dict[str, str]:
    """The exact child env for one grant (design §D1, spike-verified form).

    ``GIT_CONFIG_COUNT=3`` with the EMPTY first entry: the github.com-scoped
    helper list is reset before the brokered helper is added, so a node's
    global ``credential.helper=store`` (or any other persisting helper, global
    or repo-local) never receives the value — while other hosts keep their own
    helpers untouched (an unscoped reset would strip those too). The order is
    load-bearing: git consults same-scope entries in order and the empty value
    RESETS the accumulated list, so the reset must precede the addition.
    ``useHttpPath=true`` is what puts ``path=owner/repo.git`` in the helper's
    input, which is what the helper's allow-list check reads.
    """
    return {
        "GH_TOKEN": token,
        "GITHUB_TOKEN": token,
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_1": helper_command(),
        "GIT_CONFIG_KEY_2": "credential.https://github.com.useHttpPath",
        "GIT_CONFIG_VALUE_2": "true",
    }


#: One client per process per root, built lazily. NOT keyed per session: the
#: GrantCache inside is already keyed ``(key, session_id)``, and the refusal
#: document is per network. A ``None`` answer is deliberately NOT cached — a
#: share that arrives while this process lives must be seen by the next command
#: (the same staleness class ``MeshCredentialBroker``'s docstring describes on
#: the other side).
_GIT_CLIENTS: dict[str, Any] = {}
_GIT_CLIENTS_LOCK = threading.Lock()

#: Tokens this process has already scheduled a self-revoke for, by token string.
_SELF_REVOKE_SCHEDULED: set[str] = set()
_SELF_REVOKE_LOCK = threading.Lock()


def _client_for(root: Path | None) -> Any:
    from local_operator.network.credentials.client import MeshCredentialClient

    cache_key = str(root or "")
    with _GIT_CLIENTS_LOCK:
        client = _GIT_CLIENTS.get(cache_key)
        if client is not None:
            return client
    client = MeshCredentialClient.for_this_device(root)
    if client is not None:
        with _GIT_CLIENTS_LOCK:
            _GIT_CLIENTS[cache_key] = client
    return client


def _schedule_self_revoke(token: str, grant_exp_ms: int) -> None:
    """The borrower's best-effort belt: DELETE the token at its window end.

    Fires only while THIS process lives; the owner-side revoker is the primary,
    and the token's own ceiling is the floor under both. Duplicate scheduling is
    suppressed per token: the env path re-reads the same cached grant for every
    command in a window, and one timer per command would be a thread storm.
    """
    if not token:
        return
    with _SELF_REVOKE_LOCK:
        if token in _SELF_REVOKE_SCHEDULED:
            return
        _SELF_REVOKE_SCHEDULED.add(token)
    delay = max(0.1, grant_exp_ms / 1000.0 - time.time())

    def _fire() -> None:
        try:
            revoke_installation_token(token)
        finally:
            with _SELF_REVOKE_LOCK:
                _SELF_REVOKE_SCHEDULED.discard(token)

    timer = threading.Timer(delay, _fire)
    timer.daemon = True
    timer.start()


def revoke_installation_token(token: str, *, base_url: str | None = None) -> bool:
    """One borrower-side DELETE. Best effort; never raises.

    ``base_url`` is read at CALL time (``None`` means the module's real host), so
    a test's loopback GitHub stands in without touching the constant — the same
    reason the minter is injectable on the owner side.
    """
    try:
        return GithubMinter(base_url=base_url or GITHUB_API_BASE).revoke(token)
    except Exception:  # noqa: BLE001 — a belt, not a mechanism of record
        return False


def borrowed_git_env(
    *,
    root: Path | None = None,
    session_id: str = "",
    client: Any = None,
) -> tuple[dict[str, str], str]:
    """Fetch-on-use: ``(env, token)`` for one child, or ``({}, "")``. Never raises.

    ``{}`` is every not-now answer on purpose — this runs on the bash tool's
    spawn path, and a command must never fail because a borrow could not: the
    child then behaves exactly as it did before brokering existed. A refusal is
    left in the client's own cache for the surfaces that render sentences.

    ``session_id`` travels as ATTRIBUTION (the audit row's ``for_session``, the
    grant cache's partition) and nothing else; see the module docstring.
    """
    try:
        if client is None:
            client = _client_for(root)
        if client is None:
            return {}, ""
        if not client.should_borrow(GITHUB_KEY):
            return {}, ""
        outcome = client.request_grant_sync(GITHUB_KEY, session_id=session_id, provider=GITHUB_KEY)
    except Exception:  # noqa: BLE001 — see the docstring
        return {}, ""
    token = str(getattr(outcome, "access_token", "") or "")
    if not token:
        return {}, ""
    grant_exp_ms = int(getattr(outcome, "grant_expires_at_ms", 0) or 0)
    _schedule_self_revoke(token, grant_exp_ms)
    return git_env_for_token(token), token


# ---------------------------------------------------------------------------
# The git credential helper
# ---------------------------------------------------------------------------

#: The hosts the helper serves, exactly, lowercased. ``github.com:443``
#: is the spelling git writes when the URL carried an explicit port; both mean
#: the same HTTPS endpoint, and anything else — ``github.com.evil.com``, a
#: different host, a different scheme — is refused.
_ALLOWED_HOSTS = frozenset({"github.com", "github.com:443"})


def _parse_credential_request(lines: Iterable[str]) -> dict[str, str]:
    """``key=value`` lines from git's credential protocol, last one wins.

    Malformed lines are skipped rather than fatal: the protocol is line-oriented
    and only the three fields this helper reads (`protocol`, `host`, `path`)
    matter. ``wwwauth[]``/``capability[]``/``url`` and friends are ignored.
    """
    fields: dict[str, str] = {}
    for line in lines:
        text = str(line).rstrip("\r\n")
        if not text or text == "\n":
            continue
        name, sep, value = text.partition("=")
        if not sep:
            continue
        fields[name.strip().lower()] = value
    return fields


def _path_allowed(path: str, repositories: Sequence[str]) -> bool:
    """Whether ``path`` is a repo this helper may serve.

    FAIL-CLOSED EDGES FIRST: absent, empty or malformed paths are refused (a
    bare ``git credential fill`` has no path, and serving it would hand the
    token out for any interaction against github.com), and NO LIST CONFIGURED IS
    ALSO A REFUSAL — not a wildcard. The mint refuses to produce a token under
    an empty list in the first place, so a token that reaches a helper without a
    list is outside the mechanism; the mechanism never widens for it.
    """
    normalised = normalise_repository(path)
    if not normalised:
        return False
    return normalised in set(repositories)


def git_helper_reply(
    operation: str,
    request_lines: Iterable[str],
    *,
    token: str,
    repositories: Sequence[str] = (),
) -> str:
    """The helper's pure core: git's request in, git's response out (or ``""``).

    Semantics are the spike's, verbatim: serve ONLY ``protocol=https`` to
    ``github.com``/``github.com:443`` with a present, allow-listed path; ``get``
    prints the two lines git expects and NOTHING else; ``store``/``erase`` are
    no-ops; every refusal is silence with exit 0, because a credential helper's
    way of saying "I have nothing" is to print nothing — an error message would
    surface it as git's own failure. The helper never reads or writes a file,
    and the token it serves is whatever ``$GH_TOKEN`` its environment carries
    (the session injected it; a helper started bare has none and refuses).
    """
    op = str(operation or "").strip().lower()
    if op in ("store", "erase"):
        return ""
    if op != "get":
        return ""
    fields = _parse_credential_request(request_lines)
    if fields.get("protocol", "").lower() != "https":
        return ""
    if fields.get("host", "").lower() not in _ALLOWED_HOSTS:
        return ""
    if not _path_allowed(fields.get("path", ""), repositories):
        return ""
    if not token:
        return ""
    return f"username=x-access-token\npassword={token}\n"


def run_git_helper_cli(operation: str) -> int:
    """The ``lop credential git-helper`` entry point. Always exit 0, mostly silence."""
    import sys

    try:
        lines = sys.stdin.read().splitlines()
    except Exception:  # noqa: BLE001 — no stdin is "no request"
        lines = []
    reply = git_helper_reply(
        operation,
        lines,
        token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "",
        repositories=repositories_for(None),
    )
    if reply:
        sys.stdout.write(reply)
        sys.stdout.flush()
    return 0

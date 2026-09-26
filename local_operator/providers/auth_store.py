"""SQLite credential store + the API-key resolution cascade.

Credentials live in ``~/.local-operator/auth.db`` (0600, WAL, busy
timeout); no OS keychain is used.

The resolution cascade (``get_api_key``) is the no-auth-mode-switch design
First match wins:

1. runtime override (CLI ``--api-key``)
2. config override (``models.yml``/gateway pointer)
3. OAuth credential (auto-refresh + stickiness/round-robin)
4. API key persisted by ``login`` (``source="login"``)
5. env var — the process environment. After PR2a the plaintext
   ``credentials.env`` file is no longer read on this tier (the store's
   provider-class rows are, above), so an export is the only ambient source
6. stored API key without ``source="login"``
7. fallback resolver (custom providers)

Side effect preserved: session stickiness for a provider is cleared as soon
as resolution LEAVES the OAuth tier (before the env tier), so identity
lookups stop attributing OAuth accounts.

Refresh single-flight is a per-process ``asyncio.Lock`` PLUS a cross-process
lease in ``auth_credential_refresh_leases`` (same atomic upsert as
:meth:`UsageCacheStore.try_lease`). Two TUI+runtime processes racing a
rotating OAuth refresh token used to invalidate each other's new token
(PR-24); the loser now waits out the winner's write instead of POSTing a
consumed token. Clients are still per-process — only the refresh is leased.

Three more defences guard the same rotating token against REFRESH-TOKEN REUSE
DETECTION, which revokes the whole token family rather than failing one
request (a Radient grant died twice in ~17 hours on this machine):

* :data:`REFRESH_SEND_UNCONFIRMED_KEY` — a write-ahead marker, armed before
  the POST, recording that this row's refresh token was presented by an
  exchange whose outcome is not settled. A token whose send is unconfirmed is
  never presented again while that lasts: one exchange window
  (:data:`UNCONFIRMED_SEND_TTL_S`) for an outcome that never arrived, one block
  window for an answer that proved nothing, and not at all for a failure httpx
  reports as provably pre-send.
* :data:`AUTH_REFRESH_LEASE_MS` outliving
  :data:`PROVIDER_REFRESH_TOTAL_BUDGET_S` — the wall-clock cap this store puts
  on one exchange, which is NOT the providers' per-operation httpx timeout — so
  a peer's lease cannot expire inside the window where the holder's request may
  still be in flight.
* :data:`GRANT_DEAD_AT_KEY` — the IdP's own refusal, persisted on the row, so
  every row-reading surface says "sign-in expired" instead of "logged in", and
  only an interactive login clears it.

A caller that loses the lease is served the stored row, as it always was: a
peer's refresh is not a verdict about the credential, and the consumers that
must not spend a due bearer refuse it themselves after waiting for the peer.
What a marked row gets instead is a deferred, classified refusal
(:class:`RefreshUnconfirmedError`) rather than a presented token.
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from local_operator.paths import config_dir
from local_operator.providers.registry import (
    GetApiKeyFn,
    RefreshFn,
    credential_provider_id,
    get_provider_definition,
    provider_env_key,
)

if TYPE_CHECKING:  # the store path stays an optional, import-guarded argument
    from local_operator.providers.usage_cache import UsageCacheStore

logger = logging.getLogger("local_operator.providers.auth_store")

OAUTH_REFRESH_SKEW_MS = 60_000  # pre-emptive refresh trigger
DEFAULT_BLOCK_MS = 60_000  # rate-limit / 401 backoff

#: The per-operation budget a provider's refresh POST runs under. Every refresh
#: in ``providers/oauth/`` passes ``timeout=30.0`` (anthropic, kimi, openai,
#: xai, zai, qwencloud, radient all name the same number), and it is named here
#: rather than inline in each of them because it is the FLOOR for
#: :data:`PROVIDER_REFRESH_TOTAL_BUDGET_S`, and the two drifting apart is the
#: defect that derivation exists to prevent, not a style question.
#:
#: It is PER OPERATION and not a wall-clock budget: httpx applies it to connect,
#: to write and to each socket read separately, so a drippling endpoint can hold
#: one POST open for connect + write + read worth of it. That is not theory — see
#: :data:`PROVIDER_REFRESH_TOTAL_BUDGET_S`, where the measurement is recorded.
PROVIDER_REFRESH_HTTP_TIMEOUT_S = 30.0

#: The WALL-CLOCK budget the store itself imposes on one refresh exchange, the
#: number :data:`AUTH_REFRESH_LEASE_MS` is derived from.
#:
#: Why it has to exist at all: the providers' own ``timeout=30.0`` bounds each
#: OPERATION, not the request, so the pathological single POST is not 30 s. Two
#: measurements on this machine with the production client configuration against
#: an endpoint that stalls once and then dribbles: **41.4 s** (each read gap
#: under the per-op timeout) and **114.5 s** (three gaps). A lease sized from the
#: per-op number therefore does not outlive the request it guards — defect #2,
#: only harder to reach. ``mcp/auth.py`` carries the same lesson with its own
#: figure (a nominal 10 s timeout that produced a 140.7 s POST) and its answer is
#: the same shape: name the phases AND cap the total.
#:
#: 2x the per-op number: enough for connect + write + first read on a slow but
#: honest endpoint (the dripple case above is pathological, not slow), and small
#: enough that a stalled endpoint costs this account seconds rather than minutes.
PROVIDER_REFRESH_TOTAL_BUDGET_S = 60.0

#: How long a process may hold the cross-process refresh lease.
#:
#: The lease MUST outlive the request it guards, plus a margin. It used to be
#: 30 s — exactly the holder's per-operation HTTP timeout — so a peer's lease
#: expired at the same instant the holder's POST would time out, and the peer
#: then took the lease and re-presented the SAME rotating refresh token while the
#: holder's request could still be on the wire. Against a provider that rotates
#: and runs refresh-token reuse detection (Radient), that is the POST that
#: revokes the whole token family: the operator's grant died twice in ~17 hours
#: this way.
#:
#: Derived from :data:`PROVIDER_REFRESH_TOTAL_BUDGET_S` — the number the store
#: actually enforces on the exchange — plus one per-operation timeout of margin
#: for what the holder does around the POST (the row re-read, the guarded UPDATE,
#: the commit: sub-100 ms here, but a process can be descheduled for long
#: stretches on a loaded host). 90 s, not a multiple of the per-op number, which
#: is what the first revision of this change got wrong (review round 1, R4).
#:
#: The CEILING is what a CRASHED holder leaves behind. A peer that loses the
#: lease is handed the stored row — the behaviour every consumer's own join is
#: built on, see ``_served_while_contended`` — so a long TTL costs it a bounded
#: wait rather than a refusal, and 90 s stays inside what a caller already
#: tolerates from ``DEFAULT_BLOCK_MS`` plus one join.
AUTH_REFRESH_LEASE_MS = 90_000

#: Payload key recording that THIS row's refresh token was PRESENTED by an
#: exchange whose outcome is not settled — the request may have reached the IdP
#: and spent the token, and re-presenting a spent token is the reuse-detection
#: POST that revokes the family. Value is ``{"digest": <short digest of the
#: presented token>, "at": <epoch milliseconds>, "shape": <one of the
#: SEND_SHAPE_* constants>}``.
#:
#: Why a DIGEST rather than the token: the marker only ever answers "is the
#: token in this row the one an exchange is unsure about?", and a digest keeps
#: the payload from carrying the same live credential twice. It is not a
#: confidentiality measure — the token itself lives in the same row.
#:
#: Why it is ARMED BEFORE THE POST: a process that dies mid-request takes all
#: its in-memory knowledge with it, and the next boot would present a token that
#: may already be spent. The write-ahead marker is the only channel that
#: survives that death. The price is a false positive in the narrow window
#: between the arm and the wire, which is why the marker EXPIRES — and why the
#: SHAPE it stores decides for how long (:data:`UNCONFIRMED_SEND_TTL_S` for an
#: outcome that never arrived, :data:`ANSWERED_SEND_TTL_S` for one that did and
#: proved nothing).
#:
#: The same defence, for the same reason, already protects MCP OAuth grants
#: (``mcp.auth.GRANT_UNCONFIRMED_SEND_KEY``). Deliberately a DIFFERENT key from
#: MCP's while every other name here is shared: the two subsystems never write
#: the same row, but a marker key that read another subsystem's marker would
#: suppress a refresh on no evidence at all, and this one is cheap to keep
#: distinct.
REFRESH_SEND_UNCONFIRMED_KEY = "refresh_send_unconfirmed"

#: Payload key: the IdP REFUSED this row's grant for good. Value is the epoch
#: milliseconds at which the refusal was recorded.
#:
#: It is persisted rather than merely raised so a row-reading surface — the
#: usage panel, the model picker, the desktop status routes — can say "sign-in
#: expired" from the row alone, with no live refresh attempt to re-earn the
#: verdict. That is what makes the verdict honest on a row whose access token
#: has not expired yet, which is exactly the state the old code reported as a
#: healthy login.
#:
#: Only an INTERACTIVE login clears it: :meth:`AuthStore.upsert_credential`
#: replaces the payload wholesale, so a token minted by a refresh can never
#: resurrect the family this marker says the IdP revoked (a refresh landing
#: after the tombstone is refused, not persisted — see
#: ``AuthStore._ensure_oauth_fresh``).
#:
#: A payload key rather than ``disabled_cause``, deliberately. A
#: ``disabled_cause`` row is FILTERED OUT of :meth:`AuthStore.list_credentials`,
#: which would make the dead login vanish from every panel that has to name the
#: account and offer the remedy — the opposite of what the row is for, and it
#: contradicts the documented rule that a dead grant is REPORTED, never retired
#: (see :class:`CredentialInvalidError` and ``AuthStore.list_oauth_accesses``).
#:
#: Spelled the same as ``mcp.auth.GRANT_DEAD_AT_KEY`` because it is the same
#: concept, and in MILLISECONDS here where MCP stores seconds: each value is
#: written and read by one subsystem only (a row belongs to exactly one of
#: them) and the unit follows its writer's clock helper, ``_now_ms``.
GRANT_DEAD_AT_KEY = "grant_dead_at"

#: How long an IN-DOUBT "presented but unacknowledged" send marker stays live
#: (see :data:`REFRESH_SEND_UNCONFIRMED_KEY` and :data:`SEND_SHAPE_UNKNOWN`).
#:
#: The marker exists to stop us re-presenting a token an exchange may already
#: have spent, so the only window it may cover is the one in which such an
#: exchange could still be running — and that window is already named here: the
#: invariant is *never re-present a token while an exchange may be in flight
#: anywhere*, and :data:`AUTH_REFRESH_LEASE_MS` is the store's own statement of
#: it (the wall-clock cap on one exchange plus a per-operation timeout of margin
#: for the holder's descheduling). DERIVED from those two numbers rather than
#: typed, so this bound cannot drift below the window it must outlive: 90 + 30 =
#: 120 s. The alternative — a per-op timeout of margin on its own — is the
#: arithmetic that produced the lease's own defect, where two constants that had
#: to keep an order drifted apart.
#:
#: SUPERSEDES the deliberate one-hour bound of review round 1, R3, and the thing
#: that bound got wrong is what makes this a defect fix rather than a tuning: an
#: hour is not "at most one sign-in", it is LONGER THAN THE BEARER IT IS ARMED
#: AGAINST. Radient mints ``expires_in = 3600`` minus a 5-minute skew
#: (``providers/oauth/radient.py``) — 55 minutes of usable bearer — and a refresh
#: is triggered exactly when that bearer is due or expired, which is the moment a
#: marker gets armed. So a marker armed then outlives the bearer by construction,
#: and from bearer death to marker expiry there is no usable bearer and no
#: permitted refresh: a guaranteed outage, not the "degraded" state the hour was
#: meant to buy. Measured on the operator's machine: 640 consecutive poll
#: refusals (~1.8 h at the connector's 10 s cadence) on a credential whose bearer
#: had already expired, while every surface reported an expired login.
#:
#: 120 s is strictly inside the bearer's own life, so the worst case after one
#: unconfirmed send is a ~2-minute deferral that clears itself, with two honest
#: exits: the refresh lands (the token was never spent), or the next attempt
#: earns ``invalid_grant``, the tombstone is written and the surfaces say "sign in
#: again" — the same verdict the hour produced, 58 minutes earlier.
#:
#: The two boundaries are pinned by tests so a later edit cannot re-break either:
#: GREATER than :data:`AUTH_REFRESH_LEASE_MS` (a peer that takes the lease the
#: instant it expires arms a marker of its own for the same token, and a bound
#: that lapsed before that peer's exchange could still be on the wire would
#: re-present a token that may already be spent), GREATER than
#: :data:`DEFAULT_BLOCK_MS` (below it, the block ``_resolve`` writes would lift
#: while this marker still suppressed the row), and SHORTER than the 55-minute
#: bearer the marker is armed against.
#:
#: It bounds ONE DEFERRAL, NOT THE STATE, and that distinction matters to every
#: sentence that quotes it: the marker is armed before EVERY POST, so an endpoint
#: that stalls past the budget on each attempt arms a fresh window each time and a
#: repeating stall keeps the row deferred in ~2-minute cycles. The copy therefore
#: carries the escalation for the persisting case rather than an absolute that says
#: nothing else is ever needed.
#:
#: What this costs, stated plainly: the round-1 property "you have an hour to
#: notice and sign in" is gone. What it buys: an outage becomes a deferral.
UNCONFIRMED_SEND_TTL_S = AUTH_REFRESH_LEASE_MS / 1000 + PROVIDER_REFRESH_HTTP_TIMEOUT_S

#: How long an ANSWERED send marker stays live — :data:`SEND_SHAPE_ANSWERED`.
#:
#: An endpoint that ANSWERED (a 5xx, a 429, a non-terminal 4xx) failed to prove
#: anything about our token, so the marker is kept — a provider that commits the
#: rotation and THEN fails the response leaves the row holding a spent token, and
#: re-presenting that is the family-revoking POST. But its evidence is much
#: weaker than the in-doubt shape's: the endpoint was reachable and its own code
#: decided what to say, which is the ordinary "the provider had a bad minute"
#: shape this repo's own merged e2e test calls "not the account's fault: retry,
#: do not re-sign-in". Believing the marker for an hour turned that into a
#: suppressed account until the user signed in again (review round 1, R3).
#:
#: ``DEFAULT_BLOCK_MS`` — the window the cascade ALREADY refuses a credential
#: whose refresh failed for, in every process sharing the DB. So the observable
#: healing cadence for a transient fault is the one it always had: the account
#: is out for a block window, then the next attempt may present the token again.
#: What changes is only that the diagnosis does not re-POST inside that window.
ANSWERED_SEND_TTL_S = DEFAULT_BLOCK_MS / 1000

#: Marker shapes, stored IN the marker so the two bounds above cannot be
#: confused at read time. ``SEND_SHAPE_UNKNOWN`` is the exchange's outcome never
#: arriving; ``SEND_SHAPE_ANSWERED`` is an answer that proves nothing about our
#: token. A THIRD shape — provably never sent — is not stored at all: it is
#: resolved on the spot by clearing the marker (see ``_refresh_request_never_sent``).
SEND_SHAPE_UNKNOWN = "unknown"
SEND_SHAPE_ANSWERED = "answered"

#: Hard ceiling on ANY credential block, whatever computed it (a usage
#: reset estimate, a provider Retry-After header, a caller's block_ms). A
#: block is a cost-avoidance backoff, not a correctness gate -- every message
#: boundary re-probes usage and the cascade re-checks blocked rows -- so no
#: reading, however large or however hostile the header, may strand an
#: account for more than this. An hour outlives a working session's need to
#: stop re-hitting a spent account, and is short enough that a wrong estimate
#: self-heals on the next boundary. Guards the days-long block a raw weekly
#: reset or a Retry-After: 604800 would otherwise write.
MAX_CREDENTIAL_BLOCK_MS = 60 * 60 * 1000  # 1 hour

#: How long a provider-fault demotion keeps a credential at the back of the pool
#: (see ``AuthStore.deprioritize_credential``). Deliberately short: the mark says
#: "this account was failing a moment ago", which stops being useful information
#: quickly, and it cannot expire by being USED -- a demoted row sorts last, so it
#: is not selected, so it never earns the success that would clear it. Two
#: minutes outlives a burst of 529s without outliving the outage that caused it.
DEPRIORITIZE_TTL_MS = 120_000

#: How far (in remaining-quota fraction) an account may trail the least-loaded
#: account and still be treated as EQUALLY good by the first-resolve pick (see
#: ``AuthStore._usage_ranked_order``). Ten points, because the pick's job is
#: to keep the pool from skewing to the extremes observed in the field (three
#: accounts at 65-99% of their 5-hour window while two sat at 6% and 29%),
#: not to chase the single emptiest account: several sessions and subagents
#: start within the same minute, and with a zero-width bucket every one of
#: them would land on the same row and drain it. Within the bucket the
#: existing per-session hash spreads them.
USAGE_PICK_TOLERANCE = 0.10

#: A cached usage report older than this is treated as UNKNOWN by the
#: first-resolve pick rather than trusted -- UNLESS every window it measures
#: still names a reset in the future, in which case it stays trusted at any
#: age (see ``AuthStore._cached_remaining_fraction``). Thirty minutes: the
#: reports come from the message-boundary preflight of whichever sessions
#: are active, so on a busy machine they are minutes old. The exception is
#: what makes the cutoff safe in both directions: usage inside a rolling
#: window only rises until the window resets, so an old report whose windows
#: have not rolled over is a LOWER bound on how full the account is, and
#: discarding it let a 99%-full account with a two-hour-old report rank as
#: neutral and collect most of the new sessions (observed, live). A report
#: whose window HAS since reset says nothing about the account now, and one
#: with no reset timestamps cannot be told apart from it. Longer than the
#: report TTL on purpose -- the cache serves expired rows precisely so a
#: stale-but-recent answer beats no answer.
USAGE_PICK_MAX_REPORT_AGE_MS = 30 * 60_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_credentials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  credential_type TEXT NOT NULL,
  data TEXT NOT NULL,
  disabled_cause TEXT DEFAULT NULL,
  identity_key TEXT DEFAULT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
CREATE INDEX IF NOT EXISTS idx_auth_provider_identity
  ON auth_credentials(provider, identity_key) WHERE identity_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS auth_credential_blocks (
  credential_id INTEGER NOT NULL,
  provider_key TEXT NOT NULL,
  block_scope TEXT NOT NULL DEFAULT '',
  blocked_until_ms INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (credential_id, provider_key, block_scope)
);
CREATE INDEX IF NOT EXISTS idx_auth_credential_blocks_expires
  ON auth_credential_blocks(blocked_until_ms);

CREATE TABLE IF NOT EXISTS auth_credential_refresh_leases (
  credential_id INTEGER PRIMARY KEY,
  holder TEXT NOT NULL,
  expires_at_ms INTEGER NOT NULL
);
"""


class AuthStoreError(Exception):
    """Credential resolution/refresh failure (never a bare sqlite error)."""


class CredentialInvalidError(AuthStoreError):
    """The row's grant is dead; retrying cannot revive it, only a re-login can.

    The store-level face of
    :class:`~local_operator.providers.oauth.callback_server.InvalidGrantError`.
    It stays a subclass of :class:`AuthStoreError` so every existing handler
    (the cascade rotating to a sibling, ``ensure_oauth_fresh`` swallowing to
    ``None``) behaves exactly as before; callers that must tell a permanent
    failure from an outage -- ``/usage`` is the first -- catch this instead.

    Raising it is a STATEMENT ABOUT THE GRANT, not a routing decision: nothing
    here disables, blocks or deletes the row. The login is still the user's,
    and the panel keeps showing it.
    """


class RefreshUnconfirmedError(AuthStoreError):
    """A refresh was NOT attempted this time because the token is in doubt.

    Raised instead of presenting a refresh token whose LAST exchange is not
    settled — either its outcome never arrived (:data:`SEND_SHAPE_UNKNOWN`) or an
    answer arrived that proved nothing about it (:data:`SEND_SHAPE_ANSWERED`).
    Presenting it again is the reuse-detection POST that revokes the whole token
    family, so it is deferred instead.

    Deliberately NOT a :class:`CredentialInvalidError`: nothing here says the
    credential is bad, and the remedy is not "sign in again" even though that is
    the only way to recover *immediately* — the state also self-heals at the
    marker's expiry (:data:`UNCONFIRMED_SEND_TTL_S` / :data:`ANSWERED_SEND_TTL_S`),
    which is why the message names both. Since
    :data:`UNCONFIRMED_SEND_TTL_S` is bounded by one exchange window rather than
    by an hour, that self-healing exit is now the ordinary one and a sign-in is
    the ESCALATION for a deferral that outlives it: the message says so, and the
    surfaces that render it (``tunnels/gateway.py``'s ``RELAY_DETAIL`` and
    ``tunnels/cli.py``'s status line) say so too, because "your login expired,
    sign in" is what this state used to be reported as and it was wrong twice —
    the check had run, and nothing about the login had changed. A caller that
    switches on the class (``list_oauth_accesses`` does, to log the state rather
    than debug-swallow it) can tell "deferred" from "this credential is
    unusable", which is the same distinction the MCP sibling draws with
    ``McpRefreshContendedError``.

    ``_resolve`` treats it as an ordinary failed refresh — a ``DEFAULT_BLOCK_MS``
    block and move on — and that is deliberate after review round 1. The state it
    describes IS a credential that cannot mint a bearer right now, which is
    exactly what the block is for, and the merged failover test
    (``test_its_credential_read_neither_blocks_nor_repoints_the_turn``) needs an
    ordinary request to still block, or its control stops discriminating. What a
    peer's IN-FLIGHT refresh gets is different and is not this: it is served the
    stored row, because a peer working is not a verdict about anything (see
    ``_served_while_contended``).
    """


@dataclasses.dataclass
class StoredCredential:
    """One row of ``auth_credentials`` with its parsed payload."""

    id: int
    provider: str
    credential_type: str  # 'api_key' | 'oauth'
    data: dict[str, Any]
    disabled_cause: str | None = None
    identity_key: str | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclasses.dataclass(frozen=True)
class OAuthAccess:
    """The identity-carrying credential record handed to wire clients.

    ``get_oauth_access()``: everything a provider-specific request
    shaper needs beyond the bare bearer — which account/org pays, and
    whether the token is OAuth (needs provider-specific auth headers/routes)
    or a plain API key.
    """

    access_token: str
    credential_id: int
    account_id: str | None = None
    email: str | None = None
    org_id: str | None = None
    api_endpoint: str | None = None
    kind: str = "oauth"  # 'oauth' | 'api_key'
    #: The full stored credential dict, when this row is OAuth. The wire's
    #: bearer is ``access_token`` (already mapped through the provider's
    #: ``get_api_key``), but some providers split duties across two tokens —
    #: the QwenCloud Token Plan infers on a pasted ``sk-sp-…`` key while quota
    #: needs the OAuth ``access`` — and a usage fetcher must see the raw row
    #: to spend the right one. None for plain API-key rows.
    raw: dict[str, Any] | None = None
    #: True when this row's stored grant was refused as permanently dead, so
    #: no bearer could be minted and none ever will be until the user runs
    #: ``/login <provider>`` again. Carried on the identity record rather than
    #: signalled by omission because omission is exactly what made this state
    #: indistinguishable from a transient outage: the row simply vanished
    #: from the usage fetch and the panel reported stale numbers forever.
    #: ``access_token`` is empty whenever this is set.
    credential_invalid: bool = False


def default_db_path() -> Path:
    return config_dir() / "auth.db"


#: Process-level stores handed out by :func:`shared_auth_store`.
#:
#: Keyed on everything that can change an ANSWER, not merely on the file: the
#: resolved ``auth.db`` path, the canonical config ROOT the env tier reads its
#: provider-class store rows from, and the config-derived override tier. Two
#: callers that would resolve a provider's key differently must not be handed
#: one store — see :func:`shared_auth_store`.
#:
#: Ordered, and bounded at :data:`_SHARED_STORES_MAX`. The bound is what keeps
#: this from trading a per-call connection for a per-ROOT one that is never
#: released: a test session reaches the classification leg with a fresh
#: ``tmp_path`` per test, so an unbounded map grows a connection (three file
#: descriptors with WAL sidecars) per test in a worker. That is the exact shape
#: that already hit ``EMFILE`` under the default 256-descriptor limit here — see
#: :meth:`AuthStore._usage_ranked_order`, which closes the usage cache per
#: ranking for this reason. Eight is far above any real process's credential
#: roots (one, plus an isolated or server root) and 24 descriptors at worst.
_SHARED_STORES: (
    "OrderedDict[tuple[str, str, tuple[tuple[str, str], ...], tuple[int, int] | None], AuthStore]"
) = OrderedDict()
_SHARED_STORES_MAX = 8

#: Guards the map, and is held ACROSS the ``AuthStore()`` construction inside
#: :func:`shared_auth_store` — which is the point rather than an accident. The
#: construction is the expensive part (connect + 3 PRAGMAs + schema DDL + 3
#: chmods, measured at ~1.6 ms and up to 9.2 ms under fleet load), so two
#: threads racing for the first store must produce ONE connection, not two.
_SHARED_STORES_LOCK = threading.Lock()


def _canonical_config_root(config_root: Path | None) -> Path:
    """The config root a store built with ``config_root`` will actually read.

    ``None`` is NOT a third root. Every consumer of this argument resolves it as
    ``config_dir()`` — ``secrets/keys.secrets_dir`` is literally
    ``(base if base is not None else config_dir())`` — so keying ``None`` apart
    from ``config_dir()`` would open two connections to one root for no
    answer-level reason, which is the duplication this accessor exists to
    remove. Canonicalising is therefore exactly answer-preserving rather than
    merely convenient: the two spellings denote the same directory by
    construction.
    """
    return config_root if config_root is not None else config_dir()


def _file_identity(path: Path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` for the file at ``path``, or ``None`` if it is absent.

    The accessor keys on this, and it is a CORRECTNESS component rather than
    bookkeeping. A live SQLite connection is bound to the INODE it opened, not
    to the path it was opened by: if ``auth.db`` is deleted and recreated — a
    logout that wipes it, a restored backup, ``VACUUM``, or a test suite that
    recycles a temp directory — a cached connection keeps answering from the
    unlinked old file, and every new row written to the replacement is
    invisible to it. Measured, in the classification suite: two parametrized
    cases sharing a recreated ``tmp_path`` had a fresh connection see the new
    row while the shared connection still answered from the deleted one, which
    surfaced as a credential "stored by login" that the cascade could not find.

    So the identity is part of the key: a replacement moves the key, the next
    acquisition builds a store on the current file, and the stale entry is left
    to the map's bound. On a platform or filesystem that does not fill
    ``st_ino`` this degrades to the path-only behaviour it replaces rather than
    failing.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


def shared_auth_store(
    db_path: str | Path | None = None,
    *,
    config_dir: Path | None = None,
    config_overrides: dict[str, str] | None = None,
) -> AuthStore:
    """The process's store for one credential root — reused, not reopened.

    WHY THIS EXISTS. Constructions that only READ through the cascade were
    opening a fresh SQLite connection each: measured on this tree at seven
    connections to one ``auth.db`` per boot (``classification/vendors.py`` x3,
    ``model/configure.py`` x3, ``session_factory.py`` x1) and six per cold
    prompt, each paying connect + three ``PRAGMA`` + the schema
    ``executescript`` + three ``chmod``. They all resolve the same file with the
    same tier configuration, so the connection is shared state that was being
    rebuilt per call.

    WHO MAY USE IT, AND WHO MUST NOT.
    Only a caller whose use is a bounded read through the cascade and which
    does NOT own a close. The distinction is not stylistic — it is the one
    :func:`_release_detached_refresh` and :meth:`AuthStore.detached_refresh`
    record, and the sites that must keep their own store are:

    * The tunnels trio (``tunnels/api.py``, ``tunnels/report.py``), which sit
      inside ``with closing(AuthStore())`` precisely so the connection dies with
      the bounded call. They are the callers whose detached exchange gets its
      OWN store (:meth:`AuthStore.detached_refresh`), which is a correctness
      requirement rather than tidiness.
    * The session's store (``session_factory.py``), which
      :func:`~local_operator.session_factory.attach_auth_dispose` CLOSES on
      ``session.dispose()``. Its key would otherwise be indistinguishable from
      the classification legs' — same file, same root — and disposing one
      session would then close the store under every other reader in the
      process. That is the concrete form of the hazard
      :func:`_release_detached_refresh` warns about, and it is why ownership,
      not just the path, decides what may share.

    CLOSING. Nothing that receives this store may close it; the process's
    teardown is :func:`close_shared_auth_stores`. A caller that closes it
    anyway cannot poison the process, which is the deliberate half of the
    choice: use-after-close raises ``sqlite3.ProgrammingError`` ("Cannot operate
    on a closed database"), so this accessor REOPENS a store it finds closed
    rather than handing the corpse on. The residual window is the one no
    accessor can close — a caller already INSIDE a call when another thread
    closes — and that is why the contract is "never close what you were
    handed" rather than a promise that closing is harmless.

    THREADING. Sharing is what this store was built for, so the consolidation
    makes an existing property real rather than adding one: the connection is
    opened with ``check_same_thread=False`` and every statement goes through
    :class:`_SerializedConnection`'s reentrant mutex (see :meth:`_connect`),
    which exists so a caller on a worker thread gets a correct answer instead of
    a ``ProgrammingError``. The store already holds process-scoped routing state
    shared by every session in the process (``_round_robin``,
    ``_deprioritized``), and its per-process refresh single-flight
    (``_refresh_locks``) is per STORE — so one shared store is strictly better
    covered than three: the three classification legs previously held three
    independent "single-flight" locks for one credential. It holds no per-call
    state that a second caller could corrupt: a cursor is created and consumed
    inside one serialized statement, and every write path commits in the same
    call, so no transaction is left open across calls (checked over all thirteen
    ``commit`` sites).

    ``usage_cache`` is deliberately NOT accepted: it is an injection seam for
    tests, a per-store handle, and a caller that needs its own has no business
    sharing one.

    The map is bounded at :data:`_SHARED_STORES_MAX` roots and evicts by
    dropping its reference, never by closing — see the note there. A process
    with more distinct credential roots than that keeps working; it just stops
    reusing the least recently used ones. An entry is also keyed on the file's
    IDENTITY and not only its path, so a deleted-and-recreated ``auth.db`` is
    never served from the old connection — see :func:`_file_identity`.
    """
    resolved_db = Path(db_path) if db_path is not None else default_db_path()
    base_key = (
        str(resolved_db),
        str(_canonical_config_root(config_dir)),
        tuple(sorted((config_overrides or {}).items())),
    )
    with _SHARED_STORES_LOCK:
        # Stat BEFORE the lookup so a replaced file is a miss rather than a
        # stale hit (see :func:`_file_identity`), and again after construction
        # so the entry is filed under the inode the store actually opened —
        # which for a first call is the one the constructor just created.
        key = (*base_key, _file_identity(resolved_db))
        store = _SHARED_STORES.get(key)
        if store is None or store.closed:
            # Reopen rather than hand back a closed handle (see CLOSING above);
            # a fresh AuthStore also re-arms the lazily-built usage cache, which
            # ``close()`` had dropped.
            store = AuthStore(resolved_db, config_dir=config_dir, config_overrides=config_overrides)
            key = (*base_key, _file_identity(resolved_db))
            _SHARED_STORES[key] = store
        _SHARED_STORES.move_to_end(key)
        while len(_SHARED_STORES) > _SHARED_STORES_MAX:
            evicted_key, evicted = _SHARED_STORES.popitem(last=False)
            if evicted is store:
                # Cannot happen with a max of 2 or more, but a bound of 1 would
                # otherwise evict the store this call is about to return.
                _SHARED_STORES[evicted_key] = evicted
                break
            # DROPPED, never closed, and that is the safe direction: a caller
            # that is mid-call (or awaiting a refresh) still holds a reference,
            # so the store stays alive and valid for it, and only the map's own
            # reference goes away. Closing here would hand that caller the
            # ``ProgrammingError`` this accessor exists to prevent. The
            # descriptors return by refcount once the last holder lets go, which
            # is what bounds the process.
            logger.debug(
                "shared auth store: evicting %s (over %d roots)", evicted_key[0], _SHARED_STORES_MAX
            )
        return store


def close_shared_auth_stores() -> None:
    """Close every process-level store. The only legitimate closer of one.

    Registered with :mod:`atexit` so a process that used a shared store releases
    its file descriptor and WAL sidecars deterministically instead of relying on
    the interpreter to reclaim them. Closing at exit cannot cut short a live
    exchange: the detached refreshes that must outlive their caller own their
    own store and are closed by :func:`_release_detached_refresh`, and atexit
    runs after every non-daemon thread has finished.

    Exceptions are swallowed and the reference is dropped regardless. Shutdown
    is the wrong moment to raise — an already-broken handle must not turn a
    clean exit into a traceback — and dropping the reference is what matters,
    since the process is ending either way.
    """
    with _SHARED_STORES_LOCK:
        stores = list(_SHARED_STORES.values())
        _SHARED_STORES.clear()
    for store in stores:
        try:
            store.close()
        except Exception:  # noqa: BLE001 — a broken handle is not a shutdown failure
            logger.debug("shared auth store: close at teardown failed", exc_info=True)


atexit.register(close_shared_auth_stores)


def _identity_key_for(provider: str, credential: dict[str, Any]) -> str | None:
    """Dedupe key so one account holds one row (org scope ⇒ separate rows).

    API keys and CLI-stored credentials never dedupe (each key is its own
    row). OAuth payloads dedupe on the account identity when the IdP returns
    one; when it does not (Kimi returns none, xAI/Anthropic only with an
    id_token), a deterministic per-provider constant keeps re-login on ONE
    row — otherwise two logins leave two rows and the older one carries a
    dead rotated refresh token that the cascade keeps selecting.
    """
    if credential.get("type") == "api_key" or credential.get("source") == "login":
        return None
    for field in ("org_id", "account_id", "email", "project_id"):
        value = credential.get(field)
        if value:
            return str(value)
    # The per-provider constant asks "is this an OAuth credential?", and it used
    # to answer by testing for a refresh token. That is the same blind spot the
    # type derivation above had: an OAuth credential whose token NEVER EXPIRES
    # carries no refresh token by design (Z.AI's coding-plan sign-in mints
    # exactly that), so it fell through to `None` and every re-login left
    # another row. Five sign-ins meant five rows, and `/usage` rendered the one
    # account five times over.
    #
    # A declared type answers the question directly; the refresh/access pair
    # stays as the fallback for the callers that declare nothing, which is how
    # every existing provider reaches this line.
    if credential.get("type") == "oauth" and credential.get("access"):
        return f"oauth:{provider}"
    if credential.get("refresh") and credential.get("access"):
        return f"oauth:{provider}"
    return None


def _send_marker_owner(marker: Any) -> tuple[Any, Any, Any] | None:
    """The identity of the exchange that owns ``marker``, or ``None`` if unreadable.

    ``(holder, at, digest)`` — and deliberately NOT the whole dict. ``shape`` is
    the one field the owning exchange is allowed to rewrite when its send lands,
    so an ownership test that compared whole dicts would refuse the very rebind
    the arm authorised: the marker it wrote comes back carrying a different shape
    and an equal check would stop calling it "its own". The three fields that do
    identify the exchange are the write-ahead stamp, the token it names, and the
    ``pid:uuid`` the lease rows already carry.
    """
    if not isinstance(marker, dict):
        return None
    return (marker.get("holder"), marker.get("at"), marker.get("digest"))


def _refresh_token_digest(refresh_token: str) -> str:
    """Short stable digest of one refresh token, for the send marker.

    Never a credential in its own right: it only ever answers "is the token in
    this row the one an exchange presented?", so keeping the digest rather than
    a second copy of the token keeps the payload from carrying the same live
    secret twice. Truncated, because the question compares a handful of our own
    tokens rather than searching an adversarial keyspace.

    The same helper name exists in ``mcp.auth``, which guards the same
    reuse-detection POST for MCP grants. Not imported from there on purpose:
    that would put the MCP package on the provider store's import path for
    fourteen lines of hashing, and the two markers must stay independently
    evolvable.
    """
    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:16]


def _send_marker_is_live(marker: Any, refresh_token: str | None, *, now_ms: int) -> bool:
    """Whether a send marker says ``refresh_token`` may already be spent.

    ``True`` only for a WELL-FORMED, UNEXPIRED marker whose digest belongs to
    this token. Every other shape is ``False``, and the caller clears the marker
    on the way past rather than believing it:

    * a marker whose digest belongs to a DIFFERENT token is stale — a peer has
      rotated the row since — and suppressing a healthy, newly rotated token
      because of an unrelated send is a false positive the user pays for with a
      browser visit;
    * expiry is what stops a marker that never resolved from suppressing an
      account's refresh indefinitely, and the BOUND DEPENDS ON THE SHAPE the
      marker stores: a full :data:`UNCONFIRMED_SEND_TTL_S` for an outcome that
      never arrived, :data:`ANSWERED_SEND_TTL_S` for an answer that proved
      nothing (review round 1, R3);
    * a malformed marker — or one whose ``shape`` is not a known value, which can
      only come from a hand-edited or future payload — cannot be judged, so it is
      NOT believed. Unknown has to fall on the side that keeps trying.
    """
    if not isinstance(marker, dict):
        return False
    at = marker.get("at")
    digest = marker.get("digest")
    shape = marker.get("shape")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return False
    if shape == SEND_SHAPE_ANSWERED:
        ttl_ms = ANSWERED_SEND_TTL_S * 1000
    elif shape == SEND_SHAPE_UNKNOWN:
        ttl_ms = UNCONFIRMED_SEND_TTL_S * 1000
    else:
        return False
    if not 0 <= now_ms - at <= ttl_ms:
        return False
    return bool(refresh_token) and digest == _refresh_token_digest(refresh_token)


def _self_clearing_minutes() -> int:
    """The deferral's own bound in whole minutes, for the sentences that name it.

    Derived from the constant rather than typed into the messages that quote it:
    the reason :data:`UNCONFIRMED_SEND_TTL_S` is derived at all is that a reader
    must be told a window that matches the rule, and a hard-coded "2 minutes" in
    a log line is the same drift in prose. It quotes the LONGEST a single deferral
    can last (the :data:`SEND_SHAPE_ANSWERED` shape clears sooner).

    ONE DEFERRAL, NOT THE STATE, and the first revision of this docstring claimed
    more than the number supports (UX round 1, U1; design round 1, D5): a stalling
    endpoint that again gets no answer after this window expires arms a FRESH
    marker, because the marker is armed before every POST, so a repeating stall
    keeps renewing it. The window is therefore an upper bound on one deferral, and
    every sentence built on it is an upper bound on one deferral — which is why
    the user-facing copy carries the escalation for the case it persists rather
    than an absolute that says nothing else is ever needed.
    """
    return int(UNCONFIRMED_SEND_TTL_S // 60)


#: How the deferral's bound is SPELLED in the sentences that quote it. Words,
#: because every surface a person reads spells it that way ("about two minutes"),
#: and a log line reading "about 2 minutes" beside a terminal reading "about two
#: minutes" is two voices for one fact (design round 1, D5). Past ten the words
#: stop being clearer than the digit, so a bound that large falls back to it.
_SPELLED_MINUTES = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def self_clearing_window() -> str:
    """The deferral's bound as a spelled window (``"two minutes"``).

    The single spelling every message in this module uses, so the surfaces that name
    the window read as one voice, and so a changed bound changes all of them at once.
    Public because the window is quoted OUTSIDE this module too — `tunnels/cli.py`
    for the `Login:` line, when the screen has no other carrier for it — and that
    caller must derive it rather than type it (agent review round 1, R2).
    """
    minutes = _self_clearing_minutes()
    return f"{_SPELLED_MINUTES.get(minutes, minutes)} minutes"


def _refresh_request_never_sent(exc: BaseException) -> bool:
    """Whether an ``httpx`` failure happened BEFORE the request reached the wire.

    The distinction is httpx's own, not a guess, and it is load-bearing here for
    the same reason it is in ``mcp/auth.py``, which owns this rule: a refresh
    token presented to a rotating provider is spent by the REQUEST, so a failure
    that happened before the request was written leaves the token untouched and
    is an ordinary transient retry, while one that happened after may have spent
    it and must not be re-presented without an interactive sign-in.

    Pre-send, per httpx's taxonomy: ``ConnectError`` (refused, DNS, TLS
    handshake), ``ConnectTimeout`` / ``PoolTimeout`` (waiting for a connection or
    a pool slot), ``UnsupportedProtocol`` and ``LocalProtocolError`` (the request
    was never even formatted for a socket). Everything else — ``ReadTimeout``,
    ``ReadError``, ``WriteError``, ``WriteTimeout``, ``RemoteProtocolError``, and
    the builtin ``TimeoutError`` the store's own total budget raises — can have
    been written, so it is treated as suspect. The asymmetry is deliberate and is
    the marker's whole trade: a wrongly-suspect failure costs a bounded wait, a
    wrongly-trusted one costs the whole token family.

    The first revision of this change omitted this rule entirely, which made a
    token endpoint on a CLOSED PORT suppress the account for the marker's whole
    window over a request nothing ever received (review round 1, R2). The annotation is
    ``BaseException`` because the predicate is TOTAL over exceptions by
    construction: it answers "was this demonstrably pre-send?", and anything it
    does not recognise — including ``asyncio.CancelledError`` — answers ``False``,
    i.e. suspect, the conservative answer. That is why the caller needs no special
    case for cancellation: the store's own handler cannot catch a
    ``CancelledError``, so a cancelled send keeps the marker by PROPAGATION, while
    this predicate — were it ever asked — would keep it too. This paragraph must
    not claim a classification the caller never requests: it is the sibling's own
    round-2 correction, mirrored rather than re-derived, because the first draft of
    this docstring copied the wording that correction struck (review round 2, N1).
    """
    import httpx

    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.UnsupportedProtocol,
            httpx.LocalProtocolError,
        ),
    )


def _caused_by_never_sent(exc: BaseException) -> bool:
    """``_refresh_request_never_sent`` over the whole ``__cause__``/``__context__``
    chain.

    A provider's refresh fn may wrap the transport failure in its own error
    (``LoginError``), and the chain is the only place the httpx type survives.
    Bounded at a handful of links so a self-referential chain cannot spin, and it
    reports ``False`` the moment it cannot see a pre-send cause — the
    conservative direction, matching the predicate it wraps.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    for _ in range(8):
        if node is None or id(node) in seen:
            return False
        seen.add(id(node))
        if _refresh_request_never_sent(node):
            return True
        node = node.__cause__ or node.__context__
    return False


#: Refresh exchanges a bounded caller has handed off to this module (see
#: :meth:`AuthStore.detached_refresh`).
#:
#: A STRONG reference, and it is load-bearing rather than tidiness: asyncio keeps
#: only a weak reference to a task, so a bare ``create_task`` whose result nobody
#: awaits can be collected mid-await. Here that would cancel an exchange after its
#: marker was armed and before its answer arrived — manufacturing, in the
#: supervisor meant to prevent it, exactly the in-doubt state the marker exists to
#: record. The done-callback drops each task from this set (which is what keeps it
#: bounded) and closes the store the exchange ran on.
_DETACHED_REFRESHES: set[asyncio.Task[dict[str, Any] | None]] = set()


def _release_detached_refresh(task: asyncio.Task[dict[str, Any] | None], store: AuthStore) -> None:
    """Retire a detached exchange: drop the reference, close its own store.

    The store is closed HERE, by the supervisor, and never by the caller: the
    whole point of a detached exchange is that it outlives the call that started
    it, and a caller's ``closing(AuthStore())`` would close the connection under a
    live exchange — losing the marker resolution and the rotation to "cannot
    operate on a closed database".

    The outcome is retrieved because a caller that gave up at its own bound never
    awaits this task, and the loop logs an unretrieved exception on a task nobody
    waits for as "Task exception was never retrieved". Every outcome here is an
    expected one — a deferral, a transport failure, a dead grant — so that line
    would be noise on the ordinary path rather than a signal.
    """
    _DETACHED_REFRESHES.discard(task)
    if not task.cancelled():
        task.exception()
    store.close()


class _SerializedConnection:
    """A sqlite connection whose statements are serialized by a mutex.

    The store's connection is opened with ``check_same_thread=False`` so a
    caller on a worker thread gets a correct answer instead of a
    ``ProgrammingError`` (see :meth:`AuthStore._connect`). sqlite connection
    objects are not internally synchronised, though, so the flag alone would
    swap a loud failure for silently interleaved statements -- an INSERT's
    ``lastrowid`` read after another thread's INSERT, for one.

    Wrapping instead of editing every call site keeps the ~27 existing
    ``self._conn.execute(...)`` uses exactly as they read, and means a
    statement added later is serialized by construction rather than by the
    author remembering a lock. ``execute`` returns the cursor, so
    ``.fetchall()``, ``.fetchone()`` and ``.lastrowid`` keep working; the lock
    covers producing the cursor, which is where the shared state is touched.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        # Reentrant: `executescript` runs during construction while the schema
        # is applied, and a future caller holding the lock across a helper that
        # also executes must not deadlock against itself.
        self._lock = threading.RLock()

    def execute(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(*args, **kwargs)

    def executescript(self, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executescript(*args, **kwargs)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class AuthStore:
    """Credential persistence + the 7-step cascade.

    ``config_dir`` supplies the config ROOT the env tier resolves its
    provider-class store rows under. It no longer feeds a plaintext leg: the env
    tier reads store rows then the process environment, and the legacy
    ``credentials.env`` rung is GONE (PR2a).
    ``config_stored_values`` seeds the config-derived tier. All DB access is
    local and synchronous; async methods only exist where refresh/network
    happens.

    ``config_dir`` is a bare PATH, not a manager: PR2b deleted the
    ``CredentialManager`` that used to carry it, and every caller already has the
    path (``ConfigManager.config_dir``) or can take ``None`` for the HOME default.

    .. note::
        Refresh is single-flight per process (``asyncio.Lock``) and across
        processes (``auth_credential_refresh_leases``, same upsert as usage).
        HTTP clients stay per-process.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        config_dir: Path | None = None,
        config_overrides: dict[str, str] | None = None,
        usage_cache: "UsageCacheStore | None" = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._config_dir = config_dir
        self._config_overrides = dict(config_overrides or {})
        self._runtime_overrides: dict[str, str] = {}
        self._fallback_resolvers: dict[str, Callable[[str], str | None]] = {}
        self._sticky: dict[tuple[str, str], int] = {}
        self._round_robin: dict[str, int] = {}
        # Credentials that just failed on a PROVIDER-side fault, which is not
        # their fault and so must not block them (see ``rotate_sibling``). They
        # are merely sorted last, so an attempt moves to a sibling while the
        # deprioritised row stays available as a last resort.
        #
        # Deliberately unpersisted: the condition it describes lasts seconds,
        # and a mark surviving a restart would misroute a session for a fault
        # that had long cleared.
        #
        # Deliberately keyed by PROVIDER rather than by session, and so shared
        # by every session in the process — like ``_round_robin`` above. That is
        # the correct scope for what it records: "this account was failing at
        # this provider a moment ago" is a fact about the provider and the
        # account, not about who observed it, so a sibling session benefits from
        # the discovery instead of paying to repeat it. Nothing here can make a
        # credential unusable, so the worst a stale mark can do is reorder a
        # pool whose members are all equally valid.
        self._deprioritized: dict[str, dict[int, int]] = {}
        self._refresh_locks: dict[int, asyncio.Lock] = {}
        #: Identity of this process in lease rows, so a release only frees a
        #: lease THIS process took. Same shape as :class:`UsageCacheStore`.
        self._refresh_holder = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        # The first-resolve account pick (``_usage_ranked_order``). ON by
        # default: a store built without a settings mapping (the CLI's login
        # surface, the server's per-job stores) should still spread load, and
        # the opt-out rides in through :meth:`configure_usage_aware_pick` from
        # the one place that reads ``retry.*``. The cache handle is built
        # lazily on first use so a store that never resolves an OAuth row
        # never opens the usage database.
        self._usage_aware_pick = True
        self._usage_pick_tolerance = USAGE_PICK_TOLERANCE
        self._usage_pick_max_age_ms = USAGE_PICK_MAX_REPORT_AGE_MS
        # ``usage_cache`` is an injection seam for tests (a temp store with
        # synthetic reports); production callers leave it ``None`` and the
        # shared ``~/.local-operator/usage_cache.db`` is opened on demand.
        self._usage_cache: "UsageCacheStore | None" = usage_cache
        self._usage_cache_probed = usage_cache is not None
        #: Set by :meth:`close`. Read by :func:`shared_auth_store` to decide
        #: whether the process-level store it is holding is still usable.
        self._closed = False
        self._conn = self._connect()

    @property
    def db_path(self) -> Path:
        """The SQLite file backing this store.

        Public because a caller about to write a SECRET must be able to check
        the permissions of the file that will hold it, and only the store
        knows which file that is. Resolving it caller-side instead (via
        ``default_db_path()``) gives two sources of truth: a store built on an
        explicit path would have the WRONG file checked, so the check could
        pass while the real store was world-readable.
        """
        return self._db_path

    # -- connection ----------------------------------------------------------

    def _connect(self) -> _SerializedConnection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file 0600 BEFORE sqlite opens it: the plain
        # connect-then-chmod leaves a window where secrets sit 0644 (PR-11).
        if not self._db_path.exists():
            fd = os.open(self._db_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        # `check_same_thread=False` + an explicit mutex, rather than sqlite's
        # default thread affinity. The store is built once on whatever thread
        # first asks for it (for the desktop routes, the event loop) and is then
        # shared, so ANY caller reaching it from a worker -- an `asyncio.to_thread`
        # hop, a thread-pool executor, a background scheduler -- raised
        # `ProgrammingError`. That is how a credential-free machine came to
        # report every model as connected (D18): the raise was swallowed by a
        # broad catch and reported as "the store is unreadable", which the model
        # catalogue deliberately degrades to "show everything".
        #
        # Removing the one bad hop fixed that route; this fixes the CLASS, so
        # the next caller to touch the store from another thread gets correct
        # serialized access instead of a plausible-looking wrong answer. The
        # lock is required because sqlite connection objects are not internally
        # synchronised: `check_same_thread=False` alone would trade a loud error
        # for silent interleaving.
        conn = sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        conn.commit()
        for path in (
            self._db_path,
            self._db_path.with_suffix(self._db_path.suffix + "-wal"),
            self._db_path.with_suffix(self._db_path.suffix + "-shm"),
        ):
            # WAL sidecars hold the same plaintext; keep them 0600 too.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return _SerializedConnection(conn)

    def close(self) -> None:
        # The flag is set FIRST so a reader that races this close sees either a
        # live store or a closed one, never a store whose connection is already
        # gone but which still claims to be open. It is what
        # :func:`shared_auth_store` consults to decide whether to reopen, and
        # its only consumer: everything else detects a closed connection the way
        # sqlite does, with ``ProgrammingError`` on use.
        self._closed = True
        self._conn.close()
        if self._usage_cache is not None:
            self._usage_cache.close()
            self._usage_cache = None

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run. See :func:`shared_auth_store`."""
        return self._closed

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # -- overrides -----------------------------------------------------------

    def set_runtime_api_key(self, provider: str, api_key: str | None) -> None:
        """Tier 1: CLI ``--api-key``. ``None`` clears."""
        if api_key:
            self._runtime_overrides[provider] = api_key
        else:
            self._runtime_overrides.pop(provider, None)

    def set_config_api_key(self, provider: str, api_key: str | None) -> None:
        """Tier 2: models.yml pointer. Beats OAuth because the user aimed the
        provider at a custom base URL/gateway."""
        if api_key:
            self._config_overrides[provider] = api_key
        else:
            self._config_overrides.pop(provider, None)

    def override_keys(self, provider: str) -> list[str]:
        """The override-tier secrets (runtime, then config) for ``provider``.

        Public accessor for callers that need to know WHICH secrets the
        cascade's tiers 1/2 would resolve without re-implementing the lookup —
        the usage cache folds these into its account fingerprint so two
        sessions on different override keys never share a cache row. Reading
        the private maps from outside worked but pinned this module's field
        names: a rename here would have silently reverted that fingerprint.
        """
        keys: list[str] = []
        for tier in (self._runtime_overrides, self._config_overrides):
            secret = tier.get(provider)
            if secret:
                keys.append(secret)
        return keys

    def set_fallback_resolver(
        self, provider: str, resolver: Callable[[str], str | None] | None
    ) -> None:
        """Tier 7: custom-provider hook."""
        if resolver is None:
            self._fallback_resolvers.pop(provider, None)
        else:
            self._fallback_resolvers[provider] = resolver

    def configure_usage_aware_pick(
        self,
        enabled: bool,
        *,
        tolerance: float | None = None,
        max_report_age_ms: int | None = None,
    ) -> None:
        """Switch the usage-ranked first-resolve pick on or off.

        The store defaults to ON so every constructor site spreads load
        without knowing about the setting; the session stream, which is the
        one place that parses ``retry.*``, calls this with
        ``retry.usageAwareAccountPick`` so the operator's opt-out reaches the
        cascade. ``tolerance`` and ``max_report_age_ms`` are exposed for tests
        and diagnostics rather than as user settings: the defaults encode a
        judgement about herding and window rollover (see the module
        constants) that a per-user knob would only let people get wrong.
        """
        self._usage_aware_pick = bool(enabled)
        if tolerance is not None:
            self._usage_pick_tolerance = max(0.0, min(1.0, float(tolerance)))
        if max_report_age_ms is not None:
            self._usage_pick_max_age_ms = max(0, int(max_report_age_ms))

    # -- credential CRUD -------------------------------------------------------

    @staticmethod
    def _row_to_credential(row: tuple[Any, ...]) -> StoredCredential:
        return StoredCredential(
            id=row[0],
            provider=row[1],
            credential_type=row[2],
            data=json.loads(row[3]),
            disabled_cause=row[4],
            identity_key=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    @staticmethod
    def _oauth_key_fn(provider: str) -> GetApiKeyFn | None:
        """The extractor that pulls the WIRE token out of an OAuth row.

        Which field is the bearer is a property of the row, and the row belongs
        to the storage provider — so the storage definition's extractor is the
        authoritative one, and the flavour's own is preferred only where it has
        one. QwenCloud is why this cannot just read ``creds["access"]``: its row
        holds a management token in ``access`` and the ``sk-sp-…`` inference key
        in ``api_key``, and only ``alibaba-token-plan`` (the STORAGE id) carries
        the extractor that knows to prefer the latter. Resolving the ``-oauth``
        flavour by its own definition would authenticate inference with the
        token the inference endpoint rejects.
        """
        definition = get_provider_definition(provider)
        if definition is not None and definition.get_api_key is not None:
            return definition.get_api_key
        storage = get_provider_definition(credential_provider_id(provider))
        return storage.get_api_key if storage is not None else None

    @staticmethod
    def _storage_id(provider: str) -> str:
        """The provider id whose ROWS answer a query about ``provider``.

        Login flavours (``xai-oauth``, ``openai-device``,
        ``alibaba-token-plan-oauth``, ``zai-oauth``) deliberately store their
        credential under the base provider's name, so every query here — which
        is exact SQL —
        has to be asked in terms of the storage id or it matches nothing. Doing
        it once, at the boundary of the store, is what keeps the translation
        from having to be remembered at each of the dozen call sites that ask
        this class about a provider; forgetting it does not fail loudly, it
        silently reports the provider as having no credential at all.

        Applied to row lookups, blocks and session stickiness alike: an alias
        and its base are ONE credential, so a backoff earned by a request under
        one name must be honoured under the other, and a session that stuck to
        an account must stay on it across both spellings.
        """
        return credential_provider_id(provider)

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[StoredCredential]:
        """Enabled credentials (all providers or one), oldest first.

        ``provider`` is resolved through :meth:`_storage_id`, so asking for a
        login flavour returns the rows its login actually wrote.
        """
        if provider is not None:
            provider = self._storage_id(provider)
            rows = self._conn.execute(
                "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
                " created_at, updated_at FROM auth_credentials WHERE provider = ? ORDER BY id",
                (provider,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
                " created_at, updated_at FROM auth_credentials ORDER BY id"
            ).fetchall()
        creds = [self._row_to_credential(r) for r in rows]
        if include_disabled:
            return creds
        return [c for c in creds if c.disabled_cause is None]

    def active_local_credential(self, provider: str, endpoint: str) -> StoredCredential | None:
        """The last configured token for this endpoint, without reviving history.

        Cloud keys form a rotation pool. A local setup form instead edits ONE
        connection: a replacement token must not alternate with its predecessor.
        Inspect disabled rows too, so clearing/rejecting the latest token does
        not silently fall back to an older token the operator replaced.
        """
        for row in reversed(self.list_credentials(provider, include_disabled=True)):
            if row.credential_type == "api_key" and row.data.get("endpoint") == endpoint:
                return row if row.disabled_cause is None else None
        return None

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> StoredCredential:
        """Insert, or update the row for the same identity (org scope ⇒ rows).

        Revives soft-deleted rows for the same identity (re-login).

        The login path already resolves ``store_credentials_as`` before calling
        here, so this normalization is usually a no-op. It is applied anyway
        because the READS now alias unconditionally: a caller that passed a
        flavour id would otherwise write a row under ``xai-oauth`` that no
        lookup for either ``xai-oauth`` or ``xai`` would ever return, which is
        a worse failure than the one being fixed and an invisible one. Writes
        and reads must agree on where a credential lives.
        """
        provider = self._storage_id(provider)
        # An EXPLICIT type wins; the structural guess is the fallback for the
        # callers (and stored rows) that never declared one.
        #
        # The guess reads "has both refresh and access", which cannot see a
        # credential that is OAuth-issued but has no refresh token because it
        # never expires -- Z.AI's coding-plan sign-in mints exactly that. Such a
        # row landed as `api_key` with its secret under `data["access"]`, where
        # nothing can read it: tiers 4 and 6 read `data["key"]`, and tier 3 only
        # walks `oauth`-typed rows. The login reported success and every request
        # afterwards failed with no credential at all.
        declared = credential.get("type")
        credential_type = (
            declared
            if declared in ("oauth", "api_key")
            else ("oauth" if credential.get("refresh") and credential.get("access") else "api_key")
        )
        identity = _identity_key_for(provider, credential)
        payload = dict(credential)
        payload["type"] = credential_type
        # An INTERACTIVE write is the only thing that may clear the two
        # durability markers, and it does so by replacing the payload wholesale.
        # Dropped explicitly rather than left to that replacement: a login flow
        # that seeded its dict from the row it is replacing (a re-login that
        # reuses the stored account/org fields) would otherwise carry a dead
        # grant's tombstone, or another token's unconfirmed-send marker, onto
        # the credential the user just proved is alive.
        payload.pop(GRANT_DEAD_AT_KEY, None)
        payload.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
        now = self._now_ms()
        data_json = json.dumps(payload)

        if identity is not None:
            row = self._conn.execute(
                "SELECT id FROM auth_credentials WHERE provider = ? "
                "AND identity_key = ? ORDER BY id",
                (provider, identity),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    "UPDATE auth_credentials SET credential_type = ?, data = ?, "
                    "disabled_cause = NULL,"
                    " updated_at = ? WHERE id = ?",
                    (credential_type, data_json, now, row[0]),
                )
                self._conn.commit()
                return self._after_credential_write(provider, row[0])

        cursor = self._conn.execute(
            "INSERT INTO auth_credentials "
            "(provider, credential_type, data, identity_key, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (provider, credential_type, data_json, identity, now, now),
        )
        self._conn.commit()
        return self._after_credential_write(provider, cursor.lastrowid)

    def _after_credential_write(self, provider: str, credential_id: int | None) -> StoredCredential:
        """Read the row back, then run what a completed write implies.

        One exit for both the update and the insert path, so a later caller
        cannot land in a branch that skips the re-arm.
        """
        stored = self._reread_after_write(credential_id)
        self._rearm_parked_tunnel_login(provider, stored)
        return stored

    def _rearm_parked_tunnel_login(self, provider: str, stored: StoredCredential) -> None:
        """Let a login that fixed a dead grant bring the tunnel connector back.

        The connector cannot notice by itself: parking exits SUCCESSFULLY
        (that is what stops its supervisor retrying it, see
        ``tunnels/service.py``), so the operator's login is the only event that
        can end the park. Called from here because this one write path is shared
        by the TUI's `/login`, `lop login`, and the desktop login route, and
        because the fix has to be durable first — this runs after the commit,
        never before it.

        Unconditional and cheap on purpose: `tunnels.install.rearm_if_parked`
        decides whether the write concerns the tunnel at all, and its first
        guard is the park file, which does not exist in the ordinary case. The
        import is lazy because `tunnels.install` reaches `launchd`, `paths` and
        the tunnels package, none of which belong on the import path of a
        credential write — and because this module is imported BY that package.

        OFF THE EVENT LOOP where there is one (review round 1, m3). The guard
        chain ends in a real `launchctl kickstart` / `systemctl --user start`
        with a 20-second timeout, and this write path is reached from the desktop
        login route and from `/login` in the TUI — both on the loop that also
        serves every other request, so a hung service manager stalled the whole
        server for up to 20 seconds. There is nothing to return here, so the
        cheapest correct thing is to hand the call to a worker thread and let the
        caller proceed: the operator is told the sign-in succeeded, and the
        connector's own state file says the rest.
        """
        try:
            from local_operator.tunnels import install
        except Exception:  # noqa: BLE001 — a login must not fail over a service restart
            logger.warning(
                "could not re-arm a parked tunnel connector for %s after a login",
                provider,
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop on this thread. Deliberately the same call rather than a
            # skipped one: a script driving a login still wants its connector
            # back, and there is no loop here to stall.
            self._rearm_off_thread(install, provider, stored.id)
            return
        try:
            loop.run_in_executor(None, self._rearm_off_thread, install, provider, stored.id)
        except RuntimeError:
            # A running loop whose default executor is ALREADY SHUT DOWN — and
            # this is the one line in the guard chain that can fail there, so it
            # carries the guard (review round 2, M3). Measured: with
            # `loop.shutdown_default_executor()` done and the loop still ticking,
            # `run_in_executor` raises `RuntimeError: Executor shutdown has been
            # called`, which propagated out of here through
            # `_after_credential_write` and out of `upsert_credential` — i.e. out
            # of a credential write that had already COMMITTED, reporting a
            # successful sign-in to the operator as a failure (the house pattern
            # for a fire-and-forget dispatch under someone else's verdict is
            # `session/attached.py`'s guarded one).
            #
            # Inline rather than dropped, which is the no-loop branch's own trade
            # and the reason it makes it: the caller is a login, and the point of
            # this hook is that the connector comes back without a second
            # command. Reachability is the teardown window this signature
            # describes — a loop on its way out is not serving anyone, so the
            # blocking call it can no longer hand to a thread costs no request,
            # and `_rearm_off_thread` never raises.
            self._rearm_off_thread(install, provider, stored.id)

    @staticmethod
    def _rearm_off_thread(install: Any, provider: str, credential_id: int) -> None:
        """The re-arm itself, where a blocking call costs nobody a turn.

        Never raises: a re-arm that fails must not fail the login that triggered
        it — and on the executor path an exception would surface only as an
        unretrieved future exception, which is neither a log line nor a shrug.
        """
        try:
            note = install.rearm_if_parked(provider=provider, credential_id=credential_id)
        except Exception:  # noqa: BLE001 — a login must not fail over a service restart
            logger.warning(
                "could not re-arm a parked tunnel connector for %s after a login",
                provider,
            )
            return
        if note:
            logger.info("%s (provider=%s)", note, provider)

    def _reread_after_write(self, credential_id: int | None) -> StoredCredential:
        """Re-read a row this connection just wrote.

        The write and the read share one connection, so a miss is impossible;
        surfacing it as an error beats leaking ``None`` out of a non-optional
        return and failing somewhere further away.
        """
        stored = self.get_credential(credential_id) if credential_id is not None else None
        if stored is None:
            raise AuthStoreError("Credential row could not be read back after write")
        return stored

    def get_credential(self, credential_id: int) -> StoredCredential | None:
        row = self._conn.execute(
            "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
            " created_at, updated_at FROM auth_credentials WHERE id = ?",
            (credential_id,),
        ).fetchone()
        return self._row_to_credential(row) if row else None

    def disable_credential(self, credential_id: int, cause: str) -> None:
        """Soft-delete tombstone (keeps history, blocks selection)."""
        self._conn.execute(
            "UPDATE auth_credentials SET disabled_cause = ?, updated_at = ? WHERE id = ?",
            (cause, self._now_ms(), credential_id),
        )
        self._conn.commit()

    def delete_credential(self, credential_id: int) -> None:
        """Remove a row entirely (logout)."""
        self._conn.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?", (credential_id,)
        )
        self._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (credential_id,))
        self._conn.commit()

    def delete_credentials_for_provider(
        self, provider: str, disabled_cause: str = "logged-out"
    ) -> int:
        """Logout: wipe every credential stored under ``provider``. Returns count."""
        rows = self.list_credentials(provider)
        for row in rows:
            self.delete_credential(row.id)
        return len(rows)

    # -- blocking --------------------------------------------------------------

    def block_credential(
        self,
        credential_id: int,
        provider: str,
        block_scope: str = "",
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> None:
        """Record a backoff keyed by ``provider:type`` (storage id)."""
        credential = self.get_credential(credential_id)
        provider = self._storage_id(provider)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        # Floor keeps a 0/negative block from being a no-op; the ceiling
        # (MAX_CREDENTIAL_BLOCK_MS) guarantees no reading -- a multi-day usage
        # reset, a hostile Retry-After -- can strand an account past the point
        # where re-probing is cheaper than waiting. This is the single choke
        # point every block passes through (preflight AND reactive
        # rotate_sibling AND any future caller), so the cap protects them all
        # from a rogue provider header for free. See the constant's docstring.
        block_ms = max(1000, min(int(block_ms), MAX_CREDENTIAL_BLOCK_MS))
        until = self._now_ms() + block_ms
        self._conn.execute(
            "INSERT INTO auth_credential_blocks (credential_id, provider_key, block_scope,"
            " blocked_until_ms, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(credential_id, provider_key, block_scope)"
            " DO UPDATE SET blocked_until_ms = excluded.blocked_until_ms, "
            "updated_at = excluded.updated_at",
            (credential_id, provider_key, block_scope, until, self._now_ms()),
        )
        self._conn.commit()

    def is_blocked(self, credential_id: int, provider: str) -> bool:
        credential = self.get_credential(credential_id)
        provider = self._storage_id(provider)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        row = self._conn.execute(
            "SELECT blocked_until_ms FROM auth_credential_blocks"
            " WHERE credential_id = ? AND provider_key = ? AND block_scope = ''",
            (credential_id, provider_key),
        ).fetchone()
        return bool(row and row[0] > self._now_ms())

    def is_blocked_for_model(self, credential_id: int, provider: str, model_id: str) -> bool:
        """Whether ``credential_id`` is out of rotation for ``model_id``.

        The read side of a scoped quota block mirrors the rule the usage
        layer gates caps by: a tier row applies to a model when its slug
        appears in the model id (``fable`` in ``claude-fable-5``, ``grok-4``
        in ``grok-4.6``). A block is therefore visible to exactly the models
        whose cap wrote it, on ANY provider, without the two sides having to
        agree on a family parser — the write stores the usage tier slug, the
        read asks "does that slug gate this model". Account-wide blocks stop
        everything as before; a model no scoped slug matches still sees the
        account (the under-block direction, which heals on the next probe).
        """
        if self.is_blocked(credential_id, provider):
            return True
        lowered = (model_id or "").lower()
        if not lowered:
            return False
        credential = self.get_credential(credential_id)
        provider = self._storage_id(provider)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        rows = self._conn.execute(
            "SELECT block_scope, blocked_until_ms FROM auth_credential_blocks"
            " WHERE credential_id = ? AND provider_key = ? AND block_scope != ''",
            (credential_id, provider_key),
        ).fetchall()
        now = self._now_ms()
        return any(until > now and scope.removeprefix("model:") in lowered for scope, until in rows)

    def clear_blocks_for_model(self, credential_id: int, provider: str, model_id: str) -> None:
        """Drop the account-wide block and every scoped block gating ``model_id``.

        The recovery-probe counterpart of :meth:`is_blocked_for_model`: a
        fresh verdict that proves the model serviceable supersedes exactly
        the blocks that could have hidden it, and leaves other families'
        scoped blocks standing (a probe that proves opus serviceable says
        nothing about a Fable weekly that is still spent).
        """
        credential = self.get_credential(credential_id)
        provider = self._storage_id(provider)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        lowered = (model_id or "").lower()
        self._conn.execute(
            "DELETE FROM auth_credential_blocks"
            " WHERE credential_id = ? AND provider_key = ?"
            " AND (block_scope = '' OR (? != '' AND block_scope != ''"
            " AND instr(?, substr(block_scope, 7)) > 0))",
            (credential_id, provider_key, lowered, lowered),
        )
        self._conn.commit()

    def clear_blocks(self, credential_id: int) -> None:
        self._conn.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?", (credential_id,)
        )
        self._conn.commit()

    def clear_block(self, credential_id: int, provider: str, block_scope: str = "") -> None:
        """Drop the primary block for ONE credential, leaving other scopes alone.

        ``clear_blocks`` wipes every block the credential carries; usage-aware
        fallback only ever wants to rescind the quota backoff it placed itself,
        and must not disturb a block another mechanism (e.g. an auth failure on
        a different scope) recorded against the same row.
        """
        credential = self.get_credential(credential_id)
        provider = self._storage_id(provider)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        self._conn.execute(
            "DELETE FROM auth_credential_blocks"
            " WHERE credential_id = ? AND provider_key = ? AND block_scope = ?",
            (credential_id, provider_key, block_scope),
        )
        self._conn.commit()

    # -- OAuth refresh -----------------------------------------------------------

    def _refresh_fn(self, provider: str) -> RefreshFn | None:
        """The refresh callable for rows stored under ``provider``.

        Rows live under the STORAGE id, so the definition that owns the refresh
        is frequently not the one named by that id: a row under ``xai`` may have
        been written by ``xai-oauth``'s login, and only the flavour carries a
        ``refresh_token``. Hence the reverse scan — base definition first (a
        provider that refreshes its own rows), then the flavour that aliases
        onto it.
        """
        definition = get_provider_definition(provider)
        if definition is not None and definition.refresh_token is not None:
            return definition.refresh_token
        # Rows stored under an alias (xai-oauth ⇒ xai): find the origin def.
        from local_operator.providers.registry import PROVIDER_REGISTRY

        for other in PROVIDER_REGISTRY:
            if other.store_credentials_as == provider and other.refresh_token is not None:
                return other.refresh_token
        return None

    def _refresh_lock_for(self, credential_id: int) -> asyncio.Lock:
        lock = self._refresh_locks.get(credential_id)
        if lock is None:
            lock = asyncio.Lock()
            self._refresh_locks[credential_id] = lock
        return lock

    def _try_refresh_lease(self, credential_id: int) -> bool:
        """Take the cross-process refresh lease if it is free (or expired).

        Copied from :meth:`UsageCacheStore.try_lease`: one atomic upsert,
        judged by rowcount, because SELECT-then-INSERT is not atomic even
        inside ``with conn`` (sqlite3 defers BEGIN until the first write).
        True means THIS process POSTs. False means a peer already owns the
        refresh; the caller re-reads the row instead of joining the fan-out.
        A coordination failure returns True — efficiency depends on the lease,
        correctness of a single refresher does not.
        """
        now = self._now_ms()
        expires = now + AUTH_REFRESH_LEASE_MS
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO auth_credential_refresh_leases
                    (credential_id, holder, expires_at_ms)
                VALUES (?, ?, ?)
                ON CONFLICT(credential_id) DO UPDATE SET
                  holder = excluded.holder,
                  expires_at_ms = excluded.expires_at_ms
                WHERE auth_credential_refresh_leases.expires_at_ms <= ?
                """,
                (credential_id, self._refresh_holder, expires, now),
            )
            self._conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error:
            logger.debug("auth refresh lease failed", exc_info=True)
            return True

    def _release_refresh_lease(self, credential_id: int) -> None:
        """Free the lease if this process holds it (a refresh finished)."""
        try:
            self._conn.execute(
                "DELETE FROM auth_credential_refresh_leases "
                "WHERE credential_id = ? AND holder = ?",
                (credential_id, self._refresh_holder),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.debug("auth refresh lease release failed", exc_info=True)

    # -- durability markers for the refresh path --------------------------------
    #
    # Two pieces of state that must outlive the process that learned them, and
    # that therefore live in the row payload rather than in memory:
    #
    # * :data:`REFRESH_SEND_UNCONFIRMED_KEY` — a token was presented by an
    #   exchange whose outcome never arrived, so it may already be spent and must
    #   not be presented again until it expires;
    # * :data:`GRANT_DEAD_AT_KEY` — the IdP refused the grant for good, so every
    #   surface must say so until the user signs in again.
    #
    # Both are read-modify-written through :meth:`_update_payload` so neither can
    # invent its own staleness rule, and both are cleared the same way: by an
    # interactive login (``upsert_credential``), which replaces the payload.

    @staticmethod
    def _presented_refresh_token(creds: dict[str, Any]) -> str | None:
        """The refresh token a provider's refresh fn would present for ``creds``.

        Read with the same ``refresh``/``refresh_token`` fallback the provider
        refreshes use (``providers/oauth/radient.py`` and its siblings), so the
        digest a marker records is the digest of the token that would actually
        go on the wire. The fallback is why this is a helper rather than a bare
        ``.get("refresh")`` at each of the three call sites.
        """
        token = creds.get("refresh") or creds.get("refresh_token")
        return str(token) if token else None

    @staticmethod
    def _holds_live_bearer(creds: dict[str, Any], *, now_ms: int) -> bool:
        """Whether ``creds`` holds an access token still inside its lifetime.

        ``expires`` is written at MINT time with the provider's own skew already
        subtracted (``EXPIRY_SKEW_MS`` in ``providers/oauth/radient.py`` and
        siblings), so ``expires <= now`` is a token the provider itself treats as
        dead and no second skew belongs here. A payload with no ``expires`` is a
        static token that never expires — the Z.AI coding-plan shape — and is
        live by definition.
        """
        if not creds.get("access"):
            return False
        expires = creds.get("expires")
        if expires is None:
            return True
        try:
            return int(expires) > now_ms
        except (TypeError, ValueError):
            return False

    def _update_payload(
        self,
        credential_id: int,
        mutate: Callable[[dict[str, Any]], bool],
        *,
        moves_write_stamp: bool = False,
    ) -> bool:
        """Read-modify-write one row's payload under ``mutate``.

        The single write primitive for both durability markers. It re-reads the
        row INSIDE the call because each marker is computed from state a peer may
        have changed since the caller last looked, and every caller writes only
        while its own precondition still holds — the same check-then-write
        convention the refresh path's cross-process guard already uses, rather
        than a second one. ``mutate`` returns whether the row still wants the
        write at all.

        ``moves_write_stamp`` is why this is not simply an UPDATE: a SEND marker
        write must NOT move ``auth_credentials.updated_at``, and two memos in this
        codebase are keyed on that stamp with that premise written down —
        ``tunnels/report.py`` :func:`_verdict_key` and
        ``server/routes/desktop_radient._diagnosis_key``, both of which say "a
        FAILED refresh writes nothing, which is what lets a verdict hold across
        the very failures it describes". Arming and clearing a send marker is a
        failed refresh writing something, so bumping the stamp here would
        invalidation-memoise the verdict those caches exist to stop re-earning:
        one extra token-endpoint POST per poll, for a verdict the store had
        already computed. The MCP sibling documents the identical trap from the
        other side (``mcp/manager._grant_marker``: keying on ``updated_at`` "would
        make this session retry on its OWN writes, every tick, forever").
        A tombstone DOES pass ``True``: a grant the IdP has refused for good is a
        change to the credential, and a memo still holding the pre-refusal verdict
        would be holding a fact the row no longer supports. A login or a landed
        rotation moves the stamp through their own writes, as they always did.

        Returns whether the write happened. A row that is GONE (a racing logout)
        is ``False`` and never a re-created row: an explicitly removed credential
        stays removed. Best-effort on a sqlite error, like every other marker
        write here — losing a marker restores the pre-marker behaviour rather
        than failing the request that triggered it.
        """
        row = self.get_credential(credential_id)
        if row is None or not isinstance(row.data, dict):
            return False
        data = dict(row.data)
        if not mutate(data):
            return False
        try:
            if moves_write_stamp:
                self._conn.execute(
                    "UPDATE auth_credentials SET data = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(data), self._now_ms(), credential_id),
                )
            else:
                self._conn.execute(
                    "UPDATE auth_credentials SET data = ? WHERE id = ?",
                    (json.dumps(data), credential_id),
                )
            self._conn.commit()
        except sqlite3.Error:
            logger.debug("auth payload marker write failed", exc_info=True)
            return False
        return True

    def _arm_send_marker(
        self, credential_id: int, presented_refresh_token: str, shape: str = SEND_SHAPE_UNKNOWN
    ) -> dict[str, Any] | None:
        """Arm the write-ahead marker for a token an exchange is about to present.

        Called while the refresh lock AND the cross-process lease are held, and
        BEFORE the POST: that ordering is the whole point. A process that dies
        between the write and the response takes all in-memory knowledge with it,
        and the next boot would present a token the IdP may already have spent —
        the reuse-detection POST that revokes the family. The marker is the only
        channel that survives that death.

        ``shape`` starts at :data:`SEND_SHAPE_UNKNOWN` — the state a process that
        dies mid-request leaves behind, and the only one there is no way to
        narrow. The exception arms update it to :data:`SEND_SHAPE_ANSWERED`, whose
        bound is much shorter (see :func:`_send_marker_is_live`), and a failure
        proven to be pre-send clears the marker instead of re-arming it.

        RETURNS THE MARKER IT WROTE, and ``None`` when the write failed — that
        return value is the OWNERSHIP TOKEN for everything the exchange does to
        the marker afterwards, not a convenience. A marker is a claim about a
        token ANOTHER PROCESS may still be exchanging: ours can outlive its own
        TTL (a suspended laptop ages the wall clock the marker is stamped on while
        the exchange's own cap runs on asyncio's loop clock, which cannot fire
        while the loop is not running), and the peer that takes the resulting
        expired lease arms a marker of its own for the SAME token. Rewriting that
        peer's marker from here — which the first version of this method permitted
        by checking only that a dict was present — re-bounds an outcome that is
        still unknown from one exchange window to one block window, and a block
        window later the store presents a token the peer's exchange may already
        have spent. That is review round
        2's M1, and the reuse-detection POST this whole change exists to prevent.
        ``holder`` is the same ``pid:uuid`` the lease rows carry, compared the same
        way :meth:`_release_refresh_lease` compares it, so "this marker is mine"
        has one definition in this module rather than two.

        Best-effort: a store failure must not stop the POST, and an exchange that
        could not arm anything simply owns nothing to re-bound or clear.
        """
        marker = {
            "digest": _refresh_token_digest(presented_refresh_token),
            "at": self._now_ms(),
            "shape": shape,
            "holder": self._refresh_holder,
        }

        def write(data: dict[str, Any]) -> bool:
            data[REFRESH_SEND_UNCONFIRMED_KEY] = marker
            return True

        return marker if self._update_payload(credential_id, write) else None

    def _rebound_send_marker(
        self, credential_id: int, armed: dict[str, Any] | None, shape: str
    ) -> bool:
        """Re-bound a marker to ``shape`` after its exchange answered — OURS only.

        The arm happens before the POST, so the marker is always born
        :data:`SEND_SHAPE_UNKNOWN`. The exchange's own failure then says more than
        the arm could: an answer arrived (:data:`SEND_SHAPE_ANSWERED`, a much
        shorter bound) or the request provably never went out (handled by the
        caller, which clears it).

        ``armed`` is what :meth:`_arm_send_marker` returned, and the write happens
        only while the row's marker is STILL THAT EXACT MARKER. A marker a peer has
        armed since — the handoff review round 2 reproduced — is left untouched,
        because re-bounding it would truncate the peer's bound for a token this
        exchange is no longer the one unsure about. An exchange that never armed
        anything (``None``) writes nothing.

        Returns whether the marker was re-bound.
        """
        if armed is None:
            return False
        owner = _send_marker_owner(armed)
        if owner is None:
            return False

        def rebind(data: dict[str, Any]) -> bool:
            if _send_marker_owner(data.get(REFRESH_SEND_UNCONFIRMED_KEY)) != owner:
                return False
            data[REFRESH_SEND_UNCONFIRMED_KEY] = {**armed, "shape": shape}
            return True

        return self._update_payload(credential_id, rebind)

    def _clear_send_marker(self, credential_id: int, armed: dict[str, Any] | None) -> bool:
        """Resolve the send marker THIS exchange armed: a definitive answer landed.

        Called for a 2xx whose rotation was persisted, for an ``invalid_grant``
        refusal, and for a failure PROVEN to have happened before the request
        reached the wire (:func:`_refresh_request_never_sent` — nothing was
        presented, so there is nothing to be unsure about). ``armed`` is what
        :meth:`_arm_send_marker` returned, and the marker is removed only while it
        is STILL THAT MARKER: a peer that took the expired lease mid-exchange owns
        the row's marker now, and this exchange resolving its own send says nothing
        about the peer's (review round 2's M1 carries both halves).

        Deliberately NOT called for the two remaining shapes, which is the whole
        point of the marker:

        * the request was written and no answer arrived — the state the marker
          describes, kept for the full :data:`UNCONFIRMED_SEND_TTL_S`;
        * an answer that proves nothing about our token, such as a ``5xx`` or a
          ``429``. A provider that commits a rotation and THEN fails the response
          leaves the row holding a spent token, so clearing the marker there
          would let the next attempt re-present it: the reuse-detection POST that
          revokes the whole family. That shape is re-BOUND instead
          (:data:`SEND_SHAPE_ANSWERED`), which keeps the token un-presented while
          the failure is fresh and lets the account heal on the window the
          cascade already refuses it for, rather than for the in-doubt shape's
          whole window.

        Returns whether the marker was removed.
        """
        if armed is None:
            return False
        owner = _send_marker_owner(armed)
        if owner is None:
            return False

        def drop(data: dict[str, Any]) -> bool:
            if _send_marker_owner(data.get(REFRESH_SEND_UNCONFIRMED_KEY)) != owner:
                return False
            data.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
            return True

        return self._update_payload(credential_id, drop)

    def _drop_dead_send_marker(self, credential_id: int, target: str | None) -> bool:
        """Remove a marker that can no longer be believed, WHOEVER armed it.

        The ownership condition :meth:`_rebound_send_marker` and
        :meth:`_clear_send_marker` enforce exists so one exchange cannot rewrite
        another's LIVE marker. This is the other side of it: a marker that has
        already stopped meaning anything — expired, or about a token the row no
        longer holds — is inert, so removing it changes no answer for any process,
        and not removing it means re-deciding the same dead marker on every later
        attempt. Re-checked under the write, because the row can move between the
        read that called it dead and the write.

        ``target`` is the token the caller asked about (the row's current one);
        ``None`` means the row holds no refresh token at all. Returns whether a
        marker was removed.
        """

        def drop(data: dict[str, Any]) -> bool:
            marker = data.get(REFRESH_SEND_UNCONFIRMED_KEY)
            if marker is None:
                return False
            if _send_marker_is_live(marker, target, now_ms=self._now_ms()):
                return False
            data.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
            return True

        return self._update_payload(credential_id, drop)

    def send_unconfirmed(self, credential_id: int, refresh_token: str | None = None) -> bool:
        """Whether a LIVE marker says ``refresh_token`` may already be spent.

        ``refresh_token`` defaults to the one stored, which is the question the
        refresh path asks before it POSTs. A marker that is stale — expired, or
        about a different token — is CLEARED on the way past rather than
        believed, because the answer already decided it is not this row's
        business; leaving it in place would re-evaluate (and re-clear) it on
        every later attempt. See :func:`_send_marker_is_live` for the rule, and
        :meth:`_drop_dead_send_marker` for why that one removal is safe for a
        marker this process did not arm.
        """
        row = self.get_credential(credential_id)
        if row is None or not isinstance(row.data, dict):
            return False
        data = dict(row.data)
        marker = data.get(REFRESH_SEND_UNCONFIRMED_KEY)
        if marker is None:
            return False
        target = refresh_token if refresh_token is not None else self._presented_refresh_token(data)
        if _send_marker_is_live(marker, target, now_ms=self._now_ms()):
            return True
        self._drop_dead_send_marker(credential_id, target)
        return False

    @staticmethod
    def _grant_dead_at(creds: dict[str, Any]) -> int | None:
        """The epoch-ms a dead-grant tombstone was written, or ``None``.

        Tolerant by design: an absent, non-numeric, boolean or non-positive value
        is NOT a tombstone. An unreadable marker must never suppress a refresh
        that might work, so "unknown" has to fall on the side that keeps trying.
        """
        marker = creds.get(GRANT_DEAD_AT_KEY)
        if isinstance(marker, bool) or not isinstance(marker, (int, float)):
            return None
        return int(marker) if marker > 0 else None

    def grant_is_dead(self, credential_id: int) -> bool:
        """Whether this row carries the IdP's own refusal as a persisted verdict.

        The read half of :data:`GRANT_DEAD_AT_KEY`, for surfaces that must state
        the verdict WITHOUT attempting a refresh: a dead grant is a fact about
        the account, and re-earning it with a POST is both a wasted round trip
        and a second chance to present a revoked family's token.
        """
        row = self.get_credential(credential_id)
        return (
            row is not None
            and isinstance(row.data, dict)
            and self._grant_dead_at(row.data) is not None
        )

    def _mark_grant_dead(self, credential_id: int, *, rejected_refresh_token: str | None) -> bool:
        """Record that this row's refresh token was refused by the IdP for good.

        The tombstone is the OUTCOME of a refusal the caller has already
        classified (an ``invalid_grant`` this store turned into
        :class:`CredentialInvalidError`); it is never the classification itself,
        so the store cannot invent a dead grant out of a transient 5xx.

        ``rejected_refresh_token`` makes the freshness re-read a
        compare-and-skip, and it exists because this is a whole-payload
        read-modify-write: a sibling that persisted a fresh rotation between the
        caller's guard and this write would otherwise have a LIVE token marked
        dead (or erased by a pre-rotation snapshot) and stay suppressed until an
        interactive login. The caller holds the refresh lock and the lease, so
        the remaining gap is against writers that deliberately do not take them —
        an interactive login — which is why the compare is here as well as there.
        A row that no longer holds the token the IdP rejected is left alone.

        Returns whether the tombstone was written.
        """

        def write(data: dict[str, Any]) -> bool:
            if rejected_refresh_token is not None:
                stored = self._presented_refresh_token(data)
                if stored != rejected_refresh_token:
                    logger.warning(
                        "dead-grant marker for %s was NOT written: the row now holds a "
                        "different refresh token (a peer rotated it, or the user signed in "
                        "again), so the refusal this marker records is about a superseded "
                        "token",
                        credential_id,
                    )
                    return False
            data[GRANT_DEAD_AT_KEY] = self._now_ms()
            # The refusal IS the definitive answer to the send marker: the IdP
            # told us what happened to the presented token, so there is nothing
            # left for the marker to protect. This pop is deliberately NOT
            # ownership-gated like the rebind/clear pair: the row is going DEAD,
            # a dead row is never refreshed again, and the guard above has already
            # established the marker can only be about the token just refused — so
            # removing it costs a peer nothing it was still protecting.
            data.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
            return True

        return self._update_payload(credential_id, write, moves_write_stamp=True)

    def _served_while_contended(
        self, row: StoredCredential, now_data: dict[str, Any]
    ) -> dict[str, Any]:
        """What a caller may be served when it did NOT win the refresh lease.

        THE ANSWER IS THE STORED ROW, and getting this wrong is what the first
        revision of this change did (review round 1, R1, BLOCKER). It refused an
        expired bearer here with a bare :class:`AuthStoreError`, which reads as
        *this credential is bad* one call up: ``_resolve`` blocks the row for
        ``DEFAULT_BLOCK_MS`` in every process sharing the DB, and the stale-bearer
        return that the desktop proxy's bounded join is built on disappeared —
        two merged e2e tests went from 200 to 502 for the whole duration of a
        legitimately refreshing peer. A PEER'S REFRESH IS NOT A VERDICT ABOUT THE
        CREDENTIAL, so it must not be reported as one; the consumers that must not
        spend a due bearer already say so themselves, after waiting for the peer
        (``server/routes/desktop_radient.py``, ``_await_peer_refresh`` and
        ``_is_due``). PR #1340 put that refusal in the CONSUMER, which is where the
        information about whether a stale bearer is usable actually lives.

        What DOES belong here is the one state a return would leak: a grant the
        IdP has refused for good is served nothing, with the same
        :class:`CredentialInvalidError` the row produces with the lease free.

        The send marker is deliberately NOT consulted. A live marker in this
        branch means the PEER's exchange armed it moments ago and is on the wire —
        the mechanism working — so refusing on it would fail every request that
        races a healthy refresh. The marker is consulted where it protects
        something: in the leased branch, by the only writer, before a POST.
        """
        if self._grant_dead_at(now_data) is not None:
            raise CredentialInvalidError(
                f"OAuth grant for '{row.provider}' was refused by the identity provider; "
                f"run /login {row.provider} to sign in again"
            )
        return now_data

    @staticmethod
    def _needs_refresh(creds: dict[str, Any], *, force: bool = False) -> bool:
        if force:
            return True
        access = creds.get("access")
        if not access:
            return True
        expires = creds.get("expires")
        if expires is None:  # static token; never expires
            return False
        return int(expires) <= AuthStore._now_ms() + OAUTH_REFRESH_SKEW_MS

    async def _ensure_oauth_fresh(
        self, row: StoredCredential, *, force: bool = False
    ) -> dict[str, Any]:
        """Return usable OAuth data for ``row``, refreshing single-flight.

        Raises :class:`AuthStoreError` when a refresh is required and fails;
        callers treat that row as unusable and rotate. A grant the IdP has
        declared permanently dead raises the :class:`CredentialInvalidError`
        subclass instead, so a caller that can act on the difference (tell the
        user to re-login rather than retry) is able to.

        Org fields are restored from the stored row AFTER the merge (PR-12):
        a refresh function that (mistakenly) returns org_id/org_name/
        authorized_at can never rewrite them — identity is fixed at login.
        """
        creds = dict(row.data)
        # The IdP's own refusal, persisted on the row, is honoured BEFORE the
        # freshness check and therefore before any POST. The ordering matters: a
        # tombstoned grant is dead whether or not its access token has expired
        # yet, and a row that still looks fresh is exactly the state that used to
        # be reported as a healthy login. This is where the store CONSUMES a
        # terminal verdict; WHICH refusals are terminal is decided one layer down
        # (``oauth/callback_server.is_terminal_grant_response``, widened for
        # Radient's prose body by #1342) — the store records and honours the
        # verdict rather than re-deriving it.
        if self._grant_dead_at(creds) is not None:
            raise CredentialInvalidError(
                f"OAuth grant for '{row.provider}' was refused by the identity provider; "
                f"run /login {row.provider} to sign in again"
            )
        if not self._needs_refresh(creds, force=force):
            return creds
        refresh = self._refresh_fn(row.provider)
        if refresh is None:
            raise AuthStoreError(f"No refresh capability for provider '{row.provider}'")

        async with self._refresh_lock_for(row.id):
            # Re-read inside the lock: another coroutine may have refreshed.
            current = self.get_credential(row.id)
            if current is None or current.disabled_cause is not None:
                raise AuthStoreError(f"Credential {row.id} disappeared during refresh")
            fresh = dict(current.data)
            # The same ask as the pre-lock check, repeated under the lock: a
            # sibling may have tombstoned the grant while we waited for it.
            if self._grant_dead_at(fresh) is not None:
                raise CredentialInvalidError(
                    f"OAuth grant for '{row.provider}' was refused by the identity provider; "
                    f"run /login {row.provider} to sign in again"
                )
            if not self._needs_refresh(fresh, force=force):
                return fresh
            if not self._try_refresh_lease(row.id):
                # A peer holds the lease. Wait out a slice of it and re-read:
                # POSTing the same rotating refresh token is how the loser
                # used to earn a real invalid_grant against a live grant.
                await asyncio.sleep(0.05)
                now_row = self.get_credential(row.id)
                if now_row is None or now_row.disabled_cause is not None:
                    raise AuthStoreError(f"Credential {row.id} disappeared during refresh")
                now_data = dict(now_row.data)
                if not self._needs_refresh(now_data, force=force):
                    return now_data
                # Peer still in flight or failed without writing. Serve whatever
                # is still HONEST (see ``_served_while_contended``) and re-try the
                # lease only for a caller that has no choice about needing a
                # fresh token (force=True): a non-forced caller does not take a
                # live holder's lease, because a second POST of the same rotating
                # token is the harm the lease exists to prevent.
                if not force:
                    return self._served_while_contended(row, now_data)
                if not self._try_refresh_lease(row.id):
                    return self._served_while_contended(row, now_data)
            # Imported here, not at module scope: `callback_server` drags in
            # http.server/ssl/email (~138 ms, 150-odd modules), and three
            # separate comments in this codebase exist to stop anyone
            # re-adding it as a top-level import. This is a refresh path that
            # has already paid for the module, so the cost is zero here.
            from local_operator.providers.oauth.callback_server import (
                InvalidGrantError,
                LoginError,
            )

            presented = self._presented_refresh_token(fresh)
            # Everything from here to the return runs holding the cross-process
            # lease, and EVERY exit must free it. That used to be six separate
            # release calls, one per exit path, and a leaked lease now strands
            # peers for the whole of AUTH_REFRESH_LEASE_MS — the very window in
            # which they must not take the lease themselves — so the release is a
            # `finally` that no later edit can miss.
            try:
                if presented and self.send_unconfirmed(row.id, presented):
                    # This row's refresh token was presented by an exchange whose
                    # outcome is not settled, so it MAY already be spent, and
                    # re-presenting a spent token is the reuse-detection POST that
                    # revokes the whole family. Defer instead of presenting.
                    #
                    # Unless the row's ACCESS token is still inside its own
                    # lifetime: that is a different secret, unaffected by the
                    # refresh exchange, and refusing it would fail requests that
                    # would have worked. The marker is about the refresh token, so
                    # it only decides against presenting one — never against
                    # serving a bearer that still works.
                    if self._holds_live_bearer(fresh, now_ms=self._now_ms()):
                        return fresh
                    logger.warning(
                        "refresh for credential %s (%s) was NOT attempted: its stored "
                        "refresh token was presented by an exchange whose outcome is not "
                        "settled, and presenting it again may revoke the whole token "
                        "family — this clears by itself within about %s, and needs a "
                        "sign-in only if it persists past that",
                        row.id,
                        row.provider,
                        self_clearing_window(),
                    )
                    raise RefreshUnconfirmedError(
                        f"OAuth refresh for '{row.provider}' was deferred: the stored "
                        "refresh token was presented by an exchange whose outcome is not "
                        "settled, so it is not presented again while that lasts "
                        "(presenting it may revoke the whole token family). This is not a "
                        "verdict about the login and it clears by itself within about "
                        f"{self_clearing_window()}; sign in again only if it persists "
                        "past that"
                    )
                armed: dict[str, Any] | None = None
                if presented:
                    # WRITE-AHEAD, and before the request can be on the wire: a
                    # process that dies mid-POST takes all in-memory knowledge
                    # with it, and the next boot would present a token this
                    # exchange may already have spent. What comes back is this
                    # exchange's claim on the marker it wrote: every later write to
                    # it is gated on that identity, because a peer can own the row's
                    # marker by the time our answer arrives (review round 2, M1).
                    armed = self._arm_send_marker(row.id, presented)
                try:
                    # WALL-CLOCK cap on the exchange. The providers pass a
                    # per-OPERATION httpx timeout, which does not bound a request:
                    # connect, write and each read get it separately, so a
                    # stalling endpoint held one POST open for 41.4 s and 114.5 s
                    # in measurements on this machine — which is why the lease is
                    # derived from THIS number and not from the per-op one (review
                    # round 1, R4). A fired cap lands in the generic arm below with
                    # the marker left armed, which is right: the request was handed
                    # over, so the outcome is in doubt.
                    async with asyncio.timeout(PROVIDER_REFRESH_TOTAL_BUDGET_S):
                        refreshed = await refresh(fresh)
                except AuthStoreError:
                    # The refresh fn's OWN refusal. Nothing about it says what
                    # happened to the token on the wire, so the marker is left
                    # exactly as it is: armed if the exchange may have run, and
                    # absent if it never did.
                    raise
                except InvalidGrantError as exc:
                    # RFC 6749 SS5.2 terminal grant error. Re-raised as the store's
                    # own permanent-failure type so nothing above imports the
                    # OAuth flow module to identify it, and so the existing
                    # `except AuthStoreError` handlers still catch it.
                    #
                    # The cross-process rotation race documented on the success
                    # path below reaches HERE FIRST, and is the more dangerous
                    # arrival: with a rotating refresh token (kimi rotates), the
                    # process that loses the race POSTs a token the winner has
                    # already consumed, and the IdP answers `invalid_grant`
                    # legitimately. Declaring the credential permanently dead on
                    # that evidence condemns a LIVE grant -- the winner's token is
                    # in the DB and working -- and before this classification
                    # existed the loser raised a generic AuthStoreError and
                    # self-healed on the next cycle. So the same guard runs first:
                    # if the stored refresh token is no longer the one we sent,
                    # another process refreshed successfully and our rejection is
                    # about a superseded token, not about the account.
                    now_row = self.get_credential(row.id)
                    if (
                        now_row is not None
                        and self._presented_refresh_token(dict(now_row.data)) != presented
                    ):
                        logger.warning(
                            "refresh race on %s: our token was already rotated by another "
                            "process; keeping its token rather than declaring the grant dead",
                            row.id,
                        )
                        return dict(now_row.data)
                    # Past the race guard, the refusal IS about the token this row
                    # holds, so the verdict is recorded as DATA before it is
                    # raised: every surface that reads the row (the usage panel,
                    # ``/provider``) can then say "sign-in expired" without a live
                    # refresh attempt to re-earn it, and only an interactive login
                    # clears it. The marker is resolved by the write — the IdP told
                    # us what happened to the presented token.
                    self._mark_grant_dead(row.id, rejected_refresh_token=presented)
                    raise CredentialInvalidError(
                        f"OAuth grant for '{row.provider}' is no longer valid: {exc}"
                    ) from exc
                except Exception as exc:
                    if _caused_by_never_sent(exc):
                        # PROVABLY PRE-SEND: httpx's own taxonomy says the request
                        # was never written — a refused connection, a DNS failure,
                        # a dead TLS handshake, a pool timeout, a URL it could not
                        # even format — so the token was NOT presented and there is
                        # nothing to be unsure about. Clearing the marker is what
                        # keeps a token endpoint on a closed port from suppressing
                        # the account for the marker's window over a request nothing
                        # ever received, while the message stops asserting that a
                        # presentation happened (review round 1, R2). The sibling
                        # rule (`mcp/auth.py:_refresh_request_never_sent`) is
                        # adopted rather than re-derived, including its asymmetry:
                        # everything it does not recognise stays suspect.
                        self._clear_send_marker(row.id, armed)
                        logger.info(
                            "refresh for credential %s (%s) could not reach the token "
                            "endpoint (%s); the refresh token was never presented",
                            row.id,
                            row.provider,
                            type(exc).__name__,
                        )
                        raise AuthStoreError(
                            f"OAuth refresh for '{row.provider}' could not reach the token "
                            f"endpoint ({type(exc).__name__}); the refresh token was never "
                            "presented, so this is an ordinary transient failure"
                        ) from exc
                    if isinstance(exc, LoginError):
                        # The endpoint ANSWERED and its answer proves nothing about
                        # our token, so the marker is kept but re-bounded: the
                        # failure is fresh evidence of a provider having a bad
                        # minute, not of an exchange whose outcome never arrived.
                        # The shorter bound is what keeps a 5xx from costing the
                        # account the in-doubt shape's window (review round 1, R3).
                        self._rebound_send_marker(row.id, armed, SEND_SHAPE_ANSWERED)
                    # ``{exc}`` alone renders an empty tail for the httpx transport
                    # errors whose str() is empty, which is how a timed-out refresh
                    # came back as "OAuth refresh failed for 'radient': " with no
                    # remedy to read (review round 1, Q-4). The type name is the
                    # part that is always there.
                    detail = str(exc) or type(exc).__name__
                    raise AuthStoreError(
                        f"OAuth refresh failed for '{row.provider}': {detail}"
                    ) from exc
                merged = dict(fresh)
                merged.update(refreshed)
                # Restore identity fields from the stored credential — NEVER
                # rewritten by refresh, whatever the refresh fn returns.
                for field in ("org_id", "org_name", "authorized_at"):
                    if field in fresh:
                        merged[field] = fresh[field]
                    else:
                        merged.pop(field, None)
                # The response IS the acknowledgement the marker was waiting for:
                # the exchange ran, so the token it presented is consumed and the
                # row holds its replacement. Skipped here rather than trusted from
                # ``fresh``, which was read BEFORE the arm — and this write is not
                # ownership-gated for the same reason the tombstone's pop is not:
                # it replaces the very token any marker in this payload was about,
                # after the guard below has established that token, so what it drops
                # is dead for every process rather than live for a peer.
                merged.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
                # Cross-process guard: the server's job processes each build their
                # own AuthStore, so the per-process refresh lock does not cover
                # them. Two processes racing a rotating refresh token both POST
                # the same token; the IdP rotates and the loser's new token is
                # dead. If the stored refresh token changed under us (another
                # process won), skip our write — overwriting would clobber the
                # winner's live token with our dead one and soft-delete the row.
                now_row = self.get_credential(row.id)
                if (
                    now_row is not None
                    and self._presented_refresh_token(dict(now_row.data)) != presented
                ):
                    logger.warning(
                        "refresh race on %s: another process refreshed first; " "keeping its token",
                        row.id,
                    )
                    return dict(now_row.data)
                if now_row is not None and self._grant_dead_at(dict(now_row.data)) is not None:
                    # A sibling tombstoned the grant while our POST was in flight,
                    # and the IdP then answered US with a rotation. A token minted
                    # by a refresh belongs to the family the IdP revoked, so it
                    # must not resurrect the login: the rotation is dropped, the
                    # tombstone the sibling wrote stands, and the caller is told
                    # the truth instead of being handed a token out of a dead
                    # family. Checked AFTER the refresh-race guard because a
                    # rotated token is the more specific explanation of a moved
                    # row, and the guard's outcome keeps the peer's payload
                    # untouched either way.
                    logger.warning(
                        "refresh on %s minted a token after the grant was marked dead; "
                        "the tombstone is kept and the rotation is not persisted",
                        row.id,
                    )
                    raise CredentialInvalidError(
                        f"OAuth grant for '{row.provider}' was refused by the identity "
                        f"provider; run /login {row.provider} to sign in again"
                    )
                self._conn.execute(
                    "UPDATE auth_credentials SET data = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(merged), self._now_ms(), row.id),
                )
                self._conn.commit()
                return merged
            finally:
                self._release_refresh_lease(row.id)

    def _refreshable_row(self, credential_id: int) -> StoredCredential | None:
        """The row a refresh may act on, or ``None`` when there is no such row.

        Shared by both refresh entry points so their precondition cannot drift: a
        row that is gone, disabled, or not an OAuth credential is not a refresh
        failure in either of them, and only the raising variant can then tell a
        failed refresh from a missing row.
        """
        row = self.get_credential(credential_id)
        if row is None or row.disabled_cause is not None:
            return None
        if row.credential_type != "oauth":
            return None
        return row

    async def ensure_oauth_fresh(self, credential_id: int) -> dict[str, Any] | None:
        """Usable OAuth data for ONE specific credential, refreshed if stale.

        The per-row half of the cascade's refresh step, exposed for callers
        that must read a SPECIFIC account's state — quota recovery probes a
        blocked row's own usage, and asking the cascade would resolve to
        whichever sibling outranks the row instead. Returns ``None`` when the
        row is gone, disabled, not an OAuth credential, or its refresh fails:
        the caller keeps whatever verdict the row already carried. Raises
        nothing; a probe is a read, not a routing decision."""
        row = self._refreshable_row(credential_id)
        if row is None:
            return None
        try:
            return await self._ensure_oauth_fresh(row)
        except AuthStoreError:
            return None

    async def ensure_oauth_fresh_or_raise(self, credential_id: int) -> dict[str, Any] | None:
        """`ensure_oauth_fresh`, but a refresh that fails keeps its cause.

        The probe above collapses every unusable state into ``None``, which is
        right for a caller that only needs a verdict and wrong for one that must
        tell an operator WHERE to look: a token endpoint that could not be
        reached and a grant the IdP rejected arrive as the same ``None``, and
        they need different remedies. ``None`` still means the row is gone,
        disabled, or not an OAuth credential; a refresh that failed raises
        :class:`AuthStoreError` with the underlying transport or IdP error
        chained on ``__cause__``, so a caller can classify on it.
        """
        row = self._refreshable_row(credential_id)
        if row is None:
            return None
        return await self._ensure_oauth_fresh(row)

    def refresh_deferred(self, credential_id: int) -> bool:
        """Whether an unsettled send is holding this row's refresh off right now.

        READ-ONLY, and deliberately not :meth:`send_unconfirmed` even though the
        two answer the same question: that one CLEARS a marker it finds dead on
        the way past, and a status surface must be able to ask without changing
        anything. It is the exact condition :meth:`_ensure_oauth_fresh` defers
        on — a LIVE marker for the token this row stores, and an access token
        that is no longer inside its own lifetime. A LIVE BEARER IS NOT A
        DEFERRAL: the marker is about the refresh token and proves nothing about
        a bearer that still works, so that case keeps being served (the same
        escape ``_holds_live_bearer`` gives the refresh path). A tombstoned grant
        is not one either — the IdP refused it, which is a verdict about the
        credential rather than a wait.

        It lives here so the surfaces can say "a refresh is in doubt and is being
        retried" from the row, with the rule in the module that owns the marker,
        instead of each surface re-deriving liveness from the marker payload.
        """
        row = self.get_credential(credential_id)
        if row is None or not isinstance(row.data, dict):
            return False
        data = dict(row.data)
        if self._grant_dead_at(data) is not None:
            return False
        now_ms = self._now_ms()
        if not _send_marker_is_live(
            data.get(REFRESH_SEND_UNCONFIRMED_KEY),
            self._presented_refresh_token(data),
            now_ms=now_ms,
        ):
            return False
        return not self._holds_live_bearer(data, now_ms=now_ms)

    def detached_refresh(self, credential_id: int) -> asyncio.Task[dict[str, Any] | None]:
        """Start a refresh exchange that OUTLIVES a bounded caller's own wait.

        For a caller whose answer is bounded but whose EXCHANGE must not be
        cancelled — the tunnel status surface, whose verdict a desktop route
        polls on open. ``asyncio.wait_for`` around the exchange itself cancels it
        at the bound, and a ``CancelledError`` is a ``BaseException``: it reaches
        none of ``_ensure_oauth_fresh``'s handlers, so the write-ahead send
        marker stays armed with nothing left to resolve it. On a due bearer that
        is not a degraded state but an outage (see
        :data:`UNCONFIRMED_SEND_TTL_S`), so such a caller stops WAITING instead —
        by bounding its wait with ``asyncio.wait``, which leaves the task alone,
        rather than with a ``wait_for`` around the exchange or around a shield of
        it (both of which are the same defect with different amounts of stdlib
        ceremony). This is the task such a caller hands off.

        HOW LONG THE HAND-OFF LASTS IS THE CALLER'S LOOP, not this module's promise
        (agent review round 1, R1): a caller in a process that keeps running gets
        the full guarantee, while one under ``asyncio.run`` that returns immediately
        (``lop tunnel status``) has its loop torn down at process exit and this task
        cancelled with it — the request dies mid-wire and the marker stays armed for
        the derived bound, which is the accepted consequence rather than a defect
        (design §6.1; a drain here would restore the measured 30.9 s poll latency the
        bound exists to remove).

        The exchange gets its OWN store, and that is a correctness requirement
        rather than tidiness: a bounded caller is inside
        ``with closing(AuthStore())``, so an exchange still running when the
        caller returns would write its marker resolution and its rotation into a
        closed sqlite connection. The new store is built from this one's own
        path and config, so it reads the same database and the row the caller
        already resolved. Single-flight is NOT weakened by the second store:
        :meth:`_try_refresh_lease` is one atomic upsert judged by ``rowcount``
        against the shared database, not a per-process lock, so a second
        exchange — a second caller, or a peer process — loses the lease and is
        served the stored row instead.

        The caller must NOT try to free the lease when it gives up waiting. The call
        that used to be in `report.py` is deleted rather than re-pointed, and the
        precise reason is worth keeping: ``_release_refresh_lease`` is
        HOLDER-SCOPED and this exchange's lease belongs to the supervisor's store,
        so a release from the caller's store is a no-op — a line that reads like a
        safety mechanism while doing nothing. It would be a real hazard the moment
        it became effective (a release keyed on the credential alone, or routed
        through this store), because a freed lease lets a peer take it and re-present
        the token while this POST is on the wire. The exchange's own ``finally`` is
        the only release.
        """
        store = AuthStore(
            self._db_path,
            config_dir=self._config_dir,
            config_overrides=self._config_overrides,
        )
        task = asyncio.create_task(store.ensure_oauth_fresh_or_raise(credential_id))
        _DETACHED_REFRESHES.add(task)
        task.add_done_callback(lambda finished: _release_detached_refresh(finished, store))
        return task

    # -- selection: stickiness + round-robin -------------------------------------

    def deprioritize_credential(self, provider: str, credential_id: int) -> None:
        """Sort ``credential_id`` last for ``provider`` without blocking it.

        The half of ``rotate_sibling`` that applies when the PROVIDER failed
        rather than the credential: a 529 storm must move the next attempt onto
        another account, but blocking would strand a healthy account (and,
        repeated across the pool, strand every one of them).

        The mark EXPIRES on its own after :data:`DEPRIORITIZE_TTL_MS`. Clearing
        it on a successful request is not sufficient by itself, and the reason is
        circular: a demoted credential sorts last, so it is not selected, so it
        never gets the success that would clear it. Without a TTL a single 529
        left an account bottom-of-pool for the life of the process -- the same
        "healthy account effectively out of rotation" outcome this whole change
        exists to prevent, arrived at the slow way.

        Keyed by the STORAGE id, like blocks and stickiness: an alias and its
        base are one credential pool, so a demotion earned under one spelling
        must reorder the other's selection too.
        """
        marks = self._deprioritized.setdefault(self._storage_id(provider), {})
        marks[credential_id] = self._now_ms() + DEPRIORITIZE_TTL_MS

    def clear_deprioritized(
        self, provider: str, credential_id: int | Iterable[int] | None = None
    ) -> None:
        """Restore full priority for a credential, several, or all of them.

        Takes an explicit id set rather than always clearing the provider,
        because ``_selection_order`` runs once per cascade TIER with a different
        subset of rows: a one-row tier finding its only row demoted must not
        drop the marks belonging to rows in another tier that it never saw.
        """
        provider = self._storage_id(provider)
        marks = self._deprioritized.get(provider)
        if marks is None:
            return
        if credential_id is None:
            self._deprioritized.pop(provider, None)
            return
        ids = {credential_id} if isinstance(credential_id, int) else {int(i) for i in credential_id}
        for one in ids:
            marks.pop(one, None)
        if not marks:
            self._deprioritized.pop(provider, None)

    def _active_demotions(self, provider: str) -> set[int]:
        """Ids still demoted, dropping any whose TTL has passed.

        Expiry is evaluated on READ rather than by a timer: the marks are only
        consulted here, so a lazy sweep is both sufficient and free of a
        background task that would have to be owned and cancelled.
        """
        provider = self._storage_id(provider)
        marks = self._deprioritized.get(provider)
        if not marks:
            return set()
        now = self._now_ms()
        for cid in [cid for cid, until in marks.items() if until <= now]:
            marks.pop(cid, None)
        if not marks:
            self._deprioritized.pop(provider, None)
            return set()
        return set(marks)

    def _selection_order(
        self,
        rows: list[StoredCredential],
        provider: str,
        session_id: str | None,
        *,
        read_only: bool = False,
        model_id: str = "",
    ) -> list[StoredCredential]:
        if not rows:
            return []
        # Same key as ``_set_sticky`` writes: an alias and its base share one
        # credential, so they must share one stickiness and one round-robin
        # cursor, or a session alternating spellings would alternate accounts.
        # Normalized here, ahead of both the demotion marks and the base order,
        # so every keyed structure below sees one spelling.
        provider = self._storage_id(provider)
        ordered = self._base_selection_order(rows, provider, session_id, model_id=model_id)
        # Demotion is applied LAST, to the finished order.
        #
        # It used to run first, which did not work: both orderings below rotate
        # the list (`rows[i:] + rows[:i]`), so a row moved to the back was
        # rotated straight back towards the front. With three credentials and a
        # session whose hash landed on index 1, the account that had just failed
        # was tried SECOND -- the pool was never fully walked, which is the bug
        # the demotion exists to fix. Applying it here cannot be undone by a
        # later step, and because the partition is stable the relative order the
        # sticky/hash/round-robin choice produced is otherwise preserved.
        demoted = self._active_demotions(provider)
        if not demoted:
            return ordered
        # Every row demoted means the pool has been walked once, so the marks
        # describe an outage rather than any one account: they are stale and the
        # rows are equally good again. Only THIS tier's rows are cleared -- the
        # cascade calls this once per credential tier with a different subset,
        # so popping the whole provider would let a one-row tier wipe the marks
        # belonging to rows it never saw. Judged on the RAW mark set, before the
        # sticky exemption below, so "the whole tier is demoted" keeps meaning
        # exactly that and the stale-marks rule is unchanged by stickiness.
        #
        # This branch is NOT the cascade's safety net, and exactly one pass
        # reaches it. On :meth:`_resolve`'s FIRST pass ``_usable_key_rows`` has
        # already dropped demoted rows, so an all-demoted tier arrives empty and
        # returns at the guard above. On the second pass ``ignore_demotions``
        # suppresses that filter, so the rows arrive whole and land here --
        # meaning in practice this branch is reached only by a ``read_only``
        # second pass, because the normal one cleared the marks before
        # recursing and has no demotions left to find.
        #
        # That is precisely why the ``read_only`` gate below is load-bearing
        # rather than dead: it is the last thing standing between an isolated
        # request and a mark it is not entitled to clear. The net that makes a
        # demoted lone row resolvable at all is the ``ignore_demotions`` pass in
        # :meth:`_resolve`; do not reason about cascade-wide all-demoted
        # behaviour from here.
        if all(r.id in demoted for r in ordered):
            # Not under ``read_only``: clearing the marks is a routing DECISION,
            # and an isolated request running beside a user's turn must not be
            # able to move that turn's account. It still gets the same order --
            # all rows demoted means no reordering either way -- so the only
            # difference is that it decides nothing, which is the contract.
            if not read_only:
                self.clear_deprioritized(provider, [r.id for r in ordered])
            return ordered
        # The session's sticky row is exempt from the sort-last: see
        # ``_usable_key_rows`` for why a demotion is a preference for NEW picks
        # and never a reason to move a session off the account it is
        # transacting on. ``_base_selection_order`` already put it first, and
        # the partition is stable, so exempting it keeps it there.
        movable = demoted - {self.session_credential_id(provider, session_id)}
        preferred = [r for r in ordered if r.id not in movable]
        return preferred + [r for r in ordered if r.id in movable]

    def _base_selection_order(
        self,
        rows: list[StoredCredential],
        provider: str,
        session_id: str | None,
        *,
        model_id: str = "",
    ) -> list[StoredCredential]:
        """Stickiness, then a usage-ranked per-session pick, then round-robin.

        ``provider`` arrives already normalized by :meth:`_selection_order`,
        its only caller. ``model_id`` names the model the request will run so
        the usage ranking can ignore tier caps that do not gate it.
        """
        if session_id:
            sticky_id = self.session_credential_id(provider, session_id)
            sticky = next((r for r in rows if r.id == sticky_id), None)
            if sticky is not None:
                rest = [r for r in rows if r.id != sticky.id]
                return [sticky, *rest]
            # No sticky yet: this is the session's FIRST pick for the provider
            # (or its re-pick after a demotion cleared the sticky), the one
            # moment the choice is free. Rank by cached usage so it lands on
            # the account with the most headroom; the hash rotation is what
            # remains when usage says nothing, and what breaks ties.
            return self._usage_ranked_order(rows, provider, session_id, model_id)
        # No session: round-robin across calls.
        provider_key = f"{provider}:any"
        start = self._round_robin.get(provider_key, 0) % len(rows)
        self._round_robin[provider_key] = start + 1
        return rows[start:] + rows[:start]

    @staticmethod
    def _hash_order(rows: list[StoredCredential], session_id: str) -> list[StoredCredential]:
        """The per-session rotation: ``crc32(session_id) % len(rows)``.

        Deterministic in the session id so a session that re-resolves before
        it has a sticky account meets the same order, and spread across
        sessions so concurrent starts do not stampede one row.
        """
        if not rows:
            return []
        index = zlib.crc32(session_id.encode("utf-8")) % len(rows)
        return rows[index:] + rows[:index]

    def _usage_cache_store(self) -> "UsageCacheStore | None":
        """The shared usage cache handle; ``None`` when it cannot be built.

        Probed once: a cache that failed to construct once will fail again,
        and a store that re-tried on every resolve would pay the failure on
        each. The same contract as ``ModelConfigurator._usage_cache_store``
        -- a missing cache is a permanent miss, never an error. The handle
        opens its SQLite connection lazily and :meth:`_usage_ranked_order`
        closes it again after each ranking, so between first resolves this
        store holds no file descriptor for it (see that method).
        """
        if not self._usage_cache_probed:
            self._usage_cache_probed = True
            try:
                from local_operator.providers.usage_cache import UsageCacheStore

                self._usage_cache = UsageCacheStore()
            except Exception:  # noqa: BLE001 -- no cache = hash order, never fatal
                logger.debug("usage pick: cache unavailable", exc_info=True)
                self._usage_cache = None
        return self._usage_cache

    def _cached_remaining_fraction(
        self, row: StoredCredential, provider: str, model_id: str, now_ms: int
    ) -> float | None:
        """Remaining shared quota (0..1) for ``row`` from the cache, or ``None``.

        ``None`` is UNKNOWN, not zero: no report, an unparseable one, one
        older than :data:`USAGE_PICK_MAX_REPORT_AGE_MS`, or a report whose
        windows carry no measurable fraction. The caller treats unknown as
        neutral so a cold cache cannot demote an account.

        TWO sources feed this, and the newer report wins. The per-account
        preflight row is keyed exactly as the message-boundary preflight
        builds it (``ModelConfigurator._cached_account_usage``): the storage
        provider id plus the row's email, else account id, else ``cred:<id>``.
        But that row is only ever written while ``retry.usageAwareFallback``
        is on, which is NOT the shipped default -- with the reactive path
        off, the only usage the cache holds is the warmer's per-provider
        payload (``/usage`` and the TUI's background warm), whose reports
        carry ``identity``. Reading only the preflight row made this pick a
        silent no-op on a stock config, so the warmer payload is sliced per
        account too (:meth:`UsageCacheStore.latest_account_report`), matched
        on every label an enumerator might have attached to the row.

        Only OAuth rows carry an identity; an API-key row is keyed by a hash
        of its secret on the preflight side, and re-deriving that here would
        put the secret on a read path that never needed it, so key rows are
        simply unknown. The health reduction is the same one the reactive
        path uses (:func:`usage_health`): shared windows always count, a
        tier-scoped cap counts only when ``model_id`` is in that tier, and an
        enabled extra-usage meter supersedes the plan windows.
        """
        if row.credential_type != "oauth":
            return None
        cache = self._usage_cache_store()
        if cache is None:
            return None
        from local_operator.providers.usage import UsageReport, usage_health
        from local_operator.providers.usage_cache import account_preflight_key

        data = row.data
        identity = data.get("email") or data.get("account_id") or f"cred:{row.id}"
        candidates: list[UsageReport] = []
        reports = cache.get(account_preflight_key(provider, str(identity)), include_expired=True)
        if reports:
            candidates.append(reports[0])
        # The warmer labels an account by whichever of these its enumerator
        # found first (``list_oauth_accesses`` vs ``list_oauth_identities``
        # differ on the identity_key fallback), so offer every spelling.
        labels = {
            str(value)
            for value in (data.get("email"), data.get("account_id"), row.identity_key)
            if value and not str(value).startswith("oauth:")
        }
        labels.add(f"cred:{row.id}")
        warmed = cache.latest_account_report(provider, labels)
        if warmed is not None:
            candidates.append(warmed)
        # A report that measures nothing cannot describe the account, however
        # fresh it is. The warmer's first failure for a never-seen account
        # (``_mark_account_failure`` with no last-good) is exactly that: a
        # stub stamped ``now`` with an empty ``limits``. Letting it win the
        # recency contest below would mask a slightly older real preflight
        # report and rank a 95%-used account as neutral (review round 2, F7).
        candidates = [
            r for r in candidates if any(limit.amount.fraction() is not None for limit in r.limits)
        ]
        if not candidates:
            return None
        report = max(candidates, key=lambda r: r.fetched_at)
        if report.fetched_at <= 0:
            return None
        if now_ms - report.fetched_at > self._usage_pick_max_age_ms:
            # Past the age cutoff a report is still a lower bound on usage as
            # long as none of its measured windows has rolled over since it
            # was fetched -- and a window that reset between then and now
            # carries a ``resets_at_ms`` in the past, so "every reset is still
            # ahead" is exactly that test. A window with no reset timestamp
            # cannot be vouched for, so it makes the whole report unknown.
            measured = [limit for limit in report.limits if limit.amount.fraction() is not None]
            if not measured or any(
                limit.resets_at_ms is None or limit.resets_at_ms <= now_ms for limit in measured
            ):
                return None
        health = usage_health(report, model_id, now_ms=now_ms)
        if health.state == "unknown" or health.remaining_fraction is None:
            return None
        return health.remaining_fraction

    def _usage_ranked_order(
        self,
        rows: list[StoredCredential],
        provider: str,
        session_id: str,
        model_id: str,
    ) -> list[StoredCredential]:
        """Least-loaded first, ties broken by the per-session hash.

        WHY. Account choice used to be ``crc32(session_id) % n`` alone --
        uniform over sessions, blind to how full each account already is.
        With five OAuth accounts and ~16 concurrent sessions plus their
        subagents, three accounts sat at 65-99% of their 5-hour window while
        two sat at 6% and 29%, and sessions kept being told "All 5 OAuth
        credentials unusable" because the hash kept landing new work on the
        rows that were already spent. The REACTIVE usage-aware path
        (``ModelConfigurator.preflight_usage``) only moves a session once its
        account is at the reserve threshold; by then the skew exists. This is
        the PROACTIVE half: it runs once, at the moment a session first binds
        to an account, and steers that binding toward headroom.

        HOW. Every row's remaining shared fraction is read from the cache
        (:meth:`_cached_remaining_fraction`, no network). The rows within
        ``tolerance`` of the best-known remaining fraction form a bucket of
        "equally good" accounts; unknown rows join the bucket as neutral
        members, since no evidence is not evidence against. The bucket is
        rotated by the session hash -- the same rotation the plain order
        applied to the whole pool -- so a burst of sessions and subagents
        starting together spread across the bucket instead of herding onto
        one row. Rows known to be worse follow, best first.

        STICKINESS is untouched: the caller only reaches here without a
        sticky, and the winner is pinned by ``_resolve`` exactly as before,
        so a session never moves mid-conversation (the provider's prompt
        cache is per account, and moving would rewrite the whole prefix).

        FAIL OPEN. Any exception, a missing cache, or a pool with nothing
        known collapses to the pre-existing hash order, unchanged -- the pick
        may be no worse than it was before this existed.

        The cache connection is CLOSED after every ranking rather than held
        for the store's life. A ranking happens once per session per provider
        (plus a re-pick after a demotion), so reopening costs a few
        milliseconds a handful of times, whereas holding it would add a
        second SQLite connection (three descriptors with WAL sidecars) to
        every ``AuthStore`` -- and the test suite, which builds hundreds of
        stores per worker and does not always close them, hit ``EMFILE``
        under the default 256-descriptor limit the first time it did.
        """
        fallback = self._hash_order(rows, session_id)
        if not self._usage_aware_pick or len(rows) < 2:
            return fallback
        try:
            now_ms = self._now_ms()
            remaining = {
                r.id: self._cached_remaining_fraction(r, provider, model_id, now_ms) for r in rows
            }
            known = [v for v in remaining.values() if v is not None]
            if not known:
                return fallback
            best = max(known)
            floor = best - self._usage_pick_tolerance
            bucket: list[StoredCredential] = []
            rest: list[StoredCredential] = []
            for r in rows:
                value = remaining[r.id]
                # DELIBERATE: an unknown row outranks a known-worse one. It
                # joins the bucket beside the best-known rows rather than
                # sorting behind a row that is provably 30% full, because a
                # missing report is most often a cold cache or a window that
                # just reset -- and penalising an account for the absence of
                # data is how a fresh login gets starved by its own silence.
                (bucket if value is None or value >= floor else rest).append(r)
            # Stable sort: rows with equal remaining keep the store's row
            # order, which is the order the hash fallback also assumes.
            rest.sort(key=lambda r: -(remaining[r.id] or 0.0))
            ordered = self._hash_order(bucket, session_id) + rest
            logger.debug(
                "usage pick: %s session=%s model=%s remaining=%s -> %s",
                provider,
                session_id,
                model_id or "-",
                {cid: (None if v is None else round(v, 3)) for cid, v in remaining.items()},
                [r.id for r in ordered],
            )
            return ordered
        except Exception:  # noqa: BLE001 -- a ranking failure must never block a resolve
            logger.debug("usage pick: ranking failed; hash order", exc_info=True)
            return fallback
        finally:
            if self._usage_cache is not None:
                try:
                    self._usage_cache.close()
                except Exception:  # noqa: BLE001 -- releasing a handle can never fail a resolve
                    logger.debug("usage pick: cache close failed", exc_info=True)

    def _set_sticky(self, provider: str, session_id: str | None, credential_id: int | None) -> None:
        if not session_id:
            return
        provider = self._storage_id(provider)
        if credential_id is None:
            self._sticky.pop((provider, session_id), None)
        else:
            self._sticky[(provider, session_id)] = credential_id

    def session_credential_id(self, provider: str, session_id: str | None) -> int | None:
        """The credential ``session_id`` is sticky to for ``provider``, if any.

        The read half of :meth:`pin_session_credential`. The quota preflight
        captures this BEFORE its boundary walk resolves anything, because the
        walk's own resolve pins the session to whatever row it lands on — so
        "is the account under verdict the one this session is transacting
        on?" can only be answered from a reading taken ahead of the walk. Same
        storage-id normalisation as the write, so an alias and its base agree.
        """
        if not session_id:
            return None
        return self._sticky.get((self._storage_id(provider), session_id))

    def release_session_credential(self, provider: str, session_id: str | None) -> None:
        """Forget a session's sticky selection so its next resolve picks afresh.

        The public clear beside :meth:`pin_session_credential`. Needed by the
        quota preflight when it demotes a row the session was only just pinned
        to by the walk's own resolve (a fresh pick, nothing cached on it yet):
        the cascade keeps a demoted STICKY row in service on purpose (see
        ``_usable_key_rows``), so without dropping the pin the re-resolve would
        hand back the very row the walk is trying to move off. A no-op without
        a session id, like the write it mirrors.
        """
        self._set_sticky(provider, session_id, None)

    def pin_session_credential(
        self, provider: str, session_id: str | None, credential_id: int
    ) -> None:
        """Point a session's sticky selection at ``credential_id``.

        The public half of :meth:`_set_sticky`, for callers that have just
        PROBED a specific account and must route the session to the account
        the quota verdict was about. Quota-aware preflight re-checks blocked
        siblings one by one; without pinning, the cascade's round-robin /
        stickiness could hand the request to a different row than the one
        whose usage was just read, and the session would keep failing on an
        account the recovery walk had already judged. A no-op without a
        session id, like the sticky write it wraps."""
        self._set_sticky(provider, session_id, credential_id)

    def _usable_key_rows(
        self,
        provider: str,
        credential_type: str,
        source: str | None,
        *,
        ignore_demotions: bool = False,
        model_id: str = "",
        session_id: str | None = None,
        exclude_keys: Collection[str] | None = None,
        exclude_credential_ids: Collection[int] | None = None,
    ) -> list[StoredCredential]:
        # ``model_id`` scopes the block filter: an account blocked only for a
        # model family (a spent scoped weekly cap) still serves every other
        # family, so a resolve that names a different model must see the row.
        from local_operator.providers.local import LOCAL_PROVIDER_IDS, resolve_base_url

        if provider in LOCAL_PROVIDER_IDS:
            selected = self.active_local_credential(provider, resolve_base_url(provider))
            candidates = [selected] if selected is not None else []
        else:
            candidates = self.list_credentials(provider)
        rows = [
            r
            for r in candidates
            if r.credential_type == credential_type
            and not self.is_blocked_for_model(r.id, provider, model_id)
        ]
        if source is not None:
            rows = [r for r in rows if r.data.get("source") == source]
        if exclude_keys or exclude_credential_ids:
            # Rows the CALLER already saw rejected are hidden from THIS resolve
            # alone: nothing is blocked, demoted or repointed, so the pick comes
            # back exactly as it went in for every other resolver. This is the
            # read-only half of "an isolated errand may serve ITSELF from a
            # sibling" — the write half (blocking the failing row) belongs to
            # the turn's rotation, not to decoration.
            #
            # BOTH identifiers, because neither alone is sufficient. A bearer
            # string does not survive a refresh: an OAuth row whose token the
            # concurrent turn rotated between the errand's resolve and its retry
            # would escape a key-only exclusion and be served again, which is
            # the whole failure this exclusion exists to prevent. And an id
            # alone cannot cover a store tier that produces a bearer without a
            # row behind it (env/legacy keys resolve with no credential id).
            excluded_ids = set(exclude_credential_ids or ())
            rows = [
                r
                for r in rows
                if r.id not in excluded_ids
                and not any(self._row_matches_key(r, k) for k in (exclude_keys or ()))
            ]
        # Drop demoted rows from the TIER, not merely sort them last, when some
        # other credential is still reachable.
        #
        # Ordering alone is not enough here, because the cascade is a sequence
        # of tiers and a tier is consulted whole: an OAuth row, or an api_key
        # row with `source="login"`, wins its tier before a row in a later tier
        # is ever looked at. So a demoted row that is ALONE in its tier kept
        # winning the cascade -- and `rotate_sibling` kept reporting that a
        # sibling existed, which told the driver rotation was progressing while
        # the same failing bearer came back every time. A healthy credential one
        # tier down never received a single request.
        #
        # Dropping is safe WITHOUT a "is anything else reachable?" guard, and
        # deliberately has none. Such a guard could only count database ROWS,
        # while the cascade also resolves from the env var (tier 5) and the
        # fallback resolver (tier 7), which are not rows: a demoted lone stored
        # row would then never yield, and an exported ANTHROPIC_API_KEY beside a
        # signed-in account became unreachable where it used to be the fallback.
        #
        # The safety net for the resulting empty tier is the second pass at the
        # end of :meth:`_resolve`: if demotions are the ONLY reason the whole
        # cascade came back empty, it resolves once more with
        # ``ignore_demotions``, so a demoted lone row is still served rather
        # than reported as no credential at all. ``_selection_order``'s
        # all-demoted branch cannot be that net: on the first pass this filter
        # runs ahead of it and hands it an empty list, and on the second pass
        # the net has already fired -- that is what suppressed this filter.
        demoted = set() if ignore_demotions else self._active_demotions(provider)
        if demoted:
            # Every row in THIS tier is demoted: yield the tier so the cascade
            # moves on to whatever comes next -- another tier, the env var, or
            # the resolver -- and, if nothing else serves, the second pass at
            # the end of :meth:`_resolve` clears the marks together and
            # re-resolves (the sticky then wins as usual). Judged on the RAW
            # rows, BEFORE the sticky exemption below, and it has to be: run
            # after it, an all-demoted tier with a sticky inside came back as
            # the one sticky row, ``_selection_order`` read that one-row tier
            # as "all demoted" and cleared only the sticky's mark -- an
            # outage's marks then decayed asymmetrically and every other
            # session's fresh pick skewed onto the sticky's account for the
            # rest of the TTL. The stale-marks rule is about the tier, and
            # stickiness must not change what "the whole tier" means.
            if all(r.id in demoted for r in rows):
                return []
            # The session's STICKY row survives the drop (a BLOCKED sticky does
            # not: the block filter above ran first and blocks are verdicts
            # that the account cannot serve). A demotion is a preference about
            # where NEW picks go; it is never a reason to move a session that
            # is already transacting on the account. The provider's prompt
            # cache is per account, so moving a live conversation rewrites its
            # whole prefix (150-500k tokens at cache-write price) to buy
            # nothing — the sibling has never seen it. Measured on this host:
            # 374 such moves in 30h, 102M cache-write tokens, ~38% of every
            # Anthropic cache write, most of them 2-70s after a full cache hit
            # in the same conversation. The marks are process-wide, so this
            # exemption is also what keeps ANOTHER session's demotion of this
            # account from evicting a sibling session mid-conversation on it.
            # A session whose OWN request 529'd still moves: ``rotate_sibling``
            # clears that session's sticky for a server fault before the mark
            # is consulted, so the exemption finds nothing to keep. The
            # reactive quota path already keeps the warm-account promise
            # (``rotate_sibling``: "sticky preserved" on a usage 429); this is
            # the same rule applied to the preference marks.
            #
            # The exemption is keyed on stickiness alone, not on whether the
            # session has actually sent a request yet — the store cannot tell.
            # A caller that demotes a row the session was pinned to by a
            # resolve moments ago (a fresh pick) and wants the next resolve to
            # move must release the pin first (``release_session_credential``).
            sticky_id = self.session_credential_id(provider, session_id)
            return [r for r in rows if r.id not in demoted or r.id == sticky_id]
        return rows

    # -- the cascade ---------------------------------------------------------

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
        """Resolve the API key for ``provider`` via the 7-step cascade.

        ``read_only`` makes the resolve decide nothing about routing — see
        :meth:`_resolve`. ``model_id`` names the model the request will run,
        so model-family-scoped quota blocks (see
        :meth:`is_blocked_for_model`) only exclude the accounts that cannot
        serve THAT model. ``exclude_keys`` / ``exclude_credential_ids`` hide
        the rows they name from THIS resolve alone — see :meth:`_resolve`.
        """
        key, _row = await self._resolve(
            provider,
            session_id,
            force_refresh=force_refresh,
            read_only=read_only,
            model_id=model_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        return key

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
    ) -> OAuthAccess | None:
        """The identity-carrying record for wire clients.

        Returns :class:`OAuthAccess` for whichever credential the cascade
        picks — ``kind == "oauth"`` with account/org identity when an OAuth
        row wins, ``kind == "api_key"`` otherwise. Runtime/config overrides
        deliberately short-circuit to ``None`` (they aim at gateways where
        stored identity does not apply).

        ``read_only`` resolves without blocking a credential or moving session
        stickiness, for a decorative call running beside a live turn — see
        :meth:`_resolve` and
        :attr:`~local_operator.harness.types.ChatRequest.isolated`.
        ``exclude_keys`` / ``exclude_credential_ids`` hide the rows they name
        from THIS resolve alone — see :meth:`_resolve`.
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return None
        key, row = await self._resolve(
            provider,
            session_id,
            force_refresh=force_refresh,
            read_only=read_only,
            model_id=model_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        if key is None:
            return None
        if row is not None and row.credential_type == "oauth":
            data = row.data
            return OAuthAccess(
                access_token=key,
                credential_id=row.id,
                account_id=data.get("account_id"),
                email=data.get("email"),
                org_id=data.get("org_id"),
                api_endpoint=data.get("api_endpoint"),
                kind="oauth",
                raw=data,
            )
        from local_operator.providers.local import LOCAL_PROVIDER_IDS

        # Carry the endpoint with the selected key. Re-reading configuration at
        # dispatch alone cannot detect a change between selection and dispatch.
        endpoint = (
            row.data.get("endpoint") if row is not None and provider in LOCAL_PROVIDER_IDS else None
        )
        return OAuthAccess(
            access_token=key,
            credential_id=row.id if row is not None else 0,
            kind="api_key",
            api_endpoint=endpoint if isinstance(endpoint, str) else None,
        )

    async def list_oauth_accesses(self, provider: str) -> list[OAuthAccess]:
        """EVERY logged-in OAuth account for ``provider``, each one refreshed.

        :meth:`get_oauth_access` answers "which account will the next request
        run as", and that is the right question for the wire. It is the wrong
        question for a usage report: quota is per account, so a user with two
        accounts on one provider has two answers and the cascade can only ever
        return one of them. Worse, with no ``session_id`` the cascade's
        selection order ROUND-ROBINS, so the single account that got reported
        was not even stable between refreshes.

        Four differences from the cascade, all deliberate, and all the same
        principle: routing decisions must not become reporting decisions.

        - **Blocked credentials are INCLUDED.** ``_usable_key_rows`` drops rows
          under a backoff, which is right for "where do I send this request"
          and exactly wrong here — the commonest reason a credential is blocked
          is that it ran out of quota, so the account a user most needs to see
          on a usage screen was the one guaranteed to be missing from it. Its
          exhausted window IS the explanation for the block.
        - **Stable order, by row id.** Enumeration must not depend on which
          request happened last, or a list of accounts reshuffles itself while
          the user is reading it.
        - **No stickiness.** ``_set_sticky`` pins which credential a SESSION
          transacts on. Reading a quota must not repoint the session's account
          as a side effect.
        - **No blocking on refresh failure.** A routing resolve blocks a row that
          fails to refresh so it can rotate to a sibling for the request in
          hand. Here the row is simply omitted: taking a credential out of
          service is a routing decision, and a read is not entitled to make it.
          The last two are the same principle ``_resolve``'s ``read_only`` mode
          applies to a decorative REQUEST, which needs a bearer but is likewise
          not entitled to route the session.
        - **A PERMANENTLY dead grant is reported, not omitted.** Omission is
          right for a transient miss -- the caller keeps last-good numbers and
          retries -- and wrong for a grant the IdP has refused for good, which
          is a fact about the account the operator has to act on. Such a row
          comes back with ``credential_invalid=True`` and an EMPTY
          ``access_token``, so a caller that wanted a bearer still skips it
          (there is none) while a caller that wants the account's state can
          see why. This does not weaken the invariant above: reporting that a
          grant is dead is still not disabling, blocking or deleting the row.
          The verdict is taken from the row's own persisted tombstone first
          (:data:`GRANT_DEAD_AT_KEY`), so the state is reported at all — a
          live attempt can only report it while the access token is stale
          enough to need refreshing, and a dead grant whose token has not
          expired yet was exactly what used to render as a healthy login.

        Logged-out rows are still excluded — ``list_credentials`` filters on
        ``disabled_cause``, and an account the user signed out of is genuinely
        not theirs to report on.

        Overrides short-circuit for the same reason they do in
        :meth:`get_oauth_access` — they aim at a gateway, where stored identity
        does not apply.
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return []
        key_fn = self._oauth_key_fn(provider)
        rows = [r for r in self.list_credentials(provider) if r.credential_type == "oauth"]
        accesses: list[OAuthAccess] = []
        for row in sorted(rows, key=lambda r: r.id):
            data = row.data if isinstance(row.data, dict) else {}
            if self._grant_dead_at(data) is not None:
                # The refusal is ALREADY on the row, so this read does not have
                # to re-earn it with a POST: a dead grant is a fact about the
                # account, and re-deriving it would be both a wasted round trip
                # and a second chance to present a revoked family's token. It is
                # also the case a live attempt cannot cover — a row whose access
                # token has not expired yet, which the old code reported as a
                # perfectly healthy login while every refresh was being refused.
                logger.info(
                    "usage: credential %s for %s has a dead grant (recorded); re-login " "required",
                    row.id,
                    provider,
                )
                accesses.append(
                    OAuthAccess(
                        access_token="",
                        credential_id=row.id,
                        account_id=data.get("account_id"),
                        email=data.get("email"),
                        org_id=data.get("org_id"),
                        api_endpoint=data.get("api_endpoint"),
                        kind="oauth",
                        raw=data,
                        credential_invalid=True,
                    )
                )
                continue
            try:
                creds = await self._ensure_oauth_fresh(row)
            except CredentialInvalidError:
                # Terminal, so it earns a real log line rather than the debug
                # one a retryable miss gets: this state never clears on its
                # own and the user has to be told to sign in again.
                logger.info(
                    "usage: credential %s for %s has a dead grant; re-login required",
                    row.id,
                    provider,
                )
                accesses.append(
                    OAuthAccess(
                        access_token="",
                        credential_id=row.id,
                        account_id=data.get("account_id"),
                        email=data.get("email"),
                        org_id=data.get("org_id"),
                        api_endpoint=data.get("api_endpoint"),
                        kind="oauth",
                        raw=data,
                        credential_invalid=True,
                    )
                )
                continue
            except RefreshUnconfirmedError as error:
                # R5: the marker state is the one the operator hits after a lost
                # response, and it was invisible here — a bare ``AuthStoreError``
                # is a debug line and an OMITTED row, so the panel said nothing
                # about an account that needs a sign-in or a moment's patience.
                # Logged at INFO with the store's own sentence, which names both
                # remedies. NOT reported as ``credential_invalid``: that flag means
                # "only a re-login clears it" and this state clears itself at the
                # marker's expiry, so publishing it as a dead grant would put
                # wrong copy in front of the user (and make the model picker drop
                # a credential that is about to work again).
                logger.info(
                    "usage: credential %s for %s has a deferred refresh; omitting (%s)",
                    row.id,
                    provider,
                    error,
                )
                continue
            except AuthStoreError:
                logger.debug(
                    "usage: credential %s for %s failed to refresh; omitting",
                    row.id,
                    provider,
                    exc_info=True,
                )
                continue
            key = key_fn(creds) if key_fn else creds.get("access")
            if not key:
                continue
            accesses.append(
                OAuthAccess(
                    access_token=key,
                    credential_id=row.id,
                    account_id=creds.get("account_id"),
                    email=creds.get("email"),
                    org_id=creds.get("org_id"),
                    api_endpoint=creds.get("api_endpoint"),
                    kind="oauth",
                    raw=creds,
                )
            )
        return accesses

    def list_oauth_identities(self, provider: str) -> list[OAuthAccess]:
        """Stored OAuth identities for ``provider``, without minting a bearer.

        :meth:`list_oauth_accesses` is the right enumerator when a live usage
        probe needs a token, and it still omits a row whose refresh raises —
        taking a credential out of service is a routing decision, and a read
        is not entitled to make it. ``/usage`` still has to *name* that
        account: the operator is logged in, the email is on the row, and
        dropping the block is how a refresh-failed login vanished from the
        panel. This sibling never calls ``_ensure_oauth_fresh`` and never
        blocks a row. The token field is empty; callers that need a bearer
        still go through :meth:`list_oauth_accesses`.

        Identity is taken from the stored payload first (email / account_id /
        org_id) and falls back to ``identity_key`` so a row whose token blob
        is unreadable still has a label. Logged-out rows stay excluded —
        ``list_credentials`` already filters ``disabled_cause``.
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return []
        rows = [r for r in self.list_credentials(provider) if r.credential_type == "oauth"]
        identities: list[OAuthAccess] = []
        for row in sorted(rows, key=lambda r: r.id):
            data = row.data if isinstance(row.data, dict) else {}
            email = data.get("email")
            account_id = data.get("account_id")
            org_id = data.get("org_id")
            if not email and not account_id:
                # identity_key is the same field upsert already computed; it
                # is what the cache fingerprint names the account by when the
                # payload has no email.
                fallback = row.identity_key
                if fallback and not str(fallback).startswith("oauth:"):
                    email = fallback
            identities.append(
                OAuthAccess(
                    access_token="",
                    credential_id=row.id,
                    account_id=str(account_id) if account_id else None,
                    email=str(email) if email else None,
                    org_id=str(org_id) if org_id else None,
                    api_endpoint=data.get("api_endpoint"),
                    kind="oauth",
                    raw=data or None,
                )
            )
        return identities

    async def _resolve(
        self,
        provider: str,
        session_id: str | None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
        ignore_demotions: bool = False,
        model_id: str = "",
        exclude_keys: Collection[str] | None = None,
        exclude_credential_ids: Collection[int] | None = None,
    ) -> tuple[str | None, StoredCredential | None]:
        """The 7-step cascade; returns ``(key, winning row or None)``.

        ``ignore_demotions`` runs the cascade as if no credential were demoted.
        It is set only by this method's own second pass (see the tail), where
        demotions have been found to be the sole reason the cascade came back
        empty. Because the second pass sets it, the tail's branch cannot re-arm
        and the recursion terminates at depth two.

        ``read_only`` resolves WITHOUT making any routing decision: no
        credential blocked when its refresh fails, no session stickiness
        written and none cleared. It exists for a request that runs beside a
        user's turn and must not be able to move that turn's account — see
        :attr:`~local_operator.harness.types.ChatRequest.isolated`. The cascade
        still READS stickiness, so a read-only resolve lands on the same
        credential the turn is transacting on, which is the point. A successful
        OAuth refresh still persists the rotated token: that is the same
        account's own bookkeeping, not a decision about where requests go, and
        dropping it would throw away a single-use refresh token.

        ``exclude_keys`` and ``exclude_credential_ids`` hide the rows they name
        from this ONE resolve — a read-only way to ask for a SIBLING after a
        bearer was rejected, leaving every block, demotion and the sticky
        pointer exactly as they were. Threaded only by the failover driver's
        isolated auth re-resolve; the ordinary rotation path hides the failing
        row by blocking it instead (``rotate_sibling``), which a decorative call
        must not do. The id form is the load-bearing one for OAuth, whose bearer
        string changes under a refresh — see :meth:`_usable_key_rows`.
        """

        def pin(credential_id: int | None) -> None:
            """Write (or, with ``None``, clear) session stickiness — unless this
            resolve is read-only, in which case it is not ours to move."""
            if not read_only:
                self._set_sticky(provider, session_id, credential_id)

        # 1. Runtime override
        runtime = self._runtime_overrides.get(provider)
        if runtime:
            return runtime, None

        # 2. Config override
        config = self._config_overrides.get(provider)
        if config:
            return config, None

        # 3. OAuth credential
        oauth_rows = self._usable_key_rows(
            provider,
            "oauth",
            source=None,
            ignore_demotions=ignore_demotions,
            model_id=model_id,
            session_id=session_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        for row in self._selection_order(
            oauth_rows, provider, session_id, read_only=read_only, model_id=model_id
        ):
            try:
                creds = await self._ensure_oauth_fresh(row, force=force_refresh)
            except AuthStoreError:
                if not read_only:
                    self.block_credential(row.id, provider)  # try a sibling
                continue
            key_fn = self._oauth_key_fn(provider)
            key = key_fn(creds) if key_fn else creds.get("access")
            if key:
                pin(row.id)
                refreshed = self.get_credential(row.id)
                return key, refreshed or row
        if oauth_rows and force_refresh:
            # Every sibling failed its refresh — surface the failure so the
            # failover layer can block/back off instead of silently looping.
            raise AuthStoreError(f"All OAuth credentials for '{provider}' failed to refresh")
        # PR-15: with NO oauth rows, force_refresh falls through to tiers 4-7.

        # 4. API key persisted by interactive login
        login_rows = self._usable_key_rows(
            provider,
            "api_key",
            source="login",
            ignore_demotions=ignore_demotions,
            model_id=model_id,
            session_id=session_id,
            exclude_keys=exclude_keys,
            exclude_credential_ids=exclude_credential_ids,
        )
        for row in self._selection_order(
            login_rows, provider, session_id, read_only=read_only, model_id=model_id
        ):
            key = row.data.get("key")
            if key:
                pin(row.id)
                return key, row

        # Leaving the OAuth tier: clear session stickiness so identity
        # attribution stops for non-OAuth requests (PR-16; cleared before
        # step 5, regardless of which later tier ends up winning).
        pin(None)

        # 5. Env var tier (the process environment; the plaintext credentials.env
        # file is no longer read here, PR2a).
        env_key = self._env_api_key(provider)
        if env_key:
            return env_key, None

        # 6. Stored api_key without source="login" (e.g. broker migration)
        stored_rows = [
            row
            for row in self._usable_key_rows(
                provider,
                "api_key",
                source=None,
                ignore_demotions=ignore_demotions,
                model_id=model_id,
                session_id=session_id,
                exclude_keys=exclude_keys,
                exclude_credential_ids=exclude_credential_ids,
            )
            if row.data.get("source") != "login"
        ]
        for row in self._selection_order(
            stored_rows, provider, session_id, read_only=read_only, model_id=model_id
        ):
            key = row.data.get("key")
            if key:
                pin(row.id)
                return key, row
        # 7. Fallback resolver
        resolver = self._fallback_resolvers.get(provider)
        if resolver is not None:
            return resolver(provider), None

        # Nothing in the whole cascade -- but demotions are a ROUTING
        # preference, never a statement that a credential is unusable. If they
        # are the only reason this came back empty, they have outlived their
        # purpose (there is nowhere else to route to), so clear them and resolve
        # once more. Without this a sole demoted credential resolved to None and
        # the caller was told no credential was configured, which is exactly the
        # misdiagnosis this change set out to remove.
        #
        # A ``read_only`` resolve takes this pass too. It must: dropping a
        # demoted row from its tier is the destructive half of demotion, and a
        # resolve that is forbidden from deciding anything about routing cannot
        # be handed that half alone -- it would report "no credential" for a
        # credential that is merely deprioritised, which is the misdiagnosis at
        # issue. What it does NOT do is clear the marks: that is the routing
        # decision, and it stays reserved for the caller who owns the turn.
        # ``ignore_demotions`` gives the same answer without touching state.
        if not ignore_demotions and self._active_demotions(provider):
            if not read_only:
                self.clear_deprioritized(provider)
            return await self._resolve(
                provider,
                session_id,
                force_refresh=force_refresh,
                read_only=read_only,
                ignore_demotions=True,
                model_id=model_id,
                exclude_keys=exclude_keys,
                exclude_credential_ids=exclude_credential_ids,
            )

        return None, None

    def _env_api_key(self, provider: str) -> str | None:
        # The env leg resolves through the SHARED store-first reader
        # (``registry.provider_env_key``), which is alias-aware (a flavour
        # authenticates with its base provider's var) so the cascade,
        # ``is_usable`` and the catalogue enrichment cannot disagree about
        # whether a key runs a flavour. It reads the provider-class store row
        # first and the process environment second; the legacy ``credentials.env``
        # file leg is GONE (PR2a).
        #
        # The cascade ORDER is untouched: this is still step 5, still one value,
        # still before the stored-api_key and fallback-resolver rungs.
        #
        # ``base`` is the manager's own config root, so a store the caller
        # configured elsewhere is the one consulted; unset means the
        # HOME-derived default, which is what a CLI invocation wants.
        return provider_env_key(provider, base=self._config_dir)

    # -- failover support --------------------------------------------------------

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: BaseException,
        api_key: str | None = None,
        block_ms: int = DEFAULT_BLOCK_MS,
        *,
        model_id: str = "",
    ) -> bool:
        """a/b/c tier-1 step (c): drop the failing credential, keep a sibling.

        Usage-limit errors only get a temporary block (sticky preserved — the
        sibling rotation happens outside the backoff window). Invalidated
        tokens are soft-deleted. Returns whether another enabled credential
        of the same type remains.
        """
        from local_operator.providers.failover import (
            is_invalidated_credential_error,
            is_server_side_failure,
            is_usage_limit_error,
            retry_after_ms_from_error,
        )

        rows = self.list_credentials(provider)
        if api_key is not None:
            failing = next((r for r in rows if self._row_matches_key(r, api_key)), None)
        elif session_id:
            # Stickiness is written under the storage id (`_set_sticky`), so the
            # lookup must ask with the same spelling or a flavour id would never
            # find the failing row it is sticky to.
            failing = next(
                (
                    r
                    for r in rows
                    if r.id == self._sticky.get((self._storage_id(provider), session_id))
                ),
                None,
            )
        else:
            failing = None

        usage_limited = is_usage_limit_error(error)
        # A provider-wide fault (5xx/529 overload, timeout) is not evidence
        # against the CREDENTIAL. Blocking it would take a healthy account out
        # of the pool for a minute because the provider had a bad second, and
        # under a sustained outage that walks the whole pool into the blocked
        # state until the session has nothing left to try -- while every one of
        # those accounts would have served the very next request. So the row is
        # left usable and only the sticky pointer moves, which is enough to send
        # THIS attempt to a sibling.
        server_side = is_server_side_failure(error)
        if failing is not None:
            if server_side:
                self.deprioritize_credential(provider, failing.id)
            else:
                retry_after = retry_after_ms_from_error(error)
                # A usage-limit 429 names the family the request ran on, not
                # the window that spent: an opus request can be refused by the
                # shared 5-hour window as surely as a fable one by its scoped
                # weekly. Scoping the block to the family is the under-block
                # side of that ambiguity, and it is the side that heals — the
                # next 429 on another family writes its own scope, and the
                # preflight's usage probe upgrades to an account-wide block
                # the moment a shared window is the one binding. The
                # over-block (account-wide on a family verdict) is the side
                # that strands spendable quota behind "all credentials
                # unusable", so it is never written from a family-named
                # rotation. The family comes from the model the request ran,
                # via the same parser the usage layer's tier rows key on.
                from local_operator.model.registry import model_family

                family = model_family(model_id) if model_id else ""
                scope = f"model:{family}" if (usage_limited and family) else ""
                self.block_credential(
                    failing.id,
                    provider,
                    block_scope=scope,
                    block_ms=max(block_ms, retry_after or 0),
                )
            if usage_limited:
                # Sticky preserved: same account stays first after backoff.
                pass
            else:
                self._set_sticky(provider, session_id, None)
                if is_invalidated_credential_error(error):
                    self.disable_credential(failing.id, cause="invalidated-token")

        credential_type = failing.credential_type if failing else None
        siblings = [
            r
            for r in self.list_credentials(provider)
            if r.id != (failing.id if failing else None)
            and not self.is_blocked(r.id, provider)
            and (credential_type is None or r.credential_type == credential_type)
        ]
        if siblings:
            return True
        # No untried sibling of the SAME TYPE remains -- which is not the same
        # as "nothing else is reachable". The cascade has other tiers: another
        # credential type, the env var, the fallback resolver. Clearing the
        # demotion here erased the mark in the very call that set it whenever
        # the failing row had no same-type sibling (an OAuth account beside a
        # pasted key -- precisely the shape a Z.AI sign-in creates), so tier 3
        # re-served the identical failing row and the healthy credential one
        # tier down was never asked.
        #
        # So the mark STANDS. It is not permanent: it expires on its TTL, it is
        # cleared when the credential next serves a request, and
        # `_selection_order` drops the whole set as stale once every row it sees
        # is demoted. Any of those returns this credential to service; none of
        # them requires pretending here that the fault never happened.
        return False

    @staticmethod
    def _row_matches_key(row: StoredCredential, api_key: str) -> bool:
        if row.credential_type == "api_key":
            return row.data.get("key") == api_key
        # Compare against the SAME extractor the cascade used to produce the
        # wire key, not ``data["access"]`` directly: a QwenCloud row holds a
        # management token in ``access`` and the ``sk-sp-…`` inference key in
        # ``api_key``, so the failing bearer failover reports is the extractor's
        # output and a raw-field compare would find no row — no block, no
        # demotion, no sticky clear for a credential that just failed.
        # A malformed row (neither ``access`` nor ``api_key``) must not raise
        # here: this runs inside failover's failure path, where an exception
        # would replace "rotate away from the failing credential" with a crash.
        # The extractor reads fields directly, so fall back to the raw compare
        # when it cannot produce a key.
        key_fn = AuthStore._oauth_key_fn(row.provider)
        if key_fn is not None:
            try:
                return key_fn(row.data) == api_key
            except KeyError:
                return False
        return row.data.get("access") == api_key

    def credential_id_for_key(self, provider: str, api_key: str) -> int | None:
        """Reverse lookup used by failover to block the exact bearer."""
        for row in self.list_credentials(provider):
            if self._row_matches_key(row, api_key):
                return row.id
        return None

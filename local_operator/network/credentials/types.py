"""The credential broker's vocabulary: keys, grants, refusals, placement rows.

STDLIB ONLY AT IMPORT (``network/credentials/__init__.py`` states the rule): the
relay imports the credentials package at construction, and nothing on that path
may pull the provider auth store. Everything here is a name or a plain data
holder, so the whole broker protocol can be reasoned about — and tested — without
an ``auth.db`` in sight.

THREE FACTS THIS MODULE EXISTS TO PIN, each a decision rather than a spelling:

* **A credential KEY is not a provider name.** A provider login keys on the
  provider (``openai``); an MCP grant keys on the server URL (``mcp:<url>``),
  because one provider row per MCP server is what ``mcp/auth.py`` already stores
  (``provider='mcp-oauth'``, ``identity_key=<server_url>``). Collapsing those into
  one namespace would let an MCP server's placement shadow a provider's, which is
  the kind of collision that serves the wrong account rather than refusing.
* **A brokered credential has a SYNTHETIC id, and it is NEGATIVE.** The session
  path calls about twenty ``AuthStore`` methods, many keyed by ``credential_id``
  (build plan §0 finding 5 — the design's three-protocol wrapper was too narrow
  for the real call sites). A borrowed credential has no row on the borrower, so
  it needs an id that is stable for a turn, usable as a dict/list key, and
  **impossible for a real row to have**: SQLite's ``INTEGER PRIMARY KEY`` assigns
  from 1 upward, so every negative integer is unreachable by construction. That is
  the whole safety argument for the synthetic id, and
  :func:`synthetic_credential_id` is its only home.
* **The refusal codes are CLOSED and carry the requester's behaviour.** The design
  keys the requester's reaction on the code and never on the sentence (§3.2), so
  the code set and its cache TTLs live together in one table below: a code added
  without a TTL would make the borrower guess, which is how a refusal storm starts.
"""

from __future__ import annotations

import dataclasses
import time
import zlib
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Op names and the two legs
# ---------------------------------------------------------------------------

#: Leg 1 (runtime → its own relay), control-socket ops. Named ``credential_*``
#: rather than ``net_*``: the ``net_*`` names are the peer-scope vocabulary, and
#: the rule (``relay.py``'s control-handler comment) is that a reader can tell
#: from the frame alone which boundary it crossed. These are declared in P0's
#: ``types.LOCAL_OPS`` and registered here.
LOCAL_CREDENTIAL_OPS: tuple[str, ...] = (
    "credential_grant",
    "credential_report",
    "credential_placement",
)

#: Leg 2 (relay → owner relay). ONE peer op with a ``kind`` discriminator, as the
#: transport document reserves it: the capability model authorises per op, so four
#: peer ops would mean four capability rows for one authority.
PEER_BROKER_OP = "net_broker"

#: The capability the transport's chokepoint requires for :data:`PEER_BROKER_OP`
#: (``types.OP_CAPABILITY``). Restated here so the broker's own refusal sentence
#: can name what the peer is missing without importing the relay.
BROKER_CAPABILITY = "broker_credential"

#: Advertised in hello/welcome ``caps``. A peer that does not advertise it answers
#: ``unknown op``, which the client maps to ``unsupported`` rather than retrying.
BROKER_CAP_STRING = "credential-broker-v1"

#: The ``kind`` discriminator on a :data:`PEER_BROKER_OP` frame.
BrokerKind = Literal["grant", "report", "placement"]
BROKER_KINDS: frozenset[str] = frozenset({"grant", "report", "placement"})

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

MCP_KEY_PREFIX = "mcp:"


def credential_key_for_provider(provider: str) -> str:
    """The placement key for a provider login. Identity, so it is spelled once."""
    return provider


def credential_key_for_mcp(server_url: str) -> str:
    return f"{MCP_KEY_PREFIX}{server_url}"


def is_mcp_key(key: str) -> bool:
    return key.startswith(MCP_KEY_PREFIX)


def mcp_url_from_key(key: str) -> str:
    """The server URL inside an ``mcp:<url>`` key. Raises on a non-MCP key.

    Raising rather than returning ``""``: a caller that asked for the URL of a
    provider key has a bug, and an empty string would send it on to fetch a
    credential for the empty URL.
    """
    if not is_mcp_key(key):
        raise ValueError(f"{key!r} is not an mcp:<url> placement key")
    return key[len(MCP_KEY_PREFIX) :]


# ---------------------------------------------------------------------------
# Device-bound providers
# ---------------------------------------------------------------------------

#: Providers whose grants are bound to the DEVICE that made them, and can
#: therefore never be brokered (build plan §0 finding 7; cut line unsafe item 1).
#:
#: Kimi signs every inference call with the local device fingerprint
#: (``providers/clients.py`` → ``kimi_common_headers()`` →
#: ``oauth/kimi.py get_or_create_device_id``). A borrower replaying the owner's
#: access token presents it with ITS OWN fingerprint, which is a different
#: device's token being used by this one — the case the design itself calls
#: never. Refused by NAME rather than by a heuristic, because a heuristic that
#: missed would broker it silently.
DEVICE_BOUND_PROVIDERS: frozenset[str] = frozenset({"kimi"})


def device_bound_refusal(provider: str) -> str:
    """The sentence for a device-bound refusal, naming the local remedy."""
    return (
        f"{provider} logins are bound to the device that made them, so this one "
        f"cannot be lent to another device; run 'lop login {provider}' on the "
        "device that needs it"
    )


# ---------------------------------------------------------------------------
# Refusal codes: the closed set, and what the requester does with each
# ---------------------------------------------------------------------------

#: Codes that mean "an older peer, or a peer that does not serve this at all".
#: Both are session-scoped: the answer cannot change while the build does not.
NO_RETRY = -1

#: ``code`` → the TTL of the borrower's cached refusal, in milliseconds.
#: ``0`` means "honour the ``retry_after_ms`` the owner sent, defaulting to this
#: table's own fallback"; :data:`NO_RETRY` means "cache for the session".
#:
#: The numbers are the design's §3.2 table, and the two that matter are the
#: extremes: ``owner_offline`` is SHORT (15 s — the owner may come back at any
#: moment, and the borrower's own grant keeps working meanwhile), and
#: ``not_a_holder``/``revoked``/``grant_invalid`` are LONG (300 s — nothing the
#: borrower can do changes them, and a short TTL here is a retry storm aimed at a
#: device that has already said no).
BROKER_ERROR_TTL_MS: dict[str, int] = {
    "no_local_credential": 300_000,
    "owner_offline": 15_000,
    "not_a_holder": 60_000,
    # A device asked a device that does not hold the credential. Its own code rather
    # than `not_a_holder`, because the remedy is different: the requester was told to
    # ask the WRONG device, so re-asking the right one (which the placement names)
    # can succeed immediately.
    "not_owner": 0,
    "revoked": 300_000,
    "epoch_stale": 0,
    "grant_invalid": 300_000,
    "refresh_failed": 0,
    "quota_blocked": 0,
    "interactive_required": 300_000,
    "rate_limited": 0,
    "unsupported": NO_RETRY,
    "device_bound": 300_000,
    "not_authorised": 60_000,
    "not_implemented": NO_RETRY,
    "internal": 60_000,
}

BROKER_ERROR_CODES: frozenset[str] = frozenset(BROKER_ERROR_TTL_MS)

#: The default when the owner sent no ``retry_after_ms`` for a code that honours
#: one. Deliberately conservative — a retry after a minute is cheap next to a
#: loop that asks a rate-limited owner on every provider call.
DEFAULT_RETRY_AFTER_MS = 60_000


@dataclasses.dataclass
class BrokerError:
    """A refused grant: the machine ``code`` and the sentence for the person.

    The transport's envelope has no ``code`` field, so this rides inside an
    ``ack``'s ``detail``: a refused grant is a routed request that was ANSWERED,
    not a transport failure, and keeping that distinction matters because the
    transport's own codes (``not_authorised``, ``not_a_member``, ``unknown_op``)
    live in the same namespace and mean something else.
    """

    code: str
    message: str = ""
    key: str = ""
    owner_device: str = ""
    owner_device_name: str = ""
    retry_after_ms: int = 0

    @property
    def cache_ttl_ms(self) -> int:
        """How long the borrower caches this refusal. See :data:`BROKER_ERROR_TTL_MS`."""
        ttl = BROKER_ERROR_TTL_MS.get(self.code, DEFAULT_RETRY_AFTER_MS)
        if ttl == NO_RETRY:
            return NO_RETRY
        if ttl == 0:
            return self.retry_after_ms or DEFAULT_RETRY_AFTER_MS
        return ttl

    def to_detail(self) -> dict[str, Any]:
        """The wire shape (design §3.2's failure reply, inside ``detail``)."""
        detail: dict[str, Any] = {"kind": "error", "code": self.code, "message": self.message}
        if self.key:
            detail["key"] = self.key
        if self.owner_device:
            detail["owner_device"] = self.owner_device
        if self.owner_device_name:
            detail["owner_device_name"] = self.owner_device_name
        if self.retry_after_ms:
            detail["retry_after_ms"] = self.retry_after_ms
        return detail

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> BrokerError:
        return cls(
            code=str(detail.get("code") or "internal"),
            message=str(detail.get("message") or ""),
            key=str(detail.get("key") or ""),
            owner_device=str(detail.get("owner_device") or ""),
            owner_device_name=str(detail.get("owner_device_name") or ""),
            retry_after_ms=int(detail.get("retry_after_ms") or 0),
        )


@dataclasses.dataclass(frozen=True)
class CredentialRef:
    """Where a granted bearer came from, in the OWNER's own id space."""

    owner_device: str
    owner_device_name: str
    provider: str
    kind: str  # 'oauth' | 'api_key' | 'mcp-oauth'
    credential_id: int


@dataclasses.dataclass(frozen=True)
class GrantScope:
    """What the delegation is bounded by. ``session`` is the smaller authority."""

    kind: str  # 'session' | 'device'
    session_id: str = ""


@dataclasses.dataclass
class Grant:
    """A borrowed bearer. It lives in memory on the borrower and on no disk.

    ``grant_expires_at_ms`` is ``min(token_expires_at_ms, now + grant_ttl_s)``
    (design §3.3's narrowing rule, enforced in ``owner.py`` rather than here —
    this type only carries the decision). ``token_expires_at_ms`` is ``0`` for a
    token that does not expire (a static API key), which is the same encoding the
    auth store uses for its own ``expires`` field.
    """

    access_token: str
    kind: str  # 'bearer' | 'api_key'
    token_expires_at_ms: int
    grant_expires_at_ms: int
    credential_ref: CredentialRef
    served_by: str
    granted_at: float = dataclasses.field(default_factory=time.time)
    refreshed: bool = False
    scope: GrantScope = dataclasses.field(default_factory=lambda: GrantScope(kind="session"))
    identity: dict[str, str] = dataclasses.field(default_factory=dict)
    latency_ms: int = 0
    grant_id: str = ""

    def usable_at(self, now_ms: float | None = None) -> bool:
        """Whether this grant may still be handed to a request right now."""
        now = now_ms if now_ms is not None else time.time() * 1000.0
        return now < self.grant_expires_at_ms

    def to_detail(self) -> dict[str, Any]:
        """The wire shape: design §3.2's success reply, inside ``detail``.

        ``access_token`` is the ONE field on this wire that is material, and it is
        the one the audit writer refuses by name (``audit.FORBIDDEN_DETAIL_KEYS``
        holds ``access_token`` and ``token``) — so the token can cross the link and
        can never be recorded in a log, which is the correct asymmetry.
        """
        detail: dict[str, Any] = {
            "kind": "grant",
            "grant_id": self.grant_id,
            "access_token": self.access_token,
            "token_kind": self.kind,
            "token_expires_at_ms": self.token_expires_at_ms,
            "grant_expires_at_ms": self.grant_expires_at_ms,
            "credential_ref": dataclasses.asdict(self.credential_ref),
            "scope": dataclasses.asdict(self.scope),
            "served_by": self.served_by,
            "refreshed": self.refreshed,
            "latency_ms": self.latency_ms,
        }
        if self.identity:
            detail["identity"] = dict(self.identity)
        return detail

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> Grant:
        """Rebuild a grant from the wire shape.

        A missing ``access_token`` yields an EMPTY bearer rather than raising: the
        caller's own emptiness check is what refuses, and an exception here would
        travel out of a provider call as a transport error for what is really a
        protocol mismatch. ``credential_ref`` degrades to a blank ref for the same
        reason — it is not on the path this module's tests would notice, so a
        future field rename must not be able to make a borrow crash.
        """
        ref = detail.get("credential_ref")
        scope = detail.get("scope")
        ref_row = ref if isinstance(ref, dict) else {}
        scope_row = scope if isinstance(scope, dict) else {}
        identity = detail.get("identity")
        return cls(
            access_token=str(detail.get("access_token") or ""),
            kind=str(detail.get("token_kind") or "bearer"),
            token_expires_at_ms=int(detail.get("token_expires_at_ms") or 0),
            grant_expires_at_ms=int(detail.get("grant_expires_at_ms") or 0),
            credential_ref=CredentialRef(
                owner_device=str(ref_row.get("owner_device") or ""),
                owner_device_name=str(ref_row.get("owner_device_name") or ""),
                provider=str(ref_row.get("provider") or ""),
                kind=str(ref_row.get("kind") or "oauth"),
                credential_id=int(ref_row.get("credential_id") or 0),
            ),
            served_by=str(detail.get("served_by") or ""),
            refreshed=bool(detail.get("refreshed")),
            scope=GrantScope(
                kind=str(scope_row.get("kind") or "session"),
                session_id=str(scope_row.get("session_id") or ""),
            ),
            identity={str(k): str(v) for k, v in (identity or {}).items()},
            latency_ms=int(detail.get("latency_ms") or 0),
            grant_id=str(detail.get("grant_id") or ""),
        )


# ---------------------------------------------------------------------------
# The placement document
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Holder:
    """One device allowed to borrow one credential. Authorisation, not metadata."""

    device: str
    scope: str = "session"  # 'session' | 'device'
    granted_at: float = 0.0
    granted_by: str = ""


@dataclasses.dataclass
class CredentialPlacementEntry:
    """One credential key's placement: who owns it, who may borrow it.

    ``doc_rev`` is PER ENTRY, not per document, and that is the merge rule the
    design chose: two devices writing different keys must not have one's whole
    document win, and the resolver needs a total order within a key to break a
    tie. ``owner_device`` is the only writer of this row.
    """

    key: str
    provider: str
    kind: str  # 'oauth-rotating' | 'api-key-static' | 'mcp-rotating'
    owner_device: str
    owner_device_name: str = ""
    identity_label: str = ""
    holders: list[Holder] = dataclasses.field(default_factory=list)
    declared_at: float = 0.0
    doc_rev: int = 1

    def holder(self, device: str) -> Holder | None:
        for row in self.holders:
            if row.device == device:
                return row
        return None

    def is_holder(self, device: str) -> bool:
        return self.holder(device) is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "kind": self.kind,
            "owner_device": self.owner_device,
            "owner_device_name": self.owner_device_name,
            "identity_label": self.identity_label,
            "holders": [dataclasses.asdict(row) for row in self.holders],
            "declared_at": self.declared_at,
            "doc_rev": self.doc_rev,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> CredentialPlacementEntry:
        holders: list[Holder] = []
        for item in row.get("holders") or []:
            if not isinstance(item, dict) or not item.get("device"):
                continue
            holders.append(
                Holder(
                    device=str(item["device"]),
                    scope=str(item.get("scope") or "session"),
                    granted_at=float(item.get("granted_at") or 0.0),
                    granted_by=str(item.get("granted_by") or ""),
                )
            )
        return cls(
            key=str(row.get("key") or ""),
            provider=str(row.get("provider") or ""),
            kind=str(row.get("kind") or "oauth-rotating"),
            owner_device=str(row.get("owner_device") or ""),
            owner_device_name=str(row.get("owner_device_name") or ""),
            identity_label=str(row.get("identity_label") or ""),
            holders=holders,
            declared_at=float(row.get("declared_at") or 0.0),
            doc_rev=int(row.get("doc_rev") or 1),
        )


# ---------------------------------------------------------------------------
# The synthetic credential id
# ---------------------------------------------------------------------------

#: The magnitude the synthetic ids are drawn from. Negative integers only, so no
#: real row can ever collide (see this module's docstring). Kept well inside
#: SQLite's 64-bit range so arithmetic on it in either direction cannot overflow.
SYNTHETIC_ID_CEILING = 1 << 31


def synthetic_credential_id(key: str, owner_device: str) -> int:
    """A stable, NEGATIVE credential id for a brokered credential.

    Deterministic on ``(key, owner_device)`` so the same borrow yields the same
    id for the life of the placement: a session that recorded a verdict against
    one of these under a block, or a caller holding it across two provider calls
    in one turn, must not see the id change underneath it.
    """
    digest = zlib.crc32(f"{key}\x00{owner_device}".encode("utf-8"))
    return -(digest % SYNTHETIC_ID_CEILING) - 1


def is_synthetic_credential_id(credential_id: int) -> bool:
    """Whether an id belongs to the broker rather than to a row."""
    return credential_id < 0

"""A member's Radient Guard consent on this install (Guard RFC v0.9 §5, slice L2a).

WHAT. The host half of the one-time, host-owned consent prompt:

* :class:`ConsentOffer` -- the server's ``consent_required`` answer, parsed
  strictly (§5.2.1);
* :class:`GuardConsentStore` -- the durable queue and the sticky state, one
  SQLite file at ``<config_dir>/guard/consent.sqlite3`` (§5.2.2, §5.2.6);
* :class:`GuardConsentHost` -- offer / decline / accept / revoke, plus the
  mirror of each pending prompt onto the desktop approvals plane as a
  ``guard_consent`` record (``guard/consent_op.py``);
* :func:`check_point` -- the admission-point status read, which runs ONLY for
  an account the client already knows is on Radient Team (or a Team trial).

THE MODEL CANNOT ANSWER IT (§5.2.4). An accept is authority-increasing, so
:meth:`GuardConsentHost.accept` asks :func:`harness.approval.admit_increasing`
and nothing else: the spawn-capability proof of the attached host surface, or
a verified operator/device signature (the desktop's ``/approve-operation``
path, which signs the digest-bound typed card before :meth:`accept_signed`
runs). There is no consent tool; a ``bridge_decide``, a peer ``send``, an MCP
result, a slash relay, ``--yolo`` and ``/approvals auto`` carry neither fact,
so each is refused here, recorded as a typed host refusal, and the prompt stays
pending with NOTHING sent. A deny and "Decide later" are ordinary.

ALL OR NOTHING (R2-m2). An accept grants exactly what the queued row offered:
the offered scopes on a first grant, the not-yet-declined added scopes on a
widening. No API here takes a scope subset.

ZERO-COST FOR NON-TEAM (§5.1). :func:`check_point` returns before any
network call, any file and any credential-store access unless
``hub_org.eligibility.eligible_orgs`` -- the local, zero-network record of the
account's Team memberships -- is non-empty. The store's constructor does no
I/O, and the install key is created only inside an admitted accept.

WIRE CONTRACT. The server half (agent-server slice A2) is not built yet. This
module codes against RFC §5.2/§6 and fixes the parts the RFC leaves open, so
A2 can pin them: the status read is ``GET /v1/guard/status?tenant=<id>``
answering ``{status, prompt?}``; the grant is ``POST /v1/guard/consents`` with
``{payload, signature, alg, key_id}`` where ``signature`` is Ed25519 over
:data:`SIGNING_DOMAIN` + canonical JSON of ``payload``; the revoke is
``POST /v1/guard/consents/revoke`` with ``{tenant}`` and the JWT only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

logger = logging.getLogger(__name__)

#: Generation 2 adds ``prompts.claim`` (the per-claim token, L2a security S-6).
#: A generation-1 file is migrated in place on open (one ``ALTER TABLE``); any
#: other generation is refused as damaged, as before.
GUARD_CONSENT_SCHEMA_VERSION: Final = 2

#: The scopes the RFC defines (§15.4' C4). An offer naming another is refused:
#: a scope this build cannot describe to the member cannot be consented to.
SCOPES: Final = frozenset(
    {
        "decisions",
        "posture",
        "task_time",
        "review_cycles",
        "reports",
        "trace_detail",
        "admin_share",
    }
)
#: Scopes this build REFUSES to queue even though the RFC defines them (copy
#: review round 1, C1). Their revoke confirm must state how long backup copies
#: of the deleted step-by-step timing survive (RFC §5.4: "cannot render before
#: G2 records X"), and X is not measured yet. An offer naming either is refused
#: whole (``guard_scope_unavailable``) rather than trimmed: the grant is signed
#: over exactly the offered scopes and terms digest, so a client-side subset
#: would be a grant the server never offered. Lift an entry only together with
#: :data:`guard.surface.BACKUP_DAYS` (the measured X) and the revoke copy.
UNMEASURED_SCOPES: Final = frozenset({"trace_detail", "admin_share"})
#: ``reason`` values (§5.2.1). ``widened`` is the only one that offers
#: ``added_scopes`` alone.
REASONS: Final = frozenset(
    {"new_org", "widened", "reactivated", "plan_gap", "new_device", "rejoined"}
)

#: Domain separation for the install-key signature over a grant payload.
SIGNING_DOMAIN: Final = b"radient-guard-consent-v1\x00"

#: How long one desktop mirror card stays open. The SQLite queue row has NO
#: expiry (§5.2.2); the mirror is re-filed under a fresh request id at the next
#: admission point after it lapses, so this bounds only how stale a desktop
#: card can look, not how long the member has to answer. Seven days covers a
#: member who opens the desktop once a week; it is not a measured bound and
#: nothing correctness-bearing depends on it.
MIRROR_WINDOW_S: Final = 7 * 24 * 3600.0

_BUSY_TIMEOUT_S: Final = 5.0

#: A ``committing`` claim older than this is a crashed commit, released back to
#: ``pending`` at the next admission point so the member can answer again. The
#: live path it must outlast is two key-store calls (``install_key._TIMEOUT_S``
#: = 10 s each) plus the grant request (``RadientClient._guard_call``
#: ``timeout=30`` per connect/read): ~50 s. Ten minutes is far above that; a
#: re-answer after a real crash is harmless because the server's partial-unique
#: consent index makes a second grant for the same rev a no-op.
CLAIM_LEASE_S: Final = 600.0

_TERMS_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}")

PROMPT_PENDING: Final = "pending"
#: An admitted accept CLAIMED the row and is signing/sending its grant. Only
#: one answer can hold the claim (``BEGIN IMMEDIATE`` + ``state='pending'``),
#: so a TUI accept racing a desktop apply cannot sign two grants (§5.2.3), and
#: a revoke that lands mid-flight finds the claim and wins (§5.4/§5.7d).
PROMPT_COMMITTING: Final = "committing"
PROMPT_ACCEPTED: Final = "accepted"
PROMPT_DECLINED: Final = "declined"
PROMPT_SUPERSEDED: Final = "superseded"

STICKY_DECLINED: Final = "declined"
STICKY_REVOKED: Final = "revoked"

#: Server codes that mean "a revoke landed first" (§5.2.5 revocation wins).
_STALE_REVOCATION_CODES: Final = frozenset({"guard_revocation_stale", "guard_consent_revoked"})


class GuardConsentRefused(Exception):
    """A consent operation was refused; nothing was granted or sent."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json(payload: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8 (``network.wire.canonical_json``'s rule)."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _digest(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# The offer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentOffer:
    """One ``consent_required`` answer (§5.2.1), validated before it is queued."""

    tenant: str
    org_display_name: str
    collection_rev: int
    revocation_gen: int
    scopes: tuple[str, ...]
    added_scopes: tuple[str, ...]
    rendered_terms_digest: str
    reason: str

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> ConsentOffer:
        def bad(what: str) -> GuardConsentRefused:
            return GuardConsentRefused("guard_offer_invalid", f"the Guard offer has {what}")

        if not isinstance(data, Mapping):
            raise bad("no body")
        tenant = data.get("tenant")
        name = data.get("org_display_name")
        rev = data.get("collection_rev")
        gen = data.get("revocation_gen")
        terms = data.get("rendered_terms_digest")
        reason = data.get("reason")
        if not isinstance(tenant, str) or not 0 < len(tenant) <= 128:
            raise bad("no tenant")
        if not isinstance(name, str) or not 0 < len(name) <= 256:
            raise bad("no organization name")
        # ``bool`` is an ``int`` subclass; a JSON ``true`` must not read as rev 1.
        for value in (rev, gen):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise bad("a malformed revision")
        if not isinstance(terms, str) or _TERMS_DIGEST.fullmatch(terms) is None:
            raise bad("a malformed terms digest")
        if reason not in REASONS:
            raise bad("an unknown reason")
        scopes = cls._scopes(data.get("scopes"), bad)
        added = cls._scopes(data.get("added_scopes") or [], bad)
        if not scopes:
            raise bad("no scopes")
        if set(scopes) & UNMEASURED_SCOPES:
            raise GuardConsentRefused(
                "guard_scope_unavailable",
                "the Guard offer asks for a scope this version cannot offer yet",
            )
        if not set(added) <= set(scopes):
            raise bad("added scopes outside the offered scopes")
        if reason == "widened" and not added:
            raise bad("a widening with nothing added")
        # ``rev``/``gen`` were checked to be non-negative ints above.
        return cls(tenant, name, cast(int, rev), cast(int, gen), scopes, added, terms, reason)

    @staticmethod
    def _scopes(value: Any, bad: Callable[[str], GuardConsentRefused]) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
            raise bad("a malformed scope list")
        if any(s not in SCOPES for s in value) or len(set(value)) != len(value):
            raise bad("an unknown or repeated scope")
        return tuple(sorted(value))


@dataclass(frozen=True)
class PromptRow:
    """One queued prompt, exactly as the member is asked about it."""

    tenant: str
    collection_rev: int
    revocation_gen: int
    org_display_name: str
    scopes: tuple[str, ...]
    #: What THIS prompt asks for: the offered scopes on a first grant, the
    #: added scopes minus any the member already declined on a widening.
    asked_scopes: tuple[str, ...]
    #: What an accept grants in total (sent as ``scopes`` in the signed body).
    grant_scopes: tuple[str, ...]
    terms_digest: str
    reason: str
    created_at: float
    state: str
    approval_id: str
    mirror_seq: int
    #: The token of the accept that holds this row's ``committing`` claim ("" =
    #: unclaimed). Not part of :meth:`immutable`: it says who is answering, not
    #: what the member was asked, so it never moves the digest.
    claim: str = ""

    @property
    def widening(self) -> bool:
        return self.reason == "widened"

    def immutable(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant,
            "collection_rev": self.collection_rev,
            "revocation_gen": self.revocation_gen,
            "org_display_name": self.org_display_name,
            "scopes": list(self.scopes),
            "asked_scopes": list(self.asked_scopes),
            "grant_scopes": list(self.grant_scopes),
            "terms_digest": self.terms_digest,
            "reason": self.reason,
        }

    @property
    def digest(self) -> str:
        """The digest of the queued row: what a typed desktop card binds (§5.2.2)."""

        return _digest(self.immutable())


# ---------------------------------------------------------------------------
# The durable store
# ---------------------------------------------------------------------------


def guard_consent_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "consent.sqlite3"


_DDL: Final = (
    "CREATE TABLE prompts ("
    " tenant TEXT NOT NULL, collection_rev INTEGER NOT NULL,"
    " revocation_gen INTEGER NOT NULL, org_display_name TEXT NOT NULL,"
    " scopes TEXT NOT NULL, asked_scopes TEXT NOT NULL, grant_scopes TEXT NOT NULL,"
    " terms_digest TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,"
    " state TEXT NOT NULL, approval_id TEXT NOT NULL DEFAULT '',"
    " mirror_seq INTEGER NOT NULL DEFAULT 0, decided_at REAL, decided_via TEXT,"
    " claim TEXT, PRIMARY KEY (tenant, collection_rev))",
    "CREATE TABLE sticky (tenant TEXT PRIMARY KEY, state TEXT NOT NULL,"
    " origin TEXT NOT NULL, at REAL NOT NULL)",
    "CREATE TABLE declined_scopes (tenant TEXT NOT NULL, scope TEXT NOT NULL,"
    " at REAL NOT NULL, PRIMARY KEY (tenant, scope))",
    "CREATE TABLE consents (tenant TEXT PRIMARY KEY, collection_rev INTEGER NOT NULL,"
    " scopes TEXT NOT NULL, granted_at REAL NOT NULL, state TEXT NOT NULL)",
    "CREATE TABLE install (id INTEGER PRIMARY KEY CHECK (id = 1), install_id TEXT NOT NULL,"
    " enrolled INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE notices (id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL,"
    " kind TEXT NOT NULL, org_display_name TEXT NOT NULL, scopes TEXT NOT NULL,"
    " at REAL NOT NULL, shown INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE refusals (id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL,"
    " collection_rev INTEGER NOT NULL, source TEXT NOT NULL, code TEXT NOT NULL,"
    " at REAL NOT NULL)",
)


class GuardConsentStore:
    """The queue and the sticky state. Every method opens one short connection.

    Same house rules as ``guard/outbox.py``: construction does no I/O, the
    directory is re-asserted 0700 and the file 0600 on every open, a file
    with an unknown schema generation -- or one that is not SQLite at all -- is
    refused as damaged rather than re-created.
    """

    def __init__(self, config_dir: Path | str) -> None:
        self._path = guard_consent_path(config_dir)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        existed = self._path.exists() and self._path.stat().st_size > 0
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            os.chmod(self._path, 0o600)
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise GuardConsentRefused(
                    "consent_store_damaged", "the Guard consent store is not a database"
                ) from exc
            if existed and version == 1:
                # Generation 1 (L2a) -> 2: the per-claim token column. Additive
                # and NULL for every existing row, which reads as "unclaimed".
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("ALTER TABLE prompts ADD COLUMN claim TEXT")
                conn.execute(f"PRAGMA user_version = {GUARD_CONSENT_SCHEMA_VERSION}")
                conn.execute("COMMIT")
                version = GUARD_CONSENT_SCHEMA_VERSION
            if existed and version != GUARD_CONSENT_SCHEMA_VERSION:
                raise GuardConsentRefused(
                    "consent_store_damaged",
                    "the Guard consent store has an unknown schema generation; refusing it",
                )
            if not version:
                conn.execute("BEGIN IMMEDIATE")
                for statement in _DDL:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {GUARD_CONSENT_SCHEMA_VERSION}")
                conn.execute("COMMIT")
        except BaseException:
            conn.close()
            raise
        return conn


def _row(row: sqlite3.Row) -> PromptRow:
    return PromptRow(
        tenant=row["tenant"],
        collection_rev=int(row["collection_rev"]),
        revocation_gen=int(row["revocation_gen"]),
        org_display_name=row["org_display_name"],
        scopes=tuple(json.loads(row["scopes"])),
        asked_scopes=tuple(json.loads(row["asked_scopes"])),
        grant_scopes=tuple(json.loads(row["grant_scopes"])),
        terms_digest=row["terms_digest"],
        reason=row["reason"],
        created_at=float(row["created_at"]),
        state=row["state"],
        approval_id=row["approval_id"],
        mirror_seq=int(row["mirror_seq"]),
        claim=str(row["claim"] or ""),
    )


# ---------------------------------------------------------------------------
# The transport (the Radient client methods, or a test double)
# ---------------------------------------------------------------------------


class GuardTransport(Protocol):
    """The three Guard calls. ``RadientClient`` implements them (``clients/radient.py``)."""

    def guard_status(self, tenant_id: str) -> dict[str, Any]: ...

    def guard_grant(self, envelope: Mapping[str, Any]) -> dict[str, Any]: ...

    def guard_revoke(self, tenant_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OfferOutcome:
    """``queued`` | ``duplicate`` | ``suppressed_sticky`` | ``suppressed_declined_scopes``
    | ``superseded_by_newer`` | ``already_consented``."""

    status: str
    row: PromptRow | None = None


@dataclass(frozen=True)
class AcceptOutcome:
    """``granted`` | ``refused`` (not admitted, or another answer holds the row;
    nothing sent) | ``failed`` (sent, refused by the server or unreachable; the row
    stays pending) | ``revoked_first`` (a Console or local revoke won the race) |
    ``revoked`` (:meth:`GuardConsentHost.revoke` reached the server)."""

    status: str
    code: str = ""


# ---------------------------------------------------------------------------
# The host engine
# ---------------------------------------------------------------------------


class GuardConsentHost:
    """Offer, decline, accept and revoke for one config root."""

    def __init__(
        self,
        config_dir: Path | str,
        *,
        transport: GuardTransport | Callable[[], GuardTransport | None] | None = None,
        key_store: Any = None,
        clock: Callable[[], float] = time.time,
        mirror: bool = True,
    ) -> None:
        # No I/O here: a host built for a non-Team account leaves no file.
        self._config_dir = Path(config_dir)
        self._store = GuardConsentStore(config_dir)
        self._transport = transport
        self._key_store = key_store
        self._clock = clock
        self._mirror_enabled = mirror

    @property
    def store(self) -> GuardConsentStore:
        return self._store

    # -- reads -------------------------------------------------------------

    def pending(self) -> list[PromptRow]:
        if not self._store.exists():
            return []
        conn = self._store.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE state = ? ORDER BY created_at", (PROMPT_PENDING,)
            ).fetchall()
            return [_row(r) for r in rows]
        finally:
            conn.close()

    def prompt(self, tenant: str, collection_rev: int) -> PromptRow | None:
        if not self._store.exists():
            return None
        conn = self._store.connect()
        try:
            found = conn.execute(
                "SELECT * FROM prompts WHERE tenant = ? AND collection_rev = ?",
                (tenant, collection_rev),
            ).fetchone()
            return _row(found) if found is not None else None
        finally:
            conn.close()

    def sticky(self, tenant: str) -> str | None:
        return self._scalar("SELECT state FROM sticky WHERE tenant = ?", (tenant,))

    def declined_scopes(self, tenant: str) -> set[str]:
        return set(self._column("SELECT scope FROM declined_scopes WHERE tenant = ?", (tenant,)))

    def consent(self, tenant: str) -> dict[str, Any] | None:
        if not self._store.exists():
            return None
        conn = self._store.connect()
        try:
            found = conn.execute("SELECT * FROM consents WHERE tenant = ?", (tenant,)).fetchone()
            if found is None:
                return None
            return {
                "tenant": found["tenant"],
                "collection_rev": int(found["collection_rev"]),
                "scopes": json.loads(found["scopes"]),
                "state": found["state"],
            }
        finally:
            conn.close()

    def notices(self, *, unshown_only: bool = True) -> list[dict[str, Any]]:
        """The receipt notices (§5.7c / §15.4' C3) for a surface to render."""

        if not self._store.exists():
            return []
        conn = self._store.connect()
        try:
            sql = "SELECT * FROM notices" + (" WHERE shown = 0" if unshown_only else "")
            return [
                {
                    "id": r["id"],
                    "tenant": r["tenant"],
                    "kind": r["kind"],
                    "org_display_name": r["org_display_name"],
                    "scopes": json.loads(r["scopes"]),
                    "at": r["at"],
                }
                for r in conn.execute(sql + " ORDER BY id").fetchall()
            ]
        finally:
            conn.close()

    def mark_notice_shown(self, notice_id: int) -> None:
        self._exec("UPDATE notices SET shown = 1 WHERE id = ?", (notice_id,))

    def refusals(self) -> list[dict[str, Any]]:
        if not self._store.exists():
            return []
        conn = self._store.connect()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM refusals ORDER BY id")]
        finally:
            conn.close()

    def _scalar(self, sql: str, args: tuple[Any, ...]) -> Any:
        if not self._store.exists():
            return None
        conn = self._store.connect()
        try:
            found = conn.execute(sql, args).fetchone()
            return found[0] if found is not None else None
        finally:
            conn.close()

    def _column(self, sql: str, args: tuple[Any, ...]) -> list[Any]:
        if not self._store.exists():
            return []
        conn = self._store.connect()
        try:
            return [r[0] for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def _exec(self, sql: str, args: tuple[Any, ...]) -> None:
        conn = self._store.connect()
        try:
            conn.execute(sql, args)
        finally:
            conn.close()

    # -- offer -------------------------------------------------------------

    def offer(self, offer: ConsentOffer) -> OfferOutcome:
        """Queue ``offer`` durably unless the member's sticky answer suppresses it.

        One transaction (§5.2.2): the sticky checks, the per-scope widening
        suppression (U10), superseding older revisions and the insert see one
        state. The desktop mirror is filed after commit; a mirror failure never
        loses the queue row.

        SAME-REV RE-OFFERS (review round 1, MAJOR-1). Only an IDENTICAL pending
        row (or one an accept is committing) is a duplicate. A row at the same
        ``collection_rev`` that was declined, accepted then revoked, or
        superseded is REPLACED once the sticky state has been lifted -- that is
        "Share again" (§5.2.6, §5.4/C10, §5.9 "re-grant at the same rev
        succeeds") -- and so is a pending row whose server-side immutables
        changed (a new ``revocation_gen`` or terms digest), which the member
        would otherwise sign stale and see fail forever. The replacement takes
        the next ``mirror_seq`` so its desktop card is a new request, and the
        old card is withdrawn.
        """

        now = self._clock()
        replaced_card = ""
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            sticky = conn.execute(
                "SELECT state FROM sticky WHERE tenant = ?", (offer.tenant,)
            ).fetchone()
            if sticky is not None:
                # §5.2.6: a decline or revoke is sticky until the member opens it
                # again (``share_again``). Nothing is queued, badged or noticed.
                conn.execute("ROLLBACK")
                return OfferOutcome("suppressed_sticky")
            newer = conn.execute(
                "SELECT 1 FROM prompts WHERE tenant = ? AND collection_rev > ?",
                (offer.tenant, offer.collection_rev),
            ).fetchone()
            if newer is not None:
                conn.execute("ROLLBACK")
                return OfferOutcome("superseded_by_newer")
            found = conn.execute(
                "SELECT * FROM prompts WHERE tenant = ? AND collection_rev = ?",
                (offer.tenant, offer.collection_rev),
            ).fetchone()
            existing = None if found is None else _row(found)
            if existing is not None and existing.state == PROMPT_COMMITTING:
                # An admitted accept is mid-flight for this rev; never yank it.
                conn.execute("ROLLBACK")
                return OfferOutcome("duplicate", existing)
            if offer.reason == "widened":
                declined = {
                    r[0]
                    for r in conn.execute(
                        "SELECT scope FROM declined_scopes WHERE tenant = ?", (offer.tenant,)
                    )
                }
                asked = tuple(s for s in offer.added_scopes if s not in declined)
                if not asked:
                    # U10: every added scope was already declined with "Keep
                    # current". No row, no badge, no notice; sharing continues.
                    conn.execute("ROLLBACK")
                    return OfferOutcome("suppressed_declined_scopes")
                grant = tuple(s for s in offer.scopes if s not in declined)
            else:
                asked = offer.scopes
                grant = offer.scopes
            live = conn.execute(
                "SELECT collection_rev, scopes FROM consents WHERE tenant = ? AND state = 'live'",
                (offer.tenant,),
            ).fetchone()
            if (
                live is not None
                and int(live["collection_rev"]) >= offer.collection_rev
                and set(grant) <= set(json.loads(live["scopes"]))
            ):
                conn.execute("ROLLBACK")
                return OfferOutcome("already_consented")
            mirror_seq = 0
            if existing is not None:
                candidate = {
                    **existing.immutable(),
                    "revocation_gen": offer.revocation_gen,
                    "org_display_name": offer.org_display_name,
                    "scopes": list(offer.scopes),
                    "asked_scopes": list(asked),
                    "grant_scopes": list(grant),
                    "terms_digest": offer.rendered_terms_digest,
                    "reason": offer.reason,
                }
                if existing.state == PROMPT_PENDING and candidate == existing.immutable():
                    conn.execute("ROLLBACK")
                    return OfferOutcome("duplicate", existing)
                if existing.state == PROMPT_PENDING:
                    replaced_card = existing.approval_id
                mirror_seq = existing.mirror_seq + 1
                conn.execute(
                    "DELETE FROM prompts WHERE tenant = ? AND collection_rev = ?",
                    (offer.tenant, offer.collection_rev),
                )
            conn.execute(
                "UPDATE prompts SET state = ? WHERE tenant = ? AND state = ?",
                (PROMPT_SUPERSEDED, offer.tenant, PROMPT_PENDING),
            )
            conn.execute(
                "INSERT INTO prompts (tenant, collection_rev, revocation_gen, org_display_name,"
                " scopes, asked_scopes, grant_scopes, terms_digest, reason, created_at, state,"
                " mirror_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    offer.tenant,
                    offer.collection_rev,
                    offer.revocation_gen,
                    offer.org_display_name,
                    json.dumps(list(offer.scopes)),
                    json.dumps(list(asked)),
                    json.dumps(list(grant)),
                    offer.rendered_terms_digest,
                    offer.reason,
                    now,
                    PROMPT_PENDING,
                    mirror_seq,
                ),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        if replaced_card:
            self._withdraw_card(replaced_card)
        self._withdraw_superseded_mirrors(offer.tenant, offer.collection_rev)
        row = self.prompt(offer.tenant, offer.collection_rev)
        assert row is not None
        row = self._mirror(row) or row
        return OfferOutcome("queued", row)

    # -- decline / later ---------------------------------------------------

    def decline(self, tenant: str, collection_rev: int, *, surface: str) -> PromptRow:
        """The member said no. Ordinary (the safe direction); sticky (§5.2.6, U10).

        A first-grant decline is sticky for the whole org and clears any
        per-scope rows (the org-level state supersedes them). A widening's
        "Keep current" records one ``declined_scopes`` row per asked scope and
        keeps sharing what was already consented. Nothing is sent.
        """

        now = self._clock()
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute(
                "SELECT * FROM prompts WHERE tenant = ? AND collection_rev = ? AND state = ?",
                (tenant, collection_rev, PROMPT_PENDING),
            ).fetchone()
            if found is None:
                conn.execute("ROLLBACK")
                raise GuardConsentRefused(
                    "guard_prompt_not_pending", "this Guard prompt is not waiting for an answer"
                )
            row = _row(found)
            if row.widening:
                conn.executemany(
                    "INSERT OR IGNORE INTO declined_scopes (tenant, scope, at) VALUES (?, ?, ?)",
                    [(tenant, scope, now) for scope in row.asked_scopes],
                )
            else:
                conn.execute("DELETE FROM declined_scopes WHERE tenant = ?", (tenant,))
                conn.execute(
                    "INSERT OR REPLACE INTO sticky (tenant, state, origin, at) VALUES (?, ?, ?, ?)",
                    (tenant, STICKY_DECLINED, "local", now),
                )
            conn.execute(
                "UPDATE prompts SET state = ?, decided_at = ?, decided_via = ?"
                " WHERE tenant = ? AND collection_rev = ?",
                (PROMPT_DECLINED, now, surface[:32], tenant, collection_rev),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        self._settle_mirror(row)
        return row

    def share_again(self, tenant: str) -> None:
        """Settings / ``/guard`` "Share again": lift the org-level sticky state.

        The next admission point then queues the server's prompt (§5.2.6).
        """

        self._exec("DELETE FROM sticky WHERE tenant = ?", (tenant,))
        self._exec("DELETE FROM declined_scopes WHERE tenant = ?", (tenant,))

    def share_more(self, tenant: str) -> None:
        """Settings / ``/guard`` "Share more": forget the per-scope widening declines."""

        self._exec("DELETE FROM declined_scopes WHERE tenant = ?", (tenant,))

    # -- accept ------------------------------------------------------------

    def accept(
        self,
        tenant: str,
        collection_rev: int,
        *,
        capability: bool,
        signature: bool | None,
        source: str,
        expected_digest: str | None = None,
    ) -> AcceptOutcome:
        """Grant exactly the queued row, iff the answer is ADMITTED.

        ``expected_digest`` is the :attr:`PromptRow.digest` of the row the
        person was SHOWN. When given, the claim refuses (``guard_prompt_changed``)
        unless the row it claims still has that digest, checked inside the same
        ``BEGIN IMMEDIATE`` as the claim, so a same-rev replace by ``offer()``
        between "rendered" and "claimed" cannot sign terms nobody saw (L2a
        review round 2, MINOR-R2-2). Every host surface passes it.

        ``capability`` and ``signature`` are the two facts the runtime and the
        desktop plane already compute; :func:`admit_increasing` is the policy.
        A refusal writes a typed host refusal row, leaves the prompt pending
        and sends NOTHING -- the install key is not even looked up.

        Both facts are CALLER-ASSERTED: this method trusts the surface that
        computed them (the attached host TUI, or :meth:`accept_signed` after
        ``verify_for_run``). Same-user code that imports this module and passes
        ``capability=True`` itself admits itself; that is RFC residual (i), and
        the ``guard-credential`` guardrail refuses the obvious model spelling of
        it (defence in depth only).

        The row is CLAIMED (``pending`` -> ``committing``) before anything is
        signed, so exactly one answer can send a grant for it.
        """

        from local_operator.harness.approval import admit_increasing

        if not admit_increasing(capability=capability, signature=signature):
            self._record_refusal(tenant, collection_rev, source, "guard_answer_not_admitted")
            return AcceptOutcome("refused", "guard_answer_not_admitted")
        row = self._claim(tenant, collection_rev, expected_digest)
        if row is None:
            return AcceptOutcome("refused", "guard_prompt_not_pending")
        if set(row.grant_scopes) & UNMEASURED_SCOPES:
            # A row queued before this build refused these scopes (C1): never
            # sign it; the member is asked again once the scope is offerable.
            self._release(row)
            return AcceptOutcome("refused", "guard_scope_unavailable")
        return self._commit(row, source=source)

    def _claim(
        self, tenant: str, collection_rev: int, expected_digest: str | None = None
    ) -> PromptRow | None:
        """Move ONE pending row to ``committing`` under a fresh claim token.

        The token (``prompts.claim``) is what the post-send step settles on: a
        claim the lease released and another answer re-claimed is no longer
        "mine", so a late grant can neither settle nor revoke it by mistake
        (L2a security round 2, S-6).
        """

        if not self._store.exists():
            return None
        token = secrets.token_hex(16)
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            pending = conn.execute(
                "SELECT * FROM prompts WHERE tenant = ? AND collection_rev = ? AND state = ?",
                (tenant, collection_rev, PROMPT_PENDING),
            ).fetchone()
            if (
                pending is not None
                and expected_digest is not None
                and _row(pending).digest != expected_digest
            ):
                conn.execute("ROLLBACK")
                raise GuardConsentRefused(
                    "guard_prompt_changed",
                    "this Guard prompt changed after it was shown; nothing was shared",
                )
            claimed = conn.execute(
                "UPDATE prompts SET state = ?, decided_at = ?, claim = ? WHERE tenant = ? AND"
                " collection_rev = ? AND state = ?",
                (PROMPT_COMMITTING, self._clock(), token, tenant, collection_rev, PROMPT_PENDING),
            ).rowcount
            found = conn.execute(
                "SELECT * FROM prompts WHERE tenant = ? AND collection_rev = ?",
                (tenant, collection_rev),
            ).fetchone()
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return _row(found) if claimed == 1 and found is not None else None

    def _release(self, row: PromptRow) -> None:
        """Hand a failed claim back to ``pending``; a revoke that superseded it wins."""

        self._exec(
            "UPDATE prompts SET state = ?, decided_at = NULL, claim = NULL WHERE tenant = ? AND"
            " collection_rev = ? AND state = ? AND claim IS ?",
            (PROMPT_PENDING, row.tenant, row.collection_rev, PROMPT_COMMITTING, row.claim or None),
        )

    def _release_stale_claims(self) -> None:
        """A claim older than :data:`CLAIM_LEASE_S` is a crashed commit."""

        self._exec(
            "UPDATE prompts SET state = ?, decided_at = NULL, claim = NULL"
            " WHERE state = ? AND decided_at < ?",
            (PROMPT_PENDING, PROMPT_COMMITTING, self._clock() - CLAIM_LEASE_S),
        )

    def accept_signed(self, approval_id: str, expected: Mapping[str, Any]) -> AcceptOutcome:
        """The desktop typed op's apply: the card was signed through ``/approve-operation``.

        ``verify_for_run`` re-derives the record digest and re-verifies the
        operator signature against this machine's anchor; only a record that
        holds becomes ``signature=True``. A denied, unsigned or tampered card
        raises its own refusal and nothing is sent.
        """

        from local_operator.network import approvals as A

        verified = A.verify_for_run(approval_id, root=self._config_dir)
        payload = (verified.get("what") or {}).get("guard_consent") or {}
        tenant = str(payload.get("tenant") or "")
        rev = payload.get("collection_rev")
        row = self.prompt(tenant, rev) if isinstance(rev, int) else None
        if (
            row is None
            or row.digest != payload.get("prompt_digest")
            or expected.get("prompt_digest") != row.digest
            or row.approval_id != approval_id
        ):
            raise GuardConsentRefused(
                "guard_prompt_changed",
                "this Guard card no longer matches the queued prompt; nothing was shared",
            )
        if row.state == PROMPT_ACCEPTED:
            return AcceptOutcome("granted")
        return self.accept(
            tenant,
            row.collection_rev,
            capability=False,
            signature=True,
            source="desktop",
            expected_digest=row.digest,
        )

    def _record_refusal(self, tenant: str, rev: int, source: str, code: str) -> None:
        if not self._store.exists():
            # A refusal for a prompt that was never queued has nothing to annotate,
            # and must not create a store for a non-Team install.
            return
        self._exec(
            "INSERT INTO refusals (tenant, collection_rev, source, code, at)"
            " VALUES (?, ?, ?, ?, ?)",
            (tenant[:128], int(rev), source[:32], code, self._clock()),
        )

    def _install(self) -> tuple[str, bool]:
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute("SELECT install_id, enrolled FROM install WHERE id = 1").fetchone()
            if found is None:
                install_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO install (id, install_id, enrolled) VALUES (1, ?, 0)",
                    (install_id,),
                )
                conn.execute("COMMIT")
                return install_id, False
            conn.execute("COMMIT")
            return found["install_id"], bool(found["enrolled"])
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _resolve_transport(self) -> GuardTransport | None:
        transport = self._transport
        if callable(transport) and not hasattr(transport, "guard_grant"):
            transport = transport()
        return transport  # type: ignore[return-value]

    def _commit(self, row: PromptRow, *, source: str) -> AcceptOutcome:
        """Enrol (lazily) + grant in ONE signed request (§5.2.5). Host-only.

        ``row`` is already claimed (``committing``); every outcome that
        recorded nothing hands the claim back.
        """

        try:
            outcome = self._commit_claimed(row, source=source)
        except BaseException:
            self._release(row)
            raise
        if outcome.status == "failed":
            # Nothing was recorded; the row is answerable again (unless a revoke
            # superseded the claim meanwhile -- ``_release`` keeps that).
            self._release(row)
        return outcome

    def _commit_claimed(self, row: PromptRow, *, source: str) -> AcceptOutcome:
        from local_operator.guard import install_key

        transport = self._resolve_transport()
        if transport is None:
            return AcceptOutcome("failed", "guard_signed_out")
        install_id, enrolled = self._install()
        store = self._key_store or install_key.default_store(self._config_dir)
        # THE ONLY PLACE A GUARD KEY IS CREATED: after admission, for a row the
        # member was shown (module docstring of install_key).
        try:
            key, created = install_key.get_or_create(
                store, install_id, lock_path=install_key.lock_file(self._config_dir)
            )
        except install_key.GuardKeyStoreError:
            # Nothing was sent; the row goes back to pending for the next answer.
            return AcceptOutcome("failed", "guard_key_store_unavailable")
        public = install_key.public_raw(key)
        payload: dict[str, Any] = {
            "tenant": row.tenant,
            "collection_rev": row.collection_rev,
            "revocation_gen": row.revocation_gen,
            "scopes": list(row.grant_scopes),
            "terms_digest": row.terms_digest,
            "install_id": install_id,
            "nonce": secrets.token_hex(16),
            "ts": int(self._clock()),
        }
        if created or not enrolled:
            payload["enroll"] = {"public_key": _b64(public)}
        envelope = {
            "payload": payload,
            "alg": "Ed25519",
            "key_id": hashlib.sha256(public).hexdigest()[:32],
            "signature": _b64(key.sign(SIGNING_DOMAIN + canonical_json(payload))),
        }
        try:
            transport.guard_grant(envelope)
        except Exception as exc:  # noqa: BLE001 -- classified; the row stays pending
            code = str(getattr(exc, "code", "") or "guard_grant_failed")
            if code in _STALE_REVOCATION_CODES:
                # §5.2.5/§5.2.6: a revoke landed first. Revocation wins. When it
                # was THIS install's own revoke (sticky ``revoked`` already set
                # while the grant was in flight) it is not relabelled Console.
                if self.sticky(row.tenant) == STICKY_REVOKED:
                    self._exec(
                        "UPDATE prompts SET state = ? WHERE tenant = ? AND collection_rev = ?"
                        " AND state = ? AND claim IS ?",
                        (
                            PROMPT_SUPERSEDED,
                            row.tenant,
                            row.collection_rev,
                            PROMPT_COMMITTING,
                            row.claim or None,
                        ),
                    )
                else:
                    self._observe_console_revoke(row)
                return AcceptOutcome("revoked_first", code)
            logger.info("guard consent grant failed (%s)", code)
            return AcceptOutcome("failed", code)
        now = self._clock()
        revoked_meanwhile = False
        settled_elsewhere = False
        notice_revoke = True
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # The server enrolled this key whatever happens next.
            conn.execute("UPDATE install SET enrolled = 1 WHERE id = 1")
            current = conn.execute(
                "SELECT state, claim FROM prompts WHERE tenant = ? AND collection_rev = ?",
                (row.tenant, row.collection_rev),
            ).fetchone()
            sticky = conn.execute(
                "SELECT state, at FROM sticky WHERE tenant = ?", (row.tenant,)
            ).fetchone()
            mine = (
                current is not None
                and current["state"] == PROMPT_COMMITTING
                and (current["claim"] or "") == row.claim
            )
            # Security round 1, S-1: a revoke that arrived while the grant was in
            # flight superseded the claim and set sticky ``revoked``. Revoke wins
            # (§5.4, §5.7d): the consent is recorded ``revoke_pending`` -- never
            # ``live`` -- the sticky state stays, and the revoke is re-sent below
            # because the first one may have reached the server before this grant.
            # L2a review round 2, MINOR-R2-1: ANY sticky state counts, not only
            # ``revoked`` -- an org-level "Don't share" on a newer row that landed
            # while this grant was in flight is the member's later answer, and
            # writing ``live`` (then deleting the sticky row) would erase it.
            # L2a security round 2, S-6: a row that is no longer under THIS
            # claim, with no sticky state and not superseded, was released by the
            # lease and answered (or is answerable) again -- another answer, not
            # a revoke. It is settled below without a revoke the member never gave.
            revoked_meanwhile = (
                sticky is not None or current is None or current["state"] == PROMPT_SUPERSEDED
            )
            settled_elsewhere = not revoked_meanwhile and not mine
            if revoked_meanwhile and sticky is not None:
                # One revoke receipt per stop (L2a QA Q5): the member's own
                # revoke already wrote one if it reached the server.
                notice_revoke = (
                    conn.execute(
                        "SELECT 1 FROM notices WHERE tenant = ? AND kind = 'revoke' AND at >= ?",
                        (row.tenant, float(sticky["at"])),
                    ).fetchone()
                    is None
                )
            if settled_elsewhere:
                assert current is not None
                if current["state"] == PROMPT_PENDING:
                    # Released by the lease and not yet re-answered: this grant
                    # IS the member's answer and the server recorded it.
                    conn.execute(
                        "UPDATE prompts SET state = ?, claim = ? WHERE tenant = ?"
                        " AND collection_rev = ? AND state = ?",
                        (
                            PROMPT_COMMITTING,
                            row.claim,
                            row.tenant,
                            row.collection_rev,
                            PROMPT_PENDING,
                        ),
                    )
                    settled_elsewhere = False
                else:
                    # Accepted, or committing under another answer's claim:
                    # that answer records the outcome; nothing to undo here.
                    conn.execute("COMMIT")
            if revoked_meanwhile:
                conn.execute(
                    "INSERT OR REPLACE INTO consents (tenant, collection_rev, scopes, granted_at,"
                    " state) VALUES (?, ?, ?, ?, 'revoke_pending')",
                    (row.tenant, row.collection_rev, json.dumps(list(row.grant_scopes)), now),
                )
                conn.execute(
                    "UPDATE prompts SET state = ? WHERE tenant = ? AND collection_rev = ?"
                    " AND state = ? AND claim IS ?",
                    (
                        PROMPT_SUPERSEDED,
                        row.tenant,
                        row.collection_rev,
                        PROMPT_COMMITTING,
                        row.claim or None,
                    ),
                )
                conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
            raise
        if settled_elsewhere:
            conn.close()
            return AcceptOutcome("granted")
        if revoked_meanwhile:
            conn.close()
            self._settle_mirror(row)
            self._send_revoke(row.tenant, row.org_display_name, now, notice=notice_revoke)
            return AcceptOutcome("revoked_first", "guard_revoked_locally")
        try:
            conn.execute(
                "UPDATE prompts SET state = ?, decided_at = ?, decided_via = ?, claim = NULL"
                " WHERE tenant = ? AND collection_rev = ? AND claim IS ?",
                (
                    PROMPT_ACCEPTED,
                    now,
                    source[:32],
                    row.tenant,
                    row.collection_rev,
                    row.claim or None,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO consents (tenant, collection_rev, scopes, granted_at,"
                " state) VALUES (?, ?, ?, ?, 'live')",
                (row.tenant, row.collection_rev, json.dumps(list(row.grant_scopes)), now),
            )
            conn.execute("DELETE FROM sticky WHERE tenant = ?", (row.tenant,))
            # The receipt notice (§5.7c, §15.4' C3): first grant vs widening.
            conn.execute(
                "INSERT INTO notices (tenant, kind, org_display_name, scopes, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    row.tenant,
                    "widen" if row.widening else "grant",
                    row.org_display_name,
                    json.dumps(list(row.asked_scopes)),
                    now,
                ),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        if source != "desktop":
            self._settle_mirror(row)
        return AcceptOutcome("granted")

    # -- revoke ------------------------------------------------------------

    def revoke(self, tenant: str) -> AcceptOutcome:
        """Stop sharing with ``tenant`` from this install. Always allowed (§5.4).

        Local state first, so a failed call can never leave this install
        sharing: the consent is marked revoked and the tenant sticky before the
        JWT-only revoke is sent. A failure leaves ``revoke_pending`` for the
        next admission point to retry.

        A row an accept is still committing is superseded too, so the grant's
        post-send step sees the revoke and keeps it (security round 1, S-1).
        """

        now = self._clock()
        name = self._scalar(
            "SELECT org_display_name FROM prompts WHERE tenant = ? ORDER BY collection_rev DESC",
            (tenant,),
        )
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE consents SET state = 'revoke_pending' WHERE tenant = ?", (tenant,))
            conn.execute("DELETE FROM declined_scopes WHERE tenant = ?", (tenant,))
            conn.execute(
                "INSERT OR REPLACE INTO sticky (tenant, state, origin, at) VALUES (?, ?, ?, ?)",
                (tenant, STICKY_REVOKED, "local", now),
            )
            conn.execute(
                "UPDATE prompts SET state = ? WHERE tenant = ? AND state IN (?, ?)",
                (PROMPT_SUPERSEDED, tenant, PROMPT_PENDING, PROMPT_COMMITTING),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        self._withdraw_superseded_mirrors(tenant, None)
        return self._send_revoke(tenant, name or tenant, now)

    def _send_revoke(
        self, tenant: str, name: str, now: float, *, notice: bool = True
    ) -> AcceptOutcome:
        """The JWT-only revoke call; on success ``revoked`` + (``notice``) a receipt.

        ``notice=False`` is the post-grant re-send of a revoke the member already
        has a receipt for (L2a QA Q5: one stop, one receipt).
        """

        transport = self._resolve_transport()
        if transport is None:
            return AcceptOutcome("failed", "guard_signed_out")
        try:
            transport.guard_revoke(tenant)
        except Exception as exc:  # noqa: BLE001 -- stays revoke_pending, retried later
            return AcceptOutcome("failed", str(getattr(exc, "code", "") or "guard_revoke_failed"))
        self._exec(
            "UPDATE consents SET state = 'revoked' WHERE tenant = ? AND state = 'revoke_pending'",
            (tenant,),
        )
        if notice:
            self._exec(
                "INSERT INTO notices (tenant, kind, org_display_name, scopes, at)"
                " VALUES (?, 'revoke', ?, '[]', ?)",
                (tenant, name, now),
            )
        return AcceptOutcome("revoked")

    def _observe_console_revoke(self, row: PromptRow) -> None:
        now = self._clock()
        conn = self._store.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE consents SET state = 'revoked' WHERE tenant = ?", (row.tenant,))
            conn.execute("DELETE FROM declined_scopes WHERE tenant = ?", (row.tenant,))
            conn.execute(
                "INSERT OR REPLACE INTO sticky (tenant, state, origin, at) VALUES (?, ?, ?, ?)",
                (row.tenant, STICKY_REVOKED, "console", now),
            )
            conn.execute(
                "UPDATE prompts SET state = ? WHERE tenant = ? AND state IN (?, ?)",
                (PROMPT_SUPERSEDED, row.tenant, PROMPT_PENDING, PROMPT_COMMITTING),
            )
            conn.execute(
                "INSERT INTO notices (tenant, kind, org_display_name, scopes, at)"
                " VALUES (?, 'console_revoke', ?, '[]', ?)",
                (row.tenant, row.org_display_name, now),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        self._settle_mirror(row)

    def retry_pending_revokes(self) -> None:
        """Re-send revokes a previous call could not deliver. Never raises."""

        for tenant in self._column(
            "SELECT tenant FROM consents WHERE state = 'revoke_pending'", ()
        ):
            try:
                self.revoke(tenant)
            except Exception:  # noqa: BLE001
                logger.debug("guard revoke retry failed", exc_info=True)

    # -- the desktop approvals mirror -----------------------------------------

    def _mirror(self, row: PromptRow) -> PromptRow | None:
        """File (or re-file) the host-level ``guard_consent`` card. Never raises.

        Skipped when this machine has no operator anchor: the desktop could not
        sign the card anyway, and the queue row stays answerable from an
        attached host surface (the spawn-capability path).
        """

        if not self._mirror_enabled:
            return None
        try:
            from local_operator.guard import consent_op

            approval_id = consent_op.file_mirror(self._config_dir, row, now=self._clock())
        except Exception:  # noqa: BLE001 -- the queue row is the source of truth
            logger.debug("guard consent mirror could not be filed", exc_info=True)
            return None
        if approval_id is None:
            return None
        self._exec(
            "UPDATE prompts SET approval_id = ?, mirror_seq = ? WHERE tenant = ? AND"
            " collection_rev = ?",
            (approval_id, row.mirror_seq, row.tenant, row.collection_rev),
        )
        return self.prompt(row.tenant, row.collection_rev)

    def _settle_mirror(self, row: PromptRow) -> None:
        self._withdraw_card(row.approval_id)

    def _withdraw_card(self, approval_id: str) -> None:
        """Withdraw a card the queue already answered. Never raises."""

        if not approval_id:
            return
        try:
            from local_operator.guard import consent_op

            consent_op.settle_mirror(self._config_dir, approval_id)
        except Exception:  # noqa: BLE001
            logger.debug("guard consent mirror could not be settled", exc_info=True)

    def _withdraw_superseded_mirrors(self, tenant: str, keep_rev: int | None) -> None:
        for approval_id, rev in self._rows(
            "SELECT approval_id, collection_rev FROM prompts WHERE tenant = ? AND state = ?"
            " AND approval_id != ''",
            (tenant, PROMPT_SUPERSEDED),
        ):
            if keep_rev is not None and rev == keep_rev:
                continue
            try:
                from local_operator.guard import consent_op

                consent_op.settle_mirror(self._config_dir, approval_id)
            except Exception:  # noqa: BLE001
                logger.debug("guard consent mirror could not be withdrawn", exc_info=True)

    def _rows(self, sql: str, args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        if not self._store.exists():
            return []
        conn = self._store.connect()
        try:
            return [tuple(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def reconcile_mirrors(self) -> None:
        """Fold desktop answers back into the queue; re-file lapsed cards. Never raises.

        A desktop deny is the member's "Don't share" / "Keep current"; an
        expired card is re-filed under the next ``mirror_seq`` because the
        queue itself never expires.
        """

        self.fold_desktop_answers()
        for row in self.pending():
            try:
                from local_operator.guard import consent_op

                state = consent_op.mirror_state(self._config_dir, row.approval_id)
                if state in {"expired", "withdrawn", "missing"}:
                    bumped = PromptRow(**{**row.__dict__, "mirror_seq": row.mirror_seq + 1})
                    self._mirror(bumped)
            except Exception:  # noqa: BLE001
                logger.debug("guard consent mirror reconcile failed", exc_info=True)

    def fold_desktop_answers(self) -> None:
        """Fold desktop DENIES into the queue, and release crashed claims. Never raises.

        Runs BEFORE an admission point's offers (QA round 1, Q1): a deny the
        member gave on the desktop is only a card state until folded, and an
        offer for a newer rev would otherwise supersede the still-``pending``
        row first -- dropping the "Don't share" (org sticky) or "Keep current"
        (U10 declined scopes) and asking again.
        """

        try:
            self._release_stale_claims()
        except Exception:  # noqa: BLE001
            logger.debug("guard consent stale-claim release failed", exc_info=True)
        for row in self.pending():
            try:
                from local_operator.guard import consent_op

                if consent_op.mirror_state(self._config_dir, row.approval_id) == "denied":
                    self.decline(row.tenant, row.collection_rev, surface="desktop")
            except Exception:  # noqa: BLE001
                logger.debug("guard consent desktop fold failed", exc_info=True)


# ---------------------------------------------------------------------------
# The admission point
# ---------------------------------------------------------------------------


def team_tenants(config_dir: Path | str) -> list[tuple[str, str]]:
    """``[(tenant_id, tenant_name)]`` this account is a Team (or Team trial) member of.

    ZERO NETWORK: the org-hub gate's cached record (``hub_org.eligibility``),
    whose predicate is the server's own entitlement -- ``/me/memberships``
    reports the EFFECTIVE plan status, so a live Team trial reads ``active``
    there (agent-server ``membership_service.planSummary``, measured at
    ``42da8aa4``). Empty for every non-Team account.
    """

    from local_operator.hub_org import eligibility

    return [(org.tenant_id, org.tenant_name) for org in eligibility.eligible_orgs(Path(config_dir))]


def check_point(
    config_dir: Path | str,
    *,
    boundary: str,
    transport_factory: Callable[[], GuardTransport | None] | None = None,
    key_store: Any = None,
) -> list[OfferOutcome]:
    """One admission point (sign-in, daemon start, the 6 h tick). Never raises.

    Returns BEFORE any network call, file or credential-store access for an
    account with no Team membership (§5.1, the L2a zero-call row). The one
    exception is an install that already HAS a consent store with a
    ``revoke_pending`` row (a member who lapsed after a failed revoke): that
    revoke is retried, because revoke has no membership precondition (§5.4).
    A never-Team install has no store, so it still makes zero calls.
    """

    try:
        factory = transport_factory or (lambda: default_transport(config_dir))
        tenants = team_tenants(config_dir)
        if not tenants:
            # Review round 1, MINOR-4: revoke has no membership precondition
            # (§5.4), so a member who lapsed or left after a failed revoke still
            # gets it retried. Zero cost for a non-Team install: it has no
            # store, and the transport is resolved only for a revoke_pending row.
            if GuardConsentStore(config_dir).exists():
                GuardConsentHost(
                    config_dir, transport=factory, key_store=key_store
                ).retry_pending_revokes()
            return []
        transport = factory()
        if transport is None:
            return []
        host = GuardConsentHost(config_dir, transport=transport, key_store=key_store)
        if host.store.exists():
            host.fold_desktop_answers()
        outcomes: list[OfferOutcome] = []
        for tenant_id, _name in tenants:
            try:
                answer = transport.guard_status(tenant_id)
            except Exception as exc:  # noqa: BLE001 -- e.g. 404 before A2 ships
                logger.debug(
                    "guard status read failed (%s, %s)", boundary, getattr(exc, "code", "")
                )
                continue
            if not isinstance(answer, Mapping) or answer.get("status") != "consent_required":
                continue
            try:
                offer = ConsentOffer.from_wire(answer.get("prompt") or {})
            except GuardConsentRefused:
                logger.info("guard status carried an invalid offer (%s)", boundary)
                continue
            if offer.tenant != tenant_id:
                continue
            outcomes.append(host.offer(offer))
        if host.store.exists():
            host.reconcile_mirrors()
            host.retry_pending_revokes()
        return outcomes
    except Exception:  # noqa: BLE001 -- an admission point never fails its boundary
        logger.debug("guard consent check point failed (%s)", boundary, exc_info=True)
        return []


def default_transport(config_dir: Path | str) -> GuardTransport | None:
    """The signed-in person's ``RadientClient``, or ``None`` when signed out."""

    from local_operator.hub_org import eligibility

    resolved = eligibility.person_client(Path(config_dir))
    return None if resolved is None else resolved[0]


def iter_scopes(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


__all__ = [
    "AcceptOutcome",
    "ConsentOffer",
    "GuardConsentHost",
    "GuardConsentRefused",
    "GuardConsentStore",
    "GuardTransport",
    "OfferOutcome",
    "PromptRow",
    "REASONS",
    "SCOPES",
    "SIGNING_DOMAIN",
    "canonical_json",
    "check_point",
    "default_transport",
    "guard_consent_path",
    "team_tenants",
]

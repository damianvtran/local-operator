"""The join-time share list: candidates, per-kind defaults, renderers, bounds.

WHY THIS MODULE EXISTS (the both-ends join screen). Before a device is admitted,
BOTH humans must see the same list of what will be shared: the owner needs it to
decide what to lend, and the joiner needs it to see what it will borrow. One list
is produced here and travels as the ``net_pair_offer`` frame's ``items``; the
owner's confirm screen, the joiner's prompt and the park payload all render from
those same items, and ``lop network credentials`` (the ledger) classifies kinds
through the same helpers — a second classifier is how a screen and a share come
to disagree about what ``openai`` is.

THE LIST IS DISPLAY DATA UNTIL THE OWNER'S ADMISSION. Nothing here authorises
anything: the frame is sealed under the pair link's keys, only a peer that
already proved possession of the invite ever sees it (the ``pair-offer-v1`` gate
is on the joiner's OWN advertisement), the owner may only REDUCE what it sent,
and the grants are written at admission from the owner's own store. A per-kind
default is a renderer's mark on a row, not a grant.

``share`` DEFAULTS per ``docs/design/mesh-consent-provisioning.md`` §1.4, which
supersedes the §2.3 table this module was originally built on: **approval of a
device IS the authorisation**, so the join-time list is the default provisioning
set and every row stays reduce-only (the owner may drop rows; ``credential
share|revoke`` remain as adjustment surfaces, never prerequisites):

* ``oauth-rotating`` — **yes** (unchanged): R13's core case (a peer with no login
  must work without the operator re-authenticating), and ``scope: "session"`` at
  grant time keeps it the smallest useful authority.
* ``api-key-static`` — **yes** (flipped from no): a static key is the class that
  *works* from a second device, and the concern that kept it off ("a permanent
  capability increase", mesh-credentials.md:328) is answered by the approval gate
  and the revoke/rotate surfaces, not by making the operator type ``share`` per
  key.
* ``mcp-rotating`` — **yes** (flipped from no): the access token was always
  brokerable and the refresh grant never moves (the ``_oauth_refresh_lock`` stays
  host-local), so the v1 scope-surface caution is spent.
* ``github-app`` — **yes when the adapter resolves a source** (§3): the
  candidate gate is the source ladder (``resolve_source``: the App secret, a
  ``GITHUB_TOKEN``-class secret, or the ``gh`` login), and the ladder widens
  which devices grow the row, not this default. The "one App covers every
  designated repository" concern is bounded by the repository allow-list and the
  helper's path check.
* ``radient`` — **offered by default** for ``device`` members (flipped from
  never-auto-offered): the org bearer carries organization-write authority, but
  the operator's directive covers exactly this login — a node that cannot publish
  an agent cannot do the work it was onboarded for — and the reduce step is the
  narrowing surface. Pool exclusion is structural and unchanged (§1.3: pool
  members declare no credentials and are excluded from every credential path).
* ``secret`` (class 2) remains NEVER a candidate — it is not brokered at all.

The two exclusions that do NOT move, because they are facts rather than
postures: device-bound providers (kimi, below) and host-local credentials (the
mobile portal password lives only in the macOS Keychain and never enters the
candidate set).

Stdlib only at import (``credentials/__init__.py`` states the rule): everything
heavy — the readiness store read, the placement document, the MCP token storage —
is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from local_operator.network import wire

#: The order of the offer states both ends speak. The joiner's state and the
#: owner's registry can never carry a fourth string by accident: the wire frame
#: and the records round-trip through these constants.
OFFER_LISTED = "listed"
#: The owner advertised ``pair-offer-v1`` and sent an EMPTY list — a different
#: fact from the joiner's "absent" below, and the two render different lines
#: (§4.3 vs §4.4): "did not offer" is about a build, "offered nothing" is about a
#: device's store.
OFFER_EMPTY = "empty"
#: No offer was sent or read: the peer did not advertise the capability (an older
#: build), or the bounded drain ran out and a late frame never arrived. The
#: joiner treats both the same on screen because the remedy is the same: nothing
#: is served in this ceremony, and a share can follow the join.
OFFER_ABSENT = "absent"
#: The OWNER-side record state for "the other device never advertised the
#: capability": the confirm screen says so out loud (its build cannot be shown a
#: list) instead of silently showing an empty one.
OWNER_SKIPPED = "skipped_peer_unsupported"
#: The owner SENT a list (possibly empty). ``PendingPairing.offer_state``.
OWNER_SENT = "sent"

#: Kinds the join list can carry, and the two renderers' spelling of each. Keys
#: here are exactly the kinds ``shape_from_rows`` can return; a kind outside this
#: table is a bug in the classifier, never a wire value.
KIND_LABELS: dict[str, str] = {
    "oauth-rotating": "OAuth",
    "api-key-static": "API key",
    "mcp-rotating": "MCP login",
    # The github adapter's row (github.py). "GitHub", not "GitHub App": the row
    # can be served from any ladder arm (§3.2) — the label names the row, and
    # the receipt names the arm that actually served.
    "github-app": "GitHub",
}

#: The per-kind default posture (§1.4 of ``mesh-consent-provisioning.md``,
#: which supersedes the §2.3 table: approval of a device is the authorisation,
#: so the join-time list is the default provisioning set). A kind absent here has
#: no default and is not offered; ``share_default`` answers ``False`` for it,
#: which is the closed direction.
SHARE_DEFAULT_BY_KIND: dict[str, bool] = {
    "oauth-rotating": True,
    # Flipped from False (S1 of the provisioning design): static keys are the
    # class that works from a second device, and the approval gate + revoke/rotate
    # surfaces bound them where the old default made the operator share per key.
    "api-key-static": True,
    # Flipped from False: the access token was always brokerable, the refresh
    # grant is host-local by construction, and the MCP defs push already reports
    # the server set the node will run.
    "mcp-rotating": True,
    # Flipped from False, conditional on the adapter resolving a source (§3):
    # the candidate gate is the source ladder (``resolve_source``), so the row
    # exists whenever an arm can serve; the ladder widens which devices grow the
    # row. The "one App covers every designated repository" concern is bounded by
    # the repository allow-list and the helper's path check.
    "github-app": True,
}

#: The candidate cap. 64 is far above any real store and keeps a hostile or
#: broken store from producing a frame that strains the record budget; an
#: overflow drops the TAIL of the sorted list (the list is sorted by key, so
#: what survives is deterministic).
MAX_OFFER_ITEMS = 64

#: Display bound for a masked identity label, in characters. The label travels in
#: the frame MASKED (see ``mask_label``): the pairing screen's use for it is
#: "which of my two openai logins is this", never the account itself, and the
#: design's review explicitly refused writing it anywhere else (§8, label row:
#: masked in the UI, never in the audit).
MAX_OFFER_LABEL = 64

#: Bound for a row's ``kind`` string. Unknown kinds are TOLERATED (see
#: ``validate_items`` and the evolution rule on ``canonical_rows``): the bound is
#: what keeps an unknown kind from being a smuggling channel, not a membership
#: test over today's labels.
MAX_OFFER_KIND = 64


class OfferEnumerationError(Exception):
    """A store that EXISTS but cannot be read.

    Deliberately distinct from "no store at all" (which is an honest empty
    offer): the caller sends an empty offer either way, but only this case
    records ``enumeration: "unreadable"`` in the audit, so an operator
    investigating "why was my list empty" can tell a device with nothing to
    share from one whose store is broken.
    """


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def mask_label(label: str) -> str:
    """``damian@example.com`` → ``d***@example.com``; anything else → first char + ``***``.

    DISPLAY-LEVEL MASKING ONLY, and it lives HERE so the frame, the owner's
    confirm screen and the joiner's prompt cannot disagree about what a reader
    sees. The domain (or the tail after the first character) is kept on purpose:
    the label's whole job is telling two logins for one provider apart, and a
    fully-bulleted string does not. The value never enters the audit log.
    """
    text = str(label or "")
    if not text:
        return ""
    if "@" in text:
        local_name, _, domain = text.partition("@")
        return f"{local_name[:1]}***@{domain}"[:MAX_OFFER_LABEL]
    return f"{text[:1]}***"[:MAX_OFFER_LABEL]


def share_default(kind: str) -> bool:
    """The owner's default posture for one kind. Closed: unknown → not offered."""
    return SHARE_DEFAULT_BY_KIND.get(kind, False)


def kind_label(kind: str) -> str:
    """The row's spelling of a kind; unknown kinds print themselves."""
    return KIND_LABELS.get(kind, kind)


# ---------------------------------------------------------------------------
# The shared reads (moved out of ``network/cli.py`` so the ledger, the offer and
# the confirm screen classify through ONE implementation)
# ---------------------------------------------------------------------------


def open_store(config: Path) -> Any | None:
    """This device's credential store at ``config``, or ``None`` when none can be read.

    TWO ABSENCES, ONE ANSWER, deliberately: no database at all (the store's
    absence — ``readiness._open_store`` refuses to create one; review round 1,
    MINOR; QA round 1, Q1) and a database that EXISTS but cannot be opened
    (corrupt bytes, a ``000`` mode) both answer ``None`` here. Every caller that
    wants a degrade renders ``None`` as its own answer; the offer's own build
    path uses the STRICT read below when it needs the two apart.
    """
    from local_operator.network import readiness as readiness_mod

    try:
        return readiness_mod._open_store(config)  # noqa: SLF001 — the one read-only store guard
    except Exception:  # noqa: BLE001 — an unopenable store answers None, never raises
        return None


def close_quietly(store: Any) -> None:
    """Close a store without letting a close failure change the answer."""
    try:
        store.close()
    except Exception:  # noqa: BLE001 — closing is best-effort
        pass


def provider_rows(provider: str, config: Path) -> list[Any]:
    """This device's rows for a provider, or ``[]``. Never raises, never writes."""
    try:
        store = open_store(config)
        if store is None:
            return []
        try:
            return list(store.list_credentials(provider))
        finally:
            close_quietly(store)
    except Exception:  # noqa: BLE001 — an unreadable store is "no credential here"
        return []


def shape_from_rows(key: str, rows: list[Any]) -> tuple[str, str, str]:
    """``(kind, key, identity label)`` from a provider's enabled store rows.

    One spelling for the share verb, the ledger and the offer. An OAuth row is
    preferred over an older static row (see ``shape_for_key``); ``key`` is what
    the no-rows case names.
    """
    if not rows:
        return "oauth-rotating", key, ""
    row = next((candidate for candidate in rows if candidate.credential_type == "oauth"), rows[0])
    label = str(row.data.get("email") or row.data.get("account_id") or "")
    kind = "oauth-rotating" if row.credential_type == "oauth" else "api-key-static"
    return kind, key, label


def shape_for_key(key: str, config: Path) -> tuple[str, str, str]:
    """``(placement kind, provider, identity label)`` for a key.

    Read from the OWNER's own store, so the document records what is actually
    signed in rather than what the operator typed: a ``kind`` that disagreed with
    the row would make the broker's narrowing rule and the operator's expectation
    diverge.

    THE OAUTH ROW WINS over an older pasted key (Radient org projection): a
    provider can hold BOTH a ``radient-key`` login and the org OAuth login under
    one provider (``radient-key`` aliases into ``"radient"``), and org calls are
    served from the OAuth row. Reading the oldest row — ``list_credentials`` is
    ``ORDER BY id`` — recorded ``api-key-static``/``""`` for a store whose org
    calls are OAuth, so a share looked narrower than the login actually is. The
    aliased MIXED buckets today are ``radient``, ``xai`` and ``zai``; the OAuth
    row wins in all of them.
    """
    from local_operator.network.credentials import github as github_mod
    from local_operator.network.credentials.types import is_mcp_key, mcp_url_from_key

    if github_mod.is_github_key(key):
        # github has no store row, so its shape is the key's own name. The
        # identity label stays empty on purpose: the row is served from whichever
        # ladder arm resolves (§3.2) — an App, a token, or a gh login — and a
        # label naming one of them would become display data that is wrong the
        # moment the device's sources change.
        return github_mod.GITHUB_KIND, github_mod.GITHUB_KEY, ""
    if is_mcp_key(key):
        url = mcp_url_from_key(key)
        # OPEN FIRST, CONSTRUCT ONLY WHEN A STORE CAME BACK (review round 1,
        # MINOR; QA round 1, Q1): the ambient ``McpTokenStorage(url)`` built an
        # ``AuthStore`` that CREATES its database, so this read made a store
        # appear on a device that never signed in.
        store = open_store(config)
        if store is not None:
            try:
                from local_operator.mcp.auth import McpTokenStorage

                if McpTokenStorage(url, store=store).has_stored_row():
                    return "mcp-rotating", "mcp-oauth", ""
            except Exception:  # noqa: BLE001 — an unreadable store is "not signed in"
                pass
            finally:
                close_quietly(store)
        return "mcp-rotating", "mcp-oauth", ""
    return shape_from_rows(key, provider_rows(key, config))


def credential_here(key: str, config: Path) -> bool:
    """Whether this device actually holds ``key`` RIGHT NOW.

    The admission-time re-check, and the same read the share verb's refusal
    makes: a declared entry for a credential that is not here would put a row in
    every device's document whose owner cannot serve it. An unreadable store
    answers ``False`` — the closed direction at a grant seam.
    """
    from local_operator.network.credentials import github as github_mod
    from local_operator.network.credentials.types import is_mcp_key, mcp_url_from_key

    if github_mod.is_github_key(key):
        # "Does this device hold it" for github is whether any ladder arm
        # resolves (§3.2) — the App secret, a ``GITHUB_TOKEN``-class secret, or
        # the gh CLI's own login — read read-only, the same guard as every
        # other candidate (never constructs a store, never creates one). An
        # UNREADABLE store is not a promise: ``source_present`` folds it to no
        # (M1 — a share must not claim what the serve path would refuse).
        return github_mod.source_present(config)
    if is_mcp_key(key):
        url = mcp_url_from_key(key)
        store = open_store(config)
        if store is None:
            return False
        try:
            from local_operator.mcp.auth import McpTokenStorage

            return bool(McpTokenStorage(url, store=store).has_stored_row())
        except Exception:  # noqa: BLE001 — an unreadable store is "not signed in"
            return False
        finally:
            close_quietly(store)
    return bool(provider_rows(key, config))


# ---------------------------------------------------------------------------
# Candidates (the strict read the offer builds from)
# ---------------------------------------------------------------------------


def _strict_provider_rows(config: Path) -> list[Any]:
    """Every credential row, or ``[]`` when no store exists — raising when it exists
    and cannot be read (see :class:`OfferEnumerationError`)."""
    database = config / "auth.db"
    if not database.exists():
        return []
    try:
        from local_operator.network import readiness as readiness_mod

        store = readiness_mod._open_store(config)  # noqa: SLF001 — the one read-only store guard
        if store is None:  # pragma: no cover — exists() said otherwise; belt and braces
            return []
        try:
            return list(store.list_credentials(None))
        finally:
            close_quietly(store)
    except Exception as exc:  # noqa: BLE001 — the caller distinguishes this from "no store"
        raise OfferEnumerationError(str(exc)) from exc


def enumerate_candidates(config: Path) -> list[dict[str, Any]]:
    """Every credential this device could serve, sorted by key.

    Excluded BY NAME, each for a reason the broker would refuse anyway: MCP rows
    under the ``mcp-oauth`` provider (the server candidates below are their
    ledger) and device-bound providers (``kimi`` — a grant would be refused by
    name at placement time). Radient was a third exclusion until S1 of the
    provisioning design flipped it: it is now an ordinary candidate (its kind's
    default decides, and §1.4 decided yes for device members), while the pool
    exclusion stays structural — pool members never reach this list because they
    declare no credentials at all (§1.3). An MCP server is a candidate only when
    a login row exists HERE: offering a server with nothing behind it would
    promise the joiner something the admission's re-check drops.

    Raises :class:`OfferEnumerationError` when the store exists but cannot be
    read — the caller sends an empty offer and records WHY.
    """
    from local_operator.network.credentials.types import (
        DEVICE_BOUND_PROVIDERS,
        credential_key_for_mcp,
    )

    rows: list[dict[str, Any]] = []

    by_provider: dict[str, list[Any]] = {}
    for credential in _strict_provider_rows(config):
        provider = str(getattr(credential, "provider", "") or "")
        if not provider or provider == "mcp-oauth" or provider in DEVICE_BOUND_PROVIDERS:
            continue
        by_provider.setdefault(provider, []).append(credential)
    for provider in sorted(by_provider):
        kind, _name, label = shape_from_rows(provider, by_provider[provider])
        rows.append({"key": provider, "kind": kind, "label": label})

    from local_operator.network import readiness as readiness_mod

    fact = readiness_mod.mcp_servers_fact(config)
    for server in fact.get("servers") or []:
        if server.get("transport") not in ("http", "sse"):
            continue
        url = str(server.get("url") or "")
        if not url or server.get("has_row") is not True:
            # No URL, no login to hold and nothing to serve — and a read that
            # cannot open the store reports ``has_row: None``, which must NOT
            # read as "serve it anyway".
            continue
        rows.append({"key": credential_key_for_mcp(url), "kind": "mcp-rotating", "label": ""})

    from local_operator.network.credentials import github as github_mod

    if github_mod.source_present(config):
        # A ladder arm resolves, so there is something to serve: the row has no
        # store row, so it is named here directly. The kind's default decides its
        # posture on the join list — its ``share`` resolves through
        # ``share_default`` like every row, True for ``github-app`` since the
        # §1.4 flip — and this gate only says the row exists.
        rows.append({"key": github_mod.GITHUB_KEY, "kind": github_mod.GITHUB_KIND, "label": ""})

    rows.sort(key=lambda row: row["key"])
    return rows


def build_items(config: Path) -> list[dict[str, Any]]:
    """The offer's ``items``: candidates + per-kind defaults + masked labels, capped.

    Sorted by key (the sort is part of the contract — the digest commits to the
    list as built, and two implementations that sort differently would fail each
    other's digest check), capped at :data:`MAX_OFFER_ITEMS`.
    """
    rows = sorted(enumerate_candidates(config), key=lambda row: str(row["key"]))
    items = [
        {
            "key": row["key"],
            "kind": row["kind"],
            "label": mask_label(row["label"]),
            "share": share_default(row["kind"]),
        }
        for row in rows
    ]
    return items[:MAX_OFFER_ITEMS]


# ---------------------------------------------------------------------------
# The frame's item shape
# ---------------------------------------------------------------------------


def canonical_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The row projection both ends digest and validate to.

    THE ROW-SCHEMA EVOLUTION RULE (agent review round 1, M1): ADDITIVE evolution
    of ``pair-offer-v1`` is tolerated. Unknown per-row FIELDS never enter the
    digest — it is computed over THIS projection on both ends, so a newer sender
    may add one without refusing every pairing with a not-yet-upgraded joiner —
    and unknown KINDS render through ``kind_label``'s fallback and can never
    grant (grants are owner-side and key-based; a joiner only displays). A change
    that alters the MEANING of an existing field, or the frame's security
    properties, ships as ``pair-offer-v2``: an old joiner would read it under the
    old rules.
    """
    return [
        {
            "key": str(item["key"]),
            "kind": str(item["kind"]),
            "label": str(item["label"]),
            "share": bool(item["share"]),
        }
        for item in items
    ]


def digest_of(items: list[dict[str, Any]]) -> str:
    """The frame's ``digest``: sha256 over the canonical projection of ``items``.

    A CHECKSUM, NOT A SIGNATURE — the record's AEAD already protects the frame,
    and this cannot add integrity the link keys do not have. What it pins is the
    two screens agreeing on ONE list: the builder and the parser both recompute
    it over ``canonical_rows`` (so additive fields ride without breaking an old
    peer, per the evolution rule there), and a build that dropped, reordered or
    rewrote a row fails its own frame rather than showing the joiner a list the
    owner never offered.
    """
    return hashlib.sha256(wire.canonical_json(canonical_rows(items))).hexdigest()


def validate_items(value: Any) -> list[dict[str, Any]]:
    """Validate a frame's ``items``; raise ``ValueError`` on anything unusable.

    Raises rather than dropping: an item the parser cannot trust is a frame the
    ceremony must refuse (the caller maps this to ``protocol_error``), and a
    tolerant parse would show a list that does not match the digest anyway.

    UNKNOWN KINDS ARE TOLERATED BY THE ROW-SCHEMA RULE (``canonical_rows``):
    they are bounded, digested (so tampering still fails) and DISPLAYED through
    ``kind_label``'s fallback — and they can never grant, because grants are the
    owner's own, made against its own store, key by key. An unknown kind is what
    a newer sender's additive row looks like; refusing it here would turn every
    additive change into a whole-ceremony refusal on older joiners (agent review
    round 1, M1).
    """
    if not isinstance(value, list):
        raise ValueError("the share list is not a list")
    if len(value) > MAX_OFFER_ITEMS:
        raise ValueError(f"the share list carries more than {MAX_OFFER_ITEMS} items")
    items: list[dict[str, Any]] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"share list item {position} is not an object")
        key = raw.get("key")
        kind = raw.get("kind")
        label = raw.get("label")
        share = raw.get("share")
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError(f"share list item {position} has no usable key")
        if not isinstance(kind, str) or not kind or len(kind) > MAX_OFFER_KIND:
            raise ValueError(f"share list item {position} has an unusable kind")
        if not isinstance(label, str) or len(label) > MAX_OFFER_LABEL:
            raise ValueError(f"share list item {position} has an unusable label")
        if not isinstance(share, bool):
            raise ValueError(f"share list item {position} has no share flag")
        items.append({"key": key, "kind": kind, "label": label, "share": share})
    return items


def served_keys(items: list[dict[str, Any]]) -> list[str]:
    """The keys the owner said it WOULD serve (``share: true``), in list order."""
    return [str(item["key"]) for item in items if item.get("share")]


def owner_default_shares(items: list[dict[str, Any]]) -> list[str]:
    """The default decision: the full offered set (§3.3 — the default is what was
    offered, so a y/N-only flow behaves as "accept what was offered")."""
    return sorted(served_keys(items))


# ---------------------------------------------------------------------------
# The renderers (one list, four surfaces)
# ---------------------------------------------------------------------------


def render_rows(items: list[dict[str, Any]], *, decision: list[str] | None = None) -> list[str]:
    """The row block every surface shows, in the design's shape (§2.3/§3.1).

    The ``(kind, label)`` cell is padded to the block's widest so the STATE
    column lines up — the list exists to be scanned down that column (design
    round 1, D2).

    ``decision`` renders a CEREMONY IN PROGRESS instead of the wire's original
    flags: keys still in the list read ``will be served``, offered keys the
    person has removed read ``no longer served``, and unoffered ones stay ``not
    offered``. The confirm screen re-renders from the current decision after each
    edit, so the frame the final ``y`` lands on matches the grant (design round
    1, D1 / UX round 1, U1).
    """
    cells: list[str] = []
    for item in items:
        inner = kind_label(str(item.get("kind") or ""))
        label = str(item.get("label") or "")
        if label:
            inner = f"{inner}, {label}"
        cells.append(f"{item.get('key')} ({inner})")
    width = max((len(cell) for cell in cells), default=0)
    served = {str(key) for key in decision} if decision is not None else set()
    rows: list[str] = []
    for item, cell in zip(items, cells):
        if decision is None:
            state = "will be served" if item.get("share") else "not offered"
        elif str(item.get("key")) in served:
            state = "will be served"
        elif item.get("share"):
            state = "no longer served"
        else:
            state = "not offered"
        rows.append(f"  {cell}{' ' * (width - len(cell))}   {state}")
    return rows


JOINER_TAIL = "the inviter can remove items before admitting; nothing else will be served."
OWNER_TAIL = "the list can only shrink; nothing else will be served."
#: §4.3's sentence. The old clause "or has none to share" is §4.4's fact (a
#: DIFFERENT sentence) and it was false for the drain-expiry corner, where the
#: peer is a new build whose offer simply did not arrive — so the parenthetical
#: names the two causes the state actually covers (design round 1, D4).
NOT_OFFERED_LINE = (
    "the other device did not offer credentials (it is an older build, or the offer "
    "did not arrive)"
)
EMPTY_OFFER_LINE = "no credentials were offered"
OWNER_EMPTY_LINE = "nothing will be served in this ceremony"
OWNER_SKIPPED_LINE = (
    "the other device is an older build; it cannot be shown a share list — share after the "
    "join with `lop network credential share <key> --with <device>`"
)


def render_joiner_block(items: list[dict[str, Any]], *, inviter: str) -> list[str]:
    """The joinER's screen lines: what the other device will serve to this one."""
    return [
        f"Credentials {inviter} will serve to this device:",
        *render_rows(items),
        JOINER_TAIL,
    ]


def render_owner_block(
    items: list[dict[str, Any]], *, joiner: str, decision: list[str] | None = None
) -> list[str]:
    """The ownER's screen lines: what this device will serve to the joining one.

    ``decision`` is the in-progress set the confirm screen re-renders from
    (``render_rows``); the wire's original flags render when it is omitted.
    """
    return [
        f"Credentials this device will serve to {joiner}:",
        *render_rows(items, decision=decision),
        OWNER_TAIL,
    ]


def render_state_line(state: str) -> str:
    """The one-line render for a list that is absent or empty (``OFFER_ABSENT``
    gets §4.3's sentence, ``OFFER_EMPTY`` §4.4's — they are different facts)."""
    return EMPTY_OFFER_LINE if state == OFFER_EMPTY else NOT_OFFERED_LINE


def missing_share_lines(offered: list[str], shares: list[str], reduced: list[str]) -> list[str]:
    """The receipt's delta lines: a deliberate reduction is its OWN fact.

    ``reduced`` (from the result frame / the payload) names the keys the owner's
    PERSON removed at the confirm screen; anything else that was promised and not
    served failed on the owner's device and keeps the remedy sentence. The two
    must not read the same (UX round 1, U2): pointing the joiner at the share
    verb for a decision the owner already took re-opens that decision.

    THE SUBJECT IS THE JOINER'S, NOT THE OWNER'S (UX round 1, U5): these lines
    read ``not available here:`` — on the joining device "serve" belongs to the
    other end ("Credentials <inviter> will serve to this device"), so the receipt
    says what became AVAILABLE HERE rather than flipping the verb's direction.
    """
    served = set(shares)
    reduced_set = set(reduced)
    deliberate = [key for key in offered if key not in served and key in reduced_set]
    failed = [key for key in offered if key not in served and key not in reduced_set]
    lines: list[str] = []
    if deliberate:
        pronoun = "it" if len(deliberate) == 1 else "them"
        lines.append(
            f"not available here: {', '.join(deliberate)} — the other device chose not to "
            f"share {pronoun}"
        )
    if failed:
        lines.append(
            f"not available here: {', '.join(failed)} — ask the other device to run "
            "`lop network credential share <key> --with <device>` to lend it after the join"
        )
    return lines


def sentence_clause(state: str, items: list[dict[str, Any]], *, inviter: str) -> str:
    """The clause ``network/cli.py`` appends to the park payload's sentence.

    THE SENTENCE IS PART OF THE CONTRACT (``_awaiting_payload``): an agent that
    shows a user anything other than this sentence is paraphrasing the one
    instruction the ceremony rests on, so the share-list fact rides the sentence
    itself rather than only the payload's structured fields.
    """
    if state == OFFER_LISTED:
        served = ", ".join(item["key"] for item in items if item.get("share")) or "nothing"
        return f"{inviter or 'The other device'} will serve: {served}."
    line = render_state_line(state)
    return f"{line[:1].upper()}{line[1:]}."

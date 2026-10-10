"""Per-session CHANNEL spend: the metered ledger beside the token accumulator.

Why this module exists (design ``docs/design/spend-channels.md`` and the
project's design note): session money used to be inference only. ``SessionSpend``
(``session/spend.py``) counts model turns, and web search kept a second,
in-memory ledger (``web_search/cost.py``) that the TUI folded in at paint time
and every other surface omitted. Images, TTS and STT recorded nothing at all.
The result was surfaces that disagreed about what one session cost, and no way
to ask "what did the images cost" of a resumed conversation.

The model here is an EVENT LEDGER, not a second accumulator:

- One frozen :class:`ChannelSpendRecord` per money event, keyed by a globally
  unique ``record_id``. The record is durable in the transcript
  (``session_channel_spend.v1`` custom rows), replayed by the fold on resume,
  and mirrored into ``analytics.db`` for cross-session reporting.
- :class:`ChannelSpend` is the in-memory fold: highest ``rev`` per
  ``record_id`` wins, so a quote-to-settled upgrade is a plain revision bump and
  a fork that copies the journal dedups by construction.
- ``amount_micro is None`` means UNKNOWN, never zero: "we could not size this"
  and "this cost nothing" are different facts and only one of them may be
  rendered as a number (the rule ``spend.py`` established for tokens).
- :func:`combine` is the ONE place that turns the fold plus the session's
  inference figures into the published ``spend_channels`` wire object. Every
  surface reads that object; nothing sums locally (design §5.1). It returns
  plain JSON-shaped data so it is testable without a session.

The module is a LEAF, like ``spend.py``: it imports nothing from the session
package at module level, so tools can import it for the record shape without
dragging the harness in. Knowledge values reuse ``CostKnowledge``'s four
spellings (``unknown|partial|floor|exact``) as plain strings for the same
reason.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

#: Transcript custom type carrying one channel record per row (plus a one-time
#: ``{"kind": "start"}`` marker). A member of the transcript's
#: ``BOOKKEEPING_CUSTOM_TYPES``: bookkeeping about the session, never LLM
#: context, and ``preserve_mtime`` holds so a record write cannot keep a
#: dormant session ranked as freshly worked.
CHANNEL_SPEND_CUSTOM_TYPE = "session_channel_spend.v1"

#: Schema version of one record's ``details`` payload. A reader that does not
#: recognise the version degrades to "no record" rather than inventing one.
CHANNEL_SPEND_VERSION = 1

#: Version stamped on the ``spend_channels`` wire object (design §5.1). The
#: UI's gate is ``features.cost_channels >= 1``; this field is for the object's
#: own evolution, where an unknown version means "render nothing".
CHANNEL_SPEND_WIRE_VERSION = 1

#: Knowledge spellings, worst first. The ORDER is load-bearing: it is the
#: precedence ``SessionSpend.knowledge`` already uses (an unpriced call makes
#: the total a lower bound; a recovered floor is still a bound; only a sum with
#: nothing missing is exact), and it is applied by :func:`combine` so a session
#: whose inference half is partial cannot be presented as exact on the strength
#: of its channel half.
_KNOWLEDGE_ORDER = ("unknown", "partial", "floor", "exact")

#: Billing bases. ``subscription_api_equivalent`` is the subscription path: the
#: published API price of what a plan funded — labelled so it is never read as
#: cash, and kept in its own ``by_basis`` bucket beside ``billed`` (operator
#: rule: subscription paths carry API-equivalent dollars, separable from the
#: money that was actually charged).
BASIS_BILLED = "billed"
BASIS_SUBSCRIPTION = "subscription_api_equivalent"
BASIS_ESTIMATED = "estimated"
BASIS_NOT_TRACKED = "not_tracked"
BASIS_VALUES = frozenset({BASIS_BILLED, BASIS_SUBSCRIPTION, BASIS_ESTIMATED, BASIS_NOT_TRACKED})

#: Wire-only ``by_basis`` key: the micro-USD amount whose billing BASIS is not
#: tracked yet — the session's OWN inference (its route→basis mapping is PR-3)
#: plus the children bundle (relayed child records carry their own bases in
#: their own session's object; the children block here carries a total). It is
#: not a record-level basis, so no ``ChannelSpendRecord`` may carry it — that
#: keeps this bucket out of the per-record loop and makes the bucket sum
#: reconcile by construction: ``billed + subscription + estimated +
#: not_tracked_micro == total_micro`` once unknown-amount rows (which
#: contribute 0) are counted separately as ``not_tracked_calls`` (design round
#: 1, D1). ADDITIVE on the v1 wire: an older producer omits the key and a
#: reader must treat absence as 0 — the UI round-2 request that named this
#: field was exactly "the backend must publish the amount, a UI must never
#: re-sum inference rows to find it".
NOT_TRACKED_MICRO = "not_tracked_micro"

#: Where an amount (or its documented absence) comes from. ``server_reported``
#: is the first-party server's own figure (Radient); ``provider_reported`` is a
#: third-party provider's receipt; ``catalogue`` is a labelled client-side or
#: server-supplied price table; ``none`` means nothing stated it.
COST_SOURCE_SERVER = "server_reported"
COST_SOURCE_PROVIDER = "provider_reported"
COST_SOURCE_CATALOGUE = "catalogue"
COST_SOURCE_NONE = "none"
COST_SOURCE_VALUES = frozenset(
    {COST_SOURCE_SERVER, COST_SOURCE_PROVIDER, COST_SOURCE_CATALOGUE, COST_SOURCE_NONE}
)

#: The closed channel set. ``other`` carries a free-text ``detail`` so a new
#: metered channel never needs a schema change; ``inference`` is FOLD-ONLY (the
#: token ledger presents itself as this channel; token money is never
#: re-recorded as channel rows, which is what keeps the two sums from
#: overlapping).
CHANNEL_INFERENCE = "inference"
CHANNELS = frozenset(
    {
        CHANNEL_INFERENCE,
        "image",
        "tts",
        "stt",
        "search",
        "read",
        "classification",
        "other",
    }
)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_VALUES = frozenset({STATUS_OK, STATUS_FAILED, STATUS_CANCELLED})

#: ``price_version`` for the two known client-side rate tables (web search and
#: web read). These are the labelled exceptions to the no-rate-tables rule
#: (design §4.3): the figure IS an estimate, the id says which table produced
#: it, and ``# MIGRATE to catalogue`` marks them for the server-prices pass.
SEARCH_PRICE_VERSION = "client-search-table-2026-09"  # MIGRATE to catalogue

#: The web-search price table's version label. A constant stays because the
#: catalogue document does not exist yet (design §4.3); the ``price_version``
#: makes the estimate visible instead of hidden, and the marker comments below
#: are the migration reminder. ``legacy`` is what the transcript backfill
#: stamps on rows recovered from pre-feature journals, so a reader can tell a
#: live estimate from a recovered one.
WEB_SEARCH_PRICE_VERSION = "client-search-table-2026-09"  # MIGRATE to catalogue
WEB_SEARCH_PRICE_VERSION_LEGACY = "client-search-table-legacy"  # MIGRATE to catalogue


def usd_to_micro(value: Any) -> int | None:
    """``value`` as integer micro-USD, or ``None`` when it is not a number.

    Micro-USD (USD x 1e6) is :class:`SessionSpend`'s own convention, chosen
    there so a sum is exact; a float dollar amount reintroduces the rounding
    the integer exists to remove. ``bool`` is rejected on purpose (``True`` is
    an ``int`` in Python and a malformed payload must not become 1 micro-USD);
    strings are accepted only through ``Decimal`` so a JSON string like
    ``"0.0025"`` — the shape the Radient contract draft allows for amount
    fields — parses exactly rather than through a float.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value * 1_000_000
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
            return None
        return int(round(value * 1_000_000))
    if isinstance(value, str):
        try:
            return int(round(Decimal(value.strip()) * 1_000_000))
        except (InvalidOperation, ValueError, ArithmeticError):
            return None
    return None


def normalise_basis(value: Any) -> str | None:
    """A wire spelling of a billing basis, or ``None`` when unrecognised.

    The wave-2 image lane spells the subscription basis
    ``subscription-api-equivalent`` (hyphens, because it is a provider-facing
    label); the wire here is underscored (design §3.1). Accepting both at one
    boundary means no caller has to remember which side of the line it is on.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().replace("-", "_")
    return candidate if candidate in BASIS_VALUES else None


def map_image_cost_labels(
    route: str,
    cost_source: Any,
    billing_basis: Any,
    cost_provenance: Any,
    *,
    has_amount: bool,
) -> tuple[str, str, str]:
    """``(billing_basis, cost_source, price_version)`` for one image result.

    ONE mapping from the wave-2 labels (``imagegen``'s ``CostSource`` /
    ``BillingBasis`` / ``cost_provenance``, which may each be absent on an older
    tree) to the ledger's vocabulary, so the live emission and the transcript
    backfill cannot disagree about what a row means.

    The unlabelled fallbacks are deliberate and dated:

    - Radient's generate-time figure is a QUOTE (settlement is lazy and
      poll-driven; agent-server answers the quoted price at submit), so a
      radient amount with no explicit basis is ``estimated`` and
      ``server_reported``.
    - Every other route that reports a figure reports the provider's own
      per-request charge (OpenRouter ``usage.cost``, xAI ticks), so those are
      ``billed`` and ``provider_reported``.
    - No amount at all => ``not_tracked`` — never a fabricated zero.
    """
    source_label = str(cost_source).strip() if isinstance(cost_source, str) else ""
    basis = normalise_basis(billing_basis)
    if basis is None:
        if not has_amount:
            basis = BASIS_NOT_TRACKED
        elif source_label == "subscription":
            basis = BASIS_SUBSCRIPTION
        elif route == "radient":
            # Radient's generate-time figure is a QUOTE (the settled charge is
            # lazy and poll-driven), so an unlabelled amount from this route is
            # an estimate, never a billed charge.
            basis = BASIS_ESTIMATED
        elif source_label == "reported":
            basis = BASIS_BILLED
        else:
            basis = BASIS_ESTIMATED
    # ``reported`` on Radient is the SERVER's own figure; on the other
    # reporting routes it is the provider's; a rate table, a plan quota or an
    # already-normalised subscription basis means the number came from a
    # catalogue (the client's labelled table or the vendor document); anything
    # else carries no source at all.
    if route == "radient":
        source = COST_SOURCE_SERVER
    elif source_label == "reported":
        source = COST_SOURCE_PROVIDER
    elif source_label in ("rate_table", "subscription") or basis == BASIS_SUBSCRIPTION:
        source = COST_SOURCE_CATALOGUE
    elif route in ("openrouter", "xai"):
        source = COST_SOURCE_PROVIDER
    else:
        source = COST_SOURCE_NONE
    price_version = str(cost_provenance).strip() if isinstance(cost_provenance, str) else ""
    return basis, source, price_version


def new_record_id(channel: str, correlation: str = "") -> str:
    """``"<channel>:<correlation-or-uuid4>"`` — the ledger's idempotency key.

    A correlation id makes the record idempotent across a retry and lets the
    (future) server-side usage reconciliation match it; without one, a fresh
    uuid4 keeps two identical-looking generations two records. The prefix is
    the channel rather than the session so a log line or a ``channel_calls``
    row names what the money bought without a join.
    """
    key = (correlation or "").strip() or uuid.uuid4().hex
    return f"{channel}:{key}"


@dataclass(frozen=True)
class ChannelSpendRecord:
    """ONE metered money event outside the token path.

    Frozen because the fold keeps these by identity and a mutation would move
    money without an append; a correction is a higher ``rev`` for the same
    ``record_id``.

    ``amount_micro is None`` means the amount is UNKNOWN. ``status`` carries
    whether the event succeeded, failed or was cancelled — a failed event with
    no figure is not automatically "unknown money": the design's presumption
    (documented on the row, never hidden) is that an unreported failure was not
    charged, so it does not degrade the session's knowledge state, while a
    CANCELLED event may still be charged and therefore does.
    """

    record_id: str
    rev: int = 0
    ts_ms: int = 0
    session_id: str = ""
    parent_session_id: str = ""
    channel: str = "other"
    provider: str = ""
    model: str = ""
    units: float = 0.0
    unit: str = ""
    amount_micro: int | None = None
    billing_basis: str = BASIS_ESTIMATED
    cost_source: str = COST_SOURCE_NONE
    price_version: str = ""
    status: str = STATUS_OK
    #: Kept when a later ``rev`` replaced a quote, so the audit trail still
    #: shows what was originally claimed (design §3.1).
    quote_micro: int | None = None
    request_id: str = ""
    #: Free text, used by ``channel="other"`` and by rows whose knowledge
    #: presumption needs a word (a failed-unbilled job, a cancel awaiting
    #: settlement).
    detail: str = ""

    def __post_init__(self) -> None:
        """Normalise the label that describes an ABSENT amount.

        ``billing_basis`` is what the amount MEANS in money terms, so with no
        amount there is nothing to mean and ``not_tracked`` is the only honest
        value — a record that says ``estimated`` beside ``amount_micro=None``
        would let a UI read a basis where no figure exists ("est. $—").
        Normalised here rather than at each call site so the invariant holds
        for every constructor path, including the tools.
        """
        if self.amount_micro is None and self.billing_basis != BASIS_NOT_TRACKED:
            object.__setattr__(self, "billing_basis", BASIS_NOT_TRACKED)
        if not math.isfinite(float(self.units)):
            # A non-finite count cannot be summed, serialised (``Infinity`` is
            # not JSON) or rendered; it reads as NO count, which is exactly what
            # ``units == 0`` with an empty unit already means. Every constructor
            # path is covered here, so no emitter has to remember (review m3).
            object.__setattr__(self, "units", 0.0)

    def analytics_row(self) -> tuple[Any, ...]:
        """The recorder queue's primitive row for this record.

        Store insert order MINUS the timestamp the store stamps itself, so the
        producer (the session, which has no analytics import and reaches the
        writer through a queue) has to flatten to primitives exactly once, in
        one place; a test pins the tuple against the store's column list.
        """
        return (
            self.record_id,
            int(self.rev),
            int(self.ts_ms),
            self.session_id,
            self.parent_session_id,
            self.channel,
            self.provider,
            self.model,
            float(self.units),
            self.unit,
            self.amount_micro,
            self.billing_basis,
            self.cost_source,
            self.price_version,
            self.status,
            self.request_id,
        )

    def to_details(self) -> dict[str, Any]:
        """The transcript row's ``details`` payload."""
        details: dict[str, Any] = {
            "version": CHANNEL_SPEND_VERSION,
            "record_id": self.record_id,
            "rev": int(self.rev),
            "ts_ms": int(self.ts_ms),
            "session_id": self.session_id,
            "channel": self.channel,
            "provider": self.provider,
            "model": self.model,
            "units": float(self.units),
            "unit": self.unit,
            "amount_micro": self.amount_micro,
            "billing_basis": self.billing_basis,
            "cost_source": self.cost_source,
            "price_version": self.price_version,
            "status": self.status,
        }
        if self.parent_session_id:
            details["parent_session_id"] = self.parent_session_id
        if self.quote_micro is not None:
            details["quote_micro"] = self.quote_micro
        if self.request_id:
            details["request_id"] = self.request_id
        if self.detail:
            details["detail"] = self.detail
        return details

    @classmethod
    def from_details(cls, details: Any) -> ChannelSpendRecord | None:
        """Recall a record, or ``None`` when the payload is unusable.

        ``None`` — not a zero — for a missing, malformed or unknown-version
        row: a caller must be able to tell "no record" from "recorded zero",
        because only the second is a claim about the money.
        """
        if not isinstance(details, Mapping):
            return None
        # ``version`` is a small INTEGER, compared directly: the money
        # converter's contract is USD amounts, and routing a version through it
        # would invite exactly the "5 means five dollars" confusion this
        # module's ``None != 0`` rule exists to prevent.
        version = details.get("version")
        if isinstance(version, bool) or version != CHANNEL_SPEND_VERSION:
            return None
        record_id = details.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            return None
        rev = details.get("rev", 0)
        ts_ms = details.get("ts_ms", 0)
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
            return None
        if isinstance(ts_ms, bool) or not isinstance(ts_ms, int | float) or ts_ms < 0:
            return None
        # ``float('nan')`` passes both the type and the ``< 0`` test and then
        # RAISES in ``int()``; infinity raises too. A non-finite timestamp is a
        # corrupt row, and a corrupt row reads as "no record" (review m3).
        if isinstance(ts_ms, float) and not math.isfinite(ts_ms):
            return None
        amount = details.get("amount_micro")
        if amount is not None:
            if isinstance(amount, bool) or not isinstance(amount, int | float):
                return None
            # A JSON ``NaN``/``Infinity`` parses to a float and ``int()`` on it
            # RAISES: a corrupt row must read as "no record", never take the
            # session open down with it (review round 1, MINOR 1).
            if isinstance(amount, float) and not math.isfinite(amount):
                return None
            amount = int(amount)
        units = details.get("units", 0.0)
        if isinstance(units, bool) or not isinstance(units, int | float):
            return None
        if isinstance(units, float) and not math.isfinite(units):
            # Infinity would ride the wire as a bare ``Infinity`` token (invalid
            # JSON) and overflow every sum it joins; a non-finite count is not a
            # count (review m3).
            return None
        basis = normalise_basis(details.get("billing_basis")) or BASIS_NOT_TRACKED
        source = str(details.get("cost_source", "") or "")
        if source not in COST_SOURCE_VALUES:
            source = COST_SOURCE_NONE
        status = str(details.get("status", "") or "")
        if status not in STATUS_VALUES:
            status = STATUS_OK
        channel = str(details.get("channel", "") or "")
        if channel not in CHANNELS:
            channel = "other"
        quote = details.get("quote_micro")
        if quote is not None and not isinstance(quote, int):
            quote = None
        return cls(
            record_id=record_id,
            rev=int(rev),
            ts_ms=int(ts_ms),
            session_id=str(details.get("session_id", "") or ""),
            parent_session_id=str(details.get("parent_session_id", "") or ""),
            channel=channel,
            provider=str(details.get("provider", "") or ""),
            model=str(details.get("model", "") or ""),
            units=float(units),
            unit=str(details.get("unit", "") or ""),
            amount_micro=amount,
            billing_basis=basis,
            cost_source=source,
            price_version=str(details.get("price_version", "") or ""),
            status=status,
            quote_micro=quote,
            request_id=str(details.get("request_id", "") or ""),
            detail=str(details.get("detail", "") or ""),
        )


def now_ms() -> int:
    """Wall-clock epoch milliseconds — the ledger's timestamp unit."""
    return int(time.time() * 1000)


class ChannelSpend:
    """The folded view over one session's channel records.

    Keyed by ``record_id``; a higher ``rev`` supersedes the earlier record, and
    any lower-or-equal one is ignored. That is the whole idempotency story: the
    transcript may re-append the same row after a crash, a fork may carry rows
    its parent also has, and a retried backfill may walk the same journal
    twice — the fold cannot double count any of them.
    """

    def __init__(self) -> None:
        self._records: dict[str, ChannelSpendRecord] = {}

    def apply(self, record: ChannelSpendRecord) -> bool:
        """Fold one record in. Returns True when the fold CHANGED."""
        current = self._records.get(record.record_id)
        if current is not None and record.rev <= current.rev:
            return False
        self._records[record.record_id] = record
        return True

    def has(self, record_id: str) -> bool:
        return record_id in self._records

    def get(self, record_id: str) -> ChannelSpendRecord | None:
        return self._records.get(record_id)

    def rows(self) -> list[ChannelSpendRecord]:
        """Every folded record, oldest first (``ts_ms``, then id for stability).

        Order matters to the transcript append path and to tests: a fold that
        iterated a dict's insertion order would let a revision that arrived in
        a different sequence paint rows in a different order on two surfaces.
        """
        return sorted(self._records.values(), key=lambda rec: (rec.ts_ms, rec.record_id))

    def total_known_micro(self) -> int:
        """The sum of STATED amounts only. Unknowns contribute nothing here.

        Never render this alone: it is a lower bound whenever any record is
        unknown, which is exactly what :func:`combine`'s knowledge state says.
        """
        return sum(
            rec.amount_micro for rec in self._records.values() if rec.amount_micro is not None
        )

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class InferenceSnapshot:
    """The session's token-path money, as the fold needs to see it.

    A snapshot rather than the ``SessionSpend`` object so :func:`combine` is
    testable without a session — and so the one rule that inference is
    fold-only (never re-appended as channel rows) cannot be violated by a
    caller that reached for the accumulator itself.
    """

    micro: int = 0
    calls: int = 0
    priced_calls: int = 0
    knowledge: str = "unknown"
    #: ``SessionSpend.by_identity`` — per serving identity. ``None`` means the
    #: record predates the key (or the host cannot state one): the wire then
    #: carries ONE inference row with an empty provider/model, and the UI reads
    #: that as "by model: not tracked" rather than inventing a breakdown.
    by_identity: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class ChildrenSnapshot:
    """The subagent/fork contribution already rolled into the session's totals."""

    total_micro: int = 0
    knowledge: str = "exact"
    #: Wire-visible reason for a degraded ``knowledge`` — the resumed-parent
    #: case: children that ran in an earlier process are not re-readable, so
    #: the block may not present their spend as fully accounted (review m4/Q9).
    reason: str = ""


def _inference_rows(inference: InferenceSnapshot) -> list[dict[str, Any]]:
    """One wire row per serving identity, plus an UNATTRIBUTED row if needed.

    The unattributed row is the turn-end remainder's home (``SessionSpend``
    documents that a remainder is money with no call), the ``other`` bucket's
    home once the identity cap folds entries, and the single row an old record
    (no ``by_identity``) gets. Without it the rows would not sum to the
    inference total, which is the kind of quiet disagreement between the parts
    and the whole this project exists to remove.
    """
    if inference.micro == 0 and inference.calls == 0:
        return []
    rows: list[dict[str, Any]] = []
    attributed_micro = 0
    attributed_calls = 0
    entries = inference.by_identity or {}
    for key, entry in entries.items():
        if not isinstance(entry, Mapping):
            continue
        micro = int(entry.get("micro", 0) or 0)
        calls = int(entry.get("calls", 0) or 0)
        unpriced = int(entry.get("unpriced", 0) or 0)
        attributed_micro += micro
        attributed_calls += calls
        if calls == 0 and micro == 0 and unpriced == 0:
            continue
        if unpriced and micro == 0:
            row_knowledge = "unknown"
        elif unpriced:
            row_knowledge = "partial"
        else:
            row_knowledge = "exact"
        rows.append(
            {
                "channel": CHANNEL_INFERENCE,
                "provider": str(entry.get("provider", "") or ""),
                # ``model_id`` is the key ``SessionSpend`` itself stores (its
                # identity dicts are the wire's ``{provider, model_id}`` pair);
                # ``model`` is accepted too so a future writer spelling it that
                # way cannot silently lose the model half of the identity.
                "model": str(entry.get("model_id", entry.get("model", "")) or ""),
                "label": str(key),
                "units": float(calls),
                "unit": "calls",
                "amount_micro": micro,
                "knowledge": row_knowledge,
                "basis": [BASIS_NOT_TRACKED],
                "price_versions": [],
            }
        )
    if not rows and (inference.micro or inference.calls):
        # No by-identity breakdown: state the total and say so, rather than
        # dropping the money from the rows or splitting it at random.
        rows.append(
            {
                "channel": CHANNEL_INFERENCE,
                "provider": "",
                "model": "",
                "label": "",
                "units": float(inference.calls),
                "unit": "calls",
                "amount_micro": inference.micro,
                "knowledge": (
                    "unknown"
                    if inference.micro == 0 and inference.calls and not inference.priced_calls
                    else inference.knowledge
                ),
                "basis": [BASIS_NOT_TRACKED],
                "price_versions": [],
            }
        )
        return rows
    remainder_micro = inference.micro - attributed_micro
    remainder_calls = inference.calls - attributed_calls
    if remainder_micro or remainder_calls:
        rows.append(
            {
                "channel": CHANNEL_INFERENCE,
                "provider": "",
                "model": "",
                "label": "unattributed",
                "units": float(max(remainder_calls, 0)),
                "unit": "calls",
                "amount_micro": remainder_micro,
                "knowledge": "exact" if not remainder_calls else "partial",
                "basis": [BASIS_NOT_TRACKED],
                "price_versions": [],
            }
        )
    return rows


#: Display order of channels: the hierarchy the design note states
#: (inference → image → tts → stt → search → read → classification → other).
#: The panel sorts wire rows by this; keeping it here means every surface
#: that groups by channel ages its rows the same way (design round 1, D9).
CHANNEL_ORDER = (
    CHANNEL_INFERENCE,
    "image",
    "tts",
    "stt",
    "search",
    "read",
    "classification",
    "other",
)


def _channel_rank(channel: str) -> int:
    """The display rank of a channel; unknown channels sort last, then by name."""
    try:
        return CHANNEL_ORDER.index(channel)
    except ValueError:
        return len(CHANNEL_ORDER)


def _channel_rows(records: Sequence[ChannelSpendRecord]) -> list[dict[str, Any]]:
    """One wire row per ``(channel, provider, model, unit)`` group.

    Grouping happens HERE rather than in the UI so every surface groups
    identically and the micro-USD sums are computed once, in exact integers.
    """
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record.channel, record.provider, record.model, record.unit)
        group = groups.setdefault(
            key,
            {
                "channel": record.channel,
                "provider": record.provider,
                "model": record.model,
                "label": "",
                "units": 0.0,
                "unit": record.unit,
                "amount_micro": 0,
                "knowledge": "exact",
                "basis": set(),
                "price_versions": set(),
                "has_amount": False,
                "has_unstated": False,
                #: False until some record states a unit count. A legacy row
                #: recovered without one writes 0 units, and 0 beside a real
                #: dollar figure is the fabricated-zero defect this module
                #: refuses for money — so the row emits ``units: None`` (the
                #: panel omits the note) unless a count was actually recorded
                #: (QA round 1, Q6).
                "has_units": False,
            },
        )
        if record.units and record.units > 0:
            group["units"] = float(group["units"]) + float(record.units)
            group["has_units"] = True
        if record.amount_micro is not None:
            group["amount_micro"] = int(group["amount_micro"]) + int(record.amount_micro)
            group["has_amount"] = True
        elif record.status != STATUS_FAILED:
            # A failed job with no figure is the documented presumption of no
            # charge; anything else with no figure is money we cannot state.
            group["has_unstated"] = True
        group["basis"].add(record.billing_basis)
        if record.price_version:
            group["price_versions"].add(record.price_version)
    rows: list[dict[str, Any]] = []
    for group in groups.values():
        if group["has_unstated"]:
            group["knowledge"] = "partial" if group["has_amount"] else "unknown"
        else:
            group["knowledge"] = "exact"
        rows.append(
            {
                "channel": group["channel"],
                "provider": group["provider"],
                "model": group["model"],
                "label": group["label"],
                # None when no contributing record carried a count: "units
                # unknown" and "zero units" are as different as the money rule
                # that made this module refuse fabricated zeros.
                "units": group["units"] if group["has_units"] else None,
                "unit": group["unit"],
                # None, never 0, when nothing in the group was sized: on the wire
                # an unknown amount and a free call are different facts.
                "amount_micro": group["amount_micro"] if group["has_amount"] else None,
                "knowledge": group["knowledge"],
                "basis": sorted(group["basis"]),
                "price_versions": sorted(group["price_versions"]),
            }
        )
    rows.sort(
        key=lambda row: (
            _channel_rank(str(row["channel"])),
            str(row["provider"]),
            str(row["model"]),
            str(row["unit"]),
        )
    )
    return rows


def combine(
    records: Sequence[ChannelSpendRecord],
    *,
    inference: InferenceSnapshot | None = None,
    children: ChildrenSnapshot | None = None,
    child_records: Sequence[ChannelSpendRecord] = (),
    tracked: bool = False,
    lost: bool = False,
) -> dict[str, Any]:
    """The ONE derivation of the ``spend_channels`` wire object (design §5.1).

    ``total_micro`` is the GRAND total — the session's own inference, every
    channel record, and the subagent/fork contribution — so a surface that
    renders the total and the rows never sees them disagree, and a surface that
    renders ONLY the total (the status band) needs nothing else.

    ``child_records`` are channel records RELAYED from live child sessions
    (design §5.1: the parent total includes the child's records once, deduped
    by ``record_id``). They are NOT mixed into ``rows`` — the children block is
    their home, one line on the panel beside the session's own rows — and they
    can never double count: a child's own object carries its records under the
    child's session, and the parent only ever sees each ``record_id`` once.

    Knowledge precedence is ``unknown > partial > floor > exact``:

    - ``unknown``: nothing anywhere is stateable — no priced call and no stated
      amount, matching ``SessionSpend.knowledge``'s ``$—`` state. A session
      with no money at all also lands here rather than claiming an exact zero.
    - ``partial``: some money is known and some is not — an unreported
      successful call, an unsettled cancel, an unpriced model call, a child
      ledger with unknowns, or channel rows recovered from a session with no
      ``start`` marker (we have what we have and cannot know what we missed).
      Note the LAST clause's shape: a marker-less session degrades only when
      the journal actually CARRIES recovered channel rows (or a lost-money
      row). A pre-feature session with no recovered rows is not marked: its
      figure is the best evidence there is, and the operator's continuity
      tests (cold and in-process must spell one journal identically) settled
      that reading over degrading most of the store (review round 2, M-4).
    - ``floor``: rows were POSITIVELY reported lost (``lost_money_rows``) or a
      component's own knowledge is a floor.
    - ``exact``: everything that spent money has a stated figure.
    """
    records = list(records)
    child_records = list(child_records)
    inference = inference or InferenceSnapshot()
    children = children or ChildrenSnapshot()

    stated_channel_micro = sum(rec.amount_micro for rec in records if rec.amount_micro is not None)
    child_stated_micro = sum(
        rec.amount_micro for rec in child_records if rec.amount_micro is not None
    )
    children_total_micro = int(children.total_micro) + child_stated_micro
    total_micro = inference.micro + stated_channel_micro + children_total_micro

    unstated_records = [
        rec for rec in records if rec.amount_micro is None and rec.status != STATUS_FAILED
    ]
    # A relayed child record with no stated amount is money we know exists and
    # cannot size: the children block says partial rather than letting the
    # parent present an exact total over an undercount (QA round 1, Q2).
    children_knowledge = children.knowledge
    if any(rec.amount_micro is None and rec.status != STATUS_FAILED for rec in child_records):
        children_knowledge = "partial"
    inference_unstated = inference.knowledge == "unknown" and inference.calls > 0
    inference_partial = inference.knowledge == "partial"
    # DEGRADATION NEEDS EVIDENCE (see the docstring): marker-less sessions that
    # carry recovered rows (or a lost-money row) are lower bounds; a marker-less
    # session with no channel evidence at all keeps its own figure.
    untracked_with_evidence = not tracked and (bool(records) or lost)

    partial = bool(
        unstated_records
        or inference_unstated
        or inference_partial
        or untracked_with_evidence
        or children_knowledge == "partial"
        or children_knowledge == "unknown"
    )
    floor = bool(lost or inference.knowledge == "floor" or children.knowledge == "floor")
    nothing_stated = (
        not any(rec.amount_micro is not None for rec in records)
        and inference.priced_calls == 0
        and children.total_micro == 0
    )
    if total_micro == 0 and nothing_stated:
        knowledge = "unknown"
    elif partial:
        knowledge = "partial"
    elif floor:
        knowledge = "floor"
    else:
        knowledge = "exact"

    by_basis = {
        BASIS_BILLED: 0,
        BASIS_SUBSCRIPTION: 0,
        BASIS_ESTIMATED: 0,
        # Money whose basis is not recorded: this session's own inference (its
        # route→basis mapping is PR-3) plus the whole children bundle. This is
        # what makes the buckets sum to ``total_micro`` on every surface
        # (design round 1, D1): billed + subscription + estimated +
        # not_tracked_micro == total minus the unknown-amount rows, whose
        # money is 0 by definition and whose COUNT is stated separately.
        NOT_TRACKED_MICRO: 0,
        # A COUNT, not money: how many records have no trackable money basis.
        # Numeric because the UI renders one basis line; the key says what the
        # number is (design §5.1).
        "not_tracked_calls": 0,
    }
    record_not_tracked_micro = 0
    for record in records:
        if record.status == STATUS_FAILED and record.amount_micro is None:
            continue  # documented presumption: an unreported failure was not charged
        if record.amount_micro is None:
            # No stated amount: the money is 0 by definition and only the COUNT
            # is stateable.
            by_basis["not_tracked_calls"] += 1
            continue
        if record.billing_basis == BASIS_NOT_TRACKED or record.billing_basis not in (
            BASIS_BILLED,
            BASIS_SUBSCRIPTION,
            BASIS_ESTIMATED,
        ):
            # A STATED amount whose basis is not tracked — or is not even in
            # the vocabulary (a foreign string from an older/other producer):
            # its money must still land in a bucket or the buckets stop
            # summing to the total (review m1 / QA Q10), and it must not write
            # a fifth bucket key (round-3 review, R3-5). ``not_tracked_micro``
            # is its home — the same bucket the inference remainder uses — NOT
            # ``not_tracked_calls``, which counts rows with NO amount.
            record_not_tracked_micro += int(record.amount_micro)
            continue
        by_basis[record.billing_basis] += int(record.amount_micro)
    by_basis[NOT_TRACKED_MICRO] = (
        max(0, int(inference.micro)) + max(0, children_total_micro) + record_not_tracked_micro
    )

    rows = _inference_rows(inference) + _channel_rows(records)
    return {
        "version": CHANNEL_SPEND_WIRE_VERSION,
        "tracked": bool(tracked),
        "total_micro": int(total_micro),
        "knowledge": knowledge,
        "by_basis": by_basis,
        "rows": rows,
        "children": {
            "total_micro": int(children_total_micro),
            "knowledge": children_knowledge,
            # The user-facing reason when the children block cannot vouch for
            # itself (a resumed parent whose earlier children are not
            # re-readable — review m4/Q9). Absent when there is nothing to say.
            **({"reason": children.reason} if children.reason else {}),
        },
    }


def web_search_record(*, channel: str, provider: str, usd: float | None) -> ChannelSpendRecord:
    """One search/read record from the search ledger's own estimate (design §4.1).

    ``estimated`` with the client table's OWN id in ``price_version`` — the
    labelled exception to the no-rate-tables rule, so the figure is visible as
    an estimate and can be migrated to a server catalogue without a wire
    change. ``usd is None`` (an unpriced search) stays ``amount=None``, never
    0: an unpriced operation is not a free one.
    """
    return ChannelSpendRecord(
        record_id=new_record_id(channel),
        ts_ms=now_ms(),
        channel=channel,
        provider=provider,
        units=1,
        unit="searches" if channel == "search" else "reads",
        amount_micro=usd_to_micro(usd),
        billing_basis=BASIS_ESTIMATED,
        cost_source=COST_SOURCE_CATALOGUE,
        price_version=SEARCH_PRICE_VERSION,
        status=STATUS_OK,
    )


def emit_web_spend(callback: Any, *, channel: str, provider: str, usd: float | None) -> None:
    """Build and hand over ONE search/read record, best-effort.

    Called by the two web tools with the callback off their ``ToolContext``;
    ``None`` is the documented "this host does not track channels" value and
    skips QUIETLY. Never raises: a lost spend row must not fail a search.
    """
    if not callable(callback):
        return
    try:
        callback(web_search_record(channel=channel, provider=provider, usd=usd))
    except Exception:  # noqa: BLE001 — a lost spend row is not a failed search
        logger.debug("web channel-spend emission failed", exc_info=True)


def records_from_details(rows: Sequence[Mapping[str, Any]]) -> list[ChannelSpendRecord]:
    """Fold a transcript reader's ``details`` rows into records, dropping junk.

    PER ROW, deliberately: the caller (``Session.__init__``) also guards the
    whole call, and a whole-call guard alone loses every GOOD row beside one
    corrupt one — measured in review round 2 (m3): one ``NaN`` timestamp beside
    a healthy 8,000 µ$ row published ``total 0``. A row that cannot be recalled
    is skipped; its neighbours are not.
    """
    out: list[ChannelSpendRecord] = []
    for row in rows:
        try:
            record = ChannelSpendRecord.from_details(row)
        except Exception:  # noqa: BLE001 — a corrupt row is "no record"
            logger.debug("channel record row skipped", exc_info=True)
            continue
        if record is not None:
            out.append(record)
    return out


def fold_records(records: Sequence[ChannelSpendRecord]) -> ChannelSpend:
    """Build a fold from an unordered sequence (the transcript reader's output)."""
    fold = ChannelSpend()
    for record in records:
        fold.apply(record)
    return fold

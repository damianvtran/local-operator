"""The per-session spend accumulator: one exact number, computed once.

Why this module exists (see ``docs/design-session-spend-ledger.md``): the
session already owned a correct in-memory accumulator, but it was durable only
as a fat, UI-gated checkpoint (7.7% of the real store), so a resumed session
priced ONE restored provider reading and painted it as a FLOOR (``≥``) — a mark
that said "this is a lower bound" for a reason that had nothing to do with the
money. Measured over the operator's store, that one reading is a median 64x
below the session's own turn rows — and, because the store keeps growing, that
factor is a SNAPSHOT of one quantity rather than a constant: the same
measurement read 81.6x on the 2,053-session store the PR's census walked. Both
figures are reproducible with ``scripts/spend_ledger_probe.py --census``, and
the command is the authority rather than either number.

The fix is not a second store: it is this accumulator, made durable as one
small ``session_spend.v1`` transcript row and RECALLED in O(1) instead of
re-derived. Three properties are load-bearing, and each is why the code below
is shaped the way it is:

- **Integer micro-USD.** The sum must be exact, so the accumulator holds
  ``micro`` (USD x 1e6) as an ``int`` — the analytics ledger's own convention
  (``analytics/model.py``: "Integer so the aggregate SUM is exact"). Floats
  would drift over ten thousand accruals, and the drift would be invisible
  because the band rounds.
- **One arithmetic site.** Accruals arrive from the frontend store's per-call
  and turn-end branches and from detached leaf calls, but every addition
  happens in :meth:`SessionSpend.accrue` / :meth:`SessionSpend.adjust`. A
  second addition site is how a turn gets billed twice; see the design's §5.1.
- **Replacement state, never a delta.** The persisted row carries the running
  total, so a reader takes the newest row and never sums the file. Two writers
  therefore cannot double-bill — the same rule the frontend checkpoint already
  states (``session.py``: "The checkpoint is replacement state, never an
  additive delta, so takeover cannot double it").
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.session.frontend_state import CostKnowledge

logger = logging.getLogger(__name__)

#: Transcript custom type carrying the durable accumulator. Deliberately NOT in
#: ``session._PERSISTABLE_CUSTOM_TYPES``: this is bookkeeping about the session
#: and must never enter LLM context.
SESSION_SPEND_CUSTOM_TYPE = "session_spend.v1"

#: Schema version of the record's ``details``. A reader that does not recognise
#: the version degrades to "no record" — never to a confident zero, because zero
#: is the one reading that is certainly false for a session that spent money.
SESSION_SPEND_VERSION = 1

#: Identifies WHICH process wrote a record, so a foreign append to one session
#: directory (the attach invariant in ``transcript.py`` broken) is detectable
#: rather than silently overwriting this process's total.
_BOOT_ID = int(time.time())


def writer_stamp() -> str:
    """``"<pid>:<boot-second>"`` for the record's ``writer`` field."""
    return f"{os.getpid()}:{_BOOT_ID}"


def _as_int(value: Any) -> int | None:
    """``value`` as an ``int``, or ``None`` when it is not a plain number.

    ``bool`` is rejected on purpose: ``True`` is an ``int`` in Python, and a
    malformed row must not be able to contribute a silent ``1`` micro-dollar.
    Strings are rejected too — ``json`` writes a number as a number, so a quoted
    one is a malformed row, and coercing it would let a broken writer's
    ``"5"`` read as a confident five micro-dollars.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _identity_of(
    usage: Any, fallback_provider: str = "", fallback_model: str = ""
) -> dict[str, str]:
    """The serving identity for one call, for the record's ``last_identity``.

    The usage's own fields win: the failover layer stamps the model that
    ACTUALLY served onto ``Usage``, and reading the session's selected model
    instead would file a fallback call under the primary it never ran on.
    """
    provider = str(getattr(usage, "provider", "") or fallback_provider or "")
    model_id = str(getattr(usage, "model_id", "") or fallback_model or "")
    identity: dict[str, str] = {}
    if provider:
        identity["provider"] = provider
    if model_id:
        identity["model_id"] = model_id
    return identity


def serving_identity(usage: Any, effective_model: Any = None) -> dict[str, str]:
    """The serving identity for one call, falling back to the session's model.

    ONE rule, used by both the frontend store and the session: the usage's own
    stamp wins, and only an unstamped usage (a primary success, a direct call, a
    rehydrated row) is attributed to the effective model. Two spellings of this
    would put a fallback call's tokens on the primary's rates in one place and
    the right ones in the other.
    """
    return _identity_of(
        usage,
        str(getattr(effective_model, "provider", "") or ""),
        str(getattr(effective_model, "model_id", "") or ""),
    )


def usage_prices_known(usage: Any) -> bool:
    """Whether ``usage`` carries a provider receipt, so pricing cannot change.

    A receipt (``usd_cost``) is the provider's own bill and short-circuits both
    the paint resolver and the full one; a call that has one needs no off-loop
    re-pricing (design §5.2, step 1).
    """
    from local_operator.model.configure import _usage_cost

    return _usage_cost(usage) is not None


def price_call(provider: str, model_id: str, usage: Any) -> tuple[int, bool]:
    """``(micro-USD, known)`` for ONE provider call, priced at record time.

    Runs on a worker thread and calls the FULL ``resolve_model_info``, not the
    paint-safe one: measured on the operator's store, the paint resolver cannot
    price 6.86% of all rows at all, including 165 calls worth $58.51 in one
    session — 86% of its true cost (design §2.4). The provider's receipt wins
    when present; otherwise the table estimate does.

    The same order and the same pair :func:`analytics.model.price_snapshot`
    uses, and a test prices a fixture both ways and asserts equality: if these
    two drift, the band and ``/analytics`` report the same call differently,
    which is the defect class ``tui/costs.py`` exists to prevent.

    Never raises: an unpriceable call is not an error, and this runs in a task
    whose failure must not reach the turn.
    """
    try:
        from local_operator.model.configure import (
            _usage_cost,
            cost_for_usage,
            resolve_model_info,
        )

        reported = _usage_cost(usage)
        if reported is not None:
            return int(round(reported * 1_000_000)), True
        if not provider or not model_id:
            return 0, False
        info = resolve_model_info(provider, model_id)
        if not (info.input_price or info.output_price):
            return 0, False
        return int(round(cost_for_usage(provider, info, usage) * 1_000_000)), True
    except Exception:  # noqa: BLE001 — an unpriceable call is not an error
        return 0, False


def has_reported_tokens(usage: Any) -> bool:
    """Whether ``usage`` reports any tokens, i.e. the call really happened.

    A rebuild must count a token-carrying but unpriceable call as an UNPRICED
    call (it is evidence the session spent money we cannot name the size of),
    while a call with no tokens at all — an aborted or never-answered request —
    is not spend and contributes nothing to either count.
    """
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        value = (
            usage.get(field_name) if isinstance(usage, Mapping) else getattr(usage, field_name, 0)
        )
        if value:
            return True
    return False


def price_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, bool]]:
    """Price a whole rebuild's rows in ONE worker-thread hop.

    A single ``asyncio.to_thread`` for the batch rather than one per row: the
    resolver's cold-memo miss is the expensive part and it is per MODEL, not per
    row, so batching keeps the thread-pool traffic proportional to the session
    rather than to its call count, and it is what makes the structural claim
    testable (every price in a rebuild happens on a thread that is not the
    event loop's).

    Module-level and separately importable so a test can wrap it and record
    ``threading.get_ident()`` — the repo's established structural way to assert
    "this ran off the loop" (see ``test_store_maintenance_callbacks_run_off_the
    _event_loop_thread``) rather than a wall-clock bound.
    """
    return [
        price_call(
            str(row.get("provider", "") or ""),
            str(row.get("model_id", "") or ""),
            row,
        )
        for row in rows
    ]


@dataclass
class SessionSpend:
    """The session's own model-turn spend, as one exact recalled number.

    ``micro`` counts only dollars that were actually RECORDED — a provider
    receipt or a table price computed at the call boundary. Calls the resolver
    could not price are counted in ``unpriced_calls`` instead, because "we do
    not know" and "it was free" are different facts and only one of them may be
    painted as a number (design §5.4).
    """

    micro: int = 0
    calls: int = 0
    priced_calls: int = 0
    unpriced_calls: int = 0
    #: True only when the total is missing rows that no longer exist — a
    #: rebuilt sum over a journal carrying a compaction or prune marker. This is
    #: what ``≥`` is allowed to mean after this change, and nothing else.
    floor: bool = False
    #: True when this accumulator came from the one-time rebuild, so the record
    #: it writes is recognisable as reconstructed rather than live-accrued.
    rebuilt: bool = False
    writer: str = ""
    last_identity: dict[str, str] | None = None
    #: Per-call prices still eligible for an authoritative correction. In-memory
    #: only: a recalled record is already authoritative, and keeping the map
    #: across a resume would let a correction land against a call that this
    #: process never priced.
    _estimates: dict[int, int | None] = field(default_factory=dict, repr=False)
    _next_index: int = field(default=0, repr=False)

    # -- properties -------------------------------------------------------

    @property
    def usd(self) -> float:
        """``micro`` in whole dollars, for the wire fields that stay floats."""
        return self.micro / 1_000_000.0

    @property
    def has_calls(self) -> bool:
        return self.calls > 0

    # -- accrual ----------------------------------------------------------

    def accrue(self, micro: int | None, identity: Mapping[str, Any] | None = None) -> int:
        """Add one provider call. Returns a token for a later correction.

        ``micro is None`` (or a call the caller could not price) counts the call
        as UNPRICED, which is what makes the total an honest lower bound instead
        of a complete-looking sum that silently omitted it.
        """
        index = self._next_index
        self._next_index += 1
        self.calls += 1
        value = _as_int(micro) if micro is not None else None
        if value is None:
            self.unpriced_calls += 1
        else:
            self.micro += value
            self.priced_calls += 1
        self._estimates[index] = value
        if identity:
            provider = str(identity.get("provider", "") or "")
            model_id = str(identity.get("model_id", "") or "")
            if provider or model_id:
                self.last_identity = {
                    **({"provider": provider} if provider else {}),
                    **({"model_id": model_id} if model_id else {}),
                }
        return index

    def correct(self, index: int, micro: int | None) -> int:
        """Replace an accrued call's estimate with its authoritative price.

        The one-tick path: the band paints the paint-resolver's answer
        immediately, and this converges it to the price computed off-loop with
        the full resolver (design §5.2). Returns the DELTA the correction moved
        this total by, in micro-USD (0 for a no-op), so a caller can tell how
        much of the accumulator arrived as a re-price rather than as a new call:
        the front end's turn-end remainder measures the turn against the prices
        it already counted, and a correction it cannot see there is billed a
        second time (review R1-1). The delta stays truthy, so
        ``if spend.correct(...)`` keeps meaning "something changed".
        """
        if index not in self._estimates:
            return 0
        previous = self._estimates.pop(index)
        value = _as_int(micro) if micro is not None else None
        if value == previous:
            return 0
        if previous is None:
            if value is None:
                return 0
            self.unpriced_calls -= 1
            self.priced_calls += 1
            self.micro += value
            return value
        if value is None:
            self.priced_calls -= 1
            self.unpriced_calls += 1
            self.micro -= previous
            return -previous
        self.micro += value - previous
        return value - previous

    def adjust(self, delta_micro: int) -> bool:
        """Apply a turn-level remainder, which is not a call.

        ``FrontendStateStore`` reconciles a turn once at its end
        (``remainder = max(0, total - current_turn_accrued_cost)``) because the
        aggregate's price is not the sum of its calls' prices. That remainder is
        money, not a provider call, so it must not move ``calls`` — the count is
        what the knowledge state is derived from.
        """
        value = _as_int(delta_micro)
        if not value:
            return False
        self.micro += value
        return True

    # -- knowledge --------------------------------------------------------

    def knowledge(self) -> "CostKnowledge":
        """The ONE derivation of how much this figure is worth trusting.

        The precedence is the analytics panel's own rule, reused rather than
        re-invented: an unpriced call makes the total a lower bound (PARTIAL),
        and a sum over rows a compaction or prune removed can never be whole
        (FLOOR). UNKNOWN takes precedence over both because ``≥ unknown`` is a
        contradiction — a fully-unpriced session renders ``$—`` with no mark.
        """
        from local_operator.session.frontend_state import CostKnowledge

        if self.priced_calls == 0:
            return CostKnowledge.UNKNOWN
        if self.unpriced_calls:
            return CostKnowledge.PARTIAL
        if self.floor:
            return CostKnowledge.FLOOR
        return CostKnowledge.EXACT

    # -- persistence ------------------------------------------------------

    def to_details(self) -> dict[str, Any]:
        """The record's ``details`` payload (see the design's §5.3)."""
        details: dict[str, Any] = {
            "version": SESSION_SPEND_VERSION,
            "micro": int(self.micro),
            "calls": int(self.calls),
            "priced_calls": int(self.priced_calls),
            "unpriced_calls": int(self.unpriced_calls),
            "floor": bool(self.floor),
            "rebuilt": bool(self.rebuilt),
            "writer": self.writer or writer_stamp(),
        }
        if self.last_identity:
            details["last_identity"] = dict(self.last_identity)
        return details

    @classmethod
    def from_details(cls, details: Any) -> "SessionSpend | None":
        """Recall a record, or ``None`` when there is nothing trustworthy.

        ``None`` — not an empty accumulator — for a missing, malformed or
        unknown-version row: a caller must be able to tell "no record" from
        "recorded zero", because only the second is a claim about the money.
        """
        if not isinstance(details, Mapping):
            return None
        if _as_int(details.get("version")) != SESSION_SPEND_VERSION:
            return None
        micro = _as_int(details.get("micro", 0))
        calls = _as_int(details.get("calls", 0))
        priced = _as_int(details.get("priced_calls", 0))
        unpriced = _as_int(details.get("unpriced_calls", 0))
        if None in (micro, calls, priced, unpriced):
            return None
        if min(micro, calls, priced, unpriced) < 0:  # type: ignore[arg-type]
            return None
        # Every call is either priced or unpriced; a row that says otherwise is
        # internally inconsistent and is dropped rather than repaired into a
        # number this process would then present as fact.
        if priced + unpriced != calls:  # type: ignore[operator]
            return None
        last_identity = details.get("last_identity")
        return cls(
            micro=micro,  # type: ignore[arg-type]
            calls=calls,  # type: ignore[arg-type]
            priced_calls=priced,  # type: ignore[arg-type]
            unpriced_calls=unpriced,  # type: ignore[arg-type]
            floor=bool(details.get("floor", False)),
            rebuilt=bool(details.get("rebuilt", False)),
            writer=str(details.get("writer", "") or ""),
            last_identity=(
                {str(k): str(v) for k, v in last_identity.items()}
                if isinstance(last_identity, Mapping)
                else None
            ),
        )


def recall(transcript: Any) -> SessionSpend | None:
    """The durable accumulator this transcript carries, or ``None``.

    One dict lookup on the index the transcript's constructor already built —
    no scan, no re-parse, no re-price. That O(1) recall is the operator's
    requirement: a number to CONSULT rather than a recount to walk backward
    through.
    """
    if transcript is None:
        return None
    try:
        details = transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE)
    except Exception:  # noqa: BLE001 — a missing index is "no record", not an error
        return None
    return SessionSpend.from_details(details)

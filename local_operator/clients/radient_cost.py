"""The Radient ``cost`` object adapter: the ONE place that knows server field names.

Why this module exists: the Radient agent-server is adding a versioned ``cost``
object (draft, radientdev session 086f6187e0ba; minor naming changes still
possible), and speech already sends ``X-Radient-Cost-*`` headers. Every other
module in this tree sees :class:`~local_operator.session.channel_spend.ChannelSpendRecord`
and must never learn the server's spellings — so the parsing, the basis
mapping and the unit mapping live here, together, marked, and a contract change
is a change to this file plus its tests.

The rules this adapter keeps:

- ABSENCE IS ``None``, NEVER 0: a response with no ``cost`` object yields
  ``None`` from every reader, and the caller records an unknown amount
  (``not_tracked``) rather than a confident zero. A ``cost`` object with an
  explicit ``amount_usd`` of 0 is a KNOWN zero (that is the contract's failed
  shape) and passes through as ``amount_micro = 0``.
- A MISSING OR UNKNOWN ``v`` degrades to "parse defensively or answer None",
  never to a guess: an old server simply carries no object, and a future one
  must not be misread as v1 by accident. We accept v1 and unknown-versions
  alike for now, because the draft has not versioned a breaking change yet;
  the field is read and exposed so a caller could gate on it.
- BASIS IS MAPPED ONCE: the wire's ``quote`` becomes our ``estimated`` (a
  quote is never shown as a charge), ``billed`` stays ``billed``,
  ``estimated`` stays ``estimated``, and the wave-2 hyphen spelling
  ``subscription-api-equivalent`` normalises to our underscored one.

This module deliberately depends on nothing from the session package except
the record's own leaf module, so a PR-3 route can import it cheaply.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from local_operator.session.channel_spend import (
    BASIS_ESTIMATED,
    BASIS_NOT_TRACKED,
    ChannelSpendRecord,
    normalise_basis,
)

logger = logging.getLogger(__name__)

#: The header prefix speech uses for its cost fields (speech-only per the
#: draft: the media paths carry the object in the body).
HEADER_PREFIX = "x-radient-cost-"

#: Wire ``tool_type`` -> our channel vocabulary. ``other`` is the closed set's
#: escape hatch, and the raw value rides the record's ``detail`` so a new
#: metered tool type never needs a schema change to be visible.
_TOOL_TYPE_CHANNELS = {
    "image_generation": "image",
    "video_generation": "other",
    "speech": "tts",
    "transcription": "stt",
    "web_search": "search",
    "chat": "inference",
    "embeddings": "other",
    "decisions": "classification",
    "email": "other",
}

#: Wire ``unit`` -> our unit spelling (plural, human-facing).
_UNIT_SPELLINGS = {
    "token": "tokens",
    "char": "chars",
    "second": "seconds",
    "image": "images",
    "megapixel": "megapixels",
    "search": "searches",
    "email": "emails",
    "call": "calls",
}


@dataclass(frozen=True)
class RadientCost:
    """One parsed ``cost`` object, in OUR vocabulary (or ``None`` values).

    Every field is optional because every field of the draft is: a parser that
    demanded ``amount_usd`` would throw away a unit count we can still record
    (the PR-3 TTS/STT case: units known, amount unknown).
    """

    amount_micro: int | None = None
    basis: str = BASIS_NOT_TRACKED
    units: float | None = None
    unit: str = ""
    tool_type: str = ""
    channel: str = "other"
    provider: str = ""
    model: str = ""
    price_version: str = ""
    usage_record_id: str = ""
    version: int | None = None

    def spend_kwargs(self) -> dict[str, Any]:
        """The keyword arguments a :class:`ChannelSpendRecord` takes from a cost.

        ``cost_source`` is ``server_reported`` when the object carried a figure
        — it is the SERVER's own statement — and ``none`` otherwise. The
        price version and the usage record id ride ``price_version`` and
        ``request_id`` respectively, which is what later reconciles a row
        against ``GET /v1/tenants/:id/usage``.
        """
        return {
            "channel": self.channel,
            "provider": self.provider,
            "model": self.model,
            "units": float(self.units or 0.0),
            "unit": self.unit,
            "amount_micro": self.amount_micro,
            "billing_basis": self.basis,
            "cost_source": "server_reported" if self.amount_micro is not None else "none",
            "price_version": self.price_version,
            "request_id": self.usage_record_id,
            "detail": "" if self.tool_type in ("", self.channel) else self.tool_type,
        }

    def to_record(
        self,
        *,
        record_id: str,
        session_id: str = "",
        parent_session_id: str = "",
        status: str = "ok",
        rev: int = 0,
        ts_ms: int = 0,
    ) -> ChannelSpendRecord:
        """The frozen record for one response carrying this cost object."""
        return ChannelSpendRecord(
            record_id=record_id,
            rev=rev,
            ts_ms=ts_ms,
            session_id=session_id,
            parent_session_id=parent_session_id,
            status=status,
            **self.spend_kwargs(),
        )


def unit_from_wire(value: Any) -> str:
    """The wire's singular unit as our plural spelling (``""`` when absent)."""
    if not isinstance(value, str) or not value.strip():
        return ""
    key = value.strip().lower()
    return _UNIT_SPELLINGS.get(key, key)


def channel_from_tool_type(value: Any) -> str:
    """The wire's ``tool_type`` as one of our closed channel set."""
    if not isinstance(value, str) or not value.strip():
        return "other"
    return _TOOL_TYPE_CHANNELS.get(value.strip().lower(), "other")


def basis_from_wire(value: Any) -> str:
    """The wire's ``basis`` as ours. ``quote`` is ``estimated``, never a charge."""
    if not isinstance(value, str) or not value.strip():
        return BASIS_NOT_TRACKED
    key = value.strip().lower()
    if key == "quote":
        return BASIS_ESTIMATED
    mapped = normalise_basis(key)
    return mapped or BASIS_NOT_TRACKED


def micro_from_amount(value: Any) -> int | None:
    """``amount_usd`` as integer micro-USD, or ``None`` when absent/unusable.

    Accepts a float, an int and — because the draft has not fixed the type and
    a decimal STRING is the shape that survives JSON round-trips without
    rounding — a numeric string, converted through :class:`Decimal` so
    ``"0.053"`` cannot become ``0.052999999999999995`` on the way to micros.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            amount = Decimal(value.strip())
        elif isinstance(value, int):
            amount = Decimal(value)
        else:
            amount = Decimal(str(float(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    micro = int((amount * 1_000_000).to_integral_value(rounding="ROUND_HALF_UP"))
    return micro


def cost_from_payload(payload: Any) -> RadientCost | None:
    """Parse a response body's top-level ``cost`` object, or ``None``.

    ``None`` means the response carried no cost object at all — an old server,
    an endpoint that has not adopted it yet, or a failure shape. It NEVER means
    zero: the caller records an unknown amount, which is the honest state.
    """
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("cost")
    if not isinstance(raw, Mapping):
        return None
    return _cost_from_mapping(raw)


def cost_from_headers(headers: Any) -> RadientCost | None:
    """Parse the ``X-Radient-Cost-*`` header block, or ``None`` when absent.

    Accepts anything with ``.get`` (``httpx.Headers``, a dict) or a sequence of
    ``(name, value)`` pairs, so the caller can pass whatever its HTTP client
    produced. Header names are case-insensitive.
    """
    if headers is None:
        return None
    items: list[tuple[str, str]] = []
    lookup = getattr(headers, "items", None)
    if callable(lookup):
        try:
            items = [(str(name), str(value)) for name, value in lookup()]
        except Exception:  # noqa: BLE001 — an unreadable header block is "absent"
            return None
    elif isinstance(headers, Sequence):
        for pair in headers:
            if isinstance(pair, Sequence) and len(pair) == 2:
                items.append((str(pair[0]), str(pair[1])))
    if not items:
        return None
    found: dict[str, str] = {}
    for name, value in items:
        key = name.strip().lower()
        if key.startswith(HEADER_PREFIX):
            found[key[len(HEADER_PREFIX) :]] = value.strip()
    if not found:
        return None
    raw: dict[str, Any] = {
        "v": found.get("version", ""),
        "amount_usd": found.get("amount-usd", ""),
        "basis": found.get("basis", ""),
        "units": found.get("units", ""),
        "unit": found.get("unit", ""),
        "usage_record_id": found.get("usage-record-id", ""),
    }
    return _cost_from_mapping(raw)


def _cost_from_mapping(raw: Mapping[str, Any]) -> RadientCost:
    """Map one already-located ``cost`` mapping into our vocabulary.

    Tolerant by design: a field that is missing, empty or the wrong type reads
    as "not stated" and the caller decides what the absence means. An empty
    string is treated as absent — the header parser produces those for missing
    headers, and a blank ``basis`` must not become the string ``""``.
    """

    def text(key: str) -> str:
        value = raw.get(key)
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        return str(value).strip()

    version_text = text("v")
    try:
        version: int | None = int(version_text) if version_text else None
    except ValueError:
        version = None
    amount_text = text("amount_usd")
    amount = micro_from_amount(amount_text if amount_text else raw.get("amount_usd"))
    units_text = text("units")
    units: float | None = None
    if units_text:
        try:
            units = float(units_text)
        except ValueError:
            units = None
    if units is None and isinstance(raw.get("units"), int | float):
        units = float(raw["units"])
    tool_type = text("tool_type")
    return RadientCost(
        amount_micro=amount,
        basis=basis_from_wire(text("basis")),
        units=units,
        unit=unit_from_wire(text("unit")),
        tool_type=tool_type,
        channel=channel_from_tool_type(tool_type),
        provider=text("provider"),
        model=text("model"),
        price_version=text("price_version"),
        usage_record_id=text("usage_record_id"),
        version=version,
    )

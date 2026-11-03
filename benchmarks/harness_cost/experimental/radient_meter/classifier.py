"""Frozen Radient decisions billing contract, not a remote spending guarantee.

A price response must match the reviewed version AND alias/provider mapping.
This module never fetches prices, resolves credentials or persists payloads.
Informational usage.cost is intentionally not added to the platform charge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

from .json_types import JSONValue, amount, count, object_value, parse_json
from .model import Observation

RATE = Decimal("0.000000042")
VERSION = "decisions-rate@2026-09-19"
MODEL = "jev-1.13"
ALIASES = frozenset((MODEL, "jev-1.13.0", "typesafe/jev-1.13"))
PRICE_ALIASES = frozenset(("jev-1.13.0", "typesafe/jev-1.13"))


@dataclass(frozen=True)
class PriceQualification:
    """An exact supplied /v1/prices match; origin/freshness belongs to the host.

    digest is safe public rate-response provenance, NOT evidence of live I/O.
    The future controller must obtain these bytes directly from the API.
    """

    response_sha256: str
    version: str = VERSION


def qualify_prices(raw: bytes, status: int, *, max_bytes: int = 65536) -> PriceQualification:
    """Reject silent provider/model/version/alias changes before dispatch."""
    if status != 200 or not 0 < len(raw) <= max_bytes:
        raise ValueError("price response status/size")
    root = object_value(parse_json(raw))
    result = object_value(root.get("result"))
    rates = object_value(result.get("rates"))
    family = object_value(rates.get("decision_models"))
    rows = family.get("rates")
    if family.get("rate_version") != VERSION or not isinstance(rows, list):
        raise ValueError("unsupported prices version/shape")
    matches: list[dict[str, JSONValue]] = []
    for value in rows:
        row = object_value(value)
        aliases = row.get("aliases")
        identity = row.get("id")
        if not isinstance(identity, str):
            raise ValueError("invalid price model identity")
        if identity == MODEL:
            matches.append(row)
        elif identity in ALIASES or (
            isinstance(aliases, list)
            and any(alias in ALIASES for alias in aliases if isinstance(alias, str))
        ):
            raise ValueError("ambiguous price alias mapping")
    if len(matches) != 1:
        raise ValueError("missing/duplicate supported price row")
    row = matches[0]
    aliases = row.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
        raise ValueError("unsupported aliases")
    if (
        len(aliases) != len(PRICE_ALIASES)
        or frozenset(a for a in aliases if isinstance(a, str)) != PRICE_ALIASES
    ):
        raise ValueError("unsupported aliases")
    if (
        row.get("provider") != "typesafe"
        or row.get("unit") != "token"
        or row.get("price_version") != VERSION
        or amount(row.get("prompt_unit_price_usd")) != RATE
        or amount(row.get("completion_unit_price_usd")) != 0
    ):
        raise ValueError("unsupported price contract")
    return PriceQualification(hashlib.sha256(raw).hexdigest())


def validate_classifier_request(
    body: dict[str, JSONValue], *, max_questions: int, max_criteria: int
) -> None:
    """Validate shape/counts without changing native state, prompts or budgets."""
    model = body.get("model")
    if not isinstance(model, str) or model not in ALIASES:
        raise ValueError("unsupported requested classifier model")
    if body.get("provider") not in (None, "typesafe"):
        raise ValueError("unsupported requested classifier provider")
    for key in ("tools", "service_tier", "speed"):
        if key in body:
            raise ValueError("unsupported classifier request feature")
    if "max_tokens" in body and not 0 < count(body["max_tokens"]) <= 128000:
        raise ValueError("invalid classifier output budget")
    questions = object_value(body.get("questions"))
    if not 0 < len(questions) <= max_questions:
        raise ValueError("question bound")
    candidates = 0
    for question in questions.values():
        q = object_value(question)
        if q.get("type") != "choice":
            raise ValueError("unsupported question type")
        criteria = object_value(q.get("criteria"))
        if not criteria or any(not isinstance(v, str) for v in criteria.values()):
            raise ValueError("invalid criteria")
        candidates += len(criteria)
    if not 0 < candidates <= max_criteria:
        raise ValueError("candidate bound")


def qualify_classifier(payload: dict[str, JSONValue], status: int) -> Observation:
    """Preserve a finite stated bill even on HTTP/error/qualification failure."""
    known: Decimal | None = None
    faults: list[str] = []
    supported = False
    try:
        cost = object_value(payload.get("cost"))
        known = amount(cost.get("amount_usd"))
    except (ValueError, ArithmeticError):
        cost = {}
        faults.append("invalid-or-missing-billed-cost")
    try:
        usage = object_value(payload.get("usage"))
        tokens = count(usage.get("input_tokens"))
        count(usage.get("output_tokens"))
        if (
            type(cost.get("v")) is not int
            or cost.get("v") != 1
            or cost.get("basis") != "billed"
            or cost.get("model") != MODEL
            or cost.get("provider") != "typesafe"
            or cost.get("price_version") != VERSION
            or cost.get("unit") != "token"
            or count(cost.get("units")) != tokens
            or amount(cost.get("unit_price_usd")) != RATE
            or payload.get("model") != "jev-1.13.0"
            or payload.get("provider") != "typesafe"
            or known is None
            or abs(known - Decimal(tokens) * RATE) > Decimal("1e-12")
        ):
            raise ValueError("unsupported billing/served identity")
        supported = True
    except (ValueError, ArithmeticError):
        faults.append("unsupported-classifier-contract-or-usage")
    terminal = status == 200 and "error" not in payload and supported
    if not terminal:
        faults.append("classifier-http-error-or-incomplete")
    return Observation(
        actual_usd=known,
        terminal_complete=terminal,
        supported_boundary=supported,
        qualified_complete=terminal and not faults,
        faults=tuple(faults),
        served_model="jev-1.13.0" if payload.get("model") == "jev-1.13.0" else None,
        provider_evidence=(
            "explicit-typesafe" if payload.get("provider") == "typesafe" else "unqualified"
        ),
    )

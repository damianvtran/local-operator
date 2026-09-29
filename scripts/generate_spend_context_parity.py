#!/usr/bin/env python3
"""Regenerate the phone's spend/context parity fixture from the Python sources.

WHY THIS FILE EXISTS
--------------------
The phone's session screen reads two numbers off `SessionProjection` — what the
session has cost and how full its context window is — and spells them with a
TypeScript port of the Python spellings:

* the money ladder:      `local_operator/tui/costs.py::format_usd` /
                         `micro_from_usd` (the float half the band reaches
                         through `status_line.format_cost`, whose non-finite
                         fallback a coerced wire value cannot hit);
* the spend composite:   `FrontendSessionState.cumulative_cost` /
                         `cumulative_cost_knowledge`, plus the band's rules
                         (`tui/app.py::_spend_text`): zero renders nothing,
                         partial/floor mark `\u2265`, unknown + billed is `$—`;
* the context reading:   `tui/widgets/status_line.py::context_spelling` /
                         `context_semantic_color` (+ its two band tuples) and
                         `session/frontend_state.py::format_context_tokens` /
                         `format_window`;
* the rounding rule:     half-to-even, because `toFixed` rounds half away
                         from zero and would disagree at every representable
                         tie (the desktop strip's round-1 finding, R1/Q1).

Two spellings of one number is two answers to one question, and neither tree
can read the other: so the bridge is ONE artifact,
``local_operator/mobile/web/src/lib/spend-context.parity.json``, read from both
sides —

* ``tests/unit/mobile/test_tui_bridge.py`` asserts the Python sources against
  it;
* ``local_operator/mobile/web/src/lib/spend-context.test.ts`` asserts the TS
  port against it.

A change to either side fails a suite in ITS OWN tree, and the only way to
re-align them is to regenerate here and make the other side agree. The fixture
lives under the web bundle deliberately: that directory is the mobile-web
workflow's own path filter, so regenerating it (a web-path change) is exactly
what makes the vitest half run in CI.

CASES are derived from the real Python, never hand-written answers: spend cases
construct a real ``FrontendSessionState(**inputs)`` and read its
``cumulative_cost`` / ``cumulative_cost_knowledge`` properties; every ladder
and context case calls the source function. The list covers the crossings the
forms change at (\u00b5$ 50, $0.01, $1.00; 0.05% and every context band at
0.55/0.8 fractions and 200k/500k tokens) with \u00b1\u03b5 steps, every ledger
combination (owner ledger AND compatibility rows together — the double-count
case), and the zero / floor / unpriceable spellings.

    .venv/bin/python scripts/generate_spend_context_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.session.frontend_state import (  # noqa: E402
    CostKnowledge,
    FrontendSessionState,
)
from local_operator.tui.costs import (  # noqa: E402
    LOWER_BOUND_MARK,
    UNKNOWN_COST_CELL,
    format_usd,
    micro_from_usd,
)
from local_operator.tui.widgets.status_line import (  # noqa: E402
    context_semantic_color,
    context_spelling,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "local_operator/mobile/web/src/lib/spend-context.parity.json"
)

#: The spend half's wire inputs, in the projection's field order. Flattened per
#: case so a diff names exactly which input moved.
_SPEND_KEYS = (
    "cumulative_parent_cost",
    "child_costs",
    "subagent_cost",
    "subagent_cost_knowledge",
    "cost_knowledge",
)

#: `$—` + billed is the ONE spelling that needs the usage pair as well, so the
#: spend cases carry it explicitly (see `build`).
_USAGE_KEYS = ("input_tokens", "output_tokens")


def _spend_case(inputs: dict[str, Any], usage: dict[str, int] | None) -> list[Any]:
    """One spend case: the five wire inputs, the usage pair, and the answer.

    The answer is computed the way the band computes it — the state's own
    properties for the figure and its rung, then `_spend_text`'s spelling
    rules on top (zero drops the segment; a floor marks it; `None` + billed
    tokens is `$—`).
    """
    state = FrontendSessionState(session_id="parity", epoch="parity", **inputs)
    total = state.cumulative_cost
    knowledge = state.cumulative_cost_knowledge
    is_floor = knowledge in {CostKnowledge.PARTIAL, CostKnowledge.FLOOR}
    if total is None:
        billed = bool((usage or {}).get("input_tokens") or (usage or {}).get("output_tokens"))
        text = UNKNOWN_COST_CELL if billed else ""
    elif not total:
        # `_spend_text`: a zero spends NO segment — an unpriced model reaching
        # the band as 0.0 is what would otherwise print a confident `$0.0000`.
        text = ""
    else:
        micro = micro_from_usd(total)
        assert micro is not None  # finite JSON inputs only
        text = f"{LOWER_BOUND_MARK if is_floor else ''}{format_usd(micro)}"
    return [
        [inputs.get(key) for key in _SPEND_KEYS],
        [usage.get(key) for key in _USAGE_KEYS] if usage is not None else None,
        [total, knowledge.value, is_floor, text],
    ]


def _spend_cases() -> list[list[Any]]:
    """Every ledger combination and every spelling the spend half can take."""
    cases: list[list[Any]] = []

    def case(inputs: dict[str, Any], usage: dict[str, int] | None = None) -> None:
        cases.append(_spend_case(dict(inputs), usage))

    # -- rows only (the compatibility map): summed, and empty = none ---------
    case({"child_costs": {"job-a": 0.5, "job-b": 0.25}})
    case({"cumulative_parent_cost": 1.0, "child_costs": {"job-a": 0.5}})
    case({"cumulative_parent_cost": 0.0, "child_costs": {}})
    case({"child_costs": {"job-a": 0.0}})

    # -- the owner ledger, and the switch that makes it win ------------------
    # With `subagent_cost_knowledge` set, the owner figure REPLACES the rows —
    # never both. A case with both present pins the double-count rule.
    case(
        {
            "cumulative_parent_cost": 1.0,
            "child_costs": {"job-a": 0.5},
            "subagent_cost": 2.0,
            "subagent_cost_knowledge": "exact",
        }
    )
    case({"subagent_cost": 2.0, "subagent_cost_knowledge": "exact", "child_costs": {"job-a": 99.0}})
    case(
        {"cumulative_parent_cost": 1.0, "subagent_cost": 2.0, "subagent_cost_knowledge": "partial"}
    )
    case({"cumulative_parent_cost": 1.0, "subagent_cost": 1.0, "subagent_cost_knowledge": "floor"})
    case(
        {
            "cumulative_parent_cost": 1.0,
            "subagent_cost": 999.0,
            "subagent_cost_knowledge": None,
            "child_costs": {"job-a": 0.5},
        }
    )
    case(
        {
            "cumulative_parent_cost": 1.0,
            "subagent_cost_knowledge": "unknown",
            "child_costs": {"job-a": 0.5},
            "subagent_cost": None,
        }
    )
    case({"subagent_cost": None, "subagent_cost_knowledge": "exact"})
    case({"subagent_cost": 0.0, "subagent_cost_knowledge": "exact"})

    # -- the parent's own rungs ---------------------------------------------
    case({"cumulative_parent_cost": 12.3456, "cost_knowledge": "exact"})
    case({"cumulative_parent_cost": 3.0, "cost_knowledge": "partial"})
    case({"cumulative_parent_cost": 3.0, "cost_knowledge": "floor"})
    case({"cumulative_parent_cost": 3.0, "cost_knowledge": "unknown"})
    case({"cumulative_parent_cost": 3.0, "cost_knowledge": "exact", "child_costs": {"job-a": 0.5}})
    # An own `partial` outranks a known-exact owner ledger (the figure is
    # already a floor; a child cannot make it exact).
    case(
        {
            "cumulative_parent_cost": 3.0,
            "cost_knowledge": "partial",
            "subagent_cost": 1.0,
            "subagent_cost_knowledge": "exact",
        }
    )

    # -- the money-ladder crossings, through the full composite --------------
    for parent in (
        0.000001,
        0.00002,
        0.000049,
        0.00005,
        0.000051,
        0.0042,
        0.009999,
        0.01,
        0.010001,
        0.213,
        0.999999,
        1.0,
        1.000001,
        1234.5678,
    ):
        case({"cumulative_parent_cost": parent, "cost_knowledge": "exact"})
    case({"cumulative_parent_cost": 0.00002, "cost_knowledge": "floor"})
    case({"cumulative_parent_cost": 0.000049, "cost_knowledge": "partial"})

    # -- nothing to state, and the two nothings that differ ------------------
    case({})
    case({"cumulative_parent_cost": None, "child_costs": {}})
    case({}, {"input_tokens": 9000, "output_tokens": 100})
    case({}, {"input_tokens": 0, "output_tokens": 0})
    case({}, {"input_tokens": 10, "output_tokens": 0})
    case({"child_costs": {"job-a": 0.5}}, {"input_tokens": 9000, "output_tokens": 100})
    return cases


def _context_cases() -> list[list[Any]]:
    """Spelling + rung at every crossing, \u00b1\u03b5, and past the band unions."""
    cases: list[list[Any]] = []

    def case(tokens: int, window: int) -> None:
        cases.append(
            [
                tokens,
                window,
                context_spelling(tokens, window),
                context_semantic_color(tokens, window),
            ]
        )

    # Nothing to report / no denominator.
    case(0, 200_000)
    case(-5, 1_000)
    case(12_000, 0)
    case(1_500, 0)
    case(999, 0)

    # The `format_context_tokens` / `format_window` crossings.
    for tokens in (999, 1_000, 1_001, 12_400, 999_999, 1_000_000, 1_250_000):
        case(tokens, 0)
    for window in (999, 1_000, 1_500, 200_000, 999_999, 1_000_000, 1_050_000, 1_500_000):
        case(1500, window)

    # The `<0.1%` refusal and its two edges (percent = 100 * t / w).
    for tokens in (99, 100, 101):
        case(tokens, 200_000)

    # The proportional bands (0.55 / 0.8), strictly-greater boundaries.
    for tokens in (109_999, 110_000, 110_001, 159_999, 160_000, 160_001, 200_001):
        case(tokens, 200_000)

    # The absolute bands (200k / 500k), with window unknown and known.
    for tokens in (199_999, 200_000, 200_001, 499_999, 500_000, 500_001):
        case(tokens, 0)
        case(tokens, 1_000_000)

    # The union: whichever ladder is warmer wins.
    case(450_000, 1_000_000)
    case(600_000, 1_000_000)
    case(300_000, 300_000)
    return cases


def _fixed_cases() -> list[list[Any]]:
    """`pyFixed` cases: real ties, near-ties, and both signs."""
    values = (
        0.0,
        -0.0,
        0.05,
        0.125,
        0.375,
        1.005,
        1.25,
        2.5,
        2.675,
        3.5,
        12.345,
        -2.675,
        -12.345,
        999.995,
        12_345.678_9,
        1e20,
    )
    return [
        [value, digits, format(value, f".{digits}f")]
        for value in values
        for digits in (0, 1, 2, 3, 4)
    ]


def _money_cases() -> list[list[Any]]:
    """The `format_usd` ladder alone: every crossing and \u00b1\u03b5."""
    micros = (
        0,
        1,
        49,
        50,
        51,
        99,
        100,
        999,
        9_999,
        10_000,
        10_001,
        12_345,
        213_000,
        999_999,
        1_000_000,
        1_000_001,
        4_200_000,
        123_456_789,
    )
    return [[micro, format_usd(micro)] for micro in micros]


def _micro_cases() -> list[list[Any]]:
    """The float→micro step alone: exact ties land on half-to-even."""
    costs = (
        0.0,
        0.0000005,  # 0.5 µ$ — a real tie
        0.0000015,  # 1.5 µ$
        0.0000025,  # 2.5 µ$
        -0.0000005,
        0.00005,
        0.0042,
        0.1234565,
        1.25,
        -1.25,
        1.897_843,
        1234.56,
    )
    return [[cost, micro_from_usd(cost)] for cost in costs]


def build() -> dict[str, Any]:
    """The fixture's content: every case with the Python's own answer for it."""
    return {
        "_": (
            "Generated by scripts/generate_spend_context_parity.py — do not edit by hand. "
            "Read by the vitest suite (spend-context.test.ts) and the Python suite "
            "(tests/unit/mobile/test_tui_bridge.py) so the phone's ported spend/context "
            "spellings and the Python originals cannot drift apart in silence."
        ),
        "pyFixed": _fixed_cases(),
        "money": _money_cases(),
        "micro": _micro_cases(),
        "spend": _spend_cases(),
        "context": _context_cases(),
    }


def render() -> str:
    """The fixture's exact bytes — one case per line, so a diff names what moved."""
    data = build()
    lines = [f' "_": {json.dumps(data["_"])}']
    for section in ("pyFixed", "money", "micro", "spend", "context"):
        cases = data[section]
        body = ",\n".join(f"  {json.dumps(case, ensure_ascii=False)}" for case in cases)
        lines.append(f' "{section}": [\n{body}\n ]')
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main() -> None:
    """Write the fixture, or (``--check``) prove the committed one came from here.

    ``--check`` is the PROVENANCE half of the pin, the same one the clock
    fixture carries: both suites assert the fixture's CONTENT against their own
    formatter, which catches drift in a formatter but not in the file itself.
    ``tests/unit/mobile/test_tui_bridge.py`` makes the same comparison, so CI
    runs it on every PR; this flag is the human path for the same question.
    """
    data = build()
    total = sum(len(data[section]) for section in ("pyFixed", "money", "micro", "spend", "context"))
    if "--check" in sys.argv[1:]:
        if FIXTURE.read_text() == render():
            print(f"{FIXTURE} is current ({total} cases)")
            return
        print(f"{FIXTURE} is STALE — regenerate it from this script", file=sys.stderr)
        raise SystemExit(1)
    FIXTURE.write_text(render())
    print(f"wrote {FIXTURE} ({total} cases)")


if __name__ == "__main__":
    main()

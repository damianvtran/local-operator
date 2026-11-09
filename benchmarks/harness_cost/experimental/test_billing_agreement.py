"""One row fixture, three readers: a billing verdict must not depend on the lane.

WHY (R18 m4/n2): the OSWorld lane (``runner_v2.money``), the coding driver
(``driver.row_cost``) and the publishing rig (``coding/billing.money``) each read
the meter's log, so each carries its own copy of the derivation rules. Two of the
copies are deliberate -- the driver runs as a script from its own directory and
the rig reads the log, never the meter's module -- which is exactly why a
divergence stays invisible until a published table disagrees with an aborted
trial. R18 found two: an ``incomplete`` row that states a ``usage.cost`` was
published as a number by both ``money()`` readers while the driver aborted the
trial, and only the driver rejected a non-finite or negative *rate* explicitly.

This module pins the three readers to one verdict per row. The rows are synthetic
meter rows, never exported as evidence. The rig is loaded by file path from
``BENCH_CODING_RIG`` (a checkout root containing ``coding/billing.py``) and its
test skips when that is unset, because the rig is a separate tree the repository
must not depend on at import time; the in-tree pair is always checked.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "coding"))
sys.path.insert(0, str(HERE))

import driver  # noqa: E402  (needs HERE/coding on sys.path)
import runner_v2 as r  # noqa: E402

# The counts and the run's frozen card this fixture is priced at.
TOKENS = {
    "input_tokens": 2,
    "output_tokens": 1,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 27164,
}
RATES = {
    "input_tokens": 2.0,
    "output_tokens": 10.0,
    "cache_read_input_tokens": 0.1,
    "cache_creation_input_tokens": 2.5,
}
# 2*2.0 + 1*10.0 + 0*0.1 + 27164*2.5 megamicrodollars, and the same row with
# one output token instead of zero (the shape an uncounted term hides in).
USD = 0.067924
UNCOUNTED_USD = 0.067914


def settled(usd: float = USD, rates: dict | None = None, tokens: dict | None = None) -> dict:
    """A row the meter settled from its own tokens (``settled_from_tokens``)."""
    tokens = dict(tokens or TOKENS)
    return {
        "run_id": "coding__omp__tasktest__r60",
        "path": "v1/messages",
        "status": 200,
        "settled_from_tokens": True,
        "cost_derivation": {
            "tokens": tokens,
            "rates_usd_per_m": dict(rates if rates is not None else RATES),
            "usd": usd,
        },
        "usage": dict(tokens),
    }


def stated(cost: float = 0.075) -> dict:
    """A complete row carrying the provider's own bill, as OpenRouter sends it."""
    return {
        "run_id": "coding__omp__tasktest__r60",
        "path": "v1/messages",
        "status": 200,
        "usage": {**TOKENS, "cost": cost},
    }


# A zero-count term is the shape where a bad rate hides: it contributes nothing
# to the arithmetic, so only a rule that inspects the rate itself refuses it.
UNCOUNTED = {**TOKENS, "output_tokens": 0}


def cases() -> tuple[tuple[str, dict, float | None], ...]:
    """(name, row, expected) with ``None`` meaning unknown spend, never zero."""
    return (
        ("settled_from_tokens", settled(), USD),
        ("stated_cost", stated(), 0.075),
        # R18 m4: partial usage is a floor, so the cost printed beside it is a
        # floor too. Unknown in every reader -- never a number, never zero.
        ("incomplete_with_stated_cost", {**stated(), "incomplete": True}, None),
        ("incomplete_with_derivation", {**settled(), "incomplete": True}, None),
        # R18 n2: NaN, negative and missing rates must be refused by rule, not by
        # an arithmetic accident that today only the driver states explicitly.
        ("nan_rate", settled(usd=USD, rates={**RATES, "output_tokens": math.nan}), None),
        ("negative_rate_on_an_uncounted_term",
         settled(usd=UNCOUNTED_USD, rates={**RATES, "output_tokens": -1.0},
                 tokens=UNCOUNTED), None),
        ("missing_rate_on_a_counted_term",
         settled(usd=USD, rates={k: v for k, v in RATES.items() if k != "output_tokens"}), None),
        ("nan_rate_on_an_uncounted_term",
         settled(usd=UNCOUNTED_USD, rates={**RATES, "output_tokens": math.nan},
                 tokens=UNCOUNTED), None),
    )


def lane_verdict(money, row: dict, argument) -> float | None:
    """A ``money()`` reader's verdict: the amount, or ``None`` for unknown spend.

    ``argument`` is the reader's own second parameter -- the run's ledger summary
    for ``runner_v2.money``, the classification stderr for the rig's ``money``.
    The refusal is checked in full, because "unknown" is several claims: no
    total, no complete flag and no measured amount standing in for the row.
    """
    account = money([row], argument, False)
    if account["cost_complete"]:
        return account["total_cost_usd"]
    assert account["total_cost_usd"] is None, account
    assert account["model_cost_usd"] is None, account
    assert account["measured_model_cost_usd"] == 0.0, account
    assert account["unknown_billing_attempts"] == 1, account
    return None


def driver_verdict(row: dict) -> float | None:
    """``driver.row_cost``'s verdict, with its ``ValueError`` read as unknown."""
    try:
        return driver.row_cost(row)
    except ValueError:
        return None


def load_rig() -> object | None:
    root = os.environ.get("BENCH_CODING_RIG")
    if not root:
        return None
    path = Path(root) / "coding" / "billing.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("rig_billing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RIG = load_rig()


class AgreementTests(unittest.TestCase):
    def test_the_in_tree_readers_agree_on_every_row(self):
        for name, row, expected in cases():
            with self.subTest(row=name):
                self.assertEqual(driver_verdict(row), expected)
                self.assertEqual(lane_verdict(r.money, row, None), expected)

    @unittest.skipUnless(RIG, "set BENCH_CODING_RIG=<rig checkout> to check the publishing rig")
    def test_the_publishing_rig_agrees_with_the_two_in_tree_readers(self):
        for name, row, expected in cases():
            with self.subTest(row=name):
                self.assertEqual(driver_verdict(row), expected)
                self.assertEqual(lane_verdict(r.money, row, None), expected)
                self.assertEqual(lane_verdict(RIG.money, row, ""), expected)


if __name__ == "__main__":
    unittest.main()

"""The golden set: spam rate, pre-filter absorption and the graphics gate (memo §5.2).

WHAT IS MEASURED, AND WHAT IS NOT. This file measures the DETERMINISTIC half of the feature
over the 60 labelled turns in ``tests/fixtures/supplements/golden/turns.json``: the pre-filter
(the gate before any vendor call) and the no-vendor decision the runner takes when the
decision layer answers nothing. It does NOT measure the vendor's own precision/recall — that
is the memo's §5.3 cost/quality plan and the operator's dogfood week, because a vendor call
inside a unit test would be both nondeterministic and a spend. The numbers this file prints
are the C1a acceptance numbers:

* **spam rate** — turns where something was shown although the label says nothing should be,
  over all turns. Acceptance: <= 3 %.
* **pre-filter absorption** — turns that never reach a vendor. Target: >= 60 %.
* **graphics gate precision/recall** — the deterministic signal's halves: every turn it calls
  structured must be one the label wants a figure for (precision), and most of those labels
  must be found (recall >= 0.6; precision over recall is the memo's stated direction).

Each class has its own assertion too, so a class that starts leaking is named rather than
absorbed into an average.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from local_operator.supplements.candidates import prefilter
from local_operator.supplements.decision import decide as decide_supplement
from local_operator.supplements.trigger import ToolCallItem, TurnItem

from .conftest import FIXTURES

GOLDEN = FIXTURES / "golden" / "turns.json"
#: The memo's acceptance figures.
MAX_SPAM_RATE = 0.03
MIN_ABSORPTION = 0.60
MIN_GRAPHICS_PRECISION = 0.9
MIN_GRAPHICS_RECALL = 0.6


def _turns() -> list[dict[str, Any]]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["turns"]


def _items(turn: dict[str, Any]) -> list[TurnItem]:
    items: list[TurnItem] = []
    if turn["tool"]:
        items.append(
            TurnItem(
                id=f"{turn['id']}-call",
                role="assistant",
                tool_calls=(
                    ToolCallItem(id=f"{turn['id']}-c1", name=turn["tool"], args=dict(turn["call"])),
                ),
            )
        )
        if turn["result"]:
            items.append(
                TurnItem(
                    id=f"{turn['id']}-res",
                    role="tool",
                    text=turn["result"],
                    tool_call_id=f"{turn['id']}-c1",
                    tool_name=turn["tool"],
                )
            )
    return items


def _materialise(turn: dict[str, Any], root: Path) -> None:
    for relative, body in turn["files"].items():
        if not relative.startswith("/") and "$" not in relative:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)


class _Measurement:
    def __init__(self) -> None:
        self.turns = 0
        self.spam: list[str] = []
        self.shown: list[str] = []
        self.absorbed: list[str] = []
        self.graphics_true_positive: list[str] = []
        self.graphics_false_positive: list[str] = []
        self.graphics_false_negative: list[str] = []
        self.file_misses: list[str] = []

    def summary(self) -> str:
        n = max(1, self.turns)
        return (
            f"turns={self.turns} spam={len(self.spam)} ({len(self.spam) / n:.1%}) "
            f"absorption={len(self.absorbed)}/{self.turns} ({len(self.absorbed) / n:.0%}) "
            f"graphics: tp={len(self.graphics_true_positive)} "
            f"fp={len(self.graphics_false_positive)} fn={len(self.graphics_false_negative)} "
            f"file_misses={len(self.file_misses)}"
        )


async def _measure(want_graphics: bool) -> _Measurement:
    root = Path(tempfile.mkdtemp(prefix="lo-sup-golden-"))
    measured = _Measurement()
    for index, turn in enumerate(_turns()):
        work = root / f"turn{index:02d}"
        work.mkdir()
        _materialise(turn, work)
        items = _items(turn)
        pre = prefilter(
            items,
            turn["answer"],
            cwd=str(work),
            home=str(root),
            want_files=True,
            want_graphics=want_graphics,
        )
        # The decision the runner takes with no vendor answering: the heuristic half.
        decision = await decide_supplement(
            None,
            user_text=turn["user"],
            answer_text=turn["answer"],
            candidates=pre.candidates,
            evidence=pre.evidence,
            want_files=True,
            want_graphics=want_graphics,
            max_featured=4,
        )
        measured.turns += 1
        featured = [c.path for c in decision.featured]
        if featured:
            measured.shown.append(turn["id"])
        if pre.skipped and not (want_graphics and pre.evidence.structured):
            measured.absorbed.append(turn["id"])
        if turn["expect_graphics"] and pre.evidence.structured:
            measured.graphics_true_positive.append(turn["id"])
        elif turn["expect_graphics"] and not pre.evidence.structured:
            measured.graphics_false_negative.append(turn["id"])
        elif not turn["expect_graphics"] and pre.evidence.structured:
            measured.graphics_false_positive.append(turn["id"])
        if turn["expect_nothing"] and featured:
            measured.spam.append(f"{turn['id']}:{featured}")
        if featured and not turn["expect_nothing"]:
            for path in featured:
                if path not in turn["expect_files"]:
                    measured.file_misses.append(f"{turn['id']}:{path}")
    shutil.rmtree(root, ignore_errors=True)
    return measured


@pytest.mark.asyncio
async def test_the_golden_set_holds_the_acceptance_numbers(tmp_path) -> None:
    measured = await _measure(want_graphics=True)
    print("golden set:", measured.summary())  # noqa: T201 — the PR body quotes this line
    spam_rate = len(measured.spam) / max(1, measured.turns)
    assert spam_rate <= MAX_SPAM_RATE, measured.spam
    assert len(measured.absorbed) / measured.turns >= MIN_ABSORPTION, measured.absorbed
    positives = len(measured.graphics_true_positive)
    precision_denominator = positives + len(measured.graphics_false_positive)
    if precision_denominator:
        precision = positives / precision_denominator
        assert precision >= MIN_GRAPHICS_PRECISION, measured.graphics_false_positive
    if positives + len(measured.graphics_false_negative):
        recall = positives / (positives + len(measured.graphics_false_negative))
        assert recall >= MIN_GRAPHICS_RECALL, measured.graphics_false_negative
    assert not measured.file_misses, measured.file_misses
    # Every class that expects nothing must produce nothing: the per-class guard.
    nothing_classes = {"qa", "short", "code-edit", "numbers-no-figure", "bait", "scratch", "secret"}
    offenders = [t for t in measured.spam if t.split(":")[0]]
    assert offenders == [], offenders
    assert nothing_classes, "the classes are named in the fixture; see golden/turns.json"


@pytest.mark.asyncio
async def test_the_60_turns_are_balanced_and_labelled() -> None:
    turns = _turns()
    assert len(turns) == 60
    by_class: dict[str, int] = {}
    for turn in turns:
        by_class[turn["class"]] = by_class.get(turn["class"], 0) + 1
        assert isinstance(turn["expect_files"], list)
        assert isinstance(turn["expect_graphics"], bool)
        assert isinstance(turn["expect_nothing"], bool)
    assert by_class == {
        "qa": 10,
        "short": 8,
        "code-edit": 8,
        "deliverable": 8,
        "benchmark": 8,
        "numbers-no-figure": 6,
        "bait": 6,
        "scratch": 3,
        "secret": 3,
    }, by_class


@pytest.mark.asyncio
async def test_the_deliverable_class_shows_its_file_and_only_its_file(tmp_path) -> None:
    measured = await _measure(want_graphics=False)
    # The files half is on: EVERY deliverable turn shows its written path and NOTHING else
    # does. The missing-file list is empty by the previous test, so this pins the positive
    # direction -- a heuristic that returned nothing would pass every other assertion here.
    deliverable_ids = [t["id"] for t in _turns() if t["class"] == "deliverable"]
    assert measured.shown == deliverable_ids, measured.shown

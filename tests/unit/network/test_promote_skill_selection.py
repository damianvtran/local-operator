"""The promote reports skill selection BESIDE its wake-carry block, never inside it.

PR #4487 first folded ``skill_selection`` into ``_PromoteOutcome.carry``, which made
``carry`` non-``None`` for every promote and broke the exact wake-carry contract that
``test_move_carry_supervisor`` pins ("no wakes, no install" → ``carry is None``). The
selection now rides its own field and meets the wake block only on the RECEIPT
(``_receipt_carry``), which is where the source and the CLI read it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from local_operator.network import mobility
from local_operator.skills.manifest import skill_transfer_receipt

SESSION = "aabbccddeeff"


def _stage_without_wakes(root: Path) -> Path:
    staging = root / "network" / "staging" / SESSION
    staging.mkdir(parents=True)
    entry = {"id": "e1", "ts": 1.0, "type": "message", "payload": {}}
    (staging / "transcript.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return staging


def test_a_promote_without_wakes_reports_skill_selection_outside_carry(tmp_path: Path) -> None:
    root = tmp_path / "dest"
    root.mkdir()
    server: Any = SimpleNamespace(
        root=root,
        identity=SimpleNamespace(device_id="d_destination"),
        _own_label=lambda: "dest-device",
    )

    outcome = mobility._promote(server, _stage_without_wakes(root), SESSION)

    assert outcome.ok is True
    assert outcome.carry is None
    assert outcome.skill_selection == skill_transfer_receipt("local_routing")
    # The receipt still carries it, so the source and the CLI keep their line.
    assert mobility._receipt_carry(outcome) == {
        "skill_selection": skill_transfer_receipt("local_routing")
    }


def test_the_receipt_merges_skill_selection_into_the_wake_block_without_mutating_it() -> None:
    wake_block = {"wakes": 1, "supervisor": "running", "running_notice": "1 wake carried"}
    pinned = skill_transfer_receipt("pinned")
    outcome = mobility._PromoteOutcome(ok=True, carry=wake_block, skill_selection=pinned)

    assert mobility._receipt_carry(outcome) == {**wake_block, "skill_selection": pinned}
    assert "skill_selection" not in wake_block


def test_a_promote_without_a_selection_leaves_the_receipt_carry_untouched() -> None:
    assert mobility._receipt_carry(mobility._PromoteOutcome(ok=True, carry=None)) is None
    block = {"wakes": 2}
    assert mobility._receipt_carry(mobility._PromoteOutcome(ok=True, carry=block)) is block

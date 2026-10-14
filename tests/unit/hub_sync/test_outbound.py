"""Paired A/P projection: local edits kept, remote-only content preserved, no-op detection."""

from __future__ import annotations

from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbound import project_outbound

A = {"instructions": "# A\nbase rule\n", "description": "d"}
P = {"instructions": "# A\nbase rule\n", "description": "d"}


def test_local_edit_since_capture_is_projected() -> None:
    local = {"instructions": "# A\nbase rule\nMY EDIT\n", "description": "d"}
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P,
        current_local=local,
        remote_latest=P,
        opts=MergeOptions(allow_llm=False),
    )
    assert not result.no_op
    assert "MY EDIT" in result.fields["instructions"]


def test_remote_only_content_in_p_is_preserved() -> None:
    # P gained a remote-only section the local copy never had.
    P2 = {"instructions": "# A\nbase rule\nREMOTE ONLY\n", "description": "d"}
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P2,
        current_local=A,
        remote_latest=P2,
        opts=MergeOptions(allow_llm=False),
    )
    assert "REMOTE ONLY" in result.fields["instructions"]


def test_identical_remote_is_a_no_op() -> None:
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P,
        current_local=A,
        remote_latest=P,
        opts=MergeOptions(allow_llm=False),
    )
    assert result.no_op


def test_merge_skill_file_treats_absence_from_a_known_base_as_an_add() -> None:
    """QN4, engine level: base_present=False means ADD semantics on both sides.

    Before the flag, every path added since the baseline was an eternal
    needs-review ("baseline unknown" and "absent from base" were the same
    ``base=None``); the add-on-each-side table then lets a one-sided add land, a
    same-text add settle, and a DIFFERENT-text add conflict.
    """

    from local_operator.hub_sync.merge import MergeOptions
    from local_operator.hub_sync.outbound import merge_skill_file

    opts = MergeOptions(allow_llm=False)

    remote_add, r1 = merge_skill_file(
        "references/a.md", base=None, local=None, remote="A", opts=opts, base_present=False
    )
    assert remote_add == "A" and r1.outcome == "merged"

    local_add, r2 = merge_skill_file(
        "references/b.md", base=None, local="B", remote=None, opts=opts, base_present=False
    )
    assert local_add == "B" and r2.outcome in ("merged", "unchanged")

    same, r3 = merge_skill_file(
        "references/c.md", base=None, local="C", remote="C", opts=opts, base_present=False
    )
    assert same == "C" and not r3.counts.get("unresolved")

    conflict, r4 = merge_skill_file(
        "references/d.md", base=None, local="L", remote="R", opts=opts, base_present=False
    )
    assert conflict is None and r4.outcome == "needs-review" and r4.counts["unresolved"] == 1

    # Legacy META absence (base None with no presence flag) keeps the held meaning.
    held, r5 = merge_skill_file("references/e.md", base=None, local=None, remote="E", opts=opts)
    assert held is None and r5.outcome == "needs-review"

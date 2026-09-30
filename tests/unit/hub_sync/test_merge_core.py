"""The A4.2 table and the invariants the design promises, model-free."""

from __future__ import annotations

import random
from typing import Any

import pytest

from local_operator.hub_sync.merge import (
    ConflictProposal,
    ConflictRequest,
    FieldInput,
    MergeOptions,
    ResolverError,
    decide,
    merge_field,
    regrows,
    replace_field,
    validate_proposal,
)


def _md(base, local, remote, **opts):
    return merge_field(
        FieldInput("instructions", "markdown", base, local, remote), MergeOptions(**opts)
    )


@pytest.mark.parametrize(
    ("b", "lo", "r", "prov", "take", "by"),
    [
        (True, "same", "same", "unchanged", "l", None),
        (True, "mod", "same", "kept-local", "l", None),
        (True, "same", "mod", "taken-remote", "r", None),
        (True, None, "same", "removal-honored", "none", "local"),
        (True, "same", None, "removal-honored", "none", "remote"),
        (True, None, None, "removal-honored", "none", "both"),
        (True, None, "mod", "unresolved", "removal-vs-edit", None),
        (True, "mod", None, "unresolved", "removal-vs-edit", None),
        (False, "add", None, "kept-local", "l", None),
        (False, None, "add", "taken-remote", "r", None),
    ],
)
def test_decision_table(b, lo, r, prov, take, by) -> None:
    d = decide(b, lo, r, False)
    assert (d.prov, d.take, d.removed_by) == (prov, take, by)


def test_both_modified_is_identical_or_a_conflict() -> None:
    assert decide(True, "mod", "mod", True).prov == "kept-local"
    assert decide(True, "mod", "mod", False).take == "conflict"
    assert decide(False, "add", "add", False).take == "conflict"


def test_a_removal_never_regrows_across_random_edits() -> None:
    """Design B8.2.7: with the model off, a deleted section never comes back."""

    rng = random.Random(7)
    base = "\n".join(f"## S{i}\nRule {i} applies here. Keep it short." for i in range(6))
    for _ in range(20):
        gone = rng.randrange(6)
        local = "\n".join(
            f"## S{i}\nRule {i} applies here. Keep it short." for i in range(6) if i != gone
        )
        remote = base + f"\n## New{rng.randrange(99)}\nSomething new."
        result = _md(base, local, remote, allow_llm=False)
        assert f"## S{gone}" not in str(result.merged)
        assert result.outcome == "merged"


def test_a_local_shortening_is_not_relengthened_by_an_unrelated_remote_edit() -> None:
    base = "## A\nUse tools carefully and never run destructive commands without asking."
    local = "## A\nUse tools carefully."
    remote = base + "\n## B\nnew."
    assert "destructive" not in str(_md(base, local, remote, allow_llm=False).merged)


def test_the_result_is_stable_when_local_already_has_the_remote_change() -> None:
    assert _md("## A\nx.", "## A\nx2.", "## A\nx2.").outcome == "unchanged"


def test_large_shrink_warning_and_cap_refusal() -> None:
    big = "\n".join(f"## S{i}\n" + "word " * 60 + "." for i in range(8))
    result = _md(big, "## S0\n" + "word " * 60 + ".", big + "\n## Z\nx.")
    assert result.outcome in ("merged", "unchanged")
    over = _md("## A\nx.", "## A\nx.", "## A\n" + "y" * 500, max_chars=100)
    assert over.outcome == "refused" and "at most 100" in over.refusal


def test_unknown_baseline_is_two_way_conservative_and_needs_acknowledgement() -> None:
    local, remote = "## A\nx.\n## B\ny.", "## A\nx.\n## B\ny.\n## C\nz."
    held = _md(None, local, remote)
    assert held.outcome == "needs-review" and held.merged == local
    taken = _md(None, local, remote, acknowledge_unknown_baseline=True)
    assert taken.outcome == "merged" and "## B" in str(taken.merged) and "## C" in str(taken.merged)


def test_roster_and_scalar_rules() -> None:
    slot = lambda role, n=1: {"role": role, "kind": "agent", "count": n}  # noqa: E731
    r = merge_field(
        FieldInput(
            "members",
            "roster",
            [slot("coder"), slot("qa", 2)],
            [slot("coder")],
            [slot("coder"), slot("qa", 2), slot("ux")],
        )
    )
    assert isinstance(r.merged, list)
    assert [s["role"] for s in r.merged] == ["coder", "ux"] and r.outcome == "merged"
    # Roles compare case-insensitively.
    same = merge_field(
        FieldInput("members", "roster", [slot("Coder")], [slot("coder")], [slot("CODER")])
    )
    assert same.outcome == "unchanged"
    # A removal against a count edit is never auto-resolved.
    clash = merge_field(FieldInput("members", "roster", [slot("qa", 2)], [], [slot("qa", 4)]))
    assert clash.outcome == "needs-review"
    assert merge_field(FieldInput("manager", "scalar", "a", "a", "b")).merged == "b"


def test_replace_echoes_the_discarded_text_and_never_merges() -> None:
    r = replace_field(
        FieldInput("instructions", "markdown", "## A\nb.", "## A\nmine.", "## A\ntheirs."),
        take="remote",
    )
    assert r.engine.mode == "replace" and r.merged == "## A\ntheirs."
    assert r.regions[0].dropped == "## A\nmine." and r.regions[0].note == "replaced"


# -- validators V1-V6 (A6.2) ---------------------------------------------------------------


def _req(**kw) -> ConflictRequest:
    base: dict[str, Any] = dict(
        field="instructions",
        heading="A",
        base="Be brief.",
        local="Be brief and cite `src_id`.",
        remote="Be brief and use 3 words.",
        max_chars=200,
    )
    base.update(kw)
    return ConflictRequest(**base)


def test_validators_reject_each_violation_and_accept_a_faithful_merge() -> None:
    good = ConflictProposal("Be brief, cite `src_id`, and use 3 words.", covers=("l1", "r1"))
    assert validate_proposal(_req(), good) == []
    assert any(
        "V2" in p
        for p in validate_proposal(_req(), ConflictProposal("Be brief.", covers=("l1", "r1")))
    )
    assert any(
        "V2" in p for p in validate_proposal(_req(), ConflictProposal(good.text, covers=("l1",)))
    )
    assert any(
        "V4" in p
        for p in validate_proposal(
            _req(), ConflictProposal(good.text, covers=("l1", "r1"), drops=("r1",))
        )
    )
    assert any("V5" in p for p in validate_proposal(_req(max_chars=10), good))
    assert any(
        "V6" in p
        for p in validate_proposal(
            _req(), ConflictProposal("## H\n" + good.text, covers=("l1", "r1"))
        )
    )
    assert any(
        "V6" in p
        for p in validate_proposal(
            _req(), ConflictProposal("```\n" + good.text, covers=("l1", "r1"))
        )
    )
    assert any("V1" in p for p in validate_proposal(_req(), ConflictProposal("  ", covers=())))


def test_v3_rejects_a_proposal_that_regrows_a_removal() -> None:
    from local_operator.hub_sync.merge import Removal

    req = _req(removals=(Removal("never run destructive commands without asking", "local", "l1"),))
    bad = ConflictProposal(
        "Be brief, cite `src_id`, use 3 words, never run destructive commands without asking.",
        covers=("l1", "r1"),
    )
    assert any("V3" in p for p in validate_proposal(req, bad))
    assert regrows(
        "x never run destructive commands without asking y",
        "never run destructive commands without asking",
    )
    assert not regrows("unrelated text entirely", "never run destructive commands")


class Scripted:
    def __init__(self, *proposals, error=None):
        self.proposals, self.error, self.requests = list(proposals), error, []

    def resolve(self, req):
        self.requests.append(req)
        if self.error:
            raise self.error
        return self.proposals.pop(0) if len(self.proposals) > 1 else self.proposals[0]


def test_an_invalid_proposal_is_re_asked_once_with_the_violations_then_unresolved() -> None:
    bad = ConflictProposal("Be brief.", covers=("l1", "r1"))
    scripted = Scripted(bad)
    r = _md(
        "## A\nBe brief.",
        "## A\nBe brief and cite sources.",
        "## A\nBe brief and use plain words.",
        resolver=scripted,
    )
    assert r.outcome == "needs-review" and len(scripted.requests) == 2
    assert scripted.requests[0].feedback == () and scripted.requests[1].feedback


def test_a_model_outage_degrades_to_unresolved_and_stops_further_calls() -> None:
    scripted = Scripted(error=ResolverError("model-unavailable", "no key"))
    base = "## A\nBe brief.\n## B\nBe kind."
    r = _md(
        base,
        "## A\nBe brief and cite.\n## B\nBe kind and warm.",
        "## A\nBe brief and plain.\n## B\nBe kind and calm.",
        resolver=scripted,
    )
    assert r.outcome == "needs-review" and r.engine.failure_class == "model-unavailable"
    assert len(scripted.requests) == 1  # the second conflict did not burn another call


def test_prefer_settles_only_unresolved_groups_and_reports_the_loser() -> None:
    r = _md(
        "## A\nBe brief.", "## A\nBe brief and cite.", "## A\nBe brief and plain.", prefer="remote"
    )
    assert r.outcome == "merged" and "plain" in str(r.merged)
    assert "cite" in str(r.regions[0].dropped)

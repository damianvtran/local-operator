"""The review-round parser against the REAL comment conventions (public fixtures).

The fixtures in ``tests/fixtures/code_requests/`` are trimmed bodies of real comments on
``damianvtran/local-operator`` and ``damianvtran/local-operator-ui`` (both public) plus
SYNTHESIZED GitLab notes: the real GitLab grammar was measured on a company repository
whose note bodies must not be vendored here, so only its SHAPES are reproduced.

The two rules these tests exist to protect:

* the verdict is classified by its LEADING TOKEN — the real ``PASS — … 0 FAIL, 0
  BLOCKED`` verdict must not read as failing;
* the same round number can appear twice (a re-review of round 1), and the newest pass
  decides the lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.code_requests.rounds import (
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    STATE_CLEAN,
    STATE_FINDINGS_OPEN,
    STATE_REMEDIATION_POSTED,
    STATE_TERMINAL,
    STATE_UNSTATED,
    Comment,
    classify_verdict,
    freshness,
    parse,
    parse_comment,
    parse_remediation,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "code_requests"


def _comments(name: str) -> list[Comment]:
    document = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    comments = [
        Comment(
            id=str(key),
            body=str(value["body"]),
            created_at=str(value["created_at"]),
            url=str(value.get("url") or ""),
        )
        for key, value in document.items()
        if not key.startswith("_")
    ]
    comments.sort(key=lambda item: item.created_at)
    return comments


def test_every_real_right_here_convention_header_parses():
    report = parse(_comments("github_comments.json"))
    assert len(report.passes) == 17
    pairs = {(item.lane, item.kind, item.round) for item in report.passes}
    assert ("agent", "review", 1) in pairs and ("agent", "remediation", 1) in pairs
    assert ("design", "review", 2) in pairs and ("qa", "review", 1) in pairs
    # The negatives: a release note, a merge disclosure and an addendum are NOT reviews.
    assert len(report.ignored) == 3


def test_the_negation_trap_is_a_pass():
    """``**Verdict: PASS — 21/21 … 0 FAIL, 0 BLOCKED`` is the real QA verdict that a
    substring classifier marked as failing."""
    report = parse(_comments("github_comments.json"))
    qa = [item for item in report.passes if item.lane == "qa" and "21/21" in item.verdict]
    assert len(qa) == 1 and qa[0].verdict_class == STATE_CLEAN


def test_a_findings_open_verdict_is_not_clean():
    report = parse(_comments("github_comments.json"))
    opening = [item for item in report.passes if "one MAJOR must be fixed" in item.verdict]
    assert len(opening) == 1 and opening[0].verdict_class == STATE_FINDINGS_OPEN


def test_a_verdict_heading_far_down_the_body_is_still_read():
    """A real review carries ``#### Verdict`` on line 61; a 40-line scan reported it as
    unstated."""
    report = parse(_comments("github_comments.json"))
    not_safe = [item for item in report.passes if "Not safe to merge" in item.verdict]
    assert len(not_safe) == 1
    assert not_safe[0].verdict_class == STATE_FINDINGS_OPEN
    assert not_safe[0].reviewed_head == "8dc2b9797f"


def test_a_review_with_no_verdict_line_says_so():
    report = parse(_comments("github_comments.json"))
    silent = [item for item in report.passes if item.qualifier == "fix verification"]
    assert len(silent) == 1 and silent[0].verdict == ""
    assert silent[0].verdict_class == STATE_UNSTATED


def test_the_qualifier_keeps_its_own_words():
    report = parse(_comments("github_comments.json"))
    qualifiers = {item.qualifier for item in report.passes}
    assert "delta verification" in qualifiers
    assert "release bump" in qualifiers
    assert "fold convergence (scope, fold #8)" in qualifiers


def test_two_passes_of_one_round_are_numbered_apart():
    comments = [
        Comment(
            id="a",
            body=(
                "### Agent review — round 1\n\n**Reviewer:** r\n"
                "**Verdict: NOT clean — one MAJOR.**\n"
            ),
            created_at="2026-01-01T00:00:00Z",
        ),
        Comment(
            id="b",
            body="### Agent review — round 1 (fix verification)\n\n**Verdict: clean.**\n",
            created_at="2026-01-02T00:00:00Z",
        ),
    ]
    report = parse(comments, head_sha="a" * 40)
    assert [item.sequence for item in report.passes] == [1, 2]
    state = report.states(head_sha="a" * 40)[0]
    assert state.lane == "agent" and state.state == STATE_CLEAN and state.round == 1


def test_lane_states_and_freshness_over_one_pr():
    """States are per PR, because comments are: the fixture holds several PRs' threads."""
    document = json.loads((FIXTURES / "github_comments.json").read_text(encoding="utf-8"))
    comments = [
        Comment(id=str(key), body=str(value["body"]), created_at=str(value["created_at"]))
        for key, value in document.items()
        if not key.startswith("_") and value["number"] == 2090
    ]
    comments.sort(key=lambda item: item.created_at)
    report = parse(comments)
    states = {item.lane: item for item in report.states(head_sha="72d4fb1ead")}
    assert states["agent"].state == STATE_TERMINAL and states["agent"].freshness == FRESHNESS_FRESH
    assert states["design"].state == STATE_TERMINAL
    assert states["qa"].state == STATE_CLEAN
    # A head that has moved makes every lane's freshness stale rather than silently green.
    moved = {item.lane: item for item in report.states(head_sha="9999999999")}
    assert moved["agent"].freshness == FRESHNESS_STALE
    # No head at all is UNKNOWN, never fresh.
    unknown = {item.lane: item for item in report.states(head_sha=None)}
    assert unknown["agent"].freshness == FRESHNESS_UNKNOWN


def test_a_remediation_after_a_findings_verdict_reads_as_posted():
    comments = [
        Comment(
            id="r",
            body="### Agent review — round 1\n\n**Verdict: NOT clean — one MAJOR.**\n",
            created_at="2026-01-01T00:00:00Z",
        ),
        Comment(
            id="m",
            body="### Agent review remediation — round 1\n\n**F1 — fixed (`abc1234`).**\n",
            created_at="2026-01-01T00:01:00Z",
        ),
    ]
    states = {item.lane: item for item in parse(comments).states(head_sha="abc1234")}
    assert states["agent"].state == STATE_REMEDIATION_POSTED
    # And a clean re-review of the same round replaces the remediation as newest.
    comments.append(
        Comment(
            id="r2",
            body="### Agent review — round 1 (fix verification)\n\n**Verdict: clean.**\n",
            created_at="2026-01-01T00:02:00Z",
        )
    )
    states = {item.lane: item for item in parse(comments).states(head_sha="abc1234")}
    assert states["agent"].state == STATE_CLEAN
    assert states["agent"].round == 1


def test_remediation_dispositions_from_a_real_table():
    body = next(
        item.body for item in _comments("github_comments.json") if "fixed (`d3d2f900`)" in item.body
    )
    found = {item.finding: item for item in parse_remediation(body)}
    assert found["R1"].disposition == "fixed" and found["R1"].sha == "d3d2f900"
    # The reject/defer arm needs a synthetic body: ``rejected`` never appeared on the real
    # sample, so it is asserted against a body written to the convention.
    synthetic = (
        "### Agent review remediation — round 2\n\n"
        "| Finding | Disposition |\n|---|---|\n"
        "| **F1** (MAJOR) | **fixed (`abc1234`).** |\n"
        "| **F2** (MINOR) | **rejected** — the guard is what the gate asks for. |\n"
        "| **F3** (NIT) | **deferred — a follow-up ticket.** |\n"
    )
    dispositions = {item.finding: item.disposition for item in parse_remediation(synthetic)}
    assert dispositions == {"F1": "fixed", "F2": "rejected", "F3": "deferred"}


def test_no_findings_to_remediate_yields_nothing():
    assert (
        parse_remediation("### Agent review remediation — round 1\n\nNo findings to remediate.")
        == ()
    )


def test_non_convention_comments_are_ignored_not_guessed():
    for body in (
        "### Sir Knight Lop the Second — verdict\n\n**Head:** deadbeef\n",
        "### Merge disclosure — `--admin`\n\nMerged as admin.\n",
        "**Addendum to the round above — read the two together.**\n\n**Verdict: clean.**\n",
        "Shipped in **v0.68.15** — [Release](https://example.com)\n",
    ):
        assert parse_comment(Comment(id="x", body=body, created_at="2026-01-01T00:00:00Z")) is None


def test_gitlab_shapes_parse():
    report = parse(_comments("gitlab_notes.json"))
    lanes = {(item.lane, item.kind, item.round): item for item in report.passes}
    assert lanes[("agent", "review", 1)].verdict_class == STATE_FINDINGS_OPEN
    assert lanes[("agent", "review", 2)].verdict_class == STATE_TERMINAL
    assert lanes[("design", "review", 1)].verdict_class == STATE_FINDINGS_OPEN
    assert lanes[("qa", "review", 1)].verdict_class == STATE_CLEAN
    # The list-item field spelling (``- Reviewer:``) and the symbolic scope both land.
    assert lanes[("qa", "review", 1)].reviewer.startswith("qa-tester")
    assert lanes[("agent", "review", 1)].reviewed_head == "72bea95"
    assert {item.lane for item in report.states()} == {"agent", "design", "qa"}


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ("**Verdict: clean** — no BLOCKER, no MAJOR (one NIT).", STATE_CLEAN),
        ("**Verdict: PASS — no FAIL.**", STATE_CLEAN),
        ("**Verdict: clean at `d3d2f900d8` — TERMINAL.**", STATE_TERMINAL),
        ("**Verdict: NOT clean — one MAJOR must be fixed before merge.**", STATE_FINDINGS_OPEN),
        ("**Not terminal — one required change (D1).**", STATE_FINDINGS_OPEN),
        ("**Not safe to merge at this head.**", STATE_FINDINGS_OPEN),
        ("**changes-required (1 major)** — the refusal path answers 200.", STATE_FINDINGS_OPEN),
        ("**Clean — merge-ready.**", STATE_CLEAN),
        ("APPROVE — TERMINAL.", STATE_TERMINAL),
        ("verdict under discussion", STATE_UNSTATED),
        ("", STATE_UNSTATED),
    ],
)
def test_verdict_classification_by_leading_token(verdict, expected):
    assert classify_verdict(verdict) == expected


@pytest.mark.parametrize(
    "reviewed,head,expected",
    [
        ("a6c0d3e8e9", "a6c0d3e8e9f00d", FRESHNESS_FRESH),
        ("a6c0d3e8e9", "ffffffffffff", FRESHNESS_STALE),
        (None, "a6c0d3e8e9", FRESHNESS_UNKNOWN),
        ("a6c0d3e8e9", None, FRESHNESS_UNKNOWN),
        ("a6c0d", "a6c0d3e8e9", FRESHNESS_UNKNOWN),
    ],
)
def test_freshness_needs_a_real_prefix(reviewed, head, expected):
    assert freshness(reviewed, head) == expected


def test_a_labelled_verdict_past_the_field_window_is_still_read():
    """#2112's round-4 review: ``Reviewer:``/``Scope:`` on top, the verdict on line 44.

    The 40-line field window protects ``Reviewer``/``Scope``/``Head`` (a review quotes
    its own prompt further down), but a ``Verdict:`` label with no other spelling in
    the window is the closing summary, and bounding it turned a clean TERMINAL round
    into ``unstated``.
    """
    body = (
        "### Agent review — round 4\n\nReviewer: reviewer on a-model\n"
        "Scope: `46b12d2ff9..039476dff3`\n\n"
        + "".join(f"- note {index}\n" for index in range(50))
        + "\n**Verdict: `clean` — no BLOCKER, no MAJOR — round 4 is TERMINAL on `039476dff3`.**\n"
    )
    assert len(body.splitlines()) > 40
    report = parse([Comment(id="1", created_at="2026-10-08T00:00:00Z", body=body)])
    (item,) = report.passes
    assert item.verdict_class == STATE_TERMINAL
    assert item.reviewed_head == "039476dff3"


def test_a_scope_line_past_the_field_window_is_still_ignored():
    """The window still guards the quoted-prompt case for every field but the verdict."""
    body = (
        "### Agent review — round 1\n\nReviewer: r\nScope: `aaaaaaa..bbbbbbb`\n"
        + "".join(f"- note {index}\n" for index in range(50))
        + "Scope: `ccccccc..ddddddd`\nHead: eeeeeee\n"
    )
    (item,) = parse([Comment(id="1", created_at="2026-10-08T00:00:00Z", body=body)]).passes
    assert item.reviewed_head == "bbbbbbb"

"""The ``/review`` ledger's pure derivations: one fact, one rendered sentence.

These are the React-free equivalents of the desktop Code review pane's
``code-review-model.ts`` (``local-operator-ui`` @ ``76dda0304``), driven over
the SAME wire rows ``code_requests.service.view_rows`` produces. The tests pin
the reference's own precedences — a state pill's draft win, CI's null counts,
a lane's absent round, the stale lane's two SHAs, the row notice's fetch-state
switch, and the two-group partition — because those are the derivations a
second surface drifts from first.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui import review


def _lane(
    lane: str,
    *,
    round: int | None = 1,
    state: str = "clean",
    freshness: str = "fresh",
    reviewed_head: str | None = None,
) -> dict[str, Any]:
    return {
        "lane": lane,
        "round": round,
        "state": state,
        "freshness": freshness,
        "reviewed_head": reviewed_head,
        "qualifier": None,
        "reviewer": None,
        "verdict": None,
    }


def _summary(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "state": "open",
        "draft": False,
        "title": "feat(daemon): reap the observation stamp",
        "head_sha": "9d29452c1100",
        "head_ref": "feat/x",
        "base_ref": "main",
        "author": "damianvtran",
        "ci": {"status": "success", "total": 23, "passed": 23, "failed": 0, "pending": 0},
        "updated_at": 1_000_000,
        "comments": 6,
        "url": "https://github.com/damianvtran/local-operator/pull/1",
    }
    base.update(kw)
    return base


def _row(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "key": "gh:1",
        "url": "https://github.com/damianvtran/local-operator/pull/1",
        "forge": "github",
        "host": "github.com",
        "project": "damianvtran/local-operator",
        "number": 1,
        "relation": "opened",
        "relations": ["opened"],
        "acts": [],
        "mentions": [{"source": "assistant", "count": 1, "first_at": 90.0, "last_at": 100.0}],
        "first_at": 90.0,
        "last_at": 100.0,
        "link_only": False,
        "fetch_state": "ready",
        "summary": _summary(),
        "lanes": [],
        "fetched_at": 1_000.0,
        "stale": False,
        "refresh_error": None,
    }
    base.update(kw)
    return base


# -- state pill -------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "label", "variant"),
    [
        ({"state": "open", "draft": False}, "Open", "info"),
        # The draft win: the forge reports state "open" for a draft, and a
        # draft is the one state that must not read as mergeable.
        ({"state": "open", "draft": True}, "Draft", "outline"),
        ({"state": "merged", "draft": False}, "Merged", "success"),
        ({"state": "closed", "draft": False}, "Closed", "neutral"),
        ({"state": "queued", "draft": False}, "queued", "neutral"),
        ({"state": "  ", "draft": False}, "Unknown", "neutral"),
    ],
)
def test_state_pill_maps_the_forge_state(summary: dict[str, Any], label: str, variant: str) -> None:
    pill = review.state_pill({"summary": summary})
    assert pill is not None
    assert (pill.label, pill.variant) == (label, variant)


def test_state_pill_is_absent_without_a_summary() -> None:
    assert review.state_pill({"summary": None, "link_only": True}) is None


# -- CI ---------------------------------------------------------------------


def test_ci_passing_counts() -> None:
    clause = review.ci_clause({"status": "success", "total": 23, "passed": 23, "failed": 0})
    assert clause is not None
    assert (clause.text, clause.tone) == ("23/23 passed", "plain")


def test_ci_failing_count_outranks_the_status_word() -> None:
    clause = review.ci_clause({"status": "pending", "total": 24, "passed": 22, "failed": 2})
    assert clause is not None
    assert (clause.text, clause.tone) == ("2 failing", "danger")


def test_ci_failure_without_counts_uses_the_status_word() -> None:
    clause = review.ci_clause({"status": "failure", "passed": None, "failed": None})
    assert clause is not None
    assert (clause.text, clause.tone) == ("CI failing", "danger")


@pytest.mark.parametrize(
    "ci",
    [
        {"status": "none", "total": 0, "passed": 0, "failed": 0, "pending": 0},
        # A GitLab pipeline carries no job counts at all (QA round 1, Q-1):
        # `null/null passed` and a red pipeline read as passing were the bugs.
        {"status": "success", "total": None, "passed": None, "failed": None, "pending": None},
        {"status": "unknown", "total": None, "passed": None, "failed": None, "pending": None},
    ],
)
def test_ci_without_a_usable_fraction_keeps_a_status_word(ci: dict[str, Any]) -> None:
    clause = review.ci_clause(ci)
    assert clause is not None
    assert clause.text in {"No checks yet", "CI passed"}
    assert clause.tone in {"muted", "plain"}


def test_ci_pending_count_and_status() -> None:
    running = review.ci_clause({"status": "success", "pending": 2, "total": 5, "passed": 3})
    assert running is not None and running.text == "CI running"
    pending = review.ci_clause({"status": "pending", "pending": 0, "total": None})
    assert pending is not None and pending.text == "CI pending"


def test_ci_absent_or_unclassifiable_claims_nothing() -> None:
    assert review.ci_clause(None) is None
    assert review.ci_clause({"status": "queued"}) is None


# -- comments and age -------------------------------------------------------


def test_comment_clause_is_omitted_at_zero_and_null() -> None:
    assert review.comment_clause(None) is None
    assert review.comment_clause(0) is None
    assert review.comment_clause(1) == "1 comment"
    assert review.comment_clause(6) == "6 comments"


@pytest.mark.parametrize(
    ("seconds_ago", "expected"),
    [
        (0, "updated just now"),
        (30, "updated just now"),
        (60, "updated 1 min ago"),
        (12 * 60, "updated 12 min ago"),
        (3 * 3600, "updated 3 h ago"),
        (2 * 86400, "updated 2 d ago"),
        # A clock skewed ahead of the backend's reads as just now, never as a
        # negative age.
        (-300, "updated just now"),
    ],
)
def test_updated_clause_uses_the_design_s_register(seconds_ago: int, expected: str) -> None:
    updated_at = 1_700_000_000
    now_ms = (updated_at + seconds_ago) * 1000
    assert review.updated_clause(updated_at, now_ms) == expected


# -- lanes ------------------------------------------------------------------


def test_lane_display_names_unknown_lane_stays_raw() -> None:
    assert review.lane_display_name("agent") == "Code"
    assert review.lane_display_name("design") == "Design"
    assert review.lane_display_name("qa") == "QA"
    assert review.lane_display_name("ux") == "UX"
    assert review.lane_display_name("security") == "security"


def test_lane_state_clause_rounds_and_awaiting_rereview() -> None:
    assert (
        review.lane_state_clause(_lane("agent", round=1, state="findings_open"))
        == "Round 1 · findings open"
    )
    assert (
        review.lane_state_clause(_lane("agent", round=2, state="remediation_posted"))
        == "Round 2 · remediation posted · awaiting re-review"
    )
    assert (
        review.lane_state_clause(_lane("agent", round=None, state="awaiting_review"))
        == "No agent review yet"
    )
    # No round number, no claim about how far the review got.
    assert review.lane_state_clause(_lane("agent", round=None, state="clean")) == "clean"
    # A state this build has never seen prints raw rather than disappearing.
    assert (
        review.lane_state_clause(_lane("agent", round=1, state="needs_info"))
        == "Round 1 · needs_info"
    )
    assert (
        review.lane_state_clause(_lane("agent", round=1, state="unstated"))
        == "Round 1 · reviewed — verdict not stated"
    )


def test_lane_freshness_clause_names_both_shas() -> None:
    lane = _lane("agent", round=2, state="clean", freshness="stale", reviewed_head="a3ffd2b99")
    assert (
        review.lane_freshness_clause(lane, "9d29452c1100")
        == "stale — reviewed a3ffd2b, head 9d29452"
    )
    without_head = _lane("agent", round=2, freshness="stale", reviewed_head="a3ffd2b99")
    assert review.lane_freshness_clause(without_head, "") == "stale — reviewed a3ffd2b"
    no_head = _lane("agent", round=2, freshness="stale")
    assert review.lane_freshness_clause(no_head, "9d29452c1100") == "stale"
    fresh = _lane("agent", round=2, freshness="fresh")
    assert review.lane_freshness_clause(fresh, "9d29452c1100") is None
    unknown = _lane("agent", round=2, freshness="unknown")
    assert review.lane_freshness_clause(unknown, "9d29452c1100") is None


def test_round_segments_mark_only_the_latest_round_with_the_state_tone() -> None:
    tones = review.round_segments(_lane("agent", round=3, state="clean"))
    assert tones == ("muted", "muted", "success")
    # A stale clean is not clean now: the latest mark goes muted.
    stale = review.round_segments(_lane("agent", round=2, state="clean", freshness="stale"))
    assert stale == ("muted", "muted")
    assert review.round_segments(_lane("agent", round=0, state="clean")) == ()
    assert review.round_segments(_lane("agent", round=None, state="awaiting_review")) == ()


def test_lane_tone_by_state() -> None:
    assert review.lane_segment_tone("findings_open") == "warning"
    assert review.lane_segment_tone("remediation_posted") == "muted"
    assert review.lane_segment_tone("clean") == "success"
    assert review.lane_segment_tone("terminal") == "success"
    assert review.lane_segment_tone("unstated") == "outline"
    assert review.lane_segment_tone("awaiting_review") == "outline"


def test_lane_lines_order_and_unknown_lane_last() -> None:
    row = _row(
        lanes=[
            _lane("qa", round=1, state="clean"),
            _lane("security", round=1, state="clean"),
            _lane("agent", round=2, state="remediation_posted"),
            _lane("design", round=1, state="findings_open"),
            _lane("ux", round=1, state="terminal"),
        ]
    )
    lines = review.lane_lines(row)
    assert [line.name for line in lines] == ["Code", "Design", "QA", "UX", "security"]
    assert lines[0].clause == "Round 2 · remediation posted · awaiting re-review"
    assert [segment for segment in lines[0].segments] == ["muted", "muted"]


def test_lane_lines_insert_awaiting_review_for_an_open_request() -> None:
    row = _row(summary=_summary(state="open", draft=False), lanes=[_lane("design", round=1)])
    lines = review.lane_lines(row)
    assert [line.name for line in lines] == ["Code", "Design"]
    assert lines[0].clause == "No agent review yet"
    assert lines[0].segments == ()


def test_lane_lines_insert_awaiting_review_for_a_draft() -> None:
    # The wire reports a draft as state "open" + draft true, so the intent has
    # to read the boolean, not a `"draft"` state word the contract never sends.
    row = _row(summary=_summary(state="open", draft=True), lanes=[])
    assert [line.name for line in review.lane_lines(row)] == ["Code"]


def test_a_finished_request_without_lanes_has_no_strip() -> None:
    row = _row(summary=_summary(state="merged", draft=False), lanes=[])
    assert review.lane_lines(row) == ()


# -- relation, activity -----------------------------------------------------


def test_relation_tag_is_silent_for_opened_without_a_path() -> None:
    assert review.relation_tag({"relation": "opened", "via": None}) is None
    # The group header already says "Mentioned".
    assert review.relation_tag({"relation": "mentioned", "via": None}) is None
    assert review.relation_tag({"relation": "inherited", "via": None}) == "inherited"
    assert review.relation_tag({"relation": "unknown", "via": None}) == "unknown — possibly opened"


def test_relation_tag_names_the_subagent_path() -> None:
    row = {
        "relation": "opened",
        "via": {"path": ["coder", "reviewer"], "agent_role": "coder", "label": "job"},
    }
    assert review.relation_tag(row) == "via subagent coder › reviewer"
    flat = {"relation": "opened", "via": {"path": [], "agent_role": "coder", "label": ""}}
    assert review.relation_tag(flat) == "via subagent coder"
    bare = {"relation": "opened", "via": {"path": [], "agent_role": "", "label": ""}}
    assert review.relation_tag(bare) == "via subagent"


def test_acted_tag_joins_the_backend_s_own_act_words() -> None:
    assert review.acted_tag({"acts": []}) is None
    assert review.acted_tag({"acts": ["comment"]}) == "you commented"
    assert review.acted_tag({"acts": ["comment", "merge"]}) == "you commented, merged"
    # An unknown act word is shown raw rather than dropped.
    assert review.acted_tag({"acts": ["rebase"]}) == "you rebase"


# -- grouping ---------------------------------------------------------------


def test_group_rows_partitions_and_orders_both_groups_with_one_comparator() -> None:
    rows = [
        _row(
            key="m-old",
            relation="mentioned",
            last_at=10.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 10.0}],
        ),
        _row(
            key="open",
            relation="opened",
            last_at=5.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 5.0}],
        ),
        _row(
            key="m-new",
            relation="mentioned",
            last_at=30.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 30.0}],
        ),
        _row(
            key="unknown",
            relation="unknown",
            last_at=20.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 20.0}],
        ),
        _row(
            key="inherited",
            relation="inherited",
            last_at=15.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 15.0}],
        ),
        # A real open with an act outranks a real open without one.
        _row(
            key="acted",
            relation="opened",
            acts=["comment"],
            last_at=1.0,
            mentions=[{"source": "assistant", "count": 1, "last_at": 1.0}],
        ),
    ]
    groups = review.group_rows(rows)
    assert [row["key"] for row in groups.opened] == ["acted", "open", "unknown", "inherited"]
    assert [row["key"] for row in groups.mentioned] == ["m-new", "m-old"]


def test_group_rows_tolerates_a_missing_mentions_list() -> None:
    groups = review.group_rows([_row(key="a", mentions=None), _row(key="b", mentions=[])])
    assert [row["key"] for row in groups.opened] == ["a", "b"]


# -- row notice (the precedence that has regressed twice) -------------------


def test_row_notice_link_only_prefers_the_backend_sentence() -> None:
    cooling = "cooling — this host is rate-limited until 2026-10-18T12:00:00Z"
    row = _row(
        link_only=True,
        summary=None,
        fetch_state="unauthenticated",
        reason=cooling,
        link_only_hint="sign in with the gh CLI",
    )
    assert review.row_notice(row) == cooling


def test_row_notice_link_only_falls_back_to_the_per_forge_remedy() -> None:
    row = _row(link_only=True, summary=None, fetch_state="untracked", link_only_hint="Link only")
    assert review.row_notice(row) == "Link only"
    bare = _row(link_only=True, summary=None, fetch_state="untracked", link_only_hint="")
    assert review.row_notice(bare) == "Link only"


@pytest.mark.parametrize(
    ("fetch_state", "extra", "expected"),
    [
        ("pending", {"reason": "sign in"}, None),
        ("ready", {"refresh_error": "network dropped"}, "Couldn't refresh: network dropped"),
        ("ready", {"reason": "no state has been fetched yet"}, "no state has been fetched yet"),
        ("stale", {"refresh_error": "timeout"}, "Stale — last known, not current: timeout"),
        ("stale", {}, "Stale — last known state, not current."),
        ("unauthenticated", {"reason": "sign in with glab"}, "sign in with glab"),
        ("unauthenticated", {}, "sign in with glab"),
        ("cooling", {}, "Cooling — this host is rate-limited; nothing was fetched yet."),
        ("failed", {"reason": "boom"}, "boom"),
        ("failed", {"refresh_error": "timeout"}, "Couldn't refresh: timeout"),
        ("failed", {}, "Couldn't fetch this pass — will retry."),
        ("untracked", {"reason": "not tracked"}, "not tracked"),
    ],
)
def test_row_notice_follows_the_fetch_state(
    fetch_state: str, extra: dict[str, Any], expected: str | None
) -> None:
    row = _row(fetch_state=fetch_state, link_only_hint="sign in with glab", **extra)
    if fetch_state in {"unauthenticated", "untracked", "cooling"} and "reason" not in extra:
        row["link_only"] = True
        row["summary"] = None
    assert review.row_notice(row) == expected


def test_row_notice_legacy_wire_without_fetch_state() -> None:
    link_only = _row(link_only=True, summary=None, reason=None, link_only_hint="Link only")
    link_only.pop("fetch_state")
    assert review.row_notice(link_only) == "Link only"
    tracked = _row(refresh_error="credential rejected — sign in again")
    tracked.pop("fetch_state")
    assert review.row_notice(tracked) == "Couldn't refresh: credential rejected — sign in again"
    bare = _row(reason="why not")
    bare.pop("fetch_state")
    assert review.row_notice(bare) == "why not"


def test_fetch_state_of_rejects_unknown_words() -> None:
    assert review.fetch_state_of({"fetch_state": "ready"}) == "ready"
    assert review.fetch_state_of({"fetch_state": "teleporting"}) is None
    assert review.fetch_state_of({}) is None
    assert review.row_is_loading({"fetch_state": "pending"})
    assert not review.row_is_loading({"fetch_state": "ready"})


# -- identity ---------------------------------------------------------------


def test_identity_label_uses_the_host_s_own_sigil() -> None:
    github = _row(project="damianvtran/local-operator", number=927)
    assert review.identity_label(github) == "damianvtran/local-operator #927"
    gitlab = _row(forge="gitlab", project="minervaai/minerva-skills", number=57)
    assert review.identity_label(gitlab) == "minervaai/minerva-skills !57"


def test_row_title_falls_back_to_the_url() -> None:
    assert review.row_title(_row(summary=_summary(title="a title"))) == "a title"
    assert review.row_title(_row(summary=None)) == _row()["url"]


def test_link_only_remedy_trims_and_defaults() -> None:
    assert review.link_only_remedy({"link_only_hint": "  sign in with gh  "}) == "sign in with gh"
    assert review.link_only_remedy({"link_only_hint": "   "}) == "Link only"
    assert review.link_only_remedy({}) == "Link only"

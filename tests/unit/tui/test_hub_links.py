"""The Hub links sidebar render core: sections, the reserved slot, policy text."""

from __future__ import annotations

from rich.cells import cell_len

from local_operator.tui import hub_links as core


def _snapshot(*links):
    return {"capability": "hub_links:1", "links": list(links)}


def _link(**over):
    base = {
        "kind": "agent",
        "local_id": "a1",
        "name": "coder",
        "link_id": "L1",
        "hub_source": "https://hub.test/v1",
        "visibility": "public",
        "policy": {"subscribe": True, "auto_push": False, "revision": "rev1"},
        "eligibility": {"can_auto_push": False, "reason": "unsupported-precondition"},
        "auto_will_apply": False,
        "auto_will_push": False,
        "pull": {"state": "idle", "enabled": True, "error_class": None},
        "push": {"state": "idle", "enabled": False, "error_class": None},
    }
    base.update(over)
    return base


def test_rows_carry_sections_in_agents_teams_skills_order() -> None:
    rows = core.rows_from_snapshot(
        _snapshot(
            _link(kind="skill", local_id="s1", name="deploy"),
            _link(kind="agent", local_id="a1", name="coder"),
            _link(kind="team", local_id="t1", name="crew"),
        )
    )
    lines = core.render_rows(rows, 80)
    text = "\n".join(str(line) for line in lines)
    assert text.index("Agents") < text.index("Teams") < text.index("Skills")


def test_a_healthy_link_keeps_a_quiet_reserved_mark() -> None:
    """A healthy link is NOT unmarked: the slot is reserved and shows a quiet glyph."""
    rows = core.rows_from_snapshot(_snapshot(_link()))
    line = str(core.render_rows(rows, 80)[1])
    assert line.startswith(core.SLOT_IDLE)  # one reserved slot, muted


def test_the_slot_reflects_aggregate_priority() -> None:
    """blocked > failure > available > busy > applied > healthy in ONE slot."""
    assert (
        core.rows_from_snapshot(
            _snapshot(
                _link(pull={"state": "failed", "enabled": True, "error_class": "merge-refused"})
            )
        )[0].slot_state
        == "blocked"
    )
    assert (
        core.rows_from_snapshot(
            _snapshot(
                _link(pull={"state": "failed", "enabled": True, "error_class": "provider-error"})
            )
        )[0].slot_state
        == "failure"
    )
    assert (
        core.rows_from_snapshot(
            _snapshot(_link(pull={"state": "available", "enabled": True, "error_class": None}))
        )[0].slot_state
        == "available"
    )
    assert (
        core.rows_from_snapshot(
            _snapshot(_link(pull={"state": "merging", "enabled": True, "error_class": None}))
        )[0].slot_state
        == "busy"
    )
    # Aggregate across copies: blocked outranks a busy sibling.
    rows = core.rows_from_snapshot(
        _snapshot(
            _link(pull={"state": "merging", "enabled": True, "error_class": None}),
            _link(pull={"state": "failed", "enabled": True, "error_class": "merge-refused"}),
        )
    )
    assert core.aggregate_slot(rows) == core.SLOT_BLOCKED


def test_policy_text_names_scope_and_both_policies() -> None:
    public = core.rows_from_snapshot(_snapshot(_link(visibility="public")))[0]
    assert public.policy_text() == "Public; Subscribe on; Auto-push off"
    org = core.rows_from_snapshot(
        _snapshot(_link(visibility="org", policy={"subscribe": False, "auto_push": True}))
    )[0]
    assert org.policy_text() == "Private; Subscribe off; Auto-push on"


def test_collapsed_name_carries_no_policy_text() -> None:
    rows = core.rows_from_snapshot(_snapshot(_link(name="coder")))
    line = core.render_rows(rows, 80)[1]
    # The policy text is a dim TAIL, not part of the name span.
    names = [span for span in line.spans if span.style != "dim" and span.style != "reverse"]
    assert all("Subscribe" not in line.plain[span.start : span.end] for span in names)


def test_the_reserved_slot_keeps_row_height_stable_across_states() -> None:
    """A state change repaints the slot; it must not change the row COUNT."""
    base = core.render_rows(core.rows_from_snapshot(_snapshot(_link())), 80)
    changed = core.render_rows(
        core.rows_from_snapshot(
            _snapshot(_link(pull={"state": "available", "enabled": True, "error_class": None}))
        ),
        80,
    )
    assert len(base) == len(changed)
    # And the slot is exactly one cell in both.
    assert str(base[1])[0] == core.SLOT_IDLE and str(changed[1])[0] == core.SLOT_AVAILABLE


def test_loading_only_shows_reserved_text_when_empty() -> None:
    assert "Loading hub links" in str(core.render_rows([], 80, loading=True)[0])
    # With rows present, a refresh keeps them (never a skeleton).
    rows = core.rows_from_snapshot(_snapshot(_link()))
    lines = core.render_rows(rows, 80, loading=True)
    assert any("coder" in str(line) for line in lines)


def test_empty_inventory_says_not_linked() -> None:
    assert "Not linked" in str(core.render_rows([], 80)[0])


def test_a_very_long_name_is_fitted_to_width() -> None:
    rows = core.rows_from_snapshot(_snapshot(_link(name="x" * 200)))
    line = core.render_rows(rows, 40)[1]
    # The row fits its box: the name truncates with an ellipsis and the dim
    # tail gives at narrow width rather than overflowing.
    assert cell_len(line.plain) <= 40
    assert "…" in line.plain


def test_unresolved_legacy_links_are_listed_as_blocked_rows() -> None:
    """A legacy identity-less link appears (blocked, unresolved), never a guess."""
    snapshot = {
        "links": [],
        "unresolved": [
            {
                "kind": "agent",
                "local_id": "a9",
                "name": "legacy",
                "hub_id": "h9",
                "reason": "source-unresolved",
            }
        ],
    }
    rows = core.rows_from_snapshot(snapshot)
    assert len(rows) == 1 and rows[0].link_id is None
    assert rows[0].slot_state == "blocked"
    assert "Not linked" not in str(core.render_rows(rows, 80)[0])


def test_eligibility_and_will_flags_project_into_the_row() -> None:
    snapshot = _snapshot(
        _link(
            eligibility={"can_auto_push": True, "reason": "eligible"},
            auto_will_apply=True,
            auto_will_push=True,
        )
    )
    row = core.rows_from_snapshot(snapshot)[0]
    assert row.auto_will_apply is True and row.auto_will_push is True

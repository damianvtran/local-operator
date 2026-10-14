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


def test_policy_text_is_the_catalogued_accessible_description() -> None:
    """The description comes from `mark.healthy`, not a literal (i18n rule)."""
    row = core.rows_from_snapshot(_snapshot(_link(visibility="public")))[0]
    desc = row.policy_text()
    # The template names item, source, and both policies.
    assert "coder" in desc and "https://hub.test/v1" in desc
    assert "Subscribe On" in desc and "Auto-push Off" in desc


def test_the_compact_scope_label_is_public_or_org_name_private() -> None:
    """Frozen §30: the visible scope is `Public` or `<org> · Private`, NOT the URL."""
    public = core.rows_from_snapshot(_snapshot(_link(visibility="public")))[0]
    assert public.scope_label == "Public"
    org = core.rows_from_snapshot(_snapshot(_link(visibility="org", tenant_name="Acme")))[0]
    assert org.scope_label == "Acme · Private"
    # An org link with no resolved name still shows Private, never the URL.
    unnamed = core.rows_from_snapshot(_snapshot(_link(visibility="org")))[0]
    assert "Private" in unnamed.scope_label and "·" not in unnamed.scope_label


def test_the_rendered_row_does_not_paint_the_raw_api_url() -> None:
    rows = core.rows_from_snapshot(_snapshot(_link(visibility="public")))
    line = core.render_rows(rows, 100)[1]
    assert "https://" not in line.plain
    assert "Public" in line.plain


def test_collapsed_name_carries_no_policy_text() -> None:
    rows = core.rows_from_snapshot(_snapshot(_link(name="coder")))
    line = core.render_rows(rows, 80)[1]
    # The policy text is a muted TAIL, not part of the name span.
    names = [span for span in line.spans if span.style not in ("dim", "muted", "reverse")]
    assert all("Subscribe" not in line.plain[span.start : span.end] for span in names)


def test_the_tail_ink_is_the_supplied_muted_style_not_dim() -> None:
    """D4: the widget supplies the theme's `muted`; the core paints it, not `dim`."""
    rows = core.rows_from_snapshot(_snapshot(_link()))
    line = core.render_rows(rows, 80, tail_style="blue")[1]
    assert any(span.style == "blue" for span in line.spans)


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
    entry = core.render_rows(rows, 40)[1]
    # Round-2 D8: at a width too narrow for name + scope + policy on ONE line, the
    # policy pair WRAPS to a second indented line (the namespace context note:
    # wrap rather than truncate). Every physical line still fits the box, and the
    # policy words are not lost to truncation.
    for physical in str(entry).split("\n"):
        assert cell_len(physical) <= 40, (physical, cell_len(physical))
    assert "…" in entry.plain
    assert "Subscribe" in entry.plain and "Auto-push" in entry.plain


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


def test_rows_carry_policy_revision_and_eligibility() -> None:
    """D7: a row carries the CAS revision and the eligibility fact/refusal reason.

    The toggle needs the revision it last saw and the service-computed
    eligibility; without them the panel could not perform the CAS write or name
    the reason an enable is refused.
    """
    row = core.rows_from_snapshot(
        _snapshot(
            _link(
                policy={"subscribe": True, "auto_push": False, "revision": "rev-xyz"},
                eligibility={"can_auto_push": False, "reason": "unsafe-publication"},
            )
        )
    )[0]
    assert row.policy_revision == "rev-xyz"
    assert row.can_auto_push is False
    assert row.eligibility_reason == "unsafe-publication"


def test_narrow_row_wraps_the_policy_pair_rather_than_truncating() -> None:
    """D8: at the 40x24 content floor the policy pair moves to a second line."""
    rows = core.rows_from_snapshot(_snapshot(_link(name="coder")))
    entry = core.render_rows(rows, 24)[1]
    physical = str(entry).split("\n")
    assert len(physical) >= 2, physical
    assert all(cell_len(line) <= 24 for line in physical), physical
    # Both policy words AND both values survive on the wrapped detail line(s).
    rest = "\n".join(physical[1:])
    assert "Subscribe" in rest and "Auto-push" in rest
    assert "On" in rest and "Off" in rest


def test_every_hinted_key_has_a_live_action() -> None:
    """D17: every key the hint row teaches is a REAL binding, and vice versa.

    The hint row once advertised `enter opens policy` for a stub with no effect.
    This pins the invariant that the hinted keys and the panel's live key set are
    the same set (the confirm key excepted: it is taught by the PROMPT, not the
    ordinary row, because it only acts while the consent is armed).
    """

    from local_operator.i18n.keys import tui_hub_links as _k
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets import hub_links_panel as panel_mod

    _t = tr("tui.hub_links")
    bound = {str(binding[0]) for binding in panel_mod.HubLinksPanel.BINDINGS}
    # The ordinary hint row teaches these ACTIONS; the confirm key is taught by
    # the armed prompt's own block.
    taught = {
        "s": _t(_k.hint_subscribe()),
        "a": _t(_k.hint_auto_push()),
        "r": _t(_k.hint_refresh()),
        "escape": _t(_k.hint_close()),
    }
    for key in taught:
        assert key in bound, f"hinted key {key!r} has no binding"
    # Every actionable binding is either taught on the row or is the prompt-only
    # confirm / a navigation key.
    navigation = {"up", "down"}
    for key in bound:
        assert (
            key in taught or key in navigation or key == panel_mod.CONFIRM_EGRESS_KEY or key == "q"
        ), f"binding {key!r} is neither hinted nor an intentional per-key exception"
    # And the dead key is genuinely gone.
    assert "enter" not in bound

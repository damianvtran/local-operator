"""Paint existing sanitized Python ownership receipts, never call a browser.

Source: owner-approved HTTP QA fixtures for PR2455,
commit441464d465c52cef9813d55ec32f16a973e3d7cc.
This closes no loaded-extension or new-HTTP execution gate. The real standard
ToolCard/help producer runs inside OperatorApp with its production stylesheet.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.tui.test_host_speed_settings_rendered import (
    _prepay_session_import_warmup,
    capture_pair,
    ui_proof,
)

# Imported pytest fixtures are intentional: both modules share the same narrow
# no-service boundary, including the parent session-warmup override. Neither
# module imports OperatorApp or a provider at collection time.
__all__ = ["_prepay_session_import_warmup", "ui_proof"]

REPLAYS = {
    "happy": {
        "version": 1,
        "session_match": True,
        "scope_missing": False,
        "generation_current": False,
        "generation_previous": True,
        "generation_ancestry": True,
        "request_ancestry_count": 1,
    },
    "missing-local": {"version": 1, "local_scope_missing": True},
    "unknown": {"version": 1, "session_match": False, "scope_missing": True},
    "foreign": {"version": 1, "session_match": False},
    "unsupported": {"version": 1, "error_code": "ownership_inspection_unsupported"},
    "unavailable": {"version": 1, "error_code": "ownership_inspection_unavailable"},
    "invalid": {"version": 1, "error_code": "ownership_inspection_invalid"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [100, 60])
async def test_standard_ownership_help_and_receipts_real_app(
    ui_proof, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from local_operator.tools import builtin
    from local_operator.tools.tool_docs import render_tool_doc
    from local_operator.tui.widgets.editor import Editor
    from local_operator.tui.widgets.tool_card import ToolCard
    from tests.unit.tui.conftest import await_session_boot

    proof = ui_proof
    app = proof.app
    case = f"ownership-{width}x30"
    phases: set[str] = set()
    proof.expected_phases = {
        "collapsed-help",
        "expanded-help",
        "happy",
        "missing-local",
        "unknown",
        "unsupported",
    }
    proof.case_note = (
        "Painted responses: happy/missing-local/unknown/unsupported; "
        "foreign/unavailable/invalid are logical replay checks only. "
        "Compact production receipt cropping is not repaired by this test."
    )
    # This is a publication fixture, not live host discovery. The normal
    # builder/parameters/help renderer remain real; no browser resource or
    # client is constructed or invoked, and no capability is claimed live.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "ui_browser_advertisable", lambda: False)
    tool = builtin.build_browser_tool(None)
    assert tool is not None
    help_text = render_tool_doc(tool)
    assert "inspect_ownership" in help_text

    async with app.run_test(size=(width, 30)) as pilot:
        await await_session_boot(app)
        assert app._session is proof.session
        editor = app.query_one(Editor)

        async def shot(phase):
            phases.add(phase)
            await capture_pair(proof, pilot, case, phase)

        help_card = ToolCard("synthetic-help", "read", {"path": "tool://browser"})
        help_card.mark_done(help_text)
        app._append_block(help_card)
        await pilot.pause()
        await shot("collapsed-help")
        assert help_card.toggle_expanded()
        help_card.scroll_visible(immediate=True)
        await pilot.pause()
        assert "inspect_ownership" in help_card._build_content(width - 4).plain
        await shot("expanded-help")
        seen: set[str] = set()
        for name, payload in REPLAYS.items():
            card = ToolCard(f"synthetic-{name}", "browser", {"action": "inspect_ownership"})
            # Full approved sanitized payloads are replayed unchanged. Do not
            # strip foreign fields here and call that a privacy producer test.
            result = builtin._text(
                f"synthetic-{name}",
                "browser",
                json.dumps(payload, sort_keys=True),
                details=payload,
            )
            assert result.details == payload
            assert result.text == json.dumps(payload, sort_keys=True)
            card.mark_done(result.text, result.details)
            app._append_block(card)
            await pilot.pause()
            assert card.toggle_expanded()
            # NON-ANIMATED on purpose (architect decision, Option A). The default
            # animate=True runs a 0.2s scroll_y animation, and two failed
            # attempts to OBSERVE that animation (a pre-loop idle settle, then a
            # per-sample animator/scroll-target await) both still measured the
            # identical 2-row offset delta. With animate=False Textual takes its
            # non-animated branch and writes scroll_y and scroll_target_y
            # together, so there is no in-flight motion for the two samples to
            # straddle. The sticky tail re-lands with animate=False anyway.
            card.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            # Logical payload identity is not a painted-field assertion. The
            # production compact line may crop in the standard card; preserve
            # that real presentation for independent frame review, not a
            # prettier multi-line receipt that production never emitted.
            assert json.loads(result.text) == payload
            seen.add(name)
            if name in {"happy", "missing-local", "unknown", "unsupported"}:
                await shot(name)
        assert seen == set(REPLAYS), "a missing replay must not silently pass"
        assert phases == proof.expected_phases
        editor.focus()
        await pilot.pause()
        assert editor.has_focus

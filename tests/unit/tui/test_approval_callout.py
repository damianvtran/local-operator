"""The approval card's inline auto-mode handles — the rendered half.

The spec (`approval-auto-design-spec.md`) is a MEASURED layout: which rows exist
at 100x30, which single row replaces three at 60x16, and the promise that a
setting gesture never moves an answer row or answers the pending request. These
tests drive the real ``OperatorApp`` and read the real card text, because the
failure this feature is most exposed to is a plausible model that paints
something else.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.approval import ApprovalPolicy, ApprovalPrompt
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_approvals_ux import (
    GatedSession,
    _boot,
    _gate,
    _settle,
    _wait_prompt,
)


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — these tests WRITE one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


def _lines(card: ApprovalPrompt) -> list[str]:
    return card.render_lines_for_test()


def _saved_override(config_dir: Path) -> str | None:
    path = config_dir / "sessions" / "sess" / "approvals.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["override"]


async def _card(pilot, app: OperatorApp) -> ApprovalPrompt:
    await _wait_prompt(pilot, app)
    return app.query(ApprovalPrompt).first()


async def _settled_height(pilot, card: ApprovalPrompt) -> int:
    """The card's height once two consecutive frames agree.

    The first frames after a mount can be measured mid-composition; comparing
    against one of those makes a later, CORRECT frame look like growth. Two
    equal reads is the cheapest stable observation.
    """
    last = 0
    for _ in range(30):
        await pilot.pause()
        current = card.size.height
        if current and current == last:
            return current
        last = current
    return last


@pytest.mark.asyncio
async def test_both_destinations_render_at_the_authored_widths(
    config_dir: Path,
) -> None:
    """100x30 gets the full sentence pair; 60x16 keeps both, compactly.

    The whole feature exists to be DISCOVERABLE, and a destination that is
    clipped to ``sess…`` teaches nothing. Both scopes must be readable at the
    two sizes the design measured.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: rm -rf ./build"))
        card = await _card(pilot, app)
        text = "\n".join(_lines(card))
        assert "Set auto for this session [S]" in text
        assert "Set auto for all sessions [G]" in text
        assert "Approvals: Ask — global default." in text
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 16)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: rm -rf ./build"))
        card = await _card(pilot, app)
        lines = _lines(card)
        # One combined row: the status and both destinations share it.
        assert "[S]session" in lines[-1] and "[G]all sessions" in lines[-1]
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
async def test_the_compact_rail_keeps_all_three_answers_visible(
    config_dir: Path,
) -> None:
    """60x16 used to show ONE answer and a hint row. It shows all three now.

    The rail is a compact rendering of the existing list — same order, same
    keys — not a new interaction. Deny and Allow all must be readable without
    a scroll, because an authorisation prompt that hides a refusal is the C3/D1
    defect this card's whole allocation exists to prevent.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 16)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)
        rail = "\n".join(_lines(card))
        assert "❯ y.Allow" in rail
        assert "n.Deny" in rail
        assert "A.Allow all" in rail
        # No movement/enter hint competes for the row: the answer letters are
        # visible in the rail, which is the spec's reason for shedding it.
        assert "enter answer" not in rail
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
async def test_s_saves_this_session_and_leaves_the_request_parked(
    config_dir: Path,
) -> None:
    """`s` changes the RULE for the next request; it must not answer THIS one.

    The acknowledgement replaces the status sentence inline (no toast, no extra
    row), the override is written beside the transcript so a resume reopens in
    it, and the future is still pending until the user answers.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)
        selected = card.state.selected

        await pilot.press("S")
        await pilot.pause()

        assert not pending.done(), "a setting gesture answered the pending request"
        assert app._explicit_approvals_mode == "auto"
        assert app._approve_all is True
        assert _saved_override(config_dir) == "auto"
        text = "\n".join(_lines(card))
        assert "Auto saved for this session, with this conversation. Answer this request." in text
        # The answer rows did not move: the state sentence took the
        # spacer/status slot, not a new line. (Card GROWTH is pinned separately,
        # on the dedicated 60x16 frame, where it is the frame's whole point.)
        assert card.state.selected == selected
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False

        # The NEXT gate is auto and raises no card.
        second = asyncio.ensure_future(_gate(session)("bash", "run: two"))
        assert await asyncio.wait_for(second, 2) is True
        await _settle(pilot)
        assert not app.query(ApprovalPrompt)


@pytest.mark.asyncio
async def test_g_sets_the_global_default_and_leaves_the_request_parked(
    config_dir: Path,
) -> None:
    """`g` writes the machine default AND clears this conversation's record.

    Case f: otherwise a conversation holding an ``ask`` override would watch the
    operator choose "auto for all sessions" and go on asking, while the card
    claimed it inherited the new default.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # Start from a conversation override, so "globally auto" has something
        # to clear rather than passing trivially.
        app._explicit_approvals_mode = "ask"
        app._set_approve_all(False)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)

        await pilot.press("G")
        await pilot.pause()

        assert not pending.done(), "a setting gesture answered the pending request"
        assert app._explicit_approvals_mode is None
        assert app._approvals_default_auto is True
        assert app._approve_all is True
        assert _saved_override(config_dir) is None
        saved = yaml.safe_load((config_dir / "config.yml").read_text())
        assert saved["values"]["tool_approval_mode"] == "auto"
        text = "\n".join(_lines(card))
        assert "Auto set as global default; this session inherits it. Answer this request." in text
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_a_refused_destination_leaves_the_card_answerable(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No authority means no write — and the pending request still works.

    The refusal is painted in the same reserved slot, both destinations stay
    identifiable, and a repeated press repeats the explanation without writing
    config or resolving the future.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # An attached pane whose runtime route is absent: exactly the topology
        # where this window holds no authority and cannot pretend otherwise.
        monkeypatch.setattr(app, "_gate_is_owned_elsewhere", lambda: True)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)

        await pilot.press("S")
        await pilot.pause()
        await pilot.pause()

        assert not pending.done()
        text = "\n".join(_lines(card))
        assert "This window cannot authorise auto." in text
        assert "Unchanged: Ask (global default)." in text
        assert "Set auto for this session [S]" in text
        assert "Set auto for all sessions [G]" in text
        assert app._explicit_approvals_mode is None
        assert _saved_override(config_dir) is None

        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_the_mode_line_distinguishes_session_override_from_global() -> None:
    """The provenance wording is a fact about the gate, not decoration.

    A session override and the global default can hold the SAME mode word; the
    reader still has to know which one will survive a restart of the other.

    A real card, unmounted: `__init__` builds its future on the running loop, so
    this is an async test, and `_state_text` is the production renderer rather
    than a copy of its table.
    """
    cases = [
        (ApprovalPolicy(mode="ask", provenance="global"), "Approvals: Ask — global default."),
        (
            ApprovalPolicy(mode="ask", provenance="session"),
            "Approvals: Ask — this session overrides global Auto.",
        ),
        (
            ApprovalPolicy(mode="auto", provenance="session"),
            "Approvals: Auto — this session overrides global Ask.",
        ),
        (
            ApprovalPolicy(mode="auto", provenance="global"),
            "Approvals: Auto — global default; answer this pending request.",
        ),
    ]
    for policy, expected in cases:
        card = ApprovalPrompt("bash", "run: one", policy=policy)
        assert card._state_text(96) == expected


@pytest.mark.asyncio
async def test_a_setting_gesture_keeps_the_screen_unscrolled_at_60x16(
    config_dir: Path,
) -> None:
    """The five-row card is a promise: no growth, no scrollbar.

    A callout that bought a row from the dock would either clip the footer or
    make the screen scrollable — both are visible failures on the measured
    16-row geometry.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 16)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)
        before = await _settled_height(pilot, card)
        # Measured against the frame BEFORE the gesture, not against zero: the
        # dock may already be tight for reasons this change did not cause, and
        # what the callout must not do is make it worse.
        before_virtual = app.screen.virtual_size.height

        await pilot.press("S")
        await pilot.pause()

        assert card.size.height == before, "the saved acknowledgement grew the card"
        assert (
            app.screen.virtual_size.height == before_virtual
        ), "the callout changed the screen's virtual height"
        assert "Auto(session) saved" in "\n".join(_lines(card))
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)


class _StubView:
    """A settings page that only records what the app posts back."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def post_message(self, message: Any) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_the_settings_page_route_reports_the_owners_verdict(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attached owner's refusal must not be painted as a page success.

    The page hides the transcript the owner's receipt lands in, so the reply is
    posted back as a message. ``ok`` is read from the owner's own sentence:
    ``tool approvals: auto`` at the head is the one phrase both hosts emit for
    a switch that took.
    """
    from local_operator.tui.widgets.settings_view import SettingsGateRoute

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        monkeypatch.setattr(app, "_gate_is_owned_elsewhere", lambda: True)

        async def refusing_route(command: str, argument: str, images: Any) -> dict[str, str]:
            assert (command, argument) == ("approvals", "default auto")
            return {"text": "keeping tool approvals: ask — set with /approvals in this session"}

        stub = _StubView()
        await app._route_approvals_default_worker(stub, refusing_route, SettingsGateRoute)
        assert stub.messages and stub.messages[-1].ok is False
        assert "keeping tool approvals" in stub.messages[-1].text
        assert app._approvals_default_auto is False

        async def accepting_route(command: str, argument: str, images: Any) -> dict[str, str]:
            return {"text": "tool approvals: auto — saved to config.yml: tool_approval_mode auto"}

        await app._route_approvals_default_worker(stub, accepting_route, SettingsGateRoute)
        assert stub.messages[-1].ok is True
        assert app._approvals_default_auto is True
        assert app._approve_all is True


@pytest.mark.asyncio
async def test_the_page_routes_auto_to_an_attached_owner(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page write alone is only a DEFAULT when another process holds the gate.

    `/settings` hides the transcript the owner's receipt lands in, so the page
    must ASK the owner (via `default auto`) and paint the answer. This asserts
    the ask is made on the real page gesture; the answer's rendering is the
    handler test below.
    """
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.test_config_change_notice import _choose

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        monkeypatch.setattr(app, "_gate_is_owned_elsewhere", lambda: True)
        routed: list[Any] = []
        monkeypatch.setattr(
            app, "route_approvals_default_from_page", lambda view: routed.append(view)
        )
        app._open_settings_view()
        await pilot.pause()
        view = app.query(SettingsView).first()

        _choose(view, "tool_approval_mode", "auto")
        await pilot.pause()

        assert routed, "the page wrote auto and never asked the session's owner to adopt it"


@pytest.mark.asyncio
async def test_the_page_paints_a_refusal_as_a_refusal(config_dir: Path) -> None:
    """A refused route keeps the page's existing refusal alert.

    The page must not invent a second error vocabulary, and it must not leave
    the provisional "saved" line standing when the owner said no.
    """
    from local_operator.tui.widgets.settings_view import SettingsGateRoute, SettingsView

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._open_settings_view()
        await pilot.pause()
        view = app.query(SettingsView).first()

        view.on_settings_gate_route(
            SettingsGateRoute(False, "tool approvals: unchanged — no authority")
        )
        assert view._notice_alert is True
        assert "no authority" in view._error

        view.on_settings_gate_route(SettingsGateRoute(True, "tool approvals: auto — saved"))
        assert view._notice_alert is False
        assert "tool approvals: auto" in view._notice


@pytest.mark.asyncio
async def test_a_failed_global_write_is_not_painted_as_saved(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card's `g` must not claim a default it could not write.

    The runtime's own failure receipt begins "tool approvals: auto", which is
    how an invoker that parsed the sentence used to acknowledge a failed
    loosening as saved (agent review, 2026-10-23). Here the failure is local:
    the save itself fails, the mode does not move, and the card must say so in
    its reserved slot while leaving the pending request parked.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)
        monkeypatch.setattr(
            app,
            "_save_approvals_default",
            lambda auto: ("", "could not save default: OSError"),
        )

        await pilot.press("G")
        await pilot.pause()

        assert not pending.done(), "a setting gesture answered the pending request"
        text = "\n".join(_lines(card))
        assert "Not saved: could not save default: OSError" in text
        assert "saved as global default" not in text
        assert app._approvals_default_auto is False
        assert app._explicit_approvals_mode is None
        # Still answerable.
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_a_session_save_failure_is_disclosed_not_hidden(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode moved; the record did not. The card must say both.

    "Auto saved for this session, with this conversation" is a durability claim.
    On a sidecar write failure the live gate IS auto, so "Not saved … unchanged"
    would be the opposite lie. The third sentence names the state exactly.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        card = await _card(pilot, app)
        monkeypatch.setattr(app, "_persist_approval_override", lambda source, mode: "OSError")

        await pilot.press("S")
        await pilot.pause()

        assert not pending.done()
        text = "\n".join(_lines(card))
        assert "Auto applied — NOT saved for resume (OSError)." in text
        assert "saved for this session" not in text
        assert app._approve_all is True
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_a_routed_failure_receipt_is_not_read_as_a_save(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact defect: a failed default write's text starts "tool approvals: auto".

    The routed result now carries ``ok`` in ``data``; the card reads it instead
    of the sentence. This stub returns the runtime's real failure wording, so a
    regression to prefix-matching fails here.
    """
    from tests.unit.tui.test_approvals_ux import _wait_prompt

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        await _wait_prompt(pilot, app)
        card = app.query(ApprovalPrompt).first()

        async def refusing_owner(command, args, images):  # type: ignore[no-untyped-def]
            return {
                "kind": "notice",
                "text": (
                    "tool approvals: auto — NOT saved (OSError); this session keeps ask "
                    "and no other session was told"
                ),
                "style": "warning",
                "data": {
                    "approval_mode": "ask",
                    "approval_provenance": "global",
                    "ok": False,
                    "approval_saved_for_resume": False,
                    "approval_reason": "OSError",
                },
            }

        # The real session object, with only its route replaced: a stand-in
        # session would also have to satisfy the app's shutdown contract.
        monkeypatch.setattr(session, "route_shared_slash", refusing_owner, raising=False)
        await app._route_approvals_setting(card, "default auto", "global")

        text = "\n".join(_lines(card))
        assert "Not saved: OSError" in text
        assert "saved as global default" not in text
        assert app._approvals_default_auto is False
        assert app._approve_all is False
        assert not pending.done()
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_failed_default_write_keeps_the_conversation_record(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar clearing is the LAST step, not the first.

    The config write can fail (read-only dir). Clearing the conversation's
    record before it ran left a resume following the old config while this
    session followed a mode the receipt had just said was not saved. The record
    must survive the failure and the in-memory choice must be restored.
    """
    from tests.unit.tui.test_approvals_ux import _notices, _submit

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals auto")
        assert _saved_override(config_dir) == "auto"

        monkeypatch.setattr(
            "local_operator.config.ConfigManager.set_config_value",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only file system")),
        )
        await _submit(pilot, app, "/approvals default ask")

        assert app._explicit_approvals_mode == "auto", "the recorded choice was not restored"
        assert _saved_override(config_dir) == "auto", "the sidecar was cleared by a failed write"
        assert (
            app._approve_all is True
        ), "the gate moved even though its record could not be released"
        assert "NOT saved" in _notices(app)[-1]


@pytest.mark.asyncio
async def test_the_dense_rail_paints_the_third_label_inside_the_card(config_dir: Path) -> None:
    """24x16 is the frame the height guard used to exclude from the dense plan.

    The guard tested the TERMINAL height while the measurable box was the
    screen's 14 rows, so the rail fell back to one line, painted 30 cells into
    20, and ``A.Allow all`` clipped to a bare ``A`` that still answered while no
    Allow-all label was visible. Assert on the COMPOSITED frame, not on the
    renderer's string, because the clip happened at paint time (M2/Q2/D1).
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(24, 16)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: rm -rf ./build"))
        card = await _card(pilot, app)
        await pilot.pause()

        assert card.size.width == 20, card.size
        region = card._body.region
        strips = app.screen._compositor.render_strips()
        painted = [
            strip.text[region.x : region.x + region.width]
            for strip in strips[region.y : region.y + region.height]
        ]
        rail = [row for row in painted if "y.Allow" in row]
        assert rail, painted
        assert any("n.Deny" in row for row in rail), rail
        assert any("A.Allow all" in row for row in painted), painted
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
async def test_the_allow_consequence_follows_the_saved_policy(config_dir: Path) -> None:
    """A parked card must not describe the policy it was raised under after a save.

    Pressing S/G changes what the NEXT request will do while this one still
    asks; leaving "ask again next time" under Allow tells the reader the
    opposite of the confirmation two rows above (U3/D3). The swap is
    description-only: the card's height must not move.
    """
    from tests.unit.tui.test_approvals_ux import _submit

    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: rm -rf ./build"))
        card = await _card(pilot, app)
        # Wait for the ROOMY plan: the first frames after a mount can be
        # measured while the dock is still growing, and the descriptions are
        # the first thing the budget sheds (the card's own documented ladder).
        before = ""
        for _ in range(40):
            await pilot.pause()
            before = "\n".join(_lines(card))
            if "run this call and ask again next time" in before:
                break
        assert "run this call and ask again next time" in before, before
        before_rows = len(_lines(card))

        await pilot.press("S")
        await pilot.pause()
        assert card._policy.mode == "auto", card._policy
        after = "\n".join(_lines(card))
        assert "run this call; later calls will not ask" in after, after
        assert "run this call and ask again next time" not in after, after
        assert len(_lines(card)) == before_rows, "the swap moved a row"

        # The restore direction: tightening re-arms the prompt and the wording
        # must say so again. Focus the composer first — Enter while the card
        # holds focus ANSWERS it (that is the card's own Enter), not submits.
        from local_operator.tui.widgets.editor import Editor

        app.query_one(Editor).focus()
        await pilot.pause()
        await _submit(pilot, app, "/approvals ask")
        restored = ""
        for _ in range(40):
            await pilot.pause()
            restored = "\n".join(_lines(card))
            # Wait for the SETTLED frame: the first non-empty read after a
            # command can still be the previous paint, and the claim here is
            # about the wording the reader ends up with.
            if "run this call and ask again next time" in restored:
                break
        assert "run this call and ask again next time" in restored, (
            restored,
            card.is_mounted,
            card.display,
            app._approval is card,
        )
        await pilot.press("n")
        await asyncio.wait_for(pending, 2)


@pytest.mark.asyncio
async def test_a_clear_failure_with_no_record_does_not_claim_a_lost_choice(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n1: "could not be cleared" is only true when something was stored.

    A fresh conversation follows the file already, so a failed release of a
    non-existent record changes nothing the user can lose. Saying their saved
    choice could not be cleared would send them looking for a choice they never
    made.
    """
    from tests.unit.tui.test_approvals_ux import _notices, _submit

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert app._explicit_approvals_mode is None

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr("local_operator.session.approval_policy.write_override", _explode)
        await _submit(pilot, app, "/approvals default ask")

        receipt = _notices(app)[-1]
        # The DEFAULT write landed; only the pointless no-choice document
        # failed, and that is not a durability loss the reader needs told.
        assert "tool approvals: ask — saved to" in receipt, receipt
        assert "saved choice could not be cleared" not in receipt, receipt


@pytest.mark.asyncio
async def test_lowercase_letters_type_rather_than_change_the_gate(config_dir: Path) -> None:
    """`s`/`g` are ordinary letters; only `S`/`G` reach the setting.

    The lowercase pair were instant accelerators in the first cut, which made
    the most common letters in a sentence write the machine-wide default with no
    confirmation — the footgun `A` is already uppercase to avoid (agent review
    round 1, M1). Lowercase must fall through to the composer and leave BOTH the
    gate and the record untouched.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: rm -rf ./build"))
        await _card(pilot, app)

        await pilot.press("s")
        await pilot.press("g")
        await pilot.pause()

        assert app._approve_all is False, "a lowercase letter loosened the gate"
        assert app._explicit_approvals_mode is None
        assert _saved_override(config_dir) is None, "a lowercase letter wrote a record"

        await pilot.press("n")
        await asyncio.wait_for(pending, 2)

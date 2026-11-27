"""``/guard`` presentation safety (Guard RFC v0.9 §5.2.3, slice L2b).

Focus on the non-sharing choice, Esc = Decide later (still pending), input
within 500 ms ignored, and a report is confirmed only by the person's press.
A bare host app is enough here: these rows assert behaviour, not layout (the
frames come from the real ``OperatorApp`` in the PR evidence).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button

from local_operator.guard import consent as C
from local_operator.guard import surface
from local_operator.guard.reports import CONFIRMED, DRAFTED, GuardReports
from local_operator.hub_org.eligibility import OrgRef
from local_operator.tui.widgets import guard_view
from local_operator.tui.widgets.guard_view import DEBOUNCE_S, GuardScreen

TENANT = "ten_acme"


def _prompt_state(host: C.GuardConsentHost) -> str:
    row = host.prompt(TENANT, 1)
    assert row is not None
    return row.state


def _draft_state(reports: GuardReports, draft_id: str) -> str:
    draft = reports.get(draft_id)
    assert draft is not None
    return draft.state


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(surface, "_team_orgs", lambda _r: [OrgRef(TENANT, "Acme", "member")])
    return tmp_path


def _queue(root: Path) -> C.GuardConsentHost:
    host = C.GuardConsentHost(root, transport=None, mirror=False)
    host.offer(
        C.ConsentOffer.from_wire(
            {
                "tenant": TENANT,
                "org_display_name": "Acme",
                "collection_rev": 1,
                "revocation_gen": 0,
                "scopes": ["decisions"],
                "added_scopes": [],
                "rendered_terms_digest": "sha256:" + "ab" * 32,
                "reason": "new_org",
            }
        )
    )
    return host


class _Host(App[None]):
    def __init__(self, screen: GuardScreen) -> None:
        super().__init__()
        self._guard = screen
        self.result: list[str | None] = []

    def on_mount(self) -> None:
        self.push_screen(self._guard, self.result.append)


@pytest.mark.asyncio
async def test_focus_starts_on_dont_share_and_typeahead_is_ignored(root: Path) -> None:
    host = _queue(root)
    clock = _Clock()
    app = _Host(GuardScreen(root, clock=clock))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "guard-no"
        # A typed-ahead Enter lands inside the 500 ms window: ignored.
        await pilot.press("enter")
        await pilot.pause()
        assert _prompt_state(host) == "pending"
        assert isinstance(app.screen, GuardScreen)
        # Even a press on Share inside the window does nothing.
        app.screen.query_one("#guard-yes", Button).press()
        await pilot.pause()
        assert _prompt_state(host) == "pending"


@pytest.mark.asyncio
async def test_escape_is_decide_later_not_a_decline(root: Path) -> None:
    host = _queue(root)
    app = _Host(GuardScreen(root, clock=_Clock()))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result == ["later"]
    assert _prompt_state(host) == "pending"
    assert host.sticky(TENANT) is None


@pytest.mark.asyncio
async def test_dont_share_after_the_debounce_declines_locally(root: Path) -> None:
    host = _queue(root)
    clock = _Clock()
    app = _Host(GuardScreen(root, clock=clock))
    async with app.run_test() as pilot:
        await pilot.pause()
        clock.now += DEBOUNCE_S + 0.01
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
    assert _prompt_state(host) == "declined"
    assert host.sticky(TENANT) == "declined"


@pytest.mark.asyncio
async def test_a_report_is_sent_only_by_the_persons_press(root: Path) -> None:
    _queue(root).decline(TENANT, 1, surface="tui")
    reports = GuardReports(root)
    draft, _ = reports.draft(TENANT, "credential_exposure")
    clock = _Clock()
    app = _Host(GuardScreen(root, clock=clock))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "guard-no"  # Discard
        assert _draft_state(reports, draft.id) == DRAFTED
        clock.now += DEBOUNCE_S + 0.01
        app.screen.query_one("#guard-yes", Button).press()
        for _ in range(4):
            await pilot.pause()
    assert _draft_state(reports, draft.id) == CONFIRMED


@pytest.mark.asyncio
async def test_non_team_card_uses_trial_days_and_touches_no_guard_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(surface, "_team_orgs", lambda _r: [])
    monkeypatch.setattr(guard_view, "_has_team", lambda _r: False)
    calls: list[int] = []

    def loader() -> dict[str, object]:
        calls.append(1)
        return {"eligible": True, "trial_days": 21}

    app = _Host(GuardScreen(tmp_path, eligibility_loader=loader, clock=_Clock()))
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.pause()
        cta = app.screen.query_one("#guard-cta", Button)
        assert "21" in str(cta.label)
    assert calls == [1]
    assert not (tmp_path / "guard").exists()

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


@pytest.fixture(autouse=True)
def _a_person_at_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows run as a person unless they set a marker (pytest may itself run in an
    agent's shell, which would make every screen read-only)."""

    for name in ("LOCAL_OPERATOR_AGENT_SHELL", "LOCAL_OPERATOR_CONSOLE_SURFACE"):
        monkeypatch.delenv(name, raising=False)


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
        # UX U5: focus is "Decide later", so one stray Enter cannot discard it.
        assert app.focused is not None and app.focused.id == "guard-later"
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


# ---------------------------------------------------------------------------
# Round 1 remediation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["LOCAL_OPERATOR_AGENT_SHELL", "LOCAL_OPERATOR_CONSOLE_SURFACE"])
@pytest.mark.asyncio
async def test_an_agent_driven_terminal_cannot_accept_and_the_prompt_stays_pending(
    root: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    """Security S-1: the capability is derived, never a literal."""

    monkeypatch.setenv(marker, "1")
    # The nested-session escape hatch is for harnesses, not people.
    monkeypatch.setenv("LOCAL_OPERATOR_ALLOW_NESTED_SESSION", "1")
    assert surface.host_can_answer() is False
    host = _queue(root)
    clock = _Clock()
    screen = GuardScreen(root, clock=clock)
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not screen.query("#guard-yes")  # read-only: no Share button
        assert screen.query("#guard-no") and screen.query("#guard-later")
        lines = [str(w.render()) for w in screen.query(".guard-line")]
        assert any("an agent may be driving this terminal" in line for line in lines)
        # Even an increasing answer that reaches the engine is refused there.
        clock.now += DEBOUNCE_S + 0.01
        await screen._answer(yes=True)
        await pilot.pause()
    assert _prompt_state(host) == "pending"
    assert [r["code"] for r in host.refusals()] == ["guard_answer_not_admitted"]


@pytest.mark.asyncio
async def test_an_agent_driven_terminal_cannot_confirm_a_report(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    _queue(root).decline(TENANT, 1, surface="tui")
    reports = GuardReports(root)
    draft, _ = reports.draft(TENANT, "credential_exposure")
    clock = _Clock()
    screen = GuardScreen(root, clock=clock)
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not screen.query("#guard-yes")
        clock.now += DEBOUNCE_S + 0.01
        await screen._answer(yes=True)
        await pilot.pause()
    assert _draft_state(reports, draft.id) == DRAFTED


def test_a_person_at_a_plain_terminal_can_answer() -> None:
    assert surface.host_can_answer() is True


class _Styled(_Host):
    """The real lop stylesheet and palette: layout rows need them (D1)."""

    CSS_PATH = str(Path(guard_view.__file__).parents[1] / "local_operator.tcss")

    def get_css_variables(self) -> dict[str, str]:
        from local_operator.tui.theme import tcss_variable_map

        return {**super().get_css_variables(), **tcss_variable_map()}


@pytest.mark.parametrize("size", [(80, 24), (100, 34), (60, 20)])
@pytest.mark.asyncio
async def test_the_buttons_stay_on_screen_at_every_size(root: Path, size: tuple[int, int]) -> None:
    """Design D1 / UX U1: the body shrinks and scrolls; the buttons are painted."""

    host = C.GuardConsentHost(root, transport=None, mirror=False)
    host.offer(
        C.ConsentOffer.from_wire(
            {
                "tenant": TENANT,
                "org_display_name": "Acme Robotics",
                "collection_rev": 1,
                "revocation_gen": 0,
                "scopes": ["decisions", "posture", "task_time", "review_cycles", "reports"],
                "added_scopes": [],
                "rendered_terms_digest": "sha256:" + "ab" * 32,
                "reason": "new_org",
            }
        )
    )
    screen = GuardScreen(root, clock=_Clock())
    app = _Styled(screen)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.pause()
        width, height = size
        for button in screen.query(Button):
            region = button.region
            assert region.height > 0, button.id
            assert region.bottom <= height and region.right <= width, (button.id, region)
        assert screen.query_one("#guard-title").region.y >= 0


@pytest.mark.asyncio
async def test_share_again_keeps_the_org_listed_and_skips_a_revoke_pending_org(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX U2 + review R1-2: one predicate; the org stays listed as "will ask"."""

    monkeypatch.setattr(
        surface,
        "_team_orgs",
        lambda _r: [OrgRef("ten_a", "Alpha", "member"), OrgRef("ten_b", "Beta", "member")],
    )
    host = C.GuardConsentHost(tmp_path, transport=None, mirror=False)
    for tenant, name in (("ten_a", "Alpha"), ("ten_b", "Beta")):
        host.offer(
            C.ConsentOffer.from_wire(
                {
                    "tenant": tenant,
                    "org_display_name": name,
                    "collection_rev": 1,
                    "revocation_gen": 0,
                    "scopes": ["decisions"],
                    "added_scopes": [],
                    "rendered_terms_digest": "sha256:" + "ab" * 32,
                    "reason": "new_org",
                }
            )
        )
    host.decline("ten_b", 1, surface="tui")
    # Alpha: accepted locally, then a revoke that could not be sent.
    host._exec(
        "INSERT INTO consents (tenant, collection_rev, scopes, granted_at, state)"
        " VALUES ('ten_a', 1, '[]', 0, 'revoke_pending')",
        (),
    )
    host._exec("UPDATE prompts SET state = 'superseded' WHERE tenant = 'ten_a'", ())
    host._exec("INSERT INTO sticky VALUES ('ten_a', 'revoked', 'local', 0)", ())
    screen = GuardScreen(tmp_path, clock=_Clock())
    monkeypatch.setattr(guard_view.GuardScreen, "_ask_again_now", _no_offer_yet)
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        stopped = screen._stopped()
        assert stopped is not None and stopped.tenant == "ten_b"
        screen._armed_at = 0.0
        screen.query_one("#guard-again", Button).press()
        for _ in range(4):
            await pilot.pause()
        states = {s.tenant: s.state for s in surface.org_states(tmp_path)}
        lines = " ".join(str(w.render()) for w in screen.query(".guard-line"))
    assert host.sticky("ten_a") == "revoked"  # untouched: its stop is unsent
    assert host.sticky("ten_b") is None
    assert states == {"ten_a": surface.STATE_PAUSED, "ten_b": surface.STATE_WILL_ASK}
    assert "Beta will ask you again soon" in lines


async def _no_offer_yet(self: GuardScreen) -> None:
    return None


@pytest.mark.asyncio
async def test_a_damaged_consent_store_shows_a_card_instead_of_crashing(root: Path) -> None:
    """Review R1-3."""

    path = C.guard_consent_path(root)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a database at all" * 10)
    screen = GuardScreen(root, clock=_Clock())
    assert screen._mode == "unavailable"
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "guard-close"


@pytest.mark.asyncio
async def test_esc_on_a_prompt_gives_the_same_feedback_as_decide_later(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX U4."""

    _queue(root)
    app = _Host(GuardScreen(root, clock=_Clock()))
    seen: list[str] = []
    monkeypatch.setattr(app, "notify", lambda message, **_k: seen.append(str(message)))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert seen and "Still waiting" in seen[0]


@pytest.mark.asyncio
async def test_working_is_painted_and_answers_disabled_before_the_request(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX U3: measured from inside a slow accept."""

    import threading

    _queue(root)
    release = threading.Event()
    seen: dict[str, object] = {}
    screen = GuardScreen(root, clock=_Clock())

    class _SlowHost:
        def accept(self, *_a: object, **_k: object) -> object:
            seen["disabled"] = [b.disabled for b in screen.query(Button)]
            seen["working"] = any("Working" in str(w.render()) for w in screen.query(".guard-line"))
            release.wait(5)
            return C.AcceptOutcome("failed", "unavailable")

    monkeypatch.setattr(screen, "_host", lambda: _SlowHost())
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen._armed_at = 0.0
        screen.query_one("#guard-yes", Button).press()
        for _ in range(3):
            await pilot.pause()
        release.set()
        for _ in range(3):
            await pilot.pause()
    assert seen["disabled"] and all(seen["disabled"])  # type: ignore[arg-type]
    assert seen["working"] is True


@pytest.mark.asyncio
async def test_the_confirmed_message_names_the_kind(root: Path) -> None:
    """QA Q1: with a second draft on the card, the line says which one was saved."""

    _queue(root).decline(TENANT, 1, surface="tui")
    reports = GuardReports(root)
    reports.draft(TENANT, "credential_exposure")
    reports.draft(TENANT, "excessive_guardrail")
    screen = GuardScreen(root, clock=_Clock())
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen._armed_at = 0.0
        screen.query_one("#guard-yes", Button).press()
        for _ in range(4):
            await pilot.pause()
        lines = [str(w.render()) for w in screen.query(".guard-line")]
    assert any(
        line.startswith("A suspected exposed key or credential: saved on this computer, not sent.")
        for line in lines
    )
    # Copy round 2, C11: never a promise that Guard sends it on its own.
    assert any("Guard asks you once more before sending it to Acme" in line for line in lines)


def test_the_badge_names_a_waiting_report() -> None:
    """UX U6."""

    sharing = [surface.OrgState(TENANT, "Acme", "member", surface.STATE_SHARING)]
    assert guard_view.badge_text(sharing, reports_waiting=1) == "Guard: report waiting in /guard"
    assert guard_view.badge_text(sharing) == "Guard: sharing"
    unknown = surface.OrgState(TENANT, "Acme", "member", surface.STATE_PAUSED, "server_new")
    assert guard_view.indicator_text(unknown) == "Guard: paused"  # C3: no invented cause


@pytest.mark.asyncio
async def test_the_cta_slot_holds_its_place_while_eligibility_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design D4: "Checking…" in a fixed slot, then the CTA; Close never moves."""

    import threading

    monkeypatch.setattr(surface, "_team_orgs", lambda _r: [])
    monkeypatch.setattr(guard_view, "_has_team", lambda _r: False)
    gate = threading.Event()

    def loader() -> dict[str, object]:
        gate.wait(5)
        return {"eligible": True, "trial_days": 30}

    screen = GuardScreen(tmp_path, eligibility_loader=loader, clock=_Clock())
    app = _Styled(screen)
    async with app.run_test(size=(100, 34)) as pilot:
        await pilot.pause()
        cta = screen.query_one("#guard-cta", Button)
        close_x = screen.query_one("#guard-close", Button).region.x
        assert str(cta.label) == "Checking…" and cta.disabled
        width = cta.region.width
        gate.set()
        for _ in range(6):
            await pilot.pause()
        assert "30" in str(cta.label) and not cta.disabled
        assert cta.region.width == width
        assert screen.query_one("#guard-close", Button).region.x == close_x


# ---------------------------------------------------------------------------
# Round 2 remediation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "footer"),
    [
        ({"eligible": True, "trial_days": 21, "tenant_name": "Dana\nHome"}, True),
        ({"eligible": True, "trial_days": 21}, False),
    ],
)
@pytest.mark.asyncio
async def test_non_team_footer_names_the_org_or_says_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict[str, object], footer: bool
) -> None:
    """user-console#98 copy C2 (shared string): scoped to the org the card is
    about, never "this account"; no org name -> no footer, and focus stays put."""

    monkeypatch.setattr(surface, "_team_orgs", lambda _r: [])
    monkeypatch.setattr(guard_view, "_has_team", lambda _r: False)
    app = _Host(GuardScreen(tmp_path, eligibility_loader=lambda: body, clock=_Clock()))
    async with app.run_test() as pilot:
        await pilot.pause()
        focused = app.screen.focused
        for _ in range(6):
            await pilot.pause()
        lines = [str(w.render()) for w in app.screen.query(".guard-line")]
        assert app.screen.focused is focused
    assert not any("this account" in line for line in lines)
    expected = "Guard collects nothing for Dana Home."  # one line: the newline is inert
    assert (expected in lines) is footer
    assert footer or not any("collects nothing" in line for line in lines)


@pytest.mark.asyncio
async def test_escape_while_an_answer_is_in_flight_keeps_the_card(root: Path) -> None:
    """UX round 2, U9: Esc during "Working…" must not hide the outcome."""

    _queue(root)
    screen = GuardScreen(root, clock=_Clock())
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen._busy = True
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen and app.result == []
        screen._busy = False
        await pilot.press("escape")
        await pilot.pause()
    assert app.result == ["later"]


@pytest.mark.asyncio
async def test_a_prompt_that_follows_share_again_clears_the_will_ask_line(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 2, U10: the card that arrives replaces "will ask you again soon"."""

    host = _queue(root)
    host.decline(TENANT, 1, surface="tui")
    screen = GuardScreen(root, clock=_Clock())
    app = _Host(screen)

    def org_re_offers(_root: Path, *, boundary: str) -> None:
        del boundary
        host.offer(
            C.ConsentOffer.from_wire(
                {
                    "tenant": TENANT,
                    "org_display_name": "Acme",
                    "collection_rev": 2,
                    "revocation_gen": 0,
                    "scopes": ["decisions"],
                    "added_scopes": [],
                    "rendered_terms_digest": "sha256:" + "ab" * 32,
                    "reason": "new_org",
                }
            )
        )

    monkeypatch.setattr(C, "check_point", org_re_offers)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen._armed_at = 0.0
        screen.query_one("#guard-again", Button).press()
        for _ in range(8):
            await pilot.pause()
        lines = [str(w.render()) for w in screen.query(".guard-line")]
        assert screen._mode == "prompt"
    assert not any("will ask you again soon" in line for line in lines)


@pytest.mark.asyncio
async def test_a_read_only_overview_with_a_stopped_org_says_where_to_answer(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 2, U12: Share again is hidden, so the line says where it lives."""

    _queue(root).decline(TENANT, 1, surface="tui")
    screen = GuardScreen(root, clock=_Clock(), can_answer=False)
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not screen.query("#guard-again")
        lines = [str(w.render()) for w in screen.query(".guard-line")]
    assert any("run lop in a separate terminal window" in line for line in lines)


@pytest.mark.asyncio
async def test_a_request_replaced_before_the_answer_says_so(root: Path) -> None:
    """Review round 2, R2-3: the digest-bound claim's refusal explains the swap."""

    host = _queue(root)
    screen = GuardScreen(root, clock=_Clock())
    app = _Host(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        # The org replaces the request (same rev, new immutables) after render.
        host.offer(
            C.ConsentOffer.from_wire(
                {
                    "tenant": TENANT,
                    "org_display_name": "Acme",
                    "collection_rev": 1,
                    "revocation_gen": 3,
                    "scopes": ["decisions"],
                    "added_scopes": [],
                    "rendered_terms_digest": "sha256:" + "ab" * 32,
                    "reason": "new_org",
                }
            )
        )
        screen._armed_at = 0.0
        screen.query_one("#guard-yes", Button).press()
        for _ in range(6):
            await pilot.pause()
        lines = [str(w.render()) for w in screen.query(".guard-line")]
    assert "Acme changed this request. Check the new one below." in lines
    assert _prompt_state(host) == "pending"


def test_a_receipt_from_an_earlier_day_carries_its_date() -> None:
    """Copy round 2, C14: "Not you?" only helps if the person can place the time."""

    import time as _time

    now = _time.mktime((2026, 11, 27, 12, 0, 0, 0, 0, -1))
    today = _time.mktime((2026, 11, 27, 6, 41, 0, 0, 0, -1))
    earlier = _time.mktime((2026, 11, 25, 6, 41, 0, 0, 0, -1))
    assert guard_view._receipt_time(today, now) == "06:41"
    assert guard_view._receipt_time(earlier, now) == "2026-11-25 06:41"
    assert guard_view._receipt_time(0.0, now) == ""

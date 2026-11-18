"""``/community`` and ``/orgs`` through the REAL ``OperatorApp`` (stylesheet applied).

Each test opens the page the way a person does — the command typed into the
session composer — over ``community_fakes.FakeBackend`` and asserts on the
widget state the frames describe (TUI frames T1, T2, T5, T9, Amendment A).
The rendered frames themselves are produced by ``scripts/community_shot.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.community.service import PostOutcome
from local_operator.tui import community_backend
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.community_view import CommunityView
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.community_fakes import HOSTILE_HELP, FakeBackend
from tests.unit.tui.test_app_pilot import FakeSession, _await_session, _factory


async def _open(app: Any, pilot: Any, command: str = "/community") -> CommunityView:
    await _await_session(app, pilot)
    editor = app.query_one(Editor)
    editor.load_text(command)
    editor.focus()
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()
    return app.query_one(CommunityView)


async def _type(pilot: Any, text: str) -> None:
    for char in text:
        await pilot.press("space" if char == " " else char)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    monkeypatch.setenv("LOCAL_OPERATOR_COMMUNITY", "1")
    fake = FakeBackend()
    monkeypatch.setattr(community_backend, "LiveCommunityBackend", lambda _cfg: fake)
    return fake


@pytest.mark.asyncio
async def test_dark_community_is_an_ordinary_unknown_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_COMMUNITY", raising=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _await_session(app, pilot)
        assert app._dispatchable_slash("/community") is False
        assert app._dispatchable_slash("/orgs") is False
        assert not app.query(CommunityView)


@pytest.mark.asyncio
async def test_home_lists_channels_initial_first_and_esc_closes_without_leaving(
    backend: FakeBackend,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        assert [c.name for c in view.channels][:3] == ["welcome-and-rules", "general", "help"]
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(CommunityView)
        assert backend.left == 0, "closing the page is not leaving"


@pytest.mark.asyncio
async def test_composer_checking_cancel_restores_text(backend: FakeBackend) -> None:
    backend.submit_gate = asyncio.Event()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        view.channel_index = 2  # #help
        view._open_channel()
        await pilot.press("tab", "tab")
        await _type(pilot, "hello there")
        await pilot.press("enter")
        await pilot.pause()
        composer = view.composer()
        assert composer is not None and composer.status == "checking"
        assert view._input.value == "", "↵ clears the composer at once"
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert composer.status == "cancelled"
        assert view._input.value == "hello there"
        assert app.query(CommunityView), "the first esc cancels; it never leaves"
        backend.submit_gate.set()


@pytest.mark.asyncio
async def test_refusal_is_sender_only_and_f2_explains(backend: FakeBackend) -> None:
    backend.next_outcome = PostOutcome(
        "refused", category="harassment_or_abuse", reference="MOD-7Q4K-2M9D"
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("tab", "tab")
        await _type(pilot, "draft")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        assert "Harassment or abuse" in _painted(view._status)
        assert view._input.value == "draft"
        await pilot.press("f2")
        await pilot.pause()
        assert view.overlay == "why"
        assert "MOD-7Q4K-2M9D" in view._overlay_text.render().plain  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_leading_slash_is_never_posted(backend: FakeBackend) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _open(app, pilot)
        await pilot.press("tab", "tab")
        await _type(pilot, "/help")
        await pilot.press("enter")
        await pilot.pause()
        assert backend.submitted == []


@pytest.mark.asyncio
async def test_join_needs_the_explicit_agreement(backend: FakeBackend) -> None:
    backend.joined = False
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("j")
        await pilot.pause()
        assert view.overlay == "join"
        await pilot.press("tab", "enter")  # Join while unticked
        await pilot.pause()
        assert backend.joins == [] and view.join_focus == 0, "↵ on disabled Join goes back"
        await pilot.press("space", "tab", "enter")
        for _ in range(4):
            await pilot.pause()
        assert backend.joins == ["2026-11"]
        assert view.overlay is None and view.home.joined


@pytest.mark.asyncio
async def test_orgs_defaults_to_home_org_and_picking_a_private_org_keeps_bridge_flow(
    backend: FakeBackend,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot, "/orgs")
        for _ in range(3):
            await pilot.pause()
        every = [*view.orgs_private, *view.orgs_communities]
        assert every[view.orgs_index].name == "Tran Robotics"
        assert [e.kind for e in view.orgs_communities] == ["community"]
        await pilot.press("enter")
        await pilot.pause()
        assert backend.saved_selection == "t_tran"
        assert not app.query(CommunityView)


@pytest.mark.asyncio
async def test_leave_defaults_to_stay(backend: FakeBackend) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("s", "tab")
        await pilot.pause()
        assert view.overlay == "settings" and view.settings_tab == 2
        await pilot.press("enter")
        await pilot.pause()
        assert backend.left == 0 and view.overlay is None


def _painted(widget: Any) -> str:
    """Every segment the widget would hand the terminal, joined."""
    from textual.geometry import Region

    region = Region(0, 0, widget.outer_size.width, widget.outer_size.height)
    return "".join(segment.text for strip in widget.render_lines(region) for segment in strip)


@pytest.mark.asyncio
async def test_hostile_post_and_name_reach_the_terminal_without_control_sequences(
    backend: FakeBackend,
) -> None:
    # R1-3: OSC 52 (clipboard), ESC[2J (clear), OSC 0 (title) and SGR blink in a
    # stranger's post and display name, through the real app and page.
    backend.histories["c_help"] = list(HOSTILE_HELP)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        view.channel_index = next(i for i, c in enumerate(view.channels) if c.name == "help")
        view._open_channel()
        for _ in range(4):
            await pilot.pause()
        painted = _painted(view._pane_text)
        assert "Mallory" in painted and "hi  there" in painted
        for payload in ("\x1b", "\x9b", "PWNED", "cm0gLXJmIH4K"):
            assert payload not in painted, payload
        # The model the pane reads is the sanitised copy; the backend's is exact.
        assert "\x1b" not in (view.messages["c_help"] or [])[0].text
        assert "\x1b" in backend.histories["c_help"][0].text


@pytest.mark.asyncio
async def test_orgs_checks_the_viewed_org_apart_from_the_cursor(backend: FakeBackend) -> None:
    # R1-6 (D21): from the community page the community is the one viewed,
    # while the cursor still defaults to the home org (Amendment A).
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("g")
        for _ in range(4):
            await pilot.pause()
        rows = view._paint_orgs().plain.splitlines()
        assert any(row.startswith("  ✓ ") and "Local Operator Community" in row for row in rows)
        assert any(row.startswith("▸   ") and "Tran Robotics" in row for row in rows)


@pytest.mark.asyncio
async def test_narrow_header_drops_whole_key_hints(backend: FakeBackend) -> None:
    # R1-7: at 60 columns the name and the hints never run together.
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        view = await _open(app, pilot)
        header = _painted(view._header)
        assert "connected g" not in header and "? ke" not in header.replace("? keys", "")
        # D1: the help hint is the one that never yields.
        assert header.rstrip().endswith("? keys")


# -- design review round 1 (D1, D3) -------------------------------------------

#: The composer column at the default 80x24 (rail 22 + its border, the page's
#: and the column's padding): measured from the real layout by
#: ``test_composer_column_is_the_measured_width`` so the per-locale fit test
#: below cannot drift from the box it claims to fit.
COMPOSER_CELLS_80 = 53
#: The header row at 80 columns (the page's and the mode's padding), measured
#: by the same test.
HEADER_CELLS_80 = 76

#: Sample values for every placeholder the composer rows interpolate.
_SAMPLES = {
    "channel": "help",
    "wait": "0:42",
    "time": "10:42",
    "version": "2026-12",
    "code": "community_unreachable",
}


def _locale_rows(locale: str) -> dict[str, str]:
    """Every composer-row message of ``locale``, rendered with sample values."""
    from local_operator.i18n import catalogues
    from local_operator.i18n import runtime as msg_runtime

    source = catalogues.load_catalogue(locale, "tui.community")
    rows: dict[str, str] = {}
    for key, message in source.items():
        short = key.removeprefix("tui.community.")
        if not short.startswith(("status_", "audience_")):
            continue
        params: dict[str, object] = dict(_SAMPLES)
        if short == "status_refused":
            params["category"] = source["tui.community.category_spam_or_promotion"]
        rows[short] = msg_runtime.render_message(message, params, locale)
    return rows


#: Rows with no tail clause: they wrap onto the status row's second line.
_WRAPPED = {"status_local_command", "status_restricted", "status_signin"}
#: Rows whose tail is the reason they exist: a locale that drops the clause
#: separator here would lose it to the cut, so the test refuses that too.
_MUST_PIN = {
    "audience_private",
    "audience_public",
    "status_moderation",
    "status_rate_burst",
    "status_rate_conversation",
    "status_rate_daily",
    "status_rate_other",
    "status_rate_probation",
    "status_refused",
}


@pytest.mark.parametrize("locale", ["en", "es", "fr", "hi", "ru", "ur", "vi", "zh-CN"])
def test_composer_rows_keep_their_tail_at_80_columns(locale: str) -> None:
    # D1: every status/audience row of every shipped catalogue, fitted to the
    # real 53-cell column, keeps its essential tail whole (countdown, category
    # + f2, ↵ retry, "checked first") and never exceeds the box; the two rows
    # with no tail wrap within the status row's 2 lines instead.
    from rich.cells import cell_len
    from rich.console import Console
    from rich.text import Text

    from local_operator.tui.widgets.community_view import fit_pinned, pinned_tail

    console = Console(width=COMPOSER_CELLS_80, color_system=None)
    failures: list[str] = []
    for key, line in sorted(_locale_rows(locale).items()):
        if key in _WRAPPED:
            wrapped = Text(line).wrap(console, COMPOSER_CELLS_80)
            if len(wrapped) > 2:
                failures.append(f"{key}: {len(wrapped)} lines")
            continue
        # The refusal's tail starts at the category, the way the page pins it.
        anchor = _category_label(locale) if key == "status_refused" else ""
        tail = pinned_tail(line, anchor)
        if not tail:
            if key in _MUST_PIN:
                failures.append(f"{key}: no tail clause in {line!r}")
            elif len(Text(line).wrap(console, COMPOSER_CELLS_80)) > 2:
                failures.append(f"{key}: no tail and more than 2 lines")
            continue
        fitted = fit_pinned(line, tail, COMPOSER_CELLS_80)
        # The clause itself must survive whole; its leading " · " may go when
        # the lead is dropped entirely (fit_pinned's narrow branch).
        if cell_len(fitted) > COMPOSER_CELLS_80 or not fitted.endswith(tail.removeprefix(" · ")):
            failures.append(f"{key}: {fitted!r} ({cell_len(fitted)} cells)")
        if key.startswith("status_rate") and "0:42" not in fitted:
            failures.append(f"{key}: countdown lost in {fitted!r}")
    assert not failures, failures


def _category_label(locale: str) -> str:
    from local_operator.i18n import catalogues

    return catalogues.load_catalogue(locale, "tui.community")[
        "tui.community.category_spam_or_promotion"
    ]


@pytest.mark.parametrize("locale", ["en", "es", "fr", "hi", "ru", "ur", "vi", "zh-CN"])
def test_header_keeps_the_help_hint_at_80_columns(locale: str) -> None:
    # D1: hints yield ctrl+k first, then g / /orgs; ? / /keys never does.
    from rich.cells import cell_len

    from local_operator.i18n import catalogues
    from local_operator.i18n import runtime as msg_runtime
    from local_operator.tui.widgets.community_view import SEPARATOR, fit_header

    source = catalogues.load_catalogue(locale, "tui.community")
    state = source["tui.community.state_connected"]
    left = msg_runtime.render_message(
        source["tui.community.header"],
        {"community": "Local Operator Community", "state": state},
        locale,
    )
    for key in ("header_keys", "header_keys_typing"):
        right = source[f"tui.community.{key}"]
        for width in (HEADER_CELLS_80, 58):
            line = fit_header(left, right, width)
            last = right.split(SEPARATOR)[-1]
            assert cell_len(line) == width
            assert line.rstrip().endswith(last), (key, width, line)


def test_pinned_tail_cuts_only_the_lead() -> None:
    from rich.cells import cell_len

    from local_operator.tui.widgets.community_view import fit_pinned, pinned_tail

    line = "◷ Ralentissez : 10 par minute pour vous · envoi dans 0:42"
    fitted = fit_pinned(line, pinned_tail(line), 40)
    assert fitted.endswith("… · envoi dans 0:42") and cell_len(fitted) <= 40
    # Wide glyphs are measured in cells, not characters.
    wide = "◷ 慢一点：你每分钟最多发10条消息哦 · 0:42 后发送"
    fitted = fit_pinned(wide, pinned_tail(wide), 24)
    assert cell_len(fitted) <= 24 and fitted.endswith(" · 0:42 后发送")
    # Narrower than the tail: the tail's end (the value) survives.
    assert fit_pinned(line, pinned_tail(line), 8).endswith("0:42")
    assert fit_pinned("short", "", 53) == "short"


@pytest.mark.asyncio
async def test_composer_column_is_the_measured_width(backend: FakeBackend) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        assert view._input.region.width == COMPOSER_CELLS_80
        assert view._status.size.width == view._audience.size.width == COMPOSER_CELLS_80
        assert view._header.size.width == HEADER_CELLS_80


@pytest.mark.asyncio
async def test_rate_limit_keeps_its_countdown_in_the_real_row(backend: FakeBackend) -> None:
    backend.next_outcome = PostOutcome(
        "rate_limited", submission_id="k", scope="probation", retry_after_s=42
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("tab", "tab")
        await _type(pilot, "hi")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        status = _painted(view._status).rstrip()
        assert "send in 0:4" in status, status
        # The composer has the keys: the header offers the typed forms only.
        assert _painted(view._header).rstrip().endswith("/keys")
        assert view._status.has_class("-tone-warning")
        assert "checked first" in _painted(view._audience)


@pytest.mark.asyncio
async def test_read_only_channel_leads_with_the_reason(backend: FakeBackend) -> None:
    # D3 / T10: the reason is on the audience row, the composer offers nothing
    # to type into, and tab never lands on it (so no "[composer] ↵ send").
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        view.channel_index = 0
        view._open_channel()
        await pilot.pause()
        composer = view.composer()
        assert composer is not None and composer.status == "read_only"
        assert "only moderators and staff post" in _painted(view._audience)
        assert view._input.disabled and view._input.placeholder == ""
        for _ in range(3):
            await pilot.press("tab")
            await pilot.pause()
            assert view.area != "composer"
            assert "↵ send" not in _painted(view._footer)


@pytest.mark.parametrize(
    ("outcome", "status", "tail"),
    [
        (
            PostOutcome("rate_limited", scope="probation", retry_after_s=42),
            "rate_limited",
            " · send in 0:42",
        ),
        (
            PostOutcome("refused", category="spam_or_promotion"),
            "refused",
            "Spam or undisclosed promotion · f2 why",
        ),
        (PostOutcome("moderation_unavailable"), "moderation_unavailable", " · ↵ retry"),
    ],
)
def test_the_page_pins_the_whole_tail_clause(outcome: PostOutcome, status: str, tail: str) -> None:
    # The page's OWN choice of tail, through the renderable it paints, at a
    # width that forces a cut: the countdown keeps "send in" (an anchor on the
    # value alone once cut it to "…0:42"), the refusal keeps its category.
    from rich.console import Console

    from local_operator.tui import community_model as model
    from local_operator.tui.widgets.community_view import PinnedRow

    composer = model.Composer(status=status, outcome=outcome)  # type: ignore[arg-type]
    composer.until = composer.clock() + 41.5
    line, pinned, _tone = CommunityView(FakeBackend())._status_row(composer, "help")
    console = Console(width=40, color_system=None)
    with console.capture() as captured:
        console.print(PinnedRow(line, pinned))
    painted = captured.get().rstrip("\n")
    assert painted.endswith(tail) and "…" in painted, painted


@pytest.mark.asyncio
async def test_header_follows_the_composer_the_moment_tab_lands_there(
    backend: FakeBackend,
) -> None:
    # D3 (mixed focus signal): focus moves after the paint `tab` triggers, so
    # the header kept "? keys" — a key that types "?" — beside "[composer]".
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        view = await _open(app, pilot)
        await pilot.press("tab", "tab")
        assert view.area == "composer"
        header = _painted(view._header).rstrip()
        assert header.endswith("/keys") and "? keys" not in header, header

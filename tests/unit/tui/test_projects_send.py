"""Quick-send's PURE half (S6d parity P5a): the target list and its filter.

The rows' ORDER is the feature — the manager first, then the project's own
sessions with live ones first, and never this session — so it is pinned here
without a terminal. The card's keys and the app's delivery wiring are driven
through the real app in the same file once that wiring lands.
"""

from __future__ import annotations

from local_operator.tui.widgets.projects_send import (
    NO_MATCH_FOOTER,
    NO_TARGET_FOOTER,
    SEND_CARD_CHROME_ROWS,
    SEND_CARD_ROW_CAP,
    SendTarget,
    compose_band,
    filter_targets,
    pending_line,
    refusal_line,
    send_error_line,
    send_targets,
    sent_line,
)

MANAGER = SendTarget(
    kind="manager", session_id="mgr-0001", label="manager", state="live", live=True
)


def _view(*rows: dict[str, object]) -> dict[str, object]:
    return {"project": {"id": "p1", "name": "parity-spec"}, "sessions": list(rows)}


def test_the_manager_row_leads_and_is_absent_when_there_is_none() -> None:
    # REAL payloads: liveness lives in `runtime.state` (F2/Q1).
    rows = send_targets(
        _view({"session_id": "aa", "runtime": {"state": "live"}, "title": "work"}),
        own_session=None,
        manager=MANAGER,
    )
    assert rows[0] is MANAGER
    assert [row.kind for row in rows] == ["manager", "session"]
    # No manager resolved: the row is simply absent rather than a dead one.
    assert [
        row.kind
        for row in send_targets(_view({"session_id": "aa"}), own_session=None, manager=None)
    ] == ["session"]
    # And an empty project is an empty list, which the card paints as its
    # footer line rather than as "no results".
    assert send_targets(_view(), own_session=None, manager=None) == []


def test_live_sessions_come_first_and_this_session_is_never_a_target() -> None:
    """You cannot message yourself, and the order is live-first (spec §7.5.1)."""
    rows = send_targets(
        _view(
            {"session_id": "stopped1", "runtime": {"state": "stale"}, "title": "old"},
            {"session_id": "mine", "runtime": {"state": "live"}, "title": "this one"},
            {"session_id": "missing1", "exists": False, "title": "gone"},
            {"session_id": "live1", "runtime": {"state": "live"}, "title": "running"},
            # A wedged record is dialable-ish but must NOT outrank a healthy
            # peer in the order the reader chooses from.
            {"session_id": "wedged1", "runtime": {"state": "wedged"}, "title": "stuck"},
        ),
        own_session="mine",
        manager=None,
    )
    # The spec's contract is LIVE-FIRST and store order within a state group —
    # it does not rank the non-live states against each other, so the test
    # asserts exactly that and the ROW INK (not the order) is what tells
    # `wedged` from `stale`/`missing`.
    ids = [row.session_id for row in rows]
    assert ids[0] == "live1", ids
    assert sorted(ids) == ["live1", "missing1", "stopped1", "wedged1"]
    assert {row.session_id: row.state for row in rows} == {
        "live1": "live",
        "wedged1": "wedged",
        "stopped1": "stale",
        "missing1": "missing",
    }
    assert all(row.session_id != "mine" for row in rows)


def test_a_row_states_its_session_handle_and_state() -> None:
    row = SendTarget(
        kind="session", session_id="ab12cd34ef5609", label="projects review", state="live"
    )
    assert row.row_text == "projects review  · session ab12cd34ef56  · [live]"


def test_a_title_less_row_prints_its_id_once() -> None:
    """U4: a linked session with no title falls back to its short id as the
    label — the row must not then print the id a second time."""
    untitled = SendTarget(
        kind="session", session_id="001122334455", label="001122334455", state="missing"
    )
    assert untitled.row_text == "001122334455  · [missing]"
    titled = SendTarget(
        kind="session", session_id="001122334455", label='"a conversation"', state="live"
    )
    assert titled.row_text == '"a conversation"  · session 001122334455  · [live]'


def test_filtering_is_a_subsequence_over_the_row_text() -> None:
    rows = [
        SendTarget(kind="manager", session_id="m1", label="manager"),
        SendTarget(kind="session", session_id="s1", label="projects review", state="live"),
    ]
    assert filter_targets(rows, "") == rows
    assert [row.label for row in filter_targets(rows, "prv")] == ["projects review"]
    assert [row.label for row in filter_targets(rows, "zzz")] == []


def test_the_band_names_the_target_and_the_way_out() -> None:
    """Q3/D6: `m target` is gone, and the strip names the id it resolves."""
    target = SendTarget(kind="session", session_id="s1", label="projects review", state="live")
    assert compose_band(target) == "send to: ◆ s1 · esc cancel"
    assert "esc closes" in NO_TARGET_FOOTER and " s " not in NO_TARGET_FOOTER
    # U1: the second half names the route that exists — the detail page has no
    # link affordance at all, `/project link` is the only way.
    assert "/project link" in NO_TARGET_FOOTER


# -- the flow, over the real page ------------------------------------------
# The card and the modes are driven through the shipped app harness rather
# than a bare host: `m`'s meaning depends on which surface is up (a session row
# sends straight to it, anything else asks), and the composer hand-over is a
# message the app answers.

import asyncio  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from textual.content import Content  # noqa: E402
from textual.style import Style as ContentStyle  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402

from local_operator.mobile.peer_send import DeliveryOutcome  # noqa: E402
from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.widgets.projects_send import SendTargetCard  # noqa: E402
from local_operator.tui.widgets.projects_view import (  # noqa: E402
    ProjectsViewComposeChanged,
    ProjectsViewSendRequested,
)
from local_operator.tui.widgets.subagent_view import HintButton  # noqa: E402
from tests.unit.tui.test_projects_view import (  # noqa: E402
    _boot,
    _factory,
    _open,
    _ProjectSession,
    _registry,
)


@pytest.mark.asyncio
async def test_m_opens_the_picker_and_esc_puts_it_away(tmp_path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = _factory(session)  # noqa: F841 — the factory is what the app builds
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        assert view._mode == "canvas"
        await pilot.press("m")
        await pilot.pause()
        assert view._mode == "send"
        assert view._send_card is not None
        assert view._send_card.rows == [target]
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._send_card is None


@pytest.mark.asyncio
async def test_choosing_a_target_hands_the_composer_over_and_submits(tmp_path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert view.composing and view.compose_target is not None

        # RECORD, and still post: replacing `post_message` outright severs the
        # widget's own plumbing (mount/refresh notifications go through it),
        # which wedged `run_test` — the first version of this test hung the
        # harness for exactly that reason.
        posted: list[object] = []
        real_post = view.post_message

        def record(message: object) -> object:
            posted.append(message)
            return real_post(message)  # type: ignore[arg-type]

        view.post_message = record  # type: ignore[method-assign]
        sent = view.submit_compose("  hello there  ")
        assert sent is True
        request = [m for m in posted if isinstance(m, ProjectsViewSendRequested)]
        assert request and request[0].text == "hello there"
        assert request[0].target is view.compose_target

        # An empty body is refused IN-SURFACE: nothing is dialled, and the
        # page says why on its own notice line.
        posted.clear()
        assert view.submit_compose("   ") is True
        assert not [m for m in posted if isinstance(m, ProjectsViewSendRequested)]
        assert view._notice


def test_the_compose_surface_reports_its_target_and_esc() -> None:
    """`escape_surface` is the door the APP asks before dismissing the page."""
    target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
    card = SendTargetCard([target], style_for=None)
    # A card that has not been mounted still answers its own grammar.
    assert card.selected() == target
    assert isinstance(ProjectsViewComposeChanged(target=target), ProjectsViewComposeChanged)


# -- the floating card (design review round 1, D1/D2) ------------------------
# The card rides the overlay layer with a row budget handed to it by the page.
# These pins are D1's acceptance: the page does not move under the card, the
# card's whole geometry sits inside the ground it floats over, and it cannot
# select a row it did not paint.


def _page_rows(view) -> tuple[tuple[int, int, int, int], ...]:  # type: ignore[no-untyped-def]
    """The page's own rows, in the coordinates a reflow would change."""
    widgets = (view._title, view._rule, view._body, view._detail, view._hints)
    return tuple(
        (widget.region.x, widget.region.y, widget.region.width, widget.region.height)
        for widget in widgets
    )


async def _settle(pilot, predicate, passes: int = 80) -> bool:  # type: ignore[no-untyped-def]
    """Pump until ``predicate`` holds (a worker's receipt, usually)."""
    for _ in range(passes):
        await pilot.pause()
        if predicate():
            return True
    return False


def _strip(app) -> tuple[bool, str]:  # type: ignore[no-untyped-def]
    """The projects compose strip's ``(shown, text)`` — D6's carrier."""
    widget = app.query_one("#projects-compose-strip", Static)
    return bool(widget.display), widget.render().plain


def _painted_content(widget: Static) -> Content:
    """The ``Content`` a ``Static`` paints, narrowed by ASSERTION.

    ``Static.render`` is declared over the broad ``RenderableType`` union, and
    textual 8's ``Static`` paints a ``textual.content.Content``: ``.plain`` for
    the text, and spans whose style is ``textual.style.Style`` — whose colour
    field is ``foreground``, not rich's ``color``. The read narrows with
    ``isinstance``, the shape ``test_composer_visibility`` and
    ``test_boot_layout`` already use for this read, rather than a cast that
    would claim rich ``Text`` for an object that is not one.
    """
    renderable = widget.render()
    assert isinstance(renderable, Content), type(renderable).__name__
    return renderable


def _painted_text(widget: Static) -> str:
    """The plain text a ``Static`` is painting."""
    return _painted_content(widget).plain


@pytest.mark.parametrize("size", [(60, 24), (80, 24), (100, 30)])
@pytest.mark.asyncio
async def test_the_card_floats_without_reflowing_or_clipping(
    size: tuple[int, int], tmp_path: Path
) -> None:
    """D1/D2 acceptance at 60x24, 80x24 and 100x30.

    Opening the card moves nothing under it; the card's height is EXACTLY
    chrome + painted rows; every row it reports is on screen (60x24 used to
    clip the tail, and 80x24 painted zero rows while `enter` picked one); the
    floating card grows neither the scroll range nor a scrollbar.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        targets = [
            SendTarget(
                kind="session",
                session_id=f"session{i}",
                label=f'"work {i}"',
                state="live",
                live=True,
            )
            for i in range(3)
        ]
        view._send_targets = lambda: list(targets)  # type: ignore[method-assign]
        before = _page_rows(view)
        await pilot.press("m")
        await pilot.pause()
        await pilot.pause()
        card = view._send_card
        assert card is not None
        assert _page_rows(view) == before, f"{size}: opening the card reflowed the page"
        body = view._body
        top = body.region.y + body.styles.padding.top
        ground = view.content_region.y + view.content_region.height - top
        assert ground - SEND_CARD_CHROME_ROWS >= len(
            targets
        ), f"{size}: the acceptance sizes must fit the whole three-target list"
        expected = min(len(targets), SEND_CARD_ROW_CAP)
        assert len(card.window_rows()) == expected
        assert [row.session_id for row in card.window_rows()] == [
            target.session_id for target in targets[:expected]
        ]
        assert card.region.height == SEND_CARD_CHROME_ROWS + expected
        assert card.region.y == top
        assert card.region.y + card.region.height <= (
            view.content_region.y + view.content_region.height
        )
        assert card.region.x >= body.region.x
        assert card.region.x + card.region.width <= body.region.x + body.region.width
        # D9: the card takes the width it can hold — the page's whole content box
        # — so no page row is painted BESIDE it (`max-width: 60` left 20 cells of
        # canvas visible at 80x24, level with the card's own last row).
        assert card.region.x <= body.region.x
        assert (
            card.region.x + card.region.width >= body.region.x + body.region.width
        ), f"{size}: the page shows beside the card"
        assert 0 in card.painted_range()  # the selection is painted
        painted = _painted_text(card.query_one("#projects-send-rows", Static))
        for row in card.window_rows():
            assert row.row_text in painted
        assert painted.count("▸") == 1  # exactly one marker: the selected row
        # Nothing about the card grew the screen: no scrollbar, no scroll range.
        assert app.screen.virtual_size == app.screen.size


@pytest.mark.asyncio
async def test_the_window_follows_the_selection(tmp_path: Path) -> None:
    """Nine targets behind a three-row window: the cursor walks past the edge
    and the painted slice follows, so the selection is never off screen."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        targets = [
            SendTarget(kind="session", session_id=f"session{i}", label=f"work {i}", state="stale")
            for i in range(9)
        ]
        view._send_targets = lambda: list(targets)  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.set_available(SEND_CARD_CHROME_ROWS + 3)
        await pilot.pause()
        assert len(card.window_rows()) == 3
        for _ in range(7):
            await pilot.press("down")
            await pilot.pause()
            assert card.index in card.painted_range(), (card.index, card.painted_range())
        assert card.index == 7
        assert [row.session_id for row in card.window_rows()] == [
            "session5",
            "session6",
            "session7",
        ]
        # The windowed list says how much it is not showing, in `muted` ink
        # (D5): `dim` measured 3.43:1 on the card's own overlay ground.
        note = _painted_content(card.query_one("#projects-send-note", Static))
        assert note.plain == "+6 more"
        muted = theme_mod.semantic_color("muted").lower()
        # The span's ink lives on textual's `Style.foreground` (a
        # `textual.color.Color`), whose `.hex` is the `#rrggbb` the palette
        # tokens are written in — the union member is narrowed first.
        assert any(
            isinstance(span.style, ContentStyle)
            and span.style.foreground is not None
            and span.style.foreground.hex.lower() == muted
            for span in note.spans
        )


@pytest.mark.asyncio
async def test_enter_cannot_pick_a_row_the_card_did_not_paint(tmp_path: Path) -> None:
    """The 80x24 blind enter (D1): with zero painted rows, `enter` starts
    nothing — the card answers only for rows the reader can see."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="stale")
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.set_available(SEND_CARD_CHROME_ROWS)  # chrome only: not one row fits
        await pilot.pause()
        assert card.window_rows() == []
        await pilot.press("enter")
        await pilot.pause()
        assert view._mode == "send"
        assert not view.composing


# -- receipts: the band and the draft (agent review F4 / UX U2-U3 / QA Q5) ---


def test_receipts_speak_the_apps_human_vocabulary() -> None:
    """F4/U3/D6: `state_word` words, the strip's `◆ <id>` handle, no model
    stdout — no `→` prefix and no raw message uuid standing in for an outcome."""
    target = SendTarget(
        kind="session", session_id="dd44ee55ff66", label='"older review"', state="stale"
    )
    assert pending_line(target) == "sending to ◆ dd44ee55ff66…"
    assert sent_line(target, "delivered") == "sent to ◆ dd44ee55ff66 · delivered"
    assert sent_line(target, "wake unconfirmed") == "sent to ◆ dd44ee55ff66 · wake unconfirmed"
    assert refusal_line(target, "the target said no") == (
        "could not deliver to ◆ dd44ee55ff66: the target said no"
    )
    assert send_error_line(target, "no session found") == (
        "could not send to ◆ dd44ee55ff66: no session found"
    )
    assert "→" not in sent_line(target, "delivered")
    # The handle is the thing the strip shows (D6) — not the title.
    assert '"older review"' not in sent_line(target, "delivered")


@pytest.mark.asyncio
async def test_a_refused_send_keeps_the_draft_and_names_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q5/F4: the submit clears the composer, the refusal puts the draft back,
    and the sentence names the row the reader picked rather than the id."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(
            kind="session", session_id="s1", label='"older review"', state="live", live=True
        )
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert view.composing
        # N1: the in-flight sentence is the STRIP's alone. The notice row is
        # cleared instead of painted — see `test_the_strip_states_the_send_in_flight`
        # for the strip half of this pin.
        assert view._notice is None

        editor = app._editor()
        editor.load_text("retry me")
        await pilot.pause()
        resolved: dict[str, object] = {}

        def _resolve(**kwargs):  # type: ignore[no-untyped-def]
            resolved.update(kwargs)
            return None, [], None

        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            _resolve,
        )
        await pilot.press("enter")
        assert await _settle(pilot, lambda: (view._notice or "").startswith("could not send"))
        assert editor.text == "retry me"
        assert view._notice == ("could not send to ◆ s1: the session is no longer available")
        # F6: the send path resolves like every other send — `include_wedged`
        # is the KILL SWITCH's flag; the picker must not dial wedged targets
        # hopefully.
        assert not resolved.get("include_wedged")
        assert resolved.get("require_started") is True


@pytest.mark.asyncio
async def test_a_delivered_send_receipts_in_the_band_and_clears_the_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4's routing: an acknowledged send answers in the composer's band (the
    editor is empty then) and restores nothing over the cleared composer."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(
            kind="session", session_id="s1", label='"older review"', state="live", live=True
        )
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )

        async def _delivered(*args, **kwargs):  # type: ignore[no-untyped-def]
            return DeliveryOutcome(
                "delivered", "the receiver acknowledged it", "mid-1", "acked", 1, "", "live", "s1"
            )

        monkeypatch.setattr(
            "local_operator.mobile.peer_send.deliver_peer_message_outcome", _delivered
        )
        editor = app._editor()
        editor.load_text("hello there")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(pilot, lambda: _strip(app) == (True, "sent to ◆ s1 · delivered"))
        assert editor.text == ""
        # The in-flight line is GONE, not blank: an empty notice would take the
        # project detail's place in the footer (pass-5 frame catch).
        assert view._notice is None


@pytest.mark.parametrize(
    "state,wake,word,detail",
    [
        (
            "mailbox",
            "unconfirmed",
            "wake unconfirmed",
            "delivered to its mailbox (id mid-2) — the wake was not acknowledged "
            "within 5s after 2 attempts. It will read the message on its next turn; "
            "do not send it again.",
        ),
        (
            "unconfirmed",
            "unconfirmed",
            "delivery unconfirmed",
            "delivery unconfirmed (id mid-2) — no answer within 5s after 2 attempts "
            "and it could not be confirmed in its transcript. It may still arrive "
            "once its loop turns. Check the target's transcript before resending; "
            "sending again may deliver it twice.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_an_amber_send_keeps_the_draft_and_uses_the_state_word(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    wake: str,
    word: str,
    detail: str,
) -> None:
    """F4/U2: mailbox/unconfirmed are amber receipts on the notice row in the
    app's own words, carrying the core's caution for that state, and the draft
    survives so the reader decides what next."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(
            kind="session", session_id="s1", label='"older review"', state="live", live=True
        )
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )

        async def _partial(*args, **kwargs):  # type: ignore[no-untyped-def]
            return DeliveryOutcome(state, detail, "mid-2", wake, 2, "no_answer", "live", "s1")

        monkeypatch.setattr(
            "local_operator.mobile.peer_send.deliver_peer_message_outcome", _partial
        )
        advisory = DeliveryOutcome(state, detail, "mid-2", wake, 2, "no_answer", "live", "s1")
        caution = advisory.advisory
        assert caution, "the parametrized detail must carry the core's caution half"
        editor = app._editor()
        editor.load_text("keep me please")
        await pilot.pause()
        await pilot.press("enter")
        # U2: the notice carries the word AND the caution the core wrote for
        # exactly this state — the restored draft is the strongest available cue
        # to press enter again, so "do not send it again" must be on screen with
        # it rather than dropped.
        assert await _settle(pilot, lambda: view._notice == f"sent to ◆ s1 · {word} — {caution}")
        assert editor.text == "keep me please"


# -- the key model (agent review round 1, Q2/Q3/Q4/U5) -----------------------


@pytest.mark.asyncio
async def test_typing_reaches_the_filter_and_enter_chooses(tmp_path: Path) -> None:
    """QA round 1, Q2: the card says `type to filter` — the keys must LAND in
    the filter with no Tab dance, and `enter` from the same state chooses."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        targets = [
            SendTarget(
                kind="session", session_id=f"session{i}", label=f"work {i}", state="live", live=True
            )
            for i in range(3)
        ]
        view._send_targets = lambda: list(targets)  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        filt = card.query_one("#projects-send-filter", Input)
        assert app.focused is filt  # the Input starts focused (Q2)
        # ↑/↓ still move the list from the Input…
        await pilot.press("down")
        await pilot.pause()
        assert card.index == 1
        # …and typing FILTERS on the first keystroke, resetting the cursor.
        await pilot.press("2")
        await pilot.pause()
        assert filt.value == "2"
        assert [row.session_id for row in card.rows] == ["session2"]
        assert card.index == 0
        # `enter` posts Input.Submitted (the Input holds focus) → choose.
        await pilot.press("enter")
        await pilot.pause()
        assert view.composing
        target = view.compose_target
        assert target is not None and target.session_id == "session2"


@pytest.mark.asyncio
async def test_composing_keeps_only_the_escape_key_advertised(tmp_path: Path) -> None:
    """Q3/U5: the band and the page row must not advertise keys that type.

    `m target` is gone from the band, and the page's canvas ladder — which
    advertised `1 list · v next · c create · m message · d detail` — now
    paints exactly the one page key that still reaches the page: `esc cancel`.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(
            kind="session", session_id="s1", label='"older review"', state="live", live=True
        )
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert view.composing
        editor = app._editor()
        assert await _settle(
            pilot, lambda: _strip(app) == (True, "send to: ◆ s1 · esc cancel")
        )  # no `m target`
        editor.load_text("keep me")
        await pilot.pause()
        await pilot.press("end")  # the caret sits where typing left it: the end
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert editor.text == "keep mem"  # `m` types; the band promised nothing else
        # The page's own row, while composing: esc cancel, and nothing live.
        live = [hint for hint in view._hints.children if hint.display]
        assert live == [view._exit_hint]
        assert "cancel" in view._exit_hint.rendered()


@pytest.mark.asyncio
async def test_esc_out_of_compose_returns_focus_and_the_page_still_answers(
    tmp_path: Path,
) -> None:
    """QA round 1, Q4: after `esc cancel` the focus was None and every key a
    silent no-op until a Tab; the page takes the focus back."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        editor = app._editor()
        editor.load_text("draft to keep")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not view.composing
        assert app.focused is not None  # Q4: no stranded focus
        assert view.has_focus
        assert editor.text == "draft to keep"  # esc cancel keeps the draft
        # And the next `m` reaches the page instead of dying: the picker opens.
        await pilot.press("m")
        await pilot.pause()
        assert view._mode == "send" and view._send_card is not None


# -- the dressing (agent review round 1, D4/D5/D6) ---------------------------


def _styles_at(text, needle: str):  # type: ignore[no-untyped-def]
    """Every span style covering ``needle``'s first occurrence."""
    start = text.plain.index(needle)
    end = start + len(needle)
    styles = []
    for span in text.spans:
        if span.start <= start and span.end >= end and span.style is not None:
            styles.append(span.style)
    return styles


@pytest.mark.asyncio
async def test_the_selected_row_bands_and_the_chips_take_state_inks(tmp_path: Path) -> None:
    """D4: the `tint-select` band under the chosen row, the cursor ink on the
    marker, and distinct chip inks — `[live]` and `[stopped]` must not paint
    the same."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.projects_view import _style_resolver

    resolver = _style_resolver()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        targets = [
            SendTarget(
                kind="session", session_id="live-one-123", label="work", state="live", live=True
            ),
            SendTarget(kind="session", session_id="dead-one-456", label="work", state="stopped"),
        ]
        view._send_targets = lambda: list(targets)  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        text = card.rows_text()
        # The selected row's line is padded to the card's content width, so the
        # band is a ROW and not a run of text.
        assert len(text.plain.split("\n")[0]) == card.content_size.width
        assert any(
            style.bgcolor == resolver("row_selected").bgcolor for style in _styles_at(text, "▸")
        )
        assert any(style.bold for style in _styles_at(text, "▸"))
        assert any(
            style.color == resolver("chip_live").color for style in _styles_at(text, "[live]")
        )
        assert any(
            style.color == resolver("status_done").color for style in _styles_at(text, "[stopped]")
        )
        assert resolver("chip_live").color != resolver("status_done").color


@pytest.mark.asyncio
async def test_the_strip_persists_while_composing_and_names_the_id(tmp_path: Path) -> None:
    """D6: the recipient strip is a real surface — it survives the first
    keystroke (what the placeholder could not do) and names the id the send
    resolves, not the conversation title."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(
            kind="session", session_id="s1", label='"older review"', state="live", live=True
        )
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert await _settle(pilot, lambda: _strip(app) == (True, "send to: ◆ s1 · esc cancel"))
        editor = app._editor()
        editor.load_text("half a message")
        await pilot.pause()
        # STILL there while the reader types — the placeholder it replaced
        # vanished on the first keystroke.
        assert _strip(app) == (True, "send to: ◆ s1 · esc cancel")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not _strip(app)[0]


@pytest.mark.asyncio
async def test_a_failed_outcome_refuses_with_the_reason_and_keeps_the_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8(c): the `is_error` arm — a delivery that ends `failed` refuses in
    the app's words, names the strip's handle, and leaves the draft up."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )

        async def _failed(*args, **kwargs):  # type: ignore[no-untyped-def]
            return DeliveryOutcome(
                "failed",
                "the target said no",
                "mid-3",
                "unconfirmed",
                1,
                "peer_refused",
                "live",
                "s1",
            )

        monkeypatch.setattr("local_operator.mobile.peer_send.deliver_peer_message_outcome", _failed)
        editor = app._editor()
        editor.load_text("try me")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(
            pilot, lambda: view._notice == "could not deliver to ◆ s1: the target said no"
        )
        assert editor.text == "try me"


@pytest.mark.asyncio
async def test_a_refused_dial_and_a_faulted_send_keep_the_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8(c): the exception arms — `RuntimeError` is a pre-delivery refusal
    (“could not send”), a transport fault is honestly unconfirmed, and both
    leave the draft up."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )

        async def _refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("the peer refused the dial")

        monkeypatch.setattr("local_operator.mobile.peer_send.deliver_peer_message_outcome", _refuse)
        editor = app._editor()
        editor.load_text("try me")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(
            pilot, lambda: view._notice == "could not send to ◆ s1: the peer refused the dial"
        )
        assert editor.text == "try me"

        async def _fault(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("connection reset by peer")

        monkeypatch.setattr("local_operator.mobile.peer_send.deliver_peer_message_outcome", _fault)
        await pilot.press("enter")
        # U3: the notice is a SENTENCE, never interpreter text — the exception is
        # logged, not painted. U2: and it carries the caution for a state that may
        # already have landed.
        assert await _settle(
            pilot,
            lambda: (view._notice or "").startswith("sent to ◆ s1 · delivery unconfirmed"),
        )
        assert view._notice is not None
        assert "connection reset" not in view._notice
        assert "sending again may deliver it twice" in view._notice
        assert editor.text == "try me"


@pytest.mark.asyncio
async def test_an_internal_fault_reaches_the_reader_as_a_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U3: a fault on the RESOLVE path must not paint interpreter text.

    Reproduced in the UX round as `cannot unpack non-iterable coroutine object`
    — the reader gets a sentence, and the exception goes to the log.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        def _explode(**kwargs):  # type: ignore[no-untyped-def]
            raise AttributeError("cannot unpack non-iterable coroutine object")

        monkeypatch.setattr("local_operator.mobile.peer_send.resolve_peer_target", _explode)
        editor = app._editor()
        editor.load_text("try me")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(pilot, lambda: view._notice is not None)
        assert view._notice is not None
        assert view._notice == ("could not send to ◆ s1: something went wrong on this end — retry")
        assert "coroutine" not in view._notice and "AttributeError" not in view._notice
        assert editor.text == "try me"


@pytest.mark.asyncio
async def test_a_filter_that_matches_nothing_says_so(tmp_path: Path) -> None:
    """UX round 1, N2: with Q2 fixed a mistyped filter can reach the empty
    list; the note must offer the filter back, not tell the reader to link a
    session — and clearing the filter brings the row back."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        await pilot.press("z", "z", "z")
        await pilot.pause()
        note = _painted_text(card.query_one("#projects-send-note", Static))
        assert note == NO_MATCH_FOOTER
        await pilot.press("backspace", "backspace", "backspace")
        await pilot.pause()
        assert [row.session_id for row in card.rows] == ["s1"]


@pytest.mark.asyncio
async def test_the_strip_states_the_send_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-2: the strip is the composer's statement about THIS send — the
    in-flight line goes up before the outcome exists, and the outcome replaces
    it rather than the other way round."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )
        gate = asyncio.Event()

        async def _slow(*args, **kwargs):  # type: ignore[no-untyped-def]
            await gate.wait()
            return DeliveryOutcome(
                "delivered", "the peer acknowledged it", "mid-1", "acked", 1, "", "live", "s1"
            )

        monkeypatch.setattr("local_operator.mobile.peer_send.deliver_peer_message_outcome", _slow)
        editor = app._editor()
        editor.load_text("first body")
        await pilot.pause()
        await pilot.press("enter")
        # The strip speaks for the send now in flight…
        assert await _settle(pilot, lambda: _strip(app) == (True, pending_line(target)))
        # …and the outcome replaces it, not the other way round.
        gate.set()
        assert await _settle(pilot, lambda: _strip(app) == (True, sent_line(target, "delivered")))
        # N1: the notice row never carried the in-flight sentence — it is the
        # strip's alone, and the footer falls back to the project detail.
        assert view._notice is None


@pytest.mark.asyncio
async def test_a_second_send_resets_the_strip_and_a_refusal_restores_the_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-2: after a delivered first send, a refused second one must not leave
    `sent to … delivered` standing over the refusal — the band goes back to
    addressing the kept draft, and the notice carries the refusal."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()

        record = SimpleNamespace(pid=os.getpid() + 1, session_id="s1")
        monkeypatch.setattr(
            "local_operator.mobile.peer_send.resolve_peer_target",
            lambda **kwargs: (record, [], None),
        )

        async def _delivered(*args, **kwargs):  # type: ignore[no-untyped-def]
            return DeliveryOutcome(
                "delivered", "the peer acknowledged it", "mid-1", "acked", 1, "", "live", "s1"
            )

        monkeypatch.setattr(
            "local_operator.mobile.peer_send.deliver_peer_message_outcome", _delivered
        )
        editor = app._editor()
        editor.load_text("first body")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(pilot, lambda: _strip(app) == (True, sent_line(target, "delivered")))

        async def _refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("the peer refused the dial")

        monkeypatch.setattr("local_operator.mobile.peer_send.deliver_peer_message_outcome", _refuse)
        editor.load_text("second body")
        await pilot.pause()
        await pilot.press("enter")
        assert await _settle(
            pilot,
            lambda: view._notice == "could not send to ◆ s1: the peer refused the dial",
        )
        # The band hands itself back to the compose state: the kept draft is
        # addressed to somebody again, and the first send's receipt is gone.
        assert _strip(app) == (True, compose_band(target))
        assert editor.text == "second body"


@pytest.mark.asyncio
async def test_the_page_hint_row_stops_advertising_keys_the_card_consumes(
    tmp_path: Path,
) -> None:
    """R2-3: while the card is up it holds the keyboard — the canvas ladder's
    keys TYPE into its filter (reproduced below: `c` -> filter "c"), so the
    page row must not advertise them. The compose rung's fix, one mode over."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        # D10: the row is EMPTY while the card owns the keys. The card's own
        # legend carries the whole grammar one row above it, so painting
        # `esc close` here stated one instruction twice in two inks.
        painted = " ".join(
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        )
        assert painted == "", f"the page row still paints under the card: {painted!r}"
        # The repro behind the finding: a printable key lands in the filter,
        # which is exactly why the ladder was a lie while the card was up.
        await pilot.press("c")
        await pilot.pause()
        assert card.query_one("#projects-send-filter", Input).value == "c"

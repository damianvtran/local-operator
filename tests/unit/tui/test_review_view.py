"""The ``/review`` full-page ledger: real app, real ledger files, real keys.

Every state here is driven through the shipped page: the index file the app
reads is the one ``code_requests.ledger`` writes (seeded at its real path under
the isolated config root), the refresh seam is the service's own
``refresh_session`` (monkeypatched to record, never to reach a forge), and the
keys go through the app's dispatch. Nothing is asserted against a hand-built
frame.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.containers import ScrollableContainer

from local_operator.code_requests import cache as fetch_cache
from local_operator.code_requests import ledger, rounds, service
from local_operator.code_requests.refs import Ref
from local_operator.i18n import catalogues
from local_operator.i18n import text as i18n_text
from local_operator.paths import config_dir
from local_operator.session.frontend_state import CommandScope, _slash_capabilities
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for
from local_operator.tui import review
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.review_view import ReviewView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SESSION_ID = "sess"


def _plain(static) -> str:
    content = static.content
    assert isinstance(content, Text)
    return content.plain


@pytest.fixture(autouse=True)
def _isolated_fetch_cache():
    """The fetch cache memoises by host+key, not by config root.

    Two tests seeding ``github.com``'s ``owner/repo#1`` would otherwise share
    the first test's entry; the module's own test seam is what the server suite
    already uses for the same reason.
    """
    fetch_cache._reset_for_tests()
    yield
    fetch_cache._reset_for_tests()


@pytest.fixture
def refreshes(monkeypatch):
    """Record every ``refresh_session`` pass instead of calling a forge."""
    calls: list[tuple[str, list[Any]]] = []

    async def fake(config_root, session_id, rows, **kwargs):
        calls.append((session_id, list(rows)))
        return service.RefreshReport()

    monkeypatch.setattr(service, "refresh_session", fake)
    return calls


def _ref(number: int, *, forge: str = "github", project: str = "damianvtran/local-operator"):
    if forge == "gitlab":
        url = f"https://gitlab.com/{project}/-/merge_requests/{number}"
    else:
        url = f"https://github.com/{project}/pull/{number}"
    ref = Ref.from_payload(
        {
            "url": url,
            "forge": forge,
            "host": "gitlab.com" if forge == "gitlab" else "github.com",
            "project": project,
            "number": number,
            "full": True,
        }
    )
    assert ref is not None
    return ref


def _index_row(
    number: int,
    *,
    relation: str = "opened",
    forge: str = "github",
    project: str = "damianvtran/local-operator",
    acts: list[str] | None = None,
    last_at: float = 100.0,
) -> dict[str, Any]:
    ref = _ref(number, forge=forge, project=project)
    return {
        "key": ref.key,
        "ref": {
            "url": ref.url,
            "forge": ref.forge,
            "host": ref.host,
            "project": ref.project,
            "number": ref.number,
            "full": True,
        },
        "relation": relation,
        "relations": [relation],
        "acts": list(acts or ()),
        "mentions": [{"source": "assistant", "count": 1, "first_at": last_at, "last_at": last_at}],
        "first_at": last_at,
        "last_at": last_at,
    }


def _write_index(rows: list[dict[str, Any]], *, raw: str | None = None) -> Path:
    path = ledger.index_path(config_dir(), SESSION_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    path.write_text(
        json.dumps({"schema": ledger.INDEX_SCHEMA, "rows": rows, "updated_at": 1.0}),
        encoding="utf-8",
    )
    return path


def _entry(
    *,
    state: str = "open",
    draft: bool = False,
    title: str = "feat(daemon): reap the observation stamp",
    ci: dict[str, Any] | None = None,
    comments: int | None = 6,
    updated_at: float = 1_000_000.0,
    lanes: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "pieces": {"comments": []},
        "summary": {
            "state": state,
            "draft": draft,
            "title": title,
            "head_sha": "9d29452c1100",
            "head_ref": "feat/x",
            "base_ref": "main",
            "author": "damianvtran",
            "created_at": 900_000.0,
            "updated_at": updated_at,
        },
        "ci": (
            ci
            if ci is not None
            else {"status": "success", "total": 23, "passed": 23, "failed": 0, "pending": 0}
        ),
        "lanes": lanes or [],
        "comments_total": comments,
        "fetched_at": 1_000.0,
        "stale": False,
        **extra,
    }


def _write_entry(ref: Ref, entry: dict[str, Any]) -> None:
    assert fetch_cache.write_entry(config_dir(), entry, ref=ref)


def _populated(
    number: int = 1,
    *,
    relation: str = "opened",
    forge: str = "github",
    project: str = "damianvtran/local-operator",
    **entry: Any,
) -> None:
    ref = _ref(number, forge=forge, project=project)
    rows = [_index_row(number, relation=relation, forge=forge, project=project)]
    _write_index(rows)
    _write_entry(ref, _entry(**entry))


async def _settle(pilot, predicate, *, turns: int = 300) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


async def _boot(pilot, app: OperatorApp) -> None:
    await _settle(pilot, lambda: app._session is not None)


async def _open_review(pilot, app: OperatorApp) -> ReviewView:
    app._run_slash_command("/review")
    await pilot.pause()
    view = app.query_one(ReviewView)
    await _settle(pilot, lambda: view._state != "loading")
    return view


@pytest.mark.asyncio
async def test_review_shows_loading_until_the_cold_scan_settles_and_esc_restores_the_draft(
    refreshes, monkeypatch
) -> None:
    """No index and no cached rows: loading is the honest cold frame.

    The empty state is withheld until the scan settles — on a machine that has
    never written the derived index, "No code reviews yet" before the scan
    would be a claim the conversation can disprove one step later.
    """
    gate = asyncio.Event()
    real_scan = ledger.refresh_async

    async def gated_scan(*args, **kwargs):
        await gate.wait()
        return await real_scan(*args, **kwargs)

    monkeypatch.setattr(ledger, "refresh_async", gated_scan)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.load_text("half-typed prompt")
        await pilot.pause()

        app._run_slash_command("/review")
        await pilot.pause()
        view = app.query_one(ReviewView)
        assert view._state == "loading"
        assert "Loading code reviews" in _plain(view._body_text)

        gate.set()
        await _settle(pilot, lambda: view._state == "ready")
        assert "No code reviews yet" in _plain(view._body_text)
        assert app.screen.has_class("settings")
        assert not app._transcript_view().display

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ReviewView)) == 0
        assert app._transcript_view().display
        assert app._editor().text == "half-typed prompt"


@pytest.mark.asyncio
async def test_cached_rows_paint_and_work_while_the_forge_pass_is_blocked(
    refreshes, monkeypatch
) -> None:
    """The local ledger must not be held hostage by a slow unrelated fetch.

    An indexed, previously fetched request is displayable before any network
    work; the first frame is `ready` with the row, and selection/Enter keep
    working while `refresh_session` is still held (agent review F3, UX U1).
    """
    _populated(
        1,
        title="feat(code-review): the session's request ledger",
        lanes=[
            {
                "lane": "agent",
                "round": 2,
                "state": "remediation_posted",
                "freshness": "stale",
                "reviewed_head": "3f2a91b7",
            },
            {"lane": "qa", "round": 1, "state": "clean", "freshness": "fresh"},
        ],
    )
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def gated(config_root, session_id, rows, **kwargs):
        entered.set()
        await gate.wait()
        return service.RefreshReport()

    monkeypatch.setattr(service, "refresh_session", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "ready"
        body = _plain(view._body_text)
        assert "damianvtran/local-operator #1" in body
        assert "Round 2 · remediation posted · awaiting re-review" in body
        assert "stale — reviewed 3f2a91b, head 9d29452" in body
        assert "QA" in body
        assert "23/23 passed" in body
        assert "6 comments" in body
        # The forge pass is genuinely still in flight while the row is shown.
        await _settle(pilot, entered.is_set)
        assert view._refreshing is True

        opened: list[str] = []
        app.open_transcript_link = opened.append  # type: ignore[method-assign]
        await pilot.press("enter")
        await pilot.pause()
        assert opened == ["https://github.com/damianvtran/local-operator/pull/1"]

        gate.set()
        await _settle(pilot, lambda: view._refreshing is False)
        assert "damianvtran/local-operator #1" in _plain(view._body_text)


@pytest.mark.asyncio
async def test_empty_ledger_is_a_first_class_state(refreshes) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "ready"
        body = _plain(view._body_text)
        assert "No code reviews yet" in body
        assert "will appear here" in body


@pytest.mark.asyncio
async def test_unreadable_index_shows_the_error_state_and_r_retries(refreshes) -> None:
    _write_index([], raw="{not json")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "error"
        body = _plain(view._body_text)
        assert "Code reviews could not be read." in body
        assert "Nothing was lost" in body
        assert "Press r to try again." in body

        # The retry goes through the same read; a repaired index is read again.
        _write_index([_index_row(2)])
        await pilot.press("r")
        await _settle(pilot, lambda: view._state == "ready")
        assert "damianvtran/local-operator #2" in _plain(view._body_text)


@pytest.mark.asyncio
async def test_rows_group_opened_then_mentioned_and_enter_opens_the_forge_url(refreshes) -> None:
    _write_index(
        [
            _index_row(7, relation="mentioned", last_at=300.0),
            _index_row(1, relation="opened", last_at=200.0),
        ]
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        body = _plain(view._body_text)
        assert body.index("Opened") < body.index("Mentioned")
        assert body.index("#1") < body.index("#7")
        assert [row["key"] for row in view._visible_rows] == [
            _ref(1).key,
            _ref(7).key,
        ]

        opened: list[str] = []
        app.open_transcript_link = opened.append  # type: ignore[method-assign]
        await pilot.press("enter")
        await pilot.pause()
        assert opened == ["https://github.com/damianvtran/local-operator/pull/1"]

        # Clamped, not wrapped: one press up from the first row stays first.
        await pilot.press("up")
        await pilot.pause()
        assert view._index == 0
        await pilot.press("end")
        await pilot.pause()
        assert view._index == 1
        await pilot.press("down")
        await pilot.pause()
        assert view._index == 1


@pytest.mark.asyncio
async def test_r_refreshes_the_forge_and_re_reads_the_index(refreshes) -> None:
    _populated(1)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert len(refreshes) == 1

        _write_index(
            [
                _index_row(1),
                _index_row(3, relation="mentioned"),
            ]
        )
        await pilot.press("r")
        await _settle(pilot, lambda: "#3" in _plain(view._body_text))
        assert len(refreshes) == 2
        assert refreshes[-1][0] == SESSION_ID
        assert _ref(1).key in [row["key"] for row in refreshes[-1][1]]


@pytest.mark.asyncio
async def test_narrow_terminal_hides_the_detail_pane(refreshes) -> None:
    _populated(
        1,
        lanes=[{"lane": "agent", "round": 2, "state": "clean", "freshness": "fresh"}],
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._narrow is True
        assert view._pane.display is False

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._narrow is False
        assert view._pane.display is True
        pane = _plain(view._pane_body)
        assert "damianvtran/local-operator #1" in pane
        assert "Round 2 · clean" in pane


def test_review_is_a_registered_frontend_local_slash_command() -> None:
    from local_operator.session.frontend_state import _FRONTEND_LOCAL_SLASHES

    command = slash_command_for("/review")
    assert command is not None
    assert command.name == "review"
    # TUI-only: no palette row can forward it to a desktop destination.
    assert command.desktop_destination == ""
    assert command.arguments.name == "NONE"
    assert "review" in _FRONTEND_LOCAL_SLASHES
    assert "review" in {entry.name for entry in SLASH_COMMANDS}
    capabilities = _slash_capabilities()
    assert (
        next(c for c in capabilities if c.command == "review").scope == CommandScope.FRONTEND_LOCAL
    )


def test_the_namespace_has_key_parity_across_shipped_locales() -> None:
    """Phase B adds the five translations; the assertion goes live with them."""
    english = catalogues.load_catalogue("en", "tui.review")
    assert english
    assert all(key.startswith("tui.review.") for key in english)
    sidecar = catalogues.context("en", "tui.review")
    assert sidecar, "the context sidecar must describe the new keys"
    assert set(sidecar) <= set(english)
    for locale in catalogues.locales():
        if locale == "en":
            continue
        path = catalogues.catalogue_root() / locale / "tui.review.json"
        if not path.is_file():
            continue
        assert set(catalogues.load_catalogue(locale, "tui.review")) == set(english), locale


@pytest.mark.parametrize("locale", ["en", "fr", "es", "ru", "vi", "hi"])
def test_the_lane_name_column_is_fixed_in_every_shipped_locale(locale: str) -> None:
    """A longer translated lane name must not shift its clause.

    The body line is ``"  " + lane_name_cell(name) + " " + clause``, so a fixed
    cell here is what keeps that clause's start column identical in all six
    shipped locales — the defect a minimum-width ``f"{name:<6}"`` would ship
    silently the first time a translation ran past the budget.
    """
    assert catalogues.load_catalogue(locale, "tui.review")
    previous = i18n_text.active_locale()
    i18n_text.set_active_locale(locale)
    try:
        for lane in review.LANE_ORDER:
            name = review.lane_display_name(lane)
            cell = review.lane_name_cell(name)
            assert cell_len(cell) == review.LANE_NAME_CELLS, (locale, lane, name)
            # Body clause start: 2-cell indent + the name cell + 1 seam space.
            assert 2 + cell_len(cell) + 1 == 2 + review.LANE_NAME_CELLS + 1
    finally:
        i18n_text.set_active_locale(previous)


def test_lane_name_cell_truncates_an_overlong_name_without_moving_the_column() -> None:
    cell = review.lane_name_cell("x" * 40)
    assert cell == "x" * review.LANE_NAME_CELLS
    assert cell_len(cell) == review.LANE_NAME_CELLS


def test_the_view_does_not_shadow_textual_s_api() -> None:
    for name in ("query", "visible", "render", "_render"):
        assert name not in ReviewView.__dict__, name


def _seed_transcript(text: str) -> Path:
    """A real session journal holding one user message, and NO derived index."""
    session_dir = config_dir() / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    journal = ledger.transcript_path(session_dir)
    journal.write_text(
        json.dumps(
            {
                "id": "m1",
                "ts": 1.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": text}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return journal


@pytest.mark.asyncio
async def test_open_scans_the_transcript_when_no_index_exists(refreshes) -> None:
    """A TUI-only machine has never run the desktop route that builds the index.

    The conversation's PR link must reach the page through the SAME scan the
    routes use, not stay invisible because the derived file is absent.
    """
    url = "https://github.com/damianvtran/local-operator/pull/1904"
    _seed_transcript(f"opened this for review: {url}")
    assert not ledger.index_path(config_dir(), SESSION_ID).exists()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "ready"
        body = _plain(view._body_text)
        assert "Mentioned" in body
        assert "damianvtran/local-operator #1904" in body
        assert "Fetching current state" in body
        # The scan published the derived index, so a second reader sees it cold.
        assert ledger.index_path(config_dir(), SESSION_ID).is_file()
        entry = ledger.read_index(config_dir(), SESSION_ID)
        assert entry is not None and entry["rows"]


@pytest.mark.asyncio
async def test_a_failed_scan_keeps_cached_rows_and_shows_the_notice(refreshes, monkeypatch) -> None:
    """A torn transcript must not erase what this machine already recorded."""
    _populated(1)

    async def broken(*args, **kwargs):
        raise OSError("journal is unreadable")

    monkeypatch.setattr(ledger, "refresh_async", broken)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "ready"
        body = _plain(view._body_text)
        assert "damianvtran/local-operator #1" in body
        assert "23/23 passed" in body
        assert "Couldn't scan this conversation; showing what was already recorded." in _plain(
            view._detail
        )


@pytest.mark.asyncio
async def test_a_failed_scan_with_no_recorded_rows_is_the_error_state(
    refreshes, monkeypatch
) -> None:
    """With nothing cached, "no code reviews yet" would be a claim we cannot make."""

    async def broken(*args, **kwargs):
        raise OSError("journal is unreadable")

    monkeypatch.setattr(ledger, "refresh_async", broken)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        assert view._state == "error"
        body = _plain(view._body_text)
        assert "Code reviews could not be read." in body
        assert "No code reviews yet" not in body


# -- round-1 remediation regressions ---------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_round_strip_stays_bounded_in_the_view(refreshes) -> None:
    """`round 100000` from a remote comment must not size the render work."""
    parsed = rounds.parse(
        [rounds.Comment(id="c1", body="### Agent review — round 100000\n\nNo findings.")],
        head_sha="9d29452c1100",
        is_open=True,
    )
    lane = next(
        item.to_payload()
        for item in parsed.states("9d29452c1100", is_open=True)
        if item.lane == "agent"
    )
    assert lane["round"] == 100000
    _populated(1, lanes=[lane])
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        body = _plain(view._body_text)
        assert "Round 100000" in body
        assert "…" in body

        block = Text()
        view._append_row(block, view._visible_rows[0], selected=False, width=view._line_width())
        # The cap plus the row's own chrome, not one span per claimed round.
        assert len(block.spans) < 60
        # The omission marker sits before the strip it truncates.
        assert block.plain.index("…") < block.plain.index("▮")


@pytest.mark.asyncio
async def test_r_forces_the_forge_past_a_fresh_cache_while_a_cold_open_does_not(
    monkeypatch,
) -> None:
    """`r` is an explicit check-now; a cold open still trusts a fresh entry."""
    import time

    _populated(1, checked_at=time.time())
    fetches: list[tuple[list[str], bool]] = []

    async def fake_refresh_keys(config_root, refs, *, force=False, on_changed=None):
        fetches.append(([ref.key for ref in refs], force))
        return service.RefreshReport()

    monkeypatch.setattr(service, "refresh_keys", fake_refresh_keys)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        await _settle(pilot, lambda: view._refreshing is False)
        assert fetches == [], "a cold open must respect the fresh cache"

        await pilot.press("r")
        await _settle(pilot, lambda: view._refreshing is False and bool(fetches))
        assert fetches
        assert fetches[-1][1] is True, "r must reach the service as an explicit force"
        assert _ref(1).key in fetches[-1][0]


@pytest.mark.asyncio
async def test_r_repairs_a_torn_index_through_the_scan(refreshes, monkeypatch) -> None:
    """The advertised retry must run the scan that rewrites the broken file."""
    _write_index([], raw="{not json")
    real_scan = ledger.refresh_async
    calls = {"n": 0}

    async def fail_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("scan temporarily failed")
        return await real_scan(*args, **kwargs)

    monkeypatch.setattr(ledger, "refresh_async", fail_first)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._state == "error"
        assert "Press r to try again." in _plain(view._body_text)

        _seed_transcript(
            "opened this for review: https://github.com/damianvtran/local-operator/pull/42"
        )
        await pilot.press("r")
        await _settle(pilot, lambda: view._state == "ready")
        assert "damianvtran/local-operator #42" in _plain(view._body_text)
        assert (
            json.loads(ledger.index_path(config_dir(), SESSION_ID).read_text(encoding="utf-8"))[
                "schema"
            ]
            == ledger.INDEX_SCHEMA
        )


@pytest.mark.asyncio
async def test_r_repairs_a_stale_schema_index(refreshes, monkeypatch) -> None:
    """The upgrade path: a schema-2 file is rewritten by the same retry."""
    _write_index([], raw=json.dumps({"schema": 2, "rows": []}))
    _seed_transcript(
        "opened this for review: https://github.com/damianvtran/local-operator/pull/43"
    )
    real_scan = ledger.refresh_async
    calls = {"n": 0}

    async def fail_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("scan temporarily failed")
        return await real_scan(*args, **kwargs)

    monkeypatch.setattr(ledger, "refresh_async", fail_first)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._state == "error"

        await pilot.press("r")
        await _settle(pilot, lambda: view._state == "ready")
        assert "damianvtran/local-operator #43" in _plain(view._body_text)
        assert (
            json.loads(ledger.index_path(config_dir(), SESSION_ID).read_text(encoding="utf-8"))[
                "schema"
            ]
            == ledger.INDEX_SCHEMA
        )


def test_review_hints_use_the_readable_ink() -> None:
    """D1: the footer is the page's operation, not disabled chrome."""
    view = ReviewView(SESSION_ID)
    assert len(view._hint_order) == 4
    for button in view._hint_order:
        assert button._ink == ("muted", "muted")
        assert button._lifted_ink == ("fg", "fg")


@pytest.mark.asyncio
async def test_lane_names_and_notice_lines_paint_in_readable_ink(refreshes) -> None:
    """D1: selected-row metadata moves off `dim` (3.97:1) onto `muted`."""
    _populated(
        1,
        lanes=[{"lane": "agent", "round": 2, "state": "clean", "freshness": "fresh"}],
        refresh_error="Network unavailable",
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        line = review.lane_lines(view._visible_rows[0])[0]
        rendered = Text()
        view._append_lane_line(rendered, line, width=80, background=None)
        name_span = next(span for span in rendered.spans if span.start == 2)
        name_style = name_span.style
        assert isinstance(name_style, Style) and name_style.color is not None
        assert name_style.color.name == theme_mod.semantic_color("muted")
        assert name_style.color.name != theme_mod.semantic_color("dim")

        row = view._visible_rows[0]
        block = Text()
        view._append_row(block, row, selected=False, width=view._line_width())
        notice_start = block.plain.index("Network unavailable")
        notice = next(span for span in block.spans if span.start <= notice_start < span.end)
        notice_style = notice.style
        assert isinstance(notice_style, Style) and notice_style.color is not None
        assert notice_style.color.name == theme_mod.semantic_color("muted")


@pytest.mark.asyncio
async def test_row_offsets_count_every_line_when_a_notice_follows_the_meta(
    refreshes,
) -> None:
    """U2: the notice owns its line, so cursor offsets match the painted rows."""
    _populated(1, refresh_error="Network unavailable")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)

        block = Text()
        view._append_row(block, view._visible_rows[0], selected=False, width=view._line_width())
        lines = block.plain.splitlines()
        meta_line = next(line for line in lines if "23/23 passed" in line)
        assert "Network unavailable" not in meta_line, "the notice must start its own line"
        assert sum("Network unavailable" in line for line in lines) == 1
        assert view._row_offsets[0][1] == block.plain.count("\n")


@pytest.mark.asyncio
async def test_narrow_rows_collapse_lanes_and_stay_dense(refreshes) -> None:
    """U2: a three-lane request keeps its identity in a four-line body."""
    title = "feat(code-review): the session's request ledger"
    _populated(
        1,
        title=title,
        lanes=[
            {"lane": "agent", "round": 2, "state": "clean", "freshness": "fresh"},
            {"lane": "design", "round": 1, "state": "clean", "freshness": "fresh"},
            {"lane": "qa", "round": 1, "state": "clean", "freshness": "fresh"},
        ],
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._narrow is True

        block = Text()
        view._append_row(block, view._visible_rows[0], selected=True, width=view._line_width())
        lines = block.plain.splitlines()
        assert "damianvtran/local-operator #1" in lines[0]
        assert title[:40] in lines[1]
        assert "Code 2 · Design 1 · QA 1" in lines[2]
        assert "23/23 passed" in lines[3]
        # The prose lane lines are wide mode's job.
        assert "Round 2 · clean" not in block.plain
        # No blank separator between narrow rows: the four identity lines are
        # the first four, not pushed below a gap.
        start, height = view._row_offsets[0]
        assert height <= 5
        # A row at least as tall as the viewport is anchored at its TOP: bottom
        # alignment hid the identity and state above the fold (UX U2).
        if height >= view._body.size.height:
            assert view._body.scroll_offset.y == start


@pytest.mark.asyncio
async def test_the_detail_pane_is_focusable_scrollable_and_wraps_narrow_titles(
    refreshes,
) -> None:
    """D2: the full selected record is reachable, and a narrow title wraps."""
    long_title = "feat(code-review): the session's request ledger, in its own full-page pane"
    _populated(
        1,
        title=long_title,
        lanes=[
            {"lane": "agent", "round": 2, "state": "remediation_posted", "freshness": "stale"},
            {"lane": "design", "round": 1, "state": "clean", "freshness": "fresh"},
            {"lane": "qa", "round": 1, "state": "clean", "freshness": "fresh"},
            {"lane": "ux", "round": 1, "state": "clean", "freshness": "fresh"},
        ],
        refresh_error="Network unavailable",
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert isinstance(view._pane, ScrollableContainer)
        assert view._pane.can_focus is True
        assert view._pane.max_scroll_y > 0

        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is view._pane
        before = view._pane.scroll_offset.y
        await pilot.press("down")
        await pilot.pause()
        assert view._pane.scroll_offset.y > before

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open_review(pilot, app)
        assert view._narrow is True
        assert view._pane.display is False
        lines = _plain(view._detail).splitlines()
        assert len(lines) == 2
        assert " ".join(line.strip() for line in lines) == long_title

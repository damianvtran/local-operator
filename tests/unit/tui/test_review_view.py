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
from rich.text import Text

from local_operator.code_requests import cache as fetch_cache
from local_operator.code_requests import ledger, service
from local_operator.code_requests.refs import Ref
from local_operator.i18n import catalogues
from local_operator.paths import config_dir
from local_operator.session.frontend_state import CommandScope, _slash_capabilities
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for
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
async def test_review_shows_loading_then_cached_rows_and_esc_restores_the_draft(
    refreshes, monkeypatch
) -> None:
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

    async def gated(config_root, session_id, rows, **kwargs):
        # Blocked so the FIRST painted frame is the loading state; a real pass
        # here would reach the forge and make the frame a race.
        await gate.wait()
        return service.RefreshReport()

    monkeypatch.setattr(service, "refresh_session", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    # 150 columns: the pane takes 38, and the full lane clause is 44 cells —
    # at 100 the body legitimately truncates it, which would make this assert
    # about the body WIDTH rather than about the clause.
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
        body = _plain(view._body_text)
        assert "Opened" in body
        assert "damianvtran/local-operator #1" in body
        assert "Round 2 · remediation posted · awaiting re-review" in body
        assert "stale — reviewed 3f2a91b, head 9d29452" in body
        assert "QA" in body
        assert "23/23 passed" in body
        assert "6 comments" in body
        assert app.screen.has_class("settings")
        assert not app._transcript_view().display

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ReviewView)) == 0
        assert app._transcript_view().display
        assert app._editor().text == "half-typed prompt"


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
        pane = _plain(view._pane)
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


def test_the_view_does_not_shadow_textual_s_api() -> None:
    for name in ("query", "visible", "render", "_render"):
        assert name not in ReviewView.__dict__, name

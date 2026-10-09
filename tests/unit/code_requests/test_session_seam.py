"""The session seam (PR1b): the turn-end/wake/monitor dirty mark and its notes.

``_code_request_notes`` and ``_mark_code_requests_dirty`` are called with a
small stub session: the pieces they touch are the config root, the session id
and two pure helpers, and building a full Session for either would test the
runtime rather than the seam (the live path is exercised in the PR evidence).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.code_requests import cache, ledger
from local_operator.code_requests.refs import EMPTY_CONTEXT, Ref, parse_any
from local_operator.session.session import Session


def _ref(url: str) -> Ref:
    ref = parse_any(url)
    assert ref is not None, url
    return ref


REF = _ref("https://github.com/o/r/pull/7")
SESSION = "abcdef123456"


class _Stub:
    """The attributes the two seam methods read, and their pure helpers."""

    _ref_handle = staticmethod(Session._ref_handle)
    _tracked_line = Session._tracked_line

    def __init__(self, config_dir: Path, session_id: str = SESSION) -> None:
        self._config_dir = str(config_dir)
        self._session_id = session_id


@pytest.fixture(autouse=True)
def _clean():
    cache._reset_for_tests()
    yield
    cache._reset_for_tests()


def _seed_index(tmp_path: Path) -> None:
    from local_operator.code_requests.scan import Row, ScanResult

    assert ledger.write_index(tmp_path, SESSION, ScanResult(rows=[Row(ref=REF)]))


def test_the_turn_end_mark_is_all_rows_and_a_noop_without_rows(tmp_path: Path) -> None:
    import asyncio

    stub = _Stub(tmp_path)
    asyncio.run(Session._mark_code_requests_dirty(cast(Any, stub)))
    assert not cache.dirty_path(tmp_path, SESSION).exists(), "no rows, no mark"

    _seed_index(tmp_path)
    asyncio.run(Session._mark_code_requests_dirty(cast(Any, stub)))
    mark = cache.read_dirty(tmp_path, SESSION)
    assert mark["all"] is True, "a turn end revalidates whatever the session has"


def test_opened_and_acted_notes_carry_the_tag_and_their_actions(tmp_path: Path) -> None:
    stub = _Stub(tmp_path)
    detections = [
        SimpleNamespace(kind="opened", ref=REF, act=None),
        SimpleNamespace(kind="acted", ref=REF, act="comment"),
    ]
    notes = Session._code_request_notes(cast(Any, stub), detections, "bash", {}, EMPTY_CONTEXT)
    assert [note.event for note in notes] == ["code-requests", "code-requests"]
    assert notes[0].text.startswith("Tracked: #7 (opened).")
    assert notes[1].text.startswith("Tracked: #7 (comment).")
    assert cache.read_dirty(tmp_path, SESSION)["keys"] == [REF.key], "the acted mark"


def test_a_merge_note_names_the_rows_round_freshness_when_known(tmp_path: Path) -> None:
    entry = {
        "key": REF.key,
        "state": "open",
        "summary": {"head_sha": "9d29452abc123", "state": "open"},
        "lanes": [
            {
                "lane": "agent",
                "round": 2,
                "state": "terminal",
                "state_copy": "clean",
                "freshness": "fresh",
                "reviewed": "9d29452abc123",
            }
        ],
        "validators": {},
        "pieces": {},
        "checked_at": 0.0,
        "fetched_at": 0.0,
        "stale": False,
        "refresh_error": None,
    }
    assert cache.write_entry(tmp_path, entry, ref=REF)
    stub = _Stub(tmp_path)
    notes = Session._code_request_notes(
        cast(Any, stub),
        [SimpleNamespace(kind="acted", ref=REF, act="merge")],
        "bash",
        {},
        EMPTY_CONTEXT,
    )
    assert len(notes) == 1
    assert "Tracked: #7 (merge)." in notes[0].text
    assert "Latest: agent review r2 clean on 9d29452." in notes[0].text


def test_a_monitor_arm_naming_a_ref_gets_the_tracked_line(tmp_path: Path) -> None:
    # "tell me when round 2 lands" arms a watch rather than acting: the refs in
    # the ARM's arguments are what the note is built from.
    stub = _Stub(tmp_path)
    args = {"op": "arm", "command": f"code_requests show {REF.url}"}
    notes = Session._code_request_notes(cast(Any, stub), [], "monitor", args, EMPTY_CONTEXT)
    assert len(notes) == 1
    assert notes[0].text.startswith("Tracked: #7 (armed with monitor).")


def test_the_merge_note_omits_freshness_without_a_cache_entry(tmp_path: Path) -> None:
    stub = _Stub(tmp_path)
    notes = Session._code_request_notes(
        cast(Any, stub),
        [SimpleNamespace(kind="acted", ref=REF, act="merge")],
        "bash",
        {},
        EMPTY_CONTEXT,
    )
    assert len(notes) == 1 and "Latest:" not in notes[0].text

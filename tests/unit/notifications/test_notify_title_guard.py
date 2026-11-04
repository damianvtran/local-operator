"""The ``notify`` tool refuses a content-free title at the boundary (§3.1).

The store-side writers keep their short titles on purpose — an approval gate's
title is the tool name, a test's is a fixture — so the guard lives on the TOOL
PARAMS, the one path a model drives. These tests go through that path with the
real store behind ``ToolContext.notify_inbox``, because a unit call to the
validator proves the predicate and not the wiring.

The refusal must also name the silent alternative: a turn with nothing to
report has to cost the model nothing (`no_reply`, no notification, no error),
or the cheapest way to obey the guard becomes posting nothing at all.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Awaitable, TypeVar

import pytest

from local_operator.harness.types import TextContent, ToolContext
from local_operator.notifications.store import NotificationStore
from local_operator.tools import builtin

T = TypeVar("T")

#: A synthetic conversation id, never a live session (AGENTS.md isolation).
SID = "synthetic-session-title-guard"


def _run(coro: Awaitable[T]) -> T:
    # Tools promise Awaitable, while asyncio.run takes a coroutine on 3.12.
    async def run() -> T:
        return await coro

    return asyncio.run(run())


def _door(root: Path) -> Any:
    """The session's door shape (``Session._notify_inbox``), backed by the real store."""

    async def post(
        *,
        title: str,
        body: str,
        priority: str,
        key: str,
        links: list[dict[str, str]],
        anchor_id: str,
    ) -> str:
        store = NotificationStore(root)
        outcome = store.notify(
            SID,
            title=title,
            body_md=body,
            priority=priority,
            key=key,
            anchor_id=anchor_id,
            links=links,
        )
        return f"Notified (id {outcome.item_id})."

    return post


def _item_count(root: Path) -> int:
    """Rows actually written, read from the store file itself."""
    path = NotificationStore(root).path
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
    return int(row[0])


async def _notify(root: Path, args: dict[str, Any]) -> Any:
    context = ToolContext(cwd=".", notify_inbox=_door(root))
    return await builtin.execute_notify("call_title_guard", args, None, None, context)


def _text(result: Any) -> str:
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


def test_a_placeholder_title_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    result = _run(_notify(tmp_path, {"title": "noop", "body": ""}))
    assert result.is_error
    text = _text(result)
    # Actionable: names the defect AND the silent alternative.
    assert "noop" in text
    assert "no_reply" in text
    assert "Nothing was posted" in text
    # The refusal never reaches the store: no row for the operator to dismiss.
    assert _item_count(tmp_path) == 0


def test_a_legitimate_title_still_writes_a_row_and_reports_the_door_sentence(
    tmp_path: Path,
) -> None:
    result = _run(
        _notify(
            tmp_path,
            {
                "title": "PR #3116 merged — team bridge receiver",
                "body": "Merged and released.",
                "priority": "normal",
            },
        )
    )
    assert not result.is_error
    assert _text(result).startswith("Notified (id n_")
    assert _item_count(tmp_path) == 1


def test_a_short_but_real_title_is_accepted(tmp_path: Path) -> None:
    """The floor admits what a person would actually write; only noise is refused."""
    result = _run(_notify(tmp_path, {"title": "CI green", "body": ""}))
    assert not result.is_error
    assert _item_count(tmp_path) == 1


def test_a_substantial_body_does_not_rescue_a_placeholder_title(tmp_path: Path) -> None:
    """The list row is the title alone — the body is not what the operator sees."""
    result = _run(
        _notify(
            tmp_path,
            {"title": "placeholder", "body": "Migrated 14 tables and backfilled 2400 rows."},
        )
    )
    assert result.is_error
    assert "no_reply" in _text(result)
    assert _item_count(tmp_path) == 0


def test_a_legitimate_title_with_an_empty_body_is_accepted(tmp_path: Path) -> None:
    """A one-line event is a real notification; the body is optional by design."""
    result = _run(_notify(tmp_path, {"title": "3 fraud attempts blocked", "body": ""}))
    assert not result.is_error
    assert _item_count(tmp_path) == 1


def test_a_whitespace_only_title_is_refused_with_the_same_sentence(tmp_path: Path) -> None:
    """Currently this reaches the store and surfaces as an unhelpful dead end."""
    result = _run(_notify(tmp_path, {"title": "   ", "body": "x"}))
    assert result.is_error
    assert "no_reply" in _text(result)
    assert _item_count(tmp_path) == 0


@pytest.mark.parametrize("title", ["x", "y", "z", "ok", "done", "unused", "wip", "tbd", "N/A"])
def test_every_reported_placeholder_is_refused_through_the_tool(tmp_path: Path, title: str) -> None:
    result = _run(_notify(tmp_path, {"title": title, "body": ""}))
    assert result.is_error, f"{title!r} was accepted"
    assert _item_count(tmp_path) == 0

"""``lop wake create`` says the zone where the value is read.

The CLI row prints the due time in the READER's own clock, which is the same
clock they typed — so it cannot show them that ``18:45`` meant 18:45 in that
clock and not 18:45Z. The extra ``resolved:`` line quotes the absolute instant
for the one input form where text and instant differ; an absolute form says its
own instant and gets no line.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
    yield tmp_path
    if hasattr(time, "tzset"):
        monkeypatch.undo()
        time.tzset()


def _now_ms() -> int:
    return int(datetime(2026, 11, 5, 12, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1000)


def _session(config_dir: Path) -> None:
    directory = config_dir / "sessions" / "wakeclock01"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id":"x","ts":1,"type":"message","payload":'
        '{"kind":"message","role":"user","content":[{"type":"text","text":"hi"}]}}\n',
        encoding="utf-8",
    )


def _args(**kwargs: object) -> argparse.Namespace:
    base = {
        "session": "wakeclock01",
        "when": "at 18:45",
        "message": "check the build",
        "json": False,
        "every": "",
        "until": "",
        "limit": None,
        "notify": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_a_local_clock_create_prints_the_resolved_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The row still names the (local) clock; the extra line names the instant."""
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires tzset")

    from local_operator import cli

    monkeypatch.setattr(cli.time, "time", lambda: _now_ms() / 1000)
    _session(tmp_path)

    assert cli._wake_create(_args()) == 0

    out = capsys.readouterr().out
    assert "  resolved: 2026-11-05T18:45:00-05:00 (from '18:45', read as LOCAL time" in out
    # The existing row is unchanged: id, the human clock, the delta, the message.
    assert "w1" in out
    assert "check the build" in out


def test_an_absolute_create_prints_no_resolved_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing to add when the input already named its own instant."""
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires tzset")

    from local_operator import cli

    monkeypatch.setattr(cli.time, "time", lambda: _now_ms() / 1000)
    _session(tmp_path)

    assert cli._wake_create(_args(when="at 2026-11-05T23:45:00Z")) == 0

    out = capsys.readouterr().out
    assert "resolved:" not in out
    assert "w1" in out


@pytest.mark.asyncio
async def test_the_wake_tool_echoes_the_resolved_instant_for_a_local_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model's own surface: the sentence names the instant AND the reading.

    A tool reply is the only thing the caller sees, so a create that resolved a
    LOCAL clock into some other instant has to say so there — this is the one
    place the incident's writer would have caught it.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires tzset")

    from local_operator.harness.types import ToolContext
    from local_operator.tools import builtin
    from tests.unit.tools.test_builtin_tools import _FakeScheduler

    monkeypatch.setattr(builtin.time, "time", lambda: _now_ms() / 1000)
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="wakeclock01", wake_scheduler=scheduler)

    created = await builtin.execute_wake(
        "c", {"op": "create", "message": "check build", "at": "18:45"}, None, None, context
    )
    assert not created.is_error
    assert "2026-11-05T18:45:00-05:00" in created.text
    assert "read as LOCAL" in created.text

    absolute = await builtin.execute_wake(
        "c2",
        {"op": "create", "message": "check build", "at": "2026-11-05T23:45:00Z"},
        None,
        None,
        context,
    )
    assert not absolute.is_error
    assert "read as LOCAL" not in absolute.text

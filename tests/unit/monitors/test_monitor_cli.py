"""``lop monitor status`` / ``lop monitor cancel`` — the slice-4 CLI surface.

``status`` reads the derived index (so it answers without opening a session)
and must name a disabled monitor's reason (design §11.3) and a dormant
session's park; ``cancel`` goes through ``monitors/arm.py`` and must refuse
rather than invent a session or a monitor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pytest


def _session(config_dir: Path, session_id: str, rows: list[dict[str, Any]] | None = None) -> Path:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        '{"id":"m1","ts":1.0,"type":"message","payload":'
        '{"kind":"message","role":"user","content":[{"type":"text","text":"hi"}]}}'
    ]
    if rows is not None:
        lines.append(
            json.dumps(
                {
                    "id": "monitor-entry-1",
                    "ts": 2.0,
                    "type": "custom",
                    "payload": {
                        "custom_type": "monitor_schedules",
                        "details": {"monitors": rows, "next_seq": 2},
                    },
                }
            )
        )
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _row(monitor_id: str = "m1", **extra: Any) -> dict[str, Any]:
    row = {
        "id": monitor_id,
        "name": "watch the deploy queue",
        "tool": "bash",
        "arguments": {"command": "ls"},
        "every_ms": 60_000,
        "created_at": 1_700_000_000_000,
    }
    row.update(extra)
    return row


def _args(**kwargs: object) -> argparse.Namespace:
    base: dict[str, object] = {"monitor_command": "status", "json": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # `monitor status` (like every `lop` listing) renders to the terminal
    # width, and a monitor row's tail is longer than an 80-column fallback
    # leaves once the state and session columns are paid — so the table tests
    # pin a wide terminal rather than asserting on an ellipsis.
    monkeypatch.setenv("COLUMNS", "200")
    return tmp_path


def test_no_monitors_is_an_empty_answer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_operator.cli import monitor_command

    assert monitor_command(_args()) == 0
    assert capsys.readouterr().out.strip() == "no monitors"


def test_status_json_is_the_index_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    due = int(time.time() * 1000) + 30_000
    write_entry(tmp_path, "msess01", cwd="/w", monitors=[_row("m1", next_due_at=due)])

    assert monitor_command(_args(json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["monitor_id"] for row in payload] == ["m1"]
    assert payload[0]["session_id"] == "msess01"
    assert payload[0]["next_due_at"] == due
    assert payload[0]["dormant"] is False


def test_status_names_a_monitors_state_and_health(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    now = int(time.time() * 1000)
    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        # 150 s out and 5 min old: both land mid-unit so a second of test
        # slippage cannot tip the rendered word (a 30 s due renders "29s"
        # under load, which made this test time-flaky).
        monitors=[_row("m1", next_due_at=now + 150_000, checks=4, last_check_at=now - 300_000)],
    )

    assert monitor_command(_args()) == 0

    out = capsys.readouterr().out
    assert "in 2m" in out
    assert "msess01" in out
    assert "m1 watch the deploy queue" in out
    assert "every 1m" in out
    assert "4 checks" in out
    assert "last check 5m ago" in out


def test_a_disabled_monitor_shows_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        monitors=[
            _row(
                "m2",
                name="watch the issue tracker",
                next_due_at=None,
                disabled=True,
                disabled_reason="no longer read-only",
                consecutive_failures=5,
            )
        ],
    )

    assert monitor_command(_args()) == 0

    out = capsys.readouterr().out
    assert "disabled" in out
    assert "no longer read-only" in out
    assert "5 failed" in out
    # The legend tells the reader what to do about it.
    assert "create the" in out and "reactivate" in out


def test_a_dormant_session_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        monitors=[_row("m1")],
        preserve={"stopped_at": 1234},
    )

    assert monitor_command(_args()) == 0

    out = capsys.readouterr().out
    assert "dormant" in out
    assert "reopening it re-arms" in out


def test_cancel_success_reports_what_was_cancelled_and_what_remains(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import read_entry, write_entry

    _session(tmp_path, "cancelsess01", [_row("m1"), _row("m2")])
    write_entry(tmp_path, "cancelsess01", cwd="/w", monitors=[_row("m1"), _row("m2")])

    assert (
        monitor_command(_args(monitor_command="cancel", session="cancelsess01", monitor_id="m1"))
        == 0
    )

    out = capsys.readouterr().out
    assert "cancelled m1" in out
    assert "watch the deploy queue" in out
    assert "1 monitor left" in out
    entry = read_entry(tmp_path, "cancelsess01")
    assert entry is not None
    assert [row["id"] for row in entry["monitors"]] == ["m2"]


def test_cancel_json_reports_the_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    _session(tmp_path, "cancelsess01", [_row("m1")])
    write_entry(tmp_path, "cancelsess01", cwd="/w", monitors=[_row("m1")])

    assert (
        monitor_command(
            _args(monitor_command="cancel", session="cancelsess01", monitor_id="m1", json=True)
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["monitor_id"] == "m1"
    assert payload["remaining"] == 0


def test_cancel_refuses_an_unknown_monitor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    _session(tmp_path, "cancelsess01", [_row("m1")])
    write_entry(tmp_path, "cancelsess01", cwd="/w", monitors=[_row("m1")])

    assert (
        monitor_command(_args(monitor_command="cancel", session="cancelsess01", monitor_id="m9"))
        == 1
    )

    captured = capsys.readouterr()
    assert "No monitor with id 'm9'" in captured.err
    assert captured.out == ""


def test_cancel_refuses_an_unknown_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command

    assert monitor_command(_args(monitor_command="cancel", session="nosuch", monitor_id="m1")) == 1
    assert "no session 'nosuch'" in capsys.readouterr().err


def test_an_overdue_monitor_on_a_closed_session_reads_idle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """§D6: "next due now" on a watch nothing is hosting reads as a stuck
    check; the row has to say which of the two it is.
    """
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    now = int(time.time() * 1000)
    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        monitors=[_row("m1", checks=4, next_due_at=now - 3_600_000)],
    )

    assert monitor_command(_args()) == 0

    out = capsys.readouterr().out
    assert "idle" in out
    assert "overdue by" in out
    # And the legend earns its place only when the word is present.
    assert "no session is hosting it" in out


def test_the_idle_legend_is_absent_when_nothing_is_idle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    now = int(time.time() * 1000)
    write_entry(tmp_path, "msess01", cwd="/w", monitors=[_row("m1", next_due_at=now + 150_000)])

    assert monitor_command(_args()) == 0
    assert "no session is hosting it" not in capsys.readouterr().out


def test_zero_deliveries_after_several_checks_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live store's second silent failure: a monitor that ran and delivered
    nothing, with no way for the operator to tell that from a broken watch.
    """
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    now = int(time.time() * 1000)
    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        monitors=[_row("m1", checks=9, deliveries=0, next_due_at=now + 150_000)],
    )

    assert monitor_command(_args()) == 0

    out = capsys.readouterr().out
    assert "9 checks, 0 deliveries" in out
    assert "confirm the call observes what you expect" in out


def test_an_unavailable_episode_is_named_in_the_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import monitor_command
    from local_operator.monitors.store import write_entry

    now = int(time.time() * 1000)
    write_entry(
        tmp_path,
        "msess01",
        cwd="/w",
        monitors=[_row("m1", checks=4, unavailable_since=now - 600_000, next_due_at=now + 60_000)],
    )

    assert monitor_command(_args()) == 0
    assert "tool unavailable since" in capsys.readouterr().out

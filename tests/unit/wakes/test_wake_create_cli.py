"""``lop wake create`` — the subcommand that made install-on-demand testable.

Round 1 (Q3) found it simply absent: `status`, `list` and `serve` shipped and
`create` did not, so the matrix row for "install-on-demand from `lop wake
create`" was unexecutable and no path outside a live TUI could schedule a wake.

The properties worth pinning are the ones that make a created wake REAL: it
lands in the same derived index every other path writes (so `lop wake list`
and the supervisor both see it), and it refuses rather than inventing a
session that does not exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _session(config_dir: Path, session_id: str) -> Path:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id":"x","ts":1,"type":"message","payload":'
        '{"kind":"message","role":"user","content":[{"type":"text","text":"hi"}]}}\n',
        encoding="utf-8",
    )
    return directory


def _args(**kwargs: object) -> argparse.Namespace:
    base = {"session": "wakecreate01", "when": "in 2m", "message": "check the build", "json": True}
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_a_created_wake_lands_in_the_index_the_supervisor_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One index, one truth: `wake list` and the supervisor read this file."""
    from local_operator.cli import _wake_create
    from local_operator.wakes.store import read_entry

    _session(tmp_path, "wakecreate01")

    assert _wake_create(_args()) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["wake_id"] == "w1"

    entry = read_entry(tmp_path, "wakecreate01")
    assert entry is not None
    schedules = entry["schedules"]
    assert len(schedules) == 1
    assert schedules[0]["message"] == "check the build"


def test_a_second_wake_does_not_replace_the_first(tmp_path: Path) -> None:
    """`write_entry` REPLACES the entry, so the existing schedules must be
    carried forward — dropping them would silently cancel a live reminder."""
    from local_operator.cli import _wake_create
    from local_operator.wakes.store import read_entry

    _session(tmp_path, "wakecreate01")
    _wake_create(_args(message="first"))
    _wake_create(_args(message="second", when="45s"))

    entry = read_entry(tmp_path, "wakecreate01")
    assert entry is not None
    assert [s["message"] for s in entry["schedules"]] == ["first", "second"]
    assert [s["id"] for s in entry["schedules"]] == ["w1", "w2"]


def test_an_unknown_session_is_refused(tmp_path: Path) -> None:
    """A wake keyed on a session with no transcript is one the supervisor
    would faithfully fire into nothing."""
    from local_operator.cli import _wake_create

    assert _wake_create(_args(session="nosuchsession")) == 1


@pytest.mark.parametrize("when", ["banana", "", "60"])
def test_an_unreadable_time_is_refused(tmp_path: Path, when: str) -> None:
    """`60` is refused with the rest on purpose: it reads as both seconds and
    milliseconds, and guessing wrong schedules the wrong thing silently."""
    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(when=when)) == 1


@pytest.mark.parametrize("when", ["in 2m", "45s", "1h30m", "at 09:30"])
def test_the_advertised_time_forms_all_parse(tmp_path: Path, when: str) -> None:
    """Every shape the --help text promises. `in 2m` in particular: the
    parsers take a bare duration, so the preposition is stripped by the
    command rather than taught to both of them."""
    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(when=when)) == 0

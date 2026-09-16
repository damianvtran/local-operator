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
import time
from pathlib import Path
from typing import Any

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
    base = {
        "session": "wakecreate01",
        "when": "in 2m",
        "message": "check the build",
        "json": True,
        "every": "",
        "until": "",
        "limit": None,
    }
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


def test_a_recurring_wake_stores_its_interval(tmp_path: Path) -> None:
    """`--every` is what makes a background automation possible from the CLI.

    The model has been able to schedule a recurring wake since
    `WakeSchedule.every_ms`; only `lop wake create` could not, so the
    operator's stated use for this ("regular disk cleaning, automation
    tasks") was unreachable without opening a session (round 3, QA).
    """
    import json

    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(every="1h")) == 0

    entry = json.loads((tmp_path / "wakes" / "wakecreate01.json").read_text())
    assert entry["schedules"][0]["every_ms"] == 3_600_000


def test_a_one_shot_wake_stores_no_interval(tmp_path: Path) -> None:
    """Omitting `--every` must stay a one-shot, not a zero-interval loop."""
    import json

    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args()) == 0

    entry = json.loads((tmp_path / "wakes" / "wakecreate01.json").read_text())
    assert entry["schedules"][0]["every_ms"] is None


@pytest.mark.parametrize("every", ["banana", "3600", "40s"])
def test_an_unusable_repeat_interval_is_refused_in_words(
    tmp_path: Path, every: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused with a SENTENCE, not a pydantic traceback.

    `3600` is ambiguous between seconds and milliseconds; `40s` is below the
    one-minute floor, which exists because each wake starts a full turn and a
    sub-minute repeat starves the session it serves. Both used to reach the
    model validator and print a stack trace at the user.
    """
    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(every=every)) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "interval" in err


def test_a_bound_recurrence_stores_its_bounds(tmp_path: Path) -> None:
    """`--until` and `--limit` bound a repeat; the store has always honoured
    them (`advance_wake_schedule` retires on both) and only the CLI could not
    express one, so a scheduled automation could only be unbounded (R4)."""
    import json

    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(every="1h", limit=5)) == 0
    entry = json.loads((tmp_path / "wakes" / "wakecreate01.json").read_text())
    assert entry["schedules"][0]["limit"] == 5

    assert _wake_create(_args(every="1h", until="in 7d")) == 0
    entry = json.loads((tmp_path / "wakes" / "wakecreate01.json").read_text())
    assert entry["schedules"][1]["until_at"] is not None


@pytest.mark.parametrize("until", ["in 7d", "at 09:30", "7d"])
def test_every_advertised_until_form_parses(tmp_path: Path, until: str) -> None:
    """The help promises `in 7d` and `at 09:30`; both prepositions are stripped
    by the command, exactly as the `when` argument does it."""
    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(every="1h", until=until)) == 0


def test_a_bound_without_a_repeat_is_refused(tmp_path: Path) -> None:
    """A one-shot already fires exactly once, so accepting `--limit` on one
    would promise a behaviour the schedule does not have."""
    from local_operator.cli import _wake_create

    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(limit=3)) == 1
    assert _wake_create(_args(until="in 7d")) == 1
    assert _wake_create(_args(every="1h", limit=0)) == 1


def test_the_listing_shows_a_time_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--until` was carried in `--json` and dropped from the human listing, so
    a `--limit` wake advertised its bound while an `--until` one was
    indistinguishable from an unbounded repeat — the bound a user is most
    likely to forget was the one not rendered (round 5, R6/U16).

    The terminal width is PINNED because the listing budgets its message column
    from the host's (`shutil.get_terminal_size` reads `COLUMNS`, then the real
    stdout), and one run-to-run difference is enough to clamp the message out
    of the row: a `--until` row's tail (` · every 30m, until in 6d`) is 25
    characters against a message budget that shrinks by 7-8 characters whenever
    the WHEN cell widens from the 11-12 a same-day one takes (`8:14 PM EDT`,
    `12:39 AM UTC`) to the 18-19 of a next-day `Sep 15 12:04 AM UTC` -- which
    is what a run starting just before midnight UTC gets, because its due
    times land on the following day. At that width `room` is 8 against a
    10-character message, so the clamp prints `time bo…` and this test's own
    `next(...)` raises StopIteration. CI run 34910432097 (shard 3, started
    23:48 UTC; its two wakes due at 00:04/00:05 the next day) failed exactly
    that way, while the same file passed on the previous head at 21:06 UTC.
    The listing is correct in both runs -- the assertion here is that the BOUND
    is rendered, not how a narrow budget clamps a message -- so the width it
    renders at is part of the fixture, not of the host.
    """
    import argparse

    from local_operator.cli import _wake_create, wake_command

    monkeypatch.setenv("COLUMNS", "100")
    _session(tmp_path, "wakecreate01")
    assert _wake_create(_args(when="in 5m", message="unbounded", every="1h")) == 0
    assert _wake_create(_args(when="in 6m", message="time bound", every="30m", until="in 7d")) == 0
    capsys.readouterr()

    wake_command(argparse.Namespace(wake_command="list", json=False))
    out = capsys.readouterr().out
    unbounded = next(line for line in out.splitlines() if "unbounded" in line)
    bounded = next(line for line in out.splitlines() if "time bound" in line)

    assert "every 1h" in unbounded and "until" not in unbounded
    assert "every 30m" in bounded and "until in 6d" in bounded, bounded


# ---------------------------------------------------------------------------
# The three defects the write moved to ``wakes/arm.py`` to fix
# ---------------------------------------------------------------------------
#
# Each was reachable only through THIS command — the in-session tool and the
# validator got all three right — and each failed silently: a 17th schedule
# past the cap, a reused id, and a stopped session that quietly restarted.
# They are pinned here as well as in ``test_arm.py`` because the command is
# what a person actually types.


def _schedule_row(wake_id: str, **extra: object) -> dict[str, Any]:
    """One persisted row, in the shape the transcript stores (the model's own
    dump): the loader validates with ``extra="forbid"``, so a row missing any
    field would be dropped rather than read, and this test would be measuring
    its own fixture."""
    row = {
        "id": wake_id,
        "message": f"message {wake_id}",
        "next_due_at": int(time.time() * 1000) + 3_600_000,
        "created_at": 1_700_000_000_000,
        "every_ms": None,
        "until_at": None,
        "limit": None,
        "fired_count": 0,
    }
    row.update(extra)
    return row


def _persisted_wakes(directory: Path) -> list[dict[str, Any]]:
    """The latest ``wake_schedules`` snapshot, read from the bytes on disk."""
    latest: list[dict[str, Any]] = []
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("custom_type") == "wake_schedules":
            latest = list((payload.get("details") or {}).get("schedules") or [])
    return latest


def _seed_wakes(directory: Path, rows: list[dict[str, Any]]) -> None:
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "seed-wakes",
                    "ts": 3.0,
                    "type": "custom",
                    "payload": {
                        "custom_type": "wake_schedules",
                        "details": {"schedules": rows},
                    },
                }
            )
            + "\n"
        )


def _seed_index(
    config_dir: Path, session_id: str, rows: list[dict[str, Any]], **extra: object
) -> Path:
    entry = {
        "schema": 1,
        "session_id": session_id,
        "cwd": "/work/here",
        "updated_at": 1,
        "schedules": rows,
    }
    entry.update(extra)
    path = config_dir / "wakes" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def test_a_seventeenth_wake_is_refused_instead_of_written_past_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cap belongs to the one validator, and this command used to build its
    schedule model directly — so a 17th wake went to disk as ``w17``."""
    from local_operator.cli import _wake_create

    directory = _session(tmp_path, "wakecreate01")
    _seed_wakes(directory, [_schedule_row(f"w{i}") for i in range(1, 17)])

    assert _wake_create(_args(message="one too many")) == 1

    assert "16" in capsys.readouterr().err
    rows = _persisted_wakes(directory)
    assert len(rows) == 16
    assert not {row["id"] for row in rows} & {"w17"}


def test_a_cancelled_wake_id_is_reissued_rather_than_duplicated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``w{len(existing) + 1}`` collided the moment an id was cancelled: with
    ``[w1, w3]`` on disk it handed out ``w3`` a second time."""
    from local_operator.cli import _wake_create

    directory = _session(tmp_path, "wakecreate01")
    # ``w2`` cancelled out of ``[w1, w2, w3]`` leaves two rows.
    _seed_wakes(directory, [_schedule_row("w1"), _schedule_row("w3")])

    assert _wake_create(_args(message="replacement")) == 0

    assert json.loads(capsys.readouterr().out)["wake_id"] == "w2"
    ids = [row["id"] for row in _persisted_wakes(directory)]
    assert ids == ["w1", "w3", "w2"]
    assert len(set(ids)) == 3


def test_arming_a_stopped_session_leaves_it_stopped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``stopped_at`` is the user's stop, and it lives in the derived index. The
    index write here passed no ``preserve``, so scheduling one more wake
    un-parked the whole session and dropped the lateness stamps with it."""
    from local_operator.cli import _wake_create
    from local_operator.wakes.store import read_entry

    directory = _session(tmp_path, "wakecreate01")
    _seed_wakes(directory, [_schedule_row("w1")])
    path = _seed_index(
        tmp_path,
        "wakecreate01",
        [_schedule_row("w1")],
        stopped_at=4321,
        last_fired_at=8765,
        last_attempt_at=9876,
    )

    assert _wake_create(_args(message="one more")) == 0
    capsys.readouterr()

    entry = read_entry(tmp_path, "wakecreate01")
    assert entry is not None, path
    assert entry["stopped_at"] == 4321
    assert entry["last_fired_at"] == 8765
    assert entry["last_attempt_at"] == 9876
    assert len(entry["schedules"]) == 2


# ---------------------------------------------------------------------------
# The rules this command no longer re-words, and the cap it gained
# ---------------------------------------------------------------------------
#
# The refactor moved the message-length bound into the shared validator, so
# `lop wake create --message <2001 chars>` is refused where the old command
# (which built the model directly) armed it — a fourth user-visible change.
# These pin the SENTENCES as the validator's, because that is the property the
# code claims: the floor and the bound-on-a-repeat used to print this command's
# own prose for the same mistakes.


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("every", "30s", "wake interval must be at least 60s."),
        ("limit", 3, "'until' and 'limit' bound a repeat — add an 'every' interval."),
        ("limit", 0, "'limit' must be a positive integer."),
    ],
)
def test_a_rule_refusal_is_the_shared_validators_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, value: object, expected: str
) -> None:
    from local_operator.cli import _wake_create

    session = _session(tmp_path, "wakecreate01")

    assert _wake_create(_args(**{field: value})) == 1

    assert expected in capsys.readouterr().err
    assert _persisted_wakes(session) == []


def test_the_message_cap_is_the_shared_validators(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bound at the boundary rather than somewhere inside it: the length that
    is allowed still arms, and the one past it is refused with the validator's
    sentence — the same text the desktop dialog and the agent's tool show."""
    from local_operator.cli import _wake_create
    from local_operator.harness.wake import MAX_WAKE_MESSAGE_CHARS

    session = _session(tmp_path, "wakecreate01")

    assert _wake_create(_args(message="x" * MAX_WAKE_MESSAGE_CHARS)) == 0
    assert len(_persisted_wakes(session)[0]["message"]) == MAX_WAKE_MESSAGE_CHARS

    assert _wake_create(_args(message="x" * (MAX_WAKE_MESSAGE_CHARS + 1))) == 1
    assert f"at most {MAX_WAKE_MESSAGE_CHARS} characters" in capsys.readouterr().err
    assert len(_persisted_wakes(session)) == 1


def test_a_live_owner_is_refused_with_a_sentence_that_says_what_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI is a second external writer, so it meets the same guard the
    route's cold path does: a runtime owns the schedules, and a row appended
    behind it would be deleted by its next persist without ever firing."""
    import local_operator.wakes.supervisor as supervisor
    from local_operator.cli import _wake_create

    session = _session(tmp_path, "wakecreate01")

    async def live(config_dir: Path, session_id: str) -> bool:
        return True

    monkeypatch.setattr(supervisor, "_has_live_runtime", live)

    assert _wake_create(_args()) == 1
    assert "Retry in a moment" in capsys.readouterr().err
    assert _persisted_wakes(session) == []

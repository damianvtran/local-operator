"""``lop model`` switches a REAL second session's model (design D1–D4, D6).

The motivating incident: four live sessions had to move from one provider to
another, and the only route that worked was typing ``/model`` into each pane.
These tests boot the PRODUCTION runtime (``session/runtime/process.py``) as a
separate process on the mock hosting, then run the PRODUCTION ``lop model``
command as another process against it, and read what a user would read: the
command's own output and exit code, and the target's durable transcript.

Isolation: the ``headless_tui_env`` fixture redirects the config dir and the root
conftest redirects ``HOME``. Every child environment here is rebuilt with every
``CMUX_*`` and ``LOP_*`` variable removed, then the names each child needs are set
explicitly — a runtime that inherited a workspace id could address the
operator's live window, and an inherited ``LOP_MOBILE_CHILD_*`` would make the
child something this test did not choose. Session ids are synthetic.

What this file proves end to end: the idle switch and its receipt, the durable
``selected_model`` / ``session_model_switch`` / audit-card rows, a refused pair
that changes nothing, the already-on answer, the stored-session refusal, and —
on a busy target — that the provider call ALREADY in flight finishes on the old
model while the next call in the SAME turn runs on the new one (read off the
``model_id`` each assistant row's ``usage`` records). It also proves that
``lop sessions`` carries the new model without waiting for a heartbeat (PR
#1555's change-driven republish), polled on the record. What it does not
prove: D7's older-peer mapping, which rests on
``tests/unit/mobile/test_peer_model_wire.py`` (a registrant that answers
``unknown op``) and on QA against a real older build.

``DEEPSEEK_API_KEY`` is a placeholder: it makes ``deepseek`` USABLE for the
target-side credential check (``ProviderController.is_usable`` reads env keys),
and no turn ever runs on it — the busy case switches onto a second mock model
instead, so no request leaves the machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S
from tests.e2e.harness import NO_NOTIFY_ENV
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

SESSION_ID = "peermodele2e1"


def _seed(config_dir: Path, session_id: str) -> Path:
    """An ENGAGED session (one durable user row) on the mock provider."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    # Auto-approve so the busy case's real `bash` call never parks on a gate.
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    return directory / "transcript.jsonl"


def _clean_env(config_dir: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    env.update(NO_NOTIFY_ENV)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env["DEEPSEEK_API_KEY"] = "e2e-placeholder-not-a-key"
    env.update(extra)
    return env


def _spawn_runtime(config_dir: Path, session_id: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=_clean_env(
            config_dir,
            LOP_MOBILE_CHILD_CWD=str(config_dir),
            LOP_MOBILE_CHILD_RESUME=session_id,
            # Long enough that the idle reaper never retires the target mid-test.
            LOP_SESSION_GRACE_S="120",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _lop(config_dir: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            *argv,
        ],
        env=_clean_env(config_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _record(config_dir: Path, session_id: str) -> Any:
    for record, _state in registry.scan(config_dir):
        if getattr(record, "session_id", "") == session_id:
            return record
    return None


def _wait(predicate, timeout: float, what: str) -> Any:
    """Poll ``predicate`` until truthy. Every caller also wraps it in ``bounded``,
    so a wedged predicate dumps its stacks instead of hanging the worker."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"{what} within {timeout}s")


def _rows(transcript: Path) -> list[dict[str, Any]]:
    rows = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _custom(rows: list[dict[str, Any]], custom_type: str) -> list[dict[str, Any]]:
    return [
        row.get("payload", {})
        for row in rows
        if row.get("payload", {}).get("custom_type") == custom_type
    ]


def _kill_all(child: subprocess.Popen[bytes], config_dir: Path) -> None:
    for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config_dir))}:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    child.wait(timeout=10)


def test_lop_model_switches_a_live_session_and_audits_it(headless_tui_env: Path) -> None:
    """Idle switch, then a refused pair that changes nothing, then already-on."""
    config = headless_tui_env
    transcript = _seed(config, SESSION_ID)
    child = _spawn_runtime(config, SESSION_ID)
    try:
        with bounded(60, "peer model: target publishes its record"):
            record = _wait(
                lambda: _record(config, SESSION_ID), 30, "the target never published a record"
            )
        assert record.started, "a resumed session with a durable row is engaged"
        name = record.conversation_name or record.session_id

        with bounded(90, "peer model: idle switch"):
            switched = _lop(config, "model", "--pid", str(record.pid), "deepseek/deepseek-flash")
        assert switched.returncode == 0, switched.stdout + switched.stderr
        assert switched.stdout.strip().splitlines() == [
            "switched to deepseek/deepseek-flash (was test/mock)",
            "its next turn runs on it",
            f"→ {name} (pid {record.pid})",
        ]

        # `lop sessions` shows the new model promptly (PR #1555's republish on
        # the push tick). Polled on the RECORD, never slept for. The bound is a
        # structural one, not a speed claim: well inside the first 15 s heartbeat,
        # so only the change-driven republish can have written it — the
        # heartbeat floor that used to be the only writer cannot pass this.
        with bounded(30, "peer model: the record carries the new model"):
            _wait(
                lambda: getattr(_record(config, SESSION_ID), "model_label", "")
                == "deepseek/deepseek-flash",
                HEARTBEAT_INTERVAL_S / 3,
                "`lop sessions`'s record kept the old model until the heartbeat",
            )
        listed = _lop(config, "sessions", "--json")
        assert listed.returncode == 0, listed.stderr
        rows = [row for row in json.loads(listed.stdout) if row["session_id"] == SESSION_ID]
        assert [row["model_label"] for row in rows] == ["deepseek/deepseek-flash"], rows

        with bounded(30, "peer model: transcript rows"):
            rows = _wait(
                lambda: (
                    (
                        lambda r: (
                            r
                            if _custom(r, "selected_model") and _custom(r, "peer_message")
                            else None
                        )
                    )(_rows(transcript))
                ),
                15,
                "the switch never reached the target's transcript",
            )
        # The audit card (D6): a record-only peer card naming from and to.
        cards = _custom(rows, "peer_message")
        assert len(cards) == 1
        # New model FIRST, then the old one. A CLI from a plain terminal is the
        # sender `terminal` (short, for the card header) and the body ends with
        # where it ran; never its own short-lived pid.
        body = cards[0]["details"]["body"]
        assert body.startswith(
            "[remote model switch] now on deepseek/deepseek-flash (was test/mock) — from a "
            "terminal in "
        ), body
        assert cards[0]["details"]["sender"]["conversation_name"] == "terminal"
        # The durable selection a resume keeps, and the model-visible notice.
        assert _custom(rows, "selected_model")[-1]["details"]["selector"] == (
            "deepseek/deepseek-flash"
        )
        assert _custom(rows, "session_model_switch"), "the switch notice was not journalled"
        # Record-only: the card must not have opened a turn.
        assert not [
            row for row in rows if row.get("payload", {}).get("role") == "assistant"
        ], "the audit card started a turn"

        # A pair the target cannot serve is REFUSED and nothing moves.
        with bounded(90, "peer model: refused switch"):
            refused = _lop(config, "model", "--pid", str(record.pid), "deepseek/not-a-model")
        assert refused.returncode == 1
        assert (
            "refused: 'not-a-model' is not a model deepseek serves; still on "
            "deepseek/deepseek-flash\n→ " in refused.stderr
        ), refused.stderr
        after = _rows(transcript)
        assert len(_custom(after, "peer_message")) == 1, "a refusal must not write a card"
        assert _custom(after, "selected_model")[-1]["details"]["selector"] == (
            "deepseek/deepseek-flash"
        ), "a refusal must not half-switch"

        unknown = _lop(config, "model", "--pid", str(record.pid), "nosuchprov/x")
        assert unknown.returncode == 1
        assert "refused: 'nosuchprov' is not a known provider" in unknown.stderr

        same = _lop(config, "model", "--pid", str(record.pid), "deepseek/deepseek-flash")
        assert same.returncode == 0, same.stderr
        assert same.stdout.startswith("already on deepseek/deepseek-flash\nnothing changed\n")
    finally:
        _kill_all(child, config)


def test_lop_model_on_a_busy_session_says_the_call_in_flight_finishes(
    headless_tui_env: Path,
) -> None:
    """D2: apply immediately; the result names the mid-turn semantics."""
    config = headless_tui_env
    transcript = _seed(config, SESSION_ID)
    child = _spawn_runtime(config, SESSION_ID)
    try:
        with bounded(60, "peer model busy: record"):
            record = _wait(
                lambda: _record(config, SESSION_ID), 30, "the target never published a record"
            )
        # A turn that holds the REAL bash tool for ~8 s (the mock's [bash:N]).
        woke = _lop(config, "send", "--pid", str(record.pid), "--wake", "please [bash:8]")
        assert woke.returncode == 0, woke.stdout + woke.stderr
        with bounded(60, "peer model busy: turn starts"):
            _wait(
                lambda: (r := _record(config, SESSION_ID)) is not None and r.busy,
                30,
                "the target never reported busy",
            )
        # The mock answers the turn's first provider call at once with the bash
        # call, so a short settle puts the switch inside the 8 s tool step. The
        # assistant row for that call is only persisted when the batch ends, so
        # it cannot be the event waited on; the settle is bounded well inside
        # the sleep, and the model assertion below is what proves ordering.
        time.sleep(1.0)
        with bounded(90, "peer model busy: switch"):
            switched = _lop(config, "model", "--pid", str(record.pid), "test/other-mock")
        assert switched.returncode == 0, switched.stdout + switched.stderr
        lines = switched.stdout.splitlines()
        assert lines[0] == "switched to test/other-mock (was test/mock)", switched.stdout
        # Which of the two true sentences depends on whether the first call has
        # been answered yet; the unit tests pin each wording to its state.
        assert lines[1] in (
            "mid-turn: the current step finishes on the old model",
            "mid-turn: the call in flight finishes on the old model",
        ), switched.stdout
        # THE SEMANTICS, not just the sentence: the call already answered ran on
        # the old model, and the NEXT call in the same turn — the one after the
        # bash step — ran on the new one. The mock stamps each reply's usage
        # with the spec it was built from.
        with bounded(60, "peer model busy: the turn finishes"):
            assistant = _wait(
                lambda: (lambda r: r if len(r) >= 2 else None)(
                    [
                        row["payload"]
                        for row in _rows(transcript)
                        if row.get("payload", {}).get("role") == "assistant"
                    ]
                ),
                30,
                "the busy turn never made its second provider call",
            )
        models = [(row.get("usage") or {}).get("model_id") for row in assistant]
        assert models[:2] == ["mock", "other-mock"], models
    finally:
        _kill_all(child, config)


def test_lop_model_refuses_a_session_that_is_not_running(headless_tui_env: Path) -> None:
    """Live only (D4): a stored session is named, with the two ways to switch it."""
    config = headless_tui_env
    _seed(config, "coldpeermodel1")
    result = _lop(config, "model", "--session", "coldpeermodel1", "deepseek/deepseek-flash")
    assert result.returncode == 1
    assert "session 'coldpeermodel1' is not running — open it and use /model" in result.stderr

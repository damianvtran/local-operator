"""A departure that caught nothing must not write an error row (2026-09-17).

THE OPERATOR'S REPORT, verbatim: a desktop turn showed "Stopped with an error" /
"[session incident] cut-off: the runtime retired so the next engage would run a
newer builddeclined 3x (0.56.2 → 0.56.6)", and "it seems that the conversation
continued smoothly after that". Three separate defects rode in that one card,
and each has a cell here:

* the sentence ran the detail into the words before it (``…newer
  builddeclined``) because ``_drain_detail`` is the one caller that does not
  spell the separator its readers assume;
* the why-now named a transition the install had left hours and two generations
  earlier — five latches at 01:58 named ``(0.56.2 → 0.56.6)`` and the record
  replaying one of them at 09:56 did so while 0.56.9 was on disk;
* the run the card described was one the exit had NOT cut. The latches that
  commit a runtime to leaving (``begin_retire``) refuse while anything is in
  flight, and the disposal synthesised an end for whatever run was left
  UNSETTLED — hours after its turn had stopped, against a conversation that then
  continued normally. Six durable ``error`` rows in the reporting host's
  ``attention.db`` carry such a run, rendered as an unexplained cut-off (the
  ``idle-exit`` label is a retirement label and not a cause in
  ``incidents.CUT_OFF_CAUSES``).

These cells drive the REAL exit ordering rather than the pieces: the reaper's
own idle rung over a real ``ServingSessionHandle`` and a real ``Session``, and
for the update itself the production ``process.py`` in a subprocess against a
fake install marker that is flipped the way ``lop-update`` flips it.

Isolation: the ``headless_tui_env`` fixture redirects the config dir and the
root conftest redirects ``HOME``; every ``CMUX_*`` variable is stripped from
children (a runtime that inherited a workspace id could address the operator's
live window, #648).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.incidents import render_cut_off_reason
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime import registry
from tests.e2e.harness import NO_NOTIFY_ENV
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: What ``lop-update`` writes last: the marker naming the build the next engage
#: will run. Two of them, because the incident's shape is a flip that happened
#: while the runtime was still working and ANOTHER that landed before it left.
OLD_MARKER = "46a4e9b1234567890abcdef v0.56.2\n"
NEW_MARKER = "f4a70b991234567890abcdef v0.56.6\n"


def _completion_rows(directory: Path) -> list[dict[str, Any]]:
    """The durable turn-outcome rows this session's transcript carries.

    The durable equivalent of the viewer's cut-off vocabulary: an exit that cut
    a turn writes ``error``, a reading that classifies one writes
    ``interrupted``, and a turn that finished writes ``complete``. Read from the
    JSONL because that is where the row outlives the process that wrote it. A
    row that carries no ``kind`` is the ``eligible: False`` marker a completed
    run with nothing to show writes, which is not an outcome at all.
    """
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written tail; the writer is still appending
        payload = parsed.get("payload")
        if (
            parsed.get("type") == "custom"
            and isinstance(payload, dict)
            and payload.get("custom_type") == "completion_attention"
        ):
            rows.append(dict(payload.get("details") or {}))
    return rows


class _Runtime:
    """The runtime-side collaborator the reaper probes, and nothing else.

    Deliberately NOT a double of the code under test: ``_reaper`` takes this as
    a parameter and reads four things from it (attached viewers, the boot stamp,
    the retiring announcement and ``aclose``), while the HANDLE below is the
    production one over a production ``Session``.
    """

    def __init__(self) -> None:
        self.retiring: list[tuple[str, str, bool, str]] = []
        self.closed = False

    def attach_clients(self) -> int:
        return 0

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

    async def aclose(self) -> None:
        self.closed = True


async def _cancel_a_run_mid_stream(session: Any) -> None:
    """Leave the shape a wedged turn leaves: a run that never published.

    A turn cancelled at its await in the provider stream is the shape
    ``Session.dispose`` documents itself as covering — the loop never reaches
    the ``AgentEndEvent`` it would otherwise yield, the pipeline's ``finally``
    runs ``_publish_attention_outcome`` with no outcome, and the run is left
    UNSETTLED. It is also this host's ordinary shape: the six ``idle-exit`` rows
    were runs whose last turn row preceded the row by minutes, i.e. turns that
    had died mid-flight while the session still looked idle.
    """
    import contextlib

    task = asyncio.ensure_future(session.prompt("a turn that will be cancelled"))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not session.is_streaming:
        await asyncio.sleep(0.01)
    assert session.is_streaming, "the turn never reached the provider stream"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert session._attention_run_settled is False, "this cell needs an unsettled run"


def _blocking_stream() -> Any:
    """A provider stream that never yields: the turn stays in flight.

    An async GENERATOR, because that is what ``stream_fn`` requires — a plain
    coroutine would type-check as a mistake and raise the first time a turn ran.
    """

    async def _stream(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.Event().wait()
        yield  # pragma: no cover — unreachable: the event is never set

    return _stream


async def _idle_exit_over_a_real_handle(config: Path) -> dict[str, Any]:
    """Run the reaper's quiet rung to completion and report what it published.

    The idle rung is driven through the REAL handle and the REAL session, so
    what is exercised is the production ordering: the latch, the grace loop, the
    dispose rung that writes the note, and the teardown that publishes the run's
    outcome. Only the runtime-side collaborator is a stub (see ``_Runtime``).
    """
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.test_session import make_session

    session = make_session(config, _blocking_stream())
    handle = ServingSessionHandle(
        session,
        asyncio.get_running_loop(),
        cwd=str(config),
        install_gates=False,
        config_dir=config,
    )
    runtime = _Runtime()
    stop = asyncio.Event()
    await _cancel_a_run_mid_stream(session)
    await asyncio.wait_for(child_mod._reaper(handle, runtime, stop), timeout=30)
    assert stop.is_set(), "the idle rung never took the exit"
    assert runtime.closed, "the runtime was not torn down"
    return {"retiring": runtime.retiring, "pid_gone": True}


@pytest.mark.asyncio
async def test_an_idle_exit_that_caught_nothing_publishes_no_error(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's own rows, on the rung that wrote them.

    Six ``error`` rows in ``attention.db`` carry ``cause="idle-exit"`` — a token
    that is a RETIREMENT LABEL and not a member of ``incidents.CUT_OFF_CAUSES``,
    so every one of them rendered as "the turn was cut off and the cause could
    not be determined", and the next engage was handed a cut-off incident card
    for a run that had already ended. The quiet rung proves nothing was in
    flight before it latches; arming a cut-off there claimed the opposite.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.1")
    config = headless_tui_env

    with bounded(60, "the reaper's quiet rung over a real handle"):
        await _idle_exit_over_a_real_handle(config)

    rows = _completion_rows(config / "sess")
    errors = [row for row in rows if row.get("kind") == "error"]
    assert errors == [], f"an idle exit that caught nothing reported an error: {errors!r}"
    assert not [
        row for row in rows if row.get("cause") == "idle-exit"
    ], "the retirement label reached the durable outcome as a cut-off cause"
    # The run is SETTLED rather than left open for a successor, and settled
    # with no verdict: this is the marker a complete run with nothing to show
    # already writes, so no reader learns a new shape and no later party
    # classifies the run (see ``Session._settle_run_without_an_outcome``).
    assert rows and rows[-1].get("eligible") is False, rows


@pytest.mark.asyncio
async def test_a_retirement_cuts_nothing_and_a_live_turn_still_reports_it(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both directions of the retirement family, where they actually occur.

    THE REACHABLE ORDER (agent review round 1, MINOR-1). ``begin_retire``
    refuses while anything would be lost (``may_refresh``), so a retirement that
    SUCCEEDED is proof no turn was in flight: the only run it can reach is one
    that was already orphaned. This cell used to assert the opposite — it
    latched a retirement over an orphan and required an ``error`` row carrying
    ``runtime-retired`` — which is precisely the disposition the operator's own
    ``attention.db`` is full of (agent review round 1, MAJOR-1; QA round 1, Q1).

    Half A walks the real latch over a real orphan and asserts nothing is
    reported at all, and that the run is SETTLED rather than left for a
    successor (silence is only safe because the token is closed). Half B is the
    guard rail: a turn that IS in flight when the disposal takes the exit must
    still be recorded as a cut-off, with the cause this exit can attest to.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.test_session import make_session

    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.1")
    config = headless_tui_env

    with bounded(60, "a latched retirement over an orphan, and a live turn cut"):
        # -- half A: the latch a build drain takes, over an orphaned run ------–
        session = make_session(config, _blocking_stream())
        handle = ServingSessionHandle(
            session,
            asyncio.get_running_loop(),
            cwd=str(config),
            install_gates=False,
            config_dir=config,
        )
        await _cancel_a_run_mid_stream(session)
        assert handle.begin_retire("runtime-retired", " (0.56.2 → 0.56.6)") is True
        assert session._cut_off_cause == "", "a latch may not arm a turn"
        await handle.dispose()

        rows = _completion_rows(config / "sess")
        assert [row for row in rows if row.get("kind") == "error"] == [], rows
        assert not [row for row in rows if row.get("cause") == "runtime-retired"], rows
        assert rows and rows[-1].get("eligible") is False, rows

        # -- half B: the same disposal over a turn that IS in flight ----------
        live_root = config / "live"
        live_root.mkdir(parents=True, exist_ok=True)
        live = make_session(live_root, _blocking_stream())
        live_handle = ServingSessionHandle(
            live,
            asyncio.get_running_loop(),
            cwd=str(config),
            install_gates=False,
            config_dir=config,
        )
        task = asyncio.ensure_future(live.prompt("a turn the exit will cut"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not live.is_streaming:
            await asyncio.sleep(0.01)
        assert live.is_streaming, "the turn never reached the provider stream"
        # The latch refuses while it is live — that refusal IS half A's premise —
        # so the cut is the disposal's, and it says so.
        assert live_handle.begin_retire("runtime-retired", " (0.56.2 → 0.56.6)") is False
        await live_handle.dispose()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    cut = [row for row in _completion_rows(live_root / "sess") if row.get("kind") == "error"]
    assert len(cut) == 1, f"a cut turn must still be recorded as one: {cut!r}"
    assert cut[0]["cause"] == "runtime-shutdown", cut[0]
    assert cut[0]["reason"] == render_cut_off_reason("runtime-shutdown"), cut[0]["reason"]
    assert "builddeclined" not in str(cut[0]["reason"])


def _child_env(config_dir: Path, prefix: Path, session_id: str, **extra: str) -> dict[str, str]:
    """A runtime child's environment, with every inherited pane variable gone.

    The families are stripped rather than the ``CMUX_*`` prefix alone:
    ``LOP_RUNTIME_*`` and ``LOP_MOBILE_CHILD_*`` decide what a child is, so a
    cell run from inside another session would otherwise inherit its session,
    provider and model — a plausible-looking cell that proves nothing.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOP_RUNTIME_", "LOP_MOBILE_CHILD_"))
    }
    env.update(NO_NOTIFY_ENV)
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "LOP_MOBILE_CHILD_CWD": str(config_dir),
            "LOP_MOBILE_CHILD_RESUME": session_id,
            "LOP_BUILD_PREFIX": str(prefix),
            # Short, so the exit is observable inside one watchdog budget, and
            # long enough that a marker written mid-cell is not acted on before
            # it is whole.
            "LOP_BUILD_SETTLE_S": "0.5",
            "LOP_BUILD_STAGGER_S": "0.5",
            # Long, so the QUIET rung can never be what retires this runtime:
            # these cells are about the build handover.
            "LOP_SESSION_GRACE_S": "600",
        }
    )
    env.update(extra)
    return env


def _seed(config_dir: Path, session_id: str) -> Path:
    """A session with one durable row, on the mock provider."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    return directory


@pytest.mark.asyncio
async def test_an_idle_update_retires_writes_no_error_row_and_re_engages_cleanly(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's scenario end to end: an update under an idle runtime.

    Boot the production runtime, run a real turn through the real viewer, flip
    the install marker the way ``lop-update`` does, and assert the two things
    the operator asked for: nothing is recorded as an error, and the next engage
    runs the new build cleanly.

    WHAT THIS CELL DOES NOT DISCRIMINATE, stated rather than implied (agent
    review round 1, MINOR-3): the turn COMPLETES, and its row is asserted before
    the marker flips, so the end event is emitted long before any latch arms and
    this cell would pass against the pre-fix tree too. It is kept for the
    END-TO-END claim it does prove — a real update under a real runtime records
    no error and re-engages on the new build — while the misclassification it was
    originally written for is staged by
    ``test_a_retirement_cuts_nothing_and_a_live_turn_still_reports_it`` and by
    ``tests/unit/session/test_cut_off_turns.py``.
    """
    from tests.e2e.test_cut_off_turns_e2e import _attach, _wait_for_record

    config = headless_tui_env
    session_id = "retireclean1"
    _seed(config, session_id)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / ".lop-source").write_text(OLD_MARKER, encoding="utf-8")
    for name, value in (
        ("LOP_BUILD_PREFIX", str(prefix)),
        ("LOP_BUILD_SETTLE_S", "0.5"),
        ("LOP_BUILD_STAGGER_S", "0.5"),
    ):
        monkeypatch.setenv(name, value)

    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=_child_env(config, prefix, session_id),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        viewer = await _attach(config, session_id)
        await viewer.prompt("say hello")
        deadline = time.monotonic() + 60
        directory = config / "sessions" / session_id
        while time.monotonic() < deadline:
            rows = _completion_rows(directory)
            if any(row.get("kind") == "complete" for row in rows):
                break
            await asyncio.sleep(0.2)
        completed = [row for row in _completion_rows(directory) if row.get("kind")]
        assert completed and completed[0]["kind"] == "complete", completed

        (prefix / ".lop-source").write_text(NEW_MARKER, encoding="utf-8")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and child.poll() is None:
            await asyncio.sleep(0.05)
        assert child.poll() is not None, "the stale runtime never retired"
        assert child.returncode == 0, f"the retirement was not a clean exit: {child.returncode}"

        rows = _completion_rows(directory)
        errors = [row for row in rows if row.get("kind") == "error"]
        assert errors == [], f"an idle update produced an error trace: {errors!r}"

        # …and the next engage runs the NEW build, with no successor spawned by
        # this unwatched retirement.
        await asyncio.sleep(0.5)
        assert [r for r, _ in registry.scan(config) if r.session_id == session_id] == []
        env = _child_env(config, prefix, session_id)
        env.pop("LOP_MOBILE_CHILD_RESUME")
        env.pop("LOP_MOBILE_CHILD_CWD")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from local_operator.cli import main; sys.exit(main())",
                "send",
                "--session",
                session_id,
                "--wake",
                "hello",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        record = await _wait_for_record(config, session_id)
        assert record.source_ref == NEW_MARKER.split()[0], "the next engage ran the old build"
    finally:
        for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)

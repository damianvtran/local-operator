"""The session arm's completion verifier: bridge finish branch, inspect tool, records.

What is pinned, against the design (verifier-design.md, "Coder brief"):

* PASS accepts; FAIL/INCONCLUSIVE becomes the finish challenge carrying the
  findings and is accounted as the self-check challenge; the last round
  always accepts; a budget that cannot hold the exchange records a skip and
  takes today's path; premature/failed claims are untouched.
* ``inspect_command`` is refused outside the verification window and does
  not deadlock while the finish branch holds the bridge lock.
* rows of the verifier enter the request summary with their job id.
* ``rounds=0`` adds nothing to any record.
* end to end through the REAL ``Session.verify_completion``: the FAIL
  findings reach the worker, the worker fixes, the second finish is accepted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.evaluation.completion_verifier import (
    MAX_VERIFIER_WALL_S,
    SANDBOX_PROBE,
    SANDBOX_TOKEN,
    CompletionVerifierConfig,
    VerifierInspector,
    obvious_write,
    sandboxed,
)
from local_operator.evaluation.runner.episode import EpisodeConfig
from local_operator.evaluation.session_arm import (
    SessionArmOutcome,
    _flush_request_observation,
    _outcome_json,
    _verifier_summary,
)
from local_operator.harness.verifier import VerifierVerdict
from tests.unit.evaluation.test_session_arm import _bridge, _command_result

DONE = {"actions": [{"kind": "finish", "status": "done", "reason": "B2 now holds 42"}]}
WAIT = {"actions": [{"kind": "wait", "duration_ms": 50}]}


def _verdict(status: str, findings: str = "", ended_by: str = "verdict") -> VerifierVerdict:
    return VerifierVerdict(
        status=status, findings=findings, ended_by=ended_by, job_id="vjob"  # type: ignore[arg-type]
    )


def _armed(tmp_path: Path, verdicts: list[VerifierVerdict], *, rounds: int = 1, **kwargs: Any):
    actions: list[tuple[str, dict[str, Any]]] = []
    events: list[tuple[str, dict[str, Any]]] = []
    seen: list[tuple[Any, Any, int]] = []
    bridge = _bridge(
        tmp_path, record=lambda kind, payload: actions.append((kind, payload)), **kwargs
    )

    async def verifier(claim: Any, pending: Any, round_: int) -> VerifierVerdict:
        seen.append((claim, pending, round_))
        return verdicts[min(len(seen) - 1, len(verdicts) - 1)]

    bridge.verifier = verifier
    bridge.verifier_rounds = rounds
    bridge.verifier_envelope_s = 240.0  # verifier_envelope_s(180): 50 + 180 + 10
    bridge.record_event = lambda kind, payload: events.append((kind, payload))
    return bridge, actions, events, seen


# -- 5. the bridge's finish branch --------------------------------------------


@pytest.mark.asyncio
async def test_pass_accepts_the_first_done_claim(tmp_path: Path) -> None:
    bridge, actions, events, seen = _armed(tmp_path, [_verdict("PASS", "all met")])
    reply = await bridge.call(DONE)
    assert reply["details"]["terminal"] == "finish"
    assert bridge.end_requested == "finish" and len(seen) == 1
    # No self-check challenge on top of an independent PASS.
    assert [k for k, _ in actions if k == "completion_challenged"] == []
    [finish] = [p for k, p in actions if k == "finish"]
    assert finish["verified"] == "PASS" and finish["completion_challenged"] is False
    assert [k for k, _ in events] == ["verifier_verdict"]
    assert events[0][1]["round"] == 1 and events[0][1]["status"] == "PASS"
    assert bridge.steps == 1


@pytest.mark.asyncio
async def test_fail_becomes_the_challenge_with_findings(tmp_path: Path) -> None:
    bridge, actions, _events, _seen = _armed(
        tmp_path, [_verdict("FAIL", "B2 is empty (read: openpyxl B2=None)")]
    )
    reply = await bridge.call(DONE)
    assert reply["details"]["terminal"] == "completion-challenged"
    text = reply["content"][0].text
    assert text.startswith("An independent check of your deliverable reported (FAIL):")
    assert "B2 is empty (read: openpyxl B2=None)" in text
    # The shared challenge follows, unchanged.
    assert "That declaration is a CLAIM" in text and "B2 now holds 42" in text
    assert bridge.end_requested is None and bridge.completion_challenges_fired == 1
    # Accounted exactly as the self-check challenge: one epoch, same kind.
    assert [p for k, p in actions if k == "output_epoch"] == [
        {"epoch_kind": "finish-challenged", "steps": 1}
    ]
    [row] = [p for k, p in actions if k == "completion_challenged"]
    assert row["trigger"] == "verifier" and row["verdict"] == "FAIL"


@pytest.mark.asyncio
async def test_inconclusive_also_challenges(tmp_path: Path) -> None:
    bridge, _actions, events, _seen = _armed(
        tmp_path, [_verdict("INCONCLUSIVE", "could not open the file")]
    )
    reply = await bridge.call(DONE)
    assert reply["details"]["terminal"] == "completion-challenged"
    assert "(INCONCLUSIVE)" in reply["content"][0].text
    # Round 3: the record says which way INCONCLUSIVE went, so an ablation
    # never has to infer it from ``status``.
    [(kind, row)] = events
    assert kind == "verifier_verdict" and row["disposition"] == "challenged"


@pytest.mark.asyncio
async def test_verdict_rows_record_the_disposition(tmp_path: Path) -> None:
    for status, disposition in (("PASS", "accepted"), ("FAIL", "challenged")):
        (tmp_path / status).mkdir()
        bridge, _actions, events, _seen = _armed(tmp_path / status, [_verdict(status)])
        await bridge.call(DONE)
        assert [row["disposition"] for kind, row in events if kind == "verifier_verdict"] == [
            disposition
        ], status


def test_a_workers_own_verifier_task_is_not_the_verifier_share() -> None:
    """m3: rows of a worker's ``task(agent="verifier")`` carry the role name
    but not a recorded verifier job id, so they never enter the share -- and
    with the verifier off the record keeps no ``verifier`` key at all."""

    from local_operator.harness.types import Usage

    worker_child = _row("r2", actor_role="verifier", job_id="worker-task")
    rows = [_row("r1", actor_role=None, job_id=None), worker_child]
    usages = [Usage.model_validate(r["usage"]) for r in rows]
    written: list[tuple[str, dict[str, Any]]] = []
    sink = SimpleNamespace(write=lambda k, p: written.append((k, p)))
    off = SimpleNamespace(request_rows=rows, request_usages=usages, verifier_job_ids=set())
    _flush_request_observation(sink, off)  # type: ignore[arg-type]
    assert "verifier" not in written[-1][1]
    on = SimpleNamespace(
        request_rows=[*rows, _row("r3", actor_role="verifier", job_id="vjob")],
        request_usages=usages,
        verifier_job_ids={"vjob"},
    )
    _flush_request_observation(sink, on)  # type: ignore[arg-type]
    assert written[-1][1]["verifier"]["job_ids"] == ["vjob"]
    assert written[-1][1]["verifier"]["count"] == 1


@pytest.mark.asyncio
async def test_the_last_round_always_accepts(tmp_path: Path) -> None:
    bridge, _actions, _events, seen = _armed(tmp_path, [_verdict("FAIL", "no")], rounds=1)
    assert (await bridge.call(DONE))["details"]["terminal"] == "completion-challenged"
    second = await bridge.call(DONE)
    assert second["details"]["terminal"] == "finish" and len(seen) == 1


@pytest.mark.asyncio
async def test_two_rounds_verify_twice_then_accept(tmp_path: Path) -> None:
    bridge, _a, _e, seen = _armed(tmp_path, [_verdict("FAIL", "no")], rounds=2, max_steps=10)
    assert (await bridge.call(DONE))["details"]["terminal"] == "completion-challenged"
    assert (await bridge.call(DONE))["details"]["terminal"] == "completion-challenged"
    assert (await bridge.call(DONE))["details"]["terminal"] == "finish"
    assert [round_ for _c, _p, round_ in seen] == [1, 2]


@pytest.mark.asyncio
async def test_skipped_when_the_step_budget_cannot_hold_the_exchange(tmp_path: Path) -> None:
    bridge, _a, events, seen = _armed(tmp_path, [_verdict("FAIL", "no")], max_steps=1)
    await bridge.call(DONE)
    assert seen == []
    assert events == [
        ("verifier_skipped", {"round": 1, "reason": "step-budget", "steps": 0, "max_steps": 1})
    ]


@pytest.mark.asyncio
async def test_skipped_when_the_wall_cannot_hold_the_verifier(tmp_path: Path) -> None:
    bridge, actions, events, seen = _armed(tmp_path, [_verdict("PASS")])
    bridge.wall_remaining = lambda: 60.0  # < verifier_envelope_s 240
    reply = await bridge.call(DONE)
    assert seen == [] and events[0][1]["reason"] == "wall"
    # Today's path: the self-check challenge fires as it always did.
    assert reply["details"]["terminal"] == "completion-challenged"
    assert "An independent check" not in reply["content"][0].text


@pytest.mark.asyncio
async def test_refused_verdict_is_a_skip_on_todays_path(tmp_path: Path) -> None:
    bridge, _a, events, _seen = _armed(
        tmp_path, [_verdict("INCONCLUSIVE", "memory admission refused", ended_by="refused")]
    )
    reply = await bridge.call(DONE)
    assert events == [
        (
            "verifier_skipped",
            {"round": 1, "reason": "refused", "detail": "memory admission refused"},
        )
    ]
    assert "An independent check" not in reply["content"][0].text


@pytest.mark.asyncio
async def test_premature_and_failed_claims_are_unchanged(tmp_path: Path) -> None:
    bridge, _a, _e, seen = _armed(tmp_path, [_verdict("FAIL", "no")])
    failed = {"actions": [{"kind": "finish", "status": "failed", "reason": "cannot"}]}
    reply = await bridge.call(failed)
    # The premature-terminal rule still fires (no action executed yet) ...
    assert reply["details"]["terminal"] == "completion-challenged"
    assert "before any action has been executed" in reply["content"][0].text
    # ... and the verifier never ran for a non-``done`` claim.
    assert seen == []


@pytest.mark.asyncio
async def test_a_raising_verifier_is_a_skip(tmp_path: Path) -> None:
    bridge, _a, events, _seen = _armed(tmp_path, [])

    async def broken(*_args: Any) -> Any:
        raise RuntimeError("boom")

    bridge.verifier = broken
    reply = await bridge.call(DONE)
    assert events[0][0] == "verifier_skipped" and events[0][1]["reason"] == "error"
    assert reply["details"]["terminal"] == "completion-challenged"  # today's path


# -- 6. inspect_command: window gate and no deadlock ------------------------------


@pytest.mark.asyncio
async def test_inspect_is_refused_outside_the_window_and_reaches_nothing(tmp_path: Path) -> None:
    ran: list[str] = []

    async def run_command(command: str, timeout_s: float) -> Any:
        ran.append(command)
        return _command_result()

    bridge = _bridge(tmp_path)
    inspector = VerifierInspector(run_command=run_command, window=lambda: bridge.verification_round)
    text, is_error = await inspector.inspect({"command": "cat /etc/hostname"})
    assert is_error and "only available while a completion check is running" in text
    assert ran == []


@pytest.mark.asyncio
async def test_inspect_inside_the_finish_branch_does_not_deadlock(tmp_path: Path) -> None:
    ran: list[str] = []
    rows: list[tuple[str, dict[str, Any]]] = []

    async def run_command(command: str, timeout_s: float) -> Any:
        ran.append(command)
        if command == SANDBOX_PROBE:
            return _command_result(stdout=SANDBOX_TOKEN + "\n", stdout_bytes=18)
        return _command_result(stdout="42\n", stdout_bytes=3)

    bridge = _bridge(tmp_path)
    bridge.run_command = run_command
    inspector = VerifierInspector(
        run_command=run_command,
        window=lambda: bridge.verification_round,
        record_event=lambda kind, payload: rows.append((kind, payload)),
    )
    outputs: list[tuple[str, bool]] = []

    async def verifier(claim: Any, pending: Any, round_: int) -> VerifierVerdict:
        # The bridge lock is HELD here (``call`` is mid-finish).
        assert bridge._lock.locked()
        outputs.append(await inspector.inspect({"command": "cat B2.txt"}))
        return _verdict("PASS")

    bridge.verifier, bridge.verifier_rounds = verifier, 1
    reply = await asyncio.wait_for(bridge.call(DONE), timeout=5)
    assert reply["details"]["terminal"] == "finish"
    assert outputs[0][1] is False and "42" in outputs[0][0]
    # Sandboxed: the command was wrapped, the probe recorded.
    assert ran[0] == SANDBOX_PROBE and ran[1] == sandboxed("cat B2.txt")
    kinds = [(k, p.get("probe", False), p.get("readonly_enforcement")) for k, p in rows]
    assert kinds == [
        ("verifier_command", True, "sandboxed"),
        ("verifier_command", False, "sandboxed"),
    ]
    # After the window closed, the same tool is refused again.
    assert (await inspector.inspect({"command": "ls"}))[1] is True


@pytest.mark.asyncio
async def test_audit_mode_refuses_obvious_writes_and_sweeps_changes() -> None:
    ran: list[str] = []

    async def run_command(command: str, timeout_s: float) -> Any:
        ran.append(command)
        if command == SANDBOX_PROBE:
            return _command_result(exit_code=3, stdout="", stdout_bytes=0)
        if command == "date +%s":
            return _command_result(stdout="1700000000\n", stdout_bytes=11)
        if command.startswith("find "):
            return _command_result(stdout="/home/user/.config/x\n", stdout_bytes=21)
        return _command_result()

    rows: list[dict[str, Any]] = []
    inspector = VerifierInspector(
        run_command=run_command, window=lambda: 1, record_event=lambda _k, p: rows.append(p)
    )
    text, refused = await inspector.inspect({"command": "echo x > /home/user/file"})
    assert refused and "looks like it writes" in text
    assert inspector.mode == "audit"
    text, refused = await inspector.inspect({"command": "cat /home/user/file"})
    assert not refused and "cat /home/user/file" in ran
    last = rows[-1]
    assert last["readonly_enforcement"] == "audit"
    assert last["changed_files"] == ["/home/user/.config/x"] and last["mutation_detected"] is True
    assert "-newermt @1700000000" in next(c for c in ran if c.startswith("find "))


def test_obvious_write_lexicon() -> None:
    for command in (
        "rm -f a",
        "sed -i s/a/b/ f",
        "cp a b",
        "python3 -c \"open('f','w')\"",
        "echo hi >> f",
        "soffice --headless --convert-to pdf a.docx",
    ):
        assert obvious_write(command), command
    for command in (
        "cat a 2>/dev/null",
        "python3 -c \"import openpyxl; print(openpyxl.load_workbook('a.xlsx')['S']['B2'].value)\"",
        "ls -la ~/Desktop",
        "grep -c x f 2>&1",
        "pdftotext a.pdf -",
    ):
        assert not obvious_write(command), command


# -- 9. verifier rows in the cost summary -------------------------------------


def _row(request_id: str, *, actor_role: str | None, job_id: str | None) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "actor_role": actor_role,
        "job_id": job_id,
        "purpose": "turn",
        "usage_coverage": "reported_final",
        "usage": {"input": 10, "output": 5, "usd_cost": 0.01, "provider": "p", "model_id": "m"},
    }


def test_verifier_rows_enter_the_request_summary() -> None:
    from local_operator.harness.types import Usage

    rows = [
        _row("r1", actor_role=None, job_id=None),
        _row("r2", actor_role="verifier", job_id="vjob"),
        _row("r3", actor_role="verifier", job_id="vjob"),
    ]
    written: list[tuple[str, dict[str, Any]]] = []
    record = SimpleNamespace(write=lambda kind, payload: written.append((kind, payload)))
    handle = SimpleNamespace(
        request_rows=rows,
        request_usages=[Usage.model_validate(r["usage"]) for r in rows],
        verifier_job_ids={"vjob"},
    )
    _flush_request_observation(record, handle)  # type: ignore[arg-type]
    [(kind, payload)] = written
    assert kind == "requests" and payload["count"] == 3  # whole tree, verifier included
    share = payload["verifier"]
    assert share["count"] == 2 and share["job_ids"] == ["vjob"]
    assert share["recorded_usd"] == pytest.approx(0.02)


# -- 10. rounds=0 adds nothing -------------------------------------------------


def test_rounds_zero_adds_no_key_anywhere(tmp_path: Path) -> None:
    assert EpisodeConfig(
        evidence_root=tmp_path, artifact_root=tmp_path, rescue_root=tmp_path
    ).completion_verifier == CompletionVerifierConfig(rounds=0)
    assert CompletionVerifierConfig().enabled is False
    # The bridge's default carries no verifier.
    assert _bridge(tmp_path).verifier is None
    # No ``verifier`` share when no verifier row exists.
    written: list[tuple[str, dict[str, Any]]] = []
    handle = SimpleNamespace(
        request_rows=[_row("r1", actor_role=None, job_id=None)], request_usages=[]
    )
    sink = SimpleNamespace(write=lambda k, p: written.append((k, p)))
    _flush_request_observation(sink, handle)  # type: ignore[arg-type]
    assert "verifier" not in written[0][1]
    # And the outcome has no key at all.
    outcome = SessionArmOutcome(status="completed", episode_id="e", record_root=None)
    assert "completion_verifier" not in _outcome_json(outcome)
    assert _verifier_summary(None) is None


def test_config_refuses_out_of_range() -> None:
    bad_values: list[dict[str, Any]] = [
        {"rounds": 3},
        {"rounds": True},
        {"wall_s": MAX_VERIFIER_WALL_S + 1},
        {"max_commands": 0},
        {"local_tool_allowance": -1},
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            CompletionVerifierConfig(**bad)
    # The request cap is derived, never an independent constant: 10 commands
    # + 4 local reads + the refused call + the verdict.
    assert CompletionVerifierConfig(rounds=1).stamp() == {
        "rounds": 1,
        "max_requests": 16,
        "max_commands": 10,
        "local_tool_allowance": 4,
        "wall_s": 180.0,
    }


# -- end to end: FAIL findings reach the worker, it fixes, the next finish passes --


@pytest.mark.asyncio
async def test_end_to_end_fail_fix_pass_through_the_real_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge, the arm's wiring (``_arm_completion_verifier``), the real
    ``Session.verify_completion`` and a real verifier child session; only the
    model wire and the guest are scripted. The worker's three calls are the
    "model": a wrong ``done``, the fix, the same ``done`` again."""

    from local_operator.evaluation.session_arm import (
        _arm_completion_verifier,
        _copy_verifier_transcripts,
    )
    from local_operator.harness.types import ChatRequest, TurnEndEvent
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from tests.unit.session.test_launch_subagent import MODEL
    from tests.unit.session.test_verify_completion import (
        TreeStream,
        is_verifier,
        text_turn,
        tool_turn,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    guest = {"B2": "None"}

    async def run_command(command: str, timeout_s: float) -> Any:
        if command == SANDBOX_PROBE:
            return _command_result(stdout=SANDBOX_TOKEN + "\n", stdout_bytes=18)
        value = guest["B2"] + "\n"
        return _command_result(stdout=value, stdout_bytes=len(value))

    def reply(request: ChatRequest) -> list[Any]:
        assert is_verifier(request), "only the verifier talks to the model here"
        results = [m for m in request.messages if m.role in ("tool", "toolResult")]
        if not results:
            return tool_turn("inspect_command", {"command": "read B2"})
        seen = "".join(getattr(block, "text", "") for block in results[-1].content)
        if "42" in seen:
            return text_turn("B2 holds 42: MET\nVERDICT: PASS")
        return text_turn("B2 is empty (read None): UNMET\nVERDICT: FAIL")

    parent = Session(
        model=MODEL,
        stream_fn=TreeStream(reply),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )

    executed: list[Any] = []
    bridge = _bridge(tmp_path, max_steps=10, instruction="Put 42 in cell B2.")
    original_execute = bridge.execute

    async def execute(batch: Any) -> Any:
        executed.append(batch)
        guest["B2"] = "42"  # the worker's fix lands on the guest
        return await original_execute(batch)

    bridge.execute = execute
    bridge.run_command = run_command
    written: list[tuple[str, dict[str, Any]]] = []
    record = SimpleNamespace(write=lambda kind, payload: written.append((kind, payload)))
    handle = SimpleNamespace(session=parent, verifier_job_ids=set())
    state = _arm_completion_verifier(
        bridge=bridge,
        handle=handle,  # type: ignore[arg-type]
        record=record,  # type: ignore[arg-type]
        config=CompletionVerifierConfig(rounds=2),
        wall_remaining=None,
    )
    try:
        first = await asyncio.wait_for(bridge.call(DONE), timeout=20)
        assert first["details"]["terminal"] == "completion-challenged"
        assert "B2 is empty (read None): UNMET" in first["content"][0].text
        assert executed == []  # a challenged finish moves nothing
        # Each worker call is its own turn, as the real loop delivers them.
        bridge.fold(TurnEndEvent())
        fix = await bridge.call(WAIT)
        assert fix["is_error"] is False and len(executed) == 1
        bridge.fold(TurnEndEvent())
        second = await asyncio.wait_for(bridge.call(DONE), timeout=20)
        assert second["details"].get("terminal") == "finish", second
        assert bridge.end_requested == "finish"
    finally:
        await parent.dispose()

    kinds = [kind for kind, _ in written]
    assert kinds.count("verifier_started") == 2 and kinds.count("verifier_verdict") == 2
    verdicts = [p["status"] for k, p in written if k == "verifier_verdict"]
    assert verdicts == ["FAIL", "PASS"]
    started = [p for k, p in written if k == "verifier_started"]
    assert all(p["readonly_enforcement"] == "sandboxed" for p in started)
    assert "inspect_command" in started[0]["tools"]
    assert not {"write", "edit", "task", "web_search", "web_fetch"} & set(started[0]["tools"])
    commands = [p for k, p in written if k == "verifier_command" and not p.get("probe")]
    assert [p["job_id"] for p in commands] == [p["job_id"] for p in started]
    assert [v["status"] for v in state["verdicts"]] == ["FAIL", "PASS"]
    # The bundle gets each verifier's transcript.
    _copy_verifier_transcripts(tmp_path / "bundle", state, record)  # type: ignore[arg-type]
    assert sorted(state["bundle_transcripts"]) == sorted(p["job_id"] for p in started)
    for relative in state["bundle_transcripts"].values():
        assert (tmp_path / "bundle" / relative).stat().st_size > 0


# -- review round 1 regressions (PR #3837) -------------------------------------


def _fake_bwrap(tmp_path: Path, *, isolate: bool) -> Path:
    """A ``bwrap`` stand-in on PATH: logs its argv, then runs the command after
    ``--`` WITHOUT any isolation -- the negative control a real probe must
    catch. ``isolate=False`` is the only mode; the name keeps intent visible."""

    assert not isolate
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "bwrap"
    fake.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/argv"\n'
        'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done\n'
        "shift\n"
        'exec "$@"\n'
    )
    fake.chmod(0o755)
    return bin_dir


def test_probe_refuses_a_sandbox_that_does_not_enforce(tmp_path: Path) -> None:
    """R1: the probe runs the REAL wrapper text through ``/bin/sh``. A bwrap
    that isolates nothing must never yield the token, and the wrapper must ask
    for every isolation ``SANDBOX_GUARANTEES`` names."""
    import os
    import subprocess

    from local_operator.evaluation.completion_verifier import (
        _DESKTOP_ENV,
        _SOCKET_DIRS,
        SANDBOX_GUARANTEES,
    )

    bin_dir = _fake_bwrap(tmp_path, isolate=False)
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(home), "DISPLAY": ":0"}
    out = subprocess.run(
        ["/bin/sh", "-c", SANDBOX_PROBE], env=env, capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    assert SANDBOX_TOKEN not in out.stdout and "WRITABLE" in out.stdout, out.stdout
    argv = (tmp_path / "argv").read_text().splitlines()
    for flag in (
        "--ro-bind",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--die-with-parent",
    ):
        assert flag in argv, flag
    # Socket dirs are hidden exactly where they exist on this host.
    hidden = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--tmpfs"]
    assert hidden == [d for d in _SOCKET_DIRS if os.path.isdir(d)]
    after = argv[argv.index("--") + 1 :]
    for name in _DESKTOP_ENV:
        assert after[after.index(name) - 1] == "-u", name
    assert "private-pid-namespace" in SANDBOX_GUARANTEES

    # The wrapper really strips the desktop env from the command it runs.
    wire = sandboxed('echo "display=${DISPLAY:-unset}"')
    shown = subprocess.run(["/bin/sh", "-c", wire], env=env, capture_output=True, text=True)
    assert shown.stdout.strip() == "display=unset", shown


@pytest.mark.asyncio
async def test_sandboxed_mode_records_unknown_mutation_and_still_refuses_writes() -> None:
    """R1: no sweep in ``sandboxed`` mode means ``mutation_detected`` is
    ``None`` (not measured), and the write lexicon applies in both modes."""

    ran: list[str] = []

    async def run_command(command: str, timeout_s: float) -> Any:
        ran.append(command)
        if command == SANDBOX_PROBE:
            return _command_result(stdout=SANDBOX_TOKEN + "\n", stdout_bytes=18)
        return _command_result(stdout="ok\n", stdout_bytes=3)

    rows: list[dict[str, Any]] = []
    inspector = VerifierInspector(
        run_command=run_command, window=lambda: 1, record_event=lambda _k, p: rows.append(p)
    )
    for command in ("pkill soffice", "gsettings set org.gnome.x y z", "xdotool key ctrl+s"):
        text, refused = await inspector.inspect({"command": command})
        assert refused and "looks like it writes" in text, command
    assert ran == [SANDBOX_PROBE]
    text, refused = await inspector.inspect({"command": "cat a.txt"})
    assert not refused and ran[-1] == sandboxed("cat a.txt")
    assert rows[0]["guarantees"] and rows[0]["readonly_enforcement"] == "sandboxed"
    assert rows[-1]["mutation_detected"] is None
    assert all(row.get("mutation_detected") is not False for row in rows)


@pytest.mark.asyncio
async def test_a_failed_probe_records_which_check_failed() -> None:
    async def run_command(command: str, timeout_s: float) -> Any:
        return _command_result(stdout="PID-SHARED\n", stdout_bytes=11)

    rows: list[dict[str, Any]] = []
    inspector = VerifierInspector(
        run_command=run_command, window=lambda: 1, record_event=lambda _k, p: rows.append(p)
    )
    assert await inspector.ensure_mode() == "audit"
    assert rows[0]["probe_output"] == "PID-SHARED" and "guarantees" not in rows[0]


def test_obvious_write_lets_plain_reads_through() -> None:
    """M4: reads the first lexicon refused, each costing one of ten commands."""

    for command in (
        "python3 -c \"open('a.xlsx','rb')\"",
        "python3 -c \"print(open('/home/user/a.csv').read())\"",
        "awk '$3 > 5' f.csv",
        'python3 -c "print(1 > 0)"',
        "dpkg -l | grep libreoffice",
        "dpkg -s gimp",
        "pip list",
        "apt list --installed",
        "python3 -c \"s='a'.replace('a','b'); print(s)\"",
    ):
        assert not obvious_write(command), command
    for command in (
        "python3 -c \"open('f','w')\"",
        "python3 -c \"open('f', mode='a')\"",
        "python3 -c \"open('f','r+')\"",
        "python3 -c \"import os; os.remove('/home/user/a.xlsx')\"",
        "python3 -c \"import pathlib; pathlib.Path('a').write_text('x')\"",
        "python3 -c \"import shutil; shutil.rmtree('d')\"",
        "sh -c 'echo x > /home/user/f'",
        'bash -lc "cat a > b"',
        "echo x>f",
        "dpkg -i a.deb",
        "apt-get -y install gimp",
        "pip install x",
        "gsettings reset org.gnome.x y",
    ):
        assert obvious_write(command), command


@pytest.mark.asyncio
async def test_skipped_when_the_wall_holds_the_cap_but_not_the_envelope(tmp_path: Path) -> None:
    """M2: 200 s left holds the 180 s wall cap, not probe + wall + join."""
    from local_operator.evaluation.completion_verifier import (
        FINISH_DELIVERY_MARGIN_S,
        MCP_FINISH_DEADLINE_S,
        verifier_envelope_s,
    )

    # probe (30 s read + 20 s RPC slack) + wall + 10 s cancel join.
    assert verifier_envelope_s(180.0) == 240.0
    # The largest allowed wall still leaves the delivery margin in 300 s.
    assert MAX_VERIFIER_WALL_S == 190.0
    assert verifier_envelope_s(MAX_VERIFIER_WALL_S) + FINISH_DELIVERY_MARGIN_S <= (
        MCP_FINISH_DEADLINE_S
    )
    bridge, _actions, events, seen = _armed(tmp_path, [_verdict("PASS")])
    bridge.wall_remaining = lambda: 200.0
    await bridge.call(DONE)
    assert seen == [] and events[0][1]["reason"] == "wall"


# -- round 3: a cancelled verifier never poisons the adapter session ------------
#
# Live ``t31v-t003``: the verifier was cancelled while an ``inspect_command``
# was mid-RPC. ``run_command`` is the supervisor's ``_mutating_call``, which
# poisons on a ``CancelledError``; the worker's very next command failed
# "adapter session is poisoned" and the episode ended with no score.


def _score(root: Path) -> Any:
    """A scored result whose details artifact exists under ``root`` (the shape
    ``test_command_failure_keeps_session._score`` builds; that module needs
    the OSWorld adapter package, which this one must not)."""

    import hashlib

    from local_operator.evaluation.adapters.api import ScoreResult
    from local_operator.evaluation.evidence.models import (
        EvidenceArtifactRef,
        ScoreArtifact,
    )

    data = b'{"detail":true}'
    digest = hashlib.sha256(data).hexdigest()
    (root / digest).write_bytes(data)
    return ScoreResult(
        score=ScoreArtifact(
            status="scored",
            binary=1,
            details=EvidenceArtifactRef(
                sha256=digest, media_type="application/json", byte_count=len(data)
            ),
        )
    )


class _GuestWorker:
    """The raw-supervisor seam under the REAL ``VerifiedAdapterSession``: the
    probe answers at once, ``slow`` commands answer only when ``release`` is
    set (a guest read still running when the verifier is cut)."""

    def __init__(self, score: Any) -> None:
        from local_operator.evaluation.action_surface import LEGACY_ACTION_SURFACE

        self.action_surface = LEGACY_ACTION_SURFACE
        self.score = score
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        # Set when the slow read has ANSWERED, so a test can order it against
        # the drain rather than infer that from a release timer.
        self.read_done = asyncio.Event()
        self.commands: list[str] = []
        self.terminated = False

    async def terminate(self) -> None:
        self.terminated = True

    async def _call_raw(  # type: ignore[no-untyped-def]
        self, method, params, result_type, *, timeout, **kwargs
    ):
        from local_operator.evaluation.adapters.api import RunCommandResult

        if method == "score":
            return self.score
        assert method == "run_command", method
        self.commands.append(params.command)
        out = SANDBOX_TOKEN if params.command == SANDBOX_PROBE else "ok"
        if "slow-read" in params.command:
            self.started.set()
            await self.release.wait()
            self.read_done.set()
            out = "late"
        return RunCommandResult(
            exit_code=0,
            timed_out=False,
            stdout=out + "\n",
            stderr="",
            stdout_bytes=len(out) + 1,
            stderr_bytes=0,
            stdout_truncated=False,
            stderr_truncated=False,
        )


@pytest.mark.asyncio
async def test_cancelling_the_verifier_mid_inspect_keeps_the_session_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.evaluation.completion_verifier as cv
    from local_operator.evaluation.adapters.api import RunCommandParams, ScoreParams
    from local_operator.evaluation.adapters.supervisor import (
        HostVerifier,
        VerifiedAdapterSession,
    )
    from local_operator.evaluation.session_arm import (
        _arm_completion_verifier,
        _make_run_command,
    )
    from local_operator.harness.types import ChatRequest
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from tests.unit.evaluation.adapters.test_supervisor import observation
    from tests.unit.session.test_launch_subagent import MODEL
    from tests.unit.session.test_verify_completion import TreeStream, tool_turn

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    episode = "episode"
    verifier = HostVerifier("task", episode, tmp_path / "artifacts")
    verifier.accept_initial(observation("task", episode, 0))
    worker = _GuestWorker(_score(tmp_path / "artifacts"))
    session = VerifiedAdapterSession(worker, verifier)  # type: ignore[arg-type]

    def reply(request: ChatRequest) -> list[Any]:
        # The verifier's one move: a guest read that is still running when
        # the wall cap cancels it.
        return tool_turn("inspect_command", {"command": "cat slow-read", "timeout_s": 5})

    parent = Session(
        model=MODEL,
        stream_fn=TreeStream(reply),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    # This test pins an ORDERING, not durations: the wall cuts the verifier
    # WHILE its guest read is mid-RPC, and the arm awaits that read before the
    # worker resumes. A real 1 s wall plus a 1.5 s release timer raced the
    # child's own startup instead: at load ~30 the child sometimes had not
    # sent its first request when the wall fired (`requests: 0`), so no read
    # existed, nothing was drained, and `release` was never set. Both edges
    # are events now; the real timer arithmetic is pinned elsewhere
    # (test_verify_completion.py::test_wall_cap_cancels_a_slow_child).
    #
    # Wall edge: the verifier's settled event wakes once, still unset, the
    # moment the read reaches the guest -- the branch a timer expiry takes
    # (`first_of` returns, settle unset, no overshoot -> "wall").
    class SettledOrWall:
        def __init__(self, real: asyncio.Event) -> None:
            self._real = real
            self._armed = True

        def is_set(self) -> bool:
            return self._real.is_set()

        async def wait(self) -> bool:
            if not self._armed:
                return await self._real.wait()
            self._armed = False
            waiters = [
                asyncio.ensure_future(self._real.wait()),
                asyncio.ensure_future(worker.started.wait()),
            ]
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in waiters:
                    waiter.cancel()
            return self._real.is_set()

    real_settled_event = parent.jobs.settled_event
    monkeypatch.setattr(
        parent.jobs, "settled_event", lambda job_id: SettledOrWall(real_settled_event(job_id))
    )

    # Release edge: the guest answers only once the arm enters its drain.
    # Setting `release` before the real drain runs cannot let the read finish
    # first -- the read task needs a loop turn to resume, and drain snapshots
    # its in-flight set synchronously -- so a drain that awaits it returns 1
    # with the read already answered. No drain, no release: the read is still
    # pending when `bridge.call` returns.
    real_drain = cv.VerifierInspector.drain
    drains: list[tuple[int, bool]] = []

    async def drain_releases_the_guest(self: Any) -> int:
        worker.release.set()
        awaited = await real_drain(self)
        drains.append((awaited, worker.read_done.is_set()))
        return awaited

    monkeypatch.setattr(cv.VerifierInspector, "drain", drain_releases_the_guest)

    bridge = _bridge(tmp_path, max_steps=10, instruction="Put 42 in cell B2.")
    bridge.run_command = _make_run_command(session, episode)
    written: list[tuple[str, dict[str, Any]]] = []
    record = SimpleNamespace(write=lambda kind, payload: written.append((kind, payload)))
    _arm_completion_verifier(
        bridge=bridge,
        handle=SimpleNamespace(session=parent, verifier_job_ids=set()),  # type: ignore[arg-type]
        record=record,  # type: ignore[arg-type]
        # A hang ceiling, not the ordering under test (that is the wall edge
        # above); it also leaves the deadline clip room for the real 20 s RPC
        # slack, so the slack is no longer zeroed.
        config=CompletionVerifierConfig(rounds=1, wall_s=60.0),
        wall_remaining=None,
    )
    try:
        reply_ = await asyncio.wait_for(bridge.call(DONE), timeout=20)
        [verdict] = [p for k, p in written if k == "verifier_verdict"]
        assert (verdict["status"], verdict["ended_by"]) == ("INCONCLUSIVE", "wall"), verdict
        assert reply_["details"]["terminal"] == "completion-challenged"
        # The cancel never reached the adapter RPC: no poison, no teardown...
        assert not session._poisoned and not worker.terminated
        # ...because the abandoned read was awaited before the worker resumed.
        assert worker.release.is_set()
        assert drains == [(1, True)], drains
        # The worker's next command and the score both still work.
        after = await session.run_command(
            RunCommandParams(
                operation_id="worker-next",
                episode_id=episode,
                command="echo after",
                timeout_s=5.0,
                max_output_bytes=1024,
            ),
            timeout=5,
        )
        assert after.stdout == "ok\n" and after.error is None
        scored = await session.score(
            ScoreParams(operation_id="score", episode_id=episode), timeout=5
        )
        assert scored.score.binary == 1
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_guest_reads_are_clipped_to_the_verification_deadline() -> None:
    """A read whose RPC could not settle before the deadline never starts:
    draining it later must not stretch the finish call past the envelope."""

    import time as time_module

    seen: list[tuple[str, float]] = []

    async def run_command(command: str, timeout_s: float) -> Any:
        seen.append((command, timeout_s))
        if command == SANDBOX_PROBE:
            return _command_result(stdout=SANDBOX_TOKEN + "\n", stdout_bytes=18)
        return _command_result(stdout="v\n", stdout_bytes=2)

    inspector = VerifierInspector(run_command=run_command, window=lambda: 1)
    await inspector.ensure_mode()
    from local_operator.evaluation.completion_verifier import RUN_COMMAND_SLACK_S

    inspector.begin(time_module.monotonic() + RUN_COMMAND_SLACK_S + 4.0)
    text, is_error = await inspector.inspect({"command": "cat f", "timeout_s": 30})
    assert not is_error and seen[-1][1] <= 4.0  # clipped from 30 s
    inspector.begin(time_module.monotonic() + RUN_COMMAND_SLACK_S + 0.5)
    text, is_error = await inspector.inspect({"command": "cat g"})
    assert is_error and "give your VERDICT now" in text
    assert [c for c, _ in seen if "cat g" in c] == []  # never reached the guest
    assert await inspector.drain() == 0
    # N2 (round 3): drain closes the verification, so a later guest call is
    # not clipped against the round that just ended.
    text, is_error = await inspector.inspect({"command": "cat h", "timeout_s": 30})
    assert not is_error, text
    assert [t for c, t in seen if "cat h" in c] == [30], seen

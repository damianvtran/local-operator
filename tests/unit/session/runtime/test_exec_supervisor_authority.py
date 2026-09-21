"""Stage E: the supervisor's credential, handed UP the run to the supervisor.

WHAT CHANGED, AND WHY THE DIRECTION IS THE WHOLE DESIGN. A supervised
``lop exec --control`` run could already be DENIED by its supervisor (a deny needs
no authority anywhere) but not APPROVED: answering a parked card is
authority-increasing, and the only credential the seam accepted was the spawn
capability, which an exec run's supervisor does not have — the run holds it, in its
own memory, and nothing else can prove possession of it.

So the run MINTs its own capability and writes it UP a descriptor the supervisor
opened (``--supervisor-fd N``), then closes that descriptor. The supervisor reads
it, keys it by the pid the run's endpoint line already prints, and its
``AttachClient`` presents the proof automatically. No new crypto, no file, no
environment variable: a descriptor number in argv is not a secret, and by the time
any tool subprocess exists the descriptor is gone from the run's table too.

THE TAKEOVER CELL is the security half: inside the run, the run's OWN tool child
must not be able to answer the card its own tool call parked. It holds the record
key — it can dial — and that must still not be enough, because the credential it
would need is the one the run just handed to somebody else.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.approval import (
    OPERATOR_CAP_BYTES,
    deliver_operator_cap_to,
    mint_operator_cap,
    open_supervisor_cap_channel,
    reset_operator_caps_for_tests,
)
from local_operator.session.runtime import registry


@pytest.mark.asyncio
async def test_the_channel_carries_the_capability_upward_and_closes_both_ends() -> None:
    """The primitive, both halves, in one process.

    The run's half is ``deliver_operator_cap_to`` and the supervisor's is
    ``SupervisorCapChannel.read``; driving them against each other is the closest a
    single process can come to the real handoff, and it pins the three properties
    that matter: the bytes arrive whole, the descriptor is CLOSED on the writer's
    side (a descriptor left in the run's table is one a later tool child could find
    by number — the exact leak the downward handoff exists to avoid), and the
    channel closes both of its own ends whatever happens.
    """
    channel = open_supervisor_cap_channel()
    assert channel.argv[0] == "--supervisor-fd"
    assert channel.pass_fds == (channel.pass_fds[0],), "the descriptor pair is not a pair"
    child_fd = int(channel.argv[1])
    assert child_fd in channel.pass_fds

    cap = mint_operator_cap()
    assert deliver_operator_cap_to(child_fd, cap) is True
    # The writer closed it: a later ``os.write`` on the same number fails.
    with pytest.raises(OSError):
        await asyncio.to_thread(__import__("os").write, child_fd, b"x")

    assert channel.read(timeout_s=5.0) == cap
    assert channel.closed is True
    # ONE READ, ONE ANSWER: a second read is ``None`` rather than a second grant.
    assert channel.read(timeout_s=0.1) is None


@pytest.mark.asyncio
async def test_a_supervisor_that_never_reads_gets_nothing_and_leaks_nothing() -> None:
    """Fail-closed on the supervisor's side, and no descriptor left behind.

    ``None`` is a supported answer — the supervisor holds no credential for that
    run, so an authority-increasing frame is refused while every ordinary operation
    continues — and it must not be a hang: a supervisor blocked forever on a run
    that never writes is a worse failure than a refusal.
    """
    channel = open_supervisor_cap_channel()
    assert channel.read(timeout_s=0.25) is None
    assert channel.closed is True

    # A SHORT read is refused rather than padded: a capability this side invented
    # is one the run never held.
    partial = open_supervisor_cap_channel()
    import os

    os.write(int(partial.argv[1]), b"short")
    os.close(int(partial.argv[1]))
    assert partial.read(timeout_s=1.0) is None
    assert partial.closed is True


def test_a_detached_run_cannot_be_given_a_supervisor_descriptor() -> None:
    """``--background --supervisor-fd`` is REFUSED, and the reason is structural.

    The descriptor names one end of a socketpair the SUPERVISOR holds; ``--background``
    detaches a worker and the launcher that owns that end exits immediately, so the
    run's upward write would find a closed peer. The observable outcome would be a
    run that looks supervised and whose cards nobody can approve — the failure this
    whole stage exists to remove — so the combination is a configuration error with
    a sentence that names the real lever for an unattended run.
    """
    from local_operator.exec_mode import (
        ExecArgs,
        build_worker_argv,
        reject_detached_supervisor_fd,
    )

    background = ExecArgs(background=True, control=True, supervisor_fd=7)
    refusal = reject_detached_supervisor_fd(background)
    assert refusal is not None
    assert "--background" in refusal and "--yolo" in refusal

    # And it is never serialized into the worker's argv: a descriptor number that
    # crossed that boundary names something the worker does not hold.
    with pytest.raises(AssertionError):
        build_worker_argv("do the thing", background)

    unsupervised = ExecArgs(control=False, supervisor_fd=7)
    assert reject_detached_supervisor_fd(unsupervised) is not None
    assert reject_detached_supervisor_fd(ExecArgs(control=True, supervisor_fd=7)) is None
    assert reject_detached_supervisor_fd(ExecArgs(control=True, supervisor_fd=None)) is None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_resuming_a_live_session_with_a_supervisor_descriptor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N7, PROBED RATHER THAN REASONED ABOUT: the takeover route, on the real thing.

    The question the design asks explicitly is what happens to
    ``lop exec --resume <own live session> --control --supervisor-fd N``. It is a
    plausible-looking takeover: a supervisor attaches a descriptor to a run that
    resumes a session whose runtime is ALREADY serving, and if the resume succeeded
    it would have a credential for somebody else's live gate.

    MEASURED BEHAVIOUR (2026-09-20, this machine, worktree venv): the run exits 1
    BEFORE any control surface exists, with the session LEASE's refusal —

        session <id> is already open in another process (pid <pid>) — watch and
        steer it there, or from the phone session list

    — and the supervisor receives NO capability (its read sees EOF, not a short
    read: the run never wrote). So the takeover does not happen, and it does not
    happen for a reason that predates this stage: the lease is what stops a second
    runtime from owning a live session's gate at all. What this stage adds is the
    OTHER half — the supervisor is not left holding a descriptor with nothing behind
    it.

    Driven as a real subprocess with a real inherited descriptor, because the
    property under test is about a descriptor surviving ``exec`` — which nothing
    in-process can observe.
    """
    import subprocess
    import sys

    from local_operator.harness.approval import open_supervisor_cap_channel
    from local_operator.session.runtime import launch as launch_module
    from tests.unit.session.runtime import test_runtime_detachment as detachment

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    session_id = "supervisortakeover1"
    session_dir = config_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: ask\n",
        encoding="utf-8",
    )
    detachment._isolate(monkeypatch, config_dir)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    child = launch_module._spawn_runtime(session_id, str(config_dir), defer_materialise=False)
    try:
        deadline = time.monotonic() + 40
        live = None
        while time.monotonic() < deadline:
            for found, _state in registry.scan(config_dir):
                if getattr(found, "session_id", "") == session_id:
                    live = found
                    break
            if live is not None:
                break
            await asyncio.sleep(0.05)
        assert live is not None, "the rig never got a live runtime to resume"

        channel = open_supervisor_cap_channel()
        argv = [
            sys.executable,
            "-m",
            "local_operator.cli",
            "exec",
            "--control",
            f"--resume={session_id}",
            *channel.argv,
            "resume probe",
        ]
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True,
                text=True,
                timeout=180,
                pass_fds=channel.pass_fds,
                close_fds=True,
            )
        finally:
            pass

        assert proc.returncode != 0, proc.stdout
        assert "already open in another process" in proc.stderr, proc.stderr[-2000:]
        # AND NO CREDENTIAL CROSSED. ``None`` rather than a short read: the run never
        # wrote, so the supervisor's channel sees the closed peer.
        assert channel.read(timeout_s=5.0) is None
    finally:
        child.terminate()
        with contextlib.suppress(Exception):
            child.wait(timeout=10)
        registry.scan(config_dir)


def test_the_flag_is_registered_on_both_parsers() -> None:
    """The CLI half, read from the parsers themselves rather than re-typed.

    A flag accepted on one side of the ``exec``/``exec_worker`` boundary and not the
    other is the failure ``build_worker_argv``'s own comments record twice (for
    ``--control`` and ``--resume``): accepted at the front end and silently lost.
    """
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    args = parser.parse_args(["exec", "--control", "--supervisor-fd", "3", "hello"])
    assert args.supervisor_fd == 3
    assert args.control is True
    default = parser.parse_args(["exec", "hello"])
    assert default.supervisor_fd is None


# ---------------------------------------------------------------------------
# The end-to-end cell: the supervisor really approves through the handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_supervisor_approves_a_card_through_the_upward_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5, over the real surface: mint -> publish -> hand up -> prove -> approve.

    Every hop is production code: ``start_exec_control`` mints the capability,
    ``RuntimeServer`` holds it and serves the record, ``deliver_operator_cap_to``
    writes it up the descriptor, and the supervisor's connection presents the proof
    the way ``AttachClient`` does. The card is parked by the REAL gate, and the
    resolved value is asserted — a flag assertion would not distinguish an approval
    from a refusal that happened to look similar.
    """
    from local_operator.session.runtime.exec_control import start_exec_control
    from tests.unit.session.runtime.test_approval_authority_seam import (
        _AttachableSession,
        _dial,
        _send,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    ConfigManager(tmp_path).set_config_value("tool_approval_mode", "ask")

    session = _AttachableSession()
    channel = open_supervisor_cap_channel()
    control = await start_exec_control(
        session,
        cwd=str(tmp_path),
        supervised=True,
        supervisor_fd=int(channel.argv[1]),
    )
    try:
        # THE ENDPOINT LINE IS WHAT NAMES THE PID the supervisor keys the credential
        # by, so the rig reads it from the line rather than from the control object.
        line = control.endpoint_line
        assert "control:" in line and f"pid={control.pid}" in line
        cap = channel.read(timeout_s=5.0)
        assert cap is not None, "the run never handed its capability up"
        assert len(cap) == OPERATOR_CAP_BYTES

        # The supervisor's proof is accepted; the card really resolves.
        parked = asyncio.ensure_future(control.handle._approval_gate("bash", "rm -rf build/"))
        await asyncio.sleep(0)
        pending = control.handle._fold.projection.pending
        assert pending is not None, "the fixture did not park a card"
        supervisor = await _dial(control.runtime.record, client="attach", cap=cap)
        try:
            reply = await _send(
                supervisor,
                None,
                {
                    "op": "approval_answer",
                    "request_id": pending.request_id,
                    "approved": True,
                    "operator_cap": supervisor.proof(cap),
                },
            )
            assert reply["op"] == "ack", reply
            assert await parked is True, "the supervisor's approval did not resolve the card"
        finally:
            supervisor.close()
    finally:
        await control.aclose()
        registry.scan(tmp_path)


@pytest.mark.asyncio
async def test_the_runs_own_tool_child_cannot_answer_its_own_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N6: the takeover path, on the real surface.

    A tool child inside a supervised run holds everything a same-uid process can
    hold: the record (0600, readable by it), the control key, the loopback port —
    and it can dial and READ. What it does not hold is the run's capability, which
    the run handed UP to the supervisor and then closed the descriptor on. So it can
    deny the card (a deny needs no authority and settles the call safely) and it
    cannot approve it, which is the one thing the issue is about.
    """
    from local_operator.session.runtime.exec_control import start_exec_control
    from tests.unit.session.runtime.test_approval_authority_seam import (
        _AttachableSession,
        _dial,
        _send,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    ConfigManager(tmp_path).set_config_value("tool_approval_mode", "ask")

    session = _AttachableSession()
    channel = open_supervisor_cap_channel()
    control = await start_exec_control(
        session, cwd=str(tmp_path), supervised=True, supervisor_fd=int(channel.argv[1])
    )
    try:
        assert channel.read(timeout_s=5.0) is not None
        # The child has the RECORD and nothing else — no nonce, no proof, no
        # signature. ``_dial`` without ``cap`` is exactly that state.
        child = await _dial(control.runtime.record, client="attach")
        try:
            parked = asyncio.ensure_future(control.handle._approval_gate("bash", "rm -rf build/"))
            await asyncio.sleep(0)
            pending = control.handle._fold.projection.pending
            assert pending is not None
            refused = await _send(
                child,
                None,
                {"op": "approval_answer", "request_id": pending.request_id, "approved": True},
            )
            assert refused["op"] == "error", refused
            assert control.handle._fold.projection.pending is not None, "the card resolved anyway"

            # The same child may DENY: that settles the call in the safe direction
            # and needs no authority, which is what keeps a follower from being
            # walled off from the one answer it can safely give.
            allowed = await _send(
                child,
                None,
                {"op": "approval_answer", "request_id": pending.request_id, "approved": False},
            )
            assert allowed["op"] == "ack", allowed
            assert await parked is False
        finally:
            child.close()
    finally:
        await control.aclose()
        registry.scan(tmp_path)


@pytest.mark.asyncio
async def test_an_unsupervised_exec_run_holds_no_capability_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No descriptor, no capability — the fail-closed state, asserted rather than
    assumed.

    ``supervisor_fd=None`` is every ``lop exec --control`` run started before this
    stage and every run whose supervisor only wants to deny. The run must then hold
    NOTHING (minting a capability nobody can prove possession of would be a
    credential nothing needs), and its authority-increasing frames must be refused
    exactly as they were before.
    """
    from local_operator.session.runtime.exec_control import start_exec_control
    from tests.unit.session.runtime.test_approval_authority_seam import (
        _AttachableSession,
        _dial,
        _send,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    ConfigManager(tmp_path).set_config_value("tool_approval_mode", "ask")

    session = _AttachableSession()
    control = await start_exec_control(session, cwd=str(tmp_path), supervised=True)
    try:
        assert control.runtime._operator_cap is None
        conn = await _dial(control.runtime.record, client="attach")
        try:
            reply = await _send(
                conn,
                None,
                {
                    "op": "slash_result",
                    "command": "approvals",
                    "args": "auto",
                    "images": [],
                },
            )
            assert reply["op"] == "error", reply
            assert control.handle._auto_approve is False
        finally:
            conn.close()
    finally:
        await control.aclose()
        registry.scan(tmp_path)

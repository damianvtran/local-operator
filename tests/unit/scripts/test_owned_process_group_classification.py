"""Classification at the REAL syscall boundary, and what it costs a passing gate.

Every other authority test injects a ``signal_group`` that returns a verdict, so
the mapping from a real ``os.killpg`` failure to a verdict was exercised nowhere:
A1 (ESRCH read as refusal) was invisible to the suite. These tests keep the
classification inside the real module and move only the kernel facts.

The deletion controls assert on EXIT CODE and SIGNAL COUNT, never on a message:
a relabelled path would still pass a string match.
"""

from __future__ import annotations

import ast
import builtins
import io
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


class World:
    """Synthetic kernel facts. Nothing here is a host capability."""

    def __init__(self, pgid=10, parent=1, birth="leader", scheme="fixture"):
        self.pgid, self.parent, self.birth, self.scheme = pgid, parent, birth, scheme
        self.rows = {parent: (0, parent, parent, "creator"), pgid: (parent, pgid, pgid, birth)}
        self.now = 0.0
        self.calls = []
        self.sample_calls = 0

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        # Two samples during capture, then the child is a corpse: the loop sees it
        # through `leader_terminal` and never reaps it to find out.
        self.sample_calls += 1
        corpse = self.sample_calls > 2
        return {
            pid: owned.procstate.ProcessSample(corpse and pid == self.pgid, self.rows[pid][3])
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def capture(self, *, send, pgid=None):
        return owned.OwnedGroup.capture(
            pgid or self.pgid,
            # Explicit and finite: `capture` has no default deadline on purpose,
            # so every caller states the budget it is willing to spend.
            deadline=self.clock() + 5.0,
            parent_pid=self.parent,
            census=self.census,
            samples=self.samples,
            identity=self.identity,
            scheme=lambda: self.scheme,
            clock=self.clock,
            signal_group=send,
        )

    def forget(self, pgid=None):
        self.rows.pop(pgid or self.pgid, None)


#: Real kernel primitives: skip where the platform has no POSIX process groups.
posix_only = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process-group authority")


def _free_process_group() -> int:
    """A process group that provably does not exist. Signal 0 delivers nothing."""
    for candidate in range(900_000, 900_400):
        try:
            os.killpg(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    pytest.skip("no free process-group id available for the ESRCH control")


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    """The real wrapper module, isolated scratch, real capability object.

    Resolved by NAME rather than as a parameter: the shared fixture is imported
    into this module (that import is how pytest sees it), and a parameter of the
    same name shadows it — F811, and a trap for the next reader.
    """
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


def _fake_process(poll_status):
    return NS(pid=10, poll=lambda: poll_status, wait=lambda **_: poll_status)


def _install(wrapper, monkeypatch, world, send, process):
    cap = world.capture(send=send)
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: process, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )
    return cap


@posix_only
def test_real_killpg_esrch_is_observed_absent_not_a_refusal():
    """The classification under test is the REAL one; only the facts are fake."""
    pgid = _free_process_group()
    world = World(pgid=pgid, parent=os.getpid(), birth="leader")
    # Deliberately NOT injecting signal_group: this is os.killpg itself, and the
    # group it is asked about provably does not exist, so the kernel answers ESRCH.
    cap = world.capture(send=os.killpg)
    assert cap.last.status == "VERIFIED"
    verdict = cap.signal(signal.SIGKILL, deadline=time_budget())
    assert verdict.status == "OBSERVED_EMPTY", verdict.reason
    assert "ESRCH" in verdict.reason


def time_budget():
    import time

    return time.monotonic() + 1.0


@posix_only
def test_permission_error_from_the_checked_boundary_stays_a_refusal():
    world = World()

    def denied(_pgid, _signum):
        raise PermissionError(1, "Operation not permitted")

    cap = world.capture(send=denied)
    assert cap.signal(signal.SIGKILL, deadline=time_budget()).status == "REFUSED"


def test_a_passing_gate_keeps_rc_zero_when_the_group_is_already_gone(wrapper, monkeypatch):
    """A1's reachable window: the leader passed, the residual group exited first.

    RED on the reviewed head: the ESRCH was mapped to REFUSED, so a green gate
    returned 125 ("did not run"). The assertion is the EXIT CODE.
    """
    world = World()

    def gone(pgid, _signum):
        world.forget(pgid)
        raise ProcessLookupError(3, "No such process")

    _install(wrapper, monkeypatch, world, gone, _fake_process(0))
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output)
    assert rc == 0, output.getvalue()


def _foreign_birth_run(wrapper, monkeypatch, module):
    """One run whose leader is reaped into a DIFFERENT birth before cleanup.

    Returns ``(exit code, signal count)`` — the two values the deletion control
    discriminates on. ``module`` is the real module or a mutated copy of it.
    """
    world = World()

    def sender(pgid, signum):
        """A delivered signal on a live group: it records, and the group goes away."""
        world.calls.append((pgid, signum))
        world.rows.pop(pgid, None)

    cap = module.OwnedGroup.capture(
        world.pgid,
        deadline=world.clock() + 5.0,
        parent_pid=world.parent,
        census=world.census,
        samples=world.samples,
        identity=world.identity,
        scheme=lambda: world.scheme,
        clock=world.clock,
        signal_group=sender,
    )

    def reaped():
        world.rows[world.pgid] = (world.parent, world.pgid, world.pgid, "FOREIGN")
        return 0

    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=module.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(
            Popen=lambda *a, **k: NS(pid=10, poll=reaped, wait=lambda **_: 0),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output)
    return rc, len(world.calls), output.getvalue()


def test_deleting_the_identity_check_fails_on_exit_code_and_signal_count(wrapper, monkeypatch):
    """The mutation control: remove the identity check and BOTH assertions break.

    With the check: the final refresh sees a foreign birth and refuses, so no
    syscall is attempted and the passed gate reports 126 (cleanup unproven).
    Without it: the same run would signal a stranger (count 1) and return 0.
    """
    rc, calls, output = _foreign_birth_run(wrapper, monkeypatch, owned)
    assert calls == 0, output
    assert rc == 126, (rc, output)


def test_the_final_recensus_is_what_reports_an_unproven_cleanup(wrapper, monkeypatch):
    """Q3(a): a group that outlives every kill must not be reported as clean.

    The fake syscall accepts the signal but the member stays, so the ONLY place
    that can notice is the final re-census before the wrapper returns. Delete it
    and this run reports 0 — which is the mutant this test exists to catch.
    """
    world = World()
    _install(wrapper, monkeypatch, world, lambda *a: world.calls.append(a), _fake_process(0))
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output)
    assert world.calls, "the cleanup attempted at least one checked signal"
    assert rc == 126, (rc, output.getvalue())


def test_the_signal_path_never_enrols_a_new_member(wrapper, monkeypatch):
    """Q3(b): a member that appeared after capture is refused, never enrolled.

    With the signal-time gate: REFUSED and no syscall. Change ``signal`` to enrol
    and the same member is adopted and signalled (verdict VERIFIED, count 1).
    """
    world = World()
    cap = world.capture(send=world.calls.append)
    assert cap.last.status == "VERIFIED"
    world.rows[11] = (world.pgid, world.pgid, world.pgid, "late")
    verdict = cap.signal(signal.SIGTERM, deadline=time_budget())
    assert verdict.status == "REFUSED", verdict.reason
    assert world.calls == []


def test_a_refused_cleanup_after_a_nonzero_gate_keeps_the_child_status(wrapper, monkeypatch):
    """Only a PASSING gate is re-labelled; a failing one keeps its own code."""
    world = World()
    cap = world.capture(send=world.calls.append)
    world.rows[world.pgid] = (world.parent, world.pgid, world.pgid, "FOREIGN")
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: _fake_process(1), TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO()) == 1
    assert world.calls == []


def _load_without_killpg():
    """The real module text, executed with an ``os`` that has no ``killpg``.

    This is how Windows sees the module: the attribute is simply absent. Nothing
    here touches the real ``os`` module or the host.
    """
    source = Path(owned.__file__).read_text()
    real_import = builtins.__import__

    class NoKillpg:
        def __getattr__(self, name):
            if name == "killpg":
                raise AttributeError(name)
            return getattr(os, name)

    def fake_import(name, *args, **kwargs):
        return NoKillpg() if name == "os" else real_import(name, *args, **kwargs)

    namespace = {
        "__name__": "owned_process_group_no_killpg",
        "__file__": owned.__file__,
        "__builtins__": dict(builtins.__dict__, __import__=fake_import),
    }
    exec(compile(source, "owned_process_group_no_killpg", "exec"), namespace)
    return namespace


def test_importing_without_the_signal_primitive_succeeds_and_refuses():
    """W1: the primitive must not be evaluated at import time.

    RED on the reviewed head: `signal_group=os.killpg` is a keyword default, so
    this exec raised AttributeError while defining the class (CI
    xplat-probe-windows, job 124804637518). Now it imports, and the unsupported
    platform is a REFUSED verdict instead of an exception.
    """
    namespace = _load_without_killpg()
    world = World()
    cap = namespace["OwnedGroup"].capture(
        world.pgid,
        deadline=world.clock() + 5.0,
        parent_pid=world.parent,
        census=world.census,
        samples=world.samples,
        identity=world.identity,
        scheme=lambda: "fixture",
        clock=world.clock,
    )
    assert cap.last.status == "VERIFIED"
    verdict = cap.signal(signal.SIGTERM, deadline=time_budget())
    assert verdict.status == "REFUSED", verdict.reason
    assert "unsupported" in verdict.reason


#: ``os`` attributes Windows does not have. A default expression naming one is
#: evaluated at import, which is how W1 became an ImportError instead of a refusal.
POSIX_ONLY_OS_ATTRIBUTES = frozenset(
    {
        "killpg",
        "getpgid",
        "getsid",
        "fork",
        "forkpty",
        "setsid",
        "waitpid",
        "setpgrp",
        "wait3",
        "wait4",
    }
)


def test_no_posix_only_os_attribute_is_evaluated_in_a_default_expression():
    """The structural guard for W1, for every default in the module, present and future.

    ``os.listdir``/``os.stat`` defaults stay legal: they exist on every platform.
    """
    tree = ast.parse(Path(owned.__file__).read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d]
            for default in defaults:
                for sub in ast.walk(default):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "os"
                        and sub.attr in POSIX_ONLY_OS_ATTRIBUTES
                    ):
                        offenders.append((node.name, sub.attr))
    assert offenders == [], offenders

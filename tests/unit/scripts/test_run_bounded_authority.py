"""Real wrapper / real capability; only clocks, processes and kernel facts fake.

No synthetic PID reaches a native reader or signal. Base RED controls extract
only the former numeric-PG boundary, never run a second supervisor.
"""
from __future__ import annotations

import ast
import io
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_owned_process_group import Fake
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


@pytest.mark.parametrize("mode,expected", [
    ("reuse-finally", 125), ("finally-reuse", 125), ("grace-reuse", 124),
    ("forward-unknown", 143), ("memory-unknown", 137), ("monitor-error", None),
    ("spawn-signal", 143),
])
def test_each_wrapper_path_reauthorizes_without_unbounded_wait(memory_wrapper, monkeypatch, mode, expected):
    wrapper = memory_wrapper
    fake = Fake()
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: "fixture")
    real_capture = owned.OwnedGroup.capture
    handlers, phases = {}, []

    class Clock:
        def monotonic(self):
            fake.now += 0.001
            return fake.now
        def sleep(self, seconds):
            fake.now += seconds
    monkeypatch.setattr(wrapper, "time", Clock())

    def install(sig, handler):
        old = handlers.get(sig)
        handlers[sig] = handler
        return old
    monkeypatch.setattr(wrapper, "signal", NS(**{**vars(signal), "signal": install}))

    def foreign():
        fake.rows[10] = (1, 10, 10, "FOREIGN", False)

    class Process:
        pid = 10
        status = None
        def poll(self):
            if mode in {"reuse-finally", "finally-reuse"}:
                if mode == "reuse-finally": foreign()
                self.status = 0
            return self.status
        def wait(self, *, timeout):
            assert 0 <= timeout <= 0.3, "unconditional/unbounded wait"
            if self.status is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.status
    process = Process()

    def popen(*args, **kwargs):
        if mode in {"forward-unknown", "spawn-signal"}:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process
    monkeypatch.setattr(wrapper, "subprocess", NS(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired))

    def send(pg, sig):
        assert pg == 10
        fake.calls.append((pg, sig))
        if mode in {"grace-reuse", "finally-reuse"}: foreign()
        else:
            process.status = -int(sig)
            del fake.rows[10]

    def capture(*args, **kwargs):
        cap = real_capture(10, parent_pid=1, census=fake.census, samples=fake.samples,
                           identity=fake.identity, scheme=lambda: "fixture", clock=fake.clock,
                           signal_group=send, deadline=5)
        if mode == "forward-unknown": foreign()
        return cap
    monkeypatch.setattr(wrapper, "owned_process_group", NS(
        OwnedGroup=NS(capture=capture), Verdict=owned.Verdict))
    monkeypatch.setattr(wrapper, "_refusal_receipt", lambda owner, phase, verdict, stream: phases.append((phase, verdict)))
    budget = NS(ceiling_mb=1, reason="fake")
    monkeypatch.setattr(wrapper, "_memory_budget", lambda _: (budget, "fixture"))
    def sample():
        if mode in {"memory-unknown", "monitor-error"}: foreign()
        if mode == "monitor-error": raise RuntimeError("original monitor error")
        return NS(bytes_used=1)
    guard = NS(tick_s=0.1, sample_sync=sample, should_kill=lambda _: mode == "memory-unknown",
               over_budget_message=lambda _: "MEMORY LIMIT EXCEEDED", soft_notice=lambda _: None)
    monkeypatch.setattr(wrapper.memory_guard, "Guard", lambda *_: guard)
    if mode == "monitor-error":
        class Closed:
            def write(self, _): raise ValueError("closed stderr")
            def flush(self): raise ValueError("closed stderr")
        # Initial pre-spawn logging is not the subject of the closed-stream case.
        output = io.StringIO()
        def fail_sample():
            foreign()
            output.close()
            raise RuntimeError("original monitor error")
        guard.sample_sync = fail_sample
        with pytest.raises(RuntimeError, match="original monitor error"):
            wrapper.run(["fake"], timeout=0.05, grace=0.05, stream=output)
    else:
        assert wrapper.run(["fake"], timeout=0.01 if mode == "grace-reuse" else 0.05,
                           grace=0.05, stream=io.StringIO()) == expected
    assert all(handler is None for handler in handlers.values())
    if mode in {"reuse-finally", "forward-unknown", "memory-unknown", "monitor-error"}:
        assert fake.calls == []
        assert phases
    elif mode in {"grace-reuse", "finally-reuse"}:
        assert len(fake.calls) == 1  # reuse MUST block escalation / next final pass
        assert phases
    else:
        assert fake.calls and fake.calls[0] == (10, signal.SIGTERM)


def test_old_numeric_boundary_red_with_fake_foreign_group():
    text = Path(__file__).with_name("run_bounded_unsafe_boundary.txt").read_text()
    calls = []
    namespace = {"os": NS(killpg=lambda pg, sig: calls.append((pg, sig)))}
    exec(compile(text, "base_numeric_boundary", "exec"), namespace)
    namespace["_signal_group"](10, signal.SIGKILL)
    with pytest.raises(AssertionError):
        assert calls == [], "old boundary signalled a reused foreign numeric PG"
    print("BASE_RED_FOREIGN_NUMERIC_SIGNAL", calls)


def test_handlers_only_latch_and_all_signal_sites_take_capabilities():
    tree = ast.parse((Path(__file__).resolve().parents[3] / "scripts/run_bounded.py").read_text())
    forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward")
    assert not any(isinstance(n, ast.Call) for n in ast.walk(forward))
    assert not any(isinstance(n, ast.Attribute) and n.attr == "killpg" for n in ast.walk(tree))


@pytest.mark.parametrize("phase", ["acquire", "midrun", "zombie"])
def test_no_timeout_refusal_exits_without_unknown_signal_or_unbounded_wait(memory_wrapper, monkeypatch, phase):
    wrapper = memory_wrapper
    handlers, receipts, polls = {}, [], []
    class Clock:
        now = 0
        def monotonic(self):
            self.now += 0.001
            return self.now
        def sleep(self, seconds): self.now += seconds
    monkeypatch.setattr(wrapper, "time", Clock())
    def install(sig, handler):
        old = handlers.get(sig)
        handlers[sig] = handler
        return old
    monkeypatch.setattr(wrapper, "signal", NS(**{**vars(signal), "signal": install}))
    def wait(*, timeout):
        assert timeout == 0
        raise subprocess.TimeoutExpired("fake live or zombie child", timeout)
    def poll():
        polls.append(True)
        assert len(polls) <= 1, "unbounded run kept polling after authority loss"
        return None
    process = NS(pid=10, poll=poll, wait=wait)
    monkeypatch.setattr(wrapper, "subprocess", NS(Popen=lambda *a, **k: process,
                                                   TimeoutExpired=subprocess.TimeoutExpired))
    state = [0]
    reason = "terminal-only is not absence" if phase == "zombie" else "unknown identity"
    bad = owned.Verdict("REFUSED", reason, (10,))
    def refresh(**kwargs):
        state[0] += 1
        return owned.Verdict("VERIFIED", "first proof") if phase == "midrun" and state[0] == 1 else bad
    cap = NS(pgid=10, receipts={}, last=bad if phase != "midrun" else owned.Verdict("VERIFIED", "acquired"),
             refresh=refresh, signal=lambda *a, **k: pytest.fail("signal after proof failure"))
    monkeypatch.setattr(wrapper, "owned_process_group", NS(OwnedGroup=NS(capture=lambda *a, **k: cap),
                                                           Verdict=owned.Verdict))
    monkeypatch.setattr(wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fake"), "fixture"))
    monkeypatch.setattr(wrapper, "_refusal_receipt", lambda owner, stage, verdict, stream: receipts.append(stage))
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO()) == 125
    assert receipts and all(handler is None for handler in handlers.values())

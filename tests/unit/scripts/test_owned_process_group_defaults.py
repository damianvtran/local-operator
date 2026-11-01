"""Real default capture/sample pipeline; inject only lower kernel facts/syscalls."""
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as o


@pytest.mark.parametrize("native_available", [True, False])
def test_default_capture_subsecond_native_or_explicit_fallback_refusal(monkeypatch, native_available):
    scheme = o.procstate.birth_scheme()
    calls, signals = [], []
    monkeypatch.setattr(o.procstate, "birth_scheme", lambda: scheme)
    monkeypatch.setattr(o, "sys", NS(platform="darwin"))
    monkeypatch.setattr(o, "os", NS(getpid=lambda: 1, getpgid=lambda _: 10, getsid=lambda _: 10))
    monkeypatch.setattr(o.resources, "direct_ppid_pgid", lambda _: (1, 10))
    def listing(kind, group, buffer, capacity):
        if buffer is None: return 4
        buffer[0] = 10
        return 4
    monkeypatch.setattr(o.resources, "_darwin_libproc", lambda: NS(proc_listpids=listing))
    def native(pids):
        calls.append(tuple(pids))
        if not native_available: return None
        return {p: o.procstate.ProcessSample(False, "parent" if p == 1 else "leader") for p in pids}
    monkeypatch.setattr(o.procstate, "_sysctl_samples", native)
    monkeypatch.setattr(o.procstate, "process_samples", lambda _: pytest.fail("unbudgeted ps fallback"))
    # Deliberately omit samples/census/identity providers: these are REAL defaults.
    cap = o.OwnedGroup.capture(10, deadline=0.25, clock=lambda: 0,
                               signal_group=lambda *args: signals.append(args))
    if native_available:
        assert cap.last.status == "VERIFIED"
        assert cap.signal(9, deadline=0.25).status == "SIGNALED"
        assert signals == [(10, 9)]
    else:
        assert cap.last.status == "REFUSED"
        assert cap.signal(9, deadline=0.25).status == "REFUSED"
        assert not signals
    assert calls

"""Real default capture/sample pipeline; inject only lower kernel facts/syscalls."""

from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as o


@pytest.mark.parametrize("native_available", [True, False])
@pytest.mark.parametrize("clock_start", [0.0, 1_000_000.0])
@pytest.mark.parametrize("census_clock_offset", [0.0, 1000.0])
def test_default_capture_subsecond_native_or_explicit_fallback_refusal(
    monkeypatch, native_available, clock_start, census_clock_offset
):
    scheme = o.procstate.birth_scheme()
    calls, signals, census_clock_reads = [], [], []

    def clock():
        """A frozen lower-boundary clock: no real time reaches the default census."""
        return clock_start

    deadline = clock() + 0.25

    def census_clock():
        census_clock_reads.append(True)
        return clock() + census_clock_offset

    # Inject time at the LOWER native-census boundary, not by replacing the
    # actual default census/capture/sample pipeline. Its default was bound at
    # definition time; patching the module's time name would not reach it.
    monkeypatch.setattr(
        o.strict_darwin,
        "__kwdefaults__",
        {
            **(o.strict_darwin.__kwdefaults__ or {}),
            "clock": census_clock,
        },
    )
    monkeypatch.setattr(o.procstate, "birth_scheme", lambda: scheme)
    monkeypatch.setattr(o, "sys", NS(platform="darwin"))
    monkeypatch.setattr(
        o,
        "os",
        # `getpgrp` is real POSIX that the module legitimately calls; a double
        # that omits it makes an AttributeError look like a production defect.
        NS(
            getpid=lambda: 1,
            getpgid=lambda _: 10,
            getsid=lambda _: 10,
            getpgrp=lambda: 10,
        ),
    )
    monkeypatch.setattr(o.resources, "direct_ppid_pgid", lambda _: (1, 10))

    def listing(kind, group, buffer, capacity):
        if buffer is None:
            return 4
        buffer[0] = 10
        return 4

    monkeypatch.setattr(o.resources, "_darwin_libproc", lambda: NS(proc_listpids=listing))

    def native(pids):
        calls.append(tuple(pids))
        if not native_available:
            return None
        return {p: o.procstate.ProcessSample(False, "parent" if p == 1 else "leader") for p in pids}

    monkeypatch.setattr(o.procstate, "_sysctl_samples", native)
    monkeypatch.setattr(
        o.procstate, "process_samples", lambda _: pytest.fail("unbudgeted ps fallback")
    )
    # Deliberately omit samples/census/identity providers: these are REAL defaults.
    cap = o.OwnedGroup.capture(
        10, deadline=deadline, clock=clock, signal_group=lambda *args: signals.append(args)
    )
    assert census_clock_reads, "real default census must consume the lower fake clock"
    if census_clock_offset:
        # Deterministic domain-mismatch countercontrol: no real clock/sleep or
        # process probe is used, and native sampling MUST NOT be reached.
        assert cap.last.status == "REFUSED"
        assert not calls and not signals
        return
    if native_available:
        assert cap.last.status == "VERIFIED"
        assert cap.signal(9, deadline=deadline).status == "SIGNALED"
        assert signals == [(10, 9)]
    else:
        assert cap.last.status == "REFUSED"
        assert cap.signal(9, deadline=deadline).status == "REFUSED"
        assert not signals
    assert calls

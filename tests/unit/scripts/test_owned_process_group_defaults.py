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
            # `getpgrp` describes THIS process, so it must not coincide with
            # the fixture's target group: returning 10 made the capability
            # correctly conclude it was signalling its own group, turning a
            # passing exclusion invariant into a refused row.
            getpgrp=lambda: 999_999,
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
    # Darwin row equality is observation, never a coverage certificate. The old
    # default admission premise is intentionally unavailable, regardless of the
    # lower native sampler. Refuse before sampling and never invoke the sender.
    assert cap.last.status == "REFUSED"
    assert cap.signal(9, deadline=deadline).status == "REFUSED"
    assert not calls and not signals


def test_default_linux_capture_reaches_verified_and_signals(monkeypatch):
    """R7 positive control: the REAL default census/sample/identity pipeline,
    on an authoritative (Linux, /proc-shaped) backend, reaches VERIFIED and then
    SIGNALED. Only the /proc text and process-identity syscalls are faked.

    Without this, a change that made Linux refuse as Darwin does would pass the
    refusal matrix above unnoticed.
    """
    rows = {
        1: "1 (init) S 0 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 100 0 0",
        10: "10 (leader) S 1 10 10 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 500 0 0",
    }
    signals = []

    def read(path):
        pid = int(path.split("/")[2])
        if pid not in rows:
            raise FileNotFoundError(path)
        return rows[pid]

    monkeypatch.setattr(o, "sys", NS(platform="linux"))
    monkeypatch.setattr(o.resources, "sys", NS(platform="linux"))
    monkeypatch.setattr(o.procstate, "sys", NS(platform="linux"))
    monkeypatch.setattr(
        o.strict_linux,
        "__kwdefaults__",
        {
            **(o.strict_linux.__kwdefaults__ or {}),
            "listdir": lambda _: [str(p) for p in rows],
            "read": read,
            "stat": lambda _: None,
        },
    )

    class FakePath:
        def __init__(self, path):
            self.path = path

        def read_text(self, *args, **kwargs):
            return read(self.path)

    monkeypatch.setattr(o.procstate, "Path", FakePath)
    monkeypatch.setattr(
        o.procstate, "birth_scheme", lambda: o.procstate.BIRTH_SCHEME_PROC_STARTTIME
    )
    # capture's `scheme` default was bound to the host's function at definition
    # time; point that default (not a passed argument) at the Linux scheme.
    capture = o.OwnedGroup.capture.__func__
    monkeypatch.setattr(
        capture,
        "__kwdefaults__",
        {
            **(capture.__kwdefaults__ or {}),
            "scheme": lambda: o.procstate.BIRTH_SCHEME_PROC_STARTTIME,
        },
    )
    monkeypatch.setattr(o.resources, "direct_ppid_pgid", lambda pid: (1, 10))
    monkeypatch.setattr(
        o,
        "os",
        NS(
            getpid=lambda: 1,
            getpgid=lambda _: 10,
            getsid=lambda _: 10,
            getpgrp=lambda: 999_999,
        ),
    )
    cap = o.OwnedGroup.capture(
        10,
        deadline=10**9,
        clock=lambda: 0.0,
        signal_group=lambda pgid, signum: signals.append((pgid, signum)),
    )
    assert cap.last.status == "VERIFIED", cap.last.reason
    verdict = cap.signal(15, deadline=10**9)
    assert verdict.status == "SIGNALED", verdict.reason
    assert signals == [(10, 15)]

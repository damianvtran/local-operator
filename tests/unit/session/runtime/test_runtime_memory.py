"""The per-runtime memory guard (``session.runtime.runtime_memory``).

WHY THIS FILE EXISTS: on 2026-11-24 one session runtime cold-booted to 2.0-2.4 GB
while idle and drove the operator's Mac into swap death twice, and no guard could
end it — the machine pass deliberately never ends a runtime. These cells pin the
guard that now can, and every rule that keeps it from ending the wrong thing:

* over the ceiling for N consecutive samples spanning the sustained window → the
  run is ended (negative control: N-1 samples → nothing);
* a boot transient never matures (the ``BOOT_TRANSIENT`` invariant, and the
  default ceiling sits above the measured normal maximum);
* a birth mismatch, an unproven config root, or ``enabled: false`` → no act;
* against a REAL child process: the involuntary marker, dormant schedules, the
  SIGTERM, the escalation to SIGKILL of that exact pid, and the refusal to signal
  a pid whose birth does not match the record.

Every act-cell has a sibling showing the same setup WITHOUT the deciding fact does
nothing (or the reverse), so a guard that always or never acted fails here.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime import control, registry, runtime_memory
from local_operator.session.runtime.runtime_memory import (
    GuardAct,
    RuntimeGuardSettings,
    RuntimeGuardTracker,
    runtime_memory_pass,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")

SESSION_ID = "rtmem0000001"

#: The largest live runtime measured on the operator's machine (2026-11-23, 17
#: runtimes, six samples each; see ``RUNTIME_MEMORY_LIMIT_MB_DEFAULT``). Restated
#: here so a default lowered beneath it fails by name.
MEASURED_NORMAL_MAX_MIB = 5028

#: The #4513 first-open peak for a 1.27 GB journal, and the without-#4513 load.
MEASURED_BOOT_PEAK_MIB = 1193
MEASURED_BOOT_PEAK_WITHOUT_FIX_MIB = 3 * 1024


def _record(pid: int = 4242, started_at: float | None = None) -> Any:
    return SimpleNamespace(
        session_id=SESSION_ID,
        pid=pid,
        started_at=time.time() if started_at is None else started_at,
        cwd="/tmp",
        version="0.0.0",
        source_ref="test",
    )


class _Recorder:
    """An ``end`` seam that records what it was asked to end."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, int]] = []

    def __call__(self, config_dir: Path, record: Any, *, footprint_mb: int, limit_mb: int, **_):
        self.calls.append((record, footprint_mb, limit_mb))
        return GuardAct(
            session_id=record.session_id,
            pid=record.pid,
            footprint_mb=footprint_mb,
            limit_mb=limit_mb,
            method="sigterm",
        )


def _run_passes(
    tmp_path: Path,
    readings: list[int | None],
    *,
    settings: RuntimeGuardSettings,
    step_s: float = 60.0,
    root_ok: bool | None = True,
    record: Any = None,
) -> tuple[_Recorder, list[Any]]:
    rec = record or _record()
    end = _Recorder()
    tracker = RuntimeGuardTracker()
    reports = []
    t0 = 1_800_000_000.0
    for index, mb in enumerate(readings):
        reports.append(
            runtime_memory_pass(
                tmp_path,
                tracker,
                settings=settings,
                records=lambda _root: [rec],
                footprint_mb=lambda _pid, mb=mb: mb,
                root_proof=lambda _r, _root: root_ok,
                end=end,
                notify=lambda _root, _act: "spooled",
                now=t0 + index * step_s,
            )
        )
    return end, reports


SETTINGS = RuntimeGuardSettings(enabled=True, limit_mb=8192, samples=3, sustained_s=120.0)


def test_over_the_ceiling_for_n_samples_ends_the_runtime(tmp_path: Path) -> None:
    end, reports = _run_passes(tmp_path, [9000, 9100, 9200], settings=SETTINGS)
    assert len(end.calls) == 1
    record, mb, limit = end.calls[0]
    assert (record.pid, mb, limit) == (4242, 9200, 8192)
    assert reports[-1].state == "acted"
    assert reports[-1].acts[0].notice == "spooled"


def test_negative_control_one_sample_short_does_nothing(tmp_path: Path) -> None:
    end, reports = _run_passes(tmp_path, [9000, 9100], settings=SETTINGS)
    assert end.calls == []
    assert reports[-1].state == "watching"


def test_a_transient_that_dips_below_resets_the_streak(tmp_path: Path) -> None:
    """Over, over, UNDER, over, over: never three consecutive, so no act."""
    end, _ = _run_passes(tmp_path, [9000, 9000, 4000, 9000, 9000], settings=SETTINGS)
    assert end.calls == []


def test_an_unreadable_sample_breaks_the_streak_too(tmp_path: Path) -> None:
    """Unknown never acts — and never counts toward an act."""
    end, reports = _run_passes(tmp_path, [9000, None, 9000, 9000], settings=SETTINGS)
    assert end.calls == []
    assert reports[1].unmeasured == 1


def test_boot_transient_invariant_samples_inside_the_window_never_act(tmp_path: Path) -> None:
    """BOOT_TRANSIENT: three over-ceiling samples SECONDS apart (a supervisor
    restart runs a pass at once) must not end a booting runtime. Negative control:
    the same three samples spread across the window do act (cell above)."""
    end, _ = _run_passes(tmp_path, [9000, 9000, 9000], settings=SETTINGS, step_s=5.0)
    assert end.calls == []


def test_boot_transient_invariant_default_ceiling_is_above_every_measured_normal_load() -> None:
    """A normal big session must never be parked mid-boot: the default sits above
    the largest measured live runtime and both measured first-open peaks, and the
    sustained window is longer than one supervisor interval."""
    from local_operator.wakes.supervisor import MACHINE_MEMORY_INTERVAL_S

    default = runtime_memory.RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    assert default > MEASURED_NORMAL_MAX_MIB
    assert default > MEASURED_BOOT_PEAK_MIB
    assert default > MEASURED_BOOT_PEAK_WITHOUT_FIX_MIB
    assert runtime_memory.RUNTIME_MEMORY_SUSTAINED_S > MACHINE_MEMORY_INTERVAL_S
    assert runtime_memory.RUNTIME_MEMORY_SAMPLES >= 2


def test_a_normal_big_session_at_the_measured_max_is_never_acted_on(tmp_path: Path) -> None:
    end, reports = _run_passes(
        tmp_path,
        [MEASURED_NORMAL_MAX_MIB] * 6,
        settings=RuntimeGuardSettings(),
    )
    assert end.calls == []
    assert reports[-1].state == "ok"


def test_off_reads_nothing_and_acts_on_nothing(tmp_path: Path) -> None:
    reads: list[int] = []
    end = _Recorder()
    tracker = RuntimeGuardTracker()
    for index in range(4):
        report = runtime_memory_pass(
            tmp_path,
            tracker,
            settings=RuntimeGuardSettings(enabled=False, limit_mb=8192),
            records=lambda _root: [_record()],
            footprint_mb=lambda pid: reads.append(pid) or 9999,
            root_proof=lambda _r, _root: True,
            end=end,
            now=1_800_000_000.0 + index * 60,
        )
        assert report.state == "off"
    assert reads == [] and end.calls == []


def test_off_comes_from_config(tmp_path: Path) -> None:
    """The off switch is read from ``config.yml`` (negative control: absent → on)."""
    assert runtime_memory.read_settings(tmp_path).enabled is True
    (tmp_path / "config.yml").write_text(
        "values:\n  session:\n    runtime_memory:\n      enabled: false\n"
    )
    assert runtime_memory.read_settings(tmp_path).enabled is False


def _write_limit(root: Path, value: int) -> None:
    (root / "config.yml").write_text(
        f"values:\n  session:\n    runtime_memory:\n      limit_mb: {value}\n"
    )


def test_a_ceiling_below_the_floor_is_invalid_and_the_default_applies(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: a units typo (``8`` meant as GiB) must not become a 1 GiB fleet ceiling.

    Q2: the supervisor re-reads every pass, so the warning is said ONCE per value.
    Negative control: a valid value at or above the floor is used as written.
    """
    monkeypatch.setattr(runtime_memory, "_WARNED_LIMITS", set())
    _write_limit(tmp_path, 8)
    with caplog.at_level("WARNING", logger=runtime_memory.__name__):
        first = runtime_memory.read_settings(tmp_path)
        second = runtime_memory.read_settings(tmp_path)
    assert first.limit_mb == second.limit_mb == runtime_memory.RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    warnings = [r for r in caplog.records if "below the" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "MiB" in warnings[0].getMessage()
    # A different bad value is a new edit and earns its own line.
    _write_limit(tmp_path, 512)
    with caplog.at_level("WARNING", logger=runtime_memory.__name__):
        runtime_memory.read_settings(tmp_path)
    assert len([r for r in caplog.records if "below the" in r.getMessage()]) == 2
    for valid in (runtime_memory.RUNTIME_MEMORY_LIMIT_MB_FLOOR, 2048):
        _write_limit(tmp_path, valid)
        assert runtime_memory.read_settings(tmp_path).limit_mb == valid


# --- the operator's TUI is never ended (R5) --------------------------------------


def test_an_over_ceiling_tui_is_warned_once_and_never_ended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tui = _record()
    tui.kind = "tui"
    with caplog.at_level("WARNING", logger=runtime_memory.__name__):
        end, reports = _run_passes(tmp_path, [9000] * 5, settings=SETTINGS, record=tui)
    assert end.calls == [], "the guard ended the operator's foreground TUI"
    assert all(report.acts == () and report.over == () for report in reports)
    lines = [r.getMessage() for r in caplog.records if "TUI" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "9000 MiB" in lines[0]


def test_negative_control_the_same_load_in_a_daemon_runtime_is_ended(tmp_path: Path) -> None:
    for kind in ("daemon", "exec"):
        record = _record()
        record.kind = kind
        end, _ = _run_passes(tmp_path, [9000] * 3, settings=SETTINGS, record=record)
        assert len(end.calls) == 1, kind


# --- the supervisor hold (Q1) ------------------------------------------------------


def test_the_guard_holds_the_supervisor_while_a_guardable_runtime_is_live(
    tmp_path: Path,
) -> None:
    daemon = _record()
    daemon.kind = "daemon"
    tui = _record(pid=4343)
    tui.kind = "tui"
    on = RuntimeGuardSettings(enabled=True)
    off = RuntimeGuardSettings(enabled=False)

    def hold(records: list[Any], settings: RuntimeGuardSettings) -> bool:
        return runtime_memory.holds_supervisor(
            tmp_path, records=lambda _root: records, settings=settings
        )

    assert hold([daemon], on) is True
    assert hold([daemon, tui], on) is True
    # Negative controls: nothing live, only the operator's TUI, the guard off, and
    # this process's own record each leave retirement to its old rule.
    assert hold([], on) is False
    assert hold([tui], on) is False
    assert hold([daemon], off) is False
    own = _record(pid=os.getpid())
    own.kind = "daemon"
    assert hold([own], on) is False


def test_a_failing_hold_read_never_raises_and_never_holds(tmp_path: Path) -> None:
    def boom(_root: Path) -> list[Any]:
        raise OSError("registry unreadable")

    assert runtime_memory.holds_supervisor(tmp_path, records=boom) is False


# --- the notice (U3, U5/N1) --------------------------------------------------------


def test_the_notice_speaks_to_the_operator_in_the_settings_unit() -> None:
    text = runtime_memory.notice_text(
        GuardAct(
            session_id=SESSION_ID, pid=4242, footprint_mb=9400, limit_mb=8192, method="sigterm"
        )
    )
    assert "9,400 MiB" in text and "8,192 MiB" in text
    assert "GiB" not in text
    # The /settings route by its labels, never the raw config key.
    assert "Session memory limit" in text and "Session ceiling (MiB)" in text
    assert "session.runtime_memory.limit_mb" not in text
    assert "dormant, not deleted" in text and "resume it" in text
    assert "RUNTIME MEMORY GUARD" not in text


def test_an_unproven_config_root_never_acts(tmp_path: Path) -> None:
    for root_ok in (None, False):
        end, _ = _run_passes(tmp_path, [9000] * 4, settings=SETTINGS, root_ok=root_ok)
        assert end.calls == [], root_ok


# --- the act, against REAL processes ------------------------------------------------


def _spawn(*, ignore_term: bool) -> subprocess.Popen[bytes]:
    body = (
        "import signal, sys, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
        + "time.sleep(120)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", body], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"
    return proc


@pytest.fixture()
def child():
    procs: list[subprocess.Popen[bytes]] = []

    def make(*, ignore_term: bool) -> subprocess.Popen[bytes]:
        proc = _spawn(ignore_term=ignore_term)
        procs.append(proc)
        return proc

    yield make
    for proc in procs:
        # Exact-pid reap of anything a failing cell left behind.
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def _session_root(tmp_path: Path) -> Path:
    control.session_dir(tmp_path, SESSION_ID).mkdir(parents=True)
    return tmp_path


def test_a_sigterm_ignoring_runtime_is_escalated_to_sigkill(tmp_path: Path, child) -> None:
    root = _session_root(tmp_path)
    proc = child(ignore_term=True)
    record = _record(pid=proc.pid)
    sent: list[int] = []

    def send(pid: int, sig: int) -> None:
        sent.append(sig)
        os.kill(pid, sig)

    act = runtime_memory.end_runaway_runtime(
        root, record, footprint_mb=9000, limit_mb=8192, send=send, term_grace_s=0.5
    )
    assert act.method == "sigkill"
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert proc.wait(timeout=5) == -signal.SIGKILL
    marker = registry.read_stop_marker(control.session_dir(root, SESSION_ID))
    assert marker is not None
    assert (marker["mechanism"], marker["deliberate"], marker["pid"]) == (
        runtime_memory.MECHANISM,
        False,
        proc.pid,
    )


def test_negative_control_a_cooperative_runtime_gets_only_sigterm(tmp_path: Path, child) -> None:
    root = _session_root(tmp_path)
    proc = child(ignore_term=False)
    sent: list[int] = []

    def send(pid: int, sig: int) -> None:
        sent.append(sig)
        os.kill(pid, sig)

    act = runtime_memory.end_runaway_runtime(
        root,
        _record(pid=proc.pid),
        footprint_mb=9000,
        limit_mb=8192,
        send=send,
        term_grace_s=5.0,
    )
    assert act.method == "sigterm"
    assert sent == [signal.SIGTERM]
    assert proc.wait(timeout=5) == -signal.SIGTERM


def test_a_birth_mismatch_is_never_signalled(tmp_path: Path, child) -> None:
    """The record says the run started an hour before this process existed: the
    pid has been handed to a stranger. Nothing is sent and no marker is staged."""
    root = _session_root(tmp_path)
    proc = child(ignore_term=False)
    stale = _record(pid=proc.pid, started_at=time.time() - 3600)
    assert runtime_memory.birth_matches(stale) is False
    # Negative control: the same pid with a record written after its birth matches.
    assert runtime_memory.birth_matches(_record(pid=proc.pid)) is True
    sent: list[int] = []
    act = runtime_memory.end_runaway_runtime(
        root, stale, footprint_mb=9000, limit_mb=8192, send=lambda p, s: sent.append(s)
    )
    assert act.method == "refused"
    assert sent == []
    assert proc.poll() is None
    assert registry.read_stop_marker(control.session_dir(root, SESSION_ID)) is None


def test_a_birth_that_cannot_be_reproved_after_the_grace_withholds_sigkill(
    tmp_path: Path, child
) -> None:
    """R4: the guard's ``withheld`` branch. A SIGTERM-ignoring runtime survives the
    grace, and the second birth proof fails: SIGKILL is never sent on doubt, the
    process is left alive, and the SIGTERM's marker stays.

    Negative control: the same setup with a second proof that passes escalates
    (``test_a_sigterm_ignoring_runtime_is_escalated_to_sigkill``).
    """
    root = _session_root(tmp_path)
    proc = child(ignore_term=True)
    answers = iter([True, False])
    sent: list[int] = []

    def send(pid: int, sig: int) -> None:
        sent.append(sig)
        os.kill(pid, sig)

    act = runtime_memory.end_runaway_runtime(
        root,
        _record(pid=proc.pid),
        footprint_mb=9000,
        limit_mb=8192,
        verify=lambda _r: next(answers),
        send=send,
        term_grace_s=0.5,
    )
    assert act.method == "withheld"
    assert sent == [signal.SIGTERM]
    assert proc.poll() is None, "a pid whose birth could not be re-proved was killed"
    assert registry.read_stop_marker(control.session_dir(root, SESSION_ID)) is not None


def test_the_schedules_go_dormant_not_deleted(tmp_path: Path, child) -> None:
    from local_operator.wakes import store as wake_store

    root = _session_root(tmp_path)
    schedules = [{"id": "w1", "due_ms": int(time.time() * 1000) + 60_000, "every_s": 600}]
    wake_store.write_entry(root, SESSION_ID, cwd="/tmp", schedules=schedules)
    proc = child(ignore_term=False)
    runtime_memory.end_runaway_runtime(
        root, _record(pid=proc.pid), footprint_mb=9000, limit_mb=8192, term_grace_s=5.0
    )
    entry = wake_store.read_entry(root, SESSION_ID) or {}
    assert entry.get("schedules"), "a guard stop deleted the session's schedules"
    assert control.STOP_MARKER in entry
    assert control.PARK_MARKER not in entry
    assert wake_store.is_held(entry)


def test_the_full_pass_ends_a_real_runaway_after_n_samples(tmp_path: Path, child) -> None:
    """Pass → act → notice, end to end: only the footprint and the root proof are
    injected (the synthetic child cannot be made to name this config root)."""
    root = _session_root(tmp_path)
    proc = child(ignore_term=True)
    record = _record(pid=proc.pid)
    tracker = RuntimeGuardTracker()
    settings = RuntimeGuardSettings(enabled=True, limit_mb=1024, samples=3, sustained_s=0.0)
    reports = []
    for _ in range(3):
        reports.append(
            runtime_memory_pass(
                root,
                tracker,
                settings=settings,
                records=lambda _root: [record],
                footprint_mb=lambda _pid: 2048,
                root_proof=lambda _r, _root: True,
                end=lambda *a, **k: runtime_memory.end_runaway_runtime(*a, **k, term_grace_s=0.5),
            )
        )
        assert (proc.poll() is None) == (len(reports) < 3)
    assert [r.state for r in reports] == ["watching", "watching", "acted"]
    assert reports[-1].acts[0].method == "sigkill"
    assert reports[-1].acts[0].notice == "spooled"
    inbox = (control.session_dir(root, SESSION_ID) / "inbox.jsonl").read_text()
    assert "The memory guard ended this session" in inbox

"""Unit tests for the per-command memory guard.

Everything here drives the guard through its INJECTABLE seams (``runner`` and
``footprint_probe``) exactly as ``mobile/resources.py``'s tests do — no test
forks a real ``ps`` or spawns a real memory hog. The one integration seam that
does spawn real processes (``execute_bash``) is exercised separately; this file
pins the arithmetic and the DECISION, which is where the guard can be wrong in a
way no live process would reveal.
"""

from __future__ import annotations

import pytest

from local_operator import memory_guard as mg


def _fake_runner(*, vm_stat: str = "", swapusage: str = "", ps: str = "") -> mg.Runner:
    """A runner that answers the three probes the guard makes, by argv[0]/[1].

    A keyed fake rather than a sequence: a guard tick can call ``ps`` twice (fast
    arm then fidelity arm) and a positional fake would desynchronise the moment
    the call order changed — which is exactly the sort of brittleness that would
    make a green test hide a broken tick.
    """

    def run(argv: list[str]) -> tuple[int, str]:
        if argv[:1] == ["vm_stat"]:
            return (0, vm_stat) if vm_stat else (1, "")
        if argv[:1] == ["sysctl"]:
            return (0, swapusage) if swapusage else (1, "")
        if argv[:1] == ["ps"]:
            return (0, ps) if ps else (1, "")
        return 1, ""

    return run


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------

#: A 36 GiB host with ~6.5 GiB available, matching the contract's worked numbers
#: and this host's measured state on 2026-09-21.
_VM_STAT_36G = "\n".join(
    [
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
        "Pages free:                              180000.",
        "Pages speculative:                        40000.",
        "File-backed pages:                       235000.",
    ]
)


def _pin_host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_mb: int | None,
    total_mb: int | None,
    free_swap_mb: int | None = None,
) -> None:
    """Pin the three host probes so the arithmetic is platform-independent.

    ``compute_budget`` resolves ``_total_memory_mb``/``_available_memory_mb``/
    ``_free_swap_mb`` by GLOBAL NAME at call time, and the available/swap arms
    branch on ``sys.platform`` internally — so feeding a fake ``runner`` pins the
    parsing on macOS but does nothing on Linux, where the arm reads
    ``/proc/meminfo`` and ignores the runner. A test that wants to assert the
    ARITHMETIC (not the parsing) must therefore pin these functions, or it passes
    on this 36 GiB macOS host and fails on the 16 GiB Linux CI runner — which is
    exactly what happened in CI round 1. The parsing itself is covered separately,
    below, with ``sys.platform`` pinned to darwin.
    """
    monkeypatch.setattr(mg, "_total_memory_mb", lambda: total_mb)
    monkeypatch.setattr(mg, "_available_memory_mb", lambda runner: available_mb)
    monkeypatch.setattr(mg, "_free_swap_mb", lambda runner: free_swap_mb)


def test_budget_on_a_36gib_host_matches_the_contract_worked_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """36 GiB / 7,109 MB available: ceiling 3,554 MB, soft 2,843 MB (§3.2)."""
    # available = 455000 pages x 16 KiB = 7109 MB; total = 36864 MB; the reserve
    # is min(2048, 36864//8) = 2048, so ceiling = min(3554, 7109-2048) = 3554.
    _pin_host(monkeypatch, available_mb=7109, total_mb=36864, free_swap_mb=1192)
    budget = mg.compute_budget()
    assert budget.source == "auto"
    assert budget.total_mb == 36864
    assert budget.available_mb == 7109
    assert budget.reserve_mb == 2048
    assert budget.ceiling_mb == 3554
    assert budget.soft_mb == int(3554 * mg._SOFT_FRACTION)


def test_budget_on_a_32gib_device_keeps_the_reserve_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """32 GiB with 8 GiB available: reserve = min(2048, 4096) = 2048; ceiling
    = min(4096, 8192-2048) = 4096 MB (§3.2)."""
    _pin_host(monkeypatch, available_mb=8192, total_mb=32768, free_swap_mb=3000)
    budget = mg.compute_budget()
    assert budget.ceiling_mb == 4096


def test_the_darwin_available_arm_counts_free_speculative_and_file_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The macOS parsing itself: ``free + speculative + file-backed``, and NOT
    ``inactive``.

    ``_platform`` is pinned to darwin so this exercises the vm_stat parse on
    any host without mutating the shared ``sys`` module (review round 1, n2).
    Counting ``inactive`` reported 8,137 MB of headroom at a moment the host had
    452 MB genuinely free, so a probe answer carrying a huge inactive line must
    NOT raise the number.
    """
    monkeypatch.setattr(mg, "_platform", lambda: "darwin")
    vm_inactive = "\n".join(
        [
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
            "Pages free:                               30000.",
            "Pages speculative:                        10000.",
            "File-backed pages:                        10000.",
            "Pages inactive:                          500000.",  # must be ignored
        ]
    )
    available = mg._available_memory_mb(_fake_runner(vm_stat=vm_inactive))
    # 50000 pages x 16 KiB = 800,000,000 B -> 781 MB (floored), NOT ~8 GB.
    assert available == 781


def test_the_darwin_available_arm_reads_the_header_page_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page size comes from ``vm_stat``'s header, never a hardcoded 4096.

    This host uses 16 KiB pages, so a fixed 4096 would under-report by 4x.
    ``_platform`` is pinned so the parse runs on any host without touching the
    shared ``sys`` module (review round 1, n2).
    """
    monkeypatch.setattr(mg, "_platform", lambda: "darwin")
    vm_4k = "\n".join(
        [
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)",
            "Pages free:                               30000.",
            "Pages speculative:                        10000.",
            "File-backed pages:                        10000.",
        ]
    )
    assert mg._available_memory_mb(_fake_runner(vm_stat=vm_4k)) == 195  # 50000x4096 B


def test_swap_free_is_a_pressure_floor_never_spendable_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low free swap LOWERS the ceiling; it is never added to it (§3.3)."""
    _pin_host(monkeypatch, available_mb=7109, total_mb=36864, free_swap_mb=1192)
    healthy = mg.compute_budget()
    _pin_host(monkeypatch, available_mb=7109, total_mb=36864, free_swap_mb=64)
    pressured = mg.compute_budget()
    assert pressured.ceiling_mb < healthy.ceiling_mb
    # Free swap is a FLOOR on the effective available, not an addition: the
    # effective arm is min(available, free_swap + the 256 MB floor).
    assert pressured.available_mb == 7109  # reported arm unchanged


def test_small_device_floor_keeps_ordinary_commands_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 8 GiB host at ~1 GiB available resolves below the floor; the floor
    keeps it usable (§10's small-device risk)."""
    # available 1050 MB, total 8192 MB -> reserve 1024, so available - reserve =
    # 26 MB, below the 64 MB floor.
    _pin_host(monkeypatch, available_mb=1050, total_mb=8192, free_swap_mb=800)
    budget = mg.compute_budget()
    assert budget.ceiling_mb == mg._MIN_CEILING_MB
    assert "small-device floor" in budget.reason


# ---------------------------------------------------------------------------
# Resolution order: override / manual / disabled / unmeasurable
# ---------------------------------------------------------------------------


def test_per_call_override_wins_and_needs_no_host_probe() -> None:
    budget = mg.compute_budget(override_mb=512, runner=_fake_runner())
    assert budget.source == "override"
    assert budget.ceiling_mb == 512


def test_per_call_zero_disables_for_that_command_only() -> None:
    budget = mg.compute_budget(override_mb=0, runner=_fake_runner())
    assert budget.source == "disabled"
    assert budget.ceiling_mb == 0


def test_the_override_and_manual_arms_spend_no_host_probe() -> None:
    """An explicit ceiling must not fork ``vm_stat``/read ``/proc`` (m4).

    The arms that already know the number do not need the host's memory, so a
    probe that RAISES proves the arms never reach it.
    """

    def exploding_runner(argv: list[str]) -> tuple[int, str]:
        raise AssertionError(f"host probe must not run for an explicit ceiling: {argv}")

    override = mg.compute_budget(override_mb=512, runner=exploding_runner)
    assert override.source == "override" and override.ceiling_mb == 512
    assert override.available_mb is None  # cosmetic field honestly unset
    manual = mg.compute_budget(mode="manual", limit_mb=256, runner=exploding_runner)
    assert manual.source == "manual" and manual.ceiling_mb == 256
    assert manual.available_mb is None


def test_enabled_false_disables_the_guard() -> None:
    budget = mg.compute_budget(enabled=False, runner=_fake_runner())
    assert budget.source == "disabled"


def test_the_off_switch_beats_a_positive_per_call_override() -> None:
    """``enabled=False`` + a positive ``memory_mb`` is DISABLED (QA round 1, Q1).

    ``enabled`` is the machine-wide MASTER SWITCH (contract §6/§7), checked
    before a positive override, so a per-call ceiling cannot overrule it. This
    pins the precedence the docstring now states, so the two cannot drift back
    apart.
    """
    budget = mg.compute_budget(enabled=False, override_mb=64, runner=_fake_runner())
    assert budget.source == "disabled"
    assert budget.ceiling_mb == 0
    # `memory_mb=0` against a positive config is the OTHER direction and is
    # covered above; here the config switch wins over the per-call number.


def test_manual_mode_uses_limit_mb() -> None:
    budget = mg.compute_budget(mode="manual", limit_mb=1234, runner=_fake_runner())
    assert budget.source == "manual"
    assert budget.ceiling_mb == 1234


def test_manual_mode_with_zero_limit_falls_through_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`limit_mb=0` means "use the auto ceiling", never "zero"."""
    _pin_host(monkeypatch, available_mb=7109, total_mb=36864, free_swap_mb=1192)
    budget = mg.compute_budget(mode="manual", limit_mb=0)
    assert budget.source == "auto"


def test_unmeasurable_host_degrades_to_disabled_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F8/F11: no probe answers -> disabled, not a guessed ceiling."""
    _pin_host(monkeypatch, available_mb=None, total_mb=None, free_swap_mb=None)
    budget = mg.compute_budget()
    assert budget.source == "disabled"
    assert budget.ceiling_mb == 0


# ---------------------------------------------------------------------------
# group_rss_bytes: membership, unknowns, fail-closed
# ---------------------------------------------------------------------------

_PS_TABLE = "\n".join(
    [
        "  100   100   10240",  # group leader, 10 MiB
        "  101   100   20480",  # same group, 20 MiB
        "  102   102  999999",  # another group — must NOT be counted
    ]
)


def test_group_rss_sums_only_the_named_group() -> None:
    total = mg.group_rss_bytes(100, runner=_fake_runner(ps=_PS_TABLE))
    assert total == (10240 + 20480) * 1024


def test_group_rss_returns_none_when_the_group_is_gone() -> None:
    """A vanished group is GONE, not zero and not huge."""
    table = "  102   102  999999"
    assert mg.group_rss_bytes(100, runner=_fake_runner(ps=table)) is None


def test_group_rss_returns_none_on_a_failed_probe() -> None:
    assert mg.group_rss_bytes(100, runner=_fake_runner()) is None


# ---------------------------------------------------------------------------
# The kill decision: measured-over kills, unknown NEVER kills
# ---------------------------------------------------------------------------


def _guard(ceiling_mb: int = 100, ps: str = "") -> mg.Guard:
    budget = mg.Budget(
        ceiling_mb=ceiling_mb,
        soft_mb=int(ceiling_mb * 0.8),
        available_mb=4096,
        total_mb=16384,
        reserve_mb=2048,
        source="auto",
        reason="test",
    )
    return mg.Guard(100, budget, runner=_fake_runner(ps=ps), footprint_probe=lambda pid: None)


@pytest.mark.asyncio
async def test_over_budget_reading_kills() -> None:
    # 200 MiB in the group against a 100 MB ceiling.
    ps = "  100   100   204800"
    guard = _guard(100, ps=ps)
    sample = await guard.sample()
    assert sample.bytes_used == 204800 * 1024
    assert guard.should_kill(sample) is True


@pytest.mark.asyncio
async def test_unknown_reading_never_kills() -> None:
    """F6: a failed read must not kill. `ps` returning nothing is a no-op."""
    guard = _guard(100, ps="")  # the group is absent -> None
    sample = await guard.sample()
    assert sample.bytes_used is None
    assert guard.should_kill(sample) is False


@pytest.mark.asyncio
async def test_under_budget_reading_does_not_kill() -> None:
    guard = _guard(100, ps="  100   100   10240")
    sample = await guard.sample()
    assert guard.should_kill(sample) is False


@pytest.mark.asyncio
async def test_soft_notice_latches_once() -> None:
    """The advisory is one-shot per guard, so a group over the soft line does not
    spam the stream."""
    ps = "  100   100   92160"  # 90 MiB, over the 80 MB soft line, under 100 hard
    guard = _guard(100, ps=ps)
    sample = await guard.sample()
    first = guard.soft_notice(sample)
    assert first is not None and "approaching the command budget" in first
    assert guard.soft_notice(sample) is None


@pytest.mark.asyncio
async def test_the_guard_reads_only_the_pgid_it_was_handed() -> None:
    """F5: a tick sums a group id it holds; it never discovers one.

    The ps table carries a DECOY group holding far more; only the guarded pgid's
    bytes are ever charged.
    """
    ps = "\n".join(["  100   100   10240", "  999   999  99999999"])
    guard = _guard(100, ps=ps)
    sample = await guard.sample()
    assert sample.bytes_used == 10240 * 1024


def test_should_kill_is_pure_over_a_none_usage() -> None:
    guard = _guard(100)
    sample = mg.Sample(
        pgid=100,
        bytes_used=None,
        bytes_soft=80 * 1024 * 1024,
        bytes_hard=100 * 1024 * 1024,
        over_soft=False,
        over_hard=True,  # even if a caller sets this, a None usage must not kill
    )
    assert guard.should_kill(sample) is False


# ---------------------------------------------------------------------------
# The tool-result wording (§5)
# ---------------------------------------------------------------------------


def test_over_budget_message_is_plain_text_and_names_the_numbers() -> None:
    guard = _guard(100, ps="  100   100   204800")
    sample = mg.Sample(
        pgid=100,
        bytes_used=204800 * 1024,
        bytes_soft=80 * 1024 * 1024,
        bytes_hard=100 * 1024 * 1024,
        over_soft=True,
        over_hard=True,
    )
    message = guard.over_budget_message(sample)
    assert message.startswith("MEMORY LIMIT EXCEEDED:")
    assert "`" not in message  # the tool card paints Text; backticks land literally
    assert "was killed; the session is fine" in message
    assert "memory_mb" in message  # names the escape hatch
    assert "bash.memory.limit_mb" in message
    # ONE paragraph, no internal newline (design review D3): the card claims only
    # the first result line for its wrapping reason block, so a second line fell
    # into captured output and was cropped at canonical widths, cutting the
    # remedy and the escape hatch.
    assert "\n" not in message


# ---------------------------------------------------------------------------
# Real processes (§8.2). Marked `slow` and kept IN the default run, like the
# other subprocess-boundary tests: these are the only coverage that the guard
# actually reads a REAL group and that the kill path really ends it.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_group_is_killed_at_the_ceiling_and_the_runtime_survives() -> None:
    """A real command group that allocates past a small ceiling is killed; THIS
    process (the runtime stand-in) is untouched.

    Driven through the guard + ``terminate_process_tree`` — the exact pair
    ``execute_bash``'s ``_kill()`` uses — so the test exercises the real kill
    primitive, not a mock of it.
    """
    import asyncio
    import os
    import subprocess
    import time

    from local_operator import procstate

    proc = subprocess.Popen(  # noqa: S603 — the test's own command
        ["python3", "-c", "import time; b=bytearray(400*1024*1024); time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pgid = proc.pid
    try:
        pgid = os.getpgid(proc.pid)
        budget = mg.Budget(
            ceiling_mb=100,
            soft_mb=80,
            available_mb=4096,
            total_mb=16384,
            reserve_mb=2048,
            source="auto",
            reason="test",
        )
        guard = mg.Guard(pgid, budget)
        killed = False
        for _ in range(40):  # up to ~10 s for the allocation to land
            sample = await guard.sample()
            if guard.should_kill(sample):
                killed = True
                procstate.terminate_process_tree(pgid, force=True)
                break
            await asyncio.sleep(0.25)
        assert killed, "the guard never saw the group cross its ceiling"
        # The group is gone...
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:
            await asyncio.sleep(0.05)
        assert proc.poll() is not None, "the killed group was not reaped"
        # ...and THIS process (the runtime) is very much alive.
        os.kill(os.getpid(), 0)
    finally:
        if proc.poll() is None:
            procstate.terminate_process_tree(pgid, force=True)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_bash_reports_memory_limit_exceeded_as_an_error() -> None:
    """The tool result contract: `is_error=True`, the line at index 0, plain Text.

    A per-call ``memory_mb`` override is the smallest ceiling that is still
    deterministic — no dependence on this host's real available memory.
    """
    from local_operator.tools.builtin import execute_bash

    result = await execute_bash(
        "mem-1",
        {
            "command": "python3 -c 'import time; b=bytearray(400*1024*1024); time.sleep(30)'",
            "memory_mb": 64,
        },
        None,
        None,
        None,
    )
    assert result.is_error is True
    first = result.text.splitlines()[0]
    assert first.startswith("MEMORY LIMIT EXCEEDED:")
    assert "`" not in first
    assert result.details and result.details.get("memory_exceeded") is True


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_bash_runs_small_commands_unguarded() -> None:
    """A command well under the ceiling runs normally — the guard is silent."""
    from local_operator.tools.builtin import execute_bash

    result = await execute_bash("mem-2", {"command": "echo alive"}, None, None, None)
    assert result.is_error is False
    assert "alive" in result.text


def test_a_config_edit_is_read_on_the_next_command(tmp_path: object) -> None:
    """The keys are LIVE: ``_configured_memory_budget`` reads a FRESH
    ``ConfigManager`` per call, so a write lands on the very next command with no
    session rebuild.

    This is the live-apply proof the ``memory_guard`` section is exempted from the
    session fanout for (it is host-owned, read in the tool layer): the reader is
    exercised directly against a written config, and the resolved ceiling moves.
    """
    from pathlib import Path

    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.tools.builtin import _configured_memory_budget

    config_dir = Path(str(tmp_path))
    manager = ConfigManager(config_dir)
    settings_io.write_setting(manager, settings_io.BY_KEY["bash.memory.mode"], "manual")
    settings_io.write_setting(manager, settings_io.BY_KEY["bash.memory.limit_mb"], 777)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
        budget = _configured_memory_budget(None)
        assert budget.source == "manual"
        assert budget.ceiling_mb == 777
        assert budget.soft_mb == int(777 * 0.8)
        # A per-call override still wins over the stored manual ceiling.
        override = _configured_memory_budget(123)
        assert override.source == "override"
        assert override.ceiling_mb == 123

"""Guards for the root ``conftest.py`` worker-count hook.

The hook resolves ``-n auto`` to a count the machine can sustain. It had **no
test coverage at all** before this module, which is how two defects survived in
it:

- **The harness lied to it about ``CI``.** ``local_operator/tools/builtin.py``
  injects ``CI=1`` into every agent-run command to make CLIs non-interactive.
  The hook reads ``CI`` as "dedicated runner, take every core", so every
  agent-run suite on the operator's shared laptop disabled the CPU share that
  exists precisely to keep sibling suites from thrashing the box. Measured
  2026-09-08: six concurrent suites, 29 workers, ~11.2 GB RSS, load average
  220, 8.67 of 10 GB swap, 448 MB free.
- **The memory budget claimed a fraction of what remained**, so N sibling
  suites converged toward zero free memory instead of toward a floor.
- **The count was invisible**, which is the third defect, and it is a reporting
  defect rather than a calculation one: the resolution is a product of four
  inputs nothing prints, so a session bisecting a parallelism-sensitive failure
  does not know what parallelism it had (AGENTS.md) and a reader seeing "3
  workers" on a 14-core machine has to guess which term produced it. The line
  pinned in "The one-line decision report" below fixes that, and it is the
  shipped change.
- **A fleet divisor was implemented, measured and REJECTED**, and that is
  recorded here because it is the tempting fix for the fleet's unbounded total
  claim - dividing both arms by ``1 + live sibling suites``. On this host it
  resolves every suite to the 2-worker floor, where the slice below went from
  219.8 / 216.3 / 209.8 s to 338.4 / 314.5 / 308.1 s per instance at effectively
  identical CPU. ``conftest.py``'s module docstring carries the full measurement
  so nobody re-tries it without that data, and
  `test_the_fleet_size_is_reported_and_never_divided_by` is the executable
  version of the rule.

Every assertion below was mutation-tested against the defect it claims to
catch: the fix was reverted in the working tree, the test was shown to FAIL,
and the fix restored. That is not ceremony. The design review of this change
caught a proposed regression test that returned the same value with the fix
applied and reverted, because an unrelated clamp masked the difference - a
guard that cannot fail on the buggy code teaches every future reader that a
regression is pinned when it is not.

Three consequences of that for how these tests are written:

- **``os.cpu_count``, both memory probes and the fleet probe are pinned**, never
  read from the host. A test whose expected value depends on the developer's
  core count, current free memory, or how many sibling suites happen to be
  running cannot assert an absolute, and this suite runs on everything from a
  2-vCPU runner to a 14-core laptop that is, by design, running other suites.
- **Several tests assert on a DIFFERENCE between two environments** rather than
  on one number, because the ``2..8`` clamp can flatten both arms of a branch
  onto the same value and hide the very behaviour under test.
- **The fleet tests run in their own section** below the single-suite ones,
  because they pin two different things: what the report says, and that the
  number in it is NOT used to shrink the cap.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFTEST = REPO / "conftest.py"


def _load_hook_module() -> types.ModuleType:
    """Import the ROOT ``conftest.py`` under a private name.

    Loaded fresh under a distinct module name rather than reused from
    ``sys.modules``: pytest has already imported the real one as a plugin, and
    monkeypatching attributes on that live object would change the behaviour of
    the very session running these tests.
    """
    spec = importlib.util.spec_from_file_location("_xdist_budget_under_test", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook_module() -> types.ModuleType:
    return _load_hook_module()


def _resolve(
    module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpus: int,
    available_mb: int | None,
    total_mb: int | None = 36864,
    env: dict[str, str] | None = None,
    siblings: int | None = 0,
) -> int:
    """Run the hook against a fully synthetic machine.

    Every input the hook reads is pinned here - core count, both memory probes,
    the fleet probe, and the four environment variables - so the result is a
    pure function of the arguments and identical on a 2-vCPU runner and a
    14-core laptop. ``PYTEST_QUIET_WORKER_CAP`` is scrubbed with the rest even
    though it changes only the reporting, not the count: this module asserts on
    the report, and a developer who follows the PR's own advice ("silence it
    with PYTEST_QUIET_WORKER_CAP=1") would otherwise see 9 failures from a green
    tree. It is the same fixture-level fix AGENTS.md documents for
    AWS_DEFAULT_PROFILE - the local shell must not decide what the tests see.

    ``siblings`` defaults to 0, i.e. a solo machine. Pinning it keeps every
    assertion in this file independent of how many sibling suites happen to be
    running on the developer's laptop, which matters more here than in most
    repos because this suite is routinely run alongside five or ten others.
    ``None`` leaves the fleet probe to the caller - the probe-failure test uses
    that to exercise the real probe through a patched ``subprocess.run``.
    """
    monkeypatch.setattr(module.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(module, "_available_memory_mb", lambda: available_mb)
    monkeypatch.setattr(module, "_total_memory_mb", lambda: total_mb)
    if siblings is not None:
        monkeypatch.setattr(module, "_live_sibling_suites", lambda: siblings)
    for name in (
        "CI",
        "LOCAL_OPERATOR_AGENT_SHELL",
        "PYTEST_XDIST_AUTO_NUM_WORKERS",
        # Read from the module rather than spelled out so the name cannot drift
        # out of step with the hook's own constant.
        module._QUIET_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return module.pytest_xdist_auto_num_workers(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The loader itself (re-executing the conftest, and the guard it re-installs)
# ---------------------------------------------------------------------------


def test_reloading_the_hook_module_does_not_stack_the_real_store_guard() -> None:
    """A second load of the root conftest must REUSE the guard, not wrap it again.

    WHY it is here: ``_load_hook_module`` executes the conftest's module level,
    and that ends in ``_install_real_store_guard()``. Installing unconditionally
    wrapped whatever was already installed, so the loads this module performs
    left one wrapper per load stacked on ``os.remove``/``os.unlink``/``rmtree``
    (measured: nesting depth == number of loads, 14 tests -> 28 loads). Every
    later removal in the worker that ran them then paid one ``Path.resolve()``
    per layer, on the suite whose whole subject is wall time - a self-inflicted
    tax on unrelated tests sharing that worker.

    Falsifiable: reverting the install to unconditional makes the identity
    assertion below fail on the second load.
    """
    module = _load_hook_module()
    installed = (os.remove, os.unlink, shutil.rmtree)
    _load_hook_module()
    _load_hook_module()
    now = (os.remove, os.unlink, shutil.rmtree)
    assert now == installed, "a later load must keep the installed guard, not wrap it again"
    if module._REAL_STORE is not None:
        # Where there is a real store to protect the invariant is stronger than
        # identity: the guard is present, and it is exactly one layer deep.
        assert getattr(os.remove, module._GUARD_MARKER, False), "the guard must be installed"
        assert not getattr(
            getattr(os.remove, "__wrapped__", None), module._GUARD_MARKER, False
        ), "the wrapper must close over the real os.remove, not a previous wrapper"


# ---------------------------------------------------------------------------
# The `CI` discriminator
# ---------------------------------------------------------------------------
#
# `_MAX_WORKERS` is 8 and the CPU share halves, so on a 14-core host the two
# arms are 7 (share applied) and 8 (share lifted, clamped). That one-worker gap
# is the entire observable signal, which is why these tests pin 14 cores and
# ample memory: at 12 cores or fewer with `_MAX_WORKERS` lowered to 6 the two
# arms collapse onto the same number and the test silently stops testing.

_AMPLE_MB = 24000


def test_agent_shell_marker_denies_the_ci_core_grab(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent-run suite must NOT be treated as a dedicated runner.

    This is the incident. Reverting the denylist makes this return 8.
    """
    workers = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=_AMPLE_MB,
        env={"CI": "1", "LOCAL_OPERATOR_AGENT_SHELL": "1"},
    )
    assert workers == 7, "an agent shell must get the developer CPU share, not every core"


def test_the_marker_means_the_same_thing_to_the_hook_and_the_guard(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence was the old test; production reads ``1``/``true``/``yes``/``on``.

    ``LOCAL_OPERATOR_AGENT_SHELL=0`` used to deny the CI core grab here while
    `agent_shell.in_agent_shell` allowed the run: one variable, two answers
    (review round 1, F4). Routing the hook through the production predicate is
    what this asserts — the value must resolve to the PLAIN-CI arm.
    """
    workers = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=_AMPLE_MB,
        env={"CI": "1", "LOCAL_OPERATOR_AGENT_SHELL": "0"},
    )
    plain = _resolve(hook_module, monkeypatch, cpus=14, available_mb=_AMPLE_MB, env={"CI": "1"})
    assert workers == plain, "an off value must not read as an agent shell"


def test_agent_shell_and_plain_ci_differ(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserted as a DIFFERENCE, because an absolute can be masked by the clamp.

    If a future change lowers ``_MAX_WORKERS`` to 6, ``min(7, 6) == min(14, 6)``
    and both arms return 6 - the discriminator becomes unobservable on any host
    with 12+ cores while still looking green. This test fails loudly in that
    case instead, which is the point.
    """
    agent = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=_AMPLE_MB,
        env={"CI": "1", "LOCAL_OPERATOR_AGENT_SHELL": "1"},
    )
    real_ci = _resolve(hook_module, monkeypatch, cpus=14, available_mb=_AMPLE_MB, env={"CI": "1"})
    assert agent < real_ci, (
        "the agent-shell denylist must be observable at this core count; "
        f"got agent={agent} real_ci={real_ci}"
    )


def test_plain_ci_still_takes_every_core(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CI contract this change must not break.

    A 4-vCPU runner setting only ``CI`` - Jenkins, Azure, Travis, GitLab, and
    every self-hosted runner - keeps all 4. Applying the developer share there
    resolves 4 vCPUs to 2 workers, the halving AGENTS.md documents as "a
    regression paid on every PR".

    This is the guard against fixing the ``CI`` lie with an ALLOWLIST of
    provider variables (``GITHUB_ACTIONS``, ...) instead of a denylist: under an
    allowlist this returns 2.
    """
    workers = _resolve(hook_module, monkeypatch, cpus=4, available_mb=_AMPLE_MB, env={"CI": "1"})
    assert workers == 4, "a CI provider that sets only CI must keep every core"


def test_developer_machine_is_unchanged_by_the_marker(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain developer shell (no ``CI`` at all) behaves exactly as before."""
    workers = _resolve(hook_module, monkeypatch, cpus=14, available_mb=_AMPLE_MB)
    assert workers == 7


def test_bash_tool_exports_the_marker_the_hook_denies_on() -> None:
    """The two halves of this fix live in different files; pin them together.

    The hook's denylist is inert unless ``builtin.py`` actually exports the
    marker, and nothing else in the tree reads it - so a well-meaning cleanup
    of an "unused" env var would silently restore the incident. This is the
    only thing connecting them.
    """
    from local_operator.tools.builtin import NON_INTERACTIVE_ENV

    module = _load_hook_module()
    assert NON_INTERACTIVE_ENV[module._AGENT_SHELL_ENV] == "1"
    # CI stays exported: it is why the dict exists, and dropping it would change
    # npm/jest/yarn/playwright behaviour in every agent-run command.
    assert NON_INTERACTIVE_ENV["CI"] == "1"


# ---------------------------------------------------------------------------
# The memory reserve
# ---------------------------------------------------------------------------


def test_reserve_reduces_workers_under_memory_pressure(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below twice the reserve, the budget is the reserve arm rather than the share.

    3,600 MB available on a 36 GB host (below the 2,048 MB reserve's 2x
    boundary): the share alone would allow ``1800 // 600 = 3`` workers; holding
    the 2,048 MB reserve back allows ``1552 // 600 = 2``. Reverting the reserve
    returns 3.
    """
    workers = _resolve(hook_module, monkeypatch, cpus=14, available_mb=3600)
    assert workers == 2


def test_reserve_cannot_drive_the_result_below_min_workers(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with less free memory than the reserve still runs, at the floor.

    ``available - reserve`` is negative or worth less than two workers across
    this whole range, so the budget happens to clamp to 0 and the result must
    still be ``_MIN_WORKERS`` - never 0, never negative, and never an exception.
    A suite that refuses to start because the machine is busy would be a far
    worse failure than a slow one.
    """
    for available_mb in (0, 100, 500, 1000, 2000, 3072):
        workers = _resolve(hook_module, monkeypatch, cpus=14, available_mb=available_mb)
        assert workers == hook_module._MIN_WORKERS, f"at {available_mb} MB available"
        assert workers > 0


def test_solo_developer_at_generous_memory_is_not_double_charged(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reserve must not tax a run that is the only one on the machine.

    The shape is ``min(share, available - reserve)``. The rejected alternative,
    ``(available - reserve) * share``, applies two politeness terms to the same
    memory and costs workers even with no sibling suites at all: at 6,000 MB free
    it returns 3 where today returns 5, and at 4,000 MB it returns 2 where today
    returns 3. Both assertions below fail under that shape.

    64 cores deliberately: on 14 the CPU arm (7) or the 8-worker clamp would
    flatten the two shapes onto the same number and the test would stop testing.
    """
    assert _resolve(hook_module, monkeypatch, cpus=64, available_mb=6000) == 5
    assert _resolve(hook_module, monkeypatch, cpus=64, available_mb=4000) == 3


def test_reserve_is_invisible_above_twice_itself(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the crossover, because it is the thing most likely to be mistuned.

    ``min(a * 0.5, a - reserve) == a * 0.5`` for ``a >= 2 * reserve``. Anyone
    raising the reserve to buy fleet headroom is really moving this boundary,
    and doing so re-imposes the solo-developer penalty above.
    """
    reserve = min(hook_module._MEMORY_RESERVE_CAP_MB, 36864 // hook_module._MEMORY_RESERVE_FRACTION)
    # Asserted as reserve-vs-no-reserve at the same availability rather than
    # against an absolute: `total_mb=None` is the only way to switch the term
    # off without editing the module, and it makes the boundary the subject.
    # A high core count keeps the CPU arm slack so the memory arm is what binds.
    above = 2 * reserve + 600
    assert _resolve(hook_module, monkeypatch, cpus=64, available_mb=above) == _resolve(
        hook_module, monkeypatch, cpus=64, available_mb=above, total_mb=None
    ), "above twice the reserve the term must be invisible"

    # And it genuinely binds below the boundary, or the test above would pass
    # for the trivial reason that the reserve does nothing anywhere. SEARCHED
    # rather than pinned at a constant availability: whether the reserve changes
    # the COUNT depends on where a ``_MB_PER_WORKER`` charge boundary happens to
    # fall between the two budgets, so a hard-coded availability would silently
    # start asserting nothing the moment that charge moved (as it did in review
    # round 1, when the reserve stopped being visible at 3,500 MB and became
    # visible at 3,600 MB instead).
    binding = [
        a
        for a in range(2 * reserve - 1, reserve, -1)
        if _resolve(hook_module, monkeypatch, cpus=64, available_mb=a)
        < _resolve(hook_module, monkeypatch, cpus=64, available_mb=a, total_mb=None)
    ]
    assert binding, f"below twice the reserve ({2 * reserve} MB) the term must bind somewhere"
    assert max(binding) < 2 * reserve


def test_reserve_scales_down_on_a_small_runner(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flat 2 GB reserve would take most of a 4 GB CI container.

    ``total // 8`` is 512 MB there, so a 2-vCPU/4 GB runner is unaffected. With
    a flat 2,048 MB reserve the budget would go to 0 and the runner would be
    pinned at the floor for no reason.
    """
    small = _resolve(
        hook_module, monkeypatch, cpus=2, available_mb=3000, total_mb=4096, env={"CI": "1"}
    )
    assert small == 2
    # The clearest case for scaling: 3,600 MB free on a 4 GB runner. The scaled
    # 512 MB reserve leaves the share arm binding at 3 workers (1800 // 600). A
    # flat 2,048 MB reserve would leave 1,552 MB and 2 workers, i.e. it would
    # cost a worker on the smallest runner in the fleet for no benefit there.
    # Reverting the scaling returns 2.
    assert (
        _resolve(
            hook_module, monkeypatch, cpus=8, available_mb=3600, total_mb=4096, env={"CI": "1"}
        )
        == 3
    )


def test_the_reserve_fraction_keeps_the_small_host_floor(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fraction is 1/8. 1/18 cuts the floor on the hosts that need it most.

    WHY this is separate from the scaling test above: on a 4 GB host the two
    fractions differ by 285 MB, which is under one 600 MB charge, so NO
    assertion on a 4 GB host can tell them apart. The difference only becomes
    visible where the reserve is large and the host is not - a 16 GB box:
    1/8 holds 2,048 MB, 1/18 only 910 MB.

    WHY 3,600 MB and not 3,500 MB: this assertion has to be one a fraction cut
    CANNOT satisfy, and 3,500 MB was not. There ``min(1,750, 3,500 - 2,048) =
    1,452`` and ``min(1,750, 3,500 - 910) = 1,750`` both floor to 2 workers
    under a 600 MB charge, so a reverted fraction left the behavioural
    assertion passing and only the constant pin below failing. 3,600 MB is the
    first availability on a 16 GB host where the two diverge: 1/8 leaves 1,552
    MB (2 workers), 1/18 leaves 1,800 MB (3). That is one worker, and the
    earlier version of this docstring claimed "2 workers against 4" - which
    needed the withdrawn 400 MB charge and was wrong under it too (that charge
    gives 3 against 4: it moves the level of both arms, not their difference).
    The cost of the cut is unchanged and is the real reason to refuse it: 1/18
    promises the machine 1,138 MB less breathing room on exactly the hosts with
    the least headroom, while leaving this laptop's own 2,048 MB cap intact.
    """
    # Behaviour first, at an availability where a fraction cut changes the
    # count (1/18 resolves 3 here). The constant below is a second check naming
    # what was cut, not the only thing standing in for it.
    assert _resolve(hook_module, monkeypatch, cpus=8, available_mb=3600, total_mb=16384) == 2
    assert hook_module._MEMORY_RESERVE_FRACTION == 8


def test_unmeasurable_total_memory_degrades_to_the_old_budget(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No reserve is applied when the host size cannot be read.

    Degrading to the pre-existing share-only budget is deliberate: a guessed
    reserve on an unknown host could pin an unrelated machine to the floor.
    """
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=6000, total_mb=None) == 5


# ---------------------------------------------------------------------------
# Contracts the change must not break
# ---------------------------------------------------------------------------


def test_explicit_override_wins_unclamped(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PYTEST_XDIST_AUTO_NUM_WORKERS`` is honoured above ``_MAX_WORKERS``.

    Someone who types a number has a reason; clamping it would make the
    documented escape hatch useless. Pinned here because this hook displaces
    xdist's own provider, which is where that variable would otherwise be read.
    """
    workers = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=1000,
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "12"},
    )
    assert workers == 12, "the override must beat both the clamp and the memory floor"


def test_the_override_is_reported_as_what_bound_the_count(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The override path reports too: this hook DID choose, it just was not free to.

    ``-n0`` and an explicit ``-n`` are silent honestly - the hook never runs and
    has nothing to say. This path is the opposite: it ran, honoured the
    operator's number and returned before the calculation, so a silent run is
    indistinguishable from a serialised one and the count is the one a reader is
    least able to explain (12 workers on a machine that resolves 4). The line
    names the override as the term that bound - and only that term, because the
    arms, the reserve and the fleet had no say in this number and printing them
    would imply they did.
    """
    workers = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=1000,
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "12"},
    )
    err = capsys.readouterr().err
    assert workers == 12
    assert err.count("pytest worker cap:") == 1, "one line per run, on every path that chooses"
    assert "pytest worker cap: 12 (bound by PYTEST_XDIST_AUTO_NUM_WORKERS, unclamped)" in err
    assert "cpu arm" not in err, "the arms did not decide this count and must not be implied"


def test_hook_never_raises_when_a_probe_throws(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A throwing probe degrades to ``_FALLBACK_WORKERS`` instead of failing collection.

    Both probes are covered: the reserve added a second one, and a new
    unguarded call site would be a new way to break every ``pytest`` run.
    """

    def boom() -> int:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(hook_module.os, "cpu_count", lambda: 14)
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)

    monkeypatch.setattr(hook_module, "_available_memory_mb", boom)
    monkeypatch.setattr(hook_module, "_total_memory_mb", lambda: 36864)
    resolved = hook_module.pytest_xdist_auto_num_workers(None)  # type: ignore[arg-type]
    assert resolved == hook_module._FALLBACK_WORKERS

    monkeypatch.setattr(hook_module, "_available_memory_mb", lambda: 6000)
    monkeypatch.setattr(hook_module, "_total_memory_mb", boom)
    resolved = hook_module.pytest_xdist_auto_num_workers(None)  # type: ignore[arg-type]
    assert resolved == hook_module._FALLBACK_WORKERS


def test_total_memory_probe_reads_this_host(hook_module: types.ModuleType) -> None:
    """The reserve is inert if the probe returns ``None`` everywhere it runs.

    Unpinned on purpose - it is the one assertion that must see the real
    machine, since a probe that silently fails on the CI platform would disable
    the reserve there without any other test noticing.
    """
    total = hook_module._total_memory_mb()
    assert total is not None and total > 0


# ---------------------------------------------------------------------------
# The one-line decision report
# ---------------------------------------------------------------------------
#
# The resolved count is a product of four inputs the process never shows anyone,
# and the cost of that opacity is on the record: AGENTS.md describes a session
# bisecting a parallelism-sensitive failure without knowing what parallelism the
# run actually had, and "3 workers on a 14-core machine" is unactionable to a
# reader who cannot see which term produced it. The line pinned below names every
# input.
#
# TWO THINGS ARE PINNED IN THIS SECTION, and the second is the load-bearing one:
#
#   1. one line, on stderr, naming the cap and every term that decided it;
#   2. the FLEET SIZE IS REPORTED AND NEVER DIVIDED BY. A fleet divisor was
#      implemented and measured on 2026-09-13 (conftest.py's module docstring
#      records it): on this host it resolves every suite to the 2-worker floor,
#      where it cost **45% more wall time for the same work** than the 3 workers
#      the constants gave at the time (4 now) - 3 concurrent instances of
#      tests/unit/tui/test_slash_echo.py at real fleet depth measured 338.4 /
#      314.5 / 308.1 s at 2 workers against 219.8 / 216.3 / 209.8 s at 3, with
#      sum CPU 105.7 s vs 103.1 s. If a future change starts using this count to
#      shrink the cap, `test_the_fleet_size_is_reported_and_never_divided_by`
#      fails - which is the whole point of keeping it.

#: The memory level the report's assertions are anchored on: 5,000 MB free on a
#: 36 GB host, i.e. this machine's measured *chronic* availability during the
#: 2026-09-13/14 titration (waves saw 4,522-5,555 MB). With the shipped
#: constants - ``min(2,500, 5,000 - 2,048)`` = 2,500 MB of budget, over 600 MB
#: per worker - that resolves 4 workers against a 7-worker CPU arm, so the memory
#: arm is the one that decides. The raise over the released constants is one
#: worker at 4,964-5,313 MB free and two at 4,848 MB, from the reserve cap alone;
#: the A/B behind it measured cap 6 against cap 3 and only its DIRECTION
#: reproduced (15-16% on the independent pass, against a different baseline; the
#: first pass's cap-6-over-cap-3 speedup is withdrawn as a claim, not its CPU
#: deltas), so quote the direction and never a headline speedup figure. See
#: ``conftest.py``'s module docstring for the full table and for why the count is
#: 4 and not the 6 the earlier revision bought.
_CHRONIC_PRESSURE_MB = 5000
#: Below the reserve, where ``_MIN_WORKERS`` is the term that decides.
_FLOOR_BINDING_MB = 1000


def test_the_worker_cap_is_reported_once_with_every_term(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One line, on stderr, naming the cap and every input behind it.

    stderr, not stdout: ``-q`` output is parsed from stdout and must not gain a
    line. Reverting the report makes the ``in captured.err`` assertions fail;
    printing to stdout instead fails the ``captured.out == ""`` assertion.
    """
    workers = _resolve(
        hook_module, monkeypatch, cpus=14, available_mb=_CHRONIC_PRESSURE_MB, siblings=11
    )
    captured = capsys.readouterr()
    assert workers == 4, "14 cores -> cpu arm 7; min(2,500, 2,952) MB // 600 -> 4 workers"
    assert captured.out == "", "the report must never touch stdout: -q output is parsed from it"
    assert captured.err.count("pytest worker cap:") == 1
    assert "pytest worker cap: 4" in captured.err
    assert "bound by memory" in captured.err
    assert "cpu arm 7" in captured.err
    assert "memory arm 4" in captured.err
    assert "available 5,000 MB" in captured.err
    assert "reserve 2,048 MB" in captured.err
    assert "siblings 11" in captured.err


@pytest.mark.parametrize(
    ("cpus", "available_mb", "expected_arm"),
    [
        (14, 24000, "bound by cpu"),  # 7 against a 30-worker memory arm
        (14, 5000, "bound by memory"),  # 7 against 6
        (14, 1000, "bound by the 2-worker floor"),  # both arms below the floor
        (2, 4096, "bound by the 2-worker floor"),  # a 2-vCPU runner under pressure
    ],
)
def test_the_report_names_the_binding_arm(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cpus: int,
    available_mb: int,
    expected_arm: str,
) -> None:
    """The line has to say WHICH term decided, or it is just another number.

    This is the field that makes the diagnosis instant: "3 workers" is
    unactionable, "bound by memory, cpu arm 7, memory arm 3, available 5,140 MB,
    reserve 2,048 MB" points at the constants to move. A floor or a clamp can
    mask both arms, which is why they are named separately rather than inferred.
    """
    _resolve(hook_module, monkeypatch, cpus=cpus, available_mb=available_mb, siblings=0)
    assert expected_arm in capsys.readouterr().err


def test_the_report_names_the_clamp_that_lifted_the_arm(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On CI every core is taken, and the line says the 8-worker cap is what bound.

    The same assertion also pins that a CI run is unaffected by the fleet it can
    see: with 11 siblings reported, the count is still every core.
    """
    workers = _resolve(
        hook_module, monkeypatch, cpus=14, available_mb=_AMPLE_MB, env={"CI": "1"}, siblings=11
    )
    err = capsys.readouterr().err
    assert workers == 8
    assert "bound by the 8-worker cap" in err
    assert "cpu arm 14" in err
    assert "siblings 11" in err


def test_the_fleet_size_is_reported_and_never_divided_by(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The count in the line must not change the count of workers. THE guard.

    The refutation of the fleet divisor, executable: 11 sibling suites resolve
    exactly the same worker count as none, and only the reported field moves.
    Dividing by that field is the change that was measured and rejected, so this
    test is what stops it coming back without the argument being made again.
    """
    solo = _resolve(
        hook_module, monkeypatch, cpus=14, available_mb=_CHRONIC_PRESSURE_MB, siblings=0
    )
    solo_err = capsys.readouterr().err
    crowded = _resolve(
        hook_module, monkeypatch, cpus=14, available_mb=_CHRONIC_PRESSURE_MB, siblings=11
    )
    crowded_err = capsys.readouterr().err
    assert solo == crowded == 4, "the fleet size must not move the worker count"
    assert "siblings 0" in solo_err
    assert "siblings 11" in crowded_err


def test_the_worker_cap_report_can_be_silenced(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``PYTEST_QUIET_WORKER_CAP=1`` leaves the count alone and prints nothing.

    Set through ``_resolve``'s ``env`` rather than by hand: the fixture scrubs
    this variable with the rest (see its docstring), so an ambient value cannot
    reach the hook and only an explicit one can.
    """
    resolved = _resolve(
        hook_module,
        monkeypatch,
        cpus=14,
        available_mb=_CHRONIC_PRESSURE_MB,
        siblings=3,
        env={hook_module._QUIET_ENV: "1"},
    )
    assert resolved == 4
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_the_quiet_flag_is_read_as_a_flag_not_as_set_at_all(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    """``PYTEST_QUIET_WORKER_CAP=0`` leaves the line ON.

    The documented contract is that ``=1`` silences. An any-non-empty test made
    every other value silence it as well, including the one a reader is most
    likely to reach for when they want the report BACK - so the escape hatch
    read as broken in exactly the direction that is hardest to notice (the run
    looks normal, it just says nothing). A flag is read as a flag.

    Passed through ``_resolve``'s ``env`` for the same reason as the test above:
    the fixture owns this variable so that a developer's exported value cannot
    decide what the module asserts.
    """
    assert (
        _resolve(
            hook_module,
            monkeypatch,
            cpus=14,
            available_mb=_CHRONIC_PRESSURE_MB,
            env={hook_module._QUIET_ENV: value},
        )
        == 4
    )
    assert "pytest worker cap: 4" in capsys.readouterr().err


def test_a_broken_report_cannot_decide_the_worker_count(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing stream must not change the answer or the exit status.

    The report is called after the count is computed, and a raise here would
    otherwise be caught by the hook's outer ``except`` and turn a good resolution
    into ``_FALLBACK_WORKERS`` - a diagnostic deciding a worker count. ``print``
    is injected into the hook module's own globals, which is also how the hook
    resolves it.
    """

    def broken_print(*args, **kwargs):
        raise BrokenPipeError("stderr closed")

    monkeypatch.setattr(hook_module, "print", broken_print, raising=False)
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=_CHRONIC_PRESSURE_MB) == 4


# ---------------------------------------------------------------------------
# The sibling probe (reporting only)
# ---------------------------------------------------------------------------
#
# One read-only `ps`, one pass over its output, no lock file and no shared state
# - a probe that cannot block and cannot wedge another session. The command
# lines below are copied from `ps -A -w -w -o pid=,ppid=,command=` on the host
# this hook was tuned on (2026-09-13), so the parser is tested against the
# shapes that actually occur rather than shapes invented for the test.

_PS_DUMP = """\
    1     0 /sbin/launchd
  500   400 /Users/d/.venv/bin/python -m pytest tests/unit -q
  501   500 /Users/d/.venv/bin/python -u -c import sys;exec(eval(sys.stdin.readline()))
  600   400 .venv/bin/python -m pytest tests/unit -q
  700   400 .venv/bin/python -m pytest -n0 -q -p no:randomly tests/unit/harness
  800   400 .venv/bin/python -m pytest tests/unit -n 4
  801   800 /Users/d/.venv/bin/python -u -c import sys;exec(eval(sys.stdin.readline()))
  810   400 /opt/homebrew/bin/bash -c cd ~/x && .venv/bin/python -m pytest tests/unit -q | tail -6
  811   810 .venv/bin/python -m pytest tests/unit -q
  900   800 .venv/bin/python -m pytest tests/unit/server/test_x.py -q
  950   400 env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/unit/tui -q
"""


@pytest.fixture
def fake_ps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[str, ...], dict[str, object]]]:
    """Serve ``_PS_DUMP`` to the probe, recording every call it made."""
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(args, **kwargs):
        calls.append((tuple(args), kwargs))
        return types.SimpleNamespace(stdout=_PS_DUMP)

    # `hook_module.subprocess` IS `subprocess`; patching the attribute here is
    # the same object the hook imports.
    monkeypatch.setattr("subprocess.run", run)
    return calls


def test_sibling_probe_counts_only_live_fleet_members(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fake_ps: list[tuple[tuple[str, ...], dict[str, object]]],
) -> None:
    """The count is 4, and each exclusion is a named decision.

    Counted: ``600`` (a peer whose workers have not spawned yet - see below),
    ``800`` (workers present), ``811`` (a peer launched through a ``bash -c``
    wrapper) and ``950`` (the same, through ``env``). Excluded: ``500`` (this
    process), ``700`` (``-n0``: one process, not a share of the machine),
    ``810`` (a shell wrapper, not a suite) and ``900`` (NESTED inside ``800`` -
    a pytest a test spawned, which would otherwise make every suite on the box
    look bigger than the fleet really is).
    """
    monkeypatch.setattr(hook_module.os, "getpid", lambda: 500)
    assert hook_module._count_live_sibling_suites() == 4


def test_the_sibling_probe_makes_one_read_only_ps_call(
    hook_module: types.ModuleType,
    fake_ps: list[tuple[tuple[str, ...], dict[str, object]]],
) -> None:
    """One read-only ``ps``, unlimited width, headerless - the whole contract.

    ``-w -w`` is not cosmetic: macOS truncates ``command`` to the window width
    without it, and an xdist worker's bootstrap sits ~88 characters in, so a
    truncated line reads as "a controller with no workers". And the absence of
    lock files, temp files or any other shared state is the reason this probe is
    safe to run in every session - a wedged ``flock`` holder is what
    ``harness/group_reaper.py`` documents about propagating a freeze between
    sessions, and this hook will not import that hazard.
    """
    hook_module._count_live_sibling_suites()
    assert len(fake_ps) == 1, "the probe must read the machine once, not per process"
    args, kwargs = fake_ps[0]
    assert args == ("ps", "-A", "-w", "-w", "-o", "pid=,ppid=,command=")
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 5


def test_a_peer_without_workers_yet_is_still_counted(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer 1 s into its import phase counts; a peer that said ``-n0`` does not.

    The hook runs before any worker exists, in the window where a suite started
    moments ago is a controller with nothing but its own process. Without this
    arm a burst of suites launched together would each report a fleet of one -
    the reading a person cannot correct by hand later, which is the failure this
    line exists to prevent. The split with ``-n0`` is what keeps a QA session's
    serialised three-directory run from being reported as part of the fleet.
    """
    dump = (
        "  1     0 /sbin/launchd\n"
        "  600   400 .venv/bin/python -m pytest tests/unit -q\n"
        "  700   400 .venv/bin/python -m pytest -n0 -q tests/unit/harness\n"
    )
    monkeypatch.setattr(
        hook_module.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=dump)
    )
    monkeypatch.setattr(hook_module.os, "getpid", lambda: 500)
    assert hook_module._count_live_sibling_suites() == 1


def test_every_probe_failure_is_reported_as_unknown(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Absent ``ps``, a non-zero exit and unparsable output all give ``None``.

    Not 0, and the distinction is the point: a fleet the probe could not count
    must not look like an empty machine, so the line says ``siblings unknown``.
    The worker count is unchanged either way - the probe has no say in it.
    """

    def explode(*args, **kwargs):
        raise OSError("ps is gone")

    for behaviour in (
        explode,
        lambda *a, **k: (_ for _ in ()).throw(hook_module.subprocess.CalledProcessError(1, "ps")),
        lambda *a, **k: types.SimpleNamespace(stdout="garbage that is not a ps table\n12\n"),
        lambda *a, **k: types.SimpleNamespace(stdout=""),
    ):
        monkeypatch.setattr(hook_module.subprocess, "run", behaviour)
        monkeypatch.setattr(hook_module, "_SIBLING_PROBED", False)
        monkeypatch.setattr(hook_module, "_SIBLING_SUITES", None)
        assert hook_module._count_live_sibling_suites() is None
        capsys.readouterr()
        # ... and the hook, reading that probe, resolves and reports the same
        # count it always would.
        assert (
            _resolve(
                hook_module, monkeypatch, cpus=14, available_mb=_CHRONIC_PRESSURE_MB, siblings=None
            )
            == 4
        )
        assert "siblings unknown" in capsys.readouterr().err


def test_the_sibling_probe_is_cached_for_the_process_lifetime(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probed once; later callers get the first answer.

    Cached because the hook is consulted once, microseconds before workers
    spawn, so a second probe could only disagree about a suite that is still
    starting up - and because the visibility line is only worth leaving on by
    default if reading it costs nothing.
    """
    calls: list[int] = []

    def counted() -> int:
        calls.append(1)
        return 2

    monkeypatch.setattr(hook_module, "_count_live_sibling_suites", counted)
    assert hook_module._live_sibling_suites() == 2
    assert hook_module._live_sibling_suites() == 2
    assert len(calls) == 1, "the machine must be read once per process"


def test_sibling_probe_reads_this_host(hook_module: types.ModuleType) -> None:
    """The probe must work on the machine the suite runs on.

    Unpinned on purpose, like the total-memory probe above: every other test in
    this section patches ``subprocess.run``, so a ``ps`` invocation that fails
    everywhere would leave the report saying "siblings unknown" on every run
    with the rest of the suite green. Falsifiable rather than merely
    non-crashing: an earlier revision accepted ``None`` here, which is exactly
    the permanent-probe-failure case this docstring names as the reason the test
    exists, so it could not fail on the defect it was written for.

    The ``ps`` existence check is the one legitimate skip: on a platform with no
    ``ps`` the probe is documented to degrade to "unknown", and the parser is
    covered by the synthetic cases above.
    """
    if shutil.which("ps") is None:
        pytest.skip("no ps on this platform; the probe degrades to 'unknown' by design")
    siblings = hook_module._count_live_sibling_suites()
    assert siblings is not None, "ps is installed but the probe could not read the machine"
    assert siblings >= 0


# ---------------------------------------------------------------------------
# The titration
# ---------------------------------------------------------------------------
#
# These two tests pin the RESOLUTIONS the 2026-09-13/14 A/B bought, not the
# constants themselves - a constant is an implementation detail, the count a
# suite gets is the behaviour. The measurement (conftest.py's module docstring
# carries the full table): 3 concurrent instances of
# tests/unit/tui/test_slash_echo.py at real fleet depth, per-instance wall time
# 313.3/225.2 s at cap 3 against 188.9/161.6/188.7 s at cap 6, for ~12% more
# CPU; cap 8 was 9% better again, which did not clear the 15% bar. An independent
# A/B (review round 1) reproduced the ORDERING but not the magnitude - 15-16%
# with a different baseline - so what these tests pin is the resolution, and no
# doc quotes the percentage range. Before the titration this host resolved 3 at
# its chronic availability; the tests below say it must now resolve 4 there and
# MUST NOT have overshot elsewhere.


@pytest.mark.parametrize("available_mb", [4964, 5140, 5313, 5555])
def test_the_released_constants_resolve_four_at_measured_availability(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, available_mb: int
) -> None:
    """At the four availabilities the A/B measured, the memory arm allows 4.

    ``min(0.5 * a, a - 2048) // 600``: 2,482-2,777 MB of budget, and the CPU arm
    (7) stays clear of it, so memory is still the binding term and the raise is
    bought from the reserve CAP rather than from the cores or from the per-worker
    charge. 4, NOT the 6 an earlier revision bought by halving that charge: the
    per-worker RSS measurement (see ``_MB_PER_WORKER`` in ``conftest.py``) put the
    peak worker tree at 441.5 MB, above 400, so the charge stays at 600 and the
    count follows it down.

    These rows are NOT all one-worker raises over the released constants. The
    released file (600 / 3,072 / 8) resolves 3 at 4,964 / 5,140 / 5,313 MB and 4
    at 5,555 MB, so three of them buy one worker and 5,555 buys none; the
    two-worker row, 4,848 MB, is pinned in
    `test_the_titration_did_not_overshoot_or_undershoot` below. The per-row
    baseline is 2 / 2 / 3 / 3 / 3 / 4 released against 3 / 4 / 4 / 4 / 4 / 4
    shipped at 4,522 / 4,848 / 4,964 / 5,140 / 5,313 / 5,555 MB.
    """
    workers = _resolve(hook_module, monkeypatch, cpus=14, available_mb=available_mb)
    assert workers == 4, f"at {available_mb} MB available"


def test_the_titration_did_not_overshoot_or_undershoot(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The edges: the low end of the observed range falls to 3, the top stays at 7.

    4,522 MB is the lowest `available` any A/B wave measured, and 3 is what the
    shipped constants buy there (``min(2,261, 2,474) // 600``) - the raise must
    not be tuned so hard that a *worse* machine is handed more workers than a
    better one, so the count has to fall with availability (4 at 4,848-5,313 MB,
    3 here) instead of sitting flat across the band. The released constants gave
    **2** here, so this row is a raise too, not a return to the old count. At the
    other end, 8,498 MB and 24,000 MB must both resolve the CPU arm's 7 (not 8):
    the change moved a memory constant only, and a suite on a machine with memory
    to spare must not get a wider run than the CPU share allows.
    """
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=4522) == 3
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=8498) == 7
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=24000) == 7


@pytest.mark.parametrize(
    ("cpus", "available_mb", "total_mb", "expected"),
    [
        (2, 3000, 4096, 2),  # 2 vCPU / 4 GB runner
        (2, 7000, 8192, 2),  # 2 vCPU / 8 GB
        (4, 15000, 16384, 4),  # 4 vCPU / 16 GB
        (8, 31000, 32768, 8),  # 8 vCPU / 32 GB
    ],
)
def test_the_ci_runner_shapes_are_unchanged_by_the_titration(
    hook_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    cpus: int,
    available_mb: int,
    total_mb: int,
    expected: int,
) -> None:
    """Every documented CI runner shape resolves exactly what it did before.

    These four rows are the trace in `_MEMORY_RESERVE_CAP_MB`'s comment. The
    reserve's CAP moved (3,072 -> 2,048 MB; the 1/8 fraction is unchanged) and
    the per-worker charge did NOT, so this is the test that says the change was a
    hosted-runner no-op - a reserve cap that also widened CI would be paid on
    every PR.

    Note what these rows CANNOT show, and why the fraction is pinned elsewhere:
    all four are CPU-bound, so their worker count is set by the CPU arm under
    either constant set and the reserve's net effect on them is zero by
    construction. A fraction cut therefore passes this test untouched, which is
    how 1/18 reached review in the first place -
    `test_the_reserve_fraction_keeps_the_small_host_floor` is the assertion that
    can see it.
    """
    resolved = _resolve(
        hook_module,
        monkeypatch,
        cpus=cpus,
        available_mb=available_mb,
        total_mb=total_mb,
        env={"CI": "1"},
    )
    assert resolved == expected, f"{cpus} vCPU / {total_mb} MB runner"

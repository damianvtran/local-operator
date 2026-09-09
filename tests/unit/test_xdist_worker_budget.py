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

Every assertion below was mutation-tested against the defect it claims to
catch: the fix was reverted in the working tree, the test was shown to FAIL,
and the fix restored. That is not ceremony. The design review of this change
caught a proposed regression test that returned the same value with the fix
applied and reverted, because an unrelated clamp masked the difference - a
guard that cannot fail on the buggy code teaches every future reader that a
regression is pinned when it is not.

Two consequences of that for how these tests are written:

- **``os.cpu_count`` and both memory probes are pinned**, never read from the
  host. A test whose expected value depends on the developer's core count and
  current free memory cannot assert an absolute, and this suite runs on
  everything from a 2-vCPU runner to a 14-core laptop.
- **Several tests assert on a DIFFERENCE between two environments** rather than
  on one number, because the ``2..8`` clamp can flatten both arms of a branch
  onto the same value and hide the very behaviour under test.
"""

from __future__ import annotations

import importlib.util
import pathlib
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
) -> int:
    """Run the hook against a fully synthetic machine.

    Every input the hook reads is pinned here - core count, both memory probes,
    and the three environment variables - so the result is a pure function of
    the arguments and identical on a 2-vCPU runner and a 14-core laptop.
    """
    monkeypatch.setattr(module.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(module, "_available_memory_mb", lambda: available_mb)
    monkeypatch.setattr(module, "_total_memory_mb", lambda: total_mb)
    for name in ("CI", "LOCAL_OPERATOR_AGENT_SHELL", "PYTEST_XDIST_AUTO_NUM_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return module.pytest_xdist_auto_num_workers(None)  # type: ignore[arg-type]


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

    6,000 MB available on a 36 GB host: the share alone would allow
    ``3000 // 600 = 5`` workers; holding back the 3,072 MB reserve allows
    ``2928 // 600 = 4``. Reverting the reserve returns 5.
    """
    workers = _resolve(hook_module, monkeypatch, cpus=14, available_mb=6000)
    assert workers == 4


def test_reserve_cannot_drive_the_result_below_min_workers(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with less free memory than the reserve still runs, at the floor.

    ``available - reserve`` is negative across this whole range, so the budget
    must clamp to 0 and the result to ``_MIN_WORKERS`` - never 0, never
    negative, and never an exception. A suite that refuses to start because the
    machine is busy would be a far worse failure than a slow one.
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
    memory and costs ~2.6 workers even with no sibling suites at all: at
    8,000 MB free it returns 4 where today returns 6, and at 6,000 MB it drops
    5 workers to 2. Both assertions below fail under that shape.
    """
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=8000) == 6
    assert _resolve(hook_module, monkeypatch, cpus=14, available_mb=12000) == 7


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
    # for the trivial reason that the reserve does nothing anywhere.
    below = 2 * reserve - 1200
    assert _resolve(hook_module, monkeypatch, cpus=64, available_mb=below) < _resolve(
        hook_module, monkeypatch, cpus=64, available_mb=below, total_mb=None
    ), "below twice the reserve the term must bind"


def test_reserve_scales_down_on_a_small_runner(
    hook_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flat 3 GB reserve would take three quarters of a 4 GB CI container.

    ``total // 8`` is 512 MB there, so a 2-vCPU/4 GB runner is unaffected. With
    a flat 3,072 MB reserve the budget would go to 0 and the runner would be
    pinned at the floor for no reason.
    """
    small = _resolve(
        hook_module, monkeypatch, cpus=2, available_mb=3000, total_mb=4096, env={"CI": "1"}
    )
    assert small == 2
    # The clearest case for scaling: 3,600 MB free on a 4 GB runner. The scaled
    # 512 MB reserve leaves the share arm binding at 3 workers. A flat 3,072 MB
    # reserve would leave 528 MB, resolve to 0, and pin this runner at the
    # floor for no reason. Reverting the scaling returns 2.
    assert (
        _resolve(
            hook_module, monkeypatch, cpus=8, available_mb=3600, total_mb=4096, env={"CI": "1"}
        )
        == 3
    )


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

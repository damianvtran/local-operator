"""The second staleness question: which BUILD is a running daemon on?

WHY THIS EXISTS. ``launchd.rewrite_if_stale`` answers the first question — "does
the plist on disk say what this build would render?" — and under the generation
layout that answer is ``current`` for a daemon that is generations behind, because
every unit names the STABLE SHIM (``~/.local/share/lop/bin/python3``) and a shim
does not change when the pointer moves. Measured live on the operator's machine
2026-09-24: four byte-identical plists, two daemons on the current generation and
two two-and-three generations behind. The build a process actually runs is
therefore only visible in that process, and on macOS it is in its argv (the shim
``exec``s the generation's own image, so ``ps -o args=`` names the generation).

WHAT IS PINNED HERE, and why each one is a decision rather than a formality:

- **the generation is read from a REAL live process**, not only from a string —
  the end-to-end test spawns one and reads it through the shipped predicate, so a
  probe that never actually calls ``ps`` cannot pass;
- **an unreadable probe means no move**, in every shape that can produce one: no
  ``ps`` at all, a pid that is gone, an argv that names no generation, a pointer
  that is absent, dangling or being renamed as we read it. The direction is chosen
  (see ``update.stale_generation_of_process``): a missed reload leaves a daemon
  where it is, while a reload reasoned from a failed probe interrupts a working one;
- **a generation reached through a symlinked home still matches**, which is why
  the ancestors are resolved and the file itself is not;
- **only a generation OTHER than ``current`` is stale**, compared by NAME, because
  the two sides are spelled differently by construction (the shim's ``pwd -P``
  against ``~``-derived ``stable_root()``).

The repair that consumes this — one ``launchctl print``, one probe, at most one
``kickstart`` — is pinned in ``tests/unit/test_daemon_plist_refresh.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import update as update_mod

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason=(
        "the probe reads `ps` and a symlink layout; off POSIX it answers None, which "
        "`test_no_ps_at_all_is_an_unreadable_probe` covers as the deliberate answer"
    ),
)

#: A pid nothing can be running under (macOS caps pids at ``kern.maxproc``, well
#: below this, and Linux's default ``pid_max`` likewise). Fixed rather than a
#: spawned-and-reaped child, because a reaped pid can be REUSED and this test is
#: about the answer for a pid that does not exist.
_UNUSED_PID = 1 << 30

#: The four supervised daemons' real argv SHAPES, captured with
#: ``ps -o pid=,ppid=,args=`` on the operator's machine 2026-09-24 (only the home
#: and the generation id are elided here; the rest is verbatim). Every one of them
#: begins with the generation's own image, which is what the probe reads.
_LIVE_ARGV: dict[str, tuple[str, ...]] = {
    "tunnel": ("-m", "local_operator.tunnels.service"),
    "mobile": ("-m", "local_operator.mobile.service", "--port", "4098"),
    "wakes supervisor": ("-m", "local_operator.wakes.supervisor"),
    "browser bridge": ("-m", "local_operator.browser_bridge.daemon", "--port", "4099"),
}


def _layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> Path:
    """A generation layout under the test's own stable root; returns the root.

    ``_STABLE_ROOT`` is patched rather than ``generations_dir`` itself, so the
    spelling the probe compares against is produced by the real accessor — the
    thing a symlinked home or a moved root would change.
    """
    root = tmp_path / "lop"
    (root / "generations").mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / "generations" / name).mkdir(exist_ok=True)
    monkeypatch.setattr(update_mod, "_STABLE_ROOT", str(root))
    return root


def _argv(root: Path, generation: str, *args: str) -> str:
    """One daemon's argv, spelled the way the shim's ``exec`` produces it."""
    image = root / "generations" / generation / "tools" / "local-operator" / "bin"
    return " ".join([str(image / "Local Operator"), *args])


@pytest.mark.parametrize("name", list(_LIVE_ARGV))
def test_the_generation_is_read_from_the_first_field_of_a_live_argv(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The string half, pinned against the real lines rather than invented ones."""
    root = _layout(tmp_path, monkeypatch, "20260922T082114Z-aba13b8246fb")
    generation = "20260922T082114Z-aba13b8246fb"

    assert update_mod.generation_in_argv(_argv(root, generation, *_LIVE_ARGV[name])) == (
        root / "generations" / generation
    )


def test_a_real_process_reports_the_generation_it_was_executed_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """END TO END THROUGH THE REAL ``ps``: a probe that never calls it cannot pass.

    The child is ``sys.executable`` executed with a generation path as ``argv[0]``,
    which is exactly the shape the shim produces (``exec "$gen/…/bin/Local
    Operator" "$@"``), so this reads a live process the way the repair does on a
    real machine. The generation directory deliberately does not exist: the argv is
    a claim about what was EXECUTED, and a pruned generation is the incident this
    whole change came from.
    """
    root = _layout(tmp_path, monkeypatch)
    generation = "20260924T103058Z-509c7450dbf6"
    argv0 = str(root / "generations" / generation / "tools" / "local-operator" / "bin" / "Local")

    child = subprocess.Popen(  # noqa: S603 — a fixed argv, and this test's own child
        [argv0, "-c", "import time; time.sleep(30)"],
        executable=sys.executable,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert update_mod.generation_of_process(child.pid) == root / "generations" / generation
    finally:
        child.kill()
        child.wait(timeout=10)


@pytest.mark.parametrize(
    "argv",
    [
        "",
        "   ",
        "[kworker/0:1]",
        "python3",
        "python3 -m local_operator.tunnels.service",
        "/usr/bin/sleep 30",
        "/opt/venv/bin/Local Operator -m local_operator.tunnels.service",
        "~/.local/share/lop/bin/python3 Local Operator [tunnel] -m local_operator.tunnels.service",
    ],
)
def test_an_argv_that_names_no_generation_reads_as_no_answer(
    argv: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pip venv, a bare interpreter, a shell, a relative path: all ``None``.

    The last row is the one that matters most and is worth naming: the SHIM itself
    appears as argv[0] for a process that has not re-exec'd through the generation
    yet (or a job whose branded image could not be planted), and it is not a
    generation — reading it as one would name the directory that HOLDS the
    generations.
    """
    _layout(tmp_path, monkeypatch, "20260101T000000Z-old")

    assert update_mod.generation_in_argv(argv) is None


def test_a_sibling_directory_that_merely_starts_the_same_is_not_a_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefix matching by NAME, not by string: ``generations-old/`` is not ours.

    The generations directory is the claim (``_is_generation_install``'s rule, and
    why this walks the path's ancestors instead of testing ``startswith``).
    """
    root = _layout(tmp_path, monkeypatch, "20260101T000000Z-old")
    sibling = root / "generations-old" / "20260101T000000Z-old" / "tools" / "bin" / "Local"

    assert update_mod.generation_in_argv(str(sibling)) is None
    assert update_mod.generation_in_argv(str(root / "generations")) is None


def test_a_generation_reached_through_a_symlinked_home_still_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the ancestors are resolved: the shim runs ``pwd -P`` and ``~`` does not.

    A home reached through a symlink makes argv carry the PHYSICAL path while
    ``generations_dir()`` is spelled from ``~``. A lexical comparison would answer
    ``None`` there — silently leaving the daemon this repair exists to move exactly
    where it is — so this pins the resolve, and pins that it is applied to the
    ancestors rather than to the file (a generation's ``bin/python3`` is a symlink
    OUT to the uv-managed interpreter; resolving THAT would leave the generation).
    """
    physical = tmp_path / "physical" / "lop"
    (physical / "generations" / "20260101T000000Z-old").mkdir(parents=True)
    logical = tmp_path / "logical"
    logical.symlink_to(tmp_path / "physical")
    monkeypatch.setattr(update_mod, "_STABLE_ROOT", str(logical / "lop"))

    # argv as the shim would see it: resolved physically by `pwd -P`.
    assert (
        update_mod.generation_in_argv(
            _argv(physical, "20260101T000000Z-old", "-m", "local_operator.tunnels.service")
        )
        == physical / "generations" / "20260101T000000Z-old"
    )
    # ...and the symlinked interpreter is NOT resolved out of the generation.
    image = physical / "generations" / "20260101T000000Z-old" / "tools" / "bin" / "python3"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.symlink_to(sys.executable)
    assert (
        update_mod.generation_in_argv(f"{image} -m local_operator.mobile.service --port 4098")
        == physical / "generations" / "20260101T000000Z-old"
    )


def test_no_ps_at_all_is_an_unreadable_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without ``ps`` (a non-POSIX branch) answers ``None``, never raises.

    Injected at ``subprocess.run`` rather than by hiding ``ps`` from ``PATH``: both
    produce the same ``OSError``, and this one cannot be defeated by a ``ps`` the
    test host happens to have somewhere else on the path.
    """

    def no_ps(*args: object, **kwargs: object):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(subprocess, "run", no_ps)

    assert update_mod.generation_of_process(os.getpid()) is None


def test_a_ps_that_does_not_answer_inside_the_bound_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged ``ps`` must not spend the refresh child's own bound on one daemon."""

    def hangs(*args: object, **kwargs: object):
        raise subprocess.TimeoutExpired(cmd="ps", timeout=update_mod._PS_PROBE_TIMEOUT_S)

    monkeypatch.setattr(subprocess, "run", hangs)

    assert update_mod.generation_of_process(os.getpid()) is None


def test_a_pid_that_is_gone_reads_as_no_answer() -> None:
    """Through the real ``ps``: a non-zero exit is "no such process"."""
    assert update_mod.generation_of_process(_UNUSED_PID) is None


@pytest.mark.parametrize("pid", [0, -1])
def test_a_pid_that_is_not_a_process_is_never_probed(pid: int) -> None:
    """``ps -p 0`` means "this process's group" and ``-1`` every process there is.

    Refused before the call rather than after: either would otherwise answer with
    SOMEBODY's argv, which is the one way this probe could name the wrong build.
    """
    assert update_mod.generation_of_process(pid) is None


def test_only_a_generation_other_than_current_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison itself: current is silent, one generation behind is named."""
    root = _layout(tmp_path, monkeypatch, "20260101T000000Z-old", "20260202T000000Z-new")
    (root / "current").symlink_to(root / "generations" / "20260202T000000Z-new")

    def argv_for(*, on_current: bool) -> str:
        generation = "20260202T000000Z-new" if on_current else "20260101T000000Z-old"
        return _argv(root, generation, "-m", "local_operator.tunnels.service")

    reads = {"on_current": True}
    monkeypatch.setattr(update_mod, "_process_argv", lambda pid: argv_for(**reads))

    # The control case, and the one an over-eager repair would break: a daemon on
    # the current build is left alone, and so is a daemon on it in the OTHER
    # spelling (`current` resolves through the symlink, argv through the shim).
    assert update_mod.stale_generation_of_process(1234) is None

    reads["on_current"] = False
    assert update_mod.stale_generation_of_process(1234) == (
        root / "generations" / "20260101T000000Z-old"
    )


def test_a_daemon_whose_generation_was_pruned_is_still_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident's own shape: the tree the daemon runs is GONE.

    A pruned generation is absent from disk, and the answer must still be its name —
    otherwise the daemon that is importing from a deleted tree would be the one case
    this cannot see.
    """
    root = _layout(tmp_path, monkeypatch, "20260202T000000Z-new")
    (root / "current").symlink_to(root / "generations" / "20260202T000000Z-new")
    monkeypatch.setattr(
        update_mod,
        "_process_argv",
        lambda pid: _argv(root, "20260921T125352Z-0.61.12", "-m", "local_operator.tunnels.service"),
    )

    assert update_mod.stale_generation_of_process(1234) == (
        root / "generations" / "20260921T125352Z-0.61.12"
    )


def test_no_pointer_and_a_dangling_pointer_are_both_no_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two shapes of "there is no current build to be behind", both silent.

    ABSENT is a machine without the layout (a pip install, a pre-migration host) —
    there is no question to ask. DANGLING is the pointer mid-rename, which
    ``current_generation`` answers ``None`` for (documented: macOS raises ``EINVAL``
    from ``readlink`` while the symlink is being replaced). Neither is evidence of a
    move.
    """
    root = _layout(tmp_path, monkeypatch, "20260101T000000Z-old")
    monkeypatch.setattr(
        update_mod, "_process_argv", lambda pid: _argv(root, "20260101T000000Z-old", "-m", "x")
    )

    assert update_mod.stale_generation_of_process(1234) is None

    (root / "current").symlink_to(root / "generations" / "20260101T000000Z-gone")
    assert update_mod.stale_generation_of_process(1234) is None

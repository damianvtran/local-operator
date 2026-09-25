"""The wake DTOs must not drag the wake scheduler onto the boot path.

WHY THIS MODULE EXISTS. ``harness.types`` is imported by essentially every
entry point, and it needs ``WakeSchedule`` to annotate ``WakeSchedulerProtocol``.
While that type was DEFINED in :mod:`local_operator.harness.wake`, importing the
shared types module pulled the whole live scheduler layer with it — measured at
**203.3 ms cumulative, 40.1 ms of it pydantic constructing model classes** — so
``lop --version`` spent 0.402 s inside ``build_cli_parser`` with
``harness/wake.py`` as the largest single term, for a command that schedules
nothing. The two DTOs now live in :mod:`local_operator.harness.wake_types`, a
leaf module that costs only pydantic.

The property is asserted STRUCTURALLY (what is in ``sys.modules``, and whether
two import paths give back the same class object) rather than as a timing bound,
for the reason AGENTS.md's "Prefer a structural invariant to a numeric one"
gives: a fact about which modules loaded and which class object you hold is
identical on an idle laptop and a wedged CI runner, where a millisecond ceiling
calibrated on a developer machine has flaked this repo within a day.

The second test is the one that would catch the real hazard. A "split" that
leaves a second ``class WakeSchedule`` behind under the same name is not a
refactor but a silent correctness bug: ``isinstance`` against one import path
would be False for an object built through the other, and nothing would raise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Repo root, so the fresh interpreter below can import the package without
#: inheriting this session's ``sys.path`` (and therefore its loaded modules).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Run in a FRESH interpreter. Asserting on this process's own ``sys.modules``
#: would pass vacuously: the test session imports hundreds of modules, so
#: ``harness.wake`` is already resident and its presence would say nothing about
#: what importing ``harness.types`` costs.
_PROBE = """
import json, sys
import local_operator.harness.types  # noqa: F401
print(json.dumps({
    "wake": "local_operator.harness.wake" in sys.modules,
    "wake_types": "local_operator.harness.wake_types" in sys.modules,
    "modules": len(sys.modules),
}))
"""


def _probe() -> dict[str, object]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(_REPO_ROOT),
        "TERM": "xterm-256color",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_importing_harness_types_does_not_load_the_wake_scheduler() -> None:
    """The edge that made ``lop --version`` pay 400 ms.

    ``harness.wake`` holds the live scheduler (asyncio timers, recurrence math,
    the pydantic model classes). Nothing in ``harness.types`` needs any of it:
    the only wake name that file uses is the ``WakeSchedule`` DTO, and that is
    now defined in the leaf module it imports directly.
    """
    result = _probe()
    assert result["wake_types"] is True, "the DTO module must be the one imported"
    assert result["wake"] is False, (
        "importing local_operator.harness.types loaded local_operator.harness.wake; "
        "the wake DTOs must stay in the leaf module or every entry point pays the "
        "scheduler's import again (this is the ~203.3 ms / 0.402 s regression)"
    )


def test_both_import_paths_hand_back_the_same_class_object() -> None:
    """A split that leaves two classes behind is a silent ``isinstance`` bug.

    ``harness.wake`` re-exports the DTOs so every existing
    ``from local_operator.harness.wake import WakeSchedule`` keeps working.
    Re-exporting is only correct if it is the SAME object — a second definition
    would compare unequal under ``is`` and make ``isinstance`` false at whichever
    of the two call sites imported the other one, with nothing raised.
    """
    from local_operator.harness import wake, wake_types

    assert wake.WakeSchedule is wake_types.WakeSchedule
    assert wake.DueWake is wake_types.DueWake
    assert wake.MIN_WAKE_INTERVAL_MS == wake_types.MIN_WAKE_INTERVAL_MS == 60_000


def test_the_leaf_module_does_not_import_the_scheduler() -> None:
    """The direction of the edge is the whole point; assert it, not just today's state."""
    from local_operator.harness import wake_types

    source = Path(wake_types.__file__).read_text(encoding="utf-8")
    # The module's own name appears in its docstring, so strip it before asking
    # whether the SCHEDULER is imported — otherwise the substring test matches
    # ``harness.wake_types`` and the assertion can never fail.
    assert "harness.wake import" not in source.replace("harness.wake_types", ""), (
        "wake_types must not import harness.wake — that would restore the cycle "
        "the split exists to remove"
    )


@pytest.mark.parametrize("name", ["WakeSchedule", "DueWake", "MIN_WAKE_INTERVAL_MS"])
def test_reexports_are_declared(name: str) -> None:
    """``__all__`` on the leaf module documents the surface both paths share."""
    from local_operator.harness import wake_types

    assert name in wake_types.__all__

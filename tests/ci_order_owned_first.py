"""CI-only collection hook: run the ownership files FIRST in their shard.

WHY THIS EXISTS. `test_run_bounded_linux_spawn_load.py::test_a_healthy_gate_survives_spawn_churn_and_records_its_group`
is the last unattributed failure on this branch, and its only witness is CI. Four
consecutive runs cancelled before their shard reached that file, so the defect was
effectively unwitnessed: a slow neighbour trips the shard-boundary watchdog long
before the file is collected, and the shard dies with tests "in flight at cancel"
and no FAILED line. Moving these files to the FRONT of the item list guarantees the
row is exercised before any watchdog can end the shard.

INERT BY CONSTRUCTION. It reorders only, never selects, deselects or skips; it is a
no-op off Linux (the file it protects is `skipif not linux`); and with no
`test_run_bounded*` item it returns the list untouched, in order.
"""

from __future__ import annotations

import sys


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 - pytest hook signature
    if sys.platform != "linux":
        return
    owned = [item for item in items if item.path.name.startswith("test_run_bounded")]
    if not owned:
        return
    rest = [item for item in items if not item.path.name.startswith("test_run_bounded")]
    items[:] = owned + rest

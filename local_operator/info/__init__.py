"""What this install is, what is running on this machine, and what to paste.

Three modules, split the way ``local_operator/analytics/`` is split and for one
extra reason on top of that precedent:

* :mod:`~local_operator.info.model` — frozen dataclasses. Stdlib only, no I/O,
  no ``rich``, no ``textual``. That is what a future ``lop info`` and every unit
  test import, and ``tests/unit/info/test_model.py`` pins it.
* :mod:`~local_operator.info.collect` — the probes, every heavyweight import
  function-local, every field independently guarded.
* :mod:`~local_operator.info.render` — the redacted plain-text export.

The TUI renderer lives in ``tui/widgets/info_panel.py``, outside this package,
so nothing here depends on Textual.

Importing this package must stay cheap: ``collect`` reaches ``mobile.resources``,
``browser_bridge``, ``credentials``, ``agents`` and ``teams``, none of which
belong on the path every ``lop --version`` pays for.
"""

from typing import TYPE_CHECKING, Any

from local_operator.info.collect import (
    LiveState,
    build_subagent_tree,
    collect_info,
    collect_live,
    collect_sessions,
    collect_snapshot,
    session_rows,
)
from local_operator.info.model import (
    UNKNOWN,
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)

if TYPE_CHECKING:
    from local_operator.info.render import build_export, relativise_home


def __getattr__(name: str) -> Any:
    # Importing a DTO also executes this package. The localized renderer reaches
    # Rich through info.copy, so defer it until a caller actually requests an
    # export; preserve the public functions themselves (and their signatures).
    if name in {"build_export", "relativise_home", "render"}:
        from importlib import import_module

        render = import_module("local_operator.info.render")
        return render if name == "render" else getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UNKNOWN",
    "AgentsInfo",
    "EnvInfo",
    "InfoSnapshot",
    "InstallInfo",
    "LiveState",
    "ProcessInfo",
    "SessionLine",
    "SessionsInfo",
    "SubagentLine",
    "build_export",
    "build_subagent_tree",
    "collect_info",
    "collect_live",
    "collect_sessions",
    "collect_snapshot",
    "relativise_home",
    "session_rows",
]

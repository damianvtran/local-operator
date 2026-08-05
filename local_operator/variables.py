"""Session-visible variables behind the ``list_variables`` / ``read_variable``
tools.

The token budget is the whole point: variable VALUES are never written into
the system prompt. The agent discovers what is available through
``list_variables`` (names only — a few hundred tokens at most) and pulls a
single value on demand with ``read_variable``. That keeps large or secret
values out of the rolling context until the agent actually needs them, which
is what makes many small variables cheap to carry.

Sources, in precedence order:

1. ``config_values`` — a mapping (e.g. the config's ``variables`` section)
   injected at session creation. Highest precedence so user intent wins.
2. The process environment, resolved through ``os.environ`` dynamically so a
   value is never a stale copy and a secret never rides in a prompt.
3. A project-local ``.local-operator.env`` file in the working directory,
   parsed as ``KEY=VALUE`` lines. Useful for per-repo secrets/toggles without
   exporting them globally.

Names are always returned sorted; the same name resolved by several sources
reports once (first source wins). ``read_variable`` reads live at call time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class VariableStore:
    """Named, lazily-read variables for one session's tools."""

    def __init__(
        self,
        cwd: str | None = None,
        config_values: Mapping[str, str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        # ``env`` is overridable for tests; defaults to the real process env,
        # resolved lazily so secrets are read at call time, not frozen here.
        self._env = env
        self._config_values = dict(config_values or {})
        self._cwd = cwd or os.getcwd()

    # -- sources -----------------------------------------------------------
    def _project_file(self) -> dict[str, str]:
        """Parse ``.local-operator.env`` from the working directory."""
        path = Path(self._cwd) / ".local-operator.env"
        out: dict[str, str] = {}
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                # Trim optional surrounding quotes on the value.
                out[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        return out

    def _live_env(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    # -- public API --------------------------------------------------------
    def names(self) -> list[str]:
        """All known variable names, sorted, deduplicated (no values)."""
        names = set(self._config_values)
        names.update(self._live_env())
        names.update(self._project_file())
        return sorted(names)

    def get(self, name: str) -> str | None:
        """Resolve ``name`` live; None when unknown. Config > project > env."""
        if name in self._config_values:
            return str(self._config_values[name])
        project = self._project_file()
        if name in project:
            return project[name]
        return self._live_env().get(name)

    def read(self, name: str) -> str:
        """Read a variable, or raise ``KeyError`` when it does not exist."""
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

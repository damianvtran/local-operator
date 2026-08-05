"""Session-visible variables behind the ``list_variables`` / ``read_variable``
tools.

The token budget is the whole point: variable VALUES are never written into
the system prompt. The agent discovers what is available through
``list_variables`` (names only) and pulls a single value on demand with
``read_variable``. That keeps large or secret values out of the rolling
context until the agent actually needs them.

Security is a hard constraint, not an afterthought. The process environment
is a credential minefield (API keys, tokens, AWS secrets), so it is NOT
exposed wholesale to an auto-approved tool. What is visible:

1. ``config_values`` — a mapping (e.g. the config's ``variables`` section)
   injected at session creation. Highest precedence.
2. A project-local ``.local-operator.env`` file in the working directory,
   parsed as ``KEY=VALUE`` lines.
3. ONLY environment variables whose name starts with the ``LOCAL_OPERATOR_``
   opt-in prefix. Anything else in the environment is invisible to the agent.

Names matching secret patterns (contain the substring ``key``, ``token``,
``secret``, ``password``/``passwd``, ``credential``, ``auth``) are excluded
from BOTH listing and reading, regardless of source, so a teammate-supplied
project file cannot smuggle a credential past the denylist. Over-matching is
the safe direction.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

#: Environment variables only surface to the agent when opted in with this
#: prefix. Everything else in the process env stays invisible.
ENV_ALLOW_PREFIX = "LOCAL_OPERATOR_"

#: Secret-shaped names are never listed or read, whatever their source. This
#: targets credential KINDS (secret/token/password/.../api_key), not the bare
#: token "key" which is far too common in legitimate config names. The
#: matching is deliberately loose — over-matching only hides more.
_SECRET_RE = re.compile(
    r"(?i)(secret|token|password|passwd|credential|authorization|bearer|" r"api_?key|_key$|^key\b)"
)


def _is_secret(name: str) -> bool:
    return bool(_SECRET_RE.search(name))


class VariableStore:
    """Named, lazily-read, denylist-filtered variables for one session."""

    def __init__(
        self,
        cwd: str | None = None,
        config_values: Mapping[str, str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        # ``env`` is overridable for tests; defaults to the real process env,
        # resolved lazily so values are read at call time, never frozen.
        self._env = env
        self._config_values = dict(config_values or {})
        self._cwd = cwd or os.getcwd()

    # -- sources -----------------------------------------------------------
    def _project_file(self) -> dict[str, str]:
        """Parse ``.local-operator.env`` from the working directory."""
        path = Path(self._cwd) / ".local-operator.env"
        out: dict[str, str] = {}
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return out
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
        return out

    def _live_env(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _env_visible(self) -> dict[str, str]:
        """Only opted-in (``LOCAL_OPERATOR_*``), non-secret env variables."""
        return {
            k: v
            for k, v in self._live_env().items()
            if k.startswith(ENV_ALLOW_PREFIX) and not _is_secret(k)
        }

    # -- public API --------------------------------------------------------
    def names(self) -> list[str]:
        """All non-secret variable names, sorted, deduplicated (no values)."""
        names = set(self._config_values)
        names.update(self._project_file())
        names.update(self._env_visible())
        return sorted(n for n in names if not _is_secret(n))

    def get(self, name: str) -> str | None:
        """Resolve ``name`` live; None when unknown or secret-shaped.

        Precedence: config > project file > opted-in env. A secret-shaped
        name is never resolved regardless of source.
        """
        if not name or _is_secret(name):
            return None
        if name in self._config_values:
            return str(self._config_values[name])
        project = self._project_file()
        if name in project:
            return project[name]
        if name.startswith(ENV_ALLOW_PREFIX):
            return self._live_env().get(name)
        return None

    def read(self, name: str) -> str:
        """Read a variable, or raise ``KeyError`` when unknown/denied."""
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

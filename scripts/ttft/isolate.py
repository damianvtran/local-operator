"""Isolation and reaping for the TTFT bench: one throwaway root per run, no orphans.

WHY EVERY LINE HERE EXISTS
==========================
Each block in this module closes a failure that was actually observed on this
fleet, which is why none of it is optional:

* **A fresh ``HOME`` per run, not merely a fresh config dir.** ``AGENTS.md``
  ("Isolating a run") is explicit: ``LOCAL_OPERATOR_CONFIG_DIR`` relocates the
  config dir but NOT the cache — ``model/catalogue.default_cache_dir()`` derives
  its root from the home directory independently — so a run with only the config
  dir redirected reads and writes the operator's real cache while believing it is
  isolated. The model catalogue cache is exactly what a local provider resolves
  its model listing from, so this harness would otherwise measure the operator's
  warm cache and call it cold.
* **``CMUX_*`` and ``LOP_*`` stripped from this process.** A headless TUI or a
  spawned runtime that inherits ``CMUX_WORKSPACE_ID`` renames the operator's real
  cmux workspaces, and ``LOP_MOBILE_CHILD_*``/``LOP_RUNTIME_ADOPT_SESSION`` make
  a child ADOPT the operator's live session instead of the benchmark's — a cell
  that silently measures a different session, or idles out with no work at all.
  Stripped once at startup so the code under test, which spawns with
  ``dict(os.environ)``, cannot re-inherit them either.
* **A pinned ``TIKTOKEN_CACHE_DIR`` per invocation.** tiktoken downloads
  ``cl100k_base.tiktoken`` unless it finds the file at ``$TIKTOKEN_CACHE_DIR`` or
  ``<TMPDIR>/data-gym-cache``. A fresh ``TMPDIR`` per run therefore turned the
  first tokenizer use into a TLS round trip — 413 ms of ``SSLSocket.read``
  measured in a profile of a child. That is a property of the harness, not of
  local-operator, so the data is pinned to one directory per invocation.
* **Child reaping.** A leftover runtime keeps the session's lease, so the next
  run's engage finds a live owner and reports a suspiciously fast cold start; and
  a leaked child keeps ~200 MB resident on a host running ~25 concurrent
  sessions. Terminated by pid from the run's own registry, then the root removed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Inherited environment prefixes a spawned runtime or TUI must never see. Both
#: families, not just the cmux one: see the module docstring.
STRIPPED_PREFIXES = ("CMUX_", "LOP_")

#: Environment variables this harness OWNS and must restore or clear around a run.
MANAGED_ENV = (
    "HOME",
    "LOCAL_OPERATOR_CONFIG_DIR",
    "TMPDIR",
    "LOCAL_OPERATOR_DESKTOP_TOKEN",
    "TIKTOKEN_CACHE_DIR",
    "PYTHONPYCACHEPREFIX",
    "PYTHONDONTWRITEBYTECODE",
)


def strip_inherited_runtime_env() -> list[str]:
    """Remove ``CMUX_*``/``LOP_*`` from this process, returning what was removed.

    Called once, before anything is spawned. Returns the names so the run can
    record what it had to strip — a machine that exports a family this harness
    does not know about is worth seeing in the artefact rather than guessing at.
    """
    removed: list[str] = []
    for name in list(os.environ):
        if name.startswith(STRIPPED_PREFIXES):
            del os.environ[name]
            removed.append(name)
    return removed


def kill_registered_children(config_dir: Path) -> list[int]:
    """SIGTERM every runtime this run's config dir has a record for.

    Read from the run's OWN registry, so it can only ever reach children this
    run spawned — the operator's live sessions are in their own config dir and
    are not visible from here.
    """
    from local_operator.session.runtime import registry

    killed: list[int] = []
    try:
        records = list(registry.scan(config_dir))
    except Exception:  # noqa: BLE001 — a missing registry is "nothing to kill"
        return killed
    for record, _state in records:
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, 15)
                killed.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    return killed


@dataclass
class IsolatedRun:
    """One isolated root: a fresh home, config dir, workspace and temp dir.

    The provider endpoint and the local model are seeded into the config through
    ``ConfigManager`` rather than hand-written YAML: the metadata block carries
    fields the loader requires, and a hand-written file raises a ``KeyError``
    from inside a spawned child that reads like a startup regression.
    """

    root: Path
    config_dir: Path
    cwd: Path
    saved_env: dict[str, str | None] = field(default_factory=dict)

    @property
    def home(self) -> Path:
        return self.root

    def seed(self, *, hosting: str, model: str, base_url: str | None = None) -> None:
        """Write the minimum config a runtime needs, on the given provider.

        ``tool_approval_mode: auto`` is seeded because the model here never calls
        a tool, and a run that parks on an approval prompt would measure the ask
        gate instead of time-to-first-token. It matches what ``tests/e2e`` seeds
        for the same reason.
        """
        from local_operator.config import ConfigManager

        values: dict[str, Any] = {
            "hosting": hosting,
            "model_name": model,
            "tool_approval_mode": "auto",
        }
        if base_url is not None:
            # The local (user-operated OpenAI-compatible) provider reads its
            # endpoint from `providers.<id>.base_url` — see
            # `providers/local.py:resolve_base_url`. The model overrides are
            # load-bearing rather than cosmetic: without an explicit
            # context_window the local provider assumes 4096 tokens and REJECTS
            # this harness's system prompt with "prompt is too large", which is
            # a refusal, not a slow turn.
            values["providers"] = {
                hosting: {
                    "base_url": base_url,
                    "models": {
                        model: {
                            "context_window": 400_000,
                            "max_output_tokens": 8_192,
                            "reasoning": True,
                        }
                    },
                }
            }
        self.config_dir.mkdir(parents=True, exist_ok=True)
        ConfigManager(config_dir=self.config_dir).update_config(values)

    def activate(self) -> None:
        """Point this process at the root; remember what to restore."""
        for name in MANAGED_ENV:
            self.saved_env.setdefault(name, os.environ.get(name))
        os.environ["HOME"] = str(self.home)
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(self.config_dir)
        os.environ["TMPDIR"] = str(self.root)

    def restore(self) -> None:
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def teardown(self) -> None:
        """Kill this run's runtime children, restore the environment, remove the root."""
        kill_registered_children(self.config_dir)
        self.restore()
        shutil.rmtree(self.root, ignore_errors=True)


def make_run(*, prefix: str = "lop-ttft-bench-") -> IsolatedRun:
    """Create a fresh isolated root. The caller owns :meth:`IsolatedRun.teardown`.

    The prefix is DISTINCT from the older ``lop-ttft-`` that copies of the previous
    ``bench_ttft.py`` use in other worktrees, so a root left in ``/tmp`` can be
    attributed to the harness that made it — this machine runs several sessions on
    this same ticket, and an unattributable temp root is a leak nobody can chase.
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    cwd = root / "workspace"
    cwd.mkdir(parents=True, exist_ok=True)
    return IsolatedRun(root=root, config_dir=root / ".local-operator", cwd=cwd)


def pin_shared_caches(pycache: Path, tiktoken: Path) -> None:
    """Pin the invocation-wide bytecode and tokenizer caches onto this process.

    Both are per INVOCATION and shared by every run, which is the shape the
    desktop app creates: one bytecode cache under userData read by the daemon and
    every runtime child. Set explicitly rather than inherited — the operator's
    shell may already carry one pointing at the real app cache, and a benchmark
    that silently measured that would report the wrong number.
    """
    pycache.mkdir(parents=True, exist_ok=True)
    tiktoken.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a spawned measured child gets: this process's, cleaned.

    Kept as a function rather than inlined because a child built by hand (the TUI
    arm does) must carry the same strip rule as one the code under test spawns,
    and the two must not drift.

    THE STRIP IS NOT THE WHOLE RULE, and the notification gate is the other half.
    Every child here is a real session nobody is watching: the ``test`` hosting's
    only reply is "Hello from the mock provider!", a notification body is a
    snippet of the session's own last assistant line, and this repository has 17
    recorded banner attempts from drive-by rigs that let exactly that reach the
    operator's lock screen. ``local_operator.agent_shell.harness_child_env`` is the
    product's one carrier for the pair — it also waives the nested-session guard,
    which is what lets a bench drive the real CLI from an agent's shell at all —
    so the gate is applied HERE rather than at each spawn site: one mechanism,
    instead of three that can drift apart.
    """
    from local_operator.agent_shell import harness_child_env

    env = {key: value for key, value in os.environ.items() if not key.startswith(STRIPPED_PREFIXES)}
    if extra:
        env.update(extra)
    return harness_child_env(env)

"""Offline fixture setup for live benchmarks; never imports provider clients."""

import os
import subprocess
from collections.abc import MutableMapping
from pathlib import Path


def initialize_workspace(workspace: Path, *, environment: MutableMapping[str, str]) -> None:
    """Give ordinary Git commands a synthetic root before any model tools run.

    Without a repository here, Git can discover an ancestor repository and put
    unrelated workspace filenames into tool output. Clear the worker's inherited
    Git overrides too: GIT_DIR, GIT_WORK_TREE, or injected config can bypass that
    discovery boundary. This is fixture hygiene, not a filesystem sandbox; tools
    still run with the worker's normal filesystem permissions.
    """
    for name in list(environment):
        if name.startswith("GIT_"):
            del environment[name]
    workspace.mkdir(parents=True)
    git_environment = dict(environment)
    subprocess.run(
        ["git", "init", "--quiet", "--template=", "--initial-branch=benchmark", str(workspace)],
        env=git_environment,
        check=True,
    )
    # Avoid personal templates/hooks, monitor commands, ignore rules, signing,
    # and identity changing what the synthetic trial sees or executes. These
    # settings live only in this new repository, never in the user's config.
    local_defaults = {
        "core.hooksPath": str(workspace / ".git" / "hooks"),
        "core.fsmonitor": "false",
        "core.excludesFile": os.devnull,
        "commit.gpgSign": "false",
        "user.name": "Benchmark",
        "user.email": "benchmark@example.invalid",
    }
    for key, value in local_defaults.items():
        subprocess.run(
            ["git", "config", "--local", key, value],
            cwd=workspace,
            env=git_environment,
            check=True,
        )

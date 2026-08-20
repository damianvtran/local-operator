"""The launchd control commands have to work from whatever state launchd is
actually in — a plist can exist while the agent was never bootstrapped, and
`restart` failing with "Could not find service" is exactly that gap.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from local_operator.mobile import install


class FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakePlistPath:
    """A plist path that reports existing without touching the filesystem.

    ``service_action`` bootstraps only when the plist exists; pointing the
    mock at a real path couples the test to whatever happens to be in
    ~/Library/LaunchAgents, so the fake stands in for the file."""

    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return "/tmp/fake.plist"

    def __fspath__(self) -> str:
        return str(self)


def test_restart_bootstraps_when_the_agent_is_missing() -> None:
    calls: list[list[str]] = []

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        # _launchctl is called with the arguments only ("print", ...), not
        # the "launchctl" argv[0].
        if list(cmd)[0] == "print":
            # Real launchctl: "Bad request." on stdout, the service message on
            # stderr, exit 113.
            return FakeProc(  # type: ignore[return-value]
                returncode=113, stdout="Bad request.", stderr='Could not find service "x"'
            )
        return FakeProc(returncode=0)  # type: ignore[return-value]

    with (
        patch.object(install, "_launchctl", side_effect=run),
        patch.object(install, "plist_path", return_value=FakePlistPath()),
    ):
        result = install.service_action("restart")

    assert result["ok"] is True
    verbs = [c[0] for c in calls]
    # print (missing) -> bootstrap -> kickstart -k, in that order.
    assert verbs == ["print", "bootstrap", "kickstart"]


def test_restart_does_not_bootstrap_a_live_agent() -> None:
    calls: list[list[str]] = []

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if list(cmd)[0] == "print":
            return FakeProc(returncode=0, stdout="pid = 4242")  # type: ignore[return-value]
        return FakeProc(returncode=0)  # type: ignore[return-value]

    with (
        patch.object(install, "_launchctl", side_effect=run),
        patch.object(install, "plist_path", return_value=FakePlistPath()),
    ):
        result = install.service_action("restart")

    assert result["ok"] is True
    assert not any(c[1:2] == ["bootstrap"] for c in calls)

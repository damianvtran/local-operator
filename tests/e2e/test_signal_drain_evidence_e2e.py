"""Pins the timeout evidence block of the signal-receipt waiters.

WHY: ``_receipt_timeout_evidence`` only runs when a hosted cell has already
failed, so a bug in it would surface exactly when it is needed. These cells spawn
no runtime: they feed it stand-in children and a scratch directory. They are
diagnostics tests, not a claim about the cause of the flake they support.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from tests.e2e.test_signal_drain_e2e import _receipt_timeout_evidence


class _Child:
    def __init__(self, code: int | None) -> None:
        self.pid = 4242
        self._code = code

    def poll(self) -> int | None:
        return self._code


class _Hostile:
    pid = 1

    def poll(self) -> int | None:
        raise RuntimeError("poll exploded")


def _evidence(tmp_path: Path, child: object | None, config: Path | None = None) -> str:
    directory = tmp_path / "sessions" / "abc"
    directory.mkdir(parents=True)
    (directory / "runtime-run.json").write_text("{}")
    return _receipt_timeout_evidence("refusal wait", directory, child, config)  # type: ignore


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (None, "alive"),
        (0, "exited returncode=0"),
        (-signal.SIGTERM, "killed by signal SIGTERM (returncode=-15)"),
        (-200, "killed by signal 200"),
    ],
)
def test_child_state_line(tmp_path: Path, code: int | None, expected: str) -> None:
    text = _evidence(tmp_path, _Child(code))
    assert f"child pid=4242: {expected}" in text


def test_missing_child_and_config_are_named(tmp_path: Path) -> None:
    text = _evidence(tmp_path, None)
    assert "child: not supplied" in text
    assert "runtime log: config not supplied" in text
    assert "runtime-run.json" in text


def test_hostile_child_never_raises(tmp_path: Path) -> None:
    text = _evidence(tmp_path, _Hostile())
    assert "child state unavailable" in text
    assert "runtime-run.json" in text


def test_runtime_log_tail_is_reported(tmp_path: Path) -> None:
    config = tmp_path / "config"
    (config / "logs").mkdir(parents=True)
    (config / "logs" / "runtime.log").write_text("\n".join(f"line {i}" for i in range(100)))
    text = _evidence(tmp_path, _Child(None), config)
    assert "line 99" in text and "line 59" not in text

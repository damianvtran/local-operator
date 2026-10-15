"""``lop acp`` at the CLI seam: refused flags (Q3) and startup order (R3).

Patched below the dispatch, so nothing claims the test process's real stdio
and no ACP server starts: what is pinned is the ORDER ``cli.main`` runs the
acp branch in, and what it refuses before touching anything.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from local_operator import cli
from local_operator.acp_agent import main as acp_main
from local_operator.acp_agent import stdio as acp_stdio


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import local_operator.config_migrations as migrations

    seen: list[str] = []
    monkeypatch.setattr(acp_stdio, "claim_stdio", lambda: seen.append("claim") or (10, 11))
    monkeypatch.setattr(cli, "setup_cross_platform_environment", lambda: seen.append("path"))
    monkeypatch.setattr(
        migrations,
        "run_startup_migrations",
        lambda config_dir, **kwargs: seen.append(f"migrations:{kwargs.get('surface')}"),
    )

    def fake_run(**kwargs: Any) -> int:
        seen.append(f"run:{kwargs['stdio']}")
        return 0

    monkeypatch.setattr(acp_main, "run", fake_run)
    monkeypatch.setattr(cli, "_maybe_brand_process", lambda args: None)
    return seen


def test_migrations_run_report_only_after_the_claim(
    calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["lop", "acp"])
    assert cli.main() == 0
    assert calls == ["claim", "path", "migrations:daemon", "run:(10, 11)"]


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        (["acp", "--yolo"], "--yolo"),
        (["--yolo", "acp"], "--yolo"),
        (["acp", "--resume", "abc123"], "--resume"),
        (["acp", "--agent", "x"], "--agent"),
        (["acp", "--train"], "--train"),
        (["acp", "--run-in", "/somewhere"], "--run-in"),
    ],
)
def test_flags_acp_cannot_honour_are_refused_before_the_claim(
    calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    named: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["lop", *argv])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 2
    assert calls == [], "refused before stdio is claimed or anything runs"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert named in captured.err and "lop acp does not take" in captured.err


def test_hosting_and_model_still_pass_through(
    calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(acp_main, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["lop", "acp", "--hosting", "test", "--model", "mock"])
    assert cli.main() == 0
    assert (seen["hosting"], seen["model"]) == ("test", "mock")

"""`exec --json` stdout is a DATA channel; nothing may write prose to it.

This is the regression guard for a defect class that recurred three times: a
log record, an operator banner and an error presenter each landed on stdout,
breaking a strict `json.loads`-per-line consumer at exactly the moment it most
needed to parse — the failure. Every case here asserts on the real streams of
the real entry points, not on a mock, because the whole bug was that the code
looked correct while the stream was not.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from local_operator import cli


def _streams(fn) -> tuple[str, str]:
    """Run ``fn`` capturing both streams; returns (stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            fn()
        except SystemExit:
            pass
    return out.getvalue(), err.getvalue()


def _assert_stdout_is_json_only(stdout: str, context: str) -> None:
    """Every non-blank stdout line must parse as JSON."""
    offenders = []
    for number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except ValueError:
            offenders.append(f"line {number}: {line[:100]!r}")
    assert not offenders, f"{context} wrote non-JSON to stdout:\n  " + "\n  ".join(offenders)


def test_missing_api_key_error_goes_to_stderr() -> None:
    """The most common exec failure there is: fresh install, no key, or a
    typo'd --hosting. It used to print a coloured line to stdout."""
    stdout, stderr = _streams(lambda: cli._preflight_api_key("mistral", None))
    _assert_stdout_is_json_only(stdout, "_preflight_api_key")
    assert "MISTRAL_API_KEY" in stderr or "API key" in stderr


def test_invalid_run_in_directory_error_goes_to_stderr() -> None:
    stdout, stderr = _streams(lambda: cli._apply_run_in("/definitely/not/a/directory/xyz"))
    _assert_stdout_is_json_only(stdout, "_apply_run_in (invalid)")
    assert "Invalid working directory" in stderr


def test_run_in_success_banner_goes_to_stderr(tmp_path, monkeypatch) -> None:
    # _apply_run_in os.chdir()s as its whole purpose, and a process-global cwd
    # change leaks into every later test — it altered how tool cards shorten
    # paths and broke two TUI snapshots. monkeypatch.chdir restores on teardown.
    monkeypatch.chdir(tmp_path)
    stdout, stderr = _streams(lambda: cli._apply_run_in(str(tmp_path)))
    _assert_stdout_is_json_only(stdout, "_apply_run_in (valid)")
    assert "Setting working directory" in stderr


def test_top_level_exception_presenter_goes_to_stderr(monkeypatch) -> None:
    """`main()` wraps the exec dispatch, so its except-block is the error
    presenter for `exec --json`. It printed four decorated lines to stdout."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated preflight explosion")

    monkeypatch.setattr(cli, "build_cli_parser", boom)
    stdout, stderr = _streams(cli.main)
    _assert_stdout_is_json_only(stdout, "main() error presenter")
    assert "simulated preflight explosion" in stderr


def test_config_version_warning_goes_to_stderr(tmp_path, monkeypatch) -> None:
    """ConfigManager is constructed on the exec path, so its version warning
    is on the data channel too."""
    from local_operator.config import ConfigManager

    monkeypatch.setattr("local_operator.config.version", lambda _name: "0.0.1")
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text("version: 99.0.0\nconversation_length: 10\n")
    stdout, stderr = _streams(lambda: ConfigManager(config_dir))
    _assert_stdout_is_json_only(stdout, "ConfigManager version warning")
    assert "newer than the current version" in stderr


def test_background_job_notices_go_to_stderr(tmp_path, monkeypatch) -> None:
    """--json and --background are independent flags, so the two 'Started
    background job' lines must not precede the event stream on stdout."""
    from local_operator import exec_mode

    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda _a: None)
    monkeypatch.setattr(exec_mode, "_ensure_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(exec_mode, "_append_job_record", lambda *a, **k: None)

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(exec_mode.subprocess, "Popen", lambda *a, **k: FakeProc())
    args = exec_mode.ExecArgs(json_mode=True, background=True)
    stdout, stderr = _streams(lambda: exec_mode._spawn_background("say hi", args))
    _assert_stdout_is_json_only(stdout, "_spawn_background notices")


@pytest.mark.parametrize(
    "polluted",
    [
        "plain prose line",
        "\x1b[1;31mError: coloured banner\x1b[0m",
        "2026-08-05 12:00:00 - INFO - HTTP Request: POST https://x",
    ],
)
def test_the_purity_assertion_itself_catches_pollution(polluted: str) -> None:
    """Guard against the guard being vacuous: the helper must fail on the exact
    three shapes that shipped to stdout in the past."""
    with pytest.raises(AssertionError):
        _assert_stdout_is_json_only('{"type":"agent_start"}\n' + polluted + "\n", "fixture")

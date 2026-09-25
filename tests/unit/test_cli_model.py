"""``lop model`` — switch another live session's model from a terminal (design D4).

Grammar cases go through the production parser (``build_cli_parser``), as
``test_cli_send.py`` does, because the hazard is in how argparse SLOTS the
positionals once a ``--pid``/``--session`` selector is present. Delivery cases
dial a REAL in-process registrant, so each exit code and line is the one a
user would read.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.cli import build_cli_parser, model_command
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.mobile.test_peer_model_wire import _ModelHandle, _OldRuntime
from tests.unit.session.runtime.test_server import _wait_record


def _run(argv: "list[str]") -> int:
    return model_command(build_cli_parser().parse_args(["model", *argv]))


def _alias(own: Any, **overrides: Any) -> registry.SessionRecord:
    """A live, engaged record under a pid that is not this process's."""
    fields: dict[str, Any] = {
        "pid": os.getppid(),
        "kind": "tui",
        "session_id": "alias-session",
        "conversation_name": "experiment one",
        "cwd": "/tmp",
        "model_label": "test/model",
        "control_port": own.control_port,
        "control_key": own.control_key,
        "started": True,
    }
    fields.update(overrides)
    record = registry.SessionRecord(**fields)
    registry.publish(record)
    return record


@pytest.fixture
def no_self(monkeypatch):
    """The sender identity is this test process, never the alias pid."""
    monkeypatch.setattr("local_operator.cli._peer_sender_identity", lambda: {"pid": os.getpid()})


@pytest.mark.asyncio
async def test_a_name_and_a_model_switch_the_peer(no_self, capsys) -> None:
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        alias = _alias(await _wait_record())
        import asyncio

        rc = await asyncio.to_thread(_run, ["experiment", "deepseek/deepseek-flash"])
        out = capsys.readouterr()
        assert rc == 0, out.err
        assert out.out.strip() == (
            f"pid {alias.pid} 'experiment one': switched to deepseek/deepseek-flash "
            "(was test/model); its next turn runs on it"
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_pid_with_one_positional_binds_it_as_the_model(no_self, capsys) -> None:
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        alias = _alias(await _wait_record())
        import asyncio

        rc = await asyncio.to_thread(_run, ["--pid", str(alias.pid), "deepseek/deepseek-flash"])
        assert rc == 0, capsys.readouterr().err
        assert handle.calls[-1][0:2] == ("receive_peer_model", ("deepseek", "deepseek-flash"))
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_refusal_exits_non_zero_with_the_targets_sentence(no_self, capsys) -> None:
    runtime = RuntimeServer(_ModelHandle(), kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        alias = _alias(await _wait_record())
        import asyncio

        rc = await asyncio.to_thread(_run, ["--pid", str(alias.pid), "nosuchprov/x"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "refused: 'nosuchprov' is not a known provider; still on test/model" in err
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_an_older_peer_exits_non_zero_and_says_nothing_changed(no_self, capsys) -> None:
    runtime = _OldRuntime(_ModelHandle(), kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        alias = _alias(await _wait_record())
        import asyncio

        rc = await asyncio.to_thread(_run, ["--pid", str(alias.pid), "deepseek/deepseek-flash"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "runs an older lop that cannot switch models remotely; nothing changed" in err
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "argv,needle",
    [
        (["experiment", "deepseek/x", "--pid", "12"], "not both"),
        (["deepseek/deepseek-flash"], "name the session to switch as well"),
        ([], "usage: lop model"),
        (["experiment", "deepseek"], "<provider>/<model-id>"),
    ],
)
def test_a_malformed_command_dials_nothing(argv, needle, capsys) -> None:
    with patch("local_operator.mobile.peer_client.send_control_op") as dial:
        assert _run(argv) == 1
    assert needle in capsys.readouterr().err
    dial.assert_not_called()


def test_pid_and_session_together_is_a_parser_error() -> None:
    with pytest.raises(SystemExit):
        build_cli_parser().parse_args(["model", "--pid", "1", "--session", "s", "a/b"])


def test_a_stored_session_is_named_not_running(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "coldsess1").mkdir(parents=True)
    assert _run(["--session", "coldsess1", "deepseek/deepseek-flash"]) == 1
    assert "session 'coldsess1' is not running — open it and use /model" in (
        capsys.readouterr().err
    )


def test_an_unknown_name_is_a_plain_miss(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert _run(["nobody-by-this-name", "deepseek/deepseek-flash"]) == 1
    assert "no live session matches 'nobody-by-this-name'" in capsys.readouterr().err

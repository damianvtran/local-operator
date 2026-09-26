"""Structural guards: the CLI-startup import rule and the namespace constants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from local_operator.network import types

REPO = Path(__file__).resolve().parents[3]

_PROBE = """
import json, importlib, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""


def _modules_after_import(target: str) -> set[str]:
    """A FRESH interpreter per target: pytest has already imported half the tree, so
    an in-process sys.modules assertion would pass on a real regression."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_importing_the_package_does_not_load_the_crypto_stack() -> None:
    """``local_operator/cli.py`` imports this package to register ``lop network``, so
    importing it must not cost every ``lop`` invocation a crypto stack."""
    loaded = _modules_after_import("local_operator.network")
    assert "cryptography" not in loaded
    assert "asyncio" not in loaded
    assert "local_operator.network.relay" not in loaded
    assert "local_operator.network.handshake" not in loaded


def test_importing_the_cli_module_stays_on_the_startup_path() -> None:
    """Argument registration is stdlib-only: no crypto, no relay, no config store."""
    loaded = _modules_after_import("local_operator.network.cli")
    assert "cryptography" not in loaded
    assert "local_operator.network.relay" not in loaded
    assert "local_operator.config" not in loaded


def test_the_peers_namespace_is_its_own_and_is_a_wire_constant() -> None:
    """A2: a FOURTH namespace, because every reader of ``run/mobile`` treats each
    file there as a session."""
    assert types.PEERS_RUN_DIRNAME == "run/peers"
    from local_operator.session.runtime import types as runtime_types

    assert types.PEERS_RUN_DIRNAME not in (
        runtime_types.RUN_DIRNAME,
        runtime_types.SERVE_RUN_DIRNAME,
        runtime_types.HOST_RUN_DIRNAME,
    )


def test_the_link_version_is_separate_from_the_session_protocol() -> None:
    """Two numbers, two meanings: a link feature moves neither, and a session-level
    addition moves neither either (it is negotiated by capability string)."""
    from local_operator.session.runtime.types import PROTOCOL_VERSION

    assert types.MESH_PROTOCOL_VERSION == 1
    assert types.MESH_PROTOCOL_VERSION != PROTOCOL_VERSION
    from local_operator.network import handshake as hs_mod

    assert hs_mod.LINK_VERSION == types.MESH_PROTOCOL_VERSION


def test_the_network_group_is_registered_by_the_main_cli() -> None:
    """The one wiring edit: the parser the user actually runs knows the group."""
    from local_operator.cli import build_cli_parser
    from tests.unit.network import conftest as net_fixtures

    assert "network" in set(net_fixtures.subcommands_of(build_cli_parser()))


def _leaf(prog: list[str]) -> Any:
    """The parser for one verb, from the parser the USER runs."""
    import argparse

    from local_operator.cli import build_cli_parser
    from tests.unit.network import conftest as net_fixtures

    parser: Any = build_cli_parser()
    for name in prog:
        parser = net_fixtures.subcommands_of(parser)[name]
    assert isinstance(parser, argparse.ArgumentParser)
    return parser


def _yolo_action(parser: Any) -> Any:
    actions = [a for a in parser._actions if "--yolo" in a.option_strings]
    assert len(actions) == 1, "one declaration, never two"
    return actions[0]


def test_the_sessions_verb_does_not_promise_yolo_it_refuses() -> None:
    """QA round 1, Q3: the help advertised a flag this verb declines.

    ``cli._propagate_global_flags`` gives every subcommand ``--yolo`` with its global
    sentence ("Auto-approve all tool executions … without prompting"), and
    ``network sessions --create`` REFUSES it on both ends — a session on another device
    must not run unattended. The flag stays accepted (so a caller gets the refusal
    sentence rather than "unrecognized arguments"); only the promise goes.
    """
    sessions = _leaf(["network", "sessions"])
    action = _yolo_action(sessions)
    help_text = action.help or ""
    assert "Refused for this verb's peer create" in help_text, help_text
    assert "Auto-approve" not in help_text, help_text
    # Still parsed, and still a real bool: the guard reads it, it is not rejected by
    # argparse.
    assert sessions.parse_args(["--create", "--yolo"]).yolo is True


def test_the_other_verbs_keep_the_global_yolo_wording() -> None:
    """The marker is read per verb, so the override cannot leak into the rest."""
    exec_help = _yolo_action(_leaf(["exec"])).help or ""
    assert "Auto-approve all tool executions" in exec_help, exec_help

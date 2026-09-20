"""Structural guards: the CLI-startup import rule and the namespace constants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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

    parser = build_cli_parser()
    actions = [action for action in parser._actions if getattr(action, "choices", None)]
    subcommands = set()
    for action in actions:
        if isinstance(getattr(action, "choices", None), dict):
            subcommands |= set(action.choices)
    assert "network" in subcommands

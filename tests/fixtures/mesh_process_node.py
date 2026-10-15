"""Two-process signed-onboarding fixture: real relay, CLI, stores and protocol.

Only the absent OS root-anchor installation/hardware boundary is represented by
an isolated public fixture file. The adapter copies the actual signed anchor
statement there. No test-mode admission, fake SAS or fake join/grant commands.
RPC is a test pipe, NOT a product endpoint or a replacement mesh transport.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

# pytest's pythonpath is process-local. Standalone helper entrypoints must pin
# this checkout BEFORE imports: an editable finder may expose local_operator but
# not tests, and a wheel must never silently substitute another source revision.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    from local_operator.network import identity, relay
    from local_operator.network.credentials import github
    from local_operator.operator import trust
    from tests.unit.network.test_relay_e2e import _init_network

    SOURCE_ROOT = Path(__file__).resolve().parents[2]
    assert Path(relay.__file__).resolve() == SOURCE_ROOT / "local_operator/network/relay.py"
    HELPER_SOURCE = sys.modules[_init_network.__module__].__file__
    assert HELPER_SOURCE is not None
    assert Path(HELPER_SOURCE).resolve() == SOURCE_ROOT / "tests/unit/network/test_relay_e2e.py"

    root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    public_anchor = root / "fixture-owned-anchor.json"

    def os_anchor(uid=None):
        # Hardware/root-owned-OS installation is explicitly synthetic. Signature
        # math and all network authority verification remain production code.
        anchor = (
            trust.OperatorAnchor.from_json(json.loads(public_anchor.read_text()))
            if public_anchor.exists()
            else None
        )
        return trust.AnchorLoad(
            anchor=anchor,
            path=public_anchor,
            root_owned=anchor is not None,
            reason="fixture OS boundary",
            exists=anchor is not None,
        )

    trust.load_anchor = os_anchor
    # Match the existing network conftest's OS-discovery seam in child
    # processes too: discovering native gh must not reach the user's keychain.
    github._resolve_program = lambda name, path=None: None
    node = relay.RelayServer(
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.load_or_mint(name=sys.argv[1]),
    )
    _, port = node.bind()
    node.bind_control()
    # Config produced by an OS-managed node would agree with its bound listener.
    # This minimal isolated config lets real CLI joins advertise this real port.
    (root / "config.yml").write_text(
        f"values:\n  network:\n    port: {port}\n    listen_address: 127.0.0.1\n"
    )
    # The inviter has a real stored SYNTHETIC provider row/offer candidate;
    # the joining node has NO git/model/provider configuration. Enrollment must
    # not clone access to it or call it task-ready. No provider API is contacted.
    if sys.argv[1] == "fixture-inviter":
        from local_operator.providers.auth_store import AuthStore

        os.environ["LOP_TUNNEL_NO_REARM"] = "1"
        auth = AuthStore(db_path=root / "auth.db", config_dir=root)
        try:
            auth.upsert_credential(
                "openai", {"type": "api_key", "key": "synthetic-provider-fixture-not-a-key"}
            )
        finally:
            auth.close()
    record = _init_network(node) if sys.argv[1] == "fixture-inviter" else None
    node.start()
    print(
        json.dumps(
            {
                "device_id": node.identity.device_id,
                "port": port,
                "network_id": record.network_id if record else "",
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            if request.get("audit"):
                print(json.dumps(node.audit.tail(1000)), flush=True)
                continue
            # Execute the ACTUAL CLI entry in each already-isolated node process,
            # not a fresh interpreter per verb. The daemon thread still serves
            # real control/TCP requests and real SAS/store/signature paths. This
            # removes import/startup weight without faking join/grants responses.
            from local_operator.cli import main as cli_main
            from local_operator.network import cli as network_cli

            network_cli._read_code = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("unexpected human code prompt")
            )
            saved = sys.argv
            out, err = io.StringIO(), io.StringIO()
            try:
                sys.argv = ["lop", *request["argv"]]
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        rc = cli_main()
                    except SystemExit as exit_result:
                        rc = int(exit_result.code or 0)
            finally:
                sys.argv = saved
            print(
                json.dumps({"rc": rc, "stdout": out.getvalue(), "stderr": err.getvalue()}),
                flush=True,
            )
    finally:
        node.stop()


if __name__ == "__main__":
    main()

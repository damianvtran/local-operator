#!/usr/bin/env python3
"""Drive the REAL bridge daemon through a version-skew session, end to end.

Why this is not a unit test: the unit suite drives `BrowserResource` against a
fake `BridgeClient`, which proves the decision logic but says nothing about
whether the pieces still fit together over the real HTTP leg — the daemon's
`/health` shape, the state-file fields the resource reads through
`state_store.read()`, and the plain `close` the degraded path issues. This is
the smallest script that exercises all three against a live daemon.

It is also the reproduction for the defect: with a peer reporting an
OLDER extension it must (a) keep the link, (b) report the advisory rather than
refuse, (c) degrade the ownership lifecycle instead of failing the session.

IT NEVER TOUCHES THE OPERATOR'S BRIDGE, by construction:

  * it runs the daemon itself, in the foreground, on a random port;
  * `HOME` and `LOCAL_OPERATOR_CONFIG_DIR` point at a fresh temp tree, so the
    daemon's state, pairing file and logs are all inside it (per AGENTS.md,
    `LOCAL_OPERATOR_CONFIG_DIR` alone is not enough — the cache resolves from
    `HOME`);
  * the only socket it opens is that daemon's, and the only peer it presents is
    a synthetic extension id.

Usage::

    python scripts/browser_skew_probe.py --version 0.1.10
    python scripts/browser_skew_probe.py --version 0.1.13   # the control
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import websockets

REPO = Path(__file__).resolve().parents[1]
EXTENSION_ID = "b" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"
TOKEN = secrets.token_urlsafe(32)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def isolated_env(home: Path) -> dict[str, str]:
    """A fresh HOME plus config root, with every inherited LOP_*/CMUX_* gone.

    An inherited `CMUX_WORKSPACE_ID` is the documented way a harness renames the
    operator's real cmux workspaces; `LOP_*` is the quieter one that silently
    changes which provider/model a child runtime adopts. Neither has any business
    reaching a probe.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CMUX_") and not key.startswith("LOP_")
    }
    config = home / ".local-operator"
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)
    return env


def state_store_path(env: dict[str, str]) -> Path:
    """The daemon's discovery file for an isolated config root."""
    return Path(env["LOCAL_OPERATOR_CONFIG_DIR"]) / "run" / "browser" / "bridge.json"


def pairing_path(config: Path) -> Path:
    """Where the daemon keeps the pairing record: `config / PAIRING_FILENAME`.

    Deliberately derived from the daemon's own constant rather than spelled
    again here, so a rename cannot leave this probe writing a file nothing
    reads (which is exactly how the first run of this script produced a
    `paired: false` handshake).
    """
    from local_operator.browser_bridge.daemon import PAIRING_FILENAME

    return config / PAIRING_FILENAME


class FakeExtension:
    """A synthetic peer that answers exactly like a released extension.

    `owner_recover` deliberately answers with the BARE `internal` shape and empty
    `data` — the shape a pre-ownership release produces because it has no such
    method, and the shape a wedged current release produces when something inside
    the handler throws. Which of the two it MEANS is what the version decides.

    ``enforce_owner_guard`` makes the peer model the released extension's own
    guard verbatim (`extension/src/commands/nav.ts`: a `close` on a surface that
    carries an `allocationId`, with no `owner_proof` in the params, is refused
    with `owner_refused` / "owner-aware client required"). Every surface the
    degraded flow opens carries one, because `_browser_identity_params` folds
    `allocation_id` into the commands the session sends — so a fallback that
    closes with the tab capability ALONE is refused for real, and the tab stays
    open. That is review R1-2, and this is how it is reproduced end to end rather
    than in a fake client.
    """

    def __init__(self, port: int, version: str, *, enforce_owner_guard: bool = False) -> None:
        self.port = port
        self.version = version
        self.enforce_owner_guard = enforce_owner_guard
        self.hello_ack: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []
        self.closed: list[str] = []
        #: Whether each accepted close carried the identity params. A BOOLEAN,
        #: never the params themselves: `owner_proof` is a capability and this
        #: report is written to a file that becomes PR evidence.
        self.close_had_owner_proof: list[bool] = []
        #: Refusals, as (tab, reason) with no capability echoed back.
        self.close_refusals: list[str] = []
        self._ready = asyncio.Event()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def run(self) -> None:
        async with websockets.connect(
            f"ws://127.0.0.1:{self.port}/extension",
            additional_headers={"Origin": ORIGIN},
        ) as wire:
            await wire.send(
                json.dumps(
                    {
                        "event": "hello",
                        "proto": 1,
                        "token": TOKEN,
                        "extension_version": self.version,
                        "browser": "Chrome/153.0.8010.37",
                    }
                )
            )
            first = json.loads(await wire.recv())
            self.hello_ack = first
            self._ready.set()
            while True:
                message = json.loads(await wire.recv())
                if message.get("event") == "ping":
                    await wire.send(json.dumps({"event": "pong"}))
                    continue
                method = message.get("method", "")
                self.requests.append(message)
                if method == "close":
                    # `close` is an ORDINARY command, and a released extension
                    # implements it — including the pre-ownership ones. Only the
                    # `owner_*` lifecycle is missing on those, which is what the
                    # bare-`internal` shape below stands for.
                    params = message.get("params", {})
                    if self.enforce_owner_guard and not params.get("owner_proof"):
                        # The surface was allocated to an owner, so the guard
                        # fires regardless of what THIS frame repeats — which is
                        # why dropping the params is fatal rather than cosmetic.
                        self.close_refusals.append("owner_refused: owner-aware client required")
                        await wire.send(
                            json.dumps(
                                {
                                    "id": message.get("id", ""),
                                    "ok": False,
                                    "error": {
                                        "code": "owner_refused",
                                        "message": "owner-aware client required",
                                        "data": {},
                                    },
                                }
                            )
                        )
                        continue
                    self.closed.append(str(params.get("tab", "")))
                    self.close_had_owner_proof.append(bool(params.get("owner_proof")))
                    await wire.send(
                        json.dumps({"id": message.get("id", ""), "ok": True, "result": {}})
                    )
                    continue
                await wire.send(
                    json.dumps(
                        {
                            "id": message.get("id", ""),
                            "ok": False,
                            "error": {
                                "code": "internal",
                                "message": "the extension never answered",
                                "data": {},
                            },
                        }
                    )
                )


async def probe(version: str, *, enforce_owner_guard: bool = False) -> dict[str, Any]:
    home = Path(tempfile.mkdtemp(prefix="lo-skew-probe."))
    env = isolated_env(home)
    config = Path(env["LOCAL_OPERATOR_CONFIG_DIR"])
    # The PARENT is isolated too, not only the daemon it spawns: the in-process
    # `BrowserResource` reads the discovery file through `config_dir()`, so
    # without this it reads the OPERATOR's state (and would take its extension
    # version for this probe's — the one mistake that would make the whole
    # exercise report somebody else's answer).
    os.environ.update(
        {
            "HOME": env["HOME"],
            "LOCAL_OPERATOR_CONFIG_DIR": env["LOCAL_OPERATOR_CONFIG_DIR"],
        }
    )
    port = free_port()
    # The pairing record is written BEFORE the daemon starts, deliberately. A
    # pairing file that appears UNDER a running daemon is a pairing REVOCATION
    # from its point of view (its revocation watcher severs the live socket), so
    # writing it afterwards produced a link that died right around the call
    # under test — which is how the first run of this probe reported a wedge
    # instead of the skew it was exercising.
    pairing_path(config).parent.mkdir(parents=True, exist_ok=True)
    pairing_path(config).write_text(
        json.dumps(
            {
                "extension_id": EXTENSION_ID,
                "token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest(),
                "paired_at": time.time(),
            }
        )
    )
    daemon = subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", "browser", "serve", "--port", str(port)],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    report: dict[str, Any] = {
        "version": version,
        "port": port,
        "home": str(home),
        "owner_guard_enforced": enforce_owner_guard,
    }
    try:
        # Wait for the daemon to publish its discovery file, which is the
        # earliest moment it is serving.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if state_store_path(env).exists():
                break
            await asyncio.sleep(0.2)

        peer = FakeExtension(port, version, enforce_owner_guard=enforce_owner_guard)
        peer_task = asyncio.create_task(peer.run())
        await asyncio.wait_for(peer.wait_ready(), timeout=15)
        report["hello_ack"] = peer.hello_ack
        await asyncio.sleep(0.5)
        report["peer_alive_after_handshake"] = not peer_task.done()

        # 1. The advisory surfaces, from the CLI a user would actually run.
        status = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "browser", "status"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        report["status_exit"] = status.returncode
        report["status"] = status.stdout.splitlines()

        # 2. The degraded session round-trip, through the REAL resource code and
        #    the real HTTP leg.
        from local_operator.browser_bridge import state as state_store
        from local_operator.browser_bridge.resources import (
            BrowserOwnershipError,
            BrowserResource,
        )

        session_dir = home / "probe-session"
        session_dir.mkdir()
        resource = BrowserResource(session_dir, session_dir.name)
        resource.initialize()
        # Two honest outcomes, both evidence:
        #   * a PRE-OWNERSHIP peer (below the ownership floor) degrades, so the
        #     session keeps working and settles capability-only;
        #   * a CURRENT-but-wedged peer refuses with the OFF/ON remedy, because
        #     there is nothing to degrade to — it ships the verbs.
        try:
            recovered = await resource.recover()
        except BrowserOwnershipError as refusal:
            report["recover"] = "refused"
            report["recover_refusal"] = str(refusal)
            report["ownership_mode"] = resource.ownership
        else:
            report["recover"] = recovered
            report["ownership_mode"] = resource.ownership
            report["record_ownership"] = resource.record.get("ownership")
            resource.remember("bridge:77:probe-capability")
            report["finish"] = (await resource.finish(resource.generation, "completed")).state
            report["close_calls"] = peer.closed
            report["close_had_owner_proof"] = peer.close_had_owner_proof
            if peer.close_refusals:
                report["close_refusals"] = peer.close_refusals
        report["owner_verbs_called"] = sorted(
            {
                message.get("method", "")
                for message in peer.requests
                if message.get("method", "").startswith("owner_")
            }
        )
        state = state_store.read()
        report["state_file"] = {
            "extension_version": state.extension_version if state else None,
            "extension_proto": state.extension_proto if state else None,
            "extension_update_available": state.extension_update_available if state else None,
        }
        peer_task.cancel()
        return report
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.10", help="the version the peer reports")
    parser.add_argument(
        "--enforce-owner-guard",
        action="store_true",
        help=(
            "model the released extension's owner-aware `close` guard: refuse a close "
            "that omits owner_proof (review R1-2)"
        ),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            asyncio.run(probe(args.version, enforce_owner_guard=args.enforce_owner_guard)),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

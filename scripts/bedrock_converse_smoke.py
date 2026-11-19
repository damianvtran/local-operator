#!/usr/bin/env python3
"""Offline real SDK/exec Converse tool loops; no AWS credentials or paid calls.

Run with the branch's own venv. The reusable fake server speaks actual AWS
binary frames plus Converse JSON; there is no production loopback bridge. Each
child has isolated HOME/config/cwd, a fake aws CLI negative control, and only
synthetic bearer/SigV4 credentials. Logs show auth SCHEME, never key values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bedrock_fake_server import _isolated_env, serve

from local_operator.agent_shell import harness_child_env

# The fixture module already primes this source root; imports stay above code.
REPO = Path(__file__).resolve().parents[1]

MODELS = ("us.amazon.nova-2-lite-v1:0", "meta.llama3-3-70b-instruct-v1:0")
PROOF = "converse-tool-proof.txt"
PROOF_TEXT = "converse-tool-side-effect\n"


async def sdk_call(model: str) -> None:
    from local_operator.sdk import (
        ApprovalPolicy,
        SessionRoots,
        SessionSpec,
        open_session,
    )
    from local_operator.tui.notify import notifications_enabled

    # This is the real child, not its parent's ambient environment: fail before
    # opening a session if a future smoke builder drops notification isolation.
    assert not notifications_enabled(), "SDK smoke child must not notify the operator"
    print("SDK child notifications: disabled", flush=True)
    home = Path(os.environ["HOME"])
    roots = SessionRoots(
        config_dir=home / ".local-operator", agent_home=home / "agent", cwd=Path.cwd()
    )
    spec = SessionSpec(
        hosting="bedrock",
        model=model,
        tools=("bash",),
        approvals=ApprovalPolicy.declared(("bash",)),
    )
    async with open_session(spec, roots=roots) as session:
        reply = await session.prompt("TOOL_LOOP: write the synthetic proof then finish.")
        print("SDK reply:", reply, flush=True)
    assert Path(PROOF).read_text() == PROOF_TEXT, "SDK did not execute the real bash tool"
    print("SDK tool side effect:", Path(PROOF).read_text().strip(), flush=True)


def smoke() -> int:
    scratch = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
    if not scratch:
        raise RuntimeError("LOCAL_OPERATOR_SCRATCHPAD is required for owned scratch")
    with tempfile.TemporaryDirectory(prefix="bedrock-native-", dir=scratch) as tmp:
        root = Path(tmp)
        log = root / "requests.jsonl"
        server = serve(0, log)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        failures = 0
        try:
            for surface in ("sdk", "exec"):
                for index, model in enumerate(MODELS):
                    for signed in (False, True):
                        work = root / f"{surface}-{index}-{int(signed)}"
                        work.mkdir()
                        env = harness_child_env(_isolated_env(work, endpoint))
                        fakebin = work / "bin"
                        fakebin.mkdir()
                        aws = fakebin / "aws"
                        aws.write_text(
                            "#!/bin/sh\necho 'ERROR: aws CLI must not be called' >&2\nexit 97\n"
                        )
                        aws.chmod(0o700)
                        env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
                        if signed:
                            env.pop("AWS_BEARER_TOKEN_BEDROCK", None)
                            env.update(
                                LOP_BEDROCK_AUTH="ambient",
                                AWS_ACCESS_KEY_ID="synthetic-id",
                                AWS_SECRET_ACCESS_KEY="synthetic-not-a-real-key",
                            )
                        if surface == "sdk":
                            command = [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--sdk-call",
                                model,
                            ]
                        else:
                            command = [
                                str(REPO / ".venv/bin/local-operator"),
                                "exec",
                                "--hosting",
                                "bedrock",
                                "--model",
                                model,
                                "--tools",
                                "bash",
                                "--json",
                                "TOOL_LOOP: write the synthetic proof then finish.",
                            ]
                        print(
                            f"$ surface={surface} auth={'SigV4' if signed else 'bearer'}",
                            " ".join(command),
                            flush=True,
                        )
                        result = subprocess.run(
                            command, cwd=work, env=env, capture_output=True, text=True, timeout=240
                        )
                        print(
                            "exit=",
                            result.returncode,
                            "\n",
                            result.stdout[-4000:],
                            "\nstderr:",
                            result.stderr[-2000:],
                            flush=True,
                        )
                        proof = work / PROOF
                        passed = (
                            result.returncode == 0
                            and proof.exists()
                            and proof.read_text() == PROOF_TEXT
                        )
                        print(
                            "side effect:",
                            proof.read_text().strip() if proof.exists() else "MISSING",
                            "PASS" if passed else "FAIL",
                            flush=True,
                        )
                        failures += not passed
        finally:
            server.shutdown()
            server.server_close()
        entries = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        posts = [entry for entry in entries if entry["method"] == "POST"]
        for entry in posts:
            print(json.dumps(entry, sort_keys=True), flush=True)
            # Session naming/other harmless errands may also use this model.
            # Llama forbids streaming WITH tools, not text-only streaming; the
            # wire body, not the selected model alone, is the invariant here.
            expected = (
                "/converse"
                if "llama3-3" in entry["model_id"] and entry["tool_names"]
                else "/converse-stream"
            )
            failures += not entry["path"].endswith(expected)
        failures += not posts or not any(entry["tool_results"] for entry in posts)
        print("NATIVE SMOKE", "PASS" if not failures else f"FAIL ({failures})", flush=True)
        return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-call", choices=MODELS)
    args = parser.parse_args()
    if args.sdk_call:
        asyncio.run(sdk_call(args.sdk_call))
    else:
        raise SystemExit(smoke())

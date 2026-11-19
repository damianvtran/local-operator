#!/usr/bin/env python3
"""Offline SDK/exec AWS-chain acceptance: signed native tool loops, no LOP opt-in.

Reuses the provider's fake Bedrock protocol and SDK tool-loop driver. Every
credential source is synthetic; IMDSv2 is loopback-only. The fake aws binary
fails if invoked, and the server validates identity, region and session token
without logging authentication values. No real AWS calls occur.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bedrock_converse_smoke import MODELS, PROOF, PROOF_TEXT, REPO
from bedrock_fake_server import _isolated_env, _State, make_handler

from local_operator.agent_shell import harness_child_env


def smoke() -> int:
    scratch = Path(os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
    with tempfile.TemporaryDirectory(prefix="bedrock-chain-", dir=scratch) as tmp:
        root = Path(tmp)
        requests: list[str] = []
        signed: list[bool] = []
        expected = {"id": "", "token": ""}
        no_role = False
        delegate = make_handler(_State(root / "requests.jsonl"))
        # The factory's public return type is BaseHTTPRequestHandler, which
        # intentionally does not declare HTTP verb methods. Keep its concrete
        # POST implementation while adding signing assertions around that seam.
        delegate_post = getattr(delegate, "do_POST")

        class Bedrock(delegate):
            def do_POST(self):  # noqa: N802
                auth = self.headers.get("Authorization", "")
                signed.append(
                    auth.startswith("AWS4-HMAC-SHA256 ")
                    and f"Credential={expected['id']}/" in auth
                    and "/us-west-2/bedrock/aws4_request" in auth
                    and self.headers.get("X-Amz-Security-Token") == expected["token"]
                )
                delegate_post(self)

        class IMDS(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_PUT(self):  # noqa: N802
                requests.append(self.path)
                assert self.path == "/latest/api/token"
                assert self.headers.get("X-aws-ec2-metadata-token-ttl-seconds")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"synthetic-imds-token")

            def do_GET(self):  # noqa: N802
                requests.append(self.path)
                assert self.headers.get("X-aws-ec2-metadata-token") == "synthetic-imds-token"
                if no_role:
                    self.send_response(404)
                    self.end_headers()
                    return
                if self.path.endswith("/iam/security-credentials/"):
                    data = b"synthetic-role"
                elif self.path.endswith("/iam/security-credentials/synthetic-role"):
                    data = json.dumps(
                        dict(
                            Code="Success",
                            AccessKeyId="synthetic-role-id",
                            SecretAccessKey="synthetic-role-secret",
                            Token="synthetic-role-token",
                            Expiration="2099-01-01T00:00:00Z",
                        )
                    ).encode()
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)

        servers = [ThreadingHTTPServer(("127.0.0.1", 0), h) for h in (Bedrock, IMDS)]
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
        for thread in threads:
            thread.start()
        failures = 0
        try:
            cells = [(s, c) for s in ("sdk", "exec") for c in ("env", "profile", "imds")]
            for surface, case in cells + [("exec", "none")]:
                no_role = case == "none"
                work = root / f"{surface}-{case}"
                work.mkdir()
                env = harness_child_env(
                    _isolated_env(work, f"http://127.0.0.1:{servers[0].server_port}")
                )
                env.pop("AWS_BEARER_TOKEN_BEDROCK")
                env.pop("AWS_REGION")
                env.update(
                    AWS_CONFIG_FILE=str(work / "aws-config"),
                    AWS_SHARED_CREDENTIALS_FILE=str(work / "aws-credentials"),
                )
                (work / "aws-config").write_text(
                    "[default]\nregion = us-west-2\n[profile synthetic]\nregion = us-west-2\n"
                )
                fakebin = work / "bin"
                fakebin.mkdir()
                aws = fakebin / "aws"
                aws.write_text("#!/bin/sh\necho 'aws must not run' >&2\nexit 97\n")
                aws.chmod(0o700)
                env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
                expected.update(id=f"synthetic-{case}-id", token=f"synthetic-{case}-token")
                if case == "env":
                    env.update(
                        {
                            "AWS_ACCESS_KEY_ID": expected["id"],
                            "AWS_SECRET_ACCESS_KEY": "synthetic-secret",
                            "AWS_SESSION_TOKEN": expected["token"],
                        }
                    )
                elif case == "profile":
                    env["AWS_PROFILE"] = "synthetic"
                    (work / "aws-credentials").write_text(
                        f"[synthetic]\naws_access_key_id = {expected['id']}\n"
                        "aws_secret_access_key = synthetic-secret\n"
                        f"aws_session_token = {expected['token']}\n"
                    )
                elif case in ("imds", "none"):
                    metadata_endpoint = f"http://127.0.0.1:{servers[1].server_port}"
                    env.update(
                        AWS_EC2_METADATA_DISABLED="false",
                        AWS_EC2_METADATA_SERVICE_ENDPOINT=metadata_endpoint,
                    )
                    expected.update(id="synthetic-role-id", token="synthetic-role-token")
                assert not any(k.startswith("LOP_BEDROCK_AUTH") for k in env)
                if surface == "sdk":
                    command = [
                        sys.executable,
                        str(REPO / "scripts/bedrock_converse_smoke.py"),
                        "--sdk-call",
                        MODELS[0],
                    ]
                else:
                    command = [
                        str(REPO / ".venv/bin/local-operator"),
                        "exec",
                        "--hosting",
                        "bedrock",
                        "--model",
                        MODELS[0],
                        "--tools",
                        "bash",
                        "--json",
                        "TOOL_LOOP: write the synthetic proof then finish.",
                    ]
                before = len(signed)
                result = subprocess.run(
                    command, cwd=work, env=env, capture_output=True, text=True, timeout=120
                )
                proof = work / PROOF
                if case == "none":
                    passed = (
                        result.returncode != 0
                        and len(signed) == before
                        and "credential" in (result.stdout + result.stderr).lower()
                    )
                else:
                    passed = (
                        result.returncode == 0
                        and proof.exists()
                        and proof.read_text() == PROOF_TEXT
                        and len(signed) > before
                        and all(signed[before:])
                    )
                print(
                    json.dumps(
                        dict(
                            surface=surface,
                            source=case,
                            command=command,
                            exit=result.returncode,
                            result="PASS" if passed else "FAIL",
                            signed_requests=len(signed) - before,
                            side_effect=proof.read_text() if proof.exists() else None,
                            stdout=result.stdout[-1600:],
                            stderr=result.stderr[-1600:],
                        )
                    ),
                    flush=True,
                )
                failures += not passed
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()
        assert requests and "/latest/api/token" in requests, "IMDS control was not exercised"
        print(
            json.dumps(
                dict(
                    result="PASS" if not failures else "FAIL",
                    failures=failures,
                    imds_requests=requests,
                    all_signatures_valid=all(signed),
                )
            ),
            flush=True,
        )
        return int(bool(failures))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(smoke())

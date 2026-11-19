#!/usr/bin/env python3
"""A LOCAL fake Amazon Bedrock endpoint, plus a real-path ``lop exec`` smoke.

WHY. No Bedrock-entitled AWS credentials are available to CI or to most rigs,
yet the Bedrock provider's value is the whole stack: registry -> env tier ->
failover -> ``BedrockAnthropicClient`` -> signing -> event-stream decoding ->
usage/cost. This server speaks the two wire contracts the provider uses so QA
can drive the REAL ``lop exec`` / SDK path with no AWS at all:

* ``POST /model/{modelId}/invoke-with-response-stream`` — answers with genuine
  ``application/vnd.amazon.eventstream`` frames (encoded by the same CRC32
  encoder the unit tests use, decoded by botocore's own parser in the client),
  each carrying one Anthropic Messages event, and the final chunk's
  ``amazon-bedrock-invocationMetrics``.
* ``GET /inference-profiles`` and ``GET /foundation-models`` — discovery
  fixtures.

Point the provider at it with ``LOP_BEDROCK_ENDPOINT_URL=http://127.0.0.1:<port>``
(a test-only hook documented in ``docs/providers/BEDROCK.md``).

Behaviour by model id / prompt (so negative paths are reproducible):

* model id containing ``denied``  -> 403 ``AccessDeniedException`` "You don't
  have access to the model with the specified model ID."
* model id containing ``missing`` -> 404 ``ResourceNotFoundException``
* model id containing ``throttle`` -> 429 ``ThrottlingException``
* prompt containing ``MIDSTREAM`` -> a text delta, then an exception frame
  (``modelStreamErrorException``)
* every request whose ``system`` prefix was seen before reports
  ``cache_read_input_tokens`` > 0 (a fake prompt cache), so a 2nd turn shows
  cache reads and cost.
* a request with no ``Authorization`` header -> 403 ``UnrecognizedClientException``.

Each request is appended to ``--log`` as one JSON line: path, auth SCHEME only
(``Bearer`` / ``AWS4-HMAC-SHA256`` — never the value), the decoded body keys,
``anthropic_version``/``anthropic_beta``, and whether ``cache_control`` markers
were present. Nothing secret is logged.

Usage::

    scripts/bedrock_fake_server.py serve --port 0 --log /path/requests.jsonl
    scripts/bedrock_fake_server.py smoke     # starts a server, runs lop exec twice, checks

``smoke`` runs under an isolated HOME / config dir, with ``CMUX_*`` and every
ambient AWS variable stripped, and ``AWS_BEARER_TOKEN_BEDROCK`` set to a fake
value. It prints the commands, their actual output and the request log.
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
from typing import Any
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.unit.providers.bedrock import eventstream as es  # noqa: E402

PROFILES = {
    "inferenceProfileSummaries": [
        {
            "inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "inferenceProfileName": "US Anthropic Claude Haiku 4.5",
            "status": "ACTIVE",
            "type": "SYSTEM_DEFINED",
        },
        {
            "inferenceProfileId": "global.anthropic.claude-opus-5-5",
            "inferenceProfileName": "Global Anthropic Claude Opus 5.5",
            "status": "ACTIVE",
            "type": "SYSTEM_DEFINED",
        },
    ]
}
FOUNDATION = {
    "modelSummaries": [
        {
            "modelId": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "modelName": "Claude Haiku 4.5",
            "responseStreamingSupported": True,
            "modelLifecycle": {"status": "ACTIVE"},
        }
    ]
}


class _State:
    def __init__(self, log: Path | None) -> None:
        self.log = log
        self.seen_prefixes: set[str] = set()
        self.lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        if self.log is None:
            return
        with self.lock, self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _has_cache_control(value: Any) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(_has_cache_control(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_cache_control(v) for v in value)
    return False


def _last_user_text(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text") or "")
    return ""


def make_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet
            return

        def _json(self, status: int, payload: Any, error_type: str | None = None) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if error_type:
                self.send_header("x-amzn-ErrorType", f"{error_type}:http://internal.amazon.com/")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth_scheme(self) -> str:
            header = self.headers.get("Authorization") or ""
            return header.split(" ", 1)[0] if header else ""

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            state.record({"method": "GET", "path": path, "auth_scheme": self._auth_scheme()})
            if not self._auth_scheme():
                self._json(403, {"message": "missing auth"}, "UnrecognizedClientException")
            elif path == "/inference-profiles":
                self._json(200, PROFILES)
            elif path == "/foundation-models":
                self._json(200, FOUNDATION)
            else:
                self._json(404, {"message": "not found"}, "ResourceNotFoundException")

        def do_POST(self) -> None:  # noqa: N802
            raw_path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            parts = raw_path.split("/")
            model_id = unquote(parts[2]) if len(parts) > 3 and parts[1] == "model" else ""
            system_text = json.dumps(body.get("system"), sort_keys=True)
            with state.lock:
                cached = system_text in state.seen_prefixes
                state.seen_prefixes.add(system_text)
            state.record(
                {
                    "method": "POST",
                    "path": raw_path,
                    "model_id": model_id,
                    "auth_scheme": self._auth_scheme(),
                    "body_keys": sorted(body),
                    "anthropic_version": body.get("anthropic_version"),
                    "anthropic_beta": body.get("anthropic_beta"),
                    "cache_control_present": _has_cache_control(body),
                    "accept": self.headers.get("X-Amzn-Bedrock-Accept"),
                }
            )
            if not self._auth_scheme():
                self._json(
                    403,
                    {"message": "The security token included in the request is invalid."},
                    "UnrecognizedClientException",
                )
                return
            if "denied" in model_id:
                self._json(
                    403,
                    {"message": "You don't have access to the model with the specified model ID."},
                    "AccessDeniedException",
                )
                return
            if "missing" in model_id:
                self._json(
                    404,
                    {"message": "The provided model identifier is invalid."},
                    "ResourceNotFoundException",
                )
                return
            if "throttle" in model_id:
                self._json(
                    429,
                    {"message": "Too many requests, please wait before trying again."},
                    "ThrottlingException",
                )
                return
            prompt = _last_user_text(body)
            usage = {
                "input_tokens": 12,
                "cache_read_input_tokens": 900 if cached else 0,
                "cache_creation_input_tokens": 0 if cached else 900,
            }
            events = es.anthropic_events(
                text=f"fake-bedrock says hi ({model_id})",
                usage=usage,
                output_tokens=6,
                metrics={
                    "inputTokenCount": 12,
                    "outputTokenCount": 6,
                    "cacheReadInputTokenCount": usage["cache_read_input_tokens"],
                    "cacheWriteInputTokenCount": usage["cache_creation_input_tokens"],
                },
            )
            payload = es.stream_bytes(events)
            if "MIDSTREAM" in prompt:
                payload = es.stream_bytes(events[:3]) + es.exception(
                    "modelStreamErrorException", "fake mid-stream failure"
                )
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.amazon.eventstream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def serve(port: int, log: Path | None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(_State(log)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _isolated_env(home: Path, endpoint: str) -> dict[str, str]:
    keep = {"PATH", "TERM", "LANG", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(
        {
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"),
            "TERM": "xterm-256color",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_BEARER_TOKEN_BEDROCK": "fake-bedrock-api-key",
            "AWS_REGION": "us-east-1",
            "LOP_BEDROCK_ENDPOINT_URL": endpoint,
        }
    )
    return env


def smoke(model: str, lop: str) -> int:
    with tempfile.TemporaryDirectory(prefix="bedrock-smoke-") as tmp:
        home = Path(tmp)
        log = home / "requests.jsonl"
        server = serve(0, log)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        env = _isolated_env(home, endpoint)
        failures = 0
        try:
            for prompt in ("Say hi in three words.", "Say hi again."):
                cmd = [lop, "exec", "--hosting", "bedrock", "--model", model, "--json", prompt]
                print("$", " ".join(cmd), flush=True)
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
                print(f"exit={result.returncode}")
                print(result.stdout[-3000:])
                if result.stderr.strip():
                    print("stderr:", result.stderr[-2000:])
                if result.returncode != 0 or "fake-bedrock says hi" not in result.stdout:
                    failures += 1
        finally:
            server.shutdown()
        print("--- request log ---")
        entries = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        for entry in entries:
            print(json.dumps(entry, sort_keys=True))
        posts = [e for e in entries if e["method"] == "POST"]
        if not posts or any(e["auth_scheme"] != "Bearer" for e in posts):
            failures += 1
        if any(e["anthropic_version"] != "bedrock-2023-05-31" for e in posts):
            failures += 1
        print("SMOKE", "PASS" if failures == 0 else f"FAIL ({failures})")
        return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve_p = sub.add_parser("serve", help="run the fake endpoint until interrupted")
    serve_p.add_argument("--port", type=int, default=0)
    serve_p.add_argument("--log", type=Path)
    smoke_p = sub.add_parser("smoke", help="drive lop exec against a fresh fake endpoint")
    smoke_p.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    smoke_p.add_argument("--lop", default=str(REPO / ".venv" / "bin" / "local-operator"))
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return smoke(args.model, args.lop)
    server = serve(args.port, args.log)
    print(f"fake Bedrock listening on http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

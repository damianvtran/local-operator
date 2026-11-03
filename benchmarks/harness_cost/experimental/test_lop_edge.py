"""Offline edge-proxy checks over REAL loopback sockets, TLS and sandbox-exec.

WHY: the edge is the only place Local Operator's paid traffic is observed, so
its tests drive it the way the product does — an ``httpx`` client honouring
``HTTPS_PROXY``/``SSL_CERT_FILE`` CONNECTs, completes TLS against the per-run CA
and sends a request. Upstreams are in-memory fixtures with synthetic billing.
CONSTRAINT: credentials here are FAKE literals; nothing resolves the operator's
Radient token or OpenRouter key, and no request leaves loopback. These are
edge/meter/sandbox controls, not native ``lop exec`` evidence (S4).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib.util
import json
import os
import ssl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


meter = load("meter_proxy")
edge = load("lop_edge")
adapters = load("test_adapters")

FAKE_RADIENT = "fake-radient-token-for-offline-tests"
FAKE_OPENROUTER = "fake-openrouter-key-for-offline-tests"
DECISIONS = "https://api.radienthq.com/v1/decisions"
MESSAGES = "https://api.anthropic.com/v1/messages"


def decisions_payload(**cost_updates: object) -> dict[str, object]:
    """Synthetic Radient bill matching the reviewed decisions-rate contract."""
    cost: dict[str, object] = {
        "v": 1,
        "basis": "billed",
        "model": "jev-1.13",
        "provider": "typesafe",
        "price_version": "decisions-rate@2026-09-19",
        "unit": "token",
        "units": 100,
        "unit_price_usd": 0.000000042,
        "amount_usd": 0.0000042,
    }
    cost.update(cost_updates)
    return {
        "model": "jev-1.13.0",
        "provider": "typesafe",
        "usage": {"input_tokens": 100, "output_tokens": 2},
        "answers": {"title_fit": {"choice": "fits"}},
        "cost": cost,
        "usage_record_id": "ur_synthetic_1",
    }


DECISION_BODY: dict[str, object] = {
    "model": "jev-1.13",
    "state": "synthetic state",
    "questions": {
        "title_fit": {"type": "choice", "instructions": "x", "criteria": {"fits": "y"}}
    },
}

DECISION_BODY_BYTES = json.dumps(DECISION_BODY).encode()


class EdgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
        self.root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ,
            {
                **adapters.meter_env(self.root),
                "OPENROUTER_API_KEY": FAKE_OPENROUTER,
                "BENCH_TRIAL_BUDGET_USD": "1",
                "BENCH_ADMISSION_WAIT_S": "0.2",
            },
        )
        self.env.start()
        self.logpatch = patch.object(meter, "LOG", self.root / "meter.jsonl")
        self.logpatch.start()
        self.model_calls: list[httpx.Request] = []
        self.decision_calls: list[httpx.Request] = []
        self.model_response: tuple[int, dict[str, object]] = (
            200,
            {"model": "anthropic/claude-sonnet-5.5", "usage": {"cost": 0.01}},
        )
        self.decision_response: tuple[int, object] = (200, decisions_payload())

        def model_fixture(request: httpx.Request) -> httpx.Response:
            self.model_calls.append(request)
            status, body = self.model_response
            return httpx.Response(status, json=body)

        def decision_fixture(request: httpx.Request) -> httpx.Response:
            self.decision_calls.append(request)
            status, body = self.decision_response
            return httpx.Response(status, json=body)

        self.meter_upstream = httpx.AsyncClient(transport=httpx.MockTransport(model_fixture))
        self.clientpatch = patch.object(meter, "client", self.meter_upstream)
        self.clientpatch.start()
        # Edge -> meter goes through the meter's REAL ASGI app (one pricing core),
        # sharing the ledger file through its own startup-built guard.
        self.meter_guard = meter.BudgetGuard()
        self.to_meter = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=meter.make_app(self.meter_guard)),
            base_url="http://meter",
        )
        self.to_radient = httpx.AsyncClient(transport=httpx.MockTransport(decision_fixture))
        self.edge = edge.EdgeServer(
            run_id="trial",
            run_dir=self.root / "run",
            radient_token=FAKE_RADIENT,
            meter_base="http://meter",
            meter_client=self.to_meter,
            decisions_client=self.to_radient,
        )
        await self.edge.start()
        self.ca = self.edge.ca_path
        self.clients: list[httpx.AsyncClient] = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.aclose()
        await self.edge.close()
        self.meter_guard.close()
        await self.to_meter.aclose()
        await self.to_radient.aclose()
        await self.meter_upstream.aclose()
        self.clientpatch.stop()
        self.logpatch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def client(self, nonce: str | None = None, run_id: str = "trial") -> httpx.AsyncClient:
        """A client configured exactly as the child environment configures LoP."""
        secret = self.edge.nonce if nonce is None else nonce
        proxy = f"http://{run_id}:{secret}@127.0.0.1:{self.edge.port}"
        client = httpx.AsyncClient(
            proxy=proxy,
            verify=ssl.create_default_context(cafile=str(self.ca)),
            trust_env=False,
            timeout=5,
        )
        self.clients.append(client)
        return client

    def rows(self) -> list[dict[str, object]]:
        path = self.edge.log_path
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def ledger(self) -> dict[str, Any]:
        # The ledger is the meter's untyped JSON file; tests index it freely.
        return json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())

    # --- CA / child environment -------------------------------------------------

    def test_only_the_ca_certificate_reaches_disk(self):
        files = sorted(p.name for p in (self.root / "run").rglob("*") if p.is_file())
        self.assertEqual(files, ["edge-ca.pem"])
        pem = self.ca.read_text()
        self.assertEqual(pem.count("BEGIN CERTIFICATE"), 1)
        self.assertNotIn("PRIVATE KEY", pem)

    def test_child_environment_carries_proxy_ca_and_dummy_key_only(self):
        env = self.edge.child_env()
        self.assertEqual(env["RADIENT_API_KEY"], edge.DUMMY_RADIENT_KEY)
        self.assertEqual(env["SSL_CERT_FILE"], str(self.ca))
        self.assertEqual(env["REQUESTS_CA_BUNDLE"], str(self.ca))
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            self.assertEqual(
                env[name], f"http://trial:{self.edge.nonce}@127.0.0.1:{self.edge.port}"
            )
        self.assertEqual(env["NO_PROXY"], "127.0.0.1,localhost,::1")
        self.assertNotIn(FAKE_RADIENT, json.dumps(env))
        self.assertNotIn(FAKE_OPENROUTER, json.dumps(env))

    # --- routes --------------------------------------------------------------

    async def test_model_route_reaches_meter_without_child_credentials(self):
        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 128000,
            "messages": [{"role": "user", "content": "hi"}],
            "diagnostics": {"previous_message_id": None},
        }
        response = await self.client().post(
            MESSAGES, json=body, headers={"x-api-key": "child-dummy", "authorization": "Bearer x"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.model_calls), 1)
        sent = self.model_calls[0]
        self.assertEqual(sent.headers["authorization"], f"Bearer {FAKE_OPENROUTER}")
        self.assertNotIn("x-api-key", sent.headers)
        forwarded = json.loads(sent.content)
        self.assertEqual(forwarded["diagnostics"], {"previous_message_id": None})
        self.assertEqual(forwarded["max_tokens"], 128000)
        row = self.rows()[-1]
        self.assertEqual(
            (row["method"], row["host"], row["path"], row["route"]),
            ("POST", "api.anthropic.com", "/v1/messages", "model"),
        )
        self.assertEqual(self.ledger()["channels"]["model"], 0.01)

    async def test_decisions_route_injects_host_token_and_settles_classifier_channel(self):
        response = await self.client().post(
            DECISIONS, json=DECISION_BODY, headers={"authorization": "Bearer child-dummy"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answers"], {"title_fit": {"choice": "fits"}})
        self.assertEqual(len(self.decision_calls), 1)
        sent = self.decision_calls[0]
        self.assertEqual(str(sent.url), DECISIONS)
        self.assertEqual(sent.headers["authorization"], f"Bearer {FAKE_RADIENT}")
        ledger = self.ledger()
        self.assertAlmostEqual(ledger["channels"]["classifier"], 0.0000042)
        self.assertEqual(ledger["channels"]["model"], 0)
        self.assertFalse(ledger["blocked"])
        row = self.rows()[-1]
        self.assertEqual(row["channel"], "classifier")
        self.assertTrue(row["qualified"])
        self.assertNotIn("ur_synthetic_1", json.dumps(row))
        self.assertEqual(len(str(row["usage_record_sha256"])), 64)

    async def test_classifier_informational_usage_cost_alone_latches(self):
        payload = decisions_payload()
        del payload["cost"]
        payload["usage"] = {"input_tokens": 100, "output_tokens": 2, "cost": 0.0000042}
        self.decision_response = (200, payload)
        self.assertEqual((await self.client().post(DECISIONS, json=DECISION_BODY)).status_code, 200)
        self.assertTrue(self.ledger()["blocked"])
        self.assertEqual(len(self.ledger()["pending"]), 1)

    async def test_classifier_unqualified_contract_records_bill_and_latches(self):
        self.decision_response = (200, decisions_payload(price_version="other@2026"))
        await self.client().post(DECISIONS, json=DECISION_BODY)
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertAlmostEqual(ledger["channels"]["classifier"], 0.0000042)

    async def test_classifier_401_latches_and_reresolve_retry_is_refused(self):
        self.decision_response = (401, {"error": "unauthorized"})
        first = await self.client().post(DECISIONS, json=DECISION_BODY)
        retry = await self.client().post(DECISIONS, json=DECISION_BODY)
        self.assertEqual(first.status_code, 401)
        self.assertEqual(retry.status_code, 402)
        self.assertEqual(len(self.decision_calls), 1)
        self.assertTrue(self.ledger()["blocked"])

    async def test_unlisted_post_is_denied_and_latches_without_dispatch(self):
        response = await self.client().post("https://api.typesafe.ai/v1/systemone", json={})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.model_calls + self.decision_calls, [])
        self.assertTrue(self.ledger()["blocked"])
        row = self.rows()[-1]
        self.assertTrue(row["denied"])
        self.assertEqual(row["host"], "api.typesafe.ai")
        # The latch voids the rest of the run, including otherwise-valid routes.
        self.assertEqual((await self.client().post(DECISIONS, json=DECISION_BODY)).status_code, 402)
        self.assertEqual(self.decision_calls, [])

    async def test_unlisted_path_and_method_on_routed_host_are_denied(self):
        denied = await self.client().post("https://api.radienthq.com/v1/decisions/other", json={})
        self.assertEqual(denied.status_code, 403)
        listing = await self.client().get(DECISIONS)
        self.assertEqual(listing.status_code, 403)
        query = await self.client().post(DECISIONS + "?x=1", json=DECISION_BODY)
        self.assertEqual(query.status_code, 403)
        self.assertEqual(self.decision_calls, [])

    async def test_unlisted_get_is_denied_without_latching(self):
        response = await self.client().get("https://example.invalid/metadata")
        self.assertEqual(response.status_code, 403)
        self.assertTrue(self.rows()[-1]["denied"])
        spend = Path(os.environ["BENCH_SPEND_FILE"])
        self.assertFalse(spend.exists() and self.ledger()["blocked"])

    async def test_diagnostics_400_without_billing_latches_and_native_retry_is_refused(self):
        # The native retry (clients.py:5316-5339) resends without diagnostics;
        # the unknown bill of the first 400 must refuse it, never dispatch it.
        self.model_response = (
            400,
            {"type": "error", "error": {"message": "diagnostics: Extra inputs are not permitted"}},
        )
        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 128000,
            "messages": [{"role": "user", "content": "hi"}],
            "diagnostics": {"previous_message_id": None},
        }
        first = await self.client().post(MESSAGES, json=body)
        del body["diagnostics"]
        retry = await self.client().post(MESSAGES, json=body)
        self.assertEqual((first.status_code, retry.status_code), (400, 402))
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(self.ledger()["blocked"])

    async def test_logs_never_contain_headers_or_credentials(self):
        await self.client().post(DECISIONS, json=DECISION_BODY, headers={"x-secret": "hdr-value"})
        await self.client().post("https://api.typesafe.ai/v1/systemone", json={})
        text = "".join(p.read_text() for p in self.root.rglob("*") if p.is_file())
        for needle in (FAKE_RADIENT, FAKE_OPENROUTER, self.edge.nonce, "hdr-value", "x-secret"):
            self.assertNotIn(needle, text)

    async def test_count_tokens_is_an_unlisted_post_that_latches(self):
        # F4: LoP 500883ef never calls it; a request there is unexplained traffic.
        response = await self.client().post(MESSAGES + "/count_tokens", json={"model": "x"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.model_calls, [])
        self.assertTrue(self.ledger()["blocked"])

    async def test_decisions_admission_refusal_voids_the_run(self):
        # F5: a refused classifier admission changes the run; it must say so.
        self.meter_guard.latch("trial-elsewhere", "test-induced latch")
        response = await self.client().post(DECISIONS, json=DECISION_BODY)
        self.assertEqual(response.status_code, 402)
        self.assertIn("admission refused", " ".join(self.ledger()["voids"]["trial"]))

    async def test_edge_close_mid_decision_latches_the_reservation(self):
        # F1 path 1 (reviewer probe 6): shutdown while a decision is in flight.
        started = asyncio.Event()

        async def slow(request: httpx.Request) -> httpx.Response:
            self.decision_calls.append(request)
            started.set()
            await asyncio.sleep(30)
            return httpx.Response(200, json=decisions_payload())

        self.to_radient._transport = httpx.MockTransport(slow)
        task = asyncio.create_task(self.client().post(DECISIONS, json=DECISION_BODY))
        await asyncio.wait_for(started.wait(), 5)
        await self.edge.close()
        with contextlib.suppress(Exception):
            await task
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertIn("abandoned", " ".join(ledger["voids"]["trial"]))
        # Kept as exposure (its bill is unknown), exactly like an unknown settle.
        self.assertEqual(len(ledger["pending"]), 1)

    async def test_edge_owned_clients_are_closed(self):
        # N3: close() releases the clients the edge built for itself.
        own = edge.EdgeServer(run_id="own", run_dir=self.root / "own",
                              radient_token=FAKE_RADIENT, meter_base="http://meter")
        await own.start()
        await own.close()
        self.assertTrue(own._meter.is_closed and own._decisions.is_closed)

    def test_misconfigured_meter_environment_fails_at_construction(self):
        # F6: one guard, built when the edge is, not per request.
        with patch.dict(os.environ):
            del os.environ["BENCH_ADMISSION_WAIT_S"]
            with self.assertRaises((KeyError, ValueError)):
                edge.EdgeServer(run_id="bad", run_dir=self.root / "bad",
                                radient_token=FAKE_RADIENT, meter_base="http://meter")

    # --- proxy authorization ---------------------------------------------------

    async def test_wrong_nonce_is_407_with_zero_fixture_hits(self):
        with self.assertRaises(httpx.ProxyError) as caught:
            await self.client(nonce="wrong").post(DECISIONS, json=DECISION_BODY)
        self.assertIn("407", str(caught.exception))
        with self.assertRaises(httpx.ProxyError):
            await self.client(run_id="other").post(DECISIONS, json=DECISION_BODY)
        self.assertEqual(self.decision_calls, [])

    async def test_missing_proxy_authorization_is_407(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.edge.port)
        writer.write(
            b"CONNECT api.radienthq.com:443 HTTP/1.1\r\nHost: api.radienthq.com:443\r\n\r\n"
        )
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        writer.close()
        self.assertTrue(head.startswith(b"HTTP/1.1 407"))

    async def test_replayed_nonce_of_finished_run_is_407(self):
        self.edge.finish()
        with self.assertRaises(httpx.ProxyError):
            await self.client().post(DECISIONS, json=DECISION_BODY)
        self.assertEqual(self.decision_calls, [])

    async def test_plain_http_proxying_is_denied_and_post_latches(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.edge.port)
        auth = base64.b64encode(f"trial:{self.edge.nonce}".encode()).decode()
        writer.write(
            b"POST http://api.radienthq.com/v1/decisions HTTP/1.1\r\n"
            b"Host: api.radienthq.com\r\nContent-Length: 2\r\n"
            + f"Proxy-Authorization: Basic {auth}\r\n\r\n{{}}".encode()
        )
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        writer.close()
        self.assertTrue(head.startswith(b"HTTP/1.1 403"))
        self.assertTrue(self.ledger()["blocked"])
        self.assertEqual(self.decision_calls, [])

    async def test_connect_to_non_tls_port_is_denied_and_latches(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.edge.port)
        auth = base64.b64encode(f"trial:{self.edge.nonce}".encode()).decode()
        writer.write(
            b"CONNECT api.radienthq.com:80 HTTP/1.1\r\n"
            + f"Proxy-Authorization: Basic {auth}\r\n\r\n".encode()
        )
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        writer.close()
        self.assertTrue(head.startswith(b"HTTP/1.1 403"))
        self.assertTrue(self.ledger()["blocked"])

    # --- inherited environment, real child process, kernel sandbox ------------

    async def sandboxed_child(self, code: str, env: dict[str, str]) -> str:
        """Run ``code`` under the egress profile allowing only the edge port."""
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-p",
            edge.sandbox_profile([self.edge.port]),
            sys.executable,
            "-c",
            code,
            env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), 60)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        self.assertEqual(proc.returncode, 0, err.decode())
        return out.decode()

    async def test_default_clients_in_a_sandboxed_child_route_through_the_edge(self):
        # The product's shape: default-transport httpx (trust_env) and requests
        # clients that only inherit the environment, in another process.
        code = (
            "import json, httpx, requests\n"
            f"body = {json.dumps(DECISION_BODY)!r}\n"
            "a = httpx.Client(timeout=10).post("
            f"{DECISIONS!r}, content=body, headers={{'content-type': 'application/json'}})\n"
            f"b = requests.post({DECISIONS!r}, data=body, timeout=10,"
            " headers={'content-type': 'application/json',"
            " 'authorization': 'Bearer bench-dummy'})\n"
            "print(a.status_code, b.status_code, a.json()['answers'] == b.json()['answers'])\n"
        )
        out = await self.sandboxed_child(code, self.edge.child_env())
        self.assertEqual(out.split(), ["200", "200", "True"])
        self.assertEqual(len(self.decision_calls), 2)
        for sent in self.decision_calls:
            self.assertEqual(sent.headers["authorization"], f"Bearer {FAKE_RADIENT}")
        digest = hashlib.sha256(DECISION_BODY_BYTES).hexdigest()
        self.assertEqual([r["body_sha256"] for r in self.rows()], [digest, digest])
        self.assertAlmostEqual(self.ledger()["channels"]["classifier"], 2 * 0.0000042)

    async def test_child_without_proxy_environment_gets_eperm_and_zero_hits(self):
        env = {k: v for k, v in self.edge.child_env().items() if "proxy" not in k.lower()}
        code = (
            "import httpx\n"
            "try:\n"
            "    httpx.Client(timeout=5).post('https://1.1.1.1/v1/decisions', content=b'{}')\n"
            "    print('reached')\n"
            "except httpx.ConnectError as e:\n"
            "    print('blocked', e)\n"
        )
        out = await self.sandboxed_child(code, env)
        self.assertIn("blocked", out)
        self.assertIn("Operation not permitted", out)
        self.assertEqual(self.decision_calls + self.model_calls, [])
        self.assertEqual(self.rows(), [])


class EntrypointTests(unittest.TestCase):
    def test_stdin_entrypoint_announces_child_env_and_exits_on_close(self):
        """The host process shape: token on stdin, env out, stdin EOF finishes."""
        with tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) as tmp:
            # The full meter configuration: the edge builds its guard at startup
            # (F6), so the entrypoint without it must refuse -- checked below.
            env = {
                **adapters.meter_env(Path(tmp)),
                "PATH": "/usr/bin:/bin",
                "HOME": os.environ["HOME"],
            }
            bare = subprocess.run(
                [sys.executable, str(ROOT / "lop_edge.py")],
                input=json.dumps({"run_id": "bare", "run_dir": str(Path(tmp) / "bare"),
                                  "meter_base": "http://127.0.0.1:9"}) + "\n",
                capture_output=True, text=True, timeout=60, cwd=ROOT,
                env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
            )
            self.assertNotEqual(bare.returncode, 0)
            self.assertEqual(bare.stdout, "")
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "lop_edge.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=ROOT,
            )
            try:
                assert proc.stdin is not None and proc.stdout is not None
                config = {
                    "run_id": "entry",
                    "run_dir": str(Path(tmp) / "run"),
                    "radient_token": FAKE_RADIENT,
                    "meter_base": "http://127.0.0.1:9",
                }
                proc.stdin.write(json.dumps(config) + "\n")
                proc.stdin.flush()
                announced = json.loads(proc.stdout.readline())
                self.assertEqual(announced["env"]["RADIENT_API_KEY"], edge.DUMMY_RADIENT_KEY)
                self.assertIn(f"127.0.0.1:{announced['port']}", announced["env"]["HTTPS_PROXY"])
                self.assertNotIn(FAKE_RADIENT, json.dumps(announced))
                proc.stdin.close()
                stderr = proc.stderr.read() if proc.stderr else ""
                self.assertEqual(proc.wait(timeout=30), 0, stderr)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            files = sorted(p.name for p in (Path(tmp) / "run").iterdir())
            self.assertEqual(files, ["edge-ca.pem"])


class CertificateTests(unittest.TestCase):
    def test_run_ca_bundle_refuses_a_server_it_did_not_sign(self):
        """Direct TLS that bypasses the edge fails: the bundle trusts only the run CA."""
        ours = edge.RunCA()
        theirs = edge.RunCA()  # stands in for a real host's public chain
        with tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) as tmp:
            bundle = ours.write_certificate(Path(tmp) / "ca.pem")
            server_ctx = theirs.server_context("api.radienthq.com")
            client_ctx = ssl.create_default_context(cafile=str(bundle))
            failure = tls_handshake(server_ctx, client_ctx, "api.radienthq.com")
            self.assertIn("CERTIFICATE_VERIFY_FAILED", failure)
            ours_ctx = ours.server_context("api.radienthq.com")
            self.assertEqual(tls_handshake(ours_ctx, client_ctx, "api.radienthq.com"), "ok")

    def test_leaf_is_valid_only_for_its_host(self):
        ca = edge.RunCA()
        with tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) as tmp:
            bundle = ca.write_certificate(Path(tmp) / "c")
            client_ctx = ssl.create_default_context(cafile=str(bundle))
            leaf = ca.server_context("api.anthropic.com")
            failure = tls_handshake(leaf, client_ctx, "openrouter.ai")
            self.assertIn("CERTIFICATE_VERIFY_FAILED", failure)


def tls_handshake(server_ctx: ssl.SSLContext, client_ctx: ssl.SSLContext, name: str) -> str:
    """Handshake over a loopback socketpair; returns 'ok' or the client error text."""
    import socket
    import threading

    left, right = socket.socketpair()

    def serve():
        try:
            with server_ctx.wrap_socket(right, server_side=True) as s:
                s.recv(1)
        except (ssl.SSLError, OSError):
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with client_ctx.wrap_socket(left, server_hostname=name) as s:
            s.sendall(b"x")
        return "ok"
    except ssl.SSLError as exc:
        return str(exc)
    finally:
        left.close()
        thread.join(timeout=5)


class SandboxTests(unittest.TestCase):
    """Kernel egress backstop: only the edge (and MCP) ports are reachable."""

    def run_probe(self, code: str, allowed: list[int]) -> subprocess.CompletedProcess[str]:
        profile = edge.sandbox_profile(allowed)
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
        )

    def test_profile_rejects_non_integer_ports(self):
        for bad in (["1; (allow default)"], [0], [70000], [True]):
            with self.assertRaises(ValueError):
                edge.sandbox_profile(bad)  # type: ignore[arg-type]
        for bad_path in ('/tmp/x")(allow default)("', "relative.sock", "/tmp/new\nline"):
            with self.assertRaises(ValueError):
                edge.sandbox_profile([9], unix_sockets=[bad_path])

    def test_unix_sockets_are_denied_unless_explicitly_allowed(self):
        # F3: a host daemon's socket must not be a path around the egress fence.
        import socket

        with tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) as tmp:
            allowed_path, other_path = str(Path(tmp) / "ok.sock"), str(Path(tmp) / "daemon.sock")
            with socket.socket(socket.AF_UNIX) as a, socket.socket(socket.AF_UNIX) as o:
                a.bind(allowed_path)
                o.bind(other_path)
                a.listen()
                o.listen()
                code = (
                    "import socket,sys\n"
                    "for path in sys.argv[1:]:\n"
                    "    s=socket.socket(socket.AF_UNIX)\n"
                    "    try:\n"
                    "        s.connect(path); print('connected')\n"
                    "    except OSError as e: print('errno', e.errno)\n"
                )
                profile = edge.sandbox_profile([9], unix_sockets=[allowed_path])
                proc = subprocess.run(
                    ["/usr/bin/sandbox-exec", "-p", profile, sys.executable, "-c", code,
                     allowed_path, other_path],
                    capture_output=True, text=True, timeout=60,
                    env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
                )
                default = subprocess.run(
                    ["/usr/bin/sandbox-exec", "-p", edge.sandbox_profile([9]), sys.executable,
                     "-c", code, allowed_path],
                    capture_output=True, text=True, timeout=60,
                    env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
                )
        self.assertEqual(proc.stdout.split("\n")[:2], ["connected", "errno 1"], proc.stderr)
        self.assertEqual(default.stdout.strip(), "errno 1", default.stderr)

    def test_only_allowed_loopback_port_connects(self):
        import socket

        with socket.socket() as allowed, socket.socket() as other:
            allowed.bind(("127.0.0.1", 0))
            other.bind(("127.0.0.1", 0))
            allowed.listen()
            other.listen()
            a, o = allowed.getsockname()[1], other.getsockname()[1]
            code = (
                "import socket,sys\n"
                "for port in map(int, sys.argv[1:]):\n"
                "    s=socket.socket()\n"
                "    try:\n"
                "        s.connect(('127.0.0.1',port)); print(port,'connected')\n"
                "    except OSError as e: print(port,'errno',e.errno)\n"
            )
            proc = subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", edge.sandbox_profile([a]), sys.executable, "-c",
                 code, str(a), str(o)],
                capture_output=True,
                text=True,
                timeout=60,
                env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split("\n")[:2], [f"{a} connected", f"{o} errno 1"])

    def test_trust_env_false_client_gets_eperm_even_with_proxy_env(self):
        # A client that ignores HTTPS_PROXY (Saver/tunnels shape) must not
        # reach the network directly from inside the sandbox.
        code = (
            "import httpx\n"
            "try:\n"
            "    httpx.Client(trust_env=False, timeout=5).post('https://1.1.1.1/', content=b'x')\n"
            "    print('reached')\n"
            "except httpx.ConnectError as e:\n"
            "    print('blocked', repr(e))\n"
        )
        proc = self.run_probe(code, [9])
        self.assertIn("blocked", proc.stdout, proc.stderr)
        self.assertIn("Operation not permitted", proc.stdout)


if __name__ == "__main__":
    unittest.main()

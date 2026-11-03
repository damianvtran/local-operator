"""Cheap offline adapter checks; fake upstreams cannot spend or launch a harness.

ASGI requests exercise real routing/rewriting/accounting, while the transport
negative controls prove rejected configurations never dispatch an upstream call.
These are source-only checks, not provider billing or native harness evidence.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def until(predicate, timeout: float = 3.0) -> None:
    """Poll a test condition with a deadline so a regression fails, never hangs."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


meter = load("meter_proxy")
bridge = load("codex_toolsearch_bridge")
catalog = load("codex_catalog")


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
        root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "offline-test-key-not-a-credential",
                "BENCH_BUDGET_USD": "1",
                "BENCH_TRIAL_BUDGET_USD": "0.3",
                # 128000 output tokens at 1 USD/M plus the 10000-byte input
                # bound reserve 0.2335 USD; 0.24 keeps one request per trial.
                "BENCH_REQUEST_RESERVE_USD": "0.24",
                "BENCH_INPUT_USD_PER_MILLION": "1",
                "BENCH_OUTPUT_USD_PER_MILLION": "1",
                "BENCH_MAX_INPUT_BYTES": "10000",
                "BENCH_MAX_OUTPUT_TOKENS": "128000",
                "BENCH_ADMISSION_WAIT_S": "0.3",
                "BENCH_SERVICE_TIER": "default",
                "BENCH_SPEND_FILE": str(root / "spend.json"),
            },
        )
        self.env.start()
        self.logpatch = patch.object(meter, "LOG", root / "requests.jsonl")
        self.logpatch.start()
        self.calls = []
        self.response = {"model": "anthropic/claude-sonnet-5.5", "usage": {"cost": 0.12}}

        def upstream(request):
            self.calls.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps(self.response).encode()),
            )

        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.clientpatch = patch.object(meter, "client", self.upstream)
        self.clientpatch.start()
        self.api = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=meter.app), base_url="http://offline"
        )
        self.body = {
            "model": meter.MODEL,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }

    async def asyncTearDown(self):
        await self.api.aclose()
        await self.upstream.aclose()
        self.clientpatch.stop()
        self.logpatch.stop()
        self.env.stop()
        self.tmp.cleanup()

    async def post(self, run: str = "trial", /, **updates: object):
        return await self.api.post(f"/r/{run}/v1/messages", json={**self.body, **updates})

    async def test_real_asgi_route_settles_and_stops_trial(self):
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual(len(self.calls), 1)
        forwarded = json.loads(self.calls[0].content)
        self.assertFalse(forwarded["provider"]["allow_fallbacks"])
        self.assertEqual(forwarded["provider"]["max_price"], {"prompt": 1, "completion": 1})
        self.assertEqual(forwarded["service_tier"], "default")
        ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
        self.assertEqual(ledger["usd"], 0.12)
        self.assertEqual(ledger["trials"]["trial"], 0.12)
        self.assertNotIn(os.environ["OPENROUTER_API_KEY"], meter.LOG.read_text())
        self.assertNotIn("hello", meter.LOG.read_text())

    async def test_total_cap_independent_of_trial_ids(self):
        with patch.dict(os.environ, {"BENCH_BUDGET_USD": "0.3"}):
            self.assertEqual((await self.post("one")).status_code, 200)
            self.assertEqual((await self.post("two")).status_code, 402)
        self.assertEqual(len(self.calls), 1)

    async def test_negative_controls_never_dispatch(self):
        for update in (
            {"model": "unpriced"},
            {"max_tokens": 128001},
            {"service_tier": "priority"},
            {"plugins": []},
            {"tools": [{"type": "web_search_20250305"}]},
            {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        ):
            self.assertEqual((await self.post(**update)).status_code, 402)
        for name, value in (
            ("OPENROUTER_API_KEY", ""),
            ("BENCH_BUDGET_USD", "nan"),
            ("BENCH_SERVICE_TIER", ""),
            ("BENCH_REQUEST_RESERVE_USD", "0.001"),
            ("BENCH_MAX_INPUT_BYTES", "1"),
        ):
            with patch.dict(os.environ, {name: value}):
                self.assertEqual((await self.post()).status_code, 402)
        with patch.dict(os.environ):
            del os.environ["BENCH_REQUEST_RESERVE_USD"]
            self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual(self.calls, [])

    async def test_missing_cost_retains_reservation_and_blocks_other_trials(self):
        self.response = {"usage": {"input_tokens": 10}}
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post("other")).status_code, 402)
        ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
        self.assertTrue(ledger["blocked"])
        self.assertEqual(len(ledger["pending"]), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_malformed_ledger_and_inflight_reservation_fail_closed(self):
        path = Path(os.environ["BENCH_SPEND_FILE"])
        path.write_text("{broken")
        self.assertEqual((await self.post()).status_code, 402)
        path.write_text(json.dumps({"usd": -1, "trials": {}, "pending": {}, "blocked": False}))
        self.assertEqual((await self.post()).status_code, 402)
        path.write_text(
            json.dumps(
                {
                    "usd": 0,
                    "trials": {},
                    "pending": {},
                    "blocked": False,
                    "channels": {"model": 0, "classifier": 0},
                }
            )
        )
        guard = meter.BudgetGuard()
        # A reservation that never settles is exposure, not a refusal: the
        # second request waits for it and is refused only at the deadline.
        guard.trial = 1.0  # another trial's large in-flight worst case
        self.assertIsNotNone(guard.try_acquire("pending", "model", 0.8))
        self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual(self.calls, [])

    async def test_stream_usage_with_crlf_and_redacted_errors(self):
        chunks = (
            b'data: {"type":"message_start","message":{"model":"served",'
            b'"usage":{"input_tokens":2,"cost":0.01}}}\r\n\r\n'
            b'data: {"type":"message_delta","usage":{"cost":0.1,"output_tokens":3}}\r\n\r\n'
        )
        self.upstream._transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(chunks)
            )
        )
        response = await self.post(stream=True)
        self.assertEqual(response.content, chunks)
        ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
        self.assertEqual(ledger["usd"], 0.1)
        self.assertFalse(ledger["blocked"])

    async def test_transport_failure_keeps_budget_closed(self):
        def fail(request):
            self.calls.append(request)
            raise httpx.ConnectError("offline negative control")

        self.upstream._transport = httpx.MockTransport(fail)
        self.assertEqual((await self.post()).status_code, 502)
        self.assertEqual((await self.post("retry")).status_code, 402)
        self.assertEqual(len(self.calls), 1)

    async def test_over_reservation_charge_blocks_and_records_actual(self):
        self.response = {"usage": {"cost": 0.25}}
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post("retry")).status_code, 402)
        ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
        self.assertEqual(ledger["usd"], 0.25)
        self.assertTrue(ledger["blocked"])

    async def test_diagnostics_is_price_neutral_and_forwarded_unchanged(self):
        diagnostics = {"previous_message_id": None}
        self.assertEqual((await self.post(diagnostics=diagnostics)).status_code, 200)
        self.assertEqual(json.loads(self.calls[0].content)["diagnostics"], diagnostics)

    async def test_main_output_cap_below_native_128000_is_refused(self):
        with patch.dict(os.environ, {"BENCH_MAX_OUTPUT_TOKENS": "127999"}):
            self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual((await self.post(max_tokens=128000)).status_code, 200)
        self.assertEqual(len(self.calls), 1)

    async def test_concurrent_admissions_queue_against_pending_exposure(self):
        release = asyncio.Event()
        started = []

        async def held(request):
            self.calls.append(request)
            started.append(request)
            await release.wait()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps(self.response).encode()),
            )

        self.upstream._transport = httpx.MockTransport(held)
        with patch.dict(os.environ, {"BENCH_TRIAL_BUDGET_USD": "1", "BENCH_BUDGET_USD": "0.5"}):
            first = asyncio.create_task(self.post())
            second = asyncio.create_task(self.post())
            await until(lambda: len(started) >= 2)
            # Both are dispatched together: 0.48 pending fits a 0.5 cap.
            third = await self.post()
            self.assertEqual(third.status_code, 402)
            self.assertEqual(len(self.calls), 2)
            release.set()
            self.assertEqual([(await t).status_code for t in (first, second)], [200, 200])
        ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
        self.assertEqual(ledger["pending"], {})
        self.assertAlmostEqual(ledger["usd"], 0.24)
        self.assertAlmostEqual(ledger["channels"]["model"], 0.24)
        self.assertFalse(ledger["blocked"])

    async def test_waiting_admission_dispatches_once_exposure_settles(self):
        release = asyncio.Event()

        async def held(request):
            self.calls.append(request)
            if len(self.calls) == 1:
                await release.wait()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps(self.response).encode()),
            )

        self.upstream._transport = httpx.MockTransport(held)
        with patch.dict(
            os.environ,
            {
                "BENCH_TRIAL_BUDGET_USD": "1",
                "BENCH_BUDGET_USD": "0.4",
                "BENCH_ADMISSION_WAIT_S": "5",
            },
        ):
            first = asyncio.create_task(self.post())
            await until(lambda: bool(self.calls))
            second = asyncio.create_task(self.post())
            await asyncio.sleep(0.1)
            self.assertEqual(len(self.calls), 1)  # queued, not refused
            release.set()
            self.assertEqual([(await t).status_code for t in (first, second)], [200, 200])
        self.assertEqual(len(self.calls), 2)

    async def test_unknown_costs_latch_the_ledger(self):
        for cost in (None, float("nan"), -0.01, "0.1", True):
            with self.subTest(cost=cost):
                Path(os.environ["BENCH_SPEND_FILE"]).unlink(missing_ok=True)
                self.calls.clear()
                usage = {} if cost is None else {"cost": cost}
                self.response = {"usage": usage}
                self.assertEqual((await self.post()).status_code, 200)
                self.assertEqual((await self.post("other")).status_code, 402)
                ledger = json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())
                self.assertTrue(ledger["blocked"])
                self.assertEqual(len(self.calls), 1)

    async def test_model_rows_carry_the_model_channel(self):
        self.assertEqual((await self.post()).status_code, 200)
        row = json.loads(meter.LOG.read_text().splitlines()[-1])
        self.assertEqual(row["channel"], "model")

    async def test_bridge_stream_rewrites_search_results(self):
        item = {
            "type": "function_call",
            "name": "tool_search",
            "id": "search1",
            "call_id": "call1",
            "arguments": '{"query":"read"}',
        }
        events = [
            {"type": "response.output_item.added", "item": item},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "search1",
                "delta": "hidden",
            },
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"output": [item]}},
        ]
        payload = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(payload),
                )
            )
        ) as upstream:
            with (
                patch.object(bridge, "client", upstream),
                patch.object(bridge, "LOG", Path(self.tmp.name) / "bridge.jsonl"),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=bridge.app), base_url="http://offline"
                ) as api:
                    response = await api.post("/v1/responses", json={"tools": [], "input": []})
        self.assertNotIn(b"hidden", response.content)
        self.assertNotIn(b"output_item.added", response.content)
        data = [
            json.loads(line[5:])
            for line in response.content.splitlines()
            if line.startswith(b"data:")
        ]
        self.assertEqual(data[0]["item"]["type"], "tool_search_call")
        self.assertEqual(data[1]["response"]["output"][0]["arguments"], {"query": "read"})

    def launcher_env(self, **updates: str) -> dict[str, str]:
        ca = Path(self.tmp.name) / "edge-ca.pem"
        ca.write_text("-----BEGIN CERTIFICATE-----\n")
        env = {
            "HTTPS_PROXY": "http://trial:nonce@127.0.0.1:18080",
            "SSL_CERT_FILE": str(ca),
            "REQUESTS_CA_BUNDLE": str(ca),
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "LOP_BENCH_PYTHON": sys.executable,
            "LOP_BENCH_SOURCE_ROOT": self.tmp.name,
        }
        env.update(updates)
        return env

    def run_launcher(self, launcher, env: dict[str, str]):
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(launcher.runpy, "run_module") as run,
            patch.object(launcher.runpy, "run_path") as shim,
        ):
            try:
                launcher.main()
            except SystemExit as exc:
                return run, shim, str(exc)
            return run, shim, None

    def test_launcher_requires_the_edge_proxy_environment(self):
        launcher = load("lop_launch")
        for updates in (
            {"HTTPS_PROXY": ""},
            {"HTTPS_PROXY": "http://trial:nonce@10.0.0.1:18080"},
            {"HTTPS_PROXY": "http://127.0.0.1:18080"},  # no run identity
            {"SSL_CERT_FILE": ""},
            {"REQUESTS_CA_BUNDLE": "/elsewhere.pem"},
            {"NO_PROXY": "localhost"},
        ):
            with self.subTest(updates=updates):
                run, _, error = self.run_launcher(launcher, self.launcher_env(**updates))
                self.assertIsNotNone(error)
                run.assert_not_called()

    def test_launcher_refuses_unreleased_head_by_full_sha(self):
        launcher = load("lop_launch")
        self.assertEqual(launcher.REVIEWED_REVISION, "500883ef56ca504325ffd80c5b90bf2435d00e3f")
        for head in ("07c324e9f613b378c437936ede1978c9fc0b49db", "500883ef" + "0" * 32):
            with self.subTest(head=head):
                with patch.object(launcher.subprocess, "check_output", return_value=head + "\n"):
                    run, _, error = self.run_launcher(launcher, self.launcher_env())
                self.assertIn("not reviewed revision", str(error))
                run.assert_not_called()

    def test_launcher_refuses_dirty_or_untracked_tree_in_a_real_repository(self):
        import subprocess

        launcher = load("lop_launch")
        repo = Path(self.tmp.name) / "src"
        (repo / "local_operator").mkdir(parents=True)
        (repo / "local_operator" / "__init__.py").write_text("")
        git = ["git", "-C", str(repo)]
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run([*git, "add", "."], check=True)
        # A fixture identity with no address: the kit's publication scanner
        # rejects address-shaped literals, and git accepts an empty email.
        subprocess.run(
            [*git, "-c", "user.name=fixture", "-c", "user.email=", "commit", "-qm", "fixture"],
            check=True,
        )
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        env = self.launcher_env(LOP_BENCH_SOURCE_ROOT=str(repo))
        with patch.object(launcher, "REVIEWED_REVISION", head):
            # Clean, but the interpreter imports a different local_operator.
            _, _, error = self.run_launcher(launcher, env)
            self.assertIn("does not import", str(error))
            (repo / "local_operator" / "__init__.py").write_text("x = 1\n")
            run, _, error = self.run_launcher(launcher, env)
            self.assertIn("no modifications", str(error))
            run.assert_not_called()
            subprocess.run([*git, "checkout", "-q", "--", "."], check=True)
            # An untracked module (e.g. a stray sitecustomize.py) is importable.
            (repo / "sitecustomize.py").write_text("")
            run, _, error = self.run_launcher(launcher, env)
            self.assertIn("no modifications", str(error))
            run.assert_not_called()

    def test_launcher_runs_the_unmodified_cli_without_any_shim(self):
        launcher = load("lop_launch")
        self.assertFalse((ROOT / "lop_shim").exists())
        root = (ROOT / "../../..").resolve()  # repo root: its venv imports local_operator
        env = self.launcher_env(LOP_BENCH_SOURCE_ROOT=str(root))
        with (
            patch.object(
                launcher.subprocess, "check_output", side_effect=[launcher.REVIEWED_REVISION, ""]
            ),
            patch.object(launcher.importlib.util, "find_spec") as spec,
        ):
            spec.return_value.origin = str(root / "local_operator" / "__init__.py")
            run, shim, error = self.run_launcher(launcher, env)
        self.assertIsNone(error)
        run.assert_called_once_with("local_operator.cli", run_name="__main__")
        shim.assert_not_called()

    def test_bridge_search_loading_deduplicates(self):
        tool = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        body, stats = bridge.rewrite_request(
            {
                "tools": [{"type": "tool_search"}],
                "input": [
                    {"type": "tool_search_call", "call_id": "s", "arguments": {"query": "lookup"}},
                    {"type": "tool_search_output", "call_id": "s", "tools": [tool, tool]},
                ],
            }
        )
        self.assertEqual([t["name"] for t in body["tools"]], ["tool_search", "lookup"])
        self.assertEqual(body["input"][0]["type"], "function_call")
        self.assertEqual(stats["loaded_specs"], 1)
        item = bridge._convert_item({"call_id": "s", "arguments": '{"query":"lookup"}'})
        self.assertEqual(item["arguments"], {"query": "lookup"})

    def test_catalog_prompt_attribution_and_integrity(self):
        import hashlib

        for filename in ("codex-model-catalog.json", "codex-model-catalog.eager.json"):
            model = json.loads((ROOT / filename).read_text())["models"][0]
            prompt = model["model_messages"]["instructions_template"]
            self.assertEqual(
                hashlib.sha256(prompt.encode()).hexdigest(),
                "ac8ae107a0d72fe3476b430afb161ea4e67da2e446d778aefc44828160559807",
            )
            self.assertIn("OpenAI", prompt)
            fallback = catalog.fallback_entry(model["slug"], prompt)
            self.assertEqual(fallback["model_messages"], model["model_messages"])


if __name__ == "__main__":
    unittest.main()

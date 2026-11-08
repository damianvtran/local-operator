"""Cheap offline adapter checks; fake upstreams cannot spend or launch a harness.

ASGI requests exercise real routing/rewriting/accounting, while the transport
negative controls prove rejected configurations never dispatch an upstream call.
These are source-only checks, not provider billing or native harness evidence.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
                "BENCH_REQUEST_RESERVE_USD": "0.2",
                "BENCH_INPUT_USD_PER_MILLION": "1",
                "BENCH_OUTPUT_USD_PER_MILLION": "1",
                "BENCH_MAX_INPUT_BYTES": "10000",
                "BENCH_MAX_OUTPUT_TOKENS": "1000",
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
            {"max_tokens": 1001},
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
        path.write_text(json.dumps({"usd": 0, "trials": {}, "pending": {}, "blocked": False}))
        guard = meter.BudgetGuard()
        guard.acquire("pending")
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

    def test_launcher_negative_controls_do_not_import_cli(self):
        launcher = load("lop_launch")
        with (
            patch.dict(os.environ, {"BENCH_ANTHROPIC_URL": ""}),
            patch.object(launcher.runpy, "run_module") as run,
        ):
            with self.assertRaises(SystemExit):
                launcher.main()
            run.assert_not_called()
        with (
            patch.dict(
                os.environ,
                {
                    "BENCH_ANTHROPIC_URL": "http://127.0.0.1:8787/r/trial",
                    "LOP_BENCH_PYTHON": launcher.sys.executable,
                    "LOP_BENCH_SOURCE_ROOT": self.tmp.name,
                },
            ),
            patch.object(launcher.subprocess, "check_output", return_value="unreviewed-head"),
            patch.object(launcher.runpy, "run_module") as run,
        ):
            with self.assertRaises(SystemExit):
                launcher.main()
            run.assert_not_called()

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

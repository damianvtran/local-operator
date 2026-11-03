"""Cheap offline adapter checks; fake upstreams cannot spend or launch a harness.

ASGI requests exercise real routing/rewriting/accounting, while the transport
negative controls prove rejected configurations never dispatch an upstream call.
These are source-only checks, not provider billing or native harness evidence.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
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


RATE = "0.000001"  # 1 USD per million tokens, per-token as OpenRouter states it


def fake_listing(**overrides: dict[str, str]) -> bytes:
    """A synthetic OpenRouter ``/api/v1/models`` body (per-token USD strings).

    Opus is listed upstream but is NOT a natively requested benchmark model, so
    the rate card built from this listing must leave it out.
    """
    rates = {"prompt": RATE, "completion": RATE, "input_cache_write": RATE, "input_cache_read": "0"}
    rows: dict[str, dict[str, str]] = {
        "anthropic/claude-sonnet-5.5": dict(rates),
        "anthropic/claude-haiku-5.5": dict(rates),
        "anthropic/claude-opus-5.5": dict(rates),
    }
    rows.update(overrides)
    return json.dumps({"data": [{"id": k, "pricing": v} for k, v in rows.items()]}).encode()


def write_rate_card(path: Path, listing: bytes | None = None, fetched_at: float | None = None):
    card = meter.rate_card_from_models_listing(
        fake_listing() if listing is None else listing,
        source_url="https://openrouter.ai/api/v1/models",
        fetched_at=time.time() if fetched_at is None else fetched_at,
    )
    path.write_text(json.dumps(card))
    return card


def meter_env(root: Path) -> dict[str, str]:
    """Explicit meter configuration shared by every offline meter/edge test."""
    card = root / "rate-card.json"
    write_rate_card(card)
    return {
        "OPENROUTER_API_KEY": "offline-test-key-not-a-credential",
        "BENCH_BUDGET_USD": "1",
        # A native 128000-token main call reserves ~0.137 USD at 1 USD/M:
        # one fits a 0.2 trial cap, a second after a 0.12 bill does not.
        "BENCH_TRIAL_BUDGET_USD": "0.2",
        "BENCH_RATE_CARD": str(card),
        "BENCH_MAX_INPUT_BYTES": "10000",
        "BENCH_MAX_OUTPUT_TOKENS": "128000",
        "BENCH_ADMISSION_WAIT_S": "0.3",
        "BENCH_MAX_REQUEST_S": "900",
        "BENCH_SERVICE_TIER": "default",
        "BENCH_SPEND_FILE": str(root / "spend.json"),
    }


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
        root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, meter_env(root))
        self.env.start()
        self.logpatch = patch.object(meter, "LOG", root / "requests.jsonl")
        self.logpatch.start()
        self.calls = []
        self.response = {"model": "anthropic/claude-sonnet-5.5", "usage": {"cost": 0.12}}
        self.status = 200

        def upstream(request):
            self.calls.append(request)
            return httpx.Response(
                self.status,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps(self.response).encode()),
            )

        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.clientpatch = patch.object(meter, "client", self.upstream)
        self.clientpatch.start()
        self.guards: list[Any] = []
        self.api = None
        self.serve()
        self.body = {
            "model": meter.MODEL,
            "max_tokens": 128000,
            "messages": [{"role": "user", "content": "hello"}],
        }

    def serve(self, **env: str):
        """(Re)build the guard ONCE from the environment, as the meter's startup does."""
        for name, value in env.items():
            os.environ[name] = value
        self.guard = meter.BudgetGuard()
        self.guards.append(self.guard)
        self.api = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=meter.make_app(self.guard)), base_url="http://offline"
        )

    async def asyncTearDown(self):
        if self.api is not None:
            await self.api.aclose()
        await self.upstream.aclose()
        for guard in self.guards:
            guard.close()
        self.clientpatch.stop()
        self.logpatch.stop()
        self.env.stop()
        self.tmp.cleanup()

    async def post(self, run: str = "trial", /, **updates: object):
        assert self.api is not None
        return await self.api.post(f"/r/{run}/v1/messages", json={**self.body, **updates})

    def ledger(self):
        return json.loads(Path(os.environ["BENCH_SPEND_FILE"]).read_text())

    async def test_real_asgi_route_settles_and_stops_trial(self):
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual(len(self.calls), 1)
        forwarded = json.loads(self.calls[0].content)
        self.assertFalse(forwarded["provider"]["allow_fallbacks"])
        self.assertEqual(forwarded["provider"]["max_price"], {"prompt": 1.0, "completion": 1.0})
        self.assertEqual(forwarded["service_tier"], "default")
        ledger = self.ledger()
        self.assertEqual(ledger["usd"], 0.12)
        self.assertEqual(ledger["trials"]["trial"], 0.12)
        self.assertEqual(ledger["models"], {"anthropic/claude-sonnet-5.5": 0.12})
        # F5: the cap refusal changed the harness's run, so the run is void.
        self.assertIn("admission refused", " ".join(ledger["voids"]["trial"]))
        self.assertNotIn(os.environ["OPENROUTER_API_KEY"], meter.LOG.read_text())
        self.assertNotIn("hello", meter.LOG.read_text())

    async def test_total_cap_independent_of_trial_ids(self):
        self.serve(BENCH_BUDGET_USD="0.2", BENCH_TRIAL_BUDGET_USD="1")
        self.assertEqual((await self.post("one")).status_code, 200)
        self.assertEqual((await self.post("two")).status_code, 402)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("two", self.ledger()["voids"])

    async def test_reservation_is_per_request_not_a_flat_worst_case(self):
        # D1: a 1024-token title errand reserves its own bound, far below a main call.
        title = self.guard.reservation_usd(meter.or_model(meter.MODEL), b"x" * 500, 1024)
        main = self.guard.reservation_usd(meter.or_model(meter.MODEL), b"x" * 64000, 128000)
        self.assertAlmostEqual(title, (500 + meter.FRAMING_TOKENS + 1024) / 1e6)
        self.assertAlmostEqual(main, (64000 + meter.FRAMING_TOKENS + 128000) / 1e6)
        self.serve(BENCH_TRIAL_BUDGET_USD="0.05")
        self.assertEqual((await self.post(max_tokens=1024)).status_code, 200)

    async def test_haiku_is_priced_from_the_card_and_recorded_per_model(self):
        self.response = {"model": "anthropic/claude-haiku-5.5", "usage": {"cost": 0.01}}
        response = await self.post(model="claude-haiku-5-5", max_tokens=1024)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(self.calls[0].content)["model"], "anthropic/claude-haiku-5.5")
        self.assertEqual(self.ledger()["models"], {"anthropic/claude-haiku-5.5": 0.01})

    async def test_unpriced_model_is_refused_and_voids_the_run(self):
        # D2/D3: a native failover target outside the card (opus) never dispatches.
        response = await self.post(model="claude-opus-5-5")
        self.assertEqual(response.status_code, 402)
        self.assertEqual(self.calls, [])
        reasons = " ".join(self.ledger()["voids"]["trial"])
        self.assertIn("unpriced model anthropic/claude-opus-5.5", reasons)
        # The void holds for the rest of the run, other trials unaffected.
        self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual((await self.post("other")).status_code, 200)

    async def test_upstream_overload_voids_the_run_even_when_billed(self):
        self.status = 529
        self.response = {
            "type": "error", "error": {"type": "overloaded_error"}, "usage": {"cost": 0}
        }
        self.assertEqual((await self.post()).status_code, 529)
        ledger = self.ledger()
        self.assertIn("upstream status 529", " ".join(ledger["voids"]["trial"]))
        self.assertFalse(ledger["blocked"])  # billed zero: spend is still bounded
        self.assertEqual((await self.post()).status_code, 402)  # native retry refused
        self.assertEqual(len(self.calls), 1)

    async def test_unbilled_overload_latches_with_reason(self):
        self.status = 529
        self.response = {"type": "error", "error": {"type": "overloaded_error"}}
        await self.post()
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertIn("unbilled", " ".join(ledger["voids"]["trial"]))

    def test_configuration_errors_fail_at_startup(self):
        # F6: one guard built at startup; a bad environment never serves a request.
        root = Path(self.tmp.name)
        stale = root / "stale-card.json"
        write_rate_card(stale, fetched_at=time.time() - 2 * meter.RATE_CARD_MAX_AGE_S)
        partial = root / "partial-card.json"
        card = json.loads(Path(os.environ["BENCH_RATE_CARD"]).read_text())
        partial.write_text(json.dumps({**card, "models": {}}))
        for name, value in (
            ("BENCH_BUDGET_USD", "nan"),
            ("BENCH_SERVICE_TIER", ""),
            ("BENCH_MAX_INPUT_BYTES", "0"),
            ("BENCH_MAX_REQUEST_S", "-1"),
            ("BENCH_RATE_CARD", str(stale)),
            ("BENCH_RATE_CARD", str(partial)),
        ):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaises((ValueError, KeyError)):
                    meter.BudgetGuard()
        for name in ("BENCH_RATE_CARD", "BENCH_ADMISSION_WAIT_S", "BENCH_MAX_REQUEST_S"):
            with self.subTest(missing=name), patch.dict(os.environ):
                del os.environ[name]
                with self.assertRaises((ValueError, KeyError)):
                    meter.BudgetGuard()

    def test_rate_card_is_the_exact_native_allowlist_with_provenance(self):
        listing = fake_listing()
        card = write_rate_card(Path(self.tmp.name) / "c.json", listing)
        self.assertEqual(set(card["models"]), set(meter.NATIVE_MODELS))
        self.assertEqual(card["response_sha256"], hashlib.sha256(listing).hexdigest())
        self.assertEqual(card["source"], "https://openrouter.ai/api/v1/models")
        self.assertEqual(card["models"]["anthropic/claude-sonnet-5.5"]["prompt"], 1.0)
        for bad in (
            {"anthropic/claude-haiku-5.5": {"prompt": "nan", "completion": RATE}},
            {"anthropic/claude-haiku-5.5": {"prompt": "-1", "completion": RATE}},
            {"anthropic/claude-haiku-5.5": {"completion": RATE}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                write_rate_card(Path(self.tmp.name) / "bad.json", fake_listing(**bad))
        missing = json.dumps({"data": [{"id": "anthropic/claude-sonnet-5.5",
                                        "pricing": {"prompt": RATE, "completion": RATE}}]})
        with self.assertRaises(ValueError):
            write_rate_card(Path(self.tmp.name) / "bad.json", missing.encode())

    async def test_negative_controls_never_dispatch_and_void(self):
        for update in (
            {"max_tokens": 128001},
            {"service_tier": "priority"},
            {"plugins": []},
            {"tools": [{"type": "web_search_20250305"}]},
            {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        ):
            with self.subTest(update=update):
                run = "neg-" + hashlib.sha256(repr(update).encode()).hexdigest()[:8]
                self.assertEqual((await self.post(run, **update)).status_code, 402)
                self.assertIn(run, self.ledger()["voids"])
        self.assertEqual(self.calls, [])

    async def test_missing_cost_retains_reservation_and_blocks_other_trials(self):
        self.response = {"usage": {"input_tokens": 10}}
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post("other")).status_code, 402)
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertEqual(len(ledger["pending"]), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_malformed_ledger_fails_closed(self):
        path = Path(os.environ["BENCH_SPEND_FILE"])
        path.write_text("{broken")
        self.assertEqual((await self.post()).status_code, 402)
        path.write_text(json.dumps({"usd": -1, "trials": {}, "pending": {}, "blocked": False}))
        self.assertEqual((await self.post()).status_code, 402)
        self.assertEqual(self.calls, [])

    async def test_non_stream_body_reset_after_dispatch_latches(self):
        # F1 path 2: upstream 200 JSON that resets mid-body must not leave an
        # unlatched pending reservation behind.
        class Reset(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"usage":'
                raise httpx.ReadError("reset mid-body")

        def reset(request):
            self.calls.append(request)
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=Reset())

        self.upstream._transport = httpx.MockTransport(reset)
        with contextlib.suppress(Exception):
            await self.post()
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertIn("abandoned", " ".join(ledger["voids"]["trial"]))
        self.assertEqual((await self.post("other")).status_code, 402)

    async def test_cancelled_request_latches(self):
        # F1 path 1: cancellation (shutdown) while the upstream is in flight.
        started = asyncio.Event()

        async def hang(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            started.set()
            await asyncio.sleep(30)
            return httpx.Response(504)

        self.upstream._transport = httpx.MockTransport(hang)
        task = asyncio.create_task(self.post())
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await until(lambda: self.ledger()["blocked"])
        self.assertIn("abandoned", " ".join(self.ledger()["voids"]["trial"]))

    def test_orphaned_reservation_of_a_dead_process_refuses_the_next_request(self):
        # F1 path 3: the base behaviour (orphan pending -> next request REFUSED)
        # is restored by owner liveness, without refusing live concurrency.
        code = (
            "import os, sys; sys.path.insert(0, sys.argv[1])\n"
            "import meter_proxy as m\n"
            "g = m.BudgetGuard()\n"
            "g.try_acquire('dead', 'model', 0.01, 'anthropic/claude-sonnet-5.5')\n"
            "os._exit(0)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(ROOT)], env=dict(os.environ),
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with self.assertRaises(ValueError):
            self.guard.try_acquire("next", "model", 0.01, "anthropic/claude-sonnet-5.5")
        ledger = self.ledger()
        self.assertTrue(ledger["blocked"])
        self.assertIn("orphaned reservation", " ".join(ledger["voids"]["dead"]))

    def test_live_concurrent_owner_is_not_an_orphan(self):
        other = meter.BudgetGuard()
        self.guards.append(other)
        sonnet = "anthropic/claude-sonnet-5.5"
        self.assertIsNotNone(other.try_acquire("a", "model", 0.01, sonnet))
        self.assertIsNotNone(self.guard.try_acquire("b", "model", 0.01, sonnet))
        self.assertFalse(self.ledger()["blocked"])

    def test_stale_reservation_latches_and_teardown_latches_unsettled(self):
        token = self.guard.try_acquire("slow", "model", 0.01, "anthropic/claude-sonnet-5.5")
        self.assertIsNotNone(token)
        data = self.ledger()
        data["pending"][token]["ts"] -= 10_000
        Path(os.environ["BENCH_SPEND_FILE"]).write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.guard.try_acquire("next", "model", 0.01, "anthropic/claude-sonnet-5.5")
        self.assertIn("stale reservation", " ".join(self.ledger()["voids"]["slow"]))
        # Teardown backstop on a fresh ledger: an unsettled own token latches.
        Path(os.environ["BENCH_SPEND_FILE"]).unlink()
        g = meter.BudgetGuard()
        g.try_acquire("open", "model", 0.01, "anthropic/claude-sonnet-5.5")
        g.close()
        self.assertTrue(self.ledger()["blocked"])
        self.assertIn("teardown", " ".join(self.ledger()["voids"]["open"]))

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
        ledger = self.ledger()
        self.assertEqual(ledger["usd"], 0.1)
        self.assertFalse(ledger["blocked"])
        self.assertEqual(ledger["pending"], {})

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
        self.serve(BENCH_TRIAL_BUDGET_USD="1")
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual((await self.post("retry")).status_code, 402)
        ledger = self.ledger()
        self.assertEqual(ledger["usd"], 0.25)
        self.assertTrue(ledger["blocked"])

    async def test_diagnostics_is_price_neutral_and_forwarded_unchanged(self):
        diagnostics = {"previous_message_id": None}
        self.assertEqual((await self.post(diagnostics=diagnostics)).status_code, 200)
        self.assertEqual(json.loads(self.calls[0].content)["diagnostics"], diagnostics)

    def test_main_output_cap_below_native_128000_is_refused_at_startup(self):
        with patch.dict(os.environ, {"BENCH_MAX_OUTPUT_TOKENS": "127999"}):
            with self.assertRaises(ValueError):
                meter.BudgetGuard()

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
        # Two ~0.137 reservations fit 0.3 together; a third does not.
        self.serve(BENCH_TRIAL_BUDGET_USD="1", BENCH_BUDGET_USD="0.3")
        first = asyncio.create_task(self.post())
        second = asyncio.create_task(self.post())
        await until(lambda: len(started) >= 2)
        third = await self.post("third")
        self.assertEqual(third.status_code, 402)
        self.assertEqual(len(self.calls), 2)
        release.set()
        self.assertEqual([(await t).status_code for t in (first, second)], [200, 200])
        ledger = self.ledger()
        self.assertEqual(ledger["pending"], {})
        self.assertAlmostEqual(ledger["usd"], 0.24)
        self.assertAlmostEqual(ledger["channels"]["model"], 0.24)
        self.assertFalse(ledger["blocked"])
        self.assertIn("admission refused", " ".join(ledger["voids"]["third"]))
        self.assertNotIn("trial", ledger["voids"])

    async def test_waiting_admission_dispatches_once_exposure_settles(self):
        release = asyncio.Event()
        self.response = {"model": "anthropic/claude-sonnet-5.5", "usage": {"cost": 0.01}}

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
        self.serve(BENCH_TRIAL_BUDGET_USD="1", BENCH_BUDGET_USD="0.2", BENCH_ADMISSION_WAIT_S="5")
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
                ledger = self.ledger()
                self.assertTrue(ledger["blocked"])
                self.assertEqual(len(self.calls), 1)

    async def test_model_rows_carry_the_model_channel(self):
        self.assertEqual((await self.post()).status_code, 200)
        row = json.loads(meter.LOG.read_text().splitlines()[-1])
        self.assertEqual(row["channel"], "model")
        self.assertEqual(row["priced_model"], "anthropic/claude-sonnet-5.5")
        self.assertGreater(row["reserve_usd"], 0.128)

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
            # F2: urllib/httpx/requests give lowercase names precedence, even empty.
            {"https_proxy": ""},
            {"https_proxy": "http://other:mismatch@127.0.0.1:9"},
            {"http_proxy": "http://other:mismatch@127.0.0.1:9"},
            {"no_proxy": "*"},
            {"no_proxy": "127.0.0.1,localhost,::1,example.com"},
            {"ALL_PROXY": "socks5://127.0.0.1:9"},
            {"all_proxy": "socks5://127.0.0.1:9"},
            {"HTTP_PROXY": "http://other:mismatch@127.0.0.1:9"},
        ):
            with self.subTest(updates=updates):
                run, _, error = self.run_launcher(launcher, self.launcher_env(**updates))
                self.assertIsNotNone(error)
                run.assert_not_called()

    def test_launcher_accepts_the_edge_child_environment_verbatim(self):
        # Exactly what lop_edge.EdgeServer.child_env() exports (both spellings).
        launcher = load("lop_launch")
        proxy = "http://trial:nonce@127.0.0.1:18080"
        env = self.launcher_env(
            https_proxy=proxy, HTTP_PROXY=proxy, http_proxy=proxy,
            no_proxy="127.0.0.1,localhost,::1",
        )
        with patch.dict(os.environ, env, clear=True):
            launcher._require_edge_environment()

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

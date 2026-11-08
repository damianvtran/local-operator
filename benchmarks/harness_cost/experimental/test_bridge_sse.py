"""Exercise both real ASGI bridges without sockets or a model.

Why: helper-only tests cannot catch a route bypassing SSE parsing, and transport
chunks need not align with lines, JSON tokens, or UTF-8 characters. Standalone
file imports deliberately match the coding driver's script launch context.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def frame(data, ending=b"\n", multiline=False):
    payload = json.dumps(data, ensure_ascii=False).encode()
    if multiline:
        # A JSON newline between members is legal; SSE joins data lines with LF.
        payload = payload.replace(b', "', b',\n"', 1)
    return ending.join(b"data: " + line for line in payload.split(b"\n")) + ending * 2


class BridgeSSETests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bridges = []
        for name, relative in (
            ("search", "codex_toolsearch_bridge.py"),
            ("coding", "coding/codex_bridge.py"),
        ):
            spec = importlib.util.spec_from_file_location("sse_test_" + name, ROOT / relative)
            # A missing loader is a broken fixture, not a bridge regression.
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load bridge fixture: {relative}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            await module.client.aclose()
            setattr(module, "LOG", Path(self.temp.name) / (name + ".jsonl"))
            self.bridges.append(module)

    async def asyncTearDown(self):
        for module in self.bridges:
            await module.client.aclose()
        self.temp.cleanup()

    async def request(self, module, wire, split=False, status=200):
        stream = Chunks([bytes([byte]) for byte in wire] if split else [wire])

        def upstream(request):
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertEqual(json.loads(request.content)["tools"][0]["type"], "function")
            return httpx.Response(
                status,
                headers={"content-type": "text/event-stream", "x-public-test": "preserved"},
                stream=stream,
            )

        await module.client.aclose()
        module.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=module.app), base_url="http://bridge.test"
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={"tools": [{"type": "tool_search"}], "input": [], "stream": True},
            )
        self.assertTrue(stream.closed)
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.headers["x-public-test"], "preserved")
        return response.content

    async def test_added_done_completed_all_line_endings(self):
        item = {
            "type": "function_call",
            "id": "search-id",
            "call_id": "call-id",
            "name": "tool_search",
            "arguments": '{"query":"café"}',
        }
        usage = {"input_tokens": 7, "output_tokens": 3}
        for module in self.bridges:
            for ending in (b"\n", b"\r\n", b"\r"):
                for split in (False, True):
                    with self.subTest(bridge=module.__name__, ending=ending, split=split):
                        wire = (
                            b"".join(
                                frame(event, ending, multiline=split)
                                for event in [
                                    {"type": "response.output_item.added", "item": item},
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": "search-id",
                                        "delta": "hidden",
                                    },
                                    {
                                        "type": "response.function_call_arguments.done",
                                        "item_id": "search-id",
                                        "arguments": item["arguments"],
                                    },
                                    {"type": "response.output_item.done", "item": item},
                                    {
                                        "type": "response.completed",
                                        "response": {"output": [item], "usage": usage},
                                    },
                                ]
                            )
                            + b"data: [DONE]"
                            + ending * 2
                        )
                        result = await self.request(module, wire, split)
                        self.assertNotIn(b"response.output_item.added", result)
                        self.assertNotIn(b"response.function_call_arguments.", result)
                        self.assertNotIn(b'"function_call"', result)
                        self.assertEqual(result.count(b'"tool_search_call"'), 2)
                        self.assertIn(b'"query": "caf\\u00e9"', result)
                        self.assertIn(json.dumps(usage).encode(), result)
                        self.assertTrue(result.endswith(b"data: [DONE]" + ending * 2))
                        row = json.loads(module.LOG.read_text().splitlines()[-1])
                        self.assertEqual(row["search_calls_out"], 1)

    async def test_mixed_framing_incomplete_and_nonsearch(self):
        search = {
            "type": "function_call",
            "id": "search-id",
            "call_id": "call-id",
            "name": "tool_search",
            "arguments": '{"query":"echo"}',
        }
        ordinary = {**search, "id": "ordinary-id", "name": "echo"}
        untouched = (
            frame({"type": "response.output_item.added", "item": ordinary}, b"\r")
            + frame(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "ordinary-id",
                    "delta": "visible",
                },
                b"\r\n",
            )
            + b"event: ping\ndata\n\n"
            + b"data: []\r\n\r\n"
        )
        completed = frame(
            {
                "type": "response.incomplete",
                "response": {
                    "output": [search, ordinary],
                    "usage": {"output_tokens": 4},
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
            b"\r",
        )
        for module in self.bridges:
            with self.subTest(bridge=module.__name__):
                result = await self.request(module, untouched + completed, True)
                self.assertTrue(result.startswith(untouched))
                converted = json.loads(result[len(untouched) :].split(b"data: ", 1)[1])
                self.assertEqual(converted["response"]["output"][0]["type"], "tool_search_call")
                self.assertEqual(converted["response"]["output"][1], ordinary)
                self.assertEqual(converted["response"]["usage"], {"output_tokens": 4})
                self.assertEqual(
                    converted["response"]["incomplete_details"], {"reason": "max_output_tokens"}
                )

    async def test_passthrough_errors_usage_and_flush(self):
        for module in self.bridges:
            for ending in (b"\n", b"\r\n", b"\r"):
                with self.subTest(bridge=module.__name__, ending=ending):
                    wire = (
                        b": keepalive"
                        + ending * 2
                        + frame({"type": "response.usage", "usage": {"input_tokens": 11}}, ending)
                        + frame(
                            {"type": "error", "error": {"message": "public café"}}, ending, True
                        )
                        + b"data: not-json"
                        + ending * 2
                        + b"data: [DONE]"
                        + ending * 2
                        + b'data: {"type": "error", "error": "tail"}'
                    )
                    self.assertEqual(await self.request(module, wire, True, 429), wire)

    async def test_coding_bridge_never_forwards_a_web_search_tool(self):
        # S9e: codex-best disables Codex web search in config; the bridge is
        # the last hop before LiteLLM, so it refuses (never strips) any
        # server-side web search tool at $0 and forwards nothing upstream.
        module = self.bridges[1]
        forwarded = []

        def upstream(request):
            forwarded.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})

        await module.client.aclose()
        module.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=module.app), base_url="http://bridge.test"
        ) as client:
            for tool in ({"type": "web_search"}, {"type": "web_search_preview"},
                         {"type": "web_search", "external_web_access": False}):
                with self.subTest(tool=tool):
                    response = await client.post("/v1/responses", json={
                        "tools": [{"type": "function", "name": "shell"}, tool],
                        "input": [], "stream": True})
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json()["reason"], "server-side web search tool")
            self.assertEqual(forwarded, [])
            # A function tool merely NAMED web_search is a client tool: forwarded.
            response = await client.post("/v1/responses", json={
                "tools": [{"type": "function", "name": "web_search"}], "input": []})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(forwarded), 1)
        log = [json.loads(line) for line in module.LOG.read_text().splitlines()]
        self.assertEqual([row["status"] for row in log], [400, 400, 400, 200])


if __name__ == "__main__":
    unittest.main()

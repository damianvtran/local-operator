"""Exercise final-wire forwarding with an in-memory ASGI client, never a meter.

Why: a cache helper test alone cannot show where a forwarding adapter mutates
its request. Constraints: synthetic public text, fake upstream, scratch-only
metadata, no model/server process, socket, auth lookup or dependency install.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from starlette.testclient import TestClient
import cache_forwarder as f
from cache_namespace import namespace_tag


class FakeMeter:
    def __init__(self):
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return httpx.Response(200, content=b'{"usage":{}}', headers={"content-type": "application/json"})


class ForwarderTests(unittest.TestCase):
    def setUp(self):
        root = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
        if not root:
            raise RuntimeError("Set LOCAL_OPERATOR_SCRATCHPAD for temporary test metadata")
        self.tmp = tempfile.TemporaryDirectory(dir=root, prefix="cache-forward-test-")
        self.meter = FakeMeter()
        self.log = Path(self.tmp.name) / "metadata.jsonl"
        self.patches = [patch.object(f, "client", self.meter), patch.object(f, "LOG", self.log)]
        for p in self.patches:
            p.start()
        self.client = TestClient(f.app)

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_final_body_precedes_meter_and_log_is_metadata_only(self):
        body = {"tools": [{"name": "echo", "description": "Echo public text", "input_schema": {"type": "object"}}],
                "system": "Synthetic public rule", "messages": []}
        response = self.client.post("/r/cachedev__unit_A/v1/messages", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.meter.calls), 1)
        url, args = self.meter.calls[0]
        self.assertTrue(url.endswith("/r/cachedev__unit_A/v1/messages"))
        self.assertIn(namespace_tag("cachedev__unit_A"), args["json"]["tools"][0]["description"])
        self.assertEqual(body["tools"][0]["description"], "Echo public text")
        record = json.loads(self.log.read_text())
        self.assertIn("after_sha256", record)
        self.assertNotIn("Synthetic public rule", self.log.read_text())
        self.assertNotIn("Echo public text", self.log.read_text())

    def test_invalid_route_and_layout_never_dispatch(self):
        for route, body in [("headline__bad", {}), ("cachedev__bad", {"tools": [{"name": "missing-description"}]})]:
            response = self.client.post(f"/r/{route}/v1/messages", json=body)
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.meter.calls, [])
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()

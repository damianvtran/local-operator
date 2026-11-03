"""Exercise the actual optional ASGI guard lifecycle without upstream calls.

Why: pure reservation tests did not construct Starlette, so removal of its
legacy on_shutdown argument escaped detection. These tests enter the real
application lifespan, inspect HTTP responses and prove ledger-lock release.
Constraints: synthetic ledgers, no credentials, no sockets, no paid dispatch;
httpx and Starlette are optional development dependencies, not core analysis.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import budget_guard


class GuardASGITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        values = {
            "CODING_RESERVATION_LEDGER": str(root / "ledger.json"),
            "CODING_TOTAL_BUDGET_USD": "5",
            "CODING_TRIAL_BUDGET_USD": "1",
            "CODING_INPUT_USD_PER_M": "2.5",
            "CODING_OUTPUT_USD_PER_M": "10",
            "CODING_PRICED_MODELS": "claude-sonnet-5-5",
            "CODING_METER_URL": "http://127.0.0.1:1",
            "CODING_BRIDGE_URL": "http://127.0.0.1:1",
            "CODING_CURRENT_RUN": str(root / "current.txt"),
        }
        self.env = patch.dict(os.environ, values)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_real_lifespan_health_and_lock_reuse(self):
        from starlette.testclient import TestClient

        for _ in range(2):
            app = budget_guard.create_app()
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"ok": True, "reserved_usd": 0})
                self.assertFalse(app.state.ledger_owner.closed)
            self.assertTrue(app.state.ledger_owner.closed)

    def test_unpriced_route_and_invalid_payload_refused_before_dispatch(self):
        from starlette.testclient import TestClient

        app = budget_guard.create_app()
        with TestClient(app) as client:
            response = client.post("/unexpected", json={})
            self.assertEqual(response.status_code, 403)
            response = client.post("/r/coding__probe__task__r1/v1/messages", json={})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(client.get("/health").json()["reserved_usd"], 0)
        self.assertTrue(app.state.ledger_owner.closed)


if __name__ == "__main__":
    unittest.main()

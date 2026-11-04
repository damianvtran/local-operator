"""Offline negative controls for publication gates and durable budget accounting.

WHY: these stdlib-only tests must not bootstrap HTTP/native dependencies or send
requests. They prove selected fail-closed properties, NOT Linux runtime, Docker,
model translation or a paid pilot. Real-path evidence is deliberately absent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import budget_guard
import driver


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "reservations.json"
        self.run_id = "coding__omp__tasktest__r60"

    def budget(self, total: float = 1, trial: float = 0.6):
        return budget_guard.Budget(self.ledger, total, trial, 4, 10, ["priced-model"])

    def raw(self, **kwargs):
        return json.dumps({"model": "priced-model", "max_tokens": 20000, **kwargs}).encode()

    def test_trial_cap_is_independent_of_total(self):
        guard = self.budget()
        guard.reserve(self.run_id, self.raw())
        guard.reserve(self.run_id, self.raw())
        with self.assertRaises(OverflowError):
            guard.reserve(self.run_id, self.raw())
        guard.reserve("coding__dsh__tasktest__r60", self.raw())

    def test_total_cap_includes_every_trial_and_restart(self):
        guard = self.budget(total=0.5, trial=0.5)
        guard.reserve(self.run_id, self.raw())
        other = self.budget(total=0.5, trial=0.5)
        other.reserve("coding__dsh__tasktest__r60", self.raw())
        with self.assertRaises(OverflowError):
            other.reserve("coding__omp__tasktest__r61", self.raw())

    def test_missing_or_invalid_caps_fail_closed(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.budget(total=value)

    def test_invalid_or_unpriced_requests_never_reserve(self):
        guard = self.budget()
        for raw in (
            self.raw(model="other"),
            self.raw(max_tokens=None),
            self.raw(max_tokens=-1),
            self.raw(max_tokens=True),
            self.raw(max_tokens=1.5),
            b"broken",
            b"null",
            self.raw(messages=[{"type": "image", "source": {"url": "example"}}]),
        ):
            with self.subTest(raw=raw), self.assertRaises((ValueError, AttributeError)):
                guard.reserve(self.run_id, raw)
        self.assertFalse(self.ledger.exists())
        with self.assertRaises(ValueError):
            guard.reserve("other-lane", self.raw())

    def test_ledger_corruption_is_not_zero_spend(self):
        for raw in (
            '{"broken":',
            '{"coding__omp__tasktest__r60": -1}',
            '{"coding__omp__tasktest__r60": NaN}',
            "[]",
        ):
            self.ledger.write_text(raw)
            with self.assertRaises(ValueError):
                self.budget()

    def test_interrupted_reservation_and_wrong_tier_fail_closed(self):
        self.ledger.with_suffix(".json.pending").write_text("{}")
        with self.assertRaises(ValueError):
            self.budget()
        self.ledger.with_suffix(".json.pending").unlink()
        with patch.dict(os.environ, {"CODING_BILLING_TIER": "explicit-tier"}):
            with self.assertRaises(ValueError):
                self.budget().reserve(self.run_id, self.raw(service_tier="priority"))
        self.assertFalse(self.ledger.exists())

    def test_disk_thresholds(self):
        with patch.object(driver, "free_mb", return_value=21 * 1024):
            driver.check_space(self.root, 1024)
            with self.assertRaises(ValueError):
                driver.check_space(self.root, 2 * 1024**3)
        with patch.object(driver, "free_mb", return_value=21 * 1024 - 1):
            with self.assertRaises(ValueError):
                driver.check_space(self.root, 0)

    def test_frozen_digest_and_snapshot(self):
        path = self.root / "freeze.json"
        path.write_text('{"frozen": true}')
        self.assertTrue(driver.frozen(path, driver.digest(path))["frozen"])
        with self.assertRaises(ValueError):
            driver.frozen(path, "0" * 64)
        subset = json.loads((driver.HERE / "subset.json").read_text())
        self.assertFalse(subset["complete"])
        self.assertNotIn("python", subset)
        self.assertNotIn("min_free_mb_during_sweep", subset)
        self.assertTrue(all("oracle_reward" not in t for t in subset["tasks"]))
        self.assertEqual(len(subset["tasks"]), 21)
        self.assertEqual(subset["target_count"], 30)
        self.assertTrue((driver.PUBLIC_LOADOUT / "public_mcp_schemas.json").is_file())
        self.assertEqual(driver.HARNESS["local-operator"][1], "500883ef-v0.94.22")
        self.assertEqual(driver.SOURCE_REVISION, "500883ef56ca504325ffd80c5b90bf2435d00e3f")

    def test_coding_lop_path_has_no_endpoint_shim_and_refuses_the_candidate(self):
        # The Harbor path must not keep a second, rebinding launcher for 07c324e9.
        self.assertFalse((driver.HERE / "lop_shim.py").exists())
        launcher = (driver.HERE / "lop_launch.py").read_text()
        self.assertNotIn("lop_shim", launcher)
        self.assertNotIn("BENCH_ANTHROPIC_URL", launcher)
        self.assertIn("500883ef56ca504325ffd80c5b90bf2435d00e3f", launcher)
        adapters = (driver.HERE / "adapters.py").read_text()
        self.assertNotIn("lop_shim.py", adapters)
        self.assertNotIn("07c324e9", adapters + launcher + (driver.HERE / "driver.py").read_text())

    def test_paid_lop_rejected_even_with_claimed_attestations(self):
        task_root = self.root / "tasks"
        task = task_root / "task0"
        task.mkdir(parents=True)
        (task / "task.toml").write_text('[environment]\ndocker_image="example:tag"\n')
        subset = self.root / "subset.json"
        subset.write_text(
            json.dumps(
                {
                    "complete": True,
                    "target_count": 30,
                    "tasks": [
                        {
                            "name": f"task{i}",
                            "task8": f"task{i}",
                            "task_checksum": "checksum",
                            "image_digest": "sha256:" + "a" * 64,
                            "hub_tag_full_size_bytes": 1,
                        }
                        for i in range(30)
                    ],
                }
            )
        )
        args = argparse.Namespace(
            total_budget_usd=1.0,
            trial_budget_usd=0.5,
            input_usd_per_m=4.0,
            output_usd_per_m=10.0,
            billing_tier="explicit-tier",
            model="priced-model",
            tasks="task0",
            harnesses="local-operator",
            rep=60,
            meter_url="http://localhost:8787",
            translator_url="http://localhost:4000",
            container_guard_url="http://host.docker.internal:8788",
            output=self.root / "out",
            preregistration="prereg",
            preregistration_sha256="p" * 64,
            runtime_attestation="runtime",
            runtime_sha256="r" * 64,
            toolwire_attestation="wire",
            toolwire_sha256="w" * 64,
            oracle_attestation="oracle",
            oracle_sha256="o" * 64,
            subset=subset,
            task_root=task_root,
        )
        prereg = {
            "frozen": True,
            "billing_tier": args.billing_tier,
            "model": args.model,
            "harnesses": ["local-operator"],
            "tasks": ["task0"],
            "rep": 60,
            "total_budget_usd": 1.0,
            "trial_budget_usd": 0.5,
            "input_usd_per_m": 4.0,
            "output_usd_per_m": 10.0,
            "runtime_sha256": args.runtime_sha256,
            "toolwire_sha256": args.toolwire_sha256,
            "subset_sha256": driver.digest(subset),
            "oracle_sha256": args.oracle_sha256,
            "catalog_sha256": driver.digest(driver.HERE / "codex-model-catalog.json"),
            "source_sha256": {p.name: driver.digest(p) for p in driver.HERE.glob("*.py")},
            "task_files": {"task0": {"task.toml": driver.digest(task / "task.toml")}},
        }
        wire = {
            "verified": True,
            "tools_observed": True,
            "preregistered_tools_observed": True,
            "text_only": True,
            "runtime_sha256": args.runtime_sha256,
            "model": args.model,
            "output_limit_enforced": True,
            "billing_tier_enforced": True,
            "guard_route_verified": True,
            "models": [args.model],
            "billing_tier": args.billing_tier,
            "schema_sha256": driver.digest(driver.PUBLIC_LOADOUT / "public_mcp_schemas.json"),
        }
        oracle = {
            "frozen": True,
            "verified": True,
            "subset_sha256": driver.digest(subset),
            "tasks": {
                "task0": {
                    "reward": 1,
                    "image_digest": "sha256:" + "a" * 64,
                    "task_checksum": "checksum",
                }
            },
        }
        with patch.object(driver, "frozen", side_effect=[prereg, {"frozen": True}, wire, oracle]):
            # No container S4 record: the hold refuses before reading anything.
            with self.assertRaisesRegex(ValueError, "HARDBLOCK PAID CODING"):
                driver.preflight(args)

    def test_entrypoints_refuse_implicit_dispatch(self):
        for script in ("driver.py", "oracle_sweep.py"):
            result = subprocess.run(
                [sys.executable, str(driver.HERE / script)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)

    def test_cost_rows_never_treat_missing_usage_as_free(self):
        path = self.root / "meter.jsonl"
        path.write_text(json.dumps({"run_id": self.run_id, "usage": {"cost": 0.2}}) + "\n")
        self.assertEqual(driver.wire_rows(path, self.run_id)[0]["usage"]["cost"], 0.2)
        path.write_text(json.dumps({"run_id": self.run_id, "usage": {}}) + "\n")
        with self.assertRaises(ValueError):
            driver.wire_rows(path, self.run_id)


if __name__ == "__main__":
    unittest.main()

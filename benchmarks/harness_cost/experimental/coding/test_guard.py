"""Offline negative controls for publication gates and dispatcher refusals.

Budget accounting is the external meter's (meter_proxy, tested with it): the
lane-local reservation guard was removed in R6 F19, so its tests went with it.

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

import driver


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = "coding__omp__tasktest__r60"

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

    def _paid_fixture(self):
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
            meter_current_run_file=self.root / "current_run",
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
        return args, prereg, wire, oracle

    def test_paid_lop_rejected_even_with_claimed_attestations(self):
        args, prereg, wire, oracle = self._paid_fixture()
        with patch.object(driver, "frozen", side_effect=[prereg, {"frozen": True}, wire, oracle]):
            # No container S4 record: the hold refuses before reading anything.
            with self.assertRaisesRegex(ValueError, "HARDBLOCK PAID CODING"):
                driver.preflight(args)

    def test_toolwire_harness_set_must_equal_the_preregistration(self):
        # S9d: an attestation covering other harnesses than the preregistration
        # attests a different run; both directions and duplicates refuse.
        args, prereg, wire, oracle = self._paid_fixture()
        for harnesses in (None, [], ["omp"], ["local-operator", "omp"],
                          ["local-operator", "local-operator"]):
            with self.subTest(harnesses=harnesses), \
                    patch.object(driver, "block_paid_coding"), \
                    patch.object(driver, "check_meter_caps"), \
                    patch.object(driver, "frozen", side_effect=[
                        prereg, {"frozen": True}, {**wire, "harnesses": harnesses}, oracle]):
                with self.assertRaisesRegex(ValueError, "toolwire harness set"):
                    driver.preflight(args)
        # The matching set passes this check and stops at a later gate.
        with patch.object(driver, "block_paid_coding"), \
                patch.object(driver, "check_meter_caps"), \
                patch.object(driver, "frozen", side_effect=[
                    prereg, {"frozen": True}, {**wire, "harnesses": ["local-operator"]}, oracle]):
            with self.assertRaises(ValueError) as caught:
                driver.preflight(args)
            self.assertNotIn("toolwire harness set", str(caught.exception))

    def test_codex_best_config_disables_server_side_web_search(self):
        # S9e: LiteLLM 1.106.6 turns Codex's Responses `web_search` tool into
        # Anthropic's `web_search_20250305` server tool, which the meter
        # (rightly) refuses as unpriced. No other arm sends a server tool, so
        # codex-best runs with Codex 0.147's top-level `web_search = "disabled"`.
        config = driver.codex_config("http://bench-gate:8789/v1")
        self.assertEqual(config["web_search"], "disabled")
        self.assertEqual(config["model_provider"], "bench")
        self.assertEqual(config["model_providers"]["bench"]["base_url"],
                         "http://bench-gate:8789/v1")
        self.assertEqual(config["model_providers"]["bench"]["wire_api"], "responses")
        self.assertEqual(config["model_catalog_json"],
                         "/tmp/bench-support/codex-model-catalog.json")
        self.assertNotIn("tools", config)  # no per-tool override shadows the mode

    def test_hermes_identity_is_the_verified_commit_and_its_frozen_tags(self):
        # S9d: Hermes' CLI says "vunknown" and its dist metadata 0.0.0, so the
        # version comes from the verified source commit plus the captured tags.
        commit = "f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab"
        self.assertEqual(driver.HARNESS["hermes"][1], "0.22.6")
        self.assertEqual(driver.HERMES_SOURCE_REVISION, commit)
        evidence = json.loads((driver.HERE / driver.HERMES_TAG_EVIDENCE).read_text())
        self.assertIn(f"{commit}\trefs/tags/v0.22.6^{{}}", evidence["ls_remote_lines"])
        self.assertIn(f"{commit}\trefs/tags/rc.3-v0.22.6^{{}}", evidence["ls_remote_lines"])
        # S9e: the exact shape of the real source-verification.json record:
        # missing/mismatched/unsupported are LISTS of paths, and the tree has
        # exactly the pinned 22066 tracked files (R13 M1).
        verified = {"commit": commit, "checked_files": 22066, "missing": [], "mismatched": [],
                    "unsupported_git_entries": [], "declared_eol_normalized": ["a.bat"],
                    "all_checked_match": True}
        self.assertEqual(driver.HERMES_CHECKED_FILES, 22066)
        self.assertEqual(driver.hermes_version({"hermes_tracked_source": verified}), "0.22.6")
        for bad in ({"commit": "0" * 40}, {"all_checked_match": False},
                    {"mismatched": ["x.py"]}, {"missing": ["a", "b"]},
                    {"unsupported_git_entries": ["sub"]},
                    # The pre-S9e integer shape is not the record and refuses.
                    {"missing": 0}, {"mismatched": 0}, {"unsupported_git_entries": 0},
                    {"missing": None}, {"missing": ""},
                    {"checked_files": 0}, {"checked_files": 22065}, {"checked_files": 22067},
                    {"checked_files": "22066"}, {"checked_files": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                driver.hermes_version({"hermes_tracked_source": {**verified, **bad}})
        with self.assertRaises(ValueError):
            driver.hermes_version({})
        # Tag evidence that no longer maps the commit to v0.22.6 refuses.
        lines = [line for line in evidence["ls_remote_lines"] if "refs/tags/v0.22.6" not in line]
        with self.assertRaisesRegex(ValueError, "tag evidence"):
            driver.hermes_version({"hermes_tracked_source": verified},
                                  evidence={**evidence, "ls_remote_lines": lines})
        # Preflight's runtime record must carry that commit, not just the string.
        driver.check_source_revision("hermes", {"source_revision": commit})
        for record in ({}, {"source_revision": "0" * 40}):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "Hermes"):
                driver.check_source_revision("hermes", record)
        driver.check_source_revision("omp", {})  # other harnesses keep their version check

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

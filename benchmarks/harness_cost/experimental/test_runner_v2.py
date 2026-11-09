"""Zero-paid integration/config regression tests for runner_v2.

WHY: exercise all seven native config paths, public-only loadout isolation,
Codex-best routing, duplicate instruction prevention and fail-closed guards
without invoking any harness/model, installing dependencies, or touching shared
services/templates. Captures below are SYNTHETIC unit fixtures; they are never
exported as runtime wire attestations. A live legacy-replay refusal is tested
separately by the operator-facing preflight command, not by this test suite.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runner_v2 as r


class V2Tests(unittest.TestCase):
    """Synthetic fixtures isolate paths/credentials; no paid execution path is called."""

    def setUp(self):
        self.real = r.HERE
        self.public = r.PUBLIC_LOADOUT
        scratch = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
        if not scratch:
            raise RuntimeError("Set LOCAL_OPERATOR_SCRATCHPAD to a session-owned scratch directory")
        self.tmp = tempfile.TemporaryDirectory(prefix="v2-tests-", dir=scratch)
        self.root = Path(self.tmp.name)
        self.patches = [
            patch.object(r, "HERE", self.root),
            patch.object(r, "SCRATCH", self.root / "outside-home"),
            patch.object(r, "PUBLIC_LOADOUT", self.root),
            patch.object(r, "TASKS_FILE", self.root / "tasks.json"),
            patch.object(r, "TASK_CACHE", self.root / "cache"),
        ]
        for p in self.patches:
            p.start()
        for name in (
            "public_mcp_schemas.json",
            "public_user_instructions.md",
            "codex-model-catalog.json",
            "public_skills.sha256",
            "tasks.json",
            "codex_toolsearch_bridge.py",
            "lop_launch.py",
            "lop_edge.py",
            "meter_proxy.py",
            "runner_v2.py",
        ):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            if name == "tasks.json":
                (self.root / name).write_text("[]\n")
            else:
                source = self.public if name.startswith("public_") else self.real
                shutil.copy(source / name, self.root / name)
        shutil.copytree(self.real / "radient_meter", self.root / "radient_meter",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / "task-manifest.json").write_text("{}\n")
        (self.root / "model_metadata.json").write_text(
            json.dumps(
                {
                    "id": "anthropic/claude-sonnet-5.5",
                    "context_length": 1000000,
                    "top_provider": {"max_completion_tokens": 128000},
                }
            )
        )
        self.runtime: r.RuntimeIdentity = {
            "binary": sys.executable,
            "sha256": "synthetic-unit-fixture",
            "entry": "fixture-bin.js",
        }
        self.home = self.root / "home"
        self.home.mkdir()
        for rel in r.INSTRUCTION_PATHS.values():
            p = self.home / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((self.root / "public_user_instructions.md").read_bytes())

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_external_task_preflight_refuses_before_dispatch(self):
        from types import SimpleNamespace

        with (
            patch.object(r, "resolve_program", return_value=sys.executable),
            patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=2)) as call,
        ):
            with self.assertRaisesRegex(ValueError, "External task preflight failed"):
                r.validate_external_task({"id": "unit-public-id"})
        args = call.call_args
        self.assertIn("--preflight", args.args[0])
        self.assertEqual(args.kwargs["env"]["BENCH_TASKS_FILE"], str(r.TASKS_FILE))
        self.assertEqual(args.kwargs["env"]["BENCH_TASK_CACHE"], str(r.TASK_CACHE))

    def _evidence(self, **overrides: object) -> Path:
        """A synthetic S4/S5 record bound to the current hashes and a real artifact."""
        evidence = self.root / "evidence" / "LOP-EDGE-EVIDENCE.md"
        evidence.parent.mkdir(exist_ok=True)
        evidence.write_text("synthetic S4/S5 evidence body\n")
        record = {
            "s4_s5_passed": True,
            "lop_head": r.LOP_HEAD,
            "files": r.metering_file_hashes(),
            "evidence_path": evidence.name,
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            **overrides,
        }
        path = evidence.parent / "lop-edge-evidence.json"
        path.write_text(json.dumps(record))
        return path

    def test_paid_lop_requires_evidence_bound_to_exact_runner_meter_edge(self):
        task: r.Task = {"id": "fixture", "_domain": "", "config": [], "instruction": "",
                        "evaluator": {"result": {}}}
        with (
            patch.object(r, "prepare", side_effect=AssertionError("must not prepare")),
            patch.object(r, "radient_token", side_effect=AssertionError("must not access auth")),
        ):
            with patch.dict(os.environ, {"BENCH_LOP_EDGE_EVIDENCE": ""}):
                with self.assertRaisesRegex(ValueError, "S4/S5 evidence"):
                    r.execute("local-operator", "v2dev__unit", "clean", 40, task,
                              self.runtime, {})
            stale = {**r.metering_file_hashes(), "meter_proxy.py": "0" * 64}
            for bad in (self._evidence(files=stale), self._evidence(s4_s5_passed=False),
                        self._evidence(lop_head="07c324e9")):
                with self.subTest(bad=bad.read_text()[:60]), patch.dict(
                    os.environ, {"BENCH_LOP_EDGE_EVIDENCE": str(bad)}
                ):
                    with self.assertRaisesRegex(ValueError, "S4/S5 evidence"):
                        r.execute("local-operator", "v2dev__unit", "clean", 40, task,
                                  self.runtime, {})

    def test_evidence_preflight_accepts_exact_hashes(self):
        record = self._evidence()
        expected = json.loads(record.read_text())["evidence_sha256"]
        with patch.dict(os.environ, {"BENCH_LOP_EDGE_EVIDENCE": str(record)}):
            self.assertEqual(r.lop_edge_preflight()["lop_edge_evidence_sha256"], expected)

    def test_evidence_hash_scope_covers_launcher_and_classifier(self):
        # R4 F10: the guard and the classifier settlement are what S4/S5 proved.
        names = set(r.metering_file_hashes())
        self.assertTrue({"runner_v2.py", "meter_proxy.py", "lop_edge.py", "lop_launch.py"} <= names)
        expected = {f"radient_meter/{p.name}" for p in (self.root / "radient_meter").glob("*.py")}
        self.assertTrue(expected and expected <= names)
        before = r.metering_file_hashes()
        (self.root / "radient_meter" / "classifier.py").write_text("# changed\n")
        self.assertNotEqual(r.metering_file_hashes(), before)

    def test_evidence_record_must_bind_a_real_evidence_file(self):
        # R4 F11: a self-attested record (zeros digest, no artifact) is refused.
        cases = {
            "zeros": self._evidence(evidence_sha256="0" * 64),
            "missing file": self._evidence(evidence_path="absent.md"),
            "escapes dir": self._evidence(evidence_path="../outside.md"),
            "no path": self._evidence(evidence_path=None),
        }
        (self.root / "outside.md").write_text("synthetic S4/S5 evidence body\n")
        for name, record in cases.items():
            with self.subTest(name), patch.dict(
                os.environ, {"BENCH_LOP_EDGE_EVIDENCE": str(record)}
            ):
                with self.assertRaisesRegex(ValueError, "S4/S5 evidence"):
                    r.lop_edge_preflight()

    def test_lop_public_full_fence_admits_mcp_like_every_fenced_harness(self):
        # R4 F12: the ports execute() builds, compared across harnesses.
        with patch.object(r, "PROXY", "http://127.0.0.1:8787"), patch.object(
            r, "MCP", "http://127.0.0.1:8790"
        ):
            lop = r.fence_ports("local-operator", "public-full", edge_port=4242, bridge_port=0)
            omp = r.fence_ports("omp", "public-full", edge_port=None, bridge_port=0)
            self.assertEqual(lop, [4242, 8790])
            self.assertEqual(omp, [8787, 8790])
            self.assertEqual(r.fence_ports("local-operator", "clean", edge_port=4242,
                                           bridge_port=0), [4242])
            self.assertEqual(r.fence_ports("opencode", "public-full", edge_port=None,
                                           bridge_port=0), [])

    def test_live_prices_must_match_the_reviewed_decisions_contract(self):
        good = json.dumps({"result": {"rates": {"decision_models": {
            "rate_version": "decisions-rate@2026-09-19",
            "rates": [{"id": "jev-1.13", "aliases": ["jev-1.13.0", "typesafe/jev-1.13"],
                       "provider": "typesafe", "unit": "token",
                       "prompt_unit_price_usd": 4.2e-08, "completion_unit_price_usd": 0,
                       "price_version": "decisions-rate@2026-09-19"}]}}}}).encode()
        self.assertEqual(r.check_decision_prices(lambda: (200, good))["prices_version"],
                         "decisions-rate@2026-09-19")
        drifted = good.replace(b"4.2e-08", b"5e-08")
        for fetch in (lambda: (200, drifted), lambda: (503, good), lambda: (200, b"{}")):
            with self.assertRaisesRegex(ValueError, "prices"):
                r.check_decision_prices(fetch)

    def test_lop_launch_is_sandboxed_through_its_own_edge(self):
        argv = [sys.executable, "lop_launch.py", "exec"]
        wrapped = r.sandboxed(argv, [4242])
        self.assertEqual(wrapped[:2], ["/usr/bin/sandbox-exec", "-p"])
        self.assertIn('(remote ip "localhost:4242")', wrapped[2])
        self.assertEqual(wrapped[3:], argv)
        with self.assertRaises(ValueError):
            r.sandboxed(argv, [])

    def test_sandbox_applies_only_to_harnesses_that_passed_the_zero_paid_smoke(self):
        for harness in r.HARNESSES:
            with self.subTest(harness=harness):
                self.assertEqual(harness in r.SANDBOXED_HARNESSES,
                                 harness in r.SANDBOX_SMOKE_PASSED)
        self.assertIn("local-operator", r.SANDBOXED_HARNESSES)
        self.assertEqual(set(r.SANDBOX_SMOKE_PASSED) | set(r.SANDBOX_SMOKE_FAILED),
                         set(r.HARNESSES))
        for harness, error in r.SANDBOX_SMOKE_FAILED.items():
            self.assertTrue(error, harness)

    def test_wall_clock_limit_kills_exact_group_and_records_timeout(self):
        import subprocess
        import time

        script = self.root / "sleeper.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        assert proc.stdout is not None
        child = int(proc.stdout.readline())
        started = time.monotonic()
        outcome = r.run_with_wall_clock(proc, 0.5)
        self.assertLess(time.monotonic() - started, 20)
        self.assertTrue(outcome["timed_out"])
        self.assertEqual(outcome["outcome"], "timeout")
        self.assertIsNotNone(proc.returncode)
        with self.assertRaises(ProcessLookupError):
            os.kill(child, 0)

    def test_start_edge_hands_the_token_over_stdin_only_and_reaps(self):
        import test_adapters  # shares the offline meter configuration helper

        fake = "s6-fake-radient-bearer-not-a-credential"
        run = self.root / "run"
        run.mkdir()
        with patch.dict(os.environ, test_adapters.meter_env(self.root)):
            edge, announced = r.start_edge("v2dev__unit", run, fake)
            try:
                self.assertIsNone(edge.poll())
                env = announced["env"]
                assert isinstance(env, dict)
                import lop_edge

                self.assertEqual(env["RADIENT_API_KEY"], lop_edge.DUMMY_RADIENT_KEY)
                self.assertIn(f"127.0.0.1:{announced['port']}", env["HTTPS_PROXY"])
                self.assertNotIn(fake, json.dumps(announced))
                self.assertNotIn(fake, repr(edge.args))
                self.assertTrue((run / "edge" / "edge-ca.pem").is_file())
            finally:
                r.stop_edge(edge)
        self.assertIsNotNone(edge.returncode)
        with self.assertRaises(ProcessLookupError):
            os.killpg(edge.pid, 0)
        for path in run.rglob("*"):
            if path.is_file():
                self.assertNotIn(fake, path.read_text(errors="ignore"))

    def test_money_classifier_from_ledger_channels_not_stderr(self):
        rows = [{"path": "v1/messages", "status": 200, "usage": {"cost": 0.05}}]
        ledger = {"run": {"model": 0.05, "classifier": 0.003, "blocked": False}}
        cost = r.money(rows, ledger["run"], True)
        self.assertEqual(cost["radient_cost_usd"], 0.003)
        self.assertEqual(cost["total_cost_usd"], 0.053)
        # A latched ledger leaves the classifier total unknown, never zero.
        cost = r.money(rows, {"model": 0.05, "classifier": 0.0, "blocked": True}, True)
        self.assertIsNone(cost["radient_cost_usd"])
        self.assertFalse(cost["cost_complete"])
        # A stderr line claiming a charge is ignored entirely.
        self.assertNotIn("classification_calls_charged", cost)

    def test_run_ledger_summary_sums_channels_and_publishes_cap_overshoot(self):
        ledger = {
            "usd": 5.9, "trials": {"run": 5.9}, "pending": {}, "blocked": False,
            "channels": {"model": 5.9, "classifier": 0.0}, "models": {}, "voids": {},
            "cost_caps": {"run": {"settled_usd": 5.9, "cap_usd": 5.0}},
            "rows": [],
        }
        edge_rows = [{"channel": "classifier", "actual_usd": "0.004", "qualified": True}]
        meter_rows = [{"run_id": "run", "channel": "model", "usage": {"cost": 5.9},
                       "in_flight_at_admission": 3}]
        summary = r.run_ledger_summary(ledger, "run", meter_rows, edge_rows)
        self.assertAlmostEqual(float(str(summary["classifier"])), 0.004)
        self.assertTrue(summary["cost_capped"])
        self.assertAlmostEqual(float(str(summary["cap_overshoot_usd"])), 0.9)
        self.assertEqual(summary["max_in_flight_at_cap"], 3)
        self.assertIsNone(summary["void_reasons"])

    def test_source_launch_environment_is_explicit(self):
        with patch.dict(os.environ, {"LOP_BENCH_SOURCE_ROOT": "/explicit/reviewed-source"}):
            _, env = r.configure(
                "local-operator", "v2dev__unit", "public unit text", self.home, True, self.runtime
            )
        self.assertEqual(env["LOP_BENCH_PYTHON"], sys.executable)
        self.assertEqual(env["LOP_BENCH_SOURCE_ROOT"], "/explicit/reviewed-source")
        self.assertNotIn("OPENROUTER_API_KEY", env)
        # The released product talks to its real URLs; the edge meters them.
        self.assertNotIn("BENCH_ANTHROPIC_URL", env)
        self.assertEqual(env["RADIENT_API_KEY"], "bench-dummy")

    def test_lop_is_pinned_to_released_v0_94_22(self):
        self.assertEqual(r.LOP_HEAD, "500883ef56ca504325ffd80c5b90bf2435d00e3f")
        self.assertIn("lop_edge.py", r.integration_inputs())
        self.assertNotIn("lop_shim/sitecustomize.py", r.integration_inputs())

    def test_all_seven_commands_and_public_servers(self):
        for harness in r.HARNESSES:
            with self.subTest(harness=harness):
                argv, env = r.configure(
                    harness,
                    "v2dev__public-full__fixture",
                    "fixture",
                    self.home,
                    True,
                    self.runtime,
                    18797,
                )
                self.assertEqual(argv[0], sys.executable)
                self.assertEqual(env["HOME"], str(self.home))
                # LoP alone carries a Radient variable, and only the dummy.
                self.assertIn(env.get("RADIENT_API_KEY"), (None, "bench-dummy"))
                self.assertNotIn("BENCH_LITELLM_KEY", env)
                self.assertNotIn("AUTH_TOKEN", env)
                self.assertNotIn("--no-tools", argv)
        # dsh must INSERT new plugins rather than silently overriding nonexistent ids.
        p = json.loads((self.home / ".dsh/cordis.patch.yml").read_text())
        self.assertEqual(len(p[-1]["insert"]), 9)
        self.assertEqual(
            {e["config"]["serverName"] for e in p[-1]["insert"]},
            set(r.public_snapshot()["servers"]),
        )
        self.assertEqual(
            p[0]["config"]["providers"]["anthropic"]["models"][0]["contextWindow"], 1000000
        )
        argv, env = r.configure(
            "codex", "v2dev__clean__fixture", "fixture", self.home, False, self.runtime, 18797
        )
        self.assertIn("18797/v1", " ".join(argv))
        self.assertIn("model_catalog_json=", " ".join(argv))
        self.assertNotIn("codex-model-catalog.eager", " ".join(argv))

    def test_dsh_missing_entry_fails_before_config_write(self):
        runtime: r.RuntimeIdentity = {"binary": sys.executable, "sha256": "fixture"}
        with patch.object(r, "write_dsh_patch") as write_patch:
            with self.assertRaisesRegex(ValueError, "lacks its installed CLI entry"):
                r.configure("dsh", "v2dev__unit", "fixture", self.home, False, runtime)
            write_patch.assert_not_called()

    def test_identical_instructions_and_omp_single_source(self):
        original = (self.root / "public_user_instructions.md").read_bytes()
        for harness in r.HARNESSES:
            self.assertEqual((self.home / r.INSTRUCTION_PATHS[harness]).read_bytes(), original)
        # This is exactly the per-run selection used by prepare; validate that
        # only omp's own file survives rather than .claude/.codex copies too.
        for harness, rel in r.INSTRUCTION_PATHS.items():
            if harness != "omp":
                (self.home / rel).unlink()
        remaining = [rel for rel in r.INSTRUCTION_PATHS.values() if (self.home / rel).exists()]
        self.assertEqual(remaining, [".omp/agent/AGENTS.md"])

    def test_executable_symlink_preserves_venv_path(self):
        link = self.root / "venv/bin/python"
        link.parent.mkdir(parents=True)
        link.symlink_to(sys.executable)
        with patch.dict(os.environ, {"BENCH_LOP_PY": str(link)}):
            self.assertEqual(r.resolve_program("BENCH_LOP_PY"), str(link))
            self.assertNotEqual(r.resolve_program("BENCH_LOP_PY"), str(link.resolve()))

    def test_real_prepare_with_synthetic_skills_keeps_one_omp_instruction(self):
        base = self.root / "basehome-public"
        # Do not ship upstream skill bodies as test fixtures. This exercises the
        # real copy/filter path with explicitly synthetic, locally hashed text;
        # it does not attest real public skill exposure or pin compatibility.
        body = b"Synthetic public unit skill, not a benchmark asset.\n"
        synthetic = {name: hashlib.sha256(body).hexdigest() for name in r.expected_skills()}
        self.patches.append(patch.object(r, "expected_skills", return_value=synthetic))
        self.patches[-1].start()
        for rel in r.SKILL_ROOTS:
            for name in synthetic:
                p = base / rel / name / "SKILL.md"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
        for rel in r.INSTRUCTION_PATHS.values():
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((self.root / "public_user_instructions.md").read_bytes())
        task: r.PreparedTask = {"id": "fixture", "config": [], "_domain": "libreoffice_calc"}
        # The test fixture is session-local; simulate the production outside-
        # account-home placement check without changing the real process HOME.
        with (
            patch.dict(os.environ, {"BENCH_PUBLIC_BASEHOME": str(base)}),
            patch.object(r.Path, "home", return_value=Path("/nonexistent-account")),
        ):
            root, home = r.prepare("v2dev__public-full__omp__unit", task, "public-full", "omp")
        remaining = [p for p in r.INSTRUCTION_PATHS.values() if (home / p).exists()]
        self.assertEqual(remaining, [r.INSTRUCTION_PATHS["omp"]])
        with self.assertRaisesRegex(ValueError, "Trial id exists"):
            r.prepare("v2dev__public-full__omp__unit", task, "public-full", "omp")
        self.assertTrue(home.exists())  # refusing reuse must not delete a prior home

    def test_real_prepare_on_linux_without_reflinks(self):
        # Reproduce the runner filesystem failure against the SAME public
        # synthetic profile and real prepare/config path, not a skipped fixture.
        import subprocess

        def unsupported(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr=b"Operation not supported")

        with (
            patch.object(r.sys, "platform", "linux"),
            patch.object(r.subprocess, "run", side_effect=unsupported),
        ):
            self.test_real_prepare_with_synthetic_skills_keeps_one_omp_instruction()

    def test_public_snapshot_is_exactly_493_tools(self):
        snapshot = r.public_snapshot()
        self.assertEqual(len(snapshot["servers"]), 9)
        self.assertEqual(sum(len(s["tools"]) for s in snapshot["servers"].values()), 493)
        with patch.object(r, "HERE", self.real):
            self.assertEqual(len(r.expected_skills()), 19)

    def test_replay_mismatch_fails_closed(self):
        import io

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with patch.object(
            r.urllib.request, "urlopen", return_value=Response(b'{"result":{"tools":[]}}')
        ):
            with self.assertRaisesRegex(ValueError, "Replay schema mismatch"):
                r.verify_replay("v2dev__unit")

    def test_metadata_rejects_codex_fallback(self):
        metadata = {
            "id": "anthropic/claude-sonnet-5.5",
            "context_length": 1000000,
            "top_provider": {"max_completion_tokens": 128000},
        }
        (self.root / "model_metadata.json").write_text(json.dumps(metadata))
        r.verify_metadata()
        p = self.root / "codex-model-catalog.json"
        c = json.loads(p.read_text())
        c["models"][0]["context_window"] = 272000
        p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError, "Primary Codex"):
            r.verify_metadata()

    def test_missing_wire_and_freeze_refuse_without_credentials(self):
        with patch.dict(
            os.environ, {"BENCH_WIRE_ATTESTATIONS": str(self.root / "no-captures")}, clear=False
        ):
            with self.assertRaises(FileNotFoundError):
                r.attest("omp", "public-full", self.runtime)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "paid execution prohibited"):
                r.frozen_prereg("omp", "clean", self.runtime)

    def test_cost_split_unknown_and_zero(self):
        rows = [
            {"path": "v1/messages", "status": 200, "usage": {"cost": 0.05}},
            {"path": "v1/messages", "status": 200, "usage": {"cost": 0.002}},
        ]
        cost = r.money(rows, {"model": 0.052, "classifier": 0.003, "blocked": False}, True)
        self.assertEqual(cost["model_cost_usd"], 0.052)
        self.assertEqual(cost["radient_cost_usd"], 0.003)
        self.assertEqual(cost["total_cost_usd"], 0.055)
        self.assertIsNone(r.money(rows, None, True)["total_cost_usd"])
        self.assertEqual(r.money(rows, None, False)["total_cost_usd"], 0.052)

    def test_money_missing_attempt_never_reports_complete_total(self):
        rows = [
            {"path": "v1/messages", "status": 200, "usage": {"cost": 0.20}},
            {"path": "v1/messages", "status": 200, "usage": {}},
        ]
        cost = r.money(rows, None, False)
        self.assertFalse(cost["cost_complete"])
        self.assertIsNone(cost["model_cost_usd"])
        self.assertIsNone(cost["total_cost_usd"])
        self.assertEqual(cost["measured_model_cost_usd"], 0.20)
        self.assertEqual(cost["unknown_billing_attempts"], 1)

    def test_money_billed_error_is_included_with_status_flags(self):
        rows = [
            {"path": "v1/messages", "status": 200, "usage": {"cost": 0.20}},
            {"path": "v1/messages", "status": 500, "usage": {"cost": 0.30}},
        ]
        cost = r.money(rows, None, False)
        self.assertTrue(cost["cost_complete"])
        self.assertEqual(cost["total_cost_usd"], 0.50)
        self.assertEqual(cost["message_attempts"], 2)
        self.assertEqual(cost["non_200_message_attempts"], 1)

    def test_money_discovery_is_not_a_model_attempt(self):
        rows = [
            {"path": "mcp/tools/list", "status": 200},
            {"path": "v1/messages/count_tokens", "status": 200},
        ]
        cost = r.money(rows, None, False)
        self.assertFalse(cost["cost_complete"])
        self.assertEqual(cost["message_attempts"], 0)
        self.assertEqual(cost["non_model_rows"], 2)
        self.assertIsNone(cost["total_cost_usd"])
        cost = r.money(
            rows + [{"path": "v1/messages", "status": 429, "usage": {"cost": 0}}], None, False
        )
        self.assertTrue(cost["cost_complete"])
        self.assertEqual(cost["total_cost_usd"], 0)

    def test_money_invalid_costs_are_unknown_not_zero(self):
        for invalid in (None, -1, True, "NaN", "Infinity", "bad", {}, []):
            with self.subTest(invalid=invalid):
                cost = r.money(
                    [{"path": "v1/messages", "status": 500, "usage": {"cost": invalid}}],
                    None,
                    False,
                )
                self.assertFalse(cost["cost_complete"])
                self.assertEqual(cost["unknown_billing_attempts"], 1)
                self.assertIsNone(cost["total_cost_usd"])

    def test_money_settled_row_is_measured_and_a_truncation_is_refused(self):
        # R17 m3: this lane reads the same meter rows, so a row the meter settled
        # from its own tokens is measured at that derivation -- unknown, not zero,
        # only when the row is not a whole bill.
        tokens = {"input_tokens": 2, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 27164}
        derivation = {
            "tokens": tokens,
            "rates_usd_per_m": {"input_tokens": 2.0, "output_tokens": 10.0,
                                "cache_read_input_tokens": 0.1,
                                "cache_creation_input_tokens": 2.5},
            "usd": 0.067914,
        }
        settled = {"path": "v1/messages", "status": 200, "settled_from_tokens": True,
                   "cost_derivation": derivation, "usage": dict(tokens)}
        cost = r.money([settled], None, False)
        self.assertTrue(cost["cost_complete"])
        self.assertAlmostEqual(cost["total_cost_usd"], 0.067914, delta=1e-12)
        self.assertEqual(cost["unknown_billing_attempts"], 0)
        for broken in (
            {**settled, "incomplete": True},
            # R18 m4: a stated cost beside partial usage is a floor too. Taking
            # the cost branch first published 0.075 for this exact row.
            {"path": "v1/messages", "status": 200, "incomplete": True,
             "usage": {**tokens, "cost": 0.075}},
            {**settled, "usage": {**tokens, "output_tokens": 5000}},
            {**settled, "cost_derivation": {**derivation, "usd": 0.01}},
            {**settled, "cost_derivation": {**derivation,
                                            "rates_usd_per_m": {"input_tokens": 2.0}}},
        ):
            with self.subTest(broken=sorted(broken)):
                cost = r.money([broken], None, False)
                self.assertFalse(cost["cost_complete"])
                self.assertEqual(cost["unknown_billing_attempts"], 1)
                self.assertEqual(cost["measured_model_cost_usd"], 0.0)
                self.assertIsNone(cost["total_cost_usd"])

    def test_deferred_summary_without_raw_exchanges_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Deferred capture lacks correlated"):
            r.validate_deferred_exchanges(
                [{"model": r.MODEL, "tools": [{"name": "tool_search"}]}], [], [], False
            )

    def test_deferred_terminal_echo_cannot_impersonate_mcp_call(self):
        # Synthetic protocol fixture only, never exported as wire evidence.
        content: list[r.ContentPart] = []
        rpc: list[r.RPCExchange] = []
        for i, (server, rec) in enumerate(r.public_snapshot()["servers"].items()):
            tool = rec["tools"][0]["name"]
            result: r.RPCResult = {
                "isError": True,
                "content": [{"type": "text", "text": f"{server}: offline benchmark"}],
            }
            content.extend(
                [
                    {
                        "type": "tool_use",
                        "id": f"s{i}",
                        "name": "tool_search",
                        "input": {"query": server},
                    },
                    {"type": "tool_result", "tool_use_id": f"s{i}", "content": server},
                    {
                        "type": "tool_use",
                        "id": f"c{i}",
                        "name": "terminal",
                        "input": {"command": "echo " + server},
                    },
                    {"type": "tool_result", "tool_use_id": f"c{i}", "content": result},
                ]
            )
            rpc.extend(
                [
                    {
                        "server": server,
                        "request": {"id": i, "method": "tools/list"},
                        "response": {"id": i, "result": {"tools": rec["tools"]}},
                    },
                    {
                        "server": server,
                        "request": {
                            "id": 100 + i,
                            "method": "tools/call",
                            "params": {"name": tool, "arguments": {}},
                        },
                        "response": {"id": 100 + i, "result": result},
                    },
                ]
            )
        bodies: list[r.WireBody] = [{"messages": [{"role": "assistant", "content": content}]}]
        with self.assertRaisesRegex(ValueError, "correlated"):
            r.validate_deferred_exchanges(bodies, rpc, [], False)
        for i, (server, rec) in enumerate(r.public_snapshot()["servers"].items()):
            use = content[i * 4 + 2]
            assert use["type"] == "tool_use"
            use.update(name=f"mcp__{server}__{rec['tools'][0]['name']}", input={})
        r.validate_deferred_exchanges(bodies, rpc, [], False)
        use = content[2]
        assert use["type"] == "tool_use"
        use["input"] = {"different_arguments": True}
        with self.assertRaisesRegex(ValueError, "correlated"):
            r.validate_deferred_exchanges(bodies, rpc, [], False)

    def test_hermes_cleanup_never_uses_empty_environment(self):
        with patch.object(r.subprocess, "run") as run:
            r._hermes_teardown(self.home, {})
            run.assert_not_called()

    def test_actual_instruction_triplication_is_rejected(self):
        # Deliberately synthetic capture: this tests the parser, NOT a harness.
        root = self.root / "attest"
        root.mkdir()
        body = {
            "model": r.MODEL,
            "tools": [{"name": "native"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "system": (self.root / "public_user_instructions.md").read_text() * 3,
        }
        capture = root / "main.json"
        capture.write_text(json.dumps(body))
        counts = {k: len(v["tools"]) for k, v in r.public_snapshot()["servers"].items()}
        record = {
            "runtime_fingerprint": r.runtime_fingerprint(self.runtime),
            "runner_sha256": r.sha(Path(r.__file__)),
            "input_sha256": r.integration_inputs(),
            "public_schema_sha256": r.sha(self.root / "public_mcp_schemas.json"),
            "context_window": 1000000,
            "main_effort": "high",
            "model": r.MODEL,
            "thinking_type": "adaptive",
            "main_native_tools": 1,
            "instruction_occurrences": 1,
            "instruction_sha256": r.sha(self.root / "public_user_instructions.md"),
            "skill_count": 19,
            "mcp_server_counts": counts,
            "mcp_mode": "deferred",
            "mcp_roundtrip_servers": list(counts),
            "captures": [{"kind": "messages", "path": "main.json", "sha256": r.sha(capture)}],
        }
        (root / "omp.public-full.json").write_text(json.dumps(record))
        with patch.dict(os.environ, {"BENCH_WIRE_ATTESTATIONS": str(root)}):
            with self.assertRaisesRegex(ValueError, "instruction count differs"):
                r.attest("omp", "public-full", self.runtime)


if __name__ == "__main__":
    unittest.main()

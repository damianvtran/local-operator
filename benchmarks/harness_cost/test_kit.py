#!/usr/bin/env python3
"""Bounded regression/negative controls for the public analysis, with no spend.

Why: successful regeneration alone does not prove corrupt or incomplete inputs
fail loudly. Tests mutate disposable snapshots, never the source logs. All
outputs/temporary files stay under TMPDIR (set it to your own scratch area).
Constraints: unittest/stdlib only; no network, model APIs, harnesses, office
metric execution or subprocess dependency installation. Pixel parity is not a
test; every historical numeric field is compared with documented tolerance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import publication as p
import regenerate as r


class KitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness-cost-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "v1-snapshot"
        shutil.copytree(r.ROOT / "v1-snapshot", self.snapshot)

    def mutate(self, file, fn):
        path = self.snapshot / file
        rows = r.jsonl(path)
        fn(rows)
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))

    def test_all_historical_fields(self):
        values = r.derive(self.snapshot)
        for name, value in zip(
            ["results", "figure_data", "cost_components_long", "call_sequences"], values
        ):
            r.parity(value, r.read_json(self.snapshot / f"{name}.json"), name)
        self.assertEqual(len(values[0]["rows"]), 180)
        self.assertEqual(len(values[4]["runs"]), 186)
        r.close(values[4]["model_cost_usd"], 36.909711435, "test total")
        for row in values[0]["rows"]:
            r.close(row["total_cost_usd"], row["model_cost_usd"], "known absent classifier")
        matrix = {}
        for row in values[0]["rows"]:
            if row["arm"] == "full" and row["harness"] != "codex":
                matrix.setdefault(row["task_id"], []).append(row["score"])
        self.assertEqual(
            (sum(all(v) for v in matrix.values()), sum(not any(v) for v in matrix.values())),
            (14, 6),
        )
        # Exercise the actual exporter rather than normalizing files afterwards.
        r.export(self.root, "lf_export", {"value": 1}, [{"value": 1}, {"value": 2}])
        csv_bytes = (self.root / "lf_export.csv").read_bytes()
        self.assertNotIn(b"\r", csv_bytes)
        self.assertEqual(csv_bytes.count(b"\n"), 3)
        # Importing capture is offline: no package launch, server or API call.
        spec = importlib.util.spec_from_file_location(
            "offline_capture", r.ROOT / "public_loadout/capture_mcp_schemas.py"
        )
        assert spec is not None and spec.loader is not None
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        # No process is launched: malformed PIPE clients must fail before I/O.
        client = capture.StdioClient.__new__(capture.StdioClient)
        client.proc = SimpleNamespace(stdin=None, stdout=None, stderr=None)
        with self.assertRaises(AssertionError):
            client.send({})
        with self.assertRaises(AssertionError):
            client.request("tools/list")
        self.assertEqual(client.stderr_tail(), "")
        snap = r.read_json(r.ROOT / "public_loadout/public_mcp_schemas.json")
        pin = capture.PINS["launchdarkly"]
        self.assertRegex(pin["cmd"][-1], p.PATTERNS["provider secret"])
        self.assertEqual(capture.published_command(pin), snap["servers"]["launchdarkly"]["command"])
        self.assertEqual(capture.published_command(pin)[-1], "<offline-dummy-generated-at-runtime>")
        self.assertNotRegex(
            json.dumps(capture.published_command(pin)), p.PATTERNS["provider secret"]
        )
        for server in snap["servers"].values():
            payload = json.dumps(
                server["tools"], separators=(",", ":"), ensure_ascii=False
            ).encode()
            self.assertEqual(len(payload), server["schema_bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), server["sha256"])
        self.assertEqual(sum(len(v["tools"]) for v in snap["servers"].values()), 493)

    def test_recursive_metering_projection_refused(self):
        original = r.jsonl(self.snapshot / "requests.jsonl")
        cases = [
            lambda q: q["usage"]["cache_creation"].update(private_note="nonpublic metadata"),
            lambda q: q["usage"]["cache_creation"].update(
                ephemeral_5m_input_tokens={"private_note": "nonpublic metadata"}
            ),
            lambda q: q["usage"]["cache_creation"].update(
                ephemeral_5m_input_tokens=["nonpublic metadata"]
            ),
            lambda q: q["usage"].update(cache_creation=["nonpublic metadata"]),
            lambda q: q["usage"].update(cache_creation=None),
            lambda q: q["usage"].update(input_tokens={"private_note": "nonpublic metadata"}),
            lambda q: q["usage"]["cache_creation"].update(ephemeral_5m_input_tokens=True),
            lambda q: q.update(n_tools={"private_note": "nonpublic metadata"}),
        ]
        self.assertEqual(set(p.REQUEST_SCHEMA), p.REQUEST_FIELDS)
        self.assertEqual(set(p.USAGE_SCHEMA), p.USAGE_FIELDS)
        p.scan(self.root, ["v1-snapshot/requests.jsonl"])
        for index, mutation in enumerate(cases):
            with self.subTest(nested_case=index):
                rows = copy.deepcopy(original)
                mutation(next(q for q in rows if q.get("usage")))
                (self.snapshot / "requests.jsonl").write_text(
                    "".join(json.dumps(q) + "\n" for q in rows)
                )
                with self.assertRaisesRegex(ValueError, "Metering projection"):
                    p.scan(self.root, ["v1-snapshot/requests.jsonl"])

    def test_cost_table_numeric_cells_and_populations(self):
        rows = r.cost_table_rows(r.derive(self.snapshot)[0])
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [(x["population"], x["n"]) for x in rows],
            [("clean", 16)] * 5 + [("full", 24)] * 4 + [("codex-full", 4)],
        )
        r.export(self.root, "table_numeric", {"rows": rows}, rows)
        serialized = r.read_json(self.root / "table_numeric.json")["rows"]
        markdown = r.cost_table_markdown(rows)
        r.check_table_markdown(markdown, serialized)
        changed = markdown.replace(f"{rows[0]['observed_usd_per_task']:.6f}", "9.999999", 1)
        with self.assertRaisesRegex(ValueError, "numeric mismatch"):
            r.check_table_markdown(changed, serialized)
        changed = markdown.replace(f"{rows[0]['passed']} / {rows[0]['n']}", "0 / 4", 1)
        with self.assertRaisesRegex(ValueError, "pass/sample"):
            r.check_table_markdown(changed, serialized)
        with self.assertRaisesRegex(ValueError, "population headings"):
            r.check_table_markdown(markdown.replace("16 short tasks", "4 short tasks"), serialized)

    def test_contextual_codex_labels(self):
        rows = r.derive(self.snapshot)[0]["rows"]
        self.assertEqual(
            r.plot_labels(["codex"], [x for x in rows if x["arm"] == "clean"]), ["Codex (n=16)"]
        )
        self.assertEqual(
            r.plot_labels(["codex"], [x for x in rows if x["arm"] == "full"]), ["Codex (n=4)"]
        )

    def test_readable_figure_geometry_and_redundant_encodings(self):
        frames = r.read_json(r.ROOT / "generated/figure_geometry.json")
        self.assertEqual(
            {f["figure"] for f in frames},
            {"fig1-cost-per-task.png", "fig2-why-costs-differ.png", "cost-per-task-table.png"},
        )
        for frame in frames:
            self.assertTrue(frame["all_text_within_canvas"])
            self.assertGreaterEqual(frame["min_css_font_px_at_861"], 12)
            if frame["figure"] == "cost-per-task-table.png":
                self.assertFalse(frame["table_text_overlap"])
            else:
                encoding = frame["encoding"]
                self.assertEqual(len({e["hatch"] for e in encoding}), len(encoding))
                key = (
                    "fill_background_contrast"
                    if frame["figure"].startswith("fig1")
                    else "boundary_contrast"
                )
                self.assertTrue(all(e[key] >= 3 for e in encoding))

    def test_duplicate_run_refused(self):
        self.mutate("runs.jsonl", lambda x: x.append(copy.deepcopy(x[0])))
        with self.assertRaisesRegex(ValueError, "duplicates/count"):
            r.derive(self.snapshot)

    def test_duplicate_request_refused(self):
        def mutation(x):
            x[1] = copy.deepcopy(x[0])

        self.mutate("requests.jsonl", mutation)
        with self.assertRaisesRegex(ValueError, "Duplicate requests"):
            r.derive(self.snapshot)

    def test_missing_score_refused(self):
        def mutation(x):
            x[0]["score"] = None

        self.mutate("runs.jsonl", mutation)
        with self.assertRaisesRegex(ValueError, "Invalid task/score"):
            r.derive(self.snapshot)

    def test_missing_usage_refused(self):
        def mutation(x):
            next(q for q in x if q.get("usage"))["usage"] = None

        self.mutate("requests.jsonl", mutation)
        with self.assertRaisesRegex(ValueError, "Priced request count"):
            r.derive(self.snapshot)

    def test_openrouter_discrepancy_refused(self):
        def mutation(x):
            next(q for q in x if q.get("usage"))["usage"]["cost"] += 0.01

        self.mutate("requests.jsonl", mutation)
        with self.assertRaisesRegex(ValueError, "OpenRouter call reconciliation"):
            r.derive(self.snapshot)

    def test_cache_subtotals_refused(self):
        def mutation(x):
            next(q for q in x if q.get("usage"))["usage"]["cache_creation_input_tokens"] += 1

        self.mutate("requests.jsonl", mutation)
        with self.assertRaisesRegex(ValueError, "Cache TTL subtotals"):
            r.derive(self.snapshot)

    def test_signed_sensitivity_not_clamped(self):
        rows = r.derive(self.snapshot)[0]["rows"]
        codex = [x for x in rows if x["harness"] == "codex" and x["arm"] == "full"]
        for x in codex:
            assert "codex_cache_corrected_cost" in x
            assert "codex_sensitivity_status" in x
        self.assertEqual(
            sum(x.get("codex_cache_corrected_cost") != x["cost_usd"] for x in codex), 3
        )
        self.assertEqual(
            sum(
                x["codex_cache_corrected_cost"] > x["cost_usd"]
                for x in codex
                if "codex_cache_corrected_cost" in x
            ),
            2,
        )
        self.assertTrue(
            all(
                "NOT measured" in x["codex_sensitivity_status"]
                for x in codex
                if "codex_sensitivity_status" in x
            )
        )

    def test_unobserved_rate_branches_synthetic_only(self):
        u = dict(
            input_tokens=100001,
            output_tokens=10,
            cache_creation_input_tokens=2,
            cache_read_input_tokens=0,
            cache_creation=dict(ephemeral_1h_input_tokens=2, ephemeral_5m_input_tokens=0),
        )
        cost, _, parts, rates = r.price_row("anthropic/claude-haiku-5.5", u)
        self.assertEqual(rates["input"], 0.5)
        r.close(cost, (100001 * 0.5 + 2 * 1.0 + 10 * 2.5) / 1e6, "synthetic tier")
        r.close(parts["cache_write_1h"], 2 / 1e6, "synthetic 1h")

    def test_metering_private_field_refused(self):
        self.mutate("requests.jsonl", lambda x: x[0].update(request_body="private placeholder"))
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            p.scan(self.root, ["v1-snapshot/requests.jsonl"])

    def test_checksum_tamper_refused(self):
        shutil.copytree(
            r.ROOT,
            self.root / "kit",
            ignore=shutil.ignore_patterns(".venv", "__pycache__", "kit-src"),
        )
        tempkit = self.root / "kit"
        p.create_manifest(tempkit)
        with (tempkit / "CORRECTIONS.md").open("a") as f:
            f.write("\nchanged\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            p.verify(tempkit)

    def test_secret_scan_and_symlinks_refused(self):
        text = self.root / "test.txt"
        # Construct a fake token at runtime so the test fixture is not itself
        # mistaken for a credential by source scanning.
        examples = [
            "sk-" + "a" * 40,
            "-".join(["api", "0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12]),
            "api-" + "0" * 40,
            "sbp_" + "0" * 40,
            "ntn_" + "0" * 46,
            "gh" + "p_" + "0" * 36,
            "AK" + "IA" + "0" * 16,
        ]
        for fake in examples:
            with self.subTest(shape_length=len(fake)):
                text.write_text(fake)
                with self.assertRaisesRegex(ValueError, "provider secret"):
                    p.scan(self.root, ["test.txt"])
        text.unlink()
        text.symlink_to(self.snapshot / "requests.jsonl")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            p.inventory(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)

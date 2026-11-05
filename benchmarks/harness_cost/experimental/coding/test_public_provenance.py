"""Offline provenance controls; no Docker, model requests or task input edits."""

import asyncio
import contextlib
import copy
import hashlib
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import driver
import gold_probe
from test_adapters import _load
from test_lop_container import _load_verifier


class PublicProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
        )
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.public = self.root / "interp.py"
        self.public.write_bytes(b"public fixture")
        self.sha = hashlib.sha256(self.public.read_bytes()).hexdigest()
        self.manifest = {
            "version": 1,
            "task_payload_sha256": "a" * 64,
            "image_digest": "sha256:" + "b" * 64,
            "policy_sha256": "c" * 64,
            "public_files": [{"path": str(self.public), "sha256": self.sha}],
            "solution_digests": [],
            "gold_digests": [self.sha],
        }

    def scan(self, manifest=None):
        return gold_probe.isolation(
            {self.sha},
            root=str(self.root),
            manifest=manifest or self.manifest,
            task_payload_sha256="a" * 64,
            image_digest="sha256:" + "b" * 64,
        )

    def env(self):
        raw = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        return {
            "BENCH_PUBLIC_PROVENANCE": raw,
            "BENCH_PUBLIC_PROVENANCE_SHA256": hashlib.sha256(raw.encode()).hexdigest(),
            "BENCH_TASK_PAYLOAD_SHA256": "a" * 64,
            "BENCH_IMAGE_DIGEST": "sha256:" + "b" * 64,
            "BENCH_GOLD_DIGESTS": self.sha,
            "BENCH_GATE_PORTS": "8787",
            "BENCH_EGRESS_CANARY_PORT": "41999",
        }

    def test_raw_strict_failure_is_retained_with_public_success(self):
        result = self.scan()
        self.assertTrue(result["ok"])
        self.assertFalse(result["strict_ok"])
        self.assertEqual(result["raw_matches"], result["public_matches"])
        self.assertEqual(result["rejected_matches"], [])

    def test_solution_public_overlap_is_disclosed_and_never_approved(self):
        self.manifest["solution_digests"] = [self.sha]
        result = self.scan()
        self.assertFalse(result["ok"])
        self.assertEqual(result["solution_public_overlap"], [self.sha])
        self.assertEqual(result["public_matches"], [])

    def test_replaced_public_file_is_not_authorized(self):
        hidden_sha = hashlib.sha256(b"hidden verifier sentinel").hexdigest()
        self.public.write_bytes(b"hidden verifier sentinel")
        self.manifest["gold_digests"].append(hidden_sha)
        result = gold_probe.isolation(
            {self.sha, hidden_sha},
            root=str(self.root),
            manifest=self.manifest,
            task_payload_sha256="a" * 64,
            image_digest="sha256:" + "b" * 64,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["public_matches"], [])

    def test_missing_stale_and_malformed_manifest_fail_closed(self):
        for value in (
            None,
            [],
            {},
            "invalid",
            {**self.manifest, "version": 2},
            {**self.manifest, "image_digest": "sha256:" + "d" * 64},
            {**self.manifest, "task_payload_sha256": "d" * 64},
            {**self.manifest, "solution_digests": [None]},
            {**self.manifest, "public_files": [None]},
            {**self.manifest, "gold_digests": []},
        ):
            with self.subTest(value=value):
                result = gold_probe.isolation(
                    {self.sha},
                    root=str(self.root),
                    manifest=value,
                    task_payload_sha256="a" * 64,
                    image_digest="sha256:" + "b" * 64,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(result["provenance_error"])

    def test_forbidden_or_noncanonical_public_paths_fail_closed(self):
        for path in (
            "/solution/a",
            "/tests/a",
            "/logs/verifier/a",
            "/opt/bench/a",
            "/app/../interp.py",
            "/app//interp.py",
            "relative",
        ):
            manifest = copy.deepcopy(self.manifest)
            manifest["public_files"][0]["path"] = path
            with self.subTest(path=path):
                self.assertFalse(self.scan(manifest)["ok"])

    def test_existing_absolute_guards_and_truncation_still_refuse(self):
        original = Path.exists
        for forbidden in ("/solution", "/tests"):
            with patch.object(
                Path,
                "exists",
                lambda p, forbidden=forbidden: str(p) == forbidden or original(p),
            ):
                self.assertFalse(self.scan()["ok"])
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(
                Path, "glob", return_value=[Path("/logs/verifier/premature.log")]
            ),
        ):
            self.assertFalse(self.scan()["ok"])
        with patch.object(gold_probe, "MAX_FILES", 0):
            self.assertFalse(self.scan()["ok"])

    def test_manifest_hash_mismatch_cli_refuses(self):
        env = self.env()
        with (
            patch.dict(os.environ, env),
            patch.object(gold_probe.os, "walk", return_value=[]),
        ):
            argv = ["isolation", self.sha, *gold_probe.provenance_args()]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(gold_probe.main(argv), 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])
            argv[-3] = "e" * 64
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(gold_probe.main(argv), 1)
            self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_both_boundaries_pass_identical_manifest_and_refuse_malformed_results(self):
        env = self.env()
        good = {**self.scan(), "manifest_sha256": env["BENCH_PUBLIC_PROVENANCE_SHA256"]}
        for response in (
            json.dumps(good),
            "invalid",
            "[]",
            '{"ok":true}',
            json.dumps({**good, "ok": False}),
            json.dumps({**good, "manifest_sha256": "f" * 64}),
        ):
            commands = []
            adapter = _load()

            async def execute(command, response=response, commands=commands, **kwargs):
                commands.append(command)
                return SimpleNamespace(
                    stdout=response if " isolation " in command else '{"ok":true}',
                    return_code=0,
                )

            accepted = response == json.dumps(good)
            with patch.dict(os.environ, env):
                if accepted:
                    asyncio.run(
                        adapter._preinference_checks(
                            SimpleNamespace(logs_dir=None),
                            SimpleNamespace(exec=execute),
                        )
                    )
                else:
                    with self.assertRaises(RuntimeError):
                        asyncio.run(
                            adapter._preinference_checks(
                                SimpleNamespace(logs_dir=None),
                                SimpleNamespace(exec=execute),
                            )
                        )
                verifier, environment, calls = _load_verifier(
                    [response, '{"survivors":[]}']
                )
                paths = SimpleNamespace(verifier_dir=self.root / "verifier")
                with (
                    patch.object(verifier, "compose_project", return_value="proj"),
                    patch.object(
                        verifier, "connect_uplink", return_value="cid"
                    ) as uplink,
                ):
                    if accepted:
                        self.assertEqual(
                            asyncio.run(
                                verifier.BenchVerifier(environment, paths).verify()
                            ),
                            "graded",
                        )
                    else:
                        with self.assertRaises(RuntimeError):
                            asyncio.run(
                                verifier.BenchVerifier(environment, paths).verify()
                            )
                        uplink.assert_not_called()
                        self.assertNotIn("stock-verify", calls)
            self.assertEqual(shlex.split(commands[0])[2:], shlex.split(calls[0])[2:])

    def test_driver_refuses_stale_reviewed_inputs_without_editing_shared_tasks(self):
        task = self.root / "fixture"
        (task / "solution").mkdir(parents=True)
        (task / "environment").mkdir()
        (task / "solution/solve.sh").write_text("solution sentinel")
        (task / "environment/Dockerfile").write_text("FROM scratch\n")
        image = "sha256:" + "b" * 64
        manifest = driver.public_provenance(task, image)
        approved = {
            "task_payload_sha256": manifest["task_payload_sha256"],
            "image_digest": image,
            "dockerfile_sha256": driver.digest(task / "environment/Dockerfile"),
            "public_files": [],
        }
        policy_dir = self.root / "policy"
        policy_dir.mkdir()
        (policy_dir / "public-provenance.json").write_text(
            json.dumps({"version": 1, "tasks": {"fixture": approved}})
        )
        with patch.object(driver, "HERE", policy_dir):
            self.assertEqual(driver.public_provenance(task, image)["public_files"], [])
            with self.assertRaisesRegex(ValueError, "stale"):
                driver.public_provenance(task, "sha256:" + "c" * 64)
            (task / "environment/Dockerfile").write_text("FROM changed\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                driver.public_provenance(task, image)


if __name__ == "__main__":
    unittest.main()

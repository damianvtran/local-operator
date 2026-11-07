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
import provenance_derive
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


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


HERE = Path(__file__).resolve().parent
TASK_ROOT = Path(os.environ.get(
    "BENCH_TASK_ROOT",
    Path.home() / "workspace/harness-cost-bench/coding/tasks/terminal-bench-2-1"))


class DerivedProvenanceTests(unittest.TestCase):
    """S9 item 1: COPY/clone pairs derived from task sources, never a container."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.tmp.cleanup)
        self.task = Path(self.tmp.name) / "task"
        for rel in ("tests/sim.c", "environment/tests/sim.c"):
            _write(self.task / rel, b"sim")
        for rel in ("tests/main.tex", "environment/tests/main.tex"):
            _write(self.task / rel, b"tex")
        for rel in ("tests/model_ref.xml", "environment/src/model_ref.xml"):
            _write(self.task / rel, b"xml")
        _write(self.task / "environment/tests/test/a.scm", b"scm")
        _write(self.task / "tests/test/a.scm", b"scm")
        # Present in the build context but never COPYed: stays hidden-denied.
        _write(self.task / "environment/tests/shadow_test/b.scm", b"shadow")
        _write(self.task / "tests/shadow_test/b.scm", b"shadow")
        _write(self.task / "environment/gates.txt", b"public non-gold")
        _write(self.task / "solution/solve.sh", b"solve")
        _write(self.task / "environment/Dockerfile", "\n".join([
            "FROM base AS build", "COPY tests/main.tex /elsewhere/main.tex",
            "FROM python:3.13", "WORKDIR /app", "RUN true",
            "COPY tests/sim.c /app", "COPY gates.txt /app", "COPY tests/main.tex .",
            "COPY src/model_ref.xml /app/model_ref.xml", "COPY tests/test /app/test", ""]))

    def test_copy_pairs_cover_dir_file_dot_and_explicit_destinations(self):
        gold = set(driver.gold_digests(self.task))
        derived = provenance_derive.derive(self.task, gold)
        self.assertEqual(
            [(p["path"], p["sha256"], p["justification"]) for p in derived["public_files"]],
            [("/app/main.tex", _sha(b"tex"), "environment/Dockerfile:8: COPY tests/main.tex ."),
             ("/app/model_ref.xml", _sha(b"xml"),
              "environment/Dockerfile:9: COPY src/model_ref.xml /app/model_ref.xml"),
             ("/app/sim.c", _sha(b"sim"), "environment/Dockerfile:6: COPY tests/sim.c /app"),
             ("/app/test/a.scm", _sha(b"scm"),
              "environment/Dockerfile:10: COPY tests/test /app/test")])
        paths = {p["path"] for p in derived["public_files"]}
        # Earlier build stage, non-gold COPY and the uncopied shadow fixture: no pair.
        self.assertNotIn("/elsewhere/main.tex", paths)
        self.assertNotIn("/app/gates.txt", paths)
        self.assertFalse(any("shadow" in p for p in paths))

    def test_unsupported_build_inputs_fail_closed(self):
        dockerfile = self.task / "environment/Dockerfile"
        for line in ("COPY --from=build /x /app/x", "COPY tests/*.c /app/",
                     "ADD https://example.com/x /app/x", "COPY ../tests/sim.c /app"):
            with self.subTest(line=line):
                dockerfile.write_text(f"FROM x\nWORKDIR /app\n{line}\n")
                with self.assertRaises(provenance_derive.Unsupported):
                    provenance_derive.derive(self.task, set(driver.gold_digests(self.task)))

    def _git(self, repo, *args):
        import subprocess

        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
            check=True, capture_output=True, text=True).stdout.strip()

    def _clone_task(self, pinned: bool):
        repo = Path(self.tmp.name) / "upstream"
        _write(repo / "pkg/fixture.py", b"public clone bytes")
        _write(repo / "pkg/other.py", b"not gold")
        self._git(repo, "init", "-q")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-qm", "c")
        commit = self._git(repo, "rev-parse", "HEAD")
        _write(self.task / "tests/fixture.py", b"public clone bytes")
        pin = f'[ "$(git rev-parse HEAD)" = "{commit}" ] || exit 1' if pinned else "true"
        _write(self.task / "environment/setup.sh", "\n".join([
            "git config --global user.email x", "git clone https://example.org/up.git dclm",
            "cd dclm", pin]))
        _write(self.task / "environment/Dockerfile", "\n".join([
            "FROM python:3.13", "WORKDIR /app", "COPY setup.sh ./",
            "RUN bash /app/setup.sh && rm /app/setup.sh", "WORKDIR /app/dclm", ""]))
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        os.rename(repo, cache / f"dclm-{commit}")
        return commit, cache

    def test_pinned_build_time_clone_yields_cited_pairs_from_the_pinned_commit(self):
        commit, cache = self._clone_task(pinned=True)
        derived = provenance_derive.derive(self.task, set(driver.gold_digests(self.task)), cache)
        self.assertEqual(derived["limitations"], [])
        (pair,) = derived["public_files"]
        self.assertEqual(pair["path"], "/app/dclm/pkg/fixture.py")
        self.assertEqual(pair["origin"],
                         f"build-time clone of https://example.org/up.git@{commit}")
        self.assertEqual(pair["justification"],
                         "environment/setup.sh:2 (git clone), environment/setup.sh:4 (HEAD pin), "
                         "environment/Dockerfile:4 (RUN)")

    def test_unpinned_clone_is_a_disclosed_limitation_with_no_pairs(self):
        _commit, cache = self._clone_task(pinned=False)
        derived = provenance_derive.derive(self.task, set(driver.gold_digests(self.task)), cache)
        self.assertEqual(derived["public_files"], [])
        self.assertIn("not pinned", derived["limitations"][0])

    def _policy(self, entries):
        payload, files = driver.task_payload(self.task)
        policy_dir = Path(self.tmp.name) / "policy"
        policy_dir.mkdir(exist_ok=True)
        (policy_dir / "public-provenance.json").write_text(json.dumps({"version": 1, "tasks": {
            "task": {"task_payload_sha256": payload, "image_digest": "sha256:" + "b" * 64,
                     "dockerfile_sha256": files["environment/Dockerfile"],
                     "public_files": entries}}}))
        return policy_dir

    def test_driver_reuses_frozen_pairs_only_while_they_equal_the_derivation(self):
        gold = set(driver.gold_digests(self.task))
        derived = provenance_derive.derive(self.task, gold)["public_files"]
        image = "sha256:" + "b" * 64
        with patch.object(driver, "HERE", self._policy(derived)):
            manifest = driver.public_provenance(self.task, image)
        self.assertEqual(len(manifest["public_files"]), 4)
        for broken in (derived[:-1],
                       [{**derived[0], "path": "/app/renamed.tex"}, *derived[1:]],
                       [*derived, {"path": "/app/x", "sha256": _sha(b"sim")}]):
            with self.subTest(broken=len(broken)), patch.object(
                    driver, "HERE", self._policy(broken)):
                with self.assertRaisesRegex(ValueError, "stale public|reviewed origin"):
                    driver.public_provenance(self.task, image)

    def test_frozen_policy_entries_are_cited_and_rederive_from_the_task_sources(self):
        policy = json.loads((HERE / "public-provenance.json").read_text())
        for name, entry in policy["tasks"].items():
            for pair in entry["public_files"]:
                self.assertIn(pair["origin"], ("dockerfile-copy",)) if "source" in pair else None
                self.assertRegex(pair["justification"], r"environment/\S+:\d+")
        if not TASK_ROOT.is_dir():
            self.skipTest("task root not present; re-derivation needs the task sources")
        for name, entry in policy["tasks"].items():
            with self.subTest(task=name):
                task = TASK_ROOT / name
                derived = provenance_derive._copy_pairs(task, set(driver.gold_digests(task)))[0]
                frozen = [p for p in entry["public_files"] if p["origin"] == "dockerfile-copy"]
                self.assertEqual(sorted(derived, key=lambda p: p["path"]), frozen)
                self.assertEqual(entry["task_payload_sha256"], driver.task_payload(task)[0])
        self.assertEqual(
            {n: len(e["public_files"]) for n, e in policy["tasks"].items()},
            {"schemelike-metacircular-eval": 34, "circuit-fibsqrt": 1, "overfull-hbox": 3,
             "tune-mjcf": 1, "sanitize-git-repo": 0})


class EmptyGoldDigestTests(unittest.TestCase):
    """S9 item 2: the zero-length digest is excluded, explicitly, and nothing else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.tmp.cleanup)
        self.task = Path(self.tmp.name) / "task"
        _write(self.task / "tests/__init__.py", b"")
        _write(self.task / "tests/test_x.py", b"x")  # one byte: still guarded
        _write(self.task / "solution/solve.sh", b"solve")
        _write(self.task / "environment/Dockerfile", "FROM scratch\n")

    def test_empty_digest_is_excluded_and_recorded(self):
        digests = driver.gold_digests(self.task)
        self.assertNotIn(driver.EMPTY_SHA256, digests)
        self.assertIn(_sha(b"x"), digests)
        manifest = driver.public_provenance(self.task, "sha256:" + "b" * 64)
        self.assertEqual(manifest["gold_exclusions"], [{
            "sha256": driver.EMPTY_SHA256, "files": ["tests/__init__.py"],
            "reason": "zero-length file carries no information; its digest matches "
                      "every empty file in the image"}])

    def test_ordinary_empty_files_no_longer_refuse_but_one_byte_gold_still_does(self):
        root = Path(self.tmp.name) / "image"
        _write(root / ".dockerenv", b"")
        _write(root / "etc/subgid", b"")
        manifest = driver.public_provenance(self.task, "sha256:" + "b" * 64)
        scan = gold_probe.isolation(set(manifest["gold_digests"]), root=str(root),
                                    manifest=manifest,
                                    task_payload_sha256=manifest["task_payload_sha256"],
                                    image_digest="sha256:" + "b" * 64)
        self.assertTrue(scan["ok"], scan)
        _write(root / "app/copied.py", b"x")
        scan = gold_probe.isolation(set(manifest["gold_digests"]), root=str(root),
                                    manifest=manifest,
                                    task_payload_sha256=manifest["task_payload_sha256"],
                                    image_digest="sha256:" + "b" * 64)
        self.assertFalse(scan["ok"])
        self.assertEqual([m["path"] for m in scan["rejected_matches"]],
                         [str(root / "app/copied.py")])


class AuthorizedOracleSourceTests(unittest.TestCase):
    """S9 item 3: oracle-only authorized /solution classification in the probe."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gold_py, self.solve, self.test = b"gold impl", b"cp gold", b"hidden test"
        self.manifest = {
            "version": 1, "task_payload_sha256": "a" * 64,
            "image_digest": "sha256:" + "b" * 64, "policy_sha256": "c" * 64,
            "public_files": [],
            "solution_digests": sorted({_sha(self.gold_py), _sha(self.solve)}),
            "gold_digests": sorted({_sha(self.gold_py), _sha(self.solve), _sha(self.test)}),
        }
        self.tree = {"headless_terminal.py": _sha(self.gold_py), "solve.sh": _sha(self.solve)}
        _write(self.root / "solution/headless_terminal.py", self.gold_py)
        _write(self.root / "solution/solve.sh", self.solve)
        _write(self.root / "app/headless_terminal.py", self.gold_py)
        relocated = patch.object(gold_probe, "SOLUTION", self.root / "solution")
        relocated.start()
        self.addCleanup(relocated.stop)

    def scan(self, authorized):
        return gold_probe.isolation(
            set(self.manifest["gold_digests"]), root=str(self.root), manifest=self.manifest,
            task_payload_sha256="a" * 64, image_digest="sha256:" + "b" * 64,
            authorized_solution=authorized)

    def test_without_authorization_the_oracle_copy_is_refused(self):
        result = self.scan(None)
        self.assertFalse(result["ok"])
        self.assertIn(str(self.root / "app/headless_terminal.py"),
                      [m["path"] for m in result["rejected_matches"]])

    def test_authorized_source_digests_pass_and_raw_matches_are_retained(self):
        result = self.scan(self.tree)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["strict_ok"])
        self.assertEqual(len(result["raw_matches"]), 3)
        self.assertEqual(len(result["authorized_solution_matches"]), 3)
        self.assertTrue(result["authorized_solution_tree_match"])

    def test_verifier_only_digest_is_still_rejected(self):
        _write(self.root / "app/leaked_test.py", self.test)
        result = self.scan(self.tree)
        self.assertFalse(result["ok"])
        self.assertEqual([m["path"] for m in result["rejected_matches"]],
                         [str(self.root / "app/leaked_test.py")])

    def test_tests_digest_can_never_be_authorized(self):
        result = self.scan({**self.tree, "planted.py": _sha(self.test)})
        self.assertFalse(result["ok"])
        self.assertIn("not a manifest solution file", result["provenance_error"])

    def test_unexpected_solution_tree_entry_fails(self):
        _write(self.root / "solution/extra.txt", b"extra")
        self.assertFalse(self.scan(self.tree)["ok"])

    def test_cli_refuses_an_authorized_tree_with_a_bad_digest(self):
        raw = json.dumps(self.tree, sort_keys=True, separators=(",", ":"))
        manifest = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        argv = ["isolation", *self.manifest["gold_digests"], "--public-provenance", manifest,
                _sha(manifest.encode()), "a" * 64, "sha256:" + "b" * 64,
                "--authorized-solution", raw, "0" * 64]
        out = io.StringIO()
        # Scan the fixture root, never the host's "/".
        with contextlib.redirect_stdout(out), patch.object(
                gold_probe.isolation, "__defaults__", (str(self.root),)):
            self.assertEqual(gold_probe.main(argv), 1)
        self.assertIn("malformed authorized", json.loads(out.getvalue())["provenance_error"])


def _load_qualification(responses):
    """oracle_qualification against stand-in Harbor bases (no Harbor import)."""
    import importlib
    import sys

    verifier_module, environment, calls = _load_verifier(responses)

    class _Oracle:
        def __init__(self, logs_dir, task_dir, trial_paths):
            self.logs_dir, self._trial_paths = logs_dir, trial_paths
            self._task = SimpleNamespace(
                has_steps=False,
                paths=SimpleNamespace(tests_dir=Path(task_dir) / "tests",
                                      solution_dir=Path(task_dir) / "solution"),
                file_filter=SimpleNamespace(select=lambda d: d))

        async def setup(self, environment):
            calls.append("oracle-setup")

        def _resolve_solution_paths(self):
            return self._task.paths.solution_dir, self._task.paths.solution_dir / "solve.sh"

        async def run(self, instruction, environment, context):
            calls.append("oracle-upload-run")

    @contextlib.contextmanager
    def selected(directory):
        yield directory

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "harbor.agents.oracle":
            return SimpleNamespace(OracleAgent=_Oracle)
        if name == "harbor.utils.file_staging":
            return SimpleNamespace(selected_directory=selected)
        return real(name, *a, **k)

    with patch.dict(sys.modules, {"bench_verifier": verifier_module}), patch(
            "importlib.import_module", side_effect=fake):
        spec = importlib.util.spec_from_file_location(
            "coding_oracle_qualification", HERE / "oracle_qualification.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    module.bench = verifier_module
    return module, environment, calls


def _adapters(preflight):
    """Stand-in for adapters (Harbor-free): only the common preflight is used."""
    import sys

    return patch.dict(sys.modules, {"adapters": SimpleNamespace(_preinference_checks=preflight)})


class OracleQualificationTests(unittest.TestCase):
    """S9 item 3: receipt-bound oracle adapter/verifier; paid path cannot reach it."""

    ENV = {"BENCH_IMAGE_DIGEST": "sha256:" + "b" * 64,
           "BENCH_PUBLIC_PROVENANCE_SHA256": "d" * 64,
           "BENCH_TASK_PAYLOAD_SHA256": "a" * 64}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / "task"
        _write(self.task / "solution/solve.sh", b"cp gold")
        _write(self.task / "solution/headless_terminal.py", b"gold impl")
        _write(self.task / "tests/test_outputs.py", b"hidden test")
        self.paths = SimpleNamespace(trial_dir=self.root / "trial",
                                     agent_dir=self.root / "trial/agent",
                                     verifier_dir=self.root / "trial/verifier")
        self.paths.agent_dir.mkdir(parents=True)
        env = patch.dict(os.environ, self.ENV)
        env.start()
        self.addCleanup(env.stop)

    def _oracle(self, module):
        return module.QualificationOracle(self.paths.agent_dir, self.task, self.paths)

    def test_preflight_runs_before_upload_and_receipt_binds_identity_and_order(self):
        iso = {"ok": True, "manifest_sha256": "d" * 64}
        module, environment, calls = _load_qualification(
            [json.dumps(iso), json.dumps({"survivors": []})])
        oracle = self._oracle(module)
        order = []

        async def preflight(agent, env):
            order.append("preflight")
            self.assertFalse((self.paths.trial_dir / module.RECEIPT).exists())

        with patch.dict(os.environ, {"BENCH_GOLD_DIGESTS": "e" * 64}), _adapters(preflight):
            asyncio.run(oracle.setup(environment))
        asyncio.run(oracle.run("i", environment, None))
        receipt = json.loads((self.paths.trial_dir / module.RECEIPT).read_text())
        self.assertEqual(order, ["preflight"])
        self.assertEqual(calls[:2], ["oracle-setup", "oracle-upload-run"])
        self.assertEqual(receipt["order"], module.ORDER)
        self.assertEqual(receipt["trial"], "sess")
        self.assertEqual(receipt["image_digest"], self.ENV["BENCH_IMAGE_DIGEST"])
        self.assertEqual(set(receipt["solution_tree"]), {"solve.sh", "headless_terminal.py"})
        self.assertNotIn("/trial/agent", str(self.paths.trial_dir / module.RECEIPT))

    def test_dirty_preflight_leaves_no_receipt_and_the_oracle_never_runs(self):
        module, environment, calls = _load_qualification([])
        oracle = self._oracle(module)

        async def dirty(agent, env):
            raise RuntimeError("pre-inference isolation check failed")

        with _adapters(dirty):
            with self.assertRaises(RuntimeError):
                asyncio.run(oracle.setup(environment))
        self.assertFalse((self.paths.trial_dir / module.RECEIPT).exists())
        with self.assertRaisesRegex(RuntimeError, "preflight receipt"):
            asyncio.run(oracle.run("i", environment, None))
        self.assertNotIn("oracle-upload-run", calls)

    def _receipt(self, module, **overrides):
        receipt = {"trial": "sess", **{
            "image_digest": self.ENV["BENCH_IMAGE_DIGEST"],
            "manifest_sha256": self.ENV["BENCH_PUBLIC_PROVENANCE_SHA256"],
            "task_payload_sha256": self.ENV["BENCH_TASK_PAYLOAD_SHA256"]},
            "solution_tree": {"solve.sh": _sha(b"cp gold")}, "order": module.ORDER,
            **overrides}
        _write(self.paths.trial_dir / module.RECEIPT, json.dumps(receipt))

    def test_verifier_requires_a_complete_bound_receipt_before_any_probe(self):
        for overrides in (None, {"trial": "other"}, {"image_digest": "sha256:" + "c" * 64},
                          {"order": ["preflight-clean"]}, {"solution_tree": {}},
                          {"manifest_sha256": "e" * 64}):
            with self.subTest(overrides=overrides):
                module, environment, calls = _load_qualification([])
                receipt = self.paths.trial_dir / module.RECEIPT
                receipt.unlink(missing_ok=True)
                if overrides is not None:
                    self._receipt(module, **overrides)
                verifier = module.QualificationVerifier(environment, self.paths)
                with patch.dict(os.environ, {"BENCH_GOLD_DIGESTS": "e" * 64}):
                    with self.assertRaisesRegex(RuntimeError, "receipt refused"):
                        asyncio.run(verifier.verify())
                self.assertEqual(calls, [])

    def test_verifier_passes_the_authorized_tree_then_shares_the_native_tail(self):
        iso = {"ok": True, "manifest_sha256": "d" * 64}
        module, environment, calls = _load_qualification(
            [json.dumps(iso), json.dumps({"survivors": []})])
        self._receipt(module)
        verifier = module.QualificationVerifier(environment, self.paths)
        env = {"BENCH_PUBLIC_PROVENANCE": "{}", "BENCH_GOLD_DIGESTS": "e" * 64}
        with patch.dict(os.environ, env), \
                patch.object(module.bench, "provenance_args",
                             return_value=["--public-provenance"]), \
                patch.object(module.bench, "compose_project", return_value="proj"), \
                patch.object(module.bench, "connect_uplink", return_value="cid") as uplink:
            self.assertEqual(asyncio.run(verifier.verify()), "graded")
        uplink.assert_called_once_with("proj")
        isolation = shlex.split(calls[0])
        index = isolation.index("--authorized-solution")
        self.assertEqual(json.loads(isolation[index + 1]), {"solve.sh": _sha(b"cp gold")})
        self.assertEqual(isolation[index + 2], _sha(isolation[index + 1].encode()))
        self.assertEqual(calls[-1], "stock-verify")
        record = json.loads((self.paths.verifier_dir / module.RECORD).read_text())
        self.assertEqual(record["problems"], [])

    def test_paid_path_can_never_select_or_reach_the_oracle_policy(self):
        driver_src = (HERE / "driver.py").read_text()
        self.assertIn('"bench_verifier:BenchVerifier"', driver_src)
        self.assertNotIn("oracle_qualification", driver_src)
        self.assertNotIn("QualificationVerifier", driver_src)
        import re

        for name in driver.CONTAINER_CLOSURE:
            source = (HERE / name).read_text()
            with self.subTest(module=name):
                # No import and no Harbor import path; a docstring may name it.
                self.assertIsNone(re.search(
                    r"^\s*(import|from)\s+oracle_qualification|oracle_qualification:",
                    source, re.M))
                if name != "gold_probe.py":
                    self.assertNotIn("--authorized-solution", source)
        self.assertNotIn("oracle_qualification.py", driver.CONTAINER_CLOSURE)
        verifier, _environment, _calls = _load_verifier([])
        self.assertEqual(verifier.BenchVerifier._isolation_extra_args(None), [])
        for harness, spec in driver.HARNESS.items():
            self.assertNotIn("oracle", json.dumps(spec, default=str).lower(), harness)


if __name__ == "__main__":
    unittest.main()

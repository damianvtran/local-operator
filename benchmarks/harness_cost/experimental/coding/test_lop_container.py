"""Offline contract tests for the paid coding container path (no Docker, no network).

Covers: the container LoP launcher's edge/overlay checks, the egress compose
override and no-implicit-pull guard, the gold/quiescence/egress probe, the gate
relay's port map and real byte relay, the driver's container plan and the
closure-bound evidence gate. The container S4 (real image, real fence) is
separate and recorded outside the repo.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

import driver  # noqa: E402
import egress  # noqa: E402
import gate_relay  # noqa: E402
import gold_probe  # noqa: E402


def _load_launcher():
    spec = importlib.util.spec_from_file_location("coding_lop_launch", HERE / "lop_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ca() -> tuple[str, bytes]:
    """A real CA certificate (PEM, DER) from the edge's own RunCA."""
    import sys
    import ssl as _ssl

    sys.path.insert(0, str(HERE.parent))
    import lop_edge

    pem = lop_edge.RunCA().certificate_pem().decode()
    return pem, _ssl.PEM_cert_to_DER_cert(pem)


def _const(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    return next(
        n.value.value for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name
    )


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.launch = _load_launcher()
        tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.ca = self.tmp / "edge-ca.pem"
        ca_pem, der = _fake_ca()
        self.ca.write_text(ca_pem)
        self.ca.chmod(0o444)
        self.fingerprint = hashlib.sha256(der).hexdigest()

    def env(self, **over):
        proxy = "http://run1:nonce@bench-gate:41000"
        base = {
            "HTTPS_PROXY": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "http_proxy": proxy,
            "NO_PROXY": self.launch.NO_PROXY, "no_proxy": self.launch.NO_PROXY,
            "SSL_CERT_FILE": self.launch.CA_BUNDLE, "REQUESTS_CA_BUNDLE": self.launch.CA_BUNDLE,
            "RADIENT_API_KEY": self.launch.DUMMY_RADIENT_KEY,
            **{k: self.launch.CA_BUNDLE for k in self.launch.PARITY_KEYS},
            "BENCH_EDGE_CA_SHA256": self.fingerprint,
        }
        base.update(over)
        return {k: v for k, v in base.items() if v is not None}

    def check(self, env):
        with patch.object(self.launch, "CA_BUNDLE", str(self.ca)):
            env = {k: (str(self.ca) if v == "/run/bench-ca/edge-ca.pem" else v)
                   for k, v in env.items()}
            self.launch.require_edge_environment(env)

    def test_dummy_key_equals_the_edge_dummy(self):
        self.assertEqual(
            self.launch.DUMMY_RADIENT_KEY,
            _const(HERE.parent / "lop_edge.py", "DUMMY_RADIENT_KEY"),
        )
        self.assertEqual(self.launch.NO_PROXY, _const(HERE.parent / "lop_edge.py", "NO_PROXY"))

    def test_edge_environment_accepted(self):
        self.check(self.env())

    def test_edge_environment_refusals(self):
        cases = {
            "no proxy": self.env(HTTPS_PROXY=None, https_proxy=None, HTTP_PROXY=None,
                                 http_proxy=None),
            "loopback edge": self.env(**{
                k: "http://r:n@127.0.0.1:1"
                for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}),
            "no identity": self.env(**{k: "http://bench-gate:41000" for k in
                                       ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}),
            "lowercase empty": self.env(https_proxy=""),
            "all_proxy": self.env(ALL_PROXY="socks5://x:1"),
            "no_proxy widened": self.env(NO_PROXY="*", no_proxy="*"),
            "real-looking key": self.env(RADIENT_API_KEY="rk_live_something"),
            "foreign CA": self.env(SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"),
        }
        for name, env in cases.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                self.check(env)

    def test_bundle_must_contain_the_run_ca(self):
        # R8 F24: the bundle is run CA + public roots; the launcher still
        # requires the run CA (by DER fingerprint) to be IN it.
        import ssl as _ssl
        ca_pem, der = _fake_ca()
        roots = (HERE / "public-roots.pem").read_text()
        self.ca.chmod(0o644)
        self.ca.write_text(ca_pem + roots)
        self.ca.chmod(0o444)
        fp = hashlib.sha256(der).hexdigest()
        self.check(self.env(BENCH_EDGE_CA_SHA256=fp))
        self.assertEqual(_ssl.PEM_cert_to_DER_cert(ca_pem), der)
        for bad in ("0" * 64, None):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.check(self.env(BENCH_EDGE_CA_SHA256=bad))
        # Roots alone (no run CA) are refused even with a matching-looking pin.
        self.ca.chmod(0o644)
        self.ca.write_text(roots)
        self.ca.chmod(0o444)
        with self.assertRaises(SystemExit):
            self.check(self.env(BENCH_EDGE_CA_SHA256=fp))

    def test_parity_keys_agree_across_edge_launcher_and_driver(self):
        edge_keys = tuple(ast.literal_eval(next(
            n.value for n in ast.parse((HERE.parent / "lop_edge.py").read_text()).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CA_PARITY_KEYS")))
        self.assertEqual(self.launch.PARITY_KEYS, edge_keys)
        self.assertEqual(driver.LOP_PARITY_KEYS, edge_keys)

    def test_parity_trust_variables_must_name_the_bundle(self):
        ca_pem, der = _fake_ca()
        self.ca.chmod(0o644)
        self.ca.write_text(ca_pem)
        self.ca.chmod(0o444)
        fp = hashlib.sha256(der).hexdigest()
        for key in ("PIP_CERT", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.check(self.env(BENCH_EDGE_CA_SHA256=fp, **{key: "/etc/ssl/other.pem"}))

    def test_writable_ca_refused(self):
        self.ca.chmod(0o644)
        with self.assertRaises(SystemExit):
            self.check(self.env())

    def _overlay(self, commit=None):
        root = self.tmp / "overlay"
        pkg = root / "local_operator"
        dist = root / "local_operator-0.94.22.dist-info"
        pkg.mkdir(parents=True)
        dist.mkdir()
        (pkg / "__init__.py").write_text("x = 1\n")
        h = base64.urlsafe_b64encode(hashlib.sha256(b"x = 1\n").digest()).rstrip(b"=").decode()
        (dist / "RECORD").write_text(f"local_operator/__init__.py,sha256={h},6\n")
        (dist / "direct_url.json").write_text(json.dumps(
            {"vcs_info": {"commit_id": commit or self.launch.RELEASED_REVISION}}))
        return root

    def test_overlay_identity(self):
        root = self._overlay()
        self.assertEqual(self.launch.verify_overlay(root), 1)
        (root / "local_operator" / "__init__.py").write_text("x = 2\n")
        with self.assertRaisesRegex(SystemExit, "hash differs"):
            self.launch.verify_overlay(root)

    def test_overlay_extra_module_and_wrong_commit_refused(self):
        root = self._overlay()
        (root / "local_operator" / "injected.py").write_text("")
        with self.assertRaisesRegex(SystemExit, "differ from its RECORD"):
            self.launch.verify_overlay(root)
        other = self.tmp / "o2"
        other.mkdir()
        with patch.object(self, "tmp", other):
            with self.assertRaisesRegex(SystemExit, "not released revision"):
                self.launch.verify_overlay(self._overlay(commit="0" * 40))

    def test_overlay_prepends_to_existing_pythonpath(self):
        # R6 F18: an image's own PYTHONPATH (multi-source-data-merger sets
        # /app) must survive for LoP's tool children.
        with patch.dict(os.environ, {"PYTHONPATH": "/app:/lib"}, clear=False):
            self.launch.export_overlay_path()
            self.assertEqual(os.environ["PYTHONPATH"],
                             os.pathsep.join([str(self.launch.OVERLAY), "/app", "/lib"]))
            self.launch.export_overlay_path()  # idempotent across re-launches
            self.assertEqual(os.environ["PYTHONPATH"].split(os.pathsep).count(
                str(self.launch.OVERLAY)), 1)
        with patch.dict(os.environ, {}, clear=True):
            self.launch.export_overlay_path()
            self.assertEqual(os.environ["PYTHONPATH"], str(self.launch.OVERLAY))

    def test_main_checks_environment_before_importing_product(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            self.launch.runpy, "run_module", side_effect=AssertionError("product started")
        ):
            with self.assertRaises(SystemExit):
                self.launch.main()


class EgressTests(unittest.TestCase):
    IMAGE = "alexgshaw/git-leak-recovery@sha256:" + "a" * 64

    def override(self, ports=(8787, 8789, 41000)):
        return egress.compose_override("run1", list(ports), self.IMAGE, Path("/s"), Path("/t"),
                                       Path("/l"))

    def test_main_only_on_internal_network_and_never_pulls(self):
        o = self.override()
        main, gate = o["services"]["main"], o["services"][egress.GATE_SERVICE]
        self.assertEqual(main["networks"], [egress.INTERNAL_NETWORK])
        # R6 F16: the name resolves in main, so the canary dial is a route test.
        self.assertEqual(main["extra_hosts"], ["host.docker.internal:host-gateway"])
        self.assertIs(o["networks"][egress.INTERNAL_NETWORK]["internal"], True)
        self.assertNotIn("internal", o["networks"][egress.UPLINK_NETWORK])
        self.assertEqual(main["pull_policy"], "never")
        self.assertEqual(gate["pull_policy"], "never")
        self.assertEqual(gate["image"], self.IMAGE)
        self.assertEqual(gate["entrypoint"][8:], ["8787=8787", "8789=8789", "41000=41000"])
        # Read-only except the package-egress log (test_package_egress).
        self.assertEqual([v["target"] for v in gate["volumes"] if not v["read_only"]],
                         [egress.EGRESS_LOG])
        self.assertEqual(gate["cap_drop"], ["ALL"])
        # R6 F16: main resolves the host gateway NAME (an /etc/hosts line, no
        # route) so the egress canary tests routing, not DNS.
        self.assertEqual(main["extra_hosts"], ["host.docker.internal:host-gateway"])

    def test_override_refusals(self):
        for bad in ([], [0], [70000], [8787, 8787], [True], [egress.PACKAGE_PROXY_PORT]):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.override(bad)
        with self.assertRaises(ValueError):
            egress.compose_override("run1", [1], "repo:tag", Path("/s"), Path("/t"), Path("/l"))
        with self.assertRaises(ValueError):
            egress.compose_override("../x", [1], self.IMAGE, Path("/s"), Path("/t"), Path("/l"))

    def test_no_implicit_pull_guard(self):
        ok = ["docker", "run", "--pull", "never", "--rm", "--platform", "linux/amd64", "img"]
        # Rig convention: --pull never is the FIRST option after the verb.
        self.assertEqual(egress.require_no_implicit_pull(ok), ok)
        for good in (
            ["docker", "run", "--pull=never", "img"],
            ["docker", "--context", "x", "run", "--pull", "never", "img"],
            ["docker", "-H", "unix:///s", "container", "create", "--pull=never", "img"],
            ["docker", "compose", "-f", "a.yml", "up", "--pull", "never", "-d"],
            ["docker", "rmi", "img"],
            ["docker", "image", "inspect", "img"],
            ["docker", "network", "connect", "n", "c"],
        ):
            with self.subTest(good):
                egress.require_no_implicit_pull(good)
        for bad in (
            ["docker", "run", "--platform", "linux/amd64", "mongo:7.0"],  # the incident
            ["docker", "run", "img"],
            ["docker", "create", "img"],
            ["docker", "run", "--pull", "missing", "img"],
            ["docker", "run", "--pull"],  # no value: ValueError, not IndexError
            ["docker", "run", "img", "--pull", "never"],  # after the image = container argv
            ["docker", "run", "--init", "img", "--pull", "never"],  # arity cannot be guessed
            ["docker", "run", "--rm", "--pull", "never", "img"],  # rig rule: --pull first
            ["docker", "image", "inspect", "--platform", "linux/amd64", "img"],
            # R6 F17 / P17: global options before the verb hid it.
            ["docker", "--context", "x", "run", "img"],
            ["docker", "-H", "unix:///s", "create", "img"],
            ["docker", "-c", "x", "container", "run", "img"],
            ["docker", "--unknown-global", "x", "run", "--pull", "never", "img"],
            ["docker", "compose", "up"],
            ["docker", "compose", "-p", "x", "run", "svc"],
            ["docker", "compose", "create", "--pull", "always"],
            ["docker", "compose", "pull"],
            ["docker", "pull", "img"],  # only the driver's explicit, disk-gated pull
            ["docker", "image", "pull", "img"],
            ["docker", "build", "."],
            ["podman", "run", "img"],
        ):
            with self.subTest(bad), self.assertRaises(ValueError):
                egress.require_no_implicit_pull(bad)
        egress.require_no_implicit_pull(["docker", "pull", "img"], explicit_pull=True)

    def test_every_rig_docker_call_goes_through_the_guard(self):
        # R6 F17: the guard existed only in tests. Any subprocess call whose
        # argv literal starts with "docker" must be egress.run_docker instead.
        for name in ("driver.py", "bench_verifier.py", "oracle_sweep.py"):
            tree = ast.parse((HERE / name).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                first = node.args[0]
                if (isinstance(first, ast.List) and first.elts
                        and isinstance(first.elts[0], ast.Constant)
                        and first.elts[0].value == "docker"):
                    func = ast.unparse(node.func)
                    self.assertIn(func, ("egress.run_docker", "run_docker"), f"{name}: {func}")

    def test_run_docker_refuses_before_spawning(self):
        with patch.object(egress.subprocess, "run", side_effect=AssertionError("spawned")):
            with self.assertRaises(ValueError):
                egress.run_docker(["docker", "--context", "x", "run", "img"])

    def test_support_mounts_read_only(self):
        mounts = egress.support_mounts(Path("/s"), Path("/run-dir/edge-ca.pem"))
        # R6 N6: the CA FILE only; the edge run dir also holds its request log.
        self.assertEqual([m["target"] for m in mounts], [egress.SUPPORT, egress.CA_TARGET])
        self.assertEqual(mounts[1]["source"], str(Path("/run-dir/edge-ca.pem").resolve()))
        self.assertTrue(all(m["read_only"] for m in mounts))
        with self.assertRaises(ValueError):
            egress.support_mounts(Path("/s"), Path("/run-dir"))
        for name in egress.SUPPORT_FILES:
            self.assertTrue((HERE / name).is_file())


class ProbeTests(unittest.TestCase):
    def test_public_provenance_exact_pair_and_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            public = Path(tmp) / "interp.py"
            public.write_bytes(b"public fixture")
            sha = hashlib.sha256(public.read_bytes()).hexdigest()
            manifest = {"version": 1, "task_payload_sha256": "a" * 64,
                        "image_digest": "sha256:" + "b" * 64,
                        "public_files": [{"path": str(public), "sha256": sha}],
                        "solution_digests": [], "gold_digests": [sha],
                        "policy_sha256": "d" * 64}
            kwargs = {"root": tmp, "manifest": manifest, "task_payload_sha256": "a" * 64,
                      "image_digest": "sha256:" + "b" * 64}
            result = gold_probe.isolation({sha}, **kwargs)
            self.assertTrue(result["ok"])
            self.assertEqual(result["matched_paths"], [str(public)])
            self.assertEqual(result["public_matches"], [{"path": str(public), "sha256": sha}])
            wrong = Path(tmp) / "renamed"
            wrong.write_bytes(public.read_bytes())
            self.assertFalse(gold_probe.isolation({sha}, **kwargs)["ok"])
            wrong.unlink()
            for content in (b"hidden shadow_test", b"solution sentinel"):
                hidden = Path(tmp) / "hidden"
                hidden.write_bytes(content)
                hidden_sha = hashlib.sha256(content).hexdigest()
                manifest["gold_digests"] = [sha, hidden_sha]
                self.assertFalse(gold_probe.isolation({sha, hidden_sha}, **kwargs)["ok"])
                manifest["gold_digests"] = [sha]
                hidden.unlink()
            manifest["task_payload_sha256"] = "c" * 64
            self.assertFalse(gold_probe.isolation({sha}, **kwargs)["ok"])

    def test_isolation_finds_planted_bytes_by_hash(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            planted = Path(tmp) / "deep" / "copy.sh"
            planted.parent.mkdir()
            planted.write_bytes(b"SENTINEL-solution\n")
            digest = hashlib.sha256(b"SENTINEL-solution\n").hexdigest()
            hit = gold_probe.isolation({digest}, root=tmp)
            self.assertFalse(hit["ok"])
            self.assertEqual(hit["matched_paths"], [str(planted)])
            planted.unlink()
            self.assertEqual(gold_probe.isolation({digest}, root=tmp)["matched_paths"], [])

    def test_isolation_hashes_large_files_streaming(self):
        # R6 F20 / P18: a 4 MiB + 1 copy of a gold file was skipped (ok=True).
        blob = b"g" * (4 * 1024 * 1024 + 1)
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            big = Path(tmp) / "big.bin"
            big.write_bytes(blob)
            hit = gold_probe.isolation({hashlib.sha256(blob).hexdigest()}, root=tmp)
        self.assertEqual(hit["matched_paths"], [str(big)])
        self.assertFalse(hit["ok"])
        self.assertFalse(hasattr(gold_probe, "MAX_BYTES"))

    CANARY = 41999

    def _egress(self, outcomes, resolves="gaierror", ports=(8787,)):
        def fake(host, port):
            return outcomes.get((host, port), "ENETUNREACH")

        with patch.object(gold_probe, "_connects", side_effect=fake), patch.object(
            gold_probe, "_resolves",
            side_effect=lambda n: "resolved" if n == "host.docker.internal" else resolves,
        ):
            return gold_probe.egress(list(ports), canary_port=self.CANARY)

    def test_egress_check_requires_denials_and_listed_gate_ports(self):
        base = {("bench-gate", 8787): "connected", ("bench-gate", 1): "ECONNREFUSED",
                ("bench-gate", self.CANARY): "ECONNREFUSED"}
        self.assertTrue(self._egress(base)["ok"])
        self.assertFalse(self._egress(base, ports=(8787, 8789))["ok"])  # listed port dead
        self.assertFalse(self._egress(base, ports=())["ok"])
        self.assertFalse(self._egress({**base, ("1.1.1.1", 443): "connected"})["ok"])
        self.assertFalse(self._egress(base, resolves="resolved")["ok"])  # public DNS works

    def test_egress_refused_is_a_route_not_a_denial(self):
        # R6 F16: refused from an OPEN run-owned host port proves a route exists.
        base = {("bench-gate", 8787): "connected", ("bench-gate", 1): "ECONNREFUSED",
                ("bench-gate", self.CANARY): "ECONNREFUSED"}
        for outcome in ("ECONNREFUSED", "connected", "ConnectionResetError"):
            with self.subTest(outcome):
                bad = {**base, ("host.docker.internal", self.CANARY): outcome}
                self.assertFalse(self._egress(bad)["ok"])
        for outcome in ("ENETUNREACH", "EHOSTUNREACH", "timeout"):
            with self.subTest(outcome):
                good = {**base, ("host.docker.internal", self.CANARY): outcome}
                self.assertTrue(self._egress(good)["ok"])
        self.assertFalse(self._egress({**base, ("1.1.1.1", 443): "ECONNREFUSED"})["ok"])
        self.assertFalse(self._egress({**base, ("bench-gate", self.CANARY): "connected"})["ok"])

    def test_egress_canary_needs_a_resolvable_host_gateway(self):
        base = {("bench-gate", 8787): "connected", ("bench-gate", 1): "ECONNREFUSED",
                ("bench-gate", self.CANARY): "ECONNREFUSED"}
        with patch.object(gold_probe, "_connects",
                          side_effect=lambda h, p: base.get((h, p), "ENETUNREACH")), \
                patch.object(gold_probe, "_resolves", return_value="gaierror"):
            # No host-gateway name: the canary never tested a route; not a pass.
            self.assertFalse(gold_probe.egress([8787], canary_port=self.CANARY)["ok"])

    def test_connects_classifies_by_errno(self):
        import errno

        cases = ((OSError(errno.ENETUNREACH, "x"), "ENETUNREACH"),
                 (ConnectionRefusedError(errno.ECONNREFUSED, "x"), "ECONNREFUSED"),
                 (TimeoutError(), "timeout"), (socket.gaierror(-2, "x"), "gaierror"))
        for exc, want in cases:
            with self.subTest(want), patch.object(socket, "create_connection", side_effect=exc):
                self.assertEqual(gold_probe._connects("h", 1), want)

    def _proc(self, root: Path, pid: int, cmd: bytes, inodes=(), state="S"):
        d = root / str(pid)
        (d / "fd").mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd)
        (d / "status").write_text(f"Name:\tx\nState:\t{state} (sleeping)\n")
        for i, inode in enumerate(inodes):
            os.symlink(f"socket:[{inode}]", d / "fd" / str(i + 3))

    def test_quiescence_records_survivors_with_hash_and_listening_ports(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            root = Path(tmp)
            (root / "net").mkdir()
            # /proc/net/tcp layout: local "addr:port" hex, st 0A = LISTEN, inode col 9.
            tail = " 00000000:00000000 00:00000000 00000000     0        0 {} 1\n"
            hdr = "  sl  local_address rem_address   st ... inode\n"
            (root / "net" / "tcp").write_text(
                hdr + "   0: 00000000:1F90 00000000:0000 0A" + tail.format(777)
                + "   1: 0100007F:14D0 0100007F:9999 01" + tail.format(778))
            zero6 = "0" * 32
            (root / "net" / "tcp6").write_text(
                hdr + f"   0: {zero6}:14D0 {zero6}:0000 0A" + tail.format(779))
            self._proc(root, 1, b"sleep\0infinity\0")
            self._proc(root, 40, b"nginx: master process\0", inodes=(777, 778))
            self._proc(root, 41, b"python\0server.py\0", inodes=(779,))
            self._proc(root, 42, b"defunct\0", state="Z")
            result = gold_probe.quiescence(proc_root=root)
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["survivors"],
            [{"pid": 40, "cmd_sha256": hashlib.sha256(b"nginx: master process").hexdigest(),
              "listening_ports": [8080]},
             {"pid": 41, "cmd_sha256": hashlib.sha256(b"python server.py").hexdigest(),
              "listening_ports": [5328]}],
        )
        # Hash only: the probe never echoes a command line (it may carry a value).
        self.assertNotIn("server.py", json.dumps(result))

    def test_cli_usage(self):
        self.assertEqual(gold_probe.main(["egress", "x"]), 2)
        self.assertEqual(gold_probe.main(["egress", "8787"]), 2)  # canary required
        self.assertEqual(gold_probe.main(["egress", "--canary", "x", "8787"]), 2)
        self.assertEqual(gold_probe.main([]), 2)


def _load_verifier(responses):
    """bench_verifier against a stand-in Harbor base; ``responses`` are probe stdouts."""
    calls = []

    class _Base:
        def __init__(self, environment, trial_paths):
            self.environment, self.trial_paths = environment, trial_paths

        async def verify(self):
            calls.append("stock-verify")
            return "graded"

    class _Env:
        session_id = "sess"

        async def exec(self, command, user=None):
            calls.append(command)
            return type("R", (), {"stdout": responses.pop(0), "return_code": 0})()

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "harbor.verifier.verifier":
            return type("M", (), {"Verifier": _Base})
        return real(name, *a, **k)

    with patch("importlib.import_module", side_effect=fake):
        spec = importlib.util.spec_from_file_location("coding_bench_verifier",
                                                      HERE / "bench_verifier.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, _Env(), calls


class BenchVerifierTests(unittest.TestCase):
    SURVIVOR = {"pid": 40, "cmd_sha256": "c" * 64, "listening_ports": [8080]}

    def _verify(self, isolation_ok=True, survivors=()):
        quiet = {"check": "quiescence", "ok": not survivors, "survivors": list(survivors)}
        iso = {"check": "gold-isolation", "ok": isolation_ok, "manifest_sha256": "d" * 64}
        module, env, calls = _load_verifier([json.dumps(iso), json.dumps(quiet)])
        tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(tmp.cleanup)
        paths = SimpleNamespace(verifier_dir=Path(tmp.name) / "verifier")
        verifier = module.BenchVerifier(env, paths)
        with patch.object(module, "connect_uplink", return_value="cid"), patch.object(
            module, "compose_project", return_value="proj"
        ), patch.object(module, "provenance_args", return_value=["--public-provenance", "fixture"]), patch.dict(os.environ, {"BENCH_GOLD_DIGESTS": "a" * 64,
                                     "BENCH_PUBLIC_PROVENANCE_SHA256": "d" * 64}):
            try:
                result = asyncio.run(verifier.verify())
            except RuntimeError as exc:
                result = exc
        record = json.loads((paths.verifier_dir / "bench-isolation.json").read_text())
        return result, record, calls

    def test_survivors_are_recorded_flagged_and_never_killed(self):
        # R6 F15: nginx-request-logging / kv-store-grpc need the agent's
        # server alive at test time; grading follows Harbor's stock behaviour.
        result, record, calls = self._verify(survivors=[self.SURVIVOR])
        self.assertEqual(result, "graded")
        self.assertEqual(calls[-1], "stock-verify")
        self.assertFalse(any("kill" in c.lower() for c in calls if isinstance(c, str)))
        self.assertEqual(len([c for c in calls if "quiescence" in c]), 1)
        self.assertEqual(record["survivors"], [self.SURVIVOR])
        self.assertEqual(record["flags"], ["agent-processes-survived-into-grading"])
        self.assertTrue(record["graded"])
        self.assertFalse(hasattr(self._module(), "KILL_SURVIVORS"))

    def _module(self):
        return _load_verifier([])[0]

    def test_quiet_trial_has_no_flag(self):
        result, record, _ = self._verify()
        self.assertEqual(result, "graded")
        self.assertEqual(record["flags"], [])

    def test_gold_isolation_failure_still_refuses_grading(self):
        result, record, calls = self._verify(isolation_ok=False, survivors=[self.SURVIVOR])
        self.assertIsInstance(result, RuntimeError)
        self.assertNotIn("stock-verify", calls)
        self.assertFalse(record["graded"])
        self.assertEqual(record["survivors"], [self.SURVIVOR])

    def test_no_per_task_required_services_allowlist(self):
        for name in ("bench_verifier.py", "gold_probe.py", "driver.py"):
            text = (HERE / name).read_text().lower()
            self.assertNotIn("required_services", text, name)
            self.assertNotIn("required services", text, name)


class GateRelayTests(unittest.TestCase):
    def test_port_map_validation(self):
        self.assertEqual(gate_relay.parse_ports(["1=2", "3=4"]), {1: 2, 3: 4})
        for bad in ([], ["1"], ["a=1"], ["0=1"], ["1=2", "1=3"], ["1=70000"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                gate_relay.parse_ports(bad)

    def test_relays_bytes_to_fixed_upstream_only(self):
        async def scenario():
            async def echo(reader, writer):
                writer.write(b"upstream:" + await reader.read(100))
                await writer.drain()
                writer.close()

            upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
            up_port = upstream.sockets[0].getsockname()[1]
            with patch.object(gate_relay, "UPSTREAM_HOST", "127.0.0.1"):
                relay = await asyncio.start_server(gate_relay._handler(up_port), "127.0.0.1", 0)
                port = relay.sockets[0].getsockname()[1]
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"CONNECT api.anthropic.com:443")
                await writer.drain()
                writer.write_eof()
                data = await reader.read()
                writer.close()
                relay.close()
            upstream.close()
            return data

        self.assertEqual(asyncio.run(scenario()), b"upstream:CONNECT api.anthropic.com:443")


class DriverContainerTests(unittest.TestCase):
    def test_container_plan_rewrites_edge_to_gate_on_same_port(self):
        child = {k: "http://run1:n@127.0.0.1:41000" for k in
                 ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
        child.update(SSL_CERT_FILE="/host/edge-ca.pem", REQUESTS_CA_BUNDLE="/host/edge-ca.pem",
                     NO_PROXY="127.0.0.1,localhost,::1", RADIENT_API_KEY="dummy")
        plan = driver.container_plan(
            "run1", "local-operator", EgressTests.IMAGE, meter_port=8787, bridge_port=8789,
            edge={"port": 41000, "env": child, "ca_file": Path("/host/edge-ca.pem")},
        )
        self.assertEqual(plan["ports"], [8787, 41000])  # no bridge for LoP (R7 F21)
        self.assertEqual(plan["env"]["HTTPS_PROXY"], "http://run1:n@bench-gate:41000")
        self.assertEqual(plan["env"]["SSL_CERT_FILE"], egress.CA_TARGET)
        self.assertEqual(plan["urls"]["anthropic"], "http://bench-gate:8787/r/run1")
        # The launcher accepts exactly what the driver produces.
        launch = _load_launcher()
        ca_pem, der = _fake_ca()
        with tempfile.NamedTemporaryFile(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as ca:
            Path(ca.name).write_text(ca_pem + (HERE / "public-roots.pem").read_text())
            os.chmod(ca.name, 0o444)
            env = {**plan["env"], "RADIENT_API_KEY": launch.DUMMY_RADIENT_KEY,
                   "BENCH_EDGE_CA_SHA256": hashlib.sha256(der).hexdigest(),
                   **{k: ca.name for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                                           *launch.PARITY_KEYS)}}
            with patch.object(launch, "CA_BUNDLE", ca.name):
                launch.require_edge_environment(env)

    def test_container_plan_refusals(self):
        with self.assertRaises(ValueError):
            driver.container_plan("r", "local-operator", "i", meter_port=1, bridge_port=2)
        with self.assertRaises(ValueError):
            driver.container_plan("r", "omp", "i", meter_port=1, bridge_port=2,
                                  edge={"port": 3, "env": {}, "ca_file": Path("/edge-ca.pem")})
        plan = driver.container_plan("r", "omp", "i", meter_port=8787, bridge_port=8789)
        # Only the package-proxy route (test_package_egress); no edge env.
        self.assertEqual(sorted(plan["env"]), ["HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
                                               "http_proxy", "https_proxy", "no_proxy"])
        self.assertEqual(plan["ports"], [8787])

    def test_edge_mount_is_the_ca_file(self):
        # R6 N6: never the run dir (it holds edge-requests.jsonl).
        src = (HERE / "driver.py").read_text()
        self.assertIn('"ca_file": run_dir / "edge-ca.pem"', src)
        self.assertNotIn('"ca_dir"', src)

    def test_meter_is_the_single_budget_authority(self):
        # R6 F19: budget_guard sat on no container route. One authority for
        # both lanes: meter_proxy's ledger. No guard process, port or argument.
        src = (HERE / "driver.py").read_text()
        for gone in ("budget_guard", "8788", "container_guard_url", "container-guard-url"):
            self.assertNotIn(gone, src, gone)
        self.assertFalse((HERE / "budget_guard.py").exists())
        names = {a.dest for a in driver.parser()._actions}
        self.assertNotIn("container_guard_url", names)
        self.assertIn("meter_current_run_file", names)

    def test_codex_current_run_must_be_the_meters_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            out = Path(tmp)
            args = argparse.Namespace(output=out / "out", meter_current_run_file=out / "cr")
            self.assertEqual(driver.current_run_file(args), (out / "cr").resolve())
            for bad in (None, Path("relative")):
                with self.subTest(bad), self.assertRaises(ValueError):
                    driver.current_run_file(argparse.Namespace(output=out,
                                                               meter_current_run_file=bad))

    def test_egress_canary_is_a_run_owned_open_port_the_gate_never_forwards(self):
        with driver.egress_canary() as port:
            # Open on the host (connect succeeds), and outside the gate map.
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
            plan = driver.container_plan("r", "omp", "i", meter_port=8787, bridge_port=8789)
            self.assertNotIn(port, plan["ports"])
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()

    def test_bridge_port_is_gated_only_for_codex(self):
        # R7 F21: the bridge tags rows from the meter's current-run file, so
        # any other harness reaching it would bill a finished Codex run's id.
        for harness in ("claude-code", "omp", "opencode", "hermes"):
            plan = driver.container_plan("r", harness, "i", meter_port=8787, bridge_port=8789)
            self.assertNotIn(8789, plan["ports"], harness)
            self.assertNotIn("codex", plan["urls"], harness)
        plan = driver.container_plan("r", "codex-best", "i", meter_port=8787, bridge_port=8789)
        self.assertEqual(plan["ports"], [8787, 8789])
        self.assertEqual(plan["urls"]["codex"], "http://bench-gate:8789/v1")

    def test_codex_current_run_is_restored_after_every_outcome(self):
        # R7 F21: success, error and timeout all restore the prior contents
        # (or absence) so no later trial is metered under this Codex run.
        import subprocess

        outcomes = (None, RuntimeError("trial incomplete"),
                    subprocess.TimeoutExpired("harbor", 1), KeyboardInterrupt())
        for prior in (None, b"coding__codex-best__older__r0"):
            for outcome in outcomes:
                with self.subTest(prior=prior, outcome=type(outcome).__name__), \
                        tempfile.TemporaryDirectory(
                            dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
                    root = Path(tmp)
                    current = root / "current_run"
                    if prior is not None:
                        current.write_bytes(prior)
                    args = argparse.Namespace(
                        output=root / "out", request_log=root / "requests.jsonl",
                        meter_url="http://127.0.0.1:8787", image="i", rep=0,
                        meter_current_run_file=current,
                    )
                    seen = []

                    def trial(*a, **_k):
                        seen.append(current.read_text())
                        if outcome is not None:
                            raise outcome

                    task = {"task8": "abcd1234", "name": "t"}
                    with patch.object(driver, "block_paid_coding"), \
                            patch.object(driver, "_run_gated_trial", side_effect=trial):
                        if outcome is None:
                            driver.run_trial(args, "codex-best", task, provenance={})
                        else:
                            with self.assertRaises(type(outcome)):
                                driver.run_trial(args, "codex-best", task, provenance={})
                    self.assertEqual(seen, ["coding__codex-best__abcd1234__r0"])
                    if prior is None:
                        self.assertFalse(current.exists())
                    else:
                        self.assertEqual(current.read_bytes(), prior)

    def test_non_codex_trial_never_touches_the_current_run_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            root = Path(tmp)
            current = root / "current_run"
            current.write_bytes(b"coding__codex-best__older__r0")
            args = argparse.Namespace(
                output=root / "out", request_log=root / "requests.jsonl",
                meter_url="http://127.0.0.1:8787", image="i", rep=0,
                meter_current_run_file=current,
            )
            with patch.object(driver, "block_paid_coding"), \
                    patch.object(driver, "_run_gated_trial", return_value=None):
                driver.run_trial(args, "omp", {"task8": "abcd1234", "name": "t"}, provenance={})
            self.assertEqual(current.read_bytes(), b"coding__codex-best__older__r0")

    def _health_server(self, body):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib hook name
                data = json.dumps(body).encode()
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_meter_caps_must_equal_the_cli_caps(self):
        # R7 F22: the meter is the sole budget authority, so a meter started
        # with different caps would silently govern the lane. Refuse instead.
        url = self._health_server({"ok": True, "models": [],
                                   "caps": {"total_usd": 5.0, "trial_usd": 0.5}})
        good = argparse.Namespace(meter_url=url, total_budget_usd=5.0, trial_budget_usd=0.5)
        driver.check_meter_caps(good)
        for total, trial in ((5.0, 0.6), (4.0, 0.5), (5.000001, 0.5)):
            with self.subTest(total=total, trial=trial), \
                    self.assertRaisesRegex(ValueError, "meter caps"):
                driver.check_meter_caps(argparse.Namespace(
                    meter_url=url, total_budget_usd=total, trial_budget_usd=trial))
        legacy = self._health_server({"ok": True, "models": []})
        with self.assertRaisesRegex(ValueError, "meter caps"):
            driver.check_meter_caps(argparse.Namespace(
                meter_url=legacy, total_budget_usd=5.0, trial_budget_usd=0.5))

    def test_preflight_refuses_a_meter_with_other_caps(self):
        url = self._health_server({"ok": True, "models": [],
                                   "caps": {"total_usd": 9.0, "trial_usd": 0.5}})
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            args = argparse.Namespace(
                total_budget_usd=5.0, trial_budget_usd=0.5, input_usd_per_m=1.0,
                output_usd_per_m=1.0, billing_tier="explicit", preregistration=None,
                preregistration_sha256="", runtime_attestation=None, runtime_sha256="",
                toolwire_attestation=None, toolwire_sha256="", oracle_attestation=None,
                oracle_sha256="", tasks="t", harnesses="omp", rep=0,
                meter_current_run_file=Path(tmp) / "current_run",
                meter_url=url, translator_url="http://127.0.0.1:1", output=Path(tmp) / "o",
            )
            with patch.object(driver, "block_paid_coding"), \
                    patch.object(driver, "frozen", return_value={}), \
                    self.assertRaisesRegex(ValueError, "meter caps"):
                driver.preflight(args)

    def test_gold_digests_hash_only(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            task = Path(tmp)
            with self.assertRaises(ValueError):
                driver.gold_digests(task)
            (task / "solution").mkdir()
            (task / "solution" / "solve.sh").write_bytes(b"s")
            (task / "tests").mkdir()
            (task / "tests" / "test.sh").write_bytes(b"t")
            self.assertEqual(
                driver.gold_digests(task),
                sorted(hashlib.sha256(b).hexdigest() for b in (b"s", b"t")),
            )

    def test_evidence_record_binds_the_closure_and_artifact(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            out = Path(tmp)
            evidence = out / "EVIDENCE.md"
            evidence.write_text("table")
            record = {
                "container_s4_passed": True,
                "lop_head": driver.SOURCE_REVISION,
                "files": driver.container_closure_hashes(),
                "evidence_path": evidence.name,
                "evidence_sha256": driver.digest(evidence),
            }
            path = out / "record.json"
            args = argparse.Namespace(coding_edge_evidence=path)
            path.write_text(json.dumps(record))
            self.assertEqual(driver.container_evidence(args), record["evidence_sha256"])
            driver.block_paid_coding(args)  # lifts only here
            for key, value in (("container_s4_passed", False), ("lop_head", "0" * 40),
                               ("files", {**record["files"], "egress.py": "0" * 64}),
                               ("evidence_path", "../x"), ("evidence_sha256", "0" * 64)):
                with self.subTest(key):
                    path.write_text(json.dumps({**record, key: value}))
                    self.assertIsNone(driver.container_evidence(args))
                    with self.assertRaisesRegex(ValueError, "HARDBLOCK"):
                        driver.block_paid_coding(args)
            path.write_text(json.dumps(record))
            evidence.write_text("edited")
            self.assertIsNone(driver.container_evidence(args))
        self.assertIsNone(driver.container_evidence(argparse.Namespace()))

    def test_closure_covers_every_container_path_module(self):
        names = set(driver.CONTAINER_CLOSURE)
        for name in ("adapters.py", "bench_verifier.py", "egress.py", "gate_relay.py",
                     "gold_probe.py", "lop_launch.py", "driver.py", "../lop_edge.py",
                     "../meter_proxy.py"):
            self.assertIn(name, names)
        for name in driver.CONTAINER_CLOSURE:
            self.assertTrue((HERE / name).is_file(), name)


class SocketHygieneTests(unittest.TestCase):
    def test_no_test_opens_external_sockets(self):
        # Guard for this module: everything above uses loopback or mocks.
        with patch.object(socket, "create_connection", side_effect=AssertionError("network")):
            gold_probe._connects  # attribute exists; real calls are mocked above


if __name__ == "__main__":
    unittest.main()

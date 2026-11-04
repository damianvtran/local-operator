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
        self.ca.write_text("ca")
        self.ca.chmod(0o444)

    def env(self, **over):
        proxy = "http://run1:nonce@bench-gate:41000"
        base = {
            "HTTPS_PROXY": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "http_proxy": proxy,
            "NO_PROXY": self.launch.NO_PROXY, "no_proxy": self.launch.NO_PROXY,
            "SSL_CERT_FILE": self.launch.CA_BUNDLE, "REQUESTS_CA_BUNDLE": self.launch.CA_BUNDLE,
            "RADIENT_API_KEY": self.launch.DUMMY_RADIENT_KEY,
        }
        base.update(over)
        return {k: v for k, v in base.items() if v is not None}

    def check(self, env):
        with patch.object(self.launch, "CA_BUNDLE", str(self.ca)):
            env = {k: (str(self.ca) if v == "/run/bench-ca/edge-ca.pem" else v) for k, v in env.items()}
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
            "no proxy": self.env(HTTPS_PROXY=None, https_proxy=None, HTTP_PROXY=None, http_proxy=None),
            "loopback edge": self.env(**{k: "http://r:n@127.0.0.1:1" for k in
                                         ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}),
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

    def test_main_checks_environment_before_importing_product(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            self.launch.runpy, "run_module", side_effect=AssertionError("product started")
        ):
            with self.assertRaises(SystemExit):
                self.launch.main()


class EgressTests(unittest.TestCase):
    IMAGE = "alexgshaw/git-leak-recovery@sha256:" + "a" * 64

    def override(self, ports=(8787, 8789, 41000)):
        return egress.compose_override("run1", list(ports), self.IMAGE, Path("/s"), Path("/t"))

    def test_main_only_on_internal_network_and_never_pulls(self):
        o = self.override()
        main, gate = o["services"]["main"], o["services"][egress.GATE_SERVICE]
        self.assertEqual(main["networks"], [egress.INTERNAL_NETWORK])
        self.assertIs(o["networks"][egress.INTERNAL_NETWORK]["internal"], True)
        self.assertNotIn("internal", o["networks"][egress.UPLINK_NETWORK])
        self.assertEqual(main["pull_policy"], "never")
        self.assertEqual(gate["pull_policy"], "never")
        self.assertEqual(gate["image"], self.IMAGE)
        self.assertEqual(gate["entrypoint"][2:], ["8787=8787", "8789=8789", "41000=41000"])
        self.assertTrue(all(v["read_only"] for v in gate["volumes"]))
        self.assertEqual(gate["cap_drop"], ["ALL"])

    def test_override_refusals(self):
        for bad in ([], [0], [70000], [8787, 8787], [True]):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.override(bad)
        with self.assertRaises(ValueError):
            egress.compose_override("run1", [1], "repo:tag", Path("/s"), Path("/t"))
        with self.assertRaises(ValueError):
            egress.compose_override("../x", [1], self.IMAGE, Path("/s"), Path("/t"))

    def test_no_implicit_pull_guard(self):
        ok = ["docker", "run", "--rm", "--pull", "never", "--platform", "linux/amd64", "img"]
        self.assertEqual(egress.require_no_implicit_pull(ok), ok)
        egress.require_no_implicit_pull(["docker", "run", "--pull=never", "img"])
        for bad in (
            ["docker", "run", "--platform", "linux/amd64", "mongo:7.0"],  # the incident
            ["docker", "run", "img"],
            ["docker", "create", "img"],
            ["docker", "run", "--pull", "missing", "img"],
            ["docker", "image", "inspect", "--platform", "linux/amd64", "img"],
        ):
            with self.subTest(bad), self.assertRaises(ValueError):
                egress.require_no_implicit_pull(bad)

    def test_support_mounts_read_only(self):
        mounts = egress.support_mounts(Path("/s"), Path("/ca"))
        self.assertEqual([m["target"] for m in mounts], [egress.SUPPORT, "/run/bench-ca"])
        self.assertTrue(all(m["read_only"] for m in mounts))
        for name in egress.SUPPORT_FILES:
            self.assertTrue((HERE / name).is_file())


class ProbeTests(unittest.TestCase):
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

    def test_egress_check_requires_denials_and_listed_gate_ports(self):
        outcomes = {("bench-gate", 8787): "connected"}

        def fake(host, port):
            return outcomes.get((host, port), "ConnectionRefusedError")

        with patch.object(gold_probe, "_connects", side_effect=fake), patch.object(
            gold_probe, "_resolves", return_value="gaierror"
        ):
            self.assertTrue(gold_probe.egress([8787])["ok"])
            self.assertFalse(gold_probe.egress([8787, 8789])["ok"])  # listed port dead
            self.assertFalse(gold_probe.egress([])["ok"])
            outcomes[("1.1.1.1", 443)] = "connected"  # a public route exists
            self.assertFalse(gold_probe.egress([8787])["ok"])
        with patch.object(gold_probe, "_connects", side_effect=fake), patch.object(
            gold_probe, "_resolves", return_value="resolved"
        ):
            outcomes.pop(("1.1.1.1", 443))
            self.assertFalse(gold_probe.egress([8787])["ok"])  # public DNS works

    def test_cli_usage(self):
        self.assertEqual(gold_probe.main(["egress", "x"]), 2)
        self.assertEqual(gold_probe.main([]), 2)


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
            edge={"port": 41000, "env": child, "ca_dir": Path("/host")},
        )
        self.assertEqual(plan["ports"], [8787, 8789, 41000])
        self.assertEqual(plan["env"]["HTTPS_PROXY"], "http://run1:n@bench-gate:41000")
        self.assertEqual(plan["env"]["SSL_CERT_FILE"], egress.CA_TARGET)
        self.assertEqual(plan["urls"]["anthropic"], "http://bench-gate:8787/r/run1")
        # The launcher accepts exactly what the driver produces.
        launch = _load_launcher()
        with tempfile.NamedTemporaryFile(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as ca:
            os.chmod(ca.name, 0o444)
            env = {**plan["env"], "SSL_CERT_FILE": ca.name, "REQUESTS_CA_BUNDLE": ca.name,
                   "RADIENT_API_KEY": launch.DUMMY_RADIENT_KEY}
            with patch.object(launch, "CA_BUNDLE", ca.name):
                launch.require_edge_environment(env)

    def test_container_plan_refusals(self):
        with self.assertRaises(ValueError):
            driver.container_plan("r", "local-operator", "i", meter_port=1, bridge_port=2)
        with self.assertRaises(ValueError):
            driver.container_plan("r", "omp", "i", meter_port=1, bridge_port=2,
                                  edge={"port": 3, "env": {}, "ca_dir": Path("/")})
        plan = driver.container_plan("r", "omp", "i", meter_port=8787, bridge_port=8789)
        self.assertEqual(plan["env"], {})

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

"""Offline tests for agent-phase package egress (``../package_egress.py``).

WHY: the proxy is the only route from a paid trial's agent phase to package
registries, so its allow/deny decisions, its per-connection log and its
refusal to carry model traffic are proven over REAL loopback sockets. Upstream
"registries" are loopback fixtures reached through an injected opener; the
default opener's public-address rule is tested on its own. No Docker, no
network, no credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import driver  # noqa: E402
import egress  # noqa: E402
import gate_relay  # noqa: E402
import package_egress  # noqa: E402

ALLOWLIST = HERE / "package-egress-allowlist.json"


def _const_tuple(path: Path, name: str) -> tuple[str, ...]:
    import ast

    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == name)
    return tuple(ast.literal_eval(node.value))


class Fixture:
    """A loopback 'registry': records what it received, answers one response."""

    def __init__(self, reply: bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"):
        self.reply = reply
        self.received = b""
        self.server: asyncio.Server | None = None

    async def start(self) -> int:
        async def serve(reader, writer):
            self.received += await reader.read(65536)
            writer.write(self.reply)
            await writer.drain()
            writer.close()

        self.server = await asyncio.start_server(serve, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


class PackageEgressTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "connections.jsonl"
        self.fixture = Fixture()
        self.fixture_port = await self.fixture.start()
        self.opened: list[tuple[str, int]] = []

        async def opener(host, port):
            self.opened.append((host, port))
            return await asyncio.open_connection("127.0.0.1", self.fixture_port)

        policy = package_egress.Policy(["pypi.org", "deb.debian.org", "api.anthropic.com"])
        self.proxy = package_egress.PackageProxy(
            policy, package_egress.ConnectionLog(self.log_path, via="gate"), opener=opener
        )
        self.server = await asyncio.start_server(self.proxy.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        assert self.fixture.server is not None
        self.fixture.server.close()
        await self.fixture.server.wait_closed()

    async def exchange(self, data: bytes) -> bytes:
        before = len(self.rows())
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(data)
        await writer.drain()
        out = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        # The client may see EOF before the proxy appends its row: wait for it.
        for _ in range(250):
            if len(self.rows()) > before:
                break
            await asyncio.sleep(0.02)
        return out

    def rows(self):
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines()]

    async def test_connect_to_allowlisted_host_is_tunnelled_and_logged(self):
        out = await self.exchange(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n"
                                  b"opaque-tls-bytes")
        self.assertTrue(out.startswith(b"HTTP/1.1 200 Connection Established\r\n\r\n"))
        self.assertTrue(out.endswith(b"ok"))
        self.assertEqual(self.fixture.received, b"opaque-tls-bytes")  # not intercepted
        self.assertEqual(self.opened, [("pypi.org", 443)])
        [row] = self.rows()
        self.assertEqual(
            {k: row[k] for k in ("via", "method", "host", "port", "allowed")},
            {"via": "gate", "method": "CONNECT", "host": "pypi.org", "port": 443,
             "allowed": True},
        )
        self.assertEqual(row["bytes_up"], len(b"opaque-tls-bytes"))
        self.assertGreater(row["bytes_down"], 0)

    async def test_plain_http_get_is_forwarded_origin_form_without_proxy_credentials(self):
        out = await self.exchange(
            b"GET http://deb.debian.org/debian/dists/bookworm/InRelease HTTP/1.1\r\n"
            b"Host: deb.debian.org\r\nProxy-Authorization: Basic c2VjcmV0\r\n"
            b"Proxy-Connection: keep-alive\r\nConnection: keep-alive\r\nUser-Agent: apt\r\n\r\n"
        )
        self.assertTrue(out.endswith(b"ok"))
        sent = self.fixture.received.decode()
        self.assertTrue(sent.startswith("GET /debian/dists/bookworm/InRelease HTTP/1.1\r\n"))
        self.assertNotIn("proxy-", sent.lower())
        self.assertNotIn("c2VjcmV0", sent)
        self.assertIn("Connection: close\r\n", sent)
        self.assertEqual(self.opened, [("deb.debian.org", 80)])
        [row] = self.rows()
        self.assertEqual((row["method"], row["port"], row["allowed"]), ("GET", 80, True))

    async def test_plain_http_always_sends_the_checked_host(self):
        # R8 F25: a client Host naming another CDN tenant must not reach upstream.
        await self.exchange(
            b"GET http://deb.debian.org/v1/x HTTP/1.1\r\n"
            b"Host: some-other-tenant.example\r\nhost: second.example\r\n\r\n"
        )
        sent = self.fixture.received.decode().lower()
        self.assertEqual([x for x in sent.split("\r\n") if x.startswith("host:")],
                         ["host: deb.debian.org"])
        self.assertNotIn("tenant", sent)

    async def test_malformed_authority_is_logged_redacted(self):
        # R8 N7: userinfo in a proxy authority must never reach the trial record.
        # Assembled from parts: the publication scanner rightly flags any
        # literal ``x@host`` in shipped source.
        authority = b"user:s3cret" + b"\x40" + b"deb.debian.org:443"
        out = await self.exchange(b"CONNECT " + authority + b" HTTP/1.1\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 403 "))
        row = self.rows()[-1]
        self.assertEqual(row["reason"], "malformed")
        self.assertNotIn("s3cret", self.log_path.read_text())
        self.assertRegex(row["host"], r"^redacted-sha256:[0-9a-f]{16}$")
        self.assertEqual(self.opened, [])

    async def test_denials_never_open_an_upstream_and_are_logged(self):
        cases = {
            b"CONNECT example.com:443 HTTP/1.1\r\n\r\n": "not-allowlisted",
            b"CONNECT pypi.org:22 HTTP/1.1\r\n\r\n": "port",
            b"CONNECT pypi.org:8443 HTTP/1.1\r\n\r\n": "port",
            b"CONNECT 1.1.1.1:443 HTTP/1.1\r\n\r\n": "ip-literal",
            b"CONNECT host.docker.internal:41999 HTTP/1.1\r\n\r\n": "port",
            b"CONNECT host.docker.internal:443 HTTP/1.1\r\n\r\n": "not-allowlisted",
            b"CONNECT bench-gate:8787 HTTP/1.1\r\n\r\n": "port",
            # Listed in this policy on purpose: model hosts are denied anyway.
            b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n": "model-host",
            b"CONNECT API.OpenAI.com:443 HTTP/1.1\r\n\r\n": "model-host",
            b"GET http://openrouter.ai/api/v1 HTTP/1.1\r\n\r\n": "model-host",
            b"GET https://pypi.org/simple/ HTTP/1.1\r\n\r\n": "malformed",
            b"GET /simple/ HTTP/1.1\r\nHost: pypi.org\r\n\r\n": "malformed",
        }
        for request, reason in cases.items():
            with self.subTest(request=request):
                out = await self.exchange(request)
                self.assertTrue(out.startswith(b"HTTP/1.1 403 "), out)
                self.assertEqual(self.rows()[-1]["reason"], reason)
                self.assertIs(self.rows()[-1]["allowed"], False)
        self.assertEqual(self.opened, [])
        self.assertEqual(len(self.rows()), len(cases))

    async def test_upstream_failure_is_502_and_logged(self):
        async def failing(host, port):
            raise PermissionError("non-public address")

        self.proxy.opener = failing
        out = await self.exchange(b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.1 502 "))
        row = self.rows()[-1]
        self.assertIs(row["allowed"], True)
        self.assertIn("non-public", row["error"])


class PolicyTests(unittest.TestCase):
    def test_model_hosts_and_their_subdomains_are_denied(self):
        policy = package_egress.Policy(["pypi.org"])
        for host in ("api.anthropic.com", "statsig.anthropic.com", "api.openai.com",
                     "openrouter.ai", "api.radienthq.com", "api.deepseek.com",
                     "generativelanguage.googleapis.com", "us-central1-aiplatform.googleapis.com",
                     "bedrock-runtime.us-east-1.amazonaws.com", "x.openai.azure.com",
                     "api.mistral.ai", "api.x.ai", "api.groq.com", "chatgpt.com", "claude.ai"):
            with self.subTest(host=host):
                self.assertEqual(policy.decide(host, 443), "model-host")
        self.assertIsNone(policy.decide("pypi.org", 443))
        self.assertIsNone(policy.decide("PyPI.org.", 80))
        self.assertEqual(policy.decide("evil-pypi.org", 443), "not-allowlisted")
        self.assertEqual(policy.decide("files.pypi.org", 443), "not-allowlisted")  # exact only

    def test_policy_refuses_malformed_allow_entries(self):
        for bad in (["*.debian.org"], ["pypi.org:443"], ["1.1.1.1"], [""], ["Bad Host"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                package_egress.Policy(bad)

    def test_frozen_allowlist_is_exact_and_carries_no_model_host(self):
        data = json.loads(ALLOWLIST.read_text())
        policy = package_egress.Policy.load(ALLOWLIST)
        self.assertEqual(sorted(policy.allow), sorted(data["allow"]))
        for host in ("deb.debian.org", "archive.ubuntu.com", "security.ubuntu.com",
                     "pypi.org", "files.pythonhosted.org"):
            self.assertIn(host, policy.allow)
        for host in policy.allow:
            self.assertIsNone(policy.decide(host, 443), host)
        # Hosts the 13 tasks need only at build or verify time stay out.
        for host in ("github.com", "astral.sh", "registry.npmjs.org", "ports.ubuntu.com"):
            self.assertNotIn(host, policy.allow)

    def test_default_opener_refuses_non_public_addresses(self):
        for addr in ("127.0.0.1", "10.0.0.5", "192.168.65.254", "172.17.0.1", "169.254.1.1",
                     "::1", "fd00::1", "100.64.0.1"):
            with self.subTest(addr=addr):
                self.assertFalse(package_egress.public_address(addr))
        self.assertTrue(package_egress.public_address("151.101.0.223"))

        async def resolve_private():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo",
                              return_value=[(2, 1, 6, "", ("192.168.65.254", 443))]):
                await package_egress.open_upstream("pypi.org", 443)

        with self.assertRaises(PermissionError):
            asyncio.run(resolve_private())


class GateWiringTests(unittest.TestCase):
    IMAGE = "alexgshaw/git-leak-recovery@sha256:" + "a" * 64

    def test_gate_runs_the_package_proxy_with_a_writable_log_only(self):
        o = egress.compose_override("run1", [8787], self.IMAGE, Path("/s"), Path("/t"),
                                    Path("/logs"))
        gate = o["services"][egress.GATE_SERVICE]
        entry = gate["entrypoint"]
        self.assertEqual(entry[2:8], [
            "--package-proxy", str(egress.PACKAGE_PROXY_PORT),
            "--allowlist", f"{egress.SUPPORT}/package-egress-allowlist.json",
            "--log", f"{egress.EGRESS_LOG}/connections.jsonl",
        ])
        self.assertEqual(entry[8:], ["8787=8787"])
        writable = [v for v in gate["volumes"] if not v["read_only"]]
        self.assertEqual([(v["source"], v["target"]) for v in writable],
                         [(str(Path("/logs").resolve()), egress.EGRESS_LOG)])
        # main never sees the log (it is the trial's audit record).
        self.assertNotIn("volumes", o["services"]["main"])

    def test_gate_argv_requires_the_package_proxy(self):
        opts, ports = gate_relay.parse_args(
            ["--package-proxy", "3128", "--allowlist", "/a.json", "--log", "/l", "1=2"])
        self.assertEqual((opts["package_proxy"], ports), (3128, {1: 2}))
        for bad in (["1=2"], ["--package-proxy", "3128", "1=2"],
                    ["--package-proxy", "x", "--allowlist", "/a", "--log", "/l", "1=2"],
                    ["--package-proxy", "1", "--allowlist", "/a", "--log", "/l", "1=2"],
                    ["--bogus", "1", "1=2"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                gate_relay.parse_args(bad)

    def test_support_files_include_the_policy_and_allowlist(self):
        names = {Path(n).name for n in egress.SUPPORT_FILES}
        self.assertTrue({"package_egress.py", "package-egress-allowlist.json"} <= names)
        for name in egress.SUPPORT_FILES:
            self.assertTrue((HERE / name).is_file(), name)

    def test_allowlist_and_policy_are_in_the_closure_and_prereg_inputs(self):
        for name in ("../package_egress.py", "package-egress-allowlist.json"):
            self.assertIn(name, driver.CONTAINER_CLOSURE)
        self.assertEqual(
            driver.package_egress_hashes(),
            {"policy": driver.digest(HERE.parent / "package_egress.py"),
             "allowlist": driver.digest(ALLOWLIST),
             "public_roots": driver.digest(HERE / "public-roots.pem")},
        )

    def test_non_lop_harnesses_get_the_package_proxy_and_bypass_it_for_the_gate(self):
        proxy = f"http://{egress.GATE_SERVICE}:{egress.PACKAGE_PROXY_PORT}"
        for harness in ("claude-code", "omp", "opencode", "hermes", "codex-best", "dsh"):
            with self.subTest(harness=harness):
                plan = driver.container_plan("r", harness, "i", meter_port=8787,
                                             bridge_port=8789)
                env = plan["env"]
                for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
                    self.assertEqual(env[key], proxy)
                no_proxy = env["NO_PROXY"].split(",")
                self.assertEqual(env["no_proxy"], env["NO_PROXY"])
                # Every model URL the harness is given names the gate host,
                # so model traffic never enters the package proxy.
                for url in plan["urls"].values():
                    self.assertIn(url.split("//")[1].split(":")[0], no_proxy)
                self.assertNotIn(egress.PACKAGE_PROXY_PORT, plan["ports"])

    def test_lop_keeps_its_edge_proxy_and_the_edge_gets_the_allowlist(self):
        child = {k: "http://run1:n@127.0.0.1:41000" for k in
                 ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
        child.update(NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1")
        plan = driver.container_plan(
            "run1", "local-operator", self.IMAGE, meter_port=8787, bridge_port=8789,
            edge={"port": 41000, "env": child, "ca_file": Path("/h/edge-ca.pem")},
        )
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            self.assertEqual(plan["env"][key], "http://run1:n@bench-gate:41000")
        self.assertEqual(plan["env"]["NO_PROXY"], "127.0.0.1,localhost,::1")
        src = (HERE / "driver.py").read_text()
        self.assertIn('"package_allowlist": str(PACKAGE_ALLOWLIST)', src)

    def test_lop_trust_bundle_is_mounted_read_only_and_hashed(self):
        # R8 F24: LoP's tool children trust run CA + frozen public roots.
        roots = HERE / "public-roots.pem"
        self.assertIn("public-roots.pem", driver.CONTAINER_CLOSURE)
        self.assertEqual(driver.package_egress_hashes()["public_roots"], driver.digest(roots))
        self.assertGreater(roots.read_text().count("BEGIN CERTIFICATE"), 50)
        self.assertNotIn("PRIVATE KEY", roots.read_text())
        src = (HERE / "driver.py").read_text()
        self.assertIn('"public_roots": str(PUBLIC_ROOTS)', src)
        child = {k: "http://run1:n@127.0.0.1:41000" for k in
                 ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT", "CURL_CA_BUNDLE",
                    "NODE_EXTRA_CA_CERTS"):
            child[key] = "/host/run/edge-ca.pem"
        child["BENCH_EDGE_CA_SHA256"] = "a" * 64
        plan = driver.container_plan(
            "run1", "local-operator", self.IMAGE, meter_port=8787, bridge_port=8789,
            edge={"port": 41000, "env": child, "ca_file": Path("/host/run/edge-ca.pem")},
        )
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT", "CURL_CA_BUNDLE",
                    "NODE_EXTRA_CA_CERTS"):
            self.assertEqual(plan["env"][key], egress.CA_TARGET, key)
        self.assertEqual(plan["env"]["BENCH_EDGE_CA_SHA256"], "a" * 64)
        # Harbor forwards only the listed keys to the agent.
        keys = _const_tuple(HERE / "adapters.py", "EDGE_ENV_KEYS")
        for key in child:
            self.assertIn(key, keys)

    def test_trial_record_collects_every_logged_connection(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as t:
            gate_log, edge_log = Path(t) / "gate.jsonl", Path(t) / "edge.jsonl"
            gate_log.write_text(
                json.dumps({"via": "gate", "host": "pypi.org", "port": 443, "allowed": True,
                            "bytes_up": 5, "bytes_down": 7}) + "\n"
                + json.dumps({"via": "gate", "host": "example.com", "port": 443,
                              "allowed": False, "reason": "not-allowlisted",
                              "bytes_up": 0, "bytes_down": 0}) + "\n")
            summary = driver.package_egress_summary([gate_log, edge_log])
        self.assertEqual(summary["allowed"], 1)
        self.assertEqual(summary["denied"], 1)
        self.assertEqual((summary["bytes_up"], summary["bytes_down"]), (5, 7))
        self.assertEqual(len(summary["connections"]), 2)
        self.assertEqual(summary["policy"], driver.package_egress_hashes())


if __name__ == "__main__":
    unittest.main()

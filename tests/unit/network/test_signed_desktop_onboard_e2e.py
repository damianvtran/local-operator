"""Signed desktop card -> existing runner -> TWO process relays over real TCP.

Synthetic seams: SSH transport, already-installed software inventory, OS service
restart/root-anchor install and software (not Touch ID) signing. Actual CLI joins,
SAS comparison, stores, signatures, grants, readiness and HTTP are NOT faked.
The node pipe only transports OS-SSH command execution inside this isolated rig.
"""

from __future__ import annotations

import asyncio
import json
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import pytest_asyncio

from local_operator.network import approvals, onboard, store
from local_operator.network.types import (
    MemberRecord,
    NetworkRecord,
    capabilities_for_role,
)
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.server.test_desktop_mesh_onboard import live as live_http
from tests.unit.server.test_desktop_mesh_onboard import root as isolated_root

live = live_http
root = isolated_root


def required_member(record: NetworkRecord, device_id: str) -> MemberRecord:
    member = record.member(device_id)
    assert member is not None, "expected actual membership readback, not an absent row"
    return member


class Node:
    def __init__(self, root, name):
        self.root = root
        for path in (root, root / "home", root / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
        env.update(
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GH_CONFIG_DIR=str(root / "gh-config"),
            GLAB_CONFIG_DIR=str(root / "glab-config"),
            HOME=str(root / "home"),
            LOCAL_OPERATOR_CONFIG_DIR=str(root),
            TMPDIR=str(root / "tmp"),
        )
        helper = Path(__file__).parents[2] / "fixtures" / "mesh_process_node.py"
        self.error = (root / "node-stderr.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, str(helper), name],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.error,
            text=True,
            start_new_session=True,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.stdin, self.stdout = self.process.stdin, self.process.stdout
        self.lock = threading.Lock()
        try:
            self.ready = self.read(15)
        except BaseException:
            self.close()
            raise

    def read(self, timeout):
        with selectors.DefaultSelector() as poll:
            poll.register(self.stdout, selectors.EVENT_READ)
            assert poll.select(timeout), "node pipe exceeded its safety bound"
        line = self.stdout.readline()
        assert line, self.startup_diagnostic()
        return json.loads(line)

    def startup_diagnostic(self) -> str:
        """Only bounded known-safe import diagnostics, never arbitrary stderr.

        A dead child used to discard its only discriminating evidence. CI had
        eight identical EOF failures while local PYTHONPATH masked the cause.
        Other exception arguments/paths may contain secret data: name them but
        do not echo them, even though this fixture uses synthetic credentials.
        """
        import re

        try:
            with (self.root / "node-stderr.log").open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 8192))
                tail = stream.read(8192).decode(errors="replace")
        except OSError:
            tail = ""
        safe = []
        for line in tail.splitlines():
            if re.fullmatch(
                r"(?:ModuleNotFoundError|ImportError): No module named '[A-Za-z0-9_.]+'", line
            ):
                safe.append(line)
            elif re.match(r"^[A-Za-z][A-Za-z0-9_]*(?:Error|Exception):", line):
                safe.append(line.partition(":")[0] + " (details withheld)")
        return (
            "node exited before response; rc="
            + str(self.process.poll())
            + "; "
            + ("; ".join(safe)[-512:] or "startup stderr withheld/unavailable")
        )

    def cli(self, argv, timeout):
        with self.lock:
            self.stdin.write(json.dumps({"argv": argv, "timeout": timeout}) + "\n")
            self.stdin.flush()
            return self.read(timeout + 5)

    def audit(self):
        with self.lock:
            self.stdin.write('{"audit":true}\n')
            self.stdin.flush()
            return self.read(10)

    def close(self):
        try:
            if self.process.poll() is None:
                self.stdin.write('{"stop":true}\n')
                self.stdin.flush()
                try:
                    self.process.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(5)
        finally:
            for pipe in (self.stdin, self.stdout):
                pipe.close()
            self.error.close()


def test_startup_diagnostic_is_bounded_and_omits_untrusted_arguments(tmp_path, monkeypatch):
    from types import SimpleNamespace

    node = Node.__new__(Node)
    node.root = tmp_path
    monkeypatch.setattr(node, "process", SimpleNamespace(poll=lambda: 1), raising=False)
    secret = "SYNTHETIC_SECRET_SENTINEL"
    (tmp_path / "node-stderr.log").write_text(
        "x" * 9000
        + "\nModuleNotFoundError: No module named 'tests'\n"
        + "RuntimeError: "
        + secret
        + " /private/local/path\n"
    )
    diagnostic = node.startup_diagnostic()
    assert "No module named 'tests'" in diagnostic
    assert "RuntimeError (details withheld)" in diagnostic
    assert secret not in diagnostic and "/private/local/path" not in diagnostic
    assert len(diagnostic) < 600


def test_fixture_operator_row_never_falls_back_to_a_host_anchor(tmp_path, monkeypatch):
    """Actual producer fails under the old alias without ever reading host paths."""
    from types import SimpleNamespace

    from local_operator import operator
    from local_operator.network import readiness
    from local_operator.operator import trust
    from tests.fixtures.mesh_process_node import install_fixture_anchor

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    anchor = _make_key(tmp_path)
    (tmp_path / "fixture-owned-anchor.json").write_bytes(trust.anchor_bytes(anchor))
    original = (
        trust.load_anchor,
        operator.load_anchor,
        trust.anchor_path,
        trust._ANCHOR_ROOT_OVERRIDE,
    )
    try:
        reads = install_fixture_anchor(tmp_path)
        member = SimpleNamespace(device_id="d_fixture", name="fixture")
        fixed = readiness.operator_row(
            member, {"operator": readiness.operator_fact()}, peer_label="fixture"
        )
        assert fixed["ok"] is True and fixed["source"] == "peer"
        assert reads["native_anchor_reads"] == 0
        fixed_reads = reads["native_anchor_reads"]
        # Causal OLD defect: restore ONLY the missed package binding. The
        # original loader hits our forbidden private OS seam, not this host.
        operator.load_anchor = original[1]
        broken = readiness.operator_row(
            member, {"operator": readiness.operator_fact()}, peer_label="fixture"
        )
        assert broken["ok"] is False and reads["native_anchor_reads"] == 1
        operator.load_anchor = trust.load_anchor
        repaired = readiness.operator_row(
            member, {"operator": readiness.operator_fact()}, peer_label="fixture"
        )
        assert repaired["ok"] is True and reads["native_anchor_reads"] == 1
        print(
            json.dumps(
                {
                    "fixture_anchor_control": {
                        "fixed_operator_ok": fixed["ok"],
                        "fixed_native_attempts": fixed_reads,
                        "bad_alias_operator_ok": broken["ok"],
                        "blocked_native_attempts": reads["native_anchor_reads"],
                        "repaired_operator_ok": repaired["ok"],
                        "guard": "before-native-path-construction",
                    }
                }
            )
        )
    finally:
        (
            trust.load_anchor,
            operator.load_anchor,
            trust.anchor_path,
            trust._ANCHOR_ROOT_OVERRIDE,
        ) = original


class SshBoundary:
    """No fake success branch for join/member/grants/readiness CLI commands."""

    def __init__(self, node):
        self.node = node
        self.commands = []
        self.resolutions = []
        self.copies = {}
        self.join_calls = 0
        self.fingerprint = "SHA256:fixture"
        self.after_join = threading.Event()
        self.release_join = threading.Event()
        self.lose_join_response = False

    def probe(self):
        return onboard.Probe(
            ok=True, host="fixture-node", port=22, host_key_fp=self.fingerprint, at=0
        )

    def connect(self, credential, *, expected_fingerprint):
        assert credential is not None and credential.kind == "file"
        assert expected_fingerprint == "SHA256:fixture"
        return self.probe()

    def copy(self, local_path, remote_path):
        target = self.node.root / "tmp" / Path(remote_path).name
        shutil.copyfile(local_path, target)
        self.copies[remote_path] = str(target)
        return onboard.CommandResult((), 0)

    def run(self, argv, *, timeout, stdin=None):
        script = argv[-1]
        self.commands.append(script)

        def result(stdout=""):
            return onboard.CommandResult(tuple(argv), 0, stdout=stdout)

        if script == onboard._PRE_READ_SCRIPT:
            return result(
                "os=Linux\narch=x86_64\nuser=fixture\nlop=yes\nlop_path=/fixture/lop\n"
                f"lop_version={onboard._compiled_tag()}\nlop_update=yes\nuv=yes\n"
                "systemctl=yes\nlinger=yes\nsudo=yes\nsudo_nopass=yes\nanchor=\npre_read=done\n"
            )
        command = script.removeprefix(onboard._SCRIPT_PROLOGUE)
        for old, new in self.copies.items():
            command = command.replace(old, new)
        words = shlex.split(command)
        if words[:3] == ["lop", "operator", "install"]:
            # Root-owned OS installation ONLY is modeled. This is the ACTUAL
            # public anchor statement copied from the signed runner's local key.
            shutil.copyfile(
                words[words.index("--from") + 1], self.node.root / "fixture-owned-anchor.json"
            )
            return result()
        if words[:3] == ["lop", "operator", "trust"]:
            return result()
        if words[:3] == ["lop", "network", "restart"]:
            return result('{"ok":true}')  # OS service already runs in its child process
        if words[:2] in (["loginctl", "enable-linger"], ["loginctl", "show-user"]):
            return result("yes")
        if words[:2] == ["rm", "-f"]:
            return result()  # OS teardown only; fixture root reaped by pytest
        assert words and words[0] == "lop", "unexpected OS seam; do not invent a success"
        if words[1:3] == ["network", "join"]:
            assert "--automated" in words and "--sas-stdin" not in words
            self.join_calls += 1
        actual = self.node.cli(words[1:], timeout)
        if words[1:3] == ["network", "join"] and self.lose_join_response:
            assert actual["rc"] == 0, "real join failed before interruption barrier"
            self.after_join.set()
            assert self.release_join.wait(20), "test did not release post-admission barrier"
            self.lose_join_response = False
            return onboard.CommandResult(
                tuple(argv), 1, stderr="fixture lost the successful join response"
            )
        return onboard.CommandResult(
            tuple(argv), actual["rc"], stdout=actual["stdout"], stderr=actual["stderr"]
        )

    def close(self):
        pass


async def terminal(client, aid):
    async def wait():
        while True:
            response = await client.get(f"/v1/desktop/approvals/{aid}")
            assert response.status_code == 200
            payload = response.json()["result"]
            if not payload["run_active"]:
                return payload
            # Condition poll, not a latency assertion. This bounds load only.
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(wait(), 150)


@pytest_asyncio.fixture
async def rig(live, monkeypatch):
    client, app, root = live
    _make_key(root)
    node_a = Node(root, "fixture-inviter")
    node_b = None
    try:
        node_b = Node(root / "remote", "fixture-node")
        assert node_a.ready["pid"] != node_b.ready["pid"] != os.getpid()
        ssh = SshBoundary(node_b)
        monkeypatch.setattr(onboard, "SshTransport", lambda **kwargs: ssh)
        original_resolver = onboard.resolve_credential

        def resolve(reference, **kwargs):
            ssh.resolutions.append(True)
            return original_resolver(reference, **kwargs)

        monkeypatch.setattr(onboard, "resolve_credential", resolve)
        # TWO existing networks: the card selects the SECOND one explicitly.
        # Omitting --network at the actual invite CLI must fail, not mint on a
        # default/first network and accidentally give the test its own answer.
        created = await asyncio.to_thread(
            node_a.cli, ["network", "init", "fixture-selected", "--no-start", "--json"], 30
        )
        assert created["rc"] == 0
        parsed = onboard._json_from(created["stdout"])
        assert isinstance(parsed, dict)
        selected = parsed["network_id"]
        assert selected != node_a.ready["network_id"] and len(store.list_networks(root)) == 2
        key = root / "fixture-ssh-key"
        key.write_text("synthetic SSH boundary fixture, not a real credential")
        body = {
            "host": "fixture-node",
            "user": "fixture",
            "name": "fixture-node",
            "request_id": "req_process_fixture",
            "network_id": selected,
            "credential_ref": "path:" + str(key),
            "host_key_fp": "SHA256:fixture",
        }
        response = await client.post("/v1/desktop/approvals/request", json=body)
        assert response.status_code == 200, response.text
        aid = response.json()["result"]["approval_id"]
        yield client, root, node_a, node_b, ssh, body, aid
    finally:
        if node_b:
            node_b.close()
        node_a.close()


async def approve_enrollment(client, aid):
    response = await client.get(f"/v1/desktop/approvals/{aid}")
    snapshot = response.json()["result"]
    return await client.post(
        f"/v1/desktop/approvals/{aid}/approve-network-enrollment",
        json={
            "authority_version": 1,
            "scope": "network_only",
            "request_digest": snapshot["request_digest"],
        },
    )


@pytest.mark.asyncio
async def test_signed_card_existing_runner_reaches_two_process_members(rig):
    client, root, node_a, node_b, ssh, body, aid = rig
    assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 409
    response = await approve_enrollment(client, aid)
    assert response.status_code == 200, response.text
    approvals.verify_for_run(aid, root=root)
    response = await client.post(f"/v1/desktop/approvals/{aid}/run")
    assert response.status_code == 202, response.text
    completed = await terminal(client, aid)
    assert completed["state"] == "connected", [
        (r["step"], r["detail"]) for r in completed["receipts"]
    ]
    assert [r["step"] for r in completed["receipts"] if r["ok"]] == list(onboard.STEP_NAMES)
    assert ssh.join_calls == 1
    assert any(
        r["step"] == "provision" and r["detail"].startswith("SKIPPED:")
        for r in completed["receipts"]
    )
    assert "task readiness is not established" in completed["receipts"][-1]["detail"]
    assert "model_credential" in completed["receipts"][-1]["detail"]
    assert "git_identity" in completed["receipts"][-1]["detail"]
    report = completed["receipts"][-1]["data"]
    assert report["enrollment_verified"] is True and report["task_ready"] is False
    assert report["ready_ok"] is False
    assert {r["capability"] for r in report["equipment_warnings"]} >= {
        "git_identity",
        "model_credential",
    }
    assert all(set(r) == {"capability", "code"} for r in report["equipment_warnings"])
    actual = approvals.load_record(aid, root=root)["receipts"][-1]
    assert actual["data"] == report and actual["run_id"] == completed["run_id"]
    # A new API caller/reopened window reads the same durable, bounded facts.
    import httpx

    async with httpx.AsyncClient(base_url=str(client.base_url), headers=client.headers) as reopened:
        response = await reopened.get(f"/v1/desktop/approvals/{aid}")
        assert response.json()["result"]["receipts"][-1]["data"] == report
    from local_operator.network.credentials import placement

    for profile in (root, node_b.root):
        # Read the actual placement documents; do not substitute empty offers.
        document = placement.PlacementDocument.load(body["network_id"], profile)
        assert not document.entries
        assert not placement.placement_path(body["network_id"], profile).exists()

    assert not any("definitions push" in c or "mcp push" in c for c in ssh.commands)
    assert not (node_b.root / "auth.db").exists() and not (node_b.root / "home/.gitconfig").exists()
    left = store.load(body["network_id"], root)
    right = store.load(body["network_id"], node_b.root)
    assert left.epoch == right.epoch
    for record in (left, right):
        assert required_member(record, node_a.ready["device_id"]).active
        assert required_member(record, node_b.ready["device_id"]).active
    assert right.self_role == "drive"
    assert not store.load(node_a.ready["network_id"], root).member(node_b.ready["device_id"])
    # Real effective grant readback: exact card scopes, never admin convenience.
    card = approvals.load_record(aid, root=root)
    signed = card["what"]
    scopes = set(signed["grant"]) | ({"unattended"} if signed["unattended"] else set())
    # The invited node receives only its signed drive role. The signed grants
    # are directional: this operator device receives them ON that node, in
    # addition to its already-existing inviter role, not vice versa.
    for member in (
        required_member(left, node_b.ready["device_id"]),
        required_member(right, node_b.ready["device_id"]),
    ):
        assert member.role == signed["role"] == "drive"
        assert set(member.capabilities) == set(capabilities_for_role(signed["role"]))
        assert "admin" not in member.capabilities and "broker_credential" not in member.capabilities
    peer = required_member(right, node_a.ready["device_id"])
    baseline = required_member(left, node_a.ready["device_id"])
    assert peer.role == baseline.role
    assert set(peer.capabilities) == set(capabilities_for_role(baseline.role)) | scopes
    assert "credential_inheritance" not in signed and "credential_finite_share" not in signed
    public = json.loads((node_b.root / "fixture-owned-anchor.json").read_text())
    stored = approvals.load_record(aid, root=root)
    assert public["key_id"] == stored["what"]["anchor"]["key_id"]
    events = await asyncio.to_thread(node_a.audit)
    confirmed = [r for r in events if r.get("event") == "pairing_confirmed"]
    assert len(confirmed) == 1
    # Empty selected shares means no approved credential access. Passive sealed
    # masked descriptors may still be produced by the unchanged handshake.
    assert confirmed[0].get("detail", {}).get("shares") == 0  # producer records a count
    assert "approval:" + aid in json.dumps(confirmed)
    assert not any(r.get("event") == "sas_mismatch" for r in events)
    # The receipt names equipment warnings; configured fixture metadata is not
    # a native model completion, and readiness is never an offload success claim.
    assert "mcp" in completed["receipts"][-1]["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["no_bearer", "foreign_origin", "unsigned", "denied", "expired", "changed_identity"]
)
async def test_authority_table_has_no_unauthorized_remote_effects(rig, monkeypatch, state):
    client, root, node_a, node_b, ssh, body, aid = rig
    if state == "denied":
        assert (await client.post(f"/v1/desktop/approvals/{aid}/deny")).status_code == 200
    if state in {"expired", "changed_identity"}:
        assert (await approve_enrollment(client, aid)).status_code == 200
    if state == "expired":
        future = time.time() + 7200
        monkeypatch.setattr(approvals.time, "time", lambda: future)
    if state == "changed_identity":
        ssh.fingerprint = "SHA256:changed"
    headers = (
        {"Authorization": ""}
        if state == "no_bearer"
        else ({"Origin": "https://foreign.invalid"} if state == "foreign_origin" else {})
    )
    response = await client.post(f"/v1/desktop/approvals/{aid}/run", headers=headers)
    if state == "changed_identity":
        assert response.status_code == 202
        completed = await terminal(client, aid)
        assert completed["state"] == "failed"
        assert any(r["step"] == "pre_read" and not r["ok"] for r in completed["receipts"])
        assert len(approvals.list_records(root=root)) == 2  # corrected NEW card
        replacement_id = completed["replacement_id"]
        assert completed["next"] == "review_replacement" and replacement_id.startswith("apn_")
        assert replacement_id != aid
        replacement = approvals.load_record(replacement_id, root=root)
        assert replacement["kind"] == "device_onboard_network_only"
        assert replacement["state"] == "requested"
        assert replacement["signature"] is None and replacement["decided_at"] == 0
        assert replacement["request_id"] != approvals.load_record(aid, root=root)["request_id"]
        assert replacement["expires_at"] > replacement["created_at"]
        assert replacement["device"]["host_key_fp"] == "SHA256:changed"
        import httpx

        async with httpx.AsyncClient(
            base_url=str(client.base_url), headers=client.headers
        ) as reopened:
            again = (await reopened.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
            assert again["replacement_id"] == replacement_id
            assert again["next"] == "review_replacement"
        retry = await client.post(f"/v1/desktop/approvals/{aid}/run")
        assert retry.status_code == 409
        assert retry.json()["detail"]["code"] == "network_enrollment_review_required"
        assert len(approvals.list_records(root=root)) == 2
        # Missing replacement is a start-new-review instruction, never a bogus
        # Retry action or automatic mint. Delete ONLY this test's own fixture.
        approvals.record_path(replacement_id, root).unlink()
        missing = (await client.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
        assert missing["next"] == "start_new_review" and missing["replacement_id"] is None

    else:
        expected_status = {"no_bearer": 401, "foreign_origin": 403}.get(state, 409)
        assert response.status_code == expected_status
        expected_code = {
            "unsigned": "approval_record_tampered",
            "denied": "approval_denied",
            "expired": "approval_expired",
        }.get(state)
        if expected_code:
            assert response.json()["detail"]["code"] == expected_code
        if state == "expired":
            assert approvals.load_record(aid, root=root)["state"] == "expired"
        assert not approvals.load_record(aid, root=root).get("run")
        assert not store.load(body["network_id"], root).invites
    assert len(ssh.resolutions) == (1 if state == "changed_identity" else 0)
    assert not ssh.commands and ssh.join_calls == 0  # no install/join/grant commands
    assert not store.list_networks(node_b.root)
    assert not store.load(body["network_id"], root).member(node_b.ready["device_id"])


@pytest.mark.asyncio
async def test_lost_join_response_same_card_retry_and_single_target_writer(rig):
    client, root, node_a, node_b, ssh, body, aid = rig
    assert (await approve_enrollment(client, aid)).status_code == 200
    ssh.lose_join_response = True
    try:
        # Closing this independent response/client must not cancel the owner.
        import httpx

        async with httpx.AsyncClient(
            base_url=str(client.base_url), headers=client.headers
        ) as closer:
            assert (await closer.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 202
        assert await asyncio.to_thread(ssh.after_join.wait, 15)
        first = (await client.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
        assert first["run_active"]
        assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 409
        other = await client.post(
            "/v1/desktop/approvals/request", json={**body, "request_id": "req_same_target_second"}
        )
        other_id = other.json()["result"]["approval_id"]
        assert (await approve_enrollment(client, other_id)).status_code == 200
        assert (await client.post(f"/v1/desktop/approvals/{other_id}/run")).status_code == 409
        assert ssh.join_calls == 1
    finally:
        ssh.release_join.set()
    failed = await terminal(client, aid)
    assert failed["state"] == "failed"
    # Actual admission already happened on both processes, despite the lost ack.
    assert required_member(
        store.load(body["network_id"], node_b.root), node_a.ready["device_id"]
    ).active
    assert required_member(store.load(body["network_id"], root), node_b.ready["device_id"]).active
    before = store.load(body["network_id"], node_b.root)
    authority_before = {m.device_id: (m.role, set(m.capabilities)) for m in before.active_members()}
    assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 202
    completed = await terminal(client, aid)
    assert completed["state"] == "connected", completed["receipts"][-1]["detail"]
    assert completed["run_id"] != failed["run_id"]
    assert ssh.join_calls == 1  # existing active join was reused, not re-admitted
    assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 409
    right = store.load(body["network_id"], node_b.root)
    assert len(right.active_members()) == 2
    caps = required_member(right, node_a.ready["device_id"]).capabilities
    assert {
        m.device_id: (m.role, set(m.capabilities)) for m in right.active_members()
    } == authority_before
    assert len(caps) == len(set(caps))  # effective grants are a set, not duplicate authority

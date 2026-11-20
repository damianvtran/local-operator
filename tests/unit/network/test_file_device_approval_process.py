"""Real relay/control/desktop PID fit; native login/presence boundary is synthetic.

R owns two real loopback relays and their authenticated member handshake. D1/D2
are independent spawned processes with no R Python objects/maps. The harness Pipe
is test-only orchestration, never a new production op or file-approval mint.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from local_operator.network import account_binding as B
from local_operator.network import approvals as A
from local_operator.network import file_device_approval as F
from local_operator.network import identity, relay, store
from local_operator.network.audit import AuditLog
from local_operator.network.credentials import desktop_approval
from local_operator.network.types import MeshRefusal
from local_operator.operator import trust
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.oauth import radient
from tests.unit.network.conftest import NET_WAIT_BUDGET_S, wait_for
from tests.unit.network.test_relay_e2e import _pair, serve_shaped_relay
from tests.unit.providers.test_radient_account_retention import _verified_login_result


def _native_boundary(mp: pytest.MonkeyPatch, root: Path, keyfile: Path) -> None:
    from local_operator import operator
    from local_operator.operator import sign

    private = serialization.load_pem_private_key(keyfile.read_bytes(), password=None)
    assert isinstance(private, ec.EllipticCurvePrivateKey)
    point = private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    import hashlib

    trio = {
        "key_id": hashlib.sha256(point).hexdigest()[:32],
        "spki_fp": "sha256:" + hashlib.sha256(point).hexdigest(),
        "statement_digest": "sha256:" + hashlib.sha256(b"synthetic-process-boundary").hexdigest(),
    }
    loaded = SimpleNamespace(
        usable=True, root_owned=True, anchor=SimpleNamespace(backend=trust.SECURE_ENCLAVE)
    )
    mp.setattr(operator, "authority_level_load", lambda: (operator.LEVEL_OPERATOR_PRESENCE, loaded))
    mp.setattr(trust, "anchor_trio", lambda anchor: dict(trio))
    mp.setattr(A, "local_anchor_trio", lambda root=None: {**trio, "spki": point})
    mp.setattr(sign, "resolve_backend_name", lambda root: trust.SECURE_ENCLAVE)
    mp.setattr(AuthStore, "_rearm_parked_tunnel_login", lambda *a, **kw: None)
    # The actor's sign-in would start the org Agent Hub boundary on an unjoined
    # daemon thread (a network call with synthetic tokens, writing under
    # ``root/hub``); not what this boundary test exercises, and a late write
    # can recreate a deleted tmp dir. Same neutralisation as the ``rig`` fixture.
    mp.setattr(AuthStore, "_org_hub_boundary", lambda self, boundary: None)

    class SoftwarePresence:
        handle = SimpleNamespace(key_id=trio["key_id"])

        def sign(self, message, *, timeout=None):
            return private.sign(message, ec.ECDSA(hashes.SHA256()))

        def close(self):
            pass

    mp.setattr(sign, "load_signer", lambda **kwargs: SoftwarePresence())

    async def canonical(access, **kwargs):
        async def handler(request):
            assert str(request.url) == radient.DEFAULT_RADIENT_API_BASE_URL + "/me"
            assert request.headers["authorization"] == "Bearer " + access
            return httpx.Response(
                200,
                json={
                    "result": {
                        "identity": {"account_id": "account-a"},
                        "account": {"id": "account-a"},
                    }
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await radient.verify_canonical_account(access, http_client=client)

    mp.setattr(F, "verify_canonical_account", canonical)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)


def _member_link(server):
    def links():
        with server._links_lock:
            return [
                link
                for link in server.links.values()
                if link.result.role == "listener"
                and link.phase == "member"
                and link.alive
                and not link.closing
            ]

    assert wait_for(lambda: bool(links())), "actual authenticated incoming member link"
    return links()[0]


def _worker(pipe, role: str, base_text: str, keyfile_text: str) -> None:
    os.setsid()
    base, keyfile = Path(base_text), Path(keyfile_text)
    root = base / "owner"
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    a = b = auth = None
    row = None
    mp = pytest.MonkeyPatch()
    try:
        _native_boundary(mp, root, keyfile)
        if role == "relay":
            peer_root = base / "peer"
            own, peer = identity.load_or_mint(root), identity.load_or_mint(peer_root)
            a = serve_shaped_relay(root, mp, identity=own, audit=AuditLog(root))
            b = serve_shaped_relay(peer_root, mp, identity=peer, audit=AuditLog(peer_root))
            host, port = a.bind()
            a.bind_control()
            a.start()
            existing = store.list_networks(root)
            if not existing:
                os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(peer_root)
                record, _, _ = _pair((a, b, host, port), mp, role="read")
            else:
                record = existing[0]
            for selected_root in (root, peer_root):
                with store.mutate(record.network_id, selected_root) as current:
                    owner = current.member(own.device_id)
                    assert owner is not None
                    owner.endpoints = [f"{host}:{port}"]
                    store.save(current, selected_root)
            b.start()
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
            auth = AuthStore(root / "auth.db", config_dir=root)
            rows = auth.list_credentials("radient")
            row = (
                rows[0]
                if rows
                else auth.complete_radient_login(asyncio.run(_verified_login_result("account-a")))
            )
            link = _member_link(a)
            assert link.result.peer_public_key == b.identity.public_key
            initial = {
                "root": str(root),
                "instance": a.instance_id,
                "link_id": link.link_id,
                "listener": link.result.role,
            }
        else:
            initial = {}
        pipe.send(
            {
                "ok": True,
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "source": F.__file__,
                "prefix": sys.prefix,
                **initial,
            }
        )
        while True:
            command = pipe.recv()
            if command["cmd"] == "stop":
                pipe.send({"ok": True, "stopped": True})
                break
            try:
                op = command["cmd"]
                if role == "relay":
                    assert a is not None and auth is not None and row is not None
                    link = _member_link(a)
                    if op == "capture":
                        result = asyncio.run(
                            F.file_request(
                                link, auth, row.id, action=command.get("action", "approve")
                            )
                        )
                    elif op == "authorize":
                        result = asyncio.run(F.authorized_context(link, auth, row.id))
                    elif op == "binding":
                        # Test-only orchestration invokes the owner adapter on
                        # the ACTUAL authenticated listener. No production RPC
                        # is added and the process-local receipt never crosses
                        # this Pipe: only assertions/metadata are reported.
                        exact = B.ReadBinding(
                            owner_key=F.MeshKey(
                                device_id=a.identity.device_id, public_key=a.identity.public_key
                            ),
                            network_id=link.network_id,
                            network_epoch=link.epoch,
                            link_id=link.link_id,
                            transcript_hash=link.result.transcript_hash,
                            request_id="req_synthetic_process_read",
                            session_id="synthetic-owner-session",
                            source_row_id="synthetic-row",
                            provenance_id="synthetic-proof",
                            proof_digest="sha256:" + "a" * 64,
                            file_version="synthetic-seal-not-established-by-P1",
                            phase="fetch",
                            offset=0,
                            length=17,
                            transfer_nonce="a" * 32,
                        )
                        verified = asyncio.run(
                            B.verify_live(
                                incoming_link=link,
                                auth=auth,
                                owner_account=B.ExactNativeAccountSelection(credential_id=row.id),
                                evidence=B.OwnerFileEvidenceReference(
                                    association=command["association"]
                                ),
                                required_subject=B.CanonicalSubject(
                                    issuer=F.DEFAULT_RADIENT_API_BASE_URL,
                                    opaque_id=command.get("subject", "account-a"),
                                ),
                                required_purpose=F.PURPOSE,
                                read_binding=exact,
                            )
                        )
                        verified.require_read(
                            link,
                            (
                                exact.model_copy(update={"length": 18})
                                if command.get("substitute_read")
                                else exact
                            ),
                        )
                        assert verified.requester_key.public_key == link.result.peer_public_key
                        result = {
                            "source": B.__file__,
                            "subject": verified.subject.model_dump(),
                            "generation": verified.generation,
                            "journal_revision": verified.journal_revision,
                            "link_id": verified.read_binding.link_id,
                            "requester_key": verified.requester_key.public_key,
                        }
                    elif op == "status":
                        result = a.status()
                    else:
                        raise AssertionError(op)
                else:
                    if op == "status_probe":
                        own_record = store.find_own_relay(root)
                        assert own_record is not None
                        actual = relay.control_request(own_record, "net_status")
                        assert isinstance(actual, dict)
                        pipe.send(
                            {
                                "ok": True,
                                "pid": os.getpid(),
                                "result": {
                                    "op": actual.get("op"),
                                    "has_ok_field": "ok" in actual,
                                    "old_ok_predicate": actual.get("ok") is True,
                                    "instance": (actual.get("detail") or {}).get("instance_id"),
                                    "record_instance": own_record.instance_id,
                                },
                            }
                        )
                        continue
                    record = A.load_record(command["approval_id"], root=root)
                    facts = F.FileDeviceApprovalOp().disclosure(record)
                    args = {
                        "op": F.OP,
                        "request_digest": record["request_digest"],
                        "disclosure_digest_": desktop_approval.disclosure_digest(facts),
                    }
                    if op == "approve":
                        result = desktop_approval.approve(root, record["approval_id"], **args)
                    elif op == "apply":
                        result = desktop_approval.apply(root, record["approval_id"], **args)
                    else:
                        raise AssertionError(op)
                pipe.send({"ok": True, "pid": os.getpid(), "result": result})
            except MeshRefusal as error:
                pipe.send({"ok": False, "pid": os.getpid(), "code": error.code})
            except BaseException:
                pipe.send({"ok": False, "pid": os.getpid(), "traceback": traceback.format_exc()})
    except BaseException:
        pipe.send({"ok": False, "pid": os.getpid(), "traceback": traceback.format_exc()})
    finally:
        if auth is not None:
            auth.close()
        if b is not None:
            b.stop()
        if a is not None:
            a.stop()
        mp.undo()
        pipe.close()


class _Actor:
    def __init__(self, role, base, key):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, role, str(base), str(key)))
        self.stopped = False
        self.role = role
        self.process.start()
        child.close()
        try:
            self.ready = self.receive()
            assert self.ready["ok"], self.ready
            assert self.ready["pid"] == self.process.pid == self.ready["pgid"]
            assert Path(self.ready["source"]).resolve() == Path(F.__file__).resolve()
            assert Path(self.ready["prefix"]).resolve() == Path(sys.prefix).resolve()
        except BaseException:
            self.stop()
            raise

    def receive(self):
        assert self.pipe.poll(NET_WAIT_BUDGET_S), "actor completion under existing network backstop"
        return self.pipe.recv()

    def send(self, cmd, **fields):
        self.pipe.send({"cmd": cmd, **fields})

    def call(self, cmd, **fields):
        self.send(cmd, **fields)
        reply = self.receive()
        if not reply.get("ok"):
            # Preserve the substantive actor response before any outer bool
            # assertion. A False without its code/traceback is not diagnosis.
            print(
                "P1 ACTOR REPLY",
                json.dumps({"role": self.role, "command": cmd, **reply}, sort_keys=True),
                flush=True,
            )
        return reply

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.process.is_alive():
            try:
                self.call("stop")
            except (EOFError, BrokenPipeError, AssertionError):
                pass
            self.process.join(NET_WAIT_BUDGET_S)
        if self.process.is_alive():
            # Only this fixture's unreaped, separately owned group. No patterns,
            # peer cancellation, user service or unrecorded process is touched.
            pid = self.process.pid
            assert pid is not None
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
            else:
                # Startup failed before setsid: terminate only our child PID,
                # never the test runner's inherited process group.
                self.process.terminate()
            self.process.join(NET_WAIT_BUDGET_S)
        assert not self.process.is_alive(), "owned actor did not stop"
        self.process.close()
        self.pipe.close()


@pytest.mark.skipif(not hasattr(os, "setsid"), reason="POSIX strict-durability process fit")
def test_actual_listener_capture_crosses_desktop_pids_and_persists(tmp_path):
    key = tmp_path / "synthetic-presence.pem"
    private = ec.generate_private_key(ec.SECP256R1())
    key.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    r = d = other = None
    transcript = []
    try:
        r = _Actor("relay", tmp_path, key)
        d = _Actor("desktop", tmp_path, key)
        assert r.ready["pid"] != d.ready["pid"]
        probe = d.call("status_probe")
        assert probe["ok"], probe
        assert probe["result"]["op"] == "ack"
        assert probe["result"]["has_ok_field"] is False
        assert probe["result"]["old_ok_predicate"] is False
        assert probe["result"]["instance"] == probe["result"]["record_instance"]
        print("P1 ACTUAL CONTROL DISCRIMINATOR", json.dumps(probe, sort_keys=True), flush=True)
        unbound = r.call("binding", association="0" * 64)
        assert not unbound["ok"] and "code" in unbound, unbound
        captured = r.call("capture")
        assert captured["ok"], captured
        first = captured["result"]["approval_id"]
        status = r.call("status")
        assert status["ok"], status
        link = next(
            row for row in status["result"]["links"] if row["link_id"] == r.ready["link_id"]
        )
        assert (link["role"], link["alive"], link["closing"]) == ("listener", True, False)
        transcript.append({"R": r.ready["pid"], "D": d.ready["pid"], "status": link})
        # D restart must not lose R's durable candidate or borrow its Python map.
        d.stop()
        d = _Actor("desktop", tmp_path, key)
        assert d.call("approve", approval_id=first)["ok"]
        assert d.call("apply", approval_id=first)["ok"]
        active = r.call("authorize")
        assert active["ok"], active
        assert active["result"]["generation"] == 1
        association = active["result"]["association"]
        bound = r.call("binding", association=association)
        assert bound["ok"] and bound["result"]["generation"] == 1, bound
        assert bound["result"]["link_id"] == r.ready["link_id"]
        assert bound["result"]["subject"]["opaque_id"] == "account-a"
        wrong_subject = r.call("binding", association=association, subject="account-b")
        assert (
            not wrong_subject["ok"] and wrong_subject["code"] == "account_binding_subject_mismatch"
        )
        substituted = r.call("binding", association=association, substitute_read=True)
        assert not substituted["ok"] and substituted["code"] == "account_binding_read_mismatch"
        transcript.append({"binding": bound, "unbound": unbound, "substituted": substituted})
        stale = r.call("capture")
        assert stale["ok"], stale
        # R restart/new authenticated member link invalidates UNAPPLIED cards,
        # not completed full-key/account standing associations.
        old_instance = r.ready["instance"]
        r.stop()
        r = _Actor("relay", tmp_path, key)
        assert r.ready["instance"] != old_instance
        refused = d.call("approve", approval_id=stale["result"]["approval_id"])
        assert not refused["ok"] and refused.get("code") == "file_approval_link_changed", refused
        assert r.call("authorize")["ok"]
        rebound = r.call("binding", association=association)
        assert rebound["ok"] and rebound["result"]["link_id"] != bound["result"]["link_id"], rebound
        # Two independent workers hold decisions for the SAME expected g1.
        one, two = r.call("capture"), r.call("capture")
        assert one["ok"] and two["ok"]
        other = _Actor("desktop", tmp_path, key)
        assert d.call("approve", approval_id=one["result"]["approval_id"])["ok"]
        assert other.call("approve", approval_id=two["result"]["approval_id"])["ok"]
        d.send("apply", approval_id=one["result"]["approval_id"])
        other.send("apply", approval_id=two["result"]["approval_id"])
        outcomes = [d.receive(), other.receive()]
        assert sum(row["ok"] for row in outcomes) == 1, outcomes
        assert r.call("authorize")["result"]["generation"] == 2
        transcript.append(
            {"R2": r.ready["pid"], "D1": d.ready["pid"], "D2": other.ready["pid"], "CAS": outcomes}
        )
        # A later revoke is visible independently in R, including after the
        # current projection alone is restored by the owner test harness.
        root = Path(r.ready["root"])
        before = F._path(root).read_bytes()
        rev = r.call("capture", action="revoke")
        assert rev["ok"], rev
        assert d.call("approve", approval_id=rev["result"]["approval_id"])["ok"]
        assert d.call("apply", approval_id=rev["result"]["approval_id"])["ok"]
        assert not r.call("authorize")["ok"]
        assert not r.call("binding", association=association)["ok"]
        F._path(root).write_bytes(before)
        assert not r.call("authorize")["ok"]
        restored = r.call("binding", association=association)
        assert not restored["ok"] and restored["code"] == "file_approval_negative_fence", restored
        witness = F._load_journal(root)
        assert max(row.deny_generation for row in witness.fences.values()) == 3
        print("P1 REAL PROCESS FIT", json.dumps(transcript, sort_keys=True))
    finally:
        for actor in (other, d, r):
            if actor is not None:
                actor.stop()

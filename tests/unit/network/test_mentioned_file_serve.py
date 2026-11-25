"""``net_mentioned_file``: owner-side account accessor and the thin serving op.

Reuses the P1 rig (real SQLite/signatures, synthetic presence/canonical account)
and the B1 lane (real P2 transcript proof, audit and seal). No sockets: the
relay's shared page loop is the REAL ``RelayServer`` method bound onto the rig's
shape-double server, so the async bridge under test is the production one.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import account_binding as B
from local_operator.network import file_device_approval as F
from local_operator.network import files
from local_operator.network import mentioned_file_read as M
from local_operator.network.relay import RelayServer
from local_operator.network.types import MeshRefusal
from local_operator.providers.auth_store import AuthStore
from local_operator.session import file_provenance as P
from tests.unit.network import test_account_binding as binding_tests
from tests.unit.network import test_mentioned_file_read as read_tests
from tests.unit.providers.test_radient_account_retention import _verified_login_result

rig = binding_tests.rig
lane = read_tests.lane

ROOT = Path(__file__).resolve().parents[3]


def _serve_module():
    # Imported lazily so the module-level import graph of this test file does
    # not itself prove anything about default-off reachability.
    from local_operator.network import mentioned_file_serve

    return mentioned_file_serve


def _accessor(rig, auth=None):
    return F._owner_serving_account(rig.link, auth or rig.auth, rig.root)


def _inputs(account):
    """The three verify_live inputs, built ONLY from the accessor's result."""
    return (
        B.ExactNativeAccountSelection(credential_id=account.credential_id),
        B.OwnerFileEvidenceReference(association=account.association),
        B.CanonicalSubject(issuer=account.issuer, opaque_id=account.opaque_id),
    )


def _naive_slot_inputs(rig):
    """THE TRAP, written out so the tests can prove they discriminate.

    Credential, subject and association taken from the stored slot that
    ``_verified_context`` will itself match: verify_live's subject and evidence
    mismatch checks then compare the slot with itself and cannot fire.
    """
    document = F._load(rig.root)
    ((key, slot),) = document.slots.items()
    account = slot.intent.account
    return (
        B.ExactNativeAccountSelection(credential_id=account.credential_id),
        B.OwnerFileEvidenceReference(association=key),
        B.CanonicalSubject(issuer=account.hub, opaque_id=account.account_id),
    )


def _verify(rig, inputs, **changes):
    owner_account, evidence, subject = inputs
    return asyncio.run(
        B.verify_live(
            incoming_link=rig.link,
            auth=rig.auth,
            owner_account=owner_account,
            evidence=evidence,
            required_subject=subject,
            required_purpose=F.PURPOSE,
            read_binding=binding_tests._read(rig, **changes),
        )
    )


def _login_b(rig):
    # The real interactive completion: a second native row, A left as it was
    # (measured: complete_radient_login disables neither row).
    row = rig.auth.complete_radient_login(asyncio.run(_verified_login_result("account-b")))
    rig.observed["account"] = "account-b"
    return row


def _disable(rig, credential_id):
    rig.auth._conn.execute(
        "UPDATE auth_credentials SET disabled_cause = 'synthetic' WHERE id = ?", (credential_id,)
    )
    rig.auth._conn.commit()


def _code(error):
    return getattr(error.value, "code", None)


# --- 1a. account trap ------------------------------------------------------


def test_relogin_as_b_with_a_disabled_refuses_under_the_owner_accessor(rig):
    _ = binding_tests._active(rig)  # slot approved under native account A
    b = _login_b(rig)
    _disable(rig, rig.row.id)
    account = _accessor(rig)
    assert (account.credential_id, account.opaque_id) == (b.id, "account-b")
    with pytest.raises(B.BindingRefusal) as refused:
        _verify(rig, _inputs(account))
    assert _code(refused) == "file_approval_not_active"
    # With A's row disabled, the naive accessor is ALSO refused, but only by
    # _native's row check; its subject/evidence inputs are the slot's own, so
    # the two mismatch checks are tautological for it (stated in the report).
    naive = _naive_slot_inputs(rig)
    context_subject = naive[2]
    assert context_subject.opaque_id == "account-a"
    with pytest.raises(B.BindingRefusal) as naive_refused:
        _verify(rig, naive)
    assert _code(naive_refused) == "file_approval_native_account_required"


def test_relogin_as_b_with_a_still_enabled_naive_serves_stale_owner_refuses(rig):
    _ = binding_tests._active(rig)
    _login_b(rig)
    # The naive slot-derived accessor DOES serve, under the stale account A.
    rig.observed["account"] = "account-a"
    served = _verify(rig, _naive_slot_inputs(rig))
    assert served.subject.opaque_id == "account-a"
    # The owner-side accessor never picks one of two current logins.
    with pytest.raises(MeshRefusal) as refused:
        _accessor(rig)
    assert _code(refused) == "mentioned_file_owner_account_ambiguous"


# --- 1b. evidence trap -----------------------------------------------------


def test_slot_key_not_derived_from_link_identities_fails_evidence_check(rig, monkeypatch):
    real = F._association_of
    # Store the slot under a key that is NOT the link-authenticated hash.
    monkeypatch.setattr(F, "_association", lambda intent: real(intent.owner, intent.owner, "x"))
    _ = binding_tests._active(rig)
    naive = _naive_slot_inputs(rig)
    assert _verify(rig, naive).association == naive[1].association  # naive serves
    account = _accessor(rig)
    assert account.association != naive[1].association
    with pytest.raises(B.BindingRefusal) as refused:
        _verify(rig, _inputs(account))
    assert _code(refused) == "account_binding_evidence_mismatch"


# --- 1c. subject check stays failable --------------------------------------


@pytest.mark.parametrize("field", ["issuer", "opaque_id"])
def test_subject_differing_from_owner_native_row_is_refused(rig, field):
    _ = binding_tests._active(rig)
    owner_account, evidence, subject = _inputs(_accessor(rig))
    assert _verify(rig, (owner_account, evidence, subject)).subject == subject
    other = B.CanonicalSubject(**{**subject.model_dump(), field: "https://other.invalid"})
    with pytest.raises(B.BindingRefusal) as refused:
        _verify(rig, (owner_account, evidence, other))
    assert _code(refused) == "account_binding_subject_mismatch"


# --- 1d. inventory and incarnation refusals --------------------------------


def test_inventory_unknown_zero_and_several_are_typed_refusals(rig, monkeypatch):
    _ = binding_tests._active(rig)
    with monkeypatch.context() as patch:
        patch.setattr(AuthStore, "native_oauth_inventory", staticmethod(lambda path: None))
        with pytest.raises(MeshRefusal) as unknown:
            _accessor(rig)
    assert _code(unknown) == "mentioned_file_owner_account_unknown"
    b = _login_b(rig)
    with pytest.raises(MeshRefusal) as several:
        _accessor(rig)
    assert _code(several) == "mentioned_file_owner_account_ambiguous"
    _disable(rig, rig.row.id)
    _disable(rig, b.id)
    with pytest.raises(MeshRefusal) as zero:
        _accessor(rig)
    assert _code(zero) == "mentioned_file_owner_account_required"


def test_borrowed_or_non_oauth_rows_are_not_owner_accounts(rig, monkeypatch):
    real = AuthStore.native_oauth_inventory

    def with_noise(path):
        rows = real(path)
        assert rows is not None and len(rows) == 1
        provider, _id, incarnation, *rest = rows[0]
        return [
            *rows,
            ("radient", 90, incarnation, "x", None, None, "object", "oauth"),  # mesh_origin
            ("radient", 91, incarnation, "x", None, None, None, "api_key"),
            ("openai", 92, incarnation, "x", None, None, None, "oauth"),
            ("radient", 93, None, "x", None, None, None, "oauth"),  # no incarnation
        ]

    monkeypatch.setattr(AuthStore, "native_oauth_inventory", staticmethod(with_noise))
    assert _accessor(rig).credential_id == rig.row.id


def test_incarnation_drift_between_inventory_and_native_is_relogin(rig, monkeypatch):
    real = AuthStore.native_oauth_inventory

    def drifted(path):
        return [(*row[:2], "a-different-login", *row[3:]) for row in real(path) or []]

    monkeypatch.setattr(AuthStore, "native_oauth_inventory", staticmethod(drifted))
    with pytest.raises(MeshRefusal) as refused:
        _accessor(rig)
    assert _code(refused) == "mentioned_file_owner_relogin"


def test_native_account_changed_surfaces_as_relogin(rig, monkeypatch):
    def changed(auth, credential_id, root):
        raise F._refuse("file_approval_native_account_changed")

    monkeypatch.setattr(F, "_native", changed)
    with pytest.raises(MeshRefusal) as refused:
        _accessor(rig)
    assert _code(refused) == "mentioned_file_owner_relogin"


# --- 1e. the association hash is byte-identical ----------------------------


def test_association_delegates_to_one_byte_identical_hash(rig):
    record = binding_tests.request(rig)
    intent = F._intent(record)
    expected = hashlib.sha256(
        json.dumps(
            {
                "owner": intent.owner.model_dump(),
                "requester": intent.requester.model_dump(),
                "network_id": intent.network_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert F._association(intent) == expected
    assert F._association(intent) == F._association_of(
        intent.owner, intent.requester, intent.network_id
    )
    mesh = F._mesh(rig.link, rig.root)
    assert _accessor(rig).association == F._association_of(
        mesh["owner"], mesh["requester"], mesh["network_id"]
    )


# --- serving op ------------------------------------------------------------


@pytest.fixture
def serving(lane):
    """The real RelayServer page loop bound onto the rig's shape-double server."""
    server = lane.rig.server
    server._page_loop = None
    server._page_loop_thread = None
    server._page_loop_shelved = False
    server._page_loop_lock = threading.Lock()
    server._page_loop_main = RelayServer._page_loop_main
    server._page_loop_locked = types.MethodType(RelayServer._page_loop_locked, server)
    serve = _serve_module().MentionedFileServing(server)
    try:
        yield SimpleNamespace(lane=lane, serve=serve, server=server)
    finally:
        loop = server._page_loop
        thread = server._page_loop_thread
        if loop is not None and thread is not None:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)


def _frame(lane, **changes):
    read = lane.read
    frame = {
        "op": "net_mentioned_file",
        "req": "synthetic-req",
        "session_id": read.session_id,
        "provenance_id": read.provenance_id,
        "source_row_id": read.source_row_id,
        "proof_digest": read.proof_digest,
        "file_version": "",
        "phase": "stat",
        "offset": 0,
        "length": 0,
        "transfer_nonce": "c" * 32,
    }
    frame.update(changes)
    return {k: v for k, v in frame.items() if v is not None}


def _call(serving, **changes):
    return serving.serve.handle(serving.lane.rig.link, _frame(serving.lane, **changes))


@pytest.fixture
def no_reads(monkeypatch):
    calls = {"read_span": 0, "accessor": 0, "proof": 0}
    real_span, real_accessor, real_proof = (
        files._read_span,
        F._owner_serving_account,
        (P.lookup_proof),
    )

    def span(*args, **kwargs):
        calls["read_span"] += 1
        return real_span(*args, **kwargs)

    def accessor(*args, **kwargs):
        calls["accessor"] += 1
        return real_accessor(*args, **kwargs)

    def proof(*args, **kwargs):
        calls["proof"] += 1
        return real_proof(*args, **kwargs)

    monkeypatch.setattr(files, "_read_span", span)
    monkeypatch.setattr(F, "_owner_serving_account", accessor)
    monkeypatch.setattr(P, "lookup_proof", proof)
    return calls


def test_stat_fetch_verify_happy_path_returns_net_file_get_shapes(serving):
    content = serving.lane.target.read_bytes()
    stat = _call(serving)
    assert set(stat) == {"name", "bytes", "sha256", "attachment", "mime", "mtime_ns", "version"}
    assert stat["name"] == "report.txt" and stat["bytes"] == len(content)
    assert stat["sha256"] == "sha256:" + hashlib.sha256(content).hexdigest()
    assert stat["attachment"] == stat["sha256"][len("sha256:") :][:32]
    assert stat["version"].startswith("sealed:sha256:")
    fetch = _call(serving, phase="fetch", file_version=stat["version"], limit=10, length=None)
    assert set(fetch) == {"offset", "bytes", "data", "eof", "chunk_digest", "file_bytes"}
    assert base64.b64decode(fetch["data"]) == content[:10]
    assert fetch["eof"] is False and fetch["file_bytes"] == len(content)
    assert fetch["chunk_digest"] == "sha256:" + hashlib.sha256(content[:10]).hexdigest()
    rest = _call(serving, phase="fetch", file_version=stat["version"], offset=10, length=None)
    assert base64.b64decode(rest["data"]) == content[10:] and rest["eof"] is True
    good = "sha256:" + hashlib.sha256(content[:12]).hexdigest()
    verify = _call(
        serving, phase="verify", file_version=stat["version"], length=12, prefix_digest=good
    )
    assert verify == {"match": True}
    bad = "sha256:" + "0" * 64
    assert _call(
        serving, phase="verify", file_version=stat["version"], length=12, prefix_digest=bad
    ) == {"match": False}
    restat = _call(serving, file_version=stat["version"])
    assert restat == stat


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/etc/passwd"},
        {"name": "report.txt"},
        {"file": "x"},
        {"out": "x"},
        {"a/b": "x"},
        {"land_path": "x"},
        {"provenance_id": "../../outside"},
        {"provenance_id": "A" * 32},
        {"proof_digest": "a" * 64},
        {"transfer_nonce": "short"},
        {"phase": "resume"},
        {"offset": True},
        {"offset": -1},
        {"session_id": "a/b"},
        {"frame": {"session_id": "x"}},
        {"phase": "fetch"},  # a fetch names a sealed version
    ],
)
def test_bad_request_is_refused_before_any_lookup(serving, no_reads, change):
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, **change)
    assert _code(refused) == "mentioned_file_bad_request"
    assert no_reads == {"read_span": 0, "accessor": 0, "proof": 0}
    assert read_tests.rows(serving.lane) == []


@pytest.mark.parametrize(
    "change", [{"provenance_id": "f" * 32}, {"proof_digest": "sha256:" + "0" * 64}]
)
def test_unknown_provenance_or_digest_mismatch_reads_no_target_bytes(serving, no_reads, change):
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, **change)
    assert _code(refused) == "mentioned_file_unavailable"
    assert no_reads["read_span"] == 0
    assert read_tests.rows(serving.lane) == []


def test_unknown_sealed_version_is_refused_without_reading(serving, no_reads):
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, phase="fetch", file_version="sealed:sha256:" + "0" * 64, length=4)
    assert _code(refused) == "mentioned_file_unavailable"
    assert no_reads["read_span"] == 0


def test_versions_are_forgotten_with_their_link(serving):
    stat = _call(serving)
    serving.serve.forget_link(serving.lane.rig.link, "closed")
    with pytest.raises(MeshRefusal):
        _call(serving, phase="fetch", file_version=stat["version"], length=4)


def test_owner_relogin_mid_request_is_a_typed_refusal(serving, monkeypatch):
    stat = _call(serving)
    rig = serving.lane.rig
    real_prepare = M.OwnerMentionedFileReader.prepare
    read_calls = {"n": 0}
    real_span = files._read_span

    async def relogin_then_prepare(self, version, read):
        # Between the accessor and prepare: the owner's native login moves on.
        rig.auth._conn.execute(
            "UPDATE auth_credentials SET login_incarnation = 'relogged' WHERE id = ?",
            (rig.row.id,),
        )
        rig.auth._conn.commit()
        return await real_prepare(self, version, read)

    def span(*args, **kwargs):
        read_calls["n"] += 1
        return real_span(*args, **kwargs)

    monkeypatch.setattr(M.OwnerMentionedFileReader, "prepare", relogin_then_prepare)
    monkeypatch.setattr(files, "_read_span", span)
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, phase="fetch", file_version=stat["version"], length=4)
    assert _code(refused) == "mentioned_file_owner_relogin"
    assert read_calls["n"] == 0


# --- 4. default-off gate ---------------------------------------------------


def _recording_server(root):
    calls = []

    def register_ops(handlers, local_handlers=None, **kwargs):
        calls.append((dict(handlers), dict(kwargs)))

    # A shape double for files.install's one use of the server; typed Any so
    # the checker does not demand a whole RelayServer for two attributes.
    server: Any = SimpleNamespace(root=root, register_ops=register_ops)
    return server, calls


def test_default_off_registers_nothing_and_never_imports_the_primitive(tmp_path):
    from local_operator.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.values["mentioned_file_serving"] is False
    script = (
        "import sys, types\n"
        "from pathlib import Path\n"
        "import local_operator.network.relay\n"
        "from local_operator.network import files\n"
        "ops = []\n"
        "server = types.SimpleNamespace(root=Path(sys.argv[1]),"
        " register_ops=lambda h, l=None, **k: ops.extend(h))\n"
        "files.install(server)\n"
        "print(sorted(ops))\n"
        "print('local_operator.network.mentioned_file_read' in sys.modules,"
        " 'local_operator.network.mentioned_file_serve' in sys.modules)\n"
    )
    env = {
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "cfg"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT),
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    root = tmp_path / "owner"
    root.mkdir()
    for config in (None, "values:\n  mentioned_file_serving: false\n", "values: [\n"):
        if config is not None:
            (root / "config.yml").write_text(config)
        done = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        lines = done.stdout.strip().splitlines()
        assert lines[-2] == "['net_file_get', 'net_session_files']", config
        assert lines[-1] == "False False", config


def test_enabled_registers_the_op_with_the_slow_deadline(tmp_path):
    (tmp_path / "config.yml").write_text("values:\n  mentioned_file_serving: true\n")
    server, calls = _recording_server(tmp_path)
    files.install(server)
    served = {name: kwargs for handlers, kwargs in calls for name in handlers}
    assert "net_mentioned_file" in served
    assert served["net_mentioned_file"]["slow"] == {"net_mentioned_file": files.FILE_OP_DEADLINE_S}
    assert served["net_mentioned_file"]["on_link_closed"]


def test_the_op_is_a_closed_slice_peer_op_with_a_capability_row():
    from local_operator.network import relay
    from local_operator.network import types as mesh_types

    assert "net_mentioned_file" in mesh_types.NET_OPS
    assert mesh_types.OP_CAPABILITY["net_mentioned_file"] == "view"
    assert "net_mentioned_file" in relay.SLICE_PEER_OPS


# --- PR #4563 round-1 remediation regressions -------------------------------


def _version_dirs(lane):
    base = lane.rig.root / "sessions" / lane.read.session_id / M._DIRECTORY
    return sorted(p.name for p in base.iterdir()) if base.is_dir() else []


def _held_count(serving):
    held = serving.serve._versions.get(serving.lane.rig.link.link_id)
    return 0 if held is None else len(held[1])


@pytest.mark.parametrize("purpose", ["replica", "move"])
def test_snapshots_never_reach_a_view_members_net_sync(serving, monkeypatch, purpose):
    """S1-1: the sealed snapshot is not net_sync content (plan, fetch, copy set)."""
    from local_operator.network import sync

    _call(serving)
    lane = serving.lane
    nonce = _version_dirs(lane)[0]
    # The registered net_sync handler, driven by a link double holding only
    # ``view``: the replica watcher thread is not what this test exercises.
    monkeypatch.setattr(sync, "ensure_watcher", lambda server: None)
    handler = sync.make_handler(lane.rig.server)
    viewer = SimpleNamespace(device_id=lane.rig.link.device_id, capabilities=["view"])
    sid = lane.read.session_id
    plan = handler(
        viewer, {"op": "net_sync", "phase": "plan", "session_id": sid, "purpose": purpose}
    )
    assert not [m for m in plan["members"] if m.startswith(M._DIRECTORY)], plan["members"]
    assert not [
        n
        for n in sync.tree_entry_names(sync.session_dir(lane.rig.root, sid))
        if n.startswith(M._DIRECTORY)
    ]
    for leaf in ("snapshot", "seal.json"):
        with pytest.raises(MeshRefusal) as refused:
            handler(
                viewer,
                {
                    "op": "net_sync",
                    "phase": "fetch",
                    "session_id": sid,
                    "plan_id": plan.get("plan_id", ""),
                    "name": f"{M._DIRECTORY}/{nonce}/{leaf}",
                    "offset": 0,
                },
            )
        assert _code(refused) == "unknown_item"
    assert M._DIRECTORY in sync.NEVER_COPIED
    assert sync.unlisted_entries(sync.session_dir(lane.rig.root, sid)) == []


def test_establish_runs_off_the_shared_page_loop(serving, monkeypatch):
    """R1-1: a slow establishment never occupies the relay's page loop."""
    threads: list[str] = []
    real_span = files._read_span

    def slow_span(*args, **kwargs):
        threads.append(threading.current_thread().name)
        time.sleep(0.4)
        return real_span(*args, **kwargs)

    monkeypatch.setattr(files, "_read_span", slow_span)
    server = serving.server
    with server._page_loop_lock:
        loop = server._page_loop_locked()
    result: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: result.update(stat=_call(serving)))
    worker.start()
    deadline = time.monotonic() + 10
    while not threads and time.monotonic() < deadline:
        time.sleep(0.01)
    latencies = []
    while worker.is_alive() and len(latencies) < 5:
        started = time.monotonic()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=10)
        latencies.append(time.monotonic() - started)
        time.sleep(0.1)
    worker.join(timeout=30)
    assert "stat" in result
    assert threads and "mesh-page-loop" not in threads, threads
    assert latencies and max(latencies) < 0.2, latencies


def test_timeout_answers_typed_and_strands_nothing(serving, monkeypatch):
    """R1-3/R1-4: typed timeout reaches the peer; the late work closes cleanly."""
    serve_module = _serve_module()
    assert serve_module.MENTIONED_FILE_OP_TIMEOUT_S < files.FILE_OP_DEADLINE_S
    monkeypatch.setattr(serve_module, "MENTIONED_FILE_OP_TIMEOUT_S", 0.3)
    real_span = files._read_span
    monkeypatch.setattr(
        files, "_read_span", lambda *a, **kw: (time.sleep(0.6), real_span(*a, **kw))[1]
    )
    errors: list[BaseException] = []
    finished = threading.Event()
    real_establish = M.OwnerMentionedFileReader.establish

    async def recording(self, initial):
        try:
            return await real_establish(self, initial)
        except BaseException as error:
            errors.append(error)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(M.OwnerMentionedFileReader, "establish", recording)
    started = time.monotonic()
    with pytest.raises(MeshRefusal) as refused:
        _call(serving)
    assert _code(refused) == "mentioned_file_owner_timeout"
    assert time.monotonic() - started < 2
    assert finished.wait(30)
    deadline = time.monotonic() + 10
    while _version_dirs(serving.lane) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [type(e).__name__ for e in errors] == []
    assert _version_dirs(serving.lane) == []
    assert _held_count(serving) == 0


def test_restat_reuses_one_version_and_link_close_removes_it(serving):
    """R1-2/S1-3: three stats hold one version and one directory."""
    stats = [_call(serving) for _ in range(3)]
    assert stats[0] == stats[1] == stats[2]
    assert len(_version_dirs(serving.lane)) == 1
    assert _held_count(serving) == 1
    serving.serve.forget_link(serving.lane.rig.link, "closed")
    assert _version_dirs(serving.lane) == []


def test_a_version_that_no_longer_prepares_is_superseded_and_removed(serving, monkeypatch):
    first = _call(serving)
    old = _version_dirs(serving.lane)
    real_prepare = M.OwnerMentionedFileReader.prepare

    async def stale_once(self, version, read):
        if version.commitment == first["version"]:
            M._refuse()
        return await real_prepare(self, version, read)

    monkeypatch.setattr(M.OwnerMentionedFileReader, "prepare", stale_once)
    second = _call(serving)
    assert second["version"] != first["version"]
    now = _version_dirs(serving.lane)
    assert len(now) == 1 and now != old
    assert _held_count(serving) == 1


def test_a_refusal_after_sealing_leaves_no_snapshot_directory(serving, monkeypatch):
    """S1-4/R1-3: a stat refused after establishment does not strand its copy."""
    real = P.lookup_proof

    def gone_after_seal(root, session, opaque):
        if any(serving.lane.rig.root.glob(f"sessions/*/{M._DIRECTORY}/*/seal.json")):
            return None
        return real(root, session, opaque)

    monkeypatch.setattr(P, "lookup_proof", gone_after_seal)
    with pytest.raises(MeshRefusal) as refused:
        _call(serving)
    assert _code(refused) == "mentioned_file_unavailable"
    assert _version_dirs(serving.lane) == []


def _unapprove(rig):
    """One usable login, and no approval slot at all for any requester."""
    path = F._path(rig.root)
    path.rename(path.with_name("set-aside-approvals.json"))


def _revoke(rig):
    # Real signed revoke, as the reader's own revocation test does.
    revocation = binding_tests.approval_tests.request(rig, action="revoke")
    binding_tests.approval_tests.approve(rig, revocation)
    F.revoke(revocation["approval_id"], F.expected(revocation), rig.root)


@pytest.mark.parametrize(
    "condition", ["zero", "ambiguous", "unknown", "no_slot", "revoked", "account_unverified"]
)
def test_owner_account_state_is_generic_before_p1(serving, monkeypatch, caplog, condition):
    """S1-2/S2-2: before P1, the peer learns nothing of the owner's logins.

    ``no_slot``, ``revoked`` and ``account_unverified`` are refusals P1 itself
    raises, reachable only with exactly one usable login, so they must answer
    exactly as 0, 2 or unknown logins do.
    """
    rig = serving.lane.rig
    if condition == "zero":
        _disable(rig, rig.row.id)
    elif condition == "ambiguous":
        _login_b(rig)
    elif condition == "unknown":
        monkeypatch.setattr(AuthStore, "native_oauth_inventory", staticmethod(lambda path: None))
    elif condition == "no_slot":
        _unapprove(rig)
    elif condition == "revoked":
        _revoke(rig)
    else:
        rig.observed["error"] = RuntimeError("synthetic /me outage")
    caplog.set_level("WARNING", logger=_serve_module().__name__)
    with pytest.raises(MeshRefusal) as refused:
        _call(serving)
    assert _code(refused) == "mentioned_file_unavailable"
    assert refused.value.sentence == "the owner cannot prepare this file"
    # The owner keeps the typed reason.
    assert any(
        "answered to the peer as mentioned_file_unavailable" in record.getMessage()
        for record in caplog.records
    )
    assert _version_dirs(serving.lane) == [] and _held_count(serving) == 0


def test_an_unapproved_peer_cannot_tell_whether_the_named_file_changed(serving, no_reads):
    """S2-1: no proof lookup before P1, so file drift is not observable."""
    _unapprove(serving.lane.rig)
    answers = []
    for changed in (False, True):
        if changed:
            with serving.lane.target.open("ab") as handle:
                handle.write(b" appended after the mention")
        with pytest.raises(MeshRefusal) as refused:
            _call(serving)
        answers.append((_code(refused), refused.value.sentence))
        # The oracle itself: no proof lookup may run before P1 refuses.
        assert no_reads["proof"] == 0, (changed, answers)
    assert answers[0] == answers[1], answers
    assert answers[0] == ("mentioned_file_unavailable", "the owner cannot prepare this file")
    assert no_reads["read_span"] == 0
    assert read_tests.rows(serving.lane) == [] and _version_dirs(serving.lane) == []


def test_a_changed_file_restat_drops_the_held_version(serving):
    """Q2-1: a held version that no longer prepares is removed, even when
    its re-establishment is refused too."""
    _call(serving)
    assert len(_version_dirs(serving.lane)) == 1
    with serving.lane.target.open("ab") as handle:
        handle.write(b" appended after the stat")
    with pytest.raises(MeshRefusal):
        _call(serving)
    assert _version_dirs(serving.lane) == []
    assert _held_count(serving) == 0


def test_restat_after_relogin_is_named_relogin_once_verified(serving):
    """S2-5: a verified requester's re-stat after a relogin says relogin."""
    _call(serving)
    rig = serving.lane.rig
    rig.auth._conn.execute(
        "UPDATE auth_credentials SET login_incarnation = 'relogged' WHERE id = ?", (rig.row.id,)
    )
    rig.auth._conn.commit()
    with pytest.raises(MeshRefusal) as refused:
        _call(serving)
    assert _code(refused) == "mentioned_file_owner_relogin"
    assert _version_dirs(serving.lane) == [] and _held_count(serving) == 0


def _readers_alive():
    return [t for t in threading.enumerate() if t.name == "mesh-mentioned-file" and t.is_alive()]


def _drain(serving, timeout=30.0):
    deadline = time.monotonic() + timeout
    while (_readers_alive() or _version_dirs(serving.lane)) and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.fixture
def gated_reads(monkeypatch):
    """Every source/snapshot read blocks until ``gate`` is set (a stalled mount)."""
    gate = threading.Event()
    entered: list[threading.Thread] = []
    real_span = files._read_span

    def span(*args, **kwargs):
        entered.append(threading.current_thread())
        gate.wait(30)
        return real_span(*args, **kwargs)

    monkeypatch.setattr(files, "_read_span", span)
    try:
        yield SimpleNamespace(gate=gate, entered=entered)
    finally:
        gate.set()
        deadline = time.monotonic() + 30
        while _readers_alive() and time.monotonic() < deadline:
            time.sleep(0.02)


def test_abandoned_readers_stay_within_the_reader_cap(serving, monkeypatch, gated_reads):
    """R2-1: timed-out readers keep their slot until done; extras are refused busy."""
    serve_module = _serve_module()
    cap = 2
    # raising=False so the unfixed module (no cap at all) fails on the counts.
    monkeypatch.setattr(serve_module, "_READERS", threading.BoundedSemaphore(cap), raising=False)
    monkeypatch.setattr(serve_module, "MENTIONED_FILE_OP_TIMEOUT_S", 0.05)
    codes, peak = [], 0
    for _ in range(cap * 3):
        with pytest.raises(MeshRefusal) as refused:
            _call(serving)
        codes.append(_code(refused))
        peak = max(peak, len(_readers_alive()))
    assert peak <= cap, (peak, codes)
    assert codes == ["mentioned_file_owner_timeout"] * cap + ["mentioned_file_owner_busy"] * (
        cap * 2
    )
    gated_reads.gate.set()
    _drain(serving)
    assert _readers_alive() == [] and _version_dirs(serving.lane) == []
    assert _held_count(serving) == 0
    slots = serve_module._READERS
    taken = [slots.acquire(blocking=False) for _ in range(cap)]
    for _ in range(sum(taken)):
        slots.release()
    assert all(taken), "every reader slot came back"


def test_a_spent_budget_refuses_before_any_work(serving, monkeypatch, no_reads):
    """R2-3: a request that queued past its budget starts nothing."""
    serve_module = _serve_module()
    monkeypatch.setattr(
        serve_module, "slow_op_remaining_s", lambda: serve_module.SLOW_REPLY_MARGIN_S - 1
    )
    with pytest.raises(MeshRefusal) as refused:
        _call(serving)
    assert _code(refused) == "mentioned_file_owner_timeout"
    _drain(serving)
    assert no_reads == {"read_span": 0, "accessor": 0, "proof": 0}
    assert read_tests.rows(serving.lane) == [] and _version_dirs(serving.lane) == []


def test_concurrent_stats_of_one_row_share_one_version(serving, gated_reads):
    """R2-2: a second concurrent stat reuses the first's version, never evicts it."""
    results: dict[int, Any] = {}

    def stat(index):
        try:
            results[index] = _call(serving)
        except MeshRefusal as refusal:
            results[index] = refusal.code

    workers = [threading.Thread(target=stat, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + 10
    while len(_readers_alive()) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    # Unfixed, both reach the stalled read; fixed, the second waits its turn.
    deadline = time.monotonic() + 1
    while len({t.ident for t in gated_reads.entered}) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    gated_reads.gate.set()
    for worker in workers:
        worker.join(timeout=30)
    assert all(isinstance(results.get(i), dict) for i in range(2)), results
    assert results[0]["version"] == results[1]["version"]
    for index in range(2):
        fetch = _call(serving, phase="fetch", file_version=results[index]["version"], length=4)
        assert fetch["bytes"] == 4
    assert len(_version_dirs(serving.lane)) == 1 and _held_count(serving) == 1


def test_relogin_between_stat_and_fetch_is_named_relogin(serving):
    """Q1-2: the version carries the incarnation it was sealed under."""
    stat = _call(serving)
    rig = serving.lane.rig
    rig.auth._conn.execute(
        "UPDATE auth_credentials SET login_incarnation = 'relogged' WHERE id = ?", (rig.row.id,)
    )
    rig.auth._conn.commit()
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, phase="fetch", file_version=stat["version"], length=4)
    assert _code(refused) == "mentioned_file_owner_relogin"


def test_verify_accepts_net_file_get_prefix_bytes(serving):
    """R1-6: ``prefix_bytes`` is an alias of ``length``; two values are refused."""
    content = serving.lane.target.read_bytes()
    stat = _call(serving)
    good = "sha256:" + hashlib.sha256(content[:12]).hexdigest()
    frame = dict(phase="verify", file_version=stat["version"], length=None, prefix_digest=good)
    assert _call(serving, prefix_bytes=12, **frame) == {"match": True}
    with pytest.raises(MeshRefusal) as refused:
        _call(serving, prefix_bytes=12, **{**frame, "length": 11})
    assert _code(refused) == "mentioned_file_bad_request"


def test_the_op_names_its_owning_design_document():
    """R1-5: the default-off refusal names where P3 serving is specified."""
    from local_operator.network.relay import _owning_document

    owner = _owning_document("net_mentioned_file")
    assert owner.startswith("mesh-account-binding.md")
    assert (ROOT / "docs" / "design" / owner.split()[0]).is_file()

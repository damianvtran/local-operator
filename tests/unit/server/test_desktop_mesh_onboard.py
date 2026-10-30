"""Adapter security/ownership controls through registered app + actual loopback HTTP.

These are NOT the two-node protocol E2E: the one blocking-worker test replaces
only execution AFTER real signed-store admission, to discriminate HTTP ownership
from response lifetime. Bearer/Origin, strict models and file-only signatures are
real; file-only signing is not a Touch ID hardware assertion.
"""

from __future__ import annotations

import asyncio
import socket
import threading

import httpx
import pytest
import pytest_asyncio
import uvicorn

from local_operator.config import ConfigManager
from local_operator.network import approvals, identity, onboard, store
from local_operator.network.types import NetworkRecord
from local_operator.server.routes import desktop_approvals as routes
from tests.unit.server.test_desktop_approvals import DESKTOP_TOKEN, _file_device_record
from tests.unit.server.test_desktop_approvals import root as isolated_root

root = isolated_root


@pytest_asyncio.fixture
async def live(root, monkeypatch):
    from local_operator.paths import config_dir
    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    # Same ConfigManager constructor/root winner as real app.lifespan, without
    # starting unrelated schedulers/hub producers in a scoped adapter test.
    monkeypatch.setattr(
        app.state, "config_manager", ConfigManager(config_dir=config_dir()), raising=False
    )
    monkeypatch.setattr(app.state, "desktop_sessions", DesktopSessions(root), raising=False)
    monkeypatch.setattr(app.state, "mesh_onboard_tasks", {}, raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://desktop.fixture.test")
    started = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            app.state.fixture_loop = asyncio.get_running_loop()
            started.set()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = Server(uvicorn.Config(app, lifespan="off", log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    assert await asyncio.to_thread(started.wait, 10), "loopback HTTP did not start"
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            headers={
                "Authorization": f"Bearer {DESKTOP_TOKEN}",
                "Origin": "http://desktop.fixture.test",
            },
            timeout=10,
        ) as client:
            yield client, app, root
    finally:
        await _drain_server(app)
        server.should_exit = True
        await asyncio.to_thread(thread.join, 10)
        listener.close()
        assert not thread.is_alive(), "owned loopback HTTP did not stop"


async def _drain_server(app):
    # The real HTTP owner runs on uvicorn's loop, not pytest's waiter loop.
    future = asyncio.run_coroutine_threadsafe(routes.drain(app), app.state.fixture_loop)
    await asyncio.wrap_future(future)


@pytest.mark.asyncio
async def test_real_http_admission_and_strict_shape_precede_probe(live, monkeypatch):
    client, app, root = live
    calls = []
    monkeypatch.setattr(
        onboard,
        "probe",
        lambda *a, **kw: calls.append(kw)
        or onboard.Probe(ok=True, host="node.fixture.test", port=22, host_key_fp="SHA256:abc"),
    )
    path = "/v1/desktop/approvals/probe"
    # None of these requests may reach a public socket probe, let alone SSH.
    for headers in ({"Authorization": ""}, {"Origin": "https://foreign.fixture.test"}):
        r = await client.post(path, json={"host": "node.fixture.test"}, headers=headers)
        assert r.status_code in (401, 403), r.text
    r = await client.post(
        path, json={"host": "node.fixture.test", "credential_ref": "must-not-resolve"}
    )
    assert r.status_code == 422
    assert not calls
    r = await client.post(path, json={"host": "node.fixture.test:2200"})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["identity"] == "observed_tofu"
    assert r.json()["result"]["host"] == "node.fixture.test:2200"
    assert calls == [{"user": "", "port": 2200}]
    conflict = await client.post(path, json={"host": "node.fixture.test:2200", "port": 22})
    assert conflict.status_code == 422 and len(calls) == 1
    # The same server with a mismatched profile refuses before even the public
    # probe: no process-global env switch and no alternate-root setup writes.
    alternate = root / "other-profile"
    app.state.config_manager = ConfigManager(config_dir=alternate)
    calls.clear()
    before = sorted(p.relative_to(alternate) for p in alternate.rglob("*"))
    r = await client.post(path, json={"host": "node.fixture.test"})
    assert r.status_code == 409 and "onboard_root_mismatch" in r.text
    assert not calls
    body = {
        "host": "node.fixture.test",
        "request_id": "req_mismatch",
        "network_id": "n_none",
        "credential_ref": "fixture-reference",
        "host_key_fp": "SHA256:fixture",
    }
    refusal = await client.post("/v1/desktop/approvals/request", json=body)
    assert refusal.status_code == 409 and "onboard_root_mismatch" in refusal.text
    refusal = await client.post("/v1/desktop/approvals/ap_missing/run")
    assert refusal.status_code == 409 and "onboard_root_mismatch" in refusal.text
    assert not calls
    assert sorted(p.relative_to(alternate) for p in alternate.rglob("*")) == before


@pytest.mark.asyncio
async def test_request_same_builder_exact_network_and_no_reference_publication(live):
    from tests.unit.network.test_approvals_store import _make_key

    client, _app, root = live
    _make_key(root)
    who = identity.load_or_mint(root)
    for network in ("n_other", "n_selected"):
        store.save(
            NetworkRecord(
                network_id=network,
                name=network,
                created_by=who.device_id,
                self_device_id=who.device_id,
            ),
            root,
        )
    body = {
        "request_id": "req_same",
        "host": "node.fixture.test",
        "user": "operator",
        "network_id": "n_selected",
        "credential_ref": "secret-ref-fixture",
        "host_key_fp": "SHA256:abc",
        "advertise_host": "8.8.8.8:4097",
    }
    r = await client.post("/v1/desktop/approvals/request", json=body)
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["what"]["network_id"] == "n_selected"
    assert result["what"]["role"] == "drive" and result["what"]["grant"] == ["approve"]
    assert result["device"]["advertise_host"] == "8.8.8.8:4097"
    aid = result["approval_id"]
    again = await client.post("/v1/desktop/approvals/request", json=body)
    assert again.json()["result"]["approval_id"] == aid
    stored = approvals.load_record(aid, root=root)
    assert stored["credential_ref"]["ref"] == "secret-ref-fixture"
    assert stored["requested_by"]["surface"] == "desktop"
    for response in (r, await client.get(f"/v1/desktop/approvals/{aid}")):
        assert "secret-ref-fixture" not in response.text
        assert "credential_ref" not in response.text and "signature" not in response.text
    for field, value in (
        ("role", "admin"),
        ("session_id", "another-owner"),
        ("private_key", "plaintext"),
    ):
        response = await client.post("/v1/desktop/approvals/request", json={**body, field: value})
        assert response.status_code == 422
        if field == "private_key":
            assert "plaintext" not in response.text
    response = await client.post(
        "/v1/desktop/approvals/request", json={**body, "network_id": "n_unknown"}
    )
    assert response.status_code == 409
    assert len(approvals.list_records(root=root)) == 1


@pytest.mark.asyncio
async def test_owned_signed_admission_single_writer_and_cancellation(live, monkeypatch):
    client, app, root = live
    record = _file_device_record(root, with_anchor=True)
    aid = record["approval_id"]
    worker_entered = threading.Event()
    release = threading.Event()
    calls = []

    def worker(approval_id, *, view, root, release_target=None):
        calls.append(view.run_id)
        worker_entered.set()
        assert release.wait(10), "test did not release owned worker"
        from local_operator.network import onboard_approvals

        # The real runner frees the target just before its terminal row; this
        # fake speaks the same contract so the claim below is the real one.
        if release_target is not None:
            release_target()
        onboard_approvals.finish(
            approval_id,
            "connected",
            view.run_id,
            root=root,
            step="verify",
            detail="fixture worker complete (not protocol E2E)",
        )
        return {"ok": True}

    monkeypatch.setattr(onboard, "_execute_admitted_approval", worker)
    r = await client.post(f"/v1/desktop/approvals/{aid}/run")
    assert r.status_code == 409 and not calls  # unsigned cannot become accepted
    approved = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert approved.status_code == 200, approved.text
    approvals.verify_for_run(aid, root=root)  # real file-only signature
    try:
        r = await client.post(f"/v1/desktop/approvals/{aid}/run")
        assert r.status_code == 202, r.text
        assert await asyncio.to_thread(worker_entered.wait, 5)
        assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 409
        detail = (await client.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
        assert detail["run_active"] is True and detail["run_id"] == calls[0]
        # Releasing the HTTP response did not stop its thread/claim. Shutdown
        # drain must likewise wait rather than cancel it or admit a second writer.
        draining = asyncio.create_task(_drain_server(app))
        await asyncio.sleep(0)  # scheduling yield only; not a wall-clock oracle
        assert not draining.done()
    finally:
        release.set()
    await asyncio.wait_for(draining, 5)
    assert len(calls) == 1
    r = await client.get(f"/v1/desktop/approvals/{aid}")
    assert r.json()["result"]["state"] == "connected"
    assert r.json()["result"]["run_active"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crash", "expected_step", "expected_cause"),
    [
        # A crash inside a step: the receipt names THAT step.
        ("step", "join", "ValueError"),
        # Anything else keeps the historical fallback name, and still says what
        # kind of failure it was.
        ("bare", "worker", "RuntimeError"),
    ],
)
async def test_a_crashed_worker_receipt_names_the_step_and_the_error_class(
    live, monkeypatch, crash, expected_step, expected_cause
):
    """The catch-all used to record ``step="worker"`` and nothing else.

    ``worker`` is not one of ``onboard.STEP_NAMES`` and the sentence was fixed,
    so a run that died inside the join was indistinguishable - from every
    surface except the raw JSON store - from a credential-helper or store-write
    crash. The receipt now names the step and the exception CLASS, and it must
    never carry the exception's message: that is where paths, credentials and
    remote command lines live.
    """
    client, app, root = live
    record = _file_device_record(root, with_anchor=True)
    aid = record["approval_id"]
    secret = "top-secret-value /home/op/.ssh/lop.pem"

    def worker(approval_id, *, view, root, release_target=None):
        if crash == "step":
            raise onboard.OnboardStepCrash("join", ValueError(secret))
        # A crash that never came from a step: the fallback row must survive,
        # and its class is still the truth about what stopped the worker.
        raise RuntimeError(secret)

    monkeypatch.setattr(onboard, "_execute_admitted_approval", worker)
    assert (await client.post(f"/v1/desktop/approvals/{aid}/approve")).status_code == 200
    approvals.verify_for_run(aid, root=root)
    assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 202
    payload = await _settled(client, aid)
    row = payload["receipts"][-1]
    assert row["step"] == expected_step, row
    assert row["ok"] is False
    assert expected_cause in row["detail"] and expected_step in row["detail"], row
    assert secret not in row["detail"]
    # The durable record - not only the API view - is what a later reader sees.
    stored = approvals.load_record(aid, root=root)["receipts"][-1]
    assert stored["step"] == expected_step
    assert secret not in stored["detail"]


async def _settled(client, aid):
    """Poll the card until the run is no longer active (a load bound, not a claim)."""
    for _ in range(600):
        payload = (await client.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
        if not payload["run_active"]:
            return payload
        await asyncio.sleep(0.05)
    pytest.fail("the crashed worker never published a terminal row")


@pytest.mark.asyncio
async def test_a_finished_run_frees_the_ssh_target_before_any_retry(live, monkeypatch):
    """MAIN-RED SHAPE 2 at the seam a client sees: retry right after a failure.

    The runner frees the per-target claim BEFORE it publishes the terminal row
    (``onboard.TargetLease``), and the row is what ``run_active`` reports. So a
    retry posted once ``run_active`` is false must be admitted even while the
    previous worker is still finishing — it used to be refused 409
    ``approval_target_in_flight``, the ``assert 409 == 202`` the signed E2E went
    red on in CI.

    The fake below models only that order; the claim, the record, the admission
    and the refusal are the real ones.
    """
    client, app, root = live
    record = _file_device_record(root, with_anchor=True)
    aid = record["approval_id"]
    holds = threading.Event()
    terminal = threading.Event()

    def worker(approval_id, *, view, root, release_target=None):
        # The runner's own order: free the target, THEN publish the outcome.
        # A route that does not hand the claim over leaves it held here, which
        # is exactly what the retry below used to hit.
        if release_target is not None:
            release_target()
        from local_operator.network import onboard_approvals

        onboard_approvals.finish(
            approval_id,
            "failed",
            view.run_id,
            step="join",
            detail="the join was refused",
            root=root,
        )
        terminal.set()
        # Still holding the run open: the retry below must not depend on this
        # worker having returned, only on the claim being free.
        assert holds.wait(30), "test did not release the first worker"
        return {"ok": False}

    monkeypatch.setattr(onboard, "_execute_admitted_approval", worker)
    approved = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert approved.status_code == 200, approved.text
    approvals.verify_for_run(aid, root=root)  # real file-only signature
    try:
        assert (await client.post(f"/v1/desktop/approvals/{aid}/run")).status_code == 202
        assert await asyncio.to_thread(terminal.wait, 30)
        detail = (await client.get(f"/v1/desktop/approvals/{aid}")).json()["result"]
        assert detail["run_active"] is False
        retry = await client.post(f"/v1/desktop/approvals/{aid}/run")
        assert retry.status_code == 202, retry.text
    finally:
        holds.set()


@pytest.mark.asyncio
async def test_missing_card_is_not_accepted(live):
    client, _, _ = live
    assert (await client.post("/v1/desktop/approvals/ap_missing/run")).status_code == 404


@pytest.mark.asyncio
async def test_typed_enrollment_snapshot_is_required_before_same_signer(live, monkeypatch, caplog):
    from local_operator.network import approvals as a
    from tests.unit.network.test_enrollment_authority import create

    client, app, root = live
    record = create(root)
    aid = record["approval_id"]
    url = f"/v1/desktop/approvals/{aid}"
    ack = {
        "authority_version": 1,
        "scope": "network_only",
        "request_digest": record["request_digest"],
    }
    # Rejected secret-shaped input must never appear in response/log/card bytes.
    bad_request = {
        "host": "node.fixture.test",
        "request_id": "req_rejected_key_material",
        "network_id": "n_fixture",
        "host_key_fp": "SHA256:fixture",
    }
    for material in (
        "-----BEGIN PRIVATE KEY----- SENTINEL_PRIVATE_FIXTURE",
        "SENTINEL_PRIVATE_FIXTURE\nsecondline",
    ):
        response = await client.post(
            "/v1/desktop/approvals/request", json={**bad_request, "credential_ref": material}
        )
        assert response.status_code == 422 and "SENTINEL_PRIVATE_FIXTURE" not in response.text
    assert len(a.list_records(root=root)) == 1
    assert "SENTINEL_PRIVATE_FIXTURE" not in caplog.text
    original = a.sign_decision
    calls = []

    def signer(**kwargs):
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(a, "sign_decision", signer)
    for body in (
        None,
        {},
        ack,
        {"authority_version": True},
        {"private_key": "DO_NOT_ECHO_FIXTURE"},
    ):
        response = await client.post(url + "/approve", json=body)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "network_enrollment_confirmation_required"
        assert not calls and "DO_NOT_ECHO_FIXTURE" not in response.text
    for body in (
        {},
        {**ack, "authority_version": True},
        {**ack, "authority_version": "1"},
        {**ack, "authority_version": 2},
        {**ack, "scope": "all"},
        {**ack, "private_key": "DO_NOT_ECHO_FIXTURE"},
        {**ack, "request_digest": "bad"},
    ):
        response = await client.post(url + "/approve-network-enrollment", json=body)
        assert response.status_code == 422
        assert not calls and "DO_NOT_ECHO_FIXTURE" not in response.text
    stale = await client.post(
        url + "/approve-network-enrollment", json={**ack, "request_digest": "sha256:" + "0" * 64}
    )
    assert (
        stale.status_code == 409
        and stale.json()["detail"]["code"] == "network_enrollment_snapshot_changed"
    )
    assert not calls
    rows = (await client.get("/v1/desktop/approvals")).json()["result"]["approvals"]
    row = next(row for row in rows if row["approval_id"] == aid)
    assert row["can_approve_here"] is False
    assert row["typed_approval"] == {"action": "approve-network-enrollment", **ack}
    response = await client.post(url + "/approve-network-enrollment", json=ack)
    assert response.status_code == 200 and len(calls) == 1
    a.verify_for_run(aid, root=root)
    response = await client.post(url + "/approve-network-enrollment", json=ack)
    assert response.status_code == 409 and len(calls) == 1

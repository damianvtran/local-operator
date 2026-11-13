"""The S7 cloud transport: routes, headers, bodies, verdicts and disarm.

Driven through the REAL ``RadientTunnels.request`` (real auth store under the
autouse isolated config root, real bearer composition) with an
``httpx.MockTransport`` standing in for api.radienthq.com — the host itself is
not configurable, which is the point (``tunnels/api.py``). Bodies are compared
to the frozen fixtures in ``tests/fixtures/push/`` byte-for-byte as JSON: the
transport must send exactly what the worker composed, nothing added.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.mobile import push_devices
from local_operator.mobile.push_credentials import (
    CARRIER_EMIT,
    CARRIER_HEARTBEAT,
    CARRIER_REGISTER,
    REPORT_HEARTBEAT_ROUTE,
    REPORT_REGISTER_ROUTE,
)
from local_operator.mobile.push_payload import EMIT_ROUTE, IDEMPOTENCY_HEADER
from local_operator.mobile.push_transport import (
    PushLane,
    PushTransportDisarmed,
    RadientPushTransport,
    build_push_lane,
)
from local_operator.mobile.push_worker import (
    EMIT_RETRY_ATTEMPTS,
    EmitAccepted,
    EmitRefused,
)
from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import config as tunnel_config
from local_operator.tunnels.api import RadientTunnels
from local_operator.tunnels.config import DEFAULT_API_URL

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "push"
TUNNEL = "VZ8kQ1mT4pR7sW2xY6bN9c"
ACCESS = "push-test-access-token"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def login() -> int:
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct", "access": ACCESS}
        )
    return row.id


class Cloud:
    """A scripted api.radienthq.com: records every request, answers in order."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        self.answers = list(answers)
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def factory(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def _accepted() -> httpx.Response:
    return httpx.Response(
        202,
        json={"msg": "Push event accepted", "result": {"emit_id": "cloud-1", "accepted_at": 17}},
    )


def _transport(login: int, cloud: Cloud, **kw: Any) -> RadientPushTransport:
    async def no_sleep(_: float) -> None:
        return None

    return RadientPushTransport(
        tunnel_id=TUNNEL, credential_id=login, client_factory=cloud.factory, sleep=no_sleep, **kw
    )


@pytest.mark.parametrize(
    ("fixture", "key"),
    [
        ("emit-completion.json", "body"),
        ("emit-completion.json", "custom_body"),
        ("emit-attention.json", "body"),
        ("emit-digest.json", "body"),
    ],
)
def test_emit_posts_the_frozen_body_verbatim_to_the_frozen_route(
    login: int, fixture: str, key: str
) -> None:
    filed = _fixture(fixture)
    cloud = Cloud(_accepted())
    verdict = _transport(login, cloud).emit(filed[key], idempotency_key="attention-7")
    assert verdict == EmitAccepted(emit_id="cloud-1", accepted_at=17)
    (request,) = cloud.seen
    assert request.method == filed["method"] == "POST"
    assert str(request.url) == DEFAULT_API_URL + filed["route"].format(tunnel_id=TUNNEL)
    assert request.url.path == EMIT_ROUTE.format(tunnel_id=TUNNEL)
    assert request.headers[IDEMPOTENCY_HEADER] == "attention-7"
    assert set(filed["required_headers"]) == {IDEMPOTENCY_HEADER}
    assert request.headers["authorization"] == f"Bearer {ACCESS}"
    # Exactly the composed body: no field added, none dropped.
    assert json.loads(request.content) == filed[key]


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (httpx.Response(500, json={"error": "boom"}), 500),
        (httpx.Response(429, json={"error": "slow"}), 429),
        (httpx.Response(400, json={"error": "invalid_request"}), 400),
        (httpx.ConnectTimeout("timed out"), 0),
        # A 2xx without the API envelope is a broken cloud, not an accept.
        (httpx.Response(202, json={"emit_id": "x"}), 0),
    ],
)
def test_emit_refusals_carry_the_status_and_never_retry_in_the_transport(
    login: int, answer: Any, status: int
) -> None:
    cloud = Cloud(answer)
    transport = _transport(login, cloud)
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(status)
    assert len(cloud.seen) == 1  # the worker owns the retry, not the transport
    assert not transport.disarmed


def test_a_404_disarms_and_sends_nothing_more(login: int, caplog: pytest.LogCaptureFixture) -> None:
    cloud = Cloud(httpx.Response(404, json={"error": "tunnel_not_found"}))
    transport = _transport(login, cloud)
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(404)
    assert transport.disarmed
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(0)
    with pytest.raises(PushTransportDisarmed):
        transport.credentials({"computer": TUNNEL, "devices": []}, carrier=CARRIER_HEARTBEAT)
    assert len(cloud.seen) == 1
    assert sum("disarmed" in r.message for r in caplog.records) == 1


def test_a_missing_login_disarms(login: int) -> None:
    cloud = Cloud(_accepted())
    transport = RadientPushTransport(
        tunnel_id=TUNNEL, credential_id=login + 999, client_factory=cloud.factory
    )
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(0)
    assert transport.disarmed and cloud.seen == []


def test_credentials_posts_only_the_heartbeat_carrier(login: int) -> None:
    block = {
        "computer": TUNNEL,
        "devices": [{"device_id": "d1", "credential_live": True}],
    }
    cloud = Cloud(httpx.Response(200, json={"updated": 1}))
    transport = _transport(login, cloud)
    transport.credentials(block, carrier=CARRIER_EMIT)
    transport.credentials(block, carrier=CARRIER_REGISTER)
    assert cloud.seen == []
    transport.credentials(block, carrier=CARRIER_HEARTBEAT)
    (request,) = cloud.seen
    assert str(request.url) == DEFAULT_API_URL + REPORT_HEARTBEAT_ROUTE
    assert json.loads(request.content) == block
    assert IDEMPOTENCY_HEADER not in request.headers


def test_a_failed_heartbeat_raises_so_the_report_keeps_its_facts(login: int) -> None:
    cloud = Cloud(httpx.Response(503))
    with pytest.raises(ValueError, match="HTTP 503"):
        _transport(login, cloud).credentials(
            {"computer": TUNNEL, "devices": []}, carrier=CARRIER_HEARTBEAT
        )


def test_register_forward_body_and_bounded_retry(login: int) -> None:
    device = {
        "device_id": "d1",
        "platform": "ios",
        "environment": "production",
        "app_version": "1.0.0 (449)",
        "push_token": "ab" * 32,
        "registered_at": 1700000000,
    }
    rows = [{"device_id": "d1", "credential_live": True}]
    cloud = Cloud(httpx.Response(502), httpx.Response(500), httpx.Response(200, json={}))
    asyncio.run(_transport(login, cloud).register_forward(device, rows))
    assert len(cloud.seen) == EMIT_RETRY_ATTEMPTS
    for request in cloud.seen:
        assert str(request.url) == DEFAULT_API_URL + REPORT_REGISTER_ROUTE
        assert json.loads(request.content) == {
            "computer": TUNNEL,
            "device": device,
            "devices": rows,
        }


def test_register_forward_stops_on_a_permanent_refusal(login: int) -> None:
    cloud = Cloud(httpx.Response(400, json={"error": "invalid_request"}))
    asyncio.run(_transport(login, cloud).register_forward({"device_id": "d"}, []))
    assert len(cloud.seen) == 1


def test_api_path_cannot_repoint_the_bearer(login: int) -> None:
    cloud = Cloud(_accepted())

    async def go() -> None:
        async with cloud.factory() as client:
            await RadientTunnels(login, client).request(
                "POST", body={}, api_path="@evil.example/v1/x"
            )

    with pytest.raises(ValueError, match="/v1/"):
        asyncio.run(go())
    assert cloud.seen == []


def test_no_tunnel_no_lane(tmp_path: Path) -> None:
    assert build_push_lane(tmp_path) is None


@pytest.mark.parametrize(
    ("value", "armed"),
    [
        ({"tunnel_id": TUNNEL, "credential_id": 3, "record": {"id": TUNNEL}}, True),
        ({"tunnel_id": TUNNEL, "credential_id": 3, "record": {"revoked_at": "2026"}}, False),
        ({"tunnel_id": TUNNEL}, False),
        ({"credential_id": 3}, False),
    ],
)
def test_the_lane_arms_only_on_a_live_tunnel(
    tmp_path: Path, value: dict[str, Any], armed: bool
) -> None:
    tunnel_config.save(value)
    lane = build_push_lane(tmp_path)
    assert (lane is not None) is armed
    if lane is not None:
        assert lane.worker.computer == lane.report.computer == TUNNEL


def test_the_heartbeat_is_spaced_after_a_failure(login: int, tmp_path: Path) -> None:
    cloud = Cloud(httpx.Response(503), httpx.Response(200, json={"updated": 0}))
    push_devices.register(
        tmp_path,
        {
            "platform": "ios",
            "token": "ab" * 32,
            "environment": "production",
            "app_version": "1.0.0 (449)",
            "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
        },
    )
    lane = PushLane(_transport(login, cloud), tmp_path)
    assert lane.heartbeat(now=1000.0) is None
    assert len(cloud.seen) == 1
    assert lane.heartbeat(now=1010.0) is None  # inside the retry spacing: no call
    assert len(cloud.seen) == 1
    sent = lane.heartbeat(now=1100.0)
    assert len(cloud.seen) == 2
    assert sent is not None and json.loads(cloud.seen[1].content) == sent
    # The machine never kept the token it forwarded nothing of here.
    assert "ab" * 32 not in (tmp_path / push_devices.PUSH_DEVICES_STORE_NAME).read_text()


def test_the_register_route_forwards_the_token_and_still_stores_none(login: int) -> None:
    """The real route: local register, then the scheduled cloud forward.

    The response does not wait on the cloud (SESSION-PUSH §4.5), so the forward
    is observed on the stub afterwards; the registry file must never carry the
    token it forwarded.
    """
    import time

    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.paths import config_dir

    cloud = Cloud(httpx.Response(200, json={"device_id": "d", "registered_at": 1}))
    lane = PushLane(_transport(login, cloud), config_dir())
    token = "cd" * 32
    with TestClient(
        build_app(MobileDaemon(port=0, password="pw123", push_lane=lane)), follow_redirects=False
    ) as client:
        assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
        answer = client.post(
            "/api/push/register",
            json={
                "platform": "ios",
                "token": token,
                "environment": "production",
                "app_version": "1.0.0 (449)",
                "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
            },
        )
        assert answer.status_code == 200, answer.text
        deadline = time.monotonic() + 10
        while not cloud.seen and time.monotonic() < deadline:
            time.sleep(0.05)
    (request,) = cloud.seen
    assert str(request.url) == DEFAULT_API_URL + REPORT_REGISTER_ROUTE
    sent = json.loads(request.content)
    local = answer.json()
    assert sent["computer"] == TUNNEL
    assert sent["device"] == {
        "device_id": local["device_id"],
        "platform": "ios",
        "environment": "production",
        "app_version": "1.0.0 (449)",
        "push_token": token,
        "registered_at": local["registered_at"],
    }
    assert set(sent) == {"computer", "device", "devices"}
    stored = (config_dir() / push_devices.PUSH_DEVICES_STORE_NAME).read_text()
    assert token not in stored


def test_a_refused_register_is_not_forwarded(login: int) -> None:
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.paths import config_dir

    cloud = Cloud(httpx.Response(200, json={}))
    lane = PushLane(_transport(login, cloud), config_dir())
    with TestClient(
        build_app(MobileDaemon(port=0, password="pw123", push_lane=lane)), follow_redirects=False
    ) as client:
        client.post("/login", data={"password": "pw123"})
        answer = client.post("/api/push/register", json={"platform": "ios"})
        assert answer.status_code >= 400
    assert cloud.seen == []
    assert lane.worker is not None


def _register_one(config: Path) -> None:
    push_devices.register(
        config,
        {
            "platform": "ios",
            "token": "ab" * 32,
            "environment": "production",
            "app_version": "1.0.0 (449)",
            "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
        },
    )


def test_register_rows_never_waits_on_an_in_flight_heartbeat(login: int, tmp_path: Path) -> None:
    """R1 + R4 (review round 1 on #3875): the heartbeat's HTTP call is outside
    ``PushLane.lock``, and a second beat skips instead of queueing.

    The stub cloud parks the heartbeat request on an event, standing in for a
    slow or blackholed network. While it is parked, ``register_rows`` (the
    register route's read) must return promptly and a second ``heartbeat`` must
    return without a request; once released, the beat commits the block it
    actually sent.
    """
    import threading

    in_flight = threading.Event()
    release = threading.Event()

    def cloud(request: httpx.Request) -> httpx.Response:
        in_flight.set()
        assert release.wait(10), "test never released the parked heartbeat"
        return httpx.Response(200, json={"updated": 1})

    seen: list[httpx.Request] = []

    def factory() -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return cloud(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    _register_one(tmp_path)
    lane = PushLane(
        RadientPushTransport(tunnel_id=TUNNEL, credential_id=login, client_factory=factory),
        tmp_path,
    )
    result: dict[str, Any] = {}
    beat = threading.Thread(target=lambda: result.update(sent=lane.heartbeat(now=1000.0)))
    beat.start()
    try:
        assert in_flight.wait(10), "heartbeat never reached the cloud"
        rows: list[Any] = []
        reader = threading.Thread(target=lambda: rows.append(lane.register_rows()))
        reader.start()
        reader.join(2)
        assert not reader.is_alive(), "register_rows waited on the heartbeat's network call"
        # The beat has not committed, so the device is still unreported and
        # rides the register carrier: the read completed, with real facts.
        ((row,),) = rows
        assert row["device_id"] and "credential_live" in row
        assert lane.heartbeat(now=1000.0) is None  # R4: skipped, not queued
        assert len(seen) == 1
    finally:
        release.set()
        beat.join(10)
    assert not beat.is_alive()
    assert result["sent"] is not None and json.loads(seen[0].content) == result["sent"]


def test_a_heartbeat_overtaken_by_a_change_does_not_commit(login: int, tmp_path: Path) -> None:
    """The commit half of R1's split: a block the registry moved past while it
    was on the wire is not recorded as reported, so the next beat re-sends."""
    _register_one(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        # A second device registers while the beat is on the wire.
        push_devices.register(
            tmp_path,
            {
                "platform": "ios",
                "token": "cd" * 32,
                "environment": "production",
                "app_version": "1.0.0 (449)",
                "install_id": "1f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
            },
        )
        return httpx.Response(200, json={"updated": 1})

    lane = PushLane(
        RadientPushTransport(
            tunnel_id=TUNNEL,
            credential_id=login,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
        tmp_path,
    )
    sent = lane.heartbeat(now=1000.0)
    assert sent is not None and len(sent["devices"]) == 1  # one device went out...
    assert lane.report.heartbeat_due(now=1001.0)  # ...but the moved report stays due


@pytest.mark.parametrize("carrier", ["emit", "heartbeat", "register"])
def test_a_409_tunnel_revoked_disarms(login: int, carrier: str) -> None:
    """R3: agent-server's only push 409 is ``tunnel_revoked`` — terminal."""
    cloud = Cloud(httpx.Response(409, json={"error": "tunnel_revoked"}))
    transport = _transport(login, cloud)
    if carrier == "emit":
        assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(409)
    elif carrier == "heartbeat":
        with pytest.raises(ValueError, match="HTTP 409"):
            transport.credentials({"computer": TUNNEL, "devices": []}, carrier=CARRIER_HEARTBEAT)
    else:
        asyncio.run(transport.register_forward({"device_id": "d"}, []))
    assert transport.disarmed
    assert len(cloud.seen) == 1  # no retry spent on a certainty
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(0)
    assert len(cloud.seen) == 1


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("key_after", "advance", "rearmed", "credential"),
    [
        # Same live config: stays dark until the 1 h backoff, then one probe.
        ((TUNNEL, 1), 600.0, False, 1),
        ((TUNNEL, 1), 3600.0, True, 1),
        # A re-login rewrote the tunnel's credential: re-arms on the next check.
        ((TUNNEL, 2), 600.0, True, 2),
        # Revoked/absent, or a different tunnel: never re-armed in place.
        (None, 7200.0, False, 1),
        (("other-tunnel", 2), 7200.0, False, 1),
    ],
)
def test_a_disarmed_lane_re_arms_on_a_config_change_or_backoff(
    login: int,
    key_after: tuple[str, int] | None,
    advance: float,
    rearmed: bool,
    credential: int,
) -> None:
    """R2: the disarm is no longer until-restart."""
    from local_operator.mobile.push_transport import REARM_AFTER_S

    assert REARM_AFTER_S == 3600.0
    clock = _Clock()
    key: list[tuple[str, int] | None] = [(TUNNEL, 1)]
    cloud = Cloud(httpx.Response(404, json={"error": "tunnel_not_found"}))
    transport = RadientPushTransport(
        tunnel_id=TUNNEL,
        credential_id=1,
        client_factory=cloud.factory,
        tunnel_key=lambda: key[0],
        clock=clock,
    )
    transport._disarm(RuntimeError("x"))
    assert transport.disarmed
    key[0] = key_after
    clock.now += advance
    assert transport.disarmed is (not rearmed)
    assert transport.credential_id == credential


def test_a_re_login_before_the_failing_call_re_arms_on_the_next_check(login: int) -> None:
    """Round 2 M1: the disarm key is the credential the call USED.

    The config already names the new login when the old one's call fails; the
    disarm must still see that as a change and re-arm on the next spaced check,
    not after the 1 h backoff.
    """
    clock = _Clock()
    reads: list[int] = []

    def key() -> tuple[str, int]:
        reads.append(1)
        return (TUNNEL, login + 1)  # re-login already rewrote the config

    cloud = Cloud(httpx.Response(404, json={"error": "tunnel_not_found"}))
    transport = _transport(login, cloud, tunnel_key=key, clock=clock)
    assert transport.emit({"v": 1}, idempotency_key="k") == EmitRefused(404)
    assert reads == []  # the disarm itself read no config
    assert transport.disarmed  # inside the retry spacing: no check yet
    clock.now += 60.0
    assert not transport.disarmed
    assert transport.credential_id == login + 1
    assert len(reads) == 1


def test_the_register_forward_never_reads_the_tunnel_config_on_the_loop(login: int) -> None:
    """Round 2 M2: schedule + forward + disarm on the event loop do no file I/O."""
    reads: list[int] = []

    def key() -> tuple[str, int]:
        reads.append(1)
        return (TUNNEL, login)

    cloud = Cloud(httpx.Response(404, json={"error": "tunnel_not_found"}))
    clock = _Clock()
    transport = _transport(login, cloud, tunnel_key=key, clock=clock)

    async def go() -> None:
        task = transport.schedule_register_forward({"device_id": "d"}, [])
        assert task is not None
        await task  # the 404 disarms on the loop
        clock.now += 7200.0  # a re-arm would be due, yet the loop path must not check
        assert transport.schedule_register_forward({"device_id": "d"}, []) is None

    asyncio.run(go())
    assert len(cloud.seen) == 1
    assert reads == []
    # The scan-thread check is where the re-arm happens.
    assert not transport.disarmed
    assert reads == [1]


def test_the_scan_arms_the_lane_when_a_tunnel_appears_after_start(
    login: int, tmp_path: Path
) -> None:
    """Late tunnel: ``lop tunnel create`` after start no longer needs a restart."""
    from local_operator.mobile.daemon import MobileDaemon

    builds: list[int] = []

    def factory() -> PushLane | None:
        builds.append(1)
        return build_push_lane(tmp_path)

    daemon = MobileDaemon(port=0, password="pw123", push_lane_factory=factory)
    assert daemon.push_lane is None

    async def go() -> None:
        await daemon._maybe_arm_push_lane()  # no tunnel yet
        assert daemon.push_lane is None and len(builds) == 1
        tunnel_config.save({"tunnel_id": TUNNEL, "credential_id": login, "record": {"id": TUNNEL}})
        await daemon._maybe_arm_push_lane()  # inside the spacing: not retried
        assert daemon.push_lane is None and len(builds) == 1
        daemon._push_lane_tried_at = -1e9  # the spacing elapsed
        await daemon._maybe_arm_push_lane()
        assert daemon.push_lane is not None and len(builds) == 2
        assert daemon.push_worker is daemon.push_lane.worker
        assert daemon.push_lane.transport.computer == TUNNEL
        daemon._push_lane_tried_at = -1e9
        await daemon._maybe_arm_push_lane()  # armed: never rebuilt
        assert len(builds) == 2

    asyncio.run(go())


def test_no_factory_means_the_scan_never_builds_a_lane() -> None:
    from local_operator.mobile.daemon import MobileDaemon

    daemon = MobileDaemon(port=0, password="pw123")
    asyncio.run(daemon._maybe_arm_push_lane())
    assert daemon.push_lane is None and daemon.push_worker is None

"""Lost-guest health snapshots never substitute diagnostics for action outcomes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lop_osworld_v2_adapter import guest_loss
from lop_osworld_v2_adapter.guest_disk import CommandResult
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.aws import GuestExecutionError

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (
    CREDS,
    REGION,
)
from tests.unit.evaluation.adapters.osworld.test_guest_loss import (
    _closed_port,
    _CountingTransport,
    _NoConsoleEc2,
    _provider_over,
    _Stubs,
)

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401  isort:skip
    _hermetic_aws,
)


class Ec2:
    def __init__(self, state: str = "running") -> None:
        self.state = state
        self.calls = 0

    def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "Reservations": [
                {
                    "Instances": [
                        {"InstanceId": kwargs["InstanceIds"][0], "State": {"Name": self.state}}
                    ]
                }
            ]
        }


def test_running_guest_records_numeric_health_without_task_content(tmp_path: Path) -> None:
    calls: list[float] = []

    def probe(command: list[str], timeout: float) -> CommandResult:
        calls.append(timeout)
        assert command == ["python3", "-c", guest_loss.HEALTH_PROBE_SOURCE]
        return CommandResult(
            0,
            json.dumps(
                {
                    "disk_free_bytes": 0,
                    "MemAvailable": 32,
                    "action_server_alive": True,
                    "password": "must-not-be-persisted",
                    "SwapFree": "invalid",
                }
            ),
            "",
        )

    ec2 = Ec2()
    record = guest_loss.capture_guest_health(ec2, "i-synthetic", tmp_path, probe)
    assert record["cause"] == "instance running; guest disk full"
    assert record["guest"] == {
        "disk_free_bytes": 0,
        "MemAvailable": 32,
        "action_server_alive": True,
    }
    assert ec2.calls == len(calls) == 1
    assert 0 < calls[0] <= 4
    evidence = (tmp_path / "guest-health-execute.json").read_text()
    assert json.loads(evidence) == record
    assert "must-not-be-persisted" not in evidence


def test_terminated_guest_is_distinct_and_never_probed(tmp_path: Path) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("terminated guest must not be contacted")

    record = guest_loss.capture_guest_health(Ec2("terminated"), "i-synthetic", tmp_path, forbidden)
    assert record["cause"] == "instance lost (terminated)"
    assert json.loads((tmp_path / "guest-health-execute.json").read_text()) == record


def test_diagnostics_transport_failure_is_recorded_without_exception_text(tmp_path: Path) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("private task data")

    record = guest_loss.capture_guest_health(Ec2(), "i-synthetic", tmp_path, broken)
    assert record["guest_probe"] == "error"
    # A bare RuntimeError is not evidence the guest was unreachable: the request
    # may have been delivered. "unreachable" is reserved for a connect-class
    # verdict, which this exception cannot produce.
    assert record["cause"] == (
        "instance running; guest probe outcome unknown (the request may have been delivered)"
    )
    assert "unreachable" not in record["cause"]
    assert "private task data" not in (tmp_path / "guest-health-execute.json").read_text()


def test_hung_cloud_probe_is_abandoned_without_contacting_guest(tmp_path: Path) -> None:
    release = threading.Event()
    entered = threading.Event()

    def hung(**kwargs: Any) -> None:
        entered.set()
        release.wait(10)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unknown instance state must not trigger a guest command")

    try:
        record = guest_loss.capture_guest_health(
            SimpleNamespace(describe_instances=hung),
            "i-synthetic",
            tmp_path,
            forbidden,
            budget_s=0.05,
        )
        assert entered.is_set()
        assert record["instance_probe"] == "timeout"
        assert record["cause"] == "readiness check failed"
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostics_raises", [False, True])
async def test_real_execute_retains_delivery_outcome_and_records_instance_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, diagnostics_raises: bool
) -> None:
    """Real refused HTTP connection; diagnostics must not change action delivery.

    A probe answering 503 is EVIDENCE that the guest is down, not a reason to
    skip the bounded retry: the attempts are spent, and the failure still
    reports the same ``not_sent`` verdict with the same instance-loss cause.
    """
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    transport = _CountingTransport()
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=transport)
        ec2 = _NoConsoleEc2()
        ec2.describe_instances = Ec2("terminated").describe_instances  # type: ignore[attr-defined]
        provider._clients.ec2 = ec2
        provider._clients.http_get = lambda *args: 503
        if diagnostics_raises:

            def broken(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("private task content")

            monkeypatch.setattr(guest_loss, "capture_guest_health", broken)
        with pytest.raises(GuestExecutionError) as caught:
            await provider.execute(["first()", "second()"])
    assert caught.value.delivery == "not_sent"
    assert transport.calls == aws_mod.TRANSPORT_RETRY_ATTEMPTS
    assert "no input of this batch was applied" in str(caught.value)
    assert "private task content" not in str(caught.value)
    if diagnostics_raises:
        assert "guest down: readiness check failed" in str(caught.value)
    else:
        assert "guest down: instance lost (terminated)" in str(caught.value)
        assert provider._cache_root is not None
        record = json.loads((provider._cache_root / "guest-health-execute.json").read_text())
        assert record["instance_state"] == "terminated"


@pytest.mark.asyncio
async def test_the_health_evidence_names_the_answer_not_an_unreachable_guest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A REAL 503 listener answers both the readiness probes and the health POST.

    This is the branch whose cause reaches the terminal message and
    ``guest-health-execute.json``. It used to write "instance running; guest
    unreachable" for EVERY non-ok probe -- including this one -- and the message
    test could not catch it because its EC2 stub had no ``describe_instances``,
    so the capture failed before the clause was built. Here the stub answers the
    instance read and the server answers with a status, so the clause is
    actually rendered and the assertion can fail.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(503)
            self.end_headers()

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(503)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", server.server_address[1])
    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path)
            provider._supervisor_unit = "action-server.service"
            real_clients = aws_mod.build_clients(CREDS, REGION)
            provider._clients.http_get = real_clients.http_get
            provider._clients.http_post_json = real_clients.http_post_json
            ec2 = _NoConsoleEc2()
            ec2.describe_instances = Ec2("running").describe_instances  # type: ignore[attr-defined]
            provider._clients.ec2 = ec2
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute(["first()"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    message = str(raised.value)
    assert "guest probe answered, not ready (status=503)" in message, message
    assert "unreachable" not in message, message
    assert provider._cache_root is not None
    record = json.loads((provider._cache_root / "guest-health-execute.json").read_text())
    assert record["instance_state"] == "running"
    assert record["guest_probe"] == "error", "the probe branch must actually run"
    assert record["cause"] == "instance running; guest probe answered, not ready (status=503)"
    assert "unreachable" not in record["cause"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (CommandResult(1, "private task content", "private error"), "command_failed"),
        (CommandResult(0, "not json", ""), "ok"),
    ],
)
def test_invalid_probe_still_persists_partial_evidence(
    tmp_path: Path, result: CommandResult, expected: str
) -> None:
    record = guest_loss.capture_guest_health(
        Ec2(), "i-synthetic", tmp_path, lambda *args, **kwargs: result
    )
    assert record["instance_state"] == "running"
    assert record["guest_probe"] == expected
    persisted = (tmp_path / "guest-health-execute.json").read_text()
    assert json.loads(persisted) == record
    assert "private" not in persisted
    assert "not json" not in persisted

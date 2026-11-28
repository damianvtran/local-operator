"""``/v1/desktop/guard``'s person-only gate on a REAL ``lop serve`` start (security S-1, L2c).

WHY A SEPARATE FILE. ``test_guard_desktop_routes`` sets the agent-shell marker in the test
process itself, which is a state a served daemon never reaches: ``cli.serve_command`` strips
the markers from its own environment at startup (``drop_agent_shell_marker_from_environ``), so
a per-request ``surface.host_can_answer()`` read was ALWAYS true on a served daemon and the
``guard_answer_not_admitted`` refusal and ``can_answer: false`` hint could never fire. These
rows start the daemon the way ``lop serve`` does -- through ``serve_command`` itself, with the
environment an agent's shell gives it -- and then drive the routes and the lifespan.

``uvicorn.Server.run`` is the only thing replaced (so nothing blocks); the environment, the
marker drop, the capture and the app are the product's.
"""

# Imported pytest fixtures are re-declared as test parameters, which flake8 reads as F811.
# flake8: noqa: F811
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_operator.cli import serve_command
from local_operator.guard import consent as C
from local_operator.guard import surface
from local_operator.server.app import app as asgi_app
from tests.unit.guard.test_guard_consent import TENANT, _offer
from tests.unit.guard.test_guard_consent import (
    fixture_server as _consent_fixture_server,
)
from tests.unit.guard.test_guard_desktop_routes import (  # noqa: F401 - autouse + fixtures
    BASE,
    _a_person_backend,
    _body,
    _code,
    _host,
    _http,
    _team,
    root,
)
from tests.unit.test_cli_serve import captured_server  # noqa: F401 - fixture
from tests.unit.test_cli_serve import (  # noqa: F401 - autouse fixture
    clean_app_announcement,
)

fixture_server = _consent_fixture_server

MARKERS = {"LOCAL_OPERATOR_AGENT_SHELL": "1", "LOCAL_OPERATOR_ALLOW_NESTED_SESSION": "1"}


@pytest.fixture(autouse=True)
def _forget_the_capture() -> Iterator[None]:
    """``serve_command`` writes process-global state; leave none for the next test."""

    surface.reset_serve_capture()
    os.environ.pop(surface.HOST_CAN_ANSWER_ENV, None)
    yield
    surface.reset_serve_capture()
    os.environ.pop(surface.HOST_CAN_ANSWER_ENV, None)


def _serve(captured_server: dict[str, Any]) -> None:
    assert serve_command("127.0.0.1", 0, False) == 0
    for sock in captured_server["sockets"]:
        sock.close()


def _declined(root: Path, fixture_server: Any) -> C.GuardConsentHost:
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    return host


@pytest.mark.asyncio
async def test_a_daemon_started_in_an_agent_shell_refuses_share_again_after_the_marker_drop(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,
    captured_server: dict[str, Any],
) -> None:
    _team(monkeypatch)
    host = _declined(root, fixture_server)
    for name, value in MARKERS.items():
        monkeypatch.setenv(name, value)

    _serve(captured_server)

    # The premise of S-1, measured: the daemon's own environment is clean after start, so a
    # live read says "a person" -- which is exactly why the old per-request read was wrong.
    assert not any(name in os.environ for name in MARKERS)
    assert surface.host_can_answer() is True
    assert surface.served_host_can_answer() is False
    async with _http(root, monkeypatch) as client:
        status = _body(await client.get(BASE))
        refused = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert status["can_answer"] is False and status["orgs"][0]["can_share_again"] is False
    assert refused.status_code == 409 and _code(refused) == "guard_answer_not_admitted"
    assert host.sticky(TENANT) == "declined"  # the sticky "no" stands
    assert fixture_server.guard_requests() == []


@pytest.mark.asyncio
async def test_a_daemon_a_person_started_still_answers_share_again(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,
    captured_server: dict[str, Any],
) -> None:
    """NEGATIVE CONTROL: the same start without the markers keeps the route open."""

    _team(monkeypatch)
    host = _declined(root, fixture_server)

    _serve(captured_server)

    assert surface.served_host_can_answer() is True
    assert surface.HOST_CAN_ANSWER_ENV not in os.environ
    async with _http(root, monkeypatch) as client:
        status = _body(await client.get(BASE))
        control = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert status["can_answer"] is True
    assert control.status_code == 200 and host.sticky(TENANT) is None


@pytest.mark.asyncio
async def test_a_console_pane_start_is_still_refused_after_a_serve_start(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,
    captured_server: dict[str, Any],
) -> None:
    """``LOCAL_OPERATOR_CONSOLE_SURFACE`` survives the marker drop; the capture keeps it."""

    _team(monkeypatch)
    host = _declined(root, fixture_server)
    monkeypatch.setenv("LOCAL_OPERATOR_CONSOLE_SURFACE", "1")

    _serve(captured_server)

    async with _http(root, monkeypatch) as client:
        refused = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert _code(refused) == "guard_answer_not_admitted" and host.sticky(TENANT) == "declined"


def test_a_child_of_a_refused_daemon_inherits_the_refusal_and_cannot_widen_it(
    monkeypatch: pytest.MonkeyPatch, captured_server: dict[str, Any]
) -> None:
    """A ``--reload`` child or re-exec reads the ALREADY-CLEANED environment of its parent."""

    for name, value in MARKERS.items():
        monkeypatch.setenv(name, value)
    _serve(captured_server)
    assert os.environ[surface.HOST_CAN_ANSWER_ENV] == "0"
    # The child: no in-process capture, a clean environment, only the inherited flag.
    surface.reset_serve_capture()
    assert surface.host_can_answer() is True
    assert surface.served_host_can_answer() is False
    # Only ``"0"`` ever refuses: an agent setting ``1`` gains nothing, a missing flag decides
    # from the live environment.
    os.environ[surface.HOST_CAN_ANSWER_ENV] = "1"
    assert surface.served_host_can_answer() is True
    os.environ.pop(surface.HOST_CAN_ANSWER_ENV)
    assert surface.served_host_can_answer() is True


def test_the_lifespan_publishes_the_capture_on_app_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_server: dict[str, Any]
) -> None:
    """The real lifespan, after a real ``serve_command``: the routes read THIS value."""

    from local_operator.aida import activation

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    monkeypatch.setattr(activation, "_has_terminal", lambda: False)
    for name, value in MARKERS.items():
        monkeypatch.setenv(name, value)
    _serve(captured_server)
    with TestClient(asgi_app, headers={"Host": "127.0.0.1:1111"}):
        assert asgi_app.state.guard_host_can_answer is False
        # R2-2: the notice allow-list exists before the first request, as an empty set.
        assert asgi_app.state.guard_notices_listed == set()

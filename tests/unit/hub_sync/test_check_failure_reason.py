"""A failed org-team check must store WHY it failed (radientdev, 2026-10-28).

The operator's org team link stored ``hub-error`` with the sentence "could not
reach the hub: Could not pull the team from the Radient Agent Hub" while the
same ``GET /v1/teams/:id`` (bearer, org tenant) answered 200 on every later read.
That sentence is ``RadientClient.get_team``'s fallback for a request that got NO
response; the transport cause lived only in ``__cause__`` and was dropped, so
nothing could tell a blip from a broken link. These tests drive the REAL client
over loopback HTTP against a hub that answers in the real hub's shapes (read
off api.radienthq.com: ``{"msg", "result"}`` on success, ``{"code", "error"}``
on refusal) and assert what the check stores.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import link_origin
from tests.unit.hub_sync.test_link_origin import SOURCE, TENANT, _team

HUB_ID = "1f48dc2e-1a0a-452a-af32-629a29b06bcc"


@contextmanager
def _hub(mode: str) -> Iterator[tuple[str, list[tuple[str, bool]]]]:
    seen: list[tuple[str, bool]] = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            seen.append((self.path, self.headers.get("Authorization") is not None))
            if mode == "stall":
                # Hold the response past the client's read timeout.
                release.wait(5)
                return
            if mode == "500":
                self._send(500, {"code": "internal_error", "error": "Team lookup failed."})
                return
            if mode == "404":
                self._send(404, {"code": "team_not_found", "error": "Team not found."})
                return
            self._send(
                200,
                {
                    "msg": "Team retrieved successfully",
                    "result": {
                        "id": HUB_ID,
                        "tenant_id": TENANT,
                        "name": "crew",
                        "description": "d",
                        "manager": "manager",
                        "members": [],
                        "instructions": "i",
                        "project": "",
                    },
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def _check(tmp_path: Any, base_url: str) -> chk.ItemCheck:
    team = _team("local-1")
    prov.record_team_pull(
        tmp_path,
        team,
        {"id": HUB_ID},
        tenant_id=TENANT,
        origin=link_origin(SOURCE, "org", TENANT),
    )
    registry = SimpleNamespace(
        config_dir=tmp_path, list_teams=lambda: [team], get_team=lambda _: team
    )
    client = RadientClient(api_key=SecretStr("sk-test-org-bearer"), base_url=base_url)
    (result,) = chk.check_hub_teams(registry, client_for_tenant=lambda _: client)
    return result


def test_a_timed_out_org_team_fetch_says_it_timed_out(tmp_path, monkeypatch):
    # The production bound is 20 s; a short one keeps the stall test fast.
    monkeypatch.setattr(chk, "REQUEST_TIMEOUT_S", 0.3)
    with _hub("stall") as (base_url, seen):
        started = time.monotonic()
        result = _check(tmp_path, base_url)
        elapsed = time.monotonic() - started

    assert seen == [(f"/v1/teams/{HUB_ID}", True)]
    assert result.verdict == "unavailable" and result.reason == "hub-error"
    # Before: "could not reach the hub: Could not pull the team from the Radient
    # Agent Hub" -- the cause (a read timeout) was nowhere in it.
    assert result.detail.startswith("could not reach the hub: the hub did not answer in time")
    assert "sk-test-org-bearer" not in result.detail
    assert elapsed < 3


def test_a_failing_org_team_fetch_names_the_status_and_hub_message(tmp_path):
    with _hub("500") as (base_url, _seen):
        result = _check(tmp_path, base_url)

    assert result.reason == "hub-error"
    assert result.detail == "the hub returned an error (HTTP 500: Team lookup failed.)"


def test_a_missing_org_team_stays_hub_item_missing(tmp_path):
    with _hub("404") as (base_url, _seen):
        result = _check(tmp_path, base_url)

    assert result.reason == "hub-item-missing"


def test_a_reachable_org_team_is_checked_normally(tmp_path):
    with _hub("200") as (base_url, seen):
        result = _check(tmp_path, base_url)

    assert seen == [(f"/v1/teams/{HUB_ID}", True)]
    assert result.verdict == "up-to-date", result


@pytest.mark.parametrize(
    ("cause", "phrase"),
    [
        ("ConnectTimeout", "the connection timed out"),
        ("ReadTimeout", "the hub did not answer in time"),
        ("SSLError", "the secure connection failed"),
        ("ConnectionError", "the connection failed"),
    ],
)
def test_the_transport_cause_is_read_from_the_chain(cause, phrase):
    import requests

    from local_operator.clients._http import APIError

    error = APIError("Could not pull the team from the Radient Agent Hub", status_code=None)
    error.__cause__ = getattr(requests.exceptions, cause)("pool internals")
    reason, detail = chk._failure_reason(error)

    assert reason == "hub-error"
    assert detail == (
        f"could not reach the hub: {phrase} (Could not pull the team from the Radient Agent Hub)"
    )


def _api_error(status: int | None, message: str, cause: BaseException | None = None):
    from local_operator.clients._http import APIError

    error = APIError(message, status_code=status)
    error.__cause__ = cause
    return error


def test_the_most_specific_cause_anywhere_in_the_chain_wins():
    # Round-1 R1: the old per-exception loop returned at the FIRST exception that
    # matched ANY row, so a shallow ConnectionError hid the deeper read timeout.
    import requests
    import urllib3

    deepest = urllib3.exceptions.ReadTimeoutError(
        urllib3.HTTPConnectionPool("127.0.0.1"), "/v1/teams", "read timed out"
    )
    middle = requests.exceptions.ConnectionError("pool internals")
    middle.__cause__ = deepest
    error = _api_error(None, "Could not pull the team from the Radient Agent Hub", middle)

    reason, detail = chk._failure_reason(error)

    assert reason == "hub-error"
    assert detail == (
        "could not reach the hub: the hub did not answer in time "
        "(Could not pull the team from the Radient Agent Hub)"
    )


@pytest.mark.parametrize("cause", [TimeoutError("timed out"), __import__("socket").timeout()])
def test_a_raw_socket_timeout_is_named_not_left_generic(cause):
    # A client that lets the builtin escape matched no row and stored only the
    # generic fallback sentence -- the "why is missing" failure this module exists
    # to prevent. ``socket.timeout`` is ``TimeoutError`` on 3.10+.
    error = _api_error(None, "Could not pull the team from the Radient Agent Hub", cause)

    _reason, detail = chk._failure_reason(error)

    assert detail == (
        "could not reach the hub: the request timed out "
        "(Could not pull the team from the Radient Agent Hub)"
    )


def test_control_characters_in_a_hub_body_never_reach_the_report():
    # The text lands in ``ItemMergeReport.message``, read in a terminal: an ANSI
    # escape, CR (overwrites the line), LF or NUL must not survive.
    error = _api_error(500, "\x1b[31mboom\rfake\nline\x00end\x7f")

    _reason, detail = chk._failure_reason(error)

    assert not set(detail) & {"\x1b", "\x00", "\r", "\n", "\x7f"}
    assert "boom" in detail and "end" in detail


def test_a_control_character_becomes_a_space_not_a_deletion():
    # Round-2 R2-2: a ``None`` translation DELETED the character, so
    # "Team\nlookup" read "Teamlookup".
    _reason, detail = chk._failure_reason(_api_error(500, "line one\nline two\tend"))

    assert detail == "the hub returned an error (HTTP 500: line one line two end)"


def test_a_4xx_is_still_a_refusal():
    # A 5xx is the hub failing, not refusing; a 4xx must keep the refusal wording.
    _reason, detail = chk._failure_reason(_api_error(401, "Invalid API key."))

    assert detail == "the hub refused the request (HTTP 401: Invalid API key.)"


def test_a_5xx_html_body_does_not_repeat_the_status():
    error = _api_error(502, "Could not pull the team from the Radient Agent Hub (HTTP 502)")

    _reason, detail = chk._failure_reason(error)

    assert detail == (
        "the hub returned an error (Could not pull the team from the Radient Agent Hub (HTTP 502))"
    )


def test_a_bool_status_is_not_an_http_status():
    error = _api_error(True, "something odd")  # type: ignore[arg-type]

    _reason, detail = chk._failure_reason(error)

    assert "HTTP True" not in detail
    assert detail == "could not reach the hub: something odd"


def test_the_warning_names_the_item_that_failed(caplog):
    import requests

    error = _api_error(
        None,
        "Could not pull the team from the Radient Agent Hub",
        requests.exceptions.ConnectionError("pool internals"),
    )

    with caplog.at_level("WARNING", logger=chk.logger.name):
        chk._failure_reason(error, kind="team", hub_id=HUB_ID)

    (line,) = [r.getMessage() for r in caplog.records]
    assert "team" in line and HUB_ID in line
    assert "pool internals" not in line


def test_a_hub_supplied_id_cannot_forge_a_log_line(caplog):
    # Round-2 R2-3: the team arm's id comes from the hub document, not a UUID
    # this side minted; a newline or escape in it must not reach the log.
    import requests

    error = _api_error(None, "x", requests.exceptions.ConnectionError("pool internals"))

    with caplog.at_level("WARNING", logger=chk.logger.name):
        chk._failure_reason(error, kind="team", hub_id="x\nFAKE LOG\x1b[2J" + "y" * 300)

    (line,) = [r.getMessage() for r in caplog.records]
    assert "\n" not in line and "\x1b" not in line
    assert "team x FAKE LOG [2J" in line
    assert "y" * 200 not in line


def _closed_port() -> int:
    # Bind then close: nothing listens on the port, so a connect is refused.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def _no_proxy_env(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("_no_proxy_env")
def test_a_refused_connection_says_the_connection_failed(tmp_path):
    # Round-2 R2-1: urllib3's own ``TimeoutError`` is the base of
    # ``NewConnectionError``; matched by name it made a refused port read "the
    # request timed out". Real socket, real client, no mocked class tree.
    result = _check(tmp_path, f"http://127.0.0.1:{_closed_port()}")

    assert result.reason == "hub-error"
    assert result.detail.startswith("could not reach the hub: the connection failed (")


@pytest.mark.usefixtures("_no_proxy_env")
def test_a_dead_proxy_says_the_proxy_refused():
    import requests

    dead = f"http://127.0.0.1:{_closed_port()}"
    with pytest.raises(requests.exceptions.RequestException) as caught:
        # ``.invalid`` never resolves; the proxy is dialled first and refuses.
        requests.get("http://hub.invalid/v1/teams", proxies={"http": dead}, timeout=5)
    error = _api_error(None, "Could not pull the team from the Radient Agent Hub", caught.value)

    _reason, detail = chk._failure_reason(error)

    assert detail.startswith("could not reach the hub: the proxy refused the connection (")

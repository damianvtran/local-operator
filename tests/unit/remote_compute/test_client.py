"""The remote-compute client against a fake origin: contract, errors, no leaks.

ISOLATION. Every test writes its own credential store and its own mesh record
under ``tmp_path``, with ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` redirected,
so nothing here can read the operator's live login or networks (AGENTS.md,
"Isolating a run").

NO REAL SOCKETS. Requests travel through ``httpx.MockTransport``, and the
autouse ``no_real_network`` fixture makes the real transports raise if anything
ever escapes the mock — a test that quietly called Radient would otherwise
"pass" by reaching the internet from a developer machine.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.network import store
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.providers.auth_store import AuthStore, CredentialInvalidError
from local_operator.remote_compute import RunCreateRequest, RunSpec, resolve_network
from local_operator.remote_compute.client import (
    KNOWN_RUN_STATUSES,
    MAX_IDEMPOTENCY_KEY_BYTES,
    RUNS_PATH,
    TERMINAL_RUN_STATUSES,
    RemoteComputeClient,
    charged_amount,
    check_idempotency_key,
    is_terminal,
    new_idempotency_key,
    run_exit_code,
    run_id,
    run_max_hours,
    run_network,
    run_outcome,
    run_status,
)
from local_operator.remote_compute.errors import (
    CapacityUnavailable,
    IdempotencyConflict,
    InsufficientCredit,
    InvalidIdempotencyKey,
    InvalidRequest,
    NetworkMismatch,
    RemoteComputeError,
    RunIdentityMismatch,
    RunNotFound,
    Unreachable,
)
from local_operator.remote_compute.request import is_commit_sha, is_repo_url
from local_operator.tunnels.config import DEFAULT_API_URL
from local_operator.tunnels.errors import LoginRequired


class _Origin:
    """A fake Radient run endpoint that records what it was asked."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail loudly if a request leaves the mock transport."""
    escaped: list[str] = []

    def refuse(self: Any, request: httpx.Request) -> httpx.Response:
        escaped.append(f"{request.method} {request.url}")
        raise AssertionError(
            f"a test attempted a real network call: {request.method} {request.url}"
        )

    async def arefuse(self: Any, request: httpx.Request) -> httpx.Response:
        return refuse(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse, raising=True)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", arefuse, raising=True)
    yield escaped
    assert escaped == [], f"tests made real network calls: {escaped}"


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


@pytest.fixture()
def credential() -> int:
    with closing(AuthStore()) as store_:
        row = store_.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct", "access": "token-abc"}
        )
        return row.id


@pytest.fixture()
def mesh_root(tmp_path: Path) -> Path:
    return tmp_path / "mesh"


def _save_network(mesh_root: Path, name: str = "home") -> NetworkRecord:
    record = NetworkRecord(
        network_id=store.new_network_id(),
        name=name,
        self_device_id="d_self",
        members=[MemberRecord(device_id="d_self", name="this device")],
    )
    store.save(record, mesh_root)
    return record


@pytest.fixture()
def request_for(mesh_root: Path) -> Any:
    record = _save_network(mesh_root)
    binding = resolve_network("home", mesh_root)

    def build(**overrides: Any) -> RunCreateRequest:
        fields: dict[str, Any] = {
            "prompt": "fix the flaky test",
            "repo_url": "https://example.invalid/repo.git",
            "repo_sha": SHA,
            "hosting": "openai",
            "model": "gpt-5",
        }
        fields.update(overrides)
        return RunSpec(**fields).bind(binding)

    build.record = record  # type: ignore[attr-defined]
    return build


def _client(origin: _Origin, credential: int) -> tuple[RemoteComputeClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=origin.transport)
    return RemoteComputeClient(http, credential_id=credential), http


#: A full commit SHA: the contract's ``repo_sha`` is a commit, not a ref.
SHA = "0123456789abcdef0123456789abcdef01234567"


def _run(
    run_id_value: str = "run_1",
    status: str = "pending",
    *,
    network_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A run as the server returns it; ``network_id`` is omitted unless given."""
    payload: dict[str, Any] = {"id": run_id_value, "status": status, **extra}
    if network_id is not None:
        payload["network_id"] = network_id
    return payload


def _created(run: dict[str, Any], *, msg: str = "Run created") -> dict[str, Any]:
    """A create body: the envelope around the run's own wrapper."""
    return {"msg": msg, "result": {"run": run}}


def _read(run: dict[str, Any], *, msg: str = "Run") -> dict[str, Any]:
    """A per-run read body: the envelope around the run itself."""
    return {"msg": msg, "result": run}


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sends_the_contract_fields_and_the_tenant(
    credential: int, request_for: Any
) -> None:
    expected = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json=_created(expected)))
    client, http = _client(origin, credential)
    async with http:
        created = await client.create_run(request_for(), idempotency_key="key-1")
    assert created.run == expected
    assert created.replayed is False
    (sent,) = origin.requests
    assert sent.method == "POST"
    assert str(sent.url) == DEFAULT_API_URL + RUNS_PATH
    assert sent.headers["authorization"] == "Bearer token-abc"
    assert sent.headers["idempotency-key"] == "key-1"
    body = json.loads(sent.content)
    assert body["network_id"] == request_for.record.network_id
    assert body["prompt"] == "fix the flaky test"
    assert body["repo_url"] == "https://example.invalid/repo.git"
    assert body["repo_sha"] == SHA
    assert (body["hosting"], body["model"]) == ("openai", "gpt-5")
    assert (body["size"], body["max_hours"]) == ("m", 2.0)
    # Optional fields are OMITTED, not sent as null.
    assert "budget_usd" not in body and "name" not in body


@pytest.mark.asyncio
async def test_a_create_without_the_run_wrapper_is_refused(
    credential: int, request_for: Any
) -> None:
    """A create's result is a WRAPPER, so a bare run there is not a run at all."""
    bare = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json={"msg": "Run created", "result": bare}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RemoteComputeError):
            await client.create_run(request_for(), idempotency_key="k")


@pytest.mark.asyncio
async def test_a_replay_is_the_same_run_and_says_so(credential: int, request_for: Any) -> None:
    """201 created the run; 200 replayed an earlier submission of the same key."""
    run = _run("run_7", "running", network_id=request_for.record.network_id)
    origin = _Origin(
        httpx.Response(201, json=_created(run)),
        httpx.Response(200, json=_created(run, msg="Run already submitted")),
    )
    client, http = _client(origin, credential)
    async with http:
        first = await client.create_run(request_for(), idempotency_key="same")
        second = await client.create_run(request_for(), idempotency_key="same")
    assert (first.run, first.replayed) == (run, False)
    assert (second.run, second.replayed) == (run, True)
    assert origin.requests[0].headers["idempotency-key"] == "same"
    assert origin.requests[1].headers["idempotency-key"] == "same"


@pytest.mark.asyncio
async def test_a_run_from_another_network_is_refused(credential: int, request_for: Any) -> None:
    """SE1 — the tenant is compared on the NESTED run, not on the wrapper."""
    other = "n_" + "b" * 24
    origin = _Origin(httpx.Response(201, json=_created(_run(network_id=other))))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.create_run(request_for(), idempotency_key="k")


@pytest.mark.asyncio
async def test_a_run_without_a_network_id_is_a_mismatch(credential: int, request_for: Any) -> None:
    """The field is in the contract, so its absence is a response we cannot verify."""
    origin = _Origin(httpx.Response(201, json=_created(_run())))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.create_run(request_for(), idempotency_key="k")


@pytest.mark.asyncio
async def test_a_poll_from_another_network_stops_a_watch(credential: int, request_for: Any) -> None:
    """The watch names the network it follows, so a drifted poll is a failure."""
    origin = _Origin(
        httpx.Response(200, json=_read(_run("run_1", "running", network_id="n_" + "c" * 24)))
    )
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.get_run("run_1", expected_network=request_for.record.network_id)


@pytest.mark.asyncio
async def test_a_poll_without_an_expected_network_is_not_checked(credential: int) -> None:
    """``--remote-status`` names no network, so it takes the run as returned."""
    run = _run("run_1", "running")
    origin = _Origin(httpx.Response(200, json=_read(run)))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_1") == run


@pytest.mark.asyncio
async def test_a_run_carrying_its_own_result_field_is_not_unwrapped(credential: int) -> None:
    """E1 — only an ENVELOPE is unwrapped, not any object with a ``result`` key."""
    bare = _run("run_1", "running", result="a run field of its own")
    origin = _Origin(httpx.Response(200, json={"msg": "Run", "result": bare}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_1") == bare


@pytest.mark.asyncio
async def test_the_real_envelope_is_unwrapped(credential: int, request_for: Any) -> None:
    """The contract's envelope is ``{"msg", "result"}`` — never ``message``."""
    run = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json={"msg": "Run created", "result": {"run": run}}))
    client, http = _client(origin, credential)
    async with http:
        assert (await client.create_run(request_for(), idempotency_key="k")).run == run


@pytest.mark.asyncio
async def test_an_error_word_beside_the_result_still_reads_as_an_envelope(
    credential: int,
) -> None:
    """``error`` is part of the envelope's key set, so it does not stop the unwrap."""
    run = _run("run_1", "running")
    origin = _Origin(httpx.Response(200, json={"error": None, "result": run}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_1") == run


#: Sentinels that must survive nowhere but the pod. A create's wrapper carries them.
RUN_TOKEN = "zq-run-1"
CLOUDFLARED_TOKEN = "zq-tunnel-2"
ORIGIN_AUTH = "zq-origin-3"


@pytest.mark.asyncio
async def test_the_create_never_hands_back_the_pod_secrets(
    credential: int, request_for: Any
) -> None:
    """``run_token`` and ``tunnel`` are returned once, for the pod, and dropped here."""
    run = _run(network_id=request_for.record.network_id)
    wrapper: dict[str, Any] = {
        "run": run,
        "run_token": RUN_TOKEN,
        "tunnel": {
            "tunnel_id": "tun_1",
            "hostname": "run-1.example.invalid",
            "gateway_port": 8443,
            "cloudflared_token": CLOUDFLARED_TOKEN,
            "origin_auth": ORIGIN_AUTH,
        },
    }
    origin = _Origin(httpx.Response(201, json={"msg": "Run created", "result": wrapper}))
    client, http = _client(origin, credential)
    async with http:
        created = await client.create_run(request_for(), idempotency_key="k")
    rendered = repr(created) + str(created) + repr(created.run)
    for sentinel in (RUN_TOKEN, CLOUDFLARED_TOKEN, ORIGIN_AUTH):
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_a_reused_key_with_a_different_body_is_a_conflict(
    credential: int, request_for: Any
) -> None:
    origin = _Origin(httpx.Response(409, json={"error": "idempotency key conflict"}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(IdempotencyConflict):
            await client.create_run(request_for(), idempotency_key="same")


@pytest.mark.asyncio
async def test_get_run_reads_the_run_by_id(credential: int) -> None:
    origin = _Origin(httpx.Response(200, json=_read(_run("run_2", "running"))))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_2") == _run("run_2", "running")
    assert str(origin.requests[0].url) == DEFAULT_API_URL + RUNS_PATH + "/run_2"


@pytest.mark.asyncio
async def test_a_read_answering_for_another_run_is_refused(credential: int) -> None:
    """A body naming another run is not this run's state, whatever it says."""
    origin = _Origin(httpx.Response(200, json=_read(_run("run_other", "running"))))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RunIdentityMismatch):
            await client.get_run("run_1")


@pytest.mark.asyncio
async def test_a_cancel_answering_for_another_run_is_refused(credential: int) -> None:
    """A cancel that reported another run would look applied to work nobody stopped."""
    origin = _Origin(httpx.Response(200, json=_read(_run("run_other", "stopping"))))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RunIdentityMismatch):
            await client.cancel_run("run_1")


@pytest.mark.asyncio
async def test_a_run_id_cannot_escape_its_path_segment(credential: int) -> None:
    origin = _Origin(httpx.Response(200, json=_read(_run("../secrets", "running"))))
    client, http = _client(origin, credential)
    async with http:
        await client.get_run("../secrets")
    url = origin.requests[0].url
    # The id is QUOTED into ONE segment: a slash in it can never create a new
    # path segment, so a run id cannot address a different endpoint. Read the
    # RAW path — ``url.path`` decodes escapes, which would hide exactly this.
    raw = url.raw_path.decode()
    assert raw.endswith("/..%2Fsecrets")
    assert raw.count("/") == RUNS_PATH.count("/") + 1


@pytest.mark.asyncio
async def test_cancel_posts_to_the_cancel_path_and_is_idempotent(credential: int) -> None:
    stopping = _run("run_3", "stopping")
    origin = _Origin(
        httpx.Response(202, json=_read(stopping)),
        httpx.Response(202, json=_read(stopping)),
    )
    client, http = _client(origin, credential)
    async with http:
        assert await client.cancel_run("run_3") == stopping
        assert await client.cancel_run("run_3") == stopping
    assert str(origin.requests[0].url) == DEFAULT_API_URL + RUNS_PATH + "/run_3/cancel"
    assert origin.requests[0].method == "POST"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(204), httpx.Response(200, content=b"")])
async def test_a_bodiless_cancel_success_is_a_success(
    credential: int, response: httpx.Response
) -> None:
    """E4 — the endpoint documents no body, so its absence is not malformed."""
    origin = _Origin(response)
    client, http = _client(origin, credential)
    async with http:
        assert await client.cancel_run("run_3") == {}


@pytest.mark.asyncio
async def test_list_runs_sends_paging_and_the_optional_network(credential: int) -> None:
    page = {"runs": [_run("run_1", "running")], "limit": 5, "offset": 10}
    origin = _Origin(httpx.Response(200, json={"msg": "Runs", "result": page}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.list_runs(network_id="n_abc", limit=5, offset=10) == [
            _run("run_1", "running")
        ]
    query = origin.requests[0].url.params
    assert (query["limit"], query["offset"], query["network_id"]) == ("5", "10", "n_abc")


@pytest.mark.asyncio
async def test_list_runs_returns_the_runs_not_the_paging(credential: int) -> None:
    """``result`` is ``{runs, limit, offset}``: only the runs are the answer."""
    page = {"runs": [], "limit": 5, "offset": 0}
    origin = _Origin(httpx.Response(200, json={"msg": "Runs", "result": page}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.list_runs(limit=5) == []


@pytest.mark.asyncio
async def test_a_page_without_a_runs_list_is_refused(credential: int) -> None:
    origin = _Origin(httpx.Response(200, json={"msg": "Runs", "result": []}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RemoteComputeError):
            await client.list_runs()


@pytest.mark.asyncio
async def test_list_runs_omits_an_absent_network(credential: int) -> None:
    page = {"runs": [], "limit": 5, "offset": 0}
    origin = _Origin(httpx.Response(200, json={"msg": "Runs", "result": page}))
    client, http = _client(origin, credential)
    async with http:
        await client.list_runs(limit=5)
    assert "network_id" not in origin.requests[0].url.params


@pytest.mark.asyncio
async def test_an_empty_network_id_is_refused_before_it_is_sent(credential: int) -> None:
    origin = _Origin()
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(ValueError):
            await client.list_runs(network_id="")
    assert origin.requests == []


# ---------------------------------------------------------------------------
# Typed errors, and the redaction rule
# ---------------------------------------------------------------------------

SECRET_BODY = "sk-live-SECRET-account-42-balance-9.99"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequest),
        (401, LoginRequired),
        (402, InsufficientCredit),
        (404, RunNotFound),
        (409, IdempotencyConflict),
        (503, CapacityUnavailable),
        (500, RemoteComputeError),
        (418, RemoteComputeError),
    ],
)
@pytest.mark.asyncio
async def test_each_status_maps_to_its_typed_error_without_echoing_the_body(
    credential: int, request_for: Any, status: int, expected: type[Exception]
) -> None:
    origin = _Origin(httpx.Response(status, json={"error": SECRET_BODY}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(expected) as caught:
            await client.create_run(request_for(), idempotency_key="k")
    assert SECRET_BODY not in str(caught.value)
    assert str(status) in str(caught.value)


@pytest.mark.asyncio
async def test_an_unparseable_success_body_is_a_typed_error(credential: int) -> None:
    """A JSON decode error keeps the document; the raised error must not."""
    body = b'not json at all {"echo": "Bearer token-abc"}'
    origin = _Origin(httpx.Response(200, content=body))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RemoteComputeError) as caught:
            await client.get_run("run_1")
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "Bearer" not in _residual_text(caught.value)
    assert "not json" not in _residual_text(caught.value)


@pytest.mark.asyncio
async def test_a_transport_failure_is_unreachable_without_a_chain(credential: int) -> None:
    """SE2 — the httpx error is NOT chained: it carries the bearer header."""

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    async with http:
        with pytest.raises(Unreachable) as caught:
            await RemoteComputeClient(http, credential_id=credential).get_run("run_1")
    assert caught.value.__cause__ is None
    # The context is GONE, not merely suppressed: the error is constructed
    # outside the handler, so the httpx failure is not reachable at all. Its
    # request carries the Authorization header, and a kept `__context__` is
    # exactly how a bearer ends up one traceback away from a log.
    assert caught.value.__context__ is None
    assert "Bearer" not in _residual_text(caught.value)
    assert "token-abc" not in _residual_text(caught.value)
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert "Bearer" not in rendered and "token-abc" not in rendered


@pytest.mark.asyncio
async def test_a_revoked_grant_surfaces_login_required(
    credential: int, request_for: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(self: AuthStore, credential_id: int, *, force: bool = False) -> Any:
        raise CredentialInvalidError("grant revoked")

    monkeypatch.setattr(AuthStore, "ensure_oauth_fresh_or_raise", refuse)
    origin = _Origin()
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(LoginRequired):
            await client.create_run(request_for(), idempotency_key="k")
    assert origin.requests == []


@pytest.mark.asyncio
async def test_another_store_failure_is_a_typed_error_with_its_cause(
    credential: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.auth_store import AuthStoreError

    async def refuse(self: AuthStore, credential_id: int, *, force: bool = False) -> Any:
        raise AuthStoreError("token endpoint unreachable")

    monkeypatch.setattr(AuthStore, "ensure_oauth_fresh_or_raise", refuse)
    origin = _Origin()
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RemoteComputeError) as caught:
            await client.get_run("run_1")
    assert not isinstance(caught.value, LoginRequired)
    assert isinstance(caught.value.__cause__, AuthStoreError)
    assert origin.requests == []


@pytest.mark.asyncio
async def test_a_missing_credential_row_surfaces_login_required() -> None:
    origin = _Origin()
    client, http = _client(origin, 999)
    async with http:
        with pytest.raises(LoginRequired) as caught:
            await client.get_run("run_1")
    assert "lop login radient" in str(caught.value)
    assert origin.requests == []


# ---------------------------------------------------------------------------
# The request boundary and the key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_refuses_anything_but_a_run_create_request(credential: int) -> None:
    origin = _Origin()
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(TypeError):
            await client.create_run(
                {"network_id": "n_" + "a" * 24},  # type: ignore[arg-type]
                idempotency_key="k",
            )
    assert origin.requests == []


def test_the_key_must_be_one_to_128_bytes() -> None:
    assert check_idempotency_key("a") == "a"
    assert check_idempotency_key("x" * MAX_IDEMPOTENCY_KEY_BYTES)


def test_an_empty_key_is_refused_with_its_own_reason() -> None:
    """E2 — refused, never silently replaced by a fresh key."""
    with pytest.raises(InvalidIdempotencyKey) as caught:
        check_idempotency_key("")
    assert caught.value.reason == "empty"


def test_an_oversize_key_names_the_byte_bound() -> None:
    with pytest.raises(InvalidIdempotencyKey) as caught:
        check_idempotency_key("x" * (MAX_IDEMPOTENCY_KEY_BYTES + 1))
    assert (caught.value.reason, caught.value.max_bytes) == ("too_long", MAX_IDEMPOTENCY_KEY_BYTES)


@pytest.mark.parametrize("key", ["k\n", "k\t", "ké", "\x7f"])
def test_a_key_outside_printable_ascii_is_refused(key: str) -> None:
    """E3 — 0x20..0x7e only, and BYTES: the two rules are separate."""
    with pytest.raises(InvalidIdempotencyKey) as caught:
        check_idempotency_key(key)
    assert caught.value.reason == "not_printable"


def test_a_key_at_the_top_of_the_printable_range_is_accepted() -> None:
    assert check_idempotency_key("~ !" + "a" * 10)
    with pytest.raises(ValueError):
        check_idempotency_key("x" * (MAX_IDEMPOTENCY_KEY_BYTES + 1))
    # BYTES, not characters: 100 non-ASCII characters are over the limit.
    with pytest.raises(ValueError):
        check_idempotency_key("\u00e9" * 100)


def test_a_minted_key_is_a_uuid4_string() -> None:
    key = new_idempotency_key()
    assert len(key) == 36 and key.count("-") == 4


def test_a_key_is_refused_by_the_client_before_the_request(
    credential: int, request_for: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _Origin()
    client, http = _client(origin, credential)

    async def run() -> None:
        async with http:
            with pytest.raises(ValueError):
                await client.create_run(request_for(), idempotency_key="")

    asyncio.run(run())
    assert origin.requests == []


# ---------------------------------------------------------------------------
# Run helpers the CLI reads
# ---------------------------------------------------------------------------


def test_run_status_and_terminality() -> None:
    assert run_status({"status": "running"}) == "running"
    assert run_id({"id": "run_1"}) == "run_1"
    assert run_network({"network_id": "n_" + "a" * 24}) == "n_" + "a" * 24
    with pytest.raises(NetworkMismatch):
        run_network({"id": "run_1"})
    # TERMINAL is about the STATUS, not about how the run went: a stopped or
    # deleted run is over, and its OUTCOME says which way.
    assert is_terminal({"status": "stopped"})
    assert is_terminal({"status": "deleted"})
    assert not is_terminal({"status": "running"})
    assert not is_terminal({"status": "stopping"})
    # ``error`` is NOT terminal: the server retries it, so a watch that stopped
    # on it would report a run as finished that is about to run again.
    assert not is_terminal({"status": "error"})
    # An unknown status is NOT terminal either: a run whose state this build
    # cannot name is still running as far as anyone here can tell.
    assert not is_terminal({"status": "some_new_state"})
    assert TERMINAL_RUN_STATUSES == {"stopped", "deleted"}
    assert KNOWN_RUN_STATUSES == {
        "pending",
        "reconciling",
        "running",
        "stopping",
        "stopped",
        "revoking",
        "deleted",
        "error",
    }
    with pytest.raises(RemoteComputeError):
        run_status({"id": "run_1"})
    with pytest.raises(RemoteComputeError):
        run_id({"status": "pending"})


def test_the_outcome_is_read_as_the_server_spelled_it() -> None:
    assert run_outcome({"status": "stopped", "outcome": "succeeded"}) == "succeeded"
    assert run_outcome({"status": "stopped", "outcome": "expired"}) == "expired"
    # Absent is ``unknown``: a word the finished line can print and the exit
    # code can map to failure without inventing a success.
    assert run_outcome({"status": "stopped"}) == "unknown"
    assert run_outcome({"outcome": ""}) == "unknown"
    # A value outside the four is passed through, never mapped to ``failed``:
    # that would claim a result the server did not report.
    assert run_outcome({"outcome": "aborted_by_operator"}) == "aborted_by_operator"


def test_an_exit_code_is_an_integer_and_never_a_bool() -> None:
    assert run_exit_code({"exit_code": 0}) == 0
    assert run_exit_code({"exit_code": 137}) == 137
    # None and 0 are different answers: a run with no exit code did not exit 0.
    assert run_exit_code({}) is None
    assert run_exit_code({"exit_code": None}) is None
    assert run_exit_code({"exit_code": True}) is None
    assert run_exit_code({"exit_code": "1"}) is None


def test_the_charge_is_quoted_and_never_computed() -> None:
    assert charged_amount({"charged_usd": 0.42}) == "0.42"
    assert charged_amount({"charged_usd": "0.42"}) == "0.42"
    # The settled amount wins over the breakdown.
    assert charged_amount({"charged_usd": 0.42, "cost": {"compute_usd": 0.4}}) == "0.42"
    # No settled amount: the breakdown is shown as the server sent it.
    breakdown = {"tunnel_usd": 0.02, "compute_usd": 0.4}
    assert charged_amount({"cost": breakdown}) == '{"compute_usd": 0.4, "tunnel_usd": 0.02}'
    assert charged_amount({}) is None
    assert charged_amount({"cost": {}}) is None
    assert charged_amount({"charged_usd": True}) is None


def test_the_watch_bound_is_the_servers_max_hours() -> None:
    assert run_max_hours({"max_hours": 2}) == 2.0
    # The server may cap what was asked for, so the value can be anything.
    assert run_max_hours({"max_hours": 0.5}) == 0.5
    assert run_max_hours({}) is None
    assert run_max_hours({"max_hours": None}) is None
    assert run_max_hours({"max_hours": True}) is None
    assert run_max_hours({"max_hours": "2"}) is None


def _residual_text(error: BaseException) -> str:
    """Everything a printer or a log call could pull out of ``error``.

    One level of attributes, as the review asked: the sentence, its args, its
    ``__dict__`` and each value in it. A kept ``__context__``/``__cause__`` is
    checked separately by the caller, because that is where a provider body or
    a bearer would hide.
    """
    parts = [str(error), repr(error), repr(error.args), repr(vars(error))]
    for value in vars(error).values():
        parts.append(repr(value))
        parts.append(str(value))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The typed run body
# ---------------------------------------------------------------------------


def _fields(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "prompt": "fix the flaky test",
        "repo_url": "https://example.invalid/repo.git",
        "repo_sha": SHA,
        "hosting": "openai",
        "model": "gpt-5",
    }
    values.update(overrides)
    return values


def test_a_spec_carries_the_contract_fields_and_omits_unset_optionals() -> None:
    body = RunSpec(**_fields()).to_body()
    assert body == {
        "prompt": "fix the flaky test",
        "repo_url": "https://example.invalid/repo.git",
        "repo_sha": SHA,
        "hosting": "openai",
        "model": "gpt-5",
        "size": "m",
        "max_hours": 2.0,
    }
    assert "budget_usd" not in body and "network_id" not in body


def test_a_spec_with_optionals_sends_them() -> None:
    body = RunSpec(**_fields(size="l", max_hours=3, budget_usd=0, name="audit")).to_body()
    assert (body["size"], body["max_hours"], body["budget_usd"], body["name"]) == (
        "l",
        3,
        0,
        "audit",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"size": "xl"},
        {"max_hours": 0},
        {"max_hours": -1},
        {"budget_usd": -0.01},
        {"prompt": "   "},
        {"repo_url": ""},
        {"repo_sha": ""},
        {"hosting": ""},
        {"model": ""},
    ],
)
def test_a_spec_refuses_what_the_server_would_reject(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RunSpec(**_fields(**overrides))


def test_a_spec_cannot_name_its_own_tenant() -> None:
    # The tenant is a field of the REQUEST, and the spec has no way to spell it.
    with pytest.raises(TypeError):
        RunSpec(**_fields(), network_id="n_" + "a" * 24)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "overrides",
    [{"max_hours": True}, {"max_hours": False}, {"budget_usd": True}, {"budget_usd": False}],
)
def test_a_spec_refuses_a_bool_where_a_number_belongs(overrides: dict[str, Any]) -> None:
    """E7 — ``True`` is an ``int`` to ``isinstance`` and one hour to ``isfinite``."""
    with pytest.raises(ValueError):
        RunSpec(**_fields(**overrides))


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (SHA, True),
        (SHA.upper(), True),
        (SHA[:39], False),
        (SHA + "a", False),
        ("HEAD", False),
        ("deadbeef", False),
        ("z" * 40, False),
        (None, False),
    ],
)
def test_a_sha_is_a_full_forty_hex_commit(value: Any, accepted: bool) -> None:
    """SE4 — the predicate the CLI refuses on."""
    assert is_commit_sha(value) is accepted


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("https://example.invalid/repo.git", True),
        ("ssh://git@example.invalid/repo.git", True),
        ("git@example.invalid:team/repo.git", True),
        ("git://example.invalid/repo.git", False),
        ("file:///srv/repo", False),
        ("./repo", False),
        ("../repo", False),
        ("https://", False),
        (None, False),
    ],
)
def test_a_repo_url_is_one_the_far_end_can_clone(value: Any, accepted: bool) -> None:
    assert is_repo_url(value) is accepted


def test_bind_pairs_the_body_with_the_resolved_network(mesh_root: Path) -> None:
    record = _save_network(mesh_root)
    request = RunSpec(**_fields()).bind(resolve_network("home", mesh_root))
    assert request.network_id == record.network_id
    assert request.to_json()["network_id"] == record.network_id

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
    MAX_IDEMPOTENCY_KEY_BYTES,
    RUNS_PATH,
    TERMINAL_RUN_STATUSES,
    RemoteComputeClient,
    check_idempotency_key,
    is_terminal,
    new_idempotency_key,
    run_id,
    run_network,
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
    status: str = "queued",
    *,
    network_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A run as the server returns it; ``network_id`` is omitted unless given."""
    payload: dict[str, Any] = {"id": run_id_value, "status": status, **extra}
    if network_id is not None:
        payload["network_id"] = network_id
    return payload


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sends_the_contract_fields_and_the_tenant(
    credential: int, request_for: Any
) -> None:
    expected = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json={"result": expected}))
    client, http = _client(origin, credential)
    async with http:
        created = await client.create_run(request_for(), idempotency_key="key-1")
    assert created == expected
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
async def test_a_bare_payload_is_accepted_as_well_as_the_envelope(
    credential: int, request_for: Any
) -> None:
    bare = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json=bare))
    client, http = _client(origin, credential)
    async with http:
        assert await client.create_run(request_for(), idempotency_key="k") == bare


@pytest.mark.asyncio
async def test_a_replay_returns_the_same_run(credential: int, request_for: Any) -> None:
    replay = _run("run_7", "running", network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json=replay), httpx.Response(201, json=replay))
    client, http = _client(origin, credential)
    async with http:
        first = await client.create_run(request_for(), idempotency_key="same")
        second = await client.create_run(request_for(), idempotency_key="same")
    assert first == second == replay
    assert origin.requests[0].headers["idempotency-key"] == "same"
    assert origin.requests[1].headers["idempotency-key"] == "same"


@pytest.mark.asyncio
async def test_a_run_from_another_network_is_refused(credential: int, request_for: Any) -> None:
    """SE1 — the response's tenant is compared with the request's, not assumed."""
    other = "n_" + "b" * 24
    origin = _Origin(httpx.Response(201, json={"result": _run(network_id=other)}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.create_run(request_for(), idempotency_key="k")


@pytest.mark.asyncio
async def test_a_run_without_a_network_id_is_a_mismatch(credential: int, request_for: Any) -> None:
    """The field is in the contract, so its absence is a response we cannot verify."""
    origin = _Origin(httpx.Response(201, json={"result": _run()}))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.create_run(request_for(), idempotency_key="k")


@pytest.mark.asyncio
async def test_a_poll_from_another_network_stops_a_watch(credential: int, request_for: Any) -> None:
    """The watch names the network it follows, so a drifted poll is a failure."""
    origin = _Origin(httpx.Response(200, json=_run("run_1", "running", network_id="n_" + "c" * 24)))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(NetworkMismatch):
            await client.get_run("run_1", expected_network=request_for.record.network_id)


@pytest.mark.asyncio
async def test_a_poll_without_an_expected_network_is_not_checked(credential: int) -> None:
    """``--remote-status`` names no network, so it takes the run as returned."""
    origin = _Origin(httpx.Response(200, json=_run("run_1", "running")))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_1") == _run("run_1", "running")


@pytest.mark.asyncio
async def test_a_run_carrying_its_own_result_field_is_not_unwrapped(
    credential: int, request_for: Any
) -> None:
    """E1 — only an ENVELOPE is unwrapped, not any object with a ``result`` key."""
    bare = _run(network_id=request_for.record.network_id, result="a run field of its own")
    origin = _Origin(httpx.Response(201, json=bare))
    client, http = _client(origin, credential)
    async with http:
        assert await client.create_run(request_for(), idempotency_key="k") == bare


@pytest.mark.asyncio
async def test_an_envelope_with_a_message_beside_the_result_still_unwraps(
    credential: int, request_for: Any
) -> None:
    run = _run(network_id=request_for.record.network_id)
    origin = _Origin(httpx.Response(201, json={"result": run, "message": "accepted"}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.create_run(request_for(), idempotency_key="k") == run


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
    origin = _Origin(httpx.Response(200, json={"result": _run("run_2", "running")}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.get_run("run_2") == _run("run_2", "running")
    assert str(origin.requests[0].url) == DEFAULT_API_URL + RUNS_PATH + "/run_2"


@pytest.mark.asyncio
async def test_a_run_id_cannot_escape_its_path_segment(credential: int) -> None:
    origin = _Origin(httpx.Response(200, json=_run("x")))
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
    origin = _Origin(
        httpx.Response(202, json=_run("run_3", "cancelled")),
        httpx.Response(202, json=_run("run_3", "cancelled")),
    )
    client, http = _client(origin, credential)
    async with http:
        assert await client.cancel_run("run_3") == _run("run_3", "cancelled")
        assert await client.cancel_run("run_3") == _run("run_3", "cancelled")
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
    origin = _Origin(httpx.Response(200, json={"result": []}))
    client, http = _client(origin, credential)
    async with http:
        assert await client.list_runs(network_id="n_abc", limit=5, offset=10) == []
    query = origin.requests[0].url.params
    assert (query["limit"], query["offset"], query["network_id"]) == ("5", "10", "n_abc")


@pytest.mark.asyncio
async def test_list_runs_omits_an_absent_network(credential: int) -> None:
    origin = _Origin(httpx.Response(200, json={"result": []}))
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
    origin = _Origin(httpx.Response(200, content=b"not json"))
    client, http = _client(origin, credential)
    async with http:
        with pytest.raises(RemoteComputeError):
            await client.get_run("run_1")


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
    # The context is CUT, not merely unraised: a bare ``raise X`` would still
    # print the httpx exception under `During handling of the above exception`.
    assert caught.value.__suppress_context__ is True
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
    assert is_terminal({"status": "succeeded"})
    assert is_terminal({"status": "failed"})
    assert not is_terminal({"status": "running"})
    # An unknown status is NOT terminal: a run whose state this build cannot
    # name is still running as far as anyone here can tell.
    assert not is_terminal({"status": "some_new_state"})
    assert "succeeded" in TERMINAL_RUN_STATUSES
    with pytest.raises(RemoteComputeError):
        run_status({"id": "run_1"})
    with pytest.raises(RemoteComputeError):
        run_id({"status": "queued"})


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

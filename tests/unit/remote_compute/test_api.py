"""The shared library: submit-and-record, read-and-refresh, and the tier copy.

WHAT IS PROVEN. That a surface calling :class:`RemoteRuns` gets the client's
answer AND a local record of it, that the record follows the run as its status is
read, that a delete is recorded without hiding the row, and that the tier copy is
keyed (so a picker renders translated labels rather than English literals).

THE ISOLATION FIXTURES ARE LOCAL. ``test_client.py`` has its own; these are the
same few, restated here rather than moved into the directory's ``conftest`` so a
future file that needs them makes the move once for all three.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.network import store
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.providers.auth_store import AuthStore
from local_operator.remote_compute import api as api_module
from local_operator.remote_compute import records, resolve_network, tiers
from local_operator.remote_compute.api import RemoteRuns, RunSubmitMetadata, build_spec
from local_operator.remote_compute.client import RUNS_PATH, RemoteComputeClient
from local_operator.remote_compute.defaults import DEFAULT_MAX_HOURS, DEFAULT_SIZE
from local_operator.remote_compute.repo import UNPARSEABLE_URL, RepoRef

SHA = "0123456789abcdef0123456789abcdef01234567"


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
    escaped: list[str] = []

    def refuse(self: Any, request: httpx.Request) -> httpx.Response:
        escaped.append(f"{request.method} {request.url}")
        raise AssertionError(f"a test attempted a real network call: {request.url}")

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
    monkeypatch.delenv("LOP_REMOTE_COMPUTE_API_URL", raising=False)
    return root


@pytest.fixture()
def credential() -> int:
    with closing(AuthStore()) as store_:
        row = store_.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct", "access": "token-abc"}
        )
        return row.id


@pytest.fixture()
def request_for(tmp_path: Path) -> Any:
    record = NetworkRecord(
        network_id=store.new_network_id(),
        name="home",
        self_device_id="d_self",
        members=[MemberRecord(device_id="d_self", name="this device")],
    )
    store.save(record, tmp_path / "mesh")
    binding = resolve_network("home", tmp_path / "mesh")

    def build(**overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "prompt": "fix the flaky test",
            "repo_url": "https://example.invalid/repo.git",
            "repo_sha": SHA,
            "hosting": "openai",
            "model": "gpt-5",
        }
        fields.update(overrides)
        return build_spec(**fields).bind(binding)

    # The tenant the binding names, so a fake response can answer for it.
    build.network_id = record.network_id  # type: ignore[attr-defined]
    return build


def _library(origin: _Origin, credential: int, root: Path) -> tuple[RemoteRuns, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=origin.transport)
    client = RemoteComputeClient(http, credential_id=credential)
    return RemoteRuns(client, config_dir=root), http


def _created(*, network_id: str = "net_1", status: str = "pending") -> dict[str, Any]:
    run = {"id": "run_1", "status": status, "network_id": network_id}
    return {"msg": "Run created", "result": {"run": run}}


@pytest.mark.asyncio
async def test_a_submission_is_recorded_with_where_it_came_from(
    credential: int, request_for: Any, isolated_roots: Path
) -> None:
    origin = _Origin(httpx.Response(201, json=_created(network_id=request_for.network_id)))
    library, http = _library(origin, credential, isolated_roots)
    async with http:
        submitted = await library.create_run(
            request_for(),
            idempotency_key="k",
            recorded_from=RunSubmitMetadata(
                repo_url="https://example.invalid/repo.git",
                repo_sha=SHA,
                prompt="fix the flaky test",
                size="m",
            ),
        )
    assert submitted.replayed is False
    assert submitted.run["id"] == "run_1"
    assert submitted.record is not None
    # The record says where the run was submitted, as the client resolved it:
    # this test does not override the base, so it is the client's own answer.
    assert submitted.record.api_origin == library.client.api_url
    assert submitted.record.network_id == request_for.network_id
    assert submitted.record.status == "pending"
    assert [row.run_id for row in library.local_runs()] == ["run_1"]


@pytest.mark.asyncio
async def test_a_read_refreshes_the_recorded_status(
    credential: int, isolated_roots: Path, request_for: Any
) -> None:
    created = httpx.Response(201, json=_created(network_id=request_for.network_id))
    read = httpx.Response(200, json={"msg": "Run", "result": {"id": "run_1", "status": "running"}})
    origin = _Origin(created, read)
    library, http = _library(origin, credential, isolated_roots)
    async with http:
        await library.create_run(request_for(), idempotency_key="k")
        await library.get_run("run_1")
    assert library.local_runs()[0].status == "running"


@pytest.mark.asyncio
async def test_a_delete_marks_the_row_rather_than_dropping_it(
    credential: int, isolated_roots: Path
) -> None:
    origin = _Origin(httpx.Response(204))
    library, http = _library(origin, credential, isolated_roots)
    records.record_submission(
        run_id="run_1",
        repo_url="https://example.invalid/repo.git",
        repo_sha=SHA,
        prompt="fix it",
        size="m",
        api_origin="https://api.example.invalid",
        status="stopped",
        config_dir=isolated_roots,
    )
    async with http:
        await library.delete_run("run_1")
    assert origin.requests[0].method == "DELETE"
    assert origin.requests[0].url.path == RUNS_PATH + "/run_1"
    listed = library.local_runs()
    assert [row.run_id for row in listed] == ["run_1"]
    assert listed[0].deleted is True


@pytest.mark.asyncio
async def test_the_listing_carries_the_status_filter_through(
    credential: int, isolated_roots: Path
) -> None:
    page = {"runs": [], "limit": 50, "offset": 0}
    origin = _Origin(httpx.Response(200, json={"msg": "Runs", "result": page}))
    library, http = _library(origin, credential, isolated_roots)
    async with http:
        assert await library.list_runs(status="running") == []
    assert origin.requests[0].url.params["status"] == "running"


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_written_does_not_fail_the_submission(
    credential: int, request_for: Any, isolated_roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run exists at the far end; a full disk must not say otherwise."""

    def boom(*arguments: object, **keywords: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(records, "record_submission", boom)
    origin = _Origin(httpx.Response(201, json=_created(network_id=request_for.network_id)))
    library, http = _library(origin, credential, isolated_roots)
    async with http:
        submitted = await library.create_run(request_for(), idempotency_key="k")
    assert submitted.record is None
    assert submitted.run["id"] == "run_1"


@pytest.mark.asyncio
async def test_a_repo_url_the_scrubber_cannot_parse_still_submits_and_records(
    credential: int, request_for: Any, isolated_roots: Path
) -> None:
    """A1's repro through the real path, where the run already exists.

    ``create_run`` reaches the scrubber from ``record_submission`` only AFTER the
    server has answered 201. A scrubber that raised there told the operator the
    submission failed, with no run id to cancel, for a run the far end had
    already created; the recorded URL is now the placeholder instead.
    """
    origin = _Origin(httpx.Response(201, json=_created(network_id=request_for.network_id)))
    library, http = _library(origin, credential, isolated_roots)
    async with http:
        submitted = await library.create_run(
            request_for(),
            idempotency_key="k",
            recorded_from=RunSubmitMetadata(repo_url="https://[::1/r.git", repo_sha=SHA),
        )
    assert submitted.run["id"] == "run_1"
    assert submitted.record is not None
    assert submitted.record.repo_url == UNPARSEABLE_URL
    stored = records.records_path(isolated_roots).read_text(encoding="utf-8")
    assert "[::1" not in stored


@pytest.mark.asyncio
async def test_any_failure_from_the_record_does_not_fail_the_submission(
    credential: int,
    request_for: Any,
    isolated_roots: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The guard is wider than ``OSError``, and the message is never logged.

    What the record stores is caller-supplied, so a URL the scrubber refuses is
    the same failure class as a full disk — and the message of whatever is raised
    can quote that content, which is the thing the scrub exists to keep out of
    reach. So the exception's TYPE is logged and nothing else.
    """

    def boom(*arguments: object, **keywords: object) -> None:
        raise ValueError("https://[::1/r.git")

    monkeypatch.setattr(records, "record_submission", boom)
    origin = _Origin(httpx.Response(201, json=_created(network_id=request_for.network_id)))
    library, http = _library(origin, credential, isolated_roots)
    with caplog.at_level(logging.WARNING, logger="local_operator.remote_compute.api"):
        async with http:
            submitted = await library.create_run(request_for(), idempotency_key="k")
    assert submitted.record is None
    assert submitted.run["id"] == "run_1"
    logged = [entry.getMessage() for entry in caplog.records]
    assert any("ValueError" in message for message in logged)
    assert all("[::1" not in message for message in logged)


def test_the_defaults_are_the_documented_constants() -> None:
    chosen = api_module.run_defaults()
    assert chosen.size == DEFAULT_SIZE == "m"
    assert chosen.max_hours == DEFAULT_MAX_HOURS == 2.0


def test_build_spec_prefers_a_detected_repository_and_applies_defaults() -> None:
    detected = RepoRef(directory=Path("/tmp/repo"), url="https://x.invalid/r.git", sha=SHA, refs=())
    spec = build_spec(prompt="do it", repo=detected, hosting="openai", model="gpt-5", name=None)
    assert spec.repo_url == "https://x.invalid/r.git"
    assert spec.repo_sha == SHA
    assert spec.size == DEFAULT_SIZE
    assert spec.max_hours == DEFAULT_MAX_HOURS


def test_build_spec_lets_an_explicit_size_and_bound_win() -> None:
    spec = build_spec(
        prompt="do it",
        repo_url="https://x.invalid/r.git",
        repo_sha=SHA,
        hosting="openai",
        model="gpt-5",
        size="l",
        max_hours=6.0,
    )
    assert (spec.size, spec.max_hours) == ("l", 6.0)


def test_the_tier_labels_are_keyed_copy() -> None:
    """A label is a message code, never an English string built here."""
    assert tiers.tier_label("cloud_run").code == "wire.remote_compute.tier_cloud_run"
    assert tiers.tier_label("dedicated").code == "wire.remote_compute.tier_dedicated"
    assert tiers.option_label("spot").code == "wire.remote_compute.option_spot"


def test_spot_is_an_option_under_dedicated_and_not_a_tier() -> None:
    assert tiers.TIERS == ("cloud_run", "dedicated")
    assert tiers.tier_options("dedicated") == ("spot",)
    assert tiers.tier_options("cloud_run") == ()
    with pytest.raises(ValueError):
        tiers.tier_label("spot")


def test_an_unknown_tier_is_refused_rather_than_rendered() -> None:
    with pytest.raises(ValueError):
        tiers.tier_label("spot-market")
    with pytest.raises(ValueError):
        tiers.tier_options("nonsense")

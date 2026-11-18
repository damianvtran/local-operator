"""The desktop's Radient remote-compute routes, driven through the real app.

WHAT IS PINNED HERE, one test per property the GUI's brief depends on:

* **the flag is a consent gate, not a hint** — with ``remote_compute.enabled``
  off the summary read answers ``{"enabled": false}``, the listing answers
  ``{"runs": [], "server": "disabled"}`` and NOT ONE upstream request is made
  (measured by the fake Radient's own call log rather than by a mock's
  ``assert_not_called``), while cancel and delete stay reachable;
* **a retry after a lost response replays** — a create body may carry its own
  ``idempotency_key``, two posts under one key put the same header and body on
  the wire and get the same run id back, and a key the contract refuses is a 400
  before any request;
* **one identifier vocabulary** — every row of ``runs[]``, and both arms of the
  detail read, carry ``id`` and not the record's ``run_id``;
* **the account states are the plane's own three** — ``signed-out`` is a real
  empty ``AuthStore``, ``expired`` is ``desktop_radient``'s own
  ``_credential_refusal(grant_invalid)``, and a transient unrefreshable row still
  reads ``signed-in`` (its remedy is a retry, not a re-login);
* **``offered`` is the DEPLOYMENT's answer** — a 403 preview, a 503
  ``not_configured`` and a capacity page with no ready tier each answer ``false``,
  while an unreachable API stays ``true`` (that is a network problem, and calling
  it "not offered" paints the wrong remedy);
* **the runs listing merges the server with the LOCAL record** and still answers
  when the server does not;
* **every create refusal happens here, before a request is composed** (an
  unpushed HEAD, an ssh-only origin, a directory that is not a repository, the
  Dedicated tier, the flag off), and every upstream refusal keeps the server's
  own status and code (403, 402, 409, 429 with ``Retry-After``, 503);
* **no bearer reaches a response body or a log line**, asserted against a token
  the fake store holds and the fake upstream is handed.

Everything runs through the real ``require_desktop`` gate, the real FastAPI app
state and a real temporary filesystem; the only fakes are Radient itself (an
``httpx.MockTransport`` handler, so the library's own client, envelope unwrapping
and refusal mapping all execute) and the git repository, which is real.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import subprocess
import tarfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import store as network_store
from local_operator.network import types as network_types
from local_operator.providers.auth_store import AuthStore
from local_operator.remote_compute import records, results
from local_operator.remote_compute.errors import RunNotFound
from local_operator.server.routes import desktop_radient, desktop_remote_compute

#: THE GATE'S PLACEHOLDER TOKEN, spelled as a word the way the neighbouring
#: desktop suites spell it: this backend only checks that the bearer matches the
#: env capability, and no test here reads a real token from the environment.
DESKTOP_TOKEN = "synthetic-desktop-token"

#: The bearer the fake ``AuthStore`` holds, chosen to be unmistakable: the "no
#: leak" test greps every response body and log line for it.
RADIENT_ACCESS = "RADIENT-BEARER-MUST-NOT-LEAK"

#: The origin the fake repository is pushed to. ``https://`` because that is the
#: only form the container can clone without a key.
ORIGIN = "https://github.com/example/widget.git"

RUN_ID = "9f2c1a04-1111-4111-8111-abcdef000001"
OTHER_RUN_ID = "9f2c1a04-2222-4222-8222-abcdef000002"

NET_ID = "n_" + "1" * 24

#: The session id a fake results archive carries: 12 lowercase hex, the shape lop
#: writes a session directory with.
SESSION_ID = "e1e2e3e4e5e6"

#: The signature a fake presigned artifact URL carries. Greppable, so the cells
#: that assert "no URL leaves this process" cannot pass by accident.
SIGNATURE = "PRESIGNED-MUST-NOT-LEAK"


@dataclass
class FakeRadient:
    """``httpx.MockTransport``'s handler: answers a table, records every call.

    AN UNSTUBBED CALL IS A FAILURE rather than an empty answer: several cells
    below assert that a refusal happened BEFORE any request was composed (the
    flag off, a dedicated tier, an unusable repository), and a handler that
    answered everything would let those regress silently — which is exactly how
    "the request that should not have been made" stays invisible.
    """

    answers: dict[tuple[str, str], list[httpx.Response]] = field(default_factory=dict)
    calls: list[httpx.Request] = field(default_factory=list)
    #: When set, every call raises it instead of answering — the offline case.
    transport_failure: Exception | None = None

    def add(
        self,
        method: str,
        path: str,
        *,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Queue one answer for ``(method, path)``; several queue in order."""
        body = {} if payload is None else payload
        self.answers.setdefault((method.upper(), path), []).append(
            httpx.Response(status, json=body, headers=headers)
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.transport_failure is not None:
            raise self.transport_failure
        queue = self.answers.get((request.method, request.url.path))
        if not queue:
            raise AssertionError(
                f"the route asked Radient for {request.method} {request.url.path}, "
                "which this cell did not stub"
            )
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def paths(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.calls]

    def last_body(self) -> dict[str, Any]:
        body = self.calls[-1].content
        return json.loads(body) if body else {}


@dataclass
class Api:
    """The mounted app, its client, and the three seams a test can inspect."""

    client: AsyncClient
    root: Path
    radient: FakeRadient
    app: FastAPI
    store: AuthStore

    def configure(self, *, enabled: bool = True, model: bool = True) -> None:
        """Write this device's config, the way the TUI or the CLI would.

        The manager is rebuilt because ``is_enabled`` reads the file LIVE (that is
        the point of the switch) while the manager a route is handed is the app's
        own object — replacing it is what makes a config edit visible to a route
        that resolves the model pair through ``get_config_value``.
        """
        # THE `values:` WRAPPER IS THE STORE'S FORMAT, not decoration: a top-level
        # key is one the store does not read, and a test that wrote one would be
        # exercising a config nothing else can produce.
        lines = ["values:", "  remote_compute:", f"    enabled: {'true' if enabled else 'false'}"]
        if model:
            lines += ["  hosting: openrouter", "  model_name: example/model"]
        (self.root / "config.yml").write_text("\n".join(lines) + "\n")
        self.app.state.config_manager = ConfigManager(self.root)

    def sign_out(self) -> None:
        """This device with no Radient account at all — the plane's ``signed-out``."""
        for row in self.store.list_credentials("radient", include_disabled=True):
            self.store.delete_credential(row.id)

    def expire(self) -> None:
        """A stored sign-in whose bearer is past its life and cannot be refreshed."""
        for row in self.store.list_credentials("radient"):
            data = dict(row.data)
            data["expires"] = int(time.time() * 1000) - 60_000
            self.store.upsert_credential("radient", data)


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Api]:
    """The real router behind the real desktop gate, with a stubbed Radient."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    # A run submitted here would otherwise be pointed at whatever the operator's
    # own shell exports; the default (production Radient) is never dialled because
    # the transport is a MockTransport, and this keeps the recorded origin stable.
    monkeypatch.delenv("LOP_REMOTE_COMPUTE_API_URL", raising=False)

    radient = FakeRadient()

    @contextlib.asynccontextmanager
    async def transport() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(radient), follow_redirects=False
        ) as client:
            yield client

    monkeypatch.setattr(desktop_remote_compute, "_transport", transport)

    store = AuthStore(tmp_path / "auth.db")
    _credential(store)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_auth = SimpleNamespace(store=store)
    app.include_router(desktop_remote_compute.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {DESKTOP_TOKEN}"},
        ) as client:
            yield Api(client=client, root=tmp_path, radient=radient, app=app, store=store)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Fixtures that put a machine on disk: a network, a repository, a credential
# ---------------------------------------------------------------------------


def _network(root: Path, *, active: bool = True, name: str = "devmesh") -> None:
    """A network record this device is an active member of (or was removed from)."""
    record = network_types.NetworkRecord(
        network_id=NET_ID, name=name, self_device_id="d_" + "7" * 32, self_role="admin"
    )
    record.members.append(
        network_types.MemberRecord(
            device_id=record.self_device_id,
            name="this-machine",
            role="admin",
            capabilities=["admin"],
            lifecycle="active" if active else "expired",
        )
    )
    network_store.save(record, root)


def _git(cwd: Path, *arguments: str) -> str:
    """One git command with a pinned identity, so no global config is consulted."""
    done = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(cwd),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.org",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.org",
        },
    )
    return done.stdout


def _repository(
    root: Path, *, origin: str = ORIGIN, pushed: bool = True, directory: str = "work"
) -> Path:
    """A real repository, optionally with its HEAD on an ``origin`` tracking ref."""
    path = root / directory
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--quiet", "--initial-branch=main")
    (path / "README.md").write_text("hello\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "--quiet", "-m", "one")
    _git(path, "remote", "add", "origin", origin)
    if pushed:
        # The tracking ref is what ``repo.detect_repo`` reads: a commit is
        # "pushed" when an ``origin`` tracking ref contains it.
        _git(path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return path


def _credential(store: AuthStore, *, access: str = RADIENT_ACCESS) -> None:
    """A fresh Radient OAuth row: usable, so the account reads ``signed-in``."""
    store.upsert_credential(
        "radient",
        {
            "access": access,
            "refresh": "refresh-token",
            "token_type": "Bearer",
            "expires": int(time.time() * 1000) + 3_600_000,
            "email": "operator@example.org",
        },
    )


def _signed_out_bearer(*_args: Any, **_kwargs: Any) -> Any:
    """``radient_bearer`` for the signed-out case, as ``desktop_radient`` raises it."""
    raise desktop_radient._failure(  # noqa: SLF001 — the plane's own refusal helper
        409, "radient_no_credential", desktop_radient.NO_CREDENTIAL_MESSAGE
    )


def _dead_grant_bearer(*_args: Any, **_kwargs: Any) -> Any:
    """``radient_bearer`` for an IdP-killed grant, worded by the plane itself."""
    raise desktop_radient._credential_refusal(  # noqa: SLF001 — one diagnosis, one answer
        desktop_radient.REASON_GRANT_INVALID
    )


def _run(run_id: str = RUN_ID, **overrides: Any) -> dict[str, Any]:
    """A run as the server reports it, with the fields the panel paints."""
    payload: dict[str, Any] = {
        "id": run_id,
        "status": "running",
        "tier": "cloud_run",
        "network_id": NET_ID,
        "deadline_at": "2026-11-18T00:00:00Z",
        "metered_seconds": 252,
        "charged_usd": 0.0,
        "hold_usd": 5.09,
        # THE SECRETS THE CREATE WRAPPER CARRIES. A per-run read never includes
        # them; they are planted on a stubbed body to prove the filter holds even
        # when the body grows them.
        "run_token": "PLANTED_RUN_FIELD_ONE",
        "tunnel": {"token": "PLANTED_TUNNEL_FIELD_TWO"},
    }
    payload.update(overrides)
    return payload


def _capacity(tier: str = "cloud_run", ready: bool = True) -> dict[str, Any]:
    return {"msg": "ok", "result": {"tiers": [{"tier": tier, "ready": ready, "detail": {}}]}}


# ---------------------------------------------------------------------------
# The consent gate, and the account states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_flag_off_answers_only_the_disabled_state_and_calls_nothing(api: Api) -> None:
    """``remote_compute.enabled: false`` is a consent gate, measured at the wire."""
    api.configure(enabled=False)

    answer = await api.client.get("/v1/desktop/remote-compute")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"enabled": False}
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_create_with_the_flag_off_is_refused_before_any_request(api: Api) -> None:
    api.configure(enabled=False)
    directory = _repository(api.root)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "fix it", "tier": "cloud_run"},
    )

    assert answer.status_code == 403, answer.text
    assert answer.json()["error"]["code"] == "not_configured"
    # The sentence is the library's own "switch is off" copy, not new prose.
    assert "Remote compute is off" in answer.json()["error"]["message"]
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_the_flag_off_answers_an_empty_listing_and_calls_nothing(api: Api) -> None:
    """The listing is gated like the summary: off is a consent switch, not a hint."""
    api.configure(enabled=False)
    _record(api, RUN_ID)

    answer = await api.client.get("/v1/desktop/remote-compute/runs")

    assert answer.status_code == 200, answer.text
    # ``disabled`` rather than ``unavailable``: there is no host to retry, and its
    # own word is what keeps a panel from painting a retry over the user's choice.
    assert answer.json() == {"runs": [], "server": "disabled"}
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_cancel_and_delete_stay_reachable_with_the_flag_off(api: Api) -> None:
    """Stopping work the switch does not own: a live run still spends money.

    The CLI documents the same intent for ``--remote-cancel``, and the module
    docstring states it once for this surface.
    """
    api.configure(enabled=False)
    _record(api, RUN_ID)
    api.radient.add("POST", f"/v1/remote-compute/runs/{RUN_ID}/cancel", status=204)
    api.radient.add("DELETE", f"/v1/remote-compute/runs/{RUN_ID}", status=204)

    cancelled = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/cancel")
    deleted = await api.client.delete(f"/v1/desktop/remote-compute/runs/{RUN_ID}")

    assert cancelled.status_code == 200, cancelled.text
    assert deleted.status_code == 200, deleted.text
    assert api.radient.paths() == [
        ("POST", f"/v1/remote-compute/runs/{RUN_ID}/cancel"),
        ("DELETE", f"/v1/remote-compute/runs/{RUN_ID}"),
    ]


@pytest.mark.asyncio
async def test_an_empty_store_reads_signed_out(api: Api) -> None:
    """A real, empty ``AuthStore``: the state is decided by rows, not by a stub."""
    api.configure()
    api.sign_out()
    _network(api.root)

    answer = await api.client.get("/v1/desktop/remote-compute")

    body = answer.json()
    assert answer.status_code == 200, answer.text
    assert body["enabled"] is True
    assert body["account"] == {"state": "signed-out", "label": ""}
    # Nothing is asked of Radient for a device that has no account to ask about.
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_dead_grant_reads_expired(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """The IdP's verdict, carried by ``desktop_radient``'s own refusal."""
    api.configure()
    # An EXPIRED row, so the store's own read yields nothing and the plane's
    # diagnosis is what answers — which is the path this cell is about.
    api.expire()
    monkeypatch.setattr(desktop_remote_compute, "radient_bearer", _dead_grant_bearer)

    answer = await api.client.get("/v1/desktop/remote-compute")

    assert answer.json()["account"]["state"] == "expired"
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_transient_credential_problem_still_reads_signed_in(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``credential_unavailable``'s remedy is a retry, not a re-login.

    Reading it as ``expired`` would send the operator to sign in again over a
    problem that a second attempt fixes — and the create path is where a real
    refusal belongs.
    """
    api.configure()
    api.expire()

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise desktop_radient._credential_refusal(  # noqa: SLF001
            desktop_radient.REASON_UNAVAILABLE
        )

    monkeypatch.setattr(desktop_remote_compute, "radient_bearer", unavailable)

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["account"]["state"] == "signed-in"
    # No bearer could be minted, so nothing was asked of Radient: the account is
    # signed in, this moment is not, and the deployment has not said it is off.
    assert api.radient.calls == []
    assert body["offered"] is True
    assert body["capacity"] == []


@pytest.mark.asyncio
async def test_the_account_label_is_the_credential_identity(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``credential_identity``'s email, not a new derivation and not the row id."""
    api.configure()
    api.radient.add("GET", "/v1/remote-compute/capacity", payload=_capacity())
    api.radient.add("GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": []}})

    answer = await api.client.get("/v1/desktop/remote-compute")

    assert answer.json()["account"] == {
        "state": "signed-in",
        "label": "operator@example.org",
    }


# ---------------------------------------------------------------------------
# The one read: networks, capacity, pricing, live runs, offered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_read_answers_the_contract_shape(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole object the GUI's brief is written from, asserted field by field."""
    api.configure()
    _network(api.root)
    api.radient.add("GET", "/v1/remote-compute/capacity", payload=_capacity())
    api.radient.add(
        "GET",
        "/v1/remote-compute/runs",
        payload={"msg": "ok", "result": {"runs": [_run()], "limit": 50, "offset": 0}},
    )

    answer = await api.client.get("/v1/desktop/remote-compute")

    body = answer.json()
    assert answer.status_code == 200, answer.text
    assert body["enabled"] is True
    assert body["account"]["state"] == "signed-in"
    assert body["networks"] == [{"network_id": NET_ID, "name": "devmesh", "role": "admin"}]
    assert body["capacity"] == [{"tier": "cloud_run", "ready": True, "detail": {}}]
    # Pricing is null until the capacity read publishes a rate: the honest state,
    # not a client-side constant that drifts from the deployment's config.
    assert body["pricing"] is None
    assert [run["id"] for run in body["active"]] == [RUN_ID]
    assert body["active"][0]["hold_usd"] == 5.09
    assert body["offered"] is True


@pytest.mark.asyncio
async def test_capacity_rows_publish_only_the_three_contract_fields(api: Api) -> None:
    """Capacity is the one server-shaped body here, so it is narrowed explicitly."""
    api.configure()
    api.radient.add(
        "GET",
        "/v1/remote-compute/capacity",
        payload={
            "msg": "ok",
            "result": {
                "tiers": [
                    {
                        "tier": "cloud_run",
                        "ready": True,
                        "detail": {},
                        "hourly_usd": 9.5,
                        "pool": "internal-pool",
                    }
                ]
            },
        },
    )
    # The summary also reads the live runs; stubbed empty so the only thing this
    # cell measures is the shape of the capacity rows.
    api.radient.add("GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": []}})

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["capacity"] == [{"tier": "cloud_run", "ready": True, "detail": {}}]
    assert body["offered"] is True


@pytest.mark.asyncio
async def test_a_network_this_device_was_removed_from_is_not_offered(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resolve_network`` would refuse it, so the picker must not list it."""
    api.configure()
    _network(api.root, active=False)
    api.radient.add("GET", "/v1/remote-compute/capacity", payload=_capacity())
    api.radient.add("GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": []}})

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["networks"] == []


@pytest.mark.asyncio
async def test_a_preview_403_answers_offered_false(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider's own switch: painted, because the user turned theirs on."""
    api.configure()
    _network(api.root)
    api.radient.add(
        "GET",
        "/v1/remote-compute/capacity",
        status=403,
        payload={"error": {"code": "preview_not_enabled"}},
    )
    api.radient.add(
        "GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": [_run()]}}
    )

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["offered"] is False
    assert body["capacity"] == []
    # A run already in flight is still reported, even though the deployment will
    # admit no new one: it was admitted under the old answer and is spending now.
    assert [row["id"] for row in body["active"]] == [RUN_ID]


@pytest.mark.asyncio
async def test_a_503_not_configured_answers_offered_false(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.configure()
    _network(api.root)
    api.radient.add(
        "GET",
        "/v1/remote-compute/capacity",
        status=503,
        payload={"error": {"code": "not_configured"}},
    )
    api.radient.add(
        "GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": [_run()]}}
    )

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["offered"] is False
    assert [row["id"] for row in body["active"]] == [RUN_ID]


@pytest.mark.asyncio
async def test_capacity_with_no_ready_tier_answers_offered_false(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radient is not "offline": it is capacity that cannot be provisioned now."""
    api.configure()
    _network(api.root)
    api.radient.add("GET", "/v1/remote-compute/capacity", payload=_capacity(ready=False))
    api.radient.add("GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": []}})

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["offered"] is False
    assert body["active"] == []
    assert len(body["capacity"]) == 1
    # The tier is REPORTED rather than hidden: "capacity unavailable" and "no
    # tiers at all" are different sentences.
    assert body["capacity"][0]["ready"] is False


@pytest.mark.asyncio
async def test_an_unreachable_api_stays_offered(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """A network problem is not the deployment withdrawing the feature."""
    api.configure()
    _network(api.root)
    api.radient.transport_failure = httpx.ConnectError("no route")

    body = (await api.client.get("/v1/desktop/remote-compute")).json()

    assert body["offered"] is True
    assert body["capacity"] == []
    assert body["active"] == []


# ---------------------------------------------------------------------------
# The runs listing: the merge, and the offline answer
# ---------------------------------------------------------------------------


def _record(api: Api, run_id: str, *, status: str | None = "running") -> None:
    records.record_submission(
        run_id=run_id,
        repo_url=ORIGIN,
        repo_sha="a" * 40,
        prompt="summarised prompt",
        size="m",
        api_origin="https://api.example.org",
        network_id=NET_ID,
        status=status,
        config_dir=api.root,
    )


@pytest.mark.asyncio
async def test_the_listing_merges_server_rows_and_local_records(api: Api) -> None:
    """Three states in one cell: server-only, record-only, and both."""
    api.configure()
    _record(api, OTHER_RUN_ID)
    _record(api, RUN_ID)
    api.radient.add(
        "GET",
        "/v1/remote-compute/runs",
        payload={"msg": "ok", "result": {"runs": [_run()], "limit": 50, "offset": 0}},
    )

    body = (await api.client.get("/v1/desktop/remote-compute/runs")).json()

    assert body["server"] == "ok"
    # ONE vocabulary: every row names its run ``id``, whichever half it came from.
    assert all("id" in row for row in body["runs"])
    assert all("run_id" not in row for row in body["runs"])
    by_id = {row["id"]: row for row in body["runs"]}
    assert by_id[RUN_ID]["source"] == "server"
    assert by_id[OTHER_RUN_ID]["source"] == "record"
    # The record's own vocabulary travels: its repo and its (scrubbed) summary.
    assert by_id[OTHER_RUN_ID]["repo_url"] == ORIGIN
    assert by_id[OTHER_RUN_ID]["prompt_summary"] == "summarised prompt"
    # ONE row per run, not one per source.
    assert len(body["runs"]) == 2


@pytest.mark.asyncio
async def test_an_unreachable_server_still_lists_the_local_records(api: Api) -> None:
    api.configure()
    _record(api, RUN_ID)
    api.radient.transport_failure = httpx.ConnectError("no route")

    body = (await api.client.get("/v1/desktop/remote-compute/runs")).json()

    assert body["server"] == "unavailable"
    assert [row["id"] for row in body["runs"]] == [RUN_ID]
    assert body["runs"][0]["source"] == "record"


@pytest.mark.asyncio
async def test_a_signed_out_server_still_lists_the_local_records(api: Api) -> None:
    """A dead sign-in is the panel's most common state, and it is not an error."""
    api.configure()
    _record(api, RUN_ID)
    api.radient.add("GET", "/v1/remote-compute/runs", status=401, payload={"error": "no"})

    body = (await api.client.get("/v1/desktop/remote-compute/runs")).json()

    assert body["server"] == "signed-out"
    assert [row["id"] for row in body["runs"]] == [RUN_ID]


@pytest.mark.asyncio
async def test_a_deleted_run_stays_in_the_listing_marked(api: Api) -> None:
    """``records.mark_deleted`` keeps the row on purpose: the panel says so."""
    api.configure()
    _record(api, RUN_ID)
    records.mark_deleted(RUN_ID, config_dir=api.root)
    api.radient.add("GET", "/v1/remote-compute/runs", payload={"msg": "ok", "result": {"runs": []}})

    body = (await api.client.get("/v1/desktop/remote-compute/runs")).json()

    assert body["runs"][0]["deleted"] is True


@pytest.mark.asyncio
async def test_the_detail_read_falls_back_to_the_record_only_when_offline(api: Api) -> None:
    api.configure()
    _record(api, RUN_ID)
    api.radient.transport_failure = httpx.ConnectError("no route")

    body = (await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}")).json()

    assert body["run"]["source"] == "record"
    assert body["run"]["id"] == RUN_ID
    assert "run_id" not in body["run"]


@pytest.mark.asyncio
async def test_a_signed_out_detail_read_still_serves_the_record(api: Api) -> None:
    """An unusable sign-in is the unreachable case from the panel's point of view.

    The listing answers this device its rows, so the detail read must be able to
    open them: an offline laptop that lists a run it cannot open is the bug.
    """
    api.configure()
    _record(api, RUN_ID)
    api.sign_out()

    body = (await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}")).json()

    assert body["run"]["id"] == RUN_ID
    assert body["run"]["source"] == "record"
    # Nothing was asked of Radient: there is no credential to ask with.
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_signed_out_detail_read_with_no_record_keeps_the_sign_in_refusal(
    api: Api,
) -> None:
    """No record to fall back to, so the refusal is the answer (not a 404)."""
    api.configure()
    api.sign_out()

    answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}")

    assert answer.status_code == 409, answer.text
    assert answer.json()["error"]["code"] == "radient_no_credential"
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_refused_detail_read_is_not_papered_over_with_the_record(api: Api) -> None:
    """A 404 is an answer: the run is not there, and the local copy is not it."""
    api.configure()
    _record(api, RUN_ID)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}",
        status=404,
        payload={"error": {"code": "not_found"}},
    )

    answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}")

    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_a_codeless_404_still_answers_the_contracts_not_found(api: Api) -> None:
    """A renderer keying on ``error.code`` alone must see ``not_found`` here too.

    QA round 1, Q1: the status was right and the code was ``remote_compute_failed``,
    so "this run is gone" and "the library broke" were the same word.
    """
    api.configure()
    api.radient.add(
        "GET", f"/v1/remote-compute/runs/{RUN_ID}", status=404, payload={"error": "gone"}
    )

    answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}")

    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == "not_found"


def test_a_code_outside_the_contract_gets_the_surfaces_own_code_and_its_status() -> None:
    """The other half of Q1: an unknown code never reaches a renderer verbatim.

    The status stays exactly as the answer carried it, and the code is one of this
    surface's own words rather than the string the server sent.
    """
    failure = RunNotFound("gone", code="made_up_code", status=404)

    refusal = desktop_remote_compute.refusal_from(failure)

    assert refusal.status == 404
    assert refusal.code == "not_found"
    assert "made_up_code" not in refusal.message


# ---------------------------------------------------------------------------
# Create: the happy path, then every refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_submits_the_repository_and_records_the_run(api: Api) -> None:
    """The whole submit sequence, including the key and the local record."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={
            "msg": "submitted",
            "result": {"run": _run(status="pending"), "run_token": "s", "tunnel": {"t": "s"}},
        },
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={
            "directory": str(directory),
            "prompt": "fix the failing test",
            "tier": "cloud_run",
            "size": "l",
            "max_hours": 1,
        },
    )

    assert answer.status_code == 201, answer.text
    body = answer.json()
    assert body["run"]["id"] == RUN_ID
    assert body["replayed"] is False
    assert body["network_id"] == NET_ID
    # Server-side and per request: a renderer cannot choose it, so a retried
    # request cannot silently reuse another run's key.
    assert body["idempotency_key"]
    assert body["run"].keys().isdisjoint({"run_token", "tunnel"})

    sent = api.radient.last_body()
    assert api.radient.paths() == [("POST", "/v1/remote-compute/runs")]
    assert sent["repo_url"] == ORIGIN
    assert sent["repo_sha"] == _git(directory, "rev-parse", "HEAD").strip()
    assert sent["network_id"] == NET_ID
    assert sent["prompt"] == "fix the failing test"
    assert sent["size"] == "l"
    assert sent["max_hours"] == 1
    assert sent["hosting"] == "openrouter"
    assert sent["model"] == "example/model"
    # The minted key is the one that travelled.
    assert api.radient.calls[-1].headers["Idempotency-Key"] == body["idempotency_key"]

    recorded = records.load(api.root)
    assert [row.run_id for row in recorded] == [RUN_ID]
    assert recorded[0].network_id == NET_ID


@pytest.mark.asyncio
async def test_a_supplied_idempotency_key_replays_the_same_run(api: Api) -> None:
    """The retry that matters is a LOST RESPONSE: the renderer still holds the key.

    Two byte-identical posts under one caller-chosen key: the same key on the wire
    twice, the same body, and the same run id back — a replay (200), not a second
    run (201).
    """
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={"msg": "submitted", "result": {"run": _run(status="pending")}},
    )
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=200,
        payload={"msg": "replayed", "result": {"run": _run(status="pending")}},
    )
    body = {
        "directory": str(directory),
        "prompt": "fix the failing test",
        "tier": "cloud_run",
        "idempotency_key": "renderer-retry-0001",
    }

    first = await api.client.post("/v1/desktop/remote-compute/runs", json=body)
    second = await api.client.post("/v1/desktop/remote-compute/runs", json=body)

    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert first.json()["replayed"] is False
    assert second.json()["replayed"] is True
    assert first.json()["run"]["id"] == second.json()["run"]["id"] == RUN_ID
    assert first.json()["idempotency_key"] == "renderer-retry-0001"
    assert second.json()["idempotency_key"] == "renderer-retry-0001"
    # The pair that makes the second request a replay: same header, same body.
    sent = [
        (call.headers["Idempotency-Key"], json.loads(call.content)) for call in api.radient.calls
    ]
    assert [key for key, _ in sent] == ["renderer-retry-0001", "renderer-retry-0001"]
    assert sent[0][1] == sent[1][1]
    # One local row, folded, not one per attempt.
    assert [row.run_id for row in records.load(api.root)] == [RUN_ID]


@pytest.mark.asyncio
async def test_an_absent_idempotency_key_is_minted_per_request(api: Api) -> None:
    """A caller that does not name a key still gets one — and a fresh one each time."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={"msg": "submitted", "result": {"run": _run(status="pending")}},
    )
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={"msg": "submitted", "result": {"run": _run(OTHER_RUN_ID, status="pending")}},
    )
    body = {"directory": str(directory), "prompt": "p", "tier": "cloud_run"}

    first = await api.client.post("/v1/desktop/remote-compute/runs", json=body)
    second = await api.client.post("/v1/desktop/remote-compute/runs", json=body)

    keys = [first.json()["idempotency_key"], second.json()["idempotency_key"]]
    assert all(keys)
    assert keys[0] != keys[1]
    assert [call.headers["Idempotency-Key"] for call in api.radient.calls] == keys


@pytest.mark.asyncio
async def test_a_key_the_contract_refuses_is_a_400_before_any_request(api: Api) -> None:
    """The library's own ``check_idempotency_key``, not a second copy of its rule."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={
            "directory": str(directory),
            "prompt": "p",
            "tier": "cloud_run",
            "idempotency_key": "not a header\nvalue",
        },
    )

    assert answer.status_code == 400, answer.text
    assert answer.json()["error"]["code"] == "invalid_idempotency_key"
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_replay_answers_200_not_201(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=200,
        payload={"msg": "replayed", "result": {"run": _run()}},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 200
    assert answer.json()["replayed"] is True


@pytest.mark.asyncio
async def test_an_unpushed_head_is_refused_locally(api: Api) -> None:
    """The container clones from the remote; a local-only commit would fail there."""
    api.configure()
    directory = _repository(api.root, pushed=False)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    error = answer.json()["error"]
    assert error["code"] == "unpushed_commit"
    assert "not on any origin branch" in error["message"]
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_an_ssh_only_origin_is_refused_locally(api: Api) -> None:
    api.configure()
    directory = _repository(api.root, origin="git@github.com:example/widget.git")
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    assert answer.json()["error"]["code"] == "not_https_remote"
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_directory_that_is_not_a_repository_is_refused(api: Api) -> None:
    api.configure()
    plain = api.root / "plain"
    plain.mkdir()
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(plain), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    assert answer.json()["error"]["code"] == "no_remote"
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_the_dedicated_tier_is_a_typed_400(api: Api) -> None:
    """Acceping it and failing upstream would be work this build cannot describe."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "dedicated"},
    )

    assert answer.status_code == 400, answer.text
    error = answer.json()["error"]
    assert error["code"] == "tier_not_available"
    assert "Cloud runs" in error["message"]
    # Nothing was sent: the tier is refused before a request is composed.
    assert api.radient.calls == []
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_device_with_no_network_is_refused_before_the_repository_read(
    api: Api,
) -> None:
    """State F: the run cannot be segmented, so nothing is even read locally."""
    api.configure()
    directory = _repository(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    error = answer.json()["error"]
    assert error["code"] == "remote_compute_no_network"
    # The refusal names the remedy, from ``resolve_network``'s own sentence.
    assert "network" in error["message"].lower()
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_an_unconfigured_model_pair_is_refused_with_the_resolvers_sentence(
    api: Api,
) -> None:
    api.configure(model=False)
    directory = _repository(api.root)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    error = answer.json()["error"]
    assert error["code"] == "not_configured"
    assert error["message"] == "Hosting platform is not configured."
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_403_preview_refusal_keeps_the_servers_status(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=403,
        payload={"error": {"code": "preview_not_enabled"}},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 403, answer.text
    error = answer.json()["error"]
    assert error["code"] == "preview_not_enabled"
    assert "preview" in error["message"]


@pytest.mark.asyncio
async def test_a_402_insufficient_credits_refusal(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=402,
        payload={"error": {"code": "insufficient_credits"}},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 402, answer.text
    assert answer.json()["error"]["code"] == "insufficient_credits"


@pytest.mark.asyncio
async def test_a_409_run_limit_is_read_by_its_code_not_its_status(api: Api) -> None:
    """A 409 covers four refusals, and the remedy differs for each."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=409,
        payload={"error": {"code": "run_limit"}},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 409, answer.text
    error = answer.json()["error"]
    assert error["code"] == "run_limit"
    assert "concurrent" in error["message"]


@pytest.mark.asyncio
async def test_a_429_carries_retry_after_in_the_body_and_the_header(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=429,
        payload={"error": {"code": "capacity_unavailable"}},
        headers={"Retry-After": "7"},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 429, answer.text
    assert answer.headers["Retry-After"] == "7"
    error = answer.json()["error"]
    assert error["code"] == "capacity_unavailable"
    assert error["retry_after"] == 7.0
    assert "capacity" in error["message"]


@pytest.mark.asyncio
async def test_a_503_capacity_refusal_keeps_its_status(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.add("POST", "/v1/remote-compute/runs", status=503, payload={"error": {}})

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 503, answer.text
    assert answer.json()["error"]["code"] == "capacity_unavailable"


@pytest.mark.asyncio
async def test_an_unreachable_api_on_create_is_a_502(api: Api) -> None:
    api.configure()
    directory = _repository(api.root)
    _network(api.root)
    api.radient.transport_failure = httpx.ConnectError("no route")

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 502, answer.text
    assert answer.json()["error"]["code"] == "unreachable"


@pytest.mark.asyncio
async def test_an_unknown_body_key_is_a_422(api: Api) -> None:
    """``extra="forbid"``, the rule every desktop request model follows."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": str(directory), "prompt": "p", "tier": "cloud_run", "tierx": 1},
    )

    assert answer.status_code == 422, answer.text
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_an_oversize_prompt_or_directory_is_a_422(api: Api) -> None:
    """The plane's own bounds, refused before a body reaches a proxy and comes back 413."""
    api.configure()
    directory = _repository(api.root)
    _network(api.root)

    long_prompt = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={
            "directory": str(directory),
            "prompt": "x" * (desktop_remote_compute.PROMPT_MAX_CHARS + 1),
            "tier": "cloud_run",
        },
    )
    long_directory = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": "d" * 4097, "prompt": "p", "tier": "cloud_run"},
    )

    assert long_prompt.status_code == 422, long_prompt.text
    assert long_directory.status_code == 422, long_directory.text
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_directory_is_expanded_before_the_repository_read(api: Api) -> None:
    """``~/work`` means the operator's home, as every other cwd field here does.

    Taken raw, ``Path("~/work")`` is not a directory, so ``detect_repo`` reported a
    perfectly good repository as "not a git repository with an origin remote".
    """
    api.configure()
    _repository(api.root)
    _network(api.root)
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={"msg": "submitted", "result": {"run": _run(status="pending")}},
    )

    answer = await api.client.post(
        "/v1/desktop/remote-compute/runs",
        json={"directory": "~/work", "prompt": "p", "tier": "cloud_run"},
    )

    assert answer.status_code == 201, answer.text
    assert api.radient.last_body()["repo_url"] == ORIGIN


# ---------------------------------------------------------------------------
# Cancel and delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_answers_null_when_the_server_sends_no_body(api: Api) -> None:
    api.configure()
    _record(api, RUN_ID)
    api.radient.add("POST", f"/v1/remote-compute/runs/{RUN_ID}/cancel", status=204)

    answer = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/cancel")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"run": None}
    # The library refreshes the record from what it was told.
    assert records.load(api.root)[0].status == "running"


@pytest.mark.asyncio
async def test_cancel_refreshes_the_recorded_status(api: Api) -> None:
    api.configure()
    _record(api, RUN_ID)
    api.radient.add(
        "POST",
        f"/v1/remote-compute/runs/{RUN_ID}/cancel",
        payload={"msg": "cancelled", "result": _run(status="stopping")},
    )

    body = (await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/cancel")).json()

    assert body["run"]["status"] == "stopping"
    assert records.load(api.root)[0].status == "stopping"


@pytest.mark.asyncio
async def test_delete_marks_the_record_and_answers_the_run(api: Api) -> None:
    api.configure()
    _record(api, RUN_ID)
    api.radient.add(
        "DELETE",
        f"/v1/remote-compute/runs/{RUN_ID}",
        payload={"msg": "deleted", "result": _run(status="deleted")},
    )

    body = (await api.client.delete(f"/v1/desktop/remote-compute/runs/{RUN_ID}")).json()

    assert body["run"]["id"] == RUN_ID
    assert body["run"].keys().isdisjoint({"run_token", "tunnel"})
    assert records.load(api.root)[0].deleted is True


@pytest.mark.asyncio
async def test_cancelling_an_unknown_run_keeps_the_servers_404(api: Api) -> None:
    api.configure()
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs/ghost/cancel",
        status=404,
        payload={"error": {"code": "not_found"}},
    )

    answer = await api.client.post("/v1/desktop/remote-compute/runs/ghost/cancel")

    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Nothing secret leaves this surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_bearer_reaches_a_body_or_a_log_line(
    api: Api, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One pass over every route with a live credential and a stubbed Radient."""
    api.configure()
    _network(api.root)
    directory = _repository(api.root)
    api.radient.add("GET", "/v1/remote-compute/capacity", payload=_capacity())
    api.radient.add(
        "GET",
        "/v1/remote-compute/runs",
        payload={"msg": "ok", "result": {"runs": [_run()], "limit": 50, "offset": 0}},
    )
    api.radient.add(
        "POST",
        "/v1/remote-compute/runs",
        status=201,
        payload={"msg": "ok", "result": {"run": _run(), "run_token": "s", "tunnel": {"t": "s"}}},
    )
    api.radient.add("POST", f"/v1/remote-compute/runs/{RUN_ID}/cancel", status=204)
    api.radient.add("DELETE", f"/v1/remote-compute/runs/{RUN_ID}", status=204)

    caplog.set_level(logging.DEBUG)
    bodies: list[str] = []
    for method, path, payload in (
        ("get", "/v1/desktop/remote-compute", None),
        ("get", "/v1/desktop/remote-compute/runs", None),
        ("get", f"/v1/desktop/remote-compute/runs/{RUN_ID}", None),
        (
            "post",
            "/v1/desktop/remote-compute/runs",
            {"directory": str(directory), "prompt": "p", "tier": "cloud_run"},
        ),
        ("post", f"/v1/desktop/remote-compute/runs/{RUN_ID}/cancel", None),
        ("delete", f"/v1/desktop/remote-compute/runs/{RUN_ID}", None),
    ):
        # The detail read is stubbed separately: it is the one path this cell
        # did not queue above.
        if path.startswith("/v1/desktop/remote-compute/runs/"):
            api.radient.add(
                "GET", f"/v1/remote-compute/runs/{RUN_ID}", payload={"msg": "ok", "result": _run()}
            )
        call: Callable[..., Any] = getattr(api.client, method)
        answer = await (call(path) if payload is None else call(path, json=payload))
        bodies.append(answer.text)

    # The bearer really was sent upstream (otherwise this test proves nothing).
    assert any(
        request.headers.get("Authorization") == f"Bearer {RADIENT_ACCESS}"
        for request in api.radient.calls
    )
    assert RADIENT_ACCESS not in "\n".join(bodies)
    assert RADIENT_ACCESS not in caplog.text
    # The create wrapper's two secrets, planted on the stubbed bodies above: a
    # response that carried either would put a pod credential in a GUI transcript.
    assert "PLANTED_RUN_FIELD_ONE" not in "\n".join(bodies)
    assert "PLANTED_TUNNEL_FIELD_TWO" not in "\n".join(bodies)


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_route_is_behind_the_desktop_gate(api: Api) -> None:
    """``require_desktop``, the same dependency every other desktop router carries."""
    paths = [
        ("get", "/v1/desktop/remote-compute"),
        ("get", "/v1/desktop/remote-compute/runs"),
        ("get", f"/v1/desktop/remote-compute/runs/{RUN_ID}"),
        # The create route is listed too: the router-level dependency covers it,
        # but a route that stopped inheriting it would be a hole in the surface
        # rather than in this list.
        ("post", "/v1/desktop/remote-compute/runs"),
        ("post", f"/v1/desktop/remote-compute/runs/{RUN_ID}/cancel"),
        ("delete", f"/v1/desktop/remote-compute/runs/{RUN_ID}"),
        ("get", f"/v1/desktop/remote-compute/runs/{RUN_ID}/logs"),
        ("get", f"/v1/desktop/remote-compute/runs/{RUN_ID}/results"),
        ("post", f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch"),
        ("post", f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/apply"),
    ]
    async with AsyncClient(
        transport=ASGITransport(app=api.app), base_url="http://localhost"
    ) as anonymous:
        for method, path in paths:
            call: Callable[..., Any] = getattr(anonymous, method)
            answer = await call(path)
            assert answer.status_code in {401, 403, 503}, f"{method} {path}: {answer.text}"


# ---------------------------------------------------------------------------
# The run's logs and results: one page, no URLs, and the two writes
# ---------------------------------------------------------------------------


def _log_page(
    *, messages: tuple[str, ...] = (), token: str | None = None, status: str = "running"
) -> dict[str, Any]:
    """A log page as the server answers it, inside the usual envelope."""
    events = [{"timestamp": "2026-11-17T00:00:00Z", "message": text} for text in messages]
    return {"msg": "Logs", "result": {"events": events, "next_token": token, "run_status": status}}


def _bundle_for_run(root: Path) -> tuple[Path, bytes]:
    """A clone holding the run's base commit, and a bundle of the run's own work.

    THE CLONE IS MADE FIRST, deliberately: it must be a repository that has the
    base commit and has never seen the run's branch — which is exactly what an
    operator's working clone looks like when they apply a bundle.
    """
    source = root / "bundle-source"
    source.mkdir()
    _git(source, "init", "--quiet", "--initial-branch=main")
    (source / "README.md").write_text("hello\n")
    _git(source, "add", "README.md")
    _git(source, "commit", "--quiet", "-m", "base")
    clone = root / "bundle-clone"
    _git(root, "clone", "--quiet", str(source), str(clone))
    _git(source, "branch", f"lop/{RUN_ID}", "main")
    _git(source, "checkout", "--quiet", f"lop/{RUN_ID}")
    (source / "work.txt").write_text("the run's work\n")
    _git(source, "add", "work.txt")
    _git(source, "commit", "--quiet", "-m", "the run's commit")
    _git(source, "checkout", "--quiet", "main")
    bundle = root / "repo.bundle"
    _git(source, "bundle", "create", str(bundle), f"main..lop/{RUN_ID}")
    return clone, bundle.read_bytes()


def _repository_state(clone: Path) -> dict[str, str]:
    """Everything the operator's clone must still look like after an apply."""
    return {
        "head": _git(clone, "rev-parse", "HEAD"),
        "branch": _git(clone, "symbolic-ref", "HEAD"),
        "branches": _git(clone, "for-each-ref", "refs/heads"),
        "status": _git(clone, "status", "--porcelain"),
        "index": _git(clone, "ls-files", "--stage"),
    }


def _session_archive(session_id: str = SESSION_ID) -> bytes:
    """A results archive holding one session directory, gzipped."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, payload in (
            (f"sessions/{session_id}/transcript.jsonl", b'{"role":"user"}\n'),
            (f"sessions/{session_id}/meta.json", b"{}\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _serve(payload: bytes, *, status: int = 200) -> tuple[httpx.MockTransport, list[Any]]:
    """A presigned-download transport that answers ``payload`` and records requests."""
    seen: list[Any] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=payload)

    return httpx.MockTransport(handle), seen


def _install_download(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    """Point the presigned-download seam at ``transport`` for one cell."""

    def build() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, follow_redirects=False)

    monkeypatch.setattr(results, "artifact_client", build)


def _artifacts(*rows: dict[str, Any], git: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "msg": "Artifacts",
        "result": {"run_status": "stopped", "artifacts": list(rows), "git": git},
    }


def _artifact_row(name: str, size: int) -> dict[str, Any]:
    """One artifact, with a presigned URL on the loopback host the check allows."""
    return {
        "name": name,
        "size_bytes": size,
        "url": f"http://127.0.0.1:9/{name}?X-Amz-Signature={SIGNATURE}",
        "expires_at": "2026-11-17T00:05:00Z",
    }


@pytest.mark.asyncio
async def test_a_log_page_is_passed_through_with_its_token(api: Api) -> None:
    api.configure()
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/logs",
        payload=_log_page(messages=("cloning",), token="tok-2"),
    )

    body = (
        await api.client.get(
            f"/v1/desktop/remote-compute/runs/{RUN_ID}/logs",
            params={"next_token": "tok-1", "limit": 50},
        )
    ).json()

    assert body["run_status"] == "running"
    assert body["next_token"] == "tok-2"
    assert [event["message"] for event in body["events"]] == ["cloning"]
    assert api.radient.calls[-1].url.params["next_token"] == "tok-1"
    assert api.radient.calls[-1].url.params["limit"] == "50"
    # The account's bearer travels to Radient, as it must.
    assert api.radient.calls[-1].headers["authorization"].endswith(RADIENT_ACCESS)


@pytest.mark.asyncio
async def test_a_deployment_where_logs_are_not_readable_answers_its_own_code(api: Api) -> None:
    """``not_configured`` is a definite answer: the panel says logs are not offered."""
    api.configure()
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/logs",
        status=503,
        payload={"error": {"code": "not_configured"}},
    )

    answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}/logs")

    assert answer.status_code == 503, answer.text
    assert answer.json()["error"]["code"] == "not_configured"


@pytest.mark.asyncio
async def test_the_results_listing_carries_no_download_urls(api: Api) -> None:
    """A presigned URL is a bearer: the panel gets names and sizes, never the URL."""
    api.configure()
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(
            _artifact_row("results.tar.gz", 1234),
            git={"branch": "lop/run", "commit": "c0ffee", "bundle_present": True},
        ),
    )

    answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}/results")

    assert answer.status_code == 200, answer.text
    assert SIGNATURE not in answer.text
    assert "127.0.0.1" not in answer.text
    body = answer.json()
    assert body["run_status"] == "stopped"
    assert body["artifacts"] == [
        {"name": "results.tar.gz", "size_bytes": 1234, "expires_at": "2026-11-17T00:05:00Z"}
    ]
    assert body["git"] == {"branch": "lop/run", "commit": "c0ffee", "bundle_present": True}


@pytest.mark.asyncio
async def test_the_flag_off_refuses_logs_and_results_before_any_request(api: Api) -> None:
    """The switch is a consent gate: no log read and no listing leaves the device."""
    api.configure(enabled=False)

    for path in ("logs", "results"):
        answer = await api.client.get(f"/v1/desktop/remote-compute/runs/{RUN_ID}/{path}")
        assert answer.status_code == 403, answer.text
        assert answer.json()["error"]["code"] == "not_configured"

    fetch = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch")
    assert fetch.status_code == 403, fetch.text
    assert api.radient.calls == []


@pytest.mark.asyncio
async def test_a_fetch_extracts_the_results_and_offers_the_session(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.configure()
    payload = _session_archive()
    transport, downloads = _serve(payload)
    _install_download(monkeypatch, transport)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(_artifact_row("results.tar.gz", len(payload))),
    )

    answer = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch")

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["file_count"] == 2
    assert body["archive_bytes"] == len(payload)
    assert body["session_id"] == SESSION_ID
    assert body["session_restored"] is True
    # The default directory is the app's own, under the config root of THIS daemon.
    assert body["directory"] == str(api.root / "remote-compute" / "results" / RUN_ID)
    assert (api.root / "sessions" / SESSION_ID / "transcript.jsonl").is_file()
    # The download went out WITHOUT the account's bearer, on its own client.
    assert "authorization" not in {name.lower() for name in downloads[0].headers}
    assert RADIENT_ACCESS not in answer.text
    assert SIGNATURE not in answer.text


@pytest.mark.asyncio
async def test_a_fetch_with_an_unsafe_archive_is_a_502_naming_the_reason(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive that would escape the destination is refused, with its code."""
    api.configure()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        payload = b"escaped\n"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    transport, _downloads = _serve(buffer.getvalue())
    _install_download(monkeypatch, transport)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(_artifact_row("results.tar.gz", len(buffer.getvalue()))),
    )

    answer = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch")

    assert answer.status_code == 502, answer.text
    assert answer.json()["error"]["code"] == "outside_destination"
    assert not (api.root / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_a_fetch_whose_body_is_the_wrong_size_is_a_502(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.configure()
    payload = _session_archive()
    transport, _downloads = _serve(payload[:-5])
    _install_download(monkeypatch, transport)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(_artifact_row("results.tar.gz", len(payload))),
    )

    answer = await api.client.post(f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch")

    assert answer.status_code == 502, answer.text
    assert answer.json()["error"]["code"] == "results_size_mismatch"
    assert not (api.root / "remote-compute" / "results" / RUN_ID).exists()


@pytest.mark.asyncio
async def test_the_fetch_body_refuses_an_unknown_field(api: Api) -> None:
    api.configure()

    answer = await api.client.post(
        f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/fetch", json={"dest": "/tmp/x"}
    )

    assert answer.status_code == 422, answer.text


@pytest.mark.asyncio
async def test_an_apply_creates_the_ref_and_leaves_the_checkout_alone(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.configure()
    clone, bundle = _bundle_for_run(api.root)
    before = _repository_state(clone)
    transport, _downloads = _serve(bundle)
    _install_download(monkeypatch, transport)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(_artifact_row("repo.bundle", len(bundle))),
    )

    answer = await api.client.post(
        f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/apply",
        json={"directory": str(clone)},
    )

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["ref"] == f"refs/remotes/radient/lop/{RUN_ID}"
    assert _git(clone, "rev-parse", body["ref"]).strip() == body["commit"]
    assert _repository_state(clone) == before


@pytest.mark.asyncio
async def test_an_apply_to_a_clone_without_the_base_commit_is_a_409(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.configure()
    _source, bundle = _bundle_for_run(api.root)
    unrelated = api.root / "unrelated"
    unrelated.mkdir()
    _git(unrelated, "init", "--quiet", "--initial-branch=main")
    (unrelated / "x.txt").write_text("x\n")
    _git(unrelated, "add", "x.txt")
    _git(unrelated, "commit", "--quiet", "-m", "unrelated")
    transport, _downloads = _serve(bundle)
    _install_download(monkeypatch, transport)
    api.radient.add(
        "GET",
        f"/v1/remote-compute/runs/{RUN_ID}/artifacts",
        payload=_artifacts(_artifact_row("repo.bundle", len(bundle))),
    )

    answer = await api.client.post(
        f"/v1/desktop/remote-compute/runs/{RUN_ID}/results/apply",
        json={"directory": str(unrelated)},
    )

    assert answer.status_code == 409, answer.text
    assert answer.json()["error"]["code"] == "bundle_invalid"
    assert not (unrelated / ".git" / "refs" / "remotes" / "radient").exists()

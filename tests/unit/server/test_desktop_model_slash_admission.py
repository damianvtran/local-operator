"""The desktop ``/model`` command path pays the owner-support admission (R1-1).

WHY THIS FILE EXISTS. The desktop picker dispatches a choice as
``command.run("model", value)`` -> ``POST /v1/desktop/sessions/{id}/commands``,
and that route forwards the slash to the session's owner through
``route_shared_slash``. The owner cannot be relied on to refuse a provider it
cannot serve: a pre-Muse owner's ``_model_slash`` predates
``require_owner_provider`` entirely and reaches ``build_model_spec``, which
builds an unknown-provider spec without complaint. So the VIEWER runs the same
admission the facade's own switch runs, BEFORE the slash crosses the wire.

These cells pin both halves of that contract on the REAL route over a real
facade shell: a stale owner gets the typed ``runtime_upgrade_needed`` refusal
with zero owner RPCs (nothing that could mutate selection, spec or journal ever
reaches the owner), a bound positive owner still forwards, and the non-switch
forms (``/model saved``) remain the owner's to answer.

Every case is synthetic: fixture records, fixture settings, an isolated config
root, and a stub connect-state — no socket, no credential, no provider call.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.mobile.attach_client import AttachClient
from local_operator.model import configure
from local_operator.server.routes import desktop_sessions

TOKEN = "desktop-model-slash-admission-token"
SESSION = "synthetic-desktop-owner"
REQUEST_ID = "33333333-3333-4333-8333-333333333333"

_PRIMARY = configure.ModelSpec(provider="openai", model_id="fixture-old", context_window=128000)


class _SlashClient(AttachClient):
    """The attach client's connect state and its ONE mutating RPC, recorded."""

    def __init__(self) -> None:
        # This seam records owner RPC admission, never opens a socket; the
        # connect bit is what route_shared_slash checks before forwarding.
        self.calls: list[tuple[str, str]] = []
        self._connected = True

    async def slash_result(
        self, command: str, args: str, images: list[dict[str, str]] | None = None
    ) -> Any:
        self.calls.append((command, args))
        return {
            "kind": "notice",
            "text": f"model: fixture-old \u2192 {args} (this session)",
            "style": "info",
        }


def _facade(*, version: str, native: list[str] | None = None) -> Any:
    """A real facade shell with the REAL ``route_shared_slash`` and admission.

    ``model``/``effective_model`` are properties (the constant primary), like
    the sibling facade regression: only the actual owner can mutate selection,
    so the shell refusing without an RPC IS the no-mutation proof.
    """
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.types import SessionRecord

    class _Facade(AttachedSession):
        @property
        def model(self) -> configure.ModelSpec:
            return _PRIMARY.model_copy()

        @property
        def effective_model(self) -> configure.ModelSpec:
            return _PRIMARY.model_copy()

        def runtime_model_catalogue(self) -> list[dict[str, Any]]:
            # Truncated/absent inventory is not a negative support claim.
            return []

        def _read_state_field(self, name: str) -> Any:
            # The real store serves a copy-free allow-list only and REFUSES
            # non-scalar fields like ``model_catalogue`` by raising. Mirror
            # that refusal: a lenient double is how the routed arm's misuse of
            # this accessor slipped past these cells while the real store
            # raised (CI's fork e2e, round 1).
            raise KeyError(f"{name!r} is not a copy-free state field")

        async def bind_runtime(self) -> None:
            # The route's explicit _ensure_bound; this shell is already bound.
            return None

    facade = object.__new__(_Facade)
    facade._can_go_cold = False
    facade._recovering = False
    facade._deliberate_stop = False
    facade._client = _SlashClient()
    facade._runtime_record = SessionRecord(
        pid=12345,
        kind="daemon",
        session_id=SESSION,
        conversation_name="fixture",
        cwd="synthetic-workspace",
        model_label="openai/fixture-old",
        control_port=0,
        control_key="synthetic-unused-control",
        version=version,
        native_provider_ids=native,
    )
    return facade


class _Bridge:
    def __init__(self, remote: Any) -> None:
        self.remote = remote
        self.ingested: list[list[dict[str, str]]] = []

    async def refresh_watch(self) -> None:
        return None

    async def stage_ingested_images(self, payloads: list[dict[str, str]]) -> None:
        self.ingested.append(payloads)


class _Pool:
    """``DesktopSessions``-shaped: the route's only door to a session."""

    def __init__(self, remote: Any) -> None:
        self.remote = remote
        self.bridge = _Bridge(remote)

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False):
        # Accepted and ignored like the sibling pool doubles: the door is the
        # route's only entry, and a READ envelope is not what these cells ask.
        del read
        assert session_id == SESSION
        yield self.bridge


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The REAL command router over a facade shell and an isolated config root."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    facade = _facade(version="old-fixture")
    app.state.desktop_sessions = _Pool(facade)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, facade


async def _model(client: AsyncClient, args: str):
    return await client.post(
        f"/v1/desktop/sessions/{SESSION}/commands",
        json={
            "request_id": REQUEST_ID,
            "command": "model",
            "args": args,
            "images": [],
        },
    )


@pytest.mark.asyncio
async def test_a_stale_owner_refuses_the_forwarded_switch_before_any_owner_rpc(desktop) -> None:
    """The reported R1-1 hole: the pick lands on the owner with no admission.

    ``old-fixture`` is a version no installed build matches, and the record
    advertises no native inventory, so ``require_owner_provider`` must refuse
    with the typed code — and the refusal must fire HERE, on this side, because
    the owner running that version has no gate at all.
    """
    client, facade = desktop
    response = await _model(client, "meta-muse/muse-fixture")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "runtime_upgrade_needed", detail
    assert "saved sign-in is unchanged" in detail["text"], detail
    # Nothing reached the owner: no set_model/slash RPC, no selection change.
    assert facade._client.calls == []
    assert facade.model.provider == "openai"


@pytest.mark.asyncio
async def test_a_bound_positive_owner_still_forwards_the_switch(desktop) -> None:
    """A current owner's own advert is positive evidence; the pick is served."""
    client, facade = desktop
    facade._runtime_record = facade._runtime_record.__class__(
        **{
            **facade._runtime_record.__dict__,
            "version": "new-fixture",
            "native_provider_ids": ["meta-muse"],
        }
    )
    response = await _model(client, "meta-muse/muse-fixture")
    assert response.status_code == 200, response.text
    result = response.json()["result"]["result"]
    assert result["kind"] == "notice"
    assert facade._client.calls == [("model", "meta-muse/muse-fixture")]


@pytest.mark.asyncio
async def test_non_switch_forms_are_the_owners_to_answer(desktop) -> None:
    """``/model saved`` is a READ the owner answers; the gate must not over-refuse.

    The owner's parse answers ``saved``/``default``/a bare target itself, so a
    stale owner keeps its existing behaviour for those forms and the admission
    applies only where a provider switch is actually being attempted.
    """
    client, facade = desktop
    response = await _model(client, "saved")
    assert response.status_code == 200, response.text
    assert facade._client.calls == [("model", "saved")]

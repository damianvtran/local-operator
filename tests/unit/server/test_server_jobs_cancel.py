"""``POST /v1/desktop/sessions/{id}/jobs/{job_id}/cancel``: the job rail's write.

The route's whole job is WHICH CALL it reaches and WHAT it answers when it
cannot reach one, so that is what this file pins:

* a live, answering owner receives ``route_shared_slash("jobs", …)`` and the
  cancel lands on the owner's REAL ``AsyncJobManager`` — the same call
  ``jobs op='cancel'`` makes — with the row settling ``cancelled`` and the
  runner's abort signal fired;
* an unknown id answers 404 carrying the frozen ``not_found`` phrase from the
  engine's wording map, authored rather than left for a client to substitute
  (design spec §2.2: "the frozen ``not_found``-class phrase, never a generic
  error");
* a job that settled first refuses with the tool's own sentence (409);
* no owner, an owner that answers nothing, and an owner that does not confirm
  the cancel are all retryable 503s — a cancel that did not happen is never
  reported as one that did;
* the route is authenticated like its siblings.

Driven over HTTP against a real app over the REAL pool and its real bridge;
the only stood-in part is the attach socket's far end (the owner connection),
whose slash hop is played against the real owner-side handler and a real
manager. The serving half's own unit lives in
``tests/unit/session/runtime/test_serving_jobs_command.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.harness.jobs import AsyncJobManager
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle

PARENT_ID = "0123456789ab"
TOKEN = "jobs-cancel-token"
FROZEN_NOT_FOUND = "not cancelled — the provider no longer knows the job"

#: Sentinel: "play the real owner-side handler" rather than answering with a
#: canned envelope.
_REAL = object()


def a_session(root: Path, session_id: str = PARENT_ID) -> Path:
    """A conversation directory the pool can locate."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    return directory


class _Session:
    """The one thing the owner-side word reads: the manager."""

    def __init__(self, manager: AsyncJobManager | None = None) -> None:
        self.jobs = manager


class JobsOwner:
    """The owner connection's slash hop, played against the REAL word.

    ``AttachedSession.route_shared_slash`` sends the word over the attach
    socket; the runtime's dispatch then runs ``ServingSessionHandle``'s
    ``_jobs_slash`` against the session that owns the manager. Standing up a
    socket and a process to reach one dispatch table would test the transport,
    so this plays the hop: the real handler runs here, against the real
    manager, and answers in the runtime's own envelope shape. ``answer`` is
    the canned reply for the tests that need the transport's OTHER states
    (nothing at all; an envelope from a build that does not know the word).
    """

    connected = True

    def __init__(self, session: _Session | None = None, *, answer: Any = _REAL) -> None:
        self.session = session
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def slash_result(self, command: str, args: str, images: list[Any] | None = None) -> Any:
        self.calls.append((command, args))
        if self.answer is not _REAL:
            return self.answer
        handle = ServingSessionHandle.__new__(ServingSessionHandle)
        result = await handle._jobs_slash(self.session, args, SlashResult)
        return result.model_dump(mode="json")

    def close(self) -> None:
        return None


def _running_job(manager: AsyncJobManager, observed: list[bool]) -> tuple[str, asyncio.Event]:
    """A synthetic detached job that runs until cancelled; ``observed`` records
    whether the runner saw its abort signal fired before the task died, and the
    ``started`` event lets the test wait until the runner is live (the cancel's
    never-entered teardown path is the manager's own, exercised elsewhere)."""
    started = asyncio.Event()

    async def runner(job_id: str, signal: Any, progress: Any) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            observed.append(signal.aborted)

    return manager.register("bash", "synthetic job", runner), started


def _row(manager: AsyncJobManager, job_id: str) -> Any:
    """The job's row, asserted present — ``get`` is optional by signature."""
    job = manager.get(job_id)
    assert job is not None, f"job {job_id} is not on the roster"
    return job


async def _settle(manager: AsyncJobManager, job_id: str) -> None:
    for _ in range(200):
        if _row(manager, job_id).status != "running":
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the synthetic job did not settle")


@pytest.fixture
def desktop_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_sessions.router)
    return app, pool


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


@pytest.mark.asyncio
async def test_a_live_press_reaches_the_manager_and_settles_the_row(desktop_app, tmp_path) -> None:
    """The whole chain, one process: HTTP press → the route → the owner
    connection's slash hop → the real word → ``AsyncJobManager.cancel``."""
    app, pool = desktop_app
    a_session(tmp_path)
    manager = AsyncJobManager()
    observed: list[bool] = []
    job_id, started = _running_job(manager, observed)
    await asyncio.wait_for(started.wait(), timeout=5.0)

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(_Session(manager))  # type: ignore[assignment]
            response = await client.post(f"/v1/desktop/sessions/{PARENT_ID}/jobs/{job_id}/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"job_id": job_id, "cancelled": True}
    # The manager RECEIVED the cancel: the row settled and the runner's abort
    # signal fired — the mechanism the video runner's best_effort_cancel reads.
    assert _row(manager, job_id).status == "cancelled"
    assert observed == [True]


@pytest.mark.asyncio
async def test_an_unknown_id_answers_the_frozen_not_found_phrase(desktop_app, tmp_path) -> None:
    app, pool = desktop_app
    a_session(tmp_path)
    manager = AsyncJobManager()

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(_Session(manager))  # type: ignore[assignment]
            response = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel"
            )

    assert response.status_code == 404, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_not_found"
    assert detail["message"] == FROZEN_NOT_FOUND
    assert detail["session_id"] == PARENT_ID


@pytest.mark.asyncio
async def test_a_settled_job_refuses_with_the_tools_own_sentence(desktop_app, tmp_path) -> None:
    app, pool = desktop_app
    a_session(tmp_path)
    manager = AsyncJobManager()

    async def runner(job_id: str, signal: Any, progress: Any) -> str:
        return "done"

    job_id = manager.register("bash", "quick", runner)
    await _settle(manager, job_id)

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(_Session(manager))  # type: ignore[assignment]
            response = await client.post(f"/v1/desktop/sessions/{PARENT_ID}/jobs/{job_id}/cancel")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_not_cancelled"
    assert detail["message"] == f"job {job_id} was not cancelled (status: completed)"


@pytest.mark.asyncio
async def test_no_live_owner_is_a_retryable_503(desktop_app, tmp_path) -> None:
    """No runtime ⇒ no manager ⇒ nothing to cancel. Unlike the wake/monitor
    surfaces there is no cold path to fall back to (their state persists; a
    job's does not), so the honest answer is a refusal, never a write."""
    app, pool = desktop_app
    a_session(tmp_path)

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            assert bridge.remote is not None  # cold, and no client installed
            response = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel"
            )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_owner_unavailable"
    assert "no live runtime" in detail["message"]


@pytest.mark.asyncio
async def test_an_owner_that_answers_nothing_is_a_retryable_503(desktop_app, tmp_path) -> None:
    app, pool = desktop_app
    a_session(tmp_path)

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(answer=None)  # type: ignore[assignment]
            response = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel"
            )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_owner_unavailable"
    assert "answered nothing" in detail["message"]


@pytest.mark.asyncio
async def test_an_owner_without_a_cancel_receipt_is_never_a_200(desktop_app, tmp_path) -> None:
    """An owner that does not know the word falls through its dispatch to a
    configuration notice — a non-error envelope with no job id. Reporting that
    as a receipt would be a silently dead cancel with a 2xx; the route treats
    it as "no receipt" instead."""
    app, pool = desktop_app
    a_session(tmp_path)
    envelope = {"kind": "notice", "text": "reads this machine's configuration", "data": {}}

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(answer=envelope)  # type: ignore[assignment]
            response = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel"
            )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_owner_unavailable"
    assert "did not confirm" in detail["message"]


@pytest.mark.asyncio
async def test_an_owner_refusal_keeps_the_runtime_s_own_sentence(desktop_app, tmp_path) -> None:
    """One wording for one mistake: the CODE decides the status, and the
    runtime's sentence is what the client renders (the wake/monitor routes'
    own rule) — an owner too old to have a manager answers ``jobs_unavailable``
    with the tool's sentence, and the route must not re-word either."""
    app, pool = desktop_app
    a_session(tmp_path)
    sentence = "job tracking is not available in this session (no job manager attached)."
    envelope = {"kind": "error", "text": sentence, "data": {"code": "jobs_unavailable"}}

    async with _client(app) as client:
        async with pool.session(PARENT_ID, read=True) as bridge:
            bridge.remote._client = JobsOwner(answer=envelope)  # type: ignore[assignment]
            response = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel"
            )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "jobs_unavailable"
    assert detail["message"] == sentence


@pytest.mark.asyncio
async def test_the_route_is_authenticated(desktop_app, tmp_path) -> None:
    app, _pool = desktop_app
    a_session(tmp_path)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer wrong"},
    ) as client:
        response = await client.post(f"/v1/desktop/sessions/{PARENT_ID}/jobs/aaaaaaaaaaaa/cancel")

    assert response.status_code in (401, 403)

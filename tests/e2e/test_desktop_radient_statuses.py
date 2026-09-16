"""The batched status read, against a labelled fake Radient upstream over real loopback HTTP.

`agents.statuses` exists so the desktop hub can ask ONE question about a page of
agents instead of two per card: `GET /agents/{id}/like` and
`GET /agents/{id}/favourite` answer for one agent each, and a twelve-card page
served from the renderer cost twenty-four round trips, then a second wave after
first paint, then another on every window focus.

What this file proves, and what it deliberately does not:

- It proves the batch is ONE renderer request whose upstream calls are exactly
  the two status reads named for each requested id, that the answer is keyed by
  id with every requested id present, that an empty body is read as "no
  relation" (the upstream's own encoding of not-liked), that a per-agent
  refusal costs that one id and not the batch, and that the op's refusals (no
  ids, too many ids, a malformed id, no stored credential, a credential the
  upstream rejects) are the four statuses the UI is written against.
- It does NOT prove anything about the real Radient account behind a real
  credential: the upstream here is a labelled fake, so the like/favourite
  DOCUMENTS it serves are fixtures. The empty-body convention it impersonates is
  `GetAgentLikeHandlerEcho`'s (`c.NoContent(http.StatusOK)` when the account
  holds no like), read from agent-server's source rather than from a live call.
"""

import asyncio
import secrets
import socket
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from local_operator.server.app import app
from tests.e2e.test_desktop_controls import until

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def serve(application):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(application, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await until(lambda: server.started)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)
        listener.close()


def fixture_upstream(calls, access):
    """A fake Radient that records every path it is asked for.

    `/like` and `/favourite` answer the way the real handlers do: a document
    when the account holds the relation, and a 200 with NO body when it does
    not. `gone` is the agent delisted while the page was open; `redirect` is an
    upstream that answers with a redirect rather than a document.
    """
    fake = FastAPI()

    @fake.get("/v1/agents/{agent_id}/{relation}")
    async def status(agent_id: str, relation: str, request: Request):
        calls.append(("GET", f"agents/{agent_id}/{relation}"))
        if relation not in {"like", "favourite"}:
            return JSONResponse({"error": "not found"}, status_code=404)
        if request.headers.get("authorization") != "Bearer " + access:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if agent_id == "gone":
            return JSONResponse({"error": "agent not found"}, status_code=404)
        if agent_id == "redirect":
            return JSONResponse({}, status_code=302, headers={"Location": "https://example.org"})
        # The relation is held when the id's last character says so, which makes
        # every id in one request answer differently from its neighbours.
        if agent_id.endswith("1") or (relation == "favourite" and agent_id.endswith("2")):
            return JSONResponse({"msg": f"{relation} found", "result": {"subject_id": agent_id}})
        return Response(status_code=200)

    return fake


def store_radient_credential(access, refresh):
    """Persist a Radient credential the way a completed sign-in would.

    Written to the app's own store rather than through a route, because the
    sign-in routes need a real provider round trip; what these tests need is the
    state those routes leave behind.
    """
    app.state.desktop_auth.store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": access,
            "refresh": refresh,
            "expires": int(time.time() * 1000) + 3_600_000,
        },
    )


@pytest.mark.asyncio
async def test_one_request_reads_the_whole_page_in_two_calls_per_agent(
    headless_tui_env, monkeypatch
):
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    fake = fixture_upstream(calls, access)

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            # The first request is what builds the app's own auth state; the
            # credential is written after it, exactly as a completed sign-in
            # would leave it.
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            route = "/v1/desktop/radient"
            # Twelve ids, which is the hub's page size: the case the fan-out is
            # measured on.
            ids = [f"agent-{index}" for index in range(12)]

            # One HTTP request to the desktop plane...
            response = await client.post(
                route, json={"operation": "agents.statuses", "query": {"agent_ids": ",".join(ids)}}
            )
            assert response.status_code == 200, response.text

            # ...and exactly two upstream reads per id, and nothing else: no
            # list, no count, no per-agent document.
            assert sorted(calls) == sorted(
                ("GET", f"agents/{agent_id}/{relation}")
                for agent_id in ids
                for relation in ("like", "favourite")
            ), calls

            statuses = response.json()["result"]["data"]["result"]["statuses"]
            # Every requested id is present, keyed by id, so a caller never pairs
            # a positional list with its input.
            assert list(statuses) == ids
            for agent_id in ids:
                assert statuses[agent_id] == {
                    "liked": agent_id.endswith("1"),
                    "favourited": agent_id.endswith("1") or agent_id.endswith("2"),
                }, agent_id

            # A duplicate id is one agent's state, not four upstream reads.
            before = len(calls)
            duplicated = await client.post(
                route,
                json={
                    "operation": "agents.statuses",
                    "query": {"agent_ids": "agent-1,agent-1,agent-1"},
                },
            )
            assert duplicated.status_code == 200
            assert len(calls) - before == 2, calls[before:]
            assert duplicated.json()["result"]["data"]["result"]["statuses"] == {
                "agent-1": {"liked": True, "favourited": True}
            }

            # An agent delisted while the page was open costs that id and
            # nothing else: the batch still answers, and the absent relation is
            # the state whose toggle is still correct.
            before = len(calls)
            mixed = await client.post(
                route,
                json={"operation": "agents.statuses", "query": {"agent_ids": "gone,agent-3"}},
            )
            assert mixed.status_code == 200, mixed.text
            assert len(calls) - before == 4
            assert mixed.json()["result"]["data"]["result"]["statuses"] == {
                "gone": {"liked": False, "favourited": False},
                "agent-3": {"liked": False, "favourited": False},
            }

            # A redirect is an upstream that did not answer with a document: 502,
            # the same classification the rest of this route uses for it.
            redirected = await client.post(
                route, json={"operation": "agents.statuses", "query": {"agent_ids": "redirect"}}
            )
            assert redirected.status_code == 502, redirected.text


@pytest.mark.asyncio
async def test_the_status_batch_refuses_before_it_calls_anything(headless_tui_env, monkeypatch):
    """The four refusals the UI is written against, and the calls they do not make."""
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    fake = fixture_upstream(calls, access)

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            # The first request is what builds the app's own auth state; the
            # credential is written after it, exactly as a completed sign-in
            # would leave it.
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            route = "/v1/desktop/radient"

            # 409 when no Radient credential is stored: sign in, not "no state".
            app.state.desktop_auth.store.delete_credentials_for_provider("radient")
            signed_out = await client.post(
                route, json={"operation": "agents.statuses", "query": {"agent_ids": "agent-1"}}
            )
            assert signed_out.status_code == 409, signed_out.text

            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)

            # A malformed id never becomes a path segment, an empty list is a
            # question with no subject, and the batch has a bound. All three are
            # refusals, and none of them reaches Radient.
            refused = [
                {"agent_ids": ""},
                {"agent_ids": "agent-1,../../etc/passwd"},
                {"agent_ids": "agent-1," + ",".join(f"agent-{i}" for i in range(40))},
            ]
            before = len(calls)
            for query in refused:
                response = await client.post(
                    route, json={"operation": "agents.statuses", "query": query}
                )
                assert response.status_code == 422, (query, response.text)
            assert len(calls) == before, calls[before:]

            # A credential Radient itself refuses is the batch's answer, not
            # twelve pages of "nothing known".
            app.state.desktop_auth.store.delete_credentials_for_provider("radient")
            store = app.state.desktop_auth.store
            store.upsert_credential(
                "radient",
                {
                    "type": "oauth",
                    "access": "not-the-fixture-token",
                    "refresh": refresh,
                    "expires": int(time.time() * 1000) + 3_600_000,
                },
            )
            refused_batch = await client.post(
                route,
                json={"operation": "agents.statuses", "query": {"agent_ids": "agent-1,agent-2"}},
            )
            assert refused_batch.status_code == 401, refused_batch.text
            assert "not-the-fixture-token" not in refused_batch.text

            # And the op is a read: it needs no request id and no confirmation,
            # which is what reaching the upstream at all shows.
            assert (
                await client.post(
                    route,
                    json={"operation": "agents.statuses", "query": {"agent_ids": "agent-1"}},
                )
            ).status_code == 401

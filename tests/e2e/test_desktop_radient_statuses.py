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
- It also pins the CLASSIFICATION BOUNDARY the batch shares with the single-agent
  ops (an upstream failure is never rendered as an absence, and only a 404/410 is
  an absence), the overall deadline that bounds the whole fan-out, the wave stop
  that keeps a refused credential from costing all 2N reads, the upstream body
  ceiling, and the router's own token gate.
- It does NOT prove anything about the real Radient account behind a real
  credential: the upstream here is a labelled fake, so the like/favourite
  DOCUMENTS it serves are fixtures. The empty-body convention it impersonates is
  `GetAgentLikeHandlerEcho`'s (`c.NoContent(http.StatusOK)` when the account
  holds no like), read from agent-server's source rather than from a live call —
  so the convention is exercised here, and the live service remains QA's half.
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


def controlled_upstream(calls, state):
    """A fake Radient whose every status read is steered by ``state``.

    ``state`` is MUTABLE on purpose: one boot of the desktop plane then exercises
    a sequence of upstream behaviours — an answer, a stall, a hang, an oversized
    body, a refusal — instead of paying a server start for each. Keys:
    ``status`` (the answer for every read, 200 by default), ``delay`` (seconds to
    stall before answering) and ``size`` (answer 200 with a body of that many
    bytes, for the ceiling).
    """
    fake = FastAPI()

    @fake.get("/v1/agents/{agent_id}/{relation}")
    async def status(agent_id: str, relation: str):
        calls.append(("GET", f"agents/{agent_id}/{relation}"))
        if state.get("delay"):
            await asyncio.sleep(state["delay"])
        if state.get("size"):
            return Response(content=b"x" * state["size"], media_type="application/json")
        if state.get("status", 200) != 200:
            return JSONResponse({"error": "upstream"}, status_code=state["status"])
        return JSONResponse({"msg": f"{relation} found", "result": {"subject_id": agent_id}})

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


#: Every upstream answer the classification has to place, and what each means to
#: the batch and to the single-agent ops it shares its mapping with: the renderer
#: status, the machine code, and whether the batch answers rather than fails (the
#: ABSENCE class — 404 and its 410 sibling — is the only one that does).
UPSTREAM_CLASSES = (
    (200, 200, None, True),
    (400, 400, "radient_upstream_failed", False),
    (401, 401, "radient_credential_refused", False),
    (403, 403, "radient_credential_refused", False),
    (404, 200, None, True),
    (409, 409, "radient_upstream_failed", False),
    (410, 200, None, True),
    (422, 422, "radient_upstream_failed", False),
    (429, 429, "radient_credential_refused", False),
    (500, 502, "radient_upstream_failed", False),
    (502, 502, "radient_upstream_failed", False),
    (503, 502, "radient_upstream_failed", False),
    (504, 502, "radient_upstream_failed", False),
)


@pytest.mark.asyncio
async def test_each_upstream_answer_is_classified_like_the_single_agent_ops(
    headless_tui_env, monkeypatch
):
    """A failure is never an absence, and only 404/410 is an absence.

    Round 1 (R-1) executed this against the same fake upstream: the batch
    answered 200/all-false for an upstream 500, 503 and 404 while `agents.liked`
    answered 502, 502 and 404 for the same answers. A Radient outage, or a renamed
    upstream path, therefore rendered as a page of unliked cards with no error
    raised anywhere in either repository — permanently, and silently.

    The comparison is the point, so the single-agent op is asked the same question
    here: apart from the absence class (where the two deliberately differ — the
    batch is asked about a PAGE, so an agent the viewer cannot see is one card's
    "no relation", while `agents.liked` is asked about one agent and answers 404),
    the two must agree on the status.
    """
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    state: dict[str, object] = {}
    fake = controlled_upstream(calls, state)

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            route = "/v1/desktop/radient"
            ids = ["agent-1", "agent-2"]

            for upstream_status, batch_status, code, answers in UPSTREAM_CLASSES:
                state["status"] = upstream_status
                before = len(calls)
                response = await client.post(
                    route,
                    json={
                        "operation": "agents.statuses",
                        "query": {"agent_ids": ",".join(ids)},
                    },
                )
                assert response.status_code == batch_status, (upstream_status, response.text)

                if answers:
                    # The absence class: the page answers, and each id reads as
                    # "not liked, not favourited" rather than failing the page.
                    body = response.json()["result"]["data"]["result"]["statuses"]
                    present = upstream_status == 200
                    assert body == {
                        agent_id: {"liked": present, "favourited": present} for agent_id in ids
                    }, (upstream_status, body)
                else:
                    detail = response.json()["detail"]
                    assert detail["code"] == code, (upstream_status, detail)
                    assert detail["details"]["upstream_status"] == upstream_status, detail
                    assert detail["details"]["agent_id"] in ids, detail
                    assert detail["details"]["relation"] in {"like", "favourite"}, detail
                assert len(calls) > before, upstream_status

                # The same question, asked one agent at a time. The mapping is
                # written out here rather than read from the module, so the
                # comparison is against an independent statement of what this
                # route has always done for a single agent.
                single_agent_mapping = {
                    200: 200,
                    400: 400,
                    401: 401,
                    403: 403,
                    404: 404,
                    409: 409,
                    422: 422,
                    429: 429,
                }
                singleton = await client.post(
                    route, json={"operation": "agents.liked", "agent_id": ids[0]}
                )
                expected_singleton = single_agent_mapping.get(upstream_status, 502)
                assert singleton.status_code == expected_singleton, (
                    upstream_status,
                    singleton.text,
                )
                if not answers:
                    # A FAILURE, and the batch says the same thing the single-agent
                    # op says. Only the absence class above may differ.
                    assert singleton.status_code == batch_status, (upstream_status, singleton.text)


@pytest.mark.asyncio
async def test_the_whole_batch_is_bounded_by_one_deadline(headless_tui_env, monkeypatch):
    """A slow or hung upstream costs the page ONE budget, not one budget per wave.

    Round 1 (R-2): `httpx`'s timeout bounded one READ and nothing bounded the
    gather, while the reads run in sequential waves of `STATUS_BATCH_CONCURRENCY`
    — so the hub's twelve-card page was bounded by 4 x 30 s and the id bound by
    11 x 30 s, against 30 s for every other op on this transport. That inverts the
    case this op exists to make, in exactly the regime where the per-card fan-out
    it replaces waited only on its slowest single read.

    The budget is patched here (the shipped value is asserted below) because this
    test is about the SHAPE of the failure rather than about 30 s of waiting: an
    expiry has to be a bounded, structured answer rather than a hang.
    """
    from local_operator.server.routes import desktop_radient

    assert (
        desktop_radient.STATUS_BATCH_TIMEOUT == 30.0
    ), "the batch budget and the other ops' httpx ceiling are one number"
    budget = 0.5
    monkeypatch.setattr(desktop_radient, "STATUS_BATCH_TIMEOUT", budget)
    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    #: Every read answers just inside ONE read's budget, so nothing here is a
    #: per-read timeout: what runs out is the whole fan-out's, four waves in.
    state: dict[str, object] = {"delay": 0.35}
    fake = controlled_upstream(calls, state)

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            route = "/v1/desktop/radient"
            ids = [f"agent-{index}" for index in range(12)]
            query = {"agent_ids": ",".join(ids)}

            before = len(calls)
            started = time.monotonic()
            slow = await client.post(route, json={"operation": "agents.statuses", "query": query})
            elapsed = time.monotonic() - started
            assert slow.status_code == 502, slow.text
            detail = slow.json()["detail"]
            assert detail["code"] == "radient_upstream_timeout", detail
            # The DISTINCTION this test exists for: every read here answered
            # inside one read's budget (0.35 s against httpx's 0.5 s), so what ran
            # out is the whole fan-out's — the field a per-read timeout cannot
            # produce. Without it, four waves of these reads are twelve seconds of
            # renderer waiting.
            assert "timeout_seconds" in detail["details"], detail
            assert elapsed < 1.2, elapsed
            # Four waves at 0.35 s is 1.4 s of upstream time; the deadline is what
            # the renderer waited for, and the reads still in flight were dropped.
            assert elapsed < 1.2, elapsed
            issued = len(calls) - before
            assert 0 < issued < 24, issued

            # And the same budget bounds a read that never answers at all.
            state["delay"] = 5.0
            before = len(calls)
            started = time.monotonic()
            hung = await client.post(route, json={"operation": "agents.statuses", "query": query})
            elapsed = time.monotonic() - started
            assert hung.status_code == 502, hung.text
            assert hung.json()["detail"]["code"] == "radient_upstream_timeout", hung.text
            assert elapsed < 2.0, elapsed


@pytest.mark.asyncio
async def test_a_refused_credential_stops_the_batch_in_the_wave_that_noticed_it(
    headless_tui_env, monkeypatch
):
    """A credential Radient refused must not cost all 2N reads.

    Round 1 (R-3) measured a twelve-id batch issuing all twenty-four reads at the
    API that had just said no, on the page-refresh path, during exactly the window
    where that API is unhealthy or rate-limiting. The reads already in flight
    cannot be recalled — that floor is the concurrency limit — but the ones still
    queued behind it are cancelled rather than started.
    """
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    fake = controlled_upstream(calls, {"status": 429})

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            response = await client.post(
                "/v1/desktop/radient",
                json={
                    "operation": "agents.statuses",
                    "query": {"agent_ids": ",".join(f"agent-{index}" for index in range(12))},
                },
            )
            assert response.status_code == 429, response.text
            assert response.json()["detail"]["code"] == "radient_credential_refused", response.text
            assert len(calls) <= desktop_radient.STATUS_BATCH_CONCURRENCY, calls
            assert len(calls) < 24, calls


@pytest.mark.asyncio
async def test_an_oversized_upstream_body_is_refused_rather_than_buffered(
    headless_tui_env, monkeypatch
):
    """The module's 2 MB upstream ceiling applies to the batch too.

    Round 1 (R-5) measured a 3 MB body accepted and buffered whole, which put the
    batch's memory at six times a single op's rather than under the same ceiling —
    the single-op path has streamed and capped at this size all along.
    """
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    fake = controlled_upstream(calls, {"size": 3_000_000})

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            await client.get("/v1/auth/status")
            store_radient_credential(access, refresh)
            route = "/v1/desktop/radient"
            response = await client.post(
                route,
                json={"operation": "agents.statuses", "query": {"agent_ids": "agent-1"}},
            )
            assert response.status_code == 502, response.text
            detail = response.json()["detail"]
            assert detail["code"] == "radient_upstream_too_large", detail
            assert detail["details"]["limit_bytes"] == 2_000_000, detail

            # The single-op path answers the same body the same way: one ceiling,
            # applied by both, rather than a batch that forgot it.
            singleton = await client.post(
                route, json={"operation": "agents.liked", "agent_id": "agent-1"}
            )
            assert singleton.status_code == 502, singleton.text


@pytest.mark.asyncio
async def test_the_op_refuses_without_the_desktop_token(headless_tui_env, monkeypatch):
    """The batch rides the router's `require_desktop` gate, like every other op here.

    Round 1 noted that neither shipped test ever omitted the token, so the op's
    dependence on the gate was asserted by the router rather than exercised on
    this route. This exercises it — and proves the refusal costs no upstream call.
    """
    from local_operator.server.routes import desktop_radient

    token, refresh, access = [secrets.token_hex(32) for _ in range(3)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls: list[tuple[str, str]] = []
    fake = controlled_upstream(calls, {})

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url,
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as signed_in:
            # The first request is what builds the app's own auth state.
            await signed_in.get("/v1/auth/status")
            store_radient_credential(access, refresh)
        async with httpx.AsyncClient(base_url=desktop_url, timeout=30) as anonymous:
            response = await anonymous.post(
                "/v1/desktop/radient",
                json={"operation": "agents.statuses", "query": {"agent_ids": "agent-1"}},
            )
            assert response.status_code == 401, response.text
            assert "Desktop authorization" in response.text, response.text
            assert calls == [], calls

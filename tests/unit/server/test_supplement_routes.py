"""``GET /v1/capabilities`` and the desktop routes for turn supplements.

Lives under ``tests/unit/server`` because the ``test_app_client`` fixture does. The owner
record, the attach gate and the relay routes are in ``tests/unit/supplements/test_surfaces.py``;
this file owns the desktop routes' contract: status codes, headers verbatim, digest-only
addressing, the served document's sandbox pairing (CSP meta + JSON, never ``text/html``),
and -- for the three control write routes -- dispatch per op, the runtime receipt passed
through verbatim (identity included), every refusal, and the response policy on every
status. The identity semantics get a REAL serving handle behind the facade seam, because
that is the one place a fake client cannot honestly stand in (the runtime's own test
patterns, ``tests/unit/supplements/test_runner_generator.py``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.attachments import AttachmentStore
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.supplements import generator, policy
from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE
from local_operator.supplements.decision import Decision
from local_operator.supplements.evidence import Dataset
from tests.unit.session.test_session import ScriptedStream, wait_for

DIGEST = "04c2d29b140086cc637ad19ef34012c5"
SESSION = "a1b2c3d4e5f6"

#: A stored component body: exactly what ``SupplementRunner._store`` puts under the
#: digest (the generated body, no prelude, no wrapper).
COMPONENT = '<p class="lead">Q3 looks solid.</p>\n<div id="chart"></div>\n'


@pytest.mark.asyncio
async def test_v1_capabilities_advertises_supplements(test_app_client) -> None:
    response = await test_app_client.get("/v1/capabilities")
    assert response.status_code == 200
    features = response.json()["result"]["features"]
    # `>= 1` is the comparison clients make (`desktopFeatureState`); 0 reads as absent
    assert features.get("supplements", 0) >= 1


def _desktop_client(tmp_path, monkeypatch):
    """The real app behind the desktop bearer boundary (the shape the search-route tests use).

    ``/v1/desktop/`` answers 503 until the desktop token is configured, so a bare
    ``test_app_client`` would test the boundary and never reach the route.
    """
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    return TestClient(app, headers={"Host": "127.0.0.1:1111"}), {"Authorization": "Bearer token"}


def _session(tmp_path, session_id: str = SESSION):
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


def _store(tmp_path, blob: str = COMPONENT) -> str:
    ref = AttachmentStore(root=tmp_path / "attachments").put_bytes(
        blob.encode("utf-8"), "text/html"
    )
    assert ref is not None
    return ref.digest


def test_the_desktop_document_route_serves_the_assembled_document(tmp_path, monkeypatch) -> None:
    """The happy path, with the executing-document family's headers verbatim and the
    served bytes byte-equal to the one assembler's output."""
    from local_operator.supplements.document import assemble_stored_component

    digest = _store(tmp_path)
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
    assert response.status_code == 200
    # JSON, never a navigable document: no host may point a frame at a URL (memo §4.1).
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    html = response.json()["html"]
    assert html == assemble_stored_component(COMPONENT)
    # The sandbox pairing: the document carries its own policy (the tunnel strips
    # response headers, so the CSP must ride inside -- memo §2.7), network closed.
    assert '<meta http-equiv="Content-Security-Policy"' in html
    assert "default-src 'none'" in html and "connect-src 'none'" in html


def test_the_desktop_document_route_answers_404_for_a_missing_blob(tmp_path, monkeypatch) -> None:
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{DIGEST}/document", headers=headers
        )
    assert response.status_code == 404


def test_the_desktop_document_route_refuses_a_non_component_blob(tmp_path, monkeypatch) -> None:
    """Bytes that cannot be assembled (a malformed ``<data>`` block) are an ordinary
    404 -- the reader's remedy is the frame-level fallback line, not an error page."""
    digest = _store(tmp_path, "<data>{oops</data><p>tail</p>")
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
    assert response.status_code == 404


def test_the_desktop_document_route_keeps_the_digest_gate(tmp_path, monkeypatch) -> None:
    """A non-digest path never reaches the handler: 422, the traversal gate."""
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            "/v1/desktop/sessions/abc/supplements/not-a-digest/document", headers=headers
        )
    assert response.status_code == 422


def test_the_desktop_document_route_carries_the_policy_on_errors(tmp_path, monkeypatch) -> None:
    """D5's pair rides the family's error responses too (#2134 R4).

    The ladder's 404 and the digest gate's 422 both carry ``nosniff`` +
    ``no-store``, verbatim like the 200 (the unauthorised response is pinned in the
    bearer test below).
    """
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        missing = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{DIGEST}/document", headers=headers
        )
        non_digest = client.get(
            "/v1/desktop/sessions/abc/supplements/not-a-digest/document", headers=headers
        )
    assert missing.status_code == 404
    assert non_digest.status_code == 422
    for response in (missing, non_digest):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"


def test_the_desktop_document_route_has_no_caller_supplied_path(tmp_path, monkeypatch) -> None:
    """There is no path parameter to supply, and a smuggled one changes nothing: the
    document served is the digest's, the same as the clean request."""
    digest = _store(tmp_path)
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        clean = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
        smuggled = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document?path=/etc/passwd",
            headers=headers,
        )
    assert clean.status_code == smuggled.status_code == 200
    assert clean.json()["html"] == smuggled.json()["html"]


def test_the_desktop_document_route_is_behind_the_bearer(tmp_path, monkeypatch) -> None:
    client, _headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(f"/v1/desktop/sessions/abc/supplements/{DIGEST}/document")
    assert response.status_code in (401, 403)
    # The family policy rides the unauthorised response too (#2134 R4, D5).
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# The write half: the three supplement control routes (memo §2.7)
# ---------------------------------------------------------------------------
#
# One route per op dispatching ``bridge.remote.supplement_*`` (the interrupt route's
# own shape); the runtime's ops own every decision. These tests pin the route-level
# promises: dispatch per op, the receipt passed through VERBATIM (identity included),
# the refusals (no live runtime; a dial that cannot be served), the body bounds, and
# the response policy on every status. The identity semantics -- a stale id refused,
# never resolved onto a newer job -- are exercised through a REAL serving handle, so
# the receipt and the journal are the runtime's own.

ANCHOR = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
JOB = "3f9c1a7e5b20"
OLD_JOB = "0d1e2f3a4b5c"


class SupplementOwner:
    """A bound owner that answers the three supplement control ops.

    A plain class rather than a ``SimpleNamespace`` for the ``InterruptClient``
    reason: the facade's dispose path tests clients for set membership, and a
    namespace defines ``__eq__`` and so is unhashable. Only the ops these routes
    reach are implemented, so anything else it grows fails loudly here rather
    than being absorbed. ``ops`` records the calls VERBATIM (op, anchor, job,
    text) so identity pass-through is assertable, and ``raises`` is the
    unreachable-owner arm: the exception an ``AttachClient`` would raise from a
    lost socket.
    """

    def __init__(
        self,
        *,
        cancel: str = "cancelled",
        steer: str = "steering",
        restart: str = "restarting",
    ) -> None:
        self.connected = True
        self.cancel_receipt = cancel
        self.steer_receipt = steer
        self.restart_receipt = restart
        self.ops: list[tuple[str, str, str, str]] = []
        self.raises: BaseException | None = None

    async def supplement_cancel(self, anchor: str, job: str) -> str:
        self.ops.append(("cancel", anchor, job, ""))
        if self.raises is not None:
            raise self.raises
        return self.cancel_receipt

    async def supplement_steer(self, anchor: str, job: str, text: str) -> str:
        self.ops.append(("steer", anchor, job, text))
        if self.raises is not None:
            raise self.raises
        return self.steer_receipt

    async def supplement_restart(self, anchor: str, job: str) -> str:
        self.ops.append(("restart", anchor, job, ""))
        if self.raises is not None:
            raise self.raises
        return self.restart_receipt

    def close(self) -> None:
        pass


def _async_app(tmp_path, monkeypatch):
    """The real app on the test's own loop, with a fresh pool -- the write routes' rig.

    Unlike ``_desktop_client`` (a sync ``TestClient``, which runs the app on a
    portal thread), the live half of these tests holds the pool's bridge and
    dresses its facade ``remote`` before the request runs, so it must share ONE
    event loop with the request -- ``AsyncClient`` over ``ASGITransport``, the
    interrupt-route tests' own shape, against the FULL app so the supplement
    response-policy middleware is part of what is under test.
    """
    from types import SimpleNamespace

    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    return app, pool


def _bearer_client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer token"},
    )


@pytest.mark.asyncio
async def test_the_control_ops_dispatch_per_op_and_return_the_receipt_verbatim(
    tmp_path, monkeypatch
) -> None:
    """Each path reaches its own op, and the runtime's receipt rides out unaltered.

    The receipt is asserted BYTE-EQUAL: a layer that re-worded it would be
    composing a sentence the runtime owns (the ``InterruptReceipt`` rule), and the
    ``(anchor, job)`` the caller sent is asserted to be the pair the runtime was
    asked about -- never a substituted one.
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    try:
        sid = await pool.create(str(tmp_path))
        owner = SupplementOwner()
        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                bridge.remote._client = owner  # type: ignore[assignment]
                cancelled = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/cancel",
                    json={"anchor": ANCHOR},
                )
                steered = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/steer",
                    json={"anchor": ANCHOR, "text": "make it a table"},
                )
                restarted = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/restart",
                    json={"anchor": ANCHOR},
                )
        assert [r.status_code for r in (cancelled, steered, restarted)] == [200, 200, 200]
        assert cancelled.json()["result"] == {
            "op": "cancel",
            "anchor": ANCHOR,
            "job": JOB,
            "receipt": "cancelled",
        }
        assert steered.json()["result"]["receipt"] == "steering"
        assert restarted.json()["result"]["receipt"] == "restarting"
        assert owner.ops == [
            ("cancel", ANCHOR, JOB, ""),
            ("steer", ANCHOR, JOB, "make it a table"),
            ("restart", ANCHOR, JOB, ""),
        ]
        # The family policy rides the 200s too (the middleware stamps every status).
        for response in (cancelled, steered, restarted):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_settled_or_unknown_job_answers_the_neutral_receipt_on_a_200(
    tmp_path, monkeypatch
) -> None:
    """``already finished`` is memo §2.7's deliberate NEUTRAL receipt, never an error.

    The image-gen precedent (``ALREADY_FINISHED_NOTE``: "must never wear the error
    ink or glyph"): a press that lost its race with the job's own end -- or names a
    pair this runtime cannot act on -- must not be rendered as a failure, and the
    identity it names is the caller's, echoed back.
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    try:
        sid = await pool.create(str(tmp_path))
        owner = SupplementOwner(cancel="already finished")
        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                bridge.remote._client = owner  # type: ignore[assignment]
                response = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{OLD_JOB}/cancel",
                    json={"anchor": ANCHOR},
                )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == {
            "op": "cancel",
            "anchor": ANCHOR,
            "job": OLD_JOB,
            "receipt": "already finished",
        }
        assert owner.ops == [("cancel", ANCHOR, OLD_JOB, "")]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_session_with_no_live_runtime_refuses_and_engages_nothing(
    tmp_path, monkeypatch
) -> None:
    """A supplement job lives in a live runtime, so a cold session cannot answer this.

    The refusal is the named 503 -- not a no-op and not an ask to warm -- and the
    never-bind half is asserted with an ``_ensure_bound`` that raises: if anything
    on this path tried to engage the session, the answer would stop being a 503.
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    try:
        sid = await pool.create(str(tmp_path))
        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                assert bridge.remote is not None
                assert bridge.remote.is_cold, "the fixture session was already warm"

                async def exploding_bind(*, foreground: bool = True) -> None:
                    raise AssertionError("a supplement op must never engage a cold session")

                monkeypatch.setattr(bridge.remote, "_ensure_bound", exploding_bind)
                response = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/restart",
                    json={"anchor": ANCHOR},
                )
                assert bridge.remote.is_cold, "the refusal left a runtime behind"
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "supplements_owner_unavailable"
        assert detail["session_id"] == sid
        # The family policy rides the refusal too.
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_dial_that_cannot_be_served_is_the_ladders_503(tmp_path, monkeypatch) -> None:
    """A bound-but-dead socket is the LADDER's answer, not the route's named refusal.

    Nothing was done and the client's move is to come back (the jobs-cancel
    route's own distinction, one rung over). The attach layer's own message can
    name internal sockets and ports, so the ladder must not echo it -- asserted
    with a distinctive substring.
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    try:
        sid = await pool.create(str(tmp_path))
        owner = SupplementOwner()
        owner.raises = ConnectionError("owner connection lost: boom")
        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                bridge.remote._client = owner  # type: ignore[assignment]
                response = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/steer",
                    json={"anchor": ANCHOR, "text": "tighten the axis"},
                )
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "runtime_unreachable"
        assert "boom" not in response.text
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_steer_body_is_bounded_at_the_wire(tmp_path, monkeypatch) -> None:
    """Blank / over-bound / malformed instructions are 422s at the boundary.

    The same non-empty and 500-char rules the runtime's dispatch enforces, one
    layer earlier, where the caller's fix is a different body -- and no refused
    body ever reaches the owner.
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    try:
        sid = await pool.create(str(tmp_path))
        owner = SupplementOwner()
        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                bridge.remote._client = owner  # type: ignore[assignment]
                blank = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/steer",
                    json={"anchor": ANCHOR, "text": "   "},
                )
                over_bound = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/steer",
                    json={"anchor": ANCHOR, "text": "x" * 501},
                )
                no_anchor = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/steer",
                    json={"text": "make it a table"},
                )
                extra_field = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{JOB}/cancel",
                    json={"anchor": ANCHOR, "job": JOB},
                )
        for response in (blank, over_bound, no_anchor, extra_field):
            assert response.status_code == 422, response.text
            assert response.json()["detail"] == "The request has invalid fields."
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"
        assert owner.ops == [], "a refused body must never reach the owner"
    finally:
        await pool.close()


def test_the_control_routes_are_behind_the_bearer(tmp_path, monkeypatch) -> None:
    client, _headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        for op in ("cancel", "steer", "restart"):
            response = client.post(
                f"/v1/desktop/sessions/{SESSION}/supplements/{JOB}/{op}",
                json={"anchor": ANCHOR},
            )
            assert response.status_code in (401, 403), response.text
            # The family policy rides the unauthorised responses too (#2134 R4, D5).
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"


def test_the_control_routes_carry_the_policy_on_every_status(tmp_path, monkeypatch) -> None:
    """D5's pair rides the write family's error responses too (#2134 R4).

    The ladder's 404 (a session this machine does not hold), the path bound's 422
    and the body shape's 422 all carry ``nosniff`` + ``no-store``, verbatim like
    the 200s above.
    """
    client, headers = _desktop_client(tmp_path, monkeypatch)
    unknown = "f" * 12
    with client:
        missing = client.post(
            f"/v1/desktop/sessions/{unknown}/supplements/{JOB}/cancel",
            json={"anchor": ANCHOR},
            headers=headers,
        )
        bad_job = client.post(
            f"/v1/desktop/sessions/{SESSION}/supplements/{'a' * 65}/cancel",
            json={"anchor": ANCHOR},
            headers=headers,
        )
        bad_body = client.post(
            f"/v1/desktop/sessions/{SESSION}/supplements/{JOB}/cancel",
            json={},
            headers=headers,
        )
    assert missing.status_code == 404, missing.text
    assert bad_job.status_code == 422, bad_job.text
    assert bad_body.status_code == 422, bad_body.text
    for response in (missing, bad_job, bad_body):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"


# --- the live half: a REAL serving handle behind the route's facade seam -------------

#: A valid generator answer for the live-handle rig: one component over its
#: declared data (the runtime test suite's own fixture shape).
BLOCK = (
    '<component title="Latency" source="bench.csv">\n'
    '<data>{"lat":{"title":"Latency (ms)","columns":["region","ms"],'
    '"rows":[["us-east",120],["us-west",98.5]]}}</data>\n'
    '<html><div id="c"></div><script>LO.bar(document.getElementById("c"),"lat",'
    '{x:"region",y:"ms",unit:"ms"})</script></html>\n'
    "</component>"
)

DATASETS = (
    Dataset(
        title="Latency by region",
        source="bench.csv",
        columns=("region", "ms"),
        rows=(("us-east", "120"), ("us-west", "98.5")),
        n_rows=2,
        numeric_columns=("ms",),
    ),
)


class _Usage:
    input_tokens = 10
    output_tokens = 10


class _HandleOwner:
    """The facade seam: an ``AttachClient``-shaped stand-in that calls the REAL handle.

    The route speaks ``AttachedSession`` and the ops live on ``ServingSessionHandle``;
    the wire between them is the one layer a unit test cannot honestly fake when the
    assertion is about the RUNTIME's decision, so this stands where the daemon's own
    ``AttachClient`` stands and forwards to the handle's own methods.
    """

    def __init__(self, handle: ServingSessionHandle) -> None:
        self._handle = handle
        self.connected = True

    async def supplement_cancel(self, anchor: str, job: str) -> str:
        return await self._handle.supplement_cancel(anchor, job)

    async def supplement_steer(self, anchor: str, job: str, text: str) -> str:
        return await self._handle.supplement_steer(anchor, job, text)

    async def supplement_restart(self, anchor: str, job: str) -> str:
        return await self._handle.supplement_restart(anchor, job)

    def close(self) -> None:
        pass


def _settings() -> policy.SupplementSettings:
    """The shipped defaults (the runtime tests' own reading: the tail, not the page)."""
    return policy.SupplementSettings.from_values({})


def _decision() -> Decision:
    return Decision(vendor="heuristic", files_p={}, graphics_p=0.9, featured=(), more=())


def _rows(transcript: Transcript) -> list[dict[str, Any]]:
    return [
        entry.payload["details"]
        for entry in transcript.entries()
        if entry.type == "custom" and entry.payload.get("custom_type") == SUPPLEMENT_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_a_stale_job_is_refused_through_the_route_without_touching_the_newer_one(
    tmp_path, monkeypatch
) -> None:
    """The identity gate holds THROUGH HTTP: a cancel naming a SETTLED job answers
    the neutral receipt, the newer job keeps running, and the settled rows stay put.

    The route half of the runtime's own R1 test
    (``test_cancel_for_a_settled_job_never_touches_a_newer_job``), driven through a
    real ``ServingSessionHandle`` behind the facade seam, so the receipt and the
    journal are the runtime's own -- the only honest way to pin "a stale id is
    refused, never resolved onto a newer job" at this layer (memo §2.7; C1b's R1).
    """
    app, pool = _async_app(tmp_path, monkeypatch)
    sid = await pool.create(str(tmp_path))
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100_000),
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=Transcript(tmp_path / "sessions" / sid),
        system_blocks_provider=lambda: ["stable"],
        cwd=str(tmp_path),
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def immediate_complete(request):
        return BLOCK, _Usage()

    monkeypatch.setattr(
        handle._supplements,
        "_resolve_model",
        lambda settings: generator.DesignModel(
            ModelSpec(provider="test", model_id="m", context_window=100_000),
            "session",
            "test/m",
        ),
    )
    monkeypatch.setattr(handle._supplements, "_complete", immediate_complete)
    try:
        handle._supplements._render("anchor-old", _decision(), DATASETS, "u", "a", _settings())
        await wait_for(lambda: not handle._supplements.running, timeout=10)
        rows = _rows(session.transcript)
        assert [row["state"] for row in rows] == ["decided", "done"], rows
        old_job = rows[0]["job"]

        async def blocking_complete(request):
            started.set()
            await release.wait()
            return BLOCK, _Usage()

        monkeypatch.setattr(handle._supplements, "_complete", blocking_complete)
        handle._supplements._render("anchor-new", _decision(), DATASETS, "u", "a", _settings())
        await asyncio.wait_for(started.wait(), timeout=10)

        async with _bearer_client(app) as client:
            async with pool.session(sid) as bridge:
                bridge.remote._client = _HandleOwner(handle)  # type: ignore[assignment]
                response = await client.post(
                    f"/v1/desktop/sessions/{sid}/supplements/{old_job}/cancel",
                    json={"anchor": "anchor-old"},
                )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == {
            "op": "cancel",
            "anchor": "anchor-old",
            "job": old_job,
            "receipt": "already finished",
        }
        # The newer job was NOT cut, and the settled pair was NOT rewritten.
        assert handle._supplements.running
        old_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-old"]
        assert [row["state"] for row in old_rows] == ["decided", "done"], old_rows

        release.set()
        await wait_for(lambda: not handle._supplements.running, timeout=10)
        new_rows = [row for row in _rows(session.transcript) if row["anchor"] == "anchor-new"]
        assert [row["state"] for row in new_rows] == ["decided", "done"], new_rows
    finally:
        release.set()
        await handle.dispose()
        await pool.close()


def test_legacy_table_document_route_preserves_formatted_cells(tmp_path, monkeypatch) -> None:
    """Stored table spellings survive the actual serving path, not just JSON splitting."""
    import json
    import re

    from local_operator.supplements.document import assemble_stored_component

    data = {"d1": {"columns": ["price", "share"], "rows": [["$0.15", "45%"]]}}
    blob = "<data>" + json.dumps(data) + '</data><div id="c"></div><script>'
    blob += 'LO.table(document.getElementById("c"),"d1")</script>'
    digest = _store(tmp_path, blob)
    _session(tmp_path)
    client, headers = _desktop_client(tmp_path, monkeypatch)
    with client:
        response = client.get(
            f"/v1/desktop/sessions/{SESSION}/supplements/{digest}/document", headers=headers
        )
    assert response.status_code == 200
    html = response.json()["html"]
    assert html == assemble_stored_component(blob)
    match = re.search(r'id="lo-data">(.*?)</script>', html)
    assert match is not None
    assert json.loads(match[1]) == data

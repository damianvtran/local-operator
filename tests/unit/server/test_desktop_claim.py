"""The desktop claim handshake: the way in, and what it tightens.

``server/desktop.py``'s module docstring states the contract; these tests pin
the parts a caller depends on and the parts a reviewer would otherwise have to
take on faith:

* ONE posture predicate, truthful in all four states, with the environment
  winning when both it and a claim are present;
* the claim itself — happy path, wrong key, the one-way latch, and the promise
  that the key is never echoed, logged, or written anywhere but the record;
* that a browser cannot reach the door AT ALL, even holding the key, because a
  ``Sec-Fetch-Site``-bearing claim is refused outright, before any key is read;
* that a claim TIGHTENS the legacy control surface rather than merely turning
  the desktop routers on (the half of the bug that option A exists to close),
  for the NATIVE caller too — an empty allowlist means "no browser origin is
  admitted", never "no tightening in force";
* that the claim can install the origins its renderer declares, and that a
  declaration it cannot install is refused with its own sentence;
* that the record stops saying the plane is ungoverned, without re-minting the
  key the app proved ownership with;
* that the claim route is reachable while the plane is unclaimed — the ordering
  that would otherwise be a deadlock, asserted against the real boundary
  middleware rather than argued in a comment;
* that the two environment variables have exactly one reader in the whole
  package, so the next gate cannot grow a private second opinion (with the
  complementary routing-table walk in ``test_desktop_controls.py``);
* and that an accepted claim leaves a line an operator can find, which is the
  diagnostic for the named rollout risk.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server import desktop
from local_operator.server import registry as serve_registry
from local_operator.server.app import desktop_origin_cors, managed_desktop_boundary
from local_operator.server.routes import capabilities, config, desktop_claim, settings

#: The key the fixture publishes in the record. Fixed rather than minted so a
#: leaked value is obvious in a failing assertion; the real one is 256 bits.
CLAIM_KEY = "claim-key-under-test"
APP_ORIGIN = "http://localhost:5187"
PAGE_ORIGIN = "https://evil.example"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def plane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The desktop plane of a daemon NOBODY started with a token.

    That is the state this whole change is about: ``lop serve`` from a TUI, the
    record published, the app able to find it and unable to talk to it. The
    middleware is registered in the real app's order (CORS, then the origin
    suppressor, then the legacy boundary outermost) so the CORS assertions here
    test the stack the daemon actually runs rather than a convenient imitation.
    """
    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.delenv(desktop.ORIGINS_ENV, raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # The latch is process-wide module state; monkeypatch restores it, so one
    # test's claim can never open the plane for the next one.
    monkeypatch.setattr(desktop, "_CLAIMED", None)

    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(desktop_claim.router)
    app.include_router(settings.router)  # a desktop route: requires the posture
    app.include_router(config.router)  # a legacy control path: gated in managed mode
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(desktop_origin_cors)
    app.middleware("http")(managed_desktop_boundary)
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.credential_manager = CredentialManager(tmp_path)
    app.state.instance_id = "instance-under-test"
    app.state.serve_record = SimpleNamespace(claim_key=CLAIM_KEY)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        yield client, app


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


#: Sentinel for "no body at all", so that ``body=None`` can keep meaning "a
#: literal JSON ``null``" and the two are not conflated.
_NO_BODY = object()


async def claim(
    client: AsyncClient,
    key: str = CLAIM_KEY,
    headers: dict[str, str] | None = None,
    body: object = _NO_BODY,
):
    """POST the claim, with the bearer, any Origin/Sec-Fetch-Site headers, and
    an optional JSON body (absent unless a test supplies one)."""
    extra: dict[str, Any] = {} if body is _NO_BODY else {"json": body}
    return await client.post(
        "/v1/desktop/claim", headers={**bearer(key), **(headers or {})}, **extra
    )


def publish(app: FastAPI, key: str | None) -> None:
    """Replace what the record published, as a different boot would."""
    app.state.serve_record = None if key is None else SimpleNamespace(claim_key=key)


# --------------------------------------------------------------------------- #
# One posture predicate
# --------------------------------------------------------------------------- #


async def test_posture_is_truthful_in_all_four_states(plane, monkeypatch):
    """neither / env / claim / both — and the environment wins over a claim."""
    client, _ = plane
    assert desktop.desktop_posture() == desktop.DesktopPosture("", frozenset())
    assert not desktop.desktop_posture().enabled

    monkeypatch.setenv(desktop.TOKEN_ENV, "env-token")
    monkeypatch.setenv(desktop.ORIGINS_ENV, f"{APP_ORIGIN}, null ,")
    assert desktop.desktop_posture() == desktop.DesktopPosture("env-token", frozenset({APP_ORIGIN}))
    monkeypatch.delenv(desktop.TOKEN_ENV)
    monkeypatch.delenv(desktop.ORIGINS_ENV)

    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200
    assert desktop.desktop_posture() == desktop.DesktopPosture(CLAIM_KEY, frozenset({APP_ORIGIN}))

    # Both, forced: a claim cannot be installed while the environment governs
    # (proved below), so the state is constructed to pin the precedence rule
    # itself. The env capability is the app's own process-lifetime secret and
    # outranks anything a claim brought with it — including its origin, which
    # must not survive into a posture the environment defines.
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-token")
    assert desktop.desktop_posture() == desktop.DesktopPosture("env-token", frozenset())


async def test_the_environment_alone_refuses_a_claim(plane, monkeypatch):
    """A backend the app started has nothing to claim: it is already governed."""
    client, app = plane
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-token")
    # What `build_record` publishes in that state (asserted in
    # test_serve_registry): no key, because there is no claim to authorise.
    publish(app, "")

    response = await claim(client, key="env-token")
    assert response.status_code == 409
    assert "env-token" not in response.text
    assert desktop.desktop_posture().token == "env-token"


# --------------------------------------------------------------------------- #
# The claim itself
# --------------------------------------------------------------------------- #


async def test_claim_opens_the_plane_and_capabilities_says_so(plane):
    client, _ = plane
    # Before: the second half of the operator's complaint, exactly. The plane
    # answers 503 and the public signal says there is nothing there.
    assert (await client.get("/v1/settings")).status_code == 503
    before = (await client.get("/v1/capabilities")).json()["result"]
    assert before["desktop_available"] is False

    response = await claim(client)
    assert response.status_code == 200
    assert response.json()["result"] == {
        "claimed": True,
        "instance_id": "instance-under-test",
    }
    assert CLAIM_KEY not in response.text, "the key is never echoed"

    after = (await client.get("/v1/capabilities")).json()["result"]
    assert after["desktop_available"] is True
    # The flip IS the signal: no new key to gate a client on, and no existing
    # surface that a renderer would have to learn to read differently.
    assert set(after) == set(before)
    assert set(after["features"]) == set(before["features"])

    # And the same bearer opens the routers, with no Origin at all: Electron
    # main is a native caller (`Sec-Fetch-Site` absent) and is admitted by the
    # bearer alone.
    assert (await client.get("/v1/settings", headers=bearer(CLAIM_KEY))).status_code == 200
    assert (await client.get("/v1/settings")).status_code == 401


async def test_a_wrong_key_is_refused_and_leaves_the_plane_closed(plane):
    client, _ = plane
    response = await claim(client, key="not-the-key")
    assert response.status_code == 401
    assert "not-the-key" not in response.text
    assert not desktop.desktop_posture().enabled
    assert (await client.get("/v1/settings")).status_code == 503
    assert (await client.get("/v1/capabilities")).json()["result"]["desktop_available"] is False


async def test_a_second_claim_is_refused_even_with_the_right_key(plane):
    """The latch is one-way: one app owns this daemon for its whole life."""
    client, _ = plane
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200
    # The same caller, the right key, a fresh attempt: refused, and the refusal
    # did not widen the allowlist or invalidate the bearer already in force.
    assert (await claim(client)).status_code == 409
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 409
    # An origin that is not installed cannot even reach the latch now.
    assert (await claim(client, headers={"Origin": PAGE_ORIGIN})).status_code == 403
    assert desktop.desktop_posture().origins == frozenset({APP_ORIGIN})
    assert (await client.get("/v1/settings", headers=bearer(CLAIM_KEY))).status_code == 200


async def test_the_claim_leaves_a_line_an_operator_can_find(plane, caplog):
    """The diagnostic for the named rollout risk: a working script starts 401ing.

    Accepting a claim bearer-gates the legacy control surface for EVERY other
    local caller of this daemon, and design risk 1 is exactly that: someone's
    ``curl`` script stops working and nothing explains why. This line does, so
    it is logged at WARNING — the level the daemon's console logging defaults
    to (``LOG_LEVEL``) — and it carries the daemon's ``instance_id`` and the
    origin the caller installed, never the key.
    """
    client, _ = plane
    with caplog.at_level(logging.WARNING, logger="local_operator.server.routes.desktop_claim"):
        assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200

    accepted = [r for r in caplog.records if "desktop claim accepted" in r.getMessage()]
    assert accepted, "an accepted claim left no audit line"
    assert accepted[0].levelno == logging.WARNING
    line = accepted[0].getMessage()
    assert "instance-under-test" in line
    assert APP_ORIGIN in line
    assert CLAIM_KEY not in line


async def test_a_refused_claim_also_leaves_a_line(plane, caplog):
    """A refusal is logged too, so a lockout can be told from a wrong key."""
    client, _ = plane
    with caplog.at_level(logging.INFO, logger="local_operator.server.routes.desktop_claim"):
        assert (await claim(client, key="guessed")).status_code == 401

    refusals = [r.getMessage() for r in caplog.records if "desktop claim refused" in r.getMessage()]
    assert refusals, "a refused claim left no audit line"
    assert "instance-under-test" in refusals[-1]
    assert "401" in refusals[-1]
    assert CLAIM_KEY not in refusals[-1]
    assert "guessed" not in refusals[-1]


async def test_the_key_is_never_logged(plane, caplog):
    """The record is the ONLY channel the key is published through."""
    client, _ = plane
    probe = "log-capture-probe"
    with caplog.at_level(logging.DEBUG):
        # A probe first: without it, "the key is not in the logs" would pass on
        # a capture that never worked at all.
        logging.getLogger("local_operator.server").debug("capture %s", probe)
        assert (await claim(client)).status_code == 200
        assert (await claim(client)).status_code == 409
        await client.get("/v1/settings", headers=bearer(CLAIM_KEY))
        await client.get("/v1/capabilities")
    messages = "\n".join(record.getMessage() for record in caplog.records) + caplog.text
    assert probe in messages, "log capture is not working; the assertion below is vacuous"
    assert CLAIM_KEY not in messages
    assert CLAIM_KEY not in repr(caplog.records)


# --------------------------------------------------------------------------- #
# A browser cannot reach the door
# --------------------------------------------------------------------------- #


async def test_a_foreign_origin_cannot_claim_against_a_configured_list(plane, monkeypatch):
    """The operator's own allowlist is a narrowing, and the claim honours it."""
    client, _ = plane
    monkeypatch.setenv(desktop.ORIGINS_ENV, APP_ORIGIN)
    assert (await claim(client, headers={"Origin": "http://localhost:9999"})).status_code == 403
    assert not desktop.desktop_posture().enabled
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200


async def test_an_opaque_origin_cannot_claim(plane):
    """``"null"`` is every opaque-origin document, not one application."""
    client, _ = plane
    assert (await claim(client, headers={"Origin": "null"})).status_code == 403
    assert not desktop.desktop_posture().enabled


async def test_the_claim_admits_the_origin_it_installs(plane):
    """The one place the claim's Origin rule differs from ``require_desktop``'s.

    A claim is authorised by the 256-bit key in a ``0600`` record, which a page
    cannot read and cannot guess, and the Origin it carries is the VALUE being
    installed rather than a credential — the holder of the key could open the
    plane with the bearer alone. Requiring that value to be pre-trusted would
    refuse every desktop app that did not start this daemon, which is the only
    caller this route exists for. So an unknown origin is installed, and
    installed EXACTLY: it admits that origin and, as the tests below show,
    nothing else.

    A NATIVE caller that supplies an Origin, note: no ``Sec-Fetch-Site``
    anywhere in these headers. A page — which always carries that metadata — is
    refused outright by the test below.
    """
    client, _ = plane
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200
    assert desktop.desktop_posture().origins == frozenset({APP_ORIGIN})
    assert (
        await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": APP_ORIGIN})
    ).status_code == 200


async def test_a_browser_originated_request_cannot_claim(plane, monkeypatch):
    """A page cannot use this door even while holding the key.

    ``Sec-Fetch-Site`` is attached by the browser's own fetch/XHR stack and sits
    on the forbidden-header list, so page script can neither forge nor remove
    it: its presence is proof of a browser origin, and the intended caller (the
    app's main process) sends none. This is defence in depth rather than a
    capability boundary — the key is 256 bits and a page cannot read the record
    — but the request that installs an Origin puts the SPENDER on the allowlist,
    so a leaked key must not be spendable by script.

    The last case is the one round 1 measured as admitted: the correct key, an
    Origin the claim would happily install, and fetch metadata proving a page.
    """
    client, _ = plane
    for site in ("cross-site", "same-origin", "same-site", "none"):
        assert (await claim(client, headers={"Sec-Fetch-Site": site})).status_code == 403, site

    page_shaped = await claim(
        client,
        headers={"Origin": PAGE_ORIGIN, "Sec-Fetch-Site": "cross-site"},
    )
    assert page_shaped.status_code == 403
    assert desktop.desktop_posture().origins == frozenset(), "nothing was installed"
    assert not desktop.desktop_posture().enabled

    # ...on an empty allowlist AND on a configured one: the browser signal is
    # not conditional on what the allowlist happens to hold.
    monkeypatch.setenv(desktop.ORIGINS_ENV, APP_ORIGIN)
    crossed = await claim(
        client,
        headers={"Origin": APP_ORIGIN, "Sec-Fetch-Site": "same-origin"},
    )
    assert crossed.status_code == 403, "an admitted origin does not admit its page"
    assert not desktop.desktop_posture().enabled


async def test_the_allowlist_a_claim_installs_admits_the_app_and_nothing_else(plane):
    client, _ = plane
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200

    allowed = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": APP_ORIGIN})
    assert allowed.status_code == 200
    for origin in (PAGE_ORIGIN, "http://localhost:5188", "null"):
        refused = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": origin})
        assert refused.status_code == 403, origin


# --------------------------------------------------------------------------- #
# Origins the claim DECLARES, because a native caller has none of its own
# --------------------------------------------------------------------------- #


async def test_a_claim_installs_the_origins_it_declares(plane):
    """A native caller has no Origin to install, so it declares the ones it needs.

    Electron main sends no ``Origin``, and the packaged app's renderer loads
    from ``file://`` whose origin is the literal ``"null"`` this plane refuses
    — so without a body field there is no way for the caller the route exists
    for to admit the origin its renderer is served from. The declared origins
    are installed exactly, like a header-borne one.
    """
    client, _ = plane
    declared = ["http://localhost:5173", "https://renderer.example"]
    assert (await claim(client, body={"origins": declared})).status_code == 200
    assert desktop.desktop_posture().origins == frozenset(declared)

    for origin in declared:
        admitted = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": origin})
        assert admitted.status_code == 200, origin
    for origin in (PAGE_ORIGIN, "http://localhost:5174"):
        refused = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": origin})
        assert refused.status_code == 403, origin


async def test_a_declared_origin_is_also_admitted_to_the_cors_grant(plane):
    """The declared origin is the app's own renderer, so it keeps its grant.

    A declaration that admitted the origin to the bearer-gated routes but not
    to the CORS grant would leave a dev-server renderer unable to READ any
    reply it was authorised to get — the wall-of-its-own-making failure the
    two halves of ``desktop_posture().origins`` exist to prevent.
    """
    client, _ = plane
    assert (await claim(client, body={"origins": [APP_ORIGIN]})).status_code == 200

    granted = await client.get("/v1/capabilities", headers={"Origin": APP_ORIGIN})
    assert granted.headers["access-control-allow-origin"] == APP_ORIGIN
    refused = await client.get("/v1/capabilities", headers={"Origin": PAGE_ORIGIN})
    assert "access-control-allow-origin" not in refused.headers


@pytest.mark.parametrize(
    "declared, expected",
    [
        (["null"], "opaque"),
        (["*"], "wildcard"),
        (["http://localhost:5173/chat"], "plain origins"),
        (["http://localhost:5173/?x=1"], "plain origins"),
        (["http://localhost:5173/#fragment"], "plain origins"),
        (["file://localhost"], "absolute http(s)"),
        (["localhost:5173"], "absolute http(s)"),
        ([""], "absolute http(s)"),
        ([42], "strings"),
    ],
)
async def test_a_claim_refuses_an_origin_it_cannot_install(plane, declared, expected):
    """Each refusal carries its own sentence, and the plane stays shut.

    The caller here is the one legitimate caller, so a vague "invalid origin"
    would only cost a real app a debugging round-trip; and every refusal has to
    leave the plane unclaimed, or a rejected claim would still have spent the
    latch.
    """
    client, _ = plane
    response = await claim(client, body={"origins": declared})
    assert response.status_code == 400
    assert expected in response.json()["detail"]
    assert not desktop.desktop_posture().enabled
    assert desktop.desktop_posture().origins == frozenset()


async def test_a_claim_cannot_widen_past_the_operators_configured_list(plane, monkeypatch):
    """``LOCAL_OPERATOR_DESKTOP_ORIGINS`` is the operator's narrowing.

    Refused with the same ``403`` its header-borne twin gets: whoever started
    the daemon wrote that list down, so a body field must not be a way around
    it — the caller holds the claim key, not the operator's intent.
    """
    client, _ = plane
    monkeypatch.setenv(desktop.ORIGINS_ENV, APP_ORIGIN)

    refused = await claim(client, body={"origins": [PAGE_ORIGIN]})
    assert refused.status_code == 403
    assert not desktop.desktop_posture().enabled

    assert (await claim(client, body={"origins": [APP_ORIGIN]})).status_code == 200
    assert desktop.desktop_posture().origins == frozenset({APP_ORIGIN})


@pytest.mark.parametrize(
    "raw, expected",
    [
        (b"{not json", "JSON object"),
        (b'["http://localhost:5173"]', "JSON object"),
        (b'{"origins": "http://localhost:5173"}', "must be a list"),
    ],
)
async def test_a_malformed_claim_body_is_refused_not_ignored(plane, raw, expected):
    """A body that says something unusable is refused, never read as "nothing".

    Silently reading ``{"origins": "<one string>"}`` as an empty declaration
    would leave the caller believing its renderer was admitted while CORS walls
    it off — the failure mode this field exists to prevent.
    """
    client, _ = plane
    response = await client.post("/v1/desktop/claim", headers=bearer(CLAIM_KEY), content=raw)
    assert response.status_code == 400
    assert expected in response.json()["detail"]
    assert not desktop.desktop_posture().enabled


@pytest.mark.parametrize("body", [None, {}])
async def test_a_body_that_declares_nothing_is_accepted(plane, body):
    """The field is an addition, not a requirement on the handshake."""
    client, _ = plane
    assert (await claim(client, body=body)).status_code == 200
    assert desktop.desktop_posture().origins == frozenset()


async def test_the_claim_route_is_reachable_while_the_plane_is_unclaimed(plane):
    """The ordering that would otherwise be a deadlock, proved not argued.

    The boundary middleware really does see this path — it is under the
    ``/v1/desktop/`` sensitive prefix, so it earns the ``no-store`` header —
    and it does NOT gate it: the legacy-control check matches only the agent,
    job, schedule and flat singleton families. The contrast is asserted in the
    same test, because "reachable" is meaningless unless the rest of the plane
    is genuinely shut: ``/v1/settings`` (the same prefix, behind
    ``require_desktop``) answers 503 until the claim lands.
    """
    client, _ = plane
    blocked = await client.get("/v1/settings")
    assert blocked.status_code == 503
    assert blocked.headers["cache-control"] == "no-store"

    response = await claim(client)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


async def test_a_claim_that_never_reaches_the_record_is_refused(plane):
    """No published key means no claim: a boot that announced nothing is not a
    daemon, and the record is the key's only lawful channel."""
    client, app = plane
    publish(app, "")
    assert (await claim(client)).status_code == 503
    publish(app, None)
    assert (await claim(client)).status_code == 503


async def test_a_claim_refreshes_the_record_instead_of_re_minting_its_key(plane, tmp_path):
    """G1: after a claim the published record must stop saying "ungoverned".

    The record is the only on-disk statement of who owns this plane, and the
    ``desktop`` field is filled in by ``build_record`` before any claim can
    exist — so without this refresh the file keeps handing out the key that
    governs the daemon while telling every reader the plane is ungoverned. QA
    measured the same file being rewritten by a heartbeat with the field still
    ``false``, which is why this is a lie rather than staleness.

    The refresh goes through the publisher that OWNS the record, and that is
    the point of the test: the obvious fix — rebuild the record — re-mints
    ``claim_key`` (``build_record`` mints one whenever the posture it reads is
    ungoverned). The published key is the app's proof of ownership, so a silent
    re-mint turns the app's legitimate re-attach into a ``409`` and loses the
    credential. Hence the byte-identical assertion, and hence the follow-up
    heartbeat, which must not undo it either.
    """
    client, app = plane
    record = serve_registry.build_record(
        instance_id="instance-under-test", announced=("127.0.0.1", 1)
    )
    assert record.desktop is False
    key = record.claim_key
    assert len(key) == 43, "the record published no key, so nothing could be claimed"

    publisher = serve_registry.publisher(record, root=tmp_path)
    assert json.loads(publisher.path.read_text())["desktop"] is False
    app.state.serve_record = record
    app.state.serve_publisher = publisher

    assert (await claim(client, key=key)).status_code == 200

    on_disk = json.loads(publisher.path.read_text())
    assert on_disk["desktop"] is True, "the record still says the plane is ungoverned"
    assert on_disk["claim_key"] == key, "the refresh re-minted the key"
    assert record.claim_key == key
    assert desktop.desktop_posture().token == key

    # The timer's heartbeats rewrite the same object, so the field stays true.
    publisher.heartbeat()
    assert json.loads(publisher.path.read_text())["desktop"] is True
    publisher.close()


# --------------------------------------------------------------------------- #
# The legacy surface is TIGHTENED, not merely joined
# --------------------------------------------------------------------------- #


async def test_a_claim_tightens_the_legacy_control_surface(plane):
    """The documented drive-by-page vector, closed by the claim.

    A standalone daemon answers ``/v1/config`` unauthenticated — so a page the
    user visits can read and mutate it. A claim has to end that, or the app
    would attach to a daemon any other page can still drive.
    """
    client, _ = plane
    page = {"Origin": PAGE_ORIGIN}

    before = await client.get("/v1/config", headers=page)
    assert before.status_code == 200, "standalone posture: legacy control is open"
    assert before.headers["access-control-allow-origin"] == PAGE_ORIGIN

    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200

    native = await client.get("/v1/config")
    assert native.status_code == 401, "the claim gated the legacy control path"
    refused = await client.get("/v1/config", headers=page)
    assert refused.status_code == 403, "and a foreign page is refused even earlier"

    # ...and the app that claimed it still owns the surface, with the bearer.
    allowed = await client.get("/v1/config", headers={**bearer(CLAIM_KEY), "Origin": APP_ORIGIN})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == APP_ORIGIN


async def test_a_claim_stops_echoing_foreign_origins(plane):
    """The wildcard CORS echo, which is what made a page able to READ replies.

    The claim installs an allowlist, and from then on ``desktop_origin_cors``
    drops the CORS grant on any reply to an origin that is not on it. Before
    the claim the same request is answered with the requesting origin echoed
    back and credentials allowed — the drive-by vector at ``app.py:359``.
    """
    client, _ = plane
    page = {"Origin": PAGE_ORIGIN}
    before = await client.get("/v1/capabilities", headers=page)
    assert before.headers["access-control-allow-origin"] == PAGE_ORIGIN

    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200

    after = await client.get("/v1/capabilities", headers=page)
    assert after.status_code == 200
    assert "access-control-allow-origin" not in after.headers
    assert "access-control-allow-credentials" not in after.headers


async def test_a_native_claim_also_stops_echoing_foreign_origins(plane):
    """G2: the caller this feature exists for is the one that installs NO origin.

    Electron main sends no ``Origin``, so a native claim installs an empty
    allowlist — and the first version of ``desktop_origin_cors`` read "empty
    allowlist" as "no tightening in force", which left the wildcard echo (and
    ``allow-credentials``) on every non-gated legacy path for exactly that
    daemon. QA measured it against ``/v1/models/providers``; the middleware is
    the same one here.

    An empty allowlist on a GOVERNED plane means "no browser origin is
    admitted", which is what ``require_desktop`` already does on the gated
    families — so the app, which sends no Origin, loses nothing. That half is
    asserted below.
    """
    client, _ = plane
    page = {"Origin": PAGE_ORIGIN}
    before = await client.get("/v1/capabilities", headers=page)
    assert before.headers["access-control-allow-origin"] == PAGE_ORIGIN

    assert (await claim(client)).status_code == 200  # native: no Origin at all
    assert desktop.desktop_posture().origins == frozenset()

    for origin in (PAGE_ORIGIN, APP_ORIGIN):
        after = await client.get("/v1/capabilities", headers={"Origin": origin})
        assert after.status_code == 200
        assert "access-control-allow-origin" not in after.headers, origin
        assert "access-control-allow-credentials" not in after.headers, origin

    # The app's own shape — no Origin — is untouched, on a gated route or not.
    assert (await client.get("/v1/capabilities")).status_code == 200


async def test_an_env_token_daemon_without_a_list_admits_no_browser_origin(plane, monkeypatch):
    """G2, the app-managed half: the app is not broken by the tightening.

    A daemon the app started with a token and NO ``ORIGINS`` list already
    refuses every Origin-bearing request on the gated families (403, asserted
    below) — its renderer goes through main, which sends none. The CORS grant
    now agrees with that gate instead of contradicting it.
    """
    client, _ = plane
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-token")
    assert desktop.desktop_posture().origins == frozenset()

    for origin in (APP_ORIGIN, PAGE_ORIGIN):
        response = await client.get("/v1/capabilities", headers={"Origin": origin})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers, origin

    # The same state on a gated route: already refused before this change, so
    # nothing regresses for the app's main-process caller below.
    assert (
        await client.get(
            "/v1/settings",
            headers={"Origin": APP_ORIGIN, "Authorization": "Bearer env-token"},
        )
    ).status_code == 403
    assert (await client.get("/v1/settings", headers={**bearer("env-token")})).status_code == 200


async def test_an_env_token_daemon_with_a_list_still_admits_exactly_that_list(plane, monkeypatch):
    """The configured case is unchanged: its list, and nobody else."""
    client, _ = plane
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-token")
    monkeypatch.setenv(desktop.ORIGINS_ENV, APP_ORIGIN)

    granted = await client.get("/v1/capabilities", headers={"Origin": APP_ORIGIN})
    assert granted.headers["access-control-allow-origin"] == APP_ORIGIN
    refused = await client.get("/v1/capabilities", headers={"Origin": PAGE_ORIGIN})
    assert "access-control-allow-origin" not in refused.headers


# --------------------------------------------------------------------------- #
# One reader for the two variables
# --------------------------------------------------------------------------- #


#: The shared prefix of both variables, so an f-string that BUILDS either name
#: is a read too. It is deliberately the prefix rather than the two full
#: literals: ``f"LOCAL_OPERATOR_DESKTOP_{kind}"`` is the same variable read by
#: a second opinion, and it is exactly the spelling a future gate would reach
#: for.
_DESKTOP_ENV_PREFIX = "LOCAL_OPERATOR_DESKTOP_"

#: Names whose VALUES are the environment mapping. Anything assigned one of
#: these -- ``env = os.environ``, ``reader = os.getenv`` -- is tracked as an
#: alias and its later reads are reported.
_ENV_OBJECTS = frozenset({"os.environ", "os.getenv"})


def _mentions_a_desktop_variable(node: ast.AST | None) -> bool:
    """Whether an expression names one of the desktop variables, however spelled.

    Text-based on purpose, and it is over-inclusive rather than exact: a
    mention in a non-read position costs nothing here (only an env-object read
    site records), while a miss costs the whole invariant.
    """
    if node is None:
        return False
    text = ast.unparse(node)
    return _DESKTOP_ENV_PREFIX in text or text in {"TOKEN_ENV", "ORIGINS_ENV"}


class _EnvReaders(ast.NodeVisitor):
    """Reads of the desktop environment, including the cheap evasions.

    WHAT IT CATCHES, beyond a direct ``os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")``:

    * the constants this module declares (``TOKEN_ENV``/``ORIGINS_ENV``) and
      any other name assigned one of the two literals;
    * an f-string that builds either name from the prefix;
    * the mapping reached through a local alias (``env = os.environ`` then
      ``env.get(...)``, ``env[...]``, ``X in env``);
    * ``os.getenv`` itself, as a call, an alias, or a presence test.

    WHAT IT CANNOT SEE, precisely, so nobody reads it as complete:

    * a name built without an f-string — ``"LOCAL_OPERATOR_" + suffix``,
      ``"".join(parts)``, a name assembled in a helper function and returned;
    * ``os`` reached by another spelling (``import os as o``,
      ``getattr(os, "environ")``, ``vars(os)``, ``os.__dict__``);
    * the mapping passed IN as a parameter or read off an object, e.g. a gate
      that takes ``environ`` as an argument from a test seam;
    * an alias assigned in an inner scope and used in an outer one (aliases are
      tracked outward-in, so a name assigned after the read, or in a sibling
      scope, is not linked);
    * reflection and anything resolved at runtime.

    That is a deliberate boundary: this is a guard against a second opinion
    growing the way the original seven did (the same ``os.environ.get``
    spelling, copied), not a sandbox. Anything it cannot see, a reviewer still
    has to read.
    """

    def __init__(self) -> None:
        self.reads: list[tuple[str, int]] = []
        self._functions: list[str] = []
        # One alias map per scope, outermost first; lookups merge outward in.
        self._scopes: list[dict[str, str]] = [{}]

    # -- alias bookkeeping -------------------------------------------------- #

    def _aliases(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for scope in self._scopes:
            merged.update(scope)
        return merged

    def _is_env_object(self, node: ast.expr | None) -> bool:
        """Whether this expression evaluates to the environment mapping itself."""
        if node is None:
            return False
        text = ast.unparse(node)
        return text in _ENV_OBJECTS or self._aliases().get(text) == "environ"

    def _names_variable(self, node: ast.expr | None) -> bool:
        """As :func:`_mentions_a_desktop_variable`, plus local literal aliases."""
        if node is None:
            return False
        if _mentions_a_desktop_variable(node):
            return True
        return self._aliases().get(ast.unparse(node)) == "literal"

    def _note_targets(self, targets: list[ast.expr], value: ast.expr | None) -> None:
        """Track ``name = os.environ`` and ``name = "LOCAL_OPERATOR_DESKTOP_*"``."""
        if value is None:
            return
        if self._is_env_object(value):
            kind = "environ"
        elif _mentions_a_desktop_variable(value):
            kind = "literal"
        else:
            return
        for target in targets:
            if isinstance(target, ast.Name):
                self._scopes[-1][target.id] = kind

    def visit_Assign(self, node: ast.Assign) -> None:
        self._note_targets(list(node.targets), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._note_targets([node.target], node.value)
        self.generic_visit(node)

    # -- read sites --------------------------------------------------------- #

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._functions.append(node.name)
        self._scopes.append({})
        self.generic_visit(node)
        self._scopes.pop()
        self._functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _record(self, node: ast.expr | ast.stmt) -> None:
        self.reads.append((self._functions[-1] if self._functions else "<module>", node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if self._is_env_object(func):  # ``os.getenv("X")``
            if node.args and self._names_variable(node.args[0]):
                self._record(node)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in {"get", "pop", "setdefault"}
            and self._is_env_object(func.value)
        ):
            if node.args and self._names_variable(node.args[0]):
                self._record(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_env_object(node.value) and self._names_variable(node.slice):
            self._record(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # ``X in os.environ`` / ``os.environ == ...``: a presence test is a read
        # too, and it would be a second source of truth just the same.
        operands = [node.left, *node.comparators]
        if any(self._is_env_object(item) for item in operands) and any(
            self._names_variable(item) for item in operands
        ):
            self._record(node)
        self.generic_visit(node)


def _env_reads_by_function(path: Path) -> list[tuple[str, int]]:
    """``(enclosing function, line)`` for every READ of a desktop env var.

    Only reads: the module constants that NAME the variables are assignments,
    and a mention inside a docstring or a comment is not code at all — which is
    why this walks the AST instead of grepping (the constant declarations and
    the explanatory prose both name the variables, and neither is a second
    source of truth). See :class:`_EnvReaders` for what the walk does and does
    not catch.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    visitor = _EnvReaders()
    visitor.visit(tree)
    return visitor.reads


def test_only_desktop_posture_reads_the_desktop_environment() -> None:
    """No gate may grow a second opinion about the desktop posture.

    The bug this PR fixes WAS seven copies of one question, so the invariant is
    asserted mechanically: the two variables are read in exactly one function
    in the package, ``desktop.desktop_posture``, and nowhere else. A future
    gate that reads them directly fails here instead of shipping a claim that
    opens some surfaces and not others.

    ``tests/unit/test_ambient_env_isolation`` is the complementary test: it
    asserts these variables are scrubbed for the suite, not that they have one
    reader.
    """
    package = Path(desktop.__file__).resolve().parent.parent
    offenders: list[str] = []
    allowed = {Path(desktop.__file__).resolve(): "desktop_posture"}
    for path in sorted(package.rglob("*.py")):
        relative = path.resolve().relative_to(package.parent).as_posix()
        for function, line in _env_reads_by_function(path):
            if allowed.get(path.resolve()) == function:
                continue
            offenders.append(f"{relative}:{line} in {function}")

    assert (
        offenders == []
    ), "these read the desktop environment outside desktop_posture(): " + ", ".join(offenders)


def test_the_env_guard_catches_the_cheap_evasions(tmp_path: Path) -> None:
    """The guard above is only worth its line count if it fails on aliasing.

    Review round 1 probed it with seven synthetic readers and 4 of them got
    through: the mapping reached through a local alias (``env = os.environ``),
    the name held in a variable (``key = "LOCAL_OPERATOR_DESKTOP_TOKEN"``), and
    an f-string built from the prefix. Each of those is a realistic way for the
    next gate to be written, so each is pinned here — a walker that is silent
    on all seven would report the same happy result for a package that reads
    the variable from eight places.

    What it still cannot see is stated in :class:`_EnvReaders`'s docstring, and
    deliberately NOT asserted here: a test can only pin what the guard catches.
    """
    source = '''
"""A module that mentions LOCAL_OPERATOR_DESKTOP_TOKEN in prose only."""
import os

TOKEN_ENV = "LOCAL_OPERATOR_DESKTOP_TOKEN"
ORIGINS_ENV = "LOCAL_OPERATOR_DESKTOP_ORIGINS"


def direct_environ_get():
    return os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")


def via_the_constant():
    return os.environ.get(TOKEN_ENV)


def presence_test():
    return "LOCAL_OPERATOR_DESKTOP_ORIGINS" in os.environ


def getenv_call():
    return os.getenv("LOCAL_OPERATOR_DESKTOP_TOKEN")


def literal_in_a_name():
    key = "LOCAL_OPERATOR_DESKTOP_TOKEN"
    return os.environ.get(key)


def built_with_an_f_string():
    return os.environ.get(f"LOCAL_OPERATOR_DESKTOP_{'TOKEN'}")


def aliased_mapping():
    env = os.environ
    return env.get("LOCAL_OPERATOR_DESKTOP_TOKEN")


def an_honest_read():
    return os.environ.get("SOME_OTHER_SETTING")
'''
    probe = tmp_path / "probe_module.py"
    probe.write_text(source)

    reads = _env_reads_by_function(probe)

    # Seven synthetic readers, and the honest one is not among them.
    assert {function for function, _ in reads} == {
        "direct_environ_get",
        "via_the_constant",
        "presence_test",
        "getenv_call",
        "literal_in_a_name",
        "built_with_an_f_string",
        "aliased_mapping",
    }, f"caught {reads}"
    # The module constants and the prose mention are not reads, or the walk
    # would report every file that merely names the variable.
    assert all(function != "<module>" for function, _ in reads)

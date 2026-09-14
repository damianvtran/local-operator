"""The desktop claim handshake: the way in, and what it tightens.

``server/desktop.py``'s module docstring states the contract; these tests pin
the parts a caller depends on and the parts a reviewer would otherwise have to
take on faith:

* ONE posture predicate, truthful in all four states, with the environment
  winning when both it and a claim are present;
* the claim itself — happy path, wrong key, the one-way latch, and the promise
  that the key is never echoed, logged, or written anywhere but the record;
* that a browser cannot reach the door, because the route applies the Origin /
  ``Sec-Fetch-Site`` rule before it looks at any key;
* that a claim TIGHTENS the legacy control surface rather than merely turning
  the desktop routers on (the half of the bug that option A exists to close);
* that the claim route is reachable while the plane is unclaimed — the ordering
  that would otherwise be a deadlock, asserted against the real boundary
  middleware rather than argued in a comment;
* and that the two environment variables have exactly one reader in the whole
  package, so the next gate cannot grow a private second opinion.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server import desktop
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


async def claim(client: AsyncClient, key: str = CLAIM_KEY, headers: dict[str, str] | None = None):
    """POST the claim, with the bearer and any Origin/Sec-Fetch-Site headers."""
    return await client.post("/v1/desktop/claim", headers={**bearer(key), **(headers or {})})


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
    installed EXACTLY: it admits that origin and, as the tests above show,
    nothing else.
    """
    client, _ = plane
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200
    assert desktop.desktop_posture().origins == frozenset({APP_ORIGIN})
    assert (
        await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": APP_ORIGIN})
    ).status_code == 200


async def test_a_browser_originated_request_without_an_origin_cannot_claim(plane, monkeypatch):
    """``Sec-Fetch-Site`` is unforgeable, so its presence proves a page.

    Refused with an empty allowlist AND with a configured one: the browser
    signal is not conditional on what the allowlist happens to hold, because a
    page has nothing to install and this route exists to install something.
    """
    client, _ = plane
    assert (await claim(client, headers={"Sec-Fetch-Site": "cross-site"})).status_code == 403
    monkeypatch.setenv(desktop.ORIGINS_ENV, APP_ORIGIN)
    assert (await claim(client, headers={"Sec-Fetch-Site": "cross-site"})).status_code == 403
    # A browser that DOES present an allowed origin gets past the origin rule —
    # and still needs the key, which it has no way to read.
    crossed = await claim(
        client,
        key="guessed",
        headers={"Origin": APP_ORIGIN, "Sec-Fetch-Site": "same-origin"},
    )
    assert crossed.status_code == 401


async def test_the_allowlist_a_claim_installs_admits_the_app_and_nothing_else(plane):
    client, _ = plane
    assert (await claim(client, headers={"Origin": APP_ORIGIN})).status_code == 200

    allowed = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": APP_ORIGIN})
    assert allowed.status_code == 200
    for origin in (PAGE_ORIGIN, "http://localhost:5188", "null"):
        refused = await client.get("/v1/settings", headers={**bearer(CLAIM_KEY), "Origin": origin})
        assert refused.status_code == 403, origin


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


# --------------------------------------------------------------------------- #
# One reader for the two variables
# --------------------------------------------------------------------------- #


def _says(node: ast.AST | None) -> bool:
    """Whether an expression names one of the desktop variables."""
    if node is None:
        return False
    text = ast.unparse(node)
    return (
        "LOCAL_OPERATOR_DESKTOP_TOKEN" in text
        or "LOCAL_OPERATOR_DESKTOP_ORIGINS" in text
        or text in {"TOKEN_ENV", "ORIGINS_ENV"}
    )


def _env_reads_by_function(path: Path) -> list[tuple[str, int]]:
    """``(enclosing function, line)`` for every READ of a desktop env var.

    Only reads: the module constants that NAME the variables are assignments,
    and a mention inside a docstring or a comment is not code at all — which is
    why this walks the AST instead of grepping (the constant declarations and
    the explanatory prose both name the variables, and neither is a second
    source of truth).
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    reads: list[tuple[str, int]] = []
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def _record(self, node: ast.expr | ast.stmt) -> None:
            reads.append((stack[-1] if stack else "<module>", node.lineno))

        def visit_Call(self, node: ast.Call) -> None:
            name = ast.unparse(node.func)
            if name in {"os.environ.get", "os.environ.pop", "os.getenv"} and (
                node.args and _says(node.args[0])
            ):
                self._record(node)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if ast.unparse(node.value) == "os.environ" and _says(node.slice):
                self._record(node)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            # ``X in os.environ`` / ``os.environ == ...``: a presence test is a
            # read too, and it would be a second source of truth just the same.
            operands = [node.left, *node.comparators]
            if any(ast.unparse(item) == "os.environ" for item in operands) and any(
                _says(item) for item in operands
            ):
                self._record(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    return reads


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

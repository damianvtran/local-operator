"""A standalone daemon refuses foreign-origin browser writes; native clients are unchanged.

Exercises the REAL serve app (``local_operator.server.app.app``) so the gate is
tested in its registered position: inside ``CORSMiddleware`` and
``desktop_origin_cors``, behind ``host_admission``. ASGITransport runs no
lifespan, which is fine here: a refused request must never reach a handler, and
the one admitted write (``/v1/config/system-prompt``) needs no app state.
"""

import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from local_operator.server import desktop as desktop_module
from local_operator.server.app import _STANDALONE_ACTING_GETS, _iter_routes, app

pytestmark = pytest.mark.asyncio

EVIL = "https://evil.test"
REFUSAL = "Cross-origin browser requests cannot change this server."
_WRITES = {"POST", "PUT", "PATCH", "DELETE"}


def _write_routes() -> list[tuple[str, str]]:
    """Every (concrete path, write method) the router declares, claim excepted.

    Built from ``app.routes`` like ``legacy_gate_routes`` is, so a route added
    later is enumerated here without editing this file. Path parameters get a
    placeholder segment; the gate is keyed on the method, so its value is moot.
    """
    routes: set[tuple[str, str]] = set()
    for path, methods in _iter_routes(app.routes):
        if path == "/v1/desktop/claim":
            continue
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        routes.update((concrete, m) for m in methods & _WRITES)
    return sorted(routes)


WRITE_ROUTES = _write_routes()


@pytest.fixture(autouse=True)
def _standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No bearer, no claim, no allowlist: the plain ``local-operator serve`` posture."""
    for name in (
        desktop_module.TOKEN_ENV,
        desktop_module.ORIGINS_ENV,
        desktop_module.TOKEN_FD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(desktop_module, "_CLAIMED", None)
    monkeypatch.setattr(desktop_module, "_SPAWN_BEARER", "")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:1111") as c:
        yield c


async def test_the_enumeration_found_the_write_surface() -> None:
    # Guard against a vacuous parametrization (an import-order change that
    # left app.routes empty would make every refusal test pass trivially).
    assert ("/v1/config/system-prompt", "PATCH") in WRITE_ROUTES
    assert len(WRITE_ROUTES) > 50


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": EVIL},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        # A different port on the same loopback host is a different origin.
        {"Origin": "http://127.0.0.1:9999"},
    ],
    ids=["foreign", "null", "sfs-cross-site", "sfs-same-site", "other-port"],
)
@pytest.mark.parametrize(("path", "method"), WRITE_ROUTES)
async def test_browser_writes_from_foreign_origins_never_reach_a_handler(
    client: AsyncClient, path: str, method: str, headers: dict[str, str]
) -> None:
    response = await client.request(method, path, headers=headers, content=b"{}")
    # The gate's own sentence: no handler emits it, so its presence proves the
    # router was never reached (a 403 from a handler would carry another body).
    assert response.status_code == 403, (method, path, response.text[:200])
    assert REFUSAL in response.json()["detail"]
    # "null" included: its echo now survives on /health alone.
    assert "access-control-allow-origin" not in response.headers


async def test_foreign_write_leaves_no_file(client: AsyncClient, _standalone: Path) -> None:
    response = await client.patch(
        "/v1/config/system-prompt", json={"content": "PWNED"}, headers={"Origin": EVIL}
    )
    assert response.status_code == 403
    assert not (_standalone / "system_prompt.md").exists()


@pytest.mark.parametrize(
    ("base", "headers"),
    [
        ("http://127.0.0.1:1111", {}),  # curl / scripts / CLI
        ("http://127.0.0.1:1111", {"Origin": "http://127.0.0.1:1111"}),
        ("http://localhost:1111", {"Origin": "http://localhost:1111"}),
        ("http://LOCALHOST:1111", {"Origin": "http://localhost:1111"}),
        ("http://127.0.0.1:1111", {"Sec-Fetch-Site": "same-origin"}),
        ("http://127.0.0.1:1111", {"Sec-Fetch-Site": "none"}),
    ],
    ids=["native", "same-127", "same-localhost", "host-case", "sfs-same-origin", "sfs-none"],
)
async def test_native_and_same_origin_writes_still_work(
    _standalone: Path, base: str, headers: dict[str, str]
) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=base) as c:
        response = await c.patch(
            "/v1/config/system-prompt", json={"content": "hello"}, headers=headers
        )
    assert response.status_code == 200, response.text
    assert (_standalone / "system_prompt.md").read_text() == "hello"

    # The session loader reads exactly what the route wrote.
    from local_operator.session_factory import resolve_user_instructions

    text, _sources = resolve_user_instructions(log_provenance=False)
    assert "hello" in text


async def test_allowlisted_origin_is_the_escape_hatch(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, _standalone: Path
) -> None:
    monkeypatch.setenv(desktop_module.ORIGINS_ENV, "http://localhost:5187")
    response = await client.patch(
        "/v1/config/system-prompt",
        json={"content": "dev"},
        headers={"Origin": "http://localhost:5187"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5187"
    assert (
        await client.patch(
            "/v1/config/system-prompt", json={"content": "x"}, headers={"Origin": EVIL}
        )
    ).status_code == 403


async def test_foreign_preflight_gets_no_grant_but_null_health_still_echoes(
    client: AsyncClient,
) -> None:
    preflight = await client.options(
        "/v1/config/system-prompt",
        headers={"Origin": EVIL, "Access-Control-Request-Method": "PATCH"},
    )
    assert "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-credentials" not in preflight.headers

    foreign_health = await client.get("/health", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in foreign_health.headers

    # The packaged renderer runs at file:// and reads /health with Origin "null"
    # before it has claimed the daemon; losing this echo reads as "offline".
    null_health = await client.get("/health", headers={"Origin": "null"})
    assert null_health.headers.get("access-control-allow-origin") == "null"
    assert null_health.headers.get("access-control-allow-credentials") == "true"

    same = await client.get("/health", headers={"Origin": "http://127.0.0.1:1111"})
    assert same.headers.get("access-control-allow-origin") == "http://127.0.0.1:1111"


async def test_reads_and_static_routes_are_not_gated(
    client: AsyncClient, _standalone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.config import ConfigManager

    # The lifespan (not run under ASGITransport) is what installs this.
    monkeypatch.setattr(app.state, "config_manager", ConfigManager(_standalone), raising=False)
    # GET is outside the gate: a foreign GET reaches its handler (the CORS
    # strip, not a 403, is what keeps its body from the page).
    response = await client.get("/v1/config/system-prompt", headers={"Origin": EVIL})
    assert response.status_code != 403
    # File serving is unchanged: the UI loads these through <img>/<video>,
    # which this gate never sees.
    image = _standalone / "pic.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    static = await client.get("/v1/static/images", params={"path": str(image)})
    assert static.status_code == 200, static.text[:200]
    assert static.content == image.read_bytes()


async def test_claim_route_is_not_gated(client: AsyncClient) -> None:
    # The claim route answers browsers with its OWN refusal; this gate must not
    # pre-empt it (the app attaches through it on a standalone daemon).
    response = await client.post("/v1/desktop/claim", headers={"Origin": EVIL}, json={})
    assert REFUSAL not in response.text


async def test_managed_posture_is_untouched(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With a bearer the standalone gate stands down; whatever answers is the
    # managed boundary's own refusal (or the route), never this sentence.
    monkeypatch.setenv(desktop_module.TOKEN_ENV, "managed-token")
    response = await client.patch(
        "/v1/config/system-prompt", json={"content": "x"}, headers={"Origin": EVIL}
    )
    assert REFUSAL not in response.text
    # The managed allowlist-less posture keeps its historical CORS echo.
    health = await client.get("/health", headers={"Origin": EVIL})
    assert health.headers.get("access-control-allow-origin") == EVIL
    # ...including "null" on every path: the /health-only restriction is
    # standalone-only.
    opaque = await client.get("/v1/agents", headers={"Origin": "null"})
    assert opaque.headers.get("access-control-allow-origin") == "null"


# --- GETs that act -----------------------------------------------------------
#
# Method-keying misses a GET that imports or calls out: a cross-site <img src>
# sends exactly these headers (no Origin, no preflight) and the handler runs.

#: What Chrome/Firefox/Safari attach to a cross-site <img src> subresource GET.
IMG_HEADERS = {
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Dest": "image",
}

ACTING_GETS = sorted(re.sub(r"\{[^}]+\}", "x", path) for path in _STANDALONE_ACTING_GETS)


async def test_every_named_acting_get_is_a_live_get_route() -> None:
    # A rename that left a stale template here would silently ungate the route.
    get_paths = {path for path, methods in _iter_routes(app.routes) if "GET" in methods}
    missing = sorted(set(_STANDALONE_ACTING_GETS) - get_paths)
    assert not missing, missing
    assert "/v1/agents/{agent_id}/download" in _STANDALONE_ACTING_GETS
    assert "/v1/teams/pull/{team_id}" in _STANDALONE_ACTING_GETS


@pytest.mark.parametrize(
    "headers",
    [IMG_HEADERS, {"Sec-Fetch-Site": "same-site"}, {"Origin": EVIL}, {"Origin": "null"}],
    ids=["img-cross-site", "sfs-same-site", "foreign-origin", "null-origin"],
)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("path", ACTING_GETS)
async def test_cross_site_acting_gets_are_refused_before_the_handler(
    client: AsyncClient, path: str, method: str, headers: dict[str, str]
) -> None:
    response = await client.request(method, path, headers=headers)
    assert response.status_code == 403, (method, path, response.text[:200])
    if method == "GET":
        assert REFUSAL in response.json()["detail"]
    assert "access-control-allow-origin" not in response.headers


class _RecordingRegistry:
    """Stands in for the registry so the test sees whether the import ran."""

    def __init__(self) -> None:
        self.pulled: list[str] = []

    def download_agent_from_radient(self, _client, agent_id: str, **_kwargs):
        self.pulled.append(agent_id)
        raise RuntimeError("stub: no hub in a unit test")


@pytest.fixture
def recording_registry(_standalone: Path) -> Iterator[_RecordingRegistry]:
    from local_operator.config import ConfigManager
    from local_operator.server import dependencies

    registry = _RecordingRegistry()
    app.dependency_overrides[dependencies.get_agent_registry] = lambda: registry
    app.dependency_overrides[dependencies.get_config_manager] = lambda: ConfigManager(_standalone)
    app.dependency_overrides[dependencies.get_provider_auth_store] = lambda: object()
    try:
        yield registry
    finally:
        for dep in (
            dependencies.get_agent_registry,
            dependencies.get_config_manager,
            dependencies.get_provider_auth_store,
        ):
            app.dependency_overrides.pop(dep, None)


async def test_img_tag_cannot_import_an_agent_but_native_and_same_origin_can(
    client: AsyncClient, recording_registry: _RecordingRegistry
) -> None:
    refused = await client.get("/v1/agents/attacker-agent/download", headers=IMG_HEADERS)
    assert refused.status_code == 403
    assert recording_registry.pulled == []

    # curl / CLI (no browser headers) and the daemon's own page still reach it.
    await client.get("/v1/agents/mine/download")
    await client.get("/v1/agents/docs-page/download", headers={"Sec-Fetch-Site": "same-origin"})
    await client.get("/v1/agents/own-origin/download", headers={"Origin": "http://127.0.0.1:1111"})
    assert recording_registry.pulled == ["mine", "docs-page", "own-origin"]


async def test_allowlisted_origin_may_use_acting_gets(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    recording_registry: _RecordingRegistry,
) -> None:
    # Same escape hatch as for writes: an operator-listed dev renderer is trusted.
    monkeypatch.setenv(desktop_module.ORIGINS_ENV, "http://localhost:5187")
    await client.get(
        "/v1/agents/dev/download",
        headers={"Origin": "http://localhost:5187", "Sec-Fetch-Site": "same-site"},
    )
    assert recording_registry.pulled == ["dev"]


async def test_cross_site_media_loads_are_not_gated(
    client: AsyncClient, _standalone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.config import ConfigManager

    monkeypatch.setattr(app.state, "config_manager", ConfigManager(_standalone), raising=False)
    # The packaged renderer (file://) loads files exactly like this; gating every
    # GET instead of the named set would blank its images.
    image = _standalone / "pic.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    static = await client.get("/v1/static/images", params={"path": str(image)}, headers=IMG_HEADERS)
    assert static.status_code == 200, static.text[:200]


# --- the opaque origin -------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/config", "/v1/agents", "/v1/config/system-prompt"])
async def test_null_origin_reads_get_no_grant_off_health(
    client: AsyncClient, path: str, _standalone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager

    # The lifespan (not run under ASGITransport) is what installs these.
    monkeypatch.setattr(app.state, "config_manager", ConfigManager(_standalone), raising=False)
    monkeypatch.setattr(app.state, "agent_registry", AgentRegistry(_standalone), raising=False)
    # A sandboxed iframe or data: document on any site presents "null".
    response = await client.get(path, headers={"Origin": "null"})
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


async def test_null_origin_preflight_for_a_write_gets_no_grant(client: AsyncClient) -> None:
    preflight = await client.options(
        "/v1/config/system-prompt",
        headers={"Origin": "null", "Access-Control-Request-Method": "PATCH"},
    )
    assert "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-credentials" not in preflight.headers

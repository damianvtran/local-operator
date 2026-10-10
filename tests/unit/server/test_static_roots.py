"""The ``/v1/static/*`` path policy and response policy.

Two layers, because each can regress alone:

* the pure policy (``utils/static_roots.py``) -- which realpaths may be served;
* the live routes through the real app (``test_app_client``), because the roots
  are assembled from ``app.state`` and the headers/CORS are applied by a
  middleware in ``server/app.py`` -- neither exists at the unit layer.

Why this matters (turn-supplements S-6/F6): the routes used to serve any readable
file whose extension passed a mime allowlist, to any origin. Every guard below has
a counterpart test that fails when the guard is removed; the PR body records the
mutation runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.server import desktop
from local_operator.server.utils import static_roots
from local_operator.server.utils.static_roots import (
    ROOTS_ENV,
    ServedRoots,
    StaticPathDenied,
    build_roots,
    resolve_servable,
)

PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(autouse=True)
def _isolated_policy(monkeypatch, tmp_path):
    """No ambient roots: the agent home is a tmp dir, the env/claim state is clean."""
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "agent-home"))
    monkeypatch.delenv(ROOTS_ENV, raising=False)
    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.delenv(desktop.ORIGINS_ENV, raising=False)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    static_roots.clear_live_cache()
    yield
    static_roots.clear_live_cache()


@pytest.fixture
def workspace(tmp_path) -> Path:
    """A directory made a root through ``static.roots`` (the configured arm)."""
    directory = tmp_path / "workspace"
    directory.mkdir()
    return directory


def _roots(tmp_path: Path, workspace: Path | None = None, **kwargs) -> ServedRoots:
    config = {"static": {"roots": [str(workspace)]}} if workspace else {}
    return build_roots(tmp_path / "config", config, **kwargs)


def _denied(raw: str, roots: ServedRoots) -> StaticPathDenied:
    with pytest.raises(StaticPathDenied) as caught:
        resolve_servable(raw, roots)
    return caught.value


# --------------------------------------------------------------------------- policy


def test_a_file_inside_a_root_is_served(tmp_path, workspace):
    target = workspace / "a.png"
    target.write_bytes(PNG)
    assert resolve_servable(str(target), _roots(tmp_path, workspace)) == target.resolve()


def test_a_file_outside_every_root_is_refused_403(tmp_path, workspace):
    outside = tmp_path / "elsewhere" / "secret.png"
    outside.parent.mkdir()
    outside.write_bytes(PNG)
    assert _denied(str(outside), _roots(tmp_path, workspace)).status == 403


def test_outside_the_roots_is_403_whether_or_not_the_file_exists(tmp_path, workspace):
    """The route must not be an existence oracle for the rest of the disk."""
    roots = _roots(tmp_path, workspace)
    real = tmp_path / "real.png"
    real.write_bytes(PNG)
    assert _denied(str(real), roots).status == 403
    assert _denied(str(tmp_path / "missing.png"), roots).status == 403


def test_dotdot_is_refused_even_when_it_lands_back_inside(tmp_path, workspace):
    target = workspace / "a.png"
    target.write_bytes(PNG)
    sneaky = f"{workspace}/sub/../a.png"
    assert _denied(sneaky, _roots(tmp_path, workspace)).status == 403


def test_dotdot_escape_is_refused(tmp_path, workspace):
    (tmp_path / "secret.png").write_bytes(PNG)
    assert _denied(f"{workspace}/../secret.png", _roots(tmp_path, workspace)).status == 403


def test_a_symlink_inside_a_root_pointing_outside_is_refused(tmp_path, workspace):
    """Realpath comparison: the link's own location is inside, its target is not."""
    secret = tmp_path / "outside.png"
    secret.write_bytes(PNG)
    link = workspace / "innocent.png"
    link.symlink_to(secret)
    assert _denied(str(link), _roots(tmp_path, workspace)).status == 403


def test_a_symlink_pointing_inside_is_served_as_its_target(tmp_path, workspace):
    target = workspace / "real.png"
    target.write_bytes(PNG)
    link = workspace / "alias.png"
    link.symlink_to(target)
    assert resolve_servable(str(link), _roots(tmp_path, workspace)) == target.resolve()


def test_a_symlinked_root_is_compared_by_realpath(tmp_path):
    real = tmp_path / "real-root"
    real.mkdir()
    (real / "a.png").write_bytes(PNG)
    alias = tmp_path / "alias-root"
    alias.symlink_to(real)
    roots = _roots(tmp_path, alias)
    assert resolve_servable(str(alias / "a.png"), roots) == (real / "a.png").resolve()


def test_a_sibling_sharing_a_name_prefix_is_not_inside(tmp_path, workspace):
    """``/x/workspace-evil`` must not pass for ``/x/workspace`` (string-prefix trap)."""
    evil = tmp_path / "workspace-evil"
    evil.mkdir()
    (evil / "a.png").write_bytes(PNG)
    assert _denied(str(evil / "a.png"), _roots(tmp_path, workspace)).status == 403


def test_a_relative_path_is_refused(tmp_path, workspace):
    assert _denied("a.png", _roots(tmp_path, workspace)).status == 400


def test_a_nul_byte_is_refused(tmp_path, workspace):
    assert _denied(f"{workspace}/a\x00.png", _roots(tmp_path, workspace)).status == 400


def test_a_symlink_loop_is_refused_not_a_500(tmp_path, workspace):
    loop = workspace / "loop.png"
    loop.symlink_to(loop)
    assert _denied(str(loop), _roots(tmp_path, workspace)).status in (400, 403, 404)


def test_a_directory_is_not_a_regular_file(tmp_path, workspace):
    (workspace / "dir.png").mkdir()
    assert _denied(str(workspace / "dir.png"), _roots(tmp_path, workspace)).status == 400


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX fifos")
def test_a_fifo_is_refused_rather_than_hanging_the_read(tmp_path, workspace):
    fifo = workspace / "pipe.png"
    os.mkfifo(fifo)
    assert _denied(str(fifo), _roots(tmp_path, workspace)).status == 400


def test_a_missing_file_inside_a_root_is_404(tmp_path, workspace):
    assert _denied(str(workspace / "nope.png"), _roots(tmp_path, workspace)).status == 404


def test_dot_directories_below_a_root_are_refused(tmp_path, workspace):
    """A session rooted at ``~`` must not expose ``~/.ssh`` to the mime allowlist."""
    hidden = workspace / ".ssh"
    hidden.mkdir()
    (hidden / "key.png").write_bytes(PNG)
    (workspace / ".hidden.png").write_bytes(PNG)
    roots = _roots(tmp_path, workspace)
    assert _denied(str(hidden / "key.png"), roots).status == 403
    assert _denied(str(workspace / ".hidden.png"), roots).status == 403


def test_a_root_that_is_itself_under_a_dot_directory_still_serves(tmp_path):
    """Only the part BELOW the root is judged: ``~/.local-operator/sessions`` works."""
    dot_root = tmp_path / ".cfg" / "sessions"
    dot_root.mkdir(parents=True)
    (dot_root / "s.png").write_bytes(PNG)
    roots = build_roots(tmp_path / ".cfg", {})
    assert resolve_servable(str(dot_root / "s.png"), roots) == (dot_root / "s.png").resolve()


def test_the_filesystem_root_can_never_be_a_root(tmp_path):
    roots = build_roots(tmp_path / "config", {"static": {"roots": ["/"]}})
    assert Path("/") not in roots.roots
    assert _denied("/etc/hosts", roots).status == 403


def test_a_relative_configured_root_is_ignored(tmp_path):
    roots = build_roots(tmp_path / "config", {"static": {"roots": ["relative/dir"]}})
    assert all(root.is_absolute() for root in roots.roots)
    assert not any(root.name == "dir" for root in roots.roots)


def test_the_built_in_roots_are_agent_home_sessions_and_uploads(tmp_path):
    config_dir = tmp_path / "config"
    roots = build_roots(config_dir, {})
    expected = {
        (tmp_path / "agent-home").resolve(),
        (config_dir / "sessions").resolve(),
        (config_dir / "uploads").resolve(),
    }
    assert expected <= set(roots.roots)


def test_the_env_var_adds_roots(tmp_path, monkeypatch):
    extra = tmp_path / "from-env"
    extra.mkdir()
    (extra / "a.png").write_bytes(PNG)
    monkeypatch.setenv(ROOTS_ENV, os.pathsep.join(["", str(extra)]))
    assert resolve_servable(str(extra / "a.png"), build_roots(tmp_path / "config", {}))


def test_an_agent_working_directory_is_a_root(tmp_path):
    cwd = tmp_path / "agent-cwd"
    cwd.mkdir()
    (cwd / "a.png").write_bytes(PNG)
    roots = build_roots(tmp_path / "config", {}, agent_cwds=[str(cwd)])
    assert resolve_servable(str(cwd / "a.png"), roots)


def test_a_live_session_working_directory_is_a_root_and_a_stale_one_is_not(tmp_path):
    from local_operator.session.runtime import registry

    config_dir = tmp_path / "config"
    live, dead = tmp_path / "live-cwd", tmp_path / "dead-cwd"
    for directory in (live, dead):
        directory.mkdir()
        (directory / "a.png").write_bytes(PNG)

    def record(pid: int, cwd: Path) -> registry.SessionRecord:
        return registry.SessionRecord(
            pid=pid,
            kind="tui",
            session_id=f"s{pid}",
            conversation_name="c",
            cwd=str(cwd),
            model_label="",
            control_port=0,
            control_key="",
        )

    registry.publish(record(os.getpid(), live), config_dir)
    # A pid that cannot be running: scanned as stale, so its cwd must not widen the roots.
    registry.publish(record(2**22 + 12345, dead), config_dir)
    roots = build_roots(config_dir, {})
    assert resolve_servable(str(live / "a.png"), roots)
    assert _denied(str(dead / "a.png"), roots).status == 403


# --------------------------------------------------------------------- live routes

ROUTES = {
    "images": ("a.png", PNG),
    "audio": ("a.mp3", b"ID3\x03\x00\x00\x00\x00\x00\x00"),
    "videos": ("a.mp4", b"\x00\x00\x00\x18ftypmp42"),
    "html": ("a.html", b"<!doctype html><p>hi</p>"),
}


@pytest.fixture
def configured(test_app_client, tmp_path):
    """The live app with ``workspace`` added through ``static.roots``."""
    from local_operator.server.app import app

    directory = tmp_path / "workspace"
    directory.mkdir()
    app.state.config_manager.set_config_value("static", {"roots": [str(directory)]})
    return directory


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_every_route_serves_inside_and_refuses_outside(test_app_client, configured, route):
    name, body = ROUTES[route]
    inside = configured / name
    inside.write_bytes(body)
    outside = configured.parent / "outside" / name
    outside.parent.mkdir()
    outside.write_bytes(body)

    ok = await test_app_client.get(f"/v1/static/{route}", params={"path": str(inside)})
    assert ok.status_code == 200, ok.text
    assert ok.content == body

    denied = await test_app_client.get(f"/v1/static/{route}", params={"path": str(outside)})
    assert denied.status_code == 403
    assert "outside" in denied.json()["detail"]


@pytest.mark.asyncio
async def test_a_symlink_escape_through_the_route_is_403(test_app_client, configured, tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(PNG)
    (configured / "link.png").symlink_to(secret)
    response = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / "link.png")}
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_dotdot_through_the_route_is_403(test_app_client, configured, tmp_path):
    (tmp_path / "secret.png").write_bytes(PNG)
    response = await test_app_client.get(
        "/v1/static/images", params={"path": f"{configured}/../secret.png"}
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_a_disallowed_extension_inside_a_root_is_still_400(test_app_client, configured):
    (configured / "notes.txt").write_text("x")
    response = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / "notes.txt")}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_a_config_edit_takes_effect_without_a_restart(test_app_client, configured, tmp_path):
    from local_operator.server.app import app

    other = tmp_path / "later"
    other.mkdir()
    (other / "a.png").write_bytes(PNG)
    url = ("/v1/static/images", {"path": str(other / "a.png")})
    assert (await test_app_client.get(url[0], params=url[1])).status_code == 403
    app.state.config_manager.set_config_value("static", {"roots": [str(configured), str(other)]})
    assert (await test_app_client.get(url[0], params=url[1])).status_code == 200


@pytest.mark.asyncio
async def test_a_write_from_another_process_takes_effect_without_a_restart(
    test_app_client, configured, tmp_path
):
    """The settings row is LIVE, and a ``lop config edit`` is a different process.

    A second ``ConfigManager`` stands in for it: the server's own manager keeps the
    snapshot it loaded, so only a read of the file on disk sees this write.
    """
    from local_operator.config import ConfigManager
    from local_operator.server.app import app

    other = tmp_path / "later"
    other.mkdir()
    (other / "a.png").write_bytes(PNG)
    url = ("/v1/static/images", {"path": str(other / "a.png")})
    assert (await test_app_client.get(url[0], params=url[1])).status_code == 403
    ConfigManager(config_dir=app.state.config_manager.config_dir).set_config_value(
        "static", {"roots": [str(other)]}
    )
    assert (await test_app_client.get(url[0], params=url[1])).status_code == 200


# ------------------------------------------------------------------------- headers


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_every_route_carries_nosniff_and_a_csp_on_success(test_app_client, configured, route):
    name, body = ROUTES[route]
    (configured / name).write_bytes(body)
    response = await test_app_client.get(
        f"/v1/static/{route}", params={"path": str(configured / name)}
    )
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors" in csp


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_the_headers_ride_error_responses_too(test_app_client, configured, route):
    response = await test_app_client.get(f"/v1/static/{route}", params={"path": "/etc/hosts"})
    assert response.status_code == 403
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_html_gets_the_preview_csp_and_media_gets_the_strict_one(test_app_client, configured):
    (configured / "a.html").write_bytes(ROUTES["html"][1])
    (configured / "a.png").write_bytes(PNG)
    html = await test_app_client.get("/v1/static/html", params={"path": str(configured / "a.html")})
    image = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / "a.png")}
    )
    html_csp = html.headers["content-security-policy"]
    # The page may call https APIs but never the loopback daemon, and a direct
    # navigation runs in an opaque origin.
    assert "connect-src https:" in html_csp
    assert "sandbox allow-scripts" in html_csp
    assert "script-src" in html_csp
    media_csp = image.headers["content-security-policy"]
    assert "script-src" not in media_csp
    assert "connect-src" not in media_csp


@pytest.mark.asyncio
async def test_frame_ancestors_names_only_the_app(test_app_client, configured):
    (configured / "a.html").write_bytes(ROUTES["html"][1])
    response = await test_app_client.get(
        "/v1/static/html", params={"path": str(configured / "a.html")}
    )
    csp = response.headers["content-security-policy"]
    ancestors = [d for d in csp.split("; ") if d.startswith("frame-ancestors")][0]
    assert "'self'" in ancestors and "file:" in ancestors
    assert "*" in ancestors  # only as a PORT wildcard on loopback ...
    assert "http://evil" not in ancestors and " *" not in ancestors.replace(":*", "")


@pytest.mark.asyncio
async def test_an_admitted_origin_list_replaces_the_loopback_dev_framing(
    test_app_client, configured, monkeypatch
):
    monkeypatch.setenv(desktop.ORIGINS_ENV, "https://app.example")
    (configured / "a.html").write_bytes(ROUTES["html"][1])
    response = await test_app_client.get(
        "/v1/static/html", params={"path": str(configured / "a.html")}
    )
    ancestors = [
        d
        for d in response.headers["content-security-policy"].split("; ")
        if d.startswith("frame-ancestors")
    ][0]
    assert "https://app.example" in ancestors
    assert "localhost" not in ancestors


@pytest.mark.asyncio
async def test_other_routes_do_not_get_the_static_policy(test_app_client):
    response = await test_app_client.get("/v1/agents")
    assert "content-security-policy" not in response.headers


# ---------------------------------------------------------------------------- CORS


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_no_cors_grant_to_a_foreign_origin_on_success(test_app_client, configured, route):
    """The exploit: a visited page ``fetch``-ing a file it can name."""
    name, body = ROUTES[route]
    (configured / name).write_bytes(body)
    response = await test_app_client.get(
        f"/v1/static/{route}",
        params={"path": str(configured / name)},
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.asyncio
async def test_no_cors_grant_on_a_refusal_either(test_app_client, configured):
    response = await test_app_client.get(
        "/v1/static/images",
        params={"path": "/etc/hosts"},
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_the_opaque_null_origin_gets_no_grant(test_app_client, configured):
    """A sandboxed preview document (and a ``file:`` page) sends ``Origin: null``."""
    (configured / "a.png").write_bytes(PNG)
    response = await test_app_client.get(
        "/v1/static/images",
        params={"path": str(configured / "a.png")},
        headers={"Origin": "null"},
    )
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_an_admitted_origin_keeps_its_grant(test_app_client, configured, monkeypatch):
    monkeypatch.setenv(desktop.ORIGINS_ENV, "https://app.example")
    (configured / "a.png").write_bytes(PNG)
    response = await test_app_client.get(
        "/v1/static/images",
        params={"path": str(configured / "a.png")},
        headers={"Origin": "https://app.example"},
    )
    assert response.headers["access-control-allow-origin"] == "https://app.example"


@pytest.mark.asyncio
async def test_an_originless_request_is_unaffected(test_app_client, configured):
    """``<img>``/``<video>`` loads and curl send no Origin and need no grant."""
    (configured / "a.png").write_bytes(PNG)
    response = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / "a.png")}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_a_preflight_to_a_static_route_grants_a_foreign_origin_nothing(
    test_app_client, configured
):
    response = await test_app_client.options(
        "/v1/static/images",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_other_routes_keep_the_historical_cors_echo(test_app_client):
    """Scope guard: only /v1/static changed; ``/health`` is read cross-origin by the app."""
    response = await test_app_client.get("/v1/agents", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

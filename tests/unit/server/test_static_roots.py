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
from httpx import ASGITransport, AsyncClient

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


def _denied(raw: str, roots: ServedRoots, *, general: bool = False) -> StaticPathDenied:
    with pytest.raises(StaticPathDenied) as caught:
        resolve_servable(raw, roots, general=general)
    return caught.value


def _other_spellings_of(directory: Path) -> list[Path]:
    """``directory`` plus every spelling that names it without being that string.

    Only the ones the volume actually supports are returned: a case flip needs a
    case-insensitive filesystem, the ``/System/Volumes/Data`` twin needs the macOS
    firmlink. Reviews R2-1 and security S-1 reproduced the ``$HOME``-as-root
    exposure through exactly these two, so the guards are tested through them
    rather than by mocking ``samefile``.
    """
    spellings = [directory]
    flipped = Path(str(directory).swapcase())
    if flipped.exists() and flipped.samefile(directory):
        spellings.append(flipped)
    try:
        twin = Path("/System/Volumes/Data") / directory.relative_to("/")
    except ValueError:  # pragma: no cover -- every test path is absolute
        return spellings
    if twin.exists() and twin.samefile(directory):
        spellings.append(twin)
    return spellings


# --------------------------------------------------------------------------- policy


def test_a_file_inside_a_root_is_served(tmp_path, workspace):
    target = workspace / "a.png"
    target.write_bytes(PNG)
    assert (
        resolve_servable(str(target), _roots(tmp_path, workspace), general=False)
        == target.resolve()
    )


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
    assert (
        resolve_servable(str(link), _roots(tmp_path, workspace), general=False) == target.resolve()
    )


def test_a_symlinked_root_is_compared_by_realpath(tmp_path):
    real = tmp_path / "real-root"
    real.mkdir()
    (real / "a.png").write_bytes(PNG)
    alias = tmp_path / "alias-root"
    alias.symlink_to(real)
    roots = _roots(tmp_path, alias)
    assert (
        resolve_servable(str(alias / "a.png"), roots, general=False) == (real / "a.png").resolve()
    )


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
    assert (
        resolve_servable(str(dot_root / "s.png"), roots, general=False)
        == (dot_root / "s.png").resolve()
    )


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
    assert resolve_servable(
        str(extra / "a.png"), build_roots(tmp_path / "config", {}), general=False
    )


def _live_record(pid: int, cwd: Path):
    from local_operator.session.runtime import registry

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


def test_a_live_session_working_directory_is_a_root_and_a_stale_one_is_not(tmp_path):
    from local_operator.session.runtime import registry

    config_dir = tmp_path / "config"
    live, dead = tmp_path / "live-cwd", tmp_path / "dead-cwd"
    for directory in (live, dead):
        directory.mkdir()
        (directory / "a.png").write_bytes(PNG)

    registry.publish(_live_record(os.getpid(), live), config_dir)
    # A pid that cannot be running: scanned as stale, so its cwd must not widen the roots.
    registry.publish(_live_record(2**22 + 12345, dead), config_dir)
    roots = build_roots(config_dir, {})
    assert resolve_servable(str(live / "a.png"), roots, general=False)
    assert _denied(str(dead / "a.png"), roots).status == 403


def test_a_session_started_in_home_does_not_make_home_a_root(tmp_path, monkeypatch):
    """R1: 34 live sessions on the reference host run in ``~``; ``~/Pictures`` stays 403.

    R2-1/S-1: the rule is about the DIRECTORY, not its spelling. ``/users/me`` and
    ``/System/Volumes/Data/Users/me`` are ``$HOME`` on this filesystem and
    ``resolve()`` canonicalises neither, so a string comparison let a cwd written
    that way become a root and served ``~/Pictures``.
    """
    from local_operator.session.runtime import registry

    home = tmp_path / "home"
    (home / "Pictures").mkdir(parents=True)
    (home / "Pictures" / "a.png").write_bytes(PNG)
    monkeypatch.setenv("HOME", str(home))
    config_dir = tmp_path / "config"
    for spelling in _other_spellings_of(home):
        registry.publish(_live_record(os.getpid(), spelling), config_dir)
        static_roots.clear_live_cache()
        roots = build_roots(config_dir, {})
        assert home.resolve() not in roots.roots, spelling
        assert _denied(str(home / "Pictures" / "a.png"), roots).status == 403, spelling


def test_a_session_started_above_home_does_not_make_an_ancestor_a_root(tmp_path, monkeypatch):
    from local_operator.session.runtime import registry

    home = tmp_path / "users" / "me"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    config_dir = tmp_path / "config"
    registry.publish(_live_record(os.getpid(), home.parent), config_dir)
    assert home.parent.resolve() not in build_roots(config_dir, {}).roots


def test_a_session_inside_home_is_still_a_root(tmp_path, monkeypatch):
    """The R1 rule removes ``~`` and its ancestors, not every project under it."""
    from local_operator.session.runtime import registry

    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    (project / "a.png").write_bytes(PNG)
    monkeypatch.setenv("HOME", str(home))
    config_dir = tmp_path / "config"
    registry.publish(_live_record(os.getpid(), project), config_dir)
    assert resolve_servable(str(project / "a.png"), build_roots(config_dir, {}), general=False)


def test_home_itself_is_served_when_the_operator_configures_it(tmp_path, monkeypatch):
    """Widening past the built-ins is the operator's explicit opt-in (``static.roots``).

    Including one typed in another spelling (security S-1): the identity test that
    closes the implicit arm must not turn an operator's ``/users/me`` opt-in into a
    root that silently does nothing.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / "a.png").write_bytes(PNG)
    monkeypatch.setenv("HOME", str(home))
    for spelling in _other_spellings_of(home):
        roots = build_roots(tmp_path / "config", {"static": {"roots": [str(spelling)]}})
        assert resolve_servable(str(home / "a.png"), roots, general=False), spelling


@pytest.mark.parametrize("spelling", ["parent", "dotdot", "env"])
def test_an_ancestor_of_home_is_never_a_root_however_it_is_spelled(tmp_path, monkeypatch, spelling):
    """R6: ``/Users``, ``~/..`` and the macOS data volume all contain ``$HOME``."""
    home = tmp_path / "users" / "me"
    home.mkdir(parents=True)
    (home.parent / "other.png").write_bytes(PNG)
    monkeypatch.setenv("HOME", str(home))
    ancestor = {"parent": str(home.parent), "dotdot": "~/..", "env": str(home.parent)}[spelling]
    config = {"static": {"roots": [ancestor]}} if spelling != "env" else {}
    if spelling == "env":
        monkeypatch.setenv(ROOTS_ENV, ancestor)  # the variable bypassed the validator before
    roots = build_roots(tmp_path / "config", config)
    assert home.parent.resolve() not in roots.roots
    assert _denied(str(home.parent / "other.png"), roots).status == 403


def test_the_agent_registry_is_not_a_source_of_roots():
    """R2: ``current_working_directory`` is writable through the ungated PATCH route."""
    import inspect

    assert "agent_cwds" not in inspect.signature(build_roots).parameters


def test_an_unknown_tilde_user_is_the_same_403_not_a_500(tmp_path, workspace):
    """R5: ``expanduser`` raised RuntimeError outside the guard (a 500 and a user oracle)."""
    roots = _roots(tmp_path, workspace)
    unknown = _denied("~no-such-user-zzz/a.png", roots)
    outside = _denied(str(tmp_path / "nowhere" / "a.png"), roots)
    assert (unknown.status, unknown.detail) == (outside.status, outside.detail)
    assert unknown.status == 403


def test_a_symlink_loop_outside_the_roots_is_the_same_403(tmp_path, workspace):
    """R5: the loop used to be a distinct 400, an existence oracle for the rest of the disk."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    loop = elsewhere / "loop.png"
    loop.symlink_to(loop)
    roots = _roots(tmp_path, workspace)
    looped = _denied(str(loop), roots)
    plain = _denied(str(elsewhere / "nope.png"), roots)
    assert (looped.status, looped.detail) == (plain.status, plain.detail)


def test_an_over_long_component_inside_a_root_is_the_same_403_not_a_500(tmp_path, workspace):
    """S-4: ``ENAMETOOLONG`` escaped as the handler's 500 -- the one non-4xx answer.

    The classification is by errno, because ``Path.exists()`` swallows
    ``ENAMETOOLONG`` on Python 3.13+ and re-raises it on 3.12, which made the same
    request a 500 on CI's interpreter and a 404 here.
    """
    over_long = str(workspace / ("a" * 300) / "b.png")
    denied = _denied(over_long, _roots(tmp_path, workspace))
    assert (denied.status, denied.detail) == (403, static_roots.OUTSIDE_ROOTS_DETAIL)


def test_the_403_body_names_the_remedy(tmp_path, workspace):
    """R3(c): a thumbnail that breaks tells the user how to fix it."""
    assert (
        "static.roots"
        in _denied(str(tmp_path / "nowhere" / "a.png"), _roots(tmp_path, workspace)).detail
    )


def test_a_dot_directory_at_any_depth_is_refused(tmp_path, workspace):
    """R7: the rule applies to EVERY segment below the root, not only the first."""
    hidden = workspace / "x" / ".hidden" / "y"
    hidden.mkdir(parents=True)
    (hidden / "a.png").write_bytes(PNG)
    assert _denied(str(hidden / "a.png"), _roots(tmp_path, workspace)).status == 403


def test_an_unreadable_file_is_403_not_a_crash(tmp_path, workspace, monkeypatch):
    """R7: the ``os.access`` guard. Patched, not chmod: root-run CI reads a 0o000 file."""
    target = workspace / "a.png"
    target.write_bytes(PNG)
    monkeypatch.setattr(static_roots.os, "access", lambda *_a, **_k: False)
    denied = _denied(str(target), _roots(tmp_path, workspace))
    assert denied.status == 403
    assert "not accessible" in denied.detail


def test_a_differently_cased_parent_is_served_on_a_case_insensitive_volume(tmp_path, workspace):
    """R10: a false 403 for a file that is in the workspace. Skipped where case matters."""
    (workspace / "a.png").write_bytes(PNG)
    flipped = Path(str(tmp_path.resolve()).swapcase()) / "workspace" / "a.png"
    if not flipped.exists():
        pytest.skip("case-sensitive filesystem: a differently-cased path is a different path")
    assert resolve_servable(str(flipped), _roots(tmp_path, workspace), general=False)


# -------------------------------------------------- settings validator (write facade)


@pytest.mark.parametrize(
    "bad",
    [["relative/dir"], ["/"], [123], [None], ["~/.."]],
)
def test_static_roots_validator_refuses_what_would_widen_to_anywhere(bad):
    """R7: removing ``validate_value=_validate_static_roots`` survived every other test."""
    from local_operator import settings_io

    setting = next(item for item in settings_io.SETTINGS if item.key == "static.roots")
    assert settings_io.validate(setting, bad) is not None


def test_static_roots_validator_accepts_ordinary_absolute_directories(tmp_path):
    from local_operator import settings_io

    setting = next(item for item in settings_io.SETTINGS if item.key == "static.roots")
    assert settings_io.validate(setting, [str(tmp_path), "~/Documents"]) is None
    assert settings_io.validate(setting, []) is None


def test_static_roots_validator_refuses_an_ancestor_of_home(tmp_path, monkeypatch):
    """R6 at the write facade, with HOME pinned (the suite's own HOME is a scratch dir)."""
    from local_operator import settings_io

    home = tmp_path / "users" / "me"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    setting = next(item for item in settings_io.SETTINGS if item.key == "static.roots")
    refusal = settings_io.validate(setting, [str(home.parent)])
    assert refusal is not None and "contains your home" in refusal
    assert settings_io.validate(setting, [str(home)]) is None  # ``~`` itself is the opt-in


def test_the_macos_data_volume_firmlink_counts_as_an_ancestor_of_home():
    """R6: ``/System/Volumes/Data`` is ``/`` by another name; name matching missed it."""
    data = Path("/System/Volumes/Data")
    if not (data / Path.home().resolve().relative_to("/")).exists():
        pytest.skip("not a macOS data-volume layout")
    assert static_roots.root_refusal(data) is not None


# ------------------------------------------------------------------------ Host check


@pytest.mark.parametrize(
    ("host", "bound", "ok"),
    [
        ("127.0.0.1:1111", "127.0.0.1", True),
        ("[::1]:1111", "127.0.0.1", True),
        ("localhost:1111", "127.0.0.1", True),
        ("LocalHost", None, True),
        ("rebind.attacker.test:8080", "127.0.0.1", False),
        ("rebind.attacker.test", None, False),
        ("localhost.attacker.test", "127.0.0.1", False),
        # S-3: userinfo is not an authority, and neither is an empty Host. A browser
        # sends neither, so they need not pass -- and the ``:`` split would read the
        # attacker's name out of the first two.
        ("localhost:80@attacker.test", "127.0.0.1", False),
        ("[::1]@attacker.test", None, False),
        ("attacker@localhost", "127.0.0.1", False),
        ("", "127.0.0.1", False),
        ("   ", None, False),
        ("mac.lan:1111", "mac.lan", True),
        ("mac.lan:1111", "127.0.0.1", False),
        # Fail-closed for a wildcard/empty announced host (RFC §3.4): there is no
        # name to compare against, so a DNS name is refused; IP literals and
        # localhost still pass. The old bypass that admitted every name here is
        # deleted.
        ("anything.example", "0.0.0.0", False),
        ("anything.example", "::", False),
        ("anything.example", "", False),
        ("127.0.0.1:1111", "0.0.0.0", True),
        ("localhost:1111", "0.0.0.0", True),
        (None, "127.0.0.1", True),
    ],
)
def test_host_is_acceptable(host, bound, ok):
    assert static_roots.host_is_acceptable(host, bound) is ok


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
    # The app's own renderer dials ``http://127.0.0.1:<port>``; httpx's default
    # ``Host: test`` is a DNS name, which the rebinding guard (R8) rightly refuses.
    test_app_client.headers["Host"] = "127.0.0.1:1111"
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


@pytest.mark.asyncio
async def test_a_rebinding_host_is_refused_before_the_file_is_looked_at(
    test_app_client, configured
):
    """R8: a rebinding page is same-origin, so CORS cannot help; the Host it sends is its own."""
    (configured / "a.png").write_bytes(PNG)
    served = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / "a.png")}
    )
    assert served.status_code == 200
    rebound = await test_app_client.get(
        "/v1/static/images",
        params={"path": str(configured / "a.png")},
        headers={"Host": "rebind.attacker.test:8080"},
    )
    assert rebound.status_code == 403
    assert "nosniff" == rebound.headers["x-content-type-options"]
    # and a Host that IS the app's own spelling still works
    ok = await test_app_client.get(
        "/v1/static/images",
        params={"path": str(configured / "a.png")},
        headers={"Host": "127.0.0.1:1111"},
    )
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_other_routes_ignore_the_host_check(test_app_client):
    response = await test_app_client.get("/v1/agents", headers={"Host": "rebind.attacker.test"})
    assert response.status_code != 403 or "Host is not allowed" not in response.text


@pytest.mark.asyncio
async def test_patching_an_agent_cwd_does_not_widen_what_is_served(test_app_client, tmp_path):
    """R2, the reviewer's exact repro: PATCH a cwd with a foreign Origin, then GET.

    Before: GET 403 -> PATCH 200 (ACAO echoed, cwd persisted) -> GET 200.
    """
    from local_operator.agents import AgentEditFields
    from local_operator.server.app import app

    private = tmp_path / "private"
    private.mkdir()
    (private / "a.png").write_bytes(PNG)
    agent = app.state.agent_registry.create_agent(
        AgentEditFields(
            name="cwd-agent",
            security_prompt="",
            hosting="openai",
            model="gpt-4",
            description="",
            last_message="",
            tags=[],
            categories=[],
            temperature=0.2,
            top_p=0.5,
            top_k=10,
            max_tokens=100,
            stop=[],
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=1,
            current_working_directory=".",
        )
    )
    # Host pinned to the app's own spelling, or every 403 below would be the Host guard's.
    test_app_client.headers["Host"] = "127.0.0.1:1111"
    uploads = app.state.config_manager.config_dir / "uploads"
    uploads.mkdir(exist_ok=True)
    (uploads / "control.png").write_bytes(PNG)
    control = await test_app_client.get(
        "/v1/static/images", params={"path": str(uploads / "control.png")}
    )
    assert control.status_code == 200  # positive control: the request path itself works
    params = {"path": str(private / "a.png")}
    assert (await test_app_client.get("/v1/static/images", params=params)).status_code == 403

    patched = await test_app_client.patch(
        f"/v1/agents/{agent.id}",
        json={"current_working_directory": str(tmp_path)},
        headers={"Origin": "https://attacker.example"},
    )
    assert patched.status_code == 200  # the route stays ungated: the point is it no longer matters
    assert app.state.agent_registry.get_agent(agent.id).current_working_directory == str(tmp_path)

    after = await test_app_client.get("/v1/static/images", params=params)
    assert after.status_code == 403


@pytest.mark.asyncio
async def test_unknown_tilde_user_through_the_route_is_403_not_500(test_app_client, configured):
    """R5 end to end: the reproduced 500 was ``Could not determine home directory.``"""
    response = await test_app_client.get(
        "/v1/static/images", params={"path": "~no-such-user-zzz/a.png"}
    )
    assert response.status_code == 403
    assert "static.roots" in response.json()["detail"]


@pytest.mark.asyncio
async def test_an_over_long_component_through_the_route_is_403_not_500(test_app_client, configured):
    """S-4 end to end: the 500 the security round reproduced on a 3.12 interpreter."""
    response = await test_app_client.get(
        "/v1/static/images", params={"path": str(configured / ("a" * 300) / "b.png")}
    )
    assert response.status_code == 403
    assert "static.roots" in response.json()["detail"]


@pytest.mark.asyncio
async def test_the_daemons_own_bound_name_is_admitted_through_the_middleware(
    test_app_client, configured
):
    """The wiring half of R8: the middleware feeds ``host_is_acceptable`` the announced bind."""
    from local_operator.server import registry as serve_registry
    from local_operator.server.app import app

    (configured / "a.png").write_bytes(PNG)
    params = {"path": str(configured / "a.png")}
    headers = {"Host": "mac.lan:1111"}
    assert (
        await test_app_client.get("/v1/static/images", params=params, headers=headers)
    ).status_code == 403
    setattr(app.state, serve_registry.ANNOUNCED_STATE_ATTR, ("mac.lan", 1111))
    try:
        ok = await test_app_client.get("/v1/static/images", params=params, headers=headers)
    finally:
        delattr(app.state, serve_registry.ANNOUNCED_STATE_ATTR)
    assert ok.status_code == 200


# ------------------------------------------------------- the connection gate (P1)


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.8.8.8", True),  # 127.0.0.0/8, not just .1
        ("::1", True),
        ("::ffff:127.0.0.1", True),  # a v6 listener's v4-mapped loopback client
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.0.155", False),
        ("::ffff:192.168.0.155", False),
        ("testserver", False),
        ("rebind.attacker.test", False),
        (None, False),
    ],
)
def test_is_loopback_host(host, loopback):
    assert static_roots.is_loopback_host(host) is loopback


def test_connection_mode_is_fail_closed():
    """Anything not provably loopback -- including a missing value -- is rooted."""
    assert static_roots.connection_mode("127.0.0.1") == "general"
    assert static_roots.connection_mode("::1") == "general"
    assert static_roots.connection_mode(None) == "rooted"
    assert static_roots.connection_mode("testserver") == "rooted"
    assert static_roots.connection_mode("192.168.0.155") == "rooted"


# -------------------------------------------------------------- general mode (D1)


def _general(raw: str) -> Path:
    """``resolve_servable`` in the loopback posture, with no roots configured."""
    return resolve_servable(raw, ServedRoots(roots=()), general=True)


def test_general_mode_serves_any_readable_file_outside_every_root(tmp_path):
    target = tmp_path / "elsewhere" / "x.png"
    target.parent.mkdir()
    target.write_bytes(PNG)
    assert _general(str(target)) == target.resolve()


def test_general_mode_serves_dot_component_paths(tmp_path):
    """The class the dot rule broke: session scratchpads live under ``~/.local-operator``."""
    target = tmp_path / ".local-operator" / "sessions" / "s1" / "scratchpad" / "x.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(PNG)
    assert _general(str(target)) == target.resolve()


def test_general_mode_keeps_the_raw_dotdot_refusal(tmp_path):
    (tmp_path / "x.png").write_bytes(PNG)
    sneaky = str(tmp_path) + "/../" + tmp_path.name + "/x.png"
    denied = _denied(sneaky, ServedRoots(roots=()), general=True)
    assert denied.status == 403


def test_general_mode_refuses_directories_fifos_and_missing_files(tmp_path):
    directory = tmp_path / "dir"
    directory.mkdir()
    assert _denied(str(directory), ServedRoots(roots=()), general=True).status == 400
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    assert _denied(str(fifo), ServedRoots(roots=()), general=True).status == 400
    assert _denied(str(tmp_path / "nope.png"), ServedRoots(roots=()), general=True).status == 404


def test_general_mode_uniform_403_for_the_unplaceable_classes(tmp_path):
    """Unknown ``~user`` / an over-long component: ONE neutral body, no remedy text."""
    for raw in ("~no-such-user-zzz/a.png", str(tmp_path / ("a" * 300) / "b.png")):
        denied = _denied(raw, ServedRoots(roots=()), general=True)
        assert (denied.status, denied.detail) == (403, static_roots.UNSERVABLE_DETAIL)
    assert "static.roots" not in static_roots.UNSERVABLE_DETAIL


def test_general_mode_unreadable_file_is_403_not_a_crash(tmp_path, monkeypatch):
    target = tmp_path / "x.png"
    target.write_bytes(PNG)
    monkeypatch.setattr(static_roots.os, "access", lambda *_args, **_kwargs: False)
    assert _denied(str(target), ServedRoots(roots=()), general=True).status == 403


# ------------------------------------------------------------- live routes, general


def _loopback_client() -> AsyncClient:
    """The live app reached as a LOOPBACK-ACCEPTED connection.

    ``scope["server"]`` is what the gate reads, and httpx 0.28 builds it from
    the request URL: the default ``http://test`` client's value is the name
    ``test`` (rooted, fail-closed), a ``127.0.0.1`` client's is loopback --
    the same distinction a real socket makes, without a socket.
    """
    from local_operator.server.app import app

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:1111")
    client.headers["Host"] = "127.0.0.1:1111"
    return client


@pytest.mark.asyncio
async def test_general_mode_serves_out_of_root_through_the_route(test_app_client, configured):
    """§3.7 item 3 at the app layer: the real handler chain, loopback-accepted."""
    outside = configured.parent / "outside" / "x.png"
    outside.parent.mkdir()
    outside.write_bytes(PNG)
    client = _loopback_client()
    try:
        response = await client.get("/v1/static/images", params={"path": str(outside)})
    finally:
        await client.aclose()
    assert response.status_code == 200
    assert response.content == PNG
    # §3.7 item 7: the response policy rides general-mode responses exactly as it
    # rides rooted ones -- the middleware is posture-independent by construction.
    assert response.headers["x-content-type-options"] == "nosniff"
    policy = response.headers["content-security-policy"]
    assert policy.startswith(static_roots.MEDIA_CSP)
    assert "frame-ancestors" in policy


@pytest.mark.asyncio
async def test_general_mode_keeps_the_clean_4xx_classes(test_app_client, configured):
    client = _loopback_client()
    try:
        missing = await client.get(
            "/v1/static/images", params={"path": str(configured.parent / "outside" / "nope.png")}
        )
        directory = configured.parent / "outside" / "dir"
        directory.mkdir(parents=True)
        not_a_file = await client.get("/v1/static/images", params={"path": str(directory)})
    finally:
        await client.aclose()
    assert missing.status_code == 404
    assert not_a_file.status_code == 400


@pytest.mark.asyncio
async def test_general_mode_does_not_disable_the_host_check(test_app_client, configured):
    """§3.7 item 6: the unclamp never turns the rebinding guard off."""
    outside = configured.parent / "outside" / "x.png"
    outside.parent.mkdir()
    outside.write_bytes(PNG)
    client = _loopback_client()
    try:
        rebound = await client.get(
            "/v1/static/images",
            params={"path": str(outside)},
            headers={"Host": "rebind.attacker.test:8080"},
        )
        ok = await client.get("/v1/static/images", params={"path": str(outside)})
    finally:
        await client.aclose()
    assert rebound.status_code == 403
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_general_mode_still_enforces_the_mime_allowlists(test_app_client, configured):
    """D5: unclamped PATHS, never mimes -- an HTML document has no media route."""
    outside = configured.parent / "outside"
    outside.mkdir()
    document = outside / "doc.html"
    document.write_text("<!doctype html><script>x()</script>")
    bare = outside / "blob"
    bare.write_bytes(b"\x00")
    client = _loopback_client()
    try:
        as_html = await client.get("/v1/static/images", params={"path": str(document)})
        as_bytes = await client.get("/v1/static/images", params={"path": str(bare)})
    finally:
        await client.aclose()
    assert as_html.status_code == 400
    assert as_bytes.status_code == 400


@pytest.mark.asyncio
async def test_the_gate_reads_the_socket_not_an_announcement(test_app_client, configured):
    """§3.7 items 4-5: announced loopback, but a non-loopback ACCEPTED address
    (the embedding path) stays rooted; the same file serves on loopback."""
    from local_operator.server import registry as serve_registry
    from local_operator.server.app import app

    outside = configured.parent / "outside" / "x.png"
    outside.parent.mkdir()
    outside.write_bytes(PNG)
    params = {"path": str(outside)}
    setattr(app.state, serve_registry.ANNOUNCED_STATE_ATTR, ("127.0.0.1", 1111))
    try:
        # test_app_client's URL host is the DNS NAME "test": not provably
        # loopback, so the predicate stays rooted though the daemon "announced"
        # loopback -- the mode does not trust the announcement.
        rooted = await test_app_client.get("/v1/static/images", params=params)
        client = _loopback_client()
        try:
            general = await client.get("/v1/static/images", params=params)
        finally:
            await client.aclose()
    finally:
        delattr(app.state, serve_registry.ANNOUNCED_STATE_ATTR)
    assert rooted.status_code == 403
    assert general.status_code == 200


# ---------------------------------------------------------------- the pins (D4/D5)


def test_html_csp_network_off_directives_are_pinned_against_widening():
    """§3.7 item 10: a change here is a security change for the executing-frame
    contract -- it must go through the supplements-lane review (D4)."""
    assert static_roots.HTML_CSP == (
        "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' https:; "
        "style-src 'unsafe-inline' https:; img-src data: blob: https:; "
        "font-src data: https:; media-src data: blob: https:; connect-src https:; "
        "worker-src blob:; form-action 'none'; base-uri 'none'; sandbox allow-scripts"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["audio", "images", "videos"])
async def test_a_media_family_never_serves_html_or_untyped_bytes(
    test_app_client, configured, route
):
    """§3.7 item 8 / D5: the media families carry no executable document type."""
    document = configured / "doc.html"
    document.write_bytes(b"<!doctype html><script>x()</script>")
    bare = configured / "blob"
    bare.write_bytes(b"\x00")
    for name in (document, bare):
        response = await test_app_client.get(f"/v1/static/{route}", params={"path": str(name)})
        assert response.status_code == 400, (route, name, response.text)

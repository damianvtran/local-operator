"""``/v1/desktop/skills`` — the sessionless skill catalogue.

The requirement this surface exists for: a composer must be able to ask a
daemon for a folder's skills BEFORE any session exists (a new-chat draft has no
session record to name, and no runtime), and the answer must be the SAME
vocabulary — and the same ``version`` — the eventual session discovers for that
folder. The session arm keeps answering for released clients and the ``/skills``
panel; ``cwd`` WINS when both parameters are sent.

Every row here is discovered from DISK, so the whole file runs against a
synthetic HOME and a synthetic project folder: the root shapes the contract
names are the project root (walk-up), the home root, the ecosystem roots
(absent under the scratch HOME), and the packaged builtin catalog (always
present, and filtered out of the row assertions below). The session arm is
driven through a pool
shaped like ``DesktopSessions`` (``host(request).session(...)`` →
``bridge.remote``): the only fact the route reads from a session is
``frontend_state.cwd``, and a real runtime would add nothing to a scan that
stats the same tree either way.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_catalogues
from local_operator.skills.api import PACKAGED_SKILL_ROOT

TOKEN = "desktop-skills-route-test-token"

pytestmark = pytest.mark.asyncio


class _FakeRemote:
    """A viewer facade with the one fact the route reads."""

    def __init__(self, cwd: str) -> None:
        self.frontend_state = SimpleNamespace(cwd=cwd)


class _FakePool:
    """``DesktopSessions``-shaped: the session arm's only door."""

    def __init__(self, remotes: dict[str, _FakeRemote]) -> None:
        self.remotes = remotes

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False):
        del read
        yield SimpleNamespace(remote=self.remotes[session_id])


class _RefusingPool:
    """A pool that fails loudly: a sessionless read must never open a bridge.

    The point is not that this pool works, it is that a sessionless request
    never touches it — reaching either method fails the request, so a test
    using this pool proves the arm walked no session door at all.
    """

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False):
        raise AssertionError("a sessionless skill read opened a session")
        yield  # pragma: no cover — unreachable; keeps this an async generator


@pytest_asyncio.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real catalogue router on a synthetic HOME; no daemon, no real store."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
    # The suite allow-lists this variable (read-only dirs), but a cell asserting
    # exact rows cannot tolerate an inherited extra root it never wrote.
    monkeypatch.delenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", raising=False)
    app = FastAPI()
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = SimpleNamespace(config_dir=home / ".local-operator")
    app.state.desktop_sessions = _RefusingPool()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(client=client, app=app, home=home, tmp=tmp_path)


def _data(response) -> dict[str, Any]:
    body = response.json()
    assert "result" in body, body
    return body["result"]["data"]


def _rows(data: dict[str, Any]) -> list[dict[str, str]]:
    return [{"name": row["name"], "description": row["description"]} for row in data["skills"]]


#: The packaged builtin catalog rides in every response: ``skills/api.py``
#: appends its root LAST, so it is always discovered beside the fixture's own
#: skills. The cells here assert the FIXTURE's rows, so they filter the catalog
#: out by name; that all 14 builtins are discovered is pinned by
#: ``tests/unit/skills/test_builtin_catalog.py``.
_BUILTIN_NAMES = frozenset(
    child.name
    for child in PACKAGED_SKILL_ROOT.iterdir()
    if child.is_dir() and (child / "SKILL.md").is_file()
)


def _fixture_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["name"] not in _BUILTIN_NAMES]


def _write_skill(root: Path, name: str, description: str, *, body: str = "") -> Path:
    """One ``<root>/<name>/SKILL.md``, the on-disk shape discovery reads."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name} body{body}\n",
        encoding="utf-8",
    )
    return skill_md


async def test_the_cwd_arm_answers_with_no_session(env) -> None:
    """The whole point: a folder, and nothing else, is enough."""
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "deploy", "Project deploy helper.")

    response = await env.client.get("/v1/desktop/skills", params={"cwd": str(project)})

    assert response.status_code == 200, response.text
    data = _data(response)
    rows = _rows(data)
    assert _fixture_rows(rows) == [{"name": "deploy", "description": "Project deploy helper."}]
    assert {row["name"] for row in rows} >= _BUILTIN_NAMES
    assert data["scope"] == "discoverable"
    assert data["detail"] is None
    assert data["warning_count"] == 0
    assert re.fullmatch(r"[0-9a-f]{16}", data["version"]), data["version"]


async def test_the_cwd_arm_answers_exactly_what_the_session_arm_answers(env) -> None:
    """Draft vs attached, same folder: same rows AND the same version.

    This is the S3 pin (architect-plan-v2 §4): the composer's draft cache and a
    mounted session's read must agree, or the draft would refetch what the
    session already holds — or worse, show different rows.
    """
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "proj-deploy", "Project deploy helper.")
    _write_skill(env.home / ".local-operator" / "skills", "home-notes", "Home notes helper.")
    session_id = "abc123def456"
    env.app.state.desktop_sessions = _FakePool({session_id: _FakeRemote(str(project))})

    cwd_data = _data(await env.client.get("/v1/desktop/skills", params={"cwd": str(project)}))
    session_data = _data(
        await env.client.get("/v1/desktop/skills", params={"session_id": session_id})
    )

    assert {row["name"] for row in _fixture_rows(_rows(cwd_data))} == {
        "proj-deploy",
        "home-notes",
    }
    assert cwd_data["skills"] == session_data["skills"]
    assert cwd_data["version"] == session_data["version"]
    assert cwd_data["warning_count"] == session_data["warning_count"]


async def test_the_sessionless_arm_never_opens_a_session(env) -> None:
    """No bridge, no pool, no owner: the pool here fails on any use."""
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "deploy", "Project deploy helper.")

    response = await env.client.get("/v1/desktop/skills", params={"cwd": str(project)})

    assert response.status_code == 200, response.text
    assert [row["name"] for row in _fixture_rows(_rows(_data(response)))] == ["deploy"]


async def test_cwd_wins_when_both_parameters_are_sent(env) -> None:
    """An explicit folder beats the one implied by a conversation."""
    project = env.tmp / "proj"
    other = env.tmp / "other"
    _write_skill(project / ".local-operator" / "skills", "from-project", "Project skill.")
    _write_skill(other / ".local-operator" / "skills", "from-other", "Other skill.")
    session_id = "abc123def456"
    env.app.state.desktop_sessions = _FakePool({session_id: _FakeRemote(str(other))})

    data = _data(
        await env.client.get(
            "/v1/desktop/skills", params={"cwd": str(project), "session_id": session_id}
        )
    )

    names = [row["name"] for row in data["skills"]]
    assert "from-project" in names
    assert "from-other" not in names


async def test_the_home_arm_is_the_default_folder(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """No parameters at all: the home root — not the daemon's cwd, not a project.

    THE DAEMON-CWD TRAP IS WHAT THIS CELL DISCRIMINATES, so the request runs
    from a folder carrying its OWN project-local skill: ``default_skill_roots``
    given ``None`` walks up from the serving process's cwd (skills/api.py:110),
    which is this folder — an implementation that let ``None`` through would
    discover ``projecty`` here and fail, which is the trap the contract calls
    out ("never called with ``None``"). The correct arm passes an explicit
    ``Path.home()`` and keeps the project skill out.
    """
    _write_skill(env.home / ".local-operator" / "skills", "homey", "Home skill.")
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "projecty", "Project skill.")
    # The walk-up a ``None`` form would take, pinned so the discrimination
    # cannot be disarmed silently by a later edit that drops the chdir.
    monkeypatch.chdir(project)
    assert Path.cwd() == project.resolve()

    data = _data(await env.client.get("/v1/desktop/skills"))

    names = [row["name"] for row in data["skills"]]
    assert "homey" in names
    assert "projecty" not in names


async def test_a_literal_tilde_resolves_to_home(env) -> None:
    """``~`` is the value the desktop stores for the default folder; it expands."""
    _write_skill(env.home / ".local-operator" / "skills", "homey", "Home skill.")

    default_data = _data(await env.client.get("/v1/desktop/skills"))
    tilde_data = _data(await env.client.get("/v1/desktop/skills", params={"cwd": "~"}))

    assert tilde_data["skills"] == default_data["skills"]
    assert tilde_data["version"] == default_data["version"]


async def test_a_name_reads_the_body_through_the_closed_resolver(env) -> None:
    """``name=`` resolves a ``skill://`` URL, sessionless, and 404s unknown names."""
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "deploy", "Project deploy helper.")

    data = _data(
        await env.client.get("/v1/desktop/skills", params={"cwd": str(project), "name": "deploy"})
    )

    assert data["detail"] is not None
    assert "# deploy body" in data["detail"]

    missing = await env.client.get(
        "/v1/desktop/skills", params={"cwd": str(project), "name": "nowhere"}
    )
    assert missing.status_code == 404


async def test_a_cwd_that_is_not_an_absolute_existing_directory_is_422(env) -> None:
    """The typed refusal the client keys on, both shapes of bad folder."""
    for bad in ("relative/path", str(env.tmp / "missing")):
        response = await env.client.get("/v1/desktop/skills", params={"cwd": bad})

        assert response.status_code == 422, bad
        assert response.json()["detail"]["code"] == "invalid_cwd", bad


async def test_version_flips_on_add_touch_and_remove(env) -> None:
    """``version`` is the runtime's own change detector, so every rescan trigger moves it.

    It must ALSO stay stable across an unchanged re-read: that is the property
    a client's cache is built on ("same version, reuse the rows").
    """
    project = env.tmp / "proj"
    skills = project / ".local-operator" / "skills"
    skill_md = _write_skill(skills, "alpha", "Alpha skill.")
    params = {"cwd": str(project)}

    first = _data(await env.client.get("/v1/desktop/skills", params=params))["version"]
    again = _data(await env.client.get("/v1/desktop/skills", params=params))["version"]
    assert again == first, "an unchanged tree must keep its version"

    _write_skill(skills, "beta", "Beta skill.")
    added = _data(await env.client.get("/v1/desktop/skills", params=params))["version"]
    assert added != first, "an added skill must move the version"

    # The mtime-only case, specifically: the fingerprint compares per-file
    # (mtime_ns, size), and a rewrite that preserves size is exactly what a
    # root-mtime rule would miss (discovery.py's docstring records it).
    stat = skill_md.stat()
    os.utime(skill_md, ns=(stat.st_atime_ns + 1_000_000, stat.st_mtime_ns + 1_000_000))
    touched = _data(await env.client.get("/v1/desktop/skills", params=params))["version"]
    assert touched != added, "an in-place touch must move the version"

    shutil.rmtree(skills / "beta")
    removed = _data(await env.client.get("/v1/desktop/skills", params=params))["version"]
    assert removed != touched, "a removed skill must move the version"


async def test_a_project_skill_shadows_a_home_skill_of_the_same_name(env) -> None:
    """Earliest root wins, and the loser is COUNTED — the collision rule, verbatim."""
    _write_skill(env.home / ".local-operator" / "skills", "collide", "From home.", body=" home")
    _write_skill(
        env.tmp / "proj" / ".local-operator" / "skills", "collide", "From project.", body=" project"
    )
    params = {"cwd": str(env.tmp / "proj")}

    data = _data(await env.client.get("/v1/desktop/skills", params=params))

    assert [row for row in data["skills"] if row["name"] == "collide"] == [
        {"name": "collide", "description": "From project."}
    ]
    assert data["warning_count"] == 1

    detail = _data(await env.client.get("/v1/desktop/skills", params={**params, "name": "collide"}))
    assert "# collide body project" in detail["detail"]

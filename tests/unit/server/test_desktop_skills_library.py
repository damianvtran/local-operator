"""``/v1/desktop/skills`` rows and details for the Skills tab (``skill_library:1``).

Every cell runs against a synthetic HOME with real skill folders and real link
baselines written through the production provenance writer -- nothing in the
classification is mocked, because the whole feature is a statement about which
real path a skill lives at. The packaged builtin catalogue is always discovered
(weakest root), which is what gives the ``builtin`` label a real subject.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.server import features as svc_features
from local_operator.server.routes import desktop_catalogues
from local_operator.server.utils import desktop_skills
from local_operator.skills.api import PACKAGED_SKILL_ROOT

TOKEN = "desktop-skills-library-test-token"
HUB = "https://hub.example.test/api"

pytestmark = pytest.mark.asyncio


def _builtin_names() -> list[str]:
    return sorted(
        child.name
        for child in PACKAGED_SKILL_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )


def _write_skill(root: Path, name: str, description: str = "A skill.", *, ref: str = "") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name} body\n", encoding="utf-8"
    )
    if ref:
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "references" / "notes.md").write_text(ref, encoding="utf-8")
    return skill_dir


def _link(config_dir: Path, alias: str, *, visibility: str = "org", absent: bool = False) -> str:
    """A real baseline for ``alias``; returns the link_id the route should echo."""
    tenant = "tenant-1" if visibility == "org" else None
    local_id = f"id-{alias}"
    identity = LinkIdentity(
        "skill", local_id, HUB, visibility, tenant, f"sk-{alias}"  # type: ignore[arg-type]
    )
    fields = {
        "document_version": 1,
        "name": alias,
        "description": f"hub {alias}",
        "files": [
            {
                "path": "SKILL.md",
                "content": f"---\nname: {alias}\ndescription: hub {alias}\n---\nbody\n",
            }
        ],
    }
    record = prov.replace(
        prov.make_record("skill", local_id, f"sk-{alias}", tenant, fields, "pull"),
        identity=identity,
        local_alias=alias,
        content_state="absent" if absent else "present",
    )
    prov.write_baseline(config_dir, record)
    return identity.link_id


@pytest_asyncio.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_dir = home / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", raising=False)
    app = FastAPI()
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = SimpleNamespace(config_dir=config_dir)

    async def client_for(cfg: Path):
        app.state.config_manager = SimpleNamespace(config_dir=cfg)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            home=home,
            config_dir=config_dir,
            tmp=tmp_path,
            use_config=client_for,
        )


async def _list(env, project: Path, **params: str) -> dict[str, Any]:
    project.mkdir(parents=True, exist_ok=True)
    response = await env.client.get("/v1/desktop/skills", params={"cwd": str(project), **params})
    assert response.status_code == 200, response.text
    return response.json()["result"]["data"]


def _by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in data["skills"]}


async def test_the_five_source_labels_on_a_constructed_set_of_roots(env) -> None:
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "proj-skill")
    _write_skill(env.config_dir / "skills", "mine")
    _write_skill(env.config_dir / "skills", "pulled")
    link_id = _link(env.config_dir, "pulled")
    _write_skill(env.home / ".claude" / "skills", "claude-only")

    rows = _by_name(await _list(env, project))

    assert rows["proj-skill"]["source"] == "project"
    assert rows["mine"]["source"] == "local" and rows["mine"]["link_id"] is None
    assert rows["pulled"]["source"] == "hub" and rows["pulled"]["link_id"] == link_id
    assert rows["claude-only"]["source"] == "external"
    builtin = _builtin_names()[0]
    assert rows[builtin]["source"] == "builtin"
    assert rows[builtin]["editable"] is False
    assert rows["mine"]["editable"] is True and rows["pulled"]["editable"] is True
    assert rows["proj-skill"]["editable"] is False
    assert rows["claude-only"]["editable"] is False
    # Only the hub row names a link; hub STATE is never copied into the list.
    for row in rows.values():
        assert not {"visibility", "tenant_id", "subscribe", "update_available"} & set(row)


async def test_a_strict_superset_for_an_old_caller(env) -> None:
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "mine", "Mine.")

    data = await _list(env, project)

    row = _by_name(data)["mine"]
    assert (row["name"], row["description"]) == ("mine", "Mine.")
    assert {"skills", "scope", "detail", "warning_count", "version"} <= set(data)
    assert data["detail"] is None and data["skill_detail"] is None
    assert set(row) == {
        "name",
        "description",
        "source",
        "link_id",
        "editable",
        "installed",
        "fingerprint",
        "form",
    }
    assert row["form"] == "document" and len(row["fingerprint"]) == 64


async def test_symlinked_harness_roots_yield_one_row_labelled_by_the_real_home(env) -> None:
    """~/.claude/skills/<name> -> ~/.local-operator/skills/<name> is not a second skill."""
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "shared")
    _write_skill(env.config_dir / "skills", "linked")
    link_id = _link(env.config_dir, "linked")
    for harness in (".claude", ".codex", ".agents"):
        root = env.home / harness / "skills"
        root.mkdir(parents=True)
        for name in ("shared", "linked"):
            os.symlink(os.path.relpath(env.config_dir / "skills" / name, root), root / name)

    data = await _list(env, project)

    names = [row["name"] for row in data["skills"]]
    assert names.count("shared") == 1 and names.count("linked") == 1
    rows = _by_name(data)
    assert rows["shared"]["source"] == "local"
    assert rows["linked"]["source"] == "hub" and rows["linked"]["link_id"] == link_id


async def test_a_symlinked_managed_root_is_still_the_managed_root(env) -> None:
    """Realpath comparison: the store itself may be reached through a symlink."""
    project = env.tmp / "proj"
    real_store = env.tmp / "elsewhere" / "skills"
    _write_skill(real_store, "moved")
    env.config_dir.mkdir(parents=True)
    os.symlink(real_store, env.config_dir / "skills")

    assert _by_name(await _list(env, project))["moved"]["source"] == "local"


async def test_a_moved_config_dir_still_labels_hub_and_local(env, monkeypatch) -> None:
    """LOCAL_OPERATOR_CONFIG_DIR can put the store anywhere; nothing is hard-coded to ~.

    ``default_skill_roots`` is HOME-derived, so a moved store only reaches
    discovery when a root points at it (here the documented
    ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` override). The classifier must still
    call it the managed root -- by comparing against ``config_dir/skills``, not
    ``~/.local-operator/skills`` -- and must not give that label to the home
    store, which is no longer the managed one.
    """
    project = env.tmp / "proj"
    moved = env.tmp / "elsewhere" / "cfg"
    _write_skill(moved / "skills", "mine")
    _write_skill(moved / "skills", "pulled")
    link_id = _link(moved, "pulled")
    _write_skill(env.config_dir / "skills", "decoy")
    monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(moved / "skills"))
    await env.use_config(moved)

    rows = _by_name(await _list(env, project))

    assert rows["mine"]["source"] == "local"
    assert rows["pulled"]["source"] == "hub" and rows["pulled"]["link_id"] == link_id
    assert rows["decoy"]["source"] not in ("local", "hub")


async def test_a_project_copy_shadowing_a_hub_alias_is_labelled_project(env) -> None:
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "pulled", "project copy")
    _write_skill(env.config_dir / "skills", "pulled", "managed copy")
    _link(env.config_dir, "pulled")

    row = _by_name(await _list(env, project))["pulled"]

    assert row["source"] == "project" and row["link_id"] is None


async def test_a_subscribed_not_installed_link_is_an_explicit_row(env) -> None:
    project = env.tmp / "proj"
    link_id = _link(env.config_dir, "ghost", absent=True)

    data = await _list(env, project)

    row = _by_name(data)["ghost"]
    assert row["source"] == "hub" and row["installed"] is False
    assert row["link_id"] == link_id and row["editable"] is False
    assert row["fingerprint"] is None and row["description"] == "hub ghost"
    # Its details exist too (no body, no files) rather than a 404.
    detail = (await _list(env, project, name="ghost"))["skill_detail"]
    assert detail["installed"] is False and detail["body"] is None and detail["files"] == []


async def test_details_carry_body_bounded_files_fingerprint_form_and_budget(env) -> None:
    project = env.tmp / "proj"
    big = "x" * (desktop_skills.PREVIEW_BYTES + 500)
    _write_skill(env.config_dir / "skills", "pulled", ref=big)
    link_id = _link(env.config_dir, "pulled", visibility="public")

    data = await _list(env, project, name="pulled")

    detail = data["skill_detail"]
    assert detail["link_id"] == link_id and detail["source"] == "hub"
    # The body rides once, in the resolver's ``detail``; this field references it.
    assert detail["body"] is None and detail["body_in_detail"] is True
    assert detail["form"] == "document" and len(detail["fingerprint"]) == 64
    assert detail["publish_budget"] == {"limit": 20 * 1024, "total": 20 * 1024}
    [ref] = detail["files"]
    assert ref["path"] == "references/notes.md" and ref["size"] == len(big)
    assert len(ref["preview"]) == desktop_skills.PREVIEW_BYTES and ref["preview_truncated"] is True
    # The historical resolver detail is untouched beside the new one.
    assert "# pulled body" in data["detail"]


async def test_org_links_get_the_org_budget_and_unlinked_skills_none(env) -> None:
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "pulled")
    _write_skill(env.config_dir / "skills", "mine")
    _link(env.config_dir, "pulled", visibility="org")

    org = (await _list(env, project, name="pulled"))["skill_detail"]
    local = (await _list(env, project, name="mine"))["skill_detail"]

    assert org["publish_budget"] == {"limit": 256 * 1024, "total": 256 * 1024}
    assert local["publish_budget"] is None and local["files"] == []


async def test_unknown_name_is_still_a_404(env) -> None:
    response = await env.client.get(
        "/v1/desktop/skills", params={"cwd": str(env.tmp), "name": "no-such-skill"}
    )

    assert response.status_code == 404
    assert "wire.errors.desktop.skill_not_found" in response.text


def test_the_capability_is_advertised() -> None:
    assert svc_features.feature_flags().get("skill_library") == 1


async def test_rows_are_a_strict_superset_of_the_historical_shape(env) -> None:
    """An older UI reading ``{name, description}`` keeps working on every row."""
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "mine", "Mine.")

    data = await _list(env, project)

    for row in data["skills"]:
        assert {"name", "description"} <= set(row)
        assert isinstance(row["name"], str) and isinstance(row["description"], str)
    assert {"scope", "detail", "warning_count", "version"} <= set(data)
    # The plain list carries no detail payload.
    assert data["detail"] is None and data["skill_detail"] is None


async def test_moved_config_dir_reached_only_through_harness_symlinks_keeps_its_labels(
    env,
) -> None:
    """QA Q1, verbatim: the migration layout with ``LOCAL_OPERATOR_CONFIG_DIR`` moved.

    ``default_skill_roots`` never scans a moved ``<cfg>/skills``, so its skills
    are discovered ONLY through ``~/.claude/skills`` links. The label must come
    from the folder's real path (managed store), not from the harness root that
    reached it -- and the link/fingerprint/form must follow the symlink.
    """
    project = env.tmp / "proj"
    moved = env.tmp / "elsewhere" / "cfg"
    _write_skill(moved / "skills", "mv-local")
    _write_skill(moved / "skills", "mv-hub", ref="ref text")
    link_id = _link(moved, "mv-hub")
    # A skill under the DEFAULT store while the profile has moved: another
    # profile's, read-only here -- not a project folder's own skill.
    _write_skill(env.config_dir / "skills", "other-profile")
    claude = env.home / ".claude" / "skills"
    claude.mkdir(parents=True)
    for name in ("mv-local", "mv-hub"):
        os.symlink(os.path.relpath(moved / "skills" / name, claude), claude / name)
    await env.use_config(moved)

    data = await _list(env, project)
    rows = _by_name(data)

    assert rows["mv-hub"]["source"] == "hub" and rows["mv-hub"]["link_id"] == link_id
    assert rows["mv-hub"]["form"] == "document" and len(rows["mv-hub"]["fingerprint"]) == 64
    assert rows["mv-hub"]["editable"] is True
    assert rows["mv-local"]["source"] == "local" and rows["mv-local"]["editable"] is True
    assert rows["other-profile"]["source"] == "external"
    assert rows["other-profile"]["editable"] is False
    # The details read follows the same link: budget and file list intact.
    detail = (await _list(env, project, name="mv-hub"))["skill_detail"]
    assert detail["publish_budget"] == {"limit": 256 * 1024, "total": 256 * 1024}
    assert [f["path"] for f in detail["files"]] == ["references/notes.md"]


async def test_a_symlink_inside_a_skill_is_still_refused(env) -> None:
    """Following the skill DIRECTORY link must not relax the in-skill symlink refusal."""
    project = env.tmp / "proj"
    skill = _write_skill(env.config_dir / "skills", "sneaky")
    secret = env.tmp / "secret.md"
    secret.write_text("TOPSECRET", encoding="utf-8")
    os.symlink(secret, skill / "leak.md")

    data = await _list(env, project, name="sneaky")

    assert _by_name(data)["sneaky"]["fingerprint"] is None
    assert "TOPSECRET" not in str(data)


async def test_a_project_copy_over_a_hub_alias_gets_no_publish_budget(env) -> None:
    """Review M1: the alias link must not lend a destination to a shadowing copy."""
    project = env.tmp / "proj"
    _write_skill(project / ".local-operator" / "skills", "pulled", "project copy")
    _write_skill(env.config_dir / "skills", "pulled", "managed copy")
    _link(env.config_dir, "pulled", visibility="org")

    detail = (await _list(env, project, name="pulled"))["skill_detail"]

    assert detail["source"] == "project" and detail["link_id"] is None
    assert detail["publish_budget"] is None


async def test_a_huge_body_is_flagged_and_not_sent_twice(env) -> None:
    """QA Q4: ``body_truncated`` is explicit; the resolver text carries the bytes once."""
    from local_operator.skills.protocol import MAX_READ_BYTES

    project = env.tmp / "proj"
    skill = _write_skill(env.config_dir / "skills", "huge")
    (skill / "SKILL.md").write_text(
        "---\nname: huge\ndescription: Big.\n---\n" + "y" * (MAX_READ_BYTES * 3), encoding="utf-8"
    )

    data = await _list(env, project, name="huge")

    detail = data["skill_detail"]
    assert detail["body_truncated"] is True and detail["body_bytes"] > MAX_READ_BYTES
    assert detail["body"] is None and detail["body_in_detail"] is True
    assert data["detail"].startswith("---\nname: huge")
    assert len(data["detail"]) >= detail["body_length"]
    # Bytes appear once: the response is about one body, not two.
    assert len(str(data)) < detail["body_length"] * 1.5


async def test_a_small_body_is_not_flagged_and_stays_consistent(env) -> None:
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "small")

    data = await _list(env, project, name="small")

    detail = data["skill_detail"]
    assert detail["body_truncated"] is False and detail["body_in_detail"] is True
    assert data["detail"][: detail["body_length"]].startswith("---\nname: small")


async def test_files_total_reports_the_whole_set_past_the_listing_cap(env) -> None:
    """QA Q3: a UI can say "64 of 80" instead of guessing."""
    from local_operator.hub_sync.skill_document import MAX_FILES

    project = env.tmp / "proj"
    skill = _write_skill(env.config_dir / "skills", "many")
    refs = skill / "references"
    refs.mkdir()
    for index in range(MAX_FILES + 6):
        (refs / f"r{index:03d}.md").write_text("x", encoding="utf-8")

    detail = (await _list(env, project, name="many"))["skill_detail"]

    assert len(detail["files"]) == MAX_FILES
    assert detail["files_total"] == MAX_FILES + 6 and detail["files_truncated"] is True


def _js_slice(text: str, units: int) -> str:
    """``text.slice(0, units)`` as JavaScript does it: by UTF-16 code unit."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return raw[: units * 2].decode("utf-16-le", errors="surrogatepass")


async def test_body_length_is_utf16_units_so_a_js_slice_reproduces_the_body(env) -> None:
    """Review F1: ``detail[:body_length]`` is evaluated in JS, which counts UTF-16 units.

    The body mixes a BMP accent, CJK and two astral emoji: code points and
    UTF-16 units differ by exactly the astral count, so a code-point ``len``
    would cut the last characters off (BMP-only text would hide that).
    """
    project = env.tmp / "proj"
    skill = _write_skill(env.config_dir / "skills", "unicode")
    body = "---\nname: unicode\ndescription: Unicode.\n---\n# é 漢字 🚀 ok 🎉\n"
    (skill / "SKILL.md").write_text(body, encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "n.md").write_text("ref", encoding="utf-8")

    data = await _list(env, project, name="unicode")

    detail = data["skill_detail"]
    units = len(body.encode("utf-16-le")) // 2
    assert units == len(body) + 2  # two astral characters, one extra unit each
    assert detail["body_length"] == units
    assert detail["body_in_detail"] is True and detail["body"] is None
    # The resolver text carries a reference listing after the body; the JS-style
    # slice must stop exactly at the body's end, not before or after.
    assert len(data["detail"]) > len(body)
    assert _js_slice(data["detail"], detail["body_length"]) == body


async def test_a_concurrent_edit_inlines_the_body_and_does_not_claim_detail(
    env, monkeypatch
) -> None:
    """Review F2: when ``detail`` does not start with the body, the body is sent inline."""
    project = env.tmp / "proj"
    _write_skill(env.config_dir / "skills", "racy")
    from local_operator.skills import api as skills_api

    # The route imports the resolver inside ``_read_skills``, so the module
    # attribute it resolves at call time is the one to replace.
    real = skills_api.resolve_skill_url

    def stale(url, skills, **kwargs):
        return "SOMETHING ELSE ENTIRELY\n" + str(real(url, skills, **kwargs))

    monkeypatch.setattr(skills_api, "resolve_skill_url", stale)

    data = await _list(env, project, name="racy")

    detail = data["skill_detail"]
    assert detail["body_in_detail"] is False
    assert detail["body"].startswith("---\nname: racy")
    assert detail["body_length"] == len(detail["body"].encode("utf-16-le")) // 2


async def test_body_length_excludes_the_truncation_marker(env) -> None:
    """Review F3: the length is the file's text, not the resolver's cut-off notice."""
    from local_operator.skills.protocol import _TRUNCATION_MARKER, MAX_READ_BYTES

    project = env.tmp / "proj"
    skill = _write_skill(env.config_dir / "skills", "cut")
    head = "---\nname: cut\ndescription: Cut.\n---\n"
    (skill / "SKILL.md").write_text(head + "z" * (MAX_READ_BYTES * 2), encoding="utf-8")

    data = await _list(env, project, name="cut")

    detail = data["skill_detail"]
    assert detail["body_truncated"] is True
    assert data["detail"].startswith(head) and _TRUNCATION_MARKER in data["detail"]
    assert detail["body_length"] == MAX_READ_BYTES  # ASCII: bytes == units, marker excluded
    sliced = _js_slice(data["detail"], detail["body_length"])
    assert sliced == head + "z" * (MAX_READ_BYTES - len(head))
    assert _TRUNCATION_MARKER not in sliced

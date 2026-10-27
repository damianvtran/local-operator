"""``POST|PUT|DELETE /v1/desktop/skills``: the Skills tab's write routes (``skill_library:1``).

Real folders, real link baselines (written through the production ``apply``), the
real receipt journal and the real link check -- the acceptance is a statement about
the bytes on disk and what ``check_hub_skills`` then says about them, so nothing
in the write path is mocked. The one injected seam is the hub FETCH of the link
check, because that is the network.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.hub_sync.provenance import make_record
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage
from local_operator.server.routes import desktop_catalogues
from local_operator.skills.api import PACKAGED_SKILL_ROOT

TOKEN = "skill-writes-test-token"
HUB = "https://hub.example.test/api"

pytestmark = pytest.mark.asyncio


def _rid() -> str:
    return str(uuid.uuid4())


def _skill_md(name: str, description: str = "A skill.", body: str = "body\n") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}"


def _files(name: str, body: str = "body\n", *extra: dict[str, str]) -> list[dict[str, str]]:
    return [{"path": "SKILL.md", "content": _skill_md(name, body=body)}, *extra]


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
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(client=client, home=home, config_dir=config_dir, tmp=tmp_path)


def _link(
    config_dir: Path, name: str = "linked-skill"
) -> tuple[ManagedSkillStorage, SkillDocument]:
    """A pulled org skill: folder and baseline written by the production ``apply``."""
    document = SkillDocument.from_json(
        {
            "document_version": 1,
            "name": name,
            "description": f"hub {name}",
            "files": [
                {"path": "SKILL.md", "content": _skill_md(name, f"hub {name}", "hub body\n")},
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        }
    )
    identity = LinkIdentity("skill", f"id-{name}", HUB, "org", "tenant-1", f"sk-{name}")
    baseline = replace(
        make_record(
            "skill", identity.local_id, identity.hub_id, "tenant-1", document.to_json(), "pull"
        ),
        identity=identity,
        remote_revision='"r1"',
        lineage="clean",
        local_alias=name,
    )
    storage = ManagedSkillStorage(config_dir)
    storage.apply(identity, name, document, baseline, expected_fingerprint=None)
    return storage, document


def _fp(config_dir: Path, name: str) -> str:
    return SkillDocument.from_directory(config_dir / "skills" / name).fingerprint


def _detail(refusal: Any) -> dict[str, Any]:
    return refusal.json()["detail"]


# -- create ------------------------------------------------------------------------------


async def test_create_writes_only_into_the_managed_root(env) -> None:
    response = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "new-skill", "files": _files("new-skill")},
    )
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    assert data["name"] == "new-skill" and data["source"] == "local" and data["changed"] is True
    assert data["link_id"] is None and data["baseline_fingerprint"] is None
    assert data["fingerprint"] == _fp(env.config_dir, "new-skill")
    assert (env.config_dir / "skills" / "new-skill" / "SKILL.md").is_file()


async def test_create_over_an_existing_name_is_alias_occupied(env) -> None:
    body = {"name": "dup-skill", "files": _files("dup-skill")}
    first = await env.client.post("/v1/desktop/skills", json={"request_id": _rid(), **body})
    assert first.status_code == 200
    before = _fp(env.config_dir, "dup-skill")
    second = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "dup-skill", "files": _files("dup-skill", "other\n")},
    )
    assert second.status_code == 409
    assert _detail(second)["code"] == "skill_alias_occupied"
    assert _fp(env.config_dir, "dup-skill") == before


async def test_create_over_a_hub_link_that_owns_the_name_is_alias_occupied(env) -> None:
    storage, _document = _link(env.config_dir)
    # The link's folder is gone (unpulled), but the baseline still owns the alias.
    storage.unpull("id-linked-skill")
    response = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "linked-skill", "files": _files("linked-skill")},
    )
    assert response.status_code == 409
    assert _detail(response)["code"] == "skill_alias_occupied"
    assert not (env.config_dir / "skills" / "linked-skill").exists()


async def test_copy_from_a_builtin_creates_an_editable_copy(env) -> None:
    builtin = sorted(
        child.name
        for child in PACKAGED_SKILL_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )[0]
    response = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": builtin, "copy_from": builtin},
    )
    # Same-name copy: the managed copy shadows the builtin (the note's "Copy to my skills").
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        assert (env.config_dir / "skills" / builtin / "SKILL.md").is_file()
        assert response.json()["result"]["source"] == "local"


async def test_create_needs_exactly_one_of_files_or_copy_from(env) -> None:
    neither = await env.client.post(
        "/v1/desktop/skills", json={"request_id": _rid(), "name": "x-skill"}
    )
    both = await env.client.post(
        "/v1/desktop/skills",
        json={
            "request_id": _rid(),
            "name": "x-skill",
            "files": _files("x-skill"),
            "copy_from": "y",
        },
    )
    assert neither.status_code == both.status_code == 422


# -- hostile input ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "references/../x",
        "/etc/passwd",
        "../escape",
        ".hidden/x",
        "references/.secret",
        "references\\x",
        "references/a:b",
        "notes.md",
        "other/x.md",
        "references/\u202ex.md",
    ],
)
async def test_hostile_paths_are_422_and_nothing_is_written(env, path: str) -> None:
    files = _files("path-skill", "b\n", {"path": path, "content": "x\n"})
    response = await env.client.post(
        "/v1/desktop/skills", json={"request_id": _rid(), "name": "path-skill", "files": files}
    )
    assert response.status_code == 422, (path, response.text)
    assert not (env.config_dir / "skills" / "path-skill").exists()
    assert not (env.config_dir / "escape").exists()


@pytest.mark.parametrize("name", [".archive", ".hidden", "Upper", "a/b", "..", "-x", "x" * 65])
async def test_hostile_names_are_422(env, name: str) -> None:
    response = await env.client.post(
        "/v1/desktop/skills", json={"request_id": _rid(), "name": name, "files": _files("ok-skill")}
    )
    assert response.status_code == 422, (name, response.text)


async def test_oversized_files_are_refused_locally_at_the_local_limit(env) -> None:
    big = "x" * (200 * 1024 + 1)
    response = await env.client.post(
        "/v1/desktop/skills",
        json={
            "request_id": _rid(),
            "name": "big-skill",
            "files": _files("big-skill", "b\n", {"path": "references/big.md", "content": big}),
        },
    )
    assert response.status_code == 422
    assert not (env.config_dir / "skills" / "big-skill").exists()


async def test_a_linked_save_over_the_hub_cap_is_not_blocked(env) -> None:
    """The org per-file cap is 256 KiB; the LOCAL cap (200 KiB) is what gates a save."""
    storage, _document = _link(env.config_dir)
    current = _fp(env.config_dir, "linked-skill")
    # Over the PUBLIC hub cap (20 KiB) but under the local cap: saved, with the budget returned.
    mid = "y" * (30 * 1024)
    response = await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": _files("linked-skill", "b\n", {"path": "references/mid.md", "content": mid}),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["publish_budget"] == {"limit": 256 * 1024, "total": 256 * 1024}


async def test_a_symlinked_file_in_the_current_skill_is_422_and_unchanged(env) -> None:
    await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "sym-skill", "files": _files("sym-skill")},
    )
    folder = env.config_dir / "skills" / "sym-skill"
    current = _fp(env.config_dir, "sym-skill")
    (folder / "references").mkdir()
    target = env.tmp / "outside.txt"
    target.write_text("secret\n")
    os.symlink(target, folder / "references" / "link.md")
    response = await env.client.put(
        "/v1/desktop/skills/sym-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": _files("sym-skill", "new\n"),
        },
    )
    assert response.status_code == 422, response.text
    assert (folder / "SKILL.md").read_text() == _skill_md("sym-skill")


async def test_a_symlinked_skill_directory_is_refused_not_followed(env) -> None:
    outside = env.tmp / "elsewhere" / "linked-dir"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text(_skill_md("linked-dir"))
    (env.config_dir / "skills").mkdir(parents=True)
    os.symlink(outside, env.config_dir / "skills" / "linked-dir")
    before = (outside / "SKILL.md").read_text()
    response = await env.client.put(
        "/v1/desktop/skills/linked-dir",
        json={
            "request_id": _rid(),
            "expected_fingerprint": "0" * 64,
            "files": _files("linked-dir", "evil\n"),
        },
    )
    assert response.status_code == 422, response.text
    assert (outside / "SKILL.md").read_text() == before


async def test_a_symlinked_skills_root_is_refused_not_followed(env) -> None:
    real = env.tmp / "real-root"
    real.mkdir()
    env.config_dir.mkdir(parents=True)
    os.symlink(real, env.config_dir / "skills")
    response = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "root-skill", "files": _files("root-skill")},
    )
    assert response.status_code == 422, response.text
    assert list(real.iterdir()) == []


# -- update, CAS and the baseline invariant -----------------------------------------------


async def test_stale_expected_fingerprint_is_409_and_bytes_are_unchanged(env) -> None:
    await env.client.post(
        "/v1/desktop/skills", json={
            "request_id": _rid(),
            "name": "cas-skill",
            "files": _files("cas-skill"),
        }
    )
    folder = env.config_dir / "skills" / "cas-skill"
    before = (folder / "SKILL.md").read_bytes()
    response = await env.client.put(
        "/v1/desktop/skills/cas-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": "a" * 64,
            "files": _files("cas-skill", "new\n"),
        },
    )
    assert response.status_code == 409
    assert _detail(response)["code"] == "skill_stale"
    assert (folder / "SKILL.md").read_bytes() == before


async def test_editing_a_linked_skill_keeps_the_baseline_and_the_check_says_local_only(env) -> None:
    storage, document = _link(env.config_dir)
    baseline_file = prov.baseline_path(env.config_dir, "skill", "id-linked-skill")
    baseline_bytes = baseline_file.read_bytes()
    current = _fp(env.config_dir, "linked-skill")

    response = await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": _skill_md("linked-skill", "hub linked-skill", "EDITED\n"),
                },
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    assert data["source"] == "hub" and data["link_id"] is not None
    assert data["baseline_fingerprint"] == document.fingerprint  # unchanged
    assert data["fingerprint"] != document.fingerprint
    assert data["publish_budget"] == {"limit": 256 * 1024, "total": 256 * 1024}
    assert baseline_file.read_bytes() == baseline_bytes

    class Remote:
        api_key = "k"

    def fetch(_client: Any, _skill_id: str, **_kw: Any) -> dict[str, Any]:
        # The hub still serves the baseline bytes: only the LOCAL side moved.
        return {**document.to_json(), "revision": '"r1"'}

    checks = chk.check_hub_skills(
        config_dir=env.config_dir, client_for_tenant=lambda _t: Remote(), fetch=fetch
    )
    assert [c.classification.state for c in checks if c.classification] == ["local-only"]


async def test_update_keeps_an_executable_bit_on_scripts(env) -> None:
    storage, _document = _link(env.config_dir)
    script = env.config_dir / "skills" / "linked-skill" / "scripts" / "doctor.sh"
    os.chmod(script, 0o755)
    current = _fp(env.config_dir, "linked-skill")
    response = await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": _skill_md("linked-skill", "hub linked-skill", "x\n"),
                },
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho changed\n"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert script.read_text() == "#!/bin/sh\necho changed\n"
    assert os.stat(script).st_mode & 0o777 == 0o755


# -- read-only roots --------------------------------------------------------------------


async def test_editing_a_builtin_is_skill_not_editable(env) -> None:
    builtin = sorted(
        child.name
        for child in PACKAGED_SKILL_ROOT.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )[0]
    response = await env.client.put(
        "/v1/desktop/skills/" + builtin,
        json={"request_id": _rid(), "expected_fingerprint": "a" * 64, "files": _files(builtin)},
    )
    assert response.status_code == 409
    assert _detail(response)["code"] == "skill_not_editable"
    assert not (env.config_dir / "skills" / builtin).exists()


async def test_editing_an_ecosystem_skill_is_skill_not_editable(env) -> None:
    folder = env.home / ".claude" / "skills" / "eco-skill"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(_skill_md("eco-skill"))
    response = await env.client.put(
        "/v1/desktop/skills/eco-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": "a" * 64,
            "files": _files("eco-skill", "x\n"),
        },
    )
    assert response.status_code == 409
    assert _detail(response)["code"] == "skill_not_editable"
    assert (folder / "SKILL.md").read_text() == _skill_md("eco-skill")
    removal = await env.client.request(
        "DELETE", "/v1/desktop/skills/eco-skill", json={"request_id": _rid()}
    )
    assert removal.status_code == 409
    assert _detail(removal)["code"] == "skill_not_editable"
    assert folder.is_dir()


async def test_an_unknown_skill_is_404(env) -> None:
    response = await env.client.put(
        "/v1/desktop/skills/nothing-here",
        json={
            "request_id": _rid(),
            "expected_fingerprint": "a" * 64,
            "files": _files("nothing-here"),
        },
    )
    assert response.status_code == 404
    assert _detail(response)["code"] == "skill_not_found"


# -- remove -----------------------------------------------------------------------------


async def test_removing_a_local_skill_moves_it_to_a_kept_backup(env) -> None:
    await env.client.post(
        "/v1/desktop/skills", json={
            "request_id": _rid(),
            "name": "gone-skill",
            "files": _files("gone-skill"),
        }
    )
    response = await env.client.request(
        "DELETE", "/v1/desktop/skills/gone-skill", json={"request_id": _rid()}
    )
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    assert data["mode"] == "backup" and data["removed"] is True
    assert not (env.config_dir / "skills" / "gone-skill").exists()
    kept = env.config_dir / data["backup"]
    assert kept.is_dir() and (kept / "SKILL.md").read_text() == _skill_md("gone-skill")


async def test_removing_a_modified_linked_skill_needs_the_flag(env) -> None:
    storage, _document = _link(env.config_dir)
    current = _fp(env.config_dir, "linked-skill")
    edit = await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": _skill_md("linked-skill", "hub linked-skill", "dirty\n"),
                },
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        },
    )
    assert edit.status_code == 200
    folder = env.config_dir / "skills" / "linked-skill"

    refused = await env.client.request(
        "DELETE", "/v1/desktop/skills/linked-skill", json={"request_id": _rid()}
    )
    assert refused.status_code == 409
    assert _detail(refused)["code"] == "skill_local_changes_unpublished"
    assert folder.is_dir() and "dirty" in (folder / "SKILL.md").read_text()

    allowed = await env.client.request(
        "DELETE",
        "/v1/desktop/skills/linked-skill",
        json={"request_id": _rid(), "discard_local_changes": True},
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["result"]["mode"] == "unpull"
    assert not folder.exists()
    # Unpull semantics: the link (baseline + alias reservation) is kept.
    record = prov.read_baseline(env.config_dir, "skill", "id-linked-skill")
    assert record is not None and record.content_state == "absent"


async def test_removing_a_clean_linked_skill_unpulls_without_the_flag(env) -> None:
    _link(env.config_dir)
    response = await env.client.request(
        "DELETE", "/v1/desktop/skills/linked-skill", json={"request_id": _rid()}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["mode"] == "unpull"
    assert not (env.config_dir / "skills" / "linked-skill").exists()


# -- receipts ---------------------------------------------------------------------------


async def test_a_replayed_request_id_is_not_applied_twice(env) -> None:
    request_id = _rid()
    body = {"request_id": request_id, "name": "once-skill", "files": _files("once-skill")}
    first = await env.client.post("/v1/desktop/skills", json=body)
    assert first.status_code == 200 and first.json()["result"].get("replayed") is not True
    # Delete the folder behind the receipt's back: a REPLAY must answer from the
    # receipt and must not re-create it (a re-run would also have 409'd on nothing).
    import shutil

    shutil.rmtree(env.config_dir / "skills" / "once-skill")
    second = await env.client.post("/v1/desktop/skills", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["result"]["replayed"] is True
    assert not (env.config_dir / "skills" / "once-skill").exists()

    # The same id with DIFFERENT input is a conflict, never a second write.
    other = await env.client.post(
        "/v1/desktop/skills", json={**body, "files": _files("once-skill", "z\n")}
    )
    assert other.status_code == 409


async def test_unknown_body_fields_are_rejected(env) -> None:
    response = await env.client.post(
        "/v1/desktop/skills",
        json={"request_id": _rid(), "name": "k-skill", "files": _files("k-skill"), "surprise": 1},
    )
    assert response.status_code == 422


# -- review round 1 ----------------------------------------------------------------------


def _ctx(config_dir: Path) -> Any:
    from local_operator.hub_sync import service as svc

    return svc.HubSyncContext(
        config_dir=config_dir, config_manager=None, client_for_tenant=lambda _tenant: None
    )


def _linked_edit_doc(name: str, body: str) -> SkillDocument:
    return SkillDocument.from_json(
        {
            "document_version": 1,
            "name": name,
            "description": f"hub {name}",
            "files": [
                {"path": "SKILL.md", "content": _skill_md(name, f"hub {name}", body)},
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        }
    )


def _tree_bytes(root: Path) -> set[bytes]:
    return {p.read_bytes() for p in root.rglob("*") if p.is_file()} if root.exists() else set()


def _interleaved_remove(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, *, hold_lock: bool
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Force the order: dirty check says CLEAN -> an edit commits -> the unpull runs.

    The edit is started from the check's own return path (the exact window the
    review found), in another thread. Returns ``(edit outcome, edit marker, remove result)``.
    """
    import contextlib
    import threading

    from local_operator.server.utils import desktop_skill_writes as writes

    _link(config_dir)
    record = prov.read_baseline(config_dir, "skill", "id-linked-skill")
    assert record is not None
    marker = "FRESH-EDIT-MARKER"
    current = _fp(config_dir, "linked-skill")
    outcome: dict[str, Any] = {}

    def edit() -> None:
        try:
            outcome["result"] = writes.update(
                config_dir,
                "linked-skill",
                [
                    {
                        "path": "SKILL.md",
                        "content": _skill_md("linked-skill", "hub linked-skill", marker),
                    },
                    {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
                ],
                current,
            )
        except Exception as error:  # noqa: BLE001 - the outcome IS the observation
            outcome["error"] = error

    real = writes.folder_differs_from_baseline

    def check_then_edit(storage: Any, rec: Any) -> bool:
        answer = real(storage, rec)
        thread = threading.Thread(target=edit)
        thread.start()
        # With the lock held the edit cannot finish inside this window; without
        # it, it commits before the unpull below -- the race itself.
        thread.join(timeout=2.0 if not hold_lock else 0.5)
        outcome["thread"] = thread
        return answer

    monkeypatch.setattr(writes, "folder_differs_from_baseline", check_then_edit)
    if not hold_lock:
        monkeypatch.setattr(writes, "config_write_lock", lambda *_a, **_k: contextlib.nullcontext())
    result = writes.remove_linked(_ctx(config_dir), record, discard_local_changes=False)
    outcome["thread"].join(timeout=10)
    return outcome, marker, result


async def test_an_edit_racing_a_linked_remove_is_refused_not_destroyed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome, marker, _result = _interleaved_remove(env.config_dir, monkeypatch, hold_lock=True)
    # The edit lost the race cleanly (the skill is gone, so its CAS has nothing to
    # match) -- it was told "no", it was not silently accepted and then deleted.
    assert "error" in outcome and "result" not in outcome
    code = getattr(outcome["error"], "code", "")
    assert code in ("skill-not-editable", "skill-not-found", "local-concurrent-edit")
    assert not (env.config_dir / "skills" / "linked-skill").exists()


async def test_negative_control_without_the_lock_the_edit_is_destroyed(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same interleave, lock removed: proves the test above can fail.

    The edit is ACCEPTED (200-equivalent) and then unpulled away with no copy
    anywhere -- exactly the silent loss of review M1.
    """
    outcome, marker, _result = _interleaved_remove(env.config_dir, monkeypatch, hold_lock=False)
    assert "result" in outcome, outcome
    assert not (env.config_dir / "skills" / "linked-skill").exists()
    assert not any(marker.encode() in blob for blob in _tree_bytes(env.config_dir / "hub"))


@pytest.mark.parametrize("value", ["false", "yes", "true", "1", 0, 1, None, "0", []])
async def test_discard_local_changes_must_be_a_real_boolean(env, value: Any) -> None:
    _link(env.config_dir)
    response = await env.client.request(
        "DELETE",
        "/v1/desktop/skills/linked-skill",
        json={"request_id": _rid(), "discard_local_changes": value},
    )
    assert response.status_code == 422, (value, response.text)
    assert (env.config_dir / "skills" / "linked-skill").is_dir()


async def test_a_deliberate_discard_keeps_a_recoverable_copy(env) -> None:
    _link(env.config_dir)
    current = _fp(env.config_dir, "linked-skill")
    await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": current,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": _skill_md("linked-skill", "hub linked-skill", "KEEP-ME\n"),
                },
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        },
    )
    response = await env.client.request(
        "DELETE",
        "/v1/desktop/skills/linked-skill",
        json={"request_id": _rid(), "discard_local_changes": True},
    )
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    kept = env.config_dir / data["backup"]
    assert data["mode"] == "unpull" and kept.is_dir()
    assert "KEEP-ME" in (kept / "SKILL.md").read_text()
    # A CLEAN linked remove has nothing to keep.
    _link(env.config_dir, "clean-skill")
    clean = await env.client.request(
        "DELETE", "/v1/desktop/skills/clean-skill", json={"request_id": _rid()}
    )
    assert clean.json()["result"]["backup"] is None


async def test_the_first_request_after_a_between_renames_crash_succeeds(env) -> None:
    """m2: the alias is absent but its link owns it; recover first, never 'not editable'."""
    storage, document = _link(env.config_dir)
    edited = _linked_edit_doc("linked-skill", "crashed edit\n")

    def crash(phase: str) -> None:
        if phase == "old-renamed":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        storage.author("linked-skill", edited, document.fingerprint, checkpoint=crash)
    assert not (env.config_dir / "skills" / "linked-skill").exists()  # the gap

    response = await env.client.put(
        "/v1/desktop/skills/linked-skill",
        json={
            "request_id": _rid(),
            "expected_fingerprint": document.fingerprint,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": _skill_md("linked-skill", "hub linked-skill", "after recovery\n"),
                },
                {"path": "scripts/doctor.sh", "content": "#!/bin/sh\necho ok\n"},
            ],
        },
    )
    assert response.status_code == 200, response.text


def test_skill_write_is_advertised_only_while_every_promised_route_exists() -> None:
    """The key promises all three writes; removing one must fail HERE, not in a renderer.

    Same shape as the ``hub_skills`` pin: a capability is a reviewed act, so the
    routes it names are read from the mounted router, not from a literal.
    """
    from local_operator.server import features as svc_features

    flags = svc_features.feature_flags()
    assert flags.get("skill_write") == 1
    # The read key is separate and unchanged: a renderer needs both.
    assert flags.get("skill_library") == 1

    routes = {
        (getattr(route, "path", ""), method)
        for route in desktop_catalogues.router.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/v1/desktop/skills", "POST") in routes
    assert ("/v1/desktop/skills/{name}", "PUT") in routes
    assert ("/v1/desktop/skills/{name}", "DELETE") in routes

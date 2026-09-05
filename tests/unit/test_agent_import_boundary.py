"""An imported profile must never own a pre-existing local identity or path."""

import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_operator.agents import AgentData, AgentRegistry


def _archive(tmp_path: Path, metadata: object) -> Path:
    archive = tmp_path / "profile.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("agent.yml", yaml.safe_dump(metadata))
        zf.writestr("system_prompt.md", "Keep these instructions verbatim.")
    return archive


def _metadata(identifier: object) -> dict[str, object]:
    return {
        "id": identifier,
        "name": "../Display name is not a path",
        "created_date": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
    }


@pytest.mark.parametrize(
    "identifier", ["../outside", "../../outside", "/outside", "", None, [], {}]
)
def test_archive_identity_is_never_a_destination(tmp_path, identifier):
    registry = AgentRegistry(tmp_path / "config")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    if identifier == "/outside":
        identifier = str(outside)
    imported = registry.import_agent(_archive(tmp_path, _metadata(identifier)))
    assert str(uuid.UUID(imported.id)) == imported.id
    assert imported.id != identifier
    assert imported.name == "../Display name is not a path"
    assert sentinel.read_text() == "keep"
    destination = registry.agents_dir / imported.id
    assert destination.parent == registry.agents_dir
    assert (destination / "system_prompt.md").read_text() == "Keep these instructions verbatim."
    persisted = yaml.safe_load((destination / "agent.yml").read_text())
    assert persisted["id"] == imported.id
    assert registry.get_agent(imported.id) == imported


def test_repeated_import_preserves_existing_profile_and_private_files(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    archive = _archive(tmp_path, _metadata("original"))
    first = registry.import_agent(archive)
    original_dir = registry.agents_dir / first.id
    sentinel = original_dir / "conversation.jsonl"
    sentinel.write_text("private local history")
    archive = _archive(tmp_path, _metadata(first.id))
    second = registry.import_agent(archive)
    assert second.id != first.id
    assert sentinel.read_text() == "private local history"
    assert registry.get_agent(first.id) == first
    assert registry.get_agent(second.id) == second


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling-symlink", "memory"])
def test_generated_identity_collision_fails_closed(tmp_path, kind):
    registry = AgentRegistry(tmp_path / "config")
    identity = uuid.uuid4()
    destination = registry.agents_dir / str(identity)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    if kind == "directory":
        destination.mkdir()
        (destination / "sentinel").write_text("keep")
    elif kind == "file":
        destination.write_text("keep")
    elif kind == "symlink":
        destination.symlink_to(outside, target_is_directory=True)
    elif kind == "dangling-symlink":
        destination.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        registry._agents[str(identity)] = AgentData.model_validate(_metadata(str(identity)))
    archive = _archive(tmp_path, _metadata("archive-id"))
    with patch("local_operator.agents.uuid.uuid4", return_value=identity):
        with pytest.raises(ValueError, match="Import destination already exists"):
            registry.import_agent(archive)
    assert sentinel.read_text() == "keep"
    if kind == "directory":
        assert (destination / "sentinel").read_text() == "keep"
    elif kind == "file":
        assert destination.read_text() == "keep"
    elif "symlink" in kind:
        assert destination.is_symlink()
    else:
        assert str(identity) in registry._agents


@pytest.mark.parametrize(
    "metadata", [None, [], "text", {}, {"name": []}, {"name": "x", "version": {}}]
)
def test_invalid_metadata_has_no_registry_side_effects(tmp_path, metadata):
    registry = AgentRegistry(tmp_path / "config")
    before = set(registry.agents_dir.iterdir())
    with pytest.raises(ValueError, match="Invalid agent metadata"):
        registry.import_agent(_archive(tmp_path, metadata))
    assert set(registry.agents_dir.iterdir()) == before
    assert not registry._agents


def test_copy_failure_cleans_only_the_new_import(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    archive = _archive(tmp_path, _metadata("profile"))
    before = set(registry.agents_dir.iterdir())
    with patch("local_operator.agents.shutil.copy2", side_effect=OSError("disk unavailable")):
        with pytest.raises(Exception, match="disk unavailable"):
            registry.import_agent(archive)
    assert set(registry.agents_dir.iterdir()) == before
    assert not registry._agents


def test_archive_symlink_is_rejected_before_registry_mutation(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    archive = _archive(tmp_path, _metadata("profile"))
    with zipfile.ZipFile(archive, "a") as zf:
        link = zipfile.ZipInfo("linked-prompt")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        zf.writestr(link, "../outside")
    with pytest.raises(ValueError, match="unsupported symlink entry"):
        registry.import_agent(archive)
    assert not list(registry.agents_dir.iterdir())

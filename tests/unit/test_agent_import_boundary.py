"""An imported profile must never own a pre-existing local identity or path.

"Identity" here covers the NAME as well as the id and the path: the registry is
one flat namespace, the local lookup is exact and case-sensitive, and an
imported profile that shares a name with an existing agent leaves the resolver
choosing between two rows the user cannot tell apart. The collision rule is
specified by the cross-repo contract §3.6 and lives in
:func:`AgentRegistry.resolve_import_name`.
"""

import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_operator.agents import (
    MAX_AGENT_NAME_CHARS,
    AgentData,
    AgentRegistry,
    agent_name_key,
)


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
    imported, _ = registry.import_agent(_archive(tmp_path, _metadata(identifier)))
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
    first, _ = registry.import_agent(archive)
    original_dir = registry.agents_dir / first.id
    sentinel = original_dir / "conversation.jsonl"
    sentinel.write_text("private local history")
    archive = _archive(tmp_path, _metadata(first.id))
    second, renamed_from = registry.import_agent(archive)
    assert second.id != first.id
    assert sentinel.read_text() == "private local history"
    assert registry.get_agent(first.id) == first
    assert registry.get_agent(second.id) == second
    # The second import reuses the first's name, so it lands under a suffix and
    # says so. Without the note the caller reports a successful import of an
    # agent the user cannot find under the name they asked for.
    assert second.name == f"{first.name}-2"
    assert renamed_from == first.name


def _named_archive(directory: Path, name: str) -> Path:
    """An archive declaring ``name``; one per directory, since each writes
    ``profile.zip``."""

    directory.mkdir(parents=True, exist_ok=True)
    return _archive(directory, {**_metadata("archive-id"), "name": name})


def test_import_keeps_the_published_name_when_the_registry_does_not_hold_it(tmp_path):
    registry = AgentRegistry(tmp_path / "config")
    imported, renamed_from = registry.import_agent(_named_archive(tmp_path / "a", "Coder"))
    assert imported.name == "Coder"
    assert renamed_from is None


def test_import_suffixes_a_name_already_held_in_ANY_case(tmp_path):
    """The rule the local side was missing (contract §3.6, defect D-4).

    Before this, import called ``save_agent`` directly, which performs no name
    check, and the check ``create_agent`` uses is exact — so an incoming
    ``Coder`` landed happily beside a local ``coder`` and the resolver then
    picked one of them silently. Collisions are found with the normalised key
    and reported with the PUBLISHED name, so the row is greppable under the
    name the archive used.
    """

    registry = AgentRegistry(tmp_path / "config")
    first, _ = registry.import_agent(_named_archive(tmp_path / "a", "coder"))
    second, renamed_from = registry.import_agent(_named_archive(tmp_path / "b", "Coder"))
    assert first.name == "coder"
    assert second.name == "Coder-2"
    assert renamed_from == "Coder"
    # The existing row is untouched: a collision renames the newcomer, it never
    # overwrites or mutates what the operator already had.
    assert registry.get_agent_by_name("coder") == first
    assert registry.get_agent_by_name("Coder-2") == second


def _assert_publishable(name: str) -> None:
    """The name rules a pulled row must satisfy to be publishable at all.

    Taken from the two validators the standard names, and asserted as their
    union so a future suffix cannot quietly introduce a character either side
    refuses or exceed the cap: the hub's document rule (contract §1.4 — 1..128
    characters, no ``\\ / :`` or Unicode whitespace, no leading/trailing ``-``
    or ``.``) and the local create path's own refusal (``write_profile``: no
    whitespace, no ``/`` or ``\\``).

    ``:`` is the one character only §1.4 refuses — the local validator has no
    colon rule, so a produced name carrying one would pass the create path and
    be refused at publication, which is the silent failure this helper exists
    to catch.

    Note the stricter-than-now set of rules: the operator has ruled ordinary
    INTERNAL spaces legal in a published name (normalised to single spaces with
    the ends trimmed, the unrenderable-character refusals and the cap
    unchanged), implemented in agent-server. The names these tests produce from
    their own bases contain no spaces either way, so this stays the right
    assertion for them; do not read it as the post-ruling name spec.
    """

    assert 1 <= len(name) <= MAX_AGENT_NAME_CHARS, name
    assert not any(char.isspace() or char in "/\\:" for char in name), name
    assert not name.startswith(("-", ".")) and not name.endswith(("-", ".")), name


def test_the_publishability_check_can_see_every_rule_it_asserts():
    """A guard has to be able to fail on the rules it names, so each one is fed
    a violating name here. The colon is the one that was missing while the
    helper's docstring cited §1.4, which lists it (§1.4: ``\\ / :``)."""

    _assert_publishable("Report-ready")  # the control: a legal name passes
    for illegal in ("Report: ready", "Report:ready", "Report/ready", "Report\\ready", "x" * 129):
        with pytest.raises(AssertionError):
            _assert_publishable(illegal)


def test_import_takes_the_first_free_suffix(tmp_path):
    """``-2``, ``-3`` … — including past a row that already looks suffixed.

    The suffix is appended to the incoming name VERBATIM, so an archive whose
    published name is literally ``Scout-2`` lands under ``Scout-2-2`` rather
    than the implementation trying to parse an existing suffix out of a
    user-authored name. What matters here is the fourth line: the counter has to
    notice that ``Scout-2`` is taken and skip to ``Scout-3``.
    """

    registry = AgentRegistry(tmp_path / "config")
    names = []
    for index, incoming in (
        ("a", "Scout"),
        ("b", "Scout"),
        ("c", "Scout-2"),
        ("d", "Scout"),
    ):
        imported, _ = registry.import_agent(_named_archive(tmp_path / index, incoming))
        names.append(imported.name)
    assert names == ["Scout", "Scout-2", "Scout-2-2", "Scout-3"]
    for name in names:
        _assert_publishable(name)


def test_the_suffix_is_a_hyphen_so_the_row_can_still_be_published(tmp_path):
    """The contract's ``" (N)"`` spelling was wrong for both validators.

    A space is refused by the hub's name rule (§1.4) and by ``write_profile``,
    so a suffixed row spelled with one could never be published: the user pulls
    an agent, holds a name that collides, and is handed something the hub will
    not take. The note the route returns is what the UI turns into "Imported as
    \"Coder-2\" — you already have an agent called \"Coder\"".
    """

    registry = AgentRegistry(tmp_path / "config")
    registry.import_agent(_named_archive(tmp_path / "a", "Coder"))
    imported, renamed_from = registry.import_agent(_named_archive(tmp_path / "b", "Coder"))
    assert imported.name == "Coder-2"
    assert renamed_from == "Coder"
    assert " " not in imported.name
    _assert_publishable(imported.name)


def test_a_name_at_the_cap_truncates_the_BASE_so_the_suffix_survives(tmp_path):
    """128 characters plus ``-2`` is refused by the hub, so the base gives way.

    A hub-legal name is allowed to be exactly :data:`MAX_AGENT_NAME_CHARS` long,
    so appending a suffix to one overflows the cap by the length of the suffix.
    Truncating the suffix to fit would re-create the collision the suffix exists
    to avoid, so the base is cut instead and the result stays a legal name.
    """

    registry = AgentRegistry(tmp_path / "config")
    base = "c" * MAX_AGENT_NAME_CHARS
    assert len(base) == MAX_AGENT_NAME_CHARS
    first, _ = registry.import_agent(_named_archive(tmp_path / "a", base))
    assert first.name == base

    second, renamed_from = registry.import_agent(_named_archive(tmp_path / "b", base))
    assert renamed_from == base
    assert len(second.name) == MAX_AGENT_NAME_CHARS
    assert second.name == base[: MAX_AGENT_NAME_CHARS - 2] + "-2"
    # The published name's PREFIX is what survives, so the renamed row still
    # reads as the agent the user asked for.
    assert second.name[:-2] == base[:-2]
    _assert_publishable(second.name)

    third, _ = registry.import_agent(_named_archive(tmp_path / "c", base))
    assert len(third.name) == MAX_AGENT_NAME_CHARS
    assert third.name == base[: MAX_AGENT_NAME_CHARS - 2] + "-3"
    _assert_publishable(third.name)


def test_resolve_import_name_is_pure_and_reports_the_incoming_name(tmp_path):
    """It has to answer without importing anything: the caller may be deciding
    whether to offer a pull at all."""

    registry = AgentRegistry(tmp_path / "config")
    assert registry.resolve_import_name("Scout") == ("Scout", None)
    registry.import_agent(_named_archive(tmp_path / "a", "Scout"))
    before = len(registry.list_agents())
    assert registry.resolve_import_name("scout") == ("scout-2", "scout")
    assert len(registry.list_agents()) == before


@pytest.mark.parametrize(
    "left,right",
    [
        ("Coder", "coder"),
        ("Code   Reviewer", "code reviewer"),
        ("  coder  ", "CODER"),
        # NFKC folds COMPATIBILITY codepoints: fullwidth Coder names the same
        # agent as the ASCII spelling, and a check that compared raw strings
        # would let both exist. This is the same normalisation the hub applies to
        # its own duplicate rule (contract §3.1).
        ("\uff23\uff4f\uff44\uff45\uff52", "coder"),
    ],
)
def test_agent_name_key_folds_case_whitespace_and_compatibility_forms(left, right):
    assert agent_name_key(left) == agent_name_key(right)


def test_agent_name_key_does_not_fold_confusables_or_punctuation():
    """Declared NOT normalised, so nobody later "fixes" this into a behaviour
    change no consumer expects (contract §3.1)."""

    assert agent_name_key("Code-Reviewer") != agent_name_key("Code_Reviewer")
    # Cyrillic \u0421, not ASCII C: a different script's letter in the same slot.
    assert agent_name_key("\u0421oder") != agent_name_key("Coder")
    # A blank name has nothing to collide with; ``""`` must never match a row.
    assert agent_name_key("") == ""
    assert agent_name_key("   ") == ""


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

"""The published built-in manifest must describe exactly the packaged seeds.

``local_operator/agent_seeds/manifest.json`` is the one list of built-in names
agent-server reserves (contract §5.2/§5.4): the app cannot have a built-in the
hub does not refuse, and the hub must not reserve a name no released client can
install. Neither side can see the other at runtime, so the property is held
together HERE — by a committed, generated file plus this test — and it holds
only as long as the test exists (contract §7.3 risk 5).

Two properties are deliberately stronger than "the file looks right":

* **Equality, not a subset.** ``test_the_manifest_names_are_exactly_the_packaged_seeds``
  compares in both directions. A subset check is what the pre-existing seed
  test does (``tests/unit/test_agent_profiles.py``), so a seventh seed file
  added without regenerating the manifest passed review — the exact silent
  divergence this test exists to prevent.
* **Byte identity, not field comparison.** The committed bytes must be what the
  generator renders now. Comparing fields one at a time would let a new field,
  a reordering or a stale hash through, and every one of those is a manifest
  agent-server cannot trust.

The provenance fields (``generated_at``, ``generator_version``) are preserved
from the committed file by the generator rather than re-read from the clock and
the installed package: the release flow bumps the package version in a
pyproject-only PR, so an implicitly-timestamped manifest would make this test
red after every release.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from local_operator.agent_profiles import (
    ROLE_TAG,
    SEEDS_DIR,
    AgentProfile,
    _split_frontmatter,
    install_seed,
    list_seeds,
    load_seed,
    seed_divergence,
    seed_origin,
    seed_tags,
)
from local_operator.agents import AgentRegistry
from scripts import gen_agent_seed_manifest as generator

MANIFEST_PATH = SEEDS_DIR / "manifest.json"

#: The URL agent-server fetches (contract §5.2), and the base of the per-seed
#: ``source_url`` a ``name_reserved_builtin`` refusal links to. Pinned here as
#: well as in the generator because changing either without the other is how
#: the hub would cite a document nobody publishes.
MANIFEST_URL = (
    "https://raw.githubusercontent.com/damianvtran/local-operator/main"
    "/local_operator/agent_seeds/manifest.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _seed_file_bytes(name: str) -> str:
    return (SEEDS_DIR / f"{name}.md").read_text(encoding="utf-8", errors="replace")


def test_the_manifest_names_are_exactly_the_packaged_seeds(manifest):
    """Drift in EITHER direction: a seed added or removed without regenerating.

    The failing direction matters more than it looks. agent-server refuses a
    publish whose name is in this manifest, so a manifest that has fallen
    behind reserves nothing for a newly shipped built-in (a user can squat it),
    and a manifest that runs ahead reserves a name no client has.
    """

    assert {entry["name"] for entry in manifest["seeds"]} == set(list_seeds())


def test_every_entry_describes_the_seed_the_app_actually_installs(manifest):
    """The manifest has to describe the INSTALLED role, not the raw file.

    ``instructions_sha256`` is the loader's own instruction string
    (``agent_profiles.py`` truncates the body to ``MAX_INSTRUCTIONS_CHARS``),
    ``tags`` are what ``seed_tags`` writes into the registry, and the category
    is the one install sets. A manifest built from an independent reading of
    the file would describe bytes the app never runs.
    """

    for entry in manifest["seeds"]:
        name = entry["name"]
        profile = load_seed(name)
        assert profile is not None, name
        # The reserved name must be the name the app INSTALLS, not merely the
        # file it reads: agent-server reserves what a published agent would
        # collide with, and a seed whose frontmatter name differed from its
        # filename would reserve a name no client ever has (the generator
        # refuses that case outright).
        assert entry["name"] == profile.name, name
        meta, body = _split_frontmatter(_seed_file_bytes(name))

        assert entry["description"] == profile.description, name
        assert entry["when_to_use"] == profile.when_to_use, name
        assert entry["version"] == str(meta["version"]), name
        instructions = profile.instructions
        assert (
            entry["instructions_sha256"] == hashlib.sha256(instructions.encode("utf-8")).hexdigest()
        ), name
        assert entry["instructions_chars"] == len(instructions), name
        # A body long enough to be truncated would make the published hash
        # describe less guidance than the file contains; the generator refuses
        # it, and this is the assertion that keeps that refusal honest.
        assert len(body) <= len(instructions), name
        assert entry["tools"] == (list(profile.tools) if profile.tools is not None else None), name
        assert entry["effort"] == profile.effort, name
        assert entry["delegate"] == bool(profile.may_delegate), name
        assert entry["tags"] == list(seed_tags(profile)), name
        assert entry["categories"] == [ROLE_TAG], name
        # The cited URL must be the seed file beside the manifest agent-server
        # fetches: the refusal links here, so a path that 404s is a dead link in
        # an error message a user is trying to act on.
        assert entry["source_url"] == MANIFEST_URL.rsplit("/", 1)[0] + f"/{name}.md"


def test_the_manifest_header_matches_the_document_contract(manifest):
    """agent-server rejects a schema it does not know (§5.3), so the version is
    part of the published contract rather than documentation."""

    assert manifest["schema_version"] == 1
    assert manifest["source"].endswith("/local-operator")
    assert manifest["seeds"], "an empty manifest would reserve nothing"
    assert len(manifest["seeds"]) <= 200, "§5.3 caps a fetched manifest at 200 seeds"
    assert manifest["generated_at"].endswith("Z"), manifest["generated_at"]
    assert manifest["generator_version"], "a support report needs the version that wrote this"


def test_regeneration_reproduces_the_committed_bytes():
    """The check that makes ``manifest.json`` generated rather than authored.

    Regenerating in the tree must be a no-op for anyone who has only changed a
    seed: the generator carries the committed provenance forward, so a
    reviewer can run one command instead of reasoning about timestamps.
    """

    assert generator.render(MANIFEST_PATH) == MANIFEST_PATH.read_text(encoding="utf-8")


def test_check_reports_drift_instead_of_writing(tmp_path, capsys):
    """``--check`` is what CI (or a reviewer) runs; it must fail loudly on a
    stale file and leave it alone, and pass on the committed one."""

    assert generator.main(["--check"]) == 0
    # A hand-edit of any published field is stale, not "close enough": derive
    # the stale copy from the real one so this cannot pass by accident.
    stale = tmp_path / "manifest.json"
    committed = MANIFEST_PATH.read_text(encoding="utf-8")
    stale.write_text(committed.replace('"instructions_chars": 763', '"instructions_chars": 762'))
    assert stale.read_text(encoding="utf-8") != committed
    assert generator.main(["--check", "--out", str(stale)]) == 1
    assert stale.read_text(encoding="utf-8") != committed, "--check must not rewrite the file"


def test_a_seventh_seed_cannot_render_without_being_described(monkeypatch):
    """The gap the earlier subset assertion left open, from the other side: a
    seed the generator cannot load must fail the generation rather than land in
    the manifest as a name with no hash."""

    monkeypatch.setattr(generator, "list_seeds", lambda: [*list_seeds(), "not-a-seed"])
    with pytest.raises(SystemExit):
        generator.render(MANIFEST_PATH)


def test_the_manifest_is_package_data_a_wheel_would_carry():
    """An editable install reads package data straight from the source tree, so
    a missing ``package-data`` entry is invisible until somebody installs a
    wheel — the failure mode the entry exists for. Both halves are asserted:
    that the declaration names the file, and that it is where the imported
    package (not a repository-relative guess) says it is."""

    declared = tomllib.loads((Path(generator.REPO) / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["setuptools"]["package-data"]["local_operator"]
    assert "agent_seeds/*.json" in declared

    packaged = files("local_operator") / "agent_seeds" / "manifest.json"
    assert packaged.is_file(), packaged
    assert Path(str(packaged)).resolve() == MANIFEST_PATH.resolve()


def test_the_seed_version_key_is_inert_in_the_product(tmp_path):
    """``version:`` exists for the manifest and nothing else.

    It must not become a profile field (nothing would read it), a tag (a
    registry row would carry a marker no code understands), or a divergence
    (every installed role would report as edited). The last one is the
    dangerous one: a reset that always looks needed is a reset that silently
    overwrites an operator's edits.
    """

    assert "version" not in AgentProfile.__dataclass_fields__

    registry = AgentRegistry(tmp_path / "config")
    for name in list_seeds():
        installed = install_seed(name, registry=registry)
        assert installed is not None, name
        profile, already_installed = installed
        assert already_installed is False, name
        row = registry.get_agent_by_name(name)
        assert row is not None
        assert seed_origin(row) == name
        assert all(not tag.startswith("version:") for tag in row.tags), row.tags
        seed = load_seed(name)
        assert seed is not None
        assert seed_divergence(profile, seed) == (), name
        # Installing it again is the idempotent path, and it must stay a no-op
        # on the tags: the seed's version is not a registry field.
        before = list(row.tags)
        install_seed(name, registry=registry)
        again = registry.get_agent_by_name(name)
        assert again is not None
        assert again.tags == before, name

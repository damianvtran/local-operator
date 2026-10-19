"""The Librarian's product surface: the prompts, the seed and its manifest row.

The prompts are part of the change rather than decoration: the stable head
block is what tells EVERY agent that the Library is global and the Canvas is
not, and the files-decision sentence is what makes a deliverable a durable
catalog entry instead of scratch. The seed is the product policy and the
manifest row is what makes ``agent librarian`` resolvable uninstalled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from local_operator import agent_profiles
from local_operator.supplements.decision import FILES_INSTRUCTIONS

SEEDS = Path(__file__).resolve().parents[3] / "local_operator" / "agent_seeds"


def test_the_system_prompt_has_a_library_section() -> None:
    from local_operator.prompts_api import render_template

    text = " ".join(render_template("system.md", {}).split())
    assert "## Library" in text
    assert "global, long-term catalog" in text
    assert "Canvas Files panel is session-local" in text
    assert "Produced and highlighted files enter the Library" in text
    assert "Use the `library` tool for durable names, descriptions and tags" in text
    assert "the Librarian agent maintains the taxonomy" in text


def test_the_files_decision_prompt_names_the_library() -> None:
    flat = " ".join(FILES_INSTRUCTIONS.split())
    assert "enter the user's long-term Library" in flat
    assert "prefer durable deliverables over incidental scratch" in flat


def test_the_seed_is_the_documented_policy() -> None:
    text = " ".join((SEEDS / "librarian.md").read_text(encoding="utf-8").split())
    assert "Nothing needs organizing today." in text
    for clause in (
        "Reuse before you invent",
        "Only add a new tag when a pattern repeats",
        "Avoid near-duplicates and aliases",
        "Never overwrite a user-edited field",
        "Skip locked assets",
        "Never delete, move or rename files",
        "No secrets",
        "Aida",
        "`library` tool",
        "global and long-term",
        "Canvas Files panel is per-session",
    ):
        assert clause in text, clause


def test_the_manifest_row_is_what_makes_her_resolvable() -> None:
    manifest = json.loads((SEEDS / "manifest.json").read_text(encoding="utf-8"))
    row = next(seed for seed in manifest["seeds"] if seed["name"] == "librarian")
    assert row["delegate"] is True
    assert "class:proactive" in row["tags"]
    assert "delegate:yes" in row["tags"]
    assert row["categories"] == ["role"]
    # The digest is the generator's contract: ``profile.instructions`` — the
    # exact string the loader holds — so a hand edit to the seed without a
    # manifest regeneration is red here rather than silently publishing a
    # stale hash.
    profile = agent_profiles.resolve_profile("librarian")
    assert profile is not None
    digest = hashlib.sha256(profile.instructions.encode("utf-8")).hexdigest()
    assert row["instructions_sha256"] == digest
    assert row["instructions_chars"] == len(profile.instructions)


def test_the_packaged_seed_resolves_without_an_install() -> None:
    profile = agent_profiles.resolve_profile("librarian")
    assert profile is not None
    assert profile.name == "librarian"
    assert "Nothing needs organizing today." in profile.instructions

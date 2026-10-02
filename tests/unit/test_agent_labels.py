"""Display LABELS for agents: the key/label split and the ONE display rule.

Mirrors the teams label suite on the agent side, with the agent-side refinement
(an exact-vs-casefold comparison against the derived default) frozen where it
differs. Every surface reads ``local_operator.display_labels``; these tests pin
the rule itself, the registry's derive-and-persist behaviour across BOTH write
paths, the seed install/reset lane, the mesh round-trip, and the listing rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator import display_labels as dl
from local_operator.agent_profiles import (
    agent_listing_rows,
    install_seed,
    load_seed,
    profile_from_agent,
    seed_divergence,
)
from local_operator.agents import AgentData, AgentEditFields, AgentRegistry


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    root = tmp_path / "agents"
    root.mkdir(parents=True, exist_ok=True)
    return AgentRegistry(root)


# --------------------------------------------------------------------------- #
# The shared rule (the display_labels module)
# --------------------------------------------------------------------------- #


def test_default_label_derives_a_title_case_name_and_upcases_initialisms() -> None:
    """The shared derivation, including the six-token initialism allowlist."""
    assert dl.default_label("ux-reviewer") == "UX Reviewer"
    assert dl.default_label("tui-designer") == "TUI Designer"
    assert dl.default_label("copy-reviewer") == "Copy Reviewer"
    assert dl.default_label("data-quality") == "Data Quality"
    assert dl.default_label("my.team_name") == "My Team Name"
    # ... and stays Title Case for a plausible-looking neighbour.
    assert dl.default_label("it-ops") == "It Ops"


def test_display_form_covers_the_agent_arms() -> None:
    """The four frozen shapes, plus the agent-side casefold refinement.

    Teams freeze an EXACT comparison against the derived default; the agent
    side casefolds it, so ``UX Reviewer`` paints ALONE where teams would paint
    ``UX Reviewer (ux-reviewer)``.
    """
    form = dl.display_form
    assert form("lopdev", "") == "lopdev"  # legacy row: nothing over the key
    assert form("lopdev", "Lopdev") == "lopdev"  # derived AND casefolds: noise
    assert form("ops", "OPS") == "ops"  # chosen but casefolds to the key: raw name
    assert form("ux-reviewer", "UX Reviewer") == "UX Reviewer"  # derived, informative
    # THE REFINEMENT: a differently-cased spelling of the derived default paints
    # the label alone rather than falling to the keyed form.
    assert form("ux-reviewer", "Ux Reviewer") == "Ux Reviewer"
    assert form("ops", "Platform Reliability") == "Platform Reliability (ops)"


def test_enrichment_label_only_for_a_chosen_different_label() -> None:
    """A description column prefixes nothing a name column already says (D2)."""
    assert dl.enrichment_label("ux-reviewer", "UX Reviewer") == ""
    assert dl.enrichment_label("lopdev", "Lopdev") == ""
    assert dl.enrichment_label("ops", "OPS") == ""
    assert dl.enrichment_label("ops", "") == ""
    assert dl.enrichment_label("ops", "Platform Reliability") == "Platform Reliability"


def test_bounded_display_form_keeps_the_key_on_the_line() -> None:
    """N1: an over-cap label ellipsizes; the addressing key survives."""
    long_label = "Platform Reliability And Infrastructure Hardening For The Whole Fleet"
    form = dl.bounded_display_form("ops", long_label)
    assert len(form) <= dl.AGENT_LISTING_CAP
    assert form.endswith(" (ops)")
    assert dl.bounded_display_form("ops", "") == "ops"
    assert dl.bounded_display_form("ops", "Ops") == "ops"


def test_validate_label_collapses_whitespace_caps_and_refuses_controls() -> None:
    assert dl.validate_label("  Platform   Reliability  ") == "Platform Reliability"
    with pytest.raises(ValueError, match="at most 80"):
        dl.validate_label("x " * 41)
    with pytest.raises(ValueError, match="control characters"):
        dl.validate_label("a\u200bb")  # zero-width joiner (Cf)
    assert dl.validate_label("") == ""  # empty is legal: "no custom label"


# --------------------------------------------------------------------------- #
# The registry: derive-and-persist across BOTH writers
# --------------------------------------------------------------------------- #


def test_create_persists_the_derived_label_and_normalizes_a_custom_one(
    registry: AgentRegistry,
) -> None:
    derived = registry.create_agent(AgentEditFields(name="ux-reviewer"))
    assert derived.label == "UX Reviewer"
    assert dl.display_form(derived.name, derived.label) == "UX Reviewer"

    custom = registry.create_agent(AgentEditFields(name="ops", label="  Platform   Reliability "))
    # Whitespace collapsed on store.
    assert custom.label == "Platform Reliability"
    assert dl.display_form(custom.name, custom.label) == "Platform Reliability (ops)"


def test_the_label_round_trips_through_agent_yml(registry: AgentRegistry) -> None:
    created = registry.create_agent(AgentEditFields(name="ops", label="Platform Reliability"))
    reloaded = AgentRegistry(Path(registry.agents_dir.parent))
    row = reloaded.get_agent(created.id)
    assert row is not None and row.label == "Platform Reliability"
    raw = yaml.safe_load((Path(registry.agents_dir) / created.id / "agent.yml").read_text())
    assert raw["label"] == "Platform Reliability"


def test_a_label_less_agent_yml_still_loads_and_derives_on_the_next_write(
    registry: AgentRegistry, tmp_path: Path
) -> None:
    """A legacy row (no ``label:`` key) loads as "" and the next save derives.

    Pydantic ignores unknown extras and supplies the default, so an old file is
    not a migration: the label becomes durable on the row's next write.
    """
    created = registry.create_agent(AgentEditFields(name="ux-reviewer"))
    meta_path = Path(registry.agents_dir) / created.id / "agent.yml"
    payload = yaml.safe_load(meta_path.read_text())
    payload.pop("label", None)
    meta_path.write_text(yaml.safe_dump(payload))

    fresh = AgentRegistry(Path(registry.agents_dir.parent))
    legacy = fresh.get_agent(created.id)
    assert legacy is not None and legacy.label == ""
    fresh.update_agent(created.id, AgentEditFields(description="touched"))
    assert fresh.get_agent(created.id).label == "UX Reviewer"


def test_update_label_none_leaves_and_empty_resets_to_derived(registry: AgentRegistry) -> None:
    agent = registry.create_agent(AgentEditFields(name="ops", label="Ops Display"))
    assert registry.update_agent(agent.id, AgentEditFields(description="d")).label == "Ops Display"
    assert registry.update_agent(agent.id, AgentEditFields(label=None)).label == "Ops Display"
    # An explicit "" is a RESET to the derived default, not "no label".
    assert registry.update_agent(agent.id, AgentEditFields(label="")).label == "Ops"


def test_label_validation_refuses_over_cap_and_controls_on_both_paths(
    registry: AgentRegistry,
) -> None:
    with pytest.raises(ValueError, match="at most 80"):
        registry.create_agent(AgentEditFields(name="a", label="x " * 41))
    with pytest.raises(ValueError, match="control characters"):
        registry.create_agent(AgentEditFields(name="b", label="a\u200bb"))

    agent = registry.create_agent(AgentEditFields(name="ops"))
    with pytest.raises(ValueError, match="at most 80"):
        registry.update_agent(agent.id, AgentEditFields(label="y " * 41))
    # The refusal left the row untouched (validated before any mutation).
    assert registry.get_agent(agent.id).label == "Ops"


def test_rename_re_derives_a_derived_label_and_keeps_a_custom_one(
    registry: AgentRegistry,
) -> None:
    """D3b: a stored label that is only the OLD name's default follows the rename."""
    derived = registry.create_agent(AgentEditFields(name="ux-reviewer"))
    renamed = registry.update_agent(derived.id, AgentEditFields(name="ux-designer"))
    assert renamed.name == "ux-designer"
    assert renamed.label == "UX Designer"

    custom = registry.create_agent(AgentEditFields(name="ops", label="Platform Reliability"))
    kept = registry.update_agent(custom.id, AgentEditFields(name="ops-v2"))
    assert kept.label == "Platform Reliability"
    # An explicit label in the same write wins over the re-derive.
    again = registry.update_agent(renamed.id, AgentEditFields(name="ux-writer", label="Chosen"))
    assert again.label == "Chosen"


def test_clone_re_derives_a_derived_label_and_copies_a_custom_one(
    registry: AgentRegistry,
) -> None:
    source = registry.create_agent(AgentEditFields(name="ux-reviewer"))
    clone = registry.clone_agent(source.id, "ux-reviewer-copy")
    assert clone.label == "UX Reviewer Copy"  # derived for the COPY's own name

    custom = registry.create_agent(AgentEditFields(name="ops", label="Platform Reliability"))
    custom_clone = registry.clone_agent(custom.id, "ops-copy")
    assert custom_clone.label == "Platform Reliability"  # a chosen label is copied


def test_the_name_collision_key_stays_label_free(registry: AgentRegistry) -> None:
    """A label never joins name-collision logic: two rows may share a label."""
    registry.create_agent(AgentEditFields(name="one", label="Shared Label"))
    registry.create_agent(AgentEditFields(name="two", label="Shared Label"))
    assert registry.get_agent_by_name("one") is not None
    assert registry.get_agent_by_name("two") is not None


# --------------------------------------------------------------------------- #
# Seeds: install ships the canonical label; a label edit is not divergence
# --------------------------------------------------------------------------- #


def test_install_seed_ships_the_canonical_label(registry: AgentRegistry) -> None:
    install_seed("ux-reviewer", registry=registry)
    row = registry.get_agent_by_name("ux-reviewer")
    assert row is not None
    assert row.label == "UX Reviewer"
    assert dl.display_form(row.name, row.label) == "UX Reviewer"


def test_reset_restores_the_packaged_label_after_a_local_edit(registry: AgentRegistry) -> None:
    install_seed("reviewer", registry=registry)
    row = registry.get_agent_by_name("reviewer")
    assert row is not None

    registry.update_agent(row.id, AgentEditFields(label="My Reviewer"))
    assert registry.get_agent_by_name("reviewer").label == "My Reviewer"

    # ``overwrite`` is the reset lane: the seed's own fields are restored,
    # including the canonical label, because it rides ``install_seed._fields``.
    install_seed("reviewer", registry=registry, overwrite=True)
    assert registry.get_agent_by_name("reviewer").label == "Reviewer"


def test_a_label_only_edit_is_not_seed_divergence(registry: AgentRegistry) -> None:
    """A label is display metadata, excluded from ``_SEED_FIELDS`` (teams' stance)."""
    install_seed("reviewer", registry=registry)
    row = registry.get_agent_by_name("reviewer")
    assert row is not None
    seed = load_seed("reviewer")
    assert seed is not None
    baseline = seed_divergence(profile_from_agent(registry, row), seed)

    registry.update_agent(row.id, AgentEditFields(label="Renamed For Display"))
    edited = registry.get_agent_by_name("reviewer")
    assert edited is not None
    # The label moved, and divergence is UNCHANGED from the baseline: a
    # retitle is not "edited since install".
    assert edited.label == "Renamed For Display"
    assert seed_divergence(profile_from_agent(registry, edited), seed) == baseline


# --------------------------------------------------------------------------- #
# Listing rows and the mesh
# --------------------------------------------------------------------------- #


def test_agent_listing_rows_carries_the_raw_label_beside_the_name(
    registry: AgentRegistry,
) -> None:
    install_seed("ux-reviewer", registry=registry)
    rows = agent_listing_rows(registry)
    row = next(r for r in rows if r[0] == "ux-reviewer")
    # (name, label, facts, summary): the RAW label, never a composed form.
    assert row[1] == "UX Reviewer"
    assert "role" in row[2]


def test_the_mesh_carries_the_label_and_preserves_it_on_apply(tmp_path: Path) -> None:
    """The trap recon found: a fresh AgentData rebuild must not wipe the label."""
    from local_operator import network as _network  # noqa: F401  (import path guard)
    from local_operator.network import definitions

    author = tmp_path / "author"
    author.mkdir()
    bare = tmp_path / "bare"
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(AgentEditFields(name="ops"))
    reg.update_agent(agent.id, AgentEditFields(label="Platform Reliability"))

    assert "label" in definitions.AGENT_DEFINITION_FIELDS

    bundle = definitions.local_bundle(author)
    assert definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])

    installed = AgentRegistry(bare).get_agent(agent.id)
    assert installed is not None
    assert installed.label == "Platform Reliability"


def test_the_mesh_derives_the_label_when_an_older_peer_omits_it(tmp_path: Path) -> None:
    """Version skew: a row without ``label`` re-derives locally, never fails."""
    from local_operator.network import definitions

    author = tmp_path / "author"
    author.mkdir()
    bare = tmp_path / "bare"
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(AgentEditFields(name="ux-reviewer"))

    bundle = definitions.local_bundle(author)
    # Simulate a bundle produced by an OLDER peer: the key is absent entirely.
    for row in bundle["agents"]:
        row.pop("label", None)
    definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])

    installed = AgentRegistry(bare).get_agent(agent.id)
    assert installed is not None
    assert installed.label == "UX Reviewer"


def test_agent_data_defaults_the_label_to_empty_for_old_files() -> None:
    """The additive field's default: an object built without it reads ""."""
    import uuid
    from datetime import datetime, timezone

    data = AgentData(
        id=str(uuid.uuid4()),
        name="legacy",
        created_date=datetime.now(timezone.utc),
        version="0.0.0",
    )
    assert data.label == ""


def _unused(_: Any) -> None:  # pragma: no cover - keeps the ``Any`` import honest
    return None

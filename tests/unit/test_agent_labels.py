"""Display LABELS for agents: the key/label split and the ONE display rule.

Mirrors the teams label suite on the agent side, with the agent-side
refinement (the derived-default arm tested casefold BEFORE the casefold-to-name
arm) frozen where it differs. Every surface reads ``local_operator.display_labels``;
these tests pin the rule itself, the registry's derive-and-persist behaviour
across BOTH write paths, the seed install/reset lane, the mesh's LOCAL-ONLY
handling, and the listing rows.

``AgentEditFields`` is built through :func:`_fields`, which spells every field:
pyright reads the model's synthesised ``__init__`` as requiring all of them
(``Field(None, ...)`` is not seen as a default), so a partial construction is a
``reportCallIssue`` error in CI's whole-tree run. ``label`` itself carries a
plain default so the ~100 pre-existing construction sites did not have to gain
a parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import display_labels as dl
from local_operator.agent_profiles import (
    agent_listing_rows,
    install_seed,
    load_seed,
    profile_from_agent,
    seed_divergence,
)
from local_operator.agents import MAX_AGENT_LABEL_CHARS, AgentEditFields, AgentRegistry
from local_operator.network import definitions


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    root = tmp_path / "agents"
    root.mkdir(parents=True, exist_ok=True)
    return AgentRegistry(root)


def _fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with EVERY field spelled out, overridden as needed."""
    base: dict[str, Any] = dict(
        name=None,
        label=None,
        security_prompt=None,
        hosting=None,
        model=None,
        description=None,
        tags=None,
        categories=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


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
    """The four frozen shapes, with the derived arm tested FIRST (D1).

    A canonical label is the seed author's spelling and is painted even when it
    differs from the key only in case: ``coder`` + ``Coder`` paints ``Coder``.
    That arm order is the agent side's deliberate divergence from teams.
    """
    form = dl.display_form
    assert form("lopdev", "") == "lopdev"  # legacy row: nothing over the key
    # The derived arm first (D1): single-token slugs keep their human spelling.
    assert form("coder", "Coder") == "Coder"
    assert form("aida", "Aida") == "Aida"
    assert form("ux-reviewer", "UX Reviewer") == "UX Reviewer"
    assert form("ux-reviewer", "Ux Reviewer") == "Ux Reviewer"  # casefold derived arm
    # A single-token name's case variant IS its derived default, so a stored
    # ``LOPDEV`` paints as stored rather than collapsing to the raw key.
    assert form("lopdev", "LOPDEV") == "LOPDEV"
    # The casefold-to-name arm still collapses a label that IS the raw key text
    # -- reachable where the derived default differs from the key (multi-token).
    assert form("ux-reviewer", "ux-reviewer") == "ux-reviewer"
    assert form("ops", "Platform Reliability") == "Platform Reliability (ops)"


def test_enrichment_label_only_for_a_chosen_different_label() -> None:
    """A description column prefixes nothing a name column already says (D2)."""
    assert dl.enrichment_label("ux-reviewer", "UX Reviewer") == ""
    assert dl.enrichment_label("coder", "Coder") == ""
    assert dl.enrichment_label("lopdev", "Lopdev") == ""
    assert dl.enrichment_label("ops", "OPS") == ""
    assert dl.enrichment_label("ops", "") == ""
    assert dl.enrichment_label("ops", "Platform Reliability") == "Platform Reliability"


def test_bounded_display_form_keeps_the_key_on_the_line() -> None:
    """N1: an over-cap chosen label ellipsizes; the addressing key survives."""
    long_label = "Platform Reliability And Infrastructure Hardening For The Whole Fleet"
    form = dl.bounded_display_form("ops", long_label)
    assert len(form) <= dl.AGENT_LISTING_CAP
    assert form.endswith(" (ops)")
    assert dl.bounded_display_form("ops", "") == "ops"
    assert dl.bounded_display_form("coder", "Coder") == "Coder"


def test_validate_label_collapses_whitespace_caps_and_refuses_controls() -> None:
    assert dl.validate_label("  Platform   Reliability  ") == "Platform Reliability"
    with pytest.raises(ValueError, match="at most 80"):
        dl.validate_label("x " * 41)
    with pytest.raises(ValueError, match="control characters"):
        dl.validate_label("a\u200bb")  # zero-width space (Cf)
    assert dl.validate_label("") == ""


def test_a_derived_label_can_never_exceed_the_agent_cap() -> None:
    """R1-2: the agent cap is the NAME cap, so a derived label is always valid."""
    assert MAX_AGENT_LABEL_CHARS >= 128
    long_name = "a" * MAX_AGENT_LABEL_CHARS
    derived = dl.default_label(long_name)
    # The derive path STORES this unvalidated; the validator must accept it.
    assert dl.validate_label(derived, max_chars=MAX_AGENT_LABEL_CHARS) == derived


def test_capped_display_form_keeps_the_name_in_a_truncating_cell() -> None:
    """D2: a fixed-width cell falls back to the raw name when the form won't fit.

    The band (20 cells) and the dock (14) truncate rather than wrap, so a
    composed label past the cap would be cut mid-word and lose the key.
    """
    capped = dl.capped_display_form
    # Inside the cap: the shared display form, unchanged.
    assert capped("ops", "Platform", cap=20) == "Platform (ops)"
    assert capped("coder", "Coder", cap=20) == "Coder"
    # Past the cap: the RAW NAME (addressable), never a truncated blend of both.
    assert capped("ops", "Platform Reliability", cap=20) == "ops"
    assert capped("ops", "Platform Reliability", cap=14) == "ops"
    # A name that alone overflows is returned as-is -- nothing shorter to paint.
    assert capped("a" * 30, "", cap=20) == "a" * 30


# --------------------------------------------------------------------------- #
# The registry: derive-and-persist across both writers
# --------------------------------------------------------------------------- #


def test_create_persists_the_derived_label_and_normalizes_a_custom_one(
    registry: AgentRegistry,
) -> None:
    derived = registry.create_agent(_fields(name="ux-reviewer"))
    assert derived.label == "UX Reviewer"

    custom = registry.create_agent(_fields(name="ops", label="  Platform   Reliability "))
    assert custom.label == "Platform Reliability"
    assert dl.display_form(custom.name, custom.label) == "Platform Reliability (ops)"


def test_the_label_round_trips_through_agent_yml(registry: AgentRegistry) -> None:
    created = registry.create_agent(_fields(name="dashboard-sme"))
    meta_path = Path(registry.agents_dir) / created.id / "agent.yml"
    assert "label: Dashboard Sme" in meta_path.read_text()

    reopened = AgentRegistry(Path(registry.agents_dir.parent))
    row = reopened.get_agent(created.id)
    assert row is not None and row.label == "Dashboard Sme"


def test_a_label_less_agent_yml_still_loads_and_derives_on_the_next_write(
    registry: AgentRegistry,
) -> None:
    """A pre-upgrade row has no ``label`` key: it reads as "" and derives later."""
    created = registry.create_agent(_fields(name="ux-reviewer"))
    meta_path = Path(registry.agents_dir) / created.id / "agent.yml"
    stripped = "\n".join(
        line for line in meta_path.read_text().splitlines() if not line.startswith("label:")
    )
    meta_path.write_text(stripped)

    reopened = AgentRegistry(Path(registry.agents_dir.parent))
    row = reopened.get_agent(created.id)
    assert row is not None
    assert row.label == ""
    assert dl.display_form(row.name, row.label) == "ux-reviewer"

    reopened.update_agent(row.id, _fields(description="d"))
    refreshed = reopened.get_agent(row.id)
    assert refreshed is not None and refreshed.label == "UX Reviewer"


def test_update_label_none_leaves_and_empty_resets_to_derived(
    registry: AgentRegistry,
) -> None:
    created = registry.create_agent(_fields(name="ops", label="Platform Reliability"))

    assert (
        registry.update_agent(created.id, _fields(description="d")).label == "Platform Reliability"
    )
    assert registry.update_agent(created.id, _fields(label=None)).label == "Platform Reliability"

    # An explicit "" is a RESET to the derived default, not "no label".
    assert registry.update_agent(created.id, _fields(label="")).label == "Ops"


def test_label_validation_refuses_over_cap_and_controls_on_both_paths(
    registry: AgentRegistry,
) -> None:
    created = registry.create_agent(_fields(name="ops"))
    with pytest.raises(ValueError, match="at most 128"):
        registry.update_agent(created.id, _fields(label="x " * 65))
    with pytest.raises(ValueError, match="control characters"):
        registry.create_agent(_fields(name="other", label="a\u200bb"))
    # The refused update left the stored row untouched.
    unchanged = registry.get_agent(created.id)
    assert unchanged is not None and unchanged.label == "Ops"


def test_rename_re_derives_a_derived_label_and_keeps_a_custom_one(
    registry: AgentRegistry,
) -> None:
    derived = registry.create_agent(_fields(name="lopdev"))
    assert derived.label == "Lopdev"
    renamed = registry.update_agent(derived.id, _fields(name="lop-dev"))
    assert renamed.name == "lop-dev"
    assert renamed.label == "Lop Dev"

    custom = registry.create_agent(_fields(name="ops", label="Platform Reliability"))
    kept = registry.update_agent(custom.id, _fields(name="ops-v2"))
    assert kept.label == "Platform Reliability"
    # An explicit label in the same write still wins over the re-derive.
    again = registry.update_agent(renamed.id, _fields(name="lop-v3", label="Chosen"))
    assert again.label == "Chosen"


def test_clone_re_derives_a_derived_label_and_copies_a_custom_one(
    registry: AgentRegistry,
) -> None:
    derived = registry.create_agent(_fields(name="ux-reviewer"))
    copy = registry.clone_agent(derived.id, "design-system")
    assert copy.label == "Design System"

    custom = registry.create_agent(_fields(name="ops", label="Platform Reliability"))
    kept = registry.clone_agent(custom.id, "ops-two")
    assert kept.label == "Platform Reliability"


def test_the_name_collision_key_stays_label_free(registry: AgentRegistry) -> None:
    """A label never joins the name-collision logic (display-only)."""
    registry.create_agent(_fields(name="ops", label="Totally Different"))
    with pytest.raises(ValueError, match="already exists"):
        registry.create_agent(_fields(name="ops", label="Another"))
    # A label that collides with another row's NAME is fine: labels are not keys.
    registry.create_agent(_fields(name="other", label="ops"))


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_install_seed_ships_the_canonical_label(registry: AgentRegistry) -> None:
    install_seed("ux-reviewer", registry=registry)
    row = registry.get_agent_by_name("ux-reviewer")
    assert row is not None and row.label == "UX Reviewer"


def test_reset_restores_the_packaged_label_after_a_local_edit(
    registry: AgentRegistry,
) -> None:
    install_seed("reviewer", registry=registry)
    row = registry.get_agent_by_name("reviewer")
    assert row is not None
    registry.update_agent(row.id, _fields(label="My Reviewer"))
    retitled = registry.get_agent_by_name("reviewer")
    assert retitled is not None and retitled.label == "My Reviewer"

    # ``overwrite`` is the reset lane: the seed's own fields are restored,
    # including the canonical label, because it rides ``install_seed._fields``.
    install_seed("reviewer", registry=registry, overwrite=True)
    reset = registry.get_agent_by_name("reviewer")
    assert reset is not None and reset.label == "Reviewer"


def test_a_label_only_edit_is_not_seed_divergence(registry: AgentRegistry) -> None:
    """A label is display metadata, excluded from ``_SEED_FIELDS`` (teams' stance)."""
    install_seed("reviewer", registry=registry)
    row = registry.get_agent_by_name("reviewer")
    assert row is not None
    seed = load_seed("reviewer")
    assert seed is not None
    baseline = seed_divergence(profile_from_agent(registry, row), seed)

    registry.update_agent(row.id, _fields(label="Renamed For Display"))
    edited = registry.get_agent_by_name("reviewer")
    assert edited is not None
    assert edited.label == "Renamed For Display"
    # The label moved, and divergence is UNCHANGED from the baseline.
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


def test_the_label_is_deliberately_off_the_mesh_wire(tmp_path: Path) -> None:
    """R1-1: the label is LOCAL display metadata, never in the definition row.

    Carrying it re-shaped every mirror's recorded digest, so an unchanged
    bundle reported a false local-edits conflict on upgrade.
    """
    assert "label" not in definitions.AGENT_DEFINITION_FIELDS

    author = tmp_path / "author"
    bare = tmp_path / "bare"
    author.mkdir()
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(_fields(name="ops"))
    reg.update_agent(agent.id, _fields(label="Platform Reliability"))

    bundle = definitions.local_bundle(author)
    for row in bundle["agents"]:
        assert "label" not in row["fields"], "the label must never ride the wire"


def test_mesh_apply_preserves_a_local_retitle_and_never_conflicts_on_it(
    tmp_path: Path,
) -> None:
    """R1-1: a retitle is local, so an unchanged bundle still reports ``unchanged``."""
    author = tmp_path / "author"
    bare = tmp_path / "bare"
    author.mkdir()
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(_fields(name="ops"))

    bundle = definitions.local_bundle(author)
    first = definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])
    assert first["conflicts"] == []

    # The label re-derives on the mirror (it never travelled) ...
    mirror = AgentRegistry(bare)
    installed = mirror.get_agent(agent.id)
    assert installed is not None and installed.label == "Ops"

    # ... the operator RETITLES it locally ...
    mirror.update_agent(agent.id, _fields(label="Platform Reliability"))

    # ... and re-applying the SAME bundle is ``unchanged``, not a false conflict.
    second = definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])
    assert second["conflicts"] == [], second
    assert [row["name"] for row in second["unchanged"]] == ["ops"]
    retitled = AgentRegistry(bare).get_agent(agent.id)
    assert retitled is not None and retitled.label == "Platform Reliability"


def test_mesh_install_re_derives_the_label_from_the_name(tmp_path: Path) -> None:
    """A brand-new mirror has no local label, so the row derives one."""
    author = tmp_path / "author"
    bare = tmp_path / "bare"
    author.mkdir()
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(_fields(name="ux-reviewer"))

    bundle = definitions.local_bundle(author)
    definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])

    installed = AgentRegistry(bare).get_agent(agent.id)
    assert installed is not None
    assert installed.label == "UX Reviewer"


def test_the_apply_rebuild_carries_a_chosen_local_label_through_an_update(
    tmp_path: Path,
) -> None:
    """R1-1, the PRESERVE half: a CONTENT update must not drop the local label.

    The re-apply test above short-circuits to ``unchanged`` -- a retitle is
    invisible to the digest, which is the point of keeping the label off the wire
    -- so it never enters the branch that rebuilds a fresh ``AgentData``. This one
    FORCES that branch: the author's row changes while the mirror holds a chosen
    label, so the apply is an ``updated`` and the rebuild must carry
    ``local_label`` instead of a fresh row's derived one.
    """
    author = tmp_path / "author"
    bare = tmp_path / "bare"
    author.mkdir()
    bare.mkdir()

    reg = AgentRegistry(author)
    agent = reg.create_agent(_fields(name="ops", description="first"))

    bundle = definitions.local_bundle(author)
    assert definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])

    # The mirror picks its OWN label (a retitle the origin knows nothing about).
    mirror = AgentRegistry(bare)
    mirror.update_agent(agent.id, _fields(label="Platform Reliability"))

    # The author changes CONTENT, so the bundle differs and the rebuild runs.
    reg.update_agent(agent.id, _fields(description="second"))
    changed = definitions.local_bundle(author)
    outcome = definitions.apply_bundle(bare, changed, origin_device=changed["origin_device"])

    assert [row["name"] for row in outcome["updated"]] == ["ops"], outcome
    assert outcome["conflicts"] == [], outcome

    after = AgentRegistry(bare).get_agent(agent.id)
    assert after is not None
    # The content moved AND the mirror's chosen label survived the rebuild.
    assert after.description == "second"
    assert after.label == "Platform Reliability"

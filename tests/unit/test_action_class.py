"""``local_operator.action_class`` — encode/decode, effective-class reads, switch.

The class mechanism's three halves, in the order they run in production:
parsing/tag round-trips (``AgentProfile`` side), the effective class of a
session (attachment sidecar → registry → packaged seed), and the switch's
storage helper (flip a tag, materialize a seed when needed). The precedence
pins live here because this is where the one resolver lives; the profile-side
parse/encode is pinned in ``test_agent_profiles.py`` and the tool/TUI surfaces
in their own files.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.action_class import (
    PROACTIVE,
    REACTIVE,
    class_from_tags,
    normalize,
    session_action_class,
    set_registered_action_class,
    with_class_tag,
)
from local_operator.agents import AgentRegistry
from local_operator.resume import write_session_attachment


def _fields(**overrides: Any):
    """``AgentEditFields`` with every field spelled out (strict mode)."""
    from local_operator.agents import AgentEditFields

    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
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


class TestNormalize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("proactive", PROACTIVE),
            ("Proactive ", PROACTIVE),
            ("REACTIVE", REACTIVE),
            ("", REACTIVE),
            (None, REACTIVE),
            ("nonsense", REACTIVE),
            (7, REACTIVE),
        ],
    )
    def test_unknown_and_missing_read_reactive(self, value, expected) -> None:
        assert normalize(value) == expected

    def test_the_default_is_overridable_for_strict_callers(self) -> None:
        assert normalize("bogus", default="") == ""


class TestTags:
    def test_absent_tag_reads_reactive(self) -> None:
        assert class_from_tags(["role", "delegate:yes"]) == REACTIVE
        assert class_from_tags([]) == REACTIVE
        assert class_from_tags(None) == REACTIVE

    def test_a_tag_round_trips_case_insensitively(self) -> None:
        assert class_from_tags(["role", "Class:Proactive"]) == PROACTIVE

    def test_reactive_is_written_explicitly_not_left_absent(self) -> None:
        """A deliberate ``reactive`` is RECORDED, not implied by an absence.

        This is the encoding the class backfill's safety rests on
        (``agent_profiles.backfill_seed_action_class``): a row with no class tag
        means "never classified" — the state every row installed before the
        class feature existed is in — while a row the operator switched off
        says so. Without the distinction, a repair for those pre-class rows
        would equally re-arm a check-in the operator had switched off, which is
        the one outcome the class exists to prevent.
        """
        tags = with_class_tag(["role", "class:proactive", "seed:aida"], REACTIVE)
        assert tags == ("role", "seed:aida", "class:reactive")
        assert with_class_tag(tags, REACTIVE) == tags  # idempotent
        assert class_from_tags(tags) == REACTIVE
        # ...and absent still reads reactive, so pre-class rows and rows
        # installed from a starter that declares nothing keep working.
        assert class_from_tags(["role", "seed:aida"]) == REACTIVE

    def test_proactive_is_appended_once(self) -> None:
        tags = with_class_tag(["role"], PROACTIVE)
        assert tags == ("role", "class:proactive")
        assert with_class_tag(tags, PROACTIVE) == tags

    def test_a_flip_replaces_rather_than_stacks(self) -> None:
        # One class tag per row is the invariant the readers depend on
        # (``class_from_tags`` takes the first): flipping back and forth must
        # not leave a duplicate behind for a later flip to trip over.
        tags = with_class_tag(["role", "class:proactive"], REACTIVE)
        again = with_class_tag(tags, PROACTIVE)
        assert again == ("role", "class:proactive")
        assert [t for t in again if t.startswith("class:")] == ["class:proactive"]


class TestSessionActionClass:
    def _session(self, root, agent: str = "aida"):
        session_dir = root / "sessions" / "abc"
        session_dir.mkdir(parents=True, exist_ok=True)
        write_session_attachment(session_dir, team="", agent=agent, goal="")
        return session_dir

    def test_no_attachment_reads_reactive(self, tmp_path) -> None:
        session_dir = tmp_path / "sessions" / "abc"
        session_dir.mkdir(parents=True)
        assert session_action_class(session_dir) == REACTIVE

    def test_a_seed_attachment_resolves_the_seeds_class(self, tmp_path) -> None:
        session_dir = self._session(tmp_path)
        assert session_action_class(session_dir) == PROACTIVE  # the packaged aida seed

    def test_an_unknown_name_reads_reactive(self, tmp_path) -> None:
        session_dir = self._session(tmp_path, agent="nobody-by-this-name")
        assert session_action_class(session_dir) == REACTIVE

    def test_the_registry_row_wins_over_the_seed(self, tmp_path) -> None:
        # The switch writes the INSTALLED row; the seed is only the fallback.
        registry = AgentRegistry(tmp_path)
        set_registered_action_class(registry, "aida", REACTIVE)
        session_dir = self._session(tmp_path)
        assert session_action_class(session_dir, registry=registry) == REACTIVE

    def test_a_broken_sidecar_reads_reactive(self, tmp_path, monkeypatch) -> None:
        session_dir = self._session(tmp_path)
        from local_operator import resume as resume_module

        def _boom(_directory):
            raise OSError("unreadable")

        monkeypatch.setattr(resume_module, "read_session_attachment", _boom)
        assert session_action_class(session_dir) == REACTIVE


class TestSetRegisteredActionClass:
    def test_flips_an_installed_role_both_ways(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        set_registered_action_class(registry, "aida", REACTIVE)
        row = registry.get_agent_by_name("aida")
        assert row is not None and class_from_tags(row.tags) == REACTIVE
        set_registered_action_class(registry, "aida", PROACTIVE)
        row = registry.get_agent_by_name("aida")
        assert row is not None and class_from_tags(row.tags) == PROACTIVE

    def test_materializes_a_packaged_seed_to_hold_the_flip(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        assert registry.get_agent_by_name("aida") is None
        resolved = set_registered_action_class(registry, "aida", REACTIVE)
        assert resolved == "aida"
        row = registry.get_agent_by_name("aida")
        assert row is not None and class_from_tags(row.tags) == REACTIVE

    def test_conversational_rows_are_refused(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        # An ordinary conversational row: named, but neither a role nor a
        # specialist — the switch must not adopt it (the fail-open hijack the
        # role tag exists to stop).
        registry.create_agent(_fields(name="chatty"))
        with pytest.raises(ValueError, match="no agent named"):
            set_registered_action_class(registry, "chatty", PROACTIVE)

    def test_a_bad_class_is_refused_before_any_write(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        with pytest.raises(ValueError, match="must be one of"):
            set_registered_action_class(registry, "aida", "sideways")
        assert registry.get_agent_by_name("aida") is None

    def test_an_unknown_name_is_refused(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        with pytest.raises(ValueError, match="no agent named"):
            set_registered_action_class(registry, "nobody", PROACTIVE)

    def test_a_case_variant_spelling_reaches_the_row(self, tmp_path) -> None:
        """``/agent class Aida`` must not refuse the agent the report form names.

        The sibling resolvers fold case (``resolve_profile``,
        ``install_seed``), so the switch is the one storage path that must too:
        refusing the natural spelling made the two halves of one command
        disagree — the report form resolved ``Aida`` while the flip insisted
        on ``aida`` (agent review round 1, R2).
        """
        from local_operator.agent_profiles import install_seed

        registry = AgentRegistry(tmp_path)
        install_seed("aida", registry=registry)
        assert set_registered_action_class(registry, "Aida", PROACTIVE) == "aida"
        row = registry.get_agent_by_name("aida")
        assert row is not None and class_from_tags(row.tags) == PROACTIVE
        # The OTHER spelling flips the SAME row: one row, one flip, no
        # second copy materialized under the typed casing.
        assert set_registered_action_class(registry, "AIDA", REACTIVE) == "aida"
        assert len([a for a in registry.list_agents() if a.name.lower() == "aida"]) == 1

    def test_a_specialist_can_be_switched(self, tmp_path) -> None:
        registry = AgentRegistry(tmp_path)
        registry.create_agent(
            _fields(name="laner", description="a small lane", categories=["specialist"])
        )
        resolved = set_registered_action_class(registry, "laner", PROACTIVE)
        assert resolved == "laner"
        row = registry.get_agent_by_name("laner")
        assert row is not None and class_from_tags(row.tags) == PROACTIVE


class TestClassSwitchClause:
    def test_the_clause_uses_the_users_word_for_the_rhythm(self) -> None:
        """One clause, one wording, for all three switch handlers.

        ``cadence`` is the code's word; every user-facing surface says
        "check-in", so the receipt must too — and an outcome that did nothing
        reads as NO clause, because a dangling separator is worse than
        silence (design round 1, D2).
        """
        from local_operator.action_class import class_switch_clause

        assert class_switch_clause({"patience_cancelled": ["p1"], "cadence_dropped": True}) == (
            "; 1 pending wait(s) cancelled, her check-ins stopped"
        )
        assert class_switch_clause({"patience_cancelled": ["p1"]}) == (
            "; 1 pending wait(s) cancelled"
        )
        assert class_switch_clause({"patience_cancelled": []}) == ""
        assert class_switch_clause(None) == ""
        assert "cadence" not in class_switch_clause({"cadence_dropped": True})

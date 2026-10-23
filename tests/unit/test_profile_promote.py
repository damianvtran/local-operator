"""Promotion: copy a conversation definition into the account (LOH-9 S4).

The contract these tests pin, in the order the composite enforces it:

* the source is resolved SESSION-FIRST, and a name that only exists in the
  account is refused ("already available everywhere") rather than copied twice;
* every refusal happens BEFORE the first write, so a refusal leaves no row;
* the copy is verbatim (tags, label, prompt, both briefs) with a fresh identity,
  and the conversation keeps its own row — copy, not move;
* a team's roster is resolved through the LAUNCH resolver, so a packaged starter
  that is simply not installed is reused and a session-scoped member is copied in
  the same call, bounded by the nesting depth and the definition cap;
* a failure rolls back exactly the rows this call created, in reverse, and a
  rollback that itself fails is REPORTED — never reported as success;
* ``replace`` is in-place (the row keeps its id) and restores the user's previous
  definition if the overwrite fails;
* the origin conversation still launches its own copy while a second conversation
  gets the account one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import profile_scope
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import (
    MAX_PROMOTION_ROWS,
    PromotionIncomplete,
    ScopedAgentRegistry,
    ScopedTeamRegistry,
    scope_agent_registry,
    scope_session_registries,
    scope_team_registry,
)
from local_operator.role_provenance import read_promotions
from local_operator.teams import TeamEditFields, TeamRegistry, parse_members

ORIGIN = "surface:tui"


def _fields(**overrides: Any) -> AgentEditFields:
    base: dict[str, Any] = dict(
        name=None,
        label=None,
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


def _session_role(registry: Any, name: str, *, prompt: str | None = None, **extra: Any) -> Any:
    row = registry.create_agent(
        _fields(
            name=name,
            label=f"{name.title()} Label",
            description=f"{name} when needed",
            tags=["role", "class:proactive", "tools:read,bash", "delegate:yes"],
            categories=["scratch"],
            security_prompt="no secrets",
            model="anthropic/claude",
            **extra,
        ),
        scope="session",
    )
    registry.set_agent_system_prompt(row.id, prompt if prompt is not None else f"{name} prompt\n")
    return row


@pytest.fixture
def stacks(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    origin_dir = config / "sessions" / "aaaaaaaa-origin"
    second_dir = config / "sessions" / "bbbbbbbb-second"
    global_agents, global_teams = AgentRegistry(config), TeamRegistry(config)
    agents, teams = scope_session_registries(global_agents, global_teams, origin_dir)
    assert isinstance(agents, ScopedAgentRegistry)
    assert isinstance(teams, ScopedTeamRegistry)
    return {
        "config": config,
        "origin_dir": origin_dir,
        "second_dir": second_dir,
        "global_agents": global_agents,
        "global_teams": global_teams,
        "agents": agents,
        "teams": teams,
    }


def _second_conversation(stacks: dict[str, Any]) -> tuple[Any, Any]:
    second_agents = scope_agent_registry(stacks["global_agents"], stacks["second_dir"])
    second_teams = scope_team_registry(stacks["global_teams"], stacks["second_dir"])
    return second_agents, second_teams


# --- refusals, all before the first write ---------------------------------------


def test_a_name_only_the_account_holds_is_already_available(stacks) -> None:
    role = _session_role(stacks["agents"], "shared")
    stacks["agents"].promote("shared", authorized_by=ORIGIN)
    second_agents, _ = _second_conversation(stacks)

    with pytest.raises(ValueError, match="already in your account"):
        second_agents.promote("shared", authorized_by=ORIGIN)
    assert stacks["global_agents"].get_agent_by_name("shared").id != role.id


def test_an_unknown_name_is_a_key_error(stacks) -> None:
    with pytest.raises(KeyError):
        stacks["agents"].promote("nobody", authorized_by=ORIGIN)


def test_a_conversation_agent_that_is_not_a_role_is_refused(stacks) -> None:
    row = stacks["agents"].create_agent(
        _fields(name="chat-only", description="plain agent"), scope="session"
    )

    with pytest.raises(ValueError, match="not a role or specialist"):
        stacks["agents"].promote(row.name, authorized_by=ORIGIN)


def test_a_name_collision_refuses_without_replace(stacks) -> None:
    _session_role(stacks["agents"], "release-check")
    account = stacks["global_agents"].create_agent(_fields(name="release-check", tags=["role"]))

    with pytest.raises(ValueError, match="would overwrite it"):
        stacks["agents"].promote("release-check", authorized_by=ORIGIN)
    # Nothing was written: the account row is the one that was already there.
    assert stacks["global_agents"].get_agent_by_name("release-check").id == account.id
    assert read_promotions(stacks["config"]) == []


def test_a_target_name_this_conversation_also_holds_is_refused(stacks) -> None:
    _session_role(stacks["agents"], "one")
    _session_role(stacks["agents"], "two")

    with pytest.raises(ValueError, match="already has an agent named 'two'"):
        stacks["agents"].promote("one", as_name="two", authorized_by=ORIGIN)


def test_a_name_with_a_space_is_refused(stacks) -> None:
    _session_role(stacks["agents"], "one")

    with pytest.raises(ValueError, match="cannot contain spaces"):
        stacks["agents"].promote("one", as_name="two words", authorized_by=ORIGIN)


def test_an_unwritable_index_refuses_before_any_row_is_written(stacks, monkeypatch) -> None:
    _session_role(stacks["agents"], "one")

    def refuse(_config):
        raise OSError("read-only config root")

    monkeypatch.setattr(profile_scope.role_provenance, "require_writable", refuse)
    with pytest.raises(OSError, match="read-only"):
        stacks["agents"].promote("one", authorized_by=ORIGIN)
    assert stacks["global_agents"].get_agent_by_name("one") is None


def test_a_promotion_must_name_its_origin(stacks) -> None:
    _session_role(stacks["agents"], "one")

    with pytest.raises(ValueError, match="must name the surface"):
        stacks["agents"].promote("one")


# --- the copy itself ------------------------------------------------------------


def test_the_account_row_is_a_verbatim_copy_with_its_own_identity(stacks) -> None:
    source = _session_role(stacks["agents"], "release-check", prompt="P\x00\n")

    result = stacks["agents"].promote("release-check", authorized_by=ORIGIN)
    account = stacks["global_agents"].get_agent_by_name("release-check")

    assert result.created == [{"kind": "agent", "name": "release-check", "id": account.id}]
    assert account.id != source.id
    assert account.tags == source.tags
    assert account.categories == source.categories
    assert account.label == source.label
    assert account.description == source.description
    assert account.security_prompt == source.security_prompt
    assert account.model == source.model
    assert (
        stacks["global_agents"].get_agent_system_prompt(account.id)
        == stacks["agents"].get_agent_system_prompt(source.id)
        == "P\x00\n"
    )
    # COPY, not move: the conversation row is still there, untouched.
    assert stacks["agents"].session_registry.get_agent_by_name("release-check").id == source.id
    assert result.expanded == [] and result.replaced == []


def test_promotion_under_another_name_keeps_the_source_name(stacks) -> None:
    source = _session_role(stacks["agents"], "one")

    stacks["agents"].promote("one", as_name="two", authorized_by=ORIGIN)

    assert stacks["global_agents"].get_agent_by_name("two") is not None
    assert stacks["global_agents"].get_agent_by_name("one") is None
    assert source.name == "one"


def test_a_dry_run_names_the_rows_and_writes_nothing(stacks) -> None:
    _session_role(stacks["agents"], "one")

    plan = stacks["agents"].promote("one", authorized_by=ORIGIN, dry_run=True)

    assert plan.dry_run is True
    assert [(row["kind"], row["name"], row["id"]) for row in plan.created] == [("agent", "one", "")]
    assert plan.provenance_ids == []
    assert stacks["global_agents"].get_agent_by_name("one") is None
    assert read_promotions(stacks["config"]) == []


def test_the_index_records_the_promotion(stacks) -> None:
    source = _session_role(stacks["agents"], "one")

    result = stacks["agents"].promote("one", authorized_by=ORIGIN)
    rows = read_promotions(stacks["config"])

    assert [row["id"] for row in rows] == result.provenance_ids
    assert rows[0]["kind"] == "agent"
    assert rows[0]["name"] == "one"
    assert rows[0]["source_session"] == "aaaaaaaa-origin"
    assert rows[0]["authorized_by"] == ORIGIN
    assert rows[0]["replaced"] is None
    assert rows[0]["id"] != source.id


# --- rollback -------------------------------------------------------------------


def test_a_failed_prompt_write_leaves_no_row(stacks, monkeypatch) -> None:
    _session_role(stacks["agents"], "one")
    real = stacks["global_agents"].set_agent_system_prompt

    def boom(_id, _text):
        raise OSError("disk full")

    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", boom)
    with pytest.raises(OSError, match="disk full"):
        stacks["agents"].promote("one", authorized_by=ORIGIN)
    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", real)

    assert stacks["global_agents"].get_agent_by_name("one") is None
    assert read_promotions(stacks["config"]) == []


def test_a_rollback_that_fails_is_reported_not_hidden(stacks, monkeypatch) -> None:
    _session_role(stacks["agents"], "one")
    real_prompt = stacks["global_agents"].set_agent_system_prompt
    real_delete = stacks["global_agents"].delete_agent

    def boom(*_args):
        raise OSError("cannot write")

    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", boom)
    monkeypatch.setattr(stacks["global_agents"], "delete_agent", boom)
    with pytest.raises(PromotionIncomplete) as caught:
        stacks["agents"].promote("one", authorized_by=ORIGIN)
    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", real_prompt)
    monkeypatch.setattr(stacks["global_agents"], "delete_agent", real_delete)

    assert caught.value.remaining[0]["name"] == "one"
    assert "remains in your account" in str(caught.value)


def test_replace_restores_the_previous_definition_when_the_prompt_write_fails(
    stacks, monkeypatch
) -> None:
    _session_role(stacks["agents"], "one", prompt="take two\n")
    stacks["global_agents"].create_agent(
        _fields(name="one", tags=["role"], description="the user's own")
    )
    account = stacks["global_agents"].get_agent_by_name("one")
    stacks["global_agents"].set_agent_system_prompt(account.id, "user prompt\n")
    real = stacks["global_agents"].set_agent_system_prompt
    calls = {"n": 0}

    def failing(_id, text):
        calls["n"] += 1
        # The overwrite's prompt write is the first call; the restore's is second.
        if calls["n"] == 1:
            raise OSError("disk full")
        return real(_id, text)

    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", failing)
    with pytest.raises(OSError, match="disk full"):
        stacks["agents"].promote("one", replace=True, authorized_by=ORIGIN)
    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", real)

    restored = stacks["global_agents"].get_agent_by_name("one")
    assert restored.id == account.id
    assert restored.description == "the user's own"
    assert stacks["global_agents"].get_agent_system_prompt(restored.id) == "user prompt\n"


# --- replace --------------------------------------------------------------------


def test_replace_overwrites_in_place_and_names_what_it_replaced(stacks) -> None:
    _session_role(stacks["agents"], "one", prompt="take two\n")
    stacks["global_agents"].create_agent(
        _fields(name="one", tags=["role"], description="the user's own")
    )
    account = stacks["global_agents"].get_agent_by_name("one")

    result = stacks["agents"].promote("one", replace=True, authorized_by=ORIGIN)

    replaced = stacks["global_agents"].get_agent_by_name("one")
    assert replaced.id == account.id
    assert replaced.description != "the user's own"
    assert result.replaced == ["one"]
    assert result.created == []
    assert stacks["global_agents"].get_agent_system_prompt(replaced.id) == "take two\n"
    assert read_promotions(stacks["config"])[-1]["replaced"] == "one"


# --- teams ----------------------------------------------------------------------


def _team(registry: Any, name: str, *, manager: str, members: list[str], **extra: Any) -> Any:
    return registry.create_team(
        TeamEditFields(
            name=name,
            description=f"{name} roster",
            manager=manager,
            members=parse_members(members),
            instructions=f"{name} collaboration",
            project=f"{name} project",
            **extra,
        ),
        scope="session",
    )


def test_a_team_promotes_the_session_members_its_roster_needs(stacks) -> None:
    _session_role(stacks["agents"], "lead", prompt="lead\n")
    _session_role(stacks["agents"], "helper", prompt="helper\n")
    _team(stacks["teams"], "pod", manager="lead", members=["helper:2"])

    result = stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    assert sorted(row["name"] for row in result.created) == ["helper", "lead", "pod"]
    assert sorted(result.expanded) == ["helper", "lead"]
    account_team = stacks["global_teams"].get_team_by_name("pod")
    assert account_team.instructions == "pod collaboration"
    assert account_team.project == "pod project"
    assert [member.role for member in account_team.members] == ["helper"]
    assert [member.count for member in account_team.members] == [2]
    # The members' own rows are account-wide, with their prompts.
    assert stacks["global_agents"].get_agent_system_prompt(
        stacks["global_agents"].get_agent_by_name("helper").id
    ) == ("helper\n")
    # Every created definition is in the index, the team last.
    rows = read_promotions(stacks["config"])
    assert [row["name"] for row in rows] == ["lead", "helper", "pod"]
    assert rows[-1]["kind"] == "team"


def test_a_packaged_starter_on_the_roster_is_reused_not_copied(stacks) -> None:
    _session_role(stacks["agents"], "helper")
    # 'manager' is a packaged seed: resolvable everywhere, so the roster needs
    # no copy of it (the launch resolver's answer, not store membership).
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])

    result = stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    assert [row["name"] for row in result.created] == ["helper", "pod"]
    assert stacks["global_agents"].get_agent_by_name("manager") is None


def test_an_unresolvable_roster_refuses_before_any_write(stacks) -> None:
    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="helper", members=["ghost"])

    with pytest.raises(ValueError, match="ghost"):
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    assert stacks["global_agents"].get_agent_by_name("helper") is None
    assert stacks["global_teams"].get_team_by_name("pod") is None


def test_a_member_collision_refuses_the_whole_team(stacks) -> None:
    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    stacks["global_agents"].create_agent(_fields(name="helper", tags=["role"]))

    with pytest.raises(ValueError, match="would overwrite it"):
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    assert stacks["global_teams"].get_team_by_name("pod") is None


def test_a_team_over_the_expansion_bound_refuses(stacks, monkeypatch) -> None:
    _session_role(stacks["agents"], "a")
    _session_role(stacks["agents"], "b")
    _team(stacks["teams"], "pod", manager="a", members=["b"])
    monkeypatch.setattr(profile_scope, "MAX_PROMOTION_ROWS", 2)

    with pytest.raises(ValueError, match="past the 2-definition bound"):
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    assert stacks["global_agents"].get_agent_by_name("a") is None


def test_the_origin_and_a_second_conversation_resolve_the_right_copies(stacks) -> None:
    from local_operator.server.utils.desktop_profiles import (
        profile_detail,
        validate_target,
    )

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    second_agents, second_teams = _second_conversation(stacks)

    # The second conversation sees the ACCOUNT rows...
    assert profile_detail(second_agents, "helper", detail=False)["scope"] == "global"
    assert second_teams.get_team_by_name("pod") is not None
    assert validate_target(second_agents, second_teams, "team", "pod") == "pod"
    # ...while the origin keeps launching its own (session-first) copy.
    assert profile_detail(stacks["agents"], "helper", detail=False)["scope"] == "session"
    assert (
        stacks["agents"].get_agent_by_name("helper").id
        != second_agents.get_agent_by_name("helper").id
    )


def test_a_team_rollback_that_fails_names_the_rows_it_left(stacks, monkeypatch) -> None:
    """A remainder is reported by NAME, not as a count.

    The operator has to find the rows in their account, so the sentence lists them
    and ``PromotionIncomplete.remaining`` carries their ids.
    """

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])

    def exploding_create(_fields):
        raise OSError("team store full")

    def locked_delete(_agent_id):
        raise OSError("agent store locked")

    monkeypatch.setattr(stacks["global_teams"], "create_team", exploding_create)
    monkeypatch.setattr(stacks["global_agents"], "delete_agent", locked_delete)

    with pytest.raises(PromotionIncomplete) as caught:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    assert "agent 'helper' may still be in your account" in str(caught.value)
    assert caught.value.remaining[0]["name"] == "helper"


def test_a_team_promotion_rolling_back_removes_members_too(stacks, monkeypatch) -> None:
    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    real_create = stacks["global_teams"].create_team
    monkeypatch.setattr(
        stacks["global_teams"],
        "create_team",
        lambda _fields: (_ for _ in ()).throw(OSError("team store full")),
    )

    with pytest.raises(OSError, match="team store full"):
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    monkeypatch.setattr(stacks["global_teams"], "create_team", real_create)

    assert stacks["global_agents"].get_agent_by_name("helper") is None
    assert stacks["global_teams"].get_team_by_name("pod") is None
    assert read_promotions(stacks["config"]) == []


def test_the_index_survives_a_team_save(stacks) -> None:
    """The reason provenance is NOT a file inside the row directory.

    ``TeamRegistry._save_team_locked`` publishes a row by renaming a whole staged
    directory over it and writes exactly the row + two briefs, so a sidecar would
    be destroyed by the first edit the user makes to the team they just copied.
    """

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    before = read_promotions(stacks["config"])

    account = stacks["global_teams"].get_team_by_name("pod")
    assert not (stacks["config"] / "teams" / str(account.id) / "provenance.json").exists()
    account.description = "edited after promotion"
    stacks["global_teams"].save_team(account)

    assert read_promotions(stacks["config"]) == before
    assert stacks["global_teams"].get_team_by_name("pod").description == ("edited after promotion")


def test_a_team_promotion_needs_the_agent_registry(stacks) -> None:
    _team(stacks["teams"], "pod", manager="manager", members=[])

    with pytest.raises(ValueError, match="agent registry"):
        stacks["teams"].promote("pod", authorized_by=ORIGIN)


def test_a_team_dry_run_names_the_closure(stacks) -> None:
    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])

    plan = stacks["teams"].promote(
        "pod", authorized_by=ORIGIN, agents=stacks["agents"], dry_run=True
    )

    assert [(row["kind"], row["name"]) for row in plan.created] == [
        ("agent", "helper"),
        ("team", "pod"),
    ]
    assert plan.expanded == ["helper"]
    assert stacks["global_teams"].get_team_by_name("pod") is None


def test_the_documented_cap_is_positive() -> None:
    # The bound is a contract the refusal text quotes; a zero would make every
    # team promotion refuse, which no test above would otherwise notice.
    assert MAX_PROMOTION_ROWS > 1


def test_a_team_promotion_holds_the_agent_composites_lock(stacks, monkeypatch) -> None:
    """The team lock alone is not enough: it plans and writes AGENT rows.

    Proven by CONTENTION rather than by inspection: with the agent composite's
    lock held elsewhere, a team promotion must wait for it (and, past the bound,
    refuse in words). Before the fix it took only the team lock, so an `agent
    promote` (agent lock only) could pass the same member name as free while the
    team plan did, and both wrote — leaving two account rows with one name
    (agent review round 1, MAJOR-1).
    """

    from local_operator import profile_scope

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    monkeypatch.setattr(profile_scope, "WRITE_LOCK_TIMEOUT_S", 0.05)

    # Held from ANOTHER thread: the composite's lock is an RLock, so the same
    # thread would simply re-enter it and prove nothing.
    import threading

    lock = stacks["agents"].write_lock
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        lock.acquire()
        held.set()
        release.wait(10)
        lock.release()

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(10)
    try:
        with pytest.raises(TimeoutError, match="role write lock"):
            stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    finally:
        release.set()
        thread.join(10)

    # The refusal wrote nothing, and the team lock it took first was released.
    assert stacks["global_teams"].get_team_by_name("pod") is None
    assert stacks["global_agents"].get_agent_by_name("helper") is None
    assert stacks["teams"].write_lock.acquire(timeout=0.1)
    stacks["teams"].write_lock.release()


def test_a_double_rollback_failure_reports_BOTH_remainders(stacks, monkeypatch) -> None:
    """An inner stuck row is carried into the outer report (MINOR-2).

    ``_create_agent`` raises ``PromotionIncomplete`` naming its own row when its
    own rollback fails, and that row is not in the caller's undo list — the
    create raised before registering a step. Rebuilding the message from the
    outer list alone dropped it, so the receipt under-reported the residue.
    """

    _session_role(stacks["agents"], "alpha", prompt="alpha prompt\n")
    _session_role(stacks["agents"], "beta", prompt="beta prompt\n")
    _team(stacks["teams"], "pod", manager="manager", members=["alpha", "beta"])

    real_prompt = stacks["global_agents"].set_agent_system_prompt

    def prompt_boom(_id, text):
        if text == "beta prompt\n":
            raise OSError("prompt write failed")
        return real_prompt(_id, text)

    def delete_boom(_id):
        raise OSError("agent store locked")

    monkeypatch.setattr(stacks["global_agents"], "set_agent_system_prompt", prompt_boom)
    monkeypatch.setattr(stacks["global_agents"], "delete_agent", delete_boom)

    with pytest.raises(PromotionIncomplete) as caught:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    names = {row["name"] for row in caught.value.remaining}
    assert names == {"alpha", "beta"}, caught.value.remaining
    assert "alpha" in str(caught.value) and "beta" in str(caught.value)


def test_a_member_collision_advises_the_TEAM_verb(stacks) -> None:
    """A colliding member must not be answered with `/agent promote`.

    That advice replaces the member alone and still leaves the team unpromoted,
    so following it reproduces the same refusal (QA round 2, Q-M1).
    """

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="manager", members=["helper"])
    stacks["global_agents"].create_agent(_fields(name="helper", tags=["role"]))

    with pytest.raises(ValueError) as caught:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    text = str(caught.value)
    assert "/team promote pod replace" in text
    assert "/agent promote" not in text
    assert stacks["global_teams"].get_team_by_name("pod") is None
    assert stacks["global_agents"].get_agent_by_name("helper") is not None


def test_a_member_collision_offers_no_rename_that_cannot_work(stacks) -> None:
    """`as` renames the TEAM, so it cannot clear a colliding MEMBER.

    `/team promote pod as pod2` reaches this identical refusal — the member keeps
    its name, which is the key every roster references — so the sentence must not
    offer the rename as a second remedy (post-merge review of LOH-9 S4, MINOR-1).
    """

    _session_role(stacks["agents"], "helper")
    _team(stacks["teams"], "pod", manager="helper", members=[])
    stacks["global_agents"].create_agent(_fields(name="helper", tags=["role"]))

    with pytest.raises(ValueError) as plain:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    with pytest.raises(ValueError) as renamed:
        stacks["teams"].promote(
            "pod", as_name="pod2", authorized_by=ORIGIN, agents=stacks["agents"]
        )

    text = str(plain.value)
    assert "needs its agent 'helper'" in text
    assert "/team promote pod replace" in text
    assert "another name" not in text
    # The alternate name changes nothing: same sentence, still nothing written.
    assert str(renamed.value) == text
    assert stacks["global_teams"].get_team_by_name("pod") is None
    assert stacks["global_teams"].get_team_by_name("pod2") is None


def test_a_nested_team_collision_advises_the_TEAM_verb(stacks) -> None:
    """A colliding nested TEAM is a member too, so it gets the outer verb.

    `/team promote inner replace` replaces the inner team and still leaves `pod`
    unpromoted, reproducing this refusal — the agent-member branch was covered,
    this sibling was not (post-merge review of LOH-9 S4, MINOR-3).
    """

    _team(stacks["teams"], "inner", manager="manager", members=[])
    _team(stacks["teams"], "pod", manager="manager", members=["team:inner"])
    stacks["global_teams"].create_team(TeamEditFields(name="inner", manager="manager"))

    with pytest.raises(ValueError) as caught:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    text = str(caught.value)
    assert "needs its team 'inner'" in text
    assert "/team promote pod replace" in text
    assert "/team promote inner replace" not in text
    assert stacks["global_teams"].get_team_by_name("pod") is None


def test_a_nested_alias_collision_names_the_alias_and_a_working_remedy(stacks) -> None:
    """An alias another account row owns is not a team the account has by name.

    The generic sentence would claim an account team named 'inner' exists and
    advise `/team promote inner replace`; the account holds `other`, whose ALIAS
    is 'shared', and replacing 'inner' clears nothing (post-merge review of
    LOH-9 S4, MINOR-2). The remedy the sentence names is exercised here.
    """

    inner = _team(stacks["teams"], "inner", manager="manager", members=[], aliases=["shared"])
    _team(stacks["teams"], "pod", manager="manager", members=["team:inner"])
    stacks["global_teams"].create_team(
        TeamEditFields(name="other", manager="manager", aliases=["shared"])
    )

    with pytest.raises(ValueError) as caught:
        stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])

    text = str(caught.value)
    assert "'shared'" in text and "'other'" in text
    assert "named 'inner'" not in text
    assert "/team promote inner replace" not in text
    # The outer replace is not the remedy either: it targets the inner NAME.
    with pytest.raises(ValueError) as replaced:
        stacks["teams"].promote("pod", replace=True, authorized_by=ORIGIN, agents=stacks["agents"])
    assert str(replaced.value) == text

    # The named remedy — drop the alias on the conversation's team — works.
    stacks["teams"].update_team(inner.id, TeamEditFields(aliases=[]))
    result = stacks["teams"].promote("pod", authorized_by=ORIGIN, agents=stacks["agents"])
    assert "pod" in [row["name"] for row in result.created]

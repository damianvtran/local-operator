"""A fork carries the parent's conversation-scoped roles, and owns its copy.

The persona a session wears is re-resolved at construction from a NAME in
``attachment.json`` through the session's registry composite. That composite is
rooted at ``<session>/roles`` (``profile_scope``), so a fork that copies the
sidecar but not the tree opens wearing a persona it cannot resolve — the fork
looks like a fork that lost its team, and the restore notice is the only trace.

These tests pin the copy and the properties whose violation is silent:
byte-identical definitions, INDEPENDENCE in both directions (a fork edit or
delete must never reach the parent), lazy creation (a session with no roles
must not gain an empty tree), the byte bound, and the two things a fork must
never copy: a symlink pointing out of the tree, and the device-level stash.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator import fork as fork_module
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import SESSION_ROLES_DIRNAME, scope_session_registries
from local_operator.resume import ATTACHMENT_SIDECAR_NAME, TRANSCRIPT_NAME
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields, TeamRegistry

from .session.test_session import MODEL, ScriptedStream

PARENT_ID = "aaaaaaaaaaaa"


def _role_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` spelled out in full (it validates in strict mode)."""
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


def _registries(config_dir: Path, session_id: str) -> tuple[Any, Any]:
    """The session's registries, scoped the way ``session_factory`` scopes them.

    The same two calls (``scope_agent_registry`` / ``scope_team_registry``) over
    the same directory the factory derives, so what a fork's ``Session`` reads
    here is what it reads in a booted conversation.
    """
    agents, teams = scope_session_registries(
        AgentRegistry(config_dir), TeamRegistry(config_dir), config_dir / "sessions" / session_id
    )
    return agents, teams


def _role(agents: Any, name: str, prompt: str) -> Any:
    row = agents.create_agent(
        _role_fields(name=name, description=f"{name} for this task", tags=["role"]),
        scope="session",
    )
    agents.set_agent_system_prompt(row.id, prompt)
    return row


def _team(teams: Any, name: str, instructions: str) -> Any:
    return teams.create_team(
        TeamEditFields(
            name=name,
            description=f"{name} roster",
            manager="manager",
            members=[],
            instructions=instructions,
            project="",
        ),
        scope="session",
    )


def _seed_parent(config_dir: Path) -> Path:
    """The parent's session directory with the transcript a fork requires."""
    parent = config_dir / "sessions" / PARENT_ID
    parent.mkdir(parents=True, exist_ok=True)
    (parent / TRANSCRIPT_NAME).write_text(
        '{"id": "m1", "role": "user", "content": "audit the loader"}\n'
        '{"id": "m2", "role": "assistant", "content": "Done."}\n',
        encoding="utf-8",
    )
    return parent


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, keyed by relative path, with its bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _session(config_dir: Path, session_id: str) -> Session:
    """A session over ``<config>/sessions/<id>``, as boot/resume constructs it."""
    agents, teams = _registries(config_dir, session_id)
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(config_dir / "sessions" / session_id),
        system_blocks_provider=lambda: [],
        agent_registry=agents,
        team_registry=teams,
    )


class TestTheForkCarriesTheDefinitions:
    def test_a_fork_copies_a_conversation_scoped_role_byte_identical(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        parent = _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        row = _role(agents, "release-check", "You check releases, in order.")

        fork_id = fork_module.fork_session(config, PARENT_ID)

        fork_roles = config / "sessions" / fork_id / SESSION_ROLES_DIRNAME
        assert _tree(fork_roles) == _tree(parent / SESSION_ROLES_DIRNAME)
        # The row itself, not just bytes: the fork's registry resolves it by
        # name and by id with the SAME prompt the parent wrote.
        fork_agents, _ = _registries(config, fork_id)
        copied = fork_agents.get_agent_by_name("release-check")
        assert copied is not None and copied.id == row.id
        assert fork_agents.get_agent_system_prompt(copied.id) == "You check releases, in order."

    def test_a_fork_copies_a_conversation_scoped_team(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        parent = _seed_parent(config)
        _, teams = _registries(config, PARENT_ID)
        row = _team(teams, "audit", "Review in order, then ship.")

        fork_id = fork_module.fork_session(config, PARENT_ID)

        fork_roles = config / "sessions" / fork_id / SESSION_ROLES_DIRNAME
        assert _tree(fork_roles) == _tree(parent / SESSION_ROLES_DIRNAME)
        _, fork_teams = _registries(config, fork_id)
        copied = fork_teams.get_team_by_name("audit")
        assert copied is not None and copied.id == row.id

    def test_a_fork_of_a_session_wearing_a_session_role_restores_the_persona(
        self, tmp_path: Path
    ) -> None:
        """THE bug this slice exists for: the fork silently lost its persona.

        The parent attaches the role through the real resolver (so
        ``attachment.json`` names it), the fork is taken, and the fork's own
        ``Session`` construction is what has to bring it back — including the
        instructions, not merely the name on the band.
        """
        config = tmp_path / "config"
        _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        _role(agents, "auditor", "You audit the loader.")
        first = _session(config, PARENT_ID)
        first.attach_agent_profile("auditor")
        assert (config / "sessions" / PARENT_ID / ATTACHMENT_SIDECAR_NAME).is_file()

        fork_id = fork_module.fork_session(config, PARENT_ID)
        forked = _session(config, fork_id)

        assert forked.active_agent == "auditor"
        assert forked.attachment_restore_notice == ""
        assert getattr(forked, "_unresolved_agent", "") == ""
        assert "You audit the loader." in forked._goal_state.agent_brief


class TestTheForkIsIndependent:
    def test_a_forked_role_is_independent_of_the_parent(self, tmp_path: Path) -> None:
        """Edits and deletes, BOTH directions, never cross the fork boundary."""
        config = tmp_path / "config"
        parent = _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        row = _role(agents, "release-check", "You check releases.")
        parent_tree = _tree(parent / SESSION_ROLES_DIRNAME)

        fork_id = fork_module.fork_session(config, PARENT_ID)
        fork_agents, _ = _registries(config, fork_id)
        fork_roles = config / "sessions" / fork_id / SESSION_ROLES_DIRNAME

        # Fork edits its copy: the parent's bytes do not move.
        fork_agents.update_agent(row.id, _role_fields(description="Fork-only description"))
        assert _tree(parent / SESSION_ROLES_DIRNAME) == parent_tree
        assert fork_agents.get_agent(row.id).description == "Fork-only description"

        # Parent edits its copy: the fork's bytes do not move.
        fork_tree = _tree(fork_roles)
        agents.update_agent(row.id, _role_fields(description="Parent-only description"))
        assert _tree(fork_roles) == fork_tree
        assert fork_agents.get_agent(row.id).description == "Fork-only description"

        # Fork deletes its copy: the parent still resolves the row and prompt.
        fork_agents.delete_agent(row.id)
        assert fork_agents.get_agent_by_name("release-check") is None
        assert agents.get_agent_by_name("release-check") is not None
        assert agents.get_agent_system_prompt(row.id) == "You check releases."

        # And the reverse: deleting the parent's row leaves the fork's intact.
        fork_agents.create_agent(
            _role_fields(name="release-check", description="fork only", tags=["role"]),
            scope="session",
        )
        agents.delete_agent(row.id)
        assert agents.get_agent_by_name("release-check") is None
        assert fork_agents.get_agent_by_name("release-check") is not None

    def test_a_fork_of_a_session_with_no_roles_creates_no_roles(self, tmp_path: Path) -> None:
        """Lazy creation is an S1 invariant: absence must stay absent."""
        config = tmp_path / "config"
        _seed_parent(config)

        fork_id = fork_module.fork_session(config, PARENT_ID)

        assert not (config / "sessions" / PARENT_ID / SESSION_ROLES_DIRNAME).exists()
        assert not (config / "sessions" / fork_id / SESSION_ROLES_DIRNAME).exists()


class TestWhatTheForkRefuses:
    def test_a_fork_refuses_a_tree_over_the_bound(self, tmp_path: Path) -> None:
        """The bound is decided BEFORE the copy: nothing is written, no fork."""
        config = tmp_path / "config"
        parent = _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        row = _role(agents, "release-check", "You check releases.")
        oversized = parent / SESSION_ROLES_DIRNAME / "agents" / row.id / "blob.bin"
        # Sparse: the summed size crosses the bound without spending the disk
        # the bound exists to protect.
        with oversized.open("wb") as handle:
            handle.truncate(fork_module.MAX_ROLES_COPY_BYTES + 1)

        with pytest.raises(fork_module.ForkError) as raised:
            fork_module.fork_session(config, PARENT_ID)

        message = str(raised.value)
        assert "MiB" in message
        assert str(oversized) in message
        assert "Move those files out" in message and "fork again" in message
        # No fork, and no staging directory left behind.
        assert [p.name for p in (config / "sessions").iterdir()] == [PARENT_ID]
        assert oversized.is_file()

    def test_a_symlink_in_the_roles_tree_is_not_copied_out_of_the_tree(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / "config"
        parent = _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        row = _role(agents, "release-check", "You check releases.")
        outside = tmp_path / "outside.md"
        outside.write_text("not a definition", encoding="utf-8")
        link = parent / SESSION_ROLES_DIRNAME / "agents" / row.id / "linked.md"
        os.symlink(outside, link)

        with pytest.raises(fork_module.ForkError) as raised:
            fork_module.fork_session(config, PARENT_ID)

        assert "symlink" in str(raised.value)
        assert [p.name for p in (config / "sessions").iterdir()] == [PARENT_ID]
        # The link is still a LINK in the parent (the fork never rewrote it) and
        # no copy of it exists: the only file named linked.md is the parent's.
        assert link.is_symlink()
        assert [p for p in (config / "sessions").rglob("linked.md")] == [link]

    def test_the_stash_is_never_copied_into_a_fork(self, tmp_path: Path) -> None:
        """The device-level stash is another root; the fork reads ONE directory."""
        config = tmp_path / "config"
        _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        _role(agents, "release-check", "You check releases.")
        stash = config / "dropped-session-roles" / PARENT_ID
        stash.mkdir(parents=True)
        (stash / "stash.json").write_text('{"dropped": ["gone"]}', encoding="utf-8")

        fork_id = fork_module.fork_session(config, PARENT_ID)

        fork_dir = config / "sessions" / fork_id
        assert "dropped-session-roles" not in str(list(fork_dir.rglob("*")))
        assert all(b'{"dropped"' not in body for body in _tree(fork_dir).values())
        # The stash itself is untouched.
        assert (stash / "stash.json").read_text(encoding="utf-8") == '{"dropped": ["gone"]}'


class TestTheForkCopyDoesNotLeakIntoTheStore:
    def test_a_fork_never_writes_a_global_row(self, tmp_path: Path) -> None:
        """The copy target is the fork's own directory and nothing else."""
        config = tmp_path / "config"
        _seed_parent(config)
        agents, teams = _registries(config, PARENT_ID)
        _role(agents, "release-check", "You check releases.")
        _team(teams, "audit", "Review in order.")

        fork_module.fork_session(config, PARENT_ID)

        global_agents = AgentRegistry(config)
        global_teams = TeamRegistry(config)
        assert global_agents.get_agent_by_name("release-check") is None
        assert global_teams.get_team_by_name("audit") is None

    def test_the_parent_directory_is_not_touched_by_the_roles_copy(self, tmp_path: Path) -> None:
        """mtime and bytes of the parent's tree survive, as the file copy's do."""
        config = tmp_path / "config"
        parent = _seed_parent(config)
        agents, _ = _registries(config, PARENT_ID)
        _role(agents, "release-check", "You check releases.")
        before = _tree(parent / SESSION_ROLES_DIRNAME)
        before_stat = (parent / SESSION_ROLES_DIRNAME / "agents").stat()

        fork_id = fork_module.fork_session(config, PARENT_ID)

        assert _tree(parent / SESSION_ROLES_DIRNAME) == before
        assert (parent / SESSION_ROLES_DIRNAME / "agents").stat().st_mtime == before_stat.st_mtime
        # The staging directory is INSIDE the fork and is published away; a
        # leftover would be a directory retention could read as session content.
        assert list(parent.glob(".roles-*")) == []
        assert list((config / "sessions" / fork_id).glob(".roles-*")) == []

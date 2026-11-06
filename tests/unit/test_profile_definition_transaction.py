"""Regression: an authored save must not overwrite a competing definition write.

The durable profile approval executor needs a single cross-process definition
transaction, not a CAS check around metadata followed by an independent prompt
write. This same-process/two-registry interleaving deterministically exercises
that window without relying on timing or real operator stores.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, TypedDict

import pytest

from local_operator.agents import AgentRegistry
from local_operator.harness.types import TextContent
from local_operator.hub_sync.definition import (
    DefinitionRevisionMismatch,
    definition_revision,
)
from local_operator.profile_scope import scope_agent_registry
from local_operator.tools.agent_tool import AgentParams, write_profile
from tests.unit.tools.test_agent_tool import _edit_fields


class _TransactionKwargs(TypedDict, total=False):
    transaction_id: str
    before: str
    after: str
    read: Callable[[], str]
    write: Callable[[str], None]
    backup: Callable[[str], None]


def test_metadata_prompt_window_does_not_clobber_competing_writer(tmp_path, monkeypatch):
    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(
        _edit_fields(name="reviewer", description="Before", tags=["role"], categories=["role"])
    )
    account.set_agent_system_prompt(row.id, "Original guidance.")
    competitor = AgentRegistry(tmp_path / "account")
    composite = scope_agent_registry(account, tmp_path / "sessions" / "test-session")
    revision = definition_revision("agent", row.id, composite.definition_revision_fields(row.id))
    original_update = composite.update_agent

    def interleave(*args, **kwargs):
        updated = original_update(*args, **kwargs)
        # The competing supported writer lands after the metadata CAS but
        # before the first save's separate set_agent_system_prompt call.
        competitor.set_agent_system_prompt(row.id, "Competing guidance must survive.")
        return updated

    monkeypatch.setattr(composite, "update_agent", interleave)
    try:
        write_profile(
            composite,
            AgentParams(
                op="update", name="reviewer", description="After", instructions="Approved guidance."
            ),
            creating=False,
            scope="global",
            expected_definition_revision=revision,
        )
    except DefinitionRevisionMismatch:
        # A nested competing registry is refused before it can land; the
        # transaction must restore its own earlier metadata change as well.
        account._refresh_agents_metadata()
        assert account.get_agent(row.id).description == "Before"
        assert competitor.get_agent_system_prompt(row.id) == "Original guidance."
        return
    assert competitor.get_agent_system_prompt(row.id) == "Competing guidance must survive."


def test_fault_between_metadata_and_prompt_rolls_back_both(tmp_path, monkeypatch):
    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(
        _edit_fields(name="reviewer", description="Before", tags=["role"], categories=["role"])
    )
    account.set_agent_system_prompt(row.id, "Original guidance.")
    original = account.set_agent_system_prompt
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fault between metadata and prompt")
        return original(*args, **kwargs)

    monkeypatch.setattr(account, "set_agent_system_prompt", fail_once)
    import pytest

    with pytest.raises(OSError, match="injected fault"):
        write_profile(
            account,
            AgentParams(
                op="update", name="reviewer", description="After", instructions="Approved guidance."
            ),
            creating=False,
            scope="global",
        )
    account._refresh_agents_metadata()
    assert account.get_agent(row.id).description == "Before"
    assert account.get_agent_system_prompt(row.id) == "Original guidance."
    import json

    journals = list((account.config_dir / "profile-transactions").glob("*.json"))
    assert json.loads(journals[-1].read_text())["state"] == "rolled_back"


def test_cross_process_writer_waits_for_definition_boundary(tmp_path):
    import os
    import subprocess
    import sys

    from local_operator.definition_transaction import definition_lock

    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(_edit_fields(name="reviewer", tags=["role"], categories=["role"]))
    account.set_agent_system_prompt(row.id, "Before.")
    child = None
    script = """
import sys
from pathlib import Path
from local_operator.agents import AgentRegistry
registry = AgentRegistry(Path(sys.argv[1]))
print('ready', flush=True)
registry.set_agent_system_prompt(sys.argv[2], 'After competing writer.')
print('written', flush=True)
"""
    try:
        with definition_lock(account.config_dir):
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(account.config_dir), row.id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ),
            )
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "ready"
            assert account.get_agent_system_prompt(row.id) == "Before."
            assert child.poll() is None
        output, errors = child.communicate(timeout=10)
        assert child.returncode == 0, errors
        assert output.strip() == "written"
        assert account.get_agent_system_prompt(row.id) == "After competing writer."
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def test_signed_transaction_reconcile_stale_and_duplicate(tmp_path):
    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        prepare,
        transact,
    )

    state = {"value": "before"}
    writes, backups = [], []

    def read():
        return state["value"]

    def write(value):
        writes.append(value)
        state["value"] = value

    kwargs: _TransactionKwargs = _TransactionKwargs(
        transaction_id="signed-test",
        before="before",
        after="after",
        read=read,
        write=write,
        backup=backups.append,
    )
    # PREPARING IS A SEPARATE ACT now: the grant writes the unconsumed record and
    # the apply only executes one that already exists. A missing journal is a
    # refusal, never a fresh authorisation.
    prepare(tmp_path, transaction_id="signed-test", before="before", after="after")
    result = transact(tmp_path, **kwargs)
    assert result["state"] == "complete"
    assert writes == ["after"] and backups == ["before"]
    # This second call used to answer ``reconciled``. That re-authorised an
    # operation which had already been consumed, which is the replay this
    # change closes, so it now refuses and touches nothing.
    with pytest.raises(DefinitionTransactionConflict, match="never replay"):
        transact(tmp_path, **kwargs)
    assert writes == ["after"] and backups == ["before"]
    state["value"] = "competing"
    with pytest.raises(DefinitionTransactionConflict):
        transact(tmp_path, **kwargs)
    assert state["value"] == "competing"


def test_reusable_profile_state_save_does_not_restore_stale_prompt(tmp_path):
    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(_edit_fields(name="reviewer", tags=["role"], categories=["role"]))
    account.set_agent_system_prompt(row.id, "Old guidance.")
    stale = account.load_agent_state(row.id)
    account.set_agent_system_prompt(row.id, "Approved new guidance.")
    account.save_agent_state(row.id, stale)
    assert account.get_agent_system_prompt(row.id) == "Approved new guidance."


def test_interrupted_after_image_finishes_bookkeeping_without_second_write(tmp_path):
    import pytest

    from local_operator.definition_transaction import prepare, transact

    state, writes = {"value": "before"}, []

    def write(value):
        writes.append(value)
        state["value"] = value
        raise KeyboardInterrupt("simulated process death after canonical write")

    args: _TransactionKwargs = _TransactionKwargs(
        transaction_id="restart-test",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
    prepare(tmp_path, transaction_id="restart-test", before="before", after="after")
    with pytest.raises(KeyboardInterrupt):
        transact(tmp_path, **args)
    result = transact(tmp_path, **args)
    assert result["state"] == "complete" and result["reconciled"]
    assert writes == ["after"]


def test_stale_cas_never_takes_backup_or_changes_definition(tmp_path):
    import pytest

    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(_edit_fields(name="reviewer", description="Before"))
    account.set_agent_system_prompt(row.id, "Before.")
    revision = definition_revision("agent", row.id, account.definition_revision_fields(row.id))
    AgentRegistry(account.config_dir).set_agent_system_prompt(row.id, "Concurrent edit.")
    backups = []
    with pytest.raises(DefinitionRevisionMismatch):
        write_profile(
            account,
            AgentParams(op="update", name="reviewer", instructions="Stale."),
            creating=False,
            scope="global",
            expected_definition_revision=revision,
            before_write=lambda: backups.append("snapshot"),
        )
    assert backups == []
    assert account.get_agent_system_prompt(row.id) == "Concurrent edit."
    assert not (account.config_dir / "profile-transactions").exists()


def test_team_failure_rolls_back_metadata_and_both_briefs(tmp_path, monkeypatch):
    import pytest

    from local_operator.teams import TeamEditFields, TeamRegistry

    registry = TeamRegistry(tmp_path / "account")
    row = registry.create_team(
        TeamEditFields(
            name="lopdev",
            manager="manager",
            description="Before",
            instructions="Before instructions.",
            project="Before project.",
        )
    )
    original = registry._save_team_locked
    calls = 0

    def fail_after_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            raise OSError("injected after team swap")
        return result

    monkeypatch.setattr(registry, "_save_team_locked", fail_after_write)
    with pytest.raises(OSError, match="injected"):
        registry.update_team(row.id, TeamEditFields(description="After", instructions="After."))
    current = TeamRegistry(registry.config_dir).get_team(row.id)
    assert current.description == "Before"
    assert current.instructions == "Before instructions."
    assert current.project == "Before project."


def test_completed_transaction_never_replays_after_later_restore(tmp_path):
    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        prepare,
        transact,
    )

    state, writes = {"value": "before"}, []

    def write(value):
        writes.append(value)
        state["value"] = value

    args: _TransactionKwargs = _TransactionKwargs(
        transaction_id="consumed-test",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
    prepare(tmp_path, transaction_id="consumed-test", before="before", after="after")
    transact(tmp_path, **args)
    state["value"] = "before"  # a later supported edit deliberately restores it
    with pytest.raises(DefinitionTransactionConflict, match="never replay"):
        transact(tmp_path, **args)
    assert writes == ["after"] and state["value"] == "before"


def test_a_missing_journal_is_never_a_fresh_authorisation(tmp_path):
    """The regression this hardening exists for: deleting the crash record.

    A same-uid shell can remove ``profile-transactions/<id>.json``. Before this
    rule the next apply read that absence as "nothing has run yet" and wrote the
    change again; now only a ``prepared`` record can be applied at all.
    """
    import json as jsonlib

    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        prepare,
        transact,
    )

    state = {"value": "before"}

    def write(value):
        state["value"] = value

    args: _TransactionKwargs = _TransactionKwargs(
        transaction_id="deleted-journal",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
    with pytest.raises(DefinitionTransactionConflict, match="no prepared transaction"):
        transact(tmp_path, **args)
    assert state["value"] == "before"
    prepare(tmp_path, transaction_id="deleted-journal", before="before", after="after")
    transact(tmp_path, **args)
    assert state["value"] == "after"
    state["value"] = "before"  # a later, legitimate edit restores the bytes
    journal = tmp_path / "profile-transactions" / "deleted-journal.json"
    assert jsonlib.loads(journal.read_text())["state"] == "complete"
    journal.unlink()
    # THIS process still holds the consumed identity, so it refuses first.
    with pytest.raises(DefinitionTransactionConflict, match="already applied in this process"):
        transact(tmp_path, **args)
    # A restart starts with an empty in-process set, so this is the rule that has
    # to hold on its own: the deleted record must not read as a fresh grant.
    from local_operator import definition_transaction

    definition_transaction._CONSUMED.clear()
    with pytest.raises(DefinitionTransactionConflict, match="no prepared transaction"):
        transact(tmp_path, **args)
    assert state["value"] == "before"


def test_in_process_consumption_blocks_an_edited_back_record(tmp_path):
    """One process applies an operation once, whatever its editable journal says."""
    import json as jsonlib

    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        prepare,
        transact,
    )

    state = {"value": "before"}

    def write(value):
        state["value"] = value

    args: _TransactionKwargs = _TransactionKwargs(
        transaction_id="consumed-in-process",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
    prepare(tmp_path, transaction_id="consumed-in-process", before="before", after="after")
    transact(tmp_path, **args)
    journal = tmp_path / "profile-transactions" / "consumed-in-process.json"
    record = jsonlib.loads(journal.read_text())
    record["state"] = "prepared"  # the same-uid rollback the advisory recorded
    journal.write_text(jsonlib.dumps(record))
    state["value"] = "before"
    with pytest.raises(DefinitionTransactionConflict, match="already applied in this process"):
        transact(tmp_path, **args)
    assert state["value"] == "before"


def test_a_named_edit_refuses_an_earlier_interrupted_attempt(tmp_path):
    """A save's own journal makes a dead attempt need a new decision, not a retry."""
    import json as jsonlib
    from contextlib import ExitStack

    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        atomic_edit,
        commitment,
    )

    state = {"value": "before"}
    directory = tmp_path / "profile-transactions"
    directory.mkdir(parents=True)
    (directory / "save-abc.json").write_text(
        jsonlib.dumps(
            {
                "schema": 1,
                "state": "prepared",
                "before": commitment("before"),
                "before_image": "before",
                "after": None,
            }
        )
    )
    with pytest.raises(DefinitionTransactionConflict, match="interrupted"):
        with ExitStack() as stack:
            stack.enter_context(
                atomic_edit(
                    tmp_path,
                    owner=object(),
                    read=lambda: state["value"],
                    restore=lambda image: state.update(value=image),
                    transaction_id="save-abc",
                )
            )
            state["value"] = "after"
    assert state["value"] == "before"


def test_the_interrupted_save_surface_asks_the_person_again(tmp_path):
    """The tool says "ask the person", never "retry" (the remedy is a new grant)."""
    from local_operator.definition_transaction import DefinitionTransactionConflict
    from local_operator.harness.types import ToolContext
    from local_operator.tools import agent_tool

    context = ToolContext(
        cwd=".",
        session_dir=str(tmp_path / "session-empty"),
        agent_registry=object(),
        authorize_durable_write=_granted,
    )

    def refuse(*args, **kwargs):
        raise DefinitionTransactionConflict(
            "an earlier attempt at this exact change was interrupted"
        )

    monkeypatch_target = agent_tool.write_profile
    agent_tool.write_profile = refuse
    try:
        result = asyncio.run(
            agent_tool._op_write(
                context,
                "tc",
                agent_tool.AgentParams(op="update", name="reviewer", instructions="After."),
                creating=False,
            )
        )
    finally:
        agent_tool.write_profile = monkeypatch_target
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "ask the user to ask for this change again" in text.lower()
    assert "retry" not in text.lower()
    # Journals live in the config root; this session has no backup folder, so the
    # aged-out case is named as such instead of reading as tampering.
    assert agent_tool._backups_present(context) is False


async def _granted(intent):  # pragma: no cover — the write never reaches the seam
    return "owner-prompt:surface-test"


def test_independent_http_edits_do_not_share_an_empty_grant_identity(tmp_path):
    """User editors have no tool grant string; their later edits are independent."""
    from local_operator.tools.agent_tool import AgentParams, write_profile

    account = AgentRegistry(tmp_path / "account")
    row = account.create_agent(_edit_fields(name="reviewer", tags=["role"], categories=["role"]))
    for description in ("First edit", "Second edit"):
        write_profile(
            account,
            AgentParams(op="update", name="reviewer", description=description),
            creating=False,
            scope="global",
        )
    assert account.get_agent(row.id).description == "Second edit"


def test_named_refusal_does_not_poison_the_next_independent_edit(tmp_path):
    import pytest

    from local_operator.definition_transaction import (
        DefinitionTransactionConflict,
        atomic_edit,
    )

    state = {"value": "before"}

    def edit(identity):
        return atomic_edit(
            tmp_path,
            owner=state,
            read=lambda: state["value"],
            restore=lambda value: state.update(value=value),
            transaction_id=identity,
        )

    with edit("used"):
        state["value"] = "first"
    with pytest.raises(DefinitionTransactionConflict):
        with edit("used"):
            raise AssertionError("must not run")
    with edit("new"):
        state["value"] = "second"
    assert state["value"] == "second"


@pytest.mark.parametrize("outcome", ["rollback", "rollback_failure", "postcommit"])
def test_non_one_shot_save_fault_never_completes_a_partial_image(tmp_path, outcome):
    """Ordinary signed saves must not fall through the Revert-only handler."""
    from local_operator.definition_transaction import prepare, transact

    state = {"value": "before"}
    calls = []

    def write(value):
        calls.append(value)
        if value == "after":
            state["value"] = "after" if outcome == "postcommit" else "partial"
            raise OSError("actual forward write fault")
        if outcome == "rollback_failure":
            raise OSError("actual rollback fault")
        state["value"] = value

    prepare(tmp_path, transaction_id="ordinary-fault", before="before", after="after")
    with pytest.raises(OSError):
        transact(
            tmp_path,
            transaction_id="ordinary-fault",
            before="before",
            after="after",
            read=lambda: state["value"],
            write=write,
        )
    journal = json.loads((tmp_path / "profile-transactions/ordinary-fault.json").read_text())
    assert journal["state"] != "complete"
    assert state["value"] == ("partial" if outcome == "rollback_failure" else "before")
    assert calls == ["after", "before"]
    if outcome != "rollback_failure":
        assert journal["state"] == "rolled_back"


def test_team_snapshot_never_recovers_or_replaces_hydrated_identity(tmp_path, monkeypatch):
    from local_operator.definition_transaction import definition_lock, team_image
    from local_operator.teams import TeamEditFields, TeamRegistry

    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(name="lane", manager="manager", instructions="persisted")
    )
    loaded = registry.get_team(team.id)
    loaded.instructions = "unsaved caller edit"
    hydration = set(registry._briefs_loaded)

    def forbidden(*args, **kwargs):
        pytest.fail("a definition image must not load/recover a registry")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_load", forbidden)
        patch.setattr(registry, "get_team", forbidden)
        with definition_lock(tmp_path):
            image = team_image(registry, team.id)
    assert image is not None and image["instructions"] == "persisted"
    assert registry._teams[team.id] is loaded
    assert loaded.instructions == "unsaved caller edit"
    assert registry._briefs_loaded == hydration
    # An intentional clear remains authoritative on the exact hydrated object.
    loaded.instructions = ""
    registry.save_team(loaded)
    assert TeamRegistry(tmp_path).get_team(team.id).instructions == ""


def test_unpublished_team_backup_is_unknown_and_primary_save_error_survives(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from local_operator.definition_transaction import definition_lock, team_image
    from local_operator.teams import (
        TeamEditFields,
        TeamRegistry,
        TeamRegistryRecoveryError,
    )

    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(name="lane", manager="manager", instructions="before")
    )
    loaded = registry.get_team(team.id)
    loaded.description, loaded.instructions = "new", "after"
    replace = os.replace

    def fail_publish(source, target):
        if Path(target).name == team.id and Path(source).name.startswith("."):
            raise OSError("injected publish and recovery fault")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_publish)
        with pytest.raises(Exception, match="Failed to save team metadata") as raised:
            registry.save_team(loaded)
        assert any("recovery required" in note for note in raised.value.__notes__)
        journals = [
            json.loads(path.read_text())
            for path in (tmp_path / "profile-transactions").glob("*.json")
        ]
        assert any(item["state"] == "prepared" for item in journals)
        before_paths = sorted(path.name for path in registry.teams_dir.iterdir())
        with definition_lock(tmp_path), pytest.raises(TeamRegistryRecoveryError):
            team_image(registry, team.id)
        assert sorted(path.name for path in registry.teams_dir.iterdir()) == before_paths
    # Ordinary access AFTER the fault ends owns recovery, not the snapshot.
    fresh = TeamRegistry(tmp_path).get_team(team.id)
    assert fresh.instructions == "before" and fresh.description == ""
    same = registry.get_team(team.id)
    assert same.instructions == "before"
    same.instructions = "retry"
    registry.save_team(same)
    assert TeamRegistry(tmp_path).get_team(team.id).instructions == "retry"
    assert not any(path.name.startswith(".") for path in registry.teams_dir.iterdir())


@pytest.mark.parametrize("operation", ["save", "update", "delete"])
def test_team_invalid_identity_precedes_both_lock_wrappers(tmp_path, operation):
    from local_operator.teams import Team, TeamEditFields, TeamRegistry

    directory = tmp_path / "untouched"
    registry = TeamRegistry(directory)
    hostile = Team.model_construct(id="../outside", name="hostile", manager="manager")
    with pytest.raises(ValueError):
        if operation == "save":
            registry.save_team(hostile)
        elif operation == "update":
            registry.update_team(hostile.id, TeamEditFields(description="no"))
        else:
            registry.delete_team(hostile.id)
    assert not directory.exists()


def test_definition_file_publish_cannot_replace_or_unlink_a_directory(tmp_path):
    from local_operator.definition_transaction import _durable_json, atomic_edit
    from local_operator.profile_scope import update_profile_backup

    directory = tmp_path / "protected"
    directory.mkdir()
    (directory / "sentinel").write_text("keep")
    with pytest.raises(OSError):
        _durable_json(directory, {"value": "no"})
    session = tmp_path / "session"
    target = session / "profile-backups" / "bound.json"
    target.mkdir(parents=True)
    # Best-effort backup lookup refuses an unreadable directory before publish.
    update_profile_backup(str(session), "bound", saved_revision="no")
    journal = tmp_path / "profile-transactions" / "directory.json"
    journal.mkdir(parents=True)
    with pytest.raises(OSError):
        with atomic_edit(
            tmp_path,
            owner=object(),
            read=lambda: "before",
            restore=lambda value: None,
            transaction_id="directory",
        ):
            pytest.fail("a directory cannot be a prepared journal")
    assert (directory / "sentinel").read_text() == "keep"
    assert target.is_dir() and journal.is_dir()

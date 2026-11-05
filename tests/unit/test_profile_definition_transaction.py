"""Regression: an authored save must not overwrite a competing definition write.

The durable profile approval executor needs a single cross-process definition
transaction, not a CAS check around metadata followed by an independent prompt
write. This same-process/two-registry interleaving deterministically exercises
that window without relying on timing or real operator stores.
"""

from __future__ import annotations

from local_operator.agents import AgentRegistry
from local_operator.hub_sync.definition import (
    DefinitionRevisionMismatch,
    definition_revision,
)
from local_operator.profile_scope import scope_agent_registry
from local_operator.tools.agent_tool import AgentParams, write_profile
from tests.unit.tools.test_agent_tool import _edit_fields


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
        transact,
    )

    state = {"value": "before"}
    writes, backups = [], []

    def read():
        return state["value"]

    def write(value):
        writes.append(value)
        state["value"] = value

    kwargs = dict(
        transaction_id="signed-test",
        before="before",
        after="after",
        read=read,
        write=write,
        backup=backups.append,
    )
    result = transact(tmp_path, **kwargs)
    assert result["state"] == "complete"
    assert writes == ["after"] and backups == ["before"]
    result = transact(tmp_path, **kwargs)
    assert result["reconciled"] and writes == ["after"] and backups == ["before"]
    state["value"] = "competing"
    with pytest.raises(DefinitionTransactionConflict, match="changed"):
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

    from local_operator.definition_transaction import transact

    state, writes = {"value": "before"}, []

    def write(value):
        writes.append(value)
        state["value"] = value
        raise KeyboardInterrupt("simulated process death after canonical write")

    args = dict(
        transaction_id="restart-test",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
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
        transact,
    )

    state, writes = {"value": "before"}, []

    def write(value):
        writes.append(value)
        state["value"] = value

    args = dict(
        transaction_id="consumed-test",
        before="before",
        after="after",
        read=lambda: state["value"],
        write=write,
    )
    transact(tmp_path, **args)
    state["value"] = "before"  # a later supported edit deliberately restores it
    with pytest.raises(DefinitionTransactionConflict, match="never replay"):
        transact(tmp_path, **args)
    assert writes == ["after"] and state["value"] == "before"

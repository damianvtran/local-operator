"""Skill proposals: the ``skill`` tool decides exactly like ``agent``/``team``.

Operator ruling (2026-11): "agents MUST be able to make adjustments to
agents/teams on their own"; changes arriving from untrusted boundaries (Team
Bridge, peer agents) go through proposals reviewed in Agents → <name>, and
deletes keep their confirmation. These tests pin the skill half of that rule:

* a clean local run creates/updates a skill ITSELF (backup in hub/backups);
* a run that read untrusted input files a ``kind="skill"`` proposal with a
  per-file before/after and writes nothing;
* ``profile_proposal op=apply`` applies it from a clean run, needs the card
  from a tainted one, and refuses a stale base;
* ``remove_paths`` (a delete) keeps the account card.

Real ``Session`` + real tools + real skill storage in a synthetic config root;
no live session, profile, provider or signer is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import profile_proposals as P
from local_operator.harness.types import Message
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.server.utils import desktop_skill_writes as writes
from local_operator.tools.profile_proposal_tool import execute_profile_proposal
from local_operator.tools.skill_tool import execute_skill
from tests.unit.session.test_account_profile_consent import Gate, _session
from tests.unit.session.test_account_profile_consent import (
    configured_tiers as profile_tiers,
)
from tests.unit.session.test_account_profile_consent import stores as profile_stores
from tests.unit.session.test_profile_proposal_consent import PEER, _ctx, start

stores = profile_stores
configured_tiers = profile_tiers
pytestmark = pytest.mark.asyncio

BEFORE = "---\nname: marketing\ndescription: Marketing checklist.\n---\n# Marketing\nPlan.\n"
AFTER = BEFORE + "Measure every launch.\n"


def _root(stores) -> Path:
    return Path(stores["account"].config_dir)


def _seed(stores) -> str:
    """A managed skill on disk; returns its fingerprint (what ``show`` reports)."""
    writes.create(_root(stores), "marketing", files=[{"path": "SKILL.md", "content": BEFORE}])
    return _fingerprint(stores)


def _fingerprint(stores) -> str:
    return SkillDocument.from_directory(_root(stores) / "skills" / "marketing").fingerprint


def _disk(stores) -> str:
    return (_root(stores) / "skills" / "marketing" / "SKILL.md").read_text()


def _backups(stores) -> list[str]:
    directory = _root(stores) / "hub" / "backups"
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


def _ctx_skill(session, stores):
    ctx = _ctx(session, stores)
    ctx.memory_config_dir = str(_root(stores))
    return ctx


async def _update(session, stores, fingerprint: str, content: str = AFTER, **extra):
    args: dict[str, Any] = {
        "op": "update",
        "name": "marketing",
        "files": [{"path": "SKILL.md", "content": content}],
        "expected_fingerprint": fingerprint,
        **extra,
    }
    return await execute_skill("s1", args, None, None, _ctx_skill(session, stores))


def _said(result) -> str:
    return "".join(getattr(part, "text", "") for part in result.content)


def _details(result) -> dict[str, Any]:
    return dict(result.details or {})


def _audit(stores) -> list[dict[str, Any]]:
    path = P.proposals_dir(_root(stores)) / P.AUDIT_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_clean_local_skill_update_applies_directly_with_backup(stores):
    """Owner/clean run: written itself, previous bytes kept, no proposal filed."""
    fingerprint = _seed(stores)
    session = _session(stores, gate=Gate(answer=False))
    try:
        start(session, Message.user("Tidy up my skills."))
        result = await _update(session, stores, fingerprint)
        assert not result.is_error, result.content
        assert _details(result).get("status") != "proposed"
        assert "updated skill marketing." in _said(result)
        assert _disk(stores) == AFTER
        assert any(name.startswith("skill-") for name in _backups(stores))
        assert P.list_proposals(_root(stores)) == []
    finally:
        await session.dispose()


async def test_tainted_skill_update_files_proposal_and_writes_nothing(stores):
    fingerprint = _seed(stores)
    gate = Gate()
    session = _session(stores, gate=gate)
    try:
        start(session, Message.user("Update skill marketing."), PEER)
        result = await _update(session, stores, fingerprint)
        assert not result.is_error, result.content
        details = _details(result)
        assert details["status"] == "proposed", result.content
        assert details["profile_kind"] == "skill"
        assert "review it in Agents → marketing" in _said(result)
        assert "Mesh" not in _said(result)
        assert gate.cards == []
        assert _disk(stores) == BEFORE
        assert _backups(stores) == []
        record = P.get(_root(stores), details["proposal_id"])
        assert record is not None
        assert record["kind"] == "skill" and record["origin"]["trust"] == "untrusted"
        assert record["fields"] == ["SKILL.md"]
        detail = P.detail(_root(stores), record)
        assert detail["changes"] == [
            {
                "field": "SKILL.md",
                "label": "SKILL.md",
                "before_text": BEFORE,
                "after_text": AFTER,
                "format": "text",
                "after": AFTER,
            }
        ]
    finally:
        await session.dispose()


async def test_tainted_skill_edit_op_files_proposal(stores):
    """``op=edit`` (anchored replacements) is an update too: proposed, not written."""
    fingerprint = _seed(stores)
    session = _session(stores, gate=Gate())
    try:
        start(session, Message.user("Edit skill marketing."), PEER)
        result = await execute_skill(
            "s1",
            {
                "op": "edit",
                "name": "marketing",
                "edits": [{"path": "SKILL.md", "old_text": "Plan.", "new_text": "Plan well."}],
                "expected_fingerprint": fingerprint,
            },
            None,
            None,
            _ctx_skill(session, stores),
        )
        assert _details(result)["status"] == "proposed", result.content
        record = P.get(_root(stores), _details(result)["proposal_id"])
        assert record is not None
        assert record["after"]["files"]["SKILL.md"] == BEFORE.replace("Plan.", "Plan well.")
        assert _disk(stores) == BEFORE
    finally:
        await session.dispose()


async def test_tainted_skill_create_files_create_proposal(stores):
    session = _session(stores, gate=Gate())
    try:
        start(session, Message.user("Make a skill."), PEER)
        result = await execute_skill(
            "s1",
            {
                "op": "create",
                "name": "marketing",
                "files": [{"path": "SKILL.md", "content": BEFORE}],
            },
            None,
            None,
            _ctx_skill(session, stores),
        )
        assert _details(result)["status"] == "proposed", result.content
        record = P.get(_root(stores), _details(result)["proposal_id"])
        assert record is not None and record["operation"] == "create"
        assert record["before"] is None
        assert not (_root(stores) / "skills" / "marketing").exists()
        # A clean run's bound yes applies the create.
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "p1", {"op": "apply", "id": record["id"]}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert _disk(stores) == BEFORE
    finally:
        await session.dispose()


async def _propose(session, stores) -> str:
    fingerprint = _seed(stores)
    start(session, Message.user("Update skill marketing."), PEER)
    result = await _update(session, stores, fingerprint)
    assert _details(result)["status"] == "proposed", result.content
    return _details(result)["proposal_id"]


async def test_verbal_apply_from_clean_run_applies_with_backup_and_audit(stores):
    gate = Gate()
    session = _session(stores, gate=gate)
    try:
        pid = await _propose(session, stores)
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "p1", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert gate.cards == []
        assert _disk(stores) == AFTER
        assert any(name.startswith("skill-") for name in _backups(stores))
        assert "hub/backups" in _said(applied)
        record = P.get(_root(stores), pid)
        assert record is not None and record["state"] == "accepted"
        assert [row["action"] for row in _audit(stores)] == ["filed", "accepted"]
    finally:
        await session.dispose()


async def test_apply_from_tainted_run_requires_the_card(stores):
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        pid = await _propose(session, stores)
        refused = await execute_profile_proposal(
            "p1", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert refused.is_error, refused.content
        assert len(gate.cards) == 1
        assert gate.cards[0][1].startswith(f"Apply proposal {pid} to skill 'marketing'?\n")
        assert "Review changes in Agents → marketing." in gate.cards[0][1]
        assert _disk(stores) == BEFORE
        gate.answer = True
        applied = await execute_profile_proposal(
            "p2", {"op": "apply", "id": pid}, None, None, _ctx(session, stores)
        )
        assert not applied.is_error, applied.content
        assert len(gate.cards) == 2
        assert _disk(stores) == AFTER
    finally:
        await session.dispose()


async def test_stale_skill_proposal_is_refused(stores):
    session = _session(stores, gate=Gate())
    try:
        pid = await _propose(session, stores)
        writes.update(
            _root(stores),
            "marketing",
            [{"path": "SKILL.md", "content": BEFORE + "Concurrent.\n"}],
            _fingerprint(stores),
        )
        record = P.get(_root(stores), pid)
        assert record is not None and record["stale"] is True
        with pytest.raises(P.ProposalError) as raised:
            P.apply(_root(stores), pid, actor="test")
        assert raised.value.code == "profile_proposal_stale"
        assert _disk(stores) == BEFORE + "Concurrent.\n"
    finally:
        await session.dispose()


async def test_remove_paths_still_needs_the_card(stores):
    writes.create(
        _root(stores),
        "marketing",
        files=[
            {"path": "SKILL.md", "content": BEFORE},
            {"path": "references/a.md", "content": "a\n"},
        ],
    )
    gate = Gate(answer=False)
    session = _session(stores, gate=gate)
    try:
        start(session, Message.user("Tidy up my skills."))
        result = await execute_skill(
            "s1",
            {
                "op": "update",
                "name": "marketing",
                "remove_paths": ["references/a.md"],
                "expected_fingerprint": _fingerprint(stores),
            },
            None,
            None,
            _ctx_skill(session, stores),
        )
        assert result.is_error, result.content
        assert len(gate.cards) == 1 and gate.cards[0][1].startswith("Save skill 'marketing'")
        assert (_root(stores) / "skills" / "marketing" / "references" / "a.md").is_file()
        assert P.list_proposals(_root(stores)) == []
    finally:
        await session.dispose()

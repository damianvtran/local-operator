"""Agents edit their own agents/teams; edits of long briefs are lossless.

The operator's ruling (2026-11), verbatim: "agents are able to apply changes
to agents and teams themselves... agents MUST be able to make adjustments to
agents/teams on their own." And the incident behind the lossless half: a
12,582-char / 65-line team brief came back from ``team op=show`` as a clipped
head+tail preview, an agent appended one paragraph to that preview and saved it
as the whole field, and ~6k chars were deleted while the receipt said +0/-4.

Every test drives a REAL ``Session`` (its input classification and consent
seam) against the real ``agent``/``team``/``profile_proposal`` tools and real
registries in a synthetic, isolated config root. No live session, profile,
proposal, provider or signer is touched. The brief is SYNTHETIC: same size and
line count as the incident, none of its content.

Independent QA recipe: ``pytest -n0 tests/unit/session/test_profile_autonomy.py``
in an isolated ``HOME``/``LOCAL_OPERATOR_CONFIG_DIR``; each test's docstring
names the row it proves.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator import profile_proposals as P
from local_operator.harness.types import CustomMessage, Message
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.profile_scope import team_revision
from local_operator.profile_text_edits import field_line_stats
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.builtin import spill_truncate
from local_operator.tools.profile_proposal_tool import execute_profile_proposal
from local_operator.tools.team_tool import execute_team
from tests.unit.session.test_account_profile_consent import (
    MODEL,
    Gate,
    _never_stream,
    _text,
)
from tests.unit.session.test_account_profile_consent import (
    configured_tiers as profile_tiers,
)
from tests.unit.session.test_account_profile_consent import stores as profile_stores
from tests.unit.session.test_profile_proposal_consent import PEER, _ctx, start

stores = profile_stores
configured_tiers = profile_tiers
pytestmark = pytest.mark.asyncio

WAKE = CustomMessage(custom_type=WAKE_PROMPT_MESSAGE_TYPE, attribution="user", details={})


def _brief(chars: int = 12_582, lines: int = 65) -> str:
    """A synthetic brief of EXACTLY ``chars`` characters over ``lines`` lines."""
    rows = [f"- rule {index:02d}: synthetic collaboration guidance" for index in range(lines)]
    body = "\n".join(rows)
    pad = chars - len(body)
    assert pad > 0
    rows[lines // 2] += " " + "m" * (pad - 1)  # the MIDDLE is what a preview drops
    text = "\n".join(rows)
    assert len(text) == chars and len(text.splitlines()) == lines
    return text


@pytest.fixture()
def local(stores) -> dict[str, Any]:
    """The shared synthetic account (alias kept for the row names below)."""
    return stores


def _session(stores, *, gate=None, parent="", spawn_untrusted=False, consent=True) -> Session:
    return Session(
        model=MODEL,
        stream_fn=_never_stream,
        tools=[],
        transcript=Transcript(stores["session_dir"]),
        system_blocks_provider=lambda: ["stable"],
        agent_registry=stores["agents"],
        team_registry=stores["teams"],
        request_approval=gate,
        owner_prompt_consent=consent,
        parent_session_id=parent,
        spawn_untrusted=spawn_untrusted,
    )


async def _team(session, stores, **args):
    return await execute_team("t", args, None, None, _ctx(session, stores))


async def _agent(session, stores, **args):
    return await execute_agent("a", args, None, None, _ctx(session, stores))


def _big_team(stores, text: str) -> None:
    team = stores["teams"].get_team_by_name("lopdev")
    stores["teams"].update_team(team.id, TeamEditFields(instructions=text))
    stores["teams"].refresh()


def _stored(stores) -> str:
    stores["teams"].refresh()
    return stores["teams"].get_team_by_name("lopdev").instructions


def _revision(stores) -> str:
    stores["teams"].refresh()
    return team_revision(stores["teams"].get_team_by_name("lopdev"))


# -- lossless show / edits ---------------------------------------------------


async def test_show_returns_the_whole_12582_char_65_line_brief(local) -> None:
    """Row 1: show is lossless (no spill, no marker) and prints the revision."""
    brief = _brief()
    _big_team(local, brief)
    session = _session(local)
    try:
        start(session, WAKE)
        shown = await _team(session, local, op="show", name="lopdev")
        body = _text(shown)
        assert brief in body
        assert "spill" not in (shown.details or {})
        assert "output truncated" not in body and "SAVED at" not in body
        assert f"revision: {_revision(local)}" in body
        assert len(body) > 8 * 1024  # past the generic preview limit
    finally:
        await session.dispose()


async def test_show_then_append_preserves_every_existing_byte(local) -> None:
    """Row 2: an ``edits`` append keeps all 12,582 old bytes; autonomous, no card."""
    brief = _brief()
    _big_team(local, brief)
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, WAKE)
        result = await _team(
            session,
            local,
            op="update",
            name="lopdev",
            edits=[{"field": "instructions", "new": "- rule 65: appended"}],
        )
        assert not result.is_error, _text(result)
        after = _stored(local)
        assert after == brief + "\n- rule 65: appended"
        assert gate.cards == []
        assert result.details["authorized_by"].startswith("local-agent:")
        # One canonical count: the receipt says exactly what the values say.
        assert field_line_stats(brief, after) == (1, 0)
        assert "instructions: +1/-0 lines" in _text(result)
    finally:
        await session.dispose()


async def test_old_preview_plus_append_is_refused_even_when_acknowledged(local) -> None:
    """Row 3: the incident replay. The pre-fix show's preview (built by the
    real ``spill_truncate``) plus an append is refused with and without the
    ack; nothing is written."""
    brief = _brief()
    _big_team(local, brief)
    preview, _ = spill_truncate(brief, "team", None)
    assert len(preview) < len(brief)
    session = _session(local)
    try:
        start(session, WAKE)
        for extra in ({}, {"replace": True, "base_revision": _revision(local)}):
            refused = await _team(
                session,
                local,
                op="update",
                name="lopdev",
                instructions=preview + "\n- appended paragraph",
                **extra,
            )
            assert refused.is_error and "truncation/spill marker" in _text(refused)
        assert _stored(local) == brief
    finally:
        await session.dispose()


async def test_unacknowledged_loss_is_refused_with_derived_counts(local) -> None:
    """Row 5: a preview with its marker stripped is still a loss; refused
    without the ack, and the refusal's counts come from the real values."""
    brief = _brief()
    _big_team(local, brief)
    kept = brief.splitlines()[:20] + brief.splitlines()[45:]
    candidate = "\n".join(kept) + "\n- appended paragraph"
    added, removed = field_line_stats(brief, candidate)
    session = _session(local)
    try:
        start(session, WAKE)
        refused = await _team(session, local, op="update", name="lopdev", instructions=candidate)
        assert refused.is_error
        body = _text(refused)
        assert f"remove {removed} of 65 stored lines (+{added}/-{removed})" in body
        assert f"{len(brief) - len(candidate)} chars shorter" in body
        # The refusal never echoes the revision (a non-reader could copy it).
        assert _revision(local) not in body
        assert _stored(local) == brief
    finally:
        await session.dispose()


async def test_intentional_shortening_needs_only_the_agents_ack_and_reverts(local) -> None:
    """Row 4: replace=true + show's revision shortens with NO card; the backup
    restores the exact prior bytes; receipt counts equal the value counts."""
    brief = _brief()
    _big_team(local, brief)
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, WAKE)
        shown = await _team(session, local, op="show", name="lopdev")
        revision = next(
            line.split(": ", 1)[1]
            for line in _text(shown).splitlines()
            if line.startswith("revision: ")
        )
        short = "- one rule now"
        saved = await _team(
            session,
            local,
            op="update",
            name="lopdev",
            instructions=short,
            replace=True,
            base_revision=revision,
        )
        assert not saved.is_error, _text(saved)
        assert gate.cards == [] and _stored(local) == short
        added, removed = field_line_stats(brief, short)
        assert f"instructions: +{added}/-{removed} lines" in _text(saved)
        backup = saved.details["backup_id"]
        assert backup
        reverted = await _team(session, local, op="revert", backup_id=backup)
        assert not reverted.is_error, _text(reverted)
        assert _stored(local) == brief
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "edits,needle",
    [
        ([{"field": "instructions", "old": "no such text", "new": "x"}], "not found"),
        ([{"field": "instructions", "old": "synthetic collaboration", "new": "x"}], "found 65"),
        (
            [
                {"field": "instructions", "old": "rule 00:", "new": "RULE 00:"},
                {"field": "instructions", "old": "absent", "new": "x"},
            ],
            "edits[2]: old text not found",
        ),
        ([{"field": "label", "old": None, "new": "x"}], "cannot be patched"),
    ],
)
async def test_patch_misses_are_atomic_and_write_nothing(local, edits, needle) -> None:
    """Row 6: 0 matches, 2+ matches, a later failing edit, a non-text field."""
    brief = _brief()
    _big_team(local, brief)
    session = _session(local)
    try:
        start(session, WAKE)
        refused = await _team(session, local, op="update", name="lopdev", edits=edits)
        assert refused.is_error and needle in _text(refused), _text(refused)
        assert _stored(local) == brief
    finally:
        await session.dispose()


async def test_exact_patch_changes_only_the_named_text(local) -> None:
    brief = _brief()
    _big_team(local, brief)
    session = _session(local)
    try:
        start(session, WAKE)
        ok = await _team(
            session,
            local,
            op="update",
            name="lopdev",
            edits=[{"field": "instructions", "old": "rule 07:", "new": "rule 7b:"}],
        )
        assert not ok.is_error, _text(ok)
        assert _stored(local) == brief.replace("rule 07:", "rule 7b:", 1)
    finally:
        await session.dispose()


async def test_stale_base_revision_refuses_a_replayed_edit(local) -> None:
    """Row 7: a base_revision from before another write refuses (CAS)."""
    session = _session(local)
    try:
        start(session, WAKE)
        old = _revision(local)
        first = await _team(session, local, op="update", name="lopdev", description="moved")
        assert not first.is_error, _text(first)
        replay = await _team(
            session,
            local,
            op="update",
            name="lopdev",
            base_revision=old,
            edits=[{"field": "instructions", "new": "late"}],
        )
        assert replay.is_error and "does not match" in _text(replay)
        assert _stored(local) == "Ship."
    finally:
        await session.dispose()


async def test_agent_show_is_lossless_and_append_preserves_bytes(local) -> None:
    """The agent twin of rows 1-2 (instructions cap 10,000)."""
    brief = _brief(9_500, 40)
    account = local["account"]
    row = account.get_agent_by_name("reviewer")
    account.set_agent_system_prompt(row.id, brief)
    session = _session(local)
    try:
        start(session, WAKE)
        shown = await _agent(session, local, op="show", name="reviewer")
        assert brief in _text(shown) and "revision: " in _text(shown)
        assert "spill" not in (shown.details or {})
        saved = await _agent(
            session,
            local,
            op="update",
            name="reviewer",
            edits=[{"field": "instructions", "new": "- appended"}],
        )
        assert not saved.is_error, _text(saved)
        assert account.get_agent_system_prompt(row.id) == brief + "\n- appended"
        refused = await _agent(
            session, local, op="update", name="reviewer", instructions="- only this"
        )
        # 40 stored lines plus the append.
        assert refused.is_error and "remove 41 of 41 stored lines" in _text(refused)
    finally:
        await session.dispose()


# -- autonomy ----------------------------------------------------------------


async def test_clean_wake_creates_and_updates_agents_and_teams_with_zero_cards(local) -> None:
    """Row 8: no owner text at all (a wake), every field class, no card."""
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, WAKE)
        made = await _team(
            session,
            local,
            op="create",
            name="ops",
            scope="global",
            manager="reviewer",
            members=["reviewer"],
            instructions="Run ops.",
        )
        assert not made.is_error, _text(made)
        updated = await _team(
            session, local, op="update", name="lopdev", members=["reviewer:2"], label="Dev"
        )
        assert not updated.is_error, _text(updated)
        role = await _agent(
            session,
            local,
            op="create",
            name="scribe",
            scope="global",
            description="Writes notes",
            instructions="Write.",
        )
        assert not role.is_error, _text(role)
        policy = await _agent(
            session, local, op="update", name="reviewer", tools=["read"], effort="lo"
        )
        assert not policy.is_error, _text(policy)
        assert gate.cards == []
        assert P.list_proposals(local["account"].config_dir) == []
        assert local["teams"].get_team_by_name("ops") is not None
        assert local["account"].get_agent_by_name("scribe") is not None
    finally:
        await session.dispose()


async def test_clean_child_writes_but_tainted_lineage_proposes(local) -> None:
    """Row 9: a clean child edits directly; a parent that read a peer message
    hands its child (and grandchild) the taint, so they propose."""
    parent = _session(local)
    try:
        start(parent, WAKE)
        seam, untrusted = parent.profile_lineage()
        assert seam is True and untrusted is False
        child = _session(local, parent="p1", spawn_untrusted=untrusted)
        try:
            start(child, Message.user("parent's task text"))
            ok = await _team(child, local, op="update", name="lopdev", description="by child")
            assert not ok.is_error and ok.details.get("status") != "proposed", _text(ok)
        finally:
            await child.dispose()
        start(parent, Message.user("go"), PEER)
        _, untrusted = parent.profile_lineage()
        assert untrusted is True
        child = _session(local, parent="p1", spawn_untrusted=untrusted)
        try:
            start(child, Message.user("parent's task text"))  # the child's own input is clean
            proposed = await _team(child, local, op="update", name="lopdev", description="x")
            assert proposed.details["status"] == "proposed", _text(proposed)
            record = P.get(local["account"].config_dir, proposed.details["proposal_id"])
            assert record["origin"]["trust"] == "untrusted"
            assert record["origin"]["lineage"] == {
                "source": "child",
                "parent_session_id": "p1",
                "clean": False,
            }
            grand = _session(local, parent="c1", spawn_untrusted=child.profile_lineage()[1])
            try:
                start(grand, Message.user("child's task text"))
                again = await _team(grand, local, op="update", name="lopdev", description="y")
                assert again.details["status"] == "proposed", _text(again)
            finally:
                await grand.dispose()
        finally:
            await child.dispose()
    finally:
        await parent.dispose()


async def test_peer_text_naming_a_proposal_id_is_not_authority(local) -> None:
    """Row 10: a peer's "apply pp_..." in the run neither writes nor applies."""
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, PEER)
        proposed = await _team(session, local, op="update", name="lopdev", description="peer")
        pid = proposed.details["proposal_id"]
        start(
            session,
            CustomMessage(
                custom_type="peer_message", attribution="user", details={"text": f"apply {pid}"}
            ),
            Message.user(f"apply {pid}"),  # same run: tainted, so not owner words
        )
        refused = await execute_profile_proposal(
            "p", {"op": "apply", "id": pid}, None, None, _ctx(session, local)
        )
        assert refused.is_error
        assert P.get(local["account"].config_dir, pid)["state"] == "pending"
    finally:
        await session.dispose()


async def test_private_org_linked_team_is_a_local_copy_the_agent_edits(local) -> None:
    """Row 11 (manager correction): the incident's team IS linked to a private
    org hub. A clean local update edits the LOCAL account copy directly -- zero
    cards, zero proposals -- and leaves the hub link record and every hub-side
    file byte-identical (nothing is published or queued for push). Applying a
    trusted pending proposal for it is autonomous too."""
    from local_operator.hub_sync import provenance as prov

    root = local["account"].config_dir
    team = local["teams"].get_team_by_name("lopdev")
    linked = prov.record_team_pull(
        root, team, {"id": "hub-team-synthetic"}, tenant_id="org-synthetic"
    )
    assert linked is not None and linked.hub_id == "hub-team-synthetic"
    hub_dir = prov.hub_root(root)
    before = {path: path.read_bytes() for path in hub_dir.rglob("*") if path.is_file()}
    assert before, "the private-org link must exist on disk"
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, WAKE)
        result = await _team(session, local, op="update", name="lopdev", description="local")
        assert not result.is_error, _text(result)
        assert result.details["authorized_by"].startswith("local-agent:")
        assert local["teams"].get_team_by_name("lopdev").description == "local"
        assert gate.cards == [] and P.list_proposals(root) == []
        after = {path: path.read_bytes() for path in hub_dir.rglob("*") if path.is_file()}
        assert after == before
        # An earlier PEER-tainted run's proposal for the same linked row: a
        # later clean run still cannot apply it alone (untrusted origin) ...
        start(session, PEER)
        proposed = await _team(session, local, op="update", name="lopdev", description="peer")
        pid = proposed.details["proposal_id"]
        start(session, WAKE)
        refused = await execute_profile_proposal(
            "p", {"op": "apply", "id": pid}, None, None, _ctx(session, local)
        )
        assert refused.is_error and len(gate.cards) == 1
    finally:
        await session.dispose()


async def test_trusted_pending_proposal_applies_autonomously_and_legacy_needs_same_session(
    local,
) -> None:
    """Row 12: a clean, trusted proposal applies with no card (audit actor
    ``agent:``); a LEGACY record (no lineage) applies autonomously only from
    the session that filed it."""
    root = local["account"].config_dir
    team = local["teams"].get_team_by_name("lopdev")
    before = {
        key: team.model_dump(mode="json")[key]
        for key in (
            "id",
            "name",
            "label",
            "description",
            "manager",
            "members",
            "aliases",
            "instructions",
            "project",
        )
    }
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, WAKE)
        own = P.file(
            root,
            profile_kind="team",
            operation="update",
            target_id=before["id"],
            name="lopdev",
            before=before,
            after={**before, "description": "legacy own"},
            session_id=session._session_id,
            trust="agent",
        )
        applied = await execute_profile_proposal(
            "p", {"op": "apply", "id": own["id"]}, None, None, _ctx(session, local)
        )
        assert not applied.is_error, _text(applied)
        assert gate.cards == []
        audit = root / "profile-proposals" / "audit.jsonl"
        if audit.exists():
            assert f"agent:{session._session_id}" in audit.read_text()
        local["teams"].refresh()
        current = local["teams"].get_team_by_name("lopdev")
        now = {key: current.model_dump(mode="json")[key] for key in before}
        other = P.file(
            root,
            profile_kind="team",
            operation="update",
            target_id=before["id"],
            name="lopdev",
            before=now,
            after={**now, "description": "legacy other"},
            session_id="some-other-session",
            trust="agent",
        )
        refused = await execute_profile_proposal(
            "p2", {"op": "apply", "id": other["id"]}, None, None, _ctx(session, local)
        )
        assert refused.is_error and len(gate.cards) == 1
    finally:
        await session.dispose()


async def test_unknown_seam_fails_closed(local) -> None:
    """Row 13: a host without the session-owned seam (SDK/embedder) writes
    nothing, whatever the run looks like."""
    sdk = _session(local, consent=False)
    try:
        start(sdk, WAKE)
        refused = await _team(sdk, local, op="update", name="lopdev", description="sdk")
        assert refused.is_error
        assert local["teams"].get_team_by_name("lopdev").description == "Dev team"
    finally:
        await sdk.dispose()


async def test_receipt_and_proposal_counts_come_from_one_computation(local) -> None:
    """The +0/-4 vs +4/-34 disagreement: a peer-proposed rewrite whose new text
    is all markdown bullets reports the same counts in the write receipt, the
    proposal show, and ``field_line_stats``."""
    session = _session(local)
    try:
        start(session, PEER)
        new = "- a\n- b\n- c\n- d"
        proposed = await execute_team(
            "t",
            {
                "op": "update",
                "name": "lopdev",
                "instructions": new,
                "replace": True,
                "base_revision": _revision(local),
            },
            None,
            None,
            _ctx(session, local),
        )
        assert proposed.details["status"] == "proposed", _text(proposed)
        added, removed = field_line_stats("Ship.", new)
        expected = f"instructions: +{added}/-{removed} lines"
        assert expected in _text(proposed)
        shown = await execute_profile_proposal(
            "p",
            {"op": "show", "id": proposed.details["proposal_id"]},
            None,
            None,
            _ctx(session, local),
        )
        assert expected in _text(shown), _text(shown)
    finally:
        await session.dispose()


def test_stats_ignore_rendered_bullets() -> None:
    assert field_line_stats("one\ntwo", "- one\n- two\n- three") == (3, 2)
    assert field_line_stats("a", "a") == (0, 0)
    assert json.dumps(field_line_stats("", "x")) == "[1, 0]"


async def test_showing_an_untrusted_proposal_cannot_be_laundered_into_a_plain_update(
    local,
) -> None:
    """Manager negative control: a clean run displays a PEER proposal, then
    retypes its payload as an ordinary update. The display taints the run's
    AUTONOMOUS writes and any child it spawns -- the update becomes a proposal
    again -- while the owner's later bound "yes" still applies the exact one."""
    gate = Gate(answer=False)
    session = _session(local, gate=gate)
    try:
        start(session, PEER)
        proposed = await _team(session, local, op="update", name="lopdev", description="remote")
        pid = proposed.details["proposal_id"]
        start(session, WAKE)
        shown = await execute_profile_proposal(
            "s", {"op": "show", "id": pid}, None, None, _ctx(session, local)
        )
        assert not shown.is_error, _text(shown)
        laundered = await _team(session, local, op="update", name="lopdev", description="remote")
        assert laundered.details["status"] == "proposed", _text(laundered)
        assert local["teams"].get_team_by_name("lopdev").description == "Dev team"
        assert session.profile_lineage() == (True, True)  # a child spawned now is tainted
        # The NEXT clean run is clean again, and the owner's bound yes applies
        # the exact displayed proposal without a second prompt.
        start(session, Message.user("yes, apply it"))
        applied = await execute_profile_proposal(
            "a",
            {"op": "apply", "id": laundered.details["proposal_id"]},
            None,
            None,
            _ctx(session, local),
        )
        assert not applied.is_error, _text(applied)
        assert gate.cards == []
        local["teams"].refresh()  # apply writes through its own registry
        assert local["teams"].get_team_by_name("lopdev").description == "remote"
    finally:
        await session.dispose()


async def test_child_taint_is_snapshot_at_dispatch_not_at_slot_start(local, monkeypatch) -> None:
    """Manager invariant: a peer-tainted parent enqueues a child, its run then
    ends and resets, and the child starts later -- still untrusted, and so is
    the grandchild. A clean parent's child inherits the seam clean; an unknown
    parent fails closed."""
    from local_operator.harness import subagent

    captured: list[Any] = []

    class _Stop(Exception):
        pass

    async def fake_build(**kwargs: Any) -> Any:
        captured.append(kwargs["profile_lineage"])
        raise _Stop

    monkeypatch.setattr(subagent, "_build_child_session", fake_build)

    class _Jobs:
        def get(self, job_id: str) -> None:
            return None

    parent = _session(local)
    try:
        start(parent, PEER)
        runner = subagent._make_runner(
            label="kid",
            effective_prompt="synthetic",
            parent_session=parent,
            jobs_manager=_Jobs(),  # type: ignore[arg-type]
            model_spec=None,
        )
        start(parent, WAKE)  # the parent's NEXT run is clean
        assert parent.profile_lineage() == (True, False)
        with pytest.raises(BaseException):
            await runner("job-1", None, lambda _text: None)
        assert captured == [(True, True)]
        clean_runner = subagent._make_runner(
            label="kid2",
            effective_prompt="synthetic",
            parent_session=parent,
            jobs_manager=_Jobs(),  # type: ignore[arg-type]
            model_spec=None,
        )
        with pytest.raises(BaseException):
            await clean_runner("job-2", None, lambda _text: None)
        assert captured[-1] == (True, False)
    finally:
        await parent.dispose()
    child = _session(local, parent="p", spawn_untrusted=True)
    try:
        start(child, WAKE)
        assert child.profile_lineage() == (True, True)  # grandchild inherits taint
    finally:
        await child.dispose()
    assert subagent._snapshot_profile_lineage(object()) == (False, True)

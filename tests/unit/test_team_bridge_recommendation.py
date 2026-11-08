"""Team Bridge as the ONE channel for teammate messaging (deterministic trigger).

Why this exists: an agent asked to coordinate with a colleague's chief of staff
sent the same request over Slack DM, Team Bridge and other channels at once —
duplicate, conflicting threads with unverifiable provenance. When the session
can use Team Bridge (signed in to Radient + enrolled in a Team org), a
coordination turn now carries ``guide://team-bridge`` plus a one-line rule.

The gate is drawn from the REAL persisted path (``Store.commit_enrollment`` ->
``bridge_runtime_for_session`` -> ``build_bridge_tool`` -> the session probe),
never from a hand-set flag, and it must never touch the network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness.types import ToolContext
from local_operator.session.session import Session
from local_operator.session_factory import (
    _TEAM_BRIDGE_RULE,
    _KnowledgeHooks,
    _select_knowledge_block,
    _team_bridge_recommendation,
    _tool_roster_rows,
)
from local_operator.team_bridge.host import bridge_runtime_for_session
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import build_bridge_tool

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_synthetic",
    tenant_id="tnt_synthetic",
    endpoint_id="bep_synthetic",
    canonical_base_url="https://hub.example.invalid/v1",
    enrollment_generation=1,
)
INCIDENT = "Coordinate with Damian's chief of staff on the QA plan and get his approval."


class _Config:
    """The two fields ``TeamBridgeSettings.from_config`` reads."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.config_dir = root
        self._values: dict[str, Any] = {"team_bridge": {"enabled": enabled, "session": "s1"}}

    def get_nested_value(self, path: tuple[str, ...], default: Any = None) -> Any:
        node: Any = self._values
        for step in path:
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return node


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any HTTP request from the gate or the trigger is a failure, not a slow path."""

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the Team Bridge recommendation gate made a network call")

    monkeypatch.setattr(httpx.Client, "send", _refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", _refuse)


def _session_probe(tmp_path: Path, *, enrolled: bool, enabled: bool = True) -> Any:
    """Build the real runtime + tool from persisted state, then the session probe."""
    if enrolled:
        Store(tmp_path).commit_enrollment(
            OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1"
        )
    runtime = bridge_runtime_for_session(
        config_manager=_Config(tmp_path, enabled=enabled), session_id="s1"
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    fake = SimpleNamespace(bridge_runtime=runtime, _tools=[tool] if tool else [])
    return runtime, tool, (lambda: Session._team_bridge_available(fake))  # type: ignore[arg-type]


def _hooks(probe: Any) -> _KnowledgeHooks:
    hooks = _KnowledgeHooks()
    hooks.team_bridge_available = probe
    return hooks


def test_signed_in_on_a_team_recommends_the_guide_with_the_rule(tmp_path: Path) -> None:
    runtime, tool, probe = _session_probe(tmp_path, enrolled=True)
    assert runtime is not None and tool is not None and runtime.admission == "pending"
    hit = _team_bridge_recommendation(INCIDENT, _hooks(probe))
    assert hit is not None
    assert hit.resource_url == "guide://team-bridge"
    assert "ONE channel" in hit.description and "no Slack/email/send fan-out" in hit.description


def test_signed_in_but_not_on_a_team_does_not_recommend(tmp_path: Path) -> None:
    # Radient credential may exist, but no Team-org enrollment was committed.
    runtime, tool, probe = _session_probe(tmp_path, enrolled=False)
    assert runtime is None and tool is None
    assert _team_bridge_recommendation(INCIDENT, _hooks(probe)) is None


def test_bridge_disabled_does_not_recommend(tmp_path: Path) -> None:
    _runtime, _tool, probe = _session_probe(tmp_path, enrolled=True, enabled=False)
    assert _team_bridge_recommendation(INCIDENT, _hooks(probe)) is None


@pytest.mark.parametrize("state", ["signin", "disabled", "paused"])
def test_a_known_negative_admission_does_not_recommend(tmp_path: Path, state: str) -> None:
    runtime, _tool, probe = _session_probe(tmp_path, enrolled=True)
    runtime.admission = state
    assert _team_bridge_recommendation(INCIDENT, _hooks(probe)) is None
    runtime.admission = "admitted"
    assert _team_bridge_recommendation(INCIDENT, _hooks(probe)) is not None


def test_unknown_membership_is_silent_and_offline() -> None:
    # No probe attached (not signed in / hooks built before the session) and a
    # probe that raises: both fail closed; the autouse fixture proves no fetch.
    assert _team_bridge_recommendation(INCIDENT, _KnowledgeHooks()) is None

    def _boom() -> bool:
        raise RuntimeError("store unreadable")

    assert _team_bridge_recommendation(INCIDENT, _hooks(_boom)) is None


@pytest.mark.parametrize(
    "query",
    [
        INCIDENT,
        "ask Ben to review the rollout doc",
        "ping Aida's agent about the launch",
        "follow up with my manager about the budget approval",
        "message the teammate who owns QA coordination",
        "DM Sarah the meeting notes",
        # Review round 1, R4: a technical word in the TOPIC, not the target.
        "message Sarah about the implementation plan",
        "Ask Ben's agent about the endpoint rollout",
        # R4: the added verbs, with the person as the object.
        "email Priya that the deck is ready",
        "slack Marco about the demo",
        "send Sarah the slides",
        "send my manager the deck",
        "let Ben know the build is out",
        "let my manager know I'm late",
        # The coordination arm: phrasing that names NO person. The shared-QA
        # window is where the duplicate-thread incident actually happened, and it
        # matched neither person arm.
        "Coordinate the shared QA window with the QA team",
        "Post the shared-QA update in the QA room",
        "QA coordination is at 3pm today",
        "ask the team which review slot works",
        "sync with the team on the QA rotation",
    ],
)
def test_coordination_intent_fires(query: str) -> None:
    assert _team_bridge_recommendation(query, _hooks(lambda: True)) is not None, query


@pytest.mark.parametrize(
    "query",
    [
        "send a message to the queue",
        "ask the API for the user list",
        "notify via webhook code when the build fails",
        "tell me about the chief of staff role",
        "Ask Claude to summarise this file",
        "implement a function that pings teammates on failure",
        "fix the message broker retry for agent tasks",
        "what does the bridge module do?",
        # Review round 1, R3: capitalised non-names after the verb.
        "Tell What the config does",
        "ask How to configure it",
        "Ask Please summarise the logs",
        "ask Docker to restart the container",
        "Ping Kubernetes about the pod",
        "Notify Monday about the release",
        "tell Pergamon about the outage",
        # R4: the veto still holds inside the verb's own window, and the general
        # verbs need the person as their direct object.
        "write code to notify my manager via webhook",
        "send the logs to the endpoint",
        "send the teammates list to the API",
        "let it know",
        "Let teammates decide",
        # The coordination arm stays NOUN-SHAPED: ordinary work keeps its verbs
        # and its technical objects and must never carry the rule.
        "coordinate the deploy",
        "sync the branch with origin",
        "QA the build before you ship",
        "share the test results in the channel",
        "set up the team channel integration",
    ],
)
def test_technical_and_self_addressed_turns_stay_quiet(query: str) -> None:
    assert _team_bridge_recommendation(query, _hooks(lambda: True)) is None, query


def test_the_rule_line_stays_inside_its_token_budget() -> None:
    # Harness-owned prompt text paid on every matching turn: one short line.
    assert "\n" not in _TEAM_BRIDGE_RULE
    assert len(_TEAM_BRIDGE_RULE) <= 160
    assert count_text_tokens(_TEAM_BRIDGE_RULE) <= 30


@pytest.mark.asyncio
async def test_the_line_reaches_the_rendered_block_once_and_freezes(tmp_path: Path) -> None:
    _runtime, _tool, probe = _session_probe(tmp_path, enrolled=True)
    hooks = _hooks(probe)
    block = await _select_knowledge_block(hooks, INCIDENT, task_id="t1")
    assert "<resource_recommendations>" in block
    assert block.count("guide://team-bridge") == 1
    assert f"- guide://team-bridge — {_TEAM_BRIDGE_RULE}" in block
    # Same task: the frozen block is reused verbatim, no second line.
    assert await _select_knowledge_block(hooks, INCIDENT, task_id="t1") == block


@pytest.mark.asyncio
async def test_unrelated_turn_renders_nothing_even_when_available(tmp_path: Path) -> None:
    _runtime, _tool, probe = _session_probe(tmp_path, enrolled=True)
    block = await _select_knowledge_block(_hooks(probe), "summarise this CSV", task_id="t2")
    assert "guide://team-bridge" not in block


def test_the_bridge_tool_joins_the_classifier_roster_only_when_held(tmp_path: Path) -> None:
    _runtime, tool, _probe = _session_probe(tmp_path, enrolled=True)
    rows = {row.name: row for row in _tool_roster_rows([tool])}
    assert rows["bridge"].resource_url == "tool://bridge"
    assert "bridge" not in {row.name for row in _tool_roster_rows([])}


async def _boot_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, enrolled: bool
) -> tuple[Any, str]:
    """Real ``create_session`` on an isolated config dir, then the incident turn's block."""
    import argparse

    from local_operator import settings_io
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session_factory import create_session

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_RUNTIME_ADOPT_SESSION", raising=False)
    monkeypatch.delenv("LOP_RUNTIME_DEFER_MATERIALISE", raising=False)
    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    config = ConfigManager(config_dir)
    if enrolled:
        settings_io.write_setting(config, settings_io.BY_KEY["team_bridge.enabled"], True)
        Store(config_dir).commit_enrollment(
            OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1"
        )
    args = argparse.Namespace(
        hosting="test", model="test", agent_name=None, agent_id=None, yolo=True, train=False
    )
    session = cast(Session, await create_session(args, config, AgentRegistry(config_dir)))
    hooks = session._knowledge_hooks
    assert hooks is not None
    return session, await _select_knowledge_block(hooks, INCIDENT, task_id="real")


@pytest.mark.asyncio
async def test_a_real_factory_session_attaches_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL boot path: enrolled config dir -> create_session -> hooks -> block.

    Guards the attach point (``Session._refresh_tool_roster``): a probe only the
    fake above exercises would be a feature no live session ever turns on. The
    GATED rule line itself is asserted, not merely a guide listing: the lexical
    guide index is ungated, and when it listed the guide the advisory copy was
    deduped away (review round 1, R1/Q2) — so the guide's frontmatter stays
    receiver-shaped and the incident turn must reach THIS line.
    """
    session, block = await _boot_and_render(tmp_path, monkeypatch, enrolled=True)
    try:
        assert session.bridge_runtime is not None
        assert "bridge" in {row.name for row in session._knowledge_hooks.tool_roster}
        assert block.count(f"- guide://team-bridge — {_TEAM_BRIDGE_RULE}") == 1, block
        assert "- team-bridge:" not in block, "the ungated listing must not carry it"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_real_unenrolled_session_gets_no_team_bridge_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the boot above: not on a team -> nothing at all."""
    session, block = await _boot_and_render(tmp_path, monkeypatch, enrolled=False)
    try:
        assert session.bridge_runtime is None
        assert "team-bridge" not in block, block
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_ungated_guide_index_does_not_route_the_incident_turn(tmp_path: Path) -> None:
    """The lexical guide index has no sign-in/team gate, so it must stay quiet here.

    Review round 1 (R1/Q1/Q2): a coordination-shaped frontmatter description
    made the index list ``team-bridge`` for users who are NOT on a team, and on
    an enrolled session that listing deduped the gated rule line away. The
    gated deterministic line is the one delivery path for this rule; the
    guide's description stays receiver-shaped.
    """
    from local_operator.guides import discover_guides
    from local_operator.skills.embeddings import LocalEmbedder
    from local_operator.skills.index import SkillIndex

    index = SkillIndex(cast(Any, discover_guides()), LocalEmbedder(), cache_dir=tmp_path / "cache")
    await index.build()
    for query in (INCIDENT, "Coordinate with Damian's chief of staff on the QA plan"):
        assert "team-bridge" not in {guide.name for guide in await index.select(query)}, query

"""A create naming an attachable identity BESIDE a team is refused in /agent's own words.

Issue #2014 established the rule — a team owns the agent slot — for the local
surfaces: ``lop exec --team X --profile Y`` is refused at preflight, ``/agent`` is
closed while a team is attached, and the session's own writer journals the team's
MANAGER as the slot's occupant. The network create was the seam that rule missed: a
create naming ``--profile reviewer --team release`` resolved both halves, wrote a
sidecar carrying both, and answered a receipt claiming the profile was applied
(``instructions_applied: true``, "agent: reviewer") — while the resumed session ran
the team's manager and the profile was dropped silently by the restore's
deliberate team-wins rule (which stays as it is for sidecars older builds wrote).

The pair can never be honoured, so it is REFUSED, on both halves of the verb, and
before anything moves:

* the REQUESTING half refuses the flag pair it can see — ``--profile`` beside
  ``--team`` — before ``definitions.push_to_peer``: a create that will be refused
  must not mirror definitions onto the peer as a side effect of asking;
* the OWNING half refuses everything that would ATTACH both halves, after both
  names resolve (a name the device does not hold keeps its own, existing sentence)
  and before the mint, the stamp and the sidecar — a refused create leaves nothing
  on disk.

Both halves speak the SAME sentence family the session and ``lop exec`` use
(``team_owns_the_agent_slot_message``, imported, never restated), with the
flag-level fact in front. The ``--agent`` spelling is the same rule through the
other door: ``resolve_create_identity`` treats a role/specialist row named there as
attachable, so naming one beside a team dropped its instructions identically —
while a routing-only legacy row beside a team stays allowed, because its model
rides the birth sample and no instructions are dropped. The controls below pin
that surviving pair and the untouched single-flag creates.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.network import identity, relay, types
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

# The flag-level fact leads, then the SESSION's sentence verbatim — the exact
# construction ``lop exec`` uses (``exec_startup.resolve_startup``), because it is
# the same pair and the same rule; the shared tail is what the desktop's header
# lane keys on.
SESSION_SENTENCE_TAIL = (
    "team release owns this session: manager is the speaker, so /agent is closed. "
    "Run /team clear to detach the team first."
)
PROFILE_PAIR_SENTENCE = (
    "--profile cannot be combined with --team: a team owns the session's agent slot. "
    + SESSION_SENTENCE_TAIL
)


def _edit_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with EVERY field spelled out, overridden where it matters.

    pyright reads the model's synthesised ``__init__`` as requiring all of them, so
    a partial construction is a ``reportCallIssue`` in the whole-tree type-check —
    the same helper, with the same reason, exists in ``test_definitions.py``.
    """
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


def _mint(root: Path) -> None:
    identity.mint(root, name="cloud-node-1")


def _make_team(root: Path, name: str = "release") -> None:
    TeamRegistry(root).create_team(
        TeamEditFields(
            name=name,
            description="A roster for the release.",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Ship reviewed work.",
            project="local-operator",
        )
    )


def _make_agent(root: Path, name: str, *, tags: list[str]) -> None:
    """One registered agent row; ``tags=["role"]`` makes it attachable."""
    AgentRegistry(root).create_agent(
        _edit_fields(name=name, description=f"Use when {name} work is needed.", tags=tags)
    )


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _warmed_server(root: Path) -> relay.RelayServer:
    """A server whose spawn/engage/prompt are stubbed: the CREATE path is the subject.

    A real spawn belongs to the e2e cells (the same split ``test_unattended_create``
    documents), and none of the three stubs is on the refusal path.
    """
    server = _server(root)
    server._warm_after_create = lambda *args, **kwargs: None
    server._engage_locally = lambda *args, **kwargs: ""
    server._prompt_on = lambda *args, **kwargs: (True, "")
    return server


def _link() -> Any:
    """A stub link carrying exactly what ``_op_session_create`` reads."""
    return SimpleNamespace(
        device_id="d_" + "a" * 32,
        network_id="n_0123456789abcdef0123456789abcdef",
        epoch=1,
        context=SimpleNamespace(
            capabilities=frozenset({"prompt"}),
            device_id="d_" + "a" * 32,
            network_id="n_0123456789abcdef0123456789abcdef",
            epoch=1,
        ),
    )


def _frame(**fields: Any) -> dict[str, Any]:
    return {"op": "net_session_create", "req": 41, "cwd": "", "prompt": "hi", **fields}


def _sidecar(root: Path, session_id: str) -> dict[str, Any]:
    path = root / "sessions" / session_id / "attachment.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_owning_half_refuses_profile_with_team_and_mints_nothing(root: Path) -> None:
    """The pair is refused on the device that would run it, before the mint.

    The sentence is the shared family byte for byte (the desktop lane keys on the
    words), and the refusal lands before any directory exists: a refused create
    leaves no session, no stamp, no sidecar and no claim behind.
    """
    _mint(root)
    _make_team(root)
    server = _server(root)

    with pytest.raises(types.MeshRefusal) as refused:
        server._op_session_create(_link(), _frame(profile="reviewer", team="release"))

    assert refused.value.code == "bad_request"
    assert str(refused.value) == PROFILE_PAIR_SENTENCE
    assert not (root / "sessions").exists(), "a refused create leaves nothing on disk"


def test_the_owning_half_refuses_an_attachable_agent_row_beside_a_team(root: Path) -> None:
    """The same rule through the ``--agent`` door, and the sentence names ITS flag.

    ``resolve_create_identity`` resolves a role/specialist row named via ``--agent``
    as attachable, so the pair would attach both halves exactly as ``--profile``
    would — the refusal names the flag the caller typed instead of pretending it
    was ``--profile``.
    """
    _mint(root)
    _make_team(root)
    _make_agent(root, "auditor", tags=["role"])
    server = _server(root)

    with pytest.raises(types.MeshRefusal) as refused:
        server._op_session_create(_link(), _frame(agent_name="auditor", team="release"))

    assert str(refused.value) == (
        "--agent cannot be combined with --team: a team owns the session's agent slot. "
        + SESSION_SENTENCE_TAIL
    )
    assert not (root / "sessions").exists(), "a refused create leaves nothing on disk"


def test_the_requesting_half_refuses_the_pair_before_it_mirrors_definitions(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refuse-before-push ordering, asserted structurally.

    ``_ctl_peer_create`` reconciles the names onto the peer BEFORE the create
    frame travels; a create that can never be honoured must not leave that side
    effect behind, so the refusal is raised ahead of the push. The manager is
    resolved best-effort from THIS device's registry — this device may
    legitimately not hold the team ("run it where it lives").
    """
    _mint(root)
    _make_team(root)
    pushed: list[Any] = []
    from local_operator.network import definitions as definitions_mod

    def _push(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pushed.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(definitions_mod, "push_to_peer", _push)
    server = _server(root)

    with pytest.raises(types.MeshRefusal) as refused:
        server._ctl_peer_create({"peer": "cloud-node-1", "profile": "reviewer", "team": "release"})

    assert pushed == [], "a refused create must not mirror definitions"
    assert refused.value.code == "bad_request"
    assert str(refused.value) == PROFILE_PAIR_SENTENCE


def test_the_requesters_sentence_does_not_require_holding_the_team(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unpinned case: a team this device does not hold still refuses, in words.

    ``definitions.unpinned``'s rule is the sibling here — a name this device does
    not hold is not a reason to guess — and the sentence degrades to "its manager"
    exactly as the shared builder documents, rather than naming a speaker this
    device cannot know.
    """
    _mint(root)
    pushed: list[Any] = []
    from local_operator.network import definitions as definitions_mod

    monkeypatch.setattr(
        definitions_mod, "push_to_peer", lambda *args, **kwargs: pushed.append(args) or {"ok": True}
    )
    server = _server(root)

    with pytest.raises(types.MeshRefusal) as refused:
        server._ctl_peer_create({"peer": "cloud-node-1", "profile": "reviewer", "team": "release"})

    assert pushed == []
    assert str(refused.value) == (
        "--profile cannot be combined with --team: a team owns the session's agent slot. "
        "team release owns this session: its manager is the speaker, so /agent is closed. "
        "Run /team clear to detach the team first."
    )


def test_a_routing_only_agent_beside_a_team_still_creates_and_reports_it(root: Path) -> None:
    """The pair the rule does NOT cover, pinned so the refusal cannot over-reach.

    A legacy conversational row is deliberately not attachable — only its model and
    hosting contribute — so naming it beside a team drops nothing, and the create
    proceeds with the receipt saying "routing only". The writer journals the team
    ALONE (the agent half of the sidecar is empty), which is what makes the reply
    and the disk tell the same story.
    """
    _mint(root)
    _make_team(root)
    _make_agent(root, "my-chat", tags=[])
    server = _warmed_server(root)

    detail = server._op_session_create(_link(), _frame(agent_name="my-chat", team="release"))

    assert detail["team"]["name"] == "release"
    assert detail["agent"]["instructions_applied"] is False
    assert "not attachable" in detail["agent"]["detail"]
    assert _sidecar(root, detail["session_id"]) == {"team": "release", "agent": "", "goal": ""}


def test_a_team_alone_and_a_profile_alone_keep_todays_receipts_and_sidecars(root: Path) -> None:
    """The surviving singles are byte-for-byte what they were: nothing over-narrows.

    Each half keeps its own block in the reply and its own half of the sidecar, and
    the invariant the defect violated is asserted directly: a reply that names a
    team never also claims an attached profile was applied.
    """
    _mint(root)
    _make_team(root)
    server = _warmed_server(root)

    team_only = server._op_session_create(_link(), _frame(team="release"))
    assert team_only["agent"] is None
    assert team_only["team"]["name"] == "release"
    assert _sidecar(root, team_only["session_id"]) == {"team": "release", "agent": "", "goal": ""}

    profile_only = server._op_session_create(_link(), _frame(profile="reviewer"))
    assert profile_only["agent"]["name"] == "reviewer"
    assert profile_only["agent"]["instructions_applied"] is True
    assert profile_only["team"] is None
    assert _sidecar(root, profile_only["session_id"]) == {
        "team": "",
        "agent": "reviewer",
        "goal": "",
    }


def test_every_created_reply_stays_honest_about_the_slot(root: Path) -> None:
    """The invariant, quoted as the defect saw it: no receipt claims an applied
    profile beside a team, because no create that could produce one is accepted.

    (The two singles are covered above; this cell states the property the pair
    refusal exists to guarantee, so a future relaxation that re-admits the pair
    without a truthful reply fails HERE.)
    """
    _mint(root)
    _make_team(root)
    server = _warmed_server(root)

    for fields in ({"team": "release"}, {"profile": "reviewer"}):
        detail = server._op_session_create(_link(), _frame(**fields))
        agent = detail.get("agent") or {}
        assert not (detail.get("team") and agent.get("instructions_applied")), detail


def test_the_reply_for_a_team_names_no_attached_profile(root: Path) -> None:
    """A team-only reply's agent block is absent — not a second way to say "manager".

    The restore's rule (the manager IS the speaker) is a SESSION state, not a
    receipt line: the reply reports the binding the requester asked for, and asking
    for a team is not asking for a profile.
    """
    _mint(root)
    _make_team(root)
    server = _warmed_server(root)

    detail = server._op_session_create(_link(), _frame(team="release"))

    assert detail["agent"] is None
    assert set(detail.get("team", {})) == {"name", "id", "digest"}

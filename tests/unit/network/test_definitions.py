"""Agent and team definitions over the mesh (`network/definitions.py`).

WHAT THESE CELLS PIN, and each one is a claim the change would otherwise make on
trust:

* **The payload cannot carry a credential** — asserted as a property of the
  builder AND of the receiver, with a planted value that is a real credential
  SHAPE and a real secret VALUE: neither may appear in the serialised bundle, and
  a row carrying one is withheld by name rather than sent.
* **The conflict policy holds in both directions** — a locally authored row is
  never overwritten by a peer, a mirror follows its origin, and a mirror that the
  operator has EDITED becomes theirs (the digest is what tells the two apart).
* **Idempotence** — applying the same bundle twice reports ``unchanged`` the second
  time and writes nothing, which rests on the row round-tripping through one
  normaliser so both ends hash identically.
* **The create path refuses by name** — a name the peer cannot resolve is a
  refusal naming the name, never a session created on the default agent.

The last one is the requirement's hard edge ("never a silent fallback"), and it is
why ``resolve_create_identity`` returns a SENTENCE rather than a bool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.network import definitions, relay, store, types, wire
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _edit_fields(**overrides: Any) -> "AgentEditFields":
    """``AgentEditFields`` with EVERY field spelled out, overridden by the few a
    test cares about.

    Not a style choice: pyright reads the model's synthesised ``__init__`` as
    requiring all of them (``Field(None, ...)`` is not seen as a default), so a
    partial construction is a ``reportCallIssue`` error in CI's whole-tree
    type-check even though it runs fine. The same helper, with the same reason,
    exists in ``tests/unit/test_agent_profiles.py``.
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


def _make_agent(
    root: Path,
    name: str,
    *,
    prompt: str = "You review diffs before the tree.",
    tags: list[str] | None = None,
    model: str = "",
    hosting: str = "",
) -> str:
    """One registered agent row, built the way the product builds one."""
    registry = AgentRegistry(root)
    agent = registry.create_agent(
        _edit_fields(
            name=name,
            description=f"Use when {name} work is needed.",
            tags=list(tags if tags is not None else ["role", "tools:read,grep"]),
            model=model,
            hosting=hosting,
        )
    )
    registry.set_agent_system_prompt(agent.id, prompt)
    return str(agent.id)


def _make_team(root: Path, name: str, *, instructions: str = "Review before shipping.") -> str:
    registry = TeamRegistry(root)
    team = registry.create_team(
        TeamEditFields(
            name=name,
            description="A roster for the release.",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
            instructions=instructions,
            project="local-operator",
        )
    )
    return str(team.id)


# ---------------------------------------------------------------------------
# The credential assertion
# ---------------------------------------------------------------------------


def test_a_definition_carrying_a_credential_shape_is_withheld_not_sent(tmp_path: Path) -> None:
    """The withheld row is NAMED, so the user learns why it did not travel.

    A silent drop would be worse than the leak it prevents: the peer then cannot
    resolve a name the operator believes is synced.
    """
    root = _root(tmp_path)
    # ``sk_live_...`` is a vendor-prefixed token: a value spelled like a credential
    # even though nothing in this test told the scrubber it was one.
    poisoned = "Authenticate with " + "sk_live_" + "A" * 40 + " before calling out."
    _make_agent(root, "leaky", prompt=poisoned)
    _make_agent(root, "clean", prompt="Read the diff first.")

    bundle = definitions.local_bundle(root)
    names = [row["name"] for row in bundle["agents"]]
    assert "clean" in names
    assert "leaky" not in names, "a credential-shaped definition must not be sent"
    assert [row["name"] for row in bundle["withheld"]] == ["leaky"]
    assert bundle["withheld"][0]["shape"], "the shape label is reported, never the value"
    # AND THE VALUE ITSELF IS NOWHERE IN THE PAYLOAD.
    assert "sk_live_" not in json.dumps(bundle), "the credential value crossed the builder"


def test_a_secret_value_from_the_local_store_cannot_reach_the_bundle(tmp_path: Path) -> None:
    """The structural half: the builder never reads a credential store at all.

    A value the process HOLDS (as ``session/credential_ops`` would) is not a shape,
    so the shape pass would not necessarily catch it — what keeps it off the wire
    is that the builder reads ``agent.yml``/``system_prompt.md`` and the team row
    files and nothing else. This cell proves that by planting the value in a
    credential store and asserting the bundle is unchanged byte for byte.
    """
    root = _root(tmp_path)
    _make_agent(root, "clean", prompt="Read the diff first.")
    before = json.dumps(definitions.local_bundle(root), sort_keys=True)

    from local_operator.variables import VariableStore

    secret = "zzTOPSECRETvaluezz"
    store = VariableStore()
    # ``register_redaction`` is the public seam for "a value this session holds",
    # which is exactly the category the shape pass is not guaranteed to catch.
    store.register_redaction(secret)

    after = json.dumps(definitions.local_bundle(root), sort_keys=True)
    assert before == after
    assert secret not in after


def test_a_bundle_the_receiver_refuses_on_a_credential_shape(tmp_path: Path) -> None:
    """Defence in depth: an old or hostile peer cannot install one either."""
    root = _root(tmp_path)
    bundle = {
        "kind": definitions.BUNDLE_KIND,
        "version": 1,
        "origin_device": "d_" + "a" * 32,
        "agents": [
            {
                "kind": "agent",
                "name": "smuggled",
                "origin_id": "a_1",
                "created_date": definitions._iso(""),
                "fields": {"name": "smuggled", "description": "", "tags": []},
                "system_prompt": "token " + "ghp_" + "b" * 36,
            }
        ],
        "teams": [],
    }
    summary = definitions.apply_bundle(root, bundle, origin_device="d_" + "a" * 32)
    assert summary["installed"] == []
    assert summary["refused"], "a credential-shaped row must be refused, not installed"
    assert AgentRegistry(root).get_agent_by_name("smuggled") is None


# ---------------------------------------------------------------------------
# Round trip, idempotence, conflict policy
# ---------------------------------------------------------------------------


def test_a_bundle_round_trips_and_applying_it_twice_changes_nothing(tmp_path: Path) -> None:
    """Idempotence, and the normaliser it rests on.

    Both ends rebuild the row through ONE function, so the digest of a row the
    sender built equals the digest of the same row read back after install. A test
    that only checked "it installed" would pass on a build where every push was a
    conflict.
    """
    author = _root(tmp_path / "author")
    bare = _root(tmp_path / "bare")
    agent_id = _make_agent(
        author,
        "reviewer",
        prompt="Classify by severity before you read the tree.",
        tags=["role", "tools:read,grep", "effort:lo", "delegate:yes"],
        model="claude-opus-5-5",
        hosting="anthropic",
    )
    team_id = _make_team(author, "feature-release")

    bundle = definitions.local_bundle(author)
    origin = bundle["origin_device"]
    first = definitions.apply_bundle(bare, bundle, origin_device=origin)
    assert [row["name"] for row in first["installed"]] == ["reviewer", "feature-release"]
    assert first["conflicts"] == [] and first["refused"] == []

    # The row kept its ORIGIN id, so the agent a create names resolves to the same
    # row on both devices rather than to a lookalike.
    installed = AgentRegistry(bare).get_agent_by_name("reviewer")
    assert installed is not None and str(installed.id) == agent_id
    assert AgentRegistry(bare).get_agent_system_prompt(agent_id).startswith("Classify by severity")
    team = TeamRegistry(bare).get_team(team_id)
    assert team.name == "feature-release"
    assert "Review before shipping." in team.instructions

    # THE DIGEST AGREEMENT: the receiver's own view of the row equals the sender's.
    assert definitions.definition_state(bare)["agents"]["reviewer"] == (
        definitions.definition_state(author)["agents"]["reviewer"]
    )

    second = definitions.apply_bundle(bare, bundle, origin_device=origin)
    assert second["installed"] == [] and second["updated"] == []
    assert [row["name"] for row in second["unchanged"]] == ["reviewer", "feature-release"]
    assert second["conflicts"] == []


def test_a_locally_authored_name_is_never_overwritten_and_the_conflict_is_named(
    tmp_path: Path,
) -> None:
    """The conflict policy's first rule, from the receiving side."""
    author = _root(tmp_path / "author")
    peer = _root(tmp_path / "peer")
    _make_agent(author, "reviewer", prompt="The author's reviewer.")
    _make_agent(peer, "reviewer", prompt="MY OWN reviewer, do not touch it.")

    bundle = definitions.local_bundle(author)
    summary = definitions.apply_bundle(peer, bundle, origin_device=bundle["origin_device"])
    assert summary["conflicts"], "an occupied name must be reported"
    conflict = summary["conflicts"][0]
    assert conflict["name"] == "reviewer"
    # The sentence names the reason the operator can act on: the name is theirs,
    # so nothing was overwritten (the shape ``network/definitions.py`` writes).
    assert "not overwritten" in conflict["reason"]
    # UNTOUCHED: the operator's own definition is exactly as it was.
    registry = AgentRegistry(peer)
    mine = registry.get_agent_by_name("reviewer")
    assert mine is not None
    assert registry.get_agent_system_prompt(str(mine.id)) == "MY OWN reviewer, do not touch it."


def test_an_edited_mirror_becomes_locally_authored_and_stops_following_its_origin(
    tmp_path: Path,
) -> None:
    """The policy's second rule: the digest is what tells "mirror" from "mine".

    Without it, a mirror would follow its origin forever and an edit the operator
    made on the receiving device would be silently reverted by the next tick.
    """
    author = _root(tmp_path / "author")
    peer = _root(tmp_path / "peer")
    author_id = _make_agent(author, "reviewer", prompt="v1")
    bundle = definitions.local_bundle(author)
    origin = bundle["origin_device"]
    assert definitions.apply_bundle(peer, bundle, origin_device=origin)["installed"]

    # A mirror is NOT re-exported by the device holding it: authorship belongs to
    # the device that wrote it, so a third peer gets the row from the author.
    assert definitions.local_bundle(peer)["agents"] == []

    # The operator edits it here.
    peer_registry = AgentRegistry(peer)
    peer_registry.set_agent_system_prompt(author_id, "v2 written HERE")
    edited = definitions.local_rows(peer)["agents"][0]
    assert definitions.authored_locally(peer, kind="agents", name="reviewer", row=edited)
    # ...and it is now exported by this device, because it is this device's.
    assert [row["name"] for row in definitions.local_bundle(peer)["agents"]] == ["reviewer"]

    # A later push from the ORIGIN does not revert the edit: it is a conflict.
    author_registry = AgentRegistry(author)
    author_registry.set_agent_system_prompt(author_id, "v3 from the author")
    newer = definitions.local_bundle(author)
    summary = definitions.apply_bundle(peer, newer, origin_device=origin)
    assert summary["conflicts"], "the edited mirror must not be reverted"
    assert peer_registry.get_agent_system_prompt(author_id) == "v2 written HERE"


def test_a_mirror_follows_its_origin_and_a_different_origin_may_not_take_the_name(
    tmp_path: Path,
) -> None:
    """Update-from-the-same-origin works; a second author of the same name does not."""
    author = _root(tmp_path / "author")
    peer = _root(tmp_path / "peer")
    other = _root(tmp_path / "other")
    author_id = _make_agent(author, "scout", prompt="v1")
    bundle = definitions.local_bundle(author)
    origin = bundle["origin_device"]
    assert definitions.apply_bundle(peer, bundle, origin_device=origin)["installed"]

    AgentRegistry(author).set_agent_system_prompt(author_id, "v2")
    updated = definitions.apply_bundle(peer, definitions.local_bundle(author), origin_device=origin)
    assert [row["name"] for row in updated["updated"]] == ["scout"]
    assert AgentRegistry(peer).get_agent_system_prompt(author_id) == "v2"

    _make_agent(other, "scout", prompt="A different scout entirely.")
    theirs = definitions.local_bundle(other)
    summary = definitions.apply_bundle(peer, theirs, origin_device=theirs["origin_device"])
    assert summary["conflicts"]
    assert AgentRegistry(peer).get_agent_system_prompt(author_id) == "v2", "origin still wins"


def test_an_edited_mirror_is_reported_not_called_unchanged(tmp_path: Path) -> None:
    """THE GAP THE TWO-RELAY RIG FOUND, pinned here.

    The idempotence short-circuit compared the incoming digest with what this device
    RECORDED and returned "unchanged" — which is right for a row nobody touched, and
    wrong for a mirror the operator has since edited: the recorded digest still equals
    the sender's (nothing has pushed since), so the sync reported success while the row
    here was somebody else's work. In the rig that came back as ``unchanged`` instead of
    a conflict, meaning the conflict policy was reachable only in unit tests.
    """
    author = _root(tmp_path / "author")
    peer = _root(tmp_path / "peer")
    _make_agent(author, "reviewer", prompt="Origin's revision.")
    first = definitions.local_bundle(author)
    definitions.apply_bundle(peer, first, origin_device=first["origin_device"])

    mirror = next((peer / "agents").iterdir())
    (mirror / "system_prompt.md").write_text("Edited here.", encoding="utf-8")

    # The sender pushes the SAME revision it already sent: nothing changed on its side.
    again = definitions.local_bundle(author)
    summary = definitions.apply_bundle(peer, again, origin_device=again["origin_device"])
    assert summary["unchanged"] == []
    assert [row["name"] for row in summary["conflicts"]] == ["reviewer"]
    assert "local edits" in summary["conflicts"][0]["reason"]
    assert (mirror / "system_prompt.md").read_text(encoding="utf-8") == "Edited here."


def test_a_deleted_mirror_is_re_installed(tmp_path: Path) -> None:
    """The deliberate asymmetry: an edit is refused, a deletion is restored."""
    author = _root(tmp_path / "author")
    peer = _root(tmp_path / "peer")
    _make_agent(author, "reviewer", prompt="Origin's revision.")
    bundle = definitions.local_bundle(author)
    definitions.apply_bundle(peer, bundle, origin_device=bundle["origin_device"])

    import shutil

    shutil.rmtree(next((peer / "agents").iterdir()))
    summary = definitions.apply_bundle(peer, bundle, origin_device=bundle["origin_device"])
    assert [row["name"] for row in summary["installed"]] == ["reviewer"]
    assert summary["conflicts"] == []
    assert AgentRegistry(peer).get_agent_by_name("reviewer") is not None


def test_the_bundle_refuses_a_version_it_does_not_understand(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(definitions.DefinitionsRefused) as excinfo:
        definitions.apply_bundle(
            root,
            {"kind": definitions.BUNDLE_KIND, "version": 99, "agents": [], "teams": []},
            origin_device="d_x",
        )
    assert "version" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# The create path's half
# ---------------------------------------------------------------------------


def test_a_named_team_travels_with_its_roster(tmp_path: Path) -> None:
    """A team that resolves without its members FAILS LATER, at the first delegation.

    The row carries names, not definitions, so the create itself would succeed and the
    manager's first ``task(agent=...)`` would be the thing that could not resolve. This
    pins the fix at the payload: naming the team pulls in its members (and, through a
    nested slot, the members of the team that member names).
    """
    root = _root(tmp_path)
    _make_agent(root, "coder")
    _make_agent(root, "reviewer")
    _make_agent(root, "manager")
    _make_agent(root, "designer")
    teams = TeamRegistry(root)
    teams.create_team(
        TeamEditFields(name="pod", manager="manager", members=[TeamMember(role="designer")])
    )
    teams.create_team(
        TeamEditFields(
            name="release",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer")],
        )
    )
    # A roster slot that names ANOTHER team, so the expansion has to descend.
    teams.create_team(
        TeamEditFields(
            name="whole-org",
            manager="manager",
            # BOTH slots are team references: the expansion has to descend through a
            # nested team to reach the people in it.
            members=[TeamMember(role="release", kind="team"), TeamMember(role="pod", kind="team")],
        )
    )

    named = definitions.local_bundle(root, names={"teams": ["release"]})
    assert [row["name"] for row in named["agents"]] == ["coder", "manager", "reviewer"]
    assert [row["name"] for row in named["teams"]] == ["release"]

    nested = definitions.local_bundle(root, names={"teams": ["whole-org"]})
    assert [row["name"] for row in nested["teams"]] == ["pod", "release", "whole-org"]
    assert [row["name"] for row in nested["agents"]] == [
        "coder",
        "designer",
        "manager",
        "reviewer",
    ]

    # And a NARROW bundle still excludes everything else on the device.
    assert definitions.local_bundle(root, names={"agents": ["coder"]})["teams"] == []


def test_resolve_create_identity_names_what_is_missing(tmp_path: Path) -> None:
    """The requirement's hard edge: a refusal that NAMES the name."""
    root = _root(tmp_path)
    identity, refusal = definitions.resolve_create_identity(root, profile="nobody-here")
    assert identity is None
    assert "nobody-here" in refusal
    assert "not created" in refusal
    assert "definitions push" in refusal, "the refusal names the remedy"

    _identity, refusal = definitions.resolve_create_identity(root, team_name="lopdev")
    assert "lopdev" in refusal


def test_resolve_create_identity_reads_the_row_it_will_run(tmp_path: Path) -> None:
    """A resolved role carries its model (the birth sample) and its digest."""
    root = _root(tmp_path)
    _make_agent(root, "reviewer", tags=["role", "tools:read"], model="m-1", hosting="anthropic")
    identity, refusal = definitions.resolve_create_identity(root, profile="reviewer", effort="hi")
    assert refusal == ""
    assert identity is not None
    assert identity.agent_name == "reviewer"
    assert identity.instructions_attachable is True
    assert identity.agent_digest == definitions.definition_state(root)["agents"]["reviewer"]
    assert identity.birth is not None
    assert (identity.birth.provider, identity.birth.model_id) == ("anthropic", "m-1")
    assert identity.birth.reasoning_effort == "hi"


def test_a_packaged_seed_resolves_with_no_definitions_at_all(tmp_path: Path) -> None:
    """The one identity that needs no sync: a seed ships with the build.

    Asserted because it is what makes ``--profile reviewer`` work on a device that
    has never received anything — including a clean pod — while ``agent_digest``
    stays empty, so the create frame pins no revision for it: there is nothing on
    the far end to be stale.
    """
    root = _root(tmp_path)
    identity, refusal = definitions.resolve_create_identity(root, profile="reviewer")
    assert refusal == ""
    assert identity is not None
    assert identity.agent_kind == "seed"
    assert identity.instructions_attachable is True
    assert identity.agent_digest == ""
    assert definitions.local_bundle(root)["agents"] == []


def test_a_legacy_agent_row_is_routing_only_and_says_so(tmp_path: Path) -> None:
    """The honest half: not everything a row carries can reach a session.

    A legacy conversational agent is deliberately not attachable (its prompt is
    private to it), so the create applies its ROUTING and reports that its
    instructions did not travel — never a silent claim that the whole row applied.
    """
    root = _root(tmp_path)
    # No ``role`` tag and no ``specialist`` category: an ordinary chat agent row.
    _make_agent(root, "my-chat", tags=[], model="m-2", hosting="anthropic")
    identity, refusal = definitions.resolve_create_identity(root, agent_name="my-chat")
    assert refusal == ""
    assert identity is not None
    assert identity.instructions_attachable is False
    assert identity.birth is not None and identity.birth.model_id == "m-2"


def test_check_expected_refuses_a_revision_that_moved(tmp_path: Path) -> None:
    """Pin the revision the requester reconciled, not "whatever is there now"."""
    root = _root(tmp_path)
    agent_id = _make_agent(root, "reviewer", prompt="v1")
    digest = definitions.definition_state(root)["agents"]["reviewer"]
    assert definitions.check_expected(root, {"agents": {"reviewer": digest}}) == ""

    AgentRegistry(root).set_agent_system_prompt(agent_id, "v2")
    stale = definitions.check_expected(root, {"agents": {"reviewer": digest}})
    assert stale, "a moved revision must refuse"
    assert "reviewer" in stale and "different revision" in stale
    # An ``expect`` for a name this device never heard of is refused too, and the
    # sentence names it rather than reporting a digest mismatch against nothing.
    missing = definitions.check_expected(root, {"teams": {"lopdev": "0" * 64}})
    assert "lopdev" in missing


# ---------------------------------------------------------------------------
# The payload's own hostile-input boundary (review round 1: BLOCKER 1)
# ---------------------------------------------------------------------------


def _hostile_agent_row(name: str, origin_id: str) -> dict[str, Any]:
    """A row shaped exactly as a bundle carries one, with a chosen id."""
    return {
        "kind": "agent",
        "name": name,
        "origin_id": origin_id,
        "created_date": "2026-01-01T00:00:00+00:00",
        "system_prompt": "WRITER CONTROLLED",
        "fields": {"name": name, "description": "", "tags": [], "categories": []},
    }


def _bundle_of(*rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": definitions.BUNDLE_KIND,
        "version": definitions.BUNDLE_VERSION,
        "origin_device": "d_" + "f" * 32,
        "agents": list(rows),
        "teams": [],
    }


def test_an_origin_id_that_is_a_path_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    """BLOCKER 1: the id is a DIRECTORY NAME, and the sender chooses it.

    ``AgentRegistry.save_agent`` does ``agents_dir / id`` with ``mkdir(parents=True)``,
    so a hostile ``origin_id`` is a write anywhere this user can write. Before the
    guard this cell's payload created ``escaped-agent/`` four levels above
    ``agents/`` — a complete agent directory, with a sender-chosen
    ``system_prompt.md`` — and the summary said it had INSTALLED an agent. The
    refusal must name the row and leave the filesystem alone.
    """
    bare = _root(tmp_path / "bare")
    summary = definitions.apply_bundle(
        bare,
        _bundle_of(_hostile_agent_row("wire-escape", "../../../../escaped-agent")),
        origin_device="d_" + "f" * 32,
    )
    assert summary["installed"] == []
    assert [row["name"] for row in summary["refused"]] == ["wire-escape"]
    assert "id" in summary["refused"][0]["reason"]
    # NOTHING ANYWHERE: not the escaped directory, not an agent directory at all.
    assert sorted(path.name for path in tmp_path.rglob("escaped-agent")) == []
    assert AgentRegistry(bare).list_agents() == []
    assert not (bare / "agents").exists() or list((bare / "agents").iterdir()) == []


def test_the_read_rule_accepts_every_id_that_can_be_a_directory_name() -> None:
    """The MODEL's id rule is the SAFETY one, and that split is the whole fix.

    It runs on the read path (``_scan_agents_metadata`` builds every row through this
    model), so it must accept anything that can BE a directory name under
    ``agents_dir`` — otherwise a pre-existing row is unreadable, the scan counts it as
    incomplete, and ``require_complete_metadata`` refuses every create on the device.
    What it must still refuse is anything that is not one path segment.
    """
    from pydantic import ValidationError

    from local_operator.agents import AgentData, validate_agent_id_segment

    for good in (
        "8db36f01-1695-446d-865e-563ba6846662",
        "autosave",
        # Legacy-but-legal ids: every one of these can be, and may already be, a
        # directory under ``agents/``.
        "legacy row",
        "-leading",
        ".dotfile",
        "_under",
        "Agent \u770b",
        "x" * 129,
    ):
        assert validate_agent_id_segment(good) == good
    for bad in ("../../../../escaped-agent", "a/b", "..", ".", ""):
        with pytest.raises(ValueError):
            validate_agent_id_segment(bad)
        with pytest.raises(ValidationError):
            AgentData.model_validate(
                {
                    "id": bad,
                    "name": "x",
                    "created_date": "2026-01-01T00:00:00+00:00",
                    "version": "1.0.0",
                }
            )
    # ...and a legacy id LOADS through the same model, which is the point.
    legacy = AgentData.model_validate(
        {
            "id": "legacy row",
            "name": "x",
            "created_date": "2026-01-01T00:00:00+00:00",
            "version": "1.0.0",
        }
    )
    assert legacy.id == "legacy row"


def test_the_write_rule_still_refuses_a_non_conforming_id() -> None:
    """The STRICT rule is what a value arriving from OUTSIDE must pass.

    ``teams.validate_team_id``'s charset, kept for the write boundaries: a create, an
    import's generated id, and the mirrored-row apply (``definitions._apply_agent``,
    which is where BLOCKER 1 was). Weakening the read rule did not move this one.
    """
    from local_operator.agents import is_conforming_agent_id, validate_agent_id

    for good in ("8db36f01-1695-446d-865e-563ba6846662", "autosave", "a.b_c-1"):
        assert validate_agent_id(good) == good
        assert is_conforming_agent_id(good) is True
    for bad in (
        "../../../../escaped-agent",
        "a/b",
        "..",
        ".",
        "",
        "-leading",
        ".dotfile",
        "_under",
        "with space",
        "unicode-\u770b",
        "x" * 129,
    ):
        with pytest.raises(ValueError):
            validate_agent_id(bad)
        assert is_conforming_agent_id(bad) is False


# ---------------------------------------------------------------------------
# The create's own two guards (review round 1: BLOCKER 2, QA round 1: Q1)
# ---------------------------------------------------------------------------


def test_a_push_the_peer_half_took_stops_the_create() -> None:
    """BLOCKER 2: a row the peer would not take is a row it will resolve ITSELF.

    The measured failure: A's push was refused for a drifted mirror, the push result
    was discarded, the create went ahead unpinned, and the session ran B's text under
    the name A's user chose. The refusal has to name the row, the reason and where the
    divergent copy lives.
    """
    refused = definitions.create_refusal(
        {
            "ok": False,
            "code": "conflict",
            "conflicts": [
                {
                    "kind": "agent",
                    "name": "mesh-reviewer",
                    "reason": "the copy of that name here has local edits, so it was not "
                    "overwritten",
                }
            ],
            "refused": [],
        },
        peer_label="bare-peer",
    )
    assert "mesh-reviewer" in refused
    assert "local edits" in refused
    assert "was not created" in refused
    assert "bare-peer" in refused
    # A push that never REACHED the peer is not this case: there is no half-state, and
    # the peer's own sentence (plus the pin) is the answer.
    assert (
        definitions.create_refusal(
            {"ok": False, "code": "unreachable", "message": "not answering"},
            peer_label="bare-peer",
        )
        == ""
    )
    # A WITHHELD row is a refusal too: the row cannot travel, so nothing under that
    # name on the peer would be the user's definition.
    withheld = definitions.create_refusal(
        {
            "ok": True,
            "withheld": [{"kind": "agent", "name": "token-holder", "shape": "dsn-password"}],
        },
        peer_label="bare-peer",
    )
    assert "token-holder" in withheld and "dsn-password" in withheld
    # A clean push, and a push with an empty report, both let the create proceed.
    assert definitions.create_refusal({"ok": True, "withheld": []}, peer_label="b") == ""


def test_the_create_is_pinned_to_the_asked_for_revision(tmp_path: Path) -> None:
    """BLOCKER 2's second half: the pin must exist even when the push did not.

    ``expect_from_push`` could only pin what a push had already reconciled, so a push
    that half-completed left the create unpinned. The pin is now THIS device's revision
    of every name the frame mentions — and the peer computes the same digest for the
    same row (both ends rebuild rows through one normaliser), which is asserted here
    rather than assumed.
    """
    author = _root(tmp_path / "author")
    bare = _root(tmp_path / "bare")
    _make_agent(author, "reviewer", prompt="The author's reviewer.")
    pins = definitions.create_pins(author, agent_names=["reviewer"], team_names=[])
    assert set(pins["agents"]) == {"reviewer"}

    bundle = definitions.local_bundle(author)
    definitions.apply_bundle(bare, bundle, origin_device=bundle["origin_device"])
    # The peer holds it, so the pin agrees — this is the assertion that makes the pin
    # usable at all (a digest computed differently on each side would refuse every
    # create).
    assert definitions.check_expected(bare, pins) == ""

    # And when the peer's copy DIVERGES, the pin is what refuses it by name instead of
    # letting the create run that device's text.
    mine = AgentRegistry(bare).get_agent_by_name("reviewer")
    assert mine is not None
    AgentRegistry(bare).set_agent_system_prompt(str(mine.id), "EDITED ON THE PEER")
    stale = definitions.check_expected(bare, pins)
    assert "reviewer" in stale and "different revision" in stale

    # A name this device does not hold is not pinned: the peer's own row is then the
    # only revision that name has, and the create reports it rather than pretending.
    assert definitions.create_pins(author, agent_names=["only-on-the-peer"], team_names=[]) == {}


def test_a_named_id_resolves_to_the_name_the_bundle_selects_by(tmp_path: Path) -> None:
    """QA MINOR 7: an ``--agent-id``-only create used to reconcile nothing.

    A bundle selects by NAME and ids are per-device, so the id has to be resolved where
    it was typed. An id this device does not hold resolves to ``""``, which the create
    path refuses by name rather than sending a frame whose definition only the peer can
    resolve.
    """
    author = _root(tmp_path / "author")
    agent_id = _make_agent(author, "reviewer")
    assert definitions.name_for_agent_id(author, agent_id) == "reviewer"
    assert definitions.name_for_agent_id(author, "8db36f01-0000-0000-0000-000000000000") == ""


# ---------------------------------------------------------------------------
# The credential scan, the index lock and a registry fault (review round 1)
# ---------------------------------------------------------------------------


def test_the_credential_scan_covers_every_field_a_row_carries(tmp_path: Path) -> None:
    """MINOR 6: the guard must not describe itself wider than it is.

    Measured before the fix: a DSN in ``tags`` and a ``ghp_…`` token in ``hosting``
    BOTH installed on the receiving device, because the scan ran over the long prose
    fields only. Both are carried fields, so both are scanned now — at the sender
    (withheld, never sent) and at the receiver (refused, never installed).
    """
    dsn = "postgres://agent:sup3rs3cret@db.internal:5432/app"
    token = "ghp_" + "a1b2c3d4e5" * 4  # 40 characters, the shape the table matches
    author = _root(tmp_path / "author")
    _make_agent(author, "tag-leak", tags=["role", f"dsn:{dsn}"])
    _make_agent(author, "host-leak", hosting=token)

    bundle = definitions.local_bundle(author)
    assert {row["name"] for row in bundle["agents"]} == set()
    assert {row["name"] for row in bundle["withheld"]} == {"tag-leak", "host-leak"}

    # The receiver's half, on a bundle built by hand (a hostile or older producer).
    bare = _root(tmp_path / "bare")
    row = _hostile_agent_row("host-leak", "8db36f01-1695-446d-865e-563ba6846662")
    row["fields"]["hosting"] = token
    summary = definitions.apply_bundle(bare, _bundle_of(row), origin_device="d_" + "f" * 32)
    assert summary["installed"] == []
    assert [entry["name"] for entry in summary["refused"]] == ["host-leak"]
    assert "github-token" in summary["refused"][0]["reason"]
    assert AgentRegistry(bare).list_agents() == []


def test_two_concurrent_applies_keep_both_rows_and_both_index_entries(tmp_path: Path) -> None:
    """MAJOR 3: the index is a read-modify-write, served off a multi-worker pool.

    Measured before the lock: twenty rounds of two concurrent applies landed BOTH rows
    on disk while the index recorded ONE — after which the losing row's true origin can
    never update it again (the rule reads "that row was authored here"), a permanent
    state with no verb that clears an entry. Six rounds here, because the failure is
    statistical and a single pair can interleave benignly.
    """
    import threading

    left = _root(tmp_path / "left")
    right = _root(tmp_path / "right")
    bare = _root(tmp_path / "bare")
    names = [f"row-{index:02d}" for index in range(6)]
    for name in names:
        _make_agent(left, f"{name}-a", prompt="A")
        _make_agent(right, f"{name}-b", prompt="B")

    left_bundle = definitions.local_bundle(left)
    right_bundle = definitions.local_bundle(right)
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def _apply(bundle: dict[str, Any]) -> None:
        try:
            barrier.wait(10)
            definitions.apply_bundle(bare, bundle, origin_device=str(bundle["origin_device"]))
        except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
            failures.append(exc)

    threads = [
        threading.Thread(target=_apply, args=(left_bundle,), daemon=True),
        threading.Thread(target=_apply, args=(right_bundle,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not failures, f"an apply raised: {failures[0]!r}"

    installed = {row.name for row in AgentRegistry(bare).list_agents()}
    for name in names:
        assert name + "-a" in installed and name + "-b" in installed, sorted(installed)
    index = definitions.read_index(bare)
    assert set(index["agents"]) == {f"{name}-{side}" for name in names for side in ("a", "b")}


def test_registry_has_id_does_not_read_a_registry_fault_as_an_absent_id() -> None:
    """NIT 3: ``except (KeyError, Exception)`` is just ``except Exception``.

    A registry that cannot answer "is this id taken?" must not be heard as "the id is
    free": that installed a mirror over a row this device could not see.
    """

    class _Broken:
        def get_agent(self, agent_id: str) -> object:
            raise OSError("agents dir unreadable")

    class _Empty:
        def get_agent(self, agent_id: str) -> object:
            raise KeyError(agent_id)

    assert definitions.registry_has_id(_Empty(), "x") is False
    with pytest.raises(OSError):
        definitions.registry_has_id(_Broken(), "x")


# ---------------------------------------------------------------------------
# The read path, and the cadence's two floors (review round 2)
# ---------------------------------------------------------------------------

#: A peer device id for the syncer. Synthetic, and shaped like the real ones.
_SYNC_PEER = "d_" + "9" * 32


def _plant_legacy_row(root: Path, agent_id: str) -> Path:
    """A row exactly as a pre-#643 import left it: its own id, preserved.

    Built by renaming a REAL row and rewriting its ``id``, so the metadata shape is
    the product's own (``import_agent`` preserved an archive's id until 2026-09-05,
    and a hand-edited row can look the same). The id here is deliberately one that
    is a legal directory name but not a conforming id: that is the population the
    read path has to keep serving.
    """
    import yaml

    registry = AgentRegistry(root)
    agent = registry.create_agent(_edit_fields(name="legacy-source", description=""))
    source = root / "agents" / str(agent.id)
    target = root / "agents" / agent_id
    source.rename(target)
    payload = yaml.safe_load((target / "agent.yml").read_text(encoding="utf-8"))
    payload["id"] = agent_id
    (target / "agent.yml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return target


def test_a_pre_existing_non_conforming_row_still_lists_and_keeps_creates_working(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """REVIEW ROUND 2 (MAJOR): the id rule was on the READ path, and it took the registry.

    Measured before this fix: a device holding one row whose id predates the current
    rule lost that row from ``list_agents`` AND flipped the registry to incomplete, so
    ``require_complete_metadata`` — the FIRST call ``resolve_create_identity`` makes —
    raised ``ProfileRegistryUnavailable`` and an ordinary create against that device
    failed with "the agent registry could not be read completely…". Nothing unsafe
    happened, which is why it was a major: an affected install could not receive a
    session at all, and was told to repair definitions that were fine before the
    upgrade.

    Three facts, which are the regression's whole shape: the row LISTS, the registry
    stays COMPLETE, and a create's identity resolution still works. The rig proves the
    same thing end to end over a real link, with a real create.
    """
    root = _root(tmp_path / "root")
    _make_agent(root, "reviewer", prompt="A row written by the current build.")
    _make_agent(root, "coder", prompt="Another one.")
    _plant_legacy_row(root, "legacy row")

    with caplog.at_level("WARNING"):
        registry = AgentRegistry(root)
        rows = registry.list_agents()
    ids = sorted(str(row.id) for row in rows)
    names = sorted(row.name for row in rows)
    # LISTED, under the id that is on disk: that id is the offending value, and it is
    # the registry's key (dropping it was the regression).
    assert "legacy row" in ids, ids
    assert names == ["coder", "legacy-source", "reviewer"], names

    # NOT INCOMPLETE. This call is what a create makes first; before the fix it raised
    # ``ProfileRegistryUnavailable``.
    registry.require_complete_metadata()

    # ...and the offending id is SURFACED, by name, with the remedy — a repair
    # suggestion rather than a fault.
    warnings = [str(record.message) for record in caplog.records if record.levelname == "WARNING"]
    assert any("legacy row" in message and "mirror" in message for message in warnings), warnings

    # A create's own resolution works with the legacy row present.
    identity, refusal = definitions.resolve_create_identity(
        root, profile="", agent_name="reviewer", agent_id="", team_name="", effort=""
    )
    assert identity is not None, refusal


def test_a_new_row_cannot_be_saved_under_a_non_conforming_id(tmp_path: Path) -> None:
    """The boundary rule, at the sink every writer goes through.

    The strict rule did not move when the read rule was widened: a NEW directory is
    only ever created for a conforming id, so a sender-chosen id still cannot become a
    path (BLOCKER 1's fix). A row that is ALREADY on disk is the opposite case — it must
    stay rewritable, or a legacy row could be read but never edited or renamed.
    """
    from local_operator.agents import AgentData

    root = _root(tmp_path / "root")
    registry = AgentRegistry(root)
    # Built through ``model_validate`` like every other row in this module: the model
    # ACCEPTS this id (it has to — see the read-rule cell), so the refusal has to come
    # from the write boundary, which is what this cell pins.
    row = AgentData.model_validate(
        {
            "id": "legacy row",
            "name": "x",
            "created_date": "2026-01-01T00:00:00+00:00",
            "version": "1.0.0",
        }
    )
    with pytest.raises(ValueError):
        registry.save_agent(row)
    assert not (root / "agents" / "legacy row").exists()
    # And no phantom row: the refusal leaves the registry as it found it.
    assert list(AgentRegistry(root).list_agents()) == []

    _plant_legacy_row(root, "legacy row")
    # Looked up the way a caller does — by NAME — while the row's id is the legacy one.
    loaded = AgentRegistry(root).get_agent_by_name("legacy-source")
    assert loaded is not None and str(loaded.id) == "legacy row", loaded
    # A rewrite of the row that is already there is allowed (this is what a rename or a
    # metadata edit does), and it lands in the same directory.
    AgentRegistry(root).save_agent(loaded)
    assert (root / "agents" / "legacy row" / "agent.yml").exists()


def test_a_failed_push_is_retried_on_the_next_tick_not_a_minute_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REVIEW ROUND 2 (minor): one floor for both outcomes parked a recovered peer.

    A single failed dial used to consume the member's whole ``STATE_MIN_INTERVAL_S``
    window, so a peer that came back a second later waited ~59 s for nothing. The
    failure floor is now the syncer's own tick; a SUCCESS still parks the member for
    the full 60 s, which is what bounds the traffic. The shipped tick is asserted here
    too, because the docstring claims a number and this feature keeps being caught on
    docstrings that promise one cadence and deliver another.
    """
    from types import SimpleNamespace

    root = _root(tmp_path / "root")
    syncer = definitions.DefinitionsSyncer(SimpleNamespace(root=root))  # type: ignore[arg-type]
    assert syncer._tick_s == 15.0, syncer._tick_s  # noqa: SLF001 — the shipped number
    monkeypatch.setattr(syncer, "_targets", lambda: [_SYNC_PEER])
    responses = [
        {"ok": False, "code": "unreachable", "message": "not answering"},
        {"ok": True, "code": "in_sync", "message": "same definitions"},
        {"ok": True, "code": "in_sync", "message": "same definitions"},
    ]
    monkeypatch.setattr(
        definitions, "push_to_peer", lambda server, device_id, **fields: responses.pop(0)
    )

    assert syncer.tick(now=1000.0) == [(_SYNC_PEER, "unreachable")]
    # 16 s later: past the FAILURE floor (one tick) and far short of the probe floor.
    assert syncer.tick(now=1016.0) == [(_SYNC_PEER, "in_sync")]
    # 14 s after that: a success parks the member for the full minute.
    assert syncer.tick(now=1030.0) == []
    assert syncer.tick(now=1080.0) == [(_SYNC_PEER, "in_sync")]


def test_a_refused_push_is_parked_for_half_an_hour_not_retried_every_tick(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A POLICY REFUSAL IS NOT A FLAP (QA round 1, Q-R1-3).

    Measured on a real pair: 300 s of idle network wrote 20 rows, every one of them
    the 15 s retry of an op the peer had refused — on both sides, forever. The answer
    to "may I write definitions here" does not change between ticks, so a refusal
    parks the member; a FAILURE keeps the fast retry, because a peer that is briefly
    down is exactly the case the one-tick floor exists for. Both halves are asserted
    here, on the same syncer, so the two cannot be collapsed back into one floor.
    """
    from types import SimpleNamespace

    syncer = definitions.DefinitionsSyncer(SimpleNamespace(root=root))  # type: ignore[arg-type]
    monkeypatch.setattr(syncer, "_targets", lambda: [_SYNC_PEER])
    responses = [
        {"ok": False, "code": "unreachable", "message": "not answering"},
        {"ok": False, "code": "refused", "message": "may not do that"},
        {"ok": False, "code": "unreachable", "message": "not answering"},
        {"ok": True, "code": "in_sync", "message": "same definitions"},
    ]
    monkeypatch.setattr(
        definitions, "push_to_peer", lambda server, device_id, **fields: responses.pop(0)
    )

    # A transport failure: retried on the next tick, exactly as before.
    assert syncer.tick(now=1000.0) == [(_SYNC_PEER, "unreachable")]
    assert syncer.tick(now=1016.0) == [(_SYNC_PEER, "refused")]
    # The refusal parks it: every tick in the next half hour asks nobody, which is
    # what turns 240 refusal rows an hour into two.
    assert syncer.tick(now=1032.0) == []
    assert syncer.tick(now=2000.0) == []
    # Half an hour later it asks again — a re-attempt, not a permanent skip, so a
    # capability granted on the peer's side is discovered rather than hidden.
    assert syncer.tick(now=1016.0 + definitions.REFUSED_MIN_INTERVAL_S) == [
        (_SYNC_PEER, "unreachable")
    ]
    assert syncer.tick(now=1016.0 + definitions.REFUSED_MIN_INTERVAL_S + 16.0) == [
        (_SYNC_PEER, "in_sync")
    ]


def test_a_member_that_cannot_hold_the_op_is_not_asked_on_a_timer(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CAUSE, not the symptom: don't ask what the shared table already answers.

    ``net_definitions`` requires ``admin`` on the receiving side, so a ``drive``
    member's push can never succeed — and the cadence asked anyway, every 15 s,
    forever, costing a dial, two envelopes and an audited refusal on the peer each
    time (measured: +21 audit rows and +8,805 bytes in 300 idle seconds, 20 of them
    this op). The requirement is read from ``types.OP_CAPABILITY`` — the same table
    the receiving authoriser reads — so this cannot drift from the real rule.
    """
    record = types.NetworkRecord(
        network_id="n_0123456789abcdef01234567",
        name="home-net",
        epoch=1,
        self_device_id="d_" + "a" * 32,
        self_role="drive",
        self_capabilities=sorted(types.capabilities_for_role("drive")),
    )
    relay.admit(
        record,
        device_id="d_" + "a" * 32,
        public_key=wire.b64u(b"a" * 32),
        name="laptop",
        role="drive",
        capabilities=sorted(types.capabilities_for_role("drive")),
        added_by="d_" + "c" * 32,
        added_via="invite",
        root=root,
        persist=False,
    )
    relay.admit(
        record,
        device_id=_SYNC_PEER,
        public_key=wire.b64u(b"b" * 32),
        name="peer",
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by="d_" + "c" * 32,
        added_via="invite",
        root=root,
        persist=False,
    )
    store.save(record, root)
    from types import SimpleNamespace

    syncer = definitions.DefinitionsSyncer(SimpleNamespace(root=root))  # type: ignore[arg-type]
    monkeypatch.setattr(syncer, "_targets", lambda: [_SYNC_PEER])
    asked: list[str] = []
    monkeypatch.setattr(
        definitions,
        "push_to_peer",
        lambda server, device_id, **fields: (
            asked.append(device_id) or {"ok": True, "code": "in_sync", "message": ""}
        ),
    )

    assert syncer.tick(now=1000.0) == [(_SYNC_PEER, "skipped:no_admin")]
    assert syncer.tick(now=2000.0) == [(_SYNC_PEER, "skipped:no_admin")]
    assert asked == [], "the cadence asked a peer it cannot possibly satisfy"

    # AND THE SKIP IS NOT A LOCK-OUT: the same member, once THIS device holds the
    # capability, is asked on the very next tick. The check reads the live record
    # rather than caching, so a `member grant` here takes effect immediately.
    with store.mutate(record.network_id, root) as live:
        live.self_capabilities = sorted(types.capabilities_for_role("admin"))
        store.save(live, root)
    assert syncer.tick(now=2016.0) == [(_SYNC_PEER, "in_sync")]
    assert asked == [_SYNC_PEER]

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

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.network import definitions
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        AgentEditFields(
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

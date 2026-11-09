"""The CONDITIONAL placement pull: an idle mesh stops re-sending an unchanged document.

WHY THIS FILE EXISTS (measured on a loopback six-member mesh, 2026-11-08). After the
member-table pull was backed off (#3648) the largest remaining idle cost was
``net_broker kind=placement``: every device's cadence asks each peer for its placement
document, and the owner answered a full 1.3-1.4 KB document whose content was "nothing
changed" — 57 % of what an idle six-member mesh put on the wire. The member-table pull
solved the same problem with a digest (``wire.MEMBER_LIST_DIGEST_V1``); this is the
placement document's turn (``wire.CREDENTIAL_PLACEMENT_DIGEST_V1``).

THE SAFETY CONTRACT, and the reason the invariant cell below exists rather than only a
wire cell: ``owner_rows_digest`` must satisfy "equal digests imply the merge changes
nothing". Everything else is allowed to be conservative — a digest that never matches
costs exactly the full document the pull sent before this shipped.

The wire cells drive the product's own leg-1 op over two REAL relays (the
``test_credentials_real_link`` fixture, imported): nothing here builds a broker frame
by hand, so a field the shipped client never sends would fail here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import wire
from local_operator.network.credentials import placement as placement_mod
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures by import

OWNER = "d_00000000000000000000000000000011"
PEER = "d_00000000000000000000000000000012"
THIRD = "d_00000000000000000000000000000013"
OTHER_PEER = "d_00000000000000000000000000000014"
NETWORK = "n_placement_digest_test"


def _document(root: Path, *, owner: str = OWNER) -> placement_mod.PlacementDocument:
    return placement_mod.PlacementDocument(NETWORK, root=root, written_by=owner)


def _declared(
    root: Path, key: str = "openai", *, owner: str = OWNER
) -> placement_mod.PlacementDocument:
    document = _document(root, owner=owner)
    document.declare(
        key,
        owner_device=owner,
        owner_device_name="peer-laptop",
        provider=key,
        identity_label=f"{owner}@example.test",
        by=owner,
    )
    return document


def _copy_on_a_peer(
    root: Path, owner_doc: placement_mod.PlacementDocument, *, self_device: str = PEER
) -> placement_mod.PlacementDocument:
    """What a peer holds after one full pull: the owner's rows merged into its file.

    The owner's document is PASSED IN, never rebuilt: ``declare`` stamps ``declared_at``
    from the clock, so two "identical" documents built separately are not the same bytes
    and the digest comparison below would be comparing two creations.
    """
    peer = _document(root, owner=THIRD)
    peer.merge(owner_doc.to_json(), from_device=OWNER, self_device=self_device)
    return peer


# ---------------------------------------------------------------------------
# The invariant: equal digests mean the merge changes nothing
# ---------------------------------------------------------------------------


def test_equal_digests_mean_the_merge_changes_nothing(tmp_path: Path) -> None:
    """THE safety contract of the whole feature, checked directly.

    A peer that has pulled once hashes the same value the owner does, and the merge it
    would run on the next pull takes nothing — which is exactly what the owner promises
    when it answers ``unchanged``.
    """
    owner_doc = _declared(tmp_path / "owner")
    peer_doc = _copy_on_a_peer(tmp_path / "peer", owner_doc)

    owner_digest = placement_mod.owner_rows_digest(owner_doc, OWNER)
    peer_digest = placement_mod.owner_rows_digest(peer_doc, OWNER)
    assert owner_digest == peer_digest, "a settled pair must hash alike"
    assert peer_doc.merge(owner_doc.to_json(), from_device=OWNER, self_device=PEER) == []


def test_a_real_change_moves_the_digest_and_still_crosses(tmp_path: Path) -> None:
    """The other half: nothing the merge WOULD adopt may hide behind an equal digest."""
    owner_doc = _declared(tmp_path / "owner")
    peer_doc = _copy_on_a_peer(tmp_path / "peer", owner_doc)
    assert placement_mod.owner_rows_digest(peer_doc, OWNER) == placement_mod.owner_rows_digest(
        owner_doc, OWNER
    )

    # A new holder is a new row revision: the digest must move, and the merge must take it.
    owner_doc.grant("openai", PEER, scope="device", by=OWNER)
    assert placement_mod.owner_rows_digest(owner_doc, OWNER) != placement_mod.owner_rows_digest(
        peer_doc, OWNER
    )
    assert peer_doc.merge(owner_doc.to_json(), from_device=OWNER, self_device=PEER) == ["openai"]


def test_the_digest_covers_only_the_rows_one_device_owns(tmp_path: Path) -> None:
    """A document holds rows several devices own; each owner is authoritative for its own.

    If the digest covered the whole file, every pull between two peers whose files hold
    rows from a THIRD device would differ for ever and the saving would never arrive —
    and a digest that moved when a third device edited its own row would be answering a
    question nobody asked.
    """
    root = tmp_path / "shared"
    document = _declared(root)
    document.grant("openai", PEER, scope="device", by=OWNER)
    before = placement_mod.owner_rows_digest(document, OWNER)

    # A second owner declares its own key in the same file: rows OWNER does not own.
    other = _declared(tmp_path / "other", key="deepseek", owner=OTHER_PEER)
    document.merge(other.to_json(), from_device=OTHER_PEER, self_device=OWNER)

    assert placement_mod.owner_rows_digest(document, OWNER) == before
    assert placement_mod.owner_rows_digest(document, OTHER_PEER) != before


def test_a_document_the_peer_owns_nothing_of_hashes_to_the_empty_set(tmp_path: Path) -> None:
    """The common idle case on a mesh where nobody has shared anything yet."""
    empty = _document(tmp_path / "empty")
    assert placement_mod.owner_rows_digest(empty, OWNER) == placement_mod.owner_rows_digest(
        _document(tmp_path / "empty2"), OWNER
    )


def test_the_capability_is_advertised_exactly_once() -> None:
    assert wire.LINK_CAPABILITIES.count(wire.CREDENTIAL_PLACEMENT_DIGEST_V1) == 1
    # A DISTINCT string from the push features, on the rule that a shipped capability
    # is never reused for a new frame kind.
    assert wire.CREDENTIAL_PLACEMENT_DIGEST_V1 not in {
        wire.CREDENTIAL_PLACEMENT_PUSH_V1,
        wire.CREDENTIAL_SHARE_PUSH_V1,
        wire.CREDENTIAL_LIFECYCLE_V1,
    }


# ---------------------------------------------------------------------------
# On the wire: two real relays, the shipped op
# ---------------------------------------------------------------------------


def _mesh(request: pytest.FixtureRequest) -> Any:
    """The real-link rig, resolved BY NAME.

    ``getfixturevalue`` rather than a parameter named ``rig``: a parameter would shadow
    the module-level import that makes the fixture reachable at all (the convention these
    real-link modules follow, and what flake8's F811 is pointing at when they do not).
    """
    return request.getfixturevalue("mesh")


def _owner_declares_directly(rig: Any, key: str) -> None:
    """Add a row on the OWNER's root without the share push.

    ``lop network credential share`` also pushes the new row to the holder (so the
    holder is already up to date before it pulls and the digest legitimately matches).
    This cell needs the other direction — the owner's document moving while the holder
    has not seen it — so the row is declared the way the owner's own store does it.
    """

    with placement_mod.mutate(rig.network_id, rig.a.root, self_device=rig.owner) as document:
        document.declare(
            key,
            owner_device=rig.owner,
            owner_device_name=rig.a.identity.name,
            provider=key,
            identity_label="owner@example.test",
            by=rig.owner,
        )


def test_a_settled_pull_answers_unchanged_instead_of_the_document(
    request: pytest.FixtureRequest,
) -> None:
    """The saving itself: a pull of an unchanged document writes and sends nothing.

    A fresh pair of relays share no placement rows, so both ends hash the empty set and
    the first pull already earns the status answer — the common idle case on a rig
    where nobody has shared a credential yet, which is the state the byte table was
    measured in before this existed.
    """
    rig = _mesh(request)
    # A share is what gives this device the broker capability its pull needs (a
    # ``drive`` member cannot even ask before one), and it is also the only way an
    # owner has a row to be unchanged ABOUT — so the cell starts from one.
    _share(rig)
    first = _pull(rig)
    assert first.get("owners") == 1, first
    # The first pull may still carry the document (the share's push and this pull race
    # for the same row); what it may never do is call itself ``unchanged`` before the
    # peer has agreed about a digest.
    assert first.get("changed") in ([], [STUB_PROVIDER]), first

    second = _pull(rig)
    assert second.get("unchanged") == [rig.owner], second
    assert second.get("changed") == []
    assert second.get("skipped") is None


def test_a_document_that_moved_still_crosses(request: pytest.FixtureRequest) -> None:
    """No change may be withheld behind a digest: a new owner row gets the full answer."""
    rig = _mesh(request)
    _share(rig)
    _pull(rig)
    assert _pull(rig).get("unchanged") == [rig.owner]

    _owner_declares_directly(rig, "newshare")
    after = _pull(rig)
    assert after.get("unchanged") is None, after
    assert "newshare" in after.get("changed", [])


def test_a_peer_without_the_capability_is_still_asked_the_old_way(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MIXED VERSIONS, the member's half: an older owner never sees ``if_digest``.

    The build that predates the digest has no handler for the field, so the frame must
    not carry it. The discriminating assertion is that the answer is a DOCUMENT: the
    owner answers ``unchanged`` only to a digest, so ``unchanged`` absent proves the
    conditional field never crossed.
    """
    rig = _mesh(request)
    # The link is dialled on demand, so it exists only after a pull has run.
    _pull(rig)
    stripped = tuple(c for c in wire.LINK_CAPABILITIES if c != wire.CREDENTIAL_PLACEMENT_DIGEST_V1)
    monkeypatch.setattr(wire, "LINK_CAPABILITIES", stripped)
    links = [link for link in (rig.b.links or {}).values()]
    assert links, "the borrower holds no link to the owner"
    for link in links:
        link.capabilities = frozenset(
            c for c in link.capabilities if c != wire.CREDENTIAL_PLACEMENT_DIGEST_V1
        )

    pulled = _pull(rig)
    assert "unchanged" not in pulled, pulled
    assert pulled.get("owners") == 1


def test_an_unchanged_answer_about_another_document_is_not_believed(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer's echo is checked against what we asked: a wrong digest is named, not merged."""
    rig = _mesh(request)
    from local_operator.network.credentials.client import MeshCredentialClient

    def _forged(self: Any, owner: str, frame: dict[str, Any]) -> dict[str, Any]:
        assert frame.get("if_digest"), "the asker must have quoted a digest to reach this arm"
        return {"op": "ack", "detail": {"kind": "placement", "unchanged": True, "digest": "0" * 64}}

    monkeypatch.setattr(MeshCredentialClient, "_ask_owner_directly", _forged)
    pulled = _pull(rig)
    assert pulled.get("unchanged") is None, pulled
    assert pulled.get("changed") == []
    assert pulled.get("skipped") == [
        {"device": rig.owner, "reason": "bad_placement_digest"}
    ], pulled


def test_a_share_still_reaches_the_borrower_with_the_digest_in_place(
    request: pytest.FixtureRequest,
) -> None:
    """The user-facing path is unchanged: sharing still lands a row on the borrower."""
    rig = _mesh(request)
    _share(rig)
    pulled = _pull(rig)
    assert pulled.get("owners") == 1
    document = placement_mod.PlacementDocument.load(
        rig.network_id, rig.b.root, self_device=rig.borrower
    )
    assert any(entry.owner_device == rig.owner for entry in document.entries.values())


@pytest.mark.parametrize(
    "detail",
    [
        {"kind": "placement", "unchanged": "yes"},
        {"kind": "placement", "unchanged": 1},
        {"kind": "placement"},
        {},
        "unchanged",
    ],
)
def test_a_malformed_status_is_named_and_nothing_is_merged(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, detail: Any
) -> None:
    """QA round 1, Q1: failing safe is not enough — the operator gets a reason.

    ``unchanged: "yes"`` (or a ``detail`` that is not a mapping at all) is never believed,
    and nothing is merged, but before this the pull reported ``owners: 1, changed: []``
    with no explanation while a WRONG digest in a well-shaped frame was named. Both are
    now named; this is the malformed-status arm.
    """
    rig = _mesh(request)
    from local_operator.network.credentials.client import MeshCredentialClient

    def _malformed(self: Any, owner: str, frame: dict[str, Any]) -> dict[str, Any]:
        assert frame.get("if_digest"), "the asker must have quoted a digest to reach this arm"
        return {"op": "ack", "detail": detail}

    monkeypatch.setattr(MeshCredentialClient, "_ask_owner_directly", _malformed)
    pulled = _pull(rig)
    assert pulled.get("unchanged") is None, pulled
    assert pulled.get("changed") == []
    assert pulled.get("skipped") == [
        {"device": rig.owner, "reason": "bad_placement_status"}
    ], pulled


def test_a_full_answer_carries_no_digest(request: pytest.FixtureRequest) -> None:
    """Review round 1, M2: the digest has no reader on the full answer, so it is not sent.

    An asker that receives a document recomputes the digest from its own merged copy, so
    the field was ~80 B on every full answer — including every answer to a peer that
    predates the capability, i.e. every answer on a mixed fleet, in the message this PR
    exists to shrink. This cell reads the actual reply frame off the real link.
    """
    rig = _mesh(request)
    _share(rig)
    _pull(rig)
    # A row the borrower has never seen, declared on the owner, is what makes the next
    # answer a full document rather than a status.
    _owner_declares_directly(rig, "m2probe")

    seen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    links = [link for link in (rig.b.links or {}).values()]
    assert links, "the borrower holds no link to the owner"
    for link in links:
        original = link.request

        def capture(frame: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            reply = _original(frame, *args, **kwargs)
            if isinstance(frame, dict) and frame.get("kind") == "placement":
                seen.append((frame, reply))
            return reply

        link.request = capture

    pulled = _pull(rig)
    assert "m2probe" in pulled.get("changed", []), pulled
    assert seen, "no placement exchange was observed on the link"
    frame, reply = seen[-1]
    assert frame.get("if_digest"), "the conditional ask must have crossed for this cell"
    detail = (reply or {}).get("detail")
    assert isinstance(detail, dict) and isinstance(detail.get("document"), dict), reply
    assert "digest" not in detail, "the full answer still carries an unread digest"


def test_a_no_answer_is_not_reported_as_a_bad_status(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer that says nothing keeps the treatment it always had.

    ``_ask_owner_directly`` returns ``None`` for a dead link or a timeout. That is not a
    malformed answer, so it must not be reported under a reason word that says "status":
    nothing answered, so there is nothing to describe, and the cadence's own offline
    handling owns it. The boundary of Q1's arm is pinned here.
    """
    rig = _mesh(request)
    from local_operator.network.credentials.client import MeshCredentialClient

    def _silent(self: Any, owner: str, frame: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(MeshCredentialClient, "_ask_owner_directly", _silent)
    pulled = _pull(rig)
    assert pulled.get("owners") == 1, pulled
    assert pulled.get("skipped") is None, pulled
    assert pulled.get("refused_by") is None, pulled

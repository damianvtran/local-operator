"""The both-ends join order: what will be shared, shown before anyone confirms.

THE PROPERTY THIS FILE PINS. A pairing used to be a code both humans compared and
nothing else: the joiner learned what it could borrow only after admission, and the
owner had no screen at all. The design (moment: the both-ends join screen) adds one
sealed frame — ``net_pair_offer`` — as the FIRST record after ``welcome``, gated on
a capability both sides must advertise (``pair-offer-v1``), so an old peer never
sees it and neither protocol version moves. The offer is display data: the owner
may only REDUCE the list it sent (``PairDecision.shares``), the grants are written
at admission from the owner's own store, and a refused/widened decision admits
nothing extra.

WHERE THE CELLS RUN. The module cells here exercise ``credentials/offers.py``
directly; the relay-side cells drive the real ``_run_pair_listener`` pair path and
the CLI's ``_join_one`` over real loopback sockets (the ``devices`` fixture), the
way ``test_join_park.py`` drives the two-phase pair. A cell that needs an "old
build" simulates it the only honest way available in-repo: by removing the
capability the old build would not have advertised, from the ONE tuple both
handshake paths read (``wire.LINK_CAPABILITIES``).

Isolation: every cell runs against the pytest ``root`` config dir (``conftest``),
an AuthStore is only ever opened under it, and nothing here touches the operator's
live config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import offers

# ---------------------------------------------------------------------------
# Seeding helpers — a root with a real auth.db (and optionally mcp.json)
# ---------------------------------------------------------------------------


def _store(root: Path) -> Any:
    """A real AuthStore at ``root`` (explicit db_path: never the ambient home)."""
    from local_operator.providers.auth_store import AuthStore

    return AuthStore(db_path=root / "auth.db", config_dir=root)


def _seed(root: Path, provider: str, payload: dict[str, Any]) -> None:
    auth = _store(root)
    try:
        auth.upsert_credential(provider, payload)
    finally:
        auth.close()


def _seed_mcp_json(root: Path, servers: dict[str, dict[str, Any]]) -> None:
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


# ---------------------------------------------------------------------------
# The candidate list (memo §7 cell 2), masking, defaults
# ---------------------------------------------------------------------------


def test_offer_carries_candidates_kinds_labels_defaults(root: Path) -> None:
    """One row per serveable credential, each classified and defaulted per §2.3.

    Seeded: an OAuth provider login (default YES), a pasted static key (default
    NO), a device-bound provider (excluded by name — a grant would be refused),
    the Radient org login (never auto-offered — §1.1 "there is no join-time
    default"), an MCP login (its kind is conservative in v1) and an MCP server
    with NO login row (excluded — offering it would promise a row the admission
    re-check drops).
    """
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    _seed(root, "anthropic", {"key": "sk-ant-1", "type": "api_key"})
    _seed(root, "kimi", {"key": "sk-kimi", "type": "api_key"})
    _seed(root, "radient", {"refresh": "r2", "access": "a2", "email": "ops@corp.example"})
    _seed(
        root,
        "mcp-oauth",
        {"project_id": "https://mcp.example/sse", "refresh": "r3", "access": "a3"},
    )
    _seed_mcp_json(
        root,
        {
            "tools": {"type": "sse", "url": "https://mcp.example/sse"},
            "cold": {"type": "http", "url": "https://cold.example/mcp"},
            "local": {"command": "npx", "args": ["-y", "something"]},
        },
    )

    items = offers.build_items(root)

    assert items == [
        {
            "key": "anthropic",
            "kind": "api-key-static",
            "label": "",
            "share": False,
        },
        {
            "key": "mcp:https://mcp.example/sse",
            "kind": "mcp-rotating",
            "label": "",
            "share": False,
        },
        {
            "key": "openai",
            "kind": "oauth-rotating",
            "label": "d***@example.com",
            "share": True,
        },
    ]
    # The exclusions are pinned BY NAME so a future filter change re-reads the
    # reason comment rather than flipping silently.
    keys = [item["key"] for item in items]
    assert "kimi" not in keys and "radient" not in keys
    assert not any(key.startswith("mcp:https://cold.example") for key in keys)


def test_offer_masks_labels_and_never_carries_a_raw_identity(root: Path) -> None:
    """The label's whole job is "which of my two logins is this" — masked.

    The frame carries the label MASKED (the memo's §5.1 label row: shown masked,
    never in the audit), so the raw address must not be findable in the wire
    bytes the joiner's screen renders from.
    """
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "damian@gmail.com"})
    items = offers.build_items(root)
    assert items[0]["label"] == "d***@gmail.com"
    assert "damian@gmail.com" not in json.dumps(items)

    assert offers.mask_label("") == ""
    assert offers.mask_label("abc123") == "a***"
    assert offers.mask_label("a@b") == "a***@b"


def test_offer_bounded_sorted_digest_matches_canonical_items(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size, sort and digest: the three pins that make two implementations agree.

    A hostile or broken store cannot strain the record budget (the cap), the list
    is sorted by key (a stable digest input), and the digest is sha256 over the
    canonical serialisation of exactly the items sent.
    """
    from local_operator.network import wire

    many = [
        {"key": f"provider-{index:03d}", "kind": "api-key-static", "label": ""}
        for index in range(offers.MAX_OFFER_ITEMS + 6)
    ]
    monkeypatch.setattr(offers, "enumerate_candidates", lambda config: list(reversed(many)))

    items = offers.build_items(root)
    assert len(items) == offers.MAX_OFFER_ITEMS
    assert [item["key"] for item in items] == sorted(item["key"] for item in items)

    digest = offers.digest_of(items)
    assert digest == hashlib.sha256(wire.canonical_json(items)).hexdigest()
    # One changed byte in one row re-pins the digest.
    mutated = [dict(item) for item in items]
    mutated[-1]["share"] = not mutated[-1]["share"]
    assert offers.digest_of(mutated) != digest

    # And a list that is exactly at the cap is kept whole.
    monkeypatch.setattr(
        offers, "enumerate_candidates", lambda config: many[: offers.MAX_OFFER_ITEMS]
    )
    assert len(offers.build_items(root)) == offers.MAX_OFFER_ITEMS


def test_offer_validation_refuses_unusable_items() -> None:
    """A frame the parser cannot trust is refused, never shown partially."""
    good = {"key": "openai", "kind": "oauth-rotating", "label": "d***@x", "share": True}
    assert offers.validate_items([good]) == [good]

    for bad in (
        "not a list",
        [good] * (offers.MAX_OFFER_ITEMS + 1),
        [{"key": "", "kind": "oauth-rotating", "label": "", "share": True}],
        [{"key": "x", "kind": "secret", "label": "", "share": True}],
        [{"key": "x", "kind": "oauth-rotating", "label": "", "share": "yes"}],
        [
            {
                "key": "x",
                "kind": "oauth-rotating",
                "label": "x" * (offers.MAX_OFFER_LABEL + 1),
                "share": True,
            }
        ],
        [["nope"]],
    ):
        with pytest.raises(ValueError):
            offers.validate_items(bad)


def test_offer_reads_degrade_by_absence_and_raise_by_unreadability(root: Path) -> None:
    """No store is an honest empty offer; an unreadable store is SAYABLE.

    The two must not collapse: ``_pair_offer_for`` sends an empty offer either
    way, but only the unreadable case records ``enumeration: "unreadable"``, so
    "why was my list empty" has an answer.
    """
    assert offers.build_items(root) == []

    (root / "auth.db").mkdir()
    with pytest.raises(offers.OfferEnumerationError):
        offers.build_items(root)


# ---------------------------------------------------------------------------
# The rendered lines (one contract, four surfaces)
# ---------------------------------------------------------------------------


def test_offer_lines_and_states_are_one_contract() -> None:
    """The words both humans read, exactly — they are a contract, not formatting.

    §3.1's joiner block, §3.3's owner block, and §4.3/§4.4's two DIFFERENT
    "nothing" lines ("did not offer" is about a build; "offered nothing" is
    about a device's store).
    """
    items = [
        {"key": "openai", "kind": "oauth-rotating", "label": "d***@gmail.com", "share": True},
        {"key": "anthropic", "kind": "api-key-static", "label": "", "share": False},
    ]
    assert offers.render_joiner_block(items, inviter="damian-mbp") == [
        "Credentials damian-mbp will serve to this device:",
        "  openai (OAuth, d***@gmail.com)   will be served",
        "  anthropic (API key)   not offered",
        "the inviter can remove items before admitting; nothing else will be served.",
    ]
    assert offers.render_owner_block(items, joiner="laptop") == [
        "Credentials this device will serve to laptop:",
        "  openai (OAuth, d***@gmail.com)   will be served",
        "  anthropic (API key)   not offered",
        "the list can only shrink; nothing else will be served.",
    ]
    assert offers.render_state_line(offers.OFFER_EMPTY) == "no credentials were offered"
    assert "older build" in offers.render_state_line(offers.OFFER_ABSENT)

    assert offers.served_keys(items) == ["openai"]
    assert offers.owner_default_shares(items) == ["openai"]
    assert offers.sentence_clause(offers.OFFER_LISTED, items, inviter="damian-mbp") == (
        "damian-mbp will serve: openai."
    )
    assert offers.sentence_clause(offers.OFFER_EMPTY, [], inviter="x").startswith("No credentials")
    assert offers.sentence_clause(offers.OFFER_ABSENT, [], inviter="x").startswith(
        "The other device did not offer"
    )

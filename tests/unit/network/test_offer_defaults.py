"""The §1.4 join-time defaults: approval IS the authorisation.

PINS THE FLIP, ROW BY ROW (``docs/design/mesh-consent-provisioning.md`` §1.4),
because each one is a default posture an operator now meets without asking —
and a posture nobody asserted is a posture that silently flips back:

* ``oauth-rotating`` — share, unchanged;
* ``api-key-static`` — share (was not): a static key is the class that works
  from a second device, and the approval gate plus the revoke surfaces answer
  the old "permanent capability increase" concern;
* ``mcp-rotating`` — share (was not): the access token was always brokerable;
* ``github-app`` — share WHEN the adapter resolves a source (today: the App);
* ``radient`` — offered by default for device members (was never-auto-offered),
  pool exclusion unchanged and structural.

The two exclusions that do NOT move are asserted here too: the device-bound
provider stays out of the candidate list and stays refused by name at the
document, and the host-local mobile-portal password is structurally unreachable
— it lives only in the macOS Keychain and reaches no candidate enumeration
(``tests/unit/mobile`` owns that path's own assertions).

The base harness is ``test_pair_offer.py`` (its cells and helpers); the
admission cells drive the real ``relay._grant_pair_shares`` so the flip is
asserted at the write, not only at the listing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import relay
from local_operator.network import store as net_store
from local_operator.network import types
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import offers as offers_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_pair_offer import (  # noqa: F401 — helpers by import
    _seed,
    _seed_mcp_json,
)

OWNER = "d_owner"
JOINER = "d_joiner"
NETWORK_ID = "n_offer"


def _row(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for item in items:
        if item.get("key") == key:
            return item
    raise AssertionError(f"{key!r} is not in the offer: {[i.get('key') for i in items]}")


def _seed_app_presence(root: Path) -> None:
    """A ``GITHUB_APP`` secret's EXISTENCE, which is the row's candidate gate.

    ``build_items`` asks ``app_secret_present`` (a ``describe``, never a
    retrieval), so any bytes exercise the gate; nothing here reads the value.
    """
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    master_key = load_master_key(root, create=True)
    store = SecretStore(master_key, base=root)
    store.initialize()
    store.set(github_mod.APP_SECRET_NAME, b"{}", description="candidate-gate stub")


def _owner_record(root: Path) -> Any:
    """A real owner-side record with the joiner already a member row."""
    record = types.NetworkRecord(
        network_id=NETWORK_ID,
        name="damian-mesh",
        created_by=OWNER,
        self_device_id=OWNER,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        epoch=1,
    )
    relay.admit(
        record,
        device_id=OWNER,
        public_key="k-owner",
        name="owner-mac",
        role="admin",
        added_by=OWNER,
        added_via="self",
        capabilities=sorted(types.capabilities_for_role("admin")),
        persist=False,
    )
    relay.admit(
        record,
        device_id=JOINER,
        public_key="k-joiner",
        name="cloud-node",
        role="drive",
        added_by=OWNER,
        capabilities=sorted(types.capabilities_for_role("drive")),
        persist=False,
    )
    net_store.save(record, root)
    return record


def _admit(
    root: Path, shares: list[str]
) -> tuple[list[str], list[str], placement_mod.PlacementDocument]:
    """Run admission's grant step for one decision and read the document back."""
    record = _owner_record(root)
    audit = audit_mod.AuditLog(root)
    try:
        granted, reduced = relay._grant_pair_shares(
            record,
            joiner_id=JOINER,
            decision=types.PairDecision(
                invite_id="inv_1",
                decision="admit",
                matched=True,
                answered_by="cli",
                shares=list(shares),
            ),
            offer_items=offers_mod.build_items(root),
            owner_name="owner-mac",
            root=root,
            audit=audit,
        )
    finally:
        audit.close()
    document = placement_mod.PlacementDocument.load(NETWORK_ID, root, self_device=OWNER)
    return granted, reduced, document


# ---------------------------------------------------------------------------
# The §1.4 table, one cell per row
# ---------------------------------------------------------------------------


def test_an_oauth_login_still_defaults_to_share(root: Path) -> None:
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    row = _row(offers_mod.build_items(root), "openai")
    assert row["kind"] == "oauth-rotating"
    assert row["share"] is True


def test_a_static_api_key_now_defaults_to_share(root: Path) -> None:
    """Flipped from False: the class that works from a second device."""
    _seed(root, "anthropic", {"key": "sk-ant-1", "type": "api_key"})
    row = _row(offers_mod.build_items(root), "anthropic")
    assert row["kind"] == "api-key-static"
    assert row["share"] is True


def test_an_mcp_login_now_defaults_to_share(root: Path) -> None:
    """Flipped from False: the access token was always brokerable; the refresh
    grant stays host-local, and the MCP defs push reports the server set."""
    _seed(
        root,
        "mcp-oauth",
        {"project_id": "https://mcp.example/sse", "refresh": "r3", "access": "a3"},
    )
    _seed_mcp_json(root, {"tools": {"type": "sse", "url": "https://mcp.example/sse"}})
    row = _row(offers_mod.build_items(root), "mcp:https://mcp.example/sse")
    assert row["kind"] == "mcp-rotating"
    assert row["share"] is True


def test_the_github_row_defaults_to_share_when_the_adapter_resolves_a_source(
    root: Path,
) -> None:
    """The conditional row: today's source is the App, and its presence opens
    the gate — the S2 ladder widens which devices grow the row, not the flip."""
    _seed_app_presence(root)
    row = _row(offers_mod.build_items(root), github_mod.GITHUB_KEY)
    assert row["kind"] == "github-app"
    assert row["share"] is True


def test_without_a_source_the_github_row_does_not_exist(root: Path) -> None:
    """No store, no App: the row is absent, not a row with nothing behind it."""
    items = offers_mod.build_items(root)
    assert all(item["key"] != github_mod.GITHUB_KEY for item in items)


def test_radient_is_offered_by_default_now(root: Path) -> None:
    """Flipped from never-auto-offered (§10 Q4): reduce-only, session scope,
    and the pool exclusion is structural, not a list membership."""
    _seed(root, "radient", {"refresh": "r2", "access": "a2", "email": "ops@corp.example"})
    row = _row(offers_mod.build_items(root), "radient")
    assert row["kind"] == "oauth-rotating"
    assert row["share"] is True


def test_the_defaults_table_is_closed_and_explicit() -> None:
    assert offers_mod.share_default("oauth-rotating") is True
    assert offers_mod.share_default("api-key-static") is True
    assert offers_mod.share_default("mcp-rotating") is True
    assert offers_mod.share_default("github-app") is True
    # The closed direction: an unknown kind has no default and is not offered.
    assert offers_mod.share_default("future-kind") is False


# ---------------------------------------------------------------------------
# The reduce step at admission, and the two exclusions
# ---------------------------------------------------------------------------


def test_the_reduce_step_narrows_admission_to_the_intersection(root: Path) -> None:
    """The offer defaults to the full servable set; a decision REDUCES it and
    admission grants exactly the intersection — one placement per kept key."""
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _seed(root, "anthropic", {"key": "sk-ant-1", "type": "api_key"})
    _seed(root, "radient", {"refresh": "r2", "access": "a2", "email": "ops@corp.example"})

    granted, reduced, document = _admit(root, shares=["openai"])

    assert granted == ["openai"]
    assert reduced == ["anthropic", "radient"]
    entry = document.entry("openai")
    assert entry is not None
    assert [holder.device for holder in entry.holders] == [OWNER, JOINER]
    assert document.entry("anthropic") is None
    assert document.entry("radient") is None


def test_a_github_grant_at_admission_is_device_scoped(root: Path) -> None:
    """The flip's completion: the App key is device-scoped by construction and
    ``grant`` refuses ``session`` for it by name — a session-scoped admission
    write would drop the very grant the new default promises."""
    _seed_app_presence(root)

    granted, _reduced, document = _admit(root, shares=[github_mod.GITHUB_KEY])

    assert granted == [github_mod.GITHUB_KEY]
    entry = document.entry(github_mod.GITHUB_KEY)
    assert entry is not None
    assert entry.kind == "github-app"
    assert [(holder.device, holder.scope) for holder in entry.holders] == [
        (OWNER, "device"),
        (JOINER, "device"),
    ]


def test_the_device_bound_exclusion_does_not_move(root: Path) -> None:
    """Kimi is excluded from the candidates AND refused by name at the
    document, so the flip cannot reach it through either path."""
    _seed(root, "kimi", {"key": "sk-kimi", "type": "api_key"})
    items = offers_mod.build_items(root)
    assert all(item["key"] != "kimi" for item in items)

    document = placement_mod.PlacementDocument(NETWORK_ID, root=root, written_by=OWNER)
    with pytest.raises(MeshRefusal) as caught:
        document.declare("kimi", owner_device=OWNER, provider="kimi", by=OWNER)
    assert caught.value.code == "device_bound"

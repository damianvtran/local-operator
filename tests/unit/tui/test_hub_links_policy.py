"""Round-2 design remediation regressions for the Hub links surface.

Three defects independent design review found in the first implementation, each
pinned here by its EFFECT rather than by a description:

* D7 — the policy action must WRITE (or refuse with a named reason), through the
  same service path the CLI ``lop hub link set`` and the desktop PATCH route
  use. The first version printed a stub notice naming the raw link id, so the
  row's whole purpose ("toggle Subscribe / Auto-push") did nothing.
* D9 — the footer door must be the FIRST tail fact the sidebar's ladder drops,
  so it can never be the reason a queued-ask count or the hidden-population chip
  disappears.
* D10 — a sidebar binding must not shadow an app-level binding; the removed
  ``ctrl+n`` hub chord silently took the remappable new-session hotkey.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory

HUB = "https://api.radienthq.com/v1"


def _seed_link(
    root: Path,
    *,
    local_id: str = "a1",
    name: str = "coder",
    visibility: str = "public",
    tenant: str | None = None,
    subscribe: bool = True,
    auto_push: bool = False,
    attest_revision_cas: bool = False,
) -> LinkIdentity:
    """A real baseline + status row so the inventory enumerates a link."""

    identity = LinkIdentity("agent", local_id, HUB, visibility, tenant, f"h-{local_id}")
    record = prov.make_record(
        "agent", local_id, f"h-{local_id}", tenant, {"instructions": "x"}, "pull"
    )
    record = dataclasses.replace(
        record, identity=identity, remote_revision='"rev-1"', lineage="clean", local_alias=name
    )
    prov.write_baseline(root, record)
    hub: dict[str, Any] = {
        "links": {identity.link_id: {"subscribe": subscribe, "auto_push": auto_push}}
    }
    if tenant:
        hub["tenant_names"] = {tenant: tenant.title()}
    ConfigManager(root).set_config_value("hub", hub)

    def fold(doc: dict) -> None:
        item = doc.setdefault("items", {}).setdefault(
            f"agent:{local_id}", st.blank_item("agent", local_id, name, f"h-{local_id}", tenant)
        )
        item.update(state="up-to-date", error_class=None, last_checked_at="2026-10-14T00:00:00Z")
        if attest_revision_cas:
            doc.setdefault("server", {})["revision_cas"] = True

    st.StatusStore(root).mutate(fold)
    return identity


def _policy_values(root: Path, link_id: str) -> dict[str, Any]:
    hub = ConfigManager(root).get_config().values.get("hub", {})
    return dict(hub.get("links", {}).get(link_id, {}))


@pytest.mark.asyncio
async def test_subscribe_toggle_writes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7 EFFECT: `s` changes the persisted Subscribe value (a real write)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    before = _policy_values(tmp_path, identity.link_id)["subscribe"]
    assert before is True

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = 0
        panel.action_toggle_subscribe()
        await pilot.pause()
        await pilot.pause()

    after = _policy_values(tmp_path, identity.link_id)["subscribe"]
    assert after is False, "the toggle must persist the flipped Subscribe value"


@pytest.mark.asyncio
async def test_auto_push_public_needs_consent_then_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7: a PUBLIC auto-push enable needs the two-step consent, then writes.

    The first press arms consent (no write); only the second, with
    ``confirm_egress``, writes — the deliberate public-egress acknowledgement the
    service requires.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = 0
        # First press: arm consent, NO write.
        panel.action_toggle_auto_push()
        await pilot.pause()
        assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False
        # Second press: the deliberate confirm writes.
        panel.action_toggle_auto_push()
        await pilot.pause()
        await pilot.pause()

    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is True


@pytest.mark.asyncio
async def test_auto_push_ineligible_is_refused_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7: an ineligible origin cannot be enabled, and the refusal is NAMED.

    No revision-CAS attestation is recorded, so the service says
    ``unsupported-precondition``; the toggle must NOT write, and the app must
    report that reason (not a stub naming the link id).
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, attest_revision_cas=False, auto_push=False)
    notices: list[str] = []

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        # Capture the notices the app emits.
        monkeypatch.setattr(app, "_system_notice", lambda text, *a, **k: notices.append(str(text)))
        panel = app._hub_links_panel()
        panel._selected = 0
        panel.action_toggle_auto_push()
        await pilot.pause()
        await pilot.pause()

    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False, "no write on refusal"
    assert any("unsupported-precondition" in n for n in notices), notices


@pytest.mark.asyncio
async def test_stale_policy_revision_is_a_named_cas_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7: a stale revision is refused with the CAS error, never clobbered."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True, attest_revision_cas=True)
    notices: list[str] = []

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = 0
        # Simulate a concurrent edit: the panel's revision is now stale.
        panel._rows = [
            dataclasses.replace(row, policy_revision="stale-revision") for row in panel._rows
        ]
        monkeypatch.setattr(app, "_system_notice", lambda text, *a, **k: notices.append(str(text)))
        panel.action_toggle_subscribe()
        await pilot.pause()
        await pilot.pause()

    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is True, "no clobber"
    assert any("changed" in n.lower() or "revision" in n.lower() for n in notices), notices


def test_footer_drops_the_hub_door_before_the_asks_or_chip() -> None:
    """D9: the hub door is the FIRST tail fact the footer ladder drops.

    The door is a CONTROL with two surviving routes (the footer note's press and
    ``/hub``); the ask count and the chip are FACTS discoverable from nowhere
    else on the frame. So the door must yield before either, at every width.
    Measured through the REAL ladder with all three tail facts seeded: at 29/30
    (the sidebar's content floor) the door is gone while ``asks: 4`` and the
    chip survive; at 34 the door fits again, still without displacing a fact.
    """

    from local_operator.tui.widgets.session_sidebar import SessionSidebar

    def painted(width: int, *, focused: bool) -> str:
        sidebar = SessionSidebar()
        sidebar._asks_total = 4
        sidebar._subagent_total = 4
        sidebar._hub_total = 1
        sidebar.has_focus = focused
        return sidebar._footer_line(width)[0]

    for focused in (False, True):
        # The widths where the ask note and chip survive at all (below these the
        # pre-existing ladder already falls to the bare lead, unrelated to the
        # door). Measured: unfocused keeps facts from 24, focused from 26.
        fact_widths = (26, 29, 30, 41, 43) if focused else (24, 26, 29, 30, 41, 43)
        for width in fact_widths:
            line = painted(width, focused=focused)
            # The door NEVER rides without both facts: if it is on the line, so
            # are the ask count and the chip. So the door is never the reason a
            # fact vanished.
            if "hub:" in line:
                assert "asks: 4" in line and "⌥4" in line, (width, focused, line)
        # At the floor of the fact-bearing band the door is gone while the facts
        # survive — the door yielded first.
        floor = fact_widths[0]
        line = painted(floor, focused=focused)
        assert "asks: 4" in line and "⌥4" in line, (floor, focused, line)
        assert "hub:" not in line, (floor, focused, line)
    # At 34 there is room for the door AND both facts, unfocused.
    assert painted(34, focused=False) == "f9 focus · asks: 4 · hub: 1 · ⌥4"

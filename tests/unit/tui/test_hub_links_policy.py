"""Round-2/3 policy-action regressions for the Hub links overlay.

The policy action must WRITE through the CLI/desktop ``update_policy`` path, or
refuse with a NAMED reason — never a stub, and never a write from a stray
keystroke. Every test drives the REAL app through ``pilot.press`` (never a
direct ``action_*`` call), because round 3 found the arming key `a` could be
"confirmed" by typed-ahead text and by a second `a`, which direct calls hid.
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


async def _open_panel(app: OperatorApp, pilot: Any) -> None:
    """Open the overlay the way a user does and ENGAGE the panel (past typed-ahead)."""
    app.action_hub_links()
    await pilot.pause()
    await pilot.pause()
    # One navigation key engages the panel (round-3 D15), so subsequent presses
    # are treated as intent rather than typed-ahead.
    await pilot.press("down")
    await pilot.pause()


@pytest.mark.asyncio
async def test_subscribe_press_writes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7 EFFECT via a real keypress: `s` changes the persisted Subscribe value."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is True

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()

    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is False


@pytest.mark.asyncio
async def test_private_eligible_auto_push_enables_on_single_press(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SINGLE `a` on a PRIVATE eligible link enables (no consent needed)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(
        tmp_path, visibility="org", tenant="acme", auto_push=False, attest_revision_cas=True
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is True


@pytest.mark.asyncio
async def test_public_single_a_arms_but_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single `a` on a PUBLIC link arms the prompt but never writes (D15)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        assert panel._consent_link_id == identity.link_id, "the prompt must be armed"
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("other", ["a", "s", "r", "down", "up", "x"])
async def test_arm_then_other_then_a_never_egresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    """D15: `a`, any OTHER key, `a` must NOT publish (the old bug's core).

    `a,s,a` even flipped Subscribe (a real write) while leaving consent armed,
    and a second `a` then published. ANY other key disarms, so the trailing `a`
    only re-arms — it can never complete.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # arm
        await pilot.pause()
        await pilot.press(other)  # any other key disarms
        await pilot.pause()
        await pilot.press("a")  # re-arms at most
        await pilot.pause()
        await pilot.pause()
    assert (
        _policy_values(tmp_path, identity.link_id)["auto_push"] is False
    ), f"`a`,`{other}`,`a` must not publish"


@pytest.mark.asyncio
async def test_typed_ahead_banana_never_egresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D15: typed-ahead 'banana' (two `a`s) must NOT publish.

    `/hub` takes focus while the user may still be typing; every letter of
    'banana' arrives within the settle window, so the printable-key guard
    swallows them.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app._editor()
        editor.focus()
        editor.load_text("/hub")
        await pilot.press("enter")
        for ch in "banana":
            await pilot.press(ch)
        await pilot.pause()
        await pilot.pause()
        assert app.focused.__class__.__name__ == "HubLinksPanel"
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False


@pytest.mark.asyncio
async def test_confirm_key_after_prompt_writes_with_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D15: the NAMED confirm key (`ctrl+y`) after the prompt writes the egress."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # arm the prompt
        await pilot.pause()
        # The consent block must be ON the card, naming the confirm key.
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "Publish publicly?" in painted, painted
        assert "ctrl+y" in painted, painted
        await pilot.press("ctrl+y")  # the deliberate confirm
        await pilot.pause()
        await pilot.pause()
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is True


@pytest.mark.asyncio
async def test_esc_cancels_the_prompt_not_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D15: esc cancels an armed prompt only; the overlay stays open, no write."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # arm
        await pilot.pause()
        await pilot.press("escape")  # cancel the PROMPT
        await pilot.pause()
        panel = app._hub_links_panel()
        assert panel.is_open, "esc must cancel the prompt, not close the overlay"
        assert panel._consent_link_id is None, "esc must disarm"
        # A following `a` only re-arms.
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False


@pytest.mark.asyncio
async def test_ineligible_refusal_is_named_on_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7/D19: ineligible enable is refused with a humanized reason ON the card."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, attest_revision_cas=False, auto_push=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        # The refusal is on the CARD (D19), humanized (D19), not a raw code.
        assert panel._notice, "a refusal must be shown on the card"
        assert "unsupported precondition" in panel._notice.lower(), panel._notice
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False


@pytest.mark.asyncio
async def test_stale_policy_revision_is_a_named_cas_refusal_on_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7/D19: a stale revision is refused ON the card, never clobbered."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = app._hub_links_panel()
        # Simulate a concurrent edit: the panel's revision is now stale.
        panel._rows = [
            dataclasses.replace(row, policy_revision="stale-revision") for row in panel._rows
        ]
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert panel._notice, "a CAS refusal must be shown on the card"
        assert "changed" in panel._notice.lower(), panel._notice
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is True, "no clobber"


@pytest.mark.asyncio
async def test_tui_cannot_create_egress_grant_when_an_earlier_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OBSERVATION guard: consent only completes AFTER every earlier gate.

    The TUI's ``consent_completes`` path is reachable ONLY when the service
    reason is ``unsafe-publication``, which ``_eligibility`` returns only after
    capability, baseline and lineage all pass (service.py). Here the link has
    NO revision-CAS attestation, so the reason is ``unsupported-precondition``;
    a ``ctrl+y`` confirm must therefore write nothing and create no grant.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # refused (unsupported-precondition), not armed
        await pilot.pause()
        await pilot.press("ctrl+y")  # nothing armed; must be inert
        await pilot.pause()
        await pilot.pause()
    values = _policy_values(tmp_path, identity.link_id)
    assert values["auto_push"] is False
    grants = (
        ConfigManager(tmp_path).get_config().values.get("hub", {}).get("public_egress_grants", {})
    )
    assert identity.link_id not in grants, "no egress grant may be created"


def test_footer_drops_the_hub_door_before_the_asks_or_chip() -> None:
    """D9: the hub door is the FIRST tail fact the footer ladder drops.

    The door is a CONTROL with two surviving routes (the footer note's press and
    ``/hub``); the ask count and the chip are FACTS discoverable from nowhere
    else on the frame. Measured through the REAL ladder with all three tail
    facts seeded.
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
        fact_widths = (26, 29, 30, 41, 43) if focused else (24, 26, 29, 30, 41, 43)
        for width in fact_widths:
            line = painted(width, focused=focused)
            if "hub:" in line:
                assert "asks: 4" in line and "⌥4" in line, (width, focused, line)
        floor = fact_widths[0]
        line = painted(floor, focused=focused)
        assert "asks: 4" in line and "⌥4" in line, (floor, focused, line)
        assert "hub:" not in line, (floor, focused, line)
    assert painted(34, focused=False) == "f9 focus · asks: 4 · hub: 1 · ⌥4"

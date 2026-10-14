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
    kind: str = "agent",
    local_id: str = "a1",
    name: str = "coder",
    visibility: str = "public",
    tenant: str | None = None,
    subscribe: bool = True,
    auto_push: bool = False,
    attest_revision_cas: bool = False,
) -> LinkIdentity:
    """A real baseline + status row so the inventory enumerates a link."""

    identity = LinkIdentity(kind, local_id, HUB, visibility, tenant, f"h-{local_id}")
    # Skill baselines carry a full SkillDocument (its fingerprint covers the whole
    # manifest), so a skill needs a valid one-file document, not free-form text.
    fields: dict[str, Any]
    if kind == "skill":
        fields = {
            "document_version": 1,
            "name": name,
            "description": "d",
            "files": [
                {"path": "SKILL.md", "content": f"---\nname: {name}\ndescription: d\n---\nbody\n"}
            ],
        }
    else:
        fields = {"instructions": "x"}
    record = prov.make_record(kind, local_id, f"h-{local_id}", tenant, fields, "pull")
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
            f"{kind}:{local_id}",
            st.blank_item(kind, local_id, name, f"h-{local_id}", tenant),
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

    The carve-out is reachable ONLY when the service reason is
    ``unsafe-publication``, which ``_eligibility`` returns only after
    capability, baseline and lineage all pass (service.py). Here the link has
    NO revision-CAS attestation, so the reason is ``unsupported-precondition``;
    ``a`` refuses (not arms), and a ``ctrl+y`` confirm must therefore write
    nothing and create no grant.
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


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [15, 16, 17, 18])
async def test_selected_row_is_painted_at_every_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """D21: the SELECTED row's name must be painted, at and around the cap.

    The round-3 N3 header fix pulled a header back in AFTER the line budget was
    spent, so at exactly the cap the LAST entry — the selection — fell off the
    window while `s`/`a` would still write to it. This drives the real app with
    the selection on the last link and asserts its NAME is on screen.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for i in range(count):
        _seed_link(tmp_path, local_id=f"a{i:02d}", name=f"agent-{i:02d}", subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = count - 1
        panel._repaint()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        last = f"agent-{count - 1:02d}"
        assert last in painted, f"the selected last row {last!r} is not painted\n{painted}"


@pytest.mark.asyncio
async def test_selected_row_is_painted_with_wrapped_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D21 with WRAPPED rows: the selection survives when every row is 2 lines."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for i in range(20):
        _seed_link(tmp_path, local_id=f"a{i:02d}", name=f"agent-{i:02d}", subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    # 40 cols wraps each row's policy pair onto a second line.
    async with app.run_test(size=(40, 24)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = 19
        panel._repaint()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "agent-19" in painted, painted


@pytest.mark.asyncio
async def test_consent_block_is_fully_painted_at_small_screens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D22: the consent's full block (lead + body + keys) fits the screen.

    The card height was capped by a constant, so at 24 rows a multi-link card ran
    off the bottom and the consent keys line was clipped — a decision with no
    stated way to take it. The body budget now derives from the screen rows.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for i in range(3):
        _seed_link(
            tmp_path,
            local_id=f"a{i}",
            name=f"agent-{i}",
            subscribe=True,
            attest_revision_cas=True,
        )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(40, 24)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = app._hub_links_panel()
        panel._selected = 0
        await pilot.press("a")  # arm the consent
        await pilot.pause()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "Publish publicly?" in painted, painted
        assert "published publicly" in painted, painted
        assert "ctrl+y" in painted and "esc" in painted, painted
        # Bottom edge within the screen.
        assert panel.region.y + panel.region.height <= panel.screen.size.height, (
            panel.region,
            panel.screen.size,
        )


@pytest.mark.asyncio
async def test_consent_block_is_fully_painted_with_many_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D22: 16 links at 80x24 still paints the full consent block."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for i in range(16):
        _seed_link(
            tmp_path,
            local_id=f"a{i:02d}",
            name=f"agent-{i:02d}",
            subscribe=True,
            attest_revision_cas=True,
        )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = app._hub_links_panel()
        panel._selected = 0
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "Publish publicly?" in painted, painted
        assert "published publicly" in painted, painted
        assert "ctrl+y" in painted, painted
        assert panel.region.y + panel.region.height <= panel.screen.size.height


@pytest.mark.asyncio
@pytest.mark.parametrize("visibility,tenant", [("org", "acme"), ("public", None)])
async def test_zero_spacing_burst_with_punctuation_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visibility: str, tenant: str | None
) -> None:
    """D23: a ZERO-SPACING burst containing space/comma/period never writes.

    Textual NAMES space/comma/period (`space`/`comma`/`full_stop`), so the old
    `len(key)==1` classifier treated them as deliberate control keys and engaged
    the panel — `h<space>a` then wrote policy. This posts Key events with NO
    pauses (a true burst) and asserts nothing is written.
    """

    from textual import events

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(
        tmp_path,
        visibility=visibility,
        tenant=tenant,
        subscribe=True,
        auto_push=False,
        attest_revision_cas=True,
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        # A zero-spacing burst: call the handler with no pilot.pause between.
        # Textual names space/comma/period, so the event carries the NAMED key
        # with its real printable CHARACTER — exactly what a terminal produces.
        for ch in "h a,x.a":
            from textual.keys import _character_to_key

            panel.on_key(events.Key(_character_to_key(ch), ch))
        await pilot.pause()
    values = _policy_values(tmp_path, identity.link_id)
    assert values["auto_push"] is False, "a burst must not enable auto-push"
    assert values["subscribe"] is True, "a burst must not flip subscribe"


@pytest.mark.asyncio
async def test_highlighted_row_is_the_write_target_for_every_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D27: the HIGHLIGHTED row is exactly the row `s`/`a` writes, every index.

    The service sorts links `(kind, name)` — agent, skill, team — while the card
    paints agent, team, skill, and `selected_row`/`_selected` indexed `_rows`.
    Left in service order, the highlight and the policy-write target disagreed by
    kind. Rows are now normalised to paint order once, so the reverse-styled
    name and the write TARGET (``selected_row``, which the toggles post) are the
    same row at every index.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # Mixed kinds with names that force a DIFFERENT order under (kind,name) vs
    # paint order: an agent, a skill, a team, and a second agent.
    _seed_link(tmp_path, kind="team", local_id="t1", name="crew", tenant="acme", visibility="org")
    _seed_link(tmp_path, kind="skill", local_id="s1", name="deploy")
    _seed_link(tmp_path, kind="agent", local_id="a1", name="coder")
    _seed_link(tmp_path, kind="agent", local_id="a2", name="auditor")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = app._hub_links_panel()
        expected = ["auditor", "coder", "crew", "deploy"]  # agent, agent, team, skill
        for index in range(4):
            panel._selected = index
            panel._repaint()
            await pilot.pause()
            row = panel.selected_row
            assert (
                row is not None and row.name == expected[index]
            ), f"row at index {index} is {row and row.name!r}, expected {expected[index]!r}"
            # The painted highlight (reverse ink) must be THIS row's name — the
            # exact row the toggle writes, since the toggles post selected_row.
            painted = _highlighted_names(app)
            assert (
                row.name in painted
            ), f"the highlighted rows {painted!r} do not include the selected {row.name!r}"


def _highlighted_names(app: Any) -> list[str]:
    """The names painted with the row-ink REVERSE style (the highlight).

    ``render_strips`` splits a line into SEGMENTS (the boot-layout precedent);
    the selected row's name carries ``reverse`` in its resolved style.
    """

    names: list[str] = []
    for strip in app.screen._compositor.render_strips():
        for segment in strip._segments:
            if getattr(segment.style, "reverse", False) and segment.text.strip(" ·"):
                names.append(segment.text.strip(" ·"))
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["tab", "shift+tab"])
async def test_tab_never_strands_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """D29: tab/shift+tab must not move focus off the card while it is open.

    Left unhandled, tab moved focus to the TranscriptView, stranding the overlay
    OPEN but UNFOCUSED: `esc`/`q` no longer reached it and typed letters landed
    in the composer UNDER the card. The panel now consumes them.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_link(tmp_path, visibility="org", tenant="acme")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = app._hub_links_panel()
        await pilot.press(key)
        await pilot.pause()
        assert panel.is_open, "tab must not close the overlay"
        assert panel.has_focus, "tab must not move focus off the open overlay"
        # esc still closes it (focus was retained).
        await pilot.press("escape")
        await pilot.pause()
        assert not panel.is_open, "esc must still close the overlay after tab"


@pytest.mark.asyncio
async def test_non_printable_burst_key_does_not_engage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D30: a non-printable burst key that the panel does NOT bind must not engage.

    `enter`/ctrl/F-keys are not panel commands, so a burst like `h`,enter,`a`
    must not engage the panel and then let `a` write private auto-push.
    """

    from textual import events

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="org", tenant="acme", attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        # Zero-spacing burst: `h`, enter, `a` — enter is NOT a panel binding.
        for ev in (events.Key("h", "h"), events.Key("enter", None), events.Key("a", "a")):
            panel.on_key(ev)
        await pilot.pause()
    values = _policy_values(tmp_path, identity.link_id)
    assert values["auto_push"] is False, "a non-printable burst key must not engage"


@pytest.mark.asyncio
async def test_card_clears_the_dock_for_screens_of_at_least_24_rows_and_60_cols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D28 (D36 scope): the card clears the dock for >=24 rows and >=60 cols.

    Round-3's N3 header pull-back added a line AFTER the budget was spent, so a
    header-adorned window could paint one row past the screen and cover the
    composer. Swept here at the sizes the finding named, which are all at or
    above the fitting floor; BELOW the floor the consent decision stays fully
    painted and covers the composer by design (see HubLinksPanel._body_budget).
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for i in range(8):
        _seed_link(tmp_path, local_id=f"a{i}", name=f"agent-{i}", subscribe=True)
    for size in ((100, 24), (80, 24), (100, 30)):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.action_hub_links()
            await pilot.pause()
            await pilot.pause()
            panel = app._hub_links_panel()
            panel._selected = 7
            panel._repaint()
            await pilot.pause()
            bottom = panel.region.y + panel.region.height
            assert bottom <= panel.screen.size.height, (size, panel.region, panel.screen.size)


@pytest.mark.asyncio
async def test_mid_section_window_keeps_its_section_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D35: a window opening mid-section still shows the section title."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # 8 agents then 8 skills; select the last skill so the window opens in Skills.
    for i in range(8):
        _seed_link(tmp_path, kind="agent", local_id=f"a{i}", name=f"agent-{i}", subscribe=True)
    for i in range(8):
        _seed_link(tmp_path, kind="skill", local_id=f"s{i}", name=f"skill-{i}", subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = app._hub_links_panel()
        panel._selected = 15
        panel._repaint()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "Skills" in painted, painted
        assert "skill-7" in painted, painted


def _link(**over: Any) -> dict[str, Any]:
    base = {
        "kind": "agent",
        "local_id": "a1",
        "name": "coder",
        "link_id": "L1",
        "hub_source": "https://hub.test/v1",
        "visibility": "public",
        "tenant_name": None,
        "policy": {"subscribe": True, "auto_push": False, "revision": "rev1"},
        "eligibility": {"can_auto_push": False, "reason": "unsupported-precondition"},
        "auto_will_apply": False,
        "auto_will_push": False,
        "pull": {"state": "idle", "enabled": True, "error_class": None},
        "push": {"state": "idle", "enabled": False, "error_class": None},
    }
    base.update(over)
    return base


def _snap(*links: dict[str, Any]) -> dict[str, Any]:
    return {"capability": "hub_links:1", "links": list(links)}


@pytest.mark.asyncio
async def test_selection_follows_the_link_across_a_reorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D38: an insert above keeps the SAME link highlighted (not by index).

    `_selected` is an int into a re-ordered list, so an index-only carry moved
    the highlight to a different link on refresh, and the next `s`/`a` wrote
    THAT link. The selection now follows the link_id.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = app._hub_links_panel()
        panel.display = True
        panel.show_links(_snap(_link(name="alpha", link_id="A"), _link(name="crew", link_id="C")))
        panel._selected = 1  # crew
        await pilot.pause()
        assert panel.selected_row is not None and panel.selected_row.link_id == "C"
        # An agent is INSERTED ABOVE crew: paint order is alpha, bravo, crew.
        panel.show_links(
            _snap(
                _link(name="alpha", link_id="A"),
                _link(name="bravo", link_id="B"),
                _link(name="crew", link_id="C"),
            )
        )
        await pilot.pause()
        row = panel.selected_row
        assert (
            row is not None and row.link_id == "C"
        ), f"selection moved to {row and row.link_id!r}; it must stay on crew (C)"


@pytest.mark.asyncio
async def test_selection_clamps_and_highlights_a_real_row_when_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D38: removing the selected link clamps to a real neighbour (never None)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = app._hub_links_panel()
        panel.display = True
        panel.show_links(
            _snap(
                _link(name="alpha", link_id="A"),
                _link(name="bravo", link_id="B"),
                _link(name="crew", link_id="C"),
            )
        )
        panel._selected = 2  # crew, the last
        await pilot.pause()
        # crew disappears.
        panel.show_links(_snap(_link(name="alpha", link_id="A"), _link(name="bravo", link_id="B")))
        await pilot.pause()
        assert panel.selected_row is not None, "the selection must clamp to a real row"
        assert panel.selected_row.link_id in ("A", "B")


@pytest.mark.asyncio
async def test_removing_the_armed_link_disarms_the_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D38: an armed consent whose link is gone is DISARMED (never invisible)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = app._hub_links_panel()
        panel.display = True
        panel.show_links(
            _snap(
                _link(name="alpha", link_id="A"),
                _link(
                    name="crew",
                    link_id="C",
                    eligibility={"can_auto_push": False, "reason": "unsafe-publication"},
                ),
            )
        )
        panel._selected = 1
        panel._consent_link_id = "C"
        panel._consent_revision = "rev1"
        await pilot.pause()
        # crew disappears: the consent must not survive invisibly.
        panel.show_links(_snap(_link(name="alpha", link_id="A")))
        await pilot.pause()
        assert panel._consent_link_id is None, "the armed consent must be disarmed"
        # ctrl+y is therefore inert.
        panel.action_confirm_egress()
        await pilot.pause()
        assert panel._consent_link_id is None


@pytest.mark.asyncio
async def test_window_never_ends_on_an_orphan_section_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D37: the window must not end on a section header whose rows are below it.

    8 agents then 8 skills at 80x24: the window can admit the `Skills` header as
    its last line while the skills rows fall past the cap, showing an EMPTY
    section. The trailing header is dropped.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app._hub_links_panel()
        panel.display = True
        links = [
            _link(kind="agent", local_id=f"a{i}", name=f"agent-{i}", link_id=f"A{i}")
            for i in range(8)
        ]
        links += [
            _link(kind="skill", local_id=f"s{i}", name=f"skill-{i}", link_id=f"S{i}")
            for i in range(8)
        ]
        panel.show_links(_snap(*links))
        panel._selected = 7  # last agent
        panel._repaint()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        # If `Skills` appears, at least one skill row must appear with it.
        if "Skills" in painted:
            assert any(f"skill-{i}" in painted for i in range(8)), painted


@pytest.mark.asyncio
async def test_tui_second_write_after_the_grant_uses_the_served_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: after ctrl+y mints the grant, `s` on the freshly served revision works.

    The panel re-reads the inventory after the write, so the row carries the new
    SERVED revision; that token must equal what ``update_policy`` recomputes or
    the subscribe flip would be refused with "link policy changed".
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # arm
        await pilot.pause()
        await pilot.press("ctrl+y")  # mint the grant
        await pilot.pause()
        await pilot.pause()
        # The app re-read the inventory; `s` now uses the freshly served revision.
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "link policy changed" not in painted, painted
    values = _policy_values(tmp_path, identity.link_id)
    assert values["subscribe"] is False, "the subscribe flip must persist"
    assert values["auto_push"] is True

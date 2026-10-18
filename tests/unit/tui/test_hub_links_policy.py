"""Round-2/3 policy-action regressions for the Hub links overlay.

The policy action must WRITE through the CLI/desktop ``update_policy`` path, or
refuse with a NAMED reason — never a stub, and never a write from a stray
keystroke. Every test drives the REAL app through ``pilot.press`` (never a
direct ``action_*`` call), because round 3 found the arming key `a` could be
"confirmed" by typed-ahead text and by a second `a`, which direct calls hid.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import Kind, LinkIdentity, Visibility
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.hub_links_panel import HubLinksPanel
from tests.unit.tui.test_app_pilot import FakeSession, _factory

HUB = "https://api.radienthq.com/v1"


def _seed_link(
    root: Path,
    *,
    kind: Kind = "agent",
    local_id: str = "a1",
    name: str = "coder",
    visibility: Visibility = "public",
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
    # An AGREED POINT: since the seeding ruling (round 2, (A)), unattended push
    # needs an (A, P) receipt, so an "eligible" link must carry one -- without it
    # the honest answer is the NAMED `no-receipt` refusal.
    from local_operator.hub_sync.push import receipt_for

    record = dataclasses.replace(
        record,
        identity=identity,
        remote_revision='"rev-1"',
        lineage="clean",
        local_alias=name,
        outbound=receipt_for(kind, fields, fields, revision='"rev-1"'),
    )
    prov.write_baseline(root, record)
    hub: dict[str, Any] = {
        "links": {identity.link_id: {"subscribe": subscribe, "auto_push": auto_push}}
    }
    if tenant:
        hub["tenant_names"] = {tenant: tenant.title()}
    ConfigManager(root).set_config_value("hub", hub)

    def fold(doc: dict[str, Any]) -> None:
        item = doc.setdefault("items", {}).setdefault(
            f"{kind}:{local_id}",
            st.blank_item(kind, local_id, name, f"h-{local_id}", tenant),
        )
        item.update(state="up-to-date", error_class=None, last_checked_at="2026-10-14T00:00:00Z")
        if attest_revision_cas:
            # A per-SOURCE attestation (bumped `skills` too when this is a skill
            # link): eligibility reads the entry for the link's own hub_source.
            entry = (
                doc.setdefault("server", {})
                .setdefault("sources", {})
                .setdefault(identity.hub_source, {})
            )
            entry["revision_cas"] = 1
            if kind == "skill":
                entry["skills"] = 1

    st.StatusStore(root).mutate(fold)
    return identity


def _policy_values(root: Path, link_id: str) -> dict[str, Any]:
    hub = ConfigManager(root).get_config().values.get("hub", {})
    return dict(hub.get("links", {}).get(link_id, {}))


async def _open_panel(app: OperatorApp, pilot: Any) -> None:
    """Open the overlay the way a user does and ENGAGE the panel (past typed-ahead)."""
    app.action_hub_links()
    # The inventory is a real thread hop. Keys sent while it is loading have
    # no selected row to act on, however many message-pump pauses preceded them.
    await _await_settle(
        pilot,
        lambda: not _mounted_panel(app)._loading
        and bool(_mounted_panel(app)._rows)
        and app.focused is _mounted_panel(app),
        what="the initial inventory and panel focus",
        guard=5,
    )
    # One navigation key engages the panel (round-3 D15), so subsequent presses
    # are treated as intent rather than typed-ahead.
    await pilot.press("down")
    await pilot.pause()


def _mounted_panel(app: OperatorApp) -> HubLinksPanel:
    """The mounted overlay panel, ASSERTED present: every use opens it first."""

    panel = app._hub_links_panel()
    assert panel is not None
    return panel


#: Backstop for :func:`_await_settle`: how long a settle (a policy write+reread,
#: a fetch, a thread-landed notice) may run before the wait calls it wedged.
#: Generous on purpose -- it only ever fires on a genuine hang, and a hang must
#: fail the test, not take the shard's job ceiling with it (the conftest's
#: ``BOOT_DEADLOCK_GUARD_S`` is the same shape for the boot worker).
SETTLE_GUARD_S = 30.0


async def _await_settle(
    pilot: Any,
    predicate: Callable[[], bool],
    *,
    what: str,
    guard: float = SETTLE_GUARD_S,
) -> None:
    """Pump until ``predicate`` holds; the deadline is a deadlock backstop only.

    WHY THIS EXISTS. Every settle this file waits on (a policy write + its
    post-write re-read, a fetch, a thread-landed refusal notice) crosses real
    ``asyncio.to_thread`` hops, and ``pilot.pause()`` drains the message pump --
    it does NOT wait for a thread (the conftest's ``await_session_boot``
    documents the same reading for the boot worker's warm-up thread). A fixed
    pause count was therefore a bet on thread scheduling, and it lost on a
    loaded shard: CI run 38964693363, shard (3.13, 0), read
    ``test_double_press_is_inert_until_the_write_settles`` after 8 pauses and
    failed "the write must settle" with nothing wrong in the write. The
    reference fix (301153e7d, "wait on the facts the cells assert, not pause
    counts") established the shape: poll the FACT each turn and keep a bounded
    backstop so a genuinely wedged worker still fails loudly. This helper is
    that shape with the backstop in wall-clock time -- the unit the true wait is
    in -- rather than a frame count: ``pause()`` already sleeps ~20 ms of wall
    time per turn (``wait_for_idle``'s granularity), so a frame count is a clock
    bet in disguise. Callers keep their own assertions on the same predicate,
    which is true on return.
    """

    deadline = time.monotonic() + guard
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"the deadlock backstop ({guard:.0f}s) fired waiting for {what}")
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
        panel = _mounted_panel(app)
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

    `/hub` takes focus while the user may still be typing; a printable key can
    never engage or dispatch (round-1 Q6 made that deterministic), so every
    letter of 'banana' is swallowed whole.
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
async def test_printable_keys_never_write_until_a_bound_key_engages_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 Q6: engagement is DETERMINISTIC, not a wall-clock race.

    QA measured the former 200 ms window being defeated at realistic cadence
    (~65 ms/key): `/hub`, enter, then `h`, space, `a` wrote PRIVATE auto_push
    8/8. A printable key can now never engage or dispatch; only a key the panel
    BINDS (one harmless `down`, valid with a single link) opens the door, after
    which `a` IS the real action.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(
        tmp_path, visibility="org", tenant="acme", auto_push=False, attest_revision_cas=True
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app._editor()
        editor.focus()
        editor.load_text("/hub")
        await pilot.press("enter")
        await pilot.pause()
        # The realistic cadence that defeated the old time window.
        for key in ("h", "space", "a"):
            await pilot.press(key)
            await asyncio.sleep(0.065)
        await pilot.pause()
        assert _policy_values(tmp_path, identity.link_id)["auto_push"] is False
        # Preserve the typed-ahead burst above, but do not send the deliberate
        # action until there is a served row; loading cannot accept a write.
        panel = _mounted_panel(app)
        await _await_settle(
            pilot,
            lambda: not panel._loading and bool(panel._rows),
            what="the printable-key test's initial inventory",
            guard=5,
        )
        # A bound key engages; the next printable IS the action.
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("a")
        # Teardown cancels workers; observe the write AND its installed re-read
        # before leaving the app, not merely an idle message pump.
        await _await_settle(
            pilot,
            lambda: _policy_values(tmp_path, identity.link_id)["auto_push"] is True
            and not panel._refreshing,
            what="the engaged auto-push write and re-read",
            guard=5,
        )
    assert _policy_values(tmp_path, identity.link_id)["auto_push"] is True


@pytest.mark.asyncio
async def test_the_card_teaches_its_engage_key_until_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q6: the swallowed keys have a teacher on the frame ("↑↓ to begin")."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_link(tmp_path, visibility="org", tenant="acme", auto_push=False, attest_revision_cas=True)
    # TWO links: with one link the arrows never also teach SELECT, so the QN6
    # duplicate-hint regression ("↑↓ to begin · ↑↓ select") needs a second row.
    _seed_link(
        tmp_path,
        local_id="a2",
        name="bravo",
        visibility="org",
        tenant="acme",
        auto_push=False,
        attest_revision_cas=True,
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_hub_links()
        await pilot.pause()
        await pilot.pause()
        panel = _mounted_panel(app)
        hint = panel._hint_row(True).plain
        assert "↑↓ to begin" in hint, hint
        # QN6 (discriminating): BEFORE engagement the same arrows must not be
        # taught twice; the select hint waits until the card is engaged.
        assert "↑↓ select" not in hint, hint
        await pilot.press("down")
        await pilot.pause()
        hint = panel._hint_row(True).plain
        assert "↑↓ to begin" not in hint, hint
        assert "↑↓ select" in hint, hint


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
        panel = _mounted_panel(app)
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
    """D7/D19 + U4: ineligible enable is refused with a catalogue sentence.

    U4 (pre-release batch): the refusal used to be the raw code with hyphens
    swapped for spaces (``unsupported precondition``), which told the user
    nothing. Each known reason now has its own user-oriented sentence from the
    catalogue; the developer vocabulary must not appear.
    """

    from local_operator.i18n.keys import tui_hub_links as _k
    from local_operator.i18n.text import tr

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, attest_revision_cas=False, auto_push=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        panel = _mounted_panel(app)
        # The refusal is on the CARD (D19), as a user-oriented sentence (U4).
        assert panel._notice, "a refusal must be shown on the card"
        expected = tr("tui.hub_links")(_k.reason_unsupported_precondition())
        assert panel._notice == expected, panel._notice
        assert "unsupported precondition" not in panel._notice.lower(), panel._notice
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
        panel = _mounted_panel(app)
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


def test_footer_hub_door_outlives_the_hide_hint_but_never_asks_or_chip() -> None:
    """D2 (pre-release batch): the door yields AFTER `ctrl+b hide`, never to it.

    The pre-batch ladder spent the base rung full-then-hub-free, so at the
    sidebar's 29-cell floor (a 100-column terminal) `f9 focus · ctrl+b hide ·
    hub: 1` being one cell over budget dropped the HUB while the hide hint
    survived — and the door has no key teacher at this width. The order is now:
    every fact with the full tail, then the hide-dropped full tail, then the
    hub-free one. The door still never displaces the ask count or the chip:
    those are facts discoverable from nowhere else on the frame, and the
    hub-free variants drop the DOOR before either yields.

    All strings here are MEASURED from the real ladder (the probe ran against
    the fixed source; the quiet-29 line is the negative control — pre-fix it
    read `f9 focus · ctrl+b hide`, this test fails).
    """

    from local_operator.tui.widgets.session_sidebar import SessionSidebar

    def painted(width: int, *, focused: bool, asks: int = 0, chip: int = 0, hub: int = 0) -> str:
        sidebar = SessionSidebar()
        sidebar._asks_total = asks
        sidebar._subagent_total = chip
        sidebar._hub_total = hub
        sidebar.has_focus = focused
        return sidebar._footer_line(width)[0]

    # The door outlives the hide hint: visible at the 29-cell floor when the
    # tail is quiet, with `ctrl+b hide` dropped to pay for it; the full form
    # returns once it fits (34 cells).
    assert painted(29, focused=False, hub=1) == "f9 focus · hub: 1"
    assert painted(30, focused=False, hub=1) == "f9 focus · hub: 1"
    assert painted(34, focused=False, hub=1) == "f9 focus · ctrl+b hide · hub: 1"

    # FACTS FIRST: with asks+chip seeded the door may drop at widths where the
    # three facts cannot fit, and NEVER appears beside a dropped ask/chip.
    assert painted(29, focused=False, asks=4, chip=4, hub=1) == "f9 focus · asks: 4 · ⌥4"
    assert painted(34, focused=False, asks=4, chip=4, hub=1) == "f9 focus · asks: 4 · hub: 1 · ⌥4"
    for focused in (False, True):
        for width in (24, 26, 29, 30, 34, 41, 43):
            line = painted(width, focused=focused, asks=4, chip=4, hub=1)
            if "hub:" in line:
                assert "asks: 4" in line and "⌥4" in line, (width, focused, line)

    # No hub: the old ladder is untouched (the hide hint is still there).
    assert painted(29, focused=False) == "f9 focus · ctrl+b hide"
    assert painted(29, focused=False, asks=4, chip=4) == "f9 focus · asks: 4 · ⌥4"
    assert painted(41, focused=False, asks=4, chip=4) == "f9 focus · ctrl+b hide · asks: 4 · ⌥4"

    # Hub + chip: the chip never yields to the door either.
    assert painted(29, focused=False, chip=4, hub=1) == "f9 focus · hub: 1 · ⌥4"


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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visibility: Visibility, tenant: str | None
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
            panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
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
        panel = _mounted_panel(app)
        # U3 (pre-release batch): a press while OUR OWN write is unsettled is
        # INERT by design, so this test waits for the grant write to settle
        # (the served revision installed) before the next press — exactly what
        # a user does when they read the flip first. The main-CI failure this
        # batch also fixes (run 38648922786) was the PRE-batch version of this
        # test letting `s` fire mid-write: the exclusive hub-links worker
        # cancelled the confirm worker's re-read and a write CAS'd against the
        # stale revision (`link policy changed`).
        # The settle crosses the grant's `to_thread` write AND the re-read that
        # installs it -- wait on the fact, not a pause count (see `_await_settle`).
        await _await_settle(pilot, lambda: not panel._refreshing, what="the grant write to settle")
        assert not panel._refreshing, "the grant write must settle"
        # The app re-read the inventory; `s` now uses the freshly served revision.
        await pilot.press("s")
        # A second write has the same thread+reread boundary as the grant;
        # exiting after two pauses can cancel its worker before it settles.
        await _await_settle(
            pilot,
            lambda: _policy_values(tmp_path, identity.link_id)["subscribe"] is False
            and not panel._refreshing,
            what="the second policy write and re-read",
            guard=5,
        )
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "link policy changed" not in painted, painted
    values = _policy_values(tmp_path, identity.link_id)
    assert values["subscribe"] is False, "the subscribe flip must persist"
    assert values["auto_push"] is True


@pytest.mark.asyncio
async def test_second_press_during_the_grant_write_cannot_cancel_its_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main-CI race (run 38648922786): `s` mid-grant-write must be inert.

    PRE-batch: pressing `s` while the ctrl+y grant write was in flight started a
    second worker in the SAME exclusive group, which cancelled the confirm
    worker's re-read; the grant write committed with a revision the panel never
    learned, and a write CAS'd against a stale revision — CI painted
    `link policy changed`. POST-batch: the second press is inert while
    `_refreshing` is set (U3), so nothing cancels the confirm worker; the grant
    persists and a later `s` uses the SERVED revision. The grant write is gated
    so the race window is deterministic, not a timing coincidence.
    """

    import threading

    from local_operator.hub_sync import policies as policies_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    gate = threading.Event()
    calls = {"n": 0}
    real_update = policies_mod.update_policy

    def gated(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:  # the ctrl+y grant write blocks in its worker
            gate.wait(5)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(policies_mod, "update_policy", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")  # arm
        await pilot.pause()
        await pilot.press("ctrl+y")  # the grant write blocks on the gate
        await pilot.pause()
        panel = _mounted_panel(app)
        assert panel._refreshing, "premise: the grant write is in flight"
        await pilot.press("s")  # the racing press: inert, must not cancel the worker
        await pilot.pause()
        gate.set()
        await _await_settle(pilot, lambda: not panel._refreshing, what="the grant write to settle")
        assert not panel._refreshing, "the grant write must settle"
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "link policy changed" not in painted, painted
        # The early press wrote nothing; the grant persisted; the flag flips now.
        mid = _policy_values(tmp_path, identity.link_id)
        assert mid["auto_push"] is True, "the grant must persist (its re-read was not cancelled)"
        assert mid["subscribe"] is True, "the mid-write press must not have written"
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
    values = _policy_values(tmp_path, identity.link_id)
    assert values["subscribe"] is False, "the settled press must use the served revision"
    assert values["auto_push"] is True


# ---------------------------------------------------------------------------
# The pre-release batch (U1/U2/U3/U5/D3/U6). Each regression below FAILS on
# the pre-batch head 3cd57aaa5a; the U1/U2 repros were first executed against
# that head by hand (read-only dir -> app dead; error body -> blind write).


@pytest.mark.asyncio
async def test_a_failed_policy_write_keeps_the_session_and_reports_on_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U1a: a read-only config dir must NOT kill the TUI (the write fails).

    Pre-fix: `s` raised PermissionError out of `update_policy`, the policy
    worker's default `exit_on_error` ended the app (the repro printed
    `app_alive=False`, a WorkerFailed out of run_test), and the conversation
    went with it. Now the write failure is caught, the card shows the
    catalogue's save-failed line naming policy + item, the app keeps running,
    and the config is untouched.
    """

    from local_operator.i18n.keys import tui_hub_links as _k
    from local_operator.i18n.text import tr

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _open_panel(app, pilot)
            os.chmod(tmp_path, 0o500)  # the write must raise, the session must not die
            await pilot.press("s")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
            assert app.is_running, "the session must survive a failed write (U1)"
            panel = _mounted_panel(app)
            assert panel.is_open, "the card stays up"
            expected = tr("tui.hub_links")(
                _k.policy_save_failed(
                    policy=tr("tui.hub_links")(_k.policy_subscribe_label()), source="coder"
                )
            )
            assert panel._notice == expected, panel._notice
    finally:
        os.chmod(tmp_path, 0o700)
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is True, "nothing was saved"


@pytest.mark.asyncio
async def test_a_failed_post_write_reread_keeps_the_session_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U1b: the re-read AFTER the write fails — the write stands, session lives.

    Pre-fix the injected RuntimeError from the post-write snapshot read ended
    the app too (the repro printed `app_alive=False`), silently, after the
    write had landed.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)

        def flaky() -> Any:
            raise RuntimeError("config dir unreadable (injected)")

        app._hub_links_snapshot_sync = flaky  # type: ignore[method-assign]
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        assert app.is_running, "a failed re-read must not kill the session (U1)"
        panel = _mounted_panel(app)
        assert panel.is_open
        assert "Could not save" in panel._notice, panel._notice
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is False, "the write stands"


@pytest.mark.asyncio
async def test_policy_actions_are_inert_while_the_error_body_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U2: an error body hides the rows — `s`/`a` must not write blind.

    Pre-fix the repro flipped `subscribe true→false` while the card showed
    `hub links unavailable: ...` and no rows. Now the actions are inert, the
    hint teaches only what is live (r refresh / esc close), and the taught
    refresh restores the rows so the SAME key works again after recovery.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = _mounted_panel(app)
        panel.show_error("hub links unavailable: boom")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        assert app.is_running
        assert _policy_values(tmp_path, identity.link_id)["subscribe"] is True, "blind write (U2)"
        hint = str(panel._hint_row(True))
        assert "s subscribe" not in hint and "a auto-push" not in hint, hint
        assert "r refresh" in hint and "esc close" in hint, hint
        # Recovery is the taught key: `r` re-reads and reinstalls the rows.
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert panel._error == "" and panel._rows, "the refresh must restore the rows"
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is False


@pytest.mark.asyncio
async def test_double_press_is_inert_until_the_write_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U3: a second press while our own write is in flight must not write again.

    Pre-fix the second press CAS'd against the first one's new revision and the
    card reported the false `link policy changed` self-conflict (and `Saving…`
    was unreachable: `_refreshing` was never set). The write is gated here (a
    threading.Event) so the double press lands INSIDE the window
    deterministically; exactly one flip must persist, with no refusal line.
    """

    import threading

    from local_operator.hub_sync import policies as policies_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    gate = threading.Event()
    real_update = policies_mod.update_policy

    def gated(*args: Any, **kwargs: Any) -> Any:
        gate.wait(5)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(policies_mod, "update_policy", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("s")
        await pilot.pause()
        panel = _mounted_panel(app)
        assert panel._refreshing, "premise: the card must report the write in flight"
        await pilot.press("s")  # the double press: inert while unsettled
        await pilot.pause()
        gate.set()
        # This cell is the one CI 38964693363 (shard 3.13/0) lost to an 8-pause
        # bet: the settle crosses the write's thread hop AND the re-read's --
        # wait on the fact, not a pause count (see `_await_settle`).
        await _await_settle(pilot, lambda: not panel._refreshing, what="the write to settle")
        assert not panel._refreshing, "the write must settle"
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "link policy changed" not in painted, painted
    assert _policy_values(tmp_path, identity.link_id)["subscribe"] is False, "exactly one flip"


@pytest.mark.asyncio
async def test_the_hint_row_is_state_aware_and_names_a_failed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U5/D4/D3: hints teach only live keys; a failed row names its reason.

    Loading, the error body and the empty inventory have no fresh selectable
    rows, so `s`/`a`/↑↓ would be dead keys in those states — the row teaches
    the subset that is live. A failed selected row carries its reason (D3),
    with `r refresh` as the live recovery key beside it.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = _mounted_panel(app)
        panel.display = True
        # LOADING (first open, no rows yet): only the exit is live.
        panel.start_fetch()
        assert str(panel._hint_row(True)) == "esc close"
        # EMPTY: same — there is nothing to select.
        panel.show_links({"capability": "hub_links:1", "links": []})
        assert str(panel._hint_row(True)) == "esc close"
        # ERROR body: the re-read is live.
        panel.show_error("hub links unavailable: boom")
        assert str(panel._hint_row(True)) == "r refresh · esc close"
        # FAILED row: the reason is NAMED (D3) and `r` remains reachable — the
        # failed-row yield order drops the policy hints BEFORE `r refresh`.
        panel.show_links(
            _snap(_link(pull={"state": "failed", "enabled": True, "error_class": "provider-error"}))
        )
        assert str(panel._hint_row(True)) == (
            "↑↓ to begin · check failed: provider error · s subscribe · r refresh · esc close"
        ), str(panel._hint_row(True))


@pytest.mark.asyncio
async def test_the_consent_block_carries_no_cli_machinery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U6: the armed prompt is the decision only — no `or:` command, no raw id.

    The block named a literal `<id>` beside a real 64-hex id and read as a
    developer tool, the largest thing on the card. The prompt is now exactly
    the lead, the consequence and the two named keys.
    """

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_link(tmp_path, visibility="public", auto_push=False, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        await pilot.press("a")
        # The compositor can still hold the loading/old frame after the arm;
        # wait for the actual consent ink without weakening any copy checks.
        await _await_settle(
            pilot,
            lambda: "Publish publicly?"
            in "\n".join(strip.text for strip in app.screen._compositor.render_strips()),
            what="the public-publish consent frame",
            guard=5,
        )
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert "Publish publicly?" in painted, painted
        assert "ctrl+y" in painted and "esc to cancel" in painted, painted
        assert "lop hub link set" not in painted, painted
        assert "link id" not in painted, painted
        assert "<id>" not in painted, painted


# ---------------------------------------------------------------------------
# Round-2 reviews on #2410 (R1/R2 MAJOR, U7): the write+reread lifetime must
# stay OWNED until an installed snapshot (or the terminal save-failed line)
# settles it — a painted refusal does not. `r` must not cancel it either.


@pytest.mark.asyncio
async def test_a_cas_refusal_keeps_the_actions_blocked_until_the_reread_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1/U7: a refusal notice does NOT settle the write+reread lifetime.

    A stale revision makes the write refuse; the notice paints BEFORE the
    re-read that installs the moved revision. Through that window `s`/`a`
    must stay inert and post ZERO new intents — pre-fix the notice cleared
    `_refreshing`, a second `s` posted a write with the stale revision, and
    the new same-group exclusive worker cancelled the re-read (the U3 race,
    one state later). After the re-read installs, the SAME key writes against
    the fresh revision.
    """

    import threading

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True, attest_revision_cas=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    intents: list[object] = []
    real_handler = app.on_hub_links_panel_policy_toggle_requested

    def counting(message: Any) -> None:
        intents.append(message)
        real_handler(message)

    app.on_hub_links_panel_policy_toggle_requested = counting  # type: ignore[method-assign]
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = _mounted_panel(app)
        # Gate the RE-READ only, and only AFTER the open settled: the re-read
        # must block so the refusal window stays observable (the initial fetch
        # above already used the real snapshot).
        release = threading.Event()
        real_snapshot = app._hub_links_snapshot_sync

        def gated_snapshot() -> Any:
            release.wait(10)
            return real_snapshot()

        app._hub_links_snapshot_sync = gated_snapshot  # type: ignore[method-assign]
        # A concurrent edit moved the revision: the write will be REFUSED.
        panel._rows = [
            dataclasses.replace(row, policy_revision="stale-revision") for row in panel._rows
        ]
        await pilot.press("s")
        # The refusal notice is painted only after the worker's `to_thread` write
        # returns -- wait on the fact, not a pause count (see `_await_settle`).
        await _await_settle(pilot, lambda: bool(panel._notice), what="the CAS refusal notice")
        assert panel._notice, "the CAS refusal must be on the card"
        assert "changed" in panel._notice.lower(), panel._notice
        # R1/U7 premise: the refusal painted while the re-read is still blocked.
        assert panel._refreshing, "a refusal must not settle the write+reread lifetime"
        assert panel._policy_actions_allowed() is False
        intents.clear()  # the first intent belongs to the press above
        await pilot.press("s")
        await pilot.press("a")
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert intents == [], "no new intent may post while the re-read is pending"
        release.set()
        await _await_settle(pilot, lambda: not panel._refreshing, what="the installed re-read")
        assert not panel._refreshing, "the installed re-read settles the lifetime"
        # The installed revision is live: the same key now writes, no stale CAS.
        await pilot.press("s")
        await _await_settle(pilot, lambda: not panel._refreshing, what="the second flip to settle")
    values = _policy_values(tmp_path, identity.link_id)
    assert values["subscribe"] is False, "the post-settle flip must land on the moved revision"


@pytest.mark.asyncio
async def test_refresh_defers_while_the_write_is_pending_and_the_fresh_revision_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: `r` cannot cancel a pending policy worker's re-read.

    While OUR write+reread is unsettled, both the panel action AND a
    programmatic `RefreshRequested` must be deferred — a refresh would start a
    second exclusive worker in the same group and cancel the re-read, while
    the write keeps running in its thread. The pending worker installs the
    fresh snapshot itself and the NEXT flip accepts the served revision.
    """

    import threading

    from local_operator.hub_sync import policies as policies_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    gate = threading.Event()
    real_update = policies_mod.update_policy

    def gated(*args: Any, **kwargs: Any) -> Any:
        gate.wait(10)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(policies_mod, "update_policy", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = _mounted_panel(app)
        started: list[str] = []
        real_fetch = app._fetch_hub_links_worker

        async def counting_fetch() -> None:
            started.append("fetch")
            await real_fetch()

        app._fetch_hub_links_worker = counting_fetch  # type: ignore[method-assign]
        await pilot.press("s")
        await pilot.pause()
        assert panel._refreshing, "premise: the write is in flight"
        await pilot.press("r")  # the panel action: deferred
        await pilot.pause()
        app.post_message(HubLinksPanel.RefreshRequested())  # the handler must hold too
        await pilot.pause()
        await pilot.pause()
        assert started == [], "no refresh worker may start while the write+reread is pending"
        assert panel._refreshing, "the pending worker is untouched"
        gate.set()
        await _await_settle(
            pilot, lambda: not panel._refreshing, what="the pending worker's re-read"
        )
        assert not panel._refreshing, "the pending worker must settle via its own re-read"
        await pilot.press("s")  # the served revision accepts the next flip
        await _await_settle(pilot, lambda: not panel._refreshing, what="the second flip to settle")
    values = _policy_values(tmp_path, identity.link_id)
    assert values["subscribe"] is True, "two clean flips: False on the gated write, True after"


@pytest.mark.asyncio
async def test_reopening_during_a_pending_write_does_not_resurrect_a_stale_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 lifecycle: close+reopen mid-write settles on a FRESH fetch.

    The dismiss cancels the group and drops the armed state, so a reopen must
    not inherit `Saving…` or an armed intent; the orphaned thread write lands
    on its own and the next read shows it. Nothing stale survives.
    """

    import threading

    from local_operator.hub_sync import policies as policies_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    identity = _seed_link(tmp_path, subscribe=True)
    gate = threading.Event()
    real_update = policies_mod.update_policy

    def gated(*args: Any, **kwargs: Any) -> Any:
        gate.wait(10)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(policies_mod, "update_policy", gated)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_panel(app, pilot)
        panel = _mounted_panel(app)
        await pilot.press("s")
        await pilot.pause()
        assert panel._refreshing, "premise: the write is in flight"
        await pilot.press("escape")  # dismiss: the group is cancelled
        await pilot.pause()
        await pilot.pause()
        app.action_hub_links()  # reopen starts a NEW lifecycle
        # The reopen's fetch crosses a `to_thread` snapshot read -- wait on the
        # fact (rows installed, no stale Saving... state), not a pause count.
        await _await_settle(
            pilot,
            lambda: not _mounted_panel(app)._refreshing and bool(_mounted_panel(app)._rows),
            what="the reopen's fetch to install rows",
        )
        panel = _mounted_panel(app)
        assert not panel._refreshing, "the reopen must not inherit the stale Saving… state"
        assert panel._consent_link_id is None, "no stale intent survives a reopen"
        # The reopen starts UNENGAGED (typed-ahead protection): one navigation
        # key engages it, as the user would before pressing `r`.
        await pilot.press("down")
        await pilot.pause()
        gate.set()  # the orphaned thread write lands now
        # The landed write is observed IN THE FILE it writes; poll that fact, not
        # a pause count (see `_await_settle`).
        await _await_settle(
            pilot,
            lambda: _policy_values(tmp_path, identity.link_id).get("subscribe") is False,
            what="the orphaned write to land",
        )
        await pilot.press("r")  # the next read shows the landed write
        # The `r` re-read crosses another `to_thread` snapshot read -- wait on
        # the fact the assertion reads, not a pause count (see `_await_settle`).
        await _await_settle(
            pilot,
            lambda: bool(panel._rows) and panel._rows[0].subscribe is False,
            what="the reopened panel to settle on the landed state",
        )
        assert (
            panel._rows and panel._rows[0].subscribe is False
        ), "the reopened panel must settle on the landed state"

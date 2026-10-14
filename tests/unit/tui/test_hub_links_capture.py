"""Real-app capture rig for the Hub links sidebar.

Drives the ACTUAL ``OperatorApp`` (production stylesheet via ``run_test``), puts
it in each state the sidebar spec names, and exports a native-size SVG plus the
cell/box geometry. Not a CSS-less host and not a CLI dump: the evidence is the
frame a user sees.

Run it (isolated HOME/config, scratchpad preserved, CMUX stripped, TERM set) —
the SVGs land in ``$LOCAL_OPERATOR_SCRATCHPAD/hub-links-capture/``:

    ISO=$(mktemp -d)
    env -i HOME="$ISO/home" LOCAL_OPERATOR_CONFIG_DIR="$ISO/profile" \
        LOCAL_OPERATOR_SCRATCHPAD="$LOCAL_OPERATOR_SCRATCHPAD" \
        PATH="$PWD/.venv/bin:$PATH" TERM=xterm-256color \
        .venv/bin/python scripts/run_bounded.py --timeout 180 -- \
        .venv/bin/python -m pytest tests/unit/tui/test_hub_links_capture.py -q -n0

A designer views the SVGs; the pure-core test (:mod:`tests.unit.tui.test_hub_links`)
is what pins the rows, so this file exists to prove what those rows LOOK like.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.tui.app import OperatorApp
from scripts.visual_capture import save_capture
from tests.unit.tui.test_app_pilot import FakeSession, _factory

pytestmark = pytest.mark.asyncio

OUT = Path(os.environ.get("LOCAL_OPERATOR_SCRATCHPAD", ".")) / "hub-links-capture"
HUB = "https://api.radienthq.com/v1"

#: The rig writes EVIDENCE; it must land in the session scratchpad, never the
#: repo. A pytest run under a bare ``env -i`` has no scratchpad var, and the
#: first version then wrote ``./hub-links-capture`` into the worktree. Refuse
#: rather than guess.
if "LOCAL_OPERATOR_SCRATCHPAD" not in os.environ:  # pragma: no cover - env guard
    raise RuntimeError(
        "the hub-links capture rig needs $LOCAL_OPERATOR_SCRATCHPAD (evidence "
        "must not be written into the repo); run it through scripts/run_bounded.py "
        "within the session, which supplies it"
    )


def _snap_link(name: str) -> dict[str, Any]:
    """A minimal snapshot link row (the shape ``rows_from_snapshot`` reads)."""

    return {
        "kind": "agent",
        "local_id": name,
        "name": name,
        "link_id": name.upper()[:1] + name[1:],
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


def _seed_link(
    root: Path,
    *,
    kind: str = "agent",
    local_id: str,
    name: str,
    visibility: str,
    tenant: str | None,
    subscribe: bool,
    auto_push: bool,
    state: str,
    error: str | None,
    attest_revision_cas: bool = False,
) -> LinkIdentity:
    """Write a real baseline + status row so the inventory enumerates it as a link."""
    identity = LinkIdentity(kind, local_id, HUB, visibility, tenant, f"h-{local_id}")
    # A skill baseline carries a full SkillDocument (its fingerprint covers the
    # manifest), so a skill row needs a valid one-file document.
    if kind == "skill":
        fields: dict[str, Any] = {
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
    import dataclasses

    record = dataclasses.replace(
        record, identity=identity, remote_revision='"rev-1"', lineage="clean", local_alias=name
    )
    prov.write_baseline(root, record)
    hub: dict[str, Any] = {
        "links": {identity.link_id: {"subscribe": subscribe, "auto_push": auto_push}}
    }
    if tenant:
        # The verified persisted display name; the frozen row shows `<org> ·
        # Private` from it, so the rig seeds it rather than leaving the bare
        # `Private` fallback (D2).
        hub["tenant_names"] = {tenant: tenant.title()}
    ConfigManager(root).set_config_value("hub", hub)

    def fold(doc: dict) -> None:
        item = doc.setdefault("items", {}).setdefault(
            f"{kind}:{local_id}", st.blank_item(kind, local_id, name, f"h-{local_id}", tenant)
        )
        item.update(state=state, error_class=error, last_checked_at="2026-10-14T00:00:00Z")
        if attest_revision_cas:
            # A positively-attested atomic revision-CAS: the service's eligibility
            # precondition, seeded here so the consent frame reaches the
            # public-egress prompt rather than the named refusal.
            doc.setdefault("server", {})["revision_cas"] = True

    st.StatusStore(root).mutate(fold)
    return identity


async def _capture(name: str, *, size=(100, 30), contact_sheet: str | None = None, **seed) -> Path:
    """Capture one case in its OWN config root, so state cannot accumulate (D3).

    ``contact_sheet`` names the SURFACE to drive before capture: ``None`` opens
    the ``/hub`` overlay via the ACTION; ``"door"`` opens the sidebar so the
    footer hub DOOR is on screen (the architect's side placement); ``"command"``
    types ``/hub`` into the real editor and presses Enter (the user's own route,
    proving the registry entry's dispatch); ``"overlay-rows"`` seeds more links
    than fit to prove the body windows.
    """

    import shutil
    import tempfile

    from local_operator.tui.settings import settings_reload

    # Each case gets a fresh config root under the scratchpad. The FIRST version
    # reused one env root across all six cases, so links seeded for the healthy
    # frame were still present in the "empty" frame — the §98 empty state had no
    # valid proof at all. Isolating the root per case is what makes each frame
    # describe exactly its own state.
    case_root = Path(tempfile.mkdtemp(prefix=f"hub-links-{name}-", dir=OUT))
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(case_root)
    try:
        settings_reload()
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            for link in seed.get("links", []):
                _seed_link(case_root, **link)
            if contact_sheet == "door":
                # Drive the REAL sidebar-open path so the footer door paints.
                await pilot.press("ctrl+b")
                await pilot.pause()
                await pilot.pause()
                # Seed the OTHER two tail facts so this frame shows the footer
                # with asks + chip + hub door TOGETHER (round-2 D9). These are the
                # Sidebar's own totals, fed the way the app feeds them.
                sidebar = app._session_sidebar
                sidebar.set_asking({"a": 4})
                sidebar.set_asks_total(4)
                sidebar.set_subagent_total(4)
                app.refresh_hub_total()
                await pilot.pause()
                await pilot.pause()
            elif contact_sheet == "command":
                # The USER's route: type the command into the real editor and
                # submit it, so the capture proves the registry entry dispatches.
                editor = app._editor()
                editor.focus()
                editor.load_text("/hub")
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()
            elif contact_sheet == "consent":
                # The public auto-push consent state (round-2 D7): open the
                # overlay and arm the two-step consent on the focused public
                # link, so the frame shows the deliberate confirmation prompt.
                # A navigation key ENGAGES first (round-3 D15 typed-ahead guard),
                # then `a` arms — the real user sequence.
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
                await pilot.press("down")
                await pilot.pause()
                panel = app._hub_links_panel()
                if panel is not None:
                    panel._selected = 0
                    panel.action_toggle_auto_push()
                await pilot.pause()
            elif contact_sheet == "select-last":
                # D21: open the overlay and select the LAST link, so the frame
                # proves the chosen row's name is painted at the row cap.
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
                panel = app._hub_links_panel()
                if panel is not None:
                    panel._selected = max(0, len(panel._rows) - 1)
                    panel._repaint()
                await pilot.pause()
            elif contact_sheet == "tab-esc":
                # D29: press tab then esc — the overlay must be closed (tab did
                # not strand it open-but-unfocused), so this frame shows the
                # transcript, proving the panel released focus and dismissed.
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
                await pilot.press("tab")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
            elif contact_sheet == "reorder":
                # D38: select the SECOND link, then install a snapshot with an
                # agent INSERTED ABOVE — the highlight must stay on the same
                # link (crew), not the same index (which would be bravo).
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
                panel = app._hub_links_panel()
                if panel is not None:
                    panel._selected = 1  # crew
                    panel._repaint()
                    await pilot.pause()
                    panel.show_links(
                        {
                            "capability": "hub_links:1",
                            "links": [
                                _snap_link("alpha"),
                                _snap_link("bravo"),
                                _snap_link("crew"),
                            ],
                        }
                    )
                await pilot.pause()
            elif contact_sheet and contact_sheet.startswith("select-index:"):
                # D27/D35: open the overlay and select an EXPLICIT index, so a
                # frame can show the highlight on a chosen kind or a mid-section
                # window that must still carry its section title.
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
                panel = app._hub_links_panel()
                if panel is not None:
                    panel._selected = int(contact_sheet.split(":", 1)[1])
                    panel._repaint()
                await pilot.pause()
            else:
                # Drive the REAL open path: the sync action the binding, the
                # /hub command and the footer door all call.
                app.action_hub_links()
                await pilot.pause()
                await pilot.pause()
            path = OUT / f"{name}.svg"
            # INSIDE the run_test block: the export composites the ACTIVE screen,
            # and after the context exits there is no screen on the stack.
            save_capture(app, path)
    finally:
        # The seed root is inert once the SVG is written; reclaim it so the
        # scratchpad does not accumulate one config tree per case.
        shutil.rmtree(case_root, ignore_errors=True)
    return path


async def test_capture_hub_links_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. collapsed, healthy single link (quiet reserved mark).
    await _capture(
        "healthy-one-link",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    # 2. multi-link Public + org, differently directed.
    await _capture(
        "multi-public-org",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            ),
            dict(
                local_id="a2",
                name="auditor",
                visibility="org",
                tenant="acme",
                subscribe=False,
                auto_push=False,
                state="up-to-date",
                error=None,
            ),
        ],
    )
    # 3. available update.
    await _capture(
        "available",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="available",
                error=None,
            )
        ],
    )
    # 4. failure with a retryable class.
    await _capture(
        "failure-retry",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="failed",
                error="provider-error",
            )
        ],
    )
    # 5. very long name at minimum width.
    await _capture(
        "long-name-min-width",
        size=(40, 24),
        links=[
            dict(
                local_id="a1",
                name="an-extremely-long-agent-name-that-must-truncate",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    # 6. empty.
    await _capture("empty", links=[])

    # NOTE: the persistent SIDEBAR roster pane is NOT built — the architect
    # chose the footer-DOOR placement instead, so the sidebar captures below
    # drive the DOOR, not a rail.

    # 7. The FOOTER DOOR (the architect's placement): the sidebar open with the
    # hub note on its footer. The geometry-preserving route.
    await _capture(
        "footer-door",
        contact_sheet="door",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    # 8. More links than the body cap: the overlay must WINDOW so a selection
    # past MAX_BODY_ROWS is reachable (the architect's third bug).
    await _capture(
        "overlay-windowed",
        links=[
            dict(
                local_id=f"a{i}",
                name=f"agent-{i:02d}",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
            for i in range(20)
        ],
    )
    # 9. The `/hub` COMMAND route: typed into the real editor and submitted, so
    # this frame proves the registry entry dispatches the overlay to the user.
    await _capture(
        "command-invoked",
        contact_sheet="command",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )

    # Assert THIS test's own frames by name (the file is shared with the
    # round-2 remediation capture test, so a global count is order-coupled).
    for name in (
        "healthy-one-link",
        "available",
        "failure-retry",
        "multi-public-org",
        "long-name-min-width",
        "empty",
        "footer-door",
        "overlay-windowed",
        "command-invoked",
    ):
        svg = OUT / f"{name}.svg"
        assert svg.exists() and "<svg" in svg.read_text()[:200], name


async def test_capture_round2_remediation_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frames round-2 design review specifically asked for.

    * 40x24 overlay — the width the padding bug wrapped at (D8).
    * footer with asks + chip + hub door TOGETHER, and the control where the
      door yields while the facts survive (D9).
    * the restored hint row (D6) and the more-rows indicator (D11), captured in
      the default overlay frame.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)

    # D8: the 40x24 frame the rows wrapped at before the padding fix.
    await _capture(
        "narrow-40x24",
        size=(40, 24),
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    # D9 coexistence: the width where the door rides WITH both facts (content 34
    # at screen 120 — the sidebar quantizes, so the tested size is exact).
    await _capture(
        "footer-all-three",
        size=(120, 30),
        contact_sheet="door",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    # D9 control: the NARROWER frame where the facts survive and the door drops.
    await _capture(
        "footer-door-dropped",
        size=(116, 30),
        contact_sheet="door",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )

    # D7: the public auto-push consent prompt (the deliberate two-step state).
    await _capture(
        "consent-public",
        contact_sheet="consent",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
                attest_revision_cas=True,
            )
        ],
    )

    # D7/D16: the public auto-push consent prompt — captured at BOTH sizes the
    # designer asked for, so the wrapped block is legible at the tight card.
    await _capture(
        "consent-public",
        contact_sheet="consent",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
                attest_revision_cas=True,
            )
        ],
    )
    await _capture(
        "consent-public-40x24",
        size=(40, 24),
        contact_sheet="consent",
        links=[
            dict(
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
                attest_revision_cas=True,
            )
        ],
    )
    # D18: two links, so the hint row teaches `↑↓ select` (not scroll-only).
    await _capture(
        "multi-link-select",
        links=[
            dict(
                local_id=f"a{i}",
                name=f"agent-{i}",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
            for i in (1, 2)
        ],
    )

    # Assert THIS test's own round-2 frames by name (shared OUT dir).
    for name in (
        "narrow-40x24",
        "footer-all-three",
        "footer-door-dropped",
        "consent-public",
        "consent-public-40x24",
        "multi-link-select",
    ):
        svg = OUT / f"{name}.svg"
        assert svg.exists() and "<svg" in svg.read_text()[:200], name


async def test_hub_panel_and_host_start_hidden() -> None:
    """The overlay must not displace the transcript before it is ever opened.

    Regression guard for a real bug: ``HubLinksPanel`` was mounted like the
    usage card but never set ``display = False`` in ``__init__`` (``UsagePanel``
    does). A visible 0-row host in the toast layer shifted the whole transcript
    one cell left at boot — caught only by an unrelated slash-echo test, so this
    pins the invariant directly: both the panel and its host start at zero size.
    """
    import shutil
    import tempfile

    from local_operator.tui.settings import settings_reload
    from local_operator.tui.widgets.hub_links_panel import HubLinksPanel

    case_root = Path(tempfile.mkdtemp(prefix="hub-links-hidden-", dir=OUT))
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(case_root)
    try:
        settings_reload()
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one(HubLinksPanel)
            host = app.query_one("#hub-links-host")
            assert panel.display is False, "the hub panel must start hidden"
            assert host.region.width == 0 and host.region.height == 0, (
                "a hidden overlay host must claim no cells",
                host.region,
            )
    finally:
        shutil.rmtree(case_root, ignore_errors=True)


async def test_hint_row_is_painted_from_the_catalogue() -> None:
    """D6: the panel TEACHES its keys — the hint row is back, catalogue-built.

    The round-2 regression dropped the key-hint row (a catalogue move left the
    binding set with no hint), so the panel took focus and bound esc/q/r/↑↓/
    enter/s/a with nothing teaching them. This drives the real app and asserts
    the painted card carries the catalogue's key words.
    """
    import shutil
    import tempfile

    from local_operator.tui.settings import settings_reload

    case_root = Path(tempfile.mkdtemp(prefix="hub-links-hint-", dir=OUT))
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(case_root)
    try:
        settings_reload()
        _seed_link(
            case_root,
            local_id="a1",
            name="coder",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_hub_links()
            await pilot.pause()
            await pilot.pause()
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            # The catalogue's own words, from the real i18n runtime.
            assert "esc close" in painted, painted
            assert "s subscribe" in painted, painted
            assert "a auto-push" in painted, painted
            # D17: `enter opens policy` advertised a dead stub, so it is gone.
            assert "enter" not in painted, painted
    finally:
        shutil.rmtree(case_root, ignore_errors=True)


async def test_more_rows_indicator_reports_the_window() -> None:
    """D11: a list longer than the body cap reports its range, not silence."""

    import shutil
    import tempfile

    from local_operator.tui.settings import settings_reload

    case_root = Path(tempfile.mkdtemp(prefix="hub-links-more-", dir=OUT))
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(case_root)
    try:
        settings_reload()
        for i in range(20):
            _seed_link(
                case_root,
                local_id=f"a{i}",
                name=f"agent-{i:02d}",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_hub_links()
            await pilot.pause()
            await pilot.pause()
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            # The indicator reads `n–m/total` (the sessions-list precedent) and
            # reports the LINKS actually shown, not a fixed `1`.
            assert re.search(r"\d+–\d+/20", painted), painted
    finally:
        shutil.rmtree(case_root, ignore_errors=True)


async def test_capture_round4_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The frames round-4 design review asked for (D21/D22).

    * 16 links at 80x24 with the consent armed — the full block must paint.
    * 16 links with the selection on the LAST row — the name must be visible
      (D21: the round-3 N3 header fix dropped it at exactly the cap).
    * 3 links at 40x24 with the consent armed — the tightest width.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)

    sixteen = [
        dict(
            local_id=f"a{i:02d}",
            name=f"agent-{i:02d}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
            attest_revision_cas=True,
        )
        for i in range(16)
    ]
    # D22: 16 links at 80x24 with the consent armed (selected row 0, public).
    await _capture(
        "consent-16links-80x24",
        size=(80, 24),
        contact_sheet="consent",
        links=sixteen,
    )
    # D21: 16 links, selection on the LAST row, painted (name visible).
    await _capture(
        "selection-last-16links",
        size=(100, 40),
        contact_sheet="select-last",
        links=sixteen,
    )
    # D22: 3 links at 40x24 with the consent armed.
    await _capture(
        "consent-3links-40x24",
        size=(40, 24),
        contact_sheet="consent",
        links=sixteen[:3],
    )

    for name in ("consent-16links-80x24", "selection-last-16links", "consent-3links-40x24"):
        svg = OUT / f"{name}.svg"
        assert svg.exists() and "<svg" in svg.read_text()[:200], name


async def test_capture_round5_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The frames round-5 design review asked for (D27/D28/D29/D32/D35).

    * a mixed-kind frame with the highlight on EACH kind (agent/team/skill);
    * consent at 100x30 (composer visible, no overlap);
    * 16 links at 100x30 (no dock overlap);
    * the mid-section window state carrying its section title (D35).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)

    mixed = [
        dict(
            kind="agent",
            local_id="a1",
            name="coder",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        ),
        dict(
            kind="team",
            local_id="t1",
            name="crew",
            visibility="org",
            tenant="acme",
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        ),
        dict(
            kind="skill",
            local_id="s1",
            name="deploy",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        ),
    ]
    # D27: the highlight on each kind (indices 0,1,2 in paint order).
    for idx, kind in enumerate(("agent", "team", "skill")):
        await _capture(
            f"mixed-highlight-{kind}",
            size=(100, 30),
            contact_sheet=f"select-index:{idx}",
            links=mixed,
        )
    # D22/D28: consent at 100x30 (composer on screen, no overlap).
    await _capture(
        "consent-100x30",
        size=(100, 30),
        contact_sheet="consent",
        links=[{**mixed[0], "attest_revision_cas": True}],
    )
    # D28: 16 links at 100x30 (no dock overlap).
    sixteen = [
        dict(
            local_id=f"a{i:02d}",
            name=f"agent-{i:02d}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        for i in range(16)
    ]
    await _capture("many-16links-100x30", size=(100, 30), links=sixteen)
    # D35: a mid-section window (8 agents + 8 skills, select the last skill).
    mixed_many = [
        dict(
            kind="agent",
            local_id=f"a{i}",
            name=f"agent-{i}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        for i in range(8)
    ] + [
        dict(
            kind="skill",
            local_id=f"s{i}",
            name=f"skill-{i}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        for i in range(8)
    ]
    await _capture(
        "mid-section-skills",
        size=(60, 20),
        contact_sheet="select-index:15",
        links=mixed_many,
    )

    for name in (
        "mixed-highlight-agent",
        "mixed-highlight-team",
        "mixed-highlight-skill",
        "consent-100x30",
        "many-16links-100x30",
        "mid-section-skills",
    ):
        svg = OUT / f"{name}.svg"
        assert svg.exists() and "<svg" in svg.read_text()[:200], name


async def test_capture_tab_esc_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D29 frame: tab then esc leaves the overlay CLOSED (not stranded)."""

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)
    await _capture(
        "tab-then-esc",
        size=(100, 30),
        contact_sheet="tab-esc",
        links=[
            dict(
                kind="agent",
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            )
        ],
    )
    svg = OUT / "tab-then-esc.svg"
    assert svg.exists() and "<svg" in svg.read_text()[:200]


async def test_capture_round6_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The frames round-6 asked for (D37/D38/D39).

    * D37: 8 agents + 8 skills at 80x24, selection on the last agent, so a
      trailing `Skills` header with no rows would show as an empty section.
    * D38: the same link highlighted after an insert-above (selection follows the
      link, not the index).
    * D39: the consent escape route where `<id>` fits but the id line does not
      (the `lop hub links` pointer stands in).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    OUT.mkdir(parents=True, exist_ok=True)

    many = [
        dict(
            kind="agent",
            local_id=f"a{i}",
            name=f"agent-{i}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        for i in range(8)
    ] + [
        dict(
            kind="skill",
            local_id=f"s{i}",
            name=f"skill-{i}",
            visibility="public",
            tenant=None,
            subscribe=True,
            auto_push=False,
            state="up-to-date",
            error=None,
        )
        for i in range(8)
    ]
    # D37: the orphan-header state at 80x24 with the last agent selected.
    await _capture(
        "orphan-header-80x24",
        size=(80, 24),
        contact_sheet="select-index:7",
        links=many,
    )
    # D38: the selection follows the link across a reorder (reorder sheet).
    await _capture(
        "selection-follows-link",
        size=(100, 30),
        contact_sheet="reorder",
        links=[
            dict(
                kind="agent",
                local_id="a1",
                name="alpha",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            ),
            dict(
                kind="agent",
                local_id="a2",
                name="crew",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
            ),
        ],
    )
    # D39: an id too long for the line ⇒ the `lop hub links` pointer stands in.
    await _capture(
        "consent-id-pointer",
        size=(80, 30),
        contact_sheet="consent",
        links=[
            dict(
                kind="agent",
                local_id="a1",
                name="coder",
                visibility="public",
                tenant=None,
                subscribe=True,
                auto_push=False,
                state="up-to-date",
                error=None,
                attest_revision_cas=True,
            )
        ],
    )

    for name in ("orphan-header-80x24", "selection-follows-link", "consent-id-pointer"):
        svg = OUT / f"{name}.svg"
        assert svg.exists() and "<svg" in svg.read_text()[:200], name

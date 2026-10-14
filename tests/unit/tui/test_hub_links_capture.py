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


def _seed_link(
    root: Path,
    *,
    local_id: str,
    name: str,
    visibility: str,
    tenant: str | None,
    subscribe: bool,
    auto_push: bool,
    state: str,
    error: str | None,
) -> LinkIdentity:
    """Write a real baseline + status row so the inventory enumerates it as a link."""
    identity = LinkIdentity("agent", local_id, HUB, visibility, tenant, f"h-{local_id}")
    record = prov.make_record(
        "agent", local_id, f"h-{local_id}", tenant, {"instructions": "x"}, "pull"
    )
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
            f"agent:{local_id}", st.blank_item("agent", local_id, name, f"h-{local_id}", tenant)
        )
        item.update(state=state, error_class=error, last_checked_at="2026-10-14T00:00:00Z")

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
            elif contact_sheet == "command":
                # The USER's route: type the command into the real editor and
                # submit it, so the capture proves the registry entry dispatches.
                editor = app._editor()
                editor.focus()
                editor.load_text("/hub")
                await pilot.press("enter")
                await pilot.pause()
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

    svgs = sorted(OUT.glob("*.svg"))
    assert len(svgs) == 9
    for svg in svgs:
        assert "<svg" in svg.read_text()[:200]


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

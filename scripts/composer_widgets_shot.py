"""Rendered evidence for the composer widget-visibility flags (``display.composer.*``).

One running ``OperatorApp`` at 100x30 produces every state, so the frames differ
only by the configuration under test:

    01-default                  every key at its default (true)
    02-cost-hidden              display.composer.cost = false
    03-all-readings-hidden      all six segment keys false
    04-band-hidden              display.composer.band = false
    05-chevron-hidden           display.composer.chevron = false
    06a-live-flip-default       back to the default state
    06b-live-flip-cost-hidden   after a write from "another process"

06 is the pair the operator asked about: between 06a and 06b the script writes
``display.composer.cost: false`` into the isolated config dir BELOW the local
notify hook (the shape another process's ``lop config edit`` lands as) and
pumps the app's config watcher; the second frame is the same process still
running — the flag took effect without a restart. The other state frames use
the local delivery (``settings_io.write_setting``), which is the /settings
page's own call.

    LOP_REPO=<tree> env -u NO_COLOR TERM=xterm-256color \
        .venv/bin/python scripts/composer_widgets_shot.py OUTDIR [SIZE]

``SIZE`` defaults to ``100x30``. The band's segments are pushed explicitly
rather than left to the boot path: a capture whose inputs vary between runs
cannot back a before/after comparison (the same rule ``status_band_row_shot``
states). On a build without the composer keys (a base before-frame tree) only
``01-default`` is meaningful; the script says so and captures exactly that.

Isolation comes from ``scripts.probe_isolation``, imported before any
application module — plus the ``LOP_*`` runtime variables a session's own
environment exports, which that helper does not scrub yet (AGENTS.md,
"Isolating a run": a child that inherits them silently runs on its parent's
provider/model or adopts the wrong session). Scrubbed below, before any
``local_operator`` import, and by the runtime-hazard prefixes only so the shot
scripts' own ``LOP_REPO``/``LOP_SHOT_*`` knobs survive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("LOP_REPO") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: E402,F401  -- MUST be the first import

for _key in tuple(os.environ):
    if (
        _key.startswith("LOP_MOBILE_CHILD_")
        or _key.startswith("LOP_RUNTIME_")
        or _key == "LOP_MODEL_SELECTION_OVERRIDE"
    ):
        os.environ.pop(_key)

import asyncio  # noqa: E402

from rich.cells import cell_len  # noqa: E402
from rich.text import Text  # noqa: E402
from textual.widgets import Static  # noqa: E402

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.config_watch import process_watcher  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])  # from probe_isolation

MODEL_LABEL = "openrouter/moonshotai/kimi-k2-thinking"
CWD = "/Users/damian/local-operator"
NAME = "Composer widget visibility"

#: The six segment keys, in band order.
SEGMENT_KEYS = (
    "display.composer.model",
    "display.composer.cwd",
    "display.composer.context",
    "display.composer.rate",
    "display.composer.cost",
    "display.composer.duration",
)

#: The ladder ids the report prints, one per config suffix.
SHOWN_IDS = ("model", "cwd", "context", "last-rate", "cost", "duration", "name")


def _supported() -> bool:
    """Whether this tree reads the composer keys (false on a base before-frame)."""
    return "display.composer.band" in settings_io.BY_KEY


def _seed_transcript(app: OperatorApp) -> None:
    """A short settled conversation, so the composer sits under real content."""
    app._append_block(UserBlock("Can I hide the cost from the tok/s row in the composer?"))
    answer = AssistantBlock()
    answer.update_text(
        "Yes — each piece of the band has a display.composer key now, and edits "
        "apply to the running TUI: set display.composer.cost to false and the "
        "row re-fits without it."
    )
    answer.finalize_text()
    app._append_block(answer)


def _push_state(app: OperatorApp) -> None:
    """Pin every segment, so two runs of this script produce the same row."""
    status = app._status
    assert status is not None
    status.update(
        model_label=MODEL_LABEL,
        effort="high",
        cwd=CWD,
        context_tokens=496_000,
        context_window=1_000_000,
        last_rate="217 tok/s",
        cost="$4.21",
        conversation_name=NAME,
        streaming=False,
        failed=False,
        approvals_auto=False,
        approvals_always=False,
    )
    # 41m1s of active time, settled: activity_started_at=None keeps the clock
    # from growing between frames.
    status.seed_duration(active_seconds=2461.0, activity_started_at=None)


def _set_local(keys: tuple[str, ...], value: bool) -> None:
    """Write through the facade — the /settings page's delivery, this process."""
    manager = ConfigManager(CONFIG_DIR)
    for key in keys:
        settings_io.write_setting(manager, settings_io.BY_KEY[key], value)


def _set_below_hook(key: str, value: bool) -> None:
    """Write like ANOTHER process: below the notify hook, for the watcher to find."""
    setting = settings_io.BY_KEY[key]
    settings_io._store(ConfigManager(CONFIG_DIR), setting.path, value)


def _report(app: OperatorApp, label: str) -> None:
    """The numbers behind the still: what the widgets say, and the painted row."""
    band = app.query_one("#status-band", Static)
    chevron = app.query_one("#prompt-chevron", Static)
    status = app._status
    print(f"[{label}] band.display={band.display} chevron.display={chevron.display}")
    if not band.display:
        # The Static keeps its content when hidden; saying so keeps the line
        # below from reading as "the band is still painted".
        print("  note: band hidden — content retained but not laid out")
    content = band.content
    # ``Static.content`` is typed as the general renderable union; the band
    # always updates it with a rich ``Text``, so narrow rather than cast.
    painted = content.plain if isinstance(content, Text) else str(content)
    print(f"  painted({cell_len(painted)}): {painted!r}")
    if status is not None:
        shown = ", ".join(f"{s}={status.is_showing(s)}" for s in SHOWN_IDS)
        print(f"  is_showing: {shown}")


async def _settle(app: OperatorApp, pilot, *, subscribed: bool = True) -> None:
    for _ in range(200):
        if app._session is not None and (
            not subscribed or app._unsubscribe_config_watch is not None
        ):
            return
        await pilot.pause()
    raise AssertionError("the app never adopted a session / subscribed to config")


async def main() -> None:
    out = Path(sys.argv[1])
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    cols, rows = (int(v) for v in size.split("x"))
    out.mkdir(parents=True, exist_ok=True)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, rows)) as pilot:
        await _settle(app, pilot)
        _seed_transcript(app)
        _push_state(app)
        await pilot.pause()

        save_capture(app, out / "01-default.svg")
        _report(app, "01-default")

        if not _supported():
            print(
                "note: this tree does not read display.composer.* (base build); "
                "only 01-default is meaningful here."
            )
            return

        _set_local(("display.composer.cost",), False)
        await pilot.pause()
        save_capture(app, out / "02-cost-hidden.svg")
        _report(app, "02-cost-hidden")

        _set_local(SEGMENT_KEYS, False)
        await pilot.pause()
        save_capture(app, out / "03-all-readings-hidden.svg")
        _report(app, "03-all-readings-hidden")

        _set_local(SEGMENT_KEYS, True)
        _set_local(("display.composer.band",), False)
        await pilot.pause()
        save_capture(app, out / "04-band-hidden.svg")
        _report(app, "04-band-hidden")

        _set_local(("display.composer.band",), True)
        _set_local(("display.composer.chevron",), False)
        await pilot.pause()
        save_capture(app, out / "05-chevron-hidden.svg")
        _report(app, "05-chevron-hidden")

        # Back to the shipped shape, so the pair below is a default frame and
        # the frame after ONE config change — nothing else differs.
        _set_local(("display.composer.chevron",), True)
        await pilot.pause()
        save_capture(app, out / "06a-live-flip-default.svg")
        _report(app, "06a-live-flip-default")

        _set_below_hook("display.composer.cost", False)
        process_watcher(CONFIG_DIR).poll_now()
        await pilot.pause()
        save_capture(app, out / "06b-live-flip-cost-hidden.svg")
        _report(app, "06b-live-flip-cost-hidden")
        print("live flip: no restart — one process produced 06a and 06b")


asyncio.run(main())

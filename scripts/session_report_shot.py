"""Capture the real /session slash path with synthetic ledger data.

Usage: python scripts/session_report_shot.py OUTDIR 80x24 populated|empty|unavailable
Copy this script into a preserved base worktree to capture the pre-command path
with the same fixture. No provider request, live session or operator config is used.

The ``opener-only`` / ``stored-title`` / ``provisional`` scenarios are the
NAMING evidence: they boot the same app against a session whose store of record
is empty, which is the state a live conversation is in until its generated title
lands, and one a RESUMED conversation that never got one stays in. Each parks the
ledger read so ``first-frame.svg`` is a fact about the app's ordering — the
header is painted from memory before any disk result can land — rather than about
which side of a race the capture won.

``reported-as-is`` regenerates the headline frame of the PR that added these
scenarios: the reported conversation's state once its generated title had landed.
The transcript that frame came from is the operator's own live session, so the
scenario seeds the records the header resolves against — the opener and the
journalled title — through the same real ``Transcript`` writer, rather than a
byte copy that would not be committable. The report-time shape (same opener, no
title yet) is the ``opener-only`` scenario.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import Any  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.harness.types import Message, ModelSpec, TextContent  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402


class DiagnosticSession(FakeSession):
    frontend_state: FrontendSessionState

    @property
    def session_id(self) -> str:
        # A REAL-SHAPED ID: the harness generates ``uuid4().hex[:12]``
        # (``harness/types.py``, ``fork.py``), and every session directory on a
        # live machine is exactly 12 characters. The previous 40-character
        # demo string was ~3.3x anything reachable, so it manufactured a header
        # crop at 50 columns that no operator can hit — a defect in the
        # evidence, not in the screen.
        return "a3f9c21b7e40"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6", context_window=200000)

    @property
    def effective_model(self) -> ModelSpec:
        return ModelSpec(provider="openai", model_id="gpt-5.2", context_window=200000)

    @property
    def model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"

    @property
    def effective_model_label(self) -> str:
        return "openai/gpt-5.2"


#: The populated demo session, as (context, output, reasoning, duration_ms,
#: purpose, ok) per request. Module scope so a sibling capture script can
#: seed the SAME session — one fixture, one place to change it.
POPULATED_SHAPE = [
    # (context, output, reasoning, duration_ms, purpose, ok)
    (8_400, 900, 120, 1_850, "turn", True),
    (14_200, 1_400, 260, 2_310, "turn", True),
    (21_800, 620, 0, 1_240, "aside", True),
    (33_500, 2_900, 800, 4_120, "turn", True),
    (46_100, 1_100, 180, 2_050, "turn", True),
    (52_700, 480, 0, 980, "naming", True),
    (61_400, 3_600, 1_240, 5_870, "turn", True),
    (74_900, 1_750, 320, 2_640, "turn", True),
    (88_300, 210, 0, 1_420, "turn", False),
    (96_800, 2_150, 540, 3_180, "turn", True),
    (112_400, 1_320, 210, 2_260, "turn", True),
    (128_900, 4_800, 1_900, 7_240, "turn", True),
    (141_200, 760, 0, 1_510, "aside", True),
    (158_600, 2_400, 620, 3_450, "turn", True),
    (172_300, 1_180, 240, 2_180, "turn", True),
    (186_700, 5_200, 2_100, 8_960, "turn", True),
    # The compaction pair: expensive reads, and the context it buys back.
    (194_100, 3_100, 0, 6_320, "compaction", True),
    (42_600, 1_450, 280, 2_390, "turn", True),
]


def seed_populated(session_id: str) -> None:
    """Write the demo session into the ambient (isolated) ledger.

    A session that VARIES, because a constant fixture makes the two best new
    charts render as a wall of identical bars and a degenerate
    ``2,800 ms-2,800 ms`` timing range (QA Q3, echoed by the design and code
    rounds). Those frames are correct renderings of uniform input, but they are
    what a reader judges the feature by, and they make a working chart look
    broken. ``POPULATED_SHAPE`` is what a real session does: context grows as
    history accumulates, a compaction resets it, a couple of turns are large,
    one fails, and a cheap model handles the short asides. No number is
    load-bearing here - only the spread is.

    Shared with the cost-mode capture so the two never seed different sessions.
    """
    store = AnalyticsStore()
    snapshots = []
    for i, (context, output, reasoning, duration, purpose, ok) in enumerate(POPULATED_SHAPE):
        # The cheap model takes the short non-turn work, so `By model` has a
        # real split to draw rather than one dominant row.
        aside = purpose in ("aside", "naming")
        cache_read = int(context * 0.72)
        snapshots.append(
            replace(
                _snap(
                    session_id=session_id,
                    provider="openai" if aside else "anthropic",
                    model_id="gpt-5.2-mini" if aside else "claude-sonnet-4-6",
                    context=context,
                    input_tokens=context - cache_read - 800,
                    cache_read=cache_read,
                    cache_write=800,
                    output_tokens=output,
                    reasoning=reasoning,
                    # Roughly list price for the tokens, so `t` (cost) gives a
                    # different ordering rather than a rescale of the same one.
                    cost_micro=int(context * 0.55 + output * 8.5),
                    chars={
                        "system_prompt": 400,
                        "custom_instructions": 110,
                        "tool_inventory": 150,
                        "tool_schemas": 250,
                        # Conversation and tool results grow with context; a
                        # flat split makes `Where input went` identical on
                        # every frame.
                        "conversation": 300 + i * 95,
                        "tool_results": 120 + i * 60,
                    },
                    ts_ms=1788602400000 + i * 47000,
                    ok=ok,
                ),
                request_id=f"logical-request-{i:03d}",
                purpose=purpose,
                outcome="ok" if ok else "error",
                duration_ms=duration,
                ttft_ms=int(duration * 0.19) + 180,
                preparation_ms=18 + (i % 5) * 7,
            )
        )
    store.record_batch(snapshots)
    store.close()


#: The reported conversation's shape: opened with a question the sidebar names
#: the row by, never renamed, and named by a model call that landed nothing.
SESSION_OPENER = "There seems to be a weird issue where on resume certain tools stay open"
#: The one a naming call (or `/rename`) did store — the case the header already
#: handled, captured so the pair shows the store of record is still in charge.
SESSION_TITLE = "Resume keeps stale tool rows"
#: The title the REPORTED conversation's naming call actually stored, journalled
#: exactly as the errand writes it. The reported session is the operator's own
#: live conversation, so its transcript cannot be committed; these are the two
#: records the header resolves against (see ``seed_transcript``), which is what
#: makes the headline frame regenerable from the branch by anyone.
SESSION_REPORTED_TITLE = "Resume Duplicate Tools Frozen Calls"
#: Scenarios whose store of record is deliberately EMPTY.
NAMING_SCENARIOS = ("opener-only", "stored-title", "provisional", "reported-as-is")


async def seed_transcript(session_id: str, opener: str, *, title: str | None = None) -> None:
    """Write this session's transcript into the isolated store, and nothing else.

    Through the real ``Transcript`` writer, so the header is proved against the
    bytes a running session leaves behind and read back through the same helper
    the sidebar builds its rows with (``resume.session_name``). ``title`` is
    journalled exactly as the naming errand journals it; with no title the
    transcript is the session that was closed before its naming call landed,
    which is every session resumed from before titles were stored.
    """
    transcript = Transcript(config_dir() / "sessions" / session_id)
    await transcript.append_message(Message(role="user", content=[TextContent(text=opener)]))
    if title is not None:
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": title, "user_set": True}
        )


class _ParkedLedger:
    """``AnalyticsStore.session_report`` held open until the capture releases it.

    A plain object rather than a bound method, because it REPLACES the class
    attribute: an instance without ``__get__`` is not a descriptor, so the call
    arrives here unbound — hence the fresh store, and the source-compatible shape
    for both the pre-fix and post-fix worker.
    """

    def __init__(self, real: Any, release: threading.Event) -> None:  # noqa: ANN401
        self._real = real
        self._release = release

    def __call__(self, session_id: str, **kwargs: Any) -> Any:  # noqa: ANN401
        assert self._release.wait(10)
        return self._real(AnalyticsStore(), session_id, **kwargs)


def _header_line(app: OperatorApp) -> str:
    """The screen's own first line: the conversation label under review."""
    body = getattr(app.screen, "_body", None)
    if body is None:
        return ""
    return str(body.render()).split("\n", 1)[0]


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = sys.argv[2].split("x")
    size = (int(cols), int(rows))
    scenario = sys.argv[3] if len(sys.argv) > 3 else "populated"
    session = DiagnosticSession()
    if scenario not in NAMING_SCENARIOS:
        session.set_conversation_name("Investigate request latency")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=6,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    if scenario == "populated":
        seed_populated(session.session_id)
    elif scenario == "unavailable":
        path = default_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic corrupt ledger")
    elif scenario == "opener-only":
        await seed_transcript(session.session_id, SESSION_OPENER)
    elif scenario == "stored-title":
        await seed_transcript(session.session_id, SESSION_OPENER, title=SESSION_TITLE)
    elif scenario == "reported-as-is":
        # The frame the PR body shows for the reported session once its naming
        # call landed: the same opener the sidebar names the row by, plus the
        # generated title. Regenerable from the branch because both records are
        # constants here — the live transcript they were read from is not.
        await seed_transcript(session.session_id, SESSION_OPENER, title=SESSION_REPORTED_TITLE)
    elif scenario == "provisional":
        # A live conversation naming never returned for: the transcript carries
        # the opener AND the host wears its own stand-in. The header must prefer
        # the stand-in (it is what the band and the tab are showing), and the
        # transcript is what a resumed reload of the same session would use.
        await seed_transcript(session.session_id, SESSION_OPENER)
    ledger = default_db_path()
    before_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None
    app = OperatorApp(lambda: _factory(session))
    # Parked for the naming scenarios only: see ``_ParkedLedger``.
    park = threading.Event() if scenario in NAMING_SCENARIOS else None
    real_session_report = AnalyticsStore.session_report
    if park is not None:
        AnalyticsStore.session_report = _ParkedLedger(  # type: ignore[method-assign]
            real_session_report, park
        )
    first_frame_header: str | None = None
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if scenario == "provisional":
            # Exactly what `_submit_prompt` does with the opening message,
            # before the turn runs: the stand-in the band wears.
            app._show_provisional_name(SESSION_OPENER)
        await _submit(pilot, app, "/session")
        if park is not None:
            # THE FIRST FRAME: the screen is pushed and painted from memory while
            # the disk read is still parked. Pre-fix this read "Untitled session"
            # and corrected itself once the worker answered — the flash, and the
            # whole reason the stand-in is handed in before the push.
            save_capture(app, str(out / "first-frame.svg"))
            first_frame_header = _header_line(app)
            park.set()
        await app.workers.wait_for_complete()
        # Unpatched the moment the read it parked has finished: the rest of the
        # run — the ledger digest, the closed state — must exercise the real
        # store. No try/finally: a raise here is a capture that produced no
        # evidence, and nothing downstream reads the patched attribute.
        setattr(AnalyticsStore, "session_report", real_session_report)
        await pilot.pause()
        save_capture(app, str(out / "opened.svg"))
        await pilot.pause()
        save_capture(app, str(out / "settled.svg"))
        if type(app.screen).__name__ == "SessionScreen":
            for page in range(1, 7):
                await pilot.press("pagedown")
                await pilot.pause()
                save_capture(app, str(out / f"page-{page}.svg"))
        await pilot.press("end")
        await pilot.pause()
        save_capture(app, str(out / "bottom.svg"))
        screen = app.screen
        scroll = getattr(screen, "_scroll", None)
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "scenario": scenario,
            "size": size,
            # The surface this PR is about, as bytes: the header's own line from
            # the frame the screen is PUSHED with (before any worker result) and
            # from the settled frame, so the stills have a machine-checkable twin.
            "first_frame_header": first_frame_header,
            "settled_header": _header_line(app),
            "screen": type(screen).__name__,
            "screen_geometry": {
                "size": list(screen.size),
                "virtual_size": list(screen.virtual_size),
                "region": list(screen.region),
            },
            "prompts": session.prompts,
            "ledger_unchanged": before_digest
            == (hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None),
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_x": scroll.max_scroll_x,
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
        }
        await pilot.press("escape")
        await pilot.pause()
        save_capture(app, str(out / "closed.svg"))
        metrics["composer_focused_after_close"] = app.focused is app.query_one(Editor)
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics))


if __name__ == "__main__":
    asyncio.run(main())

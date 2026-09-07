"""Capture the /analytics per-session table where the LABELS are under stress.

Usage: python scripts/analytics_label_shot.py OUTDIR 140x45 [PAGES]

Sibling of ``analytics_collision_shot.py``, which proves the row-IDENTITY fix
(one row per session, no silent merge). This one is about what those rows SAY:
design review round 1 found the disambiguated label hard-cut mid-word with no
ellipsis, composed against a fixed 32 characters while a 140-column frame
offered 48, and overran the narrow name column so the number columns went
ragged. Those are label defects, and the frame that shows them is not the top of
the table.

**Collisions begin around rank 37 by cost**, so a top-of-table frame cannot show
them — that is precisely the mistake that let round 1 slip. ``PAGES`` (default
3) is how many page-downs to take before the shot, which is what puts the
composed ``<role> · <parent title> · <frag>`` rows on screen.

The fixture is a REAL ledger shape reduced to a readable frame: expensive
uniquely-named parents at the top (so the ranks below them are reached by
scrolling, as on the real ledger), then the mixed region the review is about —
composed sibling rows that collide, long plain-named rows that are cut without
colliding, and eval rows. No provider request, live session or operator config
is used; ``probe_isolation`` re-homes HOME and the config dir before any
application import, and the frame asserts its own ledger was untouched.
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
import re  # noqa: E402
from dataclasses import replace  # noqa: E402
from xml.etree import ElementTree as ET  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.harness.types import ModelSpec  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: Expensive uniquely-named sessions. They sort to the TOP by cost, which is
#: what pushes the interesting rows below the fold — the same reason the real
#: ledger's collisions sit at rank 37.
UNIQUE_ROWS = [
    ("8f21ac93bb04", "OSWorld benchmark evaluation and scoring", 3433.96),
    ("2b77de10c9a5", "Article search service integration", 2957.68),
    ("d40c81ff7a13", "Revise Agent Risk Assessments PDF export", 2950.37),
    ("6b1d0e97ca42", "Harden Against Credential Leak Paths in the runner", 2210.11),
    ("0f3a8d2e5c19", "Implement and release feature retention controls", 1980.44),
    ("77c4e0a1b8d3", "Fix duplicated peer message display in the sidebar", 1755.02),
    ("c19b4f7d3a08", "Improve lop README features and installation guide", 1502.75),
]

#: Composed sibling labels: one parent, one role, so the backfill mints a
#: byte-identical name for every child. Two DIFFERENT parents here, both with
#: long titles, because that is the case review F7 pointed at — two groups whose
#: names agree well past the cut.
COLLIDING_GROUPS = {
    "reviewer · Article-search-svc schema review and sign-off": [
        ("4fa0e21c7b93", 41.60),
        ("d677b3f9012a", 40.85),
        ("1a24c8e70bf6", 37.47),
        ("2c39ab5410de", 36.78),
        ("4a5e77c2b901", 36.70),
        ("9b13fe4a2c78", 35.94),
        ("c802d51b3ae6", 35.31),
        ("71c9042fbd85", 34.88),
    ],
    "qa-tester · Update Provider Onboarding and OAuth UX": [
        ("3d6a1b8e04c7", 33.90),
        ("3a58f2c91d6b", 33.42),
        ("4e6d70b2a8f1", 32.53),
    ],
    # A composed label whose parent title is short enough that a narrow budget
    # condenses it away entirely — the dangling-separator case.
    "architect · Auto-update inactive session names": [
        ("1aae5c93b207", 31.20),
        ("9085d1f4a63e", 30.75),
    ],
}

#: Plain named rows long enough to be cut WITHOUT colliding. Design D5: these
#: are the majority of rows once naming lands, and a silent cut gives the reader
#: no way to tell a complete title from a fragment.
LONG_PLAIN_ROWS = [
    ("aa01bb02cc03", "coder · Fix subagent effort levels per model", 29.80),
    ("bb02cc03dd04", "ux-reviewer · Review and merge the wake scheduler flow", 28.44),
    ("cc03dd04ee05", "copy-reviewer · Improve lop README features and guide", 27.13),
]

#: Eval rows (design D4): with a task id the row says WHICH task ran; without
#: one it falls back to the episode id, which is the shape the review saw.
EVAL_ROWS = [
    ("lop-eval-ep-0a52bce248bd", "eval osworld/chrome-0421 · claude-sonnet-4-6", 26.86),
    ("lop-eval-ep-290e1c3f86b7", "eval osworld/vscode-0117 · claude-sonnet-4-6", 25.19),
]

#: Sessions whose directory is gone, so the backfill can recover no name and the
#: row is honestly a bare id. Present so the frame shows the real MIX.
UNNAMED_ROWS = [("98bfe7686ffe", 24.99), ("21f046e9312b", 23.48)]


def seed(store: AnalyticsStore) -> None:
    """Write the fixture into the isolated ledger, names included."""
    rows: list[tuple[str, str, float]] = list(UNIQUE_ROWS)
    for label, members in COLLIDING_GROUPS.items():
        rows += [(sid, label, usd) for sid, usd in members]
    rows += LONG_PLAIN_ROWS
    rows += EVAL_ROWS
    rows += [(sid, "", usd) for sid, usd in UNNAMED_ROWS]

    snapshots = []
    index = 0
    for session_id, name, usd in rows:
        # A couple of calls per session so `calls` is not uniformly 1; the
        # dollar figure is what this frame is read for.
        for part in range(2):
            snapshots.append(
                replace(
                    _snap(
                        session_id=session_id,
                        provider="anthropic",
                        model_id="claude-sonnet-4-6",
                        context=120_000 + index * 900,
                        input_tokens=40_000,
                        cache_read=70_000,
                        cache_write=2_000,
                        output_tokens=6_000,
                        reasoning=1_200,
                        cost_micro=int(usd * 1_000_000 / 2),
                        chars={"conversation": 4000, "tool_results": 2200},
                        ts_ms=1788602400000 + index * 61000,
                        ok=True,
                    ),
                    request_id=f"req-{index:04d}-{part}",
                    purpose="turn",
                    outcome="ok",
                    duration_ms=2400,
                )
            )
            index += 1
        if name:
            store.upsert_session_name(session_id, name)
    store.record_batch(snapshots)


class LedgerSession(FakeSession):
    frontend_state: FrontendSessionState

    @property
    def session_id(self) -> str:
        return "a3f9c21b7e40"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6", context_window=200000)

    @property
    def model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"


def _svg_rows(path: Path) -> list[str]:
    """The text rows of a saved capture, in paint order.

    Read back out of the SVG rather than off the widget so the audited rows are
    the ones actually IN the image beside them — round 1's critique was that a
    claim about the table is not a claim about the frame, and a frame shows only
    a window onto a 27-row table. Rich paints spaces as U+00A0 to stop SVG
    collapsing them, so they are mapped back to real spaces before any column
    arithmetic.
    """
    namespace = {"s": "http://www.w3.org/2000/svg"}
    lines: dict[float, list[tuple[float, str]]] = {}
    for node in ET.parse(path).getroot().findall(".//s:text", namespace):
        y = float(node.get("y") or 0)
        for span in node.findall("s:tspan", namespace):
            lines.setdefault(y, []).append((float(span.get("x") or 0), span.text or ""))
    return [
        "".join(text for _, text in sorted(lines[y])).replace("\u00a0", " ").replace("\n", "")
        for y in sorted(lines)
    ]


def audit(rows: list[str]) -> dict[str, object]:
    """The defect counts the design review asked for, read off the FRAME.

    Derived from the rendered rows rather than re-deriving labels from the
    model, so the numbers describe what is actually on screen.
    """
    ends = {}
    for line in rows:
        key = line.index(" tokens") + len(" tokens")
        ends[key] = ends.get(key, 0) + 1
    # The label is everything before the run of padding that opens the number
    # columns. Leading indent is stripped FIRST: a row read back from the SVG
    # carries the frame's own left margin, and splitting before stripping made
    # every name the empty string.
    names = [re.split(r"\s{2,}", line.strip())[0] for line in rows]
    return {
        "tokens_end_columns": dict(sorted(ends.items())),
        "alignment_classes": len(ends),
        "labels_marked_cut": sum(1 for n in names if n.endswith("…") or "… ·" in n),
        "labels_with_dangling_separator": sum(
            1 for n in names if n.rpartition(" · ")[0].rstrip("…").rstrip().endswith("·")
        ),
        "distinct_labels": len(set(names)),
    }


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows_ = sys.argv[2].split("x")
    size = (int(cols), int(rows_))
    pages = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    store = AnalyticsStore()
    seed(store)
    store.close()

    session = LedgerSession()
    session.set_conversation_name("Analytics session naming")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=1,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    ledger = default_db_path()
    before_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        save_capture(app, str(out / "top.svg"))

        screen = app.screen
        # Page down into the region where the composed labels live. A top-of-
        # table frame shows only the expensive uniquely-named rows, which render
        # identically before and after and therefore prove nothing.
        for _ in range(pages):
            await pilot.press("pagedown")
            await pilot.pause()
        await pilot.pause()
        save_capture(app, str(out / "labels.svg"))

        render_lines = getattr(screen, "render_lines_for_test", None)
        body = "\n".join(render_lines()) if render_lines is not None else ""
        scroll = getattr(screen, "_scroll", None)
        session_block = body.split("By session", 1)[-1]
        table_rows = [ln.strip() for ln in session_block.splitlines() if " tokens" in ln]
        # What the FRAME shows, which is a WINDOW onto the table, not all of it.
        # Recovered from the SVG that was just saved, so these rows are
        # literally the ones a reader sees in the image beside this JSON — the
        # whole point of the round-1 critique was that a claim about the table
        # is not a claim about the frame.
        visible = [ln for ln in _svg_rows(out / "labels.svg") if " tokens" in ln]
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "size": size,
            "pages_scrolled": pages,
            "screen": type(screen).__name__,
            "session_rows_rendered": len(table_rows),
            "session_rows_seeded": len(
                UNIQUE_ROWS
                + [m for g in COLLIDING_GROUPS.values() for m in g]
                + LONG_PLAIN_ROWS
                + EVAL_ROWS
                + UNNAMED_ROWS
            ),
            "rows_on_screen": len(visible),
            "audit_all_rows": audit(table_rows),
            "audit_on_screen": audit(visible) if visible else None,
            "ledger_unchanged": before_digest
            == (hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None),
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
            "rows": table_rows,
            "on_screen": visible,
        }
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in metrics.items() if k not in ("rows", "on_screen")}, indent=2
            )
        )
        for row in visible:
            print("   ", row)


if __name__ == "__main__":
    asyncio.run(main())

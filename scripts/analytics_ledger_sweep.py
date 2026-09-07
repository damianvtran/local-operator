"""Sweep the /analytics tables across a width band against a REAL ledger copy.

Usage: python scripts/analytics_ledger_sweep.py LEDGER_COPY OUT.json [LO HI]

Sibling of ``analytics_width_band_shot.py``, which drives the real app against a
seeded fixture and exports frames. This one answers the question a fixture
cannot: does the report fit on the operator's ACTUAL data, whose shape nobody
chose? It is arithmetic-only (``build_report`` against the same content-box
width ``AnalyticsScreen._card_width`` computes) so it can sweep a wide band
across several revisions cheaply, and so it can run against a ledger far too
large and too private to commit.

It exists because of design review D11, and the shape of that finding is the
reason to keep it. Round 2 fixed a width overrun, round 3 confirmed the fix and
found the SAME class of error one column further along: ``_row_overhead``
measured the cost column from the data but budgeted the calls column at a
literal 4 cells. Every fixture in the tree renders 1-2 digit call counts, so
every test and every exported frame agreed the table fitted, while the
operator's ledger — ``anthropic`` at 317,977 calls, eight sessions past 9,999 —
lost its ``% cache`` column on the 13 most expensive rows of both tables across
terminals 104-123. Three revisions of arithmetic agreed with three revisions of
fixture and all six disagreed with the data.

So the rule this script mechanises: **a column whose width comes from a literal
rather than a measurement is the next instance of that bug**, and the only
evidence that can see it is a sweep over real magnitudes. Point it at a COPY of
a ledger (never the live file — it is opened read-only and its digest is checked
before and after, but a copy costs nothing and removes the question) and it
prints, per terminal width, the content box, the widest row the report composes,
the resulting gutter and how many rows lose their rightmost column.

    cp -c ~/.local-operator/analytics.db /tmp/ledger.db
    python scripts/analytics_ledger_sweep.py /tmp/ledger.db /tmp/sweep.json 96 160

A negative gutter is a clipped column. Run it on the branch and on the merge
base; a band that is present on the branch and absent on the base is a
regression the fixtures will not show you.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import, for the same
# reason the shot scripts do: an inherited CMUX id has renamed a real workspace.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

from rich.cells import cell_len  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.tui.widgets.analytics_panel import (  # noqa: E402
    _SCROLLBAR_GUTTER,
    build_report,
)
from local_operator.tui.widgets.tool_card import truncate_cells  # noqa: E402


def card_width(terminal: int) -> int:
    """The content box ``AnalyticsScreen._card_width`` yields for a terminal.

    Mirrors that method rather than calling it, so the sweep needs no running
    app and can be pointed at a revision whose screen class differs. The two
    must be changed together; the band shot script is the check that they agree,
    because it measures the real ``scrollable_content_region``.
    """
    return max(40, min(140, int(terminal * 0.9) - 6)) - _SCROLLBAR_GUTTER


def sweep(ledger: Path, lo: int, hi: int) -> list[dict[str, object]]:
    """Render every width in ``[lo, hi]`` and report whether the rows fit."""
    # Work on a copy inside the isolated config dir: the store opens a WAL
    # connection, and the caller's file should not even be written to by SQLite
    # journalling. The digest check below is belt to that braces.
    work = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "analytics.db"
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ledger, work)
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    store = AnalyticsStore(work)
    aggregate = store.aggregate()
    # The report reads names off this attribute; without it every session row
    # renders as a hex id and the name column is measured against the wrong data.
    setattr(aggregate, "session_names", store.session_names_map())

    rows: list[dict[str, object]] = []
    for terminal in range(lo, hi + 1):
        box = card_width(terminal)
        text = "\n".join(line.plain for line in build_report(aggregate, box))
        clipped = 0
        widest = 0
        tail = ""
        for section in ("By provider", "By session"):
            if section not in text:
                continue
            block = text.split(section, 1)[-1]
            if section == "By provider" and "By session" in block:
                block = block.split("By session", 1)[0]
            for row in (li.rstrip() for li in block.splitlines() if " tokens" in li):
                widest = max(widest, cell_len(row))
                # What the reader sees is the row painted INTO the box, so the
                # test is whether the rightmost column survives truncation.
                painted = truncate_cells(row, box)
                if not painted.endswith(" cache"):
                    clipped += 1
                    tail = tail or painted[-14:]
        rows.append(
            {
                "terminal": terminal,
                "content_box": box,
                "widest_row": widest,
                "gutter": box - widest,
                "rows_clipped": clipped,
                "sample_tail": tail,
            }
        )

    store.close()
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != before:
        raise AssertionError(f"the ledger at {ledger} was MODIFIED by this sweep")
    return rows


def main() -> None:
    ledger = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else 160

    rows = sweep(ledger, lo, hi)
    bad = [r["terminal"] for r in rows if r["rows_clipped"]]
    out.write_text(
        json.dumps(
            {
                "source": str(Path(__file__).resolve().parents[1]),
                "ledger": str(ledger),
                "band": [lo, hi],
                "widths_with_clipped_rows": bad,
                "sweep": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"{'term':>5} {'box':>4} {'widest':>7} {'gutter':>7} {'clipped':>8}  tail")
    for row in rows:
        print(
            f"{row['terminal']:>5} {row['content_box']:>4} {row['widest_row']:>7} "
            f"{row['gutter']:>7} {row['rows_clipped']:>8}  {row['sample_tail']!r}"
        )
    print()
    print("widths with clipped rows:", bad or "NONE")


if __name__ == "__main__":
    main()

"""Census a session's harness notices, per display PHASE, and optionally frame it.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/audit_notice_census.py <session-dir> [out.svg]

WHY THIS EXISTS
---------------
A display can serve a session through two phases, and a notice row can be
*carried* in one and *stored* in the other:

* the CONTEXT phase replays ``[marker, *preserved, *kept]``, where a notice is
  either a stamped render of a ``CustomMessage`` or a copy a compaction marker
  re-seated from an era before the stamp existed; and
* the AUDIT phase replays the journal itself, where a pre-stamp notice is a plain
  stored ``role="user"`` row with NO ``provider_payload`` at all.

A fix scoped to carried copies therefore looks complete on the context phase and
leaves the audit phase painting the originals — which is exactly the regression
QA found in round 2 ("the heal opened the mirror"). This script measures BOTH
phases with the tree's own fold rule, so the same command run against two
checkouts reports the delta rather than an argument.

The census uses ``is_harness_notice_row`` — the shared decision the folds use —
and the phone fold, which is a pure function of the rows; the frame (when asked
for) renders the AUDIT rows through the assembled application so the visual
delta of that rule can be looked at.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, ".")

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness.rows import (  # noqa: E402
    is_harness_notice_row,
    is_harness_notice_text,
)
from local_operator.mobile.projection import fold_messages_to_entries  # noqa: E402
from local_operator.session.transcript import Transcript, replay_entries  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    TranscriptView,
    UserBlock,
)
from tests.e2e.harness import ScriptedStream, build_session  # noqa: E402
from tests.unit.tui.test_app_pilot import _renderable_plain  # noqa: E402

#: The two phases, in the order a reader meets them.
PHASES = ("context", "audit")


def _census(rows) -> tuple[int, int, int, int]:
    """``(rows, user rows, notice rows, notice-shaped rows the fold PAINTS)``.

    The two notice figures are deliberately different measurements. ``notice_rows``
    counts what THIS tree's rule identifies, so it moves as the rule changes — the
    point of running the same command against two checkouts. ``painted`` counts
    what a human would actually SEE: user rows the fold emits whose text opens with
    a notice head. That is the symptom, and it must be zero however the rule is
    phrased.
    """
    users = [row for row in rows if getattr(row, "role", None) == "user"]
    notices = [row for row in users if is_harness_notice_row(row)]
    painted = [
        entry
        for entry in fold_messages_to_entries(list(rows))
        if entry.kind == "user" and is_harness_notice_text(entry.text or "")
    ]
    return len(rows), len(users), len(notices), len(painted)


def report(directory: Path) -> dict[str, tuple[int, int, int, int]]:
    transcript = Transcript(directory)
    entries = transcript.entries()
    out: dict[str, tuple[int, int, int, int]] = {}
    for phase in PHASES:
        rows = replay_entries(entries, None, mode=phase)
        out[phase] = _census(rows)
        total, users, notices, painted = out[phase]
        print(
            f"{phase:8}: rows={total:6} user={users:4} notice_rows={notices:3} "
            f"painted_by_the_fold={painted}"
        )
    return out


async def frame(
    directory: Path, out_path: str, phase: Literal["context", "audit"] = "audit"
) -> None:
    """Render the phase's rows through the assembled app, and count the painted
    notice rows ON SCREEN — the same rule, looked at rather than asserted.

    A WINDOW around the notices, not the whole phase: the fold bounds what it
    mounts to the newest rows, and the audit phase holds 15k of them, so handing
    it the whole list would paint the tail and leave the subject off screen. The
    slice is the same one in both checkouts (the notices and their neighbours),
    which is what makes the before/after pair comparable.
    """
    transcript = Transcript(directory)
    rows = replay_entries(transcript.entries(), None, mode=phase)
    # The slice is chosen by TEXT, not by the tree's own rule, so both checkouts
    # frame the SAME rows: a rule-based slice would frame the stamped rows on one
    # side and the plain stored ones on the other, and the pair would differ by
    # its subject rather than by the rule.
    hits = [
        index
        for index, row in enumerate(rows)
        if getattr(row, "role", None) == "user"
        and is_harness_notice_text(str(getattr(row, "text", "") or ""))
    ]
    if hits:
        start = max(0, hits[0] - 3)
        # A generous tail after the first one: enough following rows that the
        # frame shows the conversation continuing, not a bare notice list.
        end = min(len(rows), hits[0] + 16)
        rows = rows[start:end]
    session = build_session(
        Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "sessions" / "shot", ScriptedStream([])
    )

    async def factory():
        return session

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 44)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
        for _ in range(10):
            await pilot.pause()
        app._project_settled_rows(list(rows))
        for _ in range(40):
            await pilot.pause()
        view = app.query_one(TranscriptView)

        def _painted_notice(text: str) -> bool:
            # The user gutter (``▌``) rides every wrapped line of a user block,
            # so it is removed before the head test — the head is the message's,
            # not the surface's decoration.
            return is_harness_notice_text(" ".join(text.replace("▌", " ").split()))

        painted = [
            _renderable_plain(getattr(block, "renderable", ""))
            for block in view.blocks()
            if isinstance(block, UserBlock)
            # Text, not provenance: this count is "what a human sees", so it must
            # not move when the rule is rephrased — only when the pixels do.
            and _painted_notice(_renderable_plain(getattr(block, "renderable", "")))
        ]
        print(f"on screen ({phase}): user rows that are notices = {len(painted)}")
        save_capture(app, out_path)
    await session.dispose()


def main() -> None:
    source = Path(sys.argv[1])
    if source.resolve() == Path.home() / ".local-operator":
        raise SystemExit("refusing to read the live config dir; pass a copy")
    out = sys.argv[2] if len(sys.argv) > 2 else None
    report(source)
    if out:
        asyncio.run(frame(source, out))


if __name__ == "__main__":
    main()

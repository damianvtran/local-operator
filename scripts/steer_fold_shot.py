"""Capture the subagent view over a transcript that exercises the steer folds.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/steer_fold_shot.py OUT.svg [COLSxROWS]

Seeds the two states the legacy correlation arm gets wrong, so a rendered
frame answers "how many redirections does the reader actually see, and are
they labelled as what the parent did?":

  * TWO identical legacy steers behind ONE communication fact (the second
    fact sits on an unloaded page). Membership-only dedup collapsed these to a
    single row; the multiset renders both.
  * A NOTE envelope with no fact in the window, which the hardcoded label
    reported as a redirection the parent never made.
  * A MIXED-VINTAGE pair: one steer carrying its fact's id (post-upgrade) and
    an identical one with a legacy id whose own fact is off-page. The id arm
    used to suppress its row without spending the fact, leaving a count the
    legacy row was then wrongly consumed by — two redirections, one row.
    Seeded LEGACY-FIRST, which is the order that occurs on disk: the legacy
    row predates the id fix, so it is the older of the two. Resolving id
    matches during the ordered walk rather than in the pre-pass renders this
    order as a single row while the opposite order looks correct, so the frame
    has to seed the real one to show anything.

Both must render the parent's words and never the ``<parent-message>`` XML.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.harness.comms import (  # noqa: E402
    HUB_COMMUNICATION_CUSTOM_TYPE,
    SubagentComms,
)
from local_operator.harness.types import Message  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.subagent_view import SubagentView  # noqa: E402
from tests.unit.tui.test_band_panels import (  # noqa: E402
    FakeSession,
    _async_factory,
    _fake_jobs,
)
from tests.unit.tui.test_subagent_view import _job_with  # noqa: E402


async def seed(directory: Path) -> Transcript:
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("Fix the flaky retry test"))
    # One fact for two identical steers: the page-boundary shape the fallback
    # arm exists for.
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Focus on retries",
            "kind": "steer",
            "communication_id": "s-unmatched",
        },
    )
    for index in range(2):
        await transcript.append_message(
            Message.user(
                SubagentComms._format_to_child("Focus on retries", expects_reply=False, steer=True),
                id=f"legacy-{index}",
            )
        )
    # Mixed vintage: an id-correlated steer plus an identical legacy one. A
    # distinct body keeps this shape's fact budget independent of the pair
    # above, so the frame shows the two defects separately rather than as one
    # aggregate count.
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Cover the timeout path",
            "kind": "steer",
            "communication_id": "m1",
        },
    )
    for message_id in ("legacy-timeout", "m1"):
        await transcript.append_message(
            Message.user(
                SubagentComms._format_to_child(
                    "Cover the timeout path", expects_reply=False, steer=True
                ),
                id=message_id,
            )
        )
    # A note envelope with no fact at all: the label must follow its kind.
    await transcript.append_message(
        Message.user(
            SubagentComms._format_to_child(
                "The staging box is down until 14:00", expects_reply=False, steer=False
            ),
            id="lonely-note",
        )
    )
    return transcript


async def capture(output: Path, size: tuple[int, int], workdir: Path) -> None:
    transcript = await seed(workdir / "child")
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view(job.id)
        view = app.query_one(SubagentView)
        for _ in range(80):
            await pilot.pause()
            if view._history_entries:
                break
        for _ in range(4):
            await pilot.pause()
        app.save_screenshot(str(output))
        rows = view.rendered_rows()
        folded = [(entry.kind, entry.text) for entry in view._history_entries]
        print(f"size={size[0]}x{size[1]} entries={len(folded)}")
        for kind, text in folded:
            print(f"  {kind:16} {text!r}")
        print(f"  XML LEAKED: {'<parent-message>' in ' '.join(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("size", nargs="?", default="100x30")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/steer-fold-shot"))
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.lower().split("x", 1))
    args.workdir.mkdir(parents=True, exist_ok=True)
    asyncio.run(capture(args.output, (width, height), args.workdir))


if __name__ == "__main__":
    main()

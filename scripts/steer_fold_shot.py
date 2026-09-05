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
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness.comms import (  # noqa: E402
    HUB_COMMUNICATION_CUSTOM_TYPE,
    SubagentComms,
)
from local_operator.harness.types import Message  # noqa: E402
from local_operator.session.transcript import (  # noqa: E402
    TRANSCRIPT_FILENAME,
    Transcript,
)
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


# The one subdirectory of the workdir this script writes into. Everything the
# capture needs lives under it, so it is also the only thing the script is
# entitled to delete.
SEED_DIRNAME = "child"

# Filenames a seeded transcript directory legitimately contains: the rows
# themselves plus the temporary file ``Transcript.compact_file`` writes beside
# them. A seed target holding anything else was not produced by this script,
# which is the signal that the operator pointed --workdir at real work.
SEEDED_ARTIFACTS = frozenset({TRANSCRIPT_FILENAME, TRANSCRIPT_FILENAME + ".compact"})


class WorkdirRefused(Exception):
    """Raised when clearing the seed target would destroy data we did not write."""


def prepare_seed_dir(workdir: Path, *, force: bool = False) -> Path:
    """Return the cleared directory to seed into, refusing to destroy foreign data.

    This is a dev tool run with the operator's full permissions, so its blast
    radius is whatever path they typed: a mistyped or tab-completed ``--workdir
    .`` used to recursively delete the whole directory before seeding. Two rules
    keep that from costing anyone their work:

    * Only ``<workdir>/child`` -- the subdirectory this script creates -- is ever
      removed. The workdir itself and any sibling of ``child`` are left alone,
      so pointing at a populated directory can no longer eat its contents.
    * That subdirectory is cleared only when it looks like one of ours (absent,
      empty, or holding nothing but transcript artifacts). Anything else is
      refused unless the caller passes ``--force``, because reproducible frames
      are worth a re-run and someone's data is not.

    Errors are not swallowed: a clear that fails must surface, since the capture
    would otherwise append to a stale generation and render evidence that lies.
    """
    seed_dir = workdir / SEED_DIRNAME
    if seed_dir.exists() and not force:
        if not seed_dir.is_dir():
            raise WorkdirRefused(
                f"{seed_dir} exists and is not a directory; refusing to remove it. "
                "Pass --force to override, or choose another --workdir."
            )
        foreign = sorted(
            child.name for child in seed_dir.iterdir() if child.name not in SEEDED_ARTIFACTS
        )
        if foreign:
            listed = ", ".join(foreign[:5]) + (", ..." if len(foreign) > 5 else "")
            raise WorkdirRefused(
                f"{seed_dir} holds files this script did not write ({listed}); "
                "refusing to delete them. Pass --force to override, or choose "
                "another --workdir."
            )
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    return seed_dir


async def capture(output: Path, size: tuple[int, int], seed_dir: Path) -> None:
    transcript = await seed(seed_dir)
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
        save_capture(app, str(output))
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
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Parent directory to seed into. Only its 'child' subdirectory is "
            "written to and cleared; nothing else under the path is touched. "
            "Omit to use a private temporary directory that is removed on exit."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clear the 'child' subdirectory even when it holds files this script "
            "did not write. Without it, a seed target containing foreign data is "
            "refused rather than deleted."
        ),
    )
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.lower().split("x", 1))

    # This script exists to produce trustworthy visual evidence, so each
    # invocation MUST seed a transcript that contains only what `seed` writes.
    # An earlier version reused one fixed workdir and appended to whatever was
    # already there: a second run rendered two stacked generations of the same
    # conversation, and the fold's counts across the generation boundary were
    # wrong. That silently corrupted evidence twice -- it hid a real fold defect
    # in one capture and invented a phantom 18-entry frame in another. A shot
    # tool that lies is worse than no shot tool, so isolation is not optional:
    # a temp dir per run by default, and an explicit --workdir has its seed
    # subdirectory cleared rather than appended to, which keeps repeated runs
    # byte-for-byte reproducible. See prepare_seed_dir for why that clear is
    # scoped to what this script creates instead of the whole workdir.
    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="steer-fold-shot-"))
        cleanup = True
    else:
        workdir = args.workdir
        cleanup = False

    try:
        seed_dir = prepare_seed_dir(workdir, force=args.force)
    except WorkdirRefused as exc:
        parser.error(str(exc))

    try:
        asyncio.run(capture(args.output, (width, height), seed_dir))
    finally:
        if cleanup:
            # Our own mkdtemp directory, so a failure here is a real leak rather
            # than a permissions question -- report it instead of discarding it,
            # but do not fail a capture that already succeeded.
            try:
                shutil.rmtree(workdir)
            except OSError as exc:
                print(f"warning: could not remove {workdir}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

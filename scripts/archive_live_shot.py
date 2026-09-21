"""Capture the session sidebar over a store with an ARCHIVED session that is LIVE.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/archive_live_shot.py OUT.svg [COLSxROWS]

**The state QA round 1's Q1 was about, and the one a still frame is the only
honest evidence for.** A session directory the SCAN cannot rank — no transcript,
no inbox, i.e. a conversation whose owner is running but which has not been
written to yet — is re-added to the catalogue from the runtime REGISTRY by
``catalog.decorate_rows(include_live=True)``, and that walk did not know about
archives. So a conversation a user had just archived from inside it was offered
by both listing surfaces while reporting ``archived: false``.

The fixture is four real directory states, built through the real writers:

* ``control`` — an ordinary conversation with a transcript: LISTED.
* ``archived`` — archived, with a transcript: LISTED by the pre-feature tree,
  HIDDEN by the archive predicate itself (the feature's half).
* ``archived-live`` — archived AND held by a live discovery record published
  here with this process's pid, and transcript-less so only the registry path can
  carry it: HIDDEN only when ``decorate_rows`` asks the archive index. This is
  the row Q1 was about.
* ``live`` — live and transcript-less but NOT archived: LISTED, which is the
  control proving the live path still offers rows.

**Before/after**: run it once per tree. On a tree at ``60d09455`` (the feature
without the Q1 fix) ``archived-live`` is drawn; on the fixed tree it is not, and
nothing else about the frame moves. On the pre-feature base tree nothing is
archived at all, so all four are drawn — which is what makes the script run
there too (``set_archived`` is imported tolerantly).

The frame is deterministic for a given tree: the catalog is the store's own and
no clock is printed except a relative age, pinned by writing fixed mtimes.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.resume import write_session_title  # noqa: E402
from local_operator.session.cleanup import mark_store  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

try:  # the pre-feature tree draws the same store and has nothing to hide
    from local_operator.session.archived import set_archived  # noqa: E402
except ImportError:  # pragma: no cover - only reached in a before-frame worktree
    set_archived = None  # type: ignore[assignment]

#: ``(id, title, archived, live)``.
ROWS = [
    ("aaaa00000001", "Retention sweep for the analytics ledger", False, False),
    ("bbbb00000002", "Q3 pricing model review", True, False),
    ("cccc00000003", "Parser crash on nested frontmatter", True, True),
    ("dddd00000004", "Nightly catalogue refresh", False, True),
]

#: A fixed instant for every seeded mtime, so the ages on screen are a property
#: of the fixture rather than of when the capture ran.
STAMP = 1_700_000_000.0


def _seed(root: Path) -> list[subprocess.Popen[bytes]]:
    """Lay the four rows down and return the helper processes to reap.

    ONE LIVE RECORD PER PID, because the registry is keyed by pid (``publish``
    writes ``<pid>.json``): two records from one process overwrite each other and
    the second live row silently vanishes, which is a fixture that would have
    "proved" the fix by never rendering the defect.

    **EVERY LIVE ROW GETS ITS OWN HELPER, never this process's pid** (review
    round 2, MINOR-1). The first version used ``os.getpid()`` for one row, and
    that is the pid the app it drives publishes its OWN discovery record under
    during boot (``RecordPublisher``), so the row the frame exists to show was
    overwritten and a re-run of this script reproduced the AFTER list on a
    pre-fix tree — an evidence artifact that cannot reproduce what the PR quotes,
    which is the same defect class as the empty-commit probe. Helper pids are
    alive for the whole capture (which is what ``registry.classify`` reads as
    ``live``) and nothing else in the run publishes under them; a synthetic pid
    would classify ``stale``, and ``decorate_rows`` drops those as "no record".
    Both helpers are reaped by :func:`main`.
    """
    helpers: list[subprocess.Popen[bytes]] = []
    mark_store(root / "sessions")
    from local_operator.session.runtime.registry import publish
    from local_operator.session.runtime.types import SessionRecord

    live_pids: list[int] = []
    for _ in range(sum(1 for row in ROWS if row[3])):
        helpers.append(
            subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        )
        live_pids.append(helpers[-1].pid)
    live_index = 0

    for session_id, title, archived, live in ROWS:
        directory = root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "created_at.json").write_text(str(int(STAMP)), encoding="utf-8")
        if not live:
            (directory / "transcript.jsonl").write_text(
                json.dumps(
                    {
                        "type": "message",
                        "payload": {"role": "user", "content": f"{title}. Let's pick this up."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_session_title(directory, title, user_set=True, past_names=[])
        if archived and set_archived is not None:
            set_archived(root, session_id, True)
        if live:
            # A real discovery record: ``publish`` stamps the heartbeat and the
            # pid is a live process, so ``registry.classify`` reads it as
            # ``live`` — nothing here is a hand-set field.
            record = publish(
                SessionRecord(
                    pid=live_pids[live_index],
                    kind="tui",
                    session_id=session_id,
                    conversation_name=title,
                    cwd=str(root),
                    model_label="test/model",
                    control_port=0,
                    control_key="synthetic",
                    started_at=STAMP,
                ),
                root,
            )
            print(f"  record {session_id} -> {record.name} (pid {live_pids[live_index]})")
            live_index += 1
    return helpers


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2 and "x" in sys.argv[2]:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    helpers = _seed(Path(config_dir()))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # ctrl+b opens the sidebar; the app then polls the REAL catalogue into
        # it, so the rows in the frame are the ones `load_catalog` returns.
        await pilot.press("ctrl+b")
        for _ in range(200):
            await pilot.pause()
            if not getattr(app, "_sidebar_refresh_pending", False):
                break
        await pilot.pause()
        sidebar = app._session_sidebar
        print("sidebar ids:", [entry.id for entry in sidebar.entries])
        save_capture(app, out)

    for helper in helpers:
        helper.terminate()
        helper.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())

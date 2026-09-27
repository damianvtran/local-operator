"""Serve the REAL phone bundle over a seeded project store, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_projects_fixture.py PORT PASSWORD
The CAPTURE script passes the password, so this file holds no literal.

ONE STORE, SIX PROJECTS, and every row answers a different question the
Projects sheet has to render:

* ``payments-migration`` — the rich row: a description, FRESH progress, three
  milestones across all three derived states (one completed, one overdue, one
  upcoming), and two linked sessions of which one is LIVE (a real runtime record
  keyed to this process's pid, so the scan classifies it honestly rather than
  reading a state name we wrote down) and one is a durable conversation with a
  title.
* ``projects-feature`` — STALE progress (backdated, so the stale ink is real),
  a done/upcoming milestone pair, a two-session link set one of which has no
  title (the row must fall back to the id, never to blank).
* ``travel-planning`` — paused, and its one linked session is MISSING (an id
  with no directory — the state the store marks and never auto-removes).
* ``site-refresh`` — done, with every milestone completed.
* ``old-notes`` — archived, the section the board draws only when non-empty.
* ``docs-sweep`` — active with nothing set at all: no progress, no milestones,
  no sessions, no description (the row that proves "no progress"/"no sessions"
  are stated rather than left blank).

The rows are written through the REAL store (``ProjectRegistry`` — the same
validators the tool uses), and the two backdated rows are backdated by editing
the stored JSON afterwards; that is the only hand-touch, and it exists because
``progress_updated_at`` is stamped ``now`` by the store by design. Linked
session directories carry a ``transcript.jsonl`` (what makes a directory a
conversation) and a title written by the real ``write_session_title`` sidecar
writer, so the detail sheet renders the same rows the product would.

No runtime scanner and no registrant sockets (``dial_registrants=False``), and
``scripts.probe_isolation`` re-homes HOME and the config dir before any
``local_operator`` import, so this never touches the operator's live daemon,
sessions or store.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.projects import (
    EstimateUnit,
    MilestoneEdit,
    ProjectEdit,
    ProjectRegistry,
    ProjectStatus,
)
from local_operator.resume import write_session_title

#: The variable name a caller may use instead of the second argument. A NAME,
#: never a value, and written out in both this file and the capture module
#: deliberately: they are siblings, and importing one from the other would drag
#: Chrome/CDP code into a fixture that only serves a daemon.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"

LIVE_SESSION = "aa11bb22cc33"
STOPPED_SESSION = "dd44ee55ff66"
MISSING_SESSION = "deadbeef0000"
REVIEW_SESSION = "112233445566"
TITLED_ONLY_SESSION = "778899aabbcc"


def required_password(argv_rest: list[str]) -> str:
    value = argv_rest[0] if argv_rest else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. It is never defaulted and never printed."
        )
    return value


def _conversation(session_id: str, title: str | None) -> None:
    """One durable session directory: a transcript (the listing's admission) and,
    when given, a title via the real sidecar writer."""
    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    if title:
        write_session_title(directory, title, user_set=True, past_names=[])


def _live_record(session_id: str) -> None:
    """A runtime record whose pid is THIS process, so ``scan_runtime_states``
    classifies the session ``live`` for cause (the pid really is alive) rather
    than because a fixture wrote the word ``live``."""
    run_dir = config_dir() / "run" / "mobile"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "kind": "tui",
                "session_id": session_id,
                "conversation_name": "Payments cutover — ledger",
                "cwd": "/synthetic/worktree",
                "model_label": "fixture/model",
                "control_port": 1,
                "control_key": "f" * 16,
                "heartbeat_at": time.time(),
                "busy": True,
            }
        ),
        encoding="utf-8",
    )


def _backdate_progress(project_id: str, seconds: float) -> None:
    """Age one row's progress stamps so its stale ink is the server's real
    verdict over an old stamp, not a flag a fixture set."""
    path = config_dir() / "projects" / f"{project_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["progress_updated_at"] = time.time() - seconds
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _create(
    registry: ProjectRegistry,
    ids: dict[str, str],
    name: str,
    *,
    description: str | None = None,
    status: ProjectStatus | None = None,
    tags: list[str] | None = None,
    estimate: float | None = None,
    estimate_unit: EstimateUnit | None = None,
    start_date: str | None = None,
    target_date: str | None = None,
) -> str:
    """Create one seeded row, typed explicitly rather than through ``**fields``.

    The scalar parameters are not ceremony: ``**fields: object`` loses the
    ``ProjectEdit`` field types, and pyright (rightly) refuses an ``object``
    where a ``str | None`` is required — the same check CI runs, which caught
    this on the first push.
    """
    project = registry.create_project(
        ProjectEdit(
            name=name,
            description=description,
            status=status,
            tags=tags,
            estimate=estimate,
            estimate_unit=estimate_unit,
            start_date=start_date,
            target_date=target_date,
        )
    )
    ids[name] = project.id
    return project.id


def seed() -> dict[str, str]:
    """Write the six rows through the store; return ``{name: id}``."""
    registry = ProjectRegistry(config_dir())
    ids: dict[str, str] = {}

    _create(registry, ids, "docs-sweep")

    payments = _create(
        registry,
        ids,
        "payments-migration",
        description="Move billing onto the new ledger, then cut the switch-over.",
        tags=["q4", "payments"],
        status="active",
        estimate=13,
        estimate_unit="points",
        start_date="2026-09-15",
        target_date="2026-10-20",
    )
    registry.update_project(
        payments,
        ProjectEdit(progress="ledger cutover written; dry run green against staging"),
        reporter="operator",
    )
    registry.set_milestone(payments, MilestoneEdit(name="beta cut", completed=True))
    registry.set_milestone(
        payments, MilestoneEdit(name="audit", target_date="2026-09-05", completed=False)
    )
    registry.set_milestone(payments, MilestoneEdit(name="cutover window", target_date="2026-10-15"))
    registry.link_session(payments, LIVE_SESSION)
    registry.link_session(payments, STOPPED_SESSION)
    _conversation(LIVE_SESSION, "Payments cutover — ledger")
    _conversation(STOPPED_SESSION, "Staging dry run fixes")
    _live_record(LIVE_SESSION)

    feature = _create(
        registry,
        ids,
        "projects-feature",
        description="Projects primitive: store, tool, and the three view surfaces.",
        tags=["q4"],
        status="active",
        target_date="2026-10-01",
    )
    registry.update_project(
        feature,
        ProjectEdit(progress="slice 5: mobile daemon routes + the phone sheet"),
        reporter="operator",
    )
    registry.set_milestone(feature, MilestoneEdit(name="store + tool", completed=True))
    registry.set_milestone(feature, MilestoneEdit(name="views", target_date="2026-10-01"))
    registry.set_milestone(feature, MilestoneEdit(name="seed sync"))
    registry.link_session(feature, REVIEW_SESSION)
    registry.link_session(feature, TITLED_ONLY_SESSION)
    _conversation(REVIEW_SESSION, "Projects design review")
    _conversation(TITLED_ONLY_SESSION, None)
    _backdate_progress(feature, 2.2 * 86400)

    travel = _create(
        registry,
        ids,
        "travel-planning",
        description="Reykjavik in November — flights, car, northern lights.",
        status="paused",
    )
    registry.update_project(travel, ProjectEdit(progress="flights booked"), reporter="operator")
    registry.link_session(travel, MISSING_SESSION)
    _backdate_progress(travel, 5.4 * 86400)

    done = _create(
        registry,
        ids,
        "site-refresh",
        description="Marketing site: new hero, faster build.",
        status="done",
    )
    registry.update_project(done, ProjectEdit(progress="shipped"), reporter="operator")
    registry.set_milestone(done, MilestoneEdit(name="design", completed=True))
    registry.set_milestone(done, MilestoneEdit(name="launch", completed=True))

    _create(
        registry,
        ids,
        "old-notes",
        description="Scratch notes from the old setup.",
        status="archived",
    )
    return ids


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4288
    password = required_password(sys.argv[2:])
    seed()
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    app = build_app(daemon)
    print(f"Fixture projects store on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())

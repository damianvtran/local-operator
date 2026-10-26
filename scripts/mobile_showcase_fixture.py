"""Serve the REAL phone bundle over a REALISTIC populated relay, for before/after capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_showcase_fixture.py PORT PASSWORD
The runner supplies a per-run test password; the fixture stores no credential.
Use the harness browser at /showcase and its explicit viewport/theme/Audit
controls. See docs/design/mobile-relay-redesign.md §13 for the evidence contract.

WHY THIS FIXTURE EXISTS. Every sibling fixture isolates ONE defect: the overflow
rig photographs a roster that overruns, the delegating rig six rows that differ
only in their subagent chip, the glance rig the spend row. None of them answers
the question a REDESIGN has to answer before and after — *what does the phone
look like on an ordinary busy day?* — because none puts a realistic mix on one
list and a realistic conversation behind one row. The relay redesign
(``docs/design/mobile-relay-redesign.md``) is judged on exactly that frame, so
this rig composes the states the list and the session view actually carry:

LIST (``/api/sessions``), in the sections the daemon's own ranker produces:

* two PINNED rows (one streaming), through the real pin store;
* ACTIVE rows: a streaming session with running subagents, one waiting on an
  APPROVAL, one with open ASKS, an unread completion, a long name that must
  truncate at 360 px, a degraded ("not answering") row and a plain idle one;
* PREVIOUS rows: durable conversations with titles, one unread;
* TAGS: three categories through the real tag store, assigned with primaries
  so the tag line and the "group by tags" toggle have something to group.

SESSION VIEW (tap ``Relay redesign review``): a populated transcript with tool
rows (done / failed / running, a diff), markdown, a todos panel, a subagent
roster with one running child, the spend/context glance, and an open ask, so
one frame carries every panel the redesign has to re-home.

Two more session views are reachable from the list for the state matrix:
``Approve deploy script`` (a pending APPROVAL card) and ``Ended overnight
batch`` (the ENDED strip with resume).

ISOLATION. ``scripts.probe_isolation`` re-homes HOME and the config dir before
any ``local_operator`` import; no runtime scanner and no registrant sockets
(``dial_registrants=False``), so this never touches the operator's live daemon
(:4098), their tunnel or their sessions. Every id, path and name is synthetic.

WHAT IS SEEDED THROUGH REAL WRITERS, and why it matters: pins go through
``sidebar_pins.set_pin`` (the reader prunes ids without a directory), tags
through ``session_tags.write_categories``/``set_entity_tags``, unread receipts
through ``AttentionStore.publish``, titles through ``write_session_title``, and
asks through ``asks.store.write_entry`` — so the list paints what the product
would paint for the same store, not a hand-written JSON payload. Only the live
projections are injected, which is what every sibling fixture does too.

THE ``/api/usage`` AND ``/api/models`` READERS ARE SUBSTITUTED, and that is the
one place this rig does not go through the product's own store: in an isolated
sandbox the provider store is empty, so the real readers answer "no models" and
"no cached usage" — honest, but a redesign judged on those two sheets needs
their POPULATED shape. ``_list_models`` serves the committed ranked fixture the
model-sheet tests already use (``src/fixtures/models.ranked.json``) and
``_cached_phone_usage`` a two-provider snapshot in the wire shape ``api.ts``
types; both are module-level reads the route calls, so the route, its gate and
its serialisation are still the real ones. ``LOP_SHOWCASE_EMPTY_SHEETS=1``
leaves the real readers in place to photograph the empty states instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile import daemon as daemon_module
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    PendingAskWire,
    PendingRequest,
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)

# Cookies are scoped by hostname, NOT port. Parallel synthetic fixtures on
# localhost must not overwrite each other's valid signed cookie and manufacture
# a 401. Only this isolated process changes namespace; signing and auth gates
# remain the real production implementations, including absent/invalid 401s.
#
# THE NAME IS PER-PORT (fixture-only, no product code reads it). Two independent
# lanes serving localhost at the same time otherwise share one cookie name, and
# whichever signs last replaces the other's still-valid cookie — a 401 the code
# under test never caused. `LOP_MOBILE_FIXTURE_COOKIE` names it explicitly; the
# default derives from the PORT this process binds, so two fixtures on different
# ports cannot collide. The derivation lives in `fixture_cookie_name` and is
# applied in `main()` where the port is known: an IMPORT must not claim a shared
# name, and the previous constant here did exactly that (code review round 2,
# C2-2: ports 41001 and 41002 both resolved to one fixed name).
if os.environ.get("LOP_MOBILE_FIXTURE_COOKIE"):
    daemon_module.COOKIE_NAME = os.environ["LOP_MOBILE_FIXTURE_COOKIE"]


def fixture_cookie_name(port: int) -> str:
    """The signing-cookie name this fixture process will use on ``port``.

    Explicit `LOP_MOBILE_FIXTURE_COOKIE` wins; otherwise the name carries the
    port, so two lanes on localhost never share a namespace. Callers (and tests)
    ask this function instead of reading a module constant, which is how the
    per-port claim is checked against two ports rather than asserted in prose.
    """
    return os.environ.get("LOP_MOBILE_FIXTURE_COOKIE") or f"lop_probe_fixture_{port}"


#: The variable NAME (never a value) a caller may use instead of the second
#: argument — the same contract every sibling fixture documents.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"

#: A synthetic project root. Never a real path on the operator's machine.
CWD = "/synthetic/home/projects/local-operator"


#: A deliberately LONG cwd (operator finding B): the composer's workspace control
#: and the directory sheet must truncate or wrap it, never widen the phone.
LONG_CWD = (
    "/synthetic/home/projects/clients/northwind-traders/"
    "quarterly-reconciliation-2026/vendor-ledger-exports/raw"
)

#: ``1`` serves an EMPTY store (the list's empty state); the capture sets it for
#: one leg of its run.
EMPTY_ENV = "LOP_SHOWCASE_EMPTY"

#: Session ids are 12-hex because the tag store (``session_tags._entity_path``)
#: and the pin reader enforce the desktop id contract; the ``5c`` prefix marks
#: them as this rig's synthetic ids.


def required_password(argv_rest: list[str]) -> str:
    """The per-run password, or a refusal that names how to supply one (no default)."""
    value = argv_rest[0] if argv_rest else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. It is never defaulted and never printed."
        )
    return value


# ---------------------------------------------------------------------------
# The showcase conversation — the frame the session view is judged on.
# ---------------------------------------------------------------------------


def _tool(
    entry_id: str,
    name: str,
    summary: str,
    state: str = "done",
    *,
    added: int = 0,
    removed: int = 0,
    elapsed: float = 1.2,
    error: str = "",
) -> TranscriptEntry:
    return TranscriptEntry(
        id=entry_id,
        kind="tool",
        tool_call_id=f"call-{entry_id}",
        tool_name=name,
        tool_state=state,  # type: ignore[arg-type]
        summary=summary,
        diff_added=added,
        diff_removed=removed,
        elapsed_s=elapsed,
        error=error,
    )


def _showcase_transcript() -> list[TranscriptEntry]:
    return [
        TranscriptEntry(
            id="sc-1",
            kind="user",
            text="The phone relay feels crowded. Audit the session list and the "
            "session view against the desktop's borderless rules and tell me "
            "what to move.",
        ),
        TranscriptEntry(
            id="sc-2",
            kind="assistant",
            text="Starting with the list. I'll measure every bordered control, "
            "then compare the chrome height against the content area.",
        ),
        _tool("sc-3", "read", "local_operator/mobile/web/src/screens/session-list.tsx"),
        _tool("sc-4", "grep", "border-control  src/  (74 matches)", elapsed=0.4),
        _tool(
            "sc-5",
            "edit",
            "docs/design/mobile-relay-redesign.md",
            added=142,
            removed=6,
            elapsed=0.8,
        ),
        _tool(
            "sc-6",
            "bash",
            "pnpm --config.verify-deps-before-run=false run build",
            "failed",
            elapsed=14.3,
            error="exit 1: Type error in src/components/ui/sheet.tsx",
        ),
        TranscriptEntry(
            id="sc-7",
            kind="assistant",
            text=(
                "**Findings so far**\n\n"
                "- Every control at rest is a 1 px `border-control` box — five in "
                "the footer alone.\n"
                "- Rows spend two lines on a mono path and a model label most "
                "readers never scan.\n"
                "- The session header stacks a status row and three strips.\n\n"
                "I'm fanning the per-screen measurements out to two reviewers."
            ),
        ),
        _tool("sc-8", "task", "designer: measure sheets at 360x780", "running", elapsed=48.0),
        # WORST-CASE CONTENT for the no-sideways-scroll rule (operator finding B):
        # a 200+ character code line with no break opportunity, and a table with
        # six columns. Both used to become horizontal rails inside the message.
        TranscriptEntry(
            id="sc-8b",
            kind="assistant",
            text=(
                "The failing selector, verbatim:\n\n```\n"
                + "document.querySelector('"
                + "main>section>div.lo-row" * 8
                + "').getBoundingClientRect()\n```\n\n"
                "| screen | width | chrome | content | bordered | targets |\n"
                "|---|---|---|---|---|---|\n"
                "| conversation list | 320 | 96 px | 472 px | 3 spinner rings | 0 under 44 px |\n"
                "| conversation view | 390 | 56 px | 680 px | 0 | 0 under 44 px |\n"
            ),
        ),
        # THE 13-vs-14 px BAND TABLE (code review round 3, C3-1): two long single
        # words whose min-content total sits BETWEEN the total measured at the cell
        # font and the total at the wrapper's inherited font, so a verdict taken
        # with the wrong font lands on the other side of the line. The showcase's
        # "Check table band" control reports the totals at both sizes and the form
        # the component chose, so the fix is checkable rather than asserted.
        TranscriptEntry(
            id="sc-8d",
            kind="assistant",
            text=(
                "Band check:\n\n"
                "| counterreconciliation | acknowledgementledger |\n|---|---|\n"
                "| settled | confirmed |\n"
            ),
        ),
        # A TWO-COLUMN table in its OWN message: the stacked fallback has to be
        # checked against a table that FITS and therefore must stay a real table
        # (design review round 2, D4). Two tables in one message merge into one
        # block in this renderer, which would make the fitting case unmeasurable.
        TranscriptEntry(
            id="sc-8c",
            kind="assistant",
            text=(
                "Which lane owns what:\n\n"
                "| stage | owner |\n|---|---|\n"
                "| audit | designer |\n| implementation | coder |\n"
            ),
        ),
        TranscriptEntry(
            id="sc-9",
            kind="user",
            text="Keep every feature reachable — drawers and sheets are fine.",
        ),
    ]


def _showcase_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="5c00a941bb95",
        pid=930001,
        kind="tui",
        conversation_name="Relay redesign review",
        cwd=LONG_CWD,
        model_label="Claude Opus 5.5",
        model_selector="anthropic/claude-opus-5-5",
        effort="high",
        effort_ladder=["low", "medium", "high", "max"],
        streaming=True,
        activity="measuring the session column",
        activity_started_s=94,
        transcript=_showcase_transcript(),
        version=9,
    )
    projection.todos = [
        TodoPhase(
            name="Audit",
            items=[
                TodoItem(text="Capture BEFORE frames at 390 and 360", status="done"),
                TodoItem(text="Measure bordered elements per screen", status="done"),
                TodoItem(text="Measure chrome vs content height", status="pending"),
                TodoItem(text="Contrast-check the new ground pairs", status="pending"),
            ],
        ),
        TodoPhase(
            name="Direction",
            items=[
                TodoItem(text="Write the reachability table", status="pending"),
                TodoItem(text="Render a static mock", status="pending"),
            ],
        ),
    ]
    projection.subagents = [
        SubagentRow(
            job_id="sc-job-1",
            label="measure-sheets",
            agent="designer",
            status="running",
            progress="photographing the projects sheet",
            elapsed_s=48,
            session_id="5c0cb3e8fce7",
            activity="photographing the projects sheet",
        ),
        SubagentRow(
            job_id="sc-job-2",
            label="token-audit",
            agent="reviewer",
            status="completed",
            elapsed_s=31,
            session_id="5c0d07484015",
            result_text="74 border-control uses across 21 files.",
        ),
        SubagentRow(
            job_id="sc-job-3",
            label="contrast-pairs",
            agent="reviewer",
            status="failed",
            elapsed_s=12,
            session_id="5c0e78d57a49",
            error_text="palette file not found",
        ),
    ]
    # The spend/context glance the status row reads (same fields the glance
    # fixture sets), so the frame carries the row a redesign must keep.
    projection.cumulative_parent_cost = 2.47
    projection.child_costs = {}
    projection.cost_knowledge = "exact"
    projection.context_tokens = 61_800
    projection.context_window = 200_000
    projection.context_is_estimate = False
    projection.usage = {"input_tokens": 61_800, "output_tokens": 4_120}
    now = int(time.time() * 1000)
    projection.asks = [
        PendingAskWire(
            ask_id="sc-ask-1",
            created_at=now - 90_000,
            expires_at=now + 810_000,
            timeout_s=900,
            status="open",
            questions=[
                {
                    "id": "q1",
                    "question": "Should the bottom bar keep 'new session' as a "
                    "labelled button, or become a floating action button?",
                    "options": [
                        {"label": "labelled button", "description": "clearest, costs a row"},
                        {"label": "floating button", "description": "thumb zone, no row"},
                    ],
                    "multi": False,
                    "recommended": 1,
                    "secret": False,
                    "persist": False,
                }
            ],
        )
    ]
    projection.asks_open = 1
    return projection


def _approval_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="5c0159494695",
        pid=930002,
        kind="tui",
        conversation_name="Approve deploy script",
        cwd="/synthetic/home/projects/site",
        model_label="GPT-5.5",
        streaming=True,
        activity="waiting for approval",
        transcript=[
            TranscriptEntry(id="ap-1", kind="user", text="Ship the docs site."),
            TranscriptEntry(
                id="ap-2", kind="assistant", text="Building, then I need to run the deploy."
            ),
            _tool("ap-3", "bash", "pnpm build", elapsed=21.4),
        ],
        version=3,
    )
    projection.pending = PendingRequest(
        request_id="req-approval",
        kind="approval",
        title="bash",
        detail="bash — ./scripts/deploy.sh --prod\n\nPublishes the built site to the "
        "production bucket and invalidates the CDN cache.",
    )
    projection.pending_count = 1
    return projection


def _ended_projection() -> SessionProjection:
    return SessionProjection(
        session_id="5c04ba92ed73",
        pid=930003,
        kind="tui",
        conversation_name="Ended overnight batch",
        cwd="/synthetic/home/projects/batch",
        model_label="Claude Sonnet 5",
        streaming=False,
        ended=True,
        transcript=[
            TranscriptEntry(id="en-1", kind="user", text="Run the overnight reconciliation."),
            TranscriptEntry(
                id="en-2", kind="assistant", text="Reconciled 4,212 rows. 3 need review."
            ),
        ],
        version=2,
    )


def _ask_only_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="5c02a2f8909c",
        pid=930004,
        kind="tui",
        conversation_name="Pick a release window",
        cwd="/synthetic/home/projects/release",
        model_label="Claude Opus 5.5",
        streaming=False,
        transcript=[
            TranscriptEntry(id="ak-1", kind="user", text="Plan the release."),
            TranscriptEntry(id="ak-2", kind="assistant", text="I need a window from you."),
        ],
        version=2,
    )
    now = int(time.time() * 1000)
    projection.asks = [
        PendingAskWire(
            ask_id=f"ak-ask-{i}",
            created_at=now - (i + 1) * 60_000,
            expires_at=now + 600_000,
            timeout_s=900,
            status="open",
            questions=[
                {
                    "id": "q",
                    "question": question,
                    "options": [
                        {"label": "today", "description": ""},
                        {"label": "tomorrow", "description": ""},
                    ],
                    "multi": False,
                    "recommended": 0,
                    "secret": False,
                    "persist": False,
                }
            ],
        )
        for i, question in enumerate(["Which release window?", "Notify the beta channel first?"])
    ]
    projection.asks_open = 2
    return projection


def _plain_projection(
    session_id: str,
    pid: int,
    name: str,
    *,
    streaming: bool = False,
    model: str = "Claude Sonnet 5",
    cwd: str = CWD,
    degraded: bool = False,
    running: int = 0,
) -> SessionProjection:
    projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=name,
        cwd=cwd,
        model_label=model,
        streaming=streaming,
        activity="working" if streaming else "",
        degraded=degraded,
        transcript=[
            TranscriptEntry(id=f"{session_id}-1", kind="user", text=f"{name}."),
            TranscriptEntry(id=f"{session_id}-2", kind="assistant", text="On it."),
        ],
        version=2,
    )
    if running:
        projection.subagents = [
            SubagentRow(job_id=f"{session_id}-j{i}", label=f"worker-{i}", status="running")
            for i in range(running)
        ]
    return projection


#: ``(projection, subagents_running, unread, durable title)``. Order is the
#: order they are seeded; the daemon's own ranker decides the painted order.
def _live_rows() -> list[tuple[SessionProjection, int, bool]]:
    return [
        (_showcase_projection(), 1, False),
        (_approval_projection(), 0, False),
        (_ask_only_projection(), 0, False),
        (
            _plain_projection(
                "5c075dc7896d",
                930010,
                "Inbox triage",
                streaming=True,
                model="Claude Haiku 5",
                cwd="/synthetic/home/workspace",
            ),
            0,
            False,
        ),
        (
            _plain_projection(
                "5c06e542556a",
                930011,
                "Reconcile the quarterly vendor ledger against the bank export and "
                "flag every mismatch over fifty dollars",
                model="GPT-5.5",
                cwd="/synthetic/home/workspace/finance/q3-reconciliation",
            ),
            0,
            False,
        ),
        (
            _plain_projection(
                "5c0fadfa5840",
                930012,
                "Draft the launch email",
                cwd="/synthetic/home/workspace/marketing",
            ),
            0,
            True,
        ),
        (
            _plain_projection(
                "5c053b92eb34",
                930013,
                "Fan out the docs sweep",
                streaming=True,
                model="Claude Opus 5.5",
                running=3,
            ),
            3,
            False,
        ),
        (
            _plain_projection(
                "5c036dc24469",
                930014,
                "Watch the nightly build",
                degraded=True,
                cwd="/synthetic/home/projects/ci",
            ),
            0,
            False,
        ),
        (_plain_projection("5c101adbcc34", 930015, "Rename photo exports"), 0, False),
    ]


#: Durable-only conversations: ``(session_id, title, unread)``.
PREVIOUS = [
    ("5c0b51d6a6f0", "Plan the Lisbon trip", False),
    ("5c0a5df0f9c6", "Sort the 2025 tax receipts", True),
    ("5c09f2d5fcb3", "Summarise the board meeting notes", False),
    ("5c08c867b3a0", "Tighten the CV for the platform role", False),
]

#: Pins, newest first. One live, one durable.
PINS = ["5c075dc7896d", "5c09f2d5fcb3"]

#: Every tag category the rig creates, in catalogue order.
TAG_CATEGORIES = [
    "Work",
    "Design",
    "Personal",
    "Finance",
    "Travel",
    "Client Northwind Traders",
    "Research",
    "Home",
]

#: ``{session_id: (tag names, primary)}``.
TAGS = {
    "5c00a941bb95": (["Work", "Design"], "Design"),
    "5c0159494695": (["Work"], "Work"),
    "5c053b92eb34": (["Work"], "Work"),
    "5c075dc7896d": (["Personal"], "Personal"),
    "5c0b51d6a6f0": (["Personal"], "Personal"),
    "5c06e542556a": (["Work"], "Work"),
    "5c0a5df0f9c6": (["Finance", "Home"], "Finance"),
    "5c09f2d5fcb3": (["Client Northwind Traders", "Research"], "Client Northwind Traders"),
    "5c08c867b3a0": (["Travel"], "Travel"),
}


def _conversation_dir(session_id: str, title: str | None = None) -> Path:
    """A durable session directory, admitted by the listing (a transcript) and
    titled through the real sidecar writer."""
    from local_operator.paths import config_dir
    from local_operator.resume import write_session_title

    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    if title:
        write_session_title(directory, title, user_set=True, past_names=[])
    return directory


def _seed_store(live: list[tuple[SessionProjection, int, bool]]) -> None:
    from local_operator.asks.store import write_entry
    from local_operator.paths import config_dir
    from local_operator.session.attention import AttentionStore
    from local_operator.session.session_tags import set_entity_tags, write_categories
    from local_operator.tui.sidebar_pins import set_pin

    root = config_dir()
    attention = AttentionStore()
    for projection, _running, unread in live:
        _conversation_dir(projection.session_id, projection.conversation_name)
        if unread:
            attention.publish(
                f"session/{projection.session_id}", str(uuid.uuid4()), "a1", "complete"
            )
        rows = [row.to_json() for row in (projection.asks or [])]
        if rows:
            write_entry(root, projection.session_id, cwd=projection.cwd, asks=rows)
    for session_id, title, unread in PREVIOUS:
        _conversation_dir(session_id, title)
        if unread:
            attention.publish(f"session/{session_id}", str(uuid.uuid4()), "a1", "complete")
    # Pins: oldest first so PINS[0] ends up newest.
    for session_id in reversed(PINS):
        set_pin(root, session_id, True)
    # EIGHT categories (operator finding B): more than any phone line holds, so
    # the filter strip has to collapse the rest behind its `+N` door.
    categories = write_categories(root, [(None, name) for name in TAG_CATEGORIES])
    by_name = {item["name"]: item["id"] for item in categories}
    for session_id, (names, primary) in TAGS.items():
        set_entity_tags(root, "session", session_id, [by_name[n] for n in names], by_name[primary])


def _substitute_readers() -> None:
    """Swap the two provider-backed readers for populated synthetic answers.

    See the module doc: the ROUTES stay real; only what they read changes."""
    import local_operator.mobile.daemon as daemon_module

    ranked = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "local_operator/mobile/web/src/fixtures/models.ranked.json"
        ).read_text(encoding="utf-8")
    )
    now_ms = int(time.time() * 1000)

    def limit(window: str, fraction: float, *, paid: bool = False) -> dict[str, object]:
        return {
            "amount": {
                "used": None,
                "limit": None,
                "remaining": round((1 - fraction) * 100, 1),
                "used_fraction": fraction,
                "unit": "percent",
            },
            "paid_fallback": paid,
            "available": True if paid else None,
            "unlimited": False,
            "supersedes_tiers": False,
            "family_cap": False,
            "shared": True,
            "window": window,
            "family": "unknown",
            "resets_at_ms": now_ms + 3 * 3600 * 1000,
            "payable": paid,
            "exhausted": False,
        }

    usage = {
        "reports": [
            {
                "provider": "anthropic",
                "limits": [limit("5h", 0.42), limit("7d", 0.18)],
                "unavailable": False,
                "fetched_at": now_ms - 90_000,
            },
            {
                "provider": "openai",
                "limits": [limit("7d", 0.71)],
                "unavailable": False,
                "fetched_at": now_ms - 90_000,
            },
        ],
        "source": "cached",
        "cached_at": now_ms - 90_000,
        "may_be_stale": False,
    }
    daemon_module._list_models = lambda: ranked  # type: ignore[assignment]
    daemon_module._cached_phone_usage = lambda: usage  # type: ignore[assignment]


def showcase_app(daemon: MobileDaemon):
    """Serve the actual bundle in a measured phone frame for the harness browser.

    This page is fixture chrome only: it never substitutes the React tree, API
    or SSE. Width/height select the iframe viewport, not a CSS-scaled imitation
    of a phone. Theme and forced RTL controls are explicit evidence setup, not
    claims of translated-copy coverage. Capture through the browser tool; the
    historical CDP capture is not part of this browser-only recipe.
    """
    app = build_app(daemon)

    async def showcase(request):
        html = Path(__file__).with_name("mobile_showcase.html").read_text()
        src = 'src="/baseline"' if request.query_params.get("before") == "1" else 'src="/"'
        return HTMLResponse(html.replace('src="/"', src))

    # Prepend before the daemon's SPA catch-all: /showcase is fixture-only and
    # must not be resolved to the product index.html by the static route.
    async def metrics_script(_request):
        # NO-STORE, because this file IS the audit: a cached copy silently measured
        # the screen with the previous rules and reported its old numbers under the
        # new run's state label (measured in review round 1's remediation).
        return Response(
            Path(__file__).with_name("mobile_showcase_metrics.js").read_text(),
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def save_metrics(request):
        data = await request.json()
        # Caller chooses only a state label, never a destination path. The
        # evidence root is process configuration, outside the repository.
        state = str(data.get("state", ""))
        if not state or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in state):
            return JSONResponse({"error": "invalid state label"}, status_code=400)
        root = os.environ.get("LOP_SHOWCASE_EVIDENCE_DIR")
        if root:
            out = Path(root)
            out.mkdir(parents=True, exist_ok=True)
            path = out / "metrics.json"
            entries = json.loads(path.read_text()) if path.exists() else {}
            # A POST is evidence, not control: a malformed one is REFUSED with a
            # 400 and logged, never raised. This rig used to die on a body without
            # `metrics` (the band check posted its own shape), which took the whole
            # fixture down mid-run and left a stale metrics.json looking fresh.
            metrics = data.get("metrics")
            if not isinstance(metrics, dict) or "viewport" not in metrics or "theme" not in metrics:
                print(
                    "showcase: refused a /showcase/metrics POST without a metrics object",
                    flush=True,
                )
                return JSONResponse({"error": "metrics object required"}, status_code=400)
            width, height = metrics["viewport"]
            entries[f"{state}-{width}x{height}-{metrics['theme']}"] = metrics
            path.write_text(json.dumps(entries, indent=2) + "\n")
        return JSONResponse(data)

    app.router.routes.insert(0, Route("/showcase/metrics.js", metrics_script))
    app.router.routes.insert(0, Route("/showcase/metrics", save_metrics, methods=["POST"]))
    baseline_path = os.environ.get("LOP_SHOWCASE_BASELINE_DIST")
    if baseline_path:
        baseline = Path(baseline_path)
        if not (baseline / "index.html").is_file():
            raise RuntimeError("configured baseline bundle has no index.html")

        async def baseline_index(_request):
            # Assets alone are namespaced so both frozen bundles can share the
            # SAME fixture origin, API data and browser cookie without rebuilding
            # either. React/API/SSE bytes remain the actual frozen-base build.
            html = (
                (baseline / "index.html")
                .read_text()
                .replace('"./assets/', '"/baseline/assets/')
                .replace('"/assets/', '"/baseline/assets/')
            )
            return HTMLResponse(html)

        app.router.routes.insert(
            0, Mount("/baseline/assets", app=StaticFiles(directory=baseline / "assets"))
        )
        app.router.routes.insert(0, Route("/baseline", baseline_index))
    app.router.routes.insert(0, Route("/showcase", showcase))
    return app


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4290
    # The cookie namespace is decided HERE, where the port is known: the module
    # import above can only honour an explicit name (see `fixture_cookie_name`).
    daemon_module.COOKIE_NAME = fixture_cookie_name(port)
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    if os.environ.get(EMPTY_ENV) == "1":
        # THE EMPTY LIST: the same daemon over an untouched sandbox store, so the
        # empty state is the product's own answer to "no conversations", not a
        # client-side mock.
        app = showcase_app(daemon)
        print(json.dumps({"fixture": "showcase-empty", "port": port}), flush=True)
        await uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        ).serve()
        return
    if os.environ.get("LOP_SHOWCASE_EMPTY_SHEETS") != "1":
        _substitute_readers()
    live = _live_rows()
    _seed_store(live)
    for projection, running, _unread in live:
        record = SessionRecord(
            pid=projection.pid,
            kind="tui",
            session_id=projection.session_id,
            conversation_name=projection.conversation_name,
            cwd=projection.cwd,
            model_label=projection.model_label,
            control_port=1,
            control_key="fixture",
            detached=True,
            subagents_running=running,  # type: ignore[call-arg]
            subagents_queued=0,  # type: ignore[call-arg]
        )
        entry = SessionEntry(record)
        # Through the daemon's own capture, so a roster row's DETAIL (the agent
        # view the row opens) is cached exactly as a live frame would cache it.
        projection = daemon.capture_subagent_details(projection, record=record)
        entry.projection = projection
        entry.degraded = projection.degraded
        daemon.session_projections[projection.session_id] = projection
        daemon.table.entries[record.pid] = entry
    # The ENDED conversation: a durable row plus an ended entry, which is what the
    # daemon reads to set the summary's ``ended`` (the overflow fixture's recipe).
    ended = _ended_projection()
    _conversation_dir(ended.session_id, ended.conversation_name)
    ended_record = SessionRecord(
        pid=ended.pid,
        kind="tui",
        session_id=ended.session_id,
        conversation_name=ended.conversation_name,
        cwd=ended.cwd,
        model_label=ended.model_label,
        control_port=1,
        control_key="fixture",
    )
    ended_entry = SessionEntry(ended_record)
    ended_entry.projection = ended
    ended_entry.ended = True
    daemon.session_projections[ended.session_id] = ended
    daemon.table.entries[ended_record.pid] = ended_entry
    # THE PIN SET. The daemon loads ``table.pins`` on its discovery pass, which
    # this fixture never runs (``dial_registrants=False`` and no scanner), so it
    # is read once here through the same ``read_pins`` (with its store prune).
    from local_operator.paths import config_dir
    from local_operator.tui.sidebar_pins import read_pins

    daemon.table.pins = tuple(read_pins(config_dir()))
    app = showcase_app(daemon)
    print(json.dumps({"fixture": "showcase", "port": port}), flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())

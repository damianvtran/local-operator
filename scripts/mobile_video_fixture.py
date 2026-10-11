"""Serve synthetic video-generation sessions, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_video_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — see
the sibling fixtures' note on why a literal here would be a reusable credential.

Seven sessions, one per state the video card ships in round 1 of the surfaces
programme — tap each from the list with the shot script beside this file:

* ``Video gen queued``           — announced, nothing started, no datum: the
  state line alone. The card must NOT offer the cancel control (a video job is
  cancelled post-turn via ``jobs op='cancel'``; contract §4.2 / ruling R4).
* ``Video gen queue position``   — the same state with ``queue_position: 2``, so
  the ``queued · 2 ahead`` copy branch is photographed beside the withheld
  control (the shot script attempts the tap and records ``clicked: false``).
* ``Video gen running``          — the tile + log tail, and the absence of a
  control as the REAL gesture's own reading. No bar: video never carries a
  fraction (the desktop's D2 ruling, mirrored).
* ``Video gen cancelling``       — the wire's OWN hold (``stage:
  "cancelling"``): no local press can produce this face for video, so the feed
  is the only source and is photographed rather than inferred.
* ``Video gen done``             — the finished receipt line, and the ONE
  material word the card swaps: ``video ready``. No player v1 (frozen §5.4):
  the artifact ref family is a named follow-up, so no bytes render here.
* ``Video gen failed``           — the platform's failure sentence, rendered
  verbatim (the sanctioned shape: no vendor text ever reaches a surface).
* ``Video gen already finished`` — the cancel conflict
  (``error_type: media_already_completed``): stated as "already finished",
  never as an error.
* ``Video gen receipt``          — the POST-TURN delivery: a real ``job_result``
  custom row (the frozen §5.1 receipt text ``Session._job_result_message``
  writes) folded through the production fold into the notice entry the relay
  renders — the completion's v1 presence, photographed rather than argued.

The canonical live-detail bag (``stage``/``queue_position``/
``progress_fraction``/``log_lines``/``error``/``error_type``) is read from
``details`` BY THE ADAPTER (``web/src/lib/image-gen.ts``), the single place
their wire spelling appears; ``generate_video`` carries the same six because
its lifecycle is the same (surfaces programme, R1). The fixture seeds the
canonical bag — every key present, ``None`` when unsupplied.

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.projection import fold_messages_to_entries
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)

#: The variable NAME (never a value) a caller may use instead of the second
#: argument — the same contract the sibling fixtures document.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"

#: The frozen completion receipt the post-turn row carries (contract §5.1), in
#: the shape ``Session._job_result_message`` writes: a ``job_result`` custom
#: row whose ``details["text"]`` is the delivery sentence. Synthetic values
#: (job id, model, price, digest) in the frozen grammar — a fixture's numbers
#: must look like numbers, never like a claim about a real generation.
RECEIPT = (
    "background job 'Video generation (i2v, 8s, Radient)' completed:\n"
    "Video ready — job abc123 · model fal-ai/veo3.1 · 8.0s · $1.60 (reported) "
    "· attached: digest 0123456789abcdef0123456789abcdef"
)

#: The platform failure sentence, exactly as the frozen provider contract
#: shapes it: the surfaces never receive vendor free-text — `error` is a
#: stable sentence safe to render as-is, and `error_type` carries the
#: structured category beside it.
PROVIDER_ERROR = "This generation failed before producing output."
PROVIDER_ERROR_TYPE = "media_rejected"


def required_password(args: list[str]) -> str:
    """The password this run serves with, or a refusal naming the contract."""
    value = args[0] if args else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. Generate one with "
            "python -c 'import secrets;print(secrets.token_urlsafe(16))' and export it. "
            "It is never defaulted and never printed by this script."
        )
    return value


def _conversation(lead: str, followups: list[str]) -> list[TranscriptEntry]:
    """A short exchange, so the frames show the card above a REAL conversation."""
    entries = [TranscriptEntry(id="g-1", kind="user", text=lead)]
    for index, text in enumerate(followups, start=2):
        entries.append(TranscriptEntry(id=f"g-{index}", kind="assistant", text=text))
    return entries


def _video_card_entry(**over: object) -> TranscriptEntry:
    """The one tool row every synthetic conversation ends with."""
    entry = TranscriptEntry(
        id="tc-call-vid",
        kind="tool",
        tool_call_id="call-vid",
        tool_name="generate_video",
        tool_state="queued",
        summary="an otter surfing at golden hour",
        intent="generating that video",
    )
    for key, value in over.items():
        setattr(entry, key, value)
    return entry


def _live_details(**over: object) -> dict[str, object]:
    """The canonical update bag: EVERY key present, ``None`` when unsupplied.

    The harness lane freezes this shape per media tool (PR #2089 for images;
    the video lane carries the same six because its lifecycle is the same):
    every update carries all six keys, and a value no provider supplied is
    ``None`` — never a synthesized stand-in.
    """
    details: dict[str, object] = {
        "stage": None,
        "queue_position": None,
        "progress_fraction": None,
        "log_lines": None,
        "error": None,
        "error_type": None,
    }
    details.update(over)
    return details


def _projection(
    session_id: str,
    name: str,
    pid: int,
    entry: TranscriptEntry,
    *,
    running: bool,
    extra: list[TranscriptEntry] | None = None,
) -> SessionProjection:
    projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=name,
        streaming=running,
        transcript=_conversation(
            "Make me a video of an otter surfing at golden hour.",
            ["On it — generating that now."],
        )
        + [entry]
        + (extra or []),
        version=2,
    )
    if running:
        # The working line the relay would show while the job runs; the card
        # is judged beside it, not instead of it.
        projection.activity = "running generate_video"
        projection.activity_started_s = 12.0
    return projection


def _queued_projection() -> SessionProjection:
    return _projection(
        "videogen-queued",
        "Video gen queued",
        900301,
        _video_card_entry(
            tool_state="queued",
            summary="waiting to run generate_video",
            details=_live_details(stage="queued"),
        ),
        running=False,
    )


def _queued_position_projection() -> SessionProjection:
    return _projection(
        "videogen-queued-pos",
        "Video gen queue position",
        900302,
        _video_card_entry(
            tool_state="queued",
            summary="waiting to run generate_video",
            details=_live_details(stage="queued", queue_position=2),
        ),
        running=False,
    )


def _running_projection() -> SessionProjection:
    return _projection(
        "videogen-running",
        "Video gen running",
        900303,
        _video_card_entry(
            tool_state="running",
            details=_live_details(
                stage="in_progress",
                log_lines=[
                    {
                        "message": "rung 1/2: fal-ai/veo3.1 queued",
                        "timestamp": "2026-10-10T12:00:00Z",
                    },
                    {
                        "message": "rung 1/2: frames streaming",
                        "timestamp": "2026-10-10T12:00:04Z",
                    },
                ],
            ),
        ),
        running=True,
    )


def _cancelling_projection() -> SessionProjection:
    """The wire's OWN cancelling hold (``stage: "cancelling"``).

    For video there is no control to press — a video job is cancelled
    post-turn, and the turn-interrupt rail is not a cancel for it (contract
    §4.2) — so this feed-driven face is the ONLY way a ``cancelling`` card
    exists, and the hold it paints is the wire's word, not a local press.
    """
    return _projection(
        "videogen-cancelling",
        "Video gen cancelling",
        900304,
        _video_card_entry(
            tool_state="running",
            details=_live_details(stage="cancelling"),
        ),
        running=True,
    )


def _done_projection() -> SessionProjection:
    return _projection(
        "videogen-done",
        "Video gen done",
        900305,
        _video_card_entry(
            tool_state="done",
            elapsed_s=8.0,
        ),
        running=False,
    )


def _failed_projection() -> SessionProjection:
    return _projection(
        "videogen-failed",
        "Video gen failed",
        900306,
        _video_card_entry(
            tool_state="failed",
            error=PROVIDER_ERROR,
            details={"error": PROVIDER_ERROR, "error_type": PROVIDER_ERROR_TYPE},
        ),
        running=False,
    )


def _already_finished_projection() -> SessionProjection:
    """The cancel conflict: the stop raced an already-completed job."""
    return _projection(
        "videogen-finished",
        "Video gen already finished",
        900307,
        _video_card_entry(
            tool_state="failed",
            details={
                "stage": "cancelled",
                "error": (
                    "The generation had already completed when the cancel arrived; "
                    "its result was discarded."
                ),
                "error_type": "media_already_completed",
            },
        ),
        running=False,
    )


def _receipt_projection() -> SessionProjection:
    """The POST-TURN delivery: the completion as the `job_result` receipt row.

    Built the production way — a ``job_result`` ``CustomMessage``
    (``Session._job_result_message``'s shape) run through
    ``fold_messages_to_entries``, the same fold the relay serves — so the frame
    photographs the rendered receipt the phone actually shows, not a
    hand-written notice that merely resembles it.
    """
    from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    delivery = CustomMessage(
        custom_type=JOB_RESULT_MESSAGE_TYPE,
        attribution="user",
        details={"job_id": "abc123", "text": RECEIPT},
    )
    folded = fold_messages_to_entries([delivery])
    return _projection(
        "videogen-receipt",
        "Video gen receipt",
        900308,
        _video_card_entry(
            tool_state="done",
            elapsed_s=8.0,
        ),
        running=False,
        extra=folded,
    )


def _ensure_session_dir(session_id: str) -> None:
    """Create the conversation directory the session LISTING requires.

    Same reason as the sibling fixture: ``daemon._live_generation_is_user_facing``
    lists a live in-memory session only when ``config_dir()/sessions/<id>``
    exists, so without this the list route hides the sessions and every tap
    photographs the bare list instead of a card.
    """
    from local_operator.paths import config_dir

    (config_dir() / "sessions" / session_id).mkdir(parents=True, exist_ok=True)


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4191
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for projection in (
        _queued_projection(),
        _queued_position_projection(),
        _running_projection(),
        _cancelling_projection(),
        _done_projection(),
        _failed_projection(),
        _already_finished_projection(),
        _receipt_projection(),
    ):
        record = SessionRecord(
            pid=projection.pid,
            kind="tui",
            session_id=projection.session_id,
            conversation_name=projection.conversation_name,
            cwd="/synthetic",
            model_label="fixture",
            control_port=1,
            control_key="fixture",
        )
        entry = SessionEntry(record)
        entry.projection = projection
        _ensure_session_dir(projection.session_id)
        daemon.session_projections[projection.session_id] = projection
        daemon.table.entries[record.pid] = entry
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())

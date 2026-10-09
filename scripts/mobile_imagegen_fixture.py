"""Serve synthetic image-generation sessions, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_imagegen_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — see
the sibling fixtures' note on why a literal here would be a reusable credential.

Ten sessions, one per state the image-generation card ships in round 1 of the
surfaces lane — tap each from the list with the shot script beside this file:

* ``Image gen queued``           — announced, nothing started: the state line
  alone (the reduced state, before the feed carries a queue position).
* ``Image gen queue position``   — the same card with ``queue_position: 2`` in
  its live details, so the position branch is photographed rather than argued.
* ``Image gen running``          — the tile + the indeterminate bar. THE CANCEL
  TAP IS CAPTURED HERE: the shot script presses the card's own cancel and
  photographs the ``cancelling…`` hold before any confirmation lands.
* ``Image gen cancelling``       — the wire's OWN hold (``stage:
  "cancelling"``): the same card the press produces, arriving from the feed
  alone — no click, so a feed-driven hold is photographed rather than inferred.
* ``Image gen progress``         — the determinate branch: ``progress_fraction:
  0.42`` plus a ``log_lines`` tail, and a carried ``elapsed_s`` (the row's
  clock cell) so the "when the feed carries a number" branches are all visible
  at once.
* ``Image gen mid-walk failure`` — a rung-failure beat (``stage: None``, the
  semantics in ``error``/``error_type``) while the walk continues: the call is
  still live, so the card keeps its live state and the pair rides the row.
* ``Image gen done``             — a finished artifact. The fixture seeds a REAL
  transcript for this session (see ``_seed_done_transcript``), so the frame
  renders bytes served by the production ``/api/sessions/{id}/image`` route —
  not a stubbed image.
* ``Image gen failed``           — the platform's failure sentence, rendered
  verbatim (the sanctioned shape: no vendor text ever reaches a surface).
* ``Image gen already finished`` — the cancel conflict
  (``error_type: media_already_completed``): stated as "already finished",
  never as an error.
* ``Image gen interrupted``      — the settle-after-a-stop state, stated as
  ``cancelled`` with no restart control (the slot is unwired in the app; see
  the card's own tests for the wired demonstration).

The canonical live-detail bag (``stage``/``queue_position``/
``progress_fraction``/``log_lines``/``error``/``error_type``) is read from
``details`` BY THE ADAPTER (``web/src/lib/image-gen.ts``), the single place
their wire spelling appears. The fixture seeds the canonical bag — every key
present, ``None`` when unsupplied — the shape every ``generate_image`` update
carries since the harness lane's freeze (PR #2089).

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import base64
import os
import struct
import sys
import zlib

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)

#: The variable NAME (never a value) a caller may use instead of the second
#: argument — the same contract the sibling fixtures document.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"

#: The entry id the done session's artifact ref resolves against. The frozen
#: attachment contract keys the image route by the entry's ``id`` (the message
#: id the endpoint looks up), and the seeded transcript writes a message with
#: exactly this id — the production seam the emitter lane also lands on.
DONE_ENTRY_ID = "tc-call-img-done"

#: The platform failure sentence, exactly as the frozen provider contract
#: shapes it: the surfaces never receive vendor free-text — `error` is a
#: stable sentence safe to render as-is, and `error_type` carries the
#: structured category beside it.
PROVIDER_ERROR = "This generation failed before producing output."
PROVIDER_ERROR_TYPE = "media_rejected"

#: The platform sentence the canonical cancel-conflict result carries beside
#: ``error_type: media_already_completed`` (harness lane, PR #2089). The card
#: states "already finished" and does not paint it — seeded because the
#: canonical shape carries it, and the adapter's ABSENCE path for this state
#: is pinned in its unit tests.
CANCEL_CONFLICT_SENTENCE = (
    "The generation had already completed when the cancel arrived; " "its result was discarded."
)


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


def _image_card_entry(**over: object) -> TranscriptEntry:
    """The one tool row every synthetic conversation ends with."""
    entry = TranscriptEntry(
        id="tc-call-img",
        kind="tool",
        tool_call_id="call-img",
        tool_name="generate_image",
        tool_state="queued",
        summary="a red panda in a spacesuit",
        intent="generating that image",
    )
    for key, value in over.items():
        setattr(entry, key, value)
    return entry


def _live_details(**over: object) -> dict[str, object]:
    """The canonical update bag: EVERY key present, ``None`` when unsupplied.

    The harness lane freezes this shape (PR #2089): every ``generate_image``
    update carries all six keys, and a value no provider supplied is ``None``
    — never a synthesized stand-in.
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
    session_id: str, name: str, pid: int, entry: TranscriptEntry, *, running: bool
) -> SessionProjection:
    projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=name,
        streaming=running,
        transcript=_conversation(
            "Draw me a red panda in a spacesuit.",
            ["On it — generating that now."],
        )
        + [entry],
        version=2,
    )
    if running:
        # The working line the relay would show while the call runs; the card
        # is judged beside it, not instead of it.
        projection.activity = "running generate_image"
        projection.activity_started_s = 12.0
    return projection


def _queued_projection() -> SessionProjection:
    return _projection(
        "imagegen-queued",
        "Image gen queued",
        900201,
        _image_card_entry(
            tool_state="queued",
            summary="waiting to run generate_image",
            details=_live_details(stage="queued"),
        ),
        running=False,
    )


def _queued_position_projection() -> SessionProjection:
    return _projection(
        "imagegen-queued-pos",
        "Image gen queue position",
        900202,
        _image_card_entry(
            tool_state="queued",
            summary="waiting to run generate_image",
            details=_live_details(stage="queued", queue_position=2),
        ),
        running=False,
    )


def _running_projection() -> SessionProjection:
    return _projection(
        "imagegen-running",
        "Image gen running",
        900203,
        _image_card_entry(tool_state="running", details=_live_details(stage="in_progress")),
        running=True,
    )


def _cancelling_projection() -> SessionProjection:
    """The wire's OWN cancelling hold (``stage: "cancelling"``).

    Distinct from the press-driven hold the shot script captures by clicking
    the running row: this row is what a feed that already reports the
    cancellation looks like, and the card must hold the same shape without a
    local press.
    """
    return _projection(
        "imagegen-cancelling",
        "Image gen cancelling",
        900209,
        _image_card_entry(tool_state="running", details=_live_details(stage="cancelling")),
        running=True,
    )


def _progress_projection() -> SessionProjection:
    return _projection(
        "imagegen-progress",
        "Image gen progress",
        900204,
        _image_card_entry(
            tool_state="running",
            elapsed_s=42.0,
            details=_live_details(
                stage="in_progress",
                progress_fraction=0.42,
                log_lines=[
                    {"message": "diffusion step 12/30", "timestamp": "2026-10-09T12:00:00Z"},
                    {"message": "diffusion step 18/30", "timestamp": "2026-10-09T12:00:01Z"},
                    {"message": "sampling 24/30 (cfg 7.5)", "timestamp": "2026-10-09T12:00:02Z"},
                    {"message": "decoding latents", "timestamp": "2026-10-09T12:00:03Z"},
                ],
            ),
        ),
        running=True,
    )


def _mid_walk_failure_projection() -> SessionProjection:
    """A mid-walk failure beat (``stage: None``, semantics in error/error_type).

    The walk continues after a rung fails — the next update (the next rung's
    ``queued``) replaces this one — so the call is still live and the card
    keeps its live state; the pair rides the row for the settle that follows.
    """
    return _projection(
        "imagegen-mid-walk",
        "Image gen mid-walk failure",
        900210,
        _image_card_entry(
            tool_state="running",
            details=_live_details(
                stage=None,
                error="Radient exceeded its 120s generation budget.",
                error_type="timeout",
            ),
        ),
        running=True,
    )


def _done_projection() -> SessionProjection:
    return _projection(
        "imagegen-done",
        "Image gen done",
        900205,
        _image_card_entry(
            id=DONE_ENTRY_ID,
            tool_state="done",
            elapsed_s=12.1,
            images=[{"index": 0, "mime_type": "image/png"}],
        ),
        running=False,
    )


def _failed_projection() -> SessionProjection:
    return _projection(
        "imagegen-failed",
        "Image gen failed",
        900206,
        _image_card_entry(
            tool_state="failed",
            error=PROVIDER_ERROR,
            details={"error": PROVIDER_ERROR, "error_type": PROVIDER_ERROR_TYPE},
        ),
        running=False,
    )


def _already_finished_projection() -> SessionProjection:
    """The cancel conflict: the stop raced an already-completed job.

    The canonical result (harness lane, PR #2089): ``error_type:
    media_already_completed`` beside the ``stage: "cancelled"`` word and the
    platform's own sentence. The card must state "already finished" — never
    an error, because nothing failed — and the adapter's ABSENCE path for
    this state is pinned in its unit tests rather than seeded here.
    """
    return _projection(
        "imagegen-finished",
        "Image gen already finished",
        900208,
        _image_card_entry(
            tool_state="failed",
            details={
                "stage": "cancelled",
                "error": CANCEL_CONFLICT_SENTENCE,
                "error_type": "media_already_completed",
            },
        ),
        running=False,
    )


def _interrupted_projection() -> SessionProjection:
    return _projection(
        "imagegen-cancelled",
        "Image gen interrupted",
        900207,
        _image_card_entry(tool_state="interrupted"),
        running=False,
    )


def _gradient_png(width: int, height: int) -> bytes:
    """A valid PNG, generated in-process — the fixture ships no binary asset.

    A warm diagonal gradient reads unmistakably as a finished picture in the
    frame while staying obviously synthetic (nobody mistakes it for a model
    output), which is exactly what a capture fixture's artifact should be.
    """
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter byte: none
        for x in range(width):
            rows += bytes(
                (
                    40 + (x * 200) // width,
                    90 + (y * 120) // height,
                    160 - (x * 80) // width,
                )
            )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def _ensure_session_dir(session_id: str) -> None:
    """Create the conversation directory the session LISTING requires.

    ``daemon._live_generation_is_user_facing`` lists a live in-memory session
    only when ``config_dir()/sessions/<id>`` exists and its marker (when one
    is readable) is a user origin — and a directory with no marker at all is
    the fail-safe "the user's own". The synthetic projections are otherwise
    in-memory only, so without this the list route hides six of the seven
    sessions and every tap photographs the bare list instead of a card.
    """
    from local_operator.paths import config_dir

    (config_dir() / "sessions" / session_id).mkdir(parents=True, exist_ok=True)


async def _seed_done_transcript(session_id: str, entry_id: str) -> None:
    """Write the artifact the production image route serves.

    ``daemon._image_bytes`` resolves bytes from the ON-DISK transcript by
    message id, so this seeds a real ``transcript.jsonl`` through the real
    ``Transcript`` API into the isolated config root: the capture then exercises
    the production route end to end (lazy fetch, immutable caching headers),
    not a stub of it. The message id IS the entry id the ref carries — the
    contract the emitter lane lands on at the same seam.
    """
    from local_operator.harness.types import ImageContent, Message
    from local_operator.paths import config_dir
    from local_operator.session.transcript import Transcript

    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    message = Message.user(
        "the generated image",
        [
            ImageContent(
                data=base64.b64encode(_gradient_png(512, 384)).decode(),
                mime_type="image/png",
            )
        ],
        id=entry_id,
    )
    await Transcript(directory).append_message(message)


class _CaptureDaemon(MobileDaemon):
    """The real daemon, with control-command DELIVERY stubbed for capture.

    The card's cancel rides ``{op:"abort"}`` to the session's live runtime. A
    fixture has no runtime and no writer, so the real route would answer the
    abort "session not connected" — and the card, correctly, treats a refused
    abort as one that never landed and returns to ``running``. The frame this
    rig exists to photograph is the HOLD between the press and any
    confirmation, which by definition has no confirmation yet: the stub acks
    the op and the synthetic projection keeps reporting ``running``.
    """

    async def request(self, pid: int, op: str, **fields: object) -> dict[str, object]:
        return {"ok": True, "detail": "stopping the run"}


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon = _CaptureDaemon(port=port, password=password, dial_registrants=False)
    for projection in (
        _queued_projection(),
        _queued_position_projection(),
        _running_projection(),
        _cancelling_projection(),
        _progress_projection(),
        _mid_walk_failure_projection(),
        _done_projection(),
        _failed_projection(),
        _already_finished_projection(),
        _interrupted_projection(),
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
    await _seed_done_transcript("imagegen-done", DONE_ENTRY_ID)
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())

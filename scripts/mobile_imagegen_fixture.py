"""Serve synthetic image-generation sessions, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_imagegen_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — see
the sibling fixtures' note on why a literal here would be a reusable credential.

Eight sessions, one per state the image-generation card ships in round 1 of the
surfaces lane — tap each from the list with the shot script beside this file:

* ``Image gen queued``           — announced, nothing started: the state line
  alone (the reduced state, before the feed carries a queue position).
* ``Image gen queue position``   — the same card with ``queue_position: 2`` in
  its live details, so the position branch is photographed rather than argued.
* ``Image gen running``          — the tile + the indeterminate bar. THE CANCEL
  TAP IS CAPTURED HERE: the shot script presses the card's own cancel and
  photographs the ``cancelling…`` hold before any confirmation lands.
* ``Image gen progress``         — the determinate branch: ``progress: 0.42``
  plus a log tail, and a carried ``elapsed_s`` (the row's clock cell) so the
  "when the feed carries a number" branches are all visible at once.
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

The live-detail fields (``queue_position``/``progress``/``logs``) are read from
``details`` BY THE ADAPTER (``web/src/lib/image-gen.ts``), which is the single
place their wire spelling appears — when the relay freezes the names, this
fixture moves with that one module.

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
        _image_card_entry(tool_state="queued", summary="waiting to run generate_image"),
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
            details={"queue_position": 2},
        ),
        running=False,
    )


def _running_projection() -> SessionProjection:
    return _projection(
        "imagegen-running",
        "Image gen running",
        900203,
        _image_card_entry(tool_state="running"),
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
            details={
                "progress": 0.42,
                "logs": [
                    "diffusion step 12/30",
                    "diffusion step 18/30",
                    "sampling 24/30 (cfg 7.5)",
                    "decoding latents",
                ],
            },
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
            details={"error_type": PROVIDER_ERROR_TYPE},
        ),
        running=False,
    )


def _already_finished_projection() -> SessionProjection:
    """The cancel conflict: the stop raced an already-completed job.

    The frozen provider contract: this answers with
    ``error_type: media_already_completed`` and the card must state "already
    finished" — never an error, because nothing failed. Seeded with NO error
    sentence, which is also the absence path this state has to look right on.
    """
    return _projection(
        "imagegen-finished",
        "Image gen already finished",
        900208,
        _image_card_entry(
            tool_state="failed",
            details={"error_type": "media_already_completed"},
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
        _progress_projection(),
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

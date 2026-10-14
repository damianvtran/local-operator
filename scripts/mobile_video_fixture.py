"""Serve synthetic video-generation sessions, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_video_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — see
the sibling fixtures' note on why a literal here would be a reusable credential.

Ten sessions, one per state the video card ships in round 1 of the surfaces
programme plus the two player states of the bounded follow-up (§3.2) — tap
each from the list with the shot script beside this file:

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
* ``Video gen player``          — the follow-up's READY state: a done row whose
  video ref resolves through the PRODUCTION attachment route to a real,
  decodable clip (``_seed_player_transcript``) — the browser's own media
  stack ranges the bytes in, so the frame is a player over real bytes.
* ``Video gen player missing``  — the follow-up's FAILURE state: the same ref
  with nothing behind it (no transcript seeded), so the route answers 404
  and the card states ``video unavailable`` — photographed, not argued.

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


#: A real, decodable, TINY test clip (1 s, 128x72, H.264 baseline, ~4 KB),
#: EMBEDDED because the rig's fixture must serve REAL bytes through the
#: production attachment route for the browser to decode — a stubbed body
#: would prove nothing about the player or its Range requests. Clearly a test
#: asset: generated once with ffmpeg's own test source, not a user recording.
TEST_VIDEO_MP4_B64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAANNbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+"
    "gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAgAAAnd0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAA"
    "AAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAIAAAABIAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAA"
    "AAEAAAPoAAAAAAABAAAAAAHvbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAAKABVxAAAAAAALWhkbHIAAAAAAA"
    "AAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABmm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5m"
    "AAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAVpzdGJsAAAAunN0c2QAAAAAAAAAAQAAAKphdmMxAAAAAAAAAA"
    "EAAAAAAAAAAAAAAAAAAAAAAIAASABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDIgbGlieDI2NAAAAAAAAAAAAAAA"
    "GP//AAAAMGF2Y0MBQsAe/+EAGGdCwB7ZAgv5cBEAAAMAAQAAAwAUDxYuSAEABWjLgOSyAAAAEHBhc3AAAAABAAAAAQ"
    "AAABRidHJ0AAAAAAAAZpgAAAAAAAAAGHN0dHMAAAAAAAAAAQAAAAoAAAQAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAAAc"
    "c3RzYwAAAAAAAAABAAAAAQAAAAoAAAABAAAAPHN0c3oAAAAAAAAAAAAAAAoAAAVjAAAA8gAAALMAAAC6AAAAwQAAAP"
    "kAAACwAAABAgAAAL0AAADoAAAAFHN0Y28AAAAAAAAAAQAAA30AAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAA"
    "AAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYyLjEyLjEwMg"
    "AAAAhmcmVlAAAM221kYXQAAAJxBgX//23cRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY1IHIzMjIyIGIzNTYw"
    "NWEgLSBILjI2NC9NUEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDI1IC0gaHR0cDovL3d3dy52aWRlb2"
    "xhbi5vcmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MCByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgx"
    "OjB4MTExIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MT"
    "YgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0wIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3RfcHNraXA9MSBj"
    "aHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MiBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbn"
    "I9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJh"
    "bWVzPTAgd2VpZ2h0cD0wIGtleWludD0yNTAga2V5aW50X21pbj0xMCBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPT"
    "AgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9NDAuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9"
    "NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAAupliIQn+IrBgQh+KMuF4Tf19x/XtyRSbGq0lY"
    "kjtA+8IhMobgEgCbOSHAaGmEUVkSRQqWHpXF4JojERg/6eNL/DfBql6Tw3KPLI8Ihoctv2DcwAZM51EpfJ+waY1YiJ"
    "uImNihGvlqFrIN9LyCuDn6m+8T4cYdp7L69/FaDbVQdadQCbVi/saZJ1IWWKyviF3iMBEIbYjH/NkymZ5VJlNKECGa"
    "yuWVFy3MKATcSSBjEqKy6f28cQKO17RypvjjiDCEAgyGiiFL2SZWyDUx2/IdyHb8bbXJhamOOPUYTGEJM1o6Q5TShV"
    "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"
)

#: The clip's length, carried on the ref like any artifact's duration.
TEST_VIDEO_DURATION_S = 1.0


def _video_bytes() -> bytes:
    """The embedded test clip, decoded per call (fixture lifetime only)."""
    import base64

    return base64.b64decode(TEST_VIDEO_MP4_B64)


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


def _player_projection() -> SessionProjection:
    """The done card carrying a REAL video ref: the player's bytes are served
    by the production attachment route from the seeded transcript (see
    ``_seed_player_transcript``), so the frame a player over real bytes, not
    a stub."""
    return _projection(
        "videogen-player",
        "Video gen player",
        900309,
        _video_card_entry(
            tool_state="done",
            elapsed_s=8.0,
            videos=[
                {
                    "index": 0,
                    "mime_type": "video/mp4",
                    "duration_s": TEST_VIDEO_DURATION_S,
                }
            ],
        ),
        running=False,
    )


def _player_missing_projection() -> SessionProjection:
    """The done card whose ref resolves to NOTHING: no transcript is seeded for
    this session, so the production route answers 404 and the card states
    ``video unavailable`` — the designed failure state, photographed rather
    than argued."""
    return _projection(
        "videogen-player-missing",
        "Video gen player missing",
        900310,
        _video_card_entry(
            tool_state="done",
            elapsed_s=8.0,
            videos=[{"index": 0, "mime_type": "video/mp4"}],
        ),
        running=False,
    )


async def _seed_player_transcript(session_id: str) -> None:
    """Write the artifact the production attachment route serves.

    ``daemon._attachment_bytes(kind="video")`` resolves bytes from the
    ON-DISK transcript by message id — or tool call id, the row id
    ``tc-call-vid`` recovered to its call — so this seeds a real
    ``transcript.jsonl`` through the real ``Transcript`` API into the
    isolated config root and registers the clip through the real
    ``cache_media`` store call: the capture exercises the production route
    end to end (lazy fetch, Range/206, immutable caching), not a stub of it.
    """
    from local_operator.harness.types import Message, TextContent, ToolResult
    from local_operator.paths import config_dir
    from local_operator.session.attachments import cache_media
    from local_operator.session.transcript import Transcript

    artifact = cache_media(
        _video_bytes(), "video/mp4", kind="video", duration_s=TEST_VIDEO_DURATION_S
    )
    if artifact is None:
        raise SystemExit("the embedded test clip could not be registered in the store")
    message = Message.tool_result(
        ToolResult(
            tool_call_id="call-vid",
            tool_name="generate_video",
            content=[TextContent(text="Video ready"), artifact],
        )
    )
    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    await Transcript(directory).append_message(message)


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
        _player_projection(),
        _player_missing_projection(),
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
    # The player's bytes: the real clip behind the real route for the READY
    # session; the MISSING session is deliberately left unseeded so its ref
    # resolves to nothing.
    await _seed_player_transcript("videogen-player")
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())

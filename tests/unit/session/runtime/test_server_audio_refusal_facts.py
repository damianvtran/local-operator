"""The audio refusal's FACTS across the attach transport.

``AudioInputUnsupported`` names the model and carries the reason (the
resolver's report for a capability refusal, the wire's constraint for a format
one), and those two facts are what get a stuck user unstuck. That detail only
helps if it reaches the surface that shows it: the desktop 409 and the phone's
422 render the error the client REBUILDS from the frame, not the owner's
string, so facts that stop at the socket leave the surface with the bare
sentence (agent review round 1, m2 / QA observations Q3).

The facts travel in their own bounded fields (``error_model``,
``error_report``, ``error_format``) rather than inside the message, and the
decoder sanitises and caps each before rebuilding the sentence locally — the
same closed-shape carriage ``error_count``/``error_trigger`` established. A
frame without them (an older runtime, which never raised the format shape)
degrades to the bare form exactly as before.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from local_operator.session.errors import (
    AttachmentUnavailable,
    AudioInputUnsupported,
    admission_error,
)
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from tests.unit.session.runtime.test_server import FakeHandle


def _rig(exc: Exception) -> tuple[RuntimeServer, list[dict[str, Any]], _ClientConn]:
    """A server whose dispatch raises ``exc``, with its socket writes captured."""
    server = RuntimeServer(FakeHandle(), kind="tui")
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def failing_dispatch(op, frame, **_kwargs):  # noqa: ANN001
        raise exc

    server._send_to = capture  # type: ignore[assignment]
    server._dispatch = failing_dispatch  # type: ignore[assignment]
    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    server._clients[id(conn.writer)] = conn
    return server, sent, conn


async def _error_frame(exc: Exception) -> dict[str, Any]:
    server, sent, conn = _rig(exc)
    await server._on_request({"op": "prompt", "req": 1}, conn)
    errors = [f for f in sent if f.get("op") == "error"]
    assert errors, f"expected an error frame, got {sent}"
    # Round-trip through JSON: the frame really goes over a socket.
    return cast(dict[str, Any], json.loads(json.dumps(errors[0])))


@pytest.mark.asyncio
async def test_the_format_refusal_facts_reach_the_attach_client() -> None:
    """End to end: raise on the owner, decode on the client, keep both facts."""
    report = "the chat audio part takes wav or mp3 only (this capture is audio/webm)"
    frame = await _error_frame(
        AudioInputUnsupported(model="openai/gpt-audio", report=report, format_unsupported=True)
    )

    assert frame["error_code"] == AudioInputUnsupported.code
    assert frame["error_model"] == "openai/gpt-audio"
    assert frame["error_report"] == report
    assert frame["error_format"] is True

    # Exactly what attach_client.py does with the reply.
    known = admission_error(
        str(frame.get("error_code", "")),
        frame.get("error_count"),
        frame.get("error_trigger"),
        "",
        model=frame.get("error_model"),
        report=frame.get("error_report"),
        format_unsupported=frame.get("error_format"),
    )
    assert isinstance(known, AudioInputUnsupported)
    sentence = str(known)
    assert "openai/gpt-audio" in sentence, sentence
    assert "audio/webm" in sentence, sentence
    assert "No transcoding" in sentence, sentence


@pytest.mark.asyncio
async def test_the_capability_refusal_keeps_its_resolver_report() -> None:
    """The capability shape crosses with the model and the resolver's words."""
    frame = await _error_frame(
        AudioInputUnsupported(
            model="test/m",
            report="No transcription provider is available for this machine.",
        )
    )

    assert "error_format" not in frame, "the capability shape is not a format refusal"
    known = admission_error(
        str(frame.get("error_code", "")),
        None,
        None,
        "",
        model=frame.get("error_model"),
        report=frame.get("error_report"),
    )
    assert isinstance(known, AudioInputUnsupported)
    assert "test/m" in str(known)
    assert "No transcription provider is available" in str(known)


@pytest.mark.asyncio
async def test_a_frame_without_the_facts_rebuilds_the_bare_form() -> None:
    """An older runtime's frame names no facts; the sentence degrades sanely.

    The bare form is also what a NEW runtime sends for a bare raise — it is
    the same absence, and the remedy sentence is the same.
    """
    frame = await _error_frame(AudioInputUnsupported())

    assert frame["error_code"] == AudioInputUnsupported.code
    assert "error_model" not in frame
    assert "error_report" not in frame
    assert "error_format" not in frame

    known = admission_error(str(frame.get("error_code", "")))
    assert isinstance(known, AudioInputUnsupported)
    sentence = str(known)
    assert "cannot be sent to it" in sentence
    assert "Transcribe the recording first" in sentence


@pytest.mark.asyncio
async def test_other_admission_categories_carry_no_audio_facts() -> None:
    """The three fields belong to one category, not to error frames generally."""
    frame = await _error_frame(AttachmentUnavailable())

    assert frame["error_code"] == AttachmentUnavailable.code
    assert "error_model" not in frame
    assert "error_report" not in frame
    assert "error_format" not in frame


def test_a_peer_cannot_push_control_characters_into_the_rebuilt_sentence() -> None:
    """The decoder is the boundary: control characters are dropped, not rendered.

    The far side is untrusted input, so the sanitizer runs on every string the
    frame contributes, and a non-string degrades to absent (the bare form)
    rather than raising inside the decoder.
    """
    injected = admission_error(
        AudioInputUnsupported.code,
        None,
        None,
        "",
        model="openai/gpt-audio\nFAKE LINE\x00",
        report="reason\r\nwith breaks",
        format_unsupported=True,
    )
    assert isinstance(injected, AudioInputUnsupported)
    sentence = str(injected)
    assert "\n" not in sentence and "\r" not in sentence and "\x00" not in sentence
    assert "FAKE LINE" in sentence, "printable text stays; only control characters go"

    # A frame whose fields are not strings at all degrades to the bare form.
    # Splatted through a garbage dict because the point IS passing values the
    # signature forbids — an untrusted peer does not honour our annotations.
    garbage = cast(dict[str, Any], {"model": 7, "report": ["x"], "format_unsupported": "y"})
    degraded = admission_error(AudioInputUnsupported.code, None, None, "", **garbage)
    assert isinstance(degraded, AudioInputUnsupported)
    assert "cannot be sent to it" in str(degraded)


def test_an_overlong_report_is_truncated_not_dropped() -> None:
    """A clipped tail still names the model and the remedy; a dropped field
    would silently downgrade to the bare form."""
    keep = "x" * 250
    decoded = admission_error(
        AudioInputUnsupported.code,
        None,
        None,
        "",
        model="openai/gpt-audio",
        report="a" * 250 + keep,
        format_unsupported=True,
    )
    assert isinstance(decoded, AudioInputUnsupported)
    sentence = str(decoded)
    assert "openai/gpt-audio" in sentence
    assert keep not in sentence, "the report must have been capped"

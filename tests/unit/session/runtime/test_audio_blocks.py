"""The recording's ingest edge: sniffed, never trusting the declared mime.

Audio enters a session over the wire as ``[{"data_b64", "mime_type"}]`` and is
decoded into ``AudioContent`` blocks by ``audio_blocks`` — the mirror of
``image_blocks``, with one difference that is the whole point of these tests:
the OpenAI-compatible chat wire keys its ``input_audio.format`` token on the
block's mime, so a block that echoes the client's DECLARATION picks a format
token for a container the bytes are not, which is a provider 400 inside a paid
turn. The block therefore carries what ``media.sniff_audio`` verified.

The image ingest's contract carries over: bad entries are dropped, not fatal,
and empty inputs are empty lists — the callers treat that as "no recording".
"""

from __future__ import annotations

import base64
from typing import cast

from local_operator.media import sniff_audio
from local_operator.session.runtime.server import audio_blocks

#: A minimal RIFF/WAVE header plus padding — enough for the sniffer, and small
#: enough that every case below stays a unit test.
WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 32
#: EBML magic, the browser recorder's native container.
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 40


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def test_the_declared_mime_is_never_trusted() -> None:
    """A webm capture declared as wav keeps its BYTES and gains the truth.

    This is the mislabeled-phone case: the declared value is what a client
    typed, the sniffed value is what a provider will decode.
    """
    blocks = audio_blocks([{"data_b64": _b64(WEBM), "mime_type": "audio/wav"}])

    assert len(blocks) == 1
    assert blocks[0].mime_type == "audio/webm"
    sniffed = sniff_audio(WEBM)
    assert sniffed is not None
    assert blocks[0].mime_type == sniffed.mime_type
    assert blocks[0].data == _b64(WEBM)


def test_a_wav_capture_round_trips_and_matches_the_sniffer() -> None:
    blocks = audio_blocks([{"data_b64": _b64(WAV), "mime_type": "audio/wav"}])

    assert len(blocks) == 1
    assert blocks[0].mime_type == "audio/wav"
    assert base64.b64decode(blocks[0].data) == WAV


def test_bad_entries_are_dropped_not_fatal() -> None:
    """One bad entry costs that entry, never the whole prompt.

    Same contract as ``image_blocks``: not base64, not audio at all, an empty
    payload — each is dropped with a debug log, and the good block proceeds.
    """
    good = {"data_b64": _b64(WAV), "mime_type": "audio/wav"}

    blocks = audio_blocks(
        # ``cast`` to say the wrong shapes are deliberate: the entry that is a
        # bare string is the point of the test, not a type slip.
        cast(
            list[dict[str, str]],
            [
                {"data_b64": "not base64 at all!!", "mime_type": "audio/wav"},
                {"data_b64": _b64(b"this is not audio"), "mime_type": "audio/wav"},
                {"data_b64": "", "mime_type": "audio/wav"},
                "not a dict",
                good,
            ],
        )
    )

    assert len(blocks) == 1
    assert blocks[0].data == good["data_b64"]


def test_no_audio_is_an_empty_list() -> None:
    """``None`` and ``[]`` both mean "no recording", which the callers rely on."""
    assert audio_blocks(None) == []
    assert audio_blocks([]) == []

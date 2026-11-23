"""The desktop app's ``ask_attest`` frame, pinned end to end (UI dialler contract).

WHY THIS FILE EXISTS. The attestation listener (``asks/app_attest.py``, F2)
shipped in core, but for a while nothing dialled it: the desktop app's main
process never sent ``ask_attest``, so every desktop answer was stamped
``unverified`` and could not lift a delete. The dialler lives in the OTHER
repository (local-operator-ui, ``feat/ask-attested-answer``), so nothing in
either repo's CI would notice if the two sides drifted apart: a renamed key, an
image digest computed over the base64 text instead of the decoded bytes, or a
``reason`` the listener no longer accepts would silently turn every desktop
answer back into ``unverified``. This file is the cross-repo drift guard.

WHAT IS PINNED, and how it differs from ``test_app_attest.py``. That file tests
each layer on its own: the listener's replies, and the stamp matrix with rows
written straight into the table. Here every attestation goes in over the REAL
socket as the exact frame the app sends::

    {"op": "ask_attest", "session_id", "ask_id", "act", "answers", "images", "reason"}

and the stamp is read off the REAL path that consumes it (``RuntimeServer._dispatch``
through ``_ask_answer_stamp``), so the frame shape, the image-digest rule and
the content binding are checked as one contract. Only the peer verifier is a
double (the code-signature check needs the signed app; ``test_app_attest.py``
covers it with a fake Security.framework backend and the real one).

If a test here fails after a change on either side, change BOTH sides together:
the app's dialler and this file describe the same wire.
"""

from __future__ import annotations

import base64
import hashlib
import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import app_attest, surfaces
from local_operator.asks.app_attest import AttestationTable, AttestService
from local_operator.session.runtime.server import _ask_answer_stamp, _ConnectionAuthority
from tests.unit.asks.test_app_attest import (  # noqa: F401
    _Clock,
    _dial,
    _FixedVerifier,
    _token,
    short_root,
)
from tests.unit.session.runtime.test_ask_answer_surface_stamp import (  # noqa: F401
    _Rig,
    home,
)

#: The desktop window's own connection: local attach, no operator capability
#: (the wake supervisor, not the app, started the runtime). Only an
#: attestation can make its answer ``desktop``.
_DESKTOP = _ConnectionAuthority(locality="local", surface="desktop", kind="attach")
_RELAYED = _ConnectionAuthority(
    locality="remote", surface="desktop", kind="attach", operator_proved=True
)

#: The exact key set the app's dialler writes. A key added or renamed on
#: either side must show up here first.
_FRAME_KEYS = {"op", "session_id", "ask_id", "act", "answers", "images", "reason"}

_PNG = b"\x89PNG\r\n\x1a\n contract-test pixels"


def _ui_frame(
    session_id: str,
    ask_id: str,
    act: str,
    answers: dict[str, list[str]] | None = None,
    images: list[dict[str, str]] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """The frame as the app builds it: every key always present, ``reason`` null
    unless dismissing, ``images`` as ``{question_id, mime_type, sha256}``."""
    frame = {
        "op": "ask_attest",
        "session_id": session_id,
        "ask_id": ask_id,
        "act": act,
        "answers": answers if answers is not None else {},
        "images": images if images is not None else [],
        "reason": reason,
    }
    assert set(frame) == _FRAME_KEYS
    return frame


def _ui_image(data: bytes, question_id: str = "q1", mime: str = "image/png") -> dict[str, str]:
    """An attested image: sha256 hex over the DECODED bytes, not the base64 text."""
    digest = hashlib.sha256(data).hexdigest()
    return {"question_id": question_id, "mime_type": mime, "sha256": digest}


def _wire_image(data: bytes, question_id: str = "q1", mime: str = "image/png") -> dict[str, str]:
    """The same image as serve forwards it in ``ask_respond.images``."""
    data_b64 = base64.b64encode(data).decode()
    return {"question_id": question_id, "data_b64": data_b64, "mime_type": mime}


async def _listen(
    root: Path, session_id: str, table: AttestationTable | None = None
) -> AttestService:
    service = AttestService(
        session_id, verifier=_FixedVerifier(True), token_of=lambda s: _token(), table=table
    )
    assert await service.start(root / "run") is not None
    return service


async def _attest(service: AttestService, *frames: dict[str, Any]) -> list[dict[str, Any]]:
    assert service.path is not None
    replies = await _dial(service.path, list(frames))
    assert replies[0] == {"ok": True}  # the (stubbed) verifier admitted the peer
    return replies[1:]


def _stamp(
    service: AttestService, op: str, frame: dict[str, Any], authority: Any = _DESKTOP
) -> str:
    """The real stamp seam over a server carrying only the listener (no handle:
    nothing was minted here, so no cap can vouch and only attestation counts)."""
    server = types.SimpleNamespace(_handle=None, _attest=service)
    return _ask_answer_stamp(server, op, frame, authority, ask_id=str(frame["ask_id"]))


# -- positive: respond then revise, through the real dispatch ------------------


@pytest.mark.asyncio
async def test_respond_and_revise_frames_stamp_identical_answers_desktop(
    home: Any, short_root: Path  # noqa: F811
) -> None:
    rig = _Rig(*home, tag="CR")
    service = await _listen(short_root, rig.session_id)
    rig.server._attest = service
    try:
        attest = _ui_frame(rig.session_id, rig.ask_id, "respond", {"q1": ["No, keep it"]})
        assert await _attest(service, attest) == [{"ok": True}]
        reply = await rig.dispatch(
            "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["No, keep it"]}}, _DESKTOP
        )
        assert reply.surface == "desktop"
        assert rig.record()["answered_by"] == {"surface": "desktop", "proof": "app-signature"}

        attest = _ui_frame(rig.session_id, rig.ask_id, "revise", {"q1": ["Yes"]})
        assert await _attest(service, attest) == [{"ok": True}]
        reply = await rig.dispatch(
            "ask_revise", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _DESKTOP
        )
        assert reply.surface == "desktop"
        assert rig.record()["answered_by"] == {"surface": "desktop", "proof": "app-signature"}
        # The attested revision is consent the guard honours without a judge.
        outcome, judge_calls = await rig.guard()
        assert outcome is not None and outcome.blocked is False and judge_calls == 0

        # Replay: the same revise frame again, with no fresh attestation.
        reply = await rig.dispatch(
            "ask_revise", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _DESKTOP
        )
        assert reply.surface == "unverified"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_an_image_answer_binds_the_digest_of_the_decoded_bytes(
    short_root: Path,  # noqa: F811
) -> None:
    service = await _listen(short_root, "sidimg")
    try:
        await _attest(
            service,
            _ui_frame("sidimg", "a-1", "respond", {"q1": ["see screenshot"]}, [_ui_image(_PNG)]),
        )
        frame = {
            "ask_id": "a-1",
            "answers": {"q1": ["see screenshot"]},
            "images": [_wire_image(_PNG)],
        }
        assert _stamp(service, "ask_respond", frame) == "desktop"
        # A digest over the base64 TEXT (the easy mistake) attests nothing.
        text_digest = hashlib.sha256(base64.b64encode(_PNG)).hexdigest()
        wrong = {**_ui_image(_PNG), "sha256": text_digest}
        await _attest(
            service, _ui_frame("sidimg", "a-1", "respond", {"q1": ["see screenshot"]}, [wrong])
        )
        assert _stamp(service, "ask_respond", frame) == "unverified"
    finally:
        await service.close()


# -- negatives ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sent",
    [
        {"answers": {"q1": ["Yes"]}, "images": [_wire_image(_PNG)]},  # changed answer
        {"answers": {"q1": ["No, keep it"]}, "images": [_wire_image(_PNG + b"!")]},  # changed image
        {"answers": {"q1": ["No, keep it"]}, "images": []},  # image dropped
    ],
    ids=["changed-answer", "changed-image", "dropped-image"],
)
async def test_content_mismatch_is_unverified_and_burns_nothing(
    short_root: Path, sent: dict[str, Any]  # noqa: F811
) -> None:
    service = await _listen(short_root, "sidmis")
    try:
        await _attest(
            service,
            _ui_frame("sidmis", "a-1", "respond", {"q1": ["No, keep it"]}, [_ui_image(_PNG)]),
        )
        assert _stamp(service, "ask_respond", {"ask_id": "a-1", **sent}) == "unverified"
        exact = {"ask_id": "a-1", "answers": {"q1": ["No, keep it"]}, "images": [_wire_image(_PNG)]}
        assert _stamp(service, "ask_respond", exact) == "desktop"
        assert _stamp(service, "ask_respond", exact) == "unverified"  # replay
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_another_sessions_frame_is_refused_and_stamps_nothing(
    short_root: Path,  # noqa: F811
) -> None:
    service = await _listen(short_root, "sidown")
    try:
        replies = await _attest(service, _ui_frame("sidother", "a-1", "respond", {"q1": ["Yes"]}))
        assert replies == [{"ok": False, "reason": "wrong-session"}]
        frame = {"ask_id": "a-1", "answers": {"q1": ["Yes"]}}
        assert _stamp(service, "ask_respond", frame) == "unverified"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_a_relayed_frame_is_remote_and_leaves_the_attestation(
    short_root: Path,  # noqa: F811
) -> None:
    service = await _listen(short_root, "sidrem")
    try:
        await _attest(service, _ui_frame("sidrem", "a-1", "respond", {"q1": ["Yes"]}))
        frame = {"ask_id": "a-1", "answers": {"q1": ["Yes"]}}
        assert _stamp(service, "ask_respond", frame, _RELAYED) == "remote"
        # Not consumed by the remote frame: the desktop's own frame still lands.
        assert _stamp(service, "ask_respond", frame) == "desktop"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_an_expired_attestation_is_unverified(
    short_root: Path,  # noqa: F811
) -> None:
    clock = _Clock()
    service = await _listen(short_root, "sidttl", table=AttestationTable(clock=clock))
    try:
        await _attest(service, _ui_frame("sidttl", "a-1", "respond", {"q1": ["Yes"]}))
        clock.now = app_attest.ATTESTATION_TTL_S + 1
        assert _stamp(service, "ask_respond", {"ask_id": "a-1", "answers": {"q1": ["Yes"]}}) == (
            "unverified"
        )
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attested",
    [
        {"act": "decline"},
        {"act": "dismiss", "reason": "moot"},
    ],
    ids=["decline", "dismiss"],
)
async def test_a_decline_or_dismiss_attestation_never_grants_consent(
    home: Any, short_root: Path, attested: dict[str, Any]  # noqa: F811
) -> None:
    """An attested "no" (or "never mind") must not be spendable as a "yes"."""
    rig = _Rig(*home, tag=f"CN{attested['act']}")
    service = await _listen(short_root, rig.session_id)
    rig.server._attest = service
    try:
        act, reason = attested["act"], attested.get("reason")
        frame = _ui_frame(rig.session_id, rig.ask_id, act, reason=reason)
        assert await _attest(service, frame) == [{"ok": True}]
        # Act-bound: the attestation cannot stamp an answer to the delete.
        reply = await rig.dispatch(
            "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _DESKTOP
        )
        assert reply.surface == "unverified"
        outcome, _ = await rig.guard()
        assert outcome is not None and outcome.blocked is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_an_attested_decline_is_desktop_but_still_not_consent(
    home: Any, short_root: Path  # noqa: F811
) -> None:
    rig = _Rig(*home, tag="CNdd")
    service = await _listen(short_root, rig.session_id)
    rig.server._attest = service
    try:
        await _attest(service, _ui_frame(rig.session_id, rig.ask_id, "decline"))
        reply = await rig.dispatch("ask_decline", {"ask_id": rig.ask_id}, _DESKTOP)
        # The operator really declined, on the desktop: recorded as such ...
        assert reply.surface == "desktop"
        assert surfaces.consent_bearing(reply.surface) is True
        # ... and a decline is still not a yes to the delete.
        outcome, _ = await rig.guard()
        assert outcome is not None and outcome.blocked is True
    finally:
        await service.close()

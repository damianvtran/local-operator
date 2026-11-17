"""ADR 0009 Decision 1 on the REAL socket: origin comes from the connection.

The runtime decides a prompt's ``input_surface`` from auth-time facts of the
connection it arrived on (``server._input_surface_for``), never from an op
frame, and hands it to ``h.prompt``/``h.steer``. These tests dial a real
``RuntimeServer`` with each connection shape the vocabulary names, including
the two the phone uses through the production clients (``AttachClient`` with
``origin="mobile"`` is the daemon's viewer, cold-answer and cold-wake dial),
and the mesh relay's ``dial_owner``, which must never carry a ``mobile`` claim.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.server import RuntimeServer, _input_surface_for
from tests.unit.session.runtime.test_server import FakeHandle, _until, _wait_record


class SurfaceHandle(FakeHandle):
    """``FakeHandle`` whose prompt/steer take the ADR 0009 keyword."""

    def __init__(self) -> None:
        super().__init__()
        self.surfaces: list[tuple[str, str | None]] = []
        self.answer_notes: list[tuple[str | None, str]] = []

    async def prompt(
        self, text, images=None, command_id=None, input_surface=None
    ):  # noqa: ANN001, ANN202
        self.surfaces.append(("prompt", input_surface))
        return "prompt admitted"

    async def steer(self, text, images=None, input_surface=None):  # noqa: ANN001, ANN202
        self.surfaces.append(("steer", input_surface))
        return "steering queued"

    def note_answer_surface(self, ask_id, surface):  # noqa: ANN001, ANN202
        self.answer_notes.append((ask_id, surface))

    async def ask_respond(self, ask_id, answers, *, by=""):  # noqa: ANN001, ANN202
        return "answered"


async def _prompt_over(record: Any, auth_extra: dict[str, Any], op: str = "prompt") -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    try:
        writer.write(json.dumps({"key": record.control_key, **auth_extra}).encode() + b"\n")
        await writer.drain()
        welcome = await asyncio.wait_for(reader.readline(), timeout=5)
        assert json.loads(welcome)["op"] == "projection"
        writer.write(json.dumps({"op": op, "req": 1, "text": "hi"}).encode() + b"\n")
        await writer.drain()
        await _until(reader, "ack", 1)
    finally:
        writer.close()


@pytest.mark.parametrize(
    ("auth", "expected"),
    [
        # The phone's daemon control socket, after 1b: it declares the tag.
        ({"locality": "remote", "origin": "mobile"}, "mobile"),
        # A mobile daemon from before the tag still reads as the phone.
        ({"locality": "remote"}, "mobile"),
        # The desktop app's attach.
        ({"client": "attach", "surface": "desktop"}, "desktop"),
        # A local terminal attach (the TUI viewer).
        ({"client": "attach"}, "tui"),
        # A mesh peer's relayed attach.
        ({"client": "attach", "locality": "remote"}, "remote"),
        # ``lop send``/``lop stop``/any CLI carrier: daemon-kind, local.
        ({}, "unknown"),
        # Only the exact value counts: a local attach claiming anything else
        # (or a mis-cased ``mobile``) is the terminal it is.
        ({"client": "attach", "origin": "desktop"}, "tui"),
        ({"client": "attach", "origin": "Mobile"}, "tui"),
    ],
)
@pytest.mark.asyncio
async def test_each_connection_shape_stamps_its_surface(
    auth: dict[str, Any], expected: str
) -> None:
    handle = SurfaceHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        await _prompt_over(record, auth)
        await _prompt_over(record, auth, op="steer")
        assert handle.surfaces == [("prompt", expected), ("steer", expected)]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_phone_attach_client_stamps_mobile() -> None:
    """The daemon's viewer, cold-answer and cold-wake dials are all this client
    with ``origin="mobile"`` (``engage_session_client``, ``launch._deliver``)."""
    from local_operator.mobile.attach_client import AttachClient

    handle = SurfaceHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    client = AttachClient(lambda _p: None, lambda _r: None, locality="remote", origin="mobile")
    try:
        record = await _wait_record()
        await client.connect(record, record.session_id)
        await client.prompt("from the phone")
        assert handle.surfaces == [("prompt", "mobile")]
    finally:
        client.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_local_attach_client_without_the_tag_stamps_tui() -> None:
    """The negative control: the same client, untagged, is a terminal (the
    cold-wake path before 1b was indistinguishable from this)."""
    from local_operator.mobile.attach_client import AttachClient

    handle = SurfaceHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    client = AttachClient(lambda _p: None, lambda _r: None)
    try:
        record = await _wait_record()
        await client.connect(record, record.session_id)
        await client.prompt("from a terminal")
        assert handle.surfaces == [("prompt", "tui")]
    finally:
        client.close()
        runtime.close()


@pytest.mark.asyncio
async def test_the_mesh_relay_never_carries_a_mobile_claim() -> None:
    """``dial_owner`` copies only ``AUTH_FIELDS``, and ``origin`` is not one, so a
    peer that asks for ``origin: mobile`` lands as ``remote`` (ADR 0009 R2-2)."""
    from local_operator.network import dial

    assert "origin" not in dial.AUTH_FIELDS
    handle = SurfaceHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()

        def relay() -> None:
            owner, _welcome = dial.dial_owner(
                Path("."),  # unused: ``_record`` names the runtime
                record.session_id,
                auth={"origin": "mobile"},
                _record=record,
            )
            try:
                owner.send({"op": "prompt", "req": 1, "text": "relayed"})
                for _ in range(30):
                    frame = owner.recv(5)
                    if frame is None or (frame.get("op") == "ack" and frame.get("req") == 1):
                        return
            finally:
                owner.close()

        await asyncio.to_thread(relay)
        assert handle.surfaces == [("prompt", "remote")]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_queued_answer_notes_the_answering_surface() -> None:
    handle = SurfaceHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        try:
            writer.write(
                json.dumps(
                    {"key": record.control_key, "locality": "remote", "origin": "mobile"}
                ).encode()
                + b"\n"
            )
            await writer.drain()
            await asyncio.wait_for(reader.readline(), timeout=5)
            writer.write(
                json.dumps(
                    {"op": "ask_respond", "req": 2, "ask_id": "a1", "answers": {"q": ["yes"]}}
                ).encode()
                + b"\n"
            )
            await writer.drain()
            await _until(reader, "ack", 2)
        finally:
            writer.close()
        assert handle.answer_notes == [("a1", "mobile")]
    finally:
        runtime.close()


def test_the_vocabulary_first_match_wins() -> None:
    """The pure table, including the precedence of the declared tag."""
    assert (
        _input_surface_for(
            declared_origin="mobile", kind="attach", locality="local", surface="desktop"
        )
        == "mobile"
    )
    assert (
        _input_surface_for(declared_origin="", kind="daemon", locality="remote", surface="terminal")
        == "mobile"
    )
    assert (
        _input_surface_for(declared_origin="", kind="attach", locality="remote", surface="desktop")
        == "remote"
    )
    assert (
        _input_surface_for(declared_origin="", kind="attach", locality="local", surface="desktop")
        == "desktop"
    )
    assert (
        _input_surface_for(declared_origin="", kind="attach", locality="local", surface="terminal")
        == "tui"
    )
    assert (
        _input_surface_for(declared_origin="", kind="daemon", locality="local", surface="terminal")
        == "unknown"
    )


def test_the_cold_wake_errand_carries_the_origin_to_its_dial() -> None:
    """``PromptErrand.origin`` rides beside ``input_mode``; empty is the default
    every other producer keeps."""
    from local_operator.session.runtime.launch import PromptErrand

    assert PromptErrand(text="x").origin == ""
    assert PromptErrand(text="x", origin="mobile").origin == "mobile"

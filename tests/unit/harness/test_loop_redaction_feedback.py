"""The redaction FEEDBACK LOOP: a mask the model sees must never become source on disk.

**The defect (2026-10-09, sessions ``565245718d90`` and ``e94239b7eed0``).** Files edited
through the harness ended up with the literal redaction marker in place of an ordinary
identifier. The writers were never at fault: ``edit`` and ``write`` put exactly the bytes
they are handed on disk, and the executor is handed the ORIGINAL arguments (history stores
a scrubbed COPY; ``_plan_call`` uses the original). The corruption is a loop:

1. a credential shape misjudges an identifier (``vendor-prefixed-token`` on a one-word
   ``xai_<letters>`` name) and masks it in a ``read``/``grep``/``bash`` result;
2. the model sees the mask there, and in its own replayed earlier calls;
3. a model that copies what it saw emits the MARKER in its ``edit``/``write``;
4. the writer faithfully writes it.

This file drives that loop end to end through the real ``AgentLoop``, the real ``read``,
``edit`` and ``write`` tools and the session's own redaction function, with a MECHANICAL
model (it does exactly what a real one does: copy the symbol it saw). Assertions are on
the BYTES on disk (``read_bytes``), not on a display, and the identifier is assembled from
parts because the display layer masks the literal in agent-visible transcripts and a
literal spelling in this file would be unreadable to the next agent that opens it.

**DISK vs DISPLAY are separate classes on purpose.** :class:`TestDiskBytes` is about what
reaches the file; :class:`TestDisplayStillMasksSecrets` is about what the model, the
history copy and the transcript may show. A fix to one must not be allowed to pass by
weakening the other (the operator's standing rule: real credential values stay scrubbed
from transcripts, logs and display).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext, _scrub_history_arguments
from local_operator.harness.types import (
    AgentEndEvent,
    ChatRequest,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from local_operator.redaction_shapes import REDACTION_MARKER, scrub_secrets
from local_operator.tools.builtin import (
    build_edit_tool,
    build_read_tool,
    build_write_tool,
)

MODEL = ModelSpec(provider="test", model_id="m")
MARKER_BYTES = REDACTION_MARKER.encode()

#: Assembled from parts: ``xai_`` plus nine lowercase letters is the reported shape (13
#: characters). Any ``xai_`` + 8-15 lowercase letters reproduces it; nothing about this
#: word is special, which is the point of the rule being name-agnostic.
IDENT = "xai_" + "availab" + "le"
#: Same class under other prefixes from the vendor tables, so the loop test is not an
#: ``xai`` test.
OTHER_IDENTS = ("hf_" + "ready" + "probe", "tvly_" + "supports" + "streams", "npm_" + "installed")

#: A realistic xAI-shaped key (mixed case + digits). It is REAL-SHAPED, not a real secret.
REAL_KEY = "xai-" + "Ab3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hJ6kM9nP2rS5tU8wX1yA4cD7fG0iL3oQ6"


class _MechanicalModel:
    """A scripted 'model' that reads a file, then writes using ONLY what it SAW.

    ``plan`` receives the text of the read result exactly as the model would see it (after
    the session's redaction function) and returns the next tool call's ``(name, args)``.
    Because it can only copy what is in front of it, a mask in the read result is a mask
    in its call, which is the defect.
    """

    def __init__(self, read_path: Path, plan: Any) -> None:
        self.read_path = read_path
        self.plan = plan
        self.requests: list[ChatRequest] = []
        self.seen_by_model = ""

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        n = len(self.requests)
        if n == 1:
            turn = [
                StreamToolCallDelta(
                    index=0,
                    id="r1",
                    name="read",
                    argument_delta=json.dumps({"path": str(self.read_path)}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        elif n == 2:
            self.seen_by_model = "".join(
                getattr(block, "text", "")
                for message in request.messages
                if message.role == "tool"
                for block in message.content
            )
            name, args = self.plan(self.seen_by_model)
            turn = [
                StreamToolCallDelta(index=0, id="w1", name=name, argument_delta=json.dumps(args)),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        else:
            turn = [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]

        async def gen():
            for event in turn:
                yield event

        return gen()


async def _drive(model: _MechanicalModel, redact: Any = scrub_secrets) -> list[Any]:
    """Run the real loop with the real read/edit/write tools and the real scrub function."""
    context = LoopContext(tools=[build_read_tool(), build_edit_tool(), build_write_tool()])
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=model,
        redact_tool_result=redact,
    )
    events: list[Any] = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        events.append(event)
    return events


def _results(events: list[Any]) -> list[Message]:
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    return [m for m in end.messages if isinstance(m, Message) and m.role == "tool"]


def _result_text(message: Message) -> str:
    return "".join(getattr(block, "text", "") for block in message.content)


def _copy_symbol(seen: str, real: str) -> str:
    """What a model that copies the symbol it saw would write."""
    return real if real in seen else REDACTION_MARKER


# --- DISK: the bytes a tool writes ---------------------------------------------------


class TestDiskBytes:
    """The identifier reaches disk intact whichever writer and whichever hunk shape."""

    @pytest.mark.asyncio
    async def test_the_read_result_shows_the_real_identifier(self, tmp_path: Path) -> None:
        """The root fix, observed where the model looks: ``read`` does not mask a NAME."""
        src = tmp_path / "availability.py"
        src.write_bytes(f"def {IDENT}() -> bool:\n    return False\n".encode())
        model = _MechanicalModel(
            src,
            lambda seen: ("edit", {"path": str(src), "old_text": "False", "new_text": "True"}),
        )
        await _drive(model)
        assert IDENT in model.seen_by_model
        assert REDACTION_MARKER not in model.seen_by_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ident", (IDENT, *OTHER_IDENTS))
    async def test_edit_copies_the_identifier_it_saw(self, tmp_path: Path, ident: str) -> None:
        src = tmp_path / "availability.py"
        src.write_bytes(f"def {ident}() -> bool:\n    return False\n".encode())

        def plan(seen: str):
            sym = _copy_symbol(seen, ident)
            return "edit", {
                "path": str(src),
                "old_text": "return False",
                "new_text": f"return {sym}() is None",
            }

        await _drive(_MechanicalModel(src, plan))
        disk = src.read_bytes()
        assert disk.count(ident.encode()) == 2, disk
        assert disk.count(MARKER_BYTES) == 0, disk
        ast.parse(disk.decode())

    @pytest.mark.asyncio
    async def test_multi_hunk_edit(self, tmp_path: Path) -> None:
        src = tmp_path / "availability.py"
        src.write_bytes(
            f"def {IDENT}() -> bool:\n    return False\n\nFLAG = {IDENT}\nOTHER = 1\n".encode()
        )

        def plan(seen: str):
            sym = _copy_symbol(seen, IDENT)
            return "edit", {
                "path": str(src),
                "edits": [
                    {"old_text": "return False", "new_text": f"return {sym}() is None"},
                    {"old_text": "OTHER = 1", "new_text": f"OTHER = {sym}"},
                ],
            }

        await _drive(_MechanicalModel(src, plan))
        disk = src.read_bytes()
        assert disk.count(MARKER_BYTES) == 0, disk
        assert disk.count(IDENT.encode()) == 4
        ast.parse(disk.decode())

    @pytest.mark.asyncio
    async def test_replace_all_edit(self, tmp_path: Path) -> None:
        src = tmp_path / "availability.py"
        src.write_bytes(f"a = {IDENT}\nb = {IDENT}\nc = {IDENT}\n".encode())

        def plan(seen: str):
            sym = _copy_symbol(seen, IDENT)
            return "edit", {
                "path": str(src),
                "old_text": f"= {sym}",
                "new_text": f"= {sym}()",
                "replace_all": True,
            }

        await _drive(_MechanicalModel(src, plan))
        disk = src.read_bytes()
        assert disk.count(MARKER_BYTES) == 0, disk
        assert disk.count(f"{IDENT}()".encode()) == 3
        ast.parse(disk.decode())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("suffix", "body"),
        (
            (".py", "def {x}() -> bool:\n    return False\n"),
            (".ts", "export const {x} = false;\n"),
            (".json", '{{"{x}": true}}\n'),
            # The image-gen lane's README write (session e94239b7eed0): a prose document.
            (".md", "# Probe\n\nCall `{x}()` before generating.\n"),
        ),
    )
    async def test_write_rewrites_a_file_with_the_identifier(
        self, tmp_path: Path, suffix: str, body: str
    ) -> None:
        """``write`` is the same loop: the model re-emits the whole body it just read."""
        src = tmp_path / f"probe{suffix}"
        src.write_bytes(body.format(x=IDENT).encode())

        def plan(seen: str):
            sym = _copy_symbol(seen, IDENT)
            return "write", {"path": str(src), "content": body.format(x=sym) + "\n"}

        await _drive(_MechanicalModel(src, plan))
        disk = src.read_bytes()
        assert disk.count(IDENT.encode()) == 1, disk
        assert disk.count(MARKER_BYTES) == 0, disk
        if suffix == ".py":
            ast.parse(disk.decode())
        if suffix == ".json":
            json.loads(disk.decode())

    @pytest.mark.asyncio
    async def test_the_executor_receives_the_original_arguments(self, tmp_path: Path) -> None:
        """Finding 1 of the audit: no path hands SCRUBBED arguments to a writer.

        A genuine SECRET in a call is scrubbed from the stored history copy, but the
        executor must still be handed (and write) the original. This is the invariant
        'redaction never alters bytes a tool writes', pinned with a value the shapes
        really do mask so the test cannot pass by the scrub being a no-op.
        """
        src = tmp_path / "seed.txt"
        src.write_text("seed\n")
        target = tmp_path / "out.txt"
        payload = f"key={REAL_KEY}\n"
        assert scrub_secrets(payload) != payload, "the fixture must be a value the shapes mask"

        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": payload})
        )
        events = await _drive(model)
        assert target.read_bytes() == payload.encode(), "a writer was handed scrubbed bytes"
        # ...while the history copy of that very call carries no key.
        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        stored_calls = [
            call
            for m in end.messages
            if isinstance(m, Message)
            for call in (m.tool_calls or [])
            if call.name == "write"
        ]
        assert stored_calls
        assert REAL_KEY not in json.dumps(stored_calls[0].arguments)


# --- the net under the fix: edit/write say so when they put the marker on disk --------


class TestMarkerIntroductionNote:
    """``edit``/``write`` append a non-blocking note when they INTRODUCE the marker."""

    @pytest.mark.asyncio
    async def test_edit_that_introduces_the_marker_is_flagged_but_still_applied(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "a.py"
        src.write_text("x = 1\n")
        model = _MechanicalModel(
            src,
            lambda seen: (
                "edit",
                {"path": str(src), "old_text": "x = 1", "new_text": f"x = {REDACTION_MARKER}"},
            ),
        )
        events = await _drive(model)
        # Never scrubbed, never refused: the bytes are exactly what the model wrote.
        assert src.read_bytes().count(MARKER_BYTES) == 1
        text = _result_text(_results(events)[-1])
        assert "redaction marker" in text
        assert "not source" in text

    @pytest.mark.asyncio
    async def test_write_that_introduces_the_marker_is_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "a.md"
        src.write_text("# t\n")
        target = tmp_path / "b.md"
        model = _MechanicalModel(
            src,
            lambda seen: (
                "write",
                {"path": str(target), "content": f"call {REDACTION_MARKER}() first\n"},
            ),
        )
        events = await _drive(model)
        assert target.read_bytes().count(MARKER_BYTES) == 1
        assert "not source" in _result_text(_results(events)[-1])

    @pytest.mark.asyncio
    async def test_silent_when_the_file_already_held_the_marker(self, tmp_path: Path) -> None:
        """A document ABOUT the marker is legitimate: an edit that leaves it in place is silent."""
        src = tmp_path / "doc.md"
        src.write_text(f"The mask is {REDACTION_MARKER}.\n\nTODO\n")
        model = _MechanicalModel(
            src,
            lambda seen: (
                "edit",
                {"path": str(src), "old_text": "TODO", "new_text": "done"},
            ),
        )
        events = await _drive(model)
        assert src.read_bytes().count(MARKER_BYTES) == 1
        assert "redaction marker" not in _result_text(_results(events)[-1])

    @pytest.mark.asyncio
    async def test_counts_not_membership(self, tmp_path: Path) -> None:
        """A file that had ONE marker and gains a SECOND is flagged for the second."""
        src = tmp_path / "doc.md"
        src.write_text(f"The mask is {REDACTION_MARKER}.\n\nTODO\n")
        model = _MechanicalModel(
            src,
            lambda seen: (
                "edit",
                {"path": str(src), "old_text": "TODO", "new_text": REDACTION_MARKER},
            ),
        )
        events = await _drive(model)
        assert src.read_bytes().count(MARKER_BYTES) == 2
        assert "put 1 redaction marker" in _result_text(_results(events)[-1])

    @pytest.mark.asyncio
    async def test_write_over_a_file_that_already_had_the_marker_is_silent(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "doc.md"
        src.write_text(f"The mask is {REDACTION_MARKER}.\n")
        model = _MechanicalModel(
            src,
            lambda seen: (
                "write",
                {"path": str(src), "content": f"The mask is still {REDACTION_MARKER}.\n"},
            ),
        )
        events = await _drive(model)
        assert "redaction marker" not in _result_text(_results(events)[-1])

    @pytest.mark.asyncio
    async def test_the_note_does_not_leak_into_the_persisted_details(self, tmp_path: Path) -> None:
        src = tmp_path / "a.py"
        src.write_text("x = 1\n")
        model = _MechanicalModel(
            src,
            lambda seen: (
                "edit",
                {"path": str(src), "old_text": "x = 1", "new_text": f"x = {REDACTION_MARKER}"},
            ),
        )
        events = await _drive(model)
        details = (_results(events)[-1].provider_payload or {}).get("details") or {}
        assert details.get("path"), "premise: the file tools' details are persisted"
        assert not any(key.startswith("_") for key in details), details


# --- DISPLAY: real secrets stay scrubbed -----------------------------------------------


class TestDisplayStillMasksSecrets:
    """The operator's standing rule: real credential VALUES never reach transcript/display.

    The fix releases a NAME; it must not weaken any of these. Each assertion uses a value
    the shapes genuinely mask (asserted first, so the test cannot pass vacuously).
    """

    @pytest.mark.asyncio
    async def test_a_real_key_in_a_read_result_is_masked_for_the_model(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "settings.env"
        src.write_text(f"XAI_API_KEY={REAL_KEY}\nFEATURE={IDENT}\n")
        model = _MechanicalModel(
            src, lambda seen: ("edit", {"path": str(src), "old_text": "FEATURE", "new_text": "F"})
        )
        await _drive(model)
        assert REAL_KEY not in model.seen_by_model, "a real key reached the model"
        assert IDENT in model.seen_by_model, "the NAME beside it must stay readable"

    def test_a_real_key_in_a_call_is_scrubbed_from_the_history_copy(self) -> None:
        call = ToolCall(
            id="c1",
            name="bash",
            arguments={"command": f"curl -H 'Authorization: Bearer {REAL_KEY}' https://h/v1"},
        )
        message = Message(role="assistant", content=[], tool_calls=[call])
        stored = _scrub_history_arguments(message, scrub_secrets)
        assert REAL_KEY not in json.dumps(stored.tool_calls[0].arguments)
        # ...and the ORIGINAL object the executor reads is untouched.
        assert REAL_KEY in call.arguments["command"]

    @pytest.mark.parametrize(
        "secret",
        (
            REAL_KEY,
            "sk-" + "proj-" + "Zq8Lm2Vb9Nk4Pz7Rt3Yw6Hc1Dx5Fa0Ge",
            "gsk_" + "k3m9p2x7q4w8z5v1n6b0c2d4f6h8j0",
            "hf_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ",
        ),
    )
    def test_secret_values_are_scrubbed_from_results(self, secret: str) -> None:
        masked = scrub_secrets(f"value: {secret}\n")
        assert secret not in masked
        assert REDACTION_MARKER in masked

    def test_a_dsn_password_is_scrubbed_from_results(self) -> None:
        password = "Sup3r" + "SecretPw"
        masked = scrub_secrets(f"url: postgres://svc:{password}@db.invalid/app\n")
        assert password not in masked
        assert "db.invalid" in masked, "the host stays readable"

    @pytest.mark.asyncio
    async def test_a_registered_value_is_scrubbed_from_results_history_and_transcript(
        self, tmp_path: Path
    ) -> None:
        """A REGISTERED value is scrubbed by value whatever shape it has.

        This is what bounds the accepted residual of the shape rule: a short all-lowercase
        tail is released by SHAPE, but a value the operator stored (or a child fetched) is
        scrubbed by VALUE. Asserted on all three sinks the operator named: the result the
        model reads, the history copy of the call that carried it, and the persisted
        transcript.
        """
        from local_operator.session.transcript import Transcript
        from local_operator.variables import VariableStore

        value = "sk-" + "override"  # an 8-letter lowercase tail: released by the SHAPE rule
        assert scrub_secrets(value) == value, "premise: the shape rule alone does not mask it"
        store = VariableStore(cwd=str(tmp_path))
        assert store.register_redaction(value)

        src = tmp_path / "notes.txt"
        src.write_text(f"token is {value} ok\n")
        target = tmp_path / "out.txt"
        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": f"copy {value}\n"})
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        assert value not in model.seen_by_model, "result shown to the model"
        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        stored = [
            call
            for m in end.messages
            if isinstance(m, Message)
            for call in (m.tool_calls or [])
            if call.name == "write"
        ][0]
        assert value not in json.dumps(stored.arguments), "history copy of the call"
        assert target.read_text() == f"copy {value}\n", "the DISK write is the original bytes"

        # The persisted transcript, MESSAGE CONTENT and CALL ARGUMENTS. The write's
        # ``details.diff`` is a separate channel with its own rule and is deliberately not
        # asserted here: see the PR's "Not addressed" (it is the executor's own record of
        # the bytes it wrote, and the discriminator for this very defect).
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        persisted = json.dumps(
            [
                {k: v for k, v in row.get("payload", {}).items() if k != "provider_payload"}
                for row in rows
            ]
        )
        assert value not in persisted, "persisted transcript (content + call arguments)"

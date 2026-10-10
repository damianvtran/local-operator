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
import re
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
    ToolExecutionEndEvent,
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

#: A plain English word registered as a secret (``PASSWORD=<word>``). Assembled from parts for
#: the same reason as ``IDENT``. The exact-value pass matches it on LETTER-RUN BOUNDARIES, so
#: ordinary words that merely contain it (``un<word>ally``) stay readable; the standalone
#: token stays masked.
PLAIN_WORD = "synth" + "etic"
EMBEDDED_WORD = "un" + PLAIN_WORD + "ally"


def _store_that_has_scrubbed_the_word(tmp_path: Path) -> Any:
    """A real ``VariableStore`` that REGISTERED the word the way production does.

    Through the shape pass (``PASSWORD=<word>``), never ``register_redaction`` by hand: the
    over-match this class pins only exists on the registration path the store really takes.
    """
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path))
    store.redact_with_report(f"PASSWORD={PLAIN_WORD}")
    assert PLAIN_WORD in store.redaction_values(), "premise: the word is a registered value"
    return store


#: A realistic xAI-shaped key (mixed case + digits). It is REAL-SHAPED, not a real secret.
REAL_KEY = "xai-" + "Ab3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hJ6kM9nP2rS5tU8wX1yA4cD7fG0iL3oQ6"


class _MechanicalModel:
    """A scripted 'model' that reads a file, then writes using ONLY what it SAW.

    ``plan`` receives the text of the read result exactly as the model would see it (after
    the session's redaction function) and returns the next tool call's ``(name, args)``.
    ``args`` is normally a dict, JSON-encoded here into the call's raw argument text; a
    ``str`` is used VERBATIM as that text instead, for the tests that dictate a spelling
    ``json.dumps`` cannot produce (``\u000a``). Because it can only copy what is in front of
    it, a mask in the read result is a mask in its call, which is the defect.
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
            raw = args if isinstance(args, str) else json.dumps(args)
            turn = [
                StreamToolCallDelta(index=0, id="w1", name=name, argument_delta=raw),
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


class TestPlainWordOnDisk:
    """DISK for the plain-word over-match: prose copied from a result is written intact.

    The second half of the loop in the module docstring. The read result masks only the
    STANDALONE registered word, so a model copying the surrounding prose (which contains the
    word inside longer words) copies real text; before the boundary rule it copied the mask
    in place of ``un<word>ally`` and the writer put it on disk. Byte-level, by design.
    """

    @pytest.mark.asyncio
    async def test_edit_copying_prose_with_longer_words_writes_them_intact(
        self, tmp_path: Path
    ) -> None:
        store = _store_that_has_scrubbed_the_word(tmp_path)
        src = tmp_path / "notes.md"
        src.write_bytes(f"a {PLAIN_WORD} store\n{EMBEDDED_WORD} odd\n".encode())

        def plan(seen: str):
            # The mechanical model copies the longer word exactly as it SAW it.
            copied = re.search(r"un\S*?ally", seen)
            assert copied is not None, seen
            return "edit", {
                "path": str(src),
                "old_text": "odd",
                "new_text": f"odd and {copied.group(0)} again",
            }

        model = _MechanicalModel(src, plan)
        await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        # What the model saw: the standalone token masked, the longer word readable.
        assert EMBEDDED_WORD in model.seen_by_model
        assert model.seen_by_model.count(REDACTION_MARKER) == 1
        assert f"a {PLAIN_WORD} store" not in model.seen_by_model
        disk = src.read_bytes()
        assert disk.count(EMBEDDED_WORD.encode()) == 2, disk
        assert disk.count(MARKER_BYTES) == 0, disk

    @pytest.mark.asyncio
    async def test_write_of_a_prose_document_keeps_longer_words_intact(
        self, tmp_path: Path
    ) -> None:
        store = _store_that_has_scrubbed_the_word(tmp_path)
        src = tmp_path / "readme.md"
        src.write_bytes(f"# T\n\n{EMBEDDED_WORD} and {PLAIN_WORD.upper()} and more\n".encode())

        def plan(seen: str):
            copied = re.search(r"un\S*?ally", seen)
            assert copied is not None, seen
            return "write", {"path": str(src), "content": f"# T\n\n{copied.group(0)} again\n"}

        await _drive(
            _MechanicalModel(src, plan), redact=lambda text: store.redact_with_report(text)[0]
        )
        disk = src.read_bytes()
        assert disk == f"# T\n\n{EMBEDDED_WORD} again\n".encode(), disk
        assert MARKER_BYTES not in disk


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

        # The persisted transcript, MESSAGE CONTENT, CALL ARGUMENTS and — since the
        # follow-up fix this test now pins — the result's ``provider_payload``.
        # ``details.diff`` quotes the bytes the write put on disk, so it takes the
        # same mask as the result text; the disk stays the original either way
        # (asserted above) and is the only surface that must.
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        persisted = json.dumps(rows)
        assert value not in persisted, "persisted transcript (content + args + details)"
        assert REDACTION_MARKER in persisted, "premise: the mask replaced it somewhere"


class TestDisplayMasksARegisteredPlainWordAsAToken:
    """DISPLAY for a registered plain word: masked as a standalone token, never inside a word.

    Separate from :class:`TestPlainWordOnDisk` on purpose (see the module docstring): the
    operator's rule that a real value stays scrubbed from results, history and transcript
    must hold at the same time as the prose staying readable, and neither may pass by
    weakening the other.
    """

    @pytest.mark.asyncio
    async def test_results_history_and_transcript(self, tmp_path: Path) -> None:
        from local_operator.session.transcript import Transcript

        store = _store_that_has_scrubbed_the_word(tmp_path)
        src = tmp_path / "notes.txt"
        src.write_text(f"pw is {PLAIN_WORD} ok\nprose: {EMBEDDED_WORD}\n")
        target = tmp_path / "out.txt"
        content = f"copy {PLAIN_WORD} and {EMBEDDED_WORD}\n"
        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": content})
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        # The result the model reads.
        assert f"pw is {REDACTION_MARKER} ok" in model.seen_by_model
        assert f"prose: {EMBEDDED_WORD}" in model.seen_by_model

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        stored = [
            call
            for m in end.messages
            if isinstance(m, Message)
            for call in (m.tool_calls or [])
            if call.name == "write"
        ][0]
        # The history copy of the call: standalone masked, embedded intact.
        assert stored.arguments["content"] == f"copy {REDACTION_MARKER} and {EMBEDDED_WORD}\n"
        # The DISK write is still the original bytes (display never alters a writer's input).
        assert target.read_text() == content

        # The persisted transcript, now including the result's ``provider_payload``:
        # the write's ``details.diff`` quotes the bytes on disk and takes the same
        # mask as the result text (the follow-up fix; the disk is asserted above
        # and stays the original).
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        persisted = json.dumps(rows)
        assert f"{PLAIN_WORD} and" not in persisted, "the standalone word reached the transcript"
        assert f"{REDACTION_MARKER} and {EMBEDDED_WORD}" in persisted, persisted

    def test_a_non_word_registered_value_still_masks_inside_a_longer_token(
        self, tmp_path: Path
    ) -> None:
        """The unchanged half: a value with a digit keeps the substring match everywhere."""
        from local_operator.variables import VariableStore

        value = "Zq8" + "Lm2Vb9Nk4"
        store = VariableStore(cwd=str(tmp_path))
        assert store.register_redaction(value)
        for text in (f"x{value}y", f"un{value}ally", value):
            masked = store.redact(text)
            assert value not in masked, text
            assert REDACTION_MARKER in masked


# --- raw_arguments: a letter-ending escape is a boundary, not a letter (QA round 1, Q-1) --


#: The escapes a JSON encoder can leave immediately before the token, and how each case is
#: dictated: the five SHORT escapes come out of ``json.dumps`` when the content carries the
#: real character, while the ``\uXXXX`` spellings it never emits for them are dictated as
#: raw argument text. The ``\u`` cases whose last hex digit is a letter (``a``/``b``) are the
#: class that regressed: on the raw string the character before the token was that hex
#: letter, which the letter-run boundary read as a word character.
ESCAPE_CASES = (
    ("\\n", ("content", f"line1\n{PLAIN_WORD} here")),
    ("\\t", ("content", f"line1\t{PLAIN_WORD} here")),
    ("\\r", ("content", f"line1\r{PLAIN_WORD} here")),
    ("\\b", ("content", f"line1\b{PLAIN_WORD} here")),
    ("\\f", ("content", f"line1\f{PLAIN_WORD} here")),
    ("\\u000a", ("raw", "line1\\u000a" + PLAIN_WORD + " here")),
    ("\\u000b", ("raw", "line1\\u000b" + PLAIN_WORD + " here")),
    ("\\u00ab", ("raw", "line1\\u00ab" + PLAIN_WORD + " here")),
)


class TestEscapedWordFormsStayMaskedThroughRawArguments:
    """The raw provider string, its transcript row and BOTH wire replay shapes.

    ``raw_arguments`` is the provider's verbatim JSON argument text. The loop scrubs it
    (``_scrub_history_arguments``), the transcript persists it (when it does not round-trip
    to the scrubbed ``arguments``) and the wire replays it verbatim
    (``_replayable_tool_arguments_json``) or as its parsed object
    (``_replayable_tool_arguments``). Reading an escape's last letter as a word character
    released the token in all of these while the parsed ``arguments`` stayed masked — a
    leak, not prose protection (QA round 1, Q-1).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("escape_label", "case"), ESCAPE_CASES, ids=[case[0] for case in ESCAPE_CASES]
    )
    async def test_the_token_is_masked_in_the_stored_copy_transcript_and_replay(
        self, tmp_path: Path, escape_label: str, case: tuple[str, str]
    ) -> None:
        from local_operator.providers.clients import (
            _replayable_tool_arguments,
            _replayable_tool_arguments_json,
        )
        from local_operator.session.transcript import Transcript

        store = _store_that_has_scrubbed_the_word(tmp_path)
        src = tmp_path / "seed.txt"
        src.write_text("seed\n")
        target = tmp_path / "out.md"
        dictate, fragment = case

        def plan(seen: str):
            if dictate == "content":
                return "write", {"path": str(target), "content": fragment}
            return "write", '{"path": %s, "content": "%s"}' % (
                json.dumps(str(target)),
                fragment,
            )

        events = await _drive(
            _MechanicalModel(src, plan), redact=lambda text: store.redact_with_report(text)[0]
        )

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        stored = [
            call
            for m in end.messages
            if isinstance(m, Message)
            for call in (m.tool_calls or [])
            if call.name == "write"
        ][0]
        assert stored.raw_arguments is not None
        assert PLAIN_WORD not in stored.raw_arguments, escape_label
        assert PLAIN_WORD not in json.dumps(stored.arguments), escape_label
        assert REDACTION_MARKER in stored.raw_arguments, escape_label
        # BOTH wire replay shapes, exactly as the providers build them from the stored call.
        assert PLAIN_WORD not in _replayable_tool_arguments_json(stored), escape_label
        assert PLAIN_WORD not in json.dumps(_replayable_tool_arguments(stored)), escape_label

        # The persisted transcript row, including ``provider_payload``: the diff in
        # ``details`` quotes the raw argument text the content spells, so it takes
        # the same mask as everything else the row stores.
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        persisted = json.dumps(rows)
        assert PLAIN_WORD not in persisted, escape_label
        assert REDACTION_MARKER in persisted, escape_label

    @pytest.mark.asyncio
    async def test_glued_and_backslash_neighbours_keep_their_behaviour(
        self, tmp_path: Path
    ) -> None:
        """Both directions on the raw string: escapes separate, letter runs do not.

        ``my_``- and backslash-glued occurrences stay masked (the escape is a separator,
        not its last letter, and neither is an underscore); a word embedded in a longer
        letter run stays readable — the accepted residual, on this surface too.
        """
        store = _store_that_has_scrubbed_the_word(tmp_path)
        src = tmp_path / "seed.txt"
        src.write_text("seed\n")
        target = tmp_path / "out.md"
        content = f"my_{PLAIN_WORD}\n\\{PLAIN_WORD}\n{EMBEDDED_WORD}\n"

        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": content})
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        stored = [
            call
            for m in end.messages
            if isinstance(m, Message)
            for call in (m.tool_calls or [])
            if call.name == "write"
        ][0]
        raw = stored.raw_arguments
        assert raw is not None
        assert json.loads(raw)["content"] == (
            f"my_{REDACTION_MARKER}\n\\{REDACTION_MARKER}\n{EMBEDDED_WORD}\n"
        )


# --- DETAILS: written content is quoted in provider_payload, so it takes the same mask ----


#: The key this section writes through the tools. Assembled from parts like every
#: credential in this file, and REGISTERED BY VALUE so the mask is deterministic
#: rather than at the mercy of a shape table.
DETAILS_KEY = "sk-" + "followup" + "Qz9Xk2Lm5Np8" + "Rt4Yw7"


def _store_that_scrubs_details_key(tmp_path: Path) -> Any:
    """A real ``VariableStore`` with the key registered the way production does."""
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path))
    assert store.register_redaction(DETAILS_KEY)
    assert DETAILS_KEY not in store.redact(DETAILS_KEY), "premise: the value scrub is live"
    return store


class TestWrittenContentInDetailsTakesTheResultMask:
    """``details['diff']`` quotes the bytes a write/edit put on disk, so it is masked.

    ``Message.tool_result`` lifts ``ToolResult.details`` into
    ``provider_payload['details']`` — persisted in the transcript row and read by the
    settled tool card, the mobile projection and every replay. The result TEXT was
    masked at the loop's redaction seam and the diff was not, so a real key written
    through ``write`` stayed quoted there (PR #2140 follow-up, Aida-directed). These
    tests pin the new rule from both sides: the STORED copy carries the mask, and the
    WRITE ITSELF is untouched — the disk bytes are the caller's exactly, and the
    executor's own result (what the live event carries) still describes them.
    """

    @pytest.mark.asyncio
    async def test_a_key_written_through_write_is_masked_in_details_and_exact_on_disk(
        self, tmp_path: Path
    ) -> None:
        from local_operator.session.transcript import Transcript

        store = _store_that_scrubs_details_key(tmp_path)
        src = tmp_path / "seed.txt"
        src.write_text("seed\n")
        target = tmp_path / "out.txt"
        content = f"token = {DETAILS_KEY}\n"
        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": content})
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        row = _results(events)[-1]
        details = (row.provider_payload or {}).get("details") or {}
        diff = details.get("diff") or []
        assert diff, "premise: the write produced a diff"
        joined = "\n".join(str(line) for line in diff)
        assert DETAILS_KEY not in joined, "the written key is quoted in details.diff"
        assert REDACTION_MARKER in joined, "premise: the scrub replaced it, not dropped it"
        # Counters stay numbers for the UI's +N/-N badge.
        assert details.get("added") == 1 and details.get("removed") == 0, details
        # The result TEXT and the history copy of the call were already masked.
        assert DETAILS_KEY not in _result_text(row)
        assert DETAILS_KEY not in json.dumps(row.provider_payload)
        # ...and the ENFORCER of the rule, both ways: the disk holds the caller's
        # bytes exactly (which is also the proof the executor received the original
        # argument — the receipt reports the ORIGINAL length), while the executor's
        # own result, the one the live event carries, still describes those bytes.
        assert target.read_bytes() == content.encode()
        assert f"({len(content)} chars)" in _result_text(row)
        executor = [event for event in events if isinstance(event, ToolExecutionEndEvent)][-1]
        # ``or {}``/``or []`` so the Optional ``details`` is never subscripted
        # directly (pyright reportOptionalSubscript; CI type-check).
        executor_diff = (executor.result.details or {}).get("diff") or []
        assert DETAILS_KEY in "\n".join(
            executor_diff
        ), "the executor's own record must describe the bytes it wrote"

        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        persisted = json.dumps(rows)
        assert DETAILS_KEY not in persisted, "the key reached the persisted transcript"
        assert REDACTION_MARKER in persisted, "premise: the mask is persisted"

    @pytest.mark.asyncio
    async def test_a_key_edited_into_a_file_is_masked_in_details_and_exact_on_disk(
        self, tmp_path: Path
    ) -> None:
        from local_operator.session.transcript import Transcript

        store = _store_that_scrubs_details_key(tmp_path)
        target = tmp_path / "config.env"
        target.write_text("FEATURE = 1\n")
        edited = f"FEATURE = 1\ntoken = {DETAILS_KEY}\n"
        model = _MechanicalModel(
            target,
            lambda seen: (
                "edit",
                {"path": str(target), "old_text": "FEATURE = 1\n", "new_text": edited},
            ),
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        row = _results(events)[-1]
        details = (row.provider_payload or {}).get("details") or {}
        joined = "\n".join(str(line) for line in details.get("diff") or [])
        assert DETAILS_KEY not in joined, "the edited-in key is quoted in details.diff"
        assert REDACTION_MARKER in joined, "premise: the scrub replaced it, not dropped it"
        assert details.get("added") == 1 and details.get("removed") == 0, details
        # The edit landed the caller's bytes exactly, mask nowhere on disk.
        assert target.read_bytes() == edited.encode()

        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "transcript.jsonl").read_text().splitlines()
        ]
        assert DETAILS_KEY not in json.dumps(rows)

    @pytest.mark.asyncio
    async def test_a_write_with_nothing_to_mask_keeps_the_diff_verbatim(
        self, tmp_path: Path
    ) -> None:
        """The negative arm: the seam must not rewrite what it has no reason to."""
        from local_operator.tools.builtin import _diff_details
        from local_operator.variables import VariableStore

        store = VariableStore(cwd=str(tmp_path))
        assert store.register_redaction("unrelated" + "-value-0001")
        src = tmp_path / "seed.txt"
        src.write_text("seed\n")
        target = tmp_path / "plain.txt"
        content = "alpha\nbeta\n"
        model = _MechanicalModel(
            src, lambda seen: ("write", {"path": str(target), "content": content})
        )
        events = await _drive(model, redact=lambda text: store.redact_with_report(text)[0])

        details = (_results(events)[-1].provider_payload or {}).get("details") or {}
        # Byte-for-byte the difflib output — structure, counts and lines all intact.
        assert details == _diff_details(str(target), "", content), details

    @pytest.mark.asyncio
    async def test_a_write_dispatched_through_the_eval_bridge_carries_masked_details(
        self, tmp_path: Path
    ) -> None:
        """The nested bridge: eval-driven calls cross into arbitrary Python WITH details.

        ``dispatch_tool`` is the one seam every eval-driven call crosses, its contract
        is "the same text policy used for native history", and the nested result is
        handed back as a JSON dump that INCLUDES ``details`` — so the diff of a write
        driven from an eval cell must arrive masked, exactly like the stored message's
        copy, while the write on disk stays the caller's bytes.
        """
        from local_operator.harness.types import AgentTool, TextContent, ToolResult

        store = _store_that_scrubs_details_key(tmp_path)
        target = tmp_path / "from_eval.txt"
        content = f"token = {DETAILS_KEY}\n"
        captured: dict[str, Any] = {}

        async def execute(
            tool_call_id: str, args: dict[str, Any], signal: Any, on_update: Any, context: Any
        ) -> ToolResult:
            captured["nested"] = await context.dispatch_tool(
                "write", {"path": str(target), "content": content}
            )
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="eval",
                content=[TextContent(text="nested dispatch done")],
            )

        requests: list[ChatRequest] = []

        def stream_fn(request: ChatRequest, signal: Any):
            requests.append(request)
            turn: list[Any] = (
                [
                    StreamToolCallDelta(index=0, id="e1", name="eval", argument_delta="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ]
                if len(requests) == 1
                else [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
            )

            async def gen():
                for event in turn:
                    yield event

            return gen()

        context = LoopContext(
            tools=[
                AgentTool(
                    name="eval",
                    parameters={"type": "object", "properties": {}},
                    execute=execute,
                ),
                build_write_tool(),
            ]
        )
        config = LoopConfig(
            model=MODEL,
            convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
            stream_fn=stream_fn,
            redact_tool_result=lambda text: store.redact_with_report(text)[0],
        )
        async for _event in AgentLoop().run([Message.user("go")], context, config, None):
            pass

        nested = captured["nested"]
        details = nested.get("details") or {}
        joined = "\n".join(str(line) for line in details.get("diff") or [])
        assert DETAILS_KEY not in json.dumps(nested), "the key crossed back into Python"
        assert REDACTION_MARKER in joined, "premise: the scrub replaced it, not dropped it"
        assert target.read_bytes() == content.encode(), "the write itself stays byte-exact"

    @pytest.mark.asyncio
    async def test_a_mixed_sequence_masks_only_its_strings(self, tmp_path: Path) -> None:
        """A non-``str`` item rides back untouched — and must not shift the mask.

        Review round 1, R-2: the rebuild indexed ``masked`` by the item's position
        in the sequence while the collection phase banked only the strings, so
        ``{"diff": ["a", 1, "b"]}`` read one past the mask list and raised
        ``IndexError`` inside ``_append_results``. Unreachable through first-party
        writers today (``_diff_details`` always emits ``list[str]``), but the seam
        runs for every tool result, and a buggy producer must degrade to "that
        item is not text", not a crash. Both container spellings are pinned.
        """
        store = _store_that_scrubs_details_key(tmp_path)

        def redact(text: str) -> str:
            return store.redact_with_report(text)[0]

        out = await AgentLoop._redact_details(
            {"diff": [f"a {DETAILS_KEY}", 7, f"b {DETAILS_KEY}"]}, redact, "write", None
        )
        assert out is not None, "a details mapping in, a details mapping back"
        assert out["diff"][1] == 7
        assert out["diff"][0] == f"a {REDACTION_MARKER}"
        assert out["diff"][2] == f"b {REDACTION_MARKER}"
        out = await AgentLoop._redact_details(
            {"diff": (f"a {DETAILS_KEY}", None, f"b {DETAILS_KEY}")}, redact, "write", None
        )
        assert out is not None, "a details mapping in, a details mapping back"
        assert out["diff"][1] is None
        assert out["diff"][0] == f"a {REDACTION_MARKER}"
        assert out["diff"][2] == f"b {REDACTION_MARKER}"


# --- RAW ARGUMENTS: the JSON string is judged with the same line boundaries ----------------
#
# QA on PR #2133 (round 1 Q-1, round 2 Q-7): a credential right after a line break inside a
# tool call's ``raw_arguments`` survived into the history copy -- replayed on the NEXT provider
# request and persisted to ``transcript.jsonl`` -- while the same value in the decoded
# ``arguments`` was masked. In the JSON text a line break is the two characters backslash-n,
# which defeats the shape rules' word-boundary and line-start anchors. Everything below is
# synthetic: every credential-shaped value is assembled from parts at run time and is only
# ever asserted on as a boolean, never printed.

#: Shaped like a vendor key and a bearer credential (mixed case + digits); not real secrets.
_BEARER_TOKEN = "Zq8Lm2V" + "b9Nk4Pz7Rt3Yw6Hc1Dx5Fa0GeQw"
_GH_TOKEN = "ghp_" + "Zq8Lm2Vb9Nk4Pz7Rt3Yw6Hc1Dx5Fa0GeQwXy"
_CREDENTIALS = (REAL_KEY, _BEARER_TOKEN, _GH_TOKEN)

#: ``(id, tool, arguments)``. Every credential sits at the start of a line (or after a tab),
#: which is the position the escaped form of the argument cannot present to a line rule.
RAW_CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "bash-heredoc-bearer",
        "bash",
        {"command": f"cat <<'EOF' | curl -d @- https://h.invalid\nBearer {_BEARER_TOKEN}\nEOF"},
    ),
    (
        "write-content-bare-key",
        "write",
        {"path": "/w/notes.txt", "content": f"# notes\n{REAL_KEY}\nend\n"},
    ),
    (
        "env-block",
        "bash",
        {
            # Unassigned line FIRST: in the escaped text an assigned ``NAME=`` value runs on
            # across the backslash-n (``[^\\s]`` has no line to stop at) and would mask a
            # token that follows it, hiding the leak this case is for.
            "command": f"cat > .env <<'EOF'\nREGION=eu\n{_GH_TOKEN}\nXAI_API_KEY={REAL_KEY}\nEOF"
        },
    ),
    ("bearer-after-newline", "bash", {"command": f"echo hi\nBearer {_BEARER_TOKEN}"}),
    ("key-after-crlf", "write", {"path": "/w/a", "content": f"a\r\n{REAL_KEY}\r\nb"}),
    ("key-after-tab", "write", {"path": "/w/a", "content": f"a\t{REAL_KEY}"}),
    (
        "multiple-credentials-and-non-ascii",
        "bash",
        {"command": f"echo caf\u00e9\n{REAL_KEY}\n\tBearer {_BEARER_TOKEN}\n{_GH_TOKEN}\nok"},
    ),
    # QA round 2, Q-1: an assignment line directly before the Bearer line. The text pass
    # alone mangles the assignment's value AND consumes the escape before the Bearer line,
    # so the word-boundary rule cannot anchor the token at all -- these two are red under
    # the text-pass-first order and green only for the order ``_scrub_raw_arguments``
    # documents (value pass first, text pass last).
    ("assignment-then-bearer", "bash", {"command": f"TOKEN=abc123\nBearer {_BEARER_TOKEN}"}),
    ("api-key-then-bearer", "bash", {"command": f"API_KEY=abc123\nBearer {_BEARER_TOKEN}"}),
)


def _leaks(text: str) -> bool:
    """True when any synthetic credential survives in ``text`` (a boolean, never a display)."""
    return any(secret in text for secret in _CREDENTIALS)


def _session_redact(tmp_path: Path) -> Any:
    """The session's own hook: ``VariableStore.redact_with_report`` (what ``Session`` installs)."""
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path))
    return lambda text: store.redact_with_report(text)[0]


def _assistant(name: str, arguments: dict[str, Any], raw: str | None) -> Message:
    return Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(id="c1", name=name, arguments=arguments, raw_arguments=raw)],
    )


class _OneCallModel:
    """Emit ONE tool call whose argument bytes are exactly ``raw``, then record the next request.

    ``raw`` is streamed verbatim (in two deltas, as a provider would), so it need not be valid
    JSON: that is how a truncated fragment reaches the loop.
    """

    def __init__(self, name: str, raw: str) -> None:
        self.name = name
        self.raw = raw
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        if len(self.requests) == 1:
            half = len(self.raw) // 2
            turn: list[Any] = [
                StreamToolCallDelta(
                    index=0, id="c1", name=self.name, argument_delta=self.raw[:half]
                ),
                StreamToolCallDelta(index=0, argument_delta=self.raw[half:]),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        else:
            turn = [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]

        async def gen():
            for event in turn:
                yield event

        return gen()


async def _drive_raw(
    name: str, raw: str, redact: Any
) -> tuple[_OneCallModel, list[dict[str, Any]], AgentEndEvent]:
    """Run the real loop against a recording tool; return the model, executor args, end event."""
    from local_operator.harness.types import AgentTool, ToolResult

    seen_by_executor: list[dict[str, Any]] = []

    async def execute(call_id, args, signal, update, context):
        seen_by_executor.append(json.loads(json.dumps(args)))
        return ToolResult(tool_call_id=call_id, tool_name=name)

    tool = AgentTool(name=name, description=name, parameters={"type": "object"}, execute=execute)
    model = _OneCallModel(name, raw)
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=model,
        redact_tool_result=redact,
    )
    end: AgentEndEvent | None = None
    async for event in AgentLoop().run(
        [Message.user("go")], LoopContext(tools=[tool]), config, None
    ):
        if isinstance(event, AgentEndEvent):
            end = event
    assert end is not None
    return model, seen_by_executor, end


def _calls_named(messages: list[Any], name: str) -> list[ToolCall]:
    return [
        call
        for m in messages
        if isinstance(m, Message) and m.role == "assistant"
        for call in m.tool_calls
        if call.name == name
    ]


def _wire_bodies(messages: list[Message]) -> list[str]:
    """The next request as each wire family serialises it (Anthropic and OpenAI-compatible)."""
    from local_operator.providers.clients import AnthropicClient, OpenAICompatClient

    bodies = []
    for provider, build in (
        ("anthropic", lambda r: AnthropicClient()._build_body(r)),
        ("openai", lambda r: OpenAICompatClient("https://x.invalid")._build_body(r)),
    ):
        request = ChatRequest(
            model=ModelSpec(provider=provider, model_id="m"), messages=list(messages)
        )
        bodies.append(json.dumps(build(request)))
    return bodies


class TestRawArgumentsEscapedBoundaries:
    """A credential after a line break in ``raw_arguments`` is masked in every stored copy."""

    @pytest.mark.parametrize("case", RAW_CASES, ids=[c[0] for c in RAW_CASES])
    def test_premise_the_escaped_text_defeats_the_line_boundary(
        self, case: tuple[str, str, dict[str, Any]]
    ) -> None:
        """Pins WHY the fix exists, so the tests below cannot pass vacuously.

        The decoded value is masked; the same value judged as JSON text is not. If the shape
        rules ever learn to read escapes this premise goes red and the derivation can be
        revisited deliberately.
        """
        _, _, arguments = case
        assert not _leaks(
            json.dumps(
                _scrub_history_arguments(_assistant("bash", arguments, None), scrub_secrets)
                .tool_calls[0]
                .arguments
            )
        ), "decoded view is masked"
        assert _leaks(scrub_secrets(json.dumps(arguments))), "the escaped text view leaks"

    @pytest.mark.parametrize("hook", ("shape", "session"))
    @pytest.mark.parametrize("case", RAW_CASES, ids=[c[0] for c in RAW_CASES])
    def test_the_stored_copy_carries_no_credential(
        self, case: tuple[str, str, dict[str, Any]], hook: str, tmp_path: Path
    ) -> None:
        _, name, arguments = case
        redact = scrub_secrets if hook == "shape" else _session_redact(tmp_path)
        original = _assistant(name, arguments, json.dumps(arguments))
        stored = _scrub_history_arguments(original, redact).tool_calls[0]
        assert not _leaks(stored.raw_arguments or ""), "raw_arguments"
        assert not _leaks(json.dumps(stored.arguments)), "arguments"
        # Both carriers are clean. They are deliberately NOT required to be equal: the
        # text pass may mask MORE of the rendering than the decoded line (a span can run
        # across the escape, the direction the shape corpus pins as "never less"), and the
        # transcript keeps the raw row verbatim in that case -- which is the point, both
        # spellings are masked.
        assert REDACTION_MARKER in (stored.raw_arguments or "")
        assert REDACTION_MARKER in json.dumps(stored.arguments)
        # ...and the object the executor reads is untouched.
        assert original.tool_calls[0].arguments == arguments
        assert original.tool_calls[0].raw_arguments == json.dumps(arguments)

    @pytest.mark.parametrize("hook", ("shape", "session"))
    @pytest.mark.parametrize("case", RAW_CASES, ids=[c[0] for c in RAW_CASES])
    @pytest.mark.asyncio
    async def test_stored_history_next_request_wire_and_transcript_are_clean(
        self, case: tuple[str, str, dict[str, Any]], hook: str, tmp_path: Path
    ) -> None:
        """The loop end to end: every place the call is stored, replayed or persisted.

        Both hooks, because they fail differently: the shape function is the stateless
        surface every embedder gets, and the session store is the one a real ``Session``
        installs (whose containment registration can mask some rows by a side door -- the
        escaped-boundary defect must be closed for both, not only where a hit happened to
        be registered earlier).
        """
        from local_operator.session.transcript import Transcript

        _, name, arguments = case
        raw = json.dumps(arguments)
        redact = scrub_secrets if hook == "shape" else _session_redact(tmp_path)
        model, executor_args, end = await _drive_raw(name, raw, redact)

        # The executor was handed the ORIGINAL arguments (the credential intact).
        assert executor_args == [arguments]

        # 1. the history copy stored by the loop
        stored = _calls_named(end.messages, name)
        assert len(stored) == 1
        assert not _leaks(stored[0].raw_arguments or ""), "stored raw_arguments"
        assert not _leaks(json.dumps(stored[0].arguments)), "stored arguments"

        # 2. the NEXT provider request, as the loop hands it to the stream function and as
        #    two wire families serialise it
        assert len(model.requests) >= 2
        replayed = _calls_named(list(model.requests[1].messages), name)
        assert len(replayed) == 1
        assert not _leaks(replayed[0].raw_arguments or ""), "replayed raw_arguments"
        assert not _leaks(json.dumps(replayed[0].arguments)), "replayed arguments"
        assert all(not _leaks(body) for body in _wire_bodies(list(model.requests[1].messages)))

        # 3. the persisted transcript row
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        rows = (tmp_path / "session" / "transcript.jsonl").read_text()
        assert not _leaks(rows), "transcript.jsonl"
        assert REDACTION_MARKER in rows

    @pytest.mark.parametrize("hook", ("shape", "session"))
    def test_a_truncated_fragment_is_scrubbed_value_by_value(
        self, hook: str, tmp_path: Path
    ) -> None:
        """A stream cut mid-call leaves unparseable ``raw_arguments`` beside empty arguments.

        It is never replayed (the wire salvages ``arguments``) but it IS persisted and fed to the
        compaction summariser, so it is judged too -- decoded value by value, because the whole
        string is not JSON. Cut inside a string, after a complete value, and mid-escape.
        """
        redact = scrub_secrets if hook == "shape" else _session_redact(tmp_path)
        whole = json.dumps({"command": f"echo hi\n{REAL_KEY}\nBearer {_BEARER_TOKEN}\nmore"})
        cuts = {
            "inside-string": whole[: whole.index("more")],
            "after-value": whole[:-1],
            "mid-escape": whole[: whole.index("more") - 1],
            "two-fields": '{"path": "/w/a", "content": "x\\n' + REAL_KEY + '\\ny", "mode"',
        }
        for label, fragment in cuts.items():
            with pytest.raises(ValueError):
                json.loads(fragment)
            stored = _scrub_history_arguments(_assistant("bash", {}, fragment), redact)
            assert not _leaks(stored.tool_calls[0].raw_arguments or ""), label
            assert (stored.tool_calls[0].raw_arguments or "").startswith("{"), label
        # Non-secret structure survives byte-for-byte: only the masked value changed.
        kept = _scrub_history_arguments(_assistant("bash", {}, cuts["two-fields"]), redact)
        assert (kept.tool_calls[0].raw_arguments or "").startswith(
            '{"path": "/w/a", "content": "x\\n'
        )
        assert (kept.tool_calls[0].raw_arguments or "").endswith('\\ny", "mode"')

    @pytest.mark.parametrize("hook", ("shape", "session"))
    @pytest.mark.asyncio
    async def test_a_stream_cut_after_an_assignment_line_is_clean_everywhere(
        self, hook: str, tmp_path: Path
    ) -> None:
        """QA round 2, Q-1: the assignment+Bearer form under a stream cut mid-call.

        This is the shape QA reproduced the leak through on a real ``Session.prompt``: the
        cut leaves ``arguments`` empty, so no earlier pass ever sees the value and the raw
        fragment is the ONLY carrier -- it must be judged (and masked) on the characters it
        denotes. The 20-character prefix is asserted too: a truncated credential is still a
        credential, and the cut is deliberately inside the token.
        """
        from local_operator.session.transcript import Transcript

        redact = scrub_secrets if hook == "shape" else _session_redact(tmp_path)
        whole = json.dumps({"command": f"TOKEN=abc123\nBearer {_BEARER_TOKEN}\nEOF"})
        raw = whole[: whole.index(_BEARER_TOKEN) + 22]  # cut INSIDE the bearer value
        with pytest.raises(ValueError):
            json.loads(raw)
        model, executor_args, end = await _drive_raw("bash", raw, redact)
        assert executor_args == [], "an unparseable call is never executed"
        stored = _calls_named(end.messages, "bash")
        assert stored, "the call is stored"
        for call in stored:
            assert not _leaks(call.raw_arguments or "")
            assert _BEARER_TOKEN[:20] not in (call.raw_arguments or ""), "a prefix is a credential"
        replayed = _calls_named(list(model.requests[1].messages), "bash")
        assert replayed and all(not _leaks(c.raw_arguments or "") for c in replayed)
        assert all(not _leaks(body) for body in _wire_bodies(list(model.requests[1].messages)))
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        row = (tmp_path / "session" / "transcript.jsonl").read_text()
        assert not _leaks(row)
        assert _BEARER_TOKEN[:20] not in row

    @pytest.mark.asyncio
    async def test_a_truncated_call_through_the_loop_is_clean_everywhere(
        self, tmp_path: Path
    ) -> None:
        from local_operator.session.transcript import Transcript

        raw = json.dumps({"command": f"x\n{REAL_KEY}\ny"})[:-3]
        model, executor_args, end = await _drive_raw("bash", raw, _session_redact(tmp_path))
        assert executor_args == [], "an unparseable call is never executed"
        stored = _calls_named(end.messages, "bash")
        assert stored and all(not _leaks(c.raw_arguments or "") for c in stored)
        replayed = _calls_named(list(model.requests[1].messages), "bash")
        assert replayed and all(not _leaks(c.raw_arguments or "") for c in replayed)
        assert all(not _leaks(body) for body in _wire_bodies(list(model.requests[1].messages)))
        transcript = Transcript(tmp_path / "session")
        for message in end.messages:
            await transcript.append_message(message)
        assert not _leaks((tmp_path / "session" / "transcript.jsonl").read_text())

    def test_raw_that_disagrees_with_arguments_is_still_scrubbed(self) -> None:
        """A duplicate key (last wins in ``json.loads``) or a hand-edited row:
        raw is judged on its own text, not through the parsed object."""
        raw = '{"command": "ok", "command": "a\\n' + REAL_KEY + '\\nb"}'
        call_args = {"command": "ok"}
        stored = _scrub_history_arguments(_assistant("bash", call_args, raw), scrub_secrets)
        assert not _leaks(stored.tool_calls[0].raw_arguments or "")

    def test_nothing_to_mask_keeps_identity_and_byte_fidelity(self) -> None:
        """An ordinary multi-line payload round-trips untouched, whatever its encoding."""
        arguments = {"command": 'set -e\nfor f in *.py; do\n\techo caf\u00e9 "$f"\ndone\r\n'}
        encodings = (
            json.dumps(arguments),
            json.dumps(arguments, ensure_ascii=False),
            json.dumps(arguments, separators=(",", ":")),
            '{ "command" :  ' + json.dumps(arguments["command"]) + "  }",
        )
        for raw in encodings:
            message = _assistant("bash", arguments, raw)
            stored = _scrub_history_arguments(message, scrub_secrets)
            assert stored is message, "identity when nothing changed"
            assert stored.tool_calls[0].raw_arguments == raw, "byte-identical raw"
            assert stored.tool_calls[0].arguments == arguments

    def test_an_unchanged_call_beside_a_changed_one_is_the_same_object(self) -> None:
        clean = ToolCall(id="a", name="bash", arguments={"command": "ls\npwd"}, raw_arguments=None)
        secret_args = {"command": f"x\n{REAL_KEY}"}
        dirty = ToolCall(
            id="b", name="bash", arguments=secret_args, raw_arguments=json.dumps(secret_args)
        )
        message = Message(role="assistant", content=[], tool_calls=[clean, dirty])
        stored = _scrub_history_arguments(message, scrub_secrets)
        assert stored is not message
        assert stored.tool_calls[0] is clean
        assert not _leaks(stored.tool_calls[1].raw_arguments or "")
        assert message.tool_calls[1].arguments == secret_args, "the original is untouched"

    def test_six_character_escapes_are_decoded_for_judgement(self) -> None:
        """``\\u000a`` is a newline too: a model may spell the break that way, and the
        judgement must see the line it denotes (the same decode the fragment path uses)."""
        raw = '{"command": "echo hi\\u000a' + REAL_KEY + '\\u000aend"}'
        stored = _scrub_history_arguments(_assistant("bash", {}, raw), scrub_secrets)
        assert not _leaks(stored.tool_calls[0].raw_arguments or "")
        assert REDACTION_MARKER in (stored.tool_calls[0].raw_arguments or "")

    def test_a_call_with_no_raw_arguments_is_still_scrubbed(self) -> None:
        arguments = {"command": f"x\n{REAL_KEY}"}
        stored = _scrub_history_arguments(_assistant("bash", arguments, None), scrub_secrets)
        assert stored.tool_calls[0].raw_arguments is None
        assert not _leaks(json.dumps(stored.tool_calls[0].arguments))

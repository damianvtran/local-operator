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

    @pytest.mark.parametrize("case", RAW_CASES, ids=[c[0] for c in RAW_CASES])
    @pytest.mark.asyncio
    async def test_stored_history_next_request_wire_and_transcript_are_clean(
        self, case: tuple[str, str, dict[str, Any]], tmp_path: Path
    ) -> None:
        """The loop end to end: every place the call is stored, replayed or persisted."""
        from local_operator.session.transcript import Transcript

        _, name, arguments = case
        raw = json.dumps(arguments)
        model, executor_args, end = await _drive_raw(name, raw, _session_redact(tmp_path))

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

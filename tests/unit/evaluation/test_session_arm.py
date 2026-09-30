"""The session arm: declaration, bridge discipline, the MCP wire, the surface.

These tests are the PR's falsifiable core, and each maps to a claim the design
makes:

* **The action surface is a session-scoped MCP server** (design §6). The wire
  test drives the REAL stack end to end: a real ``McpManager`` spawns the real
  ``local_operator.evaluation.action_server`` child, which forwards over a
  UNIX socket to a live ``ActionBridge``, which executes and renders -- and the
  ToolResult the model would see carries the rendered step text AND the frame
  bytes as an image block.
* **cwd and the MCP config stay inside the episode scratch** (PR 1's QA MUST,
  design §5). ``assert_declaration_resolved`` refuses a cwd or a discovered
  source outside the scratch root, by resolved path.
* **One batch per observation, the turn boundary re-arms** -- the token
  discipline is the loop-driven tool's, imported not re-stated.
* **The episode session carries the shipped tool surface**: opened through
  ``sdk.open_session`` with the action server declared, the session's live
  inventory holds the harness's own tools (``task``/``hub``/``team`` among
  them) alongside the minted action tool.
* **A failed execute never wedges the bridge** (arm 1796's F1): one committed
  read-back failure re-binds through a bounded re-read, or the episode ends
  ``failed`` / ``bridge-wedged`` with its turn stopped -- never a spin to
  ``agent_stop``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator import agent_shell, sdk
from local_operator.compaction.png import encode_grayscale_png
from local_operator.evaluation.action_server import (
    WIRE_READ_LIMIT_BYTES,
    decode_response,
    encode_call,
)
from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ExecuteResult,
    ExecutionReceipt,
    observation_content_id,
)
from local_operator.evaluation.adapters.rpc import RpcErrorDetail, RpcRemoteError
from local_operator.evaluation.evidence.models import ScoreArtifact
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.session_arm import (
    BRIDGE_WEDGED_TERMINAL,
    RECOVERY_AFTER_READBACK_LOSS,
    ActionBridge,
    ObservationRenderer,
    SessionArmError,
    assert_declaration_resolved,
    declare_action_server,
    open_episode_session,
    prose_claims_completion,
    run_session_episode,
    session_tool_names,
    split_prompt_content,
)
from local_operator.harness.types import (
    AudioContent,
    ImageContent,
    Message,
    MessageEndEvent,
    TextContent,
    ToolCall,
    ToolContext,
    TurnEndEvent,
)
from local_operator.mcp.manager import McpManager
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    build_config,
    build_spec,
    selector,
)


def _roots_free() -> SessionRoots:
    """Roots for tests that never touch disk or a session factory."""

    return SessionRoots(
        config_dir=Path("/nonexistent/pr2-config"),
        agent_home=Path("/nonexistent/pr2-home"),
        cwd=Path("/nonexistent/pr2-work"),
        allow_volatile=True,
    )


class _AsyncContext:
    async def __aenter__(self) -> Any:  # pragma: no cover - never entered
        raise AssertionError("closed before entering")

    async def __aexit__(self, *exc: Any) -> None:  # pragma: no cover
        return None


@pytest.fixture
def short_scratch() -> Iterator[Path]:
    """A socket-capable scratch directory short enough for ``sun_path``.

    The action bridge binds a UNIX socket under the episode scratch, and its
    path is bounded (~104 bytes on macOS). Under xdist the ``tmp_path`` layer
    plus a descriptive test name goes over that bound (measured), so the
    run-level tests stage the session scratch here and keep the RECORD under
    ``tmp_path`` -- the same constraint ``test_record_sink``'s fixture and the
    repro harness document.
    """

    path = Path(tempfile.mkdtemp(prefix="lo-arm-t-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


#: A REAL image: the frame reader validates media (`verify_artifact`), so a
#: fixture of arbitrary bytes would be refused by the same check the runner
#: uses -- which is the property the wire test needs to exercise.
PNG = encode_grayscale_png(1, 1, b"\x00")

DIGEST = "0" * 64


def _observation(
    *,
    sequence: int = 0,
    text: str = "screen A",
    frames: tuple[FrameRef, ...] = (),
    episode_id: str = "ep-session-arm",
) -> Observation:
    draft = Observation(
        task_id="task_001",
        episode_id=episode_id,
        sequence=sequence,
        observation_id="draft",
        text=text,
        frames=frames,
    )
    return draft.model_copy(update={"observation_id": observation_content_id(draft)})


def _frame(artifact_root: Path, data: bytes) -> FrameRef:
    digest = hashlib.sha256(data).hexdigest()
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / digest).write_bytes(data)
    size = FrameSize(width=800, height=600)
    return FrameRef(
        frame_id="frame-0",
        artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
        geometry=FrameGeometry(native=size, model_visible=size),
    )


def _result(*, input_observation: Observation, output_observation: Observation) -> ExecuteResult:
    return ExecuteResult(
        receipt=ExecutionReceipt(
            operation_id="op-test",
            action_batch_id=DIGEST,
            input_observation_id=input_observation.observation_id,
            output_observation_id=output_observation.observation_id,
            sequence=output_observation.sequence,
        ),
        observation=output_observation,
    )


def _bridge(
    tmp_path: Path,
    *,
    observation: Observation | None = None,
    execute: Any = None,
    recover: Any = None,
    max_steps: int = 8,
    record: Any = None,
    instruction: str = "the task as stated",
    completion_gate: bool = True,
    completion_challenges: int = 1,
    reply_guidance: str | None = None,
) -> ActionBridge:
    obs0 = observation or _observation()
    obs1 = _observation(sequence=1, text="screen B")

    async def default_execute(batch: Any) -> ExecuteResult:
        del batch
        return _result(input_observation=obs0, output_observation=obs1)

    bridge = ActionBridge(
        endpoint=tmp_path / "action-bridge.sock",
        surface=ActionSurface(),
        render=lambda observation: [TextContent(text=f"seen {observation.sequence}")],
        execute=execute or default_execute,
        recover=recover,
        max_steps=max_steps,
        record=record,
        instruction=instruction,
        completion_gate=completion_gate,
        completion_challenges=completion_challenges,
        reply_guidance=reply_guidance,
    )
    bridge.arm(obs0)
    return bridge


def _observation_phase_error() -> RpcRemoteError:
    """The wire error a worker answers with after a batch COMMITTED and only
    the read-back was lost -- the F1 shape (arm 1796, tasks 001/013)."""

    return RpcRemoteError(
        "adapter_error",
        "adapter operation failed",
        RpcErrorDetail(
            exception_type="ObservationPhaseError",
            message="environment returned no screenshot frame",
            method="execute",
            operation_id="exec-0",
            phase="observation",
        ),
    )


def _deadline_error() -> RpcRemoteError:
    """The ambiguous class: the failure declares nothing about the batch, so
    no re-read is safe."""

    return RpcRemoteError(
        "adapter_error",
        "adapter operation failed",
        RpcErrorDetail(
            exception_type="TimeoutError",
            message="deadline exceeded",
            method="execute",
            operation_id="exec-0",
            phase="unknown",
        ),
    )


class TestDeclaration:
    def test_writes_the_server_entry_and_mints_the_name(self, tmp_path: Path) -> None:
        config = tmp_path / "home" / ".local-operator"
        config.mkdir(parents=True)
        work = tmp_path / "home" / "work"
        work.mkdir()
        decl = declare_action_server(
            config_dir=config,
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(paste_text=True),
            python_executable="/venv/python",
            cwd=work,
        )
        assert decl.tool_name == "mcp__episode_actions_apply_actions"
        document = json.loads((config / "mcp.json").read_text(encoding="utf-8"))
        entry = document["mcpServers"]["episode-actions"]
        assert entry["type"] == "stdio"
        assert entry["command"] == "/venv/python"
        assert entry["args"][:2] == ["-m", "local_operator.evaluation.action_server"]
        assert entry["args"][2] == "--endpoint"
        assert entry["args"][3] == str(tmp_path / "b.sock")
        assert entry["preloadTools"] is True
        assert entry["enabledTools"] == ["apply_actions"]
        assert entry["ownTurnOnly"] is True
        assert entry["cwd"] == str(work)

    def test_merges_an_existing_file(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        config.mkdir()
        (config / "mcp.json").write_text(
            json.dumps({"mcpServers": {"other": {"type": "stdio", "command": "true"}}}),
            encoding="utf-8",
        )
        declare_action_server(
            config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
        )
        document = json.loads((config / "mcp.json").read_text(encoding="utf-8"))
        assert set(document["mcpServers"]) == {"other", "episode-actions"}
        assert document["mcpServers"]["other"]["command"] == "true"

    def test_unreadable_existing_file_is_refused(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        config.mkdir()
        (config / "mcp.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionArmError, match="not readable JSON"):
            declare_action_server(
                config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
            )


class TestResolution:
    """The recorded MUST: the episode's MCP graph resolves inside its scratch."""

    def _declare(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        home = tmp_path / "home"
        config = home / ".local-operator"
        work = home / "work"
        config.mkdir(parents=True)
        work.mkdir()
        declare_action_server(
            config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
        )
        return home, config, work

    def test_sources_inside_scratch_are_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, config, work = self._declare(tmp_path)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        sources = assert_declaration_resolved(cwd=work, config_dir=config, scratch_root=home)
        assert sources["episode-actions"] == str(config / "mcp.json")
        assert all(Path(source).resolve().is_relative_to(home) for source in sources.values())

    def test_a_cwd_outside_scratch_is_refused(self, tmp_path: Path) -> None:
        home, config, work = self._declare(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(SessionArmError, match="cwd"):
            assert_declaration_resolved(cwd=outside, config_dir=config, scratch_root=home)

    def test_a_config_outside_scratch_is_refused(self, tmp_path: Path) -> None:
        home, config, work = self._declare(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(SessionArmError, match="config dir"):
            assert_declaration_resolved(cwd=work, config_dir=outside, scratch_root=home)

    def test_a_dropped_declaration_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, config, work = self._declare(tmp_path)
        (config / "mcp.json").unlink()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        with pytest.raises(SessionArmError, match="was not discovered"):
            assert_declaration_resolved(cwd=work, config_dir=config, scratch_root=home)


class TestBridgeDiscipline:
    @pytest.mark.asyncio
    async def test_one_batch_per_observation_until_the_turn_ends(self, tmp_path: Path) -> None:
        executed: list[Any] = []
        bridge = _bridge(tmp_path, execute=None)
        original = bridge.execute

        async def record(batch: Any) -> Any:
            executed.append(batch)
            return await original(batch)

        bridge.execute = record
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}

        first = await bridge.call(wait)
        assert first["is_error"] is False
        assert bridge.steps == 1
        assert len(executed) == 1

        second = await bridge.call(wait)
        assert second["is_error"] is True
        assert second["details"]["rejection_class"] == "second-batch"
        assert len(executed) == 1

        bridge.fold(TurnEndEvent())
        third = await bridge.call(wait)
        assert third["is_error"] is False
        assert len(executed) == 2

    @pytest.mark.asyncio
    async def test_a_dropped_sibling_field_is_stated_in_the_result(self, tmp_path: Path) -> None:
        """A carried-and-dropped field is not silent on this channel either.

        The result of the call is the next thing the model reads, so the note
        rides it ahead of the observation: the definition of a silent drop is
        that the model is never told, and a model that is never told re-sends
        the same field on the next reply instead of correcting it.
        """

        bridge = _bridge(tmp_path, execute=None)

        result = await bridge.call(
            {"actions": [{"kind": "wait", "duration_ms": 50, "frame_id": "screen"}]}
        )

        assert result["is_error"] is False
        assert result["content"][0].text == (
            'Note: "frame_id" was not accepted on a "wait" action and was ignored '
            '(a "wait" action takes "duration_ms").'
        )
        # The field was still dropped, never forwarded: the call executed and
        # the rendered observation follows the note, exactly as before.
        assert result["content"][1].text == "seen 1"

    @pytest.mark.asyncio
    async def test_a_finish_batch_is_terminal_and_never_executes(self, tmp_path: Path) -> None:
        executed: list[Any] = []
        bridge = _bridge(tmp_path, execute=None)
        original = bridge.execute

        async def record(batch: Any) -> Any:
            executed.append(batch)
            return await original(batch)

        bridge.execute = record
        claim = {"actions": [{"kind": "finish", "status": "done", "reason": "complete"}]}
        # The first `done` claim is CHALLENGED (the session-path completion
        # gate, see TestCompletionGate); the episode ends on the second. NEITHER
        # claim ever reaches the adapter -- a finish mutates nothing.
        first = await bridge.call(claim)
        assert first["details"]["terminal"] == "completion-challenged"
        reply = await bridge.call(claim)
        assert reply["is_error"] is False
        assert reply["details"]["terminal"] == "finish"
        assert bridge.terminal is True
        assert bridge.end_requested == "finish"
        assert executed == []

        after = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert after["is_error"] is True
        assert bridge.steps == 0

    @pytest.mark.asyncio
    async def test_the_step_budget_ends_the_episode_without_executing(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path, max_steps=1)
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}
        assert (await bridge.call(wait))["is_error"] is False
        # The budget is checked at the TOP of a turn, exactly as the runner
        # checks it before the next decision -- so the refusal lands on the
        # first call after the turn boundary re-arms the token, and the
        # budgeted batch never runs.
        bridge.fold(TurnEndEvent())
        refusal = await bridge.call(wait)
        assert refusal["is_error"] is False
        assert refusal["details"]["terminal"] == "max-steps"
        assert bridge.end_requested == "max-steps"
        assert bridge.steps == 1

    @pytest.mark.asyncio
    async def test_an_unparseable_batch_is_refused_with_a_class(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        reply = await bridge.call({"actions": [{"kind": "warp", "x": 1, "y": 2}]})
        assert reply["is_error"] is True
        assert "rejection_class" in reply["details"]

    @pytest.mark.asyncio
    async def test_a_missing_pending_observation_is_refused(self, tmp_path: Path) -> None:
        bridge = ActionBridge(
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=2,
        )
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert reply["is_error"] is True


class TestSplitPromptContent:
    """The rendered-blocks → ``Session.prompt(text, images)`` seam.

    ``split_prompt_content`` used to read ``block.text`` on everything that is
    not an image, behind a ``Sequence[Any]`` parameter — so widening the
    Content union with ``AudioContent`` made the first audio block an
    AttributeError pyright could not see (agent review round 1, queued latent
    crash). The guard drops what this seam was never meant to forward; these
    tests pin the new drop AND that text/images still split exactly as before.
    """

    def test_text_and_images_split_exactly_as_before(self) -> None:
        image = ImageContent(data="QUJD", mime_type="image/png")
        text, images = split_prompt_content(
            [TextContent(text="look"), image, TextContent(text="here")]
        )
        assert text == "look\nhere"
        assert images == [image]

    def test_an_audio_block_is_dropped_not_an_attribute_error(self) -> None:
        audio = AudioContent(data="QUJD", mime_type="audio/wav")
        image = ImageContent(data="QUJD", mime_type="image/png")
        text, images = split_prompt_content([TextContent(text="listen"), audio, image])
        assert text == "listen"
        assert images == [image]
        # Audio-only content yields the empty split, not a crash and not a
        # stringified block.
        assert split_prompt_content([audio]) == ("", [])


class TestCompletionGate:
    """The session path's completion gate: one challenge, then the claim stands.

    WHY THIS EXISTS. The first real-task session run (arm 1687c, task_013)
    filled a form correctly and then called ``finish`` WITHOUT submitting it:
    the evaluator's own state capture carried no ``form_response``, and the
    episode scored 0 where the reply channel's identical answers scored 1.0.
    The reply channel scored because its runner refuses a ``done`` declaration
    ONCE (``runner/completion.py``) and re-asks the model to check the claim
    against the screen; the session path accepted the claim immediately.
    These tests pin the identical contract on this channel:

    * the challenge fires ONCE, then every later finish is accepted -- the
      bound is what stops a genuinely-finished model looping forever;
    * the challenge is a refusal of the CLAIM, not a protocol error: nothing
      moves (no step counted, no terminal set, no adapter call) and a
      corrected action batch still executes afterwards;
    * ``failed`` claims are never challenged (there is no completion to
      confirm), and the gate's budget is the runner's own config knob.
    """

    @pytest.mark.asyncio
    async def test_the_first_done_claim_is_challenged_and_nothing_moves(
        self, tmp_path: Path
    ) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        obs = _observation()
        bridge = _bridge(
            tmp_path, observation=obs, record=lambda kind, payload: events.append((kind, payload))
        )
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "filled the form"}]}
        )
        assert reply["is_error"] is False
        assert reply["details"]["terminal"] == "completion-challenged"
        text = reply["content"][0].text
        assert "That declaration is a CLAIM" in text
        assert "filled the form" in text
        assert "the task as stated" in text
        assert bridge.terminal is False
        assert bridge.end_requested is None
        assert bridge.steps == 0
        assert [kind for kind, _ in events] == ["completion_challenged"]
        assert events[0][1]["reason"] == "filled the form"
        assert events[0][1]["observation_id"] == obs.observation_id

    @pytest.mark.asyncio
    async def test_the_second_done_claim_ends_the_episode(self, tmp_path: Path) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        bridge = _bridge(tmp_path, record=lambda kind, payload: events.append((kind, payload)))
        claim = {"actions": [{"kind": "finish", "status": "done", "reason": "done again"}]}
        await bridge.call(claim)
        second = await bridge.call(claim)
        assert second["details"]["terminal"] == "finish"
        assert bridge.terminal is True
        assert bridge.end_requested == "finish"
        # The record carries the CLAIM -- status and reason survive -- which is
        # what the challenge (and any reader grading this finish) reasons about.
        assert [payload for kind, payload in events if kind == "finish"] == [
            {
                "status": "done",
                "reason": "done again",
                "actions": 1,
                "completion_challenged": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_an_action_batch_after_the_challenge_still_executes(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        await bridge.call({"actions": [{"kind": "finish", "status": "done", "reason": "early"}]})
        executed = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert executed["is_error"] is False
        assert bridge.steps == 1
        # The executed batch consumed the turn's token; the turn boundary is
        # what re-arms it (`fold`), exactly as the live loop delivers it.
        bridge.fold(TurnEndEvent())
        # Exactly ONE challenge per episode: the finish that follows the action
        # ends the episode rather than repeating the challenge -- the loop this
        # gate must not create.
        end = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "now"}]}
        )
        assert end["details"]["terminal"] == "finish"
        assert bridge.terminal is True

    @pytest.mark.asyncio
    async def test_a_failed_claim_is_never_challenged(self, tmp_path: Path) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        bridge = _bridge(tmp_path, record=lambda kind, payload: events.append((kind, payload)))
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "failed", "reason": "cannot finish"}]}
        )
        assert reply["details"]["terminal"] == "finish"
        assert [kind for kind, _ in events] == ["finish"]

    @pytest.mark.asyncio
    async def test_the_gate_can_be_disabled_for_a_control_arm(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path, completion_gate=False)
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        )
        assert reply["details"]["terminal"] == "finish"

    @pytest.mark.asyncio
    async def test_a_zero_challenge_budget_accepts_the_first_claim(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path, completion_challenges=0)
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        )
        assert reply["details"]["terminal"] == "finish"

    @pytest.mark.asyncio
    async def test_the_challenge_re_attaches_the_shown_state_and_the_channel_guidance(
        self, tmp_path: Path
    ) -> None:
        from local_operator.evaluation.session_arm import CHALLENGE_REPLY_GUIDANCE

        bridge = _bridge(
            tmp_path,
            reply_guidance=CHALLENGE_REPLY_GUIDANCE.format(tool_name="apply_actions"),
        )
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        )
        text = reply["content"][0].text
        assert "Reply with a single `apply_actions` call" in text
        assert "JSON batch" not in text
        # The state the model is looking at rides WITH the challenge: the same
        # rendered blocks, attached -- never re-rendered (the renderer appends
        # a turn per render; a second render would duplicate the frame).
        assert len(reply["content"]) == 2
        assert reply["content"][1].text == "seen 0"

    @pytest.mark.asyncio
    async def test_a_challenge_does_not_re_deliver_a_dropped_fields_note(
        self, tmp_path: Path
    ) -> None:
        """A drop's note is served once, on the result, never on a challenge.

        The completion challenge re-attaches the state stored in
        ``_last_shown`` -- the rendered screen -- and the note is a correction
        about an EARLIER reply: re-attached to an unrelated challenge it reads
        as guidance for the claim under challenge. Both challenge arms (the
        finish call and the prose path) share ``_shown_blocks``, so this pin
        covers both (review round 1, MINOR 1).
        """

        bridge = _bridge(tmp_path, execute=None)

        result = await bridge.call(
            {"actions": [{"kind": "wait", "duration_ms": 50, "frame_id": "screen"}]}
        )
        assert result["content"][0].text.startswith("Note: ")

        bridge.fold(TurnEndEvent())
        challenge = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        )

        assert challenge["details"]["terminal"] == "completion-challenged"
        # The challenge re-attaches the rendered observation alone; under the
        # bug this list was ``[challenge, Note, "seen 1"]``.
        assert len(challenge["content"]) == 2
        assert challenge["content"][1].text == "seen 1"
        assert all("was not accepted" not in block.text for block in challenge["content"])


#: The prose-claim predicate's calibration table: every positive is a message
#: the SEALED session-arm corpus actually shows asserting completion (post-
#: finish summaries included -- the predicate scores the TEXT, not the ending),
#: and every negative is a real terminal or near-terminal message that must
#: NOT read as a claim: the mid-work narration that ended arm 1716's 005/006
#: (its last texts, and the empty terminal messages that followed them) and
#: the step-budget notices. A new corpus shape lands HERE first, then in the
#: pattern list in ``session_arm.py``. The whole-task positives ('I've finished
#: everything.', 'The task has been completed.') pin the object bindings and the
#: 've contraction; the two review-round tables below pin the veto set.
PROSE_CLAIM_SAMPLES = (
    "Done. Summary of what I determined and did:",
    "Done. Here's what I did: found the schedule",
    'Done. The FormCraft form is filled and submitted (confirmation: "Thanks").',
    "**Done.** the composites were applied",
    "All done.",
    "The task is complete. **Result:** `/home/user/Desktop/Checklist.docx`",
    "The Google Maps walking route is complete and displayed: 8 stops, walking mode.",
    "I have completed the task and verified the file opens.",
    "I've finished everything.",
    "The task has been completed.",
    "All deliverables have been completed.",
)

PROSE_NON_CLAIM_SAMPLES = (
    "",
    "Filenames corrected. Opening all 12 numbered document scans as an eog collection in "
    "fullscreen.",
    "Downloads mostly succeeded. Now fix #20 (Drive confirm page), inspect #15, extract PDF "
    "text with pdftotext, and find CV links in the homepages.",
    "Testing whether a dismissible overlay blocks clicks: Escape then click a profile link.",
    "The view jumped to the status fields. Filling Publishing status first.",
    "The home page bundle should reveal the API endpoints. Opening it in view-source.",
    "The viewer window stole focus; clicking back to the terminal.",
    "Impress is focused (still holding the pre-edit copy). Reloading it from disk.",
    "**task_017 — run closed early: episode step budget reached; task not completed.** The "
    "episode's action surface hit its step budget at step 3, while I was reading the "
    "registration guide.",
    "**Episode closed on the step budget while I was still surveying the registration guide "
    "— the Google Maps deliverable was never set.**",
)

#: The eight adversarial narration shapes PR #1768's round-1 review measured
#: firing on the first predicate (8 of 8) -- the regression table for the veto
#: set in ``session_arm.py``. Each is a TERMINAL-shaped mid-work message: a
#: progress report, a sub-task note, or a claim qualified into a sub-scope.
#: Per-shape mechanisms: whole-task object binding (1, 6, 7), ordinal sub-step
#: (2), partial scope (3), the "Done with/for" qualifier (4), the continuation
#: tail after the completion phrase (5, 8), with the next-work phrases as the
#: belt-and-braces veto several of them also carry.
PROSE_REVIEW_ROUND_1_PROBES = (
    "I have finished the first two files and will continue with the rest.",
    "The first chart is complete; the second still needs data.",
    "The work is done for this step; moving to the next one.",
    "Done with the first document \u2014 starting the second now.",
    "The download is done, unpacking it now.",
    "The first file has been completed; continuing with the next.",
    "I have completed the initial setup.",
    "The installation is finished; launching the app.",
)

#: The DOCUMENTED residual over-fire class (see ``prose_claims_completion``'s
#: docstring): a subject-agnostic "X is done/complete" about a non-ordinal
#: sub-object whose message carries no continuation clause. Kept deliberately:
#: every narrowing tried against the corpus's real samples cost a genuine
#: claim shape ("the route is complete and displayed"), and the cost of the
#: residual is the one bounded challenge cycle. Pinned so the boundary is
#: explicit, not assumed -- a future tightening that kills these updates the
#: docstring rather than silently drifting.
PROSE_RESIDUAL_OVERFIRE_SAMPLES = (
    "The download is done.",
    "The installation is finished.",
)


def _fold_assistant_message(
    bridge: ActionBridge, text: str, tool_calls: list[ToolCall] | None = None
) -> None:
    """Deliver one assistant message the way the live stream does (``message_end``)."""

    bridge.fold(
        MessageEndEvent(
            message=Message(
                role="assistant",
                content=[TextContent(text=text)] if text else [],
                tool_calls=tool_calls or [],
            )
        )
    )


class TestProseClaimDetector:
    """The predicate alone, against real corpus samples (see the tables above)."""

    @pytest.mark.parametrize("text", PROSE_CLAIM_SAMPLES, ids=lambda text: text[:36])
    def test_assertions_of_completion_are_claims(self, text: str) -> None:
        assert prose_claims_completion(text) is True

    @pytest.mark.parametrize("text", PROSE_NON_CLAIM_SAMPLES, ids=lambda text: repr(text[:36]))
    def test_narration_progress_reports_and_notices_are_not_claims(self, text: str) -> None:
        assert prose_claims_completion(text) is False

    @pytest.mark.parametrize("text", PROSE_REVIEW_ROUND_1_PROBES, ids=lambda text: repr(text[:36]))
    def test_round_one_review_probe_shapes_are_not_claims(self, text: str) -> None:
        """The eight adversarial narration shapes from PR #1768's round-1 review.

        They fired 8/8 on the first predicate (the review's own table); the
        veto set in ``session_arm.py`` keeps them out -- see the table's header
        for the per-shape mechanism.
        """

        assert prose_claims_completion(text) is False

    @pytest.mark.parametrize(
        "text", PROSE_RESIDUAL_OVERFIRE_SAMPLES, ids=lambda text: repr(text[:36])
    )
    def test_the_documented_residual_overfire_is_pinned(self, text: str) -> None:
        """The residual class is DELIBERATE and disclosed, not accidental.

        The detector's docstring names it; this test makes the boundary
        executable so a future narrowing cannot silently diverge from the
        documented behaviour without failing here first.
        """

        assert prose_claims_completion(text) is True


class TestProseCompletionGate:
    """The gate's answer-side arm: a terminal PROSE claim earns the one challenge.

    WHY THIS EXISTS. The gate landed in #1696 fires inside ``ActionBridge.call``,
    so it only sees TOOL-mediated claims. The first field run of arm 1748
    (task_003) ended its final answer as prose -- "Done. Summary of what I
    determined and did: ..." with NO tool call -- and the turn simply ended:
    the gate never fired and the run reads as an unverified finish, exactly the
    case the gate exists to prevent. These tests pin both sides of the fix:

    * a terminal message that CLAIMS completion earns the SAME one challenge a
      finish call earns -- the same shared budget, the same challenge text,
      delivered against the state the model last saw;
    * everything else -- the mid-work narration classes (including the eight
      adversarial shapes the round-1 review found; the predicate's tables
      above pin them), empty terminal messages (the silent-provider ending
      class 005/006 exhibit), messages that carry a tool call, and any episode
      that already ended -- does not fire.
    """

    @pytest.mark.asyncio
    async def test_a_terminal_prose_claim_is_challenged_and_recorded(self, tmp_path: Path) -> None:
        from local_operator.evaluation.session_arm import CHALLENGE_REPLY_GUIDANCE

        events: list[tuple[str, dict[str, Any]]] = []
        obs = _observation()
        bridge = _bridge(
            tmp_path,
            observation=obs,
            record=lambda kind, payload: events.append((kind, payload)),
            reply_guidance=CHALLENGE_REPLY_GUIDANCE.format(tool_name="apply_actions"),
        )
        claim = "Done. Summary of what I determined and did: both composites were applied"
        _fold_assistant_message(bridge, claim)
        challenge = bridge.prose_completion_challenge()
        assert challenge is not None
        text = challenge[0].text
        assert "That declaration is a CLAIM" in text
        assert claim in text
        assert "the task as stated" in text
        assert "Reply with a single `apply_actions` call" in text
        # The state the model was last shown rides WITH the challenge: the same
        # rendered blocks, attached -- never re-rendered.
        assert len(challenge) == 2
        assert challenge[1].text == "seen 0"
        assert bridge.end_requested is None
        assert bridge.terminal is False
        assert bridge.steps == 0
        # The record names WHICH arm of the gate fired; the finish-call arm's
        # row keeps its own historical shape.
        assert [kind for kind, _ in events] == ["completion_challenged"]
        assert events[0][1]["trigger"] == "terminal-message"
        assert events[0][1]["reason"] == claim
        assert events[0][1]["observation_id"] == obs.observation_id

    @pytest.mark.asyncio
    async def test_the_prose_challenge_re_attaches_the_last_executed_screen(
        self, tmp_path: Path
    ) -> None:
        bridge = _bridge(tmp_path)
        executed = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert executed["is_error"] is False
        # The turn boundary re-arms the token, exactly as the live loop delivers
        # it; the prose arm reads the state from the same seam.
        bridge.fold(TurnEndEvent())
        _fold_assistant_message(bridge, "Done. Summary of what I determined and did: X")
        challenge = bridge.prose_completion_challenge()
        assert challenge is not None
        assert challenge[1].text == "seen 1"

    @pytest.mark.asyncio
    async def test_the_prose_challenge_is_byte_identical_to_the_finish_call_challenge(
        self, tmp_path: Path
    ) -> None:
        """The two arms quote the SAME challenge for the same claim and screen.

        The widening must not change the challenge's kind: for a claim stated
        in the same words against the same observation, the prose arm's user
        turn is byte-identical to the finish-call arm's tool result -- same
        builder, same channel sentence.
        """
        from local_operator.evaluation.session_arm import CHALLENGE_REPLY_GUIDANCE

        guidance = CHALLENGE_REPLY_GUIDANCE.format(tool_name="apply_actions")
        claim_text = "Done. everything was applied"
        via_call = _bridge(tmp_path, reply_guidance=guidance)
        call_reply = await via_call.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": claim_text}]}
        )
        via_prose = _bridge(tmp_path, reply_guidance=guidance)
        _fold_assistant_message(via_prose, claim_text)
        prose = via_prose.prose_completion_challenge()
        assert prose is not None
        assert prose[0].text == call_reply["content"][0].text
        assert prose[1].text == call_reply["content"][1].text

    @pytest.mark.asyncio
    async def test_a_finish_after_the_prose_challenge_is_accepted(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        _fold_assistant_message(bridge, "Done. Summary of what I determined and did: X")
        assert bridge.prose_completion_challenge() is not None
        # ONE budget for both arms: the re-declaration is the episode's second
        # declaration, so it is accepted rather than challenged again -- the
        # gate cannot loop across the two paths.
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "re-checked"}]}
        )
        assert reply["details"]["terminal"] == "finish"
        assert bridge.end_requested == "finish"

    @pytest.mark.asyncio
    async def test_a_spent_finish_budget_blocks_the_prose_arm(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        await bridge.call({"actions": [{"kind": "finish", "status": "done", "reason": "claimed"}]})
        _fold_assistant_message(bridge, "Done. Summary of what I determined and did: X")
        assert bridge.prose_completion_challenge() is None

    @pytest.mark.asyncio
    async def test_the_prose_arm_does_not_fire_once_the_episode_ended(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        claim = {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        await bridge.call(claim)
        accepted = await bridge.call(claim)  # the second declaration always stands
        assert accepted["details"]["terminal"] == "finish"
        # Every completed run in the corpus ends with exactly this shape: a
        # summary prose message AFTER the accepted finish. It must not earn a
        # second exchange.
        _fold_assistant_message(bridge, "Done. Summary of what I determined and did: X")
        assert bridge.prose_completion_challenge() is None

    @pytest.mark.parametrize(
        ("gate", "challenges"),
        [(False, 1), (True, 0)],
        ids=["gate-disabled", "zero-budget"],
    )
    @pytest.mark.asyncio
    async def test_the_prose_arm_obeys_the_gate_controls(
        self, tmp_path: Path, gate: bool, challenges: int
    ) -> None:
        bridge = _bridge(tmp_path, completion_gate=gate, completion_challenges=challenges)
        _fold_assistant_message(bridge, "Done. Summary of what I determined and did: X")
        assert bridge.prose_completion_challenge() is None

    @pytest.mark.asyncio
    async def test_an_empty_terminal_message_is_never_a_claim(self, tmp_path: Path) -> None:
        # The silent-provider ending class (005/006): the model's next response
        # came back EMPTY and the loop ended. There is no claim to answer.
        bridge = _bridge(tmp_path)
        _fold_assistant_message(bridge, "")
        assert bridge.prose_completion_challenge() is None

    @pytest.mark.asyncio
    async def test_mid_work_narration_is_never_a_claim(self, tmp_path: Path) -> None:
        # Arm 1716's 005/006 ended after exactly these texts: a mid-work report
        # plus what was about to be done. They are the discriminating case for
        # the predicate -- a wider rule would trip on them.
        bridge = _bridge(tmp_path)
        _fold_assistant_message(
            bridge,
            "Filenames corrected. Opening all 12 numbered document scans as an eog collection "
            "in fullscreen.",
        )
        assert bridge.prose_completion_challenge() is None
        _fold_assistant_message(
            bridge,
            "Downloads mostly succeeded. Now fix #20 (Drive confirm page), inspect #15, "
            "extract PDF text with pdftotext, and find CV links in the homepages.",
        )
        assert bridge.prose_completion_challenge() is None

    @pytest.mark.asyncio
    async def test_a_message_that_carries_a_tool_call_is_not_this_arms_business(
        self, tmp_path: Path
    ) -> None:
        bridge = _bridge(tmp_path)
        _fold_assistant_message(
            bridge,
            "Done. Summary of what I determined and did: X",
            tool_calls=[ToolCall(id="call-1", name="apply_actions", arguments={})],
        )
        assert bridge.prose_completion_challenge() is None


def _confinement_fake_opener(installed: list[Any]) -> Any:
    """An ``sdk.open_session`` stand-in whose session records the install."""

    class _FakeSession:
        def set_tool_confinement(self, root: Any) -> None:
            installed.append(root)

        def subscribe(self, sink: Any) -> Any:
            del sink
            return lambda: None

    class _FakeContext:
        async def __aenter__(self) -> Any:
            return _FakeSession()

        async def __aexit__(self, *exc: Any) -> bool:
            del exc
            return False

    def opener(spec: Any, *, roots: Any, mode: Any) -> Any:
        del spec, roots, mode
        return _FakeContext()

    return opener


class TestConfinementInstall:
    """The episode session is opened confined, before any tool call can run."""

    @pytest.mark.asyncio
    async def test_the_episode_session_is_confined_to_the_scratch(self, tmp_path: Path) -> None:
        installed: list[Any] = []
        scratch = tmp_path / "scratch"
        handle = await open_episode_session(
            spec=SessionSpec(),
            roots=_roots_free(),
            session_opener=_confinement_fake_opener(installed),
            confinement_root=scratch,
        )
        # Installed BEFORE the sink and the first prompt: the session rebuilds
        # its tool context per turn, so this call is the earliest point at
        # which no tool call can run against an unconfined context.
        assert installed == [scratch]
        assert handle.session is not None

    @pytest.mark.asyncio
    async def test_no_confinement_root_installs_nothing(self, tmp_path: Path) -> None:
        installed: list[Any] = []
        await open_episode_session(
            spec=SessionSpec(),
            roots=_roots_free(),
            session_opener=_confinement_fake_opener(installed),
        )
        assert installed == []


class TestMcpWire:
    @pytest.mark.asyncio
    async def test_a_real_server_delivers_the_rendered_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        work = home / "work"
        config = home / ".local-operator"
        artifacts = tmp_path / "artifacts"
        for path in (work, config, artifacts):
            path.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

        obs0 = _observation(frames=(_frame(artifacts, PNG),))
        obs1 = _observation(sequence=1, text="screen B", frames=(_frame(artifacts, PNG),))
        executed: list[Any] = []

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            return _result(input_observation=obs0, output_observation=obs1)

        endpoint = tmp_path / "b.sock"
        renderer = ObservationRenderer(artifact_root=artifacts)
        bridge = ActionBridge(
            endpoint=endpoint,
            surface=ActionSurface(),
            render=renderer.render,
            execute=execute,
            max_steps=5,
        )
        bridge.arm(obs0)
        await bridge.start()
        try:
            decl = declare_action_server(
                config_dir=config,
                endpoint=endpoint,
                surface=ActionSurface(),
                python_executable=sys.executable,
                cwd=work,
            )
            manager = McpManager(str(work))
            try:
                await manager.discover_and_connect()
                await manager.wait_settled(20.0)
                tools = {tool.name for tool in manager.get_tools()}
                assert decl.tool_name in tools, tools
                tool = next(tool for tool in manager.get_tools() if tool.name == decl.tool_name)
                result = await tool.execute(
                    "c1",
                    {"actions": [{"kind": "click", "frame_id": "frame-0", "x": 3, "y": 4}]},
                    None,
                    None,
                    ToolContext(),
                )
            finally:
                await manager.disconnect_all()
        finally:
            await bridge.stop()

        assert result.is_error is False
        assert len(executed) == 1
        texts = [block.text for block in result.content if isinstance(block, TextContent)]
        images = [block for block in result.content if isinstance(block, ImageContent)]
        assert any("Step: 1" in text for text in texts), texts
        assert len(images) == 1
        assert base64.b64decode(images[0].data) == PNG

    @pytest.mark.asyncio
    async def test_a_dead_bridge_surfaces_as_an_error_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        work = home / "work"
        config = home / ".local-operator"
        for path in (work, config):
            path.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        decl = declare_action_server(
            config_dir=config,
            endpoint=tmp_path / "missing.sock",
            surface=ActionSurface(),
            python_executable=sys.executable,
            cwd=work,
        )
        manager = McpManager(str(work))
        try:
            await manager.discover_and_connect()
            await manager.wait_settled(20.0)
            tool = next(tool for tool in manager.get_tools() if tool.name == decl.tool_name)
            result = await tool.execute(
                "c1", {"actions": [{"kind": "wait", "duration_ms": 50}]}, None, None, ToolContext()
            )
        finally:
            await manager.disconnect_all()
        assert result.is_error is True
        assert "bridge" in result.text


class TestEpisodeSessionSurface:
    @pytest.fixture
    def scratch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionRoots:
        home = tmp_path / "home"
        root = home / ".local-operator"
        agent_home = home / "local-operator-home"
        work = home / "work"
        for path in (root, agent_home, work):
            path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(agent_home))
        monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
        monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
        monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
        return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)

    @pytest.mark.asyncio
    async def test_the_inventory_holds_the_shipped_tools_and_the_action_tool(
        self, scratch: SessionRoots, tmp_path: Path
    ) -> None:
        bridge = ActionBridge(
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=2,
        )
        bridge.arm(_observation())
        await bridge.start()
        try:
            decl = declare_action_server(
                config_dir=scratch.config_path,
                endpoint=tmp_path / "b.sock",
                surface=ActionSurface(),
                python_executable=sys.executable,
                cwd=scratch.cwd_path,
            )
            assert_declaration_resolved(
                cwd=scratch.cwd_path,
                config_dir=scratch.config_path,
                scratch_root=scratch.cwd_path.parent,
            )
            spec = SessionSpec(
                hosting="test", model="mock", approvals=ApprovalPolicy.auto(), name="arm-surface"
            )
            async with sdk.open_session(spec, roots=scratch) as session:
                deadline = time.monotonic() + 25.0
                names: tuple[str, ...] = ()
                while time.monotonic() < deadline:
                    names = session_tool_names(session)
                    if decl.tool_name in names:
                        break
                    await asyncio.sleep(0.1)
                assert decl.tool_name in names, names
                assert {"task", "team", "hub", "jobs"} <= set(names), names
                assert len(names) >= 20
        finally:
            await bridge.stop()


class TestActionToolGate:
    """The settle gate: prompt only once the action tool provably exists.

    MCP discovery is asynchronous by design -- the action server is spawned and
    indexed after ``open`` -- and the first request's tool array is published
    ONCE per turn. A prompt that wins that race answers every call with
    "Tool not found: <action tool>" (measured on the pilot), so the arm holds
    the prompt until the tool is in the LIVE inventory and refuses to run when
    it never arrives.
    """

    @pytest.mark.asyncio
    async def test_the_gate_holds_until_the_tool_is_live(self, tmp_path: Path) -> None:
        from local_operator.evaluation.session_arm import (
            EpisodeSession,
            _await_action_tool,
        )

        class _Session:
            def __init__(self) -> None:
                self._tools: list[Any] = [SimpleNamespace(name="bash")]
                self.mcp_startup = None
                self.mcp_manager = None

        class _Declaration:
            tool_name = "mcp__episode_actions_apply_actions"

        session = _Session()
        handle = EpisodeSession(session=session, roots=_roots_free(), _context=_AsyncContext())
        record = cast(Any, SimpleNamespace(write=lambda *a, **k: None))
        declaration = cast(Any, _Declaration())
        assert await _await_action_tool(handle, declaration, record) is False

        # The tool arriving in the LIVE inventory -- the same list the first
        # request publishes from -- flips the gate without a reopen.
        session._tools.append(SimpleNamespace(name=_Declaration.tool_name))
        assert await _await_action_tool(handle, declaration, record) is True

    @pytest.mark.asyncio
    async def test_tool_names_are_live_not_frozen(self) -> None:
        from local_operator.evaluation.session_arm import EpisodeSession

        class _Session:
            def __init__(self) -> None:
                self._tools: list[Any] = []

        session = _Session()
        handle = EpisodeSession(session=session, roots=_roots_free(), _context=_AsyncContext())
        assert handle.tool_names == ()
        session._tools.append(SimpleNamespace(name="bash"))
        assert handle.tool_names == ("bash",)

    @pytest.mark.asyncio
    async def test_an_over_long_socket_path_is_refused_before_binding(self, tmp_path: Path) -> None:
        # sockaddr_un is ~104 bytes on macOS; a scratch deep enough to trip it
        # must fail here, in words, rather than as a connect error in the child.
        long_dir = tmp_path / ("d" * 60) / ("e" * 60)
        long_dir.mkdir(parents=True)
        bridge = ActionBridge(
            endpoint=long_dir / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=1,
        )
        with pytest.raises(SessionArmError, match="too long for a UNIX socket"):
            await bridge.start()

    def test_cleanup_without_receipts_forces_rescue(self, tmp_path: Path) -> None:
        from local_operator.evaluation.session_arm import _cleanup_forces_rescue

        # No receipts is the dead-worker case: "attempted" is never evidence.
        assert _cleanup_forces_rescue(None, ()) is True  # type: ignore[arg-type]


class TestWireReadLimits:
    """Both ends of the socket must carry the wire's read limit, not the default.

    Added after the 2026-09-28 paid probe: a 476 KiB observation frame made
    ``forward_call``'s read raise "Separator is not found, and chunk exceed the
    limit" (asyncio's 64 KiB default), so every EXECUTED batch was answered to
    the model as unreachable while the desktop had already acted.
    """

    @pytest.mark.asyncio
    async def test_a_call_frame_larger_than_the_stream_default_is_served(
        self, tmp_path: Path
    ) -> None:
        # The request direction carries the same limit: a paste action may
        # legally carry ``max_type_chars`` characters in one frame, larger
        # than the default read limit.
        surface = ActionSurface(paste_text=True, max_type_chars=100_000)
        executed: list[Any] = []
        obs_in = _observation()
        obs_out = _observation(sequence=1, text="screen B")

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            return _result(input_observation=obs_in, output_observation=obs_out)

        bridge = ActionBridge(
            endpoint=tmp_path / "big-call.sock",
            surface=surface,
            render=lambda observation: [TextContent(text="seen")],
            execute=execute,
            max_steps=3,
        )
        bridge.arm(obs_in)
        await bridge.start()
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.open_unix_connection(
                str(bridge.endpoint), limit=WIRE_READ_LIMIT_BYTES
            )
            writer.write(
                encode_call(
                    {
                        "actions": [
                            {
                                "kind": "paste_text",
                                "keys": ["META", "v"],
                                "clipboard_policy": "overwrite",
                                "text": "x" * 90_000,
                            }
                        ]
                    }
                )
            )
            await writer.drain()
            reply = decode_response(await reader.readline())
        finally:
            if writer is not None:
                writer.close()
            await bridge.stop()
        assert reply.get("is_error") is False
        assert bridge.steps == 1
        assert len(executed) == 1


class TestFailedExecuteRecovery:
    """The F1 wedge (arm 1796, tasks 001/013): one failed execute must not make
    the episode terminally unable to bind a batch -- including ``finish``.

    The pre-fix shape: the token is consumed before the batch runs, a raise
    from ``execute`` left it consumed with nothing in flight, and every later
    call was refused "no action batch was run" until the model gave up as
    ``agent_stop``. These tests pin the two endings the re-bind seam makes
    possible: continue on the environment's re-read observation, or end
    legibly-wedged -- never a spin.
    """

    @pytest.mark.asyncio
    async def test_a_committed_failure_re_binds_and_the_episode_continues(
        self, tmp_path: Path
    ) -> None:
        obs0 = _observation()
        obs1 = _observation(sequence=1, text="screen B")
        executed: list[Any] = []
        re_reads: list[int] = []

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            if len(executed) == 1:
                raise _observation_phase_error()
            return _result(input_observation=obs0, output_observation=obs1)

        async def recover() -> Observation:
            re_reads.append(1)
            return obs0

        bridge = _bridge(tmp_path, observation=obs0, execute=execute, recover=recover)
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}

        first = await bridge.call(wait)
        # The call itself is an error -- its own result WAS lost -- but the
        # episode is not lost with it.
        assert first["is_error"] is True
        assert first["details"]["recovered"] is True
        assert len(first["content"]) == 1
        assert first["content"][0].text == RECOVERY_AFTER_READBACK_LOSS
        assert len(executed) == 1 and len(re_reads) == 1
        assert bridge.end_requested is None
        assert bridge.steps == 0  # the failed batch never ran as a step

        # Held in flight, exactly like a batch result: the same turn cannot
        # bind a second batch, and the turn boundary arms the recovered screen.
        same_turn = await bridge.call(wait)
        assert same_turn["details"]["rejection_class"] == "second-batch"
        bridge.fold(TurnEndEvent())

        after = await bridge.call(wait)
        assert after["is_error"] is False
        assert bridge.steps == 1
        assert len(executed) == 2
        assert len(re_reads) == 1  # one re-bind per failed batch, not a loop

    @pytest.mark.asyncio
    async def test_a_recovered_screen_not_already_shown_is_rendered(self, tmp_path: Path) -> None:
        obs1 = _observation(sequence=1, text="screen B")

        async def execute(batch: Any) -> ExecuteResult:
            del batch
            raise _observation_phase_error()

        async def recover() -> Observation:
            return obs1  # newer than the screen the model last saw

        bridge = _bridge(tmp_path, execute=execute, recover=recover)
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        # The sentence, then the one renderer's blocks for the recovered screen.
        assert reply["content"][0].text == RECOVERY_AFTER_READBACK_LOSS
        assert reply["content"][1].text == "seen 1"
        assert bridge.end_requested is None

        # The recovered screen is now the state a later challenge re-attaches.
        assert bridge._last_observation is not None
        assert bridge._last_observation.observation_id == obs1.observation_id

    @pytest.mark.asyncio
    async def test_recovery_is_attempted_once_per_failed_batch(self, tmp_path: Path) -> None:
        re_reads: list[int] = []

        async def execute(batch: Any) -> ExecuteResult:
            del batch
            raise _observation_phase_error()

        async def recover() -> Observation:
            re_reads.append(1)
            raise _observation_phase_error()  # the environment is still down

        bridge = _bridge(tmp_path, execute=execute, recover=recover)
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}
        first = await bridge.call(wait)
        assert first["details"]["terminal"] == BRIDGE_WEDGED_TERMINAL
        assert len(re_reads) == 1
        bridge.fold(TurnEndEvent())
        # A later call finds the token closed: refused, no second attempt.
        again = await bridge.call(wait)
        assert again["is_error"] is True
        assert len(re_reads) == 1

    @pytest.mark.asyncio
    async def test_a_failure_that_did_not_declare_the_commit_is_never_re_read(
        self, tmp_path: Path
    ) -> None:
        re_reads: list[int] = []

        async def execute(batch: Any) -> ExecuteResult:
            del batch
            raise _deadline_error()

        async def recover() -> Observation:
            re_reads.append(1)
            return _observation()

        bridge = _bridge(tmp_path, execute=execute, recover=recover)
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert reply["details"]["terminal"] == BRIDGE_WEDGED_TERMINAL
        assert re_reads == []
        assert bridge.end_diagnostic is not None
        assert "ambiguous" in bridge.end_diagnostic

    @pytest.mark.asyncio
    async def test_without_a_re_bind_seam_the_failure_is_still_legible(
        self, tmp_path: Path
    ) -> None:
        async def execute(batch: Any) -> ExecuteResult:
            del batch
            raise _observation_phase_error()

        bridge = _bridge(tmp_path, execute=execute)  # no recover seam
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert reply["details"]["terminal"] == BRIDGE_WEDGED_TERMINAL
        assert bridge.end_diagnostic is not None
        assert "no re-read path is wired" in bridge.end_diagnostic

    @pytest.mark.asyncio
    async def test_an_unrecoverable_failure_ends_legibly_and_never_spins(
        self, tmp_path: Path
    ) -> None:
        executed: list[Any] = []
        recorded: list[tuple[str, dict[str, Any]]] = []

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            raise _observation_phase_error()

        async def recover() -> Observation:
            raise _observation_phase_error()

        bridge = _bridge(
            tmp_path,
            execute=execute,
            recover=recover,
            record=lambda kind, payload: recorded.append((kind, payload)),
        )
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})

        # The sentence names the transport failure and what was attempted.
        assert reply["is_error"] is True
        assert reply["details"]["terminal"] == BRIDGE_WEDGED_TERMINAL
        assert bridge.end_requested == BRIDGE_WEDGED_TERMINAL
        assert bridge.end_diagnostic is not None
        assert "bridge wedged by a transport error" in bridge.end_diagnostic
        assert "RpcRemoteError" in bridge.end_diagnostic
        assert "re-read of the environment's current observation failed too" in (
            bridge.end_diagnostic
        )
        assert bridge.terminal is True

        # THE DIAGNOSIS' SECOND QUESTION, PINNED: even the terminal action
        # cannot bind -- no screen exists to bind it to -- and it must NOT
        # execute. The wedge is why "the bridge cannot accept a finish" is a
        # defect and not a policy: the harness ends the episode, so no finish
        # is ever needed from a wedged bridge.
        bridge.fold(TurnEndEvent())
        finish = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "done"}]}
        )
        assert finish["is_error"] is True
        assert finish["details"]["rejection_class"] == "second-batch"
        assert len(executed) == 1
        assert bridge.steps == 0

        # Both endings are journalled, so a bundle can tell a degraded
        # environment from a clean run.
        assert [kind for kind, _ in recorded] == ["bridge_wedged"]
        payload = recorded[0][1]
        assert payload["recovery_attempted"] is True
        assert payload["transport_error"].startswith("RpcRemoteError")
        assert payload["recover_error"].startswith("RpcRemoteError")

    @pytest.mark.asyncio
    async def test_a_successful_re_bind_is_journalled(self, tmp_path: Path) -> None:
        obs0 = _observation()
        recorded: list[tuple[str, dict[str, Any]]] = []

        async def execute(batch: Any) -> ExecuteResult:
            del batch
            raise _observation_phase_error()

        async def recover() -> Observation:
            return obs0

        bridge = _bridge(
            tmp_path,
            observation=obs0,
            execute=execute,
            recover=recover,
            record=lambda kind, payload: recorded.append((kind, payload)),
        )
        await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert [kind for kind, _ in recorded] == ["recovered"]
        assert recorded[0][1]["observation_id"] == obs0.observation_id
        assert recorded[0][1]["transport_error"].startswith("RpcRemoteError")


# ---------------------------------------------------------------------------
# The wedge as the DRIVER sees it: the outcome it reports, the turn it stops
# ---------------------------------------------------------------------------


class _OpenContext:
    """The session opener's context: hands the scripted session to the arm."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *exc: Any) -> bool:
        del exc
        return False


class _ScriptedSession:
    """The session surface ``run_session_episode`` drives, acting as BOTH the
    model and the engine's loop: every action batch is spoken over the REAL
    bridge socket (so each reply is produced by the production call path),
    and the loop stops the way the real one stops -- at the post-tool
    boundary, after the driver has asked for a graceful cancel."""

    def __init__(self, scratch_root: Path, script: list[list[dict[str, Any]]]) -> None:
        self._scratch_root = Path(scratch_root)
        self._script = list(script)
        self._sinks: list[Any] = []
        self._tools = [SimpleNamespace(name="mcp__episode_actions_apply_actions")]
        self.mcp_startup = None
        self.mcp_manager = None
        self.replies: list[dict[str, Any]] = []
        self.cancels: list[str] = []
        self.aborts: list[str] = []

    def subscribe(self, sink: Any) -> Any:
        self._sinks.append(sink)
        return lambda: None

    def set_tool_confinement(self, root: Any) -> None:
        del root

    def request_graceful_cancel(self, reason: str = "cancelled") -> None:
        self.cancels.append(reason)

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    async def prompt(
        self, text: str, *, images: Any = None, harness_injected: bool = False
    ) -> None:
        del text, images, harness_injected
        endpoint = next(iter(sorted(self._scratch_root.glob("b-*.sock"))))
        while self._script:
            reply = await self._call(endpoint, self._script.pop(0))
            self.replies.append(reply)
            self._emit(TurnEndEvent())
            if self.cancels:
                break  # the post-tool boundary the real loop stops at
            if (reply.get("details") or {}).get("terminal") == BRIDGE_WEDGED_TERMINAL:
                break

    async def _call(self, endpoint: Path, actions: list[dict[str, Any]]) -> dict[str, Any]:
        reader, writer = await asyncio.open_unix_connection(str(endpoint))
        try:
            writer.write(encode_call({"actions": actions}))
            await writer.drain()
            return decode_response(await reader.readline())
        finally:
            writer.close()
            await writer.wait_closed()

    def _emit(self, event: Any) -> None:
        for sink in self._sinks:
            sink(event)


class _WedgeAdapter(FakeAdapter):
    """FakeAdapter with the session arm's required capability, plus a scripted
    F1 failure: the first ``fail_executes`` executes raise the
    committed-but-lost-read class, and a later read (the recovery) is answered
    only when ``recoverable`` -- i.e. the environment came back."""

    def __init__(self, *args: Any, fail_executes: int, recoverable: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_executes = fail_executes
        self.recoverable = recoverable
        self.executes = 0
        self.observes = 0

    async def handshake(self, *, timeout: float = 10.0) -> Any:
        base = await super().handshake(timeout=timeout)
        capabilities = base.metadata.capabilities.model_copy(
            update={"ask_user_answer_owner": "adapter"}
        )
        return base.model_copy(
            update={"metadata": base.metadata.model_copy(update={"capabilities": capabilities})}
        )

    async def _call_raw(self, method: Any, params: Any, result_type: Any, **kwargs: Any) -> Any:
        if method == "execute":
            self.executes += 1
            if self.executes <= self.fail_executes:
                raise _observation_phase_error()
        if method == "observe":
            self.observes += 1
            # Observation zero is reset_start's own read; every later read is
            # the recovery's.
            if not self.recoverable and self.observes > 1:
                raise _observation_phase_error()
        return await super()._call_raw(method, params, result_type, **kwargs)


class TestBridgeWedgeOutcome:
    """The wedge through ``run_session_episode``: an outcome that says what
    happened, and a turn the driver stops instead of letting the model spin."""

    def _fixture(
        self,
        *,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        scratch_root: Path,
        script: list[list[dict[str, Any]]],
        fail_executes: int,
        recoverable: bool,
    ) -> tuple[Any, _ScriptedSession, _WedgeAdapter]:
        home = scratch_root / "home"
        config_dir = home / ".local-operator"
        agent_home = home / "local-operator-home"
        work = home / "work"
        for path in (config_dir, agent_home, work):
            path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(agent_home))
        roots = SessionRoots(
            config_dir=config_dir, agent_home=agent_home, cwd=work, allow_volatile=True
        )
        spec = build_spec("ep-wedge")
        config = build_config(tmp_path, observation_retry_delay=0.0)
        adapter = _WedgeAdapter(
            tmp_path,
            spec.episode_id,
            fail_executes=fail_executes,
            recoverable=recoverable,
            score=ScoreArtifact(status="scored", binary=0),
        )
        session = _ScriptedSession(scratch_root, script)
        session_spec = SessionSpec(
            hosting="test", model="mock", approvals=ApprovalPolicy.auto(), name="arm-wedge"
        )

        async def rescue(descriptor: Any, **kwargs: Any) -> Any:
            del descriptor, kwargs
            return SimpleNamespace(complete=True, receipts=(), rescue_required=False)

        def launch(selected: Any) -> Any:
            del selected
            return adapter

        def opener(selected: Any, *, roots: Any, mode: Any = None) -> Any:
            del selected, roots, mode
            return _OpenContext(session)

        async def run() -> Any:
            return await run_session_episode(
                spec=spec,
                config=config,
                selector=selector(tmp_path),
                roots=roots,
                scratch_root=scratch_root,
                session_spec=session_spec,
                secrets=(),
                launch=launch,
                rescue=rescue,
                session_opener=opener,
            )

        return run, session, adapter

    @pytest.mark.asyncio
    async def test_an_unrecoverable_failure_ends_failed_with_the_wedge_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        # The script models a model that would keep trying; the driver must
        # stop it before the second call, and the outcome must say why.
        run, session, adapter = self._fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            script=[
                [{"kind": "wait", "duration_ms": 50}],
                [{"kind": "wait", "duration_ms": 50}],
            ],
            fail_executes=4,
            recoverable=False,
        )
        outcome = await run()

        # Not agent_stop: the run died of the transport, and says so.
        assert outcome.status == "failed"
        assert outcome.terminal_reason == BRIDGE_WEDGED_TERMINAL
        assert outcome.diagnostic is not None
        assert "bridge wedged by a transport error" in outcome.diagnostic
        assert "re-read of the environment's current observation failed too" in outcome.diagnostic
        # The score failure this wedge CAUSED (the failed re-read poisons the
        # session) rides along rather than replacing the root cause.
        assert "Scoring the state reached also failed" in outcome.diagnostic

        # The stop: requested once, and the next batch never went out.
        assert len(session.cancels) == 1
        assert "bridge wedged" in session.cancels[0]
        assert len(session.replies) == 1
        assert session.replies[0]["is_error"] is True
        assert adapter.executes == 4  # initial + the three read-back retries
        # A poisoned session demands rescue, and the outcome discloses it.
        assert outcome.rescue_required is True
        # The sealed record carries the terminal event this outcome names,
        # the same way its recovered sibling carries ``action_recovered``.
        events = (outcome.record_root / "events.jsonl").read_text(encoding="utf-8")
        assert "action_bridge_wedged" in events

    @pytest.mark.asyncio
    async def test_a_recovered_failure_lets_the_episode_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_scratch: Path
    ) -> None:
        run, session, adapter = self._fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            scratch_root=short_scratch,
            script=[
                [{"kind": "wait", "duration_ms": 50}],
                [{"kind": "wait", "duration_ms": 50}],
                [{"kind": "finish", "status": "done", "reason": "recovered"}],
                [{"kind": "finish", "status": "done", "reason": "recovered"}],
            ],
            fail_executes=4,
            recoverable=True,
        )
        outcome = await run()

        assert outcome.status == "completed"
        assert outcome.terminal_reason == "finish"
        assert session.cancels == []
        assert len(session.replies) == 4
        # The first reply is the recovery: an error that says the episode is
        # NOT lost, and no re-render of the screen the model already has.
        first = session.replies[0]
        assert first["is_error"] is True
        assert first["details"]["recovered"] is True
        assert len(first["content"]) == 1
        assert first["content"][0]["text"] == RECOVERY_AFTER_READBACK_LOSS
        # The recovery ran, the model's next batch executed once, and the
        # read-back retried -- the episode cost four failed executes, not four
        # batches and not a lost episode.
        assert adapter.executes == 5  # 4 failing + the next wait batch
        assert adapter.observes == 2  # reset's read + the recovery's
        assert outcome.steps == 1
        # The record carries the recovery ("the environment degraded and the
        # harness bridged it"), not a suspiciously clean run.
        events = (outcome.record_root / "events.jsonl").read_text(encoding="utf-8")
        assert "action_recovered" in events

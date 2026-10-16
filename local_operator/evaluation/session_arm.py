"""The benchmark's session engagement: an episode that runs AS a session.

WHAT THIS IS. Design ``docs/design/sdk-engagement.md`` §6: the benchmark stops
hand-building a reply channel and drives the SAME session object the TUI,
``lop exec`` and the mobile runtime drive, with the episode's computer-use
actions exported as a per-session MCP server (``action_server.py``). This
module is the engagement glue: it declares that server in the episode's own
scratch root, hosts the bridge the server forwards calls to, opens the session
through ``local_operator.sdk``, records the one event stream, and (in
``run_session_episode``) drives one whole episode -- launch, reset, session
turn, score, cleanup, close.

WHY THIS MODULE IS NOT IN ``runner/``. The runner core (``evaluation/runner``)
must never import the application: an episode has to be reproducible from its
pinned inputs, and the isolation test asserts that boundary. This module is
deliberately on the OTHER side of that boundary -- it exists precisely to hand
the episode the real session -- so nothing under ``runner/`` may import it,
and it lives beside the evaluation package's other consumer-facing modules.
The old arm imports nothing from here; the two modes are chosen by the run
script and the reply channel's semantics are untouched.

THE TWO HALVES OF THE BRIDGE. ``ActionBridge`` (episode side) owns the token
discipline, the step budget, and the execution seam; the MCP server (child
side) owns nothing but the wire. Calls arrive one per connection, are
serialized here, and carry exactly the shape ``runner/action_tool.py``
projects -- the batch models, the refusals and their hints are imported from
there rather than re-stated, so the MCP channel and the loop-driven channel
cannot drift apart.

STATUS VOCABULARY (the pilot arm's record, not the sealed-bundle format).
``run_session_episode`` returns a :class:`SessionArmOutcome` with one of:
``completed`` (the model declared finish), ``agent_stop`` (the turn ended
without a terminal batch), ``truncated`` (the step budget ended the run; the
state reached is still scored), ``failed_pre_bundle`` (the environment never
came up, including a REFUSAL TO START: the record sink refuses before
``launch`` when the volume cannot hold the expected record plus its seal
reserve), ``failed`` (the run died mid-episode -- a bridge that could not
re-establish its observation binding after a transport failure ends here too,
with ``terminal_reason: "bridge-wedged"``, and never as ``agent_stop``, which
reads as the model giving up).

``terminal_reason`` NAMES the ending and is never left ``None``: the
terminalled paths carry ``finish`` / ``max-steps`` / ``bridge-wedged``; an
``agent_stop`` carries the end-of-turn fact that ended the turn
(``no-tool-call``, ``completion-claim``, ``empty-message``, ``provider-error``
-- with the classified category in ``diagnostic`` -- ``no-progress``,
``gate-stop``, ``wall-bound``, ``aborted``, ``stopped``, or a cut-off cause
token such as ``continuation-limit``); a ``failed`` / ``failed_pre_bundle``
carries the phase it died in (``environment-setup``, ``environment-allocation``,
``record-sink``, ``session-arm``, else its exception class kebabed). The
tokens are read from the shared end-of-turn vocabulary -- the final
``agent_end`` frame, the terminal assistant message, the driver's own wall
bound -- so no adapter-specific state is consulted (see
``_agent_stop_terminal_reason`` and ``_failure_terminal_reason``).

The record is a directory with ``events.jsonl`` (the session's own event
stream, via ``headless_print.printable_event`` -- the same projection ``exec
--json`` prints), ``outcome.json``, ``score.json``, and -- while the run is
live -- ``seal.reserve``, the pre-allocated margin the seal spends when the
volume is full. It is deliberately NOT the sealed evidence-bundle format:
converging the pilot's record onto the campaign format is the next arm's
decision, and claiming comparability early would be the one thing a benchmark
must not do.

DURABILITY IS PART OF THE RECORD, not a wrapper around it (``record_sink``
carries the full account and the measurements): a shared volume that fills
mid-run may cost the record's tail, but it must never again cost the run. A
torn record still returns the run's real status, steps and score, and says so
in ``record_incomplete``/``record_diagnostic`` -- measured 2026-09-28, two
episodes whose interactions completed were sealed ``failed / steps: 0`` by a
single sink write meeting ENOSPC, destroying $2.8 of committed spend and
reading as "the arm failed to act".
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from local_operator.owned_process import OwnerStatusChannel

from local_operator import sdk
from local_operator.evaluation.action_server import (
    SERVER_NAME,
    WIRE_READ_LIMIT_BYTES,
    decode_call,
    encode_response,
    surface_to_json,
)
from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AskUserExchangeParams,
    AskUserExchangeResult,
    CleanupParams,
    CloseParams,
    ExecuteParams,
    Handshake,
    InspectRequirementsParams,
    ObserveParams,
    PrepareParams,
    RequirementsResult,
    RescueDescriptor,
    ResetStartParams,
    ResolvedSecret,
    ScoreParams,
)
from local_operator.evaluation.adapters.supervisor import (
    AdapterSupervisor,
    HostVerifier,
    VerifiedAdapterSession,
    discard_rescue,
    persist_rescue,
    run_rescue,
)
from local_operator.evaluation.lifecycle import (
    CleanupAction,
    CleanupPlan,
    aggregate_cleanup,
)
from local_operator.evaluation.protocol import ActionBatch, FinishAction, Observation
from local_operator.evaluation.record_sink import RecordSink, RecordSinkError
from local_operator.evaluation.runner.action_tool import (
    ACTION_TOOL_NAME,
    PendingObservationToken,
    _build_batch,
    _no_pending_refusal,
    _refusal,
)
from local_operator.evaluation.runner.completion import finish_claim
from local_operator.evaluation.runner.episode import (
    _PROVISIONAL_CLEANUP_ACTION,
    EpisodeConfig,
    EpisodeSpec,
    UndeclaredDisclosedInfra,
    _observation_phase_failure,
    _rebindable_failure,
    _terminal_kind,
)
from local_operator.evaluation.runner.model import EpisodeTurn
from local_operator.evaluation.runner.provider_client import (
    DEFAULT_KEEP_RECENT_FRAMES,
    DEFAULT_REBUILD_EVERY_FRAMES,
    _ContextBuilder,
    build_completion_challenge,
)
from local_operator.evaluation.runner.public_reply import tolerated_fields_note
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    TextContent,
    ToolExecutionEndEvent,
    TurnEndEvent,
)
from local_operator.headless_print import printable_event
from local_operator.incidents import classify_incident, render_cut_off_reason
from local_operator.mcp.config import load_all_mcp_configs
from local_operator.mcp.tool_bridge import create_mcp_tool_name
from local_operator.session.request_observation import RequestFinalObservation
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec

#: The engagement mode's identifier in the run script, and in every record this
#: module writes. Spelled once so a comparison can never read two spellings of
#: the same arm as two arms.
SESSION_ARM_ID = "session"

#: How the episode's session is asked to behave. ``auto()`` is exec ``--yolo``
#: semantics -- the same posture the repo's own benchmark drives the product
#: with (``docs/BENCHMARKS.md``: "one ``local_operator.cli exec --json --yolo``
#: run per task"), and the honest posture for an unattended episode: a gate
#: that refuses a call "because nobody can be asked" would be measuring the
#: harness's refusal, not the model. The reply-channel arm has no gates at
#: all, so auto() is also its closest parity.
SESSION_APPROVALS = ApprovalPolicy.auto()

#: The first prompt's protocol paragraph. The action tool's own description
#: (shared with the loop-driven tool) carries the per-call contract; this adds
#: only what the episode knows: which tool name the actions reached the session
#: as, and how the episode ends. Everything else the model needs is on the
#: schema.
PROMPT_HEADER = (
    "Act on this computer by calling `{tool_name}`: each call carries ONE batch "
    "of actions for the screen you were just shown and returns the state that "
    "results. Send exactly one call per turn. When the task is complete, call "
    "`{tool_name}` with a single finish action.\n"
)

#: The sentence a terminal batch's tool result carries. Stated as the reason
#: the episode is ending rather than as a refusal, because a ``finish`` batch
#: is a deliberate ending -- the runner's step loop returns on it without ever
#: executing it (``_step_loop``), and the pilot keeps that semantic.
FINISH_ACK = (
    "Episode finished: the finish action was recorded and the run will be "
    "scored on the state reached. Do not call this tool again."
)

#: The step budget's terminal sentence. The runner treats reaching ``max_steps``
#: as a TRUNCATION of a normal, still-scored episode (``_step_loop``); this is
#: the model-facing half of that same decision, and the driver records the
#: truncation itself.
BUDGET_ACK = (
    "Episode step budget reached: the environment will not accept further "
    "actions and the run will be scored on the state reached. Do not call this "
    "tool again."
)

#: The ``terminal_reason`` a bridge records when its observation binding could
#: not be re-established after a failed execute (the wedge). The driver maps it
#: to ``status: "failed"`` -- NOT ``agent_stop``, which reads as a capability
#: reading and forced every F1 triage to be re-derived from ``events.jsonl`` by
#: hand. Named once; the record, the outcome and the tests all read it here.
BRIDGE_WEDGED_TERMINAL = "bridge-wedged"

#: The sentence a re-established binding delivers. The model's own batch ran
#: (the adapter declared it committed) but the screen it produced is the one
#: read-back that was lost, so the recovery must NOT imply the batch failed --
#: and it must tell the model how to get current: a ``wait`` batch changes
#: nothing and its own read-back returns the live screen.
RECOVERY_AFTER_READBACK_LOSS = (
    "Recovered: your last batch ran, but the environment could not return the "
    "screen it produced (the observation read failed after its retries). The "
    "episode continues. Do not assume that batch failed and do not repeat it "
    "blindly; re-read the current state before acting again -- a `wait` batch "
    "changes nothing and returns the current screen."
)

#: The completion challenge's channel sentence -- the ONE part of the shared
#: challenge text (``provider_client.build_completion_challenge``) that states
#: HOW to answer. The reply channel's answer is a JSON batch bound to an
#: observation id; a session's answer is another call to the action tool, so
#: this is the sentence that changes. Everything above it -- the claim named
#: as a CLAIM, the task restated, the end state named as the only evidence --
#: is byte-identical across the two channels, which is what keeps their
#: challenges comparable.
CHALLENGE_REPLY_GUIDANCE = (
    "Reply with a single `{tool_name}` call carrying either the same finish "
    "action, unchanged, or your corrective batch -- and nothing else."
)

#: VETO shapes consulted before the assertion shapes below: a message that
#: announces work still to come, limits the finished scope to a sub-step, or
#: narrates a SUB-TASK's completion is mid-work narration even when it
#: contains a completion word -- "I have finished the first two files and
#: will continue with the rest", "The work is done for this step", "The first
#: chart is complete". Found by the round-1 review of this PR (the first
#: predicate fired on all eight of its adversarial probes); all eight shapes
#: are pinned in ``tests/unit/evaluation/test_session_arm.py``.
_PROSE_MORE_WORK_PENDING_RE = re.compile(
    r"\b(?:"
    r"will\s+continue|continuing|continues?\s+(?:with|to)|continued\s+(?:with|to)|"
    r"still\s+(?:need|needs|needed|requires?|required|working\s+on)|"
    r"not\s+yet|in\s+progress|moving\s+(?:to|on)|next\s+up|"
    r"starting\s+(?:the|it|now|on|with|next)"
    r")\b",
    re.IGNORECASE,
)
#: "... for this step", "for now", "so far" scope the completion to a
#: sub-step the message itself names.
_PROSE_PARTIAL_SCOPE_RE = re.compile(
    r"\b(?:for\s+(?:this|the)\s+(?:step|stage|phase|moment)|for\s+now|so\s+far)\b",
    re.IGNORECASE,
)
#: "the first chart is complete", "the first file has been completed": the
#: completion phrase binds an ordinal SUB-TASK, not the episode's task.
_PROSE_SUB_STEP_CLAIM_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|initial)\s+(?:\w+\s+){0,3}"
    r"(?:is|are|has|have|was|were)\s+(?:(?:now|fully|finally|just)\s+)?"
    r"(?:been\s+)?(?:complete|completed|done|finished)\b",
    re.IGNORECASE,
)

#: The prose arm's assertion shapes (see :func:`prose_claims_completion`). A
#: POSITIVE list on purpose: a terminal message that matches nothing is left
#: alone, and the shapes below are the ones the sealed session-arm corpus
#: actually exhibits. Each pattern is pinned by a unit test against the real
#: sample it was calibrated from where the corpus has one, and every shape in
#: the veto set above is pinned against its measured probe. The residual
#: over-fire class is named in the detector's docstring, not hidden.
_PROSE_COMPLETION_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Done. ...", "**Done.**", "All done." -- the corpus's dominant shape.
    # "Done WITH/FOR ..." qualifies a sub-object ("Done with the first
    # document"), so it is not a whole-task claim.
    re.compile(r"^[^\w\r\n]{0,8}(?:all\s+)?done\b(?!\s+(?:with|for)\b)", re.IGNORECASE),
    # "the task is complete", "the work is done", "task completed".
    re.compile(
        r"\b(?:the\s+)?(?:task|job|work|assignment)\s+"
        r"(?:is\s+|is\s+now\s+|now\s+)?(?:complete|completed|done|finished)\b",
        re.IGNORECASE,
    ),
    # "the route is complete and displayed", "the document content is complete".
    # Subject-agnostic by design (which nouns are deliverables is not decidable
    # from one message); the vetoes above keep ordinal sub-tasks and next-work
    # tails out, and a ", <verb>ing ..." tail ("is done, unpacking it now") is
    # a continuation clause, not a sentence end.
    re.compile(
        r"\b(?:is|are)\s+(?:now\s+|fully\s+|finally\s+)?(?:complete|completed|done|finished)\b"
        r"(?!\s*[,;]\s*(?!including\b)(?:\w+ly\s+)?\w+ing\b)",
        re.IGNORECASE,
    ),
    # "I have completed", "I've finished", "I completed / finished ..." -- the
    # 've contraction takes no whitespace after "I". The phrase must bind a
    # whole-task object (or end the sentence): "I have finished the first two
    # files" / "I have completed the initial setup" are progress reports.
    re.compile(
        r"\b(?:i\s+have|i've|i)\s+(?:now\s+)?(?:completed|finished)\b"
        r"(?:\s+(?:the\s+|all\s+)?(?:task|job|work|assignment|deliverable)s?\b"
        r"|\s+everything\b|\s+it\s+all\b|(?=\s*(?:[.!?;:]|$)))",
        re.IGNORECASE,
    ),
    # "completed the task", "finished all deliverables", "finished everything".
    re.compile(
        r"\b(?:completed|finished)\s+"
        r"(?:(?:the\s+|all\s+)?(?:task|job|work|assignment|deliverable)s?"
        r"|everything|it\s+all)\b",
        re.IGNORECASE,
    ),
    # "the task has been completed", "everything has been completed" -- the
    # subject must be the whole task ("the first file has been completed" is
    # a progress report).
    re.compile(
        r"\b(?:(?:the\s+|all\s+)?(?:task|job|work|assignment|deliverable)s?"
        r"|everything|it\s+all)\s+(?:has|have)\s+(?:now\s+)?been\s+"
        r"(?:completed|finished)\b",
        re.IGNORECASE,
    ),
)

#: ``FinishAction.reason``'s field bound (``protocol.py``); a prose claim is
#: synthesised into one, so it takes that field's bound rather than inventing
#: a second one.
_PROSE_CLAIM_MAX_CHARS = 10_000


def prose_claims_completion(text: str) -> bool:
    """Whether a TERMINAL prose message asserts the episode's task is finished.

    WHY THIS EXISTS. The completion gate refuses an unverified ``done`` claim
    once and re-asks the model to compare its claim against the screen -- but
    the gate shipped in #1696 fires inside ``ActionBridge.call``, so it only
    sees TOOL-mediated claims. The first field run of arm 1748 (task_003)
    ended its final answer as prose -- "Done. Summary of what I determined
    and did: ..." with NO tool call -- and the turn simply ended
    (``agent_stop``): the gate never fired, and the run reads as an unverified
    finish, which is exactly what the gate exists to prevent. This predicate is
    the gate's answer-side detector for that terminal message (see
    ``ActionBridge.prose_completion_challenge`` for when it is consulted).

    WHAT IT IS AND IS NOT. A narrow positive list of assertion shapes the
    corpus actually exhibits -- not a general "did the model succeed"
    classifier, and it cannot be one: a terminal prose message carries no
    other signal that separates a completion claim from mid-work narration or
    a plain answer. Precision is carried by the veto set above (next-work
    announcements, partial scopes, ordinal sub-tasks) and by the whole-task
    objects the assertion shapes bind; the measured boundary is pinned by the
    discrimination tables in ``tests/unit/evaluation/test_session_arm.py``.
    The trade, in both directions:

    * a FALSE POSITIVE costs the one bounded challenge cycle the gate was
      entitled to anyway, and the challenge itself names continuing as a
      legitimate reply ("a batch of actions that closes the gap");
    * a FALSE NEGATIVE leaves exactly today's behaviour -- the ungated end --
      so this list may lag a phrasing the corpus grows into, and the unit test
      table is where a new real sample lands before it is added here.

    One residual over-fire class is KNOWN and deliberate, named here rather
    than left implicit: a subject-agnostic "X is done/complete/finished" whose
    X is a non-ordinal sub-object and whose message carries no next-work
    clause ("The download is done.") still reads as a claim. Which nouns are
    deliverables is not decidable from one message, and every narrowing tried
    against the corpus's real samples cost a genuine claim shape ("the route
    is complete and displayed") -- the cost of the residual is the one
    bounded cycle, not a wrong score. The detector is English-language and
    structural (word shapes, not semantics) and is consulted ONLY for an
    episode's terminal message, so mid-run narration normally never reaches
    it at all.
    """

    if (
        _PROSE_MORE_WORK_PENDING_RE.search(text)
        or _PROSE_PARTIAL_SCOPE_RE.search(text)
        or _PROSE_SUB_STEP_CLAIM_RE.search(text)
    ):
        return False
    return any(pattern.search(text) for pattern in _PROSE_COMPLETION_CLAIM_PATTERNS)


class SessionArmError(RuntimeError):
    """A precondition of the session engagement the caller must fix."""


# ---------------------------------------------------------------------------
# The MCP declaration: one server, in the episode's own scratch root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionServerDeclaration:
    """What was declared: the identity and the exact argv the session dials."""

    server_name: str
    tool_name: str
    config_path: Path
    endpoint: Path
    command: str
    args: tuple[str, ...]


def declare_action_server(
    *,
    config_dir: Path,
    endpoint: Path,
    surface: ActionSurface,
    python_executable: str | None = None,
    cwd: Path | None = None,
) -> ActionServerDeclaration:
    """Write the episode's ``mcp.json`` declaration; return its resolved facts.

    The declaration goes in the SESSION ROOT's own user config file
    (``<config_dir>/mcp.json``), which is the file
    ``mcp/config.load_all_mcp_configs`` reads as the user-scope source. Writing
    it anywhere else -- a shared file, the operator's tree, or the cwd's
    project slot -- would make the episode's tool surface depend on something
    outside ``SessionRoots``; the caller asserts that consequence with
    :func:`assert_declaration_resolved`.

    An EXISTING file is merged, not replaced: a resumed episode or a future
    multi-server episode keeps its other declarations, and only the one name
    this module owns is replaced. The write is atomic (same-directory temp +
    ``os.replace``), the discipline ``mcp/config.py`` itself uses, so a crash
    can never leave a half-written file that the session would read as "no
    servers".
    """

    executable = python_executable or sys.executable
    tool_name = create_mcp_tool_name(SERVER_NAME, ACTION_TOOL_NAME)
    args = [
        "-m",
        "local_operator.evaluation.action_server",
        "--endpoint",
        str(endpoint),
        "--surface",
        surface_to_json(surface),
    ]
    entry: dict[str, Any] = {
        "type": "stdio",
        "command": executable,
        "args": args,
        # The action server must be visible WITHOUT a discovery read: the
        # episode's whole point is that the model acts, and a tool the model
        # cannot see is a tool it does not use (``mcp/config.py``'s own reason
        # for ``preloadTools``). The allowlist pins the surface to the one
        # tool, so a future server-side addition cannot silently widen an
        # episode's tools.
        "preloadTools": True,
        "enabledTools": [ACTION_TOOL_NAME],
        # The surface IS the measurement: a delegated child driving it (or
        # ending it) would make the record unattributable, so the declaration
        # reserves it for the episode's own turn -- children do not inherit it
        # and the harness refuses an execution carrying a child's job id.
        "ownTurnOnly": True,
    }
    if cwd is not None:
        entry["cwd"] = str(cwd)

    config_path = Path(config_dir) / "mcp.json"
    document: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SessionArmError(
                f"the episode's mcp.json at {config_path} is not readable JSON, so the "
                f"action server cannot be declared without discarding it: {error}"
            ) from error
        if not isinstance(existing, dict):
            raise SessionArmError(f"the episode's mcp.json at {config_path} is not an object")
        document = existing

    servers = document.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_NAME] = entry
    document["mcpServers"] = servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = config_path.with_name(f".{config_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(config_path)

    return ActionServerDeclaration(
        server_name=SERVER_NAME,
        tool_name=tool_name,
        config_path=config_path,
        endpoint=Path(endpoint),
        command=executable,
        args=tuple(args),
    )


def assert_declaration_resolved(
    *,
    cwd: Path,
    config_dir: Path,
    scratch_root: Path,
    server_name: str = SERVER_NAME,
) -> Mapping[str, str]:
    """Prove, by resolved path, that the episode's MCP graph is inside scratch.

    THE RECORDED MUST (PR 1's QA round-1 checklist, design §5): MCP discovery
    is cwd-scoped, so an episode held at the operator's home would enumerate
    and dial the operator's real servers. This is the assertion that makes the
    consequence checkable at run time and in tests:

    * the ``<cwd>/.local-operator/mcp.json`` project slot resolves inside the
      scratch root (the comment above states why cwd may never be elsewhere);
    * the user slot resolves to the episode's own config file;
    * every SOURCE the discovery actually read is under the scratch root -- the
      check that catches an import (``~/.claude.json``, ``~/.cursor/mcp.json``,
      ``~/.codex/config.toml``) resolving somewhere real through a redirected
      or unredirected home.

    Returns ``(name -> source path)`` for the record; raises otherwise.
    """

    scratch = Path(scratch_root).resolve()
    resolved_cwd = Path(cwd).resolve()
    resolved_config = Path(config_dir).resolve()
    if not (resolved_cwd == scratch or resolved_cwd.is_relative_to(scratch)):
        raise SessionArmError(
            f"the episode's cwd {resolved_cwd} is not inside the scratch root {scratch}; "
            "MCP discovery is cwd-scoped and would read the operator's own tree"
        )
    if not (resolved_config == scratch or resolved_config.is_relative_to(scratch)):
        raise SessionArmError(
            f"the episode's config dir {resolved_config} is not inside the scratch root "
            f"{scratch}; the action server would be declared outside the episode"
        )

    configs, sources = load_all_mcp_configs(resolved_cwd)
    outside = {
        name: source
        for name, source in sources.items()
        if not Path(source).resolve().is_relative_to(scratch)
    }
    if outside:
        raise SessionArmError(
            f"this episode's MCP configuration resolves outside its scratch root: {outside}"
        )
    if server_name not in configs:
        raise SessionArmError(
            f"the action server {server_name!r} was not discovered from {resolved_cwd}; "
            "the declaration and the session's cwd disagree"
        )
    return sources


# ---------------------------------------------------------------------------
# Rendering: one definition of what the model is shown
# ---------------------------------------------------------------------------


class ObservationRenderer:
    """An observation as model content, rendered by the reply arm's builder.

    The reply-channel arm renders observations through
    ``provider_client._ContextBuilder`` (text lines + frame image blocks read
    through ``verify_artifact``). The session arm must not grow a SECOND
    definition of "what the model is shown" -- the two arms would drift on
    exactly the facts a comparison depends on -- so the builder itself is
    reused here: one turn appended per observation, and the newest message IS
    the content for the prompt (observation zero) or the tool result (every
    later observation).
    """

    def __init__(self, artifact_root: Path) -> None:
        self._builder = _ContextBuilder(
            artifact_root=artifact_root,
            keep_recent_frames=DEFAULT_KEEP_RECENT_FRAMES,
            rebuild_every_frames=DEFAULT_REBUILD_EVERY_FRAMES,
        )
        self._turns: list[EpisodeTurn] = []

    def render(self, observation: Observation) -> list[Any]:
        """The observation's content blocks, in the builder's own rendering."""

        self._turns.append(EpisodeTurn(observation=observation))
        self._builder.append_new_turns(self._turns)
        message = self._builder.messages[-1]
        return list(message.content)


def split_prompt_content(blocks: Sequence[Any]) -> tuple[str, list[ImageContent]]:
    """Split rendered content into (text, images) for ``Session.prompt``.

    ``TextContent`` and ``ImageContent`` are the only two shapes this seam has
    ever forwarded. NOTHING ELSE IS TOUCHED, deliberately: this splitter's
    target is ``Session.prompt(text, images)`` and the evaluation arm captures
    screenshots, never recordings, so an ``AudioContent`` (or any block a
    future union adds) is dropped rather than reaching ``block.text`` — before
    this guard the first audio block raised ``AttributeError``, invisible to
    pyright behind the ``Sequence[Any]`` parameter (agent review round 1,
    queued latent crash).
    """

    texts: list[str] = []
    images: list[ImageContent] = []
    for block in blocks:
        if isinstance(block, ImageContent):
            images.append(block)
        elif isinstance(block, TextContent):
            texts.append(block.text)
    return "\n".join(texts), images


def _bounded_cause(error: BaseException, *, limit: int = 300) -> str:
    """``Class: message``, bounded -- the one rendering the wedge sentences,
    the record payloads and the outcome diagnostic share.

    Never a repr and never a traceback: the adapter's own discipline is that
    its error text is canary-checked and safe to show, and a raw exception's
    repr is the thing that is not (it can embed a literal credential).
    """

    return f"{type(error).__name__}: {error}"[:limit]


def _bridge_wedge_sentence(
    error: BaseException,
    *,
    committed: bool,
    recovery_attempted: bool,
    recover_error: BaseException | None,
) -> str:
    """The one sentence a wedge ends with -- model-facing, record and outcome.

    It must say WHICH failure wedged the bridge and WHAT was tried, because
    the ``end_requested``-only shape this replaces (an ``agent_stop``) left
    every reader to reconstruct the cause from ``events.jsonl`` by hand. The
    three branches mirror the three states the recovery can end in; the
    sentence is deliberately singular so the transcript, the record and the
    outcome cannot drift into three different stories.
    """

    if recover_error is not None:
        attempted = (
            "a re-read of the environment's current observation failed too "
            f"({_bounded_cause(recover_error)})"
        )
    elif recovery_attempted:
        attempted = "the re-read returned no observation"
    elif committed:
        attempted = (
            "no re-read path is wired into this build, so the binding cannot " "be re-established"
        )
    else:
        attempted = (
            "the failure did not declare the batch committed, so the batch's "
            "state is ambiguous and re-reading is not safe"
        )
    return (
        f"action bridge wedged by a transport error ({_bounded_cause(error)}); "
        f"{attempted}. The observation binding could not be re-established, so "
        "no further batch -- including a finish -- can run and the episode is "
        "ending."
    )


# ---------------------------------------------------------------------------
# The bridge: the episode half of the action wire
# ---------------------------------------------------------------------------

#: Called for every executed batch; the driver records it. Receives the batch
#: and the adapter's result.
RecordBatch = Callable[[str, Mapping[str, Any]], None]

#: The execution seam. The driver fills it with the adapter call plus its
#: read-back recovery; tests fill it with anything that looks like
#: ``ExecuteResult``.
ExecuteBatch = Callable[[ActionBatch], Awaitable[Any]]

#: The re-bind seam: one bounded attempt to fetch the observation the episode
#: may continue from, called by the bridge only after an execute failure the
#: adapter declared COMMITTED (``_observation_phase_failure``). The driver
#: fills it with the read-only ``observe`` call; tests fill it with a fake.
#: ``None`` means the bridge has no re-read path and a failed execute is
#: terminal -- legibly (see ``ActionBridge._after_failed_execute``).
RecoverObservation = Callable[[], Awaitable[Observation]]


def _split_ask_delivery(delivery: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The ask seam's return, split into (canonical receipt, answer text).

    The real seam returns the adapter's ``AskUserExchangeResult`` -- the
    canonical public receipt (ask_id, request digest, accepted/declined,
    public answer). Scripted doubles may return a bare answer string (no
    receipt exists at that seam, and none is invented); ``None`` is a
    declined/empty exchange either way.
    """

    if isinstance(delivery, AskUserExchangeResult):
        return delivery.model_dump(mode="json"), delivery.answer
    if delivery is None:
        return None, None
    return None, str(delivery)


def _ask_delivery_sentence(answer: str | None, receipt: dict[str, Any] | None) -> str:
    """The ask exchange's report, as the model reads it on the tool result.

    Written to be honest in BOTH directions: an empty/declined exchange says so
    explicitly (never an invented answer, never silence), and a real answer is
    quoted verbatim -- the answer text is the exchange's product and the whole
    point of carrying it into the next root request.

    THE RECEIPT RIDES THE TEXT (R4, review round 1). The MCP forwarding layer
    delivers only the reply's CONTENT blocks to the next root request -- the
    bridge-local ``details`` never reach the model -- so the canonical
    exchange receipt is rendered INTO the text when one exists (a bare
    scripted answer has no receipt to render).
    """

    if answer is None:
        sentence = (
            "The simulated user did not provide an answer for this ask (declined "
            "or empty); continue with what you know and do not repeat the same "
            "question."
        )
    else:
        sentence = f"The simulated user answered: {answer}"
    if receipt is not None:
        sentence += "\nask_user exchange receipt: " + json.dumps(
            receipt, sort_keys=True, ensure_ascii=False
        )
    return sentence


@dataclass
class ActionBridge:
    """One episode's action bridge: token, budget, execution, refusals.

    The state machine is ``runner/action_tool.py``'s, per the design's rule
    that the session arm keeps the SAME action contract: one batch per turn,
    validate-before-consume, frames rendered by the one renderer, refusals
    carrying the shared vocabulary's class and hint. What is different is only
    WHERE it lives -- behind an MCP tool call instead of a loop-injected tool
    -- because a session cannot have tools injected into its LoopContext by a
    consumer (that would be exactly the harness coupling the design forbids).

    One deliberate ADDITION to that state machine: the completion gate
    (:attr:`completion_gate`, the runner's own contract), because the first
    real-task run measured what its absence costs -- a session that filled a
    form correctly, claimed ``finish`` without submitting, and scored binary 0
    where the reply channel's identical answers scored 1.0 through its
    challenge.
    """

    endpoint: Path
    surface: ActionSurface
    render: Callable[[Observation], list[Any]]
    execute: ExecuteBatch
    max_steps: int
    record: RecordBatch | None = None
    #: The ask seam. The real driver returns the adapter's canonical
    #: ``AskUserExchangeResult`` (receipt + answer); scripted doubles may
    #: return a bare answer string, which the bridge renders without inventing
    #: a receipt (see ``_split_ask_delivery``).
    ask: Callable[[ActionBatch], Awaitable[Any]] | None = None
    #: The re-bind seam (see :data:`RecoverObservation`): exactly ONE read of
    #: the environment's current observation, attempted only after an execute
    #: failure the adapter declared committed. The bridge owns WHEN; the
    #: driver owns the RPC. ``None`` degrades a failed execute to a legible
    #: end rather than a silent one.
    recover: RecoverObservation | None = None
    #: The task as it was stated -- the reset observation's text -- restated by
    #: the completion challenge (the same source the runner's gate uses).
    instruction: str = ""
    #: The runner's completion-gate controls, carried here so the session path
    #: has the IDENTICAL contract (see ``runner/completion.py``): at the default
    #: ON, the first ``done`` claim is challenged once and the second is always
    #: accepted, so the gate can never trap an episode in a challenge loop. A
    #: control arm flips the gate rather than the code path: the driver's
    #: ``--no-completion-gate`` disables the gate for BOTH channels, and the
    #: ``completion_challenges`` config bound applies to both (it has no CLI
    #: flag).
    completion_gate: bool = True
    completion_challenges: int = 1
    #: The channel sentence appended to the challenge -- see
    #: :data:`CHALLENGE_REPLY_GUIDANCE`.
    reply_guidance: str | None = None

    #: Set when the bridge has decided the episode should end (``finish``, the
    #: step budget). The driver reads it after the turn: a reason here is the
    #: episode's own terminal, and the driver will not re-prompt.
    end_requested: str | None = None
    #: Set beside ``end_requested`` when the terminal is the bridge's OWN (a
    #: wedge, :data:`BRIDGE_WEDGED_TERMINAL`): the one sentence naming the
    #: transport failure and what was attempted. The driver folds it into the
    #: outcome's ``diagnostic`` so the record and the driver's summary agree on
    #: WHY the run ended, instead of both saying only that it ended.
    end_diagnostic: str | None = None

    def __post_init__(self) -> None:
        # The token is armed with observation zero, which only exists once the
        # environment has reset -- so the bridge is constructed ARMED LATER
        # (``arm``) rather than with a placeholder. Serving before arming is a
        # driver bug and is answered as a plain refusal, not an exception into
        # the socket handler.
        self._token: PendingObservationToken | None = None
        self._steps = 0
        # THE ATTEMPTED OUTPUT-GROUP CLOCK (audit §45-49). Every COMPLETED
        # root output group -- a world batch, a refused frame/schema attempt,
        # a local-tool-only turn, an ask/finish proposal, a committed
        # read-back loss or terminal prose -- reserves one slot here, keyed
        # by the generated assistant message that produced it. Kept SEPARATE
        # from ``_steps`` (the admitted/bound-world epoch the public
        # ``steps`` still reports, unchanged): this is the clock the horizon
        # is ENFORCED on, so a run cannot spend a whole budget on groups the
        # world never admitted. No second ledger: its rows go to the same
        # RecordSink below.
        self._attempted_groups = 0
        self._attempted_keys: set[str] = set()
        #: The group currently being served -- its key, the outcome kind its
        #: row will carry, and whether that row was already written.
        self._attempted_current: str | None = None
        self._attempted_kind: str | None = None
        self._attempted_recorded = False
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._closed = False
        #: How many completion challenges this episode has made; bounded by
        #: ``completion_challenges``. The counter lives HERE, not on the record,
        #: for the runner's reason: the bound is a property of the episode's
        #: protocol, not of any writer.
        self._completion_fired = 0
        #: ``(observation_id, rendered blocks)`` for every screen this bridge
        #: has shown: observation zero in ``arm``, each executed result in
        #: ``call``. The completion challenge re-attaches the state the model is
        #: looking at from HERE instead of calling the renderer again -- the
        #: renderer is stateful and a second render would append a duplicate
        #: turn to the transcript the model is shown.
        self._initial_shown: tuple[str, list[Any]] | None = None
        self._last_shown: tuple[str, list[Any]] | None = None
        #: The terminal assistant message the prose arm reads. Updated by
        #: ``fold`` on every ``MessageEndEvent``, so once the turn ends this is
        #: the message the run ended on -- a prose "Done" never reaches
        #: ``call``, so the event stream is the only place the bridge can see
        #: it (see ``prose_completion_challenge``).
        self._last_assistant_message: Message | None = None
        #: The ask answer's delivery, staged by ``call`` for THIS call only and
        #: consumed by whichever result path answers it (the normal return, a
        #: recovered read-back, or the wedge end): the adapter's answer text
        #: and exchange receipt must ride the NATIVE tool result -- the content
        #: the next root request re-reads -- instead of being discarded at the
        #: ask call (the QA bridge matrix's ask-sentinel defect).
        self._pending_ask_delivery: list[Any] = []
        self._pending_ask_receipt: dict[str, Any] | None = None
        #: The most recent observation, for the prose challenge's frame
        #: re-attachment. `arm` holds observation zero; every executed batch
        #: replaces it, so it is the observation ``_last_shown`` was rendered
        #: from and the one a corrective reply binds to.
        self._last_observation: Observation | None = None

    def arm(self, observation: Observation) -> None:
        """Arm the token with the episode's first observation.

        The first screen is rendered here, ONCE, and kept: the prompt is built
        from :attr:`initial_blocks`, and the completion challenge re-attaches
        the same blocks when the episode's first call is a finish.
        """

        if self._token is not None:
            raise SessionArmError("the action bridge was already armed with an observation")
        self._token = PendingObservationToken(observation)
        self._initial_shown = (observation.observation_id, list(self.render(observation)))
        self._last_observation = observation

    @property
    def initial_blocks(self) -> list[Any]:
        """The first screen's rendered content, as the prompt shows it."""

        if self._initial_shown is None:
            raise SessionArmError("the action bridge has no first screen; arm it first")
        return self._initial_shown[1]

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def attempted_groups(self) -> int:
        """Completed root output groups reserved on the ATTEMPTED clock.

        The horizon the episode is ENFORCED on (see ``call``); the public
        ``steps`` stays the admitted/bound-world epoch, unchanged, so every
        existing reader of the outcome and records keeps its old meaning.
        """

        return self._attempted_groups

    def _group_key(self) -> str:
        """The canonical key of the output group being served.

        The assistant message that carried the calls IS the group: the fold
        stores it on ``MessageEndEvent`` before any of its tool calls execute,
        so every call of one model output reads the SAME id -- multi-call
        groups charge once and duplicate deliveries dedupe. No tool-name,
        model-name or task-id heuristic feeds this key (the audit's rule);
        the per-call fallback exists only for a driver bug (a call before any
        assistant message), where over-counting one group beats a silently
        free one.
        """

        message = self._last_assistant_message
        if message is None:
            return f"pre-message:{uuid.uuid4().hex[:12]}"
        return f"message:{message.id}"

    def _reserve_output_group(self) -> str | None:
        """Reserve the current output group on the attempted clock.

        Called at ``call`` ENTRY, BEFORE validation: the Nth group is
        reserved even when its batch is refused -- a schema/frame refusal
        after explicit action output consumed a model output group, and the
        runner charges those to the horizon even though they never bound a
        screen token (the token is untouched here).

        Returns the group key when the call may proceed, ``None`` when this
        is group N+1 (the budget is spent). A call whose group is ALREADY
        reserved -- a second tool call or action in the SAME output group --
        is never refused by this gate: the group was charged once and its
        remaining calls pair exactly as before.
        """

        key = self._group_key()
        if key in self._attempted_keys:
            return key
        if self._attempted_groups >= self.max_steps:
            return None
        self._attempted_keys.add(key)
        self._attempted_groups += 1
        self._attempted_current = key
        self._attempted_kind = None
        self._attempted_recorded = False
        return key

    def _mark_group_kind(self, kind: str) -> None:
        """Record the group's outcome kind -- FIRST writer wins.

        A same-group follow-up call (a second action of one model output, a
        duplicate delivery) must not rename the group: its late refusal would
        otherwise overwrite an admitted batch's ``batch`` kind with the
        refusal class. The first call's outcome describes how the group
        actually completed.
        """

        if self._attempted_kind is None:
            self._attempted_kind = kind

    def _reserve_message_group(self, message: Message) -> None:
        """Charge the completed root message's group at its own boundary.

        WHY THIS EXISTS BESIDE THE TURN-END CHARGE (R1 C1, measured on the
        real stack: ``wire_calls=2`` for a local-tool-only Nth group, with
        the extra turn's tool executing). The loop asks its graceful-cancel
        question at the end-of-tools boundary BEFORE it yields
        ``TurnEndEvent`` (``loop.py``: the check precedes the event). A stop
        requested when the sink sees that TurnEnd is therefore first READ at
        the next boundary -- one more root request went out. The completed
        message is the earliest boundary that precedes the loop's check:
        charging here -- before its calls are validated or dispatched --
        both keeps the attempted-group accounting identical for every
        call-set outcome and lets the sink's MessageEnd stop be read before
        the next root prediction. The Nth message's own calls still dispatch
        and pair; the graceful cancel is honoured only after the batch
        completes.

        TYPED, never guessed: a message whose ``stop_reason`` is ``"error"``
        did not complete (wordless provider fault) and is never charged;
        call-set kinds (batch/finish/ask/refusal) are written by the calls
        themselves exactly as before; the message id keeps multi-call and
        duplicate folds once.
        """

        if message.stop_reason == "error":
            return
        key = f"message:{message.id}"
        if key in self._attempted_keys:
            return
        if self._attempted_groups >= self.max_steps:
            return
        self._attempted_keys.add(key)
        self._attempted_groups += 1
        self._attempted_current = key
        self._attempted_kind = None
        self._attempted_recorded = False

    def _finalize_output_group(self) -> None:
        """Charge/record the turn's group at its end boundary.

        Two duties, one boundary (``TurnEndEvent``/``AgentEndEvent``):

        * a group the STREAM completed without any bridge call -- a
          local-tool-only turn or terminal prose -- was already charged at
          its message's own boundary (``_reserve_message_group``); its
          call-less kind (``local-tool``/``prose-terminal``) is stamped
          HERE, because the charge deliberately precedes any call-set
          classification, and this method is the next boundary that knows
          the message never reached ``call``;
        * the row of an already-reserved group is written here with the
          outcome kind its calls recorded (admitted batch, ask, finish,
          refusal class, read-back loss...).

        NOT a completed group: a turn whose assistant message's own typed
        ``stop_reason`` is ``"error"`` -- a wordless provider fault the loop
        may continue internally or end on. (The wire-level
        ``retryable_interruption`` marker it keys on lives upstream of this
        bridge; the message's stop_reason is the typed fact this layer sees.)
        Counting fault frames as output groups would charge the horizon for
        compute the model never got to complete; that is a TYPED fact, never
        a tool/model-name guess. An explicit next root request after a fault
        is a new group and charges normally; a read-back recovery inside one
        call is the same group, not another.
        """

        message = self._last_assistant_message
        # A turn whose assistant message carries the typed ``stop_reason``
        # "error" did not COMPLETE -- the loop may even continue it internally
        # (the wire-level ``retryable_interruption`` marker lives upstream of
        # this bridge; the message's own stop_reason is the typed fact this
        # layer sees). Non-completed faults are never output groups.
        faulted = message is not None and message.stop_reason == "error"
        if message is None:
            return
        key = f"message:{message.id}"
        if key in self._attempted_keys:
            # Charged at the message's own boundary (``_reserve_message_
            # group``): a call-less turn gets its typed kind NOW -- the
            # message carries the only facts the old turn-end charge read.
            # First-writer-wins keeps every call-set kind (batch, finish,
            # refusal class) exactly as the calls recorded it.
            if key == self._attempted_current:
                self._mark_group_kind("local-tool" if message.tool_calls else "prose-terminal")
            self._record_output_group()
            return
        if faulted:
            return
        if self._attempted_groups >= self.max_steps:
            return
        self._attempted_keys.add(key)
        self._attempted_groups += 1
        self._attempted_current = key
        self._attempted_kind = None
        self._mark_group_kind("local-tool" if message.tool_calls else "prose-terminal")
        self._attempted_recorded = False
        self._record_output_group()

    def _record_output_group(self) -> None:
        """Write the reserved group's row once, with its outcome kind.

        Same RecordSink ledger as every other row -- the admitted epoch's
        row and this one coexist; ``group_kind`` (never ``kind``: the row
        merge collision ``_consume_output_epoch`` documents) carries the
        per-kind reason.
        """

        if self._attempted_current is None or self._attempted_recorded:
            return
        self._attempted_recorded = True
        if self.record is not None:
            self.record(
                "attempted_group",
                {
                    "group_kind": self._attempted_kind or "unknown",
                    "attempted_groups": self._attempted_groups,
                    "message_id": self._attempted_current,
                },
            )

    @property
    def terminal(self) -> bool:
        return self._token is not None and self._token.terminal

    @property
    def last_assistant_message(self) -> Message | None:
        """The last assistant message the stream ended on.

        ``fold`` keeps it for the prose arm; the driver reads it back here so
        an ``agent_stop``'s ``terminal_reason`` can name the message the turn
        ended on without re-reading the record's ``events.jsonl``.
        """

        return self._last_assistant_message

    @property
    def completion_challenges_fired(self) -> int:
        """How many completion-gate challenges this episode delivered.

        One counter for BOTH arms -- the finish-call challenge and the prose
        claim's -- exactly as the budget is one, so a reader of an
        ``agent_stop`` can tell an ending that survived the gate from one the
        gate never saw.
        """

        return self._completion_fired

    def _shown_blocks(self, observation: Observation) -> list[Any]:
        """The rendered state a finish can bind to, for the challenge.

        Every observation a finish can bind to was rendered once -- observation
        zero by ``arm``, each executed result by ``call`` -- so the match is
        total by construction. The empty fallback exists only so a future
        arming path cannot make the bridge raise inside a socket handler; it is
        not a state either shipped path can produce.
        """

        for shown in (self._last_shown, self._initial_shown):
            if shown is not None and shown[0] == observation.observation_id:
                return shown[1]
        return []

    def fold(self, event: AgentEvent) -> None:
        """Fold session events; the turn boundary re-arms the token.

        Folding is the DRIVER's job (as in the loop-driven design): the
        engine does not know the token exists, so the arming happens on
        ``TurnEndEvent`` and the next call reads the result. The prose arm of
        the completion gate reads the stream from the same seam: the terminal
        assistant message is the only copy of a prose "Done" the bridge can
        see, because a message without a tool call never reaches ``call``.
        """

        if self._token is not None:
            self._token.fold(event)
        if isinstance(event, MessageEndEvent):
            message = event.message
            # ``AgentMessage`` is a union (``Message | CustomMessage``); the
            # prose arm only ever reads a real assistant ``Message``, and the
            # isinstance check is what tells the type checker so.
            if isinstance(message, Message) and message.role == "assistant":
                self._last_assistant_message = message
                # EARLIEST boundary, BEFORE this message's calls are
                # validated or dispatched: the charge lands here so the
                # sink's MessageEnd stop (in ``run_session_episode``) is
                # read by the loop's own boundary check before it spends
                # the NEXT root request. A turn-end charge was first read
                # one boundary late and let a local-tool-only Nth group's
                # request through (R1 C1: wire_calls=2, the extra tool
                # executed).
                self._reserve_message_group(message)
        if isinstance(event, (TurnEndEvent, AgentEndEvent)):
            self._finalize_output_group()

    def prose_completion_challenge(self) -> list[Any] | None:
        """The gate's re-prompt for a terminal message that CLAIMS completion.

        THE BYPASS THIS CLOSES. The gate inside ``call`` only ever sees a claim
        the model made THROUGH the action tool. Arm 1748's first field run
        (task_003) ended its final answer as prose -- "Done. Summary of what I
        determined and did: ..." with no tool call at all -- and the turn
        simply ended: ``agent_stop``, no challenge, no chance for the model to
        compare its claim against the screen. This method is the gate's
        answer-side arm; the driver calls it the moment the turn ends, and the
        challenge itself -- text builder, reply guidance, bound -- is the ONE
        already shipped on this channel, so the two arms cannot drift.

        WHEN IT FIRES (all required, each load-bearing):

        * the episode has NOT already ended (``end_requested is None`` and the
          token is not terminal): a finish or a truncation has its own ending,
          and the summary prose that often FOLLOWS an accepted finish (every
          completed run in the corpus ends that way) must not earn a second
          exchange;
        * the gate is enabled and the SHARED budget is not spent -- the prose
          arm draws from the same ``completion_challenges`` counter as the
          finish-call arm, so one episode can never exceed the configured
          number of challenges across BOTH paths;
        * the terminal assistant message carries NO tool call -- a message with
          one is ``call``'s business, not this arm's;
        * its text matches :func:`prose_claims_completion` -- which the
          round-1 review's eight adversarial narration shapes are the pinned
          regression table for: next-work announcements, partial-scope
          statements and ordinal sub-tasks do not fire, nor do plain answers
          or the silent-provider ending class's empty terminal messages (the
          residual class is named in the detector's docstring).

        RETURNS the challenge content (the challenge text plus the SAME
        rendered blocks the model was last shown, re-attached -- never
        re-rendered) for the driver to deliver as one harness-injected user
        turn, or ``None`` when the gate does not apply. The counter and the
        record are updated HERE, so a firing is counted and auditable even if
        the delivery then fails.
        """

        if self.end_requested is not None:
            return None
        if self._token is None or self._token.terminal:
            return None
        if not self.completion_gate or self._completion_fired >= self.completion_challenges:
            return None
        message = self._last_assistant_message
        if message is None or message.tool_calls:
            return None
        text = message.text.strip()
        if not prose_claims_completion(text):
            return None
        observation = self._last_observation
        if observation is None:  # pragma: no cover - arm() always sets it
            return None
        # The claim is a CLAIM whether it arrived as a finish action or as
        # prose; quoting it back through the same builder is what keeps the two
        # arms' challenges byte-identical above the channel sentence. The
        # synthesised status is ``done`` because that is what the message
        # asserts, and it binds to the same last observation the challenge
        # re-attaches -- the screen the claim is about.
        claim = FinishAction(
            observation_id=observation.observation_id,
            status="done",
            reason=text[:_PROSE_CLAIM_MAX_CHARS],
        )
        challenge = build_completion_challenge(
            claim=claim,
            instruction=self.instruction,
            observation=observation,
            reply_guidance=self.reply_guidance,
        )
        self._completion_fired += 1
        if self.record is not None:
            # Same record kind as the finish-call arm -- one counter, one
            # budget -- and the ``trigger`` key is the discriminator for
            # readers counting which arm fired; the finish-call row keeps its
            # own historical shape.
            self.record(
                "completion_challenged",
                {
                    "status": "done",
                    "reason": claim.reason,
                    "challenge": challenge,
                    "observation_id": observation.observation_id,
                    "trigger": "terminal-message",
                },
            )
        return [TextContent(text=challenge), *self._shown_blocks(observation)]

    async def start(self) -> None:
        self.endpoint.parent.mkdir(parents=True, exist_ok=True)
        # The wire is a UNIX socket inside the episode scratch: nothing off the
        # machine can reach it, and nothing outside the scratch can guess it.
        # Path length is the one platform trap (sockaddr_un is ~104 bytes on
        # macOS), so an over-long path fails loudly here rather than as a
        # confusing connect error inside the child.
        if len(str(self.endpoint).encode("utf-8")) >= 100:
            raise SessionArmError(
                f"action bridge endpoint {self.endpoint} is too long for a UNIX socket; "
                "put the episode scratch on a shorter path"
            )
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.endpoint), limit=WIRE_READ_LIMIT_BYTES
        )

    async def stop(self) -> None:
        self._closed = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        try:
            self.endpoint.unlink()
        except OSError:
            pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = decode_call(line)
            reply = await self.call(request.get("arguments") or {})
            writer.write(encode_response(**reply))
            await writer.drain()
        except Exception as error:  # noqa: BLE001 - every frame gets an answer
            # The server validates the reply before the model sees it; a broken
            # frame is answered with an error result rather than killing the
            # bridge -- the next call must still find a live endpoint. Broad on
            # purpose: the caller is a child process waiting on a line, and a
            # driver-side bug must surface as a tool result it can act on, not
            # as a hung socket.
            try:
                writer.write(
                    encode_response(
                        [
                            TextContent(
                                text=(
                                    "action bridge error: the episode driver could not "
                                    f"answer this call ({type(error).__name__})"
                                )
                            )
                        ],
                        is_error=True,
                    )
                )
                await writer.drain()
            except (OSError, ValueError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    def _consume_output_epoch(self, kind: str) -> None:
        """Consume ONE external output epoch at ADMISSION (blueprint §3).

        WHY AT ADMISSION, NOT ON SUCCESS. The official clock charges one
        ``agent.predict`` OUTPUT GROUP per epoch: the epoch is spent the moment
        a decision is admitted, whether the group ends in executed actions, a
        finish proposal, an ask exchange or a committed read-back loss.
        Advancing ``_steps`` only on the post-success path left finish
        proposals, challenge re-asks, ask exchanges and lost read-backs
        uncharged -- the QA bridge matrix measured finish=0 and readback=0 --
        so ``max_steps`` bounded the wrong quantity. Refusals BEFORE admission
        (schema, stale binding, second batch) charge compute/wall only: they
        never reached a decision. The record row keeps the epoch and its
        reason durable in the SAME RecordSink ledger -- not a second counter.
        """

        self._steps += 1
        if self.record is not None:
            # ``epoch_kind``, deliberately NOT ``kind``: a RecordSink row
            # merges payload keys into its own JSON object, and a payload
            # ``kind`` would overwrite the row's kind (measured on the offline
            # rig: epoch rows arrived in events.jsonl renamed to the epoch
            # kind, invisible to every counter -- the merge collision class).
            self.record("output_epoch", {"epoch_kind": kind, "steps": self._steps})

    async def call(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One model call, decided exactly as the loop-driven tool decides it."""

        async with self._lock:
            call_id = f"step-{self._steps}-{uuid.uuid4().hex[:6]}"
            # THE ATTEMPTED CLOCK'S GATE (audit §48): reserve the group BEFORE
            # validation -- so a refused Nth group is still spent -- and
            # refuse group N+1 HERE, before any of its work happens. The
            # admitted check further down stays as the fail-closed fallback;
            # ``steps`` and the public outcome fields are unchanged.
            if self._reserve_output_group() is None:
                blocked_token = self._token
                if blocked_token is not None:
                    blocked_token.mark_terminal()
                self.end_requested = self.end_requested or "max-steps"
                if self.record is not None:
                    self.record(
                        "budget",
                        {
                            "steps": self._steps,
                            "attempted_groups": self._attempted_groups,
                        },
                    )
                return {
                    "content": [TextContent(text=BUDGET_ACK)],
                    "is_error": False,
                    "details": {"terminal": "max-steps"},
                }
            self._pending_ask_delivery = []
            self._pending_ask_receipt = None
            token = self._token
            pending = token.pending if token is not None else None
            if token is None or pending is None or token.terminal:
                self._mark_group_kind("refused:no-pending")
                refusal = _no_pending_refusal(call_id)
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            from pydantic import ValidationError

            from local_operator.evaluation.runner.provider_client import (
                classify_admission_error,
                classify_validation_error,
                validation_diagnostic,
            )

            try:
                batch, tolerated_fields = _build_batch(arguments, pending)
            except ValidationError as error:
                self._mark_group_kind(f"refused:{classify_validation_error(error)}")
                refusal = _refusal(
                    call_id,
                    validation_diagnostic(error),
                    class_key=classify_validation_error(error),
                    observation=pending,
                    surface=self.surface,
                )
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            try:
                batch.validate_for(pending)
                self.surface.validate_batch(batch)
            except ValueError as error:
                self._mark_group_kind(f"refused:{classify_admission_error(error)}")
                refusal = _refusal(
                    call_id,
                    f"action batch does not match this observation: {error}",
                    class_key=classify_admission_error(error),
                    observation=pending,
                    surface=self.surface,
                )
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }

            if self._steps >= self.max_steps:
                # The runner's own ordering: the budget is checked BEFORE the
                # model's next decision, so batch N+1 never runs. The episode
                # is still scored on the state it reached.
                token.mark_terminal()
                self.end_requested = self.end_requested or "max-steps"
                if self.record is not None:
                    self.record("budget", {"steps": self._steps})
                return {
                    "content": [TextContent(text=BUDGET_ACK)],
                    "is_error": False,
                    "details": {"terminal": "max-steps"},
                }

            terminal = _terminal_kind(batch)
            if terminal == "finish":
                claim = finish_claim(batch)
                if (
                    self.completion_gate
                    and self._completion_fired < self.completion_challenges
                    and claim.status == "done"
                ):
                    # The runner's completion gate (``runner/completion.py``),
                    # enforced on this channel: the FIRST ``done`` claim is
                    # refused once and re-asked, and every later finish -- the
                    # same declaration or a corrected one -- is accepted, so the
                    # gate can never drive an episode into a challenge loop.
                    # NOTHING MOVES: the token stays armed and unspent and no
                    # ``action_batch`` is written -- but the claim ITSELF is an
                    # admitted output decision and consumes one external output
                    # epoch (blueprint §3), exactly as the runner's challenge
                    # does at the official clock. A ``failed`` claim is
                    # NOT challenged (there is no completion to confirm; the
                    # runner's rule), and a claim is only ever challenged if the
                    # shared text can name the task back, which is why the
                    # instruction rides on the bridge.
                    # The proposal is ADMITTED here: one epoch, consumed
                    # before the challenge is built and recorded (blueprint §3).
                    self._consume_output_epoch("finish-challenged")
                    self._mark_group_kind("finish-challenged")
                    self._completion_fired += 1
                    challenge = build_completion_challenge(
                        claim=claim,
                        instruction=self.instruction,
                        observation=pending,
                        reply_guidance=self.reply_guidance,
                    )
                    if self.record is not None:
                        self.record(
                            "completion_challenged",
                            {
                                "status": claim.status,
                                "reason": claim.reason,
                                "challenge": challenge,
                                "observation_id": pending.observation_id,
                            },
                        )
                    # Evidence first, re-prompt second (the runner's ordering):
                    # on this channel the challenge IS the re-prompt, and the
                    # state it names rides WITH it -- the same rendered blocks
                    # the model already has for this observation, never a
                    # re-render.
                    return {
                        "content": [TextContent(text=challenge), *self._shown_blocks(pending)],
                        "is_error": False,
                        "details": {"terminal": "completion-challenged"},
                    }
                # Mirrors the runner: a finish batch is recorded and ends the
                # episode; it is never sent to the adapter (no action in it
                # mutates the environment). A valid finish PROPOSAL is an
                # admitted output decision and consumes ONE external output
                # epoch (blueprint §3) -- the challenged first claim consumed
                # its own above, so the official clock charges each proposal
                # exactly once. The record carries the CLAIM -- its
                # status and reason -- because that is what the challenge (and
                # any reader grading a finish) reasons about; an action count
                # alone cannot answer "what did it claim".
                self._consume_output_epoch("finish")
                self._mark_group_kind("finish")
                token.mark_terminal()
                self.end_requested = "finish"
                if self.record is not None:
                    self.record(
                        "finish",
                        {
                            "status": claim.status,
                            "reason": claim.reason,
                            "actions": len(batch.actions),
                            "completion_challenged": self._completion_fired > 0,
                        },
                    )
                return {
                    "content": [TextContent(text=FINISH_ACK)],
                    "is_error": False,
                    "details": {"terminal": "finish"},
                }
            if terminal == "ask_user":
                if self.ask is None:
                    return {
                        "content": [
                            TextContent(
                                text=(
                                    "This episode cannot answer host-owned ask_user "
                                    "actions; continue without asking."
                                )
                            )
                        ],
                        "is_error": True,
                        "details": {"rejection_class": "ask_user-unsupported"},
                    }
                self._consume_output_epoch("ask_user")
                self._mark_group_kind("ask_user")
                delivery = await self.ask(batch)
                # The exchange's THIRD consumer: the answer text AND the
                # canonical receipt ride the native result's CONTENT (staged
                # here; attached to whichever path answers this call). The
                # ask-sentinel defect was this value being dropped on the
                # floor while the refresh execute below still ran -- and the
                # MCP forwarding layer only carries CONTENT to the next root
                # request, so a receipt kept solely in ``details`` would be
                # dropped again at that boundary (R4, review round 1).
                receipt, answer_text = _split_ask_delivery(delivery)
                self._pending_ask_delivery = [
                    TextContent(text=_ask_delivery_sentence(answer_text, receipt))
                ]
                self._pending_ask_receipt = {
                    "answered": answer_text is not None,
                    "answer": answer_text,
                    "receipt": receipt,
                }
                if self.record is not None and receipt is not None:
                    self.record("ask_exchange", {"receipt": receipt})
            else:
                # The batch is ADMITTED here: ONE external output epoch,
                # consumed BEFORE the guest-execution await. The old increment
                # on the post-success path made finish proposals, ask
                # exchanges and committed read-back losses invisible to the
                # official clock (all measured zero-epoch by the QA bridge
                # matrix); refusals ABOVE this point still charge compute
                # only, and resum/re-bind below never charges a second one.
                self._consume_output_epoch("batch")
                self._mark_group_kind("batch")

            observation = token.consume_pending()
            if observation is None:  # pragma: no cover - the lock makes this unreachable
                refusal = _no_pending_refusal(call_id)
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            try:
                result = await self.execute(batch)
            except Exception as error:  # noqa: BLE001 - _after_failed_execute owns it
                self._mark_group_kind("read-back")
                # The token was consumed above, so a raise from here used to
                # leave the bridge permanently unable to bind any later batch
                # -- including a finish -- and the episode spun to an
                # ``agent_stop``. The re-bind (or the legible end) is below.
                return await self._after_failed_execute(error=error)
            # NOTE: the epoch was consumed at ADMISSION (above); the old
            # post-success increment that lived here is what made the
            # non-execute output forms uncharged. Do not reintroduce it.
            token.record_in_flight(result.observation)
            if self.record is not None:
                self.record("batch", {"batch": batch, "result": result})
            rendered = self.render(result.observation)
            # ``_last_shown`` holds the render WITHOUT the correction below:
            # ``_shown_blocks`` re-attaches exactly this list to a later
            # completion challenge, and a challenge must carry the screen
            # state alone -- the note is a correction of an EARLIER reply, and
            # re-attached to an unrelated claim it reads as guidance for the
            # claim instead. It is delivered once, on the call's own result.
            self._last_shown = (result.observation.observation_id, rendered)
            note = tolerated_fields_note(tolerated_fields)
            if note is not None:
                # The correction for a sibling field this call carried and the
                # tolerance dropped rides the call's OWN result, ahead of the
                # observation: the definition of a silent drop is that the
                # model is never told, and this result is the next thing it
                # reads (the same note the loop-driven channel renders into
                # the next observation's message).
                rendered = [TextContent(text=note), *rendered]
            self._last_observation = result.observation
            details: dict[str, Any] = {"receipt": result.receipt.model_dump(mode="json")}
            if self._pending_ask_receipt is not None:
                details["ask"] = self._pending_ask_receipt
            return {
                "content": [*self._pending_ask_delivery, *rendered],
                "is_error": False,
                "details": details,
            }

    async def _after_failed_execute(self, *, error: Exception) -> dict[str, Any]:
        """One bounded re-bind after an execute failure, or a legible end.

        THE WEDGE THIS CLOSES (F1, arm 1796). ``call`` consumes the token
        BEFORE the batch runs, so a batch can never bind to a screen nobody was
        shown; until this method existed, a raise from ``execute`` left the
        token consumed with nothing in flight. From then on every call -- the
        next batch AND a ``finish`` -- was refused with "no action batch was
        run: this call found no observation to bind to", and the episode could
        only spin until the model gave up (``agent_stop``, reading as
        capability). Measured cost: one action-phase RPC error destroyed 2 of
        10 episodes (tasks 001 and 013), both sealed zero.

        WHY RECOVERY, NOT MORE RETRIES. Raising the read-back retry count is a
        comparability-affecting policy change (INFRA.md; it changes what the
        arm measures). The defect here is STATE: the binding is destroyed by a
        failure the ADAPTER declared committed, and the episode stays dead even
        when the environment is healthy again. So this method attempts exactly
        ONE re-bind through the driver's ``recover`` seam (a read-only fetch of
        the environment's current observation), and only for that committed
        class (``_observation_phase_failure``) -- the same predicate the
        read-back retries trust. For any other failure the batch's state is
        ambiguous, the supervisor has poisoned the session, and re-reading is
        not safe; the episode ends legibly instead.

        THE BOUND AND ITS COST. One ``recover`` call per failed batch, and none
        at all on the healthy path; the call itself is bounded by the caller's
        own step timeout plus the RPC layer's 1 s cancel grace (one RPC,
        181 s worst case in the campaign config: the 180 s budget + the 1 s
        grace).
        When it cannot yield an observation the episode ends HERE, terminally
        (:meth:`_end_wedged`). ``finish`` stays refused for a wedged bridge on
        purpose: no screen exists to bind it to, and the driver stops the turn
        instead of leaving the model to discover that by refusing it.
        """

        recover = self.recover
        # ``committed`` keeps its STRICT meaning for the wedge sentence ("the
        # adapter declared the batch committed"); the RE-BIND gate is wider --
        # it also admits a pre-dispatch refusal (``invalid_state``), which is
        # provably non-committed but still safe to re-read. See
        # ``_refused_before_dispatch`` for the accuracy loss this closes.
        committed = _observation_phase_failure(error)
        recovery_attempted = recover is not None and _rebindable_failure(error)
        recovered: Observation | None = None
        recover_error: Exception | None = None
        if recovery_attempted and recover is not None:
            try:
                recovered = await recover()
            except Exception as failure:  # noqa: BLE001 - the wedge sentence names it
                recover_error = failure
        if recovered is not None:
            return self._deliver_recovery(error=error, recovered=recovered)
        return self._end_wedged(
            error=error,
            committed=committed,
            recovery_attempted=recovery_attempted,
            recover_error=recover_error,
        )

    def _deliver_recovery(self, *, error: Exception, recovered: Observation) -> dict[str, Any]:
        """Re-arm the token with the re-bound observation and answer the call.

        Held IN FLIGHT, never armed directly: ``fold`` at the turn end arms
        it, so the one-batch-per-turn invariant survives the recovery -- a
        second call in the same turn still finds nothing pending and is
        refused, exactly as after a normal batch.

        THE SCREEN IS ALREADY IN THE TRANSCRIPT when the adapter's current
        observation IS the one the model last saw (the wedge case: the batch
        committed but its read-back was the thing that was lost). It is NOT
        re-rendered then, because the renderer dedups a byte-identical repeat
        into its "your last action changed nothing visible" note -- a claim
        this path cannot make, since the lost read is exactly the one that
        would have shown whether anything changed. The recovery sentence says
        what happened instead and tells the model to re-read before acting.
        """

        token = self._token
        if token is not None:
            token.record_in_flight(recovered)
        shown_before = self._last_observation
        content: list[Any] = [
            *self._pending_ask_delivery,
            TextContent(text=RECOVERY_AFTER_READBACK_LOSS),
        ]
        if shown_before is None or shown_before.observation_id != recovered.observation_id:
            rendered = self.render(recovered)
            self._last_shown = (recovered.observation_id, rendered)
            self._last_observation = recovered
            content = [*content, *rendered]
        if self.record is not None:
            self.record(
                "recovered",
                {
                    "steps": self._steps,
                    "attempted_epoch": True,
                    "transport_error": _bounded_cause(error, limit=500),
                    "observation_id": recovered.observation_id,
                },
            )
        details: dict[str, Any] = {
            "recovered": True,
            "observation_id": recovered.observation_id,
        }
        if self._pending_ask_receipt is not None:
            details["ask"] = self._pending_ask_receipt
        return {
            "content": content,
            "is_error": True,
            "details": details,
        }

    def _end_wedged(
        self,
        *,
        error: Exception,
        committed: bool,
        recovery_attempted: bool,
        recover_error: Exception | None,
    ) -> dict[str, Any]:
        """End the episode: the observation binding cannot be re-established.

        Terminal AND legible. The token is closed so no later call can bind,
        and the terminal reason plus its diagnostic name the transport failure
        -- the driver maps both into an outcome of ``failed`` /
        ``bridge-wedged`` instead of the ``agent_stop`` that used to read as
        the model giving up. The turn itself is stopped by the driver, which
        watches this state from its event sink (``request_graceful_cancel``
        keeps the failed call's result paired in the transcript).
        """

        token = self._token
        if token is not None:
            token.mark_terminal()
        sentence = _bridge_wedge_sentence(
            error,
            committed=committed,
            recovery_attempted=recovery_attempted,
            recover_error=recover_error,
        )
        self.end_requested = BRIDGE_WEDGED_TERMINAL
        self.end_diagnostic = sentence
        if self.record is not None:
            self.record(
                "bridge_wedged",
                {
                    "steps": self._steps,
                    # ``attempted_epoch``: the failed execute's epoch was
                    # consumed at admission and stays counted under the
                    # declared attempted-output convention -- this receipt is
                    # what keeps it apart from the upstream-crash comparison,
                    # where no completed episode exists at all (blueprint §3).
                    # No score below pretends the reach happened.
                    "attempted_epoch": True,
                    "transport_error": _bounded_cause(error, limit=500),
                    "recovery_attempted": recovery_attempted,
                    "recover_error": (
                        _bounded_cause(recover_error, limit=500)
                        if recover_error is not None
                        else None
                    ),
                },
            )
        details: dict[str, Any] = {"terminal": BRIDGE_WEDGED_TERMINAL}
        if self._pending_ask_receipt is not None:
            details["ask"] = self._pending_ask_receipt
        return {
            "content": [*self._pending_ask_delivery, TextContent(text=sentence)],
            "is_error": True,
            "details": details,
        }


# ---------------------------------------------------------------------------
# Opening the episode's session
# ---------------------------------------------------------------------------


#: The tool surface an episode session runs. THE BENCHMARK CALLER'S POLICY,
#: stated once here so nothing reconstructs it by memory: the ordinary local
#: working set a task needs (``read``/``write``/``edit``/``glob``/``grep``/
#: ``bash``), delegation and tracking (``task``/``hub``/``todo`` — plus
#: ``jobs``/``wait``, because child work is a BACKGROUND job: a parent that
#: cannot wait cannot collect its children), the packaged-agent and
#: isolated-team surfaces (``agent``/``team``; guides and skills ride the
#: knowledge block, not tools), plus the ONE session-scoped action tool the
#: episode exports. Everything else in the shipped inventory is a CONTROLLER
#: capability reachable in-process — ``secret``, ``list_variables``/
#: ``read_variable``, ``setting``, ``network``, ``send``/``sessions``,
#: ``monitor``/``wake``, ``code_requests``, ``eval``, ``web_*``/``browser``,
#: ``project``/``team_delete`` — and a filesystem boundary does not sandbox
#: an in-process API, so they are named out here rather than hoped about.
#: The enforcement is the EXISTING inventory mechanism (``SessionSpec.tools``
#: + ``ApprovalPolicy.declared`` -> ``Session.set_tool_inventory``); this
#: tuple is only WHICH names, and ``validate_episode_session_spec`` pins
#: membership.
EPISODE_TOOL_SURFACE: tuple[str, ...] = (
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "bash",
    "task",
    "hub",
    "todo",
    "jobs",
    "wait",
    "agent",
    "team",
    create_mcp_tool_name(SERVER_NAME, ACTION_TOOL_NAME),
)


def validate_episode_session_spec(
    *,
    spec: SessionSpec,
) -> SessionSpec:
    """The session spec an episode runs under, from the run's own spec.

    THE CONTRACT (world-boundary slice): an episode session is refused unless
    it carries (a) an explicit ``world_root`` — the model's tools and
    discovery clipped to the world, with the store roots controller-side (see
    ``sdk._resolve_world_root``) — and (b) the pinned tool surface it
    declares, approved by that same declaration. The previous posture ("full
    tool surface, ``ApprovalPolicy.auto()``") handed an unattended model
    every controller-backed capability in the shipped inventory; it is
    refused here, at the arm's single seam, so no future caller can quietly
    run a different reach than the one that was measured.
    """

    if spec.world_root is None:
        raise SessionArmError(
            "an episode session requires an explicit world_root: without one the "
            "model's tools and discovery reach the controller's own tree"
        )
    if spec.tools is None:
        raise SessionArmError(
            "an episode session must DECLARE its tool surface (SessionSpec.tools); "
            "the undeclared shipped surface is a controller capability set, not a "
            "task surface"
        )
    declared = set(spec.tools)
    if declared != set(EPISODE_TOOL_SURFACE):
        missing = sorted(set(EPISODE_TOOL_SURFACE) - declared)
        extra = sorted(declared - set(EPISODE_TOOL_SURFACE))
        raise SessionArmError(
            "the episode tool surface is pinned; a different set is a different arm "
            f"(missing={missing}, extra={extra})"
        )
    if spec.approvals.mode != "declared" or set(spec.approvals.tools) != declared:
        raise SessionArmError(
            "the episode's approvals must be ApprovalPolicy.declared(EPISODE_TOOL_SURFACE): "
            "the declaration is the run's approval for exactly the tools it names"
        )
    return spec


@dataclass
class EpisodeSession:
    """An open episode session: its live tool inventory and exit handle."""

    session: Any
    roots: SessionRoots
    _context: Any = field(repr=False)
    _unsubscribe: Callable[[], None] | None = field(default=None, repr=False)
    #: The request observer's unsubscribe. Detached in ``aclose`` AFTER the
    #: SDK context exits -- deliberately NOT with the event sink above:
    #: request-final records keep arriving through cancellation/shutdown (the
    #: whole point of subscribing for accounting), and an early detach would
    #: lose exactly the rows the caller's totals need.
    _request_unsubscribe: Callable[[], None] | None = field(default=None, repr=False)
    #: Deduplicated request-final rows (scalars and the detached usage dump
    #: only; no prompts, arguments or credentials) plus the usage OBJECTS
    #: behind them for the one canonical pricing projection. The observer
    #: callback only appends here; serialization and pricing happen once at
    #: close-out (the callback contract forbids doing that work on the model
    #: loop).
    request_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)
    request_usages: list[Any] = field(default_factory=list, repr=False)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The inventory LIVE, never a snapshot taken at open.

        The action server is an MCP server, and MCP discovery is asynchronous:
        its tools merge on settle, after ``open`` returns. A frozen field here
        read 28 names while the session's live inventory already held 29 with
        the action tool in it (measured), and the first request publishes its
        array from the LIVE inventory -- so anything that wants to know what
        the turn can actually call must read it at the moment it asks.
        """

        return session_tool_names(self.session)

    async def aclose(self) -> None:
        """Detach the sink, exit the SDK's open context, then the observer.

        The ORDER is the point: the event sink detaches first (its delivery
        is the running turn's), the SDK context exit runs the canonical
        disposal, and only THEN is the request observer detached -- finals
        for cancellation/shutdown land during that close, and a subscription
        dropped earlier would miss them.
        """

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        await self._context.__aexit__(None, None, None)
        if self._request_unsubscribe is not None:
            self._request_unsubscribe()
            self._request_unsubscribe = None


async def open_episode_session(
    *,
    spec: SessionSpec,
    roots: SessionRoots,
    on_event: Callable[[AgentEvent], Any] | None = None,
    session_opener: Callable[..., Any] | None = None,
    confinement_root: Path | None = None,
) -> EpisodeSession:
    """Open the session an episode runs, subscribing the event sink first.

    The subscription is registered BEFORE the first prompt, because events are
    not replayed: a sink attached after the turn began would miss its open.
    The sink is the RECORD and the token's fold input. Design §6 sketches the
    same job as iteration over ``sdk.events``; subscribing instead keeps the
    fold on the engine's own dispatch order rather than one queue behind it,
    which is what the token's turn-boundary arming requires.

    ``confinement_root`` confines the session's local tools to the episode
    scratch (see ``Session.set_tool_confinement`` and
    ``local_operator.tools.confinement``). Installed BEFORE the sink and the
    first prompt, because the session rebuilds its tool context per turn: a
    tool call served in the first turn must already run against the confined
    context, and there is no later point that is as early.
    """

    opener = session_opener or sdk.open_session
    context = opener(spec, roots=roots, mode="own")
    # ``Any`` on purpose, the same idiom ``sdk._build_session`` uses for its
    # post-open attachment calls: the declared session type is
    # ``SessionProtocol``, and confinement is installed through a concrete
    # ``Session`` member the protocol deliberately does not carry (the one
    # caller that needs it is this arm, not a host surface).
    session: Any = await context.__aenter__()
    if confinement_root is not None:
        session.set_tool_confinement(confinement_root)
    unsubscribe = session.subscribe(on_event) if on_event is not None else None
    handle = EpisodeSession(
        session=session,
        roots=roots,
        _context=context,
        _unsubscribe=unsubscribe,
    )
    # REQUEST OBSERVATION, attached immediately after open and BEFORE the
    # first prompt (records are not replayed): the episode's cost/actor
    # accounting must see every provider request of the ORIGINAL owning root
    # tree -- parent turns, compaction/naming helpers, delegated children and
    # grandchildren -- and must keep seeing them through cancellation and
    # shutdown, which is why the subscription outlives the event sink in
    # ``EpisodeSession.aclose``. The callback itself only deduplicates and
    # queues (contract: synchronous, no I/O, no pricing); serialization and
    # pricing happen once at close-out in ``_flush_request_observation``.
    _subscribe_request_observation(handle)
    return handle


def _subscribe_request_observation(handle: EpisodeSession) -> None:
    """Attach the episode's request observer to the owning session tree.

    ``request_id`` is the ONE deduplication key (provider/model/cache keys are
    not unique per request; the protocol says so where it defines them). A
    record without an id is refused rather than queued -- an unkeyed row could
    never be proven distinct. The handler appends and returns: no awaiting, no
    serialization, no pricing on the model loop.
    """

    seen: set[str] = set()

    def _observe(observation: RequestFinalObservation) -> None:
        request_id = observation.request_id
        if not request_id or request_id in seen:
            return
        seen.add(request_id)
        handle.request_rows.append(
            {
                "request_id": request_id,
                "tree_id": observation.tree_id,
                "session_id": observation.session_id,
                "parent_session_id": observation.parent_session_id,
                "job_id": observation.job_id,
                "actor_role": observation.actor_role,
                "purpose": observation.purpose,
                "requested_provider": observation.requested_provider,
                "requested_model_id": observation.requested_model_id,
                "served_provider": observation.served_provider,
                "served_model_id": observation.served_model_id,
                "served_identity_source": observation.served_identity_source,
                "upstream_host": observation.upstream_host,
                "usage_coverage": observation.usage_coverage,
                "outcome": observation.outcome,
                "usage": (
                    observation.usage.model_dump(mode="json")
                    if observation.usage is not None
                    else None
                ),
            }
        )
        if observation.usage is not None:
            handle.request_usages.append(observation.usage)

    handle._request_unsubscribe = handle.session.subscribe_requests(_observe)


def session_tool_names(session: Any) -> tuple[str, ...]:
    """The session's live tool inventory, by name.

    ``_tools`` first (the live inventory every built session carries),
    ``tools`` second (a host that exposes one), empty last -- the same
    getattr-probe ``session_factory`` itself uses, kept identical so the two
    can never disagree about which list is "the" surface.
    """

    live = getattr(session, "_tools", None) or getattr(session, "tools", None) or ()
    return tuple(tool.name for tool in live)


# ---------------------------------------------------------------------------
# One whole episode, driven as a session
# ---------------------------------------------------------------------------


def _record_diagnostic(error: BaseException) -> str:
    """The sentence a record-level failure leaves behind, classified not repr'd.

    A sink failure already carries its own legible sentence (naming the volume
    and the bytes that did land); anything else on the record path is reported
    by its class and message. The distinction this preserves is the one the
    ``steps=0`` artifact destroyed: an environment that ran out of room is not
    the run misbehaving.
    """

    if isinstance(error, RecordSinkError):
        return error.sentence
    return f"{type(error).__name__}: {error}"


def _first_record_failure(
    record: RecordSink, errors: Sequence[BaseException]
) -> BaseException | None:
    """The first failure that makes this record incomplete, sink-first.

    The sink's own failure is preferred because it is the classified one (it
    names the volume and the bytes that landed); ``errors`` carries the rest of
    the record path's failures in the order they were captured.
    """

    if record.failure is not None:
        return record.failure
    return errors[0] if errors else None


#: The phases a dying run names, in the order that keeps the ACTIONABLE class
#: when several ride one sentence. ``RpcRemoteError`` folds its structured
#: cause into ``str()`` (see ``adapters/rpc.py``: anything invisible to ``str``
#: never reaches the record), so the inner class -- the phase that died -- is
#: what the record already SHOWS, and this scan reads that same sentence
#: rather than a second representation of it.
_FAILURE_REASON_MARKERS: tuple[tuple[str, str], ...] = (
    ("EnvironmentSetupError", "environment-setup"),
    ("UpstreamAllocationRefused", "environment-allocation"),
    ("the record sink failed", "record-sink"),
    ("SessionArmError", "session-arm"),
)


def _failure_terminal_reason(error: BaseException) -> str:
    """The stable token naming why a run died before its record could close.

    ``terminal_reason`` used to be ``None`` on every ``failed`` /
    ``failed_pre_bundle`` outcome, so the record said a run died but never
    where -- the same defect the ``agent_stop`` reasons fix, on the other
    status that carried a ``None``. The token reads the SAME sentence the
    diagnostic carries (the close-out's ``Class: message``), so a reader
    bucketing on ``terminal_reason`` sees the phase and a reader reading the
    diagnostic sees the phase's own words; an error that matches no known
    phase falls back to its exception class, kebabed -- stable across runs,
    and honest about being a class rather than an environment fact.
    """

    if isinstance(error, RecordSinkError):
        return "record-sink"
    rendered = f"{type(error).__name__}: {error}"
    for marker, token in _FAILURE_REASON_MARKERS:
        if marker in rendered:
            return token
    return re.sub(r"(?<!^)(?=[A-Z])", "-", type(error).__name__).lower()


def _bounded_text(text: str, limit: int) -> str:
    """One bounded, single-line excerpt for a diagnostic field.

    Outcome fields stay summaries: the full text is in the record's
    ``events.jsonl`` a line away, so the diagnostic carries enough to triage
    from the outcome alone and no more.
    """

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _bounded_text_keeping_tail(text: str, limit: int, *, tail: int) -> str:
    """One bounded excerpt that keeps both ends of ``text``.

    ``_bounded_text`` truncates from the END, which is right when the signal
    leads. The repeated-error sentence inverts that: it OPENS with the
    (roster-sized) tool-name list and ends with the reason the turn stopped
    ("returned the same errors for N unchanged tool batches"), so a head-only
    excerpt would drop the reason exactly when the roster grows. Same budget,
    both ends kept; the full text stays in ``events.jsonl``.
    """

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    head = collapsed[: limit - tail - 3].rstrip()
    # Start the kept tail at a word boundary so the excerpt never opens
    # mid-token; advancing only ever shortens it, so the budget holds.
    start = len(collapsed) - tail
    boundary = collapsed.find(" ", start)
    kept = collapsed[boundary + 1 :] if boundary != -1 else collapsed[start:]
    return f"{head} … {kept}"


def _agent_stop_terminal_reason(
    *,
    wall_fired: bool,
    end: AgentEndEvent | None,
    message: Message | None,
    challenges: int,
) -> tuple[str, str | None]:
    """Name why a turn ended with no terminal batch, plus the detail a reader needs.

    THE RECORD USED TO SAY NOTHING. Every ending that is not a finish, a
    truncation or a wedge fell through to ``status: "agent_stop"`` with
    ``terminal_reason: None`` -- indistinguishable between the model stopping
    on prose, the model claiming completion on prose, a message with nothing
    in it, the wall bound, a cut-off, and a provider error. Arm 1796 re-derived
    exactly that by hand, per record, from ``events.jsonl``, because a quarter
    of its sample read as capability where it was infrastructure.

    THE INPUTS ARE THE SHARED FACTS, never adapter state: the final
    ``AgentEndEvent`` (``aborted`` / ``error`` / ``cut_off_cause`` -- the frame
    the TUI, the phone and ``exec --json`` all read), the terminal assistant
    message the bridge folds for the prose arm, and the driver's own wall
    bound, whose abort reason never rides the event. Nothing here changes when
    a turn ends, what it scored, or which status it lands in: the same episode
    ends the same way, and the label only names which way that was.

    Order is precedence: the wall bound is the driver's own act and outranks
    whatever the provider was doing when it fired; a stamped cut-off cause
    outranks the error text, because the session REWRITES an involuntary
    cut-off into ``aborted=False, error=<cut-off notice>`` before any sink
    sees it while the cause rides the same frame -- reading the error branch
    first filed every cut-off as the provider's error (review round 1, R1-1);
    else an error names the provider (classified through the shared incident
    rules) or a host gate; an abort with no cause is named an abort; a clean
    stop is read from the terminal message, the only account of it.
    """

    if wall_fired:
        return "wall-bound", "the episode wall budget aborted the turn"
    if end is None:
        return "stopped", "the turn ended with no end-of-turn event recorded"
    if end.cut_off_cause:
        # BEFORE the error branch: the frame a sink sees for an involuntary
        # cut-off is the session's rewrite (``aborted=False``, an error whose
        # text reads like a provider failure) with the cause preserved beside
        # it, so reading ``error`` first filed every cut-off as the
        # provider's (review round 1, R1-1). The rewrite runs in
        # ``Session._classify_cut_off`` before any handler is called, and
        # "consumption on the end event is the established rule for this
        # field" (session.py).
        return end.cut_off_cause, _bounded_text(render_cut_off_reason(end.cut_off_cause), 400)
    if end.error:
        if end.error == "stopped by gate":
            return "gate-stop", end.error
        if end.error.startswith("No progress: "):
            # The sentence names EVERY tool in the failing batch, so it grows
            # with the model's roster (QA measured 4376 chars at 120 names);
            # bound it like the siblings -- keeping the sentence's reason,
            # which sits at its tail.
            return "no-progress", _bounded_text_keeping_tail(end.error, 400, tail=140)
        incident = classify_incident(end.error)
        return "provider-error", _bounded_text(f"{incident.category}: {end.error}", 400)
    if end.aborted:
        return "aborted", "the turn was aborted and no cut-off cause was named"
    if message is None:
        return "empty-message", "the turn ended with no terminal assistant message"
    text = (message.text or "").strip()
    if not text:
        # "no text content", not "no content" (QA round 1, Q-2): an image-only
        # terminal message carried content; it just carried no text for this
        # reader.
        return "empty-message", "the terminal assistant message carried no text content"
    note = f" (the completion gate challenged it {challenges} time(s))" if challenges else ""
    if prose_claims_completion(text):
        return "completion-claim", (
            "the terminal message claims completion without acting: "
            f"{_bounded_text(text, 200)!r}{note}"
        )
    return "no-tool-call", (
        f"the terminal message carries no tool call: {_bounded_text(text, 200)!r}{note}"
    )


@dataclass(frozen=True)
class SessionArmOutcome:
    """What happened in the pilot arm, and where its record lives."""

    status: str
    episode_id: str
    record_root: Path | None
    score: Any | None = None
    steps: int = 0
    #: The ATTEMPTED output-group clock beside ``steps`` (audit §45-49):
    #: completed root output groups, including refused and local-tool-only
    #: groups the admitted epoch never charged. ADDITIVE field -- historical
    #: rows and the ``steps`` semantics stay exactly as shipped.
    attempted_groups: int = 0
    #: The stable token naming why the run ended: ``finish`` / ``max-steps`` /
    #: ``bridge-wedged`` on the terminalled paths; for an ``agent_stop`` or a
    #: ``failed`` the reason read from the end-of-turn facts (see
    #: ``_agent_stop_terminal_reason`` / ``_failure_terminal_reason``). Never
    #: ``None`` on a returned outcome; a reader buckets on this rather than
    #: re-deriving the ending from ``events.jsonl``.
    terminal_reason: str | None = None
    diagnostic: str | None = None
    rescue_required: bool = False
    rescue_complete: bool | None = None
    tool_names: tuple[str, ...] = ()
    resolved_sources: Mapping[str, str] = field(default_factory=dict)
    #: Whether the record is a complete archive of the run. ``True`` means the
    #: sink met a storage failure -- almost always a full volume on this shared
    #: host -- and the record stops at its last complete line; ``record_diagnostic``
    #: carries the classified sentence (out-of-room vs anything else) and how
    #: many bytes did land. The status/steps/score beside these fields are the
    #: RUN's own facts and stay true: measured 2026-09-28, a torn record must
    #: never again read as "the arm failed to act".
    record_incomplete: bool = False
    record_diagnostic: str | None = None
    duration_ms: int = 0


async def run_session_episode(
    *,
    spec: EpisodeSpec,
    config: EpisodeConfig,
    selector: Any,
    roots: SessionRoots,
    scratch_root: Path,
    session_spec: SessionSpec,
    secrets: Sequence[ResolvedSecret] = (),
    display_name: str | None = None,
    max_wall_s: float | None = None,
    launch: Any = AdapterSupervisor.launch,
    rescue: Any = run_rescue,
    confinement_root: Path | None = None,
    session_opener: Any = None,
    owner_status: OwnerStatusChannel | None = None,
) -> SessionArmOutcome:
    """Run ONE episode as a session: launch, reset, prompt, score, clean up.

    The order of calls mirrors ``EpisodeRunner`` deliberately, comment for
    comment where the two could drift: prepare is allocation-free, the rescue
    descriptor is re-persisted before ``reset_start`` (the side-effect
    boundary), the artifact root is created by the PARENT, and the score is
    taken after the turn but before cleanup. What differs is the middle: the
    decision loop and the reply channel are replaced by one
    ``sdk.open_session`` turn whose actions arrive as MCP tool calls, and the
    record is the pilot's JSONL rather than the sealed bundle format.
    """

    started_ms = int(time.time() * 1000)
    # Record-level failures that are not the sink's own storage failures (an
    # event-fold error, a write into a closed sink). Declared before the try so
    # the failure path can report them too.
    record_errors: list[BaseException] = []
    record_root = config.evidence_root / f"{spec.episode_id}-{SESSION_ARM_ID}"
    # PRE-FLIGHT, BEFORE LAUNCH. The sink refuses to open when the volume cannot
    # hold the expected record plus its seal reserve (see ``record_sink``), so a
    # volume that cannot finish the job is refused while the refusal is still
    # free -- measured 2026-09-28: two episodes whose interaction completed were
    # voided at the seal when a shared host filled mid-run, $2.8 of committed
    # spend with nothing reportable. The refusal is a normal outcome carrying the
    # sink's own sentence, never a traceback: nothing was allocated, so there is
    # nothing to rescue.
    try:
        record = RecordSink(record_root / "events.jsonl")
    except RecordSinkError as error:
        return SessionArmOutcome(
            status="failed_pre_bundle",
            episode_id=spec.episode_id,
            record_root=record_root,
            terminal_reason="record-sink",
            diagnostic=error.sentence,
            duration_ms=int(time.time() * 1000) - started_ms,
        )

    supervisor: Any = None
    adapter_session: VerifiedAdapterSession | None = None
    bridge: ActionBridge | None = None
    descriptor: RescueDescriptor | None = None
    rescue_required = False
    cleanup_done = False
    plan: CleanupPlan | None = None
    rescue_complete: bool | None = None
    #: Bound by ``open_episode_session``; declared BEFORE the try so the
    #: exception path can always read them (an early failure -- launch,
    #: handshake, prepare -- must not NameError its own close-out).
    handle: EpisodeSession | None = None
    #: The owned-work shutdown's receipt (``OwnedShutdownResult``), bound by
    #: the close-out and read by the score/release gates: ``complete=False``
    #: refuses the score and retains the worker. ``None`` means no receipt was
    #: produced at all -- unverified, never read as success.
    shutdown_result: Any = None
    #: The release gate's disposition (R2): True when owned work is unverified
    #: and the worker must NOT be released. Read by the cleanup paths and the
    #: closing ``_close_adapter_session``.
    retain_resources = False
    #: One re-join failure is recorded ONCE (R2 round 3): a sustained raise in
    #: the join surface must not flood the record, and must not end the join
    #: either -- the canonical task is still the thing being waited on, and a
    #: record failure never releases ownership (see the module's failure rule).
    outcome: SessionArmOutcome

    def _mark_rescue_required() -> None:
        nonlocal rescue_required
        rescue_required = True

    try:
        # --- launch + prepare (mirrors EpisodeRunner._launch_and_prepare) ----
        supervisor = launch(selector)
        handshake: Handshake = await supervisor.handshake(timeout=config.handshake_timeout)
        surface: ActionSurface = handshake.metadata.capabilities.action_surface()
        answer_owner = handshake.metadata.capabilities.ask_user_answer_owner
        if answer_owner != "adapter":
            # Preflight rather than a mid-episode refusal: the session arm has
            # no user responder yet, so a host-owned ask cannot be answered at
            # all, and an episode that discovers that at step 7 has already
            # paid for the steps before it.
            raise SessionArmError(
                "the session arm answers asks only through the adapter; this "
                f"adapter's ask_user_answer_owner is {answer_owner!r}"
            )
        verifier = HostVerifier(spec.task_id, spec.episode_id, config.artifact_root)
        adapter_session = VerifiedAdapterSession(
            supervisor,
            verifier,
            rescue_required=_mark_rescue_required,
            answer_owner=answer_owner,
            execution_overhead_seconds_per_action=(config.execution_overhead_seconds_per_action),
        )
        requirements = await adapter_session.inspect_requirements(
            InspectRequirementsParams(), timeout=config.prepare_timeout
        )
        _refuse_undeclared_disclosed_infra(spec, selector, requirements)

        provisional = CleanupPlan(
            episode_id=spec.episode_id,
            actions=(
                CleanupAction(
                    action_id=_PROVISIONAL_CLEANUP_ACTION,
                    kind="close_session",
                    resource_ref=spec.episode_id,
                    timeout_ms=int(config.cleanup_timeout * 1000),
                    max_attempts=2,
                ),
            ),
        )
        descriptor = _persist_descriptor(
            config=config, spec=spec, selector=selector, handshake=handshake, plan=provisional
        )
        adapter_session.mark_rescue_persisted(descriptor.descriptor_id)
        prepared = await adapter_session.prepare(
            PrepareParams(
                operation_id=f"prepare-{spec.episode_id}",
                episode_id=spec.episode_id,
                secret_refs=spec.secret_refs,
                infra_values=spec.infra_values,
            ),
            timeout=config.prepare_timeout,
        )
        plan = prepared.cleanup_plan
        descriptor = _persist_descriptor(
            config=config, spec=spec, selector=selector, handshake=handshake, plan=plan
        )
        adapter_session.mark_rescue_persisted(descriptor.descriptor_id)

        # The parent owns this root (same reason as the runner's: the worker
        # must not get a say in the one directory the parent later opens).
        config.artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        observation = (
            await adapter_session.reset_start(
                ResetStartParams(
                    operation_id=f"reset-{spec.episode_id}",
                    task_id=spec.task_id,
                    episode_id=spec.episode_id,
                    artifact_root=str(config.artifact_root),
                    secrets=tuple(secrets),
                ),
                timeout=config.reset_timeout,
            )
        ).observation
        record.write("reset", {"observation_id": observation.observation_id})

        # --- the action server, declared inside the episode scratch ---------
        # A SHORT, session-unique name, deliberately: ``sun_path`` is bounded
        # (~104 bytes on macOS) and the real arm's scratch is already deep
        # (``<worktree>/scratch-homes/<run-name>``), so a descriptive
        # ``action-bridge-<episode>.sock`` landed at the limit and the guard in
        # ``ActionBridge.start`` refused it (measured: 100+ bytes).
        endpoint = Path(scratch_root) / f"b-{uuid.uuid4().hex[:8]}.sock"
        declaration = declare_action_server(
            config_dir=roots.config_path,
            endpoint=endpoint,
            surface=surface,
            cwd=roots.cwd_path,
        )
        resolved_sources = assert_declaration_resolved(
            cwd=roots.cwd_path,
            config_dir=roots.config_path,
            scratch_root=Path(scratch_root),
        )
        record.write(
            "declaration",
            {
                "server": declaration.server_name,
                "tool": declaration.tool_name,
                "config": str(declaration.config_path),
                "endpoint": str(declaration.endpoint),
                "sources": {name: str(path) for name, path in resolved_sources.items()},
            },
        )

        renderer = ObservationRenderer(config.artifact_root)
        bridge = ActionBridge(
            endpoint=endpoint,
            surface=surface,
            render=renderer.render,
            execute=_make_execute(adapter_session, config, record),
            # The re-bind seam: one read-only ``observe`` after a failed
            # execute the adapter declared committed (see ``_make_recover``).
            recover=_make_recover(adapter_session, config, spec.episode_id),
            max_steps=config.max_steps,
            record=lambda kind, payload: record.write(
                "action_" + kind,
                {
                    key: (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
                    for key, value in payload.items()
                },
            ),
            ask=_make_ask(adapter_session, config, answer_owner),
            # The completion challenge's inputs: the task as the reset published
            # it (the same text the runner's gate restates), the runner's own
            # gate controls (one channel-agnostic switch), and this channel's
            # reply sentence.
            instruction=observation.text or "",
            completion_gate=config.completion_gate,
            completion_challenges=config.completion_challenges,
            reply_guidance=CHALLENGE_REPLY_GUIDANCE.format(tool_name=declaration.tool_name),
        )
        bridge.arm(observation)
        await bridge.start()

        # --- the session turn (the engagement this module exists for) -------
        episode_session = validate_episode_session_spec(spec=session_spec)
        if episode_session.name is None and display_name:
            episode_session = replace(episode_session, name=display_name)

        #: The wedge stop: sent once, when the bridge ends the episode because
        #: its observation binding could not be re-established (see ``_sink``).
        wedge_stop_sent = False
        #: The budget stop: sent once, when the tool result that consumed the
        #: Nth admitted output epoch has been PAIRED into the transcript (see
        #: ``_sink``). Closes guest admission AT the budget instead of one
        #: decision later, via the loop's own graceful boundary.
        budget_stop_sent = False
        #: Whether the wall bound fired (``_on_wall`` below). The driver's own
        #: fact -- the abort reason never rides the end event -- and the
        #: terminal-reason classifier reads it after the turn.
        wall_fired = False
        #: The LAST ``agent_end`` the sink saw: the shared statement of why the
        #: episode's final turn ended, captured outside the record path so a
        #: torn record cannot cost the outcome its reason.
        last_turn_end: AgentEndEvent | None = None
        # ``handle``/``shutdown_result`` are declared BEFORE the try so the
        # exception path can always read them; this is where ``handle`` -- the
        # session the sink reads late (a wedge can only be set by a call, which
        # only happens once the session is prompting) -- comes into scope.

        def _sink(event: AgentEvent) -> None:
            # The sink sits on the engine's own dispatch path, where handler
            # errors are isolated and swallowed. A storage failure is already
            # classified by ``RecordSink.write`` itself and never raises; this
            # catch is for everything ELSE the record path can raise (an
            # event-fold error, a write after close) -- captured here, reported
            # in the outcome, and the run continues: a torn record stays
            # visible, and a paid episode is not lost to a logging fault.
            nonlocal wedge_stop_sent, budget_stop_sent, last_turn_end
            try:
                _on_event(record, bridge, event)
            except BaseException as error:  # noqa: BLE001 - see above
                if not record_errors:
                    record_errors.append(error)
            # The end frame is captured OUTSIDE the record path for the same
            # reason the wedge stop is separated below: a torn record must not
            # cost the outcome its terminal reason, and the reason is the run's
            # own fact just like its status and steps.
            if isinstance(event, AgentEndEvent):
                last_turn_end = event
            # Separated from the record path deliberately: the stop below is
            # the driver's reaction to the WEDGE, not record bookkeeping, so a
            # raise in it must not read as a torn record (and a record error
            # must not skip the stop). The engine catches a handler raise and
            # logs it; the outcome already carries the wedge reason either way.
            if not wedge_stop_sent and bridge.end_requested == BRIDGE_WEDGED_TERMINAL:
                # THE WEDGE STOP. The bridge could not re-establish its
                # observation binding and has ended the episode itself;
                # without this stop the model would spend whole turns on
                # refusals (the shape that cost arm 1796 tasks 001/013) and
                # the run would only end when the model gave up. The graceful
                # cancel is the boundary-respecting stop: honoured after the
                # failed call's result is paired into the transcript, before
                # the next model request is spent -- and it is requested from
                # the sink because this is the only driver-side seam that
                # observes the session while the prompt is awaited.
                wedge_stop_sent = True
                if handle is not None:
                    handle.session.request_graceful_cancel(
                        bridge.end_diagnostic or "action bridge wedged"
                    )
            # CLOSE GUEST ADMISSION AT THE BUDGET, not one decision later
            # (blueprint §3): the moment the tool result that consumed the Nth
            # admitted output epoch is PAIRED into the transcript, request the
            # loop's own graceful boundary -- the current group's remaining
            # calls finish and pair, and the next root request is never
            # spent. ``end_requested`` is deliberately NOT read here: a
            # terminal finish ends guest admission but NOT the SDK turn, and
            # without this stop the model spends one more root request on a
            # decision the terminal bridge can only refuse (measured by
            # review round 1: finish at epoch N recorded N+1 root requests
            # with no stop marker). The bridge's own budget refusal stays the
            # fail-closed fallback for any call that still arrives; the check
            # is a state test, never a tool-call-name heuristic (the
            # declaration names the ONE action tool the arm installed itself).
            # The stop seam (audit §48), three branches:
            # * the action tool's own result -- the boundary the R3 rounds
            #   pinned: fires on the Nth ADMITTED epoch's paired result
            #   regardless of the terminal ``end_requested``, so a terminal
            #   finish at epoch N never spends request N+1, and refused
            #   groups are covered too (their error result still ends
            #   through the action tool); the attempted clock raises the
            #   same boundary for groups the admitted count has not reached;
            # * the completed message itself, when it carries NO action-tool
            #   call -- a TurnEnd-fired request was read one boundary LATE
            #   (the loop asks its graceful question BEFORE yielding the
            #   event, ``loop.py``), so a local-tool-only Nth group let one
            #   more root request go out and its tool run (R1 C1, measured
            #   wire_calls=2). Requested AT MessageEnd, the loop's
            #   end-of-tools check reads it before the next prediction,
            #   while the message's own calls still dispatch and pair (the
            #   cancel is honoured only after the batch completes). Typed
            #   non-completion (``stop_reason == "error"``) is never a
            #   group;
            # * a TurnEnd -- kept as the backstop for any driver path that
            #   produced no MessageEnd fold -- fires while the episode is
            #   still LIVE (``end_requested is None``): a completed
            #   terminal turn has nothing left to stop.
            if (
                not budget_stop_sent
                and handle is not None
                and (
                    (
                        isinstance(event, MessageEndEvent)
                        and isinstance(event.message, Message)
                        and event.message.role == "assistant"
                        and event.message.stop_reason != "error"
                        and not any(
                            call.name == declaration.tool_name for call in event.message.tool_calls
                        )
                        and bridge.end_requested is None
                        and bridge.attempted_groups >= config.max_steps
                    )
                    or (
                        isinstance(event, ToolExecutionEndEvent)
                        and event.tool_name == declaration.tool_name
                        and (
                            bridge.steps >= config.max_steps
                            or bridge.attempted_groups >= config.max_steps
                        )
                    )
                    or (
                        isinstance(event, TurnEndEvent)
                        and bridge.end_requested is None
                        and bridge.attempted_groups >= config.max_steps
                    )
                )
            ):
                budget_stop_sent = True
                record.write(
                    "budget_stop",
                    {
                        "steps": bridge.steps,
                        "attempted_groups": bridge.attempted_groups,
                        "surface": declaration.tool_name,
                    },
                )
                handle.session.request_graceful_cancel("episode step budget reached")

        handle = await open_episode_session(
            spec=episode_session,
            roots=roots,
            on_event=_sink,
            session_opener=session_opener,
            confinement_root=confinement_root,
        )
        if owner_status is not None:
            owner_status.phase("running", detail="episode session open; arm starting")
        try:
            if not await _await_action_tool(handle, declaration, record):
                raise SessionArmError(
                    "the episode's action tool "
                    f"{declaration.tool_name} is not in the session after MCP settle; "
                    "refusing to prompt into a session that cannot act"
                )
            tool_names = handle.tool_names
            record.write("tools", {"names": list(tool_names)})
            # The first screen's blocks come from the bridge: ``arm`` rendered
            # it once and the completion challenge re-attaches the same blocks,
            # so rendering again here would append a duplicate turn to the
            # transcript the model is shown.
            text, images = split_prompt_content(bridge.initial_blocks)
            prompt = PROMPT_HEADER.format(tool_name=declaration.tool_name) + "\n" + text
            wall_timer: asyncio.TimerHandle | None = None

            def _on_wall() -> None:
                nonlocal wall_fired
                wall_fired = True
                handle.session.abort("episode wall budget")

            if max_wall_s is not None:
                # The wall bound is a KILL switch, not a cancellation of the
                # await: abort() ends the turn through the normal stop path,
                # so the record keeps the shape every other end has. The timer
                # is cancelled the moment the turn ends on its own.
                wall_timer = asyncio.get_running_loop().call_later(max_wall_s, _on_wall)
            try:
                await handle.session.prompt(prompt, images=images or None)
                # The completion gate's prose arm: a terminal message that
                # CLAIMS completion without a tool call gets the SAME challenge
                # a finish call gets, delivered against the state the model
                # last saw. Arm 1748's task_003 ended with "Done. Summary ..."
                # as prose and escaped the gate entirely; the re-prompt is one
                # harness-injected user turn (the reply channel's re-prompt, in
                # this channel's vocabulary -- the challenge text is the same
                # builder the finish-call arm uses). The loop is bounded by the
                # SHARED challenge budget inside the bridge, and it is skipped
                # when the wall bound fired: the wall is a kill switch, and a
                # re-prompt after it would spend past it.
                while not wall_fired:
                    challenge = bridge.prose_completion_challenge()
                    if challenge is None:
                        break
                    challenge_text, challenge_images = split_prompt_content(challenge)
                    await handle.session.prompt(
                        challenge_text,
                        images=challenge_images or None,
                        harness_injected=True,
                    )
            finally:
                if wall_timer is not None:
                    wall_timer.cancel()
        finally:
            # EXPLICIT OWNED-WORK SHUTDOWN, bounded, BEFORE any score
            # (blueprint §3). The caller owns its tree, so it says so here:
            # delegation admission closes, the turn aborts, and the canonical
            # disposal joins every child/grandchild job -- the same retained
            # task ordinary dispose joins, so a race with an already-running
            # disposal waits for THAT cleanup rather than starting a second
            # one. The request observer deliberately outlives this call (see
            # ``EpisodeSession.aclose``), because cancellation-finals for the
            # joining work arrive DURING it. A truthful receipt is recorded:
            # ``complete=False`` keeps pending ids and errors attached, and
            # the score below is refused rather than certifying unjoined
            # work. Sessions WITHOUT the owning surface (scripted doubles on
            # the older contract) get no receipt, and the same refusal
            # applies: unproven is not complete.
            try:
                shutdown_result = await handle.session.shutdown_owned_work(
                    "episode-turn-complete", timeout_s=config.cleanup_timeout
                )
            except (
                BaseException
            ) as error:  # noqa: BLE001 - the close-out must still reach the close
                # A RAISED first join is UNVERIFIED like an incomplete one: the
                # shutdown_result stays None and the retained hold below applies
                # (no release).
                record.write(
                    "error",
                    {"scope": "shutdown", "diagnostic": f"{type(error).__name__}: {error}"},
                )
            # THE RETAINED HOLD (R2 / review round 3, rewritten in round 4).
            # An incomplete receipt used to fall through to a bounded context
            # exit and a failed outcome whose durable story was a descriptor --
            # but the caller's own event loop then exited, and ``asyncio.run``
            # cancelled the canonical cleanup with it (measured in round 2).
            # Now EVERY state that is not a VERIFIED join -- None/raised,
            # cancelled, hook-failed, settled, complete=False -- is held inside
            # ``_hold_retained_join``, which returns only a complete receipt.
            # NOTHING is torn down while unverified: no context exit, no bridge
            # stop, no adapter close, no rescue, no descriptor discard, and no
            # elapsed-time state classifier (round 3 rejected timing as a state
            # proof). The episode keeps every scope, subscriber, RecordSink,
            # socket and owner-FD it holds; a frontend that gives up waiting may
            # detach, and the host goes on joining.
            owned_complete = bool(shutdown_result is not None and shutdown_result.complete)
            if not owned_complete:
                shutdown_result = await _hold_retained_join(
                    handle,
                    record,
                    owner_status,
                    first_result=shutdown_result,
                    record_errors=record_errors,
                    bridge=bridge,
                )
                owned_complete = True
            if owner_status is not None:
                owner_status.phase("closing", detail="canonical join completed")
            close_joined = False
            if owned_complete:
                try:
                    close_joined = await _bounded_close(handle, config.cleanup_timeout)
                except BaseException as error:  # noqa: BLE001 - a close-out must reach the flush
                    record.write(
                        "error",
                        {"scope": "close", "diagnostic": f"{type(error).__name__}: {error}"},
                    )
                if not close_joined:
                    record.write(
                        "close_retained",
                        {"timeout_s": config.cleanup_timeout, "steps": bridge.steps},
                    )
            try:
                _flush_request_observation(record, handle, close_joined=close_joined)
            except BaseException as error:  # noqa: BLE001 - a close-out must complete
                if not record_errors:
                    record_errors.append(error)

        steps = bridge.steps
        terminal_reason = bridge.end_requested
        stop_diagnostic: str | None = None
        if terminal_reason == "finish":
            status = "completed"
        elif terminal_reason == BRIDGE_WEDGED_TERMINAL:
            # The bridge could not re-establish its observation binding after a
            # transport failure and ended the episode itself. That is an
            # ENVIRONMENT death, not the model giving up: mapping it to the
            # default ``agent_stop`` read as capability and sent every F1
            # triage back into ``events.jsonl`` by hand.
            status = "failed"
        elif terminal_reason == "max-steps" or steps >= config.max_steps:
            status = "truncated"
            terminal_reason = terminal_reason or "max-steps"
        else:
            # The turn ended with no terminal batch. The runner calls the
            # close-equivalents of this an agent stop; the state reached is
            # still scored, exactly as a truncation is -- and the reason now
            # NAMES what ended it, where it used to be left unlabelled (see
            # ``_agent_stop_terminal_reason``). Status, steps and score are
            # untouched: only the record's account of the ending changes.
            status = "agent_stop"
            terminal_reason, stop_diagnostic = _agent_stop_terminal_reason(
                wall_fired=wall_fired,
                end=last_turn_end,
                message=bridge.last_assistant_message,
                challenges=bridge.completion_challenges_fired,
            )

        # THE OWNED-WORK GATE (blueprint §3): the score certifies the state
        # the episode reached, and it may only be taken once every piece of
        # owned work is joined. An incomplete or absent receipt is UNVERIFIED
        # -- a score over it would read as a finished episode while children
        # it owns might still touch the world. No score, no readiness claim.
        owned_complete = bool(shutdown_result is not None and shutdown_result.complete)
        #: R2's release gate: owned work STARTED (a session was opened) and is
        #: not verified joined -- so the worker must not be released.
        retain_resources = handle is not None and not owned_complete
        if shutdown_result is not None:
            record.write(
                "shutdown",
                {
                    "complete": shutdown_result.complete,
                    "reason": shutdown_result.reason,
                    "elapsed_ms": shutdown_result.elapsed_ms,
                    "pending_job_ids": list(shutdown_result.pending_job_ids),
                    "errors": list(shutdown_result.errors),
                    "observer_callback_errors": shutdown_result.observer_callback_errors,
                    "descendant_requests_in_flight": (
                        shutdown_result.descendant_requests_in_flight
                    ),
                },
            )
        if not owned_complete:
            # UNREACHABLE BY CONSTRUCTION (R3-A1): the retained hold above
            # returns only a VERIFIED join. Fail CLOSED if a future edit finds
            # a way here -- raising routes into the outer handler, which
            # RETAINS unverified owned work rather than sealing a failed
            # outcome over it.
            raise AssertionError("unverified owned work reached the close-out gate")
        # --- score, then cleanup, then close (mirrors _close_out) -----------
        score: Any = None
        score_error: BaseException | None = None
        if not owned_complete:
            # UNREACHABLE BY CONSTRUCTION (R3-A1), the same invariant as the
            # close-out gate above: a score over unverified owned work would
            # read as a finished episode while its children may still touch
            # the world. Fail closed into the retaining outer handler.
            raise AssertionError("unverified owned work reached the score gate")
        else:
            try:
                score = (
                    await adapter_session.score(
                        ScoreParams(
                            operation_id=f"score-{spec.episode_id}",
                            episode_id=spec.episode_id,
                        ),
                        timeout=config.score_timeout,
                    )
                ).score
                try:
                    # Through the sink's seal path, which spends the reserve when
                    # the volume is full. Its failure is a RECORD failure, not a
                    # score failure: the object above still reaches the outcome, so
                    # the paid score is never orphaned by the disk (measured: the
                    # voided runs' ``score.json`` survived while the outcome that
                    # would have linked it said ``score: null``).
                    record.seal_write(
                        record_root / "score.json",
                        json.dumps(score.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    )
                except RecordSinkError as error:
                    record_errors.append(error)
            except BaseException as error:
                # ADAPTER CONTRACT: score() returns a SCORED artifact or raises,
                # and a raise is not a reason to skip cleanup -- the worker is
                # still holding the resource. The failure is recorded here and
                # turned into the outcome by the handler below.
                score_error = error
                record.write(
                    "error",
                    {"scope": "score", "diagnostic": f"{type(error).__name__}: {error}"},
                )
        receipts: tuple[Any, ...] = ()
        # NOT-COMPLETE NEVER REACHES THIS LINE (R3-A1): the retained hold above
        # returns only a VERIFIED join, so the release path always runs here.
        # A future editor who finds a way past the hold must come back to this
        # boundary instead of re-adding a silent retained-release branch.
        try:
            receipts = await adapter_session.cleanup(
                CleanupParams(
                    operation_id=f"cleanup-{spec.episode_id}",
                    cleanup_plan=plan,
                    action_ids=tuple(action.action_id for action in plan.actions),
                ),
                timeout=config.cleanup_timeout,
            )
        except BaseException as error:
            record.write(
                "error",
                {"scope": "cleanup", "diagnostic": f"{type(error).__name__}: {error}"},
            )
            # A dead worker cannot produce evidence, and "attempted" alone is
            # never evidence of cleanup -- ``aggregate_cleanup`` is the one
            # statement of that rule (exactly one receipt per declared action;
            # only ``succeeded`` and ``not_needed`` aggregate clean).
            rescue_required = rescue_required or _cleanup_forces_rescue(plan, receipts)
            if rescue_required and descriptor is not None:
                rescue_complete = await _attempt_rescue(
                    descriptor=descriptor, config=config, secrets=secrets, rescue=rescue
                )
            elif descriptor is not None:
                _discard_descriptor(config)
        cleanup_done = True
        # A score failure on a run that already ended wedged is a CONSEQUENCE
        # of the wedge (a failed re-read poisons the session, and a poisoned
        # session cannot score), so the wedge stays the reported diagnostic --
        # the root cause, not its sequel. Everywhere else a score failure is
        # the outcome's reason, exactly as before.
        diagnostic = bridge.end_diagnostic or stop_diagnostic
        if score_error is not None:
            if terminal_reason != BRIDGE_WEDGED_TERMINAL:
                raise score_error
            diagnostic = (
                f"{diagnostic or 'action bridge wedged'} "
                f"Scoring the state reached also failed: {_record_diagnostic(score_error)}."
            )
        # The record's own failure is a property of the RECORD, not of the run:
        # the run is reported with its real status, steps and score, and the
        # record says what it lost. The old shape raised ``SessionArmError``
        # here, which replaced the whole outcome with ``failed / steps: 0`` and
        # burned the product of the spend.
        record_failure = _first_record_failure(record, record_errors)
        outcome = SessionArmOutcome(
            status=status,
            episode_id=spec.episode_id,
            record_root=record_root,
            score=score,
            steps=steps,
            attempted_groups=bridge.attempted_groups,
            terminal_reason=terminal_reason,
            diagnostic=diagnostic,
            record_incomplete=record_failure is not None,
            record_diagnostic=(
                _record_diagnostic(record_failure) if record_failure is not None else None
            ),
            rescue_required=rescue_required,
            rescue_complete=rescue_complete,
            tool_names=tool_names,
            resolved_sources=resolved_sources,
            duration_ms=int(time.time() * 1000) - started_ms,
        )
    except BaseException as error:
        # Every failure still tears the worker down; a leaked cloud resource is
        # a failed episode regardless of what the model did. Cleanup is
        # attempted even when the failure landed mid-turn -- the runner's
        # failure path runs the same close-out for the same reason -- and a
        # cleanup that cannot complete forces the rescue path. THE SAME
        # RELEASE GATE AS THE NORMAL PATH (R2): when owned work STARTED and is
        # unverified, nothing is released here either; the descriptor stays
        # for a later verified teardown.
        exc_retain = handle is not None and not (
            shutdown_result is not None and bool(shutdown_result.complete)
        )
        if exc_retain and handle is not None:
            # RETAIN FIRST (R3-A1): an exception that landed before the join
            # completed does not make owned work verified -- a mid-turn
            # BaseException, a failure inside the retained hold, or a path
            # that never got a receipt. The host HOLDS in the same shared
            # retained join until the join is VERIFIED (it returns only
            # complete), and only then does this failure path proceed to
            # report and release. Nothing here exits in front of live owned
            # scopes.
            shutdown_result = await _hold_retained_join(
                handle,
                record,
                owner_status,
                first_result=shutdown_result,
                record_errors=record_errors,
                bridge=bridge,
            )
            exc_retain = False  # the retained join returned VERIFIED
        retain_resources = exc_retain
        if adapter_session is not None and plan is not None and not cleanup_done:
            if exc_retain:
                rescue_required = True
                record.write(
                    "cleanup_retained",
                    {
                        "reason": "owned work unverified; release deferred",
                        "pending_job_ids": (
                            list(shutdown_result.pending_job_ids)
                            if shutdown_result is not None
                            else []
                        ),
                        "descriptor_retained": descriptor is not None,
                        "descriptor_id": (
                            descriptor.descriptor_id if descriptor is not None else None
                        ),
                    },
                )
            else:
                try:
                    receipts = await adapter_session.cleanup(
                        CleanupParams(
                            operation_id=f"cleanup-{spec.episode_id}",
                            cleanup_plan=plan,
                            action_ids=tuple(action.action_id for action in plan.actions),
                        ),
                        timeout=config.cleanup_timeout,
                    )
                    rescue_required = rescue_required or _cleanup_forces_rescue(plan, receipts)
                except BaseException:
                    rescue_required = True
        if rescue_required and descriptor is not None and not exc_retain:
            rescue_complete = await _attempt_rescue(
                descriptor=descriptor, config=config, secrets=secrets, rescue=rescue
            )
        # The terminal error line comes FIRST: it is itself a sink write the
        # full volume may refuse, and a record failure it raises belongs in the
        # outcome below rather than after it was computed.
        record.write("error", {"diagnostic": f"{type(error).__name__}: {error}"})
        record_failure = _first_record_failure(record, record_errors)
        outcome = SessionArmOutcome(
            status="failed_pre_bundle" if adapter_session is None else "failed",
            episode_id=spec.episode_id,
            record_root=record_root,
            terminal_reason=_failure_terminal_reason(error),
            diagnostic=f"{type(error).__name__}: {error}",
            rescue_required=rescue_required,
            rescue_complete=rescue_complete,
            record_incomplete=record_failure is not None,
            record_diagnostic=(
                _record_diagnostic(record_failure) if record_failure is not None else None
            ),
            duration_ms=int(time.time() * 1000) - started_ms,
        )
    finally:
        # THE BRIDGE GATE (R2, review rounds 2-3): the action server stops only
        # when this close-out retains NO owned work. The old comment here
        # asserted that no retained child could call the bridge because
        # children are denied the action tool and the root turn is over --
        # round 2 rejected that assertion as unproven, and the conservative
        # direction is to leave the socket up whenever ownership is retained:
        # it is caller-local plumbing, so a host that goes on joining may
        # still need it, and a host that exits takes it down with the process.
        # Nothing here releases a worker resource either way; the worker-side
        # gate lives in ``_close_adapter_session``.
        if bridge is not None and not retain_resources:
            await bridge.stop()
        await _close_adapter_session(
            adapter_session,
            supervisor,
            spec,
            config,
            rescue_required,
            retain_resources=retain_resources,
        )

    # --- the seal -----------------------------------------------------------
    # Every other writer has stopped above, so nothing is competing with these
    # writes and the record's own failure state is final. The terminal marker is
    # tolerant (it names what the record lost, in the record itself); the outcome
    # goes through the seal path (reserve-backed), and the reserve is released
    # no matter how either went. The outcome OBJECT returns in every case -- the
    # runner prints it to the log -- so a dead volume costs the disk copy of the
    # summary, never the summary.
    if record.failure is not None:
        record.write(
            "record_incomplete",
            {
                "diagnostic": record.failure.sentence,
                "out_of_room": record.failure.out_of_room,
                "bytes_written": record.bytes_written,
                "lines_written": record.lines_written,
                "failures": record.failures,
            },
        )
    record.close()
    seal_error: RecordSinkError | None = None
    try:
        record.seal_write(
            record_root / "outcome.json",
            json.dumps(_outcome_json(outcome), indent=2, sort_keys=True) + "\n",
        )
    except RecordSinkError as error:
        # A failed outcome seal is not a stderr-only note: the disk copy of the
        # summary is part of the archive ``record_incomplete`` describes, so the
        # fact re-enters the outcome below -- the object the runner prints and
        # returns -- rather than stopping at the console (review round 1,
        # R1-F3). ``SessionArmOutcome`` is frozen, so this is a ``replace``.
        seal_error = error
        print(
            f"the episode outcome could not be written to {record_root}: {error.sentence}",
            file=sys.stderr,
        )
    finally:
        record.release_reserve()
    if seal_error is not None:
        # An EARLIER record failure stays the diagnostic when there is one (it
        # is the classified root cause; this is its consequence); the seal
        # failure's own sentence is used only when the record path was clean up
        # to the seal.
        outcome = replace(
            outcome,
            record_incomplete=True,
            record_diagnostic=outcome.record_diagnostic or _record_diagnostic(seal_error),
        )
    return outcome


async def _await_action_tool(
    handle: Any,
    declaration: ActionServerDeclaration,
    record: RecordSink,
    *,
    timeout_s: float = 30.0,
) -> bool:
    """Hold the prompt until the action tool provably exists in the session.

    MCP discovery is asynchronous BY DESIGN: a server deferred past the 250 ms
    startup gate connects on a background continuation and its tools merge on
    settle (``session_factory`` -> ``apply_preload``). The episode's FIRST
    request is the one request that must not race that settle -- it is the
    request whose tool array carries the action tool -- and a session that
    prompts without it answers every model call with "Tool not found: <action
    tool>" (measured: the whole pilot turn failed exactly that way when the
    prompt went out ~10 ms after open). So the arm waits, bounded, and REFUSES
    to prompt when the tool never arrives: absence is a loud, recorded failure,
    never a silent episode that cannot act. The wait is the SDK's typed
    ``Session.wait_for_tools``: the arm no longer hand-rolls the manager +
    live-inventory composition, so this gate and every external caller share
    ONE readiness definition.
    """

    session = handle.session
    readiness = await session.wait_for_tools((declaration.tool_name,), timeout_s=timeout_s)
    present = readiness.status == "ready"
    startup = getattr(session, "mcp_startup", None)
    record.write(
        "mcp_settle",
        {
            "settled": readiness.discovery_settled,
            "readiness_status": readiness.status,
            "action_tool_present": present,
            "configured": list(getattr(startup, "configured", ()) or ()),
            "connected": list(getattr(startup, "connected", ()) or ()),
            "failures": {
                name: str(message)
                for name, message in (getattr(startup, "failures", None) or {}).items()
            },
        },
    )
    return present


def _cleanup_forces_rescue(plan: CleanupPlan, receipts: Sequence[Any]) -> bool:
    """Whether cleanup leaves the worker unconfirmed: the runner's own rule.

    ``aggregate_cleanup`` requires exactly one receipt per declared action and
    only aggregates ``succeeded``/``not_needed`` as clean; an empty or
    incomplete receipt set therefore forces the rescue path, exactly as the
    runner's ``_run_cleanup`` does (which mints incomplete receipts for a dead
    worker so this same aggregation turns red rather than reading as success).
    """

    if not receipts:
        return True
    try:
        return bool(aggregate_cleanup(plan, tuple(receipts)).rescue_required)
    except ValueError:
        return True


#: The bound for each bounded RE-QUERY of the owning receipt while the host is
#: RETAINED on unverified owned work. The canonical task is either live (the
#: call waits on it, so the loop cannot spin) or settled (the call returns
#: quickly and the pause below paces the loop).
_OWNED_REJOIN_TIMEOUT_S = 5.0

#: Pause between retained-join attempts once an attempt returned WITHOUT
#: waiting a real interval. A settled canonical (cancelled, hook-failed) is
#: never recreated, so re-asking cannot change its answer -- the hold is a
#: lifetime guarantee for the scopes it keeps alive, not a retry loop, and
#: the pause is what keeps it from spinning (review round 4 rejected the
#: elapsed-time classifier that used to END the hold instead).
_OWNED_REJOIN_PACE_S = 5.0


async def _hold_retained_join(
    handle: Any,
    record: Any,
    owner_status: Any,
    *,
    first_result: Any,
    record_errors: list[BaseException],
    bridge: Any,
) -> Any:
    """Hold the host until owned work is a VERIFIED join; return only complete.

    THE ONE PLACE THE RETENTION CONTRACT LIVES (R2, review rounds 3-4; used by
    the normal close-out AND the failure path so the two cannot drift). Every
    state that is not ``complete`` -- a None/raised first join, cancelled or
    hook-failed settled canonicals, ``complete=False`` with pending jobs -- is
    RETAINED here: the caller's context, bridge, RecordSink, subscriber,
    socket and owner-FD stay alive, no final seal or terminal status is
    written, and the loop re-asks the PUBLIC bounded join (paced when an
    attempt returns without waiting) until it completes. A settled canonical
    is terminal and is NEVER recreated or re-admitted: the loop's only job is
    to keep the scopes alive and keep asking, and the caller proceeds only
    when a receipt proves the join happened. Cancellation does not outrank
    retention: ``CancelledError`` is a BaseException and is caught here like
    any other join failure -- a cancelling host is not a verdict about owned
    work.
    """
    if first_result is not None and first_result.complete:
        return first_result
    provisional_receipt = (
        {
            "pending_job_ids": list(first_result.pending_job_ids),
            "errors": list(first_result.errors)[:4],
            "elapsed_ms": first_result.elapsed_ms,
        }
        if first_result is not None
        else {
            "pending_job_ids": [],
            "errors": ["the first bounded join returned no receipt"],
            "elapsed_ms": None,
        }
    )
    try:
        record.write(
            "provisional",
            {
                "reason": "owned work unverified; holding the host until "
                "the canonical disposal joins",
                **provisional_receipt,
                "steps": bridge.steps,
            },
        )
    except BaseException as error:  # noqa: BLE001 - the join outranks the record
        if not record_errors:
            record_errors.append(error)
    if owner_status is not None:
        try:
            owner_status.phase(
                "provisional",
                detail="owned work unverified; host retained until the join",
                receipt=provisional_receipt,
            )
        except BaseException as error:  # noqa: BLE001 - observation is not disposition
            # A status writer is advisory, like RecordSink. Losing either must
            # never unwind the owner before the public join proves cleanup.
            if not record_errors:
                record_errors.append(error)
    rejoin_error_recorded = False
    while True:
        pause = _OWNED_REJOIN_PACE_S
        try:
            refreshed = await handle.session.shutdown_owned_work(
                "episode-turn-complete", timeout_s=_OWNED_REJOIN_TIMEOUT_S
            )
            if refreshed is not None and refreshed.complete:
                return refreshed
        except BaseException as error:  # noqa: BLE001 - keep joining
            if not rejoin_error_recorded:
                rejoin_error_recorded = True
                try:
                    record.write(
                        "error",
                        {
                            "scope": "shutdown-rejoin",
                            "diagnostic": f"{type(error).__name__}: {error}",
                        },
                    )
                except BaseException:  # noqa: BLE001 - the join outranks the record
                    pass
            pause = _OWNED_REJOIN_TIMEOUT_S
        try:
            await asyncio.sleep(pause)
        except BaseException:  # noqa: BLE001 - cancellation is not a verified disposition
            pass


async def _bounded_close(handle: EpisodeSession, timeout_s: float) -> bool:
    """Exit the session's context under a BOUND; True when it joined and returned.

    WHY THIS EXISTS (R2, review round 1). ``handle.aclose()`` runs the SDK
    context exit, which JOINS the canonical disposal -- the same cleanup
    ``shutdown_owned_work`` joins. After an incomplete shutdown that join used
    to be awaited with no bound, so a stuck cleanup hung the whole close-out
    (measured: still pending at .03 s against a held hook). Here the exit is
    awaited inside ``asyncio.wait``: when it completes, the join is verified;
    when the bound fires the exit is CANCELLED and abandoned -- safely,
    because ``dispose`` joins the canonical task under its own shield, so
    cancelling THIS waiter never cancels the shared cleanup.

    WHAT RETENTION MEANS, EXACTLY (architect note, 2026-10-12). Within a LIVE
    process the canonical cleanup keeps running and stays joinable by a later
    in-process entrant; the waiter cancellation only drops THIS join attempt.
    The CLI, however, returns through ``asyncio.run`` in ``run_episode``,
    which CANCELS pending tasks at teardown -- so nothing here may claim the
    owned task survives process exit, and no receipt may read as completed
    cleanup. The DURABLE retention across the caller's exit is the rescue
    DESCRIPTOR plus the TTL/outer-supervisor story, never the in-process
    task; see ``_close_adapter_session`` for the no-release gate that pairs
    with it. Deliberately NOT ``asyncio.wait_for``: ``wait_for`` waits
    indefinitely for the cancelled coroutine to actually finish, which is
    exactly the unbounded re-entry this bound exists to prevent.
    """

    exit_task = asyncio.ensure_future(handle.aclose())
    done, _pending = await asyncio.wait({exit_task}, timeout=timeout_s)
    if exit_task in done:
        failure = exit_task.exception()
        if failure is not None:
            raise failure
        return True
    exit_task.cancel()
    return False


def _flush_request_observation(
    record: RecordSink, handle: EpisodeSession, *, close_joined: bool | None = None
) -> None:
    """Serialize the queued request-final rows and price them ONCE.

    THE TWO MONEY FIGURES, kept separate (blueprint §2; ``cost_summary`` is
    the ONE canonical pricer -- no second ledger): ``recorded_usd`` is the
    receipts actually stamped, i.e. known charged money; ``estimated_usd``
    is the same canonical table projection for rows without a stamp, so a
    direct-route run keeps an estimate even though ``recorded_only`` alone
    would lose it. ``missing_usage_request_ids`` names rows where NO usage
    frame arrived: unknown, NEVER a priced zero. Cache read/write and
    reasoning ride inside the usage buckets the canonical pricer already
    separates; nothing here re-adds them to input/output.

    INVOICE COVERAGE IS CONSERVATIVE BY CONSTRUCTION (R1, review round 1).
    The observer reports LOGICAL-REQUEST finals, and a final's
    ``usage_coverage == "reported_final"`` covers only the LAST attempt's
    receipt: a retry that billed an earlier attempt is not reconciled here
    (measured: a two-attempt request billed .03 + .12 while the deduplicated
    final recorded .12 with coverage "reported_final"). So
    ``recorded_is_lower_bound`` is TRUE whenever any rows were observed --
    the recorded total may understate the bill -- and it flips False only
    under a future invoice-wide reconciliation, never from the per-request
    coverage label alone. ``invoice_coverage`` names the coverage actually
    held (``logical_final_only``).
    """

    from local_operator.model.costs import cost_summary

    rows = handle.request_rows
    usages = handle.request_usages
    recorded_usd, recorded_unknown = cost_summary(usages, recorded_only=True)
    estimated_usd, estimated_unknown = cost_summary(usages, recorded_only=False)
    purposes: dict[str, int] = {}
    coverage_counts: dict[str, int] = {}
    missing_usage_request_ids: list[str] = []
    for row in rows:
        purpose = str(row.get("purpose") or "unknown")
        purposes[purpose] = purposes.get(purpose, 0) + 1
        coverage = str(row.get("usage_coverage") or "missing")
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
        if row.get("usage") is None:
            missing_usage_request_ids.append(str(row.get("request_id") or ""))
    record.write(
        "requests",
        {
            "count": len(rows),
            "purpose_counts": purposes,
            "usage_coverage_counts": coverage_counts,
            "rows": rows,
            "missing_usage_request_ids": missing_usage_request_ids,
            "recorded_usd": recorded_usd,
            "recorded_unknown": recorded_unknown,
            "recorded_is_lower_bound": bool(rows) or bool(recorded_unknown),
            "invoice_coverage": "logical_final_only",
            "estimated_usd": estimated_usd,
            "estimated_unknown": estimated_unknown,
            # PROVISIONAL ACCOUNTING (R2 acceptance): when the session's close
            # did not join, final rows can still arrive and this snapshot is
            # explicitly NOT a sealed complete account -- the flag says which
            # it is. ``None`` keeps the legacy callers' shape.
            "close_joined": close_joined,
        },
    )


def _on_event(record: RecordSink, bridge: ActionBridge, event: AgentEvent) -> None:
    """One sink for the whole session stream: record it, fold it, count it."""

    bridge.fold(event)
    record.write("agent_event", {"event": printable_event(event)})


def _make_execute(
    adapter_session: VerifiedAdapterSession, config: EpisodeConfig, record: RecordSink
) -> ExecuteBatch:
    """The execution seam: one batch, with the runner's read-back recovery.

    Copied in ORDER from ``EpisodeRunner._execute_with_observation_recovery``,
    which is the one place the contract "retry only what the adapter declared
    committed" is implemented; the two must not drift, so this mirrors its
    attempt loop, its fresh operation ids, and its decision that any other
    failure propagates.
    """

    async def execute(batch: ActionBatch) -> Any:
        from local_operator.evaluation.runner.episode import _adapter_batch_id

        operation_id = f"exec-{uuid.uuid4().hex[:12]}"
        params = ExecuteParams(
            operation_id=operation_id,
            action_batch=batch,
            action_batch_id=_adapter_batch_id(batch),
        )
        attempts = max(0, config.observation_retry_attempts)
        for attempt in range(attempts + 1):
            try:
                if attempt == 0:
                    if config.execution_overhead_seconds_per_action:
                        return await adapter_session.execute(
                            params,
                            timeout=config.step_timeout,
                            execution_overhead_seconds_per_action=(
                                config.execution_overhead_seconds_per_action
                            ),
                        )
                    return await adapter_session.execute(params, timeout=config.step_timeout)
                return await adapter_session.resume_observation(
                    params.model_copy(update={"operation_id": f"{operation_id}-obs{attempt}"}),
                    timeout=config.step_timeout,
                )
            except Exception as error:
                if attempt == attempts or not _rebindable_failure(error):
                    raise
                record.write(
                    "observation_retry",
                    {
                        "attempt": attempt + 1,
                        "attempts": attempts,
                        "detail": str(error)[:500],
                    },
                )
                await asyncio.sleep(config.observation_retry_delay)

    return execute


def _make_recover(
    adapter_session: VerifiedAdapterSession, config: EpisodeConfig, episode_id: str
) -> RecoverObservation:
    """The re-bind seam: ONE bounded read of the environment's current state.

    WHY ONE, AND WHY THIS CALL. The bridge calls this only after an execute
    failure the adapter declared committed (``phase == "observation"``: the
    batch applied, only the read-back lost) and whose configured read-back
    retries have already run out -- so this is NOT a retry of that read-back
    (raising those is a comparability-affecting change, see INFRA.md) but a
    fresh state read through the read-only ``observe`` method every adapter
    already answers (``reset_start`` reads observation zero through it). It
    re-establishes what the bridge lost: a current observation the token can
    be re-armed with. When it cannot -- the environment is still unable to
    answer, or the read is refused -- the bridge ends the episode with the
    wedge reason instead of leaving the model to spin (see
    ``ActionBridge._end_wedged``).

    THE BOUND AND ITS COST. Exactly one call per failed batch, none at all on
    the healthy path; the call carries the same ``step_timeout`` every other
    environment call in this arm carries, plus the RPC layer's 1 s cancel
    grace (181 s of wall in the campaign's config: the 180 s budget + the 1 s
    grace), so the added worst case is one step-timeout plus its cancel grace
    on a run that would otherwise have died anyway.
    """

    async def recover() -> Observation:
        result = await adapter_session.observe(
            ObserveParams(episode_id=episode_id), timeout=config.step_timeout
        )
        return result.observation

    return recover


def _make_ask(
    adapter_session: VerifiedAdapterSession, config: EpisodeConfig, answer_owner: str
) -> Callable[[ActionBatch], Awaitable[AskUserExchangeResult | None]] | None:
    """The ask seam, adapter-owned answers only (the pilot arm's posture).

    Returns the CANONICAL ``AskUserExchangeResult`` (R4, review round 1): the
    public receipt -- ask_id, request digest, accepted/declined, public answer
    -- crosses to the bridge so it can ride the native tool result's content.
    Returning only ``result.answer`` (the old shape) threw the receipt away at
    this very boundary, and the MCP forwarding layer drops ``details``, so the
    next root request could never see it. One exchange RPC, unchanged.
    """

    if answer_owner != "adapter":
        # Host-owned answers need a responder surface the pilot arm does not
        # carry; refusing here turns a silent "the model asked and nobody could
        # answer" into a named precondition. The reply-channel arm is the one
        # that answers these today.
        return None

    async def ask(batch: ActionBatch) -> AskUserExchangeResult:
        from local_operator.evaluation.runner.episode import _ask_prompt

        ask_id, prompt = _ask_prompt(batch)
        begin = AskUserExchangeParams(
            operation_id=f"ask-begin-{ask_id}",
            episode_id=batch.episode_id,
            ask_id=ask_id,
            prompt=prompt,
        )
        adapter_session.begin_ask(begin)
        result = await adapter_session.finish_ask(begin, timeout=config.step_timeout)
        return result

    return ask


def _refuse_undeclared_disclosed_infra(
    spec: EpisodeSpec, selector: Any, requirements: RequirementsResult
) -> None:
    """Refuse a disclosed infra value the adapter build does not understand.

    Same rule as the runner's (``EpisodeRunner._refuse_undeclared_disclosed_infra``):
    an override the adapter would silently drop is worse than a failed episode,
    because the record would then disagree with the run.
    """

    from local_operator.evaluation.runner.episode import _DISCLOSED_INFRA_VALUES

    declared = {requirement.name for requirement in requirements.requirements}
    supplied = {value.name for value in spec.infra_values}
    undeclared = sorted((supplied & _DISCLOSED_INFRA_VALUES) - declared)
    if undeclared:
        raise UndeclaredDisclosedInfra(
            f"adapter {selector.adapter_id!r} version {selector.version!r} does not "
            f"declare {undeclared}, so the value would be silently ignored while the "
            "recorded run showed it applied; rebuild the adapter workspace and selector, "
            "or drop the value"
        )


def _persist_descriptor(
    *,
    config: EpisodeConfig,
    spec: EpisodeSpec,
    selector: Any,
    handshake: Handshake,
    plan: CleanupPlan,
) -> RescueDescriptor:
    """Persist the rescue descriptor exactly as the runner does, twice."""

    descriptor = RescueDescriptor(
        schema_version=ADAPTER_SCHEMA_VERSION,
        selector=selector,
        handshake=handshake,
        episode_id=spec.episode_id,
        cleanup_plan=plan,
        secret_refs=spec.secret_refs,
        infra_values=spec.infra_values,
        artifact_root=str(config.artifact_root),
    )
    persist_rescue(config.rescue_root, descriptor)
    return descriptor


def _discard_descriptor(config: EpisodeConfig) -> None:
    try:
        discard_rescue(config.rescue_root)
    except OSError:
        pass


async def _attempt_rescue(
    *,
    descriptor: RescueDescriptor,
    config: EpisodeConfig,
    secrets: Sequence[ResolvedSecret],
    rescue: Any,
) -> bool:
    try:
        aggregate = await rescue(descriptor, secrets=tuple(secrets))
    except BaseException:
        return False
    complete = bool(aggregate.complete)
    if complete:
        _discard_descriptor(config)
    return complete


async def _close_adapter_session(
    adapter_session: VerifiedAdapterSession | None,
    supervisor: Any,
    spec: EpisodeSpec,
    config: EpisodeConfig,
    rescue_required: bool,
    *,
    retain_resources: bool = False,
) -> None:
    """Mirror ``EpisodeRunner._close_session``: close, then terminate.

    THE RETENTION GATE (R2): with ``retain_resources`` set (owned work
    unverified), NOTHING is released -- no ``close_session`` RPC and no
    supervisor terminate. The adapter close would tear the session down
    under possibly-live owned calls, and the terminate kills the local
    worker process out from under a cleanup that may still be using it. The
    worker process is self-reaping in the long run (when this caller exits,
    its owner-fd EOF ends it -- the same rescue invariant used when a parent
    dies unexpectedly), and the DURABLE retention is the rescue descriptor
    plus the TTL schedule: a fresh verified join, or a separately authorized
    supervisor teardown, precedes any release. This is a caller-side
    disposition only -- it neither claims the in-process cleanup completed
    nor that it survived the caller's exit.
    """

    if retain_resources:
        return
    if adapter_session is not None and not rescue_required:
        try:
            await adapter_session.close(
                CloseParams(
                    operation_id=f"close-{spec.episode_id}",
                    episode_id=spec.episode_id,
                ),
                timeout=config.cleanup_timeout,
            )
        except BaseException:
            pass
    if supervisor is not None:
        try:
            await supervisor.terminate()
        except BaseException:
            pass


def _outcome_json(outcome: SessionArmOutcome) -> dict[str, Any]:
    score = outcome.score
    score_payload: Any = None
    if score is not None:
        score_payload = score.model_dump(mode="json") if hasattr(score, "model_dump") else score
    return {
        "arm": SESSION_ARM_ID,
        "status": outcome.status,
        "episode_id": outcome.episode_id,
        "record_root": str(outcome.record_root) if outcome.record_root else None,
        "score": score_payload,
        "steps": outcome.steps,
        "attempted_groups": outcome.attempted_groups,
        "terminal_reason": outcome.terminal_reason,
        "diagnostic": outcome.diagnostic,
        "rescue_required": outcome.rescue_required,
        "rescue_complete": outcome.rescue_complete,
        "record_incomplete": outcome.record_incomplete,
        "record_diagnostic": outcome.record_diagnostic,
        "tool_names": list(outcome.tool_names),
        "resolved_sources": {name: str(path) for name, path in outcome.resolved_sources.items()},
        "duration_ms": outcome.duration_ms,
    }

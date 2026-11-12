"""The harness completion verifier: a fresh-context check of a finished task.

WHY THIS EXISTS. A model that declares a task done is the worst-placed reader of
whether it is: on the OSWorld session arm (main29, deepseek-flash) four of five
analysed non-apparatus failures were wrong deliverables that an independent read
against the verbatim request would plausibly have caught -- a wrong cell plus
false arithmetic in the model's own finish text, opacity set where transparency
was asked, a chart the task said to follow deliberately deleted, ``*-filled.pdf``
copies made while the named originals stayed untouched -- and the model's own
self-check challenge caught none. The model never delegated (0 ``task`` calls in
10 episodes), and a delegated child could not reach the remote machine anyway.
So the HARNESS launches the check: deterministic, bounded and attributable.

It is a generic capability on ``Session`` (:meth:`Session.verify_completion`),
built from the same subagent machinery every child uses. An embedder supplies
whatever lets the check SEE the deliverable as ``host_tools`` (the OSWorld arm
hands it a lock-free ``inspect_command``); nothing here knows about OSWorld.

THE ROLE IS DEFINED HERE, NOT AS A SEED. A harness-launched check runs
unattended on a capability set chosen by code, so it must not depend on a
mutable registry row: an operator row named ``verifier`` with ``write`` in it
must not widen it, and an unresolvable name must not degrade it into a full
child (which is what ``_resolve_role`` does for a typo'd ``task`` role). The
scout has the same rule (``SCOUT_TOOL_ALLOWLIST``). It is also not offered to
the delegating model as a roster role: nothing here is a ``task`` target.

Constraints:

* No ``write``/``edit``/``task``/``web_*``: the allowlist plus the
  :class:`~local_operator.harness.subagent.ChildSurface` ceiling, which also
  removes the ``web_*`` floor every role-limited child otherwise gets.
* ``bash`` is the write-confined, no-network shell of
  :data:`~local_operator.harness.subagent.READ_ONLY_SHELL_ROLES`.
* Bounded: :class:`VerifierLimits` cancels the child at the first of its
  request, host-command and wall caps.
* The verdict is PARSED from a fixed last line; anything else is
  ``INCONCLUSIVE``, never a pass.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

if TYPE_CHECKING:
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness.types import AgentTool

VERIFIER_ROLE = "verifier"

#: Read-only reach. ``bash`` is confined by role (``READ_ONLY_SHELL_ROLES``).
VERIFIER_TOOL_ALLOWLIST: tuple[str, ...] = ("read", "glob", "grep", "bash")

VerdictStatus = Literal["PASS", "FAIL", "INCONCLUSIVE"]

#: The role text. It is generic on purpose: the embedder's prompt carries the
#: request, the claim and the evidence; this carries only how to judge them.
VERIFIER_INSTRUCTIONS = """\
You are an INDEPENDENT completion verifier. Another agent says it finished the
task below. You did not do the work, and you must not do or repair any of it.

Judge the DELIVERABLE against the REQUEST as written, not against the claim.
The claim tells you where to look; it is not evidence. Inspect the actual
artifacts (files, cells, settings, documents) with the tools you have.

1. Split the request into its concrete requirements, including the ones that
   are easy to miss: exact names and paths (an edited copy is not the named
   file), exact values and units, things that must stay unchanged, and
   wording such as "transparency" versus "opacity".
2. For each requirement, look at the real state and mark it MET, UNMET or
   UNVERIFIED, with the evidence you saw (command and output, value read).
   Re-derive any arithmetic yourself.
3. Change nothing. Read only; never write, move, delete, save or open an
   editor on the artifacts.

Be brief. End with exactly one final line:
VERDICT: PASS     -- every requirement is MET.
VERDICT: FAIL     -- at least one requirement is UNMET; the lines above name it.
VERDICT: INCONCLUSIVE -- you could not check something that matters.
"""

#: How much of each prompt field rides along. The request and claim are short in
#: practice; the observation text (an accessibility tree) is not, and the
#: verifier's cost is bounded by its request cap only if its prompt is bounded.
MAX_REQUEST_CHARS = 8_000
MAX_CLAIM_CHARS = 4_000
MAX_OBSERVATION_CHARS = 12_000
#: Findings travel back into the parent's challenge text; bound them too.
MAX_FINDINGS_CHARS = 4_000

_VERDICT_LINE = re.compile(r"^\s*\**\s*VERDICT\s*:\s*\**\s*(PASS|FAIL|INCONCLUSIVE)\b", re.I)


def verifier_profile() -> "AgentProfile":
    """The code-defined verifier role (see the module docstring for why)."""

    from local_operator.agent_profiles import AgentProfile

    return AgentProfile(
        name=VERIFIER_ROLE,
        label="Verifier",
        description="Independent read-only check of a finished task's deliverable.",
        instructions=VERIFIER_INSTRUCTIONS,
        tools=VERIFIER_TOOL_ALLOWLIST,
    )


@dataclass(frozen=True)
class VerifierLimits:
    """Caps that END the verifier, each recorded as ``ended_by`` when it fires.

    Defaults are the design's (verifier-design.md "Flag"): 12 requests, 10 host
    commands, 180 s. They are NOT measured distributions -- no verifier has run
    yet -- so they are deliberately loose against the work (a typical check is
    a handful of reads) and are recorded on every ``verifier_started`` row so a
    first run can calibrate them. ``max_requests`` counts request-final rows
    carrying the verifier's job id; the child is serial, so cancellation lands
    at most one request past the cap.
    """

    max_requests: int = 12
    max_commands: int = 10
    wall_s: float = 180.0

    def __post_init__(self) -> None:
        for name in ("max_requests", "max_commands"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        wall = self.wall_s
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall):
            raise ValueError("wall_s must be a finite number")
        if wall <= 0:
            raise ValueError("wall_s must be positive")

    def as_record(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_commands": self.max_commands,
            "wall_s": float(self.wall_s),
        }


@dataclass(frozen=True)
class VerifierVerdict:
    """What one verification produced. ``status`` is never inferred from prose.

    ``ended_by``: ``"verdict"`` (the child answered), ``"requests"``,
    ``"commands"``, ``"wall"`` (a cap cancelled it), ``"failed"`` (the child
    errored) or ``"refused"`` (it never launched: admission, shutdown).
    """

    status: VerdictStatus
    findings: str
    ended_by: str
    job_id: str | None = None
    requests: int = 0
    commands: int = 0
    wall_s: float = 0.0
    model: str | None = None
    tools: tuple[str, ...] = ()
    transcript_dir: str | None = None
    usage: dict[str, Any] | None = field(default=None)

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": self.findings,
            "ended_by": self.ended_by,
            "job_id": self.job_id,
            "requests": self.requests,
            "commands": self.commands,
            "wall_s": round(self.wall_s, 3),
            "usage": self.usage,
        }


def _bounded(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters omitted]"


def build_verifier_prompt(*, request: str, claim: str, observation: str | None = None) -> str:
    """The verifier's task message: the request, the claim, the last observation.

    NOTHING ELSE, by contract: no evaluator config, no adapter fields, no
    expected answer. The check is only meaningful if it sees what the worker
    saw; handing it grading data would turn it into an oracle (and leak the
    benchmark's answer into the episode).
    """

    parts = [
        "REQUEST (verbatim, as the worker received it):",
        _bounded(request, MAX_REQUEST_CHARS) or "(empty)",
        "",
        "WORKER'S COMPLETION CLAIM:",
        _bounded(claim, MAX_CLAIM_CHARS) or "(no text)",
    ]
    if observation and observation.strip():
        parts += [
            "",
            "LAST OBSERVATION THE WORKER SAW (text part only):",
            _bounded(observation, MAX_OBSERVATION_CHARS),
        ]
    return "\n".join(parts) + "\n"


def parse_verdict(text: str | None) -> tuple[VerdictStatus, str]:
    """``(status, findings)`` from the child's final text.

    The LAST verdict line wins (a model may quote the format before using it);
    no line at all is ``INCONCLUSIVE`` -- an unparsable answer must never read
    as a pass.
    """

    body = (text or "").strip()
    lines = body.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        match = _VERDICT_LINE.match(lines[index])
        if match is not None:
            status = match.group(1).upper()
            findings = "\n".join(lines[:index]).strip()
            return status, _bounded(findings, MAX_FINDINGS_CHARS)  # type: ignore[return-value]
    return "INCONCLUSIVE", _bounded(body, MAX_FINDINGS_CHARS)


def counted_host_tools(
    tools: Sequence["AgentTool"],
    *,
    limit: int,
    on_call: Callable[[int], None],
    on_exhausted: Callable[[], None],
) -> tuple[tuple["AgentTool", ...], Callable[[], int]]:
    """Wrap host tools so every call counts against ``limit``.

    Call ``limit + 1`` is refused (it never reaches the host) and fires
    ``on_exhausted`` so the launcher can end the child; the refusal text tells
    the model to answer with what it has, in case the cancel loses the race.
    Returns the wrapped tools and a reader for the count.
    """

    from local_operator.harness.types import TextContent, ToolResult

    count = 0

    def _wrap(tool: "AgentTool") -> "AgentTool":
        inner = tool.execute

        async def execute(tool_call_id, args, signal=None, on_update=None, context=None):  # type: ignore[no-untyped-def]
            nonlocal count
            if count >= limit:
                on_exhausted()
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    content=[
                        TextContent(
                            text=(
                                f"Command budget spent ({limit}). Give your verdict now "
                                "from what you have already seen."
                            )
                        )
                    ],
                    is_error=True,
                )
            count += 1
            on_call(count)
            return await inner(tool_call_id, args, signal, on_update, context)

        return tool.model_copy(update={"execute": execute})

    return tuple(_wrap(tool) for tool in tools), (lambda: count)

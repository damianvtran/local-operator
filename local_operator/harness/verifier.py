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
* ``bash`` is the write-confined, no-network shell a scout gets, switched on
  by the launch (``ChildSurface.readonly_shell``), NOT by the ``verifier``
  name: an operator registry role called ``verifier`` keeps its own shell.
  In a world-confined session (the OSWorld arm sets ``world_root``) the
  inherited full ``ToolConfinement`` takes precedence over the write
  confinement (``tools/builtin.py``), so that shell can write inside the
  world root -- the controller-side world, never the guest; the same rule
  applies to scouts there and predates this module (review M1).
* Bounded: :class:`VerifierLimits`. The host-command budget is enforced by
  REFUSAL (the refusal asks for the verdict); the request and wall caps
  cancel the child.
* The verdict is PARSED from the last non-empty line of the child's final
  assistant message; anything else is ``INCONCLUSIVE``, never a pass. A
  verdict that message carries wins over any cap, even when the job was
  cancelled before it settled (PR #3837 round 3, the live t31v episode).
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

#: Read-only reach. ``bash`` is confined by the launch (``ChildSurface.readonly_shell``).
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

#: Requests the request cap reserves beyond the tool budgets: the one spent
#: on a REFUSED call (the model asked for one more command and was told to
#: stop) and the one that carries the verdict. Measured, not guessed: the
#: first live verifier (OSWorld ``t31v-t001-20261111-214934``, n=1) spent 1
#: request on two local ``bash`` reads, 10 on commands, 1 on a refused 11th
#: command and gave ``VERDICT: PASS`` on request 13 -- one past the old
#: independent 12-request cap, so the answer was discarded.
VERDICT_REQUEST_RESERVE = 2
#: Local read-only tool calls (``read``/``glob``/``grep``/the confined
#: ``bash``) the request cap makes room for. Not enforced per call -- those
#: tools are the child's own, not the launcher's -- only budgeted: a child
#: that spends more on them reaches the request cap sooner. Counted on the
#: verdict (``local_tools``) so later runs can calibrate it; the live run used 2.
DEFAULT_LOCAL_TOOL_ALLOWANCE = 4

#: What a refused host call says. It is the child's cue to answer, which is
#: why the command budget no longer cancels the child: cancelling it on the
#: same request that tells it to answer is what discarded the live verdict.
COMMAND_BUDGET_REFUSAL = (
    "Command budget used ({limit} of {limit}): give your VERDICT now from what you "
    "have already seen, ending with the VERDICT line."
)
REQUEST_BUDGET_REFUSAL = (
    "Request budget used: give your VERDICT now from what you have already seen, "
    "ending with the VERDICT line."
)

#: The verdict line, tolerant of the decorations models add to a last line:
#: markdown emphasis/heading/quote/bullet/backtick prefixes and an optional
#: "final" (``> **Final VERDICT:** `PASS` ``). Anything else stays
#: ``INCONCLUSIVE``; tolerance here only ever saves a PASS from becoming a
#: wasted challenge turn (review M3), it never invents one.
#: Anchored at the END too: after the status only closing decoration or a
#: full stop may follow. The role asks for exactly ``VERDICT: PASS|FAIL``;
#: a line with more after the status is a per-requirement note
#: (``- Verdict: PASS (B2 = 42)``) or a qualified one (``PASS except C3``),
#: and taking either as the verdict is how a partial answer became a pass
#: (review round 3, M1).
_VERDICT_LINE = re.compile(
    r"^\s*(?:[#>*_`~\-]+\s*)*(?:final\s+)?VERDICT\s*[*_`]*\s*:\s*[*_`\s]*"
    r"(PASS|FAIL|INCONCLUSIVE)[*_`.!\s]*$",
    re.I,
)

#: How long a cancelled verifier job is awaited to settle. Part of the
#: verifier's worst-case envelope (``wall_s + CANCEL_JOIN_S``), which an
#: embedder with a deadline must budget for.
CANCEL_JOIN_S = 10.0


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
    """The verifier's budgets, recorded on every ``verifier_started`` row.

    ``max_commands`` (host commands, 10) and ``wall_s`` (180 s) are the
    design's (verifier-design.md "Flag"). ``max_requests`` is DERIVED from
    them by default -- ``max_commands + local_tool_allowance +``
    :data:`VERDICT_REQUEST_RESERVE` -- because two independent constants let
    a child spend its whole request budget on its command budget and have no
    request left to answer (the live t31v episode: 12 requests could not hold
    10 commands plus a verdict). An explicit ``max_requests`` is allowed (a
    tighter test bound) but must still hold every command, the refused call
    and the verdict.

    What each does when reached: a spent command budget REFUSES further host
    calls and asks for the verdict (never cancels); the request cap cancels
    the child once a request PAST it asks for more tools (a request past it
    that answers is the verdict, and is kept); the wall cancels. ``max_requests``
    counts request-final rows carrying the verifier's job id. At most one
    request past the cap is made when requests end with an assistant message;
    at most two when they end without one (the backstop in
    ``Session.verify_completion`` cannot tell the first from a message still
    in flight).
    """

    max_requests: int | None = None
    max_commands: int = 10
    local_tool_allowance: int = DEFAULT_LOCAL_TOOL_ALLOWANCE
    wall_s: float = 180.0

    def __post_init__(self) -> None:
        for name in ("max_commands", "local_tool_allowance"):
            value = getattr(self, name)
            minimum = 1 if name == "max_commands" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        floor = self.max_commands + VERDICT_REQUEST_RESERVE
        requests = self.max_requests
        if requests is None:
            # Frozen: the derived value is written once, here, so every reader
            # (and ``as_record``) sees the number actually enforced.
            object.__setattr__(self, "max_requests", floor + self.local_tool_allowance)
        elif isinstance(requests, bool) or not isinstance(requests, int) or requests < floor:
            raise ValueError(
                f"max_requests must be an integer of at least max_commands + "
                f"{VERDICT_REQUEST_RESERVE} ({floor}): every command, the refused call "
                "and the verdict must fit"
            )
        wall = self.wall_s
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall):
            raise ValueError("wall_s must be a finite number")
        if wall <= 0:
            raise ValueError("wall_s must be positive")

    @property
    def request_budget(self) -> int:
        """``max_requests`` as the int ``__post_init__`` guarantees it is."""

        assert self.max_requests is not None
        return self.max_requests

    def as_record(self) -> dict[str, Any]:
        return {
            "max_requests": self.request_budget,
            "max_commands": self.max_commands,
            "local_tool_allowance": self.local_tool_allowance,
            "wall_s": float(self.wall_s),
        }


@dataclass(frozen=True)
class VerifierVerdict:
    """What one verification produced. ``status`` is never inferred from prose.

    ``ended_by``: ``"verdict"`` (the child's final message carried a verdict
    line -- even if a cap fired or the job was cancelled after it),
    ``"requests"`` or ``"wall"`` (a cap cancelled it before any verdict),
    ``"failed"`` (the child errored) or ``"refused"`` (it never launched:
    admission, shutdown). ``caps_hit`` lists every budget reached, in order
    (``commands``, ``requests``, ``wall``), so a verdict that won over a cap
    still says the cap fired. ``local_tools`` counts the child's own
    read-only tool calls (the :data:`DEFAULT_LOCAL_TOOL_ALLOWANCE` input).
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
    caps_hit: tuple[str, ...] = ()
    local_tools: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": self.findings,
            "ended_by": self.ended_by,
            "caps_hit": list(self.caps_hit),
            "job_id": self.job_id,
            "requests": self.requests,
            "commands": self.commands,
            "local_tools": self.local_tools,
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


def find_verdict(text: str | None) -> tuple[VerdictStatus, str] | None:
    """``(status, findings)`` if the LAST NON-EMPTY line is a verdict, else ``None``.

    Only that line (review round 2, m1): the decorations :data:`_VERDICT_LINE`
    tolerates include ``>`` and ``-``, so scanning upward for any match would
    turn a QUOTED ``> VERDICT: PASS`` above trailing prose into a pass. The
    role text asks for the verdict as the final line; a reply that ends
    otherwise did not give one.
    """

    body = (text or "").strip()
    if not body:
        return None
    lines = body.splitlines()
    match = _VERDICT_LINE.match(lines[-1])
    if match is None:
        return None
    status = match.group(1).upper()
    findings = "\n".join(lines[:-1]).strip()
    return status, _bounded(findings, MAX_FINDINGS_CHARS)  # type: ignore[return-value]


def parse_verdict(text: str | None) -> tuple[VerdictStatus, str]:
    """``(status, findings)`` from the child's final text; no verdict line is
    ``INCONCLUSIVE`` -- an unparsable answer must never read as a pass."""

    found = find_verdict(text)
    if found is not None:
        return found
    return "INCONCLUSIVE", _bounded((text or "").strip(), MAX_FINDINGS_CHARS)


def counted_host_tools(
    tools: Sequence["AgentTool"],
    *,
    limit: int,
    on_call: Callable[[int], None],
    on_exhausted: Callable[[], None],
    refuse_when: Callable[[], str | None] | None = None,
) -> tuple[tuple["AgentTool", ...], Callable[[], int]]:
    """Wrap host tools so every call counts against ``limit``.

    Call ``limit + 1`` is refused (it never reaches the host) and fires
    ``on_exhausted`` so the launcher can record the cap. The refusal is the
    enforcement AND the cue: its text asks for the verdict now, so the
    launcher lets the child's next request answer instead of cancelling it.
    ``refuse_when`` adds a launcher-side refusal (the request budget) with its
    own text, checked after the command budget (the more specific fact when
    both are spent); a refused call does not count. Returns the wrapped
    tools and a reader for the count.
    """

    from local_operator.harness.types import TextContent, ToolResult

    count = 0

    def _wrap(tool: "AgentTool") -> "AgentTool":
        inner = tool.execute

        async def execute(  # type: ignore[no-untyped-def]
            tool_call_id, args, signal=None, on_update=None, context: Any = None
        ):
            nonlocal count
            refusal: str | None = None
            if count >= limit:
                on_exhausted()
                refusal = COMMAND_BUDGET_REFUSAL.format(limit=limit)
            elif refuse_when is not None:
                refusal = refuse_when()
            if refusal is not None:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    content=[TextContent(text=refusal)],
                    is_error=True,
                )
            count += 1
            on_call(count)
            return await inner(tool_call_id, args, signal, on_update, context)

        return tool.model_copy(update={"execute": execute})

    return tuple(_wrap(tool) for tool in tools), (lambda: count)

"""On-demand tool documentation for ``tool://`` reads.

Every core tool ships its full JSON Schema on **every** API request, in the
same prompt-cache prefix as the system prompt, whether or not the tool is ever
called (``AGENTS.md``'s tool-surface footprint ladder). That is the right
default for DISPATCH — the model needs schemas to call tools at all — but it
is a poor home for REFERENCE detail: per-op acceptance rules, field semantics
and failure modes are read once and then re-sent on every message.

This module makes that detail readable on demand instead. ``read
tool://<name>`` renders one tool's purpose, its authored per-op notes, and a
recursive walk of its parameters — names, types, enum literals and defaults
verbatim; ``read tool://`` lists the tools this session actually holds. The
bytes cost nothing until a reader asks for them, and the resolver is a pure
read: it never routes through the tool's ``execute`` (approval tiers, side
effects) and never changes dispatch.

Contract, mirroring ``skills/api.py`` so the chain behaves identically:

* :func:`make_tool_doc_resolver` returns ``None`` for every non-``tool://``
  URL and never raises; an unknown name is served AS CONTENT naming the
  available set, which is the model's one-round self-correction path.
* :func:`chain_tool_docs` returns a :class:`ToolDocsLink` — an ID-BEARING
  wrapper, not a bare closure — because ``Session.__init__`` installs it in
  place of the host's resolver, and the host-field parity guard must still be
  able to recognise the wrapper and recover the host value from it
  (:func:`is_tool_docs_link` / :func:`unwrap_tool_docs`).
* Property names, types and enum literals are NEVER truncated — they are the
  point of the mechanism. Prose is elided only at authoring time.
* Rendering is deterministic: a pure function of (name, label, description,
  parameters, notes) — no timestamps or ids, sorted listings, insertion-order
  properties.
* The soft 8 KiB/doc cap is enforced by TEST, never at render time; the
  runtime hard bound stays ``read``'s 16 KiB internal-document shaping.

Authored attachments live in module-level mappings rather than a new
``AgentTool`` field — a field would ride the provider tools array this
mechanism exists to shrink. :data:`TOOL_NOTES` carries per-tool notes and ops;
:data:`SPECIAL_RENDERERS` (see :func:`register_tool_doc_renderer`) lets a
surface that must be byte-identical to another view of the same information —
the sessions pilot's ``op='help'`` — register its own pure renderer here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard
from urllib.parse import unquote, urlsplit

from local_operator.harness.types import AgentTool

#: The URL scheme this module serves. The resolver is chained AHEAD of the
#: guide/skill/mcp walker in ``Session.__init__`` (and, structurally, in every
#: subagent — each constructs its own Session and wraps it over its own live
#: inventory), so a scheme that is not ``tool://`` costs the chain one
#: ``startswith`` and reaches the base walker exactly as before.
TOOL_DOC_SCHEME = "tool://"

#: Nested property levels rendered BELOW the root properties. 2 covers every
#: schema on the default surface today (deepest: ``ask`` — questions[] ->
#: options[] fields — plus one nested model on edit/todo/task/project). The
#: drift test fails if the live surface needs a third level, so a future nested
#: model forces an explicit cap decision instead of silently dropping its
#: field names from every reference.
_PARAM_DEPTH_CAP = 2

#: Section headings, as constants so the drift tests can assert against the
#: same strings the renderer writes rather than re-spelling them.
_PURPOSE_HEADING = "## Purpose"
_OPS_HEADING = "## Ops"
_PARAMS_HEADING = "## Parameters"
_NOTES_HEADING = "## Notes"


@dataclass(frozen=True)
class ToolDocOp:
    """One op in an authored per-op table.

    The sessions pilot's ``op='help'`` renders exactly this table, which is the
    only place the per-op accepted-field contract can live for op-based tools
    whose fields all sit in one flat schema union; ``blurb`` and ``fields`` are
    both optional so a pilot that only names its ops is served too.
    """

    op: str
    blurb: str = ""
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDocNotes:
    """Authored attachments for one tool's reference.

    ``ops`` renders as the per-op table (op, blurb, accepted fields); ``notes``
    renders as a trailing prose section. Both optional: a tool with no entry in
    :data:`TOOL_NOTES` gets the generic render.
    """

    ops: tuple[ToolDocOp, ...] = ()
    notes: str = ""


#: Authored notes, keyed by tool name. Deliberately a MODULE-LEVEL mapping and
#: not a new ``AgentTool`` field: the field would serialize into every
#: provider request's tools array — the cost this mechanism exists to reduce —
#: while this mapping costs nothing until a ``tool://`` read looks it up. A
#: drift test flags a key that is not a real tool name. Promote to a field
#: only if it ever grows beyond reference prose.
#:
#: The wave-1 slimming entries (audit items 2+3, 2026-10-01) are the paid-for
#: side of a binding trade: prose was REMOVED from the always-loaded tool
#: surface (parameter descriptions, per-op field descriptions, class
#: docstrings that pydantic copies into the schema), and it must be reachable
#: HERE, or the cut loses teaching the model needed at the moment of a call.
#: Each moved block keeps the original constraint wording wherever the shape
#: of the sentence allowed it — ``test_cut_wire_prose_moved_into_the_tool_doc``
#: pins one representative phrase per tool BOTH ways (present in the render,
#: absent from the wire), so a later "cleanup" that deletes a note fails there
#: with the phrase named, instead of silently re-opening the gap.
TOOL_NOTES: dict[str, ToolDocNotes] = {
    # -- agent --------------------------------------------------------------
    # Wire cut: the op field's per-op exegesis (~360 chars, every request);
    # kind/action_class tails. The tool description and the enum literals stay.
    "agent": ToolDocNotes(
        ops=(
            ToolDocOp(op="list", blurb="What exists: every role and specialist."),
            ToolDocOp(
                op="show",
                blurb=(
                    "What a role says; also prints the packaged text when an "
                    "installed role has diverged from it."
                ),
            ),
            ToolDocOp(op="search", blurb="Find a role by meaning."),
            ToolDocOp(op="install", blurb="Add a packaged starter."),
            ToolDocOp(
                op="reset",
                blurb=(
                    "Restore a role over an edited one to its packaged version, "
                    "reporting what it replaced."
                ),
            ),
            ToolDocOp(
                op="sync",
                blurb="Pull the latest for installed roles (merges hub updates with local edits).",
            ),
            ToolDocOp(
                op="create/update",
                blurb="Author or fix a role, or a specialist profile.",
            ),
        ),
        notes=(
            "Detail moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- ``action_class``: 'proactive' lets a profile attach hidden "
            "patience waits and run proactive deliveries — set it ONLY when the "
            "user clearly asked for a proactive use case (companion agents are "
            "the canonical one); it can message them unprompted, so it is "
            "sparing by default.\n"
            "- ``kind``: a 'role' is a reusable delegation target tagged for "
            "``task(agent=...)``; a 'specialist' is a durable named agent with "
            "its own instruction set, and can sit on a team roster without "
            "being a role. Ignored on update: a profile cannot change kind.\n"
            "- A specialist is the reusable base a team layers collaboration "
            "and project briefs on top of; a role is launched with "
            "``task(agent='<name>')``.\n"
            "- When a role's guidance proves wrong, fix it with the `agent` "
            "tool rather than patching one prompt."
        ),
    ),
    # -- ask ----------------------------------------------------------------
    # Wire cut: the restraint constant's mechanics half (lettered options,
    # recommendation hoisting, the free-text channel, one-call batching,
    # credential storage detail) — the RESTRAINT pins themselves stay on the
    # wire because the ask-tool tests assert them there.
    "ask": ToolDocNotes(
        notes=(
            "Mechanics moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- A question buried in a report is not seen and nothing waits on "
            "it, so writing one and continuing means you decided anyway. If you "
            "are not going to stop for an answer, do not phrase it as a "
            "question — state the decision and what would change it.\n"
            "- When you do ask, do it here INSTEAD of writing lettered options "
            "into your reply and waiting. Give each question at least two "
            "options, put the consequence of each in its description, and mark "
            "the one you recommend (it is moved to the top of the list and "
            "preselected).\n"
            "- Every question also offers the user a free-text answer, so the "
            "options do not have to be exhaustive.\n"
            "- Ask everything you need in ONE call; large option lists and "
            "calibration all live in the field descriptions below.\n"
            "- Timeouts: a late answer still reaches you and says it was late. "
            "An urgent ask's timeout notice tells you to resolve the question "
            "without the operator (a `task` subagent); do that rather than "
            "waiting. A second ask with identical question text, or a second "
            "open secret question for a key already asked for, is refused.\n"
            "- Credentials: set ``secret=true`` on that question (options "
            "empty, id is the env-var name). The value is stored in session "
            "memory and injected into ``bash``; you will only ever see the key "
            "name. Add ``persist=true`` when the credential will be needed "
            "again after this session, and it is also saved to the operator's "
            "encrypted long-term store."
        ),
    ),
    # -- browser ------------------------------------------------------------
    # Wire cut: the action field's inline action dictionary (~490 chars) and
    # the lifecycle/file-transfer/tab clauses from the description. The
    # description keeps the persistence, tab-ownership and never-install pins
    # the browser tests assert.
    "browser": ToolDocNotes(
        ops=(
            ToolDocOp(op="open", blurb="Start a surface at a URL."),
            ToolDocOp(op="goto", blurb="Navigate the session's existing surface."),
            ToolDocOp(op="read", blurb="The page's text."),
            ToolDocOp(op="snapshot", blurb="Accessibility tree with click refs."),
            ToolDocOp(op="screenshot", blurb="Capture the page to a file."),
            ToolDocOp(op="click", blurb="Click a selector or snapshot ref."),
            ToolDocOp(op="type", blurb="Type text into a field."),
            ToolDocOp(op="scroll", blurb="Move the viewport."),
            ToolDocOp(op="logs", blurb="Console + errors."),
            ToolDocOp(
                op="styles",
                blurb=(
                    "Rect + computed styles for a selector's matches (max 5, "
                    "rounded to 2dp); 'properties' names extras (max 20)."
                ),
            ),
            ToolDocOp(
                op="hit_test",
                blurb="The element stack at viewport (x, y), topmost first (max 8).",
            ),
            ToolDocOp(
                op="ancestors",
                blurb="The chain from the element up to <html> (depth <= 16, default 12).",
            ),
            ToolDocOp(op="tabs", blurb="List agent-driven tabs."),
            ToolDocOp(
                op="request_access",
                blurb="Raise the site-approval prompt for a not-yet-allowed origin.",
            ),
            ToolDocOp(op="await_access", blurb="Wait for the user's decision."),
            ToolDocOp(op="cancel_access", blurb="Cancel YOUR pending exact-origin request."),
            ToolDocOp(op="recover", blurb="Recover YOUR tab."),
            ToolDocOp(op="retain", blurb="Hold it."),
            ToolDocOp(op="release", blurb="End that hold."),
            ToolDocOp(op="close", blurb="End YOUR tab."),
        ),
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- 'scroll', 'logs', 'tabs' and the geometry reads "
            "('styles'/'hit_test'/'ancestors') need a non-cmux host (cmux says "
            "so).\n"
            "- On a non-cmux host, 'download' saves what the page offers into "
            "this session's private download directory, and 'upload' attaches "
            "local files to a page's file input.\n"
            "- The geometry reads answer with numbers, not pictures, and are "
            "bounded at the source: 'styles' reports up to 5 matches of a "
            "selector (rect + computed styles + the element's inline `--*` "
            "tokens); 'hit_test' the element stack at viewport (x, y) pixels, "
            "topmost first; 'ancestors' the chain from an element up to and "
            "including `<html>`, with the clip/layout style set. A selector "
            "that matches nothing is the typed element-not-found, never an "
            "empty success.\n"
            "- Tab ownership: 'tabs' lists every agent-driven tab including "
            "other sessions' (handles are redacted), and a redacted handle is "
            "not yours to drive.\n"
            "- The full playbook — hosts, setup, per-site approvals, what each "
            "error means — is ``guide://browser``."
        ),
    ),
    # -- console ------------------------------------------------------------
    # Wire cut: the method field's ten-method dictionary (~290 chars) and the
    # reveal/surface/keys/on tails. The description keeps the five jobs its
    # test asserts (bash contrast, con: handle, user-opened surfaces, guide
    # pointer, no 'approval' claim).
    "console": ToolDocNotes(
        ops=(
            ToolDocOp(
                op="list",
                blurb=(
                    "Surface handles; includes surfaces the USER opened — read "
                    "those rather than asking the user to repeat their output."
                ),
            ),
            ToolDocOp(op="create", blurb="Start a pty (command, args, cwd, env, cols, rows)."),
            ToolDocOp(op="status", blurb="Where a surface and its process stand."),
            ToolDocOp(op="read", blurb="Text: viewport (default) or scrollback."),
            ToolDocOp(op="screenshot", blurb="Write a PNG of the grid."),
            ToolDocOp(op="input", blurb="Type text, or a stored secret via 'secret_ref'."),
            ToolDocOp(op="keys", blurb="Send named keys; spellings and synonyms: guide://console."),
            ToolDocOp(op="resize", blurb="Set cols/rows."),
            ToolDocOp(op="secure", blurb="Turn the surface's do-not-capture span on/off."),
            ToolDocOp(op="close", blurb="Ask the process to exit, or kill it."),
        ),
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- ``reveal`` values: none (default, pane untouched) | session "
            "(open the pane only if the app is showing THIS session) | open "
            "(claim and focus the pane, only when the app's window is already "
            "focused). No value raises the OS window; a downgrade comes back "
            "as revealed=false.\n"
            "- A console handle starts with 'con:': a terminal in another "
            "window has none and is not readable by this tool.\n"
            "- `bash` cannot wedge on a prompt or leave a process running "
            "behind the turn (that is what the console is for), and the "
            "surface outlives the call — which is also why it is NOT for "
            "ordinary commands.\n"
            "- The user can toggle the same secure switch from the pane."
        ),
    ),
    # -- hub ----------------------------------------------------------------
    # Wire cut: the op field's per-op exegesis (~1,100 chars — the single
    # biggest field on the surface at audit time) and the to-field's fan-out
    # sentence. The tool description keeps the verb vocabulary and the
    # quiet-child punchline; the ops table below carries each verb's contract.
    "hub": ToolDocNotes(
        ops=(
            ToolDocOp(
                op="list",
                blurb=(
                    "Every subagent you launched with its status and whether it "
                    "can be resumed — including finished, failed and paused ones "
                    "the 'jobs' tool no longer shows."
                ),
            ),
            ToolDocOp(
                op="peek",
                blurb=(
                    "READ the subagent's transcript (ranged, cheap) to see its "
                    "current progress without spending its attention — the fast "
                    "way to check on a running child."
                ),
            ),
            ToolDocOp(op="send", blurb="A note, no reply waited for."),
            ToolDocOp(op="ask", blurb="A question; blocks for the subagent's answer."),
            ToolDocOp(
                op="steer",
                blurb="Change what it is doing (becomes part of its instructions).",
            ),
            ToolDocOp(op="pause", blurb="Stop it now but keep it resumable."),
            ToolDocOp(op="cancel", blurb="Stop it for good."),
            ToolDocOp(
                op="resume",
                blurb=(
                    "Relaunch a stopped, paused or failed subagent against its "
                    "own transcript so it continues where it left off; names "
                    "several targets to fan one message out to a whole batch at once."
                ),
            ),
        ),
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- Several ids address several subagents; 'ask' and 'peek' take "
            "exactly one; 'resume' fans one message out to every target you "
            "name, so a batch of failed subagents can be resumed in a single call.\n"
            "- ``peek`` is usually the last few steps when neither `range` nor "
            "`steps` is given."
        ),
    ),
    # -- network ------------------------------------------------------------
    # Wire cut: three field-tail clauses (the expires example, peer's
    # requirement list, the ready-clause phrasing). Small by design: this
    # schema was already tight, and its playbook is ``guide://network``.
    "network": ToolDocNotes(
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- ``peer`` is required by create/engage/stop/delete.\n"
            "- ``expires`` takes a duration such as 30m or 2h (default 10m).\n"
            "- The full playbook is ``guide://network``."
        ),
    ),
    # -- eval / todo (system-prompt cuts, audit item 3) ---------------------
    "eval": ToolDocNotes(
        notes=(
            "Reading ten files, filtering them, and summarizing is one `eval` "
            "call that prints the summary — not ten `read` calls."
        ),
    ),
    # Wire cut (context diet, deferred-tools PR): the ``op`` field's per-op
    # prose. The enum literals stay on the wire and are self-describing.
    "todo": ToolDocNotes(
        ops=(
            ToolDocOp(
                op="init",
                blurb="Replace the whole list, optionally grouped into named phases "
                "(pass `phases`).",
            ),
            ToolDocOp(
                op="add",
                blurb="Append newly discovered work without rewriting the list, "
                "optionally into a named `phase`.",
            ),
            ToolDocOp(op="done", blurb="Mark items finished."),
            ToolDocOp(
                op="block",
                blurb="Mark items that cannot proceed until a user decides or an "
                "external service answers (requires 'reason').",
            ),
            ToolDocOp(op="drop", blurb="Abandon items that are no longer needed."),
            ToolDocOp(op="view", blurb="Show the list."),
        ),
        notes=(
            "`block` a pending item with a reason naming the decision or "
            "service it is waiting on; `add` a mid-turn requirement instead of "
            "rewriting the list."
        ),
    ),
    # -- send ---------------------------------------------------------------
    # Wire cut (context diet, deferred-tools PR): the description's restatement
    # of the addressing rules (kept once, on the ``target`` field) and the
    # mesh/fallback detail.
    "send": ToolDocNotes(
        notes=(
            "Moved off the wire (context diet):\n"
            "\n"
            "- Delivery: by default the message lands in the peer's mailbox AND "
            "wakes the peer if it is idle, so an idle peer responds right away; "
            "`wake=False` is the quiet mailbox drop (read on the peer's next "
            "turn), and `now=True` steers mid-turn (opens a turn if the peer is "
            "idle). The result says how the peer received it.\n"
            "- Finding a peer: the `sessions` tool lists what is running "
            "(`lop sessions` is the fallback; `--all` adds stored ones), and "
            "`target` matches them by name.\n"
            "- A session with no message sent in it yet (a fresh `/new`) is not a "
            "recipient: sends to it are refused.\n"
            "- With `peer`, the target addresses a session on that device (mesh): "
            "the send drives a turn there and returns the owner's reply; "
            "`wake`/`now`/`patience`/`model` are local-only and refused there."
        ),
    ),
    # -- project ------------------------------------------------------------
    # Wire cut: the ProjectMilestone model docstring's 'derived, not stored'
    # rationale (pydantic copies it into $defs on every request) and several
    # field tails (status enum recital, progress edge cases, description's
    # markdown shape). The enum literals still render from the schema itself.
    "project": ToolDocNotes(
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- Milestone status (completed / overdue / upcoming) is deliberately "
            "NOT stored: it is derived at render from ``completed_at`` and "
            '``target_date`` — the same "derived, not stored" rule the view '
            "composer applies to session liveness — so a stored status can "
            "never drift from the dates that contradict it.\n"
            "- ``description`` may be markdown: multiple paragraphs, headings, "
            "lists and code.\n"
            "- ``progress``: an identical re-send records a refresh that keeps "
            "the freshness clock; a near-identical line is NEW — it appends. "
            "Write a line only for real movement.\n"
            "- Deleting a project is its own tool, ``project_delete`` (write "
            "tier); delete artifacts are never touched.\n"
            "\n"
            "Field rules moved off the wire (context diet):\n"
            "\n"
            "- ``''`` clears an optional text/date field (owner, team, title, "
            "start_date, target_date, completed_at, milestone_target_date).\n"
            "- Dates are ISO YYYY-MM-DD; ``target_date`` is never before "
            "``start_date``.\n"
            "- ``status='done'`` needs every milestone complete (or "
            "``force_done=true``) and stamps ``completed_at`` unless given.\n"
            "- ``progress``: one dated line (markdown); a NEW line appends and "
            "moves the freshness clock.\n"
            "- ``estimate``: > 0 and <= 1000, fractional allowed; "
            "``estimate_unit`` is 'points' (default) or 'days'.\n"
            "- ``milestones``: create stores the list (<= 20, names unique); "
            "update replaces the WHOLE list and is refused unless "
            "``replace_milestones=true``. Use op='milestone' for one, "
            "add-or-update by name; ``milestone_completed`` true sets "
            "``completed_at`` to today, false clears it.\n"
            "- ``attach``: local file paths (screenshots/evidence) stored on "
            "the history entry the NEW progress line appends; <= 10 files, "
            "<= 5 MB each."
        ),
    ),
    # -- task ---------------------------------------------------------------
    # Wire cut: the TaskItem model docstring (prompt text on every request —
    # see the comment at its definition in builtin.py) and the tasks-field
    # tail. The effort field's config-sensitive descriptions stay on the wire
    # because their tests assert the rendered bytes there.
    "task": ToolDocNotes(
        notes=(
            "Moved off the wire in the slimming wave (audit item 2):\n"
            "\n"
            "- ``agent`` names the ROLE the child runs as — a registered "
            "profile or a packaged starter (reviewer, coder, architect, "
            "manager, designer, scout); the role supplies standing guidance "
            "and may restrict the child's tools.\n"
            "- ``subagents.model_choice=model`` is the operator's switch that "
            "hands the child's model back; ``effort`` swaps the child's MODEL "
            "(not its reasoning level), and omitting it inherits this session's "
            "model and reasoning effort."
        ),
    ),
    # -- item-7 notes (failure semantics + the system-prompt cuts) ----------
    # These are the tools the self-presentation audit flagged as carrying no
    # failure semantics anywhere. The clauses live HERE rather than on the
    # wire because the wave's budget goal is a net REDUCTION; each states a
    # refusal or bound that is already enforced in code and pinned by that
    # tool's own tests, so the doc teaches it before the call instead of the
    # caller learning it from an error.
    "bash": ToolDocNotes(
        notes=(
            "For multi-step Python work, one `eval` call with a compact digest "
            "beats a chain of shell round-trips; see `tool://eval`."
        ),
    ),
    # Wire cut: the cost/failover/cancel clauses live here rather than in the
    # always-loaded description (the slimming-wave trade the module docstring
    # describes); the description keeps only the capability sentence and the
    # two pointers.
    "generate_image": ToolDocNotes(
        notes=(
            "Providers, in priority order: Radient (signed-in account), then "
            "FAL (stored key), then OpenAI (stored API key). Failover is "
            "automatic — if one refuses, the next runs, and the result's "
            "`details.attempts` lists what each provider said.\n"
            "\n"
            "Cost: Radient reports `cost_usd` per generation (prices come "
            "from its live model list) and bills your account credits; FAL "
            "and OpenAI bill your own key, OpenAI per image — `num_images` "
            "multiplies every provider's cost. The harness prompts for "
            "approval before spending (write tier); `image_size` and "
            "`num_images` are the spend knobs.\n"
            "\n"
            "`source_image_path` makes it an edit (image-to-image): the file "
            "is read locally and uploaded to the provider as a data URI; "
            "`strength` (0..1) only applies with it. `model` picks a "
            "provider model id — omit it and the provider's default runs; "
            "Radient model ids are read from its live list at call time.\n"
            "\n"
            "Cancelling: stopping the turn (Esc / steering) stops the wait "
            "and best-effort-cancels the provider-side job; a cancelled "
            "generation is not resumed — call again with the same or an "
            "edited prompt for a fresh one. Nothing is resumed after a "
            "restart of any kind.\n"
            "\n"
            "Failure modes: no provider configured (set one up via "
            "`guide://image-generation`); insufficient credits (402 — top "
            "up and retry); all providers failed — the error lists every "
            "attempt so you can decide what to change.\n"
            "\n"
            "The generated images are attached to the session — that IS the "
            "delivery; do not re-save them to disk unless the user asks for "
            "a file."
        ),
    ),
    "grep": ToolDocNotes(
        notes=(
            "Detail moved off the wire in the slimming wave (audit items 2+7):\n"
            "\n"
            "- `context_lines` adds surrounding lines per match (like `grep "
            "-C`); `skip` pages past the first 200 matches.\n"
            "- `grep` and `glob` both respect the project's ignore files.\n"
            "- Failure semantics: a scan that hits its wall-clock deadline "
            "returns what it has so far, and the display caps at 200 matches "
            "with the rest written to a `spill://` handle you can search."
        ),
    ),
    "jobs": ToolDocNotes(
        notes=(
            "Failure semantics: an unknown job id is refused (`unknown job "
            "<id>`), never silently treated as still running."
        ),
    ),
    "monitor": ToolDocNotes(
        notes=(
            "Failure semantics: a non-read-only target is refused with the "
            'reason (e.g. `monitor can\'t watch "eval": arbitrary code.`); a '
            "`create` also needs the tool name for the same reason it cannot "
            "watch a writer."
        ),
    ),
    "read": ToolDocNotes(
        notes=(
            "From the system prompt's tool note (moved off the wire, slimming "
            "wave): reading a Python file whole returns its declaration "
            "outline with line ranges — re-read the exact ranges you need "
            "instead of the whole file."
        ),
    ),
    "write": ToolDocNotes(
        notes=(
            "Failure semantics: a path with an unrecognised scheme is refused "
            "rather than treated as a relative path, so `notes://x.py` cannot "
            "silently create a `notes:` directory in the working directory."
        ),
    ),
}

#: Tool-specific renderers for surfaces that must be BYTE-IDENTICAL to another
#: view of the same information (the sessions pilot's ``op='help'``). Consulted
#: at RESOLVE time, so registration order does not matter. A renderer's
#: contract: deterministic, no raises — a raising renderer degrades to the
#: generic render below, never to an error result.
SPECIAL_RENDERERS: dict[str, Callable[[AgentTool], str]] = {}


def register_tool_doc_renderer(name: str, fn: Callable[[AgentTool], str]) -> None:
    """Register ``fn`` as the renderer for ``tool://<name>`` (last call wins)."""
    SPECIAL_RENDERERS[name] = fn


def render_tool_doc(tool: AgentTool, *, notes: ToolDocNotes | None = None) -> str:
    """The full reference document for one tool.

    Pure and never raises; on an unreadable schema it degrades to
    :func:`_fallback` — name, description and a plain "unavailable" note —
    because a reader has no retry path for "renderer crashed". ``notes=None``
    looks the tool's entry up in :data:`TOOL_NOTES`; pass an explicit value to
    render candidate notes that are not registered yet (tests).
    """
    if notes is None:
        notes = TOOL_NOTES.get(tool.name)
    try:
        return _render(tool, notes)
    except Exception:  # noqa: BLE001 — the public contract is "never raises"
        return _fallback(tool)


def _render(tool: AgentTool, notes: ToolDocNotes | None) -> str:
    """Section order per the design note §2: title, purpose, ops, parameters,
    authored notes, re-read footer."""
    lines = [_title(tool)]
    description = (tool.description or "").strip()
    if description:
        lines += ["", _PURPOSE_HEADING, "", description]
    if notes is not None and notes.ops:
        lines += ["", _OPS_HEADING, ""]
        lines += [_op_line(op) for op in notes.ops]
    parameters = tool.parameters if isinstance(tool.parameters, dict) else {}
    param_lines: list[str] = []
    _render_properties(parameters, root=parameters, depth=0, out=param_lines)
    if param_lines:
        lines += ["", _PARAMS_HEADING, ""]
        lines += param_lines
    extras = (notes.notes if notes is not None else "").strip()
    if extras:
        lines += ["", _NOTES_HEADING, "", extras]
    lines += ["", f"Read again: `{TOOL_DOC_SCHEME}{tool.name}`"]
    return "\n".join(lines)


def _fallback(tool: AgentTool) -> str:
    head = f"# Tool: `{tool.name}`"
    description = (tool.description or "").strip()
    parts = [head]
    if description:
        parts += ["", description]
    parts += ["", "(tool reference unavailable)"]
    return "\n".join(parts)


def _title(tool: AgentTool) -> str:
    title = f"# Tool: `{tool.name}`"
    label = (tool.label or "").strip()
    # A label that repeats the name in different case ("Read" for ``read``)
    # says nothing the title does not; only a genuinely different label
    # ("Shell", "Agent roles", "Mesh network") earns its bytes.
    if label and label.lower() != tool.name.lower():
        title += f" — {label}"
    return title


def _op_line(op: ToolDocOp) -> str:
    line = f"- {op.op}"
    blurb = " ".join(op.blurb.split())
    if blurb:
        line += f": {blurb}"
    if op.fields:
        line += f" — fields: {', '.join(op.fields)}"
    return line


def _render_properties(
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    depth: int,
    out: list[str],
    indent: str = "",
) -> None:
    """Walk one ``properties`` container, one line per property.

    Property order is the schema's insertion order (pydantic emits declaration
    order, which the tools array already depends on for prompt-cache
    stability), so the doc reads in the same order as the schema the model
    dispatches against.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    for name, subschema in properties.items():
        if not isinstance(name, str):
            continue
        out.append(
            _property_line(
                name,
                subschema,
                root=root,
                required=name in required_names,
                indent=indent,
            )
        )
        if depth >= _PARAM_DEPTH_CAP:
            continue
        for child in _child_schemas(subschema, root):
            _render_properties(child, root=root, depth=depth + 1, out=out, indent=indent + "  ")


def _property_line(
    name: str,
    subschema: Any,
    *,
    root: dict[str, Any],
    required: bool,
    indent: str,
) -> str:
    annotations = [_type_of(subschema, root)]
    if required:
        annotations.append("required")
    literals = _enum_literals(subschema, root)
    if literals:
        annotations.append("enum: " + " | ".join(str(value) for value in literals))
    resolved = _resolved(subschema, root)
    if isinstance(resolved, dict) and "default" in resolved:
        annotations.append("default: " + _format_default(resolved["default"]))
    line = f"{indent}- `{name}` ({', '.join(annotations)})"
    description = ""
    if isinstance(resolved, dict):
        raw = resolved.get("description")
        if isinstance(raw, str):
            description = " ".join(raw.split())
    if description:
        line += f": {description}"
    return line


def _child_schemas(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    """Object schemas whose properties belong one level deeper in the doc.

    A property's fields can arrive through three shapes in these schemas:
    directly (an inline object), through ``items`` (an array of a ``$defs``
    model — ``edits``/``questions``/``milestones``), or through an ``anyOf``
    branch (a union whose object arm carries fields). All three are walked so
    a field name can never hide behind the shape of its container.
    """
    children: list[dict[str, Any]] = []
    for candidate in _object_candidates(subschema, root):
        if isinstance(candidate.get("properties"), dict) and candidate["properties"]:
            children.append(candidate)
    return children


def _object_candidates(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(subschema, dict):
        return []
    resolved = _resolved(subschema, root)
    if not isinstance(resolved, dict):
        return []
    candidates = [resolved]
    for branch in resolved.get("anyOf") or ():
        branch_schema = _resolved(branch, root)
        if isinstance(branch_schema, dict):
            candidates.append(branch_schema)
    items = resolved.get("items")
    if isinstance(items, dict):
        item_schema = _resolved(items, root)
        if isinstance(item_schema, dict):
            candidates.append(item_schema)
    return candidates


def _resolved(subschema: Any, root: dict[str, Any]) -> Any:
    """Dereference a local ``$ref`` against the root ``$defs``; non-refs pass
    through unchanged. A dangling ref resolves to itself, which the walkers
    treat as a schema with nothing to add rather than an error."""
    if not isinstance(subschema, dict):
        return subschema
    ref = subschema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        defs = root.get("$defs")
        target = defs.get(ref[len("#/$defs/") :]) if isinstance(defs, dict) else None
        if isinstance(target, dict):
            return target
    return subschema


def _type_of(subschema: Any, root: dict[str, Any]) -> str:
    if not isinstance(subschema, dict):
        return "any"
    ref = subschema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        # Name the model, not "object": "array of EditHunk" is the useful
        # answer to "what goes in this list".
        return ref[len("#/$defs/") :]
    any_of = subschema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        parts: list[str] = []
        for branch in any_of:
            part = _type_of(branch, root)
            if part not in parts:
                parts.append(part)
        return " | ".join(parts)
    declared = subschema.get("type")
    if isinstance(declared, str):
        if declared == "array":
            items = subschema.get("items")
            if isinstance(items, dict):
                return f"array of {_type_of(items, root)}"
            return "array"
        return declared
    return "any"


def _enum_literals(subschema: Any, root: dict[str, Any]) -> list[Any]:
    """Enum literals attached to a property, through unions and item schemas.

    Collected recursively because the same JSON Schema expresses "one of these
    values" three ways in this codebase: a direct ``enum``, an ``enum`` on one
    ``anyOf`` branch (nullable enums), and an ``enum`` on ``items``. All three
    reach the doc — a dropped literal is a wrong answer to the exact question
    the reader came with.
    """
    if not isinstance(subschema, dict):
        return []
    resolved = _resolved(subschema, root)
    if not isinstance(resolved, dict):
        return []
    literals: list[Any] = []
    values = resolved.get("enum")
    if isinstance(values, list):
        literals.extend(values)
    for branch in resolved.get("anyOf") or ():
        literals.extend(_enum_literals(branch, root))
    items = resolved.get("items")
    if isinstance(items, dict):
        literals.extend(_enum_literals(items, root))
    return literals


def _format_default(value: Any) -> str:
    """JSON-literal rendering, so ``false``/``null``/``""`` read as the schema
    spells them rather than Python's spellings."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _url_name(url: str) -> str:
    """The decoded name a ``tool://`` URL addresses, or ``""``.

    Netloc is where the name lives (mirrors ``skills/api.py::_url_name``); a
    malformed URL is simply not a name rather than an error. The name is never
    case-folded: tool names are lowercase, and a case-mismatch answer that
    names the real tool is a better self-correction than a silent hit.
    """
    try:
        return unquote(urlsplit(url).netloc)
    except Exception:  # noqa: BLE001 — a malformed URL is simply not a name
        return ""


def make_tool_doc_resolver(
    inventory: Callable[[], Sequence[AgentTool]],
    on_tool_read: Callable[[str], str | None] | None = None,
) -> Callable[[str], str | None]:
    """Build the ``tool://`` adapter for the ``read`` tool.

    Contract (mirrors ``skills/api.py``): returns content for ``tool://`` URLs,
    ``None`` for every other scheme (the caller chains other resolvers), and
    never raises — so a reviewer can reason about this link exactly as they do
    about the skill and guide links it is chained with.

    ``inventory`` is a CALLABLE, not a snapshot: a session's tool list is
    rebound mid-session (an MCP enable, a declared inventory narrowing, the
    prune in ``harness.subagent``), and the reader must see the list that is
    live at READ time, not the one that existed when the resolver was built.
    A hidden tool is omitted from LISTINGS but still served on a direct hit —
    hidden tools remain callable (``prompts_api``), and a reader who knows the
    name is asking precisely because it is not listed.

    ``on_tool_read(name)`` runs after a SUCCESSFUL doc hit and may return a
    line to append. The session uses it to publish a deferred tool's schema
    (``tools/deferral.py``): reading a tool's reference is the explicit
    request for it, the way ``mcp://server/tool`` is for an MCP tool. A hook
    that raises appends nothing — the doc still answers.
    """

    def resolver(url: str) -> str | None:
        if not url.startswith(TOOL_DOC_SCHEME):
            return None
        try:
            doc = _resolve_tool_url(url, inventory)
        except Exception as exc:  # noqa: BLE001 — the resolver contract is "never raises"
            return f"Tool reference unavailable: {exc}"
        name = _url_name(url)
        if on_tool_read is None or not name or not any(t.name == name for t in inventory()):
            return doc
        try:
            note = on_tool_read(name)
        except Exception:  # noqa: BLE001 — a hook failure must not cost the doc
            return doc
        return f"{doc}\n\n{note}" if note else doc

    return resolver


def _resolve_tool_url(url: str, inventory: Callable[[], Sequence[AgentTool]]) -> str:
    tools = list(inventory())
    name = _url_name(url)
    available = ", ".join(sorted(tool.name for tool in tools if not tool.hidden)) or "(none)"
    if not name:
        # Bare ``tool://`` is the discovery door, mirroring ``read skill://``:
        # the listing is what tells the reader which names exist at all.
        return f"Tool URL missing a name: expected tool://<name>\nAvailable tools: {available}"
    tool = next((candidate for candidate in tools if candidate.name == name), None)
    if tool is None:
        # Served AS CONTENT (like ``Unknown skill``): the available set is the
        # model's self-correction path, in one tool round.
        return f"Unknown tool: {name}\nAvailable: {available}"
    renderer = SPECIAL_RENDERERS.get(name)
    if renderer is not None:
        try:
            return renderer(tool)
        except Exception:  # noqa: BLE001 — a broken special renderer must not
            # cost the reader the generic reference entirely; fall through to
            # it. The special renderer's own drift test pins the byte parity.
            pass
    return render_tool_doc(tool, notes=TOOL_NOTES.get(name))


class ToolDocsLink:
    """The resolver link :func:`chain_tool_docs` installs — id-bearing on purpose.

    A plain closure would work identically at CALL time, and this class exists
    for the one consumer that must not treat it as identical: the host-field
    parity guard (``tests/unit/session/test_tool_context_parity.py``) asserts
    by IDENTITY that every value the host hands ``Session.__init__`` still
    reaches the executor, and the session legitimately replaces the host's
    resolver with this link (``Session.__init__`` chains ``tool://`` ahead of
    it). That guard may keep its drop-detection only if the wrapper is
    RECOGNISABLE and the host value is RECOVERABLE from it —
    :func:`is_tool_docs_link` / :func:`unwrap_tool_docs` are the seam, and the
    guard accepts the wrapper in place of identity ONLY when unwrapping
    recovers the host's own object. Do not collapse this back into a closure
    without carrying that permission somewhere else.
    """

    __slots__ = ("_base", "_tool_resolver")

    def __init__(
        self,
        base: Callable[[str], str | None] | None,
        tool_resolver: Callable[[str], str | None],
    ) -> None:
        self._base = base
        self._tool_resolver = tool_resolver

    def __call__(self, url: str) -> str | None:
        if url.startswith(TOOL_DOC_SCHEME):
            handled = self._tool_resolver(url)
            if handled is not None:
                return handled
        base = self._base
        return base(url) if base is not None else None


def is_tool_docs_link(value: object) -> TypeGuard[ToolDocsLink]:
    """True for a resolver link :func:`chain_tool_docs` built.

    Exported for consumers that need to tell a tool-docs wrapper apart from
    any other callable (the parity guard's acceptance path); everything else
    should just call the resolver. A ``TypeGuard``, so ``assert
    is_tool_docs_link(x)`` narrows ``x`` for the type checker as well as the
    runtime — the guard's call sites depend on that.
    """
    return isinstance(value, ToolDocsLink)


def unwrap_tool_docs(value: object) -> Callable[[str], str | None] | None:
    """The host resolver a :class:`ToolDocsLink` wraps, or ``None``.

    ``None`` for anything that is not a link (including a dropped field's
    ``None``) AND for a link wrapping no base; a caller that must tell those
    two apart checks :func:`is_tool_docs_link` first.
    """
    return value._base if isinstance(value, ToolDocsLink) else None


def chain_tool_docs(
    base: Callable[[str], str | None] | None,
    inventory: Callable[[], Sequence[AgentTool]],
    *,
    on_tool_read: Callable[[str], str | None] | None = None,
) -> Callable[[str], str | None]:
    """Put the ``tool://`` resolver AHEAD of ``base``; every other scheme
    reaches ``base`` exactly as before, and with no base configured the
    ``tool://`` link still answers (a session with no knowledge resolver has
    nowhere else to route its own tool docs to).

    ``base`` is the guide->skill->mcp walker the session factory composes;
    this wrapper is installed in ``Session.__init__`` so every session — root
    or subagent, each with its own live inventory — answers ``tool://`` without
    touching the factory chain or the subagent wiring. Returns a
    :class:`ToolDocsLink` rather than a bare closure so the identity-sensitive
    parity guard can recognise it (see the class docstring). ``on_tool_read``:
    see :func:`make_tool_doc_resolver`.
    """
    return ToolDocsLink(base, make_tool_doc_resolver(inventory, on_tool_read))

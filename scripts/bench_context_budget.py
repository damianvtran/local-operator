#!/usr/bin/env python3
"""Start-of-session context budget guard.

Enforces the performance contract in ``docs/REWRITE.md``: a fresh conversation
MUST start under a bounded context. Exits non-zero when the budget is blown,
so CI fails loudly rather than letting the start context drift upward one
harmless-looking paragraph at a time.

What it measures, and why that matters
--------------------------------------
This script used to measure a FICTION, and the fiction was optimistic in three
separate ways — which is worse than no guard, because a guard that cannot go
red is believed:

1. It built tools from a bare ``ToolContext``, so the createIf gates dropped a
   third of the default tools. It measured a surface no user has, and
   undercounted the single largest payload after the system prompt. It now
   builds the full surface via ``scripts/real_tool_surface``, which owns the
   exact counts and the reason for each gate — one place, not two.
2. It passed no ``user_instructions`` and no ``repo_guidance``. Both ride the
   HEAD block of a real session and both are commonly kilobytes. They are now
   included, at representative sizes.
3. It counted only the SEMANTIC skills block plus schemas, dropping the
   instruction and inventory blocks from the total. All four blocks plus the
   tools array are now counted, because all five are on the wire.

Units
-----
CHARACTERS are the ground truth here. The analytics ledger apportions a call's
billed prompt tokens across components in proportion to each component's
character count, and the measured rate on this prompt surface is ~2.78
chars/billed-token — NOT cl100k's ~4.22. A projection made with a generic
tokenizer understates a saving by roughly 45%, so the budget is expressed in
billed tokens derived from characters at the measured rate.

Run:
    .venv/bin/python scripts/bench_context_budget.py
    .venv/bin/python scripts/bench_context_budget.py --verbose

DO NOT PIPE IT. This script's entire value is its EXIT CODE — 0 pass, 1 budget
or surface violation, 2 provenance mismatch — and a pipeline reports the LAST
command's status, so ``bench_context_budget.py | tail`` prints a failure while
exiting 0. That is not hypothetical: a reviewer's piped run reported ``EXIT=0``
while the real code was 1, and only checking ``$?`` outside a pipeline caught
it. AGENTS.md documents the same rc-swallowing for the lint gates. Read the
status directly, or use ``${PIPESTATUS[0]}`` if you must page the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import local_operator  # noqa: E402
from local_operator.harness.types import AgentTool  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.tools.registry import DEFAULT_TOOL_NAMES  # noqa: E402
from scripts.real_tool_surface import build_real_tools  # noqa: E402

#: Measured chars-per-billed-token for this prompt surface. See the module
#: docstring — this is the ledger's own apportioning rate, not a tokenizer's.
CHARS_PER_BILLED_TOKEN = 2.78

#: The enforced ceiling, in billed tokens.
#:
#: A RATCHET set to where the tree honestly stands today, not to the 30,000
#: aspiration in docs/REWRITE.md. Two things are going on and they pull in
#: opposite directions:
#:
#: - The contract's 30,000 was written against a measurement that omitted the
#:   instruction and inventory blocks, built two-thirds of the tool surface,
#:   and supplied neither custom instructions nor repo guidance. Measured
#:   honestly, ``origin/main`` before this change was ~31,700 billed tokens —
#:   i.e. the contract was ALREADY breached and the guard reported nothing.
#: - This change brings that to ~26,165 (measured, same script both sides).
#:
#: So the ceiling is set just above the current real figure. It is green by
#: MEASUREMENT rather than green by fiction, and it is the number a future
#: reduction tightens. The repo-guidance lazy-loading work lands separately
#: and should pull this down again; each context-reduction change is expected
#: to lower this line, and one that cannot has not reduced anything.
#:
#: Do NOT raise it to make a red run green without stating in the PR what
#: regressed — that converts a guard into a rubber stamp.
#:
#: Headroom is deliberately small (~2%): enough that an incidental wording
#: edit does not trip it, tight enough that a new paragraph or a new tool has
#: to be a decision rather than a surprise.
#:
#: Measured on the full 24-tool surface (see ``real_tool_surface``); on
#: ``origin/main`` with this same script the figure was 31,100.
#:
#: RAISED 26,500 -> 27,250 for the 25th tool, ``secret`` (design §5.2), with the
#: numbers rather than a wave of the hand: ``origin/main`` measured 26,492 (8
#: tokens of headroom — the ceiling was already at its limit), and the tool as
#: first written cost 828 billed tokens. That is what the ladder means by
#: "measure it": the schema was then cut to **556** by moving the rationale out
#: of the field descriptions and into ``guide://credentials``, which is read on
#: demand and costs nothing until it is. Verbs were NOT split into separate
#: tools for the same reason — six schemas instead of one.
#:
#: The remaining 556 is the price of the capability and is paid only where it
#: is usable: ``build_secret_tool`` is a createIf factory that returns ``None``
#: when the store modules will not import, so a session that cannot reach a
#: store carries no schema for it. R7: the branch measures 27,201, which is
#: main's 26,492 + ~709 — the tool's 556 plus ~150 from the ask ``persist``
#: schema growth, the registry inventory line and the guide's own row, so the
#: arithmetic closes without re-measuring. Headroom at 27,250 is 49 tokens
#: (~0.2% — tighter than the ~2% this comment's sibling describes, in the safe
#: direction), and the next context reduction tightens it.
#:
#: RAISED 27,250 -> 27,320 for the `edit` tool's not-found diagnostics, stated
#: here because the guard exists to make this an explicit decision. Measured on
#: this branch's base (``origin/main`` 0d6cde1f0) the figure was 27,237 — 13
#: tokens of headroom, so no useful sentence fits under the old ceiling and the
#: description had to be paid for rather than squeezed in. The added sentence
#: (`edit` now says a non-matching hunk writes nothing and that the error names
#: the closest file lines, so a caller re-reads instead of re-sending the
#: batch) is 118 characters, i.e. 42 billed tokens at 2.78 chars/token — the
#: whole of the increase, with 41 tokens of headroom left. The trade is not
#: close: one avoided blind retry of an edit batch costs far more than 42
#: tokens, and the observed session paid that retry nine times.
#:
#: RAISED 27,320 -> 27,800 for the ``web_read`` tool, stated here because the
#: guard exists to make this an explicit decision. Measured on this branch's base
#: (``origin/main`` 37f3494fd) the figure is 27,274 — 46 tokens of headroom, so no
#: usable tool description fits under the old ceiling: the new tool is 552
#: characters of description plus a 684-character schema, ~445 billed tokens at
#: 2.78 chars/token, and the branch measures 27,729. The alternative to paying it
#: is not a smaller number but a second tool the model cannot use correctly: the
#: description is where "only pages a previous search in THIS session captured"
#: and "refuses rather than fetching" are stated, and a model that does not know
#: the second will call ``web_fetch`` instead and pay a network round trip for
#: every page. Headroom at 27,800 is 71 tokens, and the next context reduction
#: tightens it.
#: RAISED 27,800 -> 28,000 for the ``scratchpad://`` pointer in ``system.md``,
#: stated here because the guard exists to make this an explicit decision, and
#: with the THREE measurements rather than two, because the first revision of
#: this comment derived a delta from a contaminated base (review round 1, R4).
#: All three are this script, this machine, and the same deterministic char
#: arithmetic CI runs — CI's reading on the pre-remediation head was 27,998, to
#: the token, so there is no local-vs-CI gap to leave slack for:
#:
#:   base, ``origin/main`` ba225070        27,787   (13 tokens under the old
#:                                                   ceiling: no room for a new
#:                                                   pointer at all)
#:   + the ``system.md`` pointer            27,950   (+163)
#:   + ``ReadParams.path``'s scheme clause  27,998   (+48)
#:
#: The 48 was measured into the "base" the first revision compared against — a
#: tree with the schema change already in it — so that comment called 163 "the
#: whole of the increase" when the branch's real delta was 211. Round 1 then
#: removed the duplication that clause created (the pointer says the scheme, the
#: schema field does not have to), which gives the current head 27,949 — net
#: +162 against the base: pointer +163 tokens, schema -1 token (44,228 -> 44,225
#: characters, a 3-character edit). The round-1 trim was the LARGER step and it is
#: not the -1: it ran 44,362 -> 44,228, i.e. -137 characters = -49 tokens measured
#: against the pre-trim head. The ceiling is
#: set 51 above that, the same order of headroom as the ``secret`` (49) and
#: ``web_read`` (71) raises, so the ratchet stays tight — and the tighten band
#: below (1,200) is nowhere near tripped.
#:
#: The alternative to paying the pointer is not a smaller number: an agent that
#: does not know the scheme exists has nowhere to put its own scratch work and
#: puts it in the user's working directory, where it is indistinguishable from
#: an output the user asked for. The pointer is the only discovery channel that
#: costs nothing extra — the guide body is progressive disclosure and never
#: rides the start context, and its description only rides a turn that selects
#: it.
#:
#: RAISED 28,000 -> 29,950 for the ``console`` tool (design ui-console-tab §17.1
#: row C), stated here with the arithmetic rather than a wave of the hand,
#: because the guard exists to make this an explicit decision and because the
#: delta is large enough that a reader deserves to see where it went. Both
#: sides are this script, this machine, and the deterministic char arithmetic;
#: the base was measured by running THIS script against the console-less tree
#: (``origin/main`` a8fe0c1d, which is what ``~/local-operator-worktrees/
#: console-tab-design`` carries besides the design doc itself) rather than
#: derived from the branch, so the two numbers cannot share a mistake:
#:
#:   base, ``origin/main`` a8fe0c1d          27,912   (88 tokens under the old
#:                                                     ceiling: no room for a
#:                                                     10-method tool at all)
#:   + the tool's schema                     48,841 chars of tool_schemas vs
#:                                           44,124 = +4,717 chars = +1,697
#:   + its ``system.md`` section             33,947 chars of instructions vs
#:                                           33,136  =   +811 chars =   +292
#:   + its one inventory line                +10 chars = +4
#:   = measured on this head                  83,134 chars = ~29,904 billed
#:
#: Two reviews later the head moved by ONE more edit, and it is recorded here for
#: the same reason: QA round 2 found the shipped `keys` field and the guide
#: documenting `ctrl-c`/`shift-tab` — spellings the app's encoder refuses, since it
#: spells them `ctrl+c`/`shift+tab` — so the field names the canonical spelling and
#: its synonyms in +79 characters (+28 tokens): 83,213 chars, ~29,933 billed, 17
#: tokens of headroom. The full synonym list lives in `guide://console`, which is
#: progressive disclosure and never rides the start context, so only the pointer to
#: the canonical spelling is paid for here.
#:
#: (components rounded; the total is taken from the char counts, so the three
#: component figures are floors — 4,717/2.78, 811/2.78 and 10/2.78 are 1,696.8,
#: 291.7 and 3.6, and a reader who adds the printed integers gets 1,993 rather
#: than the 1,992 the endpoints give. The total is the measured one; the parts
#: are reported to the nearest token so a reader can see where it went.)
#:
#: The schema is 85% of it, and the schema is the capability: ten methods with
#: one method parameter is ONE tool, where ten tools would be ten schemas of
#: permanent tax (the ladder's rung 1). It was measured and then CUT once — the
#: parameter descriptions were shortened and the class docstring dropped, since
#: pydantic copies a docstring into the emitted schema's ``description`` —
#: taking the schema from 5,673 to 4,717 characters, i.e. -344 billed tokens
#: before this raise was written. What remains is the irreducible part of
#: "describe ten methods' arguments so a model can call them correctly", and the
#: per-method playbook lives in ``guide://console`` where it costs nothing until
#: it is read.
#:
#: And it is paid only where it is usable: ``build_console_tool`` is a createIf
#: factory that returns ``None`` unless the desktop app publishes a
#: console-capable record, so a session on a machine without the app carries no
#: console schema at all — this benchmark forces the gate ON precisely so the
#: figure reported is the worst case rather than the common one.
#:
#: The OTHER path is outside this arithmetic and is stated rather than hidden: with
#: no record at all, the tool is absent but the inventory carries its one-line
#: prohibition (``_NO_CONSOLE_NOTE``), so the inventory block goes 211 -> 727 chars
#: (+516 chars, ~+186 billed at 2.78 chars/token) for every session on a machine
#: without the desktop app. It is not in the figure above because the gate is forced
#: ON here; it is smaller than the tool-present case it trades against, and §14.5
#: chose the prohibition deliberately — an agent that does not know the capability
#: exists cannot ask for it — but a reader comparing this number to a session's real
#: start context should know which side of the gate they are reading.
#:
#: The ceiling is set 46 above the measured head, the same order of headroom as
#: the ``secret`` (49), ``web_read`` (71) and ``scratchpad://`` (51) raises, so
#: the ratchet stays tight; the tighten band below (1,200) is nowhere near
#: tripped and the next context reduction tightens it.
#:
#: RAISED 30,025 -> 30,150 for the per-command memory guard's ``memory_mb``
#: field (2026-09-21), stated with the same arithmetic the guard exists to
#: force. The base — ``origin/main`` 40544beb, the tree the PR diffs against —
#: measures 83,336 chars = ~29,977 billed on THIS machine, i.e. 48 tokens of
#: headroom; a ``BashParams`` field costs ~51 even with the shortest honest
#: description, so the head with the shipped three-semantics description
#: measures ~30,088 here. The alternative the ladder prefers — OFFSET the cost
#: — was measured and is NOT available: no schema clause in the prefix is
#: filler, and the field is not droppable because the tool-result text ("pass
#: memory_mb on the bash call") and the escape from F7 both name it.
#:
#: THE CEILING CLEARS **CI**, NOT JUST THIS MACHINE, and that is why it sits
#: ~60 above the local head rather than ~12 (remediation round 1, measured):
#: there IS a local-vs-CI gap for this change, because `tool_schemas` is
#: computed from the real pydantic models and one of them is platform-shaped.
#: Local (macOS, py3.12.13) measured `tool_schemas` 49,353 chars / head 30,088;
#: CI (ubuntu, py3.12) measured 49,420 chars / head **30,113** — 67 more
#: schema chars, ~25 billed tokens — so a ceiling set 12 above the LOCAL head
#: failed CI by exactly 13 tokens. The ceiling is therefore set with CI as the
#: binding reading: 37 above CI's head and 62 above this machine's. Only the
#: CI figure matters for the gate, so the 37 is the one to read; it sits just
#: UNDER the 46-51 band this file's ``secret`` (49), ``web_read`` (71) and
#: ``scratchpad://`` (51) raises chose, and the 62 above local is simply the
#: same ceiling seen from the smaller reading. The band is descriptive of prior
#: raises, not a constraint this one satisfies — the local-vs-CI gap, not the
#: band, decides the number here, and the tighten band below (1,200) is nowhere
#: near tripped. Earlier raises in this file found NO local-vs-CI gap; this one
#: does, so it is recorded rather than carried as the assumption that the two
#: always agree.
#: RAISED 29,950 -> 30,025 for the scratchpad-salience change, stated here with
#: the arithmetic because the guard exists to make this an explicit decision.
#: The base — THIS BRANCH'S base, ``origin/main`` 0bc5fb3a, the tree the PR
#: diffs against — measured by THIS script on this machine reads 83,185 chars =
#: ~29,923 billed, i.e. 27 tokens of headroom — no room for any new surface —
#: and the head measures 83,336 = ~29,977:
#:
#:   + the tool schemas                49,043 vs 48,920 = +123 chars = +44
#:     (``read`` names ``scratchpad://`` in its scheme list, which it already
#:      serves; ``write`` and ``edit`` each add a clause saying where the
#:      agent's OWN scratch goes)
#:   + the ``system.md`` paragraph      33,947 vs 33,919 =   +28 chars = +10
#:     (it now names both traps — the working directory and ``/tmp`` — and is
#:      scoped to TEXT scratch, which is what the store actually holds)
#:   = measured on this head             83,336 chars = ~29,977 billed
#:
#: The 28 is AFTER the offsetting rewrite, and that is the point: the paragraph
#: was rewritten to pay for its own clause where it could, not appended to.
#: "It is this session's own folder and is deleted with the session" became "it
#: dies with the session" (-42 chars), so the second trap is named for 23 net
#: characters rather than the ~65 the clause itself costs; the review round's
#: TEXT scoping added the other 5 ("Text "), and nothing else in the prompt or
#: the schemas moved with it. Offsetting cuts elsewhere were NOT taken because
#: the remaining candidates are not filler: the scheme list is what tells a
#: model the scheme exists at all (``read`` already serves it, so the omission
#: was a bug, not a saving), and the two write/edit clauses are the only place
#: the scratch convention reaches the moment of a tool call. Where the cost is
#: in doubt the ladder says measure it and justify it, and all three clauses
#: ride a prefix that already carries those three schemas.
#:
#: The CEILING did not move in the remediation round, and that is a measurement
#: rather than a stale figure left behind: the round-1 findings changed the
#: nudge's runtime STRING (result text, which rides no request prefix) and added
#: tests, so the only prefix bytes they touched are the 5 above. The
#: re-measured head is 83,336 and the ceiling sits 48 above it — inside the
#: 46-51 band this file's ``secret`` (49), ``web_read`` (71) and
#: ``scratchpad://`` (51) raises hold to — so the ratchet stays as tight as it
#: was, and the tighten band below (1,200) is nowhere near tripped.
#: RAISED 29,950 -> 30,700 for the ``network`` tool (the mesh's agent surface,
#: ``mesh-transport-identity.md`` §12.5 / ``mesh-ui.md`` §3.2), stated with the
#: same arithmetic and for the same reason as the raise above. BOTH sides were
#: measured by running THIS script, each in its own tree, rather than derived
#: from one another:
#:
#:   base, ``origin/main`` 87a0cf70           83,185 chars = ~29,923 billed
#:                                            (27 tools, 27 under the old
#:                                             ceiling)
#:   + the tool's schema           50,827 chars of tool_schemas vs
#:                                  48,920 = +1,907 chars = +686
#:   + its one inventory line      +10 chars = +4
#:   + instructions                       33,919 vs 33,919  = 0
#:   = measured on this head                  85,102 chars = ~30,612 billed
#:                                            (28 tools)
#:
#: So the whole delta is one tool, and it is the ladder's tax paid as its rung 5
#: describes: a new core tool ships its schema on every request, in every
#: session, whether or not it is called. It is UNGATED on purpose and the design
#: records why (§12.5: an agent must be able to create the FIRST network, so a
#: gate on "a relay exists" would strip the tool from exactly the session that
#: has to run ``lop network init``), which means there is no host on which this
#: raise is not paid. Two things make that the right trade anyway, and both are
#: stated rather than implied: the tool is the only way the operator's R19 brief
#: ("an agent can set a network up from a verbal request") is satisfiable on a
#: fresh machine, and its schema is already the trimmed one — a twelve-value
#: ``action`` enum with one field per flag the CLI takes, which is one schema
#: where twelve tools would be twelve.
#:
#: A reader comparing this number to a real session's start context should know
#: the one place the mesh's "nothing changes on a device with no network" claim
#: is not literally true: that session does gain this tool. It is the claim
#: ``mesh-ui.md`` §3.2 makes about every OTHER surface (no relay, no daemon, no
#: listener, no state), and the tool's own ``description`` is what keeps the
#: schema from being paid for nothing — it names the capability and says what
#: this build cannot do.
#: THE MERGED HEAD, measured by running THIS script after folding `main` in —
#: re-measured at the fold that produced the head this branch merges at, not
#: carried over: an earlier revision of this comment named a figure taken before
#: `main` moved again, and a ceiling is only a guard if the number under it is the
#: one the tree actually produces.
#:   85,630 chars = ~30,802 billed  (28 tools)
#: The two raises above are separate changes that both land here — the
#: scratchpad-salience clause and the `network` tool — so this ceiling is the
#: single number covering both, with 48 billed tokens of headroom under the
#: ratchet those raises describe. ONE assignment: the guard reads the module
#: constant, so a second value above it would be a figure nothing asserts.
BUDGET_BILLED_TOKENS = 30_850


#: How much slack is allowed before the guard demands the ratchet be TIGHTENED.
#:
#: This is what stops the ceiling above from being aspirational. A comment
#: asking future agents to lower the budget enforces nothing; a failing build
#: that names the new number does. Set to roughly double the current headroom
#: so incidental wording edits stay quiet, while a real saving — the smallest
#: individual optimization in this PR was ~600 billed tokens — opens enough
#: slack to require the ratchet to follow it down.
#:
#: Suppressed under ``--inflate``, which deliberately measures a fiction.
TIGHTEN_WHEN_HEADROOM_EXCEEDS = 1_200

#: Stand-ins for the operator-supplied text that rides the head block of a
#: real session. Fixed sizes, because a guard whose threshold moves with the
#: developer's own ~/.local-operator contents is not a guard — it would pass
#: on a machine with no custom instructions and fail on one with a long file,
#: for reasons having nothing to do with the diff. These lengths are typical
#: of a configured install (measured on this machine: ~5.7k chars of custom
#: instructions, ~8k of repo guidance after truncation).
_SAMPLE_USER_INSTRUCTIONS = "- A standing operator preference line.\n" * 150
_SAMPLE_REPO_GUIDANCE = "Project convention paragraph explaining a rule.\n" * 170


def tool_schema_chars(tools: list[AgentTool]) -> int:
    """Characters the provider tools array costs on every request.

    Mirrors what ``Session.context_breakdown`` counts: name, description and
    the JSON-serialized parameter schema, per tool.
    """
    return sum(
        len(tool.name) + len(tool.description or "") + len(json.dumps(tool.parameters or {}))
        for tool in tools
    )


def measure_start_context(
    *,
    user_instructions: str = _SAMPLE_USER_INSTRUCTIONS,
    repo_guidance: str = _SAMPLE_REPO_GUIDANCE,
    inflate_schemas: int = 0,
) -> dict[str, Any]:
    """Character cost of everything a fresh session puts on the wire.

    ``inflate_schemas`` pads the TOOL SCHEMAS specifically. The fail-proof
    lever has to reach the largest component — the schemas are ~56% of the
    total and are what this PR is about — not only the operator text, or
    "prove it can fail" demonstrates the guard over the wrong half.
    """
    tools = build_real_tools(str(REPO))
    if inflate_schemas:
        # A new property on the first tool: the shape a real schema regression
        # takes (a tool grows an argument), rather than opaque filler.
        padded = dict(tools[0].parameters or {})
        props = dict(padded.get("properties") or {})
        props["_bench_filler"] = {"type": "string", "description": "x" * inflate_schemas}
        padded["properties"] = props
        tools = [tools[0].model_copy(update={"parameters": padded}), *tools[1:]]
    blocks = build_system_blocks(
        tools,
        skills_block="<skills/>",
        env_details="cwd: /Users/example/project\nplatform: Darwin 25.6.0 (arm64)",
        date_str="2026-01-01",
        user_instructions=user_instructions,
        repo_guidance=repo_guidance,
    )
    parts: dict[str, Any] = {
        "instructions": len(blocks[0]),
        "tool_inventory": len(blocks[1]),
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "tool_schemas": tool_schema_chars(tools),
    }
    parts["TOTAL"] = sum(parts.values())
    parts["n_tools"] = len(tools)
    # Names, so a short surface can say WHICH tool a host-dependent gate ate.
    parts["names"] = [t.name for t in tools]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET_BILLED_TOKENS,
        help="ceiling in billed tokens (default: the enforced ratchet)",
    )
    parser.add_argument(
        "--inflate",
        type=int,
        default=0,
        help=(
            "append N characters of filler to the sample custom instructions. "
            "Exists to PROVE the guard can go red — a guard that cannot fail "
            "is worse than none (AGENTS.md, 'Prove the test can still fail')."
        ),
    )
    parser.add_argument(
        "--inflate-schemas",
        type=int,
        default=0,
        help=(
            "grow a TOOL SCHEMA by N characters. The schemas are the largest "
            "component, so the fail-proof lever must reach them too."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # A measurement tool that can silently measure the WRONG TREE is the same
    # defect class as one that measures the wrong surface. `sys.path.insert(0,
    # REPO)` above does not win against a cwd that already holds a
    # `local_operator/` package (under `python -c`, cwd precedes PYTHONPATH),
    # so a reviewer comparing two checkouts can get two numbers from one tree
    # and no error. Assert what we actually imported, and say it out loud.
    resolved = Path(local_operator.__file__).resolve().parent
    expected = (REPO / "local_operator").resolve()
    if resolved != expected:
        print(
            f"PROVENANCE MISMATCH: measuring {resolved}\n"
            f"                     expected  {expected}\n"
            "      Refusing to report a figure for a tree this script does not live in."
        )
        return 2
    print(f"measuring: {resolved}")

    parts = measure_start_context(
        user_instructions=_SAMPLE_USER_INSTRUCTIONS + ("x" * args.inflate),
        inflate_schemas=args.inflate_schemas,
    )
    total_chars = parts["TOTAL"]
    billed = round(total_chars / CHARS_PER_BILLED_TOKEN)

    # The surface is stated ALWAYS, not only under --verbose: a silent drop
    # from the full tool set is exactly how this guard would go back to
    # measuring a machine no user has. See real_tool_surface._forced_browser_backend.
    n_tools = int(parts["n_tools"])
    parts_names = list(parts["names"])
    expected_tools = len(DEFAULT_TOOL_NAMES)
    print(f"tool surface: {n_tools}/{expected_tools} tools")
    if n_tools != expected_tools:
        # Name the tools, not just the count: the whole point of this check is
        # that a host-dependent gate dropped something, and "which one" is the
        # first question anyone reading a CI log will ask.
        missing = sorted(set(DEFAULT_TOOL_NAMES) - set(parts_names))
        print(
            f"FAIL: measured {n_tools} of {expected_tools} default tools.\n"
            f"      missing: {', '.join(missing) or '(none — duplicate names?)'}\n"
            "      The benchmark must measure the FULL surface on every host, or it "
            "reports headroom a real session does not have.\n"
            "      If a tool was deliberately removed, update DEFAULT_TOOL_NAMES; "
            "otherwise fix the gate that dropped it."
        )
        return 1

    if args.verbose or args.inflate or args.inflate_schemas:
        for key in ("instructions", "tool_inventory", "environment", "knowledge", "tool_schemas"):
            chars = parts[key]
            print(f"  {key:16} {chars:>8,} chars  ({chars / CHARS_PER_BILLED_TOKEN:>7,.0f} billed)")

    print(
        f"start context: {total_chars:,} chars = ~{billed:,} billed tokens "
        f"(at {CHARS_PER_BILLED_TOKEN} chars/token) vs budget {args.budget:,}"
    )
    if billed > args.budget:
        print(
            f"FAIL: start context exceeds the budget by {billed - args.budget:,} tokens.\n"
            "      Reduce the system prompt, the tool inventory or the tool schemas — "
            "or state in the PR what regressed before raising the budget."
        )
        return 1

    # SELF-TIGHTENING. Without this the ratchet is aspirational: a comment
    # saying "each reduction should lower this line" enforces nothing, so the
    # likeliest outcome is that today's number becomes the permanent ceiling
    # and the docs/REWRITE.md target is never revisited. Slack beyond the band
    # is therefore also a failure — a loud, trivially-fixed one that hands the
    # next agent the exact number to write down.
    #
    # It is a BAND rather than a hard equality so ordinary wording edits do not
    # trip it; only a real reduction (or a deliberate tool removal) opens
    # enough slack to require the ratchet to follow it down.
    headroom = args.budget - billed
    inflating = bool(args.inflate or args.inflate_schemas)
    if not inflating and headroom > TIGHTEN_WHEN_HEADROOM_EXCEEDS:
        print(
            f"FAIL: {headroom:,} billed tokens of headroom exceeds the "
            f"{TIGHTEN_WHEN_HEADROOM_EXCEEDS:,}-token slack band.\n"
            "      The context got smaller — good. Tighten the ratchet so the saving "
            "is defended:\n"
            f"      set BUDGET_BILLED_TOKENS = {billed + TIGHTEN_WHEN_HEADROOM_EXCEEDS // 2:,} "
            "in scripts/bench_context_budget.py."
        )
        return 1
    print(f"PASS ({headroom:,} billed tokens of headroom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

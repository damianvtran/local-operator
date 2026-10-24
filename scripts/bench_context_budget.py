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
from local_operator.i18n.resolve import preferred_language_line  # noqa: E402
from local_operator.prompts_api import (  # noqa: E402
    CHANNEL_ASK,
    CHANNEL_HUB,
    build_system_blocks,
)
from local_operator.tools.deferral import deferred_tool_names  # noqa: E402
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
#:
#: RAISED BY EXACTLY WHAT THE ATTACHED-INTERFACE BLOCK COSTS (2026-09-23).
#: ``prompts_api.build_system_blocks`` now emits a POSITIVE ``<interactivity>``
#: body for an attached session — the text that says a question WILL be
#: presented — where it previously emitted no block at all, so this guard's
#: default render (``interactive=True``) grew by it. Measured with the command
#: below, same tree otherwise: 83,408 chars = 30,003 billed at the merge base
#: ``1392324b`` (147 under the old 30,150), 84,005 chars = 30,218 billed with
#: the block. The block therefore costs 215 billed tokens on EVERY request of
#: an attached session (design:
#: ``docs/design/attached-interface-signal.md`` §3.4, where it was estimated at
#: ~90 — that gap is why this raise was not in the plan).
#:
#: RE-MEASURED IN ROUND 2 (2026-09-23, review remediation): 84,009 chars =
#: 30,219 billed, 146 headroom. The block itself is UNCHANGED — the round-2 work
#: channelled it (six bodies keyed on two measured facts) and left the attached
#: ``ask`` body byte-identical, which is the shape this guard measures. The 4
#: characters are the browser tool's own description: "NOTIFY the user" became
#: "NOTIFY the operator", so one person is named one way across the access flow
#: (round 1, D8). The ratchet therefore stays where the raise put it — headroom
#: 146, and NOT re-raised to 147: an addition that costs tokens costs headroom.
#:
#: The raise is deliberate and is NOT a loosening: 30,365 restores exactly the
#: 147-token headroom the tree had at the merge base, so the next addition finds
#: the ratchet as tight as this one did. It is not paid for out of the block,
#: because the block's sentences ARE the fix — the incident was a model told
#: "nobody is watching a screen" while the operator was reading the session, and
#: silence gave it nothing else to read. A future round that wants these tokens
#: back should take them from the block's style hint ("Write for a reader who
#: may answer minutes later…", ~51 tokens), not from the two sentences that
#: state the fact.
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
#: fresh machine, and its schema is the trimmed one — a sixteen-value
#: ``action`` enum with one field per flag the CLI takes, which is one schema
#: where sixteen tools would be sixteen. The four values added since (``sessions``,
#: ``trust``, ``credentials``, ``definitions_state``) are the same trade at a
#: smaller size: they close surfaces the tool could already SEE (a peer and
#: nothing it held), and every one of them is a flag the CLI already takes.
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
#:   85,926 chars = ~30,909 billed  (28 tools; origin/main 6bd703e5 alone:
#:   84,009 = ~30,219, 27 tools — so the tool is still +1,917 chars = +690)
#: The three raises above are separate changes that all land here — the
#: scratchpad-salience clause, the attached-interface ``<interactivity>`` block
#: (main's 30,365) and the `network` tool — so this ceiling is the single number
#: covering all three: main's 30,365 plus the tool's +690, which keeps main's own
#: 146 billed tokens of headroom exactly, so the network raise costs the ratchet
#: nothing it does not describe. ONE assignment: the guard reads the module
#: constant, so a second value above it would be a figure nothing asserts.
#: RAISED 31,055 -> 32,737 for the PROJECTS PAIR (``project`` + ``project_delete``,
#: the projects primitive's slice 1), stated with the arithmetic rather than a
#: wave of the hand, and only after the trim this file's ``secret`` entry
#: demands was taken. Measured by running THIS script with the pair filtered out
#: of the same surface in the same tree, so the difference is the pair and
#: nothing else:
#:
#:   baseline (this tree, pair excluded)   86,208 chars = ~31,010 billed
#:   head (pair included)                  90,883 chars = ~32,692 billed
#:     = +4,675 chars = +1,682 billed: the two tool schemas (+4,648 chars) and
#:       the two tool-inventory lines (+27). The guide's own row rides no block
#:       here — measured by hiding ``guide://projects`` from discovery: block 3
#:       moves by 0 chars.
#:
#: The pair as this branch first wrote it cost 1,799 billed (the reviewer's
#: 32,809 on the untrimmed head, against the 31,010 baseline above); moving the
#: rationale out of the field descriptions and into ``guide://projects`` — which
#: costs nothing until it is read — cut it to 1,682, the same remedy the
#: ``secret`` entry records. What is left is the price of the capability: 18
#: option fields plus a nested milestone model, both required by the design's
#: §V2.A model extensions, in two tools rather than one merged behind the write
#: tier (which would save one schema and one inventory line) because the
#: destructive-split convention every neighbouring pair follows is worth more
#: than the ~100 tokens it costs. Merging the pair is the one lever a future
#: round could pull; dropping the date/estimate/milestone fields is the other,
#: and it would delete the interface this slice exists to ship.
#:
#: The raise RESTORES the headroom rather than loosening it: the tree carried
#: 45 tokens before the pair (31,055 - 31,010) and carries 45 after
#: (32,737 - 32,692), so the next addition finds the ratchet as tight as this
#: one did, and the tighten band below (1,200) is not in play. The pair is
#: createIf-gated on a store being attached, so a session without one carries
#: neither schema; on a healthy host it is paid on every request, which is why
#: it was trimmed before the ceiling moved.
#: RAISED 32,737 -> 32,791 for the streamlined-sudo handover (PR #1657), stated
#: with the arithmetic and only after the trims this block's ``secret`` entry
#: demands were taken. The pair: the ``has_console`` note now teaches the
#: ask->relay loop in place of the old "never attempt a password yourself", and
#: the ``secret_ref`` field description names the session credential source.
#: As first written that cost +103 chars (note +42, field +61) against the 15
#: billed tokens of headroom the tree carried, so both strings were trimmed
#: before the ceiling moved — "type it into the console" became "relay it", and
#: the field's rationale took the design doc's own phrasing — which is the
#: remedy the ``secret`` entry records, applied again. Measured by running THIS
#: script on the same tree:
#:
#:   base (d4da37298)   90,966 chars = ~32,722 billed
#:   head (trimmed)     91,030 chars = ~32,745 billed
#:     = +64 chars = +23 billed: the has_console note (+24) and the secret_ref
#:       description (+40), the two strings the flow needs.
#:
#: The raise is 54 tokens and lands the ceiling at head + 46 — the headroom band
#: the entry above keeps — so the next addition finds the ratchet as tight as
#: this one did, and the tighten band below (1,200) is not in play.
#: RAISED 32,737 -> 33,193 for the ``network`` SESSION PLANE (the agent tool's
#: session verbs and the two-phase pair). Measured by running THIS script on the
#: SAME host from two trees, one after the other, so the difference is this
#: change and nothing else:
#:
#:   baseline (origin/main a937895ce)      90,883 chars = ~32,692 billed
#:   head (this branch, rebased on it)     92,014 chars = ~33,099 billed
#:     = +1,131 chars = +407 billed: the four added ``action`` values
#:       (sessions / trust / credentials / definitions_state), the fields that
#:       drive them (peer, create, prompt, engage, stop, delete, all_peers,
#:       trust_state, role, device, expires) — one schema where sixteen tools
#:       would be sixteen, the trade this tool's entry further up records.
#:
#: ``confirm`` IS NOT IN THAT FIGURE, and its absence is the point: the field
#: this branch first shipped was REMOVED in the same PR's remediation round on the
#: operator's ruling (a tool able to echo the code it printed satisfies a
#: comparison both devices derive from — see the tool's own docstring), so a
#: saving sits inside the figure rather than a cost.
#:
#: The raise RESTORES the headroom rather than loosening it, the same way the
#: projects entry does: the tree carried 45 tokens before this change
#: (32,737 - 32,692) and carries 94 after (33,193 - 33,099) — the 49-token
#: difference is exactly the removed field, banked rather than spent, and the
#: ceiling stays where this change put it because a later round can only put a
#: field back by arguing for it. The tighten band below (1,200) is not in play.
#:
#: What was NOT paid for out of the budget: the tool DESCRIPTION. It replaced a
#: sentence claiming sessions on other devices were unreachable — a capability
#: the relay has served for rounds — so that half is a correction, not a cost.
#:
#: RE-MEASURED AT THE FOLD onto origin/main 7737441e, the tree this branch now
#: merges at: the figures in the entry above were taken against the branch's
#: original base (a937895ce), and this file's header is explicit that a ceiling
#: is only a guard if the number under it is the one the MERGED tree produces, so
#: they are re-taken here by the same method — one host, two trees, one after the
#: other:
#:
#:   baseline (origin/main 7737441e)       90,993 chars = ~32,731 billed
#:   head (this branch, folded)            92,124 chars = ~33,138 billed
#:     = +1,131 chars = +407 billed — the SAME delta the entry above records, so
#:       the fold changed nothing about what this change costs; what moved is the
#:       baseline, by 39 billed tokens of ``main``'s own incoming surface (the
#:       streamlined-sudo handover above and the remote-interaction wiring).
#:
#: The ceiling stays where this change put it, and on the merged head it carries
#: 55 tokens of headroom (33,193 - 33,138) rather than the 94 the pre-fold pair
#: above produces, because ``main`` raised its OWN ceiling from 32,737 to 32,791
#: while this branch was open: main's ceiling now sits 60 above main's head, and
#: this change asks for +402 while costing +407, so five of those tokens net out.
#: Still far inside the tighten band below, so the ratchet does not move.
#: RAISED 33,193 -> 33,330 for the ``project`` tool's milestone full-replace
#: guard: ``op='update'`` with ``milestones`` is refused unless
#: ``replace_milestones=true``, naming the count at risk and both safe paths
#: (the surgical ``op='milestone'`` upsert, or the deliberate replace). The
#: guard exists because the bare "FULL replace" phrasing in the field
#: description read as "update my milestones" to an agent that then wiped
#: every sibling. Measured by running THIS script on the SAME host from two
#: trees, one after the other (the network entry's method):
#:
#:   baseline (origin/main 9f4e9d8b2)      92,124 chars = ~33,138 billed
#:   head (this branch)                    92,476 chars = ~33,265 billed
#:     = +352 chars = +127 billed: the ``replace_milestones`` boolean and its
#:       description, plus the ``milestones`` description rewrite that drops
#:       the bare "FULL replace" phrasing and names the refusal and the safe
#:       path.
#:
#: The raised ceiling lands 65 above the head — the band the entries above
#: keep (49, 51, 71; 37-62 where CI binds). The trim alternative was measured
#: and does not fit: the flag plus one honest sentence about when to pass it
#: is the discoverable half of a data-loss fix (an agent that cannot see the
#: flag cannot deliberately replace), and the earlier ``memory_mb`` entry
#: records that no other schema clause in the prefix is spare. The tighten
#: band below (1,200) is not in play.
#:
#: RAISED 33,193 -> 33,533 for the S6d PROJECT DATA slice (the ``project``
#: tool's ``owner`` / ``team`` / ``title`` fields, the bounded history tail,
#: and ``attach``'s copy-in files — the fields every project surface reads).
#: Stated with the arithmetic and only after the trim this block's ``secret``
#: entry demands was taken: the first writing cost +396 billed, and moving the
#: rationale into ``guide://projects`` (title's fallback phrasing, attach's
#: "screenshots/evidence" note, the tool description's parenthetical) cut it to
#: +340. Measured by running THIS script on the same host from two trees, one
#: after the other:
#:
#:   baseline (origin/main fc7ec229a)      92,124 chars = ~33,138 billed
#:   head (this branch)                    93,069 chars = ~33,478 billed
#:     = +945 chars = +340 billed: five optional fields on ONE tool
#:       (owner, team, title, attach, history) — a display name, the two
#:       attribution labels and the two write/read handles the desktop detail
#:       page consumes; the alternative was a second tool or surfaces that
#:       cannot be addressed.
#:
#: The raise lands the ceiling at head + 55 — the headroom band the entries
#: above keep — so the next addition finds the ratchet as tight as this one
#: did, and the tighten band below (1,200) is not in play.
#:
#: MERGED (fold of main at bfc0bcf2a): the two raises above COMPOSE on this
#: branch — main's ``replace_milestones`` guard and the S6d fields both live on
#: the merged head — so the ceiling is re-measured on the merged trees rather
#: than summed (some schema text both changes carried deduplicates through the
#: fold):
#:
#:   baseline (origin/main bfc0bcf2a)      92,476 chars = ~33,265 billed
#:   head (merged)                         93,421 chars = ~33,605 billed
#:     = +945 chars = +340 billed — exactly the S6d delta its own entry above
#:       records, because main's guard is IN the baseline; the fold adds no
#:       surface of its own, only the merge.
#:
#: The merged ceiling lands at head + 55 — the band this file keeps — and the
#: tighten band below (1,200) is not in play.
#:
#: RAISED 33,660 -> 33,760 for the project STATUS LIFECYCLE slice: the status
#: vocabulary (planning|active|qa|validation|paused|done|archived) in the
#: field description and the ``force_done`` switch that opens the done gate —
#: both on the ONE ``project`` tool every surface reads. Measured same-host,
#: one tree after the other:
#:
#:   baseline (origin/main c95d1ae01)      93,421 chars = ~33,605 billed
#:   head (this branch)                    93,699 chars = ~33,705 billed
#:     = +278 chars = +100 billed: the lengthened ``status`` description (the
#:       seven words and the gate sentence) plus the ``force_done`` field; the
#:       alternative was a status word set only the refusal knows, which
#:       cannot teach the vocabulary before the model's first mistake.
#:
#: The raise lands at head + 55 — the band this file keeps — and the tighten
#: band below (1,200) is not in play.
#:
#: RAISED 33,760 -> 34,587 for the MONITOR tool (docs/design/monitor-tool.md
#: §19.1 slice 1 — delta-watch over repeated read-only calls). Measured
#: same-host, one tree after the other; the baseline tree is ``origin/main``
#: AFTER this file's own last raise, so the delta is this change and nothing
#: else:
#:
#:   baseline (origin/main 283e4ff3b)      93,699 chars = ~33,705 billed
#:                                          (30 tools — the tool does not
#:                                           exist on main)
#:   head (this branch)                    95,999 chars = ~34,532 billed
#:                                          (31 tools)
#:     = +2,300 chars = +827 billed: the tool's own schema (``MonitorParams``)
#:       plus its inventory line, and the system.md sentences that teach
#:       wake-vs-monitor (§13) — the ladder's rung 5 tax, stated rather than
#:       implied. The tool is GATED on the session's monitor scheduler
#:       (createIf), and a bare ``ToolContext`` therefore gates it OFF, which
#:       is why ``scripts/real_tool_surface.py`` gains the stub in this same
#:       change: without it the benchmark measures 30 of 31 tools and reports
#:       headroom no real session has — the same green-by-fiction the
#:       projects raise above records.
#:
#: The raise lands at head + 55 — the band this file keeps, and together with
#: the known CI-vs-local offset (~25 billed) it clears CI rather than this
#: machine alone — and the tighten band below (1,200) is not in play.
#:
#: RAISED 34,587 -> 34,716 for slice 3 of the same Monitor design (§14 —
#: origin-aware notifications: the ``notify`` parameter carried on wake and
#: monitor deliveries, plus the §13.4 guidance sentences). Measured same-host,
#: one tree after the other; the baseline is ``origin/main`` (a5007c8e6):
#:
#:   baseline (origin/main a5007c8e6)       96,044 chars = ~34,548 billed
#:                                           (PASS, 39 billed of headroom)
#:   head (this branch)                     96,357 chars = ~34,661 billed
#:     = +313 chars = +113 billed: the wake tool's ``notify`` field and its
#:       guidance sentence (+230 chars), and system.md's §13.4 sentence
#:       (+83). The guide bullet rides ``guide://monitor`` and never enters
#:       the prefix.
#:
#: THE TRIM ALTERNATIVE WAS ASSESSED AND DOES NOT FIT: the field ALONE costs
#: +200 chars (~+72 billed) against main's 39 billed of headroom, so dropping
#: both guidance sentences would still blow the ceiling by ~33; and the
#: field's description IS §14.4's discovery surface, so shrinking it to fit
#: would trade the feature for the number. The raise lands at head + 55 — the
#: band this file keeps, and together with the known CI-vs-local offset (~25
#: billed) it clears CI rather than this machine alone — and the tighten band
#: below (1,200) is not in play.
#:
#: RAISED 34,716 -> 34,789 for the PROJECTS CREATE-DEFAULTS wording slice —
#: folded OVER the monitor-notify raise above: the ``project`` tool's
#: ``description`` / ``title`` / ``progress`` / ``attach`` field text now says
#: what those fields already do (a markdown description that is rendered, a
#: short display title, every NEW progress line appended to the history,
#: attachments stored on that entry), and the empty-listing receipt teaches
#: the same create default. Measured on the MERGED tree with the sanctioned
#: command; the slice's own delta (+203 chars = +73 billed, measured
#: same-host on its pre-fold trees) carries over exactly — the four field
#: texts ride ``tool_schemas`` and are the only block this slice moves:
#:
#:   baseline (origin/main 8c5b762df)      96,357 chars = ~34,661 billed
#:   head (this branch, folded)            96,560 chars = ~34,734 billed
#:     = +203 chars = +73 billed: the four field descriptions on ONE tool.
#:       The empty-listing receipt that teaches the same create default is
#:       runtime text — it never enters the measured start context and costs
#:       nothing until a listing is read. The schema is the only text an agent
#:       reads before its first create, and the alternative was a refusal (an
#:       over-long description) or a misread of "one line on the workstream"
#:       that the guide then has to undo; the rationale stays in the guide and
#:       the seeds, which cost nothing until read.
#:
#: The raise lands at head + 55 — the band this file keeps, and together with
#: the known CI-vs-local offset (~25 billed) it clears CI rather than this
#: machine alone — and the tighten band below (1,200) is not in play.
#:
#: RAISED 34,789 -> 34,961 for the `backend` hint on `BrowserParams` (issue
#: #1723: the guide and the source comment shipped the promise while the field
#: was absent — `extra="forbid"` rejected the documented argument — and design
#: §16.1's escape hatch is what lets a device-trust / hardware-key /
#: conditional-access task reach the user's real profile while the app runs).
#: Re-measured on the REBASED tree (the base moved twice under this branch:
#: f76d2b368 -> 7d781ccbd -> d497e91e1); the baseline is this same tree with
#: the added field and clause removed — byte-identical to the base schema,
#: because the JSON is deterministic:
#:
#:   baseline (origin/main d497e91e1)       96,573 chars = ~34,738 billed
#:                                           (PASS, 51 billed of headroom)
#:   head (this branch)                     97,039 chars = ~34,906 billed
#:     = +466 chars = +168 billed: one optional field on the ONE `browser`
#:       tool — rung 1, "extend an existing tool", so no new schema — plus the
#:       one clause in its description that names the escape hatch. Anything
#:       leaner is a `backend` with no model-facing semantics, which is the
#:       bug this change fixes.
#:
#: The raise lands at head + 55 — the band this file keeps, and together with
#: the known CI-vs-local offset (~25 billed) it clears CI rather than this
#: machine alone — and the tighten band below (1,200) is not in play.
#: FOLDED with the PROACTIVE CLASS + PATIENCE slice (design §8; R29–R38) — the
#: ``patience`` tool a proactive session can see and the ``patience`` field on
#: ``send`` — and, on the latest fold, with main's own ``backend`` raise above.
#: Re-measured ON THE FOLDED TREE with every raise present — 99,471 chars =
#: ~35,781 billed, 32/32 tools; the enforced ceiling is that measurement + 55
#: (the band this file keeps, which also clears the CI-vs-local offset
#: recorded above).
#:
#: RAISED 33,760 -> 34,634 for the PROACTIVE CLASS + PATIENCE slice (design
#: §8; R29–R38): the ``patience`` tool a proactive session can see, and the
#: ``patience`` field on ``send``. Measured by running THIS script from two
#: trees, one after the other (the base tree checked out at ``origin/main``
#: 11fc505e4, which is this branch's merge base — the two numbers are the same
#: tree pair the gates ran on):
#:
#:   baseline (origin/main 11fc505e4)      93,699 chars = ~33,705 billed
#:   head (this branch)                    96,131 chars = ~34,579 billed
#:     = +2,432 chars = +874 billed, and the delta is two pieces exactly:
#:       the ``patience`` tool's schema and its one inventory line
#:       (+2,421 chars of tool_schemas, +11 of tool_inventory), and the
#:       ``patience`` field on ``send`` (+308 chars, measured on
#:       ``SendParams.model_json_schema()`` in each tree). The first writing
#:       cost +948 billed; moving the "defaults to the configured 5m" clause
#:       out of ``send``'s field description — the tool that owns the default
#:       says it — cut it to +874, the same trim discipline this block's
#:       ``secret`` entry demands.
#:
#: What makes the raise the right trade is that the GATED part is paid by no
#: session that exists: ``patience`` is createIf-gated on the proactive class
#: AND a scheduler (design §8.2.5, rung 3 on the tool-footprint ladder), and
#: R37 makes the class opt-in — the packaged Aida seed is the only profile
#: that ships proactive — so the maximum-surface measurement this guard bounds
#: carries a schema no reactive session ever receives. The one unconditional
#: cost is the ``patience`` field on ``send``, and that is the price of the
#: attach-on-send API §8.2.5 requires: arming from the tool alone cannot
#: cover "the message this turn is about to send", which is the case the
#: mechanism exists for.
#:
#: RAISED 35,836 -> 36,935 for the ``sessions`` tool (design
#: ``docs/design/sessions-tool.md``; PR A of the sessions-tool workstream),
#: stated with the arithmetic because the guard exists to make a new tool an
#: explicit decision. BOTH sides are measured — the two CI readings are the
#: ``context-budget`` job logs this branch's push and its base produced — and
#: the tool's delta is identical on both platforms (its schema is
#: platform-free):
#:
#:   base, CI (ubuntu, py3.12)            99,464 chars = ~35,778 billed
#:                                        (PASS, 58 under the old ceiling)
#:   base, this machine (macOS, py3.12)  100,226 chars = ~36,053 billed
#:   head, CI                            102,526 chars = ~36,880 billed
#:                                        (the failing run's own line)
#:   head, this machine                  103,288 chars = ~37,154 billed
#:
#: and the delta is the tool and nothing else:
#:
#:   + tool_schemas    66,974 vs 63,923 = +3,051 chars = +1,098
#:   + inventory line                       +11 chars =    +4
#:   =                                      +3,062 chars = +1,102
#:
#: (components rounded up; the total is the char counts': 3,062 / 2.78 =
#: 1,101.4. The tool's own schema was trimmed once against the note's draft
#: before this raise was written — the note's measured 781 cl100k param tokens
#: to the shipped 699, description 788 chars — and the remaining cost is the
#: price of the capability: six ops in ONE schema where six tools would be
#: six, each op a flag surface the CLI already takes, with the visibility
#: default (the incident fix) and the receipts in the text because a guide
#: that is not read cannot state a default.)
#:
#: What makes the raise the right trade is where it is NOT paid: the builder
#: is createIf rung 3 and returns ``None`` unless ``context.subagent_launcher``
#: is present, so every child that may not delegate — the population that can
#: never call ``spawn`` — carries zero schema for it, and the figure above is
#: the forced-gate worst case (the benchmark gates it ON deliberately). The
#: ladder's rung 1 (extend an existing tool) is not available: the design's
#: §3 and §14 lock the ops onto one schema because the capability needs
#: structured parameters, per-op approval tiers and the gate, which is rung
#: 3's own example.
#:
#: THE GAP HERE RUNS THE OTHER WAY from the ``memory_mb`` entry above: this
#: tree reads LARGER on this machine than on CI (762 chars, ~274 billed), and
#: the 762 sits entirely in ``tool_schemas`` — 67,736 local vs 66,974 CI at
#: head, and the same 762 at base (64,685 vs 63,923) — i.e. in schemas OTHER
#: than this tool's, present before the diff and unmoved by it. So the ceiling
#: is set from the CI reading + 55 (the band this file keeps) = 36,935, which
#: is what the gate runs on. On this machine the reading sits above the
#: ceiling BOTH before and after the raise (36,053 vs 35,836 pre-existing;
#: 37,154 vs 36,935 after, a 219 overshoot against a 217 one), so a local
#: run's overshoot here is this recorded gap, not this change — CI remains
#: the binding reading. The tighten band below (1,200) is not in play.
#:
#: RAISED 36,935 -> 37,164 for the ``sessions`` tool's ``peek`` surface
#: (design ``docs/design/sessions-tool.md`` §8; PR B of the sessions-tool
#: workstream), stated with the arithmetic because the guard exists to make
#: schema growth an explicit decision. Measured with THIS script on both trees
#: of the stack, same machine: the base (PR A's merged state, ``00774a36`` on
#: ``origin/main``; its predecessors ``65f686a08``/``d91a208bd`` read
#: identically, so neither A's review remediations nor the merge moved any
#: context component) reads 103,288 chars = ~37,154 billed — the 219-token
#: local overshoot its entry above records — and this head reads 103,924
#: chars = ~37,383 billed, so the delta is +636 chars = +229 billed and it is
#: the peek surface and nothing else: six window fields on the ONE schema
#: (``steps``/``head``/``before_id``/``around_id``/``regex``/``digest``),
#: where four tools would have been four permanent schemas. On CI (where this
#: gate runs) the stack's base read 102,526 = ~36,880, so this head is
#: 103,162 chars = ~37,109 and the ceiling is the CI
#: head + 55, the band this file keeps. A local run still reads above it by the
#: same recorded 762-char platform gap, so CI remains the binding reading. The
#: peel-off a future reduction can act on: dropping the peek fields from the
#: schema should take this ceiling back down ~229 billed.
#: RAISED 37,164 -> 37,300 for ``ask``'s queued-deadline ``timeout`` field
#: (design ``docs/design/ask-nonblocking.md`` §2.1; PR A1 of the ask-queue
#: workstream), stated with the arithmetic because the guard exists to make
#: schema growth an explicit decision.
#:
#: The raise is EXACTLY this change's delta, and that is deliberate: base and
#: head were measured with THIS script on the same machine, so the recorded
#: platform gap between a local and a CI reading cancels out of the subtraction
#: and the ceiling keeps whatever headroom the base had. Local base
#: (``origin/main`` = ``40ca7910e``, measured in a detached worktree of it)
#: reads 103,994 chars = ~37,408 billed; this head reads 104,304 chars =
#: ~37,519, so the delta is +310 chars = +111 billed and it is the ONE field
#: and nothing else. The description is deliberately terse — §9 puts the
#: calibration copy in the flip PR — and a first draft measured +173, which is
#: why it is this short: the ladder's rung 1 (extend an existing tool) is the
#: right rung for a deadline on a tool that already exists, and it should still
#: cost as little as it can.
#:
#: A LOCAL run reads above the ceiling both before and after the raise
#: (37,408 vs 37,164 at base; 37,519 vs 37,300 at head), which is the 762-char
#: local-vs-CI gap the ``sessions`` entry above records — so a local overshoot
#: here is that gap, not this change, and CI remains the binding reading. The
#: ceiling follows THIS file's own method rather than the raw delta: CI head is
#: the local head minus that same recorded gap (~274 billed, so ~37,245), and
#: the ceiling is that plus the 55-token band this file keeps = 37,300. If the
#: gap has moved for reasons unrelated to this diff, CI's own reading is the
#: one that decides, and it prints the number to set — and the 55-token band is
#: the reason the field can grow by its own 111 without the ceiling chasing it.
#:
#: RAISED 37,300 -> 37,378 for the ``tool://`` on-demand tool reference (PR-A
#: of the prompt/tool-surface audit; design note in the audit session's
#: scratchpad, ``audit/mechanism_design.md``). The mechanism's own wire cost is
#: deliberately ZERO — a doc renders only when ``read tool://<name>`` asks for
#: it — so the raise is the always-loaded pieces that make it reachable, plus
#: two cue fixes co-landed with it and one duplication cut:
#:
#:   base (origin/main a2dbe29cc)       104,304 chars = ~37,519 billed
#:   head (this branch)                 104,521 chars = ~37,597 billed
#:     = +217 chars = +78 billed, across four edits:
#:       + the ``tool://`` cue in system.md's Tools section, +211 chars
#:         (already tightened once against the design draft)
#:       + C1, so listing or inspecting a peer no longer reads as gated
#:         behind "when the user asked" — only spawning is, +30
#:       + C4, the restart/update cause in the incident list, +46
#:       + the read description's one-word scheme list, +9
#:       - MINUS the console description's restatement of system.md's own
#:         prose: the "cannot wedge on a prompt" tail and its clause go, the
#:         boundary and the pinned "`bash` returns output directly"
#:         consequence stay, -79
#:
#: Both sides measured with THIS script on this machine (the base in a
#: detached checkout of origin/main) under a tiers-CONFIGURED
#: ``values.subagents``. That config is the one variable the ``agent``/``task``
#: docs read, and measuring it alone explains the number the entries above
#: record as a "platform gap": the 762-char difference between this machine's
#: configured reading and CI's is exactly the config delta in ``tool_schemas``
#: (68,665 vs 67,903, byte-exact) — with the config equal, the machine and CI
#: read identically. The ceiling follows the CLEAN arm, because that is what
#: CI renders and this gate compares: 104,521 - 762 = 103,759 chars = ~37,323
#: billed, + the 55-token band = 37,378. The remaining cue cost is the price
#: of making the reference discoverable at all — with no cue, no agent knows
#: it exists, which is exactly the op-ambiguity failure the audit measured.
#: The schema-slimming wave (audit fix list item 2) works this surface next
#: and should ratchet the ceiling back down. A LOCAL run under a clean config
#: reads 103,759 = ~37,323 and passes with the same 55 headroom; a
#: tiers-configured box renders the 762-char-larger surface above (219 over
#: the ceiling), which is that config dependence, not an unfixable platform
#: offset. The tighten band below (1,200) is not in play.
#:
#: RAISED 37,378 -> 37,491 for the PROJECTS-SEMANTICS backend slice (PR #1831:
#: coordination links, content-age staleness, refresh ≠ update), stated with
#: the arithmetic because the guard exists to make copy growth an explicit
#: decision. The slice's own delta is +187 chars = +67 billed, entirely in
#: ``tool_schemas`` (67,903 -> 68,090 on the clean arm: a main-only tree at
#: this fold's base vs the folded head) and NOTHING else — the ``project``
#: tool's description and its ``progress`` field description teach the link
#: roles and the refresh-vs-update split, and ``refresh`` joins the verb
#: enum — while the other counters are byte-identical. A first pass cost
#: +347 chars (+125 billed: CI run
#: 36714584926 failed the gate by 82), and the review round's remediation
#: trimmed the genuinely redundant halves — the "re-sending the identical
#: line refreshes too" clause and "rather than a working link" — before this
#: raise was written.
#:
#: This entry has survived TWO folds, and it now records both arithmetic
#: lessons in its own numbers. The first push measured 103,419 = ~37,201
#: against a 103,385/~37,189 branch-tree prediction: the +34 was MAIN's
#: growth entering through the CI merge ref (the agent tool's schema, +36 on
#: the tool itself), present because CI tests the merge with current main
#: while any branch-tree prediction misses it. So this entry states the
#: MERGE-REF numbers, BOTH arms, re-measured at the folded head with this
#: script on this machine — the clean arm in ``env -i`` is the arm CI
#: renders (same comparison as the tool:// entry above):
#:
#:   clean arm (CI), folded head      104,073 chars = ~37,436 billed
#:   tiers-configured arm, same head  104,835 chars = ~37,710 billed
#:                                    (104,835 - 104,073 = 762, the exact
#:                                    config delta the tool:// entry
#:                                    explains)
#:
#: A local run under that config therefore reads above this ceiling by the
#: same 762 chars; the clean arm is what CI renders and the binding reading.
#:
#: Of the +314 clean-arm growth since the tool:// entry (103,759 -> 104,073,
#: main kept moving between that merge and this fold), +187 is this slice's
#: project-tool copy and the remaining ~127 is main-side merges, none of them
#: here. The ceiling is the measured clean head + 55 (the band this file
#: keeps): 37,491. That is ~152 further chars of main-side growth before this
#: branch's CI run would eat the band, and CI's own reading then prints the
#: number that replaces this one. The peel-off a future reduction can act
#: on: moving the role wording into ``guide://projects`` (read on demand)
#: takes this back toward the base.
#:
#: RAISED 37,491 -> 37,515 for the ``sessions`` tool's per-op advertisement and
#: the eval bridge's failure notice (``fix/sessions-resume-0930-9d2e``), on top
#: of the projects entry above, stated with the arithmetic because the guard
#: exists to make copy growth an explicit decision. THIS branch was cut at
#: ``0dada4792`` and folded three times — onto ``fff390360``, then ``85ed0f7fd``
#: (+ the tool:// reader), then ``ebb5fd496`` — so the ceiling is re-derived on
#: the MERGED tree with THIS file's own method: the clean arm (the one CI
#: renders) reads 104,139 chars = ~37,460 billed, the tiers-configured arm
#: reads 104,901 = ~37,734 (the same 762-char config delta the entries above
#: explain), and the ceiling is the clean reading plus the 55-token band =
#: 37,515. The change's own component is +66 chars = +24 billed over the
#: projects entry's clean folded head (104,073 -> 104,139), which is exactly
#: the three platform-free strings measured during the first fold: sessions
#: description 738 -> 746 = +8; sessions schema 3,241 -> 3,238 = -3 (the
#: op-field description trimmed against the added ``help`` enum value); eval
#: description 693 -> 754 = +61 (the eval result now reports failed ``tool()``
#: calls in its own text). A local tiers-configured run reads 219 over
#: (37,734 vs 37,515) — the recorded gap minus the band — and CI remains the
#: binding reading; if any reading disagrees, the script wins.
#: RAISED 37,515 -> 37,756 for the ``team`` tool's ``label``/``aliases``
#: fields (the teams-label lane's core half: teams gain a local display label
#: and extra addressing keys), stated with the arithmetic because the guard
#: exists to make schema growth an explicit decision. Both are ADDITIONS to
#: one existing schema -- rung 1 of the footprint ladder, the cheapest rung --
#: and they are create/update fields only: no new tool, no new op, nothing on
#: the list/show path, nothing on any other tool.
#:
#: FOLDED ONTO THE HEAD ABOVE (a fourth fold for this branch: the
#: sessions-resume raise and the main-side merges landed while it was open),
#: re-derived against the merge-ref tree CI tests: the clean head measures
#: 104,809 chars = ~37,701 billed, so the ceiling is that + the 55-token band
#: this file keeps = 37,756. The composition is exact, measured rather than
#: summed: this branch's reading was 104,743 at the previous fold and the
#: sessions-resume slice above adds its own +66 chars (68,760 -> 68,826 in
#: ``tool_schemas``; every other counter byte-identical). This branch's own
#: delta underneath remains +670 chars = +241 billed against its base
#: (103,232 -> 103,902, measured ISOLATED with this script; the two fields
#: and nothing else):
#:
#:   + tool_schemas    68,333 vs 67,663 = +670 chars = +241
#:
#: A NON-isolated local run reads the 762-char subagents-tiers config delta
#: larger (see the ``tool://`` entry), so CI remains the binding reading and
#: an isolated local run can be compared to CI directly.
#:
#: The trade: labels/aliases on the tool schema are the ONLY model-facing
#: documentation of the new fields, and each sentence prevents a real misuse
#: (the empty-string reset; the whole-list replacement; the collision rule).
#: The peel-off a future reduction can act on: dropping the two fields from
#: the schema should take this ceiling back down ~241 billed.
#:
#: UPDATED (NO RAISE), 2026-09-30 — issue #1815: the ``project`` tool's
#: ``description`` field text now ends "(<= 2000 chars)." where it said
#: "(<= 240 chars)."; the schema's markdown promise is the contract, the old
#: cap refused the prose it advertised, and the fix raises DESCRIPTION_MAX to
#: 2000 with a remedy-bearing refusal. Neither the guide nor the refusal text
#: rides the prefix, so the field rename is the whole measured delta.
#: Re-derived after the fold onto ``origin/main`` ``2d58270b2`` (the #1831,
#: #1856 and team-label folds rewrote the measured trees), same machine, the
#: base in a detached worktree of that commit, the clean arm via ``env -i``
#: (the arm CI renders):
#:
#:   base (origin/main 2d58270b2, clean)  104,809 chars = ~37,701 billed
#:   head (this branch, clean)            104,810 chars = ~37,701 billed
#:     = +1 char = +0.4 billed under the rounding — the 240 -> 2000 rename,
#:       nothing else. Tiers arms read +762 (the documented config delta).
#:
#: Review rounds then moved the REFUSAL copy (remedy phrasing), the
#: ``@project:`` element's cut order and the listing clamps — all of it
#: runtime text, so the delta above is untouched: the script re-reads
#: 104,810 on the clean arm, and "over the ...-character cap" appears nowhere
#: in the ``project`` tool's serialized schema (checked directly against that
#: JSON), which is the only text the ratchet can move.
#:
#: NO RAISE: a one-character edit cannot be why the permanent per-call ceiling
#: moves, and it does not breach the binding reading: the clean arm reads
#: ~37,701 billed against the 37,756 ceiling — ~55 billed of headroom, +0.4
#: of it this change — while the tiers-configured reading sits above by the
#: same documented 762-char config gap, not by anything here.

#: RAISED 37,756 -> 37,938 for the bulk-resume SET form
#: (``feat/sessions-bulk-resume-1001``), stated with the arithmetic because the
#: guard exists to make copy growth an explicit decision. Re-derived on the
#: MERGED tree (this branch folded onto ``8a7ba1dd0``, the team-label raise
#: above already in) with this file's own method, the clean arm via ``env -i``:
#: 105,315 chars = ~37,883 billed, so the ceiling is that reading plus the
#: 55-token band this file keeps = 37,938. The change's own components,
#: measured on the assembled tool entry (this file's own ``tool_schema_chars``
#: shape, name+description+JSON schema): ``sessions`` description 746 -> 838
#: chars (the four new accepted inputs in the derived per-op summary and one
#: prose clause naming the set form) and ``sessions`` schema 2,938 -> 3,351
#: chars (four boolean fields -- ``paused``/``failed``/``all``/``dry_run`` --
#: with lean one-clause descriptions); +505 of the +506-char fold delta
#: (104,809 -> 105,315), the two raises composing because their measured
#: surfaces are disjoint (one tool each).
#:
#: LOWERED 37,756 -> 34,746 for the slimming wave (audit items 2+3,
#: ``feat/tool-slimming-1001``): the JSON schema descriptions of the eight
#: top-offender tools (project, console, browser, ask, hub, agent, network,
#: task) and the ``## Tools`` section of ``system.md`` gave up detail that now
#: lives in the ``read tool://<name>`` docs — which cost nothing until read —
#: and in the existing guides (browser/console/system-tools/peer-messaging/
#: network/sessions). The move is pinned, not asserted: one representative cut
#: phrase per tool is asserted BOTH ways in
#: ``tests/unit/tools/test_tool_docs.py::test_cut_wire_prose_moved_into_the_tool_doc``
#: (absent from the wire bytes AND present in the doc render), and every
#: cue/boundary the audit named as keep-here stayed put (C1, C4, the
#: ``tool://`` cue, the query-discipline bullet, the send/console/browser
#: prohibitions).
#:
#: RE-DERIVED 2026-10-01 after the fold onto ``origin/main`` ``0dac063b9``
#: (#1837/#1901 moved the base): same machine, the base in a detached
#: worktree of that commit, the clean arm via ``env -i`` (the arm CI
#: renders), both arms of THIS script, deterministic char counts each side:
#:
#:   clean arm        before 104,810 chars = ~37,701   after 96,442 = ~34,691
#:   configured arm   before 105,572 chars = ~37,976   after 97,204 = ~34,965
#:
#: (The configured arm still sits exactly the +762-char subagents-tiers delta
#: above the clean arm that the entries above document; the gap is expected
#: and not this change's.)
#:
#: The saving is 8,368 clean-arm chars = 3,010 billed, split:
#:
#:   + tool_schemas    68,827 -> 63,529 = -5,298 chars = -1,906 billed
#:     (project -579, browser -1,136, console -708, ask -697, hub -996,
#:      agent -828, task -334, network -58; the bash `eval` boundary clause
#:      ADDS +38 back. Per-tool figures use this script's own per-tool
#:      formula — ``len(name) + len(description) + len(json.dumps(parameters))``
#:      — so they sum EXACTLY to the -5,298 total; every other tool is
#:      byte-identical.)
#:   + instructions    34,971 -> 31,901 = -3,070 chars = -1,104 billed
#:     (the ``## Tools`` section; the content pins in
#:      ``tests/unit/test_prompts_api.py``, ``test_prompts_console_flags.py``
#:      and ``tests/unit/guides`` are what floor this number)
#:
#: Target was ~-3.7k billed (the audit's -15% of the tool schemas); the
#: actual is -3,010, and the gap is the pins rather than the effort: the
#: ask-tool and browser/console description phrases are asserted at the
#: PHRASE level by their own suites, the effort-tier mappings are
#: config-bound (their tests pin the rendered bytes in both arms), and the
#: ``i`` field's pointer is documented in ``harness/intent.py`` as a
#: deliberate trade. The next ~700 billed needs one of those decisions
#: revisited, not another copy edit.
#:
#: The ceiling is the measured clean head plus the 55-token band this file
#: keeps: 34,691 + 55 = 34,746. (The first cut of this constant left 41 and
#: then 22 of slack — the item-7 bash clause and the #1837 fold each moved
#: the head after the constant was written; this re-derivation applies the
#: band exactly, so every term closes to the token.) A local configured run
#: reads 219 over (34,965 vs 34,746) — the recorded +762 config gap minus
#: the band — as every configured local run has since the tiers delta
#: existed; CI renders the clean arm and is the binding reading. If any
#: reading disagrees, the script wins.
#:
#: RAISED 34,746 -> 34,928 for the bulk-resume SET form
#: (``feat/sessions-bulk-resume-1001``), on top of the entry above, stated with
#: the arithmetic because the guard exists to make copy growth an explicit
#: decision. Re-derived on the MERGED tree (this branch folded onto
#: ``31f981550``, the slimming wave above already in) with this file's own
#: method, the clean arm via ``env -i`` (the arm CI renders): 96,947 chars =
#: ~34,873 billed, so the ceiling is that reading plus the 55-token band this
#: file keeps = 34,928. The change's own components, measured on the assembled
#: tool entry (this file's own ``tool_schema_chars`` shape,
#: name+description+JSON schema): ``sessions`` description 746 -> 838 chars
#: (the four new accepted inputs in the derived per-op summary and one prose
#: clause naming the set form) and ``sessions`` schema 2,938 -> 3,351 chars
#: (four boolean fields — ``paused``/``failed``/``all``/``dry_run`` — with lean
#: one-clause descriptions); +505 chars = ~+182 billed, EXACTLY the fold delta
#: (96,442 -> 96,947), because the two changes' surfaces are disjoint — the
#: wave slimmed the eight worked tools, this one grows ``sessions`` alone,
#: re-measured on the merged tree rather than summed from the two entries.
#:
#: The raise is not a regression of the wave: the wave's own -3,010 billed
#: stands, and the net after this fold (96,947 = ~34,873) is still 2,828
#: billed under the pre-wave head (105,315 = ~37,883).
#:
#: RAISED 34,928 -> 35,088 for the browser read actions
#: (``feat/browser-read-actions``), on top of the entry above and with the
#: same arithmetic. Re-derived on the MERGED tree (this branch folded onto
#: ``65ca8eaee``, the slimming wave already in) with this file's own method,
#: the clean arm via ``env -i`` (the arm CI renders): base 96,947 chars =
#: ~34,873 billed, head 97,393 = ~35,033 billed; the ceiling is the head
#: reading plus the 55-token band this file keeps = 35,088. The change's own
#: components, measured on the assembled ``browser`` entry (this file's own
#: ``tool_schema_chars`` shape): description 979 -> 1,008 chars (the three
#: verb names — the per-op detail went to ``tool://browser``, where the three
#: new ops and their caps render on demand) and schema 2,583 -> 3,000 chars
#: (two fields — ``properties`` list[str], ``depth`` int|None — plus the
#: hit_test clause on the shared x/y descriptions); +446 chars = ~+160
#: billed, EXACTLY the fold delta (96,947 -> 97,393). The two new fields are
#: the feature's floor, so the raise is not a wording trade: with their
#: descriptions trimmed to empty the delta shrinks by ~75 chars and the
#: feature would still sit outside the old band. Review round 1 (R1-2) then
#: widened the ``properties`` copy to name underscores: +13 chars, head
#: re-read at 97,406 = ~35,038, still inside the band (50 left).
#:
#: RAISED 35,088 -> 35,150 for the agent display label's ``AgentParams.label``
#: (remediation round 1, Q4), with the arithmetic the guard exists to force.
#: The base is THIS branch's base, ``origin/main`` c3b2500bf, measured with
#: this same script on this machine: 97,406 chars = ~35,038 billed, i.e. the
#: 50 tokens of headroom the paragraph above recorded. The added field costs
#: 139 schema chars (39 of description, the rest the JSON scaffolding an
#: optional ``str | None`` property carries), so the head reads 97,545 =
#: ~35,088 — EXACTLY the old ceiling, 0 headroom, and therefore a CI failure:
#: the ~25-token local-vs-CI gap the ``memory_mb`` raise below recorded applies
#: here too (``tool_schemas`` is built from the real pydantic models and one is
#: platform-shaped). The ceiling is set with CI as the binding reading, the
#: same derivation that raise used: 62 above this machine's head, i.e. 37 above
#: CI's — just under the 46-51 band the ``secret`` (49), ``web_read`` (71) and
#: ``scratchpad://`` (51) raises chose. The field is the feature's floor, not a
#: wording trade: ``agent create``/``update`` must be able to set a label the
#: way ``team create``/``update`` already can, and with its description trimmed
#: to the 39-char minimum the head still lands on the old ceiling. The tighten
#: band (1,200) is nowhere near tripped.
#:
#: RAISED 35,150 -> 35,874 for the ASK FLIP (``feat/ask-flip-default``):
#: ``asks.policy.NONBLOCKING_ASK`` now defaults ON, so the QUEUED ask text is
#: what every session renders and the blocking text is what the operator's kill
#: switch (``LOP_ASK_NONBLOCKING=0``) selects. NO WORDING MOVED in this PR — the
#: two arms already existed, selected by the flag — so the raise is exactly the
#: measured DIFFERENCE between the arms, which is what the flip makes permanent.
#:
#: Both arms measured with THIS script on the same tree, on the CLEAN arm (an
#: isolated HOME and config dir, which is what CI renders — a machine with
#: ``values.subagents`` tiers configured reads 762 chars larger in
#: ``tool_schemas``, the config dependence the ``tool://`` entry above
#: explains):
#:
#:   arm                        chars      billed (2.78)
#:   kill switch / blocking     97,545     ~35,088  (today's main head, 62 left)
#:   shipped default / queued   99,557     ~35,812
#:   delta                       2,012        ~724
#:
#: The delta is entirely OUR text, so it renders identically here and on CI, and
#: it decomposes exactly into the three surfaces the flip moves: +671 chars in
#: the instructions block (``system.md``'s ask paragraph), +248 in the tail (the
#: ``<interactivity>`` body), +1,093 in ``tool_schemas`` (the ``ask`` tool's
#: description). 671 + 248 + 1,093 = 2,012 chars = ~724 billed, and
#: 35,150 + 724 = 35,874 — the arithmetic the guard exists to force.
#:
#: Cross-checked against the real CI log for ``main`` at ``e5a7a63b7`` (run
#: 37092370975, job 111115267258), which read 97,545 chars = ~35,088 billed with
#: all four non-schema counters byte-identical to CI's. So the ceiling leaves CI
#: 62 tokens of headroom on the SHIPPED DEFAULT — the same 62 main carries today
#: on the blocking one: the flip is paid for and not a token more. The
#: tiers-configured machine reads 100,319 = ~36,086 on the shipped default, 212
#: over this ceiling, which is the recorded config dependence rather than this
#: change; CI remains the binding reading.
#:
#: RAISED 35,874 -> 35,937 for the PEER-ADDRESS ROLE-WORD GUARD
#: (``fix/role-word-recipient``). No tool, field or counter moves; the delta is
#: the two ``target`` field descriptions the fix requires and nothing else.
#: ``SendParams.target`` and ``SessionsParams.target`` now state the two rules
#: the resolver applies — an EXACT conversation name / session id / cwd basename
#: wins over any substring, and a team role word (``manager``) is refused — so a
#: model reading the schema learns the guard exists. The guard itself lives in
#: the resolver; the copy is not a gate, which is why the failure being fixed
#: (a bare ``manager`` silently landing on a namesake TITLE) could not be
#: prevented by wording alone — but a model that is never told still types it.
#:
#: Measured with THIS script, CLEAN arm (an isolated HOME and config dir, which
#: is what CI renders — the config dependence the ``tool://`` entry above
#: explains does not apply). The BASE reads 99,557 at BOTH this branch's
#: merge-base (``cd3d03e9f``) and at ``origin/main`` (``e01d66d0d``) — the five
#: main commits since are schema-neutral — so the delta below is entirely this
#: branch's:
#:
#:   cd3d03e9f == origin/main   99,557 chars = ~35,812  (62 left)
#:   head (this branch)         99,733 chars = ~35,875
#:   delta                         176 chars =   ~63
#:
#: and it decomposes EXACTLY into the two edited fields:
#:
#:   +112 chars   ``SendParams.target``      (that schema's JSON 2,004 -> 2,116)
#:   + 59 chars   ``SessionsParams.target``  (3,725 -> 3,784)
#:   +  5 chars   the ONE extra em dash the new ``send`` copy adds — this block
#:                counts the provider array, where ``json.dumps`` escapes a
#:                non-ASCII char as ``\u2014`` (6 chars for 1)
#:   = 176 chars  =  ~63 billed
#:
#: The ceiling is the measured head + 62, the same band main carried before this
#: change. The tighten band (1,200) is nowhere near tripped.
#:
#: RAISED 35,937 -> 36,552 for the AGENT-SIDE SETTLE (``feat/ask-withdraw``,
#: design ``docs/design/ask-nonblocking.md`` §12): the ``ask_withdraw`` tool, a
#: createIf factory on the queued engine's own door (rung 3 — the door's
#: presence IS the mode, so the blocking arm, headless hosts and every subagent
#: pay zero schema for it). Measured with THIS script on both trees, same
#: machine, one after the other, the CLEAN arm via ``env -i`` (the arm CI
#: renders — the 762-char subagents-tiers delta below does not apply):
#:
#:   base (a8238e8ef, main + the §12 note)  99,879 chars = ~35,928  (9 left)
#:   head (this branch)                    101,461 chars = ~36,497
#:   delta                                   1,582 chars =   ~569
#:
#: and it decomposes exactly into the new surface:
#:
#:   +1,567 chars  ``tool_schemas`` (name + description + JSON schema of the
#:                 ONE new tool; ``real_tool_surface`` now binds the door the
#:                 way it binds ``ask_user``, because leaving it unbound would
#:                 measure a surface no user has — the queued arm is the
#:                 default since the flip)
#:   +   15 chars  the one inventory line
#:   =  1,582 chars = ~569 billed
#:
#: The raise exists because the capability needs structured params (the two
#: reasons, the per-question cells map) and it is exactly the rung-3 case the
#: ladder describes: it is paid only where the queue exists. Rung 1 is not
#: available — a mode on ``ask`` was rejected in the design because
#: ``AskParams`` is question-shaped and the op targets an EXISTING ask. The
#: wording was trimmed once before this entry was written (the description
#: carries the detection rule, the two reasons' consequences and the secret
#: refusal, and nothing else); the remaining cost is the price of the
#: capability. The ceiling is the measured head + 55, the band this file
#: keeps, so the base's 9 tokens of standing headroom are restored rather than
#: loosened. The tighten band below (1,200) is not in play.
#:
#: RAISED 36,552 -> 37,094 for the SESSIONS-REMOTE TOOL SURFACES
#: (``feat/sessions-remote-tools-b``, design ``docs/design/sessions-remote-tools.md``
#: §4): the ``network`` tool gains the pilot verbs (``send``/``steer``/``slash``
#: + ``text``), the ``send`` tool gains ``peer``, and the ``sessions`` tool gains
#: ``peer``/``scope`` with five ops now peer-capable — the mesh's three front
#: doors in one change, measured with THIS script on both trees, same machine,
#: one after the other, the CLEAN arm via ``env -i`` (the arm CI renders):
#:
#:   base (origin/main 4c492effd)   101,461 chars = ~36,497   (55 left)
#:   head (this branch)             102,969 chars = ~37,039
#:   delta                            1,508 chars =  ~542
#:
#: and it decomposes exactly into the three tools (name/description/schema, per
#: ``tool_schema_chars``, descriptions and schemas counted separately):
#:
#:   + 370 chars  ``send``     — description +187 (the §3B ``peer`` field text
#:                and the one description clause), schema +183 (the ``peer``
#:                property with its description and JSON scaffold)
#:   + 608 chars  ``network``  — description +36 (the sessions-action tail now
#:                advertises the pilot verbs), schema +572 (the four new fields'
#:                properties, their §4 descriptions and scaffold)
#:   + 530 chars  ``sessions`` — description +164 (the mesh clause plus the
#:                derived per-op summary across five ops), schema +366 (``peer``
#:                and ``scope``, the §4 strings and scaffold)
#:   = 1,508 chars = ~542 billed
#:
#: NO other block moves: instructions, tool_inventory, environment and knowledge
#: are byte-identical across the two arms. The field descriptions are the design
#: §4 strings (drafted lean there), and the guides are lazy — read on demand —
#: so none of this rides the start context until a caller pulls it. The ceiling
#: is the measured head + 55, the band this file keeps; the tighten band below
#: (1,200) is not in play.
#:
#: LOWERED 37,094 -> 34,810 for the CONTEXT DIET, slice B (system prompt trim),
#: measured with THIS script on both trees, the clean ``env -i`` arm:
#:
#:   base (origin/main e3176403b6)  102,969 chars = ~37,039
#:   head (this branch)              94,872 chars = ~34,127
#:   delta                           -8,097 chars = ~-2,912
#:
#: The whole delta is the ``instructions`` block (32,699 -> 24,602 chars):
#: ``prompts_md/system.md`` was cut ~44% (18,439 -> 10,342 rendered chars, the
#: browser arm) by moving tool mechanics that the tool descriptions,
#: ``tool://`` docs and guides already carry behind those resources, keeping
#: only the rules a model needs before it knows to look anything up
#: (pinned by ``tests/unit/test_system_prompt_invariants.py``). Tool schemas,
#: inventory and knowledge are byte-identical across the arms. Two savings in
#: the same change do NOT show here because this script feeds fixed stand-ins:
#: the repo-guidance head (8 -> 4 KiB, ``context_files.GUIDANCE_HEAD_BYTES``)
#: and the selected-skills short form (``skills.index.short_description``);
#: their real-input figures are in the PR. Headroom is ~2% (683 tokens), as
#: the context-diet brief asked, inside the 1,200 tighten band.
#:
#: Review round 1 restored rules the first trim dropped rather than moved (the
#: wake/monitor triggers, "don't notify", no lettered options, never
#: force-activate a tab, MCP before local-config discovery, the guide rule's
#: why): +360 chars = ~+129 billed, measured head 95,232 chars = ~34,256. The
#: ceiling is NOT raised for it — the 554 tokens left (~1.6%) are the headroom.
#:
#: LOWERED 34,810 -> 27,210 by the context diet, slice A (deferred tool schemas,
#: ``perf/deferred-tools``), RE-MEASURED after review rounds 1-2 and CI round 4. Measured
#: with THIS script, CLEAN arm via ``env -i``, both trees on the same machine
#: (base re-measured after slice B merged, so the delta is slice A's alone):
#:
#:   base (origin/main 5e59e0cd0)    95,232 chars = ~34,256
#:   head (this branch)              74,159 chars = ~26,676
#:   delta                          -21,073 chars = ~-7,580
#:
#: Instructions, environment and knowledge are byte-identical across the arms;
#: the delta is the tools array and one inventory line. Components, each
#: measured rather than apportioned:
#:
#:   -11,752 chars  ``tool_schemas``: 9 tools' schemas DEFERRED — still held,
#:                  callable, approval-gated and documented, but published only
#:                  once activated (``tools/deferral.py``). ``tool_schemas``
#:                  counts only what is published, as ``Session._publishable``
#:                  sends it. Measured as ``--no-defer`` minus the default arm,
#:                  net of the inventory line below.
#:   - 4,752 chars  ``tool_schemas``: optional ``anyOf: [T, null]`` +
#:                  ``default: null`` collapsed to ``T`` (95 ROOT properties on
#:                  the published surface; ``tools.registry.collapse_optional_nulls``),
#:                  measured as the published set with the collapse on against
#:                  the same set with it neutralised
#:   - 4,874 chars  ``tool_schemas``: description trims (``ask``, ``send``,
#:                  ``sessions``, ``browser``, ``todo.op``, ``project``, the
#:                  ``i`` intent) whose prose moved into ``read tool://<name>``
#:                  — the residual, so the four components sum to the delta
#:   +   305 chars  ``tool_inventory``: the one "schema on demand" line naming
#:                  the 9 deferred tools with a purpose phrase each
#:   = -21,073 chars = ~-7,580 billed
#:
#: ``--no-defer`` (the ``tools.defer: false`` kill switch) measures ~30,794 —
#: the two trims alone — the inverse canary that proves the deferral share is
#: real. The ceiling is the measured head + ~2% (534), inside the 1,200 band.
#:
#: RE-MEASURED DOWN twice, and each time the measurement changed the design.
#: Round 1 un-deferred ``network`` and the five child-only extras, each measured
#: collapsing adoption on a live model when its schema was absent. Round 2 moved
#: the collapse's "I rewrote this" flag OUT of the schema: carried in-schema it
#: put 121 keys on every published request, which cost 1,209 billed tokens here
#: (27,354 -> 26,145 across that fix) and asked strict providers to accept a
#: keyword their dialect does not list. CI round 4 RAISED it by 531 (26,145 ->
#: 26,676) on purpose: ``ask_withdraw`` was un-deferred after a live probe found
#: the model settled a queued ask 0/3 with its schema withheld and 3/3 with it
#: published (``tools/deferral.py``). The child arm did not move — a child holds
#: no ``ask_withdraw`` (no ask hook) — which is the check that the 531 is that
#: tool and nothing else. Re-derive with ``--verbose`` after any change to the
#: deferred set or to the collapse.
#:
#: RAISED 27,210 -> 27,217 by the IMAGE-GENERATION RESTORE
#: (``feat/image-generation-tool``), which merged the same day the diet landed,
#: so both ledgers here were measured ONCE on the MERGED tree — the same policy
#: the diet's own entries follow (re-measure on merge, never add two parent
#: deltas). Same machine, clean arm via ``env -i``:
#:
#:   merged head (this branch)       75,510 chars = ~27,162
#:
#: against the diet's own head (74,159 = ~26,676): +1,351 chars = ~+486 billed,
#: exactly the new tool's wire cost rendered through THIS branch's collapse —
#: 14 name + 186 description + the collapsed JSON schema + its inventory row.
#: Everything else is byte-identical to the diet's reading. The tool is
#: createIf-gated on the machine's image credentials, so a session without a
#: provider pays none of it; the guard measures the fully-capable surface
#: deliberately (``real_tool_surface`` now binds that predicate the way it
#: binds the browser and console gates). The band is normalized to the 55-token
#: band this file keeps — the diet left 48 tokens above its head, so the
#: ceiling moves by 7: measured head + 55.
#:
#: RAISED 27,217 -> 27,270 for the IMAGE-EDITING lane (``feat/imagegen-editing``,
#: #2157), stated here with the arithmetic because the guard exists to make
#: this an explicit decision. Same machine, clean arm via ``env -i``, both
#: trees measured on the merged base (``origin/main`` ca31b74574):
#:
#:   base  75,565 chars = ~27,182   (35 tokens of headroom at the old ratchet)
#:   head  75,659 chars = ~27,215   (this branch)
#:   delta      +94 chars = ~+34
#:
#: Components, each measured: the ``generate_image`` parameter schema carries
#: one new field — ``source_attachment`` (+113 chars: the serialized property
#: entry, 111, plus the 2-char ``, `` separator the serializer adds) — less
#: 19 chars saved by the reworded ``prompt``/``strength`` descriptions
#: (prompt -9, strength -10): +113 - 19 = the +94-char delta, closed exactly.
#: Everything else on the surface is byte-identical to base.
#:
#: The FIRST head measured 27,354 billed (+172): its class docstring rendered
#: VERBATIM into the schema's ``description`` and cost ~336 chars on the
#: wire, so the detail moved into comments + ``read tool://generate_image`` +
#: the guide and the docstring went back to its one line; the trim (the
#: docstring's 336 plus 50 of field-description text) took the delta from
#: +172 to +94 chars (~+34 billed).
#:
#: The residual ~+34 is the honest cost of ONE new wire parameter: a field a
#: model must see to use the feature, with the shortest description that
#: keeps it clear (docs-first — the guide carries the detail). Hiding it
#: would re-ship the silent-T2I class this lane exists to kill. The ratchet
#: moves to the file's 55-token band (measured head + 55) so the next
#: addition finds it as tight as this one did.
#:
#: RAISED 27,270 -> 28,159 for the VIDEO PHASE's harness lane
#: (``feat/video-generation``, #2191): the two new createIf tools
#: (``generate_video`` + ``video_status``) land their schemas on every session
#: whose machine carries a video credential — the same machinery
#: ``generate_image`` has. Measured with THIS script on the clean arm
#: (``env -i`` HOME + config dir, the arm CI renders), both trees on the
#: merge base (``origin/main`` 8b6ed6b854):
#:
#:   base  75,756 chars = ~27,250   (20 tokens of headroom at the old ratchet)
#:   head  78,128 chars = ~28,104   (this branch)
#:   delta     +2,372 chars = ~+854
#:
#: Components, each measured and closed exactly: +2,340 chars in
#: ``tool_schemas`` (the two new parameter schemas — ``generate_video``
#: 1,844 + ``video_status`` 490 — plus +6 from the ``jobs`` description's new
#: "(task/bash/video)" word, per round-1 review O2) and +32 chars in
#: ``tool_inventory`` (the two names on the listing lines); every other
#: counter byte-identical to base. The child arm moves by the SAME +2,372
#: chars (below), which is the check that the delta is these two tools and
#: nothing else — both are createIf-gated on the credential, not on the
#: parent session.
#:
#: The raise is the capability's honest cost, not a regression to trim: one
#: tool is what a model must SEE to start a generation at all (modes, source
#: fields, the knobs), the other is the only callable that answers a pending
#: job after a restart. Deferral was considered and rejected in the PR: it
#: needs its measured adoption bar (``tools/deferral.py``) and a brand-new
#: tool has no measurement to sit on.
#:
#: RAISED 27,270 -> 27,337 for SYS1 (the user's preferred language rides the
#: stable prefix): every real session's head now carries the model-facing
#: language sentence, so the guard must measure the shape that ships. The
#: worst case is rendered — the longest display name (``zh-CN``) through the
#: real ``preferred_language_line``, so the guard's text cannot drift from
#: production. Measured with this script on both trees, one after the other:
#:
#:   baseline (origin/main f29d9e7fbe)     75,756 chars = ~27,250 billed
#:   head (this branch)                    75,942 chars = ~27,317 billed
#:     = +186 chars = +67 billed: one sentence, parameterised by the one
#:       backend language setting — the mechanism that lets the model answer,
#:       and address the user, in their language at all.
#:
#: MERGED RE-MEASURE 2026-10-10 (the #2191 fold): the raises above land on
#: ONE tree for the first time. Measured with this script on the clean arm,
#: both trees, origin/main (9de8199ef5) vs the folded branch head:
#:
#:   base (origin/main 9de8199ef5)  75,942 chars = ~27,317 billed
#:   merged head                    78,279 chars = ~28,158 billed
#:   delta                          +2,337 chars = ~+841
#:
#: The base figure already contains the SYS1 sentence, so it is not
#: double-counted here; the delta is the video pair's contribution at its
#: merged revision (the entry above's +2,372 was measured before the
#: round-1..3 wire-text edits) and it is IDENTICAL on the child arm
#: (below) — the check that the delta is the pair and nothing else.
#: Ceiling = merged head + the 55-token band.
#:
#: RAISED 28,213 -> 28,506 for wave 4 slice A (the agent-writable `language`
#: setting): the unconditionally published `setting` tool rides both arms
#: (schema + description; its `tool://` doc adds no wire bytes). CLEAN ARM,
#: NAMED: fresh `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` per tree — a developer's
#: own config can move the config-sensitive docs (the 2026-10-10 review round
#: found exactly that: an ambient-config base measured 666 chars high and read
#: as "pre-existing overage"). Measured with this script, both trees back to
#: back:
#:
#:   base (origin/main ef4b4b19eb)  78,279 chars = ~28,158 billed
#:   head (this branch)             79,095 chars = ~28,451 billed
#:   delta                              +816 chars = ~+293
#:
#: The delta is the one tool and nothing else: the child arm moves by the SAME
#: +816 chars (reported as +294 there — the arms' totals round the same 816
#: differently), and the base figure equals the video fold's merged head
#: (28,158) byte for byte, so nothing else has moved the prefix since.
#: Ceiling = head + the 55-token band.
#: RAISED 28,506 -> 28,729 for the end-state verification principle added
#: to ``prompts_md/system.md`` ("Verify the end state before you call it
#: done."): one added bullet to the standing-instruction head, which rides the
#: cached prefix of every request on every surface. CLEAN ARM, NAMED: fresh
#: `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` per tree, both trees measured with
#: this script back to back on the same machine:
#:
#:   base (origin/main 7ec9a8e25b)  79,246 chars = ~28,506 billed
#:   head (this branch)             79,713 chars = ~28,674 billed
#:   delta                             +467 chars = ~+168
#:
#: The delta is the one bullet and nothing else: ``tool_inventory`` (785),
#: ``environment`` (81), ``knowledge`` (854) and ``tool_schemas`` (52,130)
#: are byte-identical to base on both trees, and the child arm moves by the
#: SAME +467 chars (reported as +167 there — the arms' totals round the same
#: 467 differently). The bullet exists because a read of finished evaluation
#: trajectories found agents reporting success with the requested outcome
#: unverified; the principle is general to every surface (CLI, TUI, mobile,
#: evaluation runs), and it cannot move to a guide: it must be resident
#: BEFORE the model forms a completion claim, which is exactly the moment a
#: lookup would not happen. Ceiling = head + the 55-token band.
#: RAISED 28,729 -> 28,818 for the multi-source edit lane (the
#: `generate_image` tool's `source_image_paths`/`source_attachments` list
#: fields — the two added schema properties; they first measured +279
#: chars, and the redundant "(multi-source)" description suffixes were
#: trimmed (-30) under review round 1, leaving +249). CLEAN ARM, NAMED:
#: fresh `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` per tree, both trees measured
#: with this script back to back on the same machine:
#:
#:   base (origin/main e8b728f8c6)  79,713 chars = ~28,674 billed
#:   head (this branch)             79,962 chars = ~28,763 billed
#:   delta                             +249 chars = ~+89
#:
#: The delta is the two fields and nothing else: ``tool_schemas`` (52,379
#: vs 52,130) is the only component that moves, the child arm moves by the
#: SAME +249 chars (reported as +90 there — the arms' totals round the
#: shared 249 differently), and ``instructions`` (25,863),
#: ``tool_inventory`` (785), ``environment`` (81) and ``knowledge`` (854)
#: are byte-identical to base on both arms. Ceiling = head + the 55-token
#: band.
#: Per-run subagent model selectors (base 23613202e3): measured with isolated
#: HOME/config, top 79,962 -> 80,508 chars (+546), ~28,763 -> ~28,960
#: billed. The child arm moves by the identical +546 chars (71,745 -> 72,291,
#: ~25,808 -> ~26,004). The additive task/batch/hub model field is required
#: to express inherit without erasing a role; no profiles/defaults were changed.
#: Preserve the existing 55-token headroom band on both measured arms.
#: A 41st tool, `bridge` (H5d, Team Bridge). Measured with the repo's own
#: script, isolated HOME/config, on three trees: the merge base `703aa047`,
#: this head, and the shared checkout (`2115d3e194`, for unrelated drift).
#:
#:   component             base 703aa047     head     delta
#:   instructions                25,863   25,863         0
#:   tool_inventory                 785      794        +9
#:   environment                     81       81         0
#:   knowledge                      854      854         0
#:   tool_schemas                53,537   54,766    +1,229
#:   ------------------------------------------------------------
#:   top total (chars)           81,120   82,358    +1,238  = ~+445 billed
#:   child total (chars)         72,903   74,141    +1,238  = ~+445 billed
#:   (child tool_schemas        45,614   46,843    +1,229)
#:
#: The inventory delta is not a reworded line: the surface gains exactly one
#: name (`bridge` — 40 tools to 41, verified by listing both), so the line
#: grows by the name plus its separator.
#:
#: Two facts to read separately:
#:
#: - THE MERGE BASE ALREADY EXCEEDED THE OLD CEILING. 81,120 chars is ~29,180
#:   billed against 29,015: +165 on the top arm, +165 on the child. No ceiling
#:   covering this branch can sit at the old value whatever this branch does —
#:   the ratchet was already stale, behind a path-gated job that does not run
#:   when a schema changes.
#: - UNRELATED DRIFT, reproduced: the shared checkout reads tool_schemas
#:   53,669 (base +132) and ~29,227 billed (+47 over the base) — main moved
#:   after the merge base without the gate re-running.
#:
#: Our share of the CI-reported overage is +445 billed, all of it the bridge
#: tool's published schema (the description was trimmed twice; what remains is
#: the schema of the ops themselves, which cannot shrink without unpublishing
#: one). The ceiling is head + the existing 55-token band on both arms —
#: 29,680 / 26,724 — and that raise is NOT self-approved: it is stated here
#: and in the PR thread for a reviewer to verify before it settles.
# RAISED 29,680 -> 30,000 for the ``harness_import`` agent tool (LOH-14 agent
# slice), after trimming it first: removing every per-field description and the
# action list from its wire description took tool_schemas 56,700 -> 56,036
# chars (-664 chars, -239 billed tokens). The remainder is the published schema
# and its one inventory line, which cannot shrink further without removing the
# capability. Measured on THIS head with this script:
#   top    83,198 chars = ~29,927 billed  (73 tokens under 30,000)
#   child  75,057 chars = ~26,999 billed  (101 tokens under 27,100)
# The tool is a setup-time capability, not an every-session one, but it is in
# the default surface today; the raise is the explicit decision the guard exists
# to force, not an unrelated budget increase.
# RAISED 30,000 -> 31,000 for the sound-generation wave's audio pair
# (``generate_voiceover`` + ``generate_sound_effect``, 2026-10-17). Both tools'
# schemas were trimmed as far as the frozen contract allows (terse per-field
# descriptions, one-sentence wire descriptions) and the remainder is the
# published field list itself, which cannot shrink without removing a field the
# contract fixes. Attribution measured on THIS head with this script:
#   without the pair  29,707 billed  (293 tokens under the old 30,000)
#   with the pair     30,616 billed  (+909: +2,482 tool_schemas chars = ~893
#                                     billed, plus the two inventory lines)
#   child arm         27,688 billed  (+588 over the old 27,100)
# Deferring the pair instead was refused on the image lane's measured
# adoption regression for a brand-new capability (``image_tool``'s D3 note:
# deferring a new tool whose description carries its "when to use it" text
# regressed adoption on #2066). The raise is stated here AND in the PR thread
# for a reviewer to verify before it settles — it is the explicit decision this
# guard exists to force, not an unrelated budget increase.
# RAISED 31,000 -> 31,025 for the memory move hint (#2616), stated with the
# arithmetic because this guard exists to make that an explicit decision, not an
# unrelated budget increase. Measured by THIS script on an isolated HOME at each
# tree, same machine:
#   673cb4de62  true first parent of #2616's merge  86,162 chars = ~30,994 billed  PASS, 6 headroom
#   20f23b8b73  #2616 merged                        86,204 chars = ~31,009 billed  FAIL by 9
#   = +42 chars of ``tool_schemas``, and all of it is ONE clause in
#     ``build_memory_tool``'s description: 'move with edit
#     new_text={"move":"scope"}' (+15 billed). #2616's review round 1 raised as a
#     NIT that the memory tool description did not name the move payload; its
#     remediation chose to address that nit, and the clause landed on a tree
#     with 6 tokens of headroom.
# Trimming was MEASURED on the head tree with this script, not assumed:
#   bare 'move with edit'                    86,178 chars = ~30,999  PASS by 1 at 31,000
#   payload-keeping 'move {"move":"scope"}'  86,185 chars = ~31,002  FAIL by 2 at 31,000
# So a trim that clears the old ceiling does exist — but only by dropping the
# payload shape the nit asked for, and every variant that keeps it fails. The
# raise is chosen to keep the payload clue AND leave margin, not to fit by
# deleting the thing the nit was about.
# RAISED to 31,466 for the Team Bridge agent channel tool (#2665), measured with
# THIS script on the merged base a68b18bdb5 in an isolated HOME:
#   top arm:   87,323 chars = ~31,411 billed  -> 31,411 + 55-token band = 31,466
#   child arm: 79,182 chars = ~28,483 billed  -> 28,483 + 55-token band = 28,538
# The channel tool adds the four ops (channels/create/post/history), eleven
# parameters and the nested member model; every field is load-bearing (name,
# topic and visibility are request facts, members is the invite shape). The
# prose was trimmed before this raise; what remains is structural.
# RAISED 31,466 -> 31,982 for the V2-4 agent ``tags`` tool, measured with THIS
# script on this branch in an isolated HOME (final wording, after the primary
# fallback description landed):
#   top arm:   88,825 chars = ~31,951 billed  -> 31,982 leaves 31 headroom
#   child arm: 80,684 chars = ~29,023 billed  -> 29,053 leaves 30 headroom
# The delta is the tool itself: +1,188 chars of ``tool_schemas`` (its
# description and flat params schema), +307 chars of instructions (the
# system-prompt guidance line the design requires, reworded in review rounds 1
# and 2 to qualify the receipt claim to ``assign``) and +7 chars of inventory.
# The schema and the prompt line are both load-bearing — the design's own
# prompt rule is "never spell a category the store does not contain" — and the
# reference vocabulary was kept OFF the wire in ``read tool://tags``, whose
# ledger entry in tests/unit/tools/test_tool_docs.py moves with this one.
# RAISED 31,982 -> 32,104 for the ask RELAY field (#2520). This raise COMPOSES
# onto the two above rather than replacing them: the branch's pre-rebase value
# (31,025 -> 31,160) was measured against a tree that carried neither #2665's
# channel tool nor the V2-4 tags tool, so merging it as-is would have silently
# LOWERED the ceiling under two shipped surfaces. Measured on the merged tree,
# one run per tree on the same host with an isolated HOME:
#   base (origin/main 2c55ac459b)  88,825 chars = ~31,951 billed
#   head (this branch)             89,094 chars = ~32,049 billed
#   delta                             +269 chars = ~+97
# The whole delta is the ``relay`` property on ``AskParams`` plus its one clause
# in the queued description. The child arm does not carry ``ask`` and does not
# move (80,684 chars = ~29,023 vs the 29,053 ceiling), so
# BUDGET_CHILD_BILLED_TOKENS stays where main put it.
#
# COMPOSED WITH LOH-9 S2 (rebased onto 4bad041283), measured by THIS script on
# both trees in an isolated HOME — NOT by taking either branch's number:
#   base (origin/main 4bad041283, #2520)  89,094 chars = ~32,048 billed
#   head (this branch)                    89,290 chars = ~32,119 billed
#   delta                                     +196 chars = ~+71
# The delta is one ``scope`` enum on ``AgentParams`` and one on ``TeamParams``,
# the ``delete`` op literal on the agent tool, and a one-line description on
# each scope field (their descriptions were trimmed first). The ladder itself
# stays OFF the wire in ``tool://agent``/``tool://team``, whose ledger entries
# moved with this change.
# Ceiling = measured head + the 55-token band (32,119 + 55), so the next
# field-size edit is a deliberate raise rather than silent creep.
#
# RAISED 32,174 -> 32,396 for the redaction policy's public-copy reader
# (PR #2796): the new tool is 121 doc tokens plus its schema and adds 167 billed
# tokens to this arm, measured by THIS script on the branch with an isolated
# HOME. The tool is a security control (the sanctioned route for public copy),
# not incidental prose; ceiling = measured head 32,341 + the 55 band.
#
# COMPOSED WITH #2796 ON THE MOVED BASE (media artifact naming, PR #2792),
# measured by THIS script on both trees with an isolated HOME, one run each —
# NOT by adding this branch's old-base number to main's:
#   base (origin/main 8426d1cac1, #2796)  89,908 chars = ~32,341 billed
#   head (this branch)                    90,905 chars = ~32,700 billed
#   delta                                     +997 chars = ~+359
# The delta is the four ``filename`` properties (each carries its ~150-char
# teaching description plus the JSON anyOf/maxLength around it), the
# ``SoundEffectRequest.name`` description (now that item's filename stem) and
# the ``sounds`` description's ``filename`` exclusivity clause. Those
# descriptions are the discovery surface for a capability whose point is that
# the agent names the asset; the sanitising and collision rules ride the three
# generation guides, read on demand. Ceiling = measured head + the 55 band.
#
# COMPOSED WITH LOH-9 S4 promotion on this moved base: promotion adds one
# `promote` enum value plus `as_name`/`confirm_replace` on BOTH authoring tools.
# Measured by THIS script on the rebased head with an isolated HOME:
#   base  32,755 ceiling (media naming already counted on this arm)
#   head  91,231 chars = ~32,817 billed  +62
# Ceiling = measured head + the 55 band. Do not pick either side of a budget
# conflict: re-measure the composed tree.
#
# RAISED 32,872 -> 33,364 for STALE ASKS (PR-1, design ask-nonblocking §16 /
# SA-D1): the guidance that tells a model WHEN to settle an ask it no longer
# needs. Measured by THIS script on both trees with an isolated HOME, one run
# each — tree 1 is origin/main eb21bbb5d, tree 2 this branch's review-round 1
# head (the wording grew once in the review: the rule now names its set,
# excludes age and a change of task order, and the description reconciles
# "latest or earlier" with VERBATIM):
#   base (origin/main eb21bbb5d)  91,310 chars = ~32,845 billed  PASS, 27 headroom
#   head (this branch)            92,599 chars = ~33,309 billed
#   delta                         +1,289 chars = ~+464
# The whole delta is deliberate copy, in two blocks:
#   +626 chars of instructions — ``system.md``'s queued-ask paragraph gains the
#     settle rule (settle what the user's messages answered or made irrelevant
#     in the same turn, an unaddressed ask stays open however old it is, an
#     answer is never a `moot`, a 24 h timeout only for a day-long decision).
#     It is the one place a model can learn WHEN, and the investigation measured
#     16 of 40 asks left open until prompted without it;
#   +663 chars of ``tool_schemas`` — one clause on the ``ask`` queued
#     description, the moot cases, the empty-list recipe, the all-empty refusal
#     and the earlier-or-latest rule on ``ask_withdraw``'s description, and the
#     ids rule on ``AskWithdrawParams.answers``. The child arm does not carry
#     either tool.
# Both blocks are model-facing English (no i18n keys). The copy was trimmed
# twice, once on each round; what remains states facts a model cannot infer, and
# the alternative — keeping the floor by not saying them — is the defect the PR
# exists to fix.
# Ceiling = measured head + the 55 band (33,309 + 55).
#
# RAISED 33,364 -> 33,466 for the TEAM BRIDGE surface after #2757 and #2760
# merged: the threads lane added the ``channel_thread`` op and the
# ``thread_root_id`` parameter, and the reactions lane added the ``react`` op
# and its ``emoji``/``remove`` parameters. Measured by THIS script on one tree
# with an isolated HOME, before and after trimming the tool's prose:
#   before trim   92,912 chars = ~33,422 billed   FAIL, 58 over
#   after trim    92,814 chars = ~33,386 billed   FAIL, 22 over
#   budget        33,466 (measured head + 80 headroom)
# The trim is real and model-facing: the bridge description loses the
# repeated op roster (Pydantic publishes it in the schema anyway), the
# thread-reply clause moves entirely to ``thread_root_id``'s own field note in
# ``read tool://bridge``, and ``ChannelMemberParams``' docstring keeps its rule
# in one shorter sentence. Ops are NOT removed: all five channel ops stay
# discoverable with their parameter names.
# What remains is unavoidably the two ops' schema surface; the 80-token margin
# is deliberate — the brief's point was that landing at the measured number
# would let the next small op tip it again.
BUDGET_BILLED_TOKENS = 33_466

#: The SUBAGENT ceiling (``--kind child``), in billed tokens. Same ratchet rules
#: as ``BUDGET_BILLED_TOKENS`` above. It is a separate number because a child's
#: INVENTORY is smaller (``CHILD_NEVER_HOLDS``: no ``ask``/``ask_withdraw``/
#: ``wake``/``monitor``/``patience``) and its prompt carries the ``hub``
#: channel, not because the deferred set differs — it does not, see
#: ``tools/deferral.py`` for the measurement that dropped the child-only extras.
#: ~93% of the 30-day ledger's sessions are children, so this is the number most
#: requests actually pay. Measured with THIS script's child arm on both trees
#: (base re-derived over an exported ``origin/main`` 5e59e0cd0 with the same
#: tool filter and channel):
#:
#:   base (origin/main 5e59e0cd0)    83,243 chars = ~29,944
#:   head (this branch)              65,974 chars = ~23,732
#:   delta                          -17,269 chars = ~-6,212
#:
#: ``--no-defer`` measures ~27,334. The ceiling is the measured head + ~2% (475).
#:
#: RAISED (with the top arm, same day, same merge) 24,207 -> 24,273 by the
#: IMAGE-GENERATION RESTORE: the child arm ALSO carries ``generate_image`` (it is
#: createIf-gated on the machine, not on the parent), measured on the merged
#: tree —
#:
#:   merged child head (this branch) 67,325 chars = ~24,218
#:
#: vs the diet's child head (65,974 = ~23,732): the same +1,351 chars = ~+486
#: billed the top arm carries, which is the check that the delta is that one
#: tool and nothing else. Band normalized to the 55-token band; the diet's
#: child head had 475 of headroom, so the ceiling moves by 66: head + 55.
#:
#: RAISED (with the top arm, same day, same trees) 24,273 -> 24,315: the
#: child arm carries ``generate_image`` too and moves by the SAME +94 chars
#: = ~+34 (the check that the delta is that one field and nothing else):
#:
#:   base  67,348 chars = ~24,226   (47 tokens of headroom at the old ratchet)
#:   head  67,442 chars = ~24,260
#:   delta      +94 chars = ~+34
#:
#: Band normalized to the 55-token band, as the top arm's entry states.
#: Band normalized to the 55-token band, as the top arm's entry states.
#:
#: RAISED (with the top arm, same day, same trees) 24,315 -> 25,203: the
#: child arm carries the video pair too (createIf on the machine, not the
#: parent) and moves by the SAME +2,372 chars = ~+853:
#:
#:   base  67,539 chars = ~24,295   (20 tokens of headroom at the old ratchet)
#:   head  69,911 chars = ~25,148
#:   delta     +2,372 chars = ~+853
#:
#: RAISED 24,315 -> 24,382 for SYS1: the same sentence the top arm carries,
#: same +186 chars = +67 billed (only the tool inventory differs between the
#: arms). Measured on both trees with ``--kind child``:
#:
#:   baseline (origin/main f29d9e7fbe)     67,539 chars = ~24,295 billed
#:   head (this branch)                    67,725 chars = ~24,362 billed
#:
#: The raise is exactly the feature's cost, as the top arm's entry states.
#:
#: MERGED RE-MEASURE (with the top arm, same fold, same trees):
#:
#:   base (origin/main 9de8199ef5)  67,725 chars = ~24,362 billed
#:   merged head                    70,062 chars = ~25,202 billed
#:   delta                          +2,337 chars = ~+840
#:
#: The same +2,337 the top arm moves by. Ceiling = merged head + the
#: 55-token band.
#:
#: RAISED 25,257 -> 25,551 (with the top arm, same trees, same clean-arm
#: environment): the child arm carries `setting` too and moves by the SAME
#: +816 chars — reported as +294 here because this arm's total rounds the
#: shared 816 chars differently (the check that the delta is that one tool
#: and nothing else):
#:
#:   base (origin/main ef4b4b19eb)  70,062 chars = ~25,202 billed
#:   head (this branch)             70,878 chars = ~25,496 billed
#:   delta                              +816 chars = ~+294
#:
#: Ceiling = head + the 55-token band.
#: RAISED 25,551 -> 25,773 (with the top arm, same tree, same clean-arm
#: environment): the child arm carries the same standing-instruction head
#: and moves by the SAME +467 chars — reported as +167 here because this
#: arm's total rounds the shared 467 chars differently (the check that the
#: delta is that one bullet and nothing else):
#:
#:   base (origin/main 7ec9a8e25b)  71,029 chars = ~25,550 billed
#:   head (this branch)             71,496 chars = ~25,718 billed
#:   delta                             +467 chars = ~+167
#:
#: Ceiling = head + the 55-token band.
#: RAISED 25,773 -> 25,863 (with the top arm, same trees, same clean-arm
#: environment): the child arm carries `generate_image` too and moves by
#: the SAME +249 chars — reported as +90 here because this arm's total
#: rounds the shared 249 chars differently (the check that the delta is
#: the two fields and nothing else):
#:
#:   base (origin/main e8b728f8c6)  71,496 chars = ~25,718 billed
#:   head (this branch)             71,745 chars = ~25,808 billed
#:   delta                             +249 chars = ~+90
#:
#: Ceiling = head + the 55-token band.
# RAISED 27,100 -> 28,100 with ``BUDGET_BILLED_TOKENS`` above, for the same
# audio pair and on the same measurement pass (child arm measured 27,688 billed
# on this head). The two ceilings move together because the pair is in every
# session's surface, child included; the difference between them is unchanged.
# RAISED 28,100 -> 28,466 with the top arm's #2665 channel-tool raise, on the
# same clean-arm pass (env -i, isolated HOME, same machine):
#   base (origin/main 8714b008c1)  78,063 chars = ~28,080 billed
#   head (this branch)             78,982 chars = ~28,411 billed
#   = +919 chars of ``tool_schemas`` = ~+331 billed, the SAME delta as the top
#     arm: the channel tool is in every session's surface, children included,
#     and no child counter moved. Ceiling = measured head + the 55-token band.
# COMPOSED WITH LOH-9 S2 (rebased onto 4bad041283), measured on both trees with
# THIS script's child arm, isolated HOME:
#   base (origin/main 4bad041283, #2520)  80,684 chars = ~29,023 billed
#   head (this branch)                    80,880 chars = ~29,094 billed
#   delta                                     +196 chars = ~+71
# The same +196 chars as the top arm, which is the check that the delta is the
# two ``scope`` enums and their one-line descriptions and nothing else. Ceiling
# = measured head + the 55-token band (29,094 + 55), so 29,149.
#
# RAISED 29,149 -> 29,371 with the top arm's public-copy reader (same PR, same
# delta): the child surface carries the tool too, measured 29,316 billed on this
# head; ceiling = 29,316 + the 55 band.
#
# COMPOSED WITH #2796 ON THE MOVED BASE (media artifact naming, same pass as
# the top arm): the child surface carries all four media tools, so it grows by
# the SAME +997 chars:
#   base  81,498 chars = ~29,316 billed
#   head  82,495 chars = ~29,674 billed  -> ceiling = head + 55.
#
# COMPOSED WITH LOH-9 S4 promotion on the same moved base: the SAME +63 billed
# tokens as the top arm (both authoring tools ride every surface, children
# included), measured 82,821 chars = ~29,792. Ceiling = 29,792 + the 55 band.
#
# RAISED (with the top arm, same day, same trees) 29,847 -> 30,100 for STALE
# ASKS (PR-1, design ask-nonblocking §16 / SA-D1): the child arm carries neither
# ``ask`` nor ``ask_withdraw``, so ``tool_schemas`` does NOT move and every char
# of this arm's delta is the shared ``system.md`` paragraph:
#   base (origin/main eb21bbb5d)  82,900 chars = ~29,820 billed  PASS, 27 headroom
#   head (this branch)            83,526 chars = ~30,045 billed
#   delta                           +626 chars = ~+225
# The +626 is the same +626 the top arm's instructions block moves by — the
# check that the child delta is that one paragraph and nothing else.
# KNOWN LIMITATION, measured here rather than assumed (review round 1 m4): this
# paragraph renders for EVERY queued-arm session, and a subagent
# (``CHILD_NEVER_HOLDS``) holds no ``ask_withdraw``, so a child pays these 626
# chars for an instruction it cannot follow. That is the existing
# ``ask_queued``/``ask_inline`` limit — gating the paragraph on tool presence
# needs a third state in that flag pair, which ``prompts_api`` refuses today —
# so it is recorded as ``deferred — needs a third flag-pair state`` (SA-D2) and
# not solved by a new flag here.
# Ceiling = measured head + the 55 band (30,045 + 55).
# RAISED 30,100 -> 30,203 with the top arm above, for the same Team Bridge
# surface (#2757 threads + #2760 reactions): the child surface publishes the
# bridge schema too, so it carries the same two ops. Measured on this tree with
# the same trim applied:
#   child head  83,741 chars = ~30,123 billed
#   child budget 30,203 (measured head + 80 headroom, same margin as the top arm)
BUDGET_CHILD_BILLED_TOKENS = 30_203

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
#: SYS1: every real session's head now carries the user's preferred-language
#: line, and the guard measures the WORST case — the longest display name
#: (``zh-CN``, 15 chars above ``en``). Rendered through the real function so
#: the guard cannot drift from the shipped sentence.
_SAMPLE_LANGUAGE_LINE = preferred_language_line("zh-CN")


def tool_schema_chars(tools: list[AgentTool]) -> int:
    """Characters the provider tools array costs on every request.

    Mirrors what ``Session.context_breakdown`` counts: name, description and
    the JSON-serialized parameter schema, per tool.
    """
    return sum(
        len(tool.name) + len(tool.description or "") + len(json.dumps(tool.parameters or {}))
        for tool in tools
    )


#: Tools a SUBAGENT never holds, whatever its role: ``ask``/``ask_withdraw``
#: need a front end's ask hook (a child's route to the operator is ``hub``),
#: ``wake``/``monitor``/``patience`` are pruned or gated off because a
#: child ends after one prompt (``harness/subagent.py``, the prune after
#: construction), and ``no_reply``'s door is ``None`` for every child — a
#: parent's ``wait`` reads the child's final text AS the report
#: (docs/design/quiet-turns.md §4) — so the constructor's capability merge
#: never mounts it and there is nothing for the prune to undo. The child arm
#: measures the unrestricted child — the widest surface a subagent can carry
#: — so it is the full surface minus these.
CHILD_NEVER_HOLDS = frozenset({"ask", "ask_withdraw", "wake", "monitor", "patience", "no_reply"})


def measure_start_context(
    *,
    user_instructions: str = _SAMPLE_USER_INSTRUCTIONS,
    repo_guidance: str = _SAMPLE_REPO_GUIDANCE,
    inflate_schemas: int = 0,
    kind: str = "top",
    defer: bool = True,
) -> dict[str, Any]:
    """Character cost of everything a fresh session puts on the wire.

    ``inflate_schemas`` pads the TOOL SCHEMAS specifically. The fail-proof
    lever has to reach the largest component — the schemas are ~56% of the
    total and are what this PR is about — not only the operator text, or
    "prove it can fail" demonstrates the guard over the wrong half.
    """
    tools = build_real_tools(str(REPO))
    if kind == "child":
        tools = [tool for tool in tools if tool.name not in CHILD_NEVER_HOLDS]
    # What a session withholds from the wire (``tools/deferral.py``). ONE set
    # for both kinds — the module records the measurement that dropped the
    # child-only extras. The inventory block lists these on its "schema on
    # demand" line instead, and ``tool_schemas`` below counts only what is
    # published — exactly what ``Session._publishable`` sends. ``defer=False``
    # is the inverse canary: the pre-deferral surface, every schema published.
    held = {tool.name for tool in tools}
    deferred = (deferred_tool_names() & held) if defer else frozenset()
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
        language_line=_SAMPLE_LANGUAGE_LINE,
        # THE WORST CASE, AND IT MUST BE STATED: ``interactive`` defaults to
        # ``None``, which renders NO ``<interactivity>`` block at all (a host with
        # no runtime probe — an ``exec`` run, a scheduled run, a plain CLI). The
        # guard is here to hold the ceiling for the host that DOES carry the
        # block, so it renders the attached session's shape explicitly:
        # ``interactive=True`` and the channel whose body is the longest of the
        # three (``CHANNEL_ASK``, the TUI/desktop/app case — see
        # ``prompts_api.build_system_blocks``).
        interactive=True,
        # A child's channel to the operator is ``hub`` (its parent), which is
        # what ``harness/subagent.py`` states for every child it builds.
        channel=CHANNEL_HUB if kind == "child" else CHANNEL_ASK,
        deferred_tools=deferred,
    )
    published = [tool for tool in tools if tool.name not in deferred]
    parts: dict[str, Any] = {
        "instructions": len(blocks[0]),
        "tool_inventory": len(blocks[1]),
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "tool_schemas": tool_schema_chars(published),
    }
    parts["TOTAL"] = sum(parts.values())
    parts["n_tools"] = len(tools)
    parts["n_published"] = len(published)
    # Names, so a short surface can say WHICH tool a host-dependent gate ate.
    parts["names"] = [t.name for t in tools]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="ceiling in billed tokens (default: the enforced ratchet for --kind)",
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
    parser.add_argument(
        "--kind",
        choices=("top", "child"),
        default="top",
        help=(
            "which session to measure: a top-level session (default) or an "
            "unrestricted subagent, which has a smaller inventory and its own "
            "budget (BUDGET_CHILD_BILLED_TOKENS)"
        ),
    )
    parser.add_argument(
        "--no-defer",
        action="store_true",
        help=(
            "measure with every schema published (``tools.defer: false``) — the "
            "inverse canary: it must bring the pre-deferral number back"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.budget is None:
        args.budget = BUDGET_CHILD_BILLED_TOKENS if args.kind == "child" else BUDGET_BILLED_TOKENS

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
        kind=args.kind,
        defer=not args.no_defer,
    )
    total_chars = parts["TOTAL"]
    billed = round(total_chars / CHARS_PER_BILLED_TOKEN)

    # The surface is stated ALWAYS, not only under --verbose: a silent drop
    # from the full tool set is exactly how this guard would go back to
    # measuring a machine no user has. See real_tool_surface._forced_browser_backend.
    n_tools = int(parts["n_tools"])
    parts_names = list(parts["names"])
    expected_tools = len(DEFAULT_TOOL_NAMES) - (
        len(CHILD_NEVER_HOLDS) if args.kind == "child" else 0
    )
    print(
        f"tool surface ({args.kind}): {n_tools}/{expected_tools} tools, "
        f"{parts['n_published']} schemas published"
    )
    if n_tools != expected_tools:
        # Name the tools, not just the count: the whole point of this check is
        # that a host-dependent gate dropped something, and "which one" is the
        # first question anyone reading a CI log will ask.
        expected_names = set(DEFAULT_TOOL_NAMES) - (
            CHILD_NEVER_HOLDS if args.kind == "child" else set()
        )
        missing = sorted(expected_names - set(parts_names))
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
    # The canary measures a deliberate fiction too (deferral off), so it is
    # exempt from the tighten band exactly as ``--inflate`` is.
    inflating = bool(args.inflate or args.inflate_schemas or args.no_defer)
    if not inflating and headroom > TIGHTEN_WHEN_HEADROOM_EXCEEDS:
        constant = "BUDGET_CHILD_BILLED_TOKENS" if args.kind == "child" else "BUDGET_BILLED_TOKENS"
        print(
            f"FAIL: {headroom:,} billed tokens of headroom exceeds the "
            f"{TIGHTEN_WHEN_HEADROOM_EXCEEDS:,}-token slack band.\n"
            "      The context got smaller — good. Tighten the ratchet so the saving "
            "is defended:\n"
            f"      set {constant} = {billed + TIGHTEN_WHEN_HEADROOM_EXCEEDS // 2:,} "
            "in scripts/bench_context_budget.py."
        )
        return 1
    print(f"PASS ({headroom:,} billed tokens of headroom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

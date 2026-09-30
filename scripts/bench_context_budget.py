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
from local_operator.prompts_api import CHANNEL_ASK, build_system_blocks  # noqa: E402
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
BUDGET_BILLED_TOKENS = 37_756
#:
#: UPDATED (NO RAISE), 2026-09-30 — issue #1815: the ``project`` tool's
#: ``description`` field text now ends "(<= 2000 chars)." where it said
#: "(<= 240 chars)."; the schema's markdown promise is the contract, the old
#: cap refused the prose it advertised, and the fix raises DESCRIPTION_MAX to
#: 2000 with a remedy-bearing refusal. Neither the guide nor the refusal text
#: rides the prefix, so the field rename is the whole measured delta.
#: Measured with THIS script, same machine, one tree after the other:
#:
#:   base (origin/main 302a061e5)     103,960 chars = ~37,396 billed
#:   head (this branch)               103,961 chars = ~37,396 billed
#:     = +1 char = +0.4 billed — the 240 -> 2000 rename, nothing else.
#:
#: NO RAISE: a one-character edit cannot be why the permanent per-call ceiling
#: moves, and it does not breach the binding reading — by the recorded
#: ~762-char platform gap that lives in ``tool_schemas``, CI's head is ~37,122
#: against the 37,164 ceiling (~42 billed of headroom). The local reading
#: sitting above the ceiling by the SAME 232 billed on both sides is the
#: pre-existing recorded gap, not this change.

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
        # THE WORST CASE, AND IT MUST BE STATED: ``interactive`` defaults to
        # ``None``, which renders NO ``<interactivity>`` block at all (a host with
        # no runtime probe — an ``exec`` run, a scheduled run, a plain CLI). The
        # guard is here to hold the ceiling for the host that DOES carry the
        # block, so it renders the attached session's shape explicitly:
        # ``interactive=True`` and the channel whose body is the longest of the
        # three (``CHANNEL_ASK``, the TUI/desktop/app case — see
        # ``prompts_api.build_system_blocks``).
        interactive=True,
        channel=CHANNEL_ASK,
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

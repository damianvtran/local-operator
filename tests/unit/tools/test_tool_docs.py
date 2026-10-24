"""The ``tool://`` on-demand tool reference: renderer, resolver and drift tests.

``tool://`` is the reference half of the tool surface: schemas stay lean and
ride every request, while the detail a reader needs ONCE — per-op accepted
fields, parameter semantics — renders only when ``read tool://<name>`` asks
for it. That trade holds only if the on-demand copy cannot rot, so these tests
pin the properties that make it trustworthy:

* COMPLETENESS — every property name and enum literal on the default surface
  reaches its doc. Names, types and enum literals are NEVER truncated; that is
  the point of the mechanism, and a drop here is a wrong answer to the exact
  question the reader came with.
* DETERMINISM — a render is a pure function of the tool: two renders are
  byte-equal, including across a JSON round-trip of the schema (the shape a
  persisted/reloaded schema has).
* RESOLVER CONTRACT — a hit serves the render bytes, bare ``tool://`` lists,
  an unknown name names the set (hidden tools excluded from listings but still
  served on a direct hit), and every non-``tool://`` URL returns ``None`` so
  the chained guide/skill/mcp walker is preserved.
* NEVER RAISES — malformed schemas and a broken inventory still produce text,
  because a reader has no retry path for "renderer crashed".
* BUDGET — each doc stays under the 8 KiB soft cap (the runtime hard bound is
  ``read``'s 16 KiB internal-document shaping) and the :data:`MEASURED_TOKENS`
  ledger is current: a diff here is a deliberate, visible cost change to the
  reference surface, not an accident to wave through.

Plus integration tests against a real ``Session``, because the wiring — not
the renderer — is what a subagent inherits: ``Session.__init__`` chains the
``tool://`` resolver ahead of the knowledge resolver over the session's LIVE
inventory, and ``read`` reaches it through the turn's own ``ToolContext``.
One of them is the sessions pilot's byte parity — ``read tool://sessions``
serves exactly the ``help`` op's payload, and NOT the generic schema render —
which is the drift pin on the registration that lifts it (PR #1862/#1863's
missing seam).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness.types import (
    AgentTool,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.tools import builtin
from local_operator.tools.registry import DEFAULT_TOOL_NAMES, create_tools
from local_operator.tools.tool_docs import (
    SPECIAL_RENDERERS,
    TOOL_NOTES,
    ToolDocNotes,
    ToolDocOp,
    chain_tool_docs,
    is_tool_docs_link,
    make_tool_doc_resolver,
    render_tool_doc,
    unwrap_tool_docs,
)
from scripts.real_tool_surface import build_real_tools
from tests.unit.session.test_session import ScriptedStream, make_session
from tests.unit.tools.test_effort_tier_schema import THREE, write_tiers

#: Soft cap for ONE rendered doc, in characters. Enforced by TEST, never at
#: render time — names/types/enums must not be truncated to fit, so an
#: oversized doc is fixed by authoring less prose, not by clipping.
DOC_SOFT_CAP_CHARS = 8 * 1024

#: The nesting level of ``properties`` containers the renderer walks below the
#: root (design note §2). Spelled here rather than imported from the module so
#: this walker is a check ON the renderer, not a mirror of its internals —
#: :func:`test_default_surface_fits_the_documented_depth_cap` fails if the live
#: surface ever outgrows it.
_WALK_DEPTH_CAP = 2

#: Token ledger of the rendered docs, ``count_text_tokens`` under cl100k_base
#: (tiktoken is a BASE dependency, so the ruler is identical on every install).
#: When this drifts the test prints the new table; paste it back only after
#: deciding the change is worth its cost — the whole mechanism exists to make
#: reference detail opt-in, and the ledger is what keeps it honest.
#:
#: CANONICAL ARM: every doc here is rendered against an ISOLATED, empty config
#: (see :func:`hermetic_config`) — the arm CI runs. ``agent`` and ``task`` are
#: the two docs a tiers-configured ``values.subagents`` changes (they advertise
#: ``subagents.model_choice`` / ``subagents.models``); without the isolation
#: this table could only ever be green on one of the two machines. The OTHER
#: arm is pinned by :func:`test_the_tier_configured_arm_is_pinned_too`.
#:
#: UPDATE (folded head dfb109697, teams-label lane): ``team`` moved 412 -> 540
#: (+128) for the ``label``/``aliases`` create/update fields this branch adds
#: to the team schema — the same addition the context-budget entry records as
#: +670 schema chars on its own ruler. Every other entry was unmoved across
#: the fold (measured isolated, this file's own fixture).
#:
#: RE-MEASURED 2026-10-01 by ``fix/sessions-resume-0930-9d2e`` (PR #1863,
#: folded after #1862): ``sessions`` 718 -> 738 and ``eval`` 333 -> 348 are
#: the ONLY entries that moved — the sessions per-op advertisement (plus the
#: ``help`` op's enum value) and the eval failure notice — and they were
#: re-measured through this file's own renderer, not pasted from another
#: machine. Every other entry is byte-identical to #1862's table.
#:
#: RE-MEASURED 2026-10-01 by the slimming wave (audit items 2+3): the EIGHT
#: worked tools moved — ``project`` 968 -> 1077, ``browser`` 1050 -> 1069,
#: ``console`` 1006 -> 1141, ``ask`` 837 -> 948, ``hub`` 638 -> 675,
#: ``agent`` 687 -> 826, ``network`` 697 -> 742, ``task`` 422 -> 505 — and
#: the item-7 failure-semantics notes added ``bash`` 271 -> 313, ``eval``
#: 348 -> 379, ``grep`` 237 -> 346, ``jobs`` 339 -> 365, ``monitor`` 369 ->
#: 420, ``read`` 312 -> 358, ``todo`` 463 -> 498 and ``write`` 100 -> 144.
#: Every other entry is byte-identical. Re-derived again after the fold onto
#: ``origin/main`` ``0dac063b9`` (#1837): ``project`` alone moved vs the
#: pre-fold reading, 1067 -> 1077, because that lane re-joined its
#: description contract to the wire and keeps the dynamic cap — so the
#: wave's pin for this tool rests on the milestone-rationale phrase. These
#: docs grew ON PURPOSE: prose cut from the always-loaded wire (descriptions
#: and parameter schemas, the
#: ledger the context-budget guard measures) moved HERE, where it costs
#: nothing until ``read tool://<name>`` asks for it. The pairing is pinned by
#: :func:`test_cut_wire_prose_moved_into_the_tool_doc`: a doc growing without
#: a wire shrinking (or a wire shrinking with the detail deleted outright)
#: fails there by phrase.
#: RE-MEASURED AGAIN 2026-10-01 by ``feat/sessions-bulk-resume-1001`` (the
#: bulk-resume SET form), on top of the wave's table above: ``sessions``
#: 738 -> 850 — in THIS change the only entry that moves, for the four new
#: accepted inputs on ``resume``
#: (``paused``/``failed``/``all``/``dry_run``, plus ``limit`` joining the
#: op's set) and the rewritten summary/refusal lines — the ``op='help'``
#: reference is exactly where the added vocabulary is supposed to land, one
#: read away instead of on every request. Re-measured through this file's own
#: renderer; every other entry is byte-identical.
#:
#: RE-MEASURED 2026-10-01 by the browser read actions
#: (``feat/browser-read-actions``): ``browser`` 1069 -> 1363, for the three
#: read ops (styles/hit_test/ancestors) and their notes bullet — the detail
#: the wire deliberately carries as verb NAMES only (the caps, the viewport-
#: pixel semantics and the no-match behaviour live here and in
#: ``guide://browser``). Measured through this file's own renderer; every
#: other entry is byte-identical. Review round 1 then moved 1363 -> 1365:
#: the ``properties`` field copy widened to name underscores (R1-2, ``_`` is
#: a legal ident character), and this doc's parameters panel quotes the field
#: copy, so the +11 chars are the panel's, counted rather than assumed.
#:
#: RE-MEASURED 2026-10-02 by the agent display labels
#: (``feat/agent-labels-1001``): ``agent`` alone moved, 826 -> 849, for the new
#: ``label`` parameter on ``AgentParams`` (create/update; ``team``'s label param
#: has the same shape). The name stays the addressing key and the label is
#: display-only, so the wire carries the field and nothing else — the how/why
#: lives in ``guide://agents`` and ``tool://agent``, which is where the rest of
#: the label vocabulary already is. Measured through this file's own renderer on
#: the remediation head; every other entry is byte-identical.
#:
#: RE-MEASURED 2026-10-03 by the ask flip (``feat/ask-flip-default``): ``ask``
#: alone moved, 948 -> 1205, and the move IS the change rather than a side
#: effect of it — ``asks.policy.NONBLOCKING_ASK`` now defaults ON, so
#: ``render_tool_doc`` renders the QUEUED tail (the receipt, a receipt-is-not-
#: consent rule, the deadline calibration and the secret shape in one long
#: paragraph) instead of the inline one. The BLOCKING tail still renders 948
#: byte-for-byte, which is why the kill switch does not disturb this table, and
#: why a future edit to either tail must re-measure BOTH arms here and in
#: ``scripts/bench_context_budget.py`` — the two ledgers move together.
#: Measured through this file's own renderer on the flip head; every other
#: entry is byte-identical.
#:
#: RE-MEASURED 2026-10-04 by the peer-address role-word guard
#: (``fix/role-word-recipient``): ``send`` and ``sessions`` alone moved,
#: 614 -> 639 and 850 -> 864. Both moves ARE the change: each tool's ``target``
#: field description now states the two rules the resolver applies — an EXACT
#: conversation name / session id / cwd basename wins over any substring, and a
#: team role word (``manager``) is refused — which is what a model reads before
#: it types a target. Every other entry is byte-identical, and the same two
#: descriptions are the whole of ``scripts/bench_context_budget.py``'s raise
#: (``35,874 -> 35,937``) — the two ledgers move together.
#:
#: RE-MEASURED 2026-10-05 by the agent-side settle (``feat/ask-withdraw``,
#: design §12): ``ask_withdraw`` is NEW — 327 — because the default surface
#: itself grew. The tool is createIf-gated on the queued engine's door, and
#: ``scripts/real_tool_surface`` now binds that door the way it binds
#: ``ask_user`` (leaving it unbound would measure a surface no user has — the
#: queued arm is the default since the flip). Every other entry is
#: byte-identical, and the same schema is the whole of
#: ``scripts/bench_context_budget.py``'s raise (``35,937 -> 36,552``, the
#: clean-arm arithmetic in its entry) — the two ledgers move together.
#: RE-MEASURED 2026-10-06 by the sessions-remote tool surfaces
#: (``feat/sessions-remote-tools-b``, PR #2011): ``network`` 742 -> 877,
#: ``send`` 639 -> 728 and ``sessions`` 864 -> 981 — the three docs whose
#: schemas and reference text the mesh surfaces grew (the §4 field strings;
#: ``sessions`` also renders the per-op reference, which gained the peer/scope
#: refusals and summaries). Every other entry is byte-identical, and the same
#: three tools are the whole of ``scripts/bench_context_budget.py``'s raise
#: (``36,552 -> 37,094``) — the two ledgers move together.
#: RE-MEASURED 2026-10-08 by the context diet (deferred tool schemas,
#: ``perf/deferred-tools``). EVERY entry moved, and the whole table moved
#: DOWN: ``create_tools`` now collapses an optional field's
#: ``anyOf: [T, null]`` + ``default: null`` to ``T`` and the injected ``i``
#: description is shorter, and the doc renders the schema it is given. Only
#: two entries moved UP, and both are moves rather than growth: ``project``
#: (1077 -> 1102) and ``todo`` (498 -> 524) gained the per-field/per-op prose
#: their wire schemas gave up (pinned both ways in ``MOVED_WIRE_DETAIL``).
#: ``send``/``ask`` also gained notes, but their shorter descriptions (which
#: the doc renders too) outweigh them; ``sessions`` fell 981 -> 802 for the
#: same reason.
#: RE-MEASURED 2026-10-08 on the MERGED tree (context diet + image-generation
#: restore, ``feat/image-generation-tool``) — the merge of the two landed the
#: same day, so this table was measured once, after both: ``generate_image``
#: is NEW at 608 — the default surface grew by one createIf-gated tool (bound
#: in ``scripts/real_tool_surface`` the way the browser/console gates are,
#: because an unbound answer would depend on which machine ran it), and the
#: diet's collapse applies to its doc too (635 pre-diet -> 608 here). EVERY
#: other entry is byte-identical to the diet's table above, and the same
#: schema is the whole of ``scripts/bench_context_budget.py``'s raise — the
#: two ledgers move together.
#: RE-MEASURED 2026-10-09 (media wave-2 provider breadth,
#: ``feat/media-provider-breadth``): ``generate_image`` 608 -> 730 is the
#: ONLY entry that moved — the deliberate rewrite of the tool:// notes (seven
#: provider rungs, per-provider cost posture, the one-image-per-call skips)
#: in ``tool_docs.py``; the wire description SHRANK in the same wave (186 ->
#: 160 chars), which is one reason both ``bench_context_budget.py`` arms
#: still PASS. Every other entry is byte-identical.
#: RE-MEASURED 2026-10-10 (blocked-todo reconciliation, ``fix/todo-guard-blocked``):
#: ``todo`` 524 -> 540 is the ONLY entry that moved — one clause added to the
#: tool:// notes (blocked items are accounted for at the turn boundary). Off the
#: wire, so ``scripts/bench_context_budget.py`` is untouched and the two ledgers
#: do NOT move together this time.
# Re-measured 2026-10-15 through both isolated render arms: the shared intent
# shape stays "2-6 words, present participle"; only its repeated pointer shrank.
# Native personal-memory docs are newly accounted for; URL docs remain opt-in.
# 782 -> 804 (#2509): the tools/effort/delegate descriptions now say WHICH
# kind may carry them, so the agent doc's rendered schema grew by those
# qualifiers (the tier arm below moves with it).
# 1001 -> 1063 (#2520, ask relay): the ``AskParams.relay`` property and the
# queued tail's one relay clause are the whole move, and BOTH ledgers were
# re-measured in the same pass — ``scripts/bench_context_budget.py``'s top arm
# moves 88,825 -> 89,094 chars (+269 = ~+97 billed) and its ceiling with it,
# while every other entry here is byte-identical. The field is mode-independent,
# so the blocking arm's bench number moved by the same +269 chars and still
# passes under that ceiling.
# RE-MEASURED (LOH-9 S2): agent 804 -> 1044 and team 779 -> 1018 are the ONLY
# entries that moved. The agent doc gains the scope ladder (reuse -> ephemeral
# -> durable), the delete op, and the consent refusal's user surfaces; the team
# doc gains the same scope rule and its delete semantics. Both were measured
# through this test's own printed table on the rebased tree, whose upstream
# baseline is the #2520 ask raise above (ask 1063, unmoved here).
# RE-MEASURED 2026-10-23 (stale asks, ``feat/ask-stale-guidance``; design
# ask-nonblocking §16/SA-D1): ``ask`` 1063 -> 1079 and ``ask_withdraw``
# 314 -> 452 are the ONLY entries that moved, and both moves ARE the change:
# the ``ask`` queued tail gains one clause (settle it the moment it stops
# mattering — the tool the clause names is NOT in this doc's schema, so the
# property list is unchanged), and ``ask_withdraw``'s doc renders the rewritten
# description plus the widened ``answers`` field copy (the empty-list recipe,
# the ids rule, the all-empty refusal, and "latest or earlier" beside VERBATIM —
# the last two taken in review round 1, which is the 418 -> 452 part). Measured
# through this test's own printed table on the branch; every other entry is
# byte-identical. BOTH ledgers move together: the same two descriptions are the
# +663 chars of ``tool_schemas`` in ``scripts/bench_context_budget.py``'s
# top-arm raise (32,872 -> 33,364), whose child arm carries neither tool and
# moves only by the shared ``system.md`` paragraph.
MEASURED_TOKENS: dict[str, int] = {
    # RE-MEASURED (LOH-9 S3): 1044 -> 1056. The delta is the one-clause
    # fork exception on the ephemeral bullet ("a fork starts with a copy of
    # this conversation's roles"), added because the guides and the two
    # ``show`` messages now carry it. Measured through this test's own
    # printed table; the ``team`` entry below moves by the same 12 for its
    # twin sentence, and no other entry moved.
    # RE-MEASURED (LOH-9 S4): 1056 -> 1119. The delta is promotion: the ``op``
    # enum gains one word and the payload gains ``as_name`` and
    # ``confirm_replace`` (the second is the overwrite AND the user's answer, so
    # its wording is the safety-critical part). Trimmed once already; the same
    # printed table measured it.
    # RE-MEASURED (LOH-9 S5): 1119 -> 1365. The delta is the completed guidance:
    # a ``promote`` row in the authored ops table and the promote step in the
    # notes, the corrected refusal remedy (the TUI form and the create/install
    # retry, with update/reset/sync naming no retry), the collision clause and
    # the host-dependent default. The same printed table measured it; the
    # 8 KiB soft cap still holds.
    "agent": 1365,
    "ask": 1079,
    "ask_withdraw": 452,
    "bash": 295,
    # NEW (H5d, Team Bridge): the ``bridge`` doc — take-up's hash rule, the
    # claim lifecycle, and why "decide" is not an op. First measured at 572
    # tokens DELIBERATELY in the canonical (hermetic, tiktoken-exact) arm; every
    # pre-existing entry was re-checked against the same ruler in the same
    # pass (no other entry moved). The tool's schema itself is new, so
    # ``scripts/bench_context_budget.py``'s arms move with it — that bench is
    # the pairing check, run with this change's verification bundle.
    # RE-MEASURED (round 17, PR #2378 CI): 572 -> 563 is the ONLY entry that
    # moved. The tool's wire DESCRIPTION was reworded later on the branch
    # ("organization bridge messages" -> "organization messages"; the take-up
    # and ack clauses tightened) while this entry kept its H5d value. Measured
    # by this test's own printed table (the ``count_text_tokens`` ledger path)
    # and pasted; no other tool's bytes moved, and the 9-token drop is the reword alone.
    # RE-MEASURED (#2665 round 1): 559 -> 729. The channel lane publishes four
    # op literals, eleven parameters and the nested ``ChannelMemberParams``
    # model; its prose was trimmed first (761 -> 729 measured: wire description
    # 376 -> 240 chars, both model docstrings to one line each). 729 is the
    # structural floor for a capability the model must be able to call. The
    # paired start-context raise is stated in scripts/bench_context_budget.py.
    "bridge": 729,
    # Re-measured with this test's canonical isolated renderer: the read-only
    # inspect_ownership operation adds 19 opt-in tokens, not wire prompt cost.
    # 1327 -> 1424 (2026-10-16): the browser `type` secret_ref convention adds
    # the field's own description to the doc — the deliberate cost of the value
    # never appearing in the transcript (incident: a literal password in `type`
    # reached a subagent's transcript, where no shape pass could see it).
    "browser": 1380,
    # NEW (code-requests PR1b): the ``code_requests`` doc — relation and lane
    # vocabulary, the freshness rule and the data-not-instructions framing.
    # 475 tokens is the deliberate cost of the reference detail staying OFF the
    # wire (the description keeps one sentence; this holds the rest).
    "code_requests": 471,
    # RE-MEASURED via this test's own print (the count_text_tokens ledger path):
    # 1096 -> 1232 is the deliberate cost of advertising `console_resources` and
    # `console_cleanup` in the console doc — the ONLY entry that moved. The two
    # new methods and the capacity/cleanup detail are the whole delta.
    "console": 1228,
    "edit": 347,
    "eval": 372,
    #: Media wave-2 edit lane (2026-10-10): the source paragraph now names
    #: the by-reference input (``source_attachment``) and the edit rungs;
    #: +63 tokens over the 730 the slimming wave measured — re-measured
    #: deliberately with this change, then again after the context-budget
    #: round trimmed the schema description (795 -> 793).
    #: RE-MEASURED 2026-10-10 (hub media slice harness co-flip,
    #: ``feat/radient-edit-sources``): 793 -> 838 is the ONLY entry that moved
    #: — Radient re-joins the edit sentence as a capable rung, the ChatGPT-plan
    #: skip is stated alone, and the un-pinned edit's hub default (sunburst
    #: edit + its drift caveat) is documented. Doc-only: the wire description
    #: and schema are untouched, so ``bench_context_budget.py``'s arms are
    #: unmoved by this change.
    # 838 -> 918 (2026-10-11, multi-source edit lane): the source clause now
    # names the plural fields and the 16-source ceiling.
    # 918 -> 935 (2026-10-11, round-1 remediation F2): the cost sentence no
    # longer lumps OpenAI with the never-reporting rungs — an OpenAI edit
    # shows an estimated figure when the response carries usage.
    # 931 -> 975 (media artifact naming): the new ``filename`` stem field's
    # teaching description renders in the parameter table. The sanitising and
    # collision rules live in the image guide, so the wire keeps one sentence.
    "generate_image": 975,
    # NEW (sound-generation wave, 2026-10-17): the ``generate_sound_effect``
    # doc — the two rungs and the no-OpenAI rule, the REQUIRED duration and
    # what it costs, the estimate rules and the honest cancel paragraphs.
    # The wire description carries one sentence; this holds the rest. Both
    # audio entries were measured by this test's own printed table (the
    # ``count_text_tokens`` ledger path) in the canonical arm.
    # RE-MEASURED 2026-10-18 (#2595, B1/Q-1): reviewed batch schema was
    # 487 (+56 from 431). Round-1 M4/M5 clarifies the exclusive single/batch
    # fields and caps names at 100 chars: final render is 494 (+7 from 487).
    # Both measurements use render_tool_doc + count_text_tokens in this test's
    # canonical isolated arm, not a copied or guessed token allowance.
    # 494 -> 568 (media artifact naming): the new ``filename`` field for the
    # single form and the ``name`` field re-described as that item's filename
    # stem. Load-bearing naming discovery; the guide holds the rules.
    # 568 -> 570 (round-1 m1/Q-1): the ``sounds`` description now names
    # ``filename`` in the exclusivity clause, so the inert combination is
    # refused rather than silently ignored.
    # 570 -> 739 (sfx-quality, 2026-10-23): the doc now carries the 192/128
    # MP3 preference and its plan fallback, the duration/price split (per second
    # on Radient, per generation on a BYOK key), the negative-prompt rule and
    # the BYOK vendor-logging limit. Measured by this test's own printed table
    # under the canonical hermetic arm; only the two audio entries moved.
    # 739 -> 725 (review round 1, M1): the "per generation ~200 credits" claim
    # was wrong for the request shape this tool always sends; the corrected
    # sentence (40 credits per second of requested duration) is shorter.
    "generate_sound_effect": 725,
    # NEW (video phase, 2026-10-10): the ``generate_video`` doc — the async
    # lifecycle (receipt, completion, restart, cancel), the two rungs, the
    # mode/source rules and the family quirks. The wire description carries
    # one sentence; this holds the rest. Re-measured deliberately via this
    # test's own print; the ONLY other count that moved is ``jobs`` below.
    # 796 -> 794 (round-1 design D3): the wire description's first sentence is
    # now the contract §2.3 string verbatim (the shipped wording stated the
    # same fact twice); re-measured deliberately, and it is the ONLY entry
    # this remediation moved.
    # 790 -> 833 (media artifact naming): the ``filename`` stem field's
    # teaching description; the destination/collision rules live in the guide.
    "generate_video": 833,
    # NEW (sound-generation wave, 2026-10-17): the ``generate_voiceover`` doc
    # — the three rungs and the failover rule, cost and estimates, the
    # language-on-flash limit (no pronunciation dictionaries), MP3-only and
    # the honest cancel paragraphs.
    # 628 -> 668 (media artifact naming): the ``filename`` stem field's
    # teaching description renders in the parameter table.
    # 668 -> 788 (sfx-quality, 2026-10-23): the MP3 sentence now carries the
    # 192/128 preference and fallback, and the doc states the BYOK
    # vendor-logging limit. Same measured table, same canonical arm.
    # 788 -> 812 (review round 1, M1): one sentence names the BYOK voiceover
    # meter (per character; the estimate is per 1,000 characters).
    "generate_voiceover": 812,
    "glob": 87,
    "grep": 333,
    # NEW (LOH-14 agent slice): the ``harness_import`` doc advertises the six
    # actions plus the preview-first/confirm/selection contract. Measured with
    # this test's canonical isolated renderer; no existing entry moved. Round-1
    # remediation trimmed the wire schema descriptions: 297 -> 182.
    "harness_import": 182,
    # 675 -> 673 (child-attribution v1): the ``message`` field description was
    # rewritten to say that pause/cancel RECORD it as the stop reason. The new
    # wording is SHORTER than the old ("Body. Required for send/ask/steer/resume;
    # recorded as the stop reason for pause/cancel; ignored by list/peek."), so
    # the hub doc got two tokens CHEAPER while gaining the fact — re-measured
    # deliberately, not pasted.
    # 673 -> 677 (round 1 remediation, D2/D4): the verb became mechanism-neutral
    # ("the reason the child ENDED"), because the same slot carries a PAUSE and a
    # paused child is halted-and-resumable, not stopped. Re-measured.
    "hub": 721,
    # 350 -> 351 (video phase, 2026-10-10): the ``jobs`` description now says
    # "(task/bash/video)" — the video job type is a model-visible part of what
    # ``op='list'`` shows. One token, re-measured.
    "jobs": 347,
    "list_variables": 54,
    # NEW (Library): the six-op catalog tool's reference — filters, tag and
    # provenance rules, and the read-tier rationale. Measured through this
    # file's own renderer; no other entry moved.
    "library": 683,
    "lsp": 272,
    # 116 -> 127 (memory move, round 1): the description now names
    # ``edit new_text={"move":"scope"}``, which makes the verb discoverable
    # without a guide lookup. Re-measured deliberately via this test's print.
    "memory": 127,
    "monitor": 353,
    "network": 870,
    # NEW (quiet-turn S0a, docs/design/quiet-turns.md §4): the ``no_reply``
    # doc — the argumentless call's description plus the "when NOT to call
    # it" notes. Re-measured via this test's own print, and it is the ONLY
    # entry that moved: every other count above is byte-identical.
    "no_reply": 109,
    "patience": 330,
    # RE-MEASURED 2026-10-20 (evidence-attachment discoverability): ``project``
    # 1098 -> 1143 is the ONLY entry that moved — the ``attach`` description now
    # names its occasion (frame/chart/log) and the copy-and-limit contract;
    # re-measured through this test's own printed table, not pasted.
    "project": 1143,
    "project_delete": 91,
    "read": 345,
    # NEW (redaction policy, PR #2796): the sanctioned public-copy reader.
    # 237 tokens measured through this test's own table; deliberately small —
    # it exists to avoid a bypass, not to be a second file browser.
    "read_public_copy": 121,
    "read_variable": 67,
    "secret": 184,
    # Advertised surface re-measured: explicit repeat + confirmed-reuse detail
    # costs +86 opt-in tokens; 3182 chars remains below the 8192-char soft cap.
    "send": 779,
    # Spawn's tags field/description deliberately adds 21 reference tokens.
    # RE-MEASURED 2026-10-23 (session self-identity): the description gains one
    # sentence pointing at the listing's `(this session)` marker, so the doc
    # moves 819 -> 840. That sentence is the ONLY thing that moved; every other
    # count above is byte-identical, re-measured through this test's own
    # printed table rather than pasted.
    "sessions": 840,
    # NEW (wave 4 slice A, 2026-10-10): the ``setting`` doc — the key
    # vocabulary, the op split and the validation-by-registry contract. 191
    # tokens is the deliberate cost of the reference detail staying off the
    # wire (the description keeps one sentence); re-measured via this test's
    # own print, and it is the ONLY count this change moved.
    "setting": 187,
    # NEW (V2-4 agent tags, 2026-10-20): the ``tags`` doc — the names-only
    # assignment rule, the caps, the remote refusal and the receipt contract.
    # Deliberate cost of that reference being one read away; the same tool's
    # schema is the whole of ``scripts/bench_context_budget.py``'s raise
    # (measurement recorded there), so the two ledgers move together.
    # 528 -> 531 in review round 2: the receipt claim is qualified to
    # ``assign``, because ``create``/``list``/``rename`` name categories only.
    "tags": 531,
    # NEW (claude.ai artifact data store, 2026-10-21): the ``artifact_data``
    # doc — the per-action field table, the path/`me` rules and the
    # ``if_version`` contract. The tool is DEFERRED, so this doc is what the
    # first ``tool://`` read or call teaches; its schema adds nothing to the
    # start context (``scripts/bench_context_budget.py`` unchanged).
    "artifact_data": 836,
    "task": 616,
    # 485 -> 779 (#2521, team-member model tier / reasoning effort): the whole
    # delta is the new ``overrides`` parameter, and it is SCHEMA rather than
    # prose — the same arm renders the doc back at 485 when that property and its
    # ``$defs.TeamOverride`` entry are stripped, i.e. the cost is the nested
    # list-of-objects shape and its three field descriptions, not added
    # sentences. It is the ONLY entry that moved. Re-measured deliberately
    # through this test's own printed table; the tier arm below is unmoved
    # because this doc does not read config. RE-MEASURED (LOH-9 S2): 779 ->
    # 1018, the scope rule and delete semantics; measured with the same printed
    # table, and the tier arm below still does not read config.
    # RE-MEASURED (LOH-9 S3): 1018 -> 1030, the fork-copy clause; only this
    # and the ``agent`` entry moved.
    # RE-MEASURED (LOH-9 S4): 1030 -> 1109, the same promotion delta; the team
    # doc also states that a replace may cover a member, which the agent twin
    # does not need.
    # RE-MEASURED (LOH-9 S5): 1109 -> 1400 — the promote row, the promote and
    # member-copy step, the corrected refusal copy and the collision clause;
    # the same printed table.
    "team": 1400,
    "team_delete": 86,
    "todo": 536,
    # NEW (video phase, 2026-10-10): the ``video_status`` doc — the cold-safe
    # read side: states, the at-most-once registration and the watcher handoff.
    "video_status": 202,
    "wait": 268,
    "wake": 308,
    "web_fetch": 348,
    "web_read": 221,
    "web_search": 189,
    "write": 137,
}

#: The SAME ledger's other arm: ``agent``/``task`` rendered against a
#: tiers-configured ``values.subagents`` (the synthetic selectors the effort
#: tests already use, so the config shape lives in one place). Recorded so a
#: change to the config-sensitive branch is a visible edit here rather than a
#: machine-dependent surprise. RE-MEASURED 2026-10-01 by the slimming wave:
#: 716 -> 855 and 585 -> 668, the same moved-here-on-purpose growth the clean
#: arm documents (the per-op tables and notes now carry what the wire used
#: to).
#: 878/672 -> 815/639 (2026-10-08, context diet): the same collapse and
#: intent-description cut as the canonical table above.
#: 639 -> 635 (and `task` 488 -> 484 above): the sentence the depth gate added to
#: the model_choice note is gone again (revert of the nested tier-choice gate).
#: 811 -> 833 (#2509): the same +22 the canonical table above takes, because
#: the qualified tools/effort/delegate descriptions render on both arms.
#: 833 -> 1073 (LOH-9 S2): the same scope ladder the canonical arm takes; the
#: tier arm renders it identically, so `task` stays 764.
#: 1073 -> 1085 (LOH-9 S3): the one-clause fork exception, the same +12 as
#: the canonical arm; `task` stays 764.
#: RE-MEASURED (LOH-9 S4): agent 1085 -> 1148; ``task`` is unmoved because the
#: promotion fields are on ``agent``/``team``, not on the launch tool.
#: RE-MEASURED (LOH-9 S5): agent 1148 -> 1394, the same +246 the canonical arm
#: takes because both arms render the completed notes; ``task`` stays 764.
TIER_ARM_TOKENS: dict[str, int] = {"agent": 1394, "task": 764}


async def _noop_execute(*_args: Any, **_kwargs: Any) -> ToolResult:
    raise AssertionError("tool bodies are never executed by these tests")


def _tool(
    name: str,
    *,
    label: str = "",
    description: str = "",
    parameters: Any = None,
    hidden: bool = False,
) -> AgentTool:
    return AgentTool(
        name=name,
        label=label,
        description=description or f"{name} tool",
        parameters=parameters if parameters is not None else {"type": "object", "properties": {}},
        hidden=hidden,
        execute=_noop_execute,
    )


def _text_of(result: ToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


@pytest.fixture()
def hermetic_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An ISOLATED, EMPTY config directory — the arm CI renders.

    ``agent`` and ``task`` build their docs from ``values.subagents`` config
    (``model_choice`` / ``models``, read through ``local_operator.config``), so
    a run without this renders whatever the RECORDER's machine is configured
    with: measured, the two docs are 687/422 tokens clean and 714/581 with
    tiers configured — exactly how this suite once passed on CI and failed on
    a tiers box (B2/Q-2). Empty here = the canonical arm.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return config


@pytest.fixture()
def default_surface(hermetic_config: Path) -> list[AgentTool]:
    """The full default tool surface, exactly as the budget bench measures it.

    Reuses ``scripts/real_tool_surface.build_real_tools`` so these drift tests
    and the ``context-budget`` CI guard cannot disagree about what "the default
    surface" is: it forces the three machine-probing createIf gates (browser,
    console and the image provider) deterministically ON, which is what makes a
    CI runner and a developer box measure the same surface — the count is
    ``len(DEFAULT_TOOL_NAMES)``, asserted below rather than spelled here.
    Function-scoped (not module-scoped)
    because the builders read the config at BUILD time, so each test's
    :func:`hermetic_config` must apply to its own render; the fixture is taken
    for its env side effect alone.
    """
    tools = build_real_tools(".")
    assert {tool.name for tool in tools} == set(DEFAULT_TOOL_NAMES), (
        "the default surface changed shape (a builder is gated off, or a tool "
        "was added/removed); update this suite's expectations with it"
    )
    return tools


# ---------------------------------------------------------------------------
# An independent schema walker (the drift oracle for completeness above)
# ---------------------------------------------------------------------------


def _deref(schema: Any, root: dict[str, Any]) -> Any:
    """Resolve a local ``$ref`` against the root ``$defs``; other schemas pass."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            defs = root.get("$defs")
            if isinstance(defs, dict):
                return defs.get(ref.split("/")[-1], schema)
    return schema


def _nested_containers(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    """Object schemas whose properties belong one level deeper in the doc.

    A field name can hide behind three container shapes — an inline object, an
    array's ``items``, or an ``anyOf`` branch with an object arm — and all
    three are how this codebase's schemas nest their models.
    """
    resolved = _deref(subschema, root)
    if not isinstance(resolved, dict):
        return []
    candidates = [resolved]
    for branch in resolved.get("anyOf") or ():
        candidates.append(_deref(branch, root))
    if isinstance(resolved.get("items"), dict):
        candidates.append(_deref(resolved["items"], root))
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("properties"), dict)
        and candidate["properties"]
    ]


def _collect_enums(subschema: Any, root: dict[str, Any], enums: set[str]) -> None:
    resolved = _deref(subschema, root)
    if not isinstance(resolved, dict):
        return
    for literal in resolved.get("enum") or []:
        enums.add(str(literal))
    for branch in resolved.get("anyOf") or ():
        _collect_enums(branch, root, enums)
    if isinstance(resolved.get("items"), dict):
        _collect_enums(resolved["items"], root, enums)


def _collect_entries(
    schema: Any,
    root: dict[str, Any],
    depth: int,
    entries: dict[tuple[int, str], set[str]],
) -> None:
    """``(depth, name) -> enum literals`` the doc must carry at that position.

    ``depth`` is the CONTAINER level: root properties live at 0, and the
    renderer indents two spaces per level, so the pair pins both the name AND
    the line it must appear on — not merely that the bytes occur somewhere.
    """
    resolved = _deref(schema, root)
    if not isinstance(resolved, dict):
        return
    properties = resolved.get("properties")
    if not isinstance(properties, dict) or not properties:
        return
    for name, subschema in properties.items():
        literals = entries.setdefault((depth, str(name)), set())
        _collect_enums(subschema, root, literals)
        if depth >= _WALK_DEPTH_CAP:
            continue
        for child in _nested_containers(subschema, root):
            _collect_entries(child, root, depth + 1, entries)


def _structured_lines(doc: str, depth: int, name: str) -> list[str]:
    """The rendered parameter line(s) for ``name`` at container ``depth``.

    The renderer writes one line per property, ``{indent}- `name` (annotations)``
    with two spaces of indent per nesting level. Matching THAT form — rather
    than "the name occurs somewhere in the doc" — is what lets the oracle see
    a dropped line: a name surviving only in prose (a description quoting it,
    the Purpose paragraph) does not count (M1).
    """
    prefix = f"{'  ' * depth}- `{name}` ("
    return [line for line in doc.splitlines() if line.startswith(prefix)]


def _max_properties_depth(schema: Any, root: dict[str, Any], depth: int = 0) -> int:
    """Deepest ``properties`` container in the schema, unbounded."""
    resolved = _deref(schema, root)
    if not isinstance(resolved, dict):
        return -1
    best = -1
    properties = resolved.get("properties")
    if isinstance(properties, dict) and properties:
        best = depth
        for subschema in properties.values():
            for child in _nested_containers(subschema, root):
                best = max(best, _max_properties_depth(child, root, depth + 1))
    for branch in resolved.get("anyOf") or ():
        best = max(best, _max_properties_depth(branch, root, depth))
    if isinstance(resolved.get("items"), dict):
        best = max(best, _max_properties_depth(resolved["items"], root, depth + 1))
    return best


# ---------------------------------------------------------------------------
# Completeness (§3.1) and the depth cap it depends on
# ---------------------------------------------------------------------------


def test_every_default_surface_property_name_and_enum_literal_reaches_its_doc(
    default_surface: list[AgentTool],
) -> None:
    """No name, type or enum literal may be dropped from a rendered doc.

    This is the reason the mechanism can replace reference prose: the reader
    gets the FULL accepted surface, not a summary of it. A missing enum
    literal is a wrong answer to the exact question ``read tool://<tool>`` is
    asked; a missing property name reads as "this field does not exist".

    The name check is STRUCTURAL (M1): each collected ``(depth, name)`` pair
    must appear as the rendered parameter line at its own indentation, and
    each enum literal must appear INSIDE that line. A substring check would
    pass whenever a dropped line's name (or literal) still occurs in the
    surrounding prose — e.g. ``timeout`` inside another field's description —
    which is exactly the drift this oracle exists to catch.
    """
    for tool in default_surface:
        doc = render_tool_doc(tool)
        entries: dict[tuple[int, str], set[str]] = {}
        _collect_entries(tool.parameters or {}, tool.parameters or {}, 0, entries)
        missing_lines: list[str] = []
        missing_literals: list[str] = []
        for (depth, name), literals in sorted(entries.items()):
            lines = _structured_lines(doc, depth, name)
            rendered = f"{'  ' * depth}- `{name}` ("
            if not lines:
                missing_lines.append(rendered)
                continue
            for literal in sorted(literals):
                if not any(literal in line for line in lines):
                    missing_literals.append(f"{rendered} lacks enum literal {literal!r}")
        assert (
            not missing_lines
        ), f"{tool.name}: parameter line(s) missing from its tool:// doc: {missing_lines}"
        assert not missing_literals, (
            f"{tool.name}: enum literals missing from their parameter line(s): "
            f"{missing_literals}"
        )


def test_default_surface_fits_the_documented_depth_cap(
    default_surface: list[AgentTool],
) -> None:
    """The depth cap is a standing decision, not a silent truncation.

    If a future schema nests its models one level deeper than the renderer
    walks, the doc would silently drop those field names (and the completeness
    test above, written to the same cap, would not notice). This fails FIRST,
    by name, so the choice is explicit: trim the nesting, or raise the cap and
    the walkers together.
    """
    too_deep = {
        tool.name: _max_properties_depth(tool.parameters or {}, tool.parameters or {})
        for tool in default_surface
    }
    offenders = {name: depth for name, depth in too_deep.items() if depth > _WALK_DEPTH_CAP}
    assert not offenders, (
        f"schemas nested deeper than the {_WALK_DEPTH_CAP}-level cap: {offenders}. "
        "The doc drops those field names; raise the cap deliberately or flatten "
        "the schema."
    )


# ---------------------------------------------------------------------------
# Determinism (§3.2)
# ---------------------------------------------------------------------------


def test_render_is_deterministic_including_a_schema_json_round_trip(
    default_surface: list[AgentTool],
) -> None:
    """Two renders are byte-equal; a JSON round-tripped schema renders equal too.

    The round trip is the shape a schema has after persistence, and the one
    the drift tests compare against (``json.dumps/loads``) — an implementation
    leaning on dict identity, insertion quirks or formatting locale would pass
    the first assertion and fail this one. It also pins that the renderer does
    not mutate the tool it reads.
    """
    for tool in default_surface:
        before = json.dumps(tool.parameters, sort_keys=True)
        first = render_tool_doc(tool)
        assert render_tool_doc(tool) == first, f"{tool.name}: two renders differ"
        reloaded = tool.model_copy(deep=True)
        reloaded.parameters = json.loads(json.dumps(tool.parameters))
        assert (
            render_tool_doc(reloaded) == first
        ), f"{tool.name}: render differs after a JSON schema round-trip"
        assert (
            json.dumps(tool.parameters, sort_keys=True) == before
        ), f"{tool.name}: the renderer mutated its input schema"


# ---------------------------------------------------------------------------
# Token ledger + soft cap (§3.3)
# ---------------------------------------------------------------------------


def test_docs_stay_within_the_soft_cap_and_the_token_ledger_is_current(
    default_surface: list[AgentTool],
) -> None:
    """The 8 KiB soft cap holds, and the ledger changes are DELIBERATE.

    The cap keeps ``read``'s runtime shaping (16 KiB) dormant; the ledger
    makes a doc getting more expensive a visible, reviewed edit in the same
    diff rather than creep nobody notices. The failure prints the whole new
    table so the update is a paste, not an archaeology dig.
    """
    current: dict[str, int] = {}
    oversized: list[str] = []
    for tool in default_surface:
        doc = render_tool_doc(tool)
        if len(doc) > DOC_SOFT_CAP_CHARS:
            oversized.append(f"{tool.name}: {len(doc)} chars")
        current[tool.name] = count_text_tokens(doc)
    assert not oversized, (
        f"tool:// docs exceed the {DOC_SOFT_CAP_CHARS}-char soft cap: {oversized}. "
        "Names/types/enums are never truncated — trim authored prose instead."
    )
    if current != MEASURED_TOKENS:
        raise AssertionError(
            "The tool:// token ledger moved. Re-measure deliberately before "
            "updating MEASURED_TOKENS in this file:\n\n"
            f"MEASURED_TOKENS = {json.dumps(current, indent=4, sort_keys=True)}"
        )


def test_the_tier_configured_arm_is_pinned_too(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The config-sensitive branch is PINNED, not ignored (B2/Q-2).

    ``agent``/``task`` advertise ``values.subagents.models`` and
    ``subagents.model_choice``, so their docs move with the operator's config —
    the non-hermeticity that once made the canonical ledger green on one
    machine and red on the other. The canonical table above renders the CLEAN
    arm under :func:`hermetic_config`; this test renders the tiers arm against
    a synthetic config and pins its sizes too, so a change to EITHER branch is
    a visible edit rather than a machine-dependent surprise.
    """
    config = tmp_path / "tier-config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_tiers(config, THREE)
    tools = {tool.name: tool for tool in build_real_tools(".")}
    measured = {
        "agent": count_text_tokens(render_tool_doc(tools["agent"])),
        "task": count_text_tokens(render_tool_doc(tools["task"])),
    }
    assert measured == TIER_ARM_TOKENS, (
        f"The tier-configured rendering moved: {measured}. Re-measure "
        "deliberately and update TIER_ARM_TOKENS."
    )
    # The branch is real: both docs differ from the clean arm's bytes.
    assert measured["agent"] != MEASURED_TOKENS["agent"]
    assert measured["task"] != MEASURED_TOKENS["task"]


#: The slimming wave's no-loss map (audit items 2+3): prose CUT from the
#: always-loaded wire must be reachable in ``read tool://<name>`` — that
#: pairing is what makes the cut a MOVE rather than a deletion. Each phrase
#: below was removed from the tool's own wire bytes (description + parameter
#: schema, the bytes ``scripts/bench_context_budget.py``'s ``tool_schemas``
#: counter measures) by the wave, and each must render in the doc's ops table
#: or notes section.
#:
#: Asserted BOTH ways on purpose. ``in doc`` alone would pass while the
#: phrase still rode every request (no saving, no move); ``not in wire``
#: alone would pass with the detail deleted outright. The pair is the
#: contract, and it is why the wave's per-tool PR mapping table can say
#: "moved" rather than "cut".
MOVED_WIRE_DETAIL: dict[str, tuple[str, ...]] = {
    "agent": ("companion agents are the canonical one",),
    "ask": ("Every question also offers the user a free-text answer",),
    "browser": (
        "a redacted handle is not yours to drive",
        # The read actions' representative moved phrase: the hit-test ORDER is
        # operationally load-bearing (the first entry is what receives the
        # click) and now lives in tool://browser, not on the wire.
        "topmost first",
    ),
    "console": ("a terminal in another window has none",),
    "hub": (
        "the fast way to check on a running child",
        "Several ids address several subagents",
    ),
    "network": ("required by create/engage/stop/delete",),
    # Project's second candidate phrase, "multiple paragraphs, headings, lists
    # and code", was dropped from this map by the #1837 fold: that lane
    # deliberately keeps it on the wire (its description contract is built
    # from DESCRIPTION_MAX and pinned by
    # test_project_tool.py::test_the_schema_text_names_the_cap_the_refusal_enforces),
    # so the both-ways assertion cannot hold for it. The milestone-rationale
    # phrase stays as this tool's representative move.
    "project": ("derived, not stored", "<= 10 files, <= 5 MB each"),
    # Context diet (deferred-tools PR): the delivery modes and the fresh
    # session refusal left the description; the per-op prose left todo.op.
    "send": ("is not a recipient: sends to it are refused",),
    "task": ("a registered profile or a packaged starter",),
    "todo": ("Abandon items that are no longer needed",),
}


@pytest.mark.parametrize("name", sorted(MOVED_WIRE_DETAIL))
def test_cut_wire_prose_moved_into_the_tool_doc(
    default_surface: list[AgentTool], name: str
) -> None:
    """Wave-1 cut phrases, pinned both ways: in the doc, gone from the wire.

    A failure of the first assertion means the doc lost detail the wire gave
    up; a failure of the second means the phrase never left the wire (the
    "move" saved nothing) or a later edit re-inlined it — move it back out
    of the schema or the description, or pick a different representative
    phrase for this tool.
    """
    tool = next(tool for tool in default_surface if tool.name == name)
    doc = render_tool_doc(tool)
    wire = (tool.description or "") + json.dumps(tool.parameters or {})
    for phrase in MOVED_WIRE_DETAIL[name]:
        assert phrase in doc, f"{name}: cut phrase not reachable in tool://{name}: {phrase!r}"
        assert phrase not in wire, (
            f"{name}: phrase still on the always-loaded wire — the move saved "
            f"nothing, so either cut it or drop it from MOVED_WIRE_DETAIL: {phrase!r}"
        )


@pytest.mark.parametrize("name", ["agent", "team"])
def test_the_promote_op_and_step_are_reachable_in_the_rendered_doc(
    default_surface: list[AgentTool], name: str
) -> None:
    """The ledger pins SIZE, not content: a same-length edit could drop the row.

    The design's guidance-reachability metric is an assertion, not eyeballing, so
    both halves the model reads are pinned by phrase: the authored Ops-table row
    and the narrative PROMOTE step.
    """
    doc = render_tool_doc(next(tool for tool in default_surface if tool.name == name))
    assert "\n- promote: " in doc, f"tool://{name} lost its promote row"
    assert "\n- PROMOTE: " in doc, f"tool://{name} lost the promote step in its notes"


def test_authored_notes_and_renderers_name_real_tools(
    default_surface: list[AgentTool],
) -> None:
    """``TOOL_NOTES``/``SPECIAL_RENDERERS`` keys must be real tool names.

    Both are module-level mappings consulted at resolve time, so a typo in a
    key would not crash anything — the entry would simply never be used, and
    the doc would silently miss its authored ops. This makes the typo loud.
    """
    names = {tool.name for tool in default_surface}
    assert set(TOOL_NOTES) <= names, f"TOOL_NOTES keys not real: {sorted(set(TOOL_NOTES) - names)}"
    assert (
        set(SPECIAL_RENDERERS) <= names
    ), f"SPECIAL_RENDERERS keys not real: {sorted(set(SPECIAL_RENDERERS) - names)}"


# ---------------------------------------------------------------------------
# Authored attachments: ops table + notes hook
# ---------------------------------------------------------------------------


def test_authored_ops_and_notes_render_in_the_documented_shape() -> None:
    """The pilot seam: ``ToolDocNotes`` renders as the per-op table (§2.3)."""
    notes = ToolDocNotes(
        ops=(
            ToolDocOp(op="list", blurb="Show things.", fields=("id", "owner")),
            ToolDocOp(op="delete", fields=("id",)),
        ),
        notes="Only the owner may delete.",
    )
    doc = render_tool_doc(_tool("widget"), notes=notes)
    assert "## Ops" in doc
    assert "- list: Show things. — fields: id, owner" in doc
    assert "- delete — fields: id" in doc
    assert "## Notes" in doc
    assert "Only the owner may delete." in doc


def test_special_renderer_wins_and_a_broken_one_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered renderer serves ``tool://<name>``; a raising one degrades
    to the generic doc instead of an error result — the seam's whole contract
    is that the sessions pilot can pin byte parity without the reader ever
    losing the fallback."""
    tool = _tool("widget")
    monkeypatch.setitem(SPECIAL_RENDERERS, "widget", lambda _tool: "SPECIAL BYTES")
    resolver = make_tool_doc_resolver(lambda: [tool])
    assert resolver("tool://widget") == "SPECIAL BYTES"

    def _boom(_tool: AgentTool) -> str:
        raise RuntimeError("renderer exploded")

    monkeypatch.setitem(SPECIAL_RENDERERS, "widget", _boom)
    fallback = resolver("tool://widget")
    assert fallback == render_tool_doc(tool)


# ---------------------------------------------------------------------------
# Resolver contract (§3.4, §3.6)
# ---------------------------------------------------------------------------


def test_resolver_serves_a_hit_with_the_render_bytes() -> None:
    tool = _tool("bash", label="Shell", description="Run a shell command.")
    resolver = make_tool_doc_resolver(lambda: [tool])
    doc = resolver("tool://bash")
    assert doc == render_tool_doc(tool)
    # The name is URL-decoded, mirroring ``skills/api.py::_url_name``: a tool
    # whose name needs escaping must stay addressable.
    assert resolver("tool://ba%73h") == doc


def test_bare_tool_url_lists_visible_tools_in_sorted_order() -> None:
    resolver = make_tool_doc_resolver(
        lambda: [_tool("zeta"), _tool("alpha"), _tool("hidden-one", hidden=True)]
    )
    text = resolver("tool://")
    assert text == ("Tool URL missing a name: expected tool://<name>\nAvailable tools: alpha, zeta")
    assert "hidden-one" not in text


def test_unknown_name_names_the_available_set() -> None:
    resolver = make_tool_doc_resolver(lambda: [_tool("zeta"), _tool("alpha", hidden=True)])
    text = resolver("tool://bogus")
    assert text == "Unknown tool: bogus\nAvailable: zeta"
    assert "alpha" not in text


def test_a_direct_hit_on_a_hidden_tool_still_serves() -> None:
    """Hidden tools remain callable; a reader who knows the name is asking
    precisely because it is not listed."""
    hidden = _tool("hidden-one", hidden=True)
    resolver = make_tool_doc_resolver(lambda: [hidden])
    assert resolver("tool://hidden-one") == render_tool_doc(hidden)


def test_non_tool_urls_return_none_so_the_chain_is_preserved() -> None:
    resolver = make_tool_doc_resolver(lambda: [_tool("bash")])
    for url in ("skill://alpha", "guide://beta", "mcp://server/tool", "https://example.com"):
        assert resolver(url) is None, url


def test_resolver_never_raises_when_the_inventory_itself_fails() -> None:
    def broken() -> list[AgentTool]:
        raise RuntimeError("inventory down")

    resolver = make_tool_doc_resolver(broken)
    text = resolver("tool://bash")
    assert isinstance(text, str)
    assert "Tool reference unavailable" in text
    # The scheme check runs FIRST: a foreign URL never touches the inventory.
    assert resolver("skill://alpha") is None


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"properties": None},
        {"properties": {}},
        {"properties": {"x": "not-a-schema"}},
        {"properties": {"x": {"type": "array"}}},
        {"properties": {"x": {"$ref": "#/$defs/Missing"}}},
        {"properties": {"x": {"anyOf": []}}},
        # A self-referential $defs graph must terminate (the depth cap bounds
        # the walk) rather than recurse forever.
        {
            "$defs": {"A": {"properties": {"y": {"$ref": "#/$defs/A"}}}},
            "properties": {"x": {"anyOf": [{"$ref": "#/$defs/A"}]}},
        },
    ],
)
def test_render_never_raises_on_malformed_schemas(parameters: Any) -> None:
    doc = render_tool_doc(_tool("widget", parameters=parameters))
    assert isinstance(doc, str)
    assert "# Tool: `widget`" in doc


# ---------------------------------------------------------------------------
# The chain: ``tool://`` in front, everything else untouched
# ---------------------------------------------------------------------------


def test_chain_serves_tool_urls_ahead_of_the_base_walker() -> None:
    calls: list[str] = []

    def base(url: str) -> str | None:
        calls.append(url)
        return {"skill://s": "SKILL", "guide://g": "GUIDE"}.get(url)

    tools = [_tool("bash")]
    chained = chain_tool_docs(base, lambda: tools)
    assert chained("tool://bash") == render_tool_doc(tools[0])
    assert calls == [], "tool:// must never reach the base walker"
    assert chained("skill://s") == "SKILL"
    assert chained("mcp://m/t") is None
    assert calls == ["skill://s", "mcp://m/t"]
    # The link is ID-BEARING (B1/Q-1): the host-field parity guard recognises
    # the wrapper and recovers the host resolver through it instead of failing
    # an identity check that a plain closure could not satisfy.
    assert is_tool_docs_link(chained)
    assert unwrap_tool_docs(chained) is base
    assert is_tool_docs_link(base) is False
    assert unwrap_tool_docs(base) is None


def test_chain_still_answers_tool_urls_with_no_base_configured() -> None:
    """A session constructed with no knowledge resolver still serves its own
    tool docs; every other scheme degrades to None exactly as before."""
    solo = chain_tool_docs(None, lambda: [_tool("bash")])
    assert solo("tool://bash") == render_tool_doc(_tool("bash"))
    assert solo("skill://s") is None
    # The seam reports the (absent) base honestly: a link with no host
    # resolver unwraps to None, so the parity guard still REJECTS it for a
    # host value — that is the drop class the guard must keep seeing.
    assert is_tool_docs_link(solo)
    assert unwrap_tool_docs(solo) is None


# ---------------------------------------------------------------------------
# Integration: the Session wiring a subagent inherits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_wrapper_serves_read_tool_urls_over_live_inventory(tmp_path) -> None:
    """Exercise the real path: Session -> ToolContext -> read tool.

    The wrapper is installed in ``Session.__init__`` over a live-inventory
    lambda, and the turn's own ``ToolContext`` is what hands it to ``read`` —
    so this drives the same objects a real session does, not a test stub
    standing in for them. The refresh half pins the lambda (a snapshot would
    keep serving the construction-time list forever).
    """
    bash_tool = _tool("bash", label="Shell", description="Run a shell command.")
    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[bash_tool],
    )
    try:
        resolver = session._skill_resolver
        assert resolver is not None
        context = session._build_tool_context()
        assert context.resolve_internal_url is resolver
        read = {tool.name: tool for tool in create_tools(context)}["read"]

        result = await read.execute("call-1", {"path": "tool://bash"}, None, None, context)
        assert result.is_error is False
        assert _text_of(result) == render_tool_doc(bash_tool)
        assert result.details is not None
        assert result.details["url"] == "tool://bash"

        # ``refresh_tools`` rebinds ``self._tools``; the resolver must follow.
        session.refresh_tools([_tool("zsh", description="Other.")])
        zsh_doc = resolver("tool://zsh")
        assert zsh_doc is not None and "# Tool: `zsh`" in zsh_doc
        bash_doc = resolver("tool://bash")
        assert bash_doc is not None and bash_doc.startswith("Unknown tool: bash")

        # This session configured NO knowledge resolver: every other scheme
        # returns None, exactly as before the wrapper existed.
        assert resolver("skill://anything") is None
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The sessions pilot's byte parity (the #1862/#1863 seam)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_tool_url_sessions_serves_the_help_op_and_renderer_bytes(tmp_path) -> None:
    """One surface, two entry points: ``read tool://sessions`` == the ``help`` op.

    The sessions pilot promises byte parity between its ``help`` payload and
    the audit lane's reader, and this drives the REAL path for both — Session
    -> ``_build_tool_context`` -> the actual ``read`` tool over
    ``tool://sessions``, and the sessions tool's own executor — against one
    renderer. The generic-render inequality is the falsifier: without the
    registration (the tool-side half of the seam) the read serves
    :func:`render_tool_doc`'s schema render instead, so (1) and (3) both go
    red — mutation-checked by reverting the registration locally.
    """
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    try:
        context = session._build_tool_context()
        tools = {tool.name: tool for tool in create_tools(context)}
        sessions_tool = tools["sessions"]
        # The resolver reads the session's LIVE inventory (the lambda chained
        # in ``Session.__init__``), so the tool must be in it to be served.
        session.refresh_tools([sessions_tool])

        result = await tools["read"].execute(
            "call-read", {"path": "tool://sessions"}, None, None, context
        )
        assert result.is_error is False
        text = _text_of(result)

        reference = builtin.render_sessions_reference()
        # (1) The read serves the registered renderer's bytes, exactly.
        assert text == reference
        # (2) ...which are the ``help`` op's payload, exactly.
        help_result = await sessions_tool.execute("call-help", {"op": "help"}, None, None, context)
        assert help_result.is_error is False
        assert _text_of(help_result) == reference
        # (3) The registration is what makes it so: the generic render for the
        # same tool is DIFFERENT bytes, so an unregistered read cannot pass
        # (1). Determinism rides along — a second read is byte-equal.
        assert text != render_tool_doc(sessions_tool)
        again = await tools["read"].execute(
            "call-read-2", {"path": "tool://sessions"}, None, None, context
        )
        assert _text_of(again) == text
    finally:
        await session.dispose()


def test_sessions_special_renderer_is_registered_to_the_one_renderer() -> None:
    """The registration itself: keyed to ``sessions`` and pointing at the one
    renderer both entry points render with — a second renderer, or a copy,
    is the drift this pins shut."""
    assert SPECIAL_RENDERERS.get("sessions") is builtin.render_sessions_reference


#: Fresh-interpreter arms for the registration's import behaviour. Each boots
#: ``sys.executable`` (the interpreter the suite runs under) and asserts the
#: registration landed through imports a REAL startup performs — not because
#: pytest's collection order imported a side module first, and with no import
#: cycle that only some orders survive. ``tool_docs_first`` also pins WHO
#: registers: the leaf module must stay importable alone and register nothing
#: by itself.
_FRESH_IMPORT_ARMS: dict[str, str] = {
    "builtin_alone": (
        "import local_operator.tools.builtin as builtin\n"
        "from local_operator.tools.tool_docs import SPECIAL_RENDERERS\n"
        "assert SPECIAL_RENDERERS['sessions'] is builtin.render_sessions_reference\n"
    ),
    "tool_docs_first": (
        "import local_operator.tools.tool_docs as tool_docs\n"
        "assert 'sessions' not in tool_docs.SPECIAL_RENDERERS\n"
        "import local_operator.tools.builtin as builtin\n"
        "assert tool_docs.SPECIAL_RENDERERS['sessions'] is builtin.render_sessions_reference\n"
    ),
    "session_startup": (
        "import local_operator.session.session\n"
        "from local_operator.tools.tool_docs import SPECIAL_RENDERERS\n"
        "from local_operator.tools.builtin import render_sessions_reference\n"
        "assert SPECIAL_RENDERERS['sessions'] is render_sessions_reference\n"
    ),
}


@pytest.mark.parametrize("arm", sorted(_FRESH_IMPORT_ARMS))
def test_the_registration_is_live_for_normal_startup_import_orders(arm: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT_ARMS[arm]],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"{arm}: rc={proc.returncode}\n{proc.stderr}"


def test_read_description_advertises_the_tool_scheme() -> None:
    """The cue lives in system.md; the read description carries only the bare
    scheme list — and ``tool://`` must be in it, or the model never learns the
    address from the tool it would read the doc with."""
    description = builtin.build_read_tool().description
    assert "tool://" in description

# Agent facts — private implementation and actual rendered evidence

Scope: remove English agent kind/provenance/action-class badges in French and
all seven target locales without translating names, descriptions, model IDs,
effort-tier identifiers or machine fields. Source base:
`d8d81b6ea98095d982a213cb21513b1fa4742502`. Copy freeze:
`7d03eb5874ddbf8d3d16aeed8a12d7d204d68603`; independent A/B copy-only admission:
`e1435564433f23f53a50c6cc4d7ab07060a642f9`.

This is implementation evidence, not independent QA/design approval or a claim
that the wider i18n queue is complete. No PR, push, version bump or release.

## Why stable facts, and which copy

`AgentFact(code, value)` is immutable. The ONE existing roster enumeration now
returns ordered tuples of these facts; `agent_listing_rows` is its four-string
localized adapter. Settings takes an in-process three-slot snapshot containing
raw facts and resolves labels at paint. Live language switches therefore need
no registry reload, credential/network I/O, new callback or watcher. Public
local/routed listings retain their existing four/triple string shapes and raw
name addressing; no bidi controls enter the wire adapter. Sorting, filtering,
session-first deduplication, seed fallback and definition/scheduling fields are
unchanged. Installed seed copies do **not** acquire a packaged badge.

Exact semantic pairs select ink and scope lifting. A model named `proactive`,
a model equal to a translated scope label, or text containing ` · ` stays an
opaque technical fact, never an action/scope tag. Unknown kinds/values remain
raw, never guessed as roles. Reactive agents gain no new badge. Legacy agent
string fixtures are opaque; an explicitly separate legacy Teams painter keeps
its existing parsing, scope-line geometry and ink. No new Team API was added.

| Meaning | Existing generated helper reused |
|---|---|
| role / specialist | `tui_authoring.role()` / `.specialist()` |
| proactive | `tui_authoring.proactive()` |
| conversation scope | `tui_authoring.scope_conversation()` |
| effort prose | `tui_small.model_picker_effort(value=raw_tier)` |

The effort tier is configured technical vocabulary, not ordinary product prose;
custom tier/model/name/summary bytes remain literal. Effort's paint adapter uses
standalone technical projection rather than widening the restricted ASCII
command-span grammar. Raw content is fitted before render-only bidi controls.

Only one new English/context key exists: `wire.agent_badges.packaged`, meaning
an available built-in starter absent from the registry, not installation,
trust, security or permission. Source raw SHA-256:
`5caa688fa6d6a78f315e32f7bb1db3b8cae08c77f727dd36af22e45a0206a23b`;
context raw SHA-256:
`8ab96370595b180339f4a6392daaef10a6f8324f202f3738dfc97574df5591e7`.
Author/audit receipts and hash-domain distinctions remain in the parent folder.
No other namespace's copy, baseline, audit history or holds were edited here.
Existing authoring/effort contexts describe these same meanings; the new compact
usage is exercised below, not presented as a new audit of those namespaces.

## Actual regressions and controls

- Before source fix: `badges-before.log` records **2 failing assertions** at the
  existing painter: opaque `proactive` received marker ink, and opaque text
  matching the conversation label was extracted to another line.
- Final changed-surface run: **50 passed in 3.45s**, two existing Pydantic
  deprecation warnings; exact output in `badges-product-targeted.log`.
- The focused file covers semantic role/specialist × absent/reactive/proactive/
  unknown action class, seed suppression, one enumeration, unknown facts,
  technical collisions, immutable facts, raw effort, legacy Teams, fallback
  provenance, raw-fit-before-controls and same-object en→synthetic-fr→en live
  repaint with enumeration/load calls forbidden.
- Added actual-catalogue width/provenance cases cover en/fr/es/zh-CN/ru/vi/ur/hi
  × role/specialist: the whole proactive and scope labels survive a shared
  27-cell payload; raw model and custom effort identifiers stay unchanged.
- Direct neighbours: the five named painter/class tests in
  `test_agent_class_slash.py`, raw-label listing, scoped-list/session-shadowing,
  and both detached routed `agent_list` tests. The final log lists the actual
  execution result; no directory/full suite was run.
- Counterfactual on final executable code: the preserved test-only plugin
  restores literal English role copy; both synthetic-locale and mounted-App
  repaint regressions fail (**2 failed in 2.39s**), recorded in
  `badges-product-counterfactual.log`. Production source was not patched for it.

Tests used the primary checkout's Python **3.12.13** with `PYTHONPATH` selecting
this worktree (verified `local_operator.__file__`), serial `-n 0`, fresh scratch
HOME/config, `env -i`, explicit `LOP_LANG=en` and desktop/notification suppression.
Each targeted run was bounded below ten minutes. No installs or Pyright wrapper.

## Rendered frames: looked at, not inferred from tests

`capture_agent_facts.py.txt` is the exact runnable source used with the real
`OperatorApp.run_test`, its production stylesheet and the existing FakeSession
fixture. It never launches a browser/native window or a model request.

- Production: actual producer→settings snapshot→paint, **120×40** terminal grid.
- Priority tall: structured snapshot fixtures with both configured kinds,
  packaged/proactive/session facts, opaque model/effort, plus a 40-row roster;
  **100×40**. This is a painter/layout stress fixture, not a claim that the
  current specialist producer emits packaged seeds.
- Priority narrow: same stress roster at **100×30**; legitimate folding leaves
  one complete row in Russian/Urdu and two in the other captured locales.
- All eight locales in all three scenarios, two consecutive frames each:
  **48 frames / 24 pairs, all pixel-identical** after independent librsvg
  rasterization. `pixel-proof.json` records the comparison.
- Original English before frame and final production English are pixel-identical.
  Original English/French SVG/PNG and geometry files remain byte-exact here.
- Personally viewed the original English/French frames, final production
  French/Chinese/Urdu, and both eight-locale priority contact sheets. French
  now paints `rôle · fourni · proactif`; Chinese paints `角色 · 内置 · 主动`.
  Long native labels retain the whole muted proactive marker on a spill line;
  scope remains whole on its own line. Raw model text can deliberately resemble
  a marker but remains faint. Existing English summaries and footer outside this
  slice remain visible and are not claimed localized by this change.

The pane has **31 content cells**, four paid once for indentation, leaving ONE
**27-cell facts payload**. This is not 27 cells per translated word. Optional
technical tails give way first; a proactive marker that cannot fit after the
kind/source prefix spills whole, as does scope. The existing final `_fit_pane`
counts actual entry lines and reserves caption/count; its initial roster
estimate and all Teams geometry remain unchanged.

`runtime-geometry.json` records content/pinned/virtual sizes, pane height,
scrollbars, text, visible names and exact hidden counts for every frame. At
100×30 the pane is 31×14 with virtual size 34×14, payload 27 and no vertical
pane scrollbar; the wider virtual extent is recorded rather than represented
as a new shrink-to-fit guarantee. Narrow hidden counts are 38 except ru/ur=39.
Every count was compared with the production `_more()` resolver; caption and
height budgets were asserted. All first/settled pairs retain identical text.

Two capture-fixture failures are preserved in logs: the first incorrectly
required two visible rows even when legitimate folding retained one; the second
used a different existing overflow message key, whose Urdu wording differed.
The final rig checks visible rows and the actual production overflow resolver.
No product code was changed to satisfy these harness mistakes.

**Native Urdu/Hindi font shaping remains UNVERIFIED.** These are headless
Textual SVG/librsvg captures; Urdu's degraded shaping/bidi appearance is visible
and is not recast as native terminal proof. Independent code, QA and design
rounds still belong to Root on one pushed head. No new interaction was added;
UX review is not requested merely for label/ink projection.

## Artifacts and reproducibility

`rendered-frames.zip` preserves all 48 SVGs, 48 PNGs, per-frame capture geometry,
contact sheets and proof JSON; every member hash is in `manifest.json`, and the
archive was read back and CRC/hash checked. Representative full-size PNGs and
both contacts are also directly readable beside this file.

From this worktree, using a new scratch HOME/config and a new output directory:

```sh
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
 PATH="$PATH" TERM=xterm-256color LOP_LANG=en \
 LOCAL_OPERATOR_NO_NOTIFICATIONS=1 LOCAL_OPERATOR_NO_DESKTOP_LAUNCH=1 \
 LOCAL_OPERATOR_NO_TERMINAL_TITLE=1 LOCAL_OPERATOR_NO_SHIMMER=1 \
 PYTHONPATH="$PWD" BADGE_CAPTURE_DIR="$OUT" \
 ~/local-operator/.venv/bin/python \
 docs/pr-evidence/agent-badges/product/capture_agent_facts.py.txt
# After the isolated App exits, rasterize with the existing normal-HOME tool:
rsvg-convert "$OUT/production-fr-settled.svg" -o "$OUT/production-fr-settled.png"
```

Generated source was regenerated normally; `generate.py --check` reported
**201 artifacts byte-identical**, saved in its log. The combined initial normal
`check.py --audit-base` invocation exceeded its 180-second tool budget after
both generation steps had succeeded; no success is claimed for that check.
The separate frozen-base checker also timed out at 120 seconds while waiting
in `ledger.validate_base` for a scoped `git ls-tree`; its 45-second stack is
archived. A direct invocation of the same Git read succeeded (59,825 bytes),
so the cause of the enclosing checker stall is unresolved. Catalogue-only
validation passed with zero problems. See `checker-qualification.txt`: the
full base-aware checker remains a qualification gap, not an assumed pass.
No baseline allowance growth or checker bypass was used.

Coder: actor `05ab22c0ed05`, actual `anthropic/claude-opus-5-5`, no fallback.
The independent copy author/A/B models are their recorded GPT quota fallbacks,
not this coder's model; no self-translation or self-audit occurred.

# Agent Hub auto-update and three-way resolution — design

Status: implemented (pull side), 2026-09-29. Author: architect (lopdev); implementation: coder (lopdev).
Worktree base: `origin/main` c67fb53ff. Every `path:line` below was read in that tree; line
numbers describe the base, not the implementation. Deviations forced by code reality are
listed in "Implementation notes" at the end.

Two parts:

- **Part A — Resolution semantics contract.** SHARED with the push-side half
  (radientdev session e6e2e7693a7b). Direction-agnostic: the same rules decide a
  pull merge and a push merge. Anything in Part A changes only by agreement of both halves.
- **Part B — Pull-side design.** Interfaces, storage, runner, surfaces, UI brief,
  guides, tests, risks.

## Contents (skeleton)

- Part A — Resolution semantics contract
  - A1 Vocabulary
  - A2 The three-way model and where the baseline lives
  - A3 Segmentation (normative)
  - A4 Presence analysis and the decision table
  - A5 Reductions (shortened, not deleted)
  - A6 Rewrite-to-combine and its validator
  - A7 Change classification and per-field rules (agents, teams)
  - A8 Output shape: merged text + provenance
  - A9 What the user is told
  - A10 Shared repro shapes
- Part B — Pull-side design
  - B0 What is wrong today (findings that shape the design)
  - B1 Teams provenance and the team arm
  - B2 The merge engine (`local_operator/hub_sync/merge.py`)
  - B3 Auto-update: settings and the check runner
  - B4 Availability / failure state store
  - B5 Surfaces: routes, CLI, tool
  - B6 UI brief
  - B7 Guides
  - B8 Test and evidence plan
  - B9 Risks and open questions

---

# Part A — Resolution semantics contract (SHARED with the push side)

Scope: how two divergent copies of the same hub-linked item are reconciled, in
EITHER direction. **Pull** = the hub copy (remote) arrives at this machine; the
merged text is written locally. **Push** = the local copy goes to the hub; the
merged text is written to the hub (and back locally). The rules are identical;
only the write target differs. Anything below marked *(normative)* is what the
two implementations must agree on byte-for-byte; the rest is guidance.

**Design consequence stated up front:** the contract is split into a
**deterministic core** (segmentation, alignment, presence analysis, change
classification, output validation) and an **LLM layer** (rewrite-to-combine of
regions that are genuinely in conflict). The core is model-free, so both
directions test against the same vectors (A10) with no model in the loop. The
LLM can only ever *propose* text; the core *accepts or rejects* it. That is what
makes "never blindly overwrite either side's intentional edits/removals" a
checkable property rather than a prompt instruction.

## A1 Vocabulary

| Term | Meaning |
|---|---|
| **B** (baseline) | The text of the field as it was at the last successful pull OR publish — the last moment local and hub were known identical. |
| **L** (local) | The field's current text on this machine. |
| **R** (remote) | The field's current text on the hub, freshly fetched. |
| **field** | One mergeable unit: an agent's `instructions` / `description`; a team's `name`, `description`, `manager`, `members`, `instructions`, `project` (A7). |
| **region** | A heading section of a markdown field (A3). The unit of *reporting*. |
| **atom** | The smallest unit of presence analysis inside a region: a list item, a sentence, a fenced code block, a table row (A3). The unit of *decision*. |
| **triple** | `(b, l, r)` — the aligned versions of one atom in B, L, R; any may be absent (∅). |
| **intentional removal** | b present, and ∅ or a strict shortening on exactly one side. |

## A2 The three-way model and where the baseline lives *(normative)*

Two-way comparison (L vs R) cannot say who removed what: an atom in R and not in L
is either "remote added it" or "local deleted it", and the two demand opposite
outcomes. B decides it. **Every merge takes B as an input; a merge with no B is
degraded (A2.3) and never auto-applies.**

### A2.1 What exists today (verified)

- Agents record only a **fingerprint** of B, not B: `hub_sha256:<hex>` over
  `json.dumps([instructions.strip(), description.strip()])`
  (`local_operator/agents.py:574-590`, marker prefix `agents.py:499`, stamped at
  import by `_stamp_hub_provenance` `agents.py:2446-2500`, refreshed by
  `_apply_hub_update` `agents.py:2996-3041`). A fingerprint answers "is L still
  equal to B?" (`sync_hub_agents` `agents.py:3136-3140`) but cannot answer *what B
  said*, so it cannot attribute a removal. Three-way needs **B's text**.
- Teams record nothing: `import_hub_team` (`teams.py:1598`) mints a fresh uuid
  and stores no hub id, no fingerprint (`HubTeamImport`, `teams.py:618-632`).
  `Team` has no tags (`teams.py:202-232`), and `team.yml` is
  `model_dump(exclude={"instructions","project"})` (`teams.py:1430`).
- A team row is a DIRECTORY that `_save_team_locked` replaces whole via
  `_swap_row_directory_locked` from a staging dir holding exactly `team.yml`,
  `instructions.md`, `project.md` (`teams.py:1417-1500`, `_write_row_files`
  `teams.py:717-757`). **Any extra file placed inside a team row directory is
  destroyed by the next edit.** So a team baseline cannot live in the row.

### A2.2 Where B lives *(normative for both directions)*

A **baseline record** per linked item, stored OUTSIDE the item's own directory,
under the config root: `<config_dir>/hub/baselines/<kind>-<local_id>.json`
(`kind` ∈ `agent`, `team`; `local_id` the registry uuid; `config_dir` from the
`ConfigManager`, never `Path.home()` — AGENTS.md "Isolating a run"). Atomic
write (temp + `os.replace`, the `monitors/state.py:155-175` shape). Schema:

```json
{
  "schema": 1,
  "kind": "agent",
  "local_id": "<uuid>",
  "hub_id": "<marketplace agent id | hub team id>",
  "tenant_id": "<org tenant id | null for public>",
  "fingerprint": "<sha256 hex of canonical fields, A2.4>",
  "fields": { "instructions": "…B text…", "description": "…B text…" },
  "recorded_at": "2026-09-29T12:00:00Z",
  "recorded_by": "pull | publish | merge-pull | merge-push | adopt"
}
```

Writers (the ONLY writers): pull/import, publish, and a successful merge apply
(either direction). After any of these, **B := the text now identical on both
sides**.

> **Amended after the pull-side implementation (Implementation note 9): the
> "identical" clause holds only when no local edit survives the merge.** After a
> merge that KEEPS local edits (or local deletions) no text is identical on both
> sides. The rule that is actually normative, and that the shared vectors encode, is
> **B := the REMOTE text that was integrated** (for a pull-merge) and **B := the
> LOCAL text that was published** (for a push-merge). Recording the merged result
> instead would make every surviving local edit read as "unchanged since base" and
> let the next hub edit to that region overwrite it. When L == R (the common
> no-conflict case) the two formulas coincide, which is why the letter of the
> sentence above was never contradicted by a vector. Agents keep their existing `hub:`/`hub_sha256:` tags too (they remain
the cheap "is this row linked/clean?" probe and are read by the CLI/tool today);
the baseline record adds the text. The tag and the record must agree; on
disagreement the record's fingerprint wins and the tag is repaired at the next
write (one derivation: `fingerprint_of_fields`, A2.4).

The record is **not** part of an export/publish bundle. Because it lives outside
the agent directory it is automatically absent from `export_agent_archive`
(`agents.py:2637-2650` walks only the agent dir) — no `_EXPORT_SKIP_NAMES`
(`agents.py:2513`) change needed. A published archive can therefore never plant a
baseline (same trust rule as `strip_provenance_tags` `agents.py:519-541`).

### A2.3 Missing or untrusted baseline (existing pulls; conservative rules) *(normative)*

- **Agents with a valid `hub_sha256` and no record**: if
  `fingerprint(L) == hub_sha256`, L still equals B, so **B := L** (exact, not a
  guess) — write the record lazily on first check. This covers every agent
  pulled before this feature that has not been edited.
- **Agents with a `hub_sha256` that does NOT match L** (edited since pull) and
  **teams with no link at all**: B is *unknown* (`baseline: "unknown"`). Rules:
  1. Check-only. Auto-update NEVER applies (state `available`, reason
     `baseline-unknown`). The indicator still shows.
  2. If `L == R` (after normalization): adopt — write B := L, state
     `up-to-date`.
  3. Otherwise a manual apply runs the **two-way conservative merge**: L-only
     atoms kept; R-only atoms are *offered* (`provenance: "taken-remote"`,
     warning `baseline-unknown: a deliberate local removal cannot be
     distinguished from a remote addition`); nothing is deleted from L; atoms
     that differ go to conflict resolution (A6). The apply requires the caller to
     pass `acknowledge_unknown_baseline=true` (routes/CLI expose it as
     `--accept-unknown-baseline`); without it the result is `needs-review`.
- **Teams**: existing local teams that were pulled before this feature have no
  `hub_id` link (the row never stored it). They are treated as unlinked and are
  never touched; a user can link one explicitly (`lop teams link`, B5.4) which
  records B := current L only after showing R and requiring `L == R` or an
  acknowledgement (same unknown-baseline rule).

### A2.4 Canonical fingerprint *(normative)*

Text normalization `norm(t)`: CRLF/CR → LF; strip trailing whitespace on each
line; collapse ≥2 consecutive blank lines to one; strip leading/trailing blank
lines. Agents: `fp = sha256(json.dumps([norm(instructions), norm(description)],
ensure_ascii=False, separators=(",", ":")))` — identical to today's
`hub_fingerprint` modulo `norm` being applied instead of bare `.strip()`; to keep
every existing marker valid, the comparison first tries the legacy `.strip()`
form and accepts either (one-way compat; new writes use `norm`). Teams:
`fp = sha256(json.dumps({"description":…, "manager":…, "members":[{"role","kind",
"count"}…sorted by (kind, role.casefold())], "instructions":…, "project":…},
sort_keys=True, ensure_ascii=False, separators=(",",":")))` over `norm`'d texts.
`name` is deliberately excluded (identity, A7). `version` is excluded:
`hub_team_document` always sends the constant `"1.0.0"` (`teams.py:560`), so it
carries no change signal — remote change detection is content-based.

## A3 Segmentation *(normative)*

Deterministic, no model, no locale dependence.

1. Normalize with `norm` (A2.4).
2. **Regions.** Split at ATX heading lines (`^#{1,6}[ \t]+\S`), excluding lines
   inside fenced code (```` ``` ```` / `~~~`). Text before the first heading is
   region `""` (preamble). `region.key = casefold(collapse_ws(heading text
   without #s)) + "#" + ordinal` where `ordinal` counts prior identical
   normalized headings (so duplicate headings stay distinct and stable).
   Heading level is part of the region text, not the key: a demotion
   (`##`→`###`) is a *change* to the region, not a delete+add.
3. **Atoms inside a region body**, in order:
   - a fenced code block = one atom (atomic; never split);
   - a list item (`-`, `*`, `+`, `1.`) including its indented continuation lines
     and nested items = one atom;
   - a table row = one atom;
   - a paragraph = one atom **per sentence** (split on `(?<=[.!?])\s+(?=[A-Z0-9"'(\[`])`
     outside inline code/links); a paragraph with no terminal punctuation is
     one atom.
   `atom.norm = collapse_ws(lowercase-insensitive? NO)`: atoms compare after
   `collapse_ws` only — case is preserved (a capitalization change is an edit).
4. **Scalar fields** (`manager`, `name`, one-line `description` ≤ 1 line): a
   single atom, no regions.
5. **Roster** (`members`): the atoms are slots keyed `(kind, casefold(role))`
   (A7); there is no text segmentation.

## A4 Alignment and presence analysis *(normative)*

### A4.1 Alignment

Regions align by `key`. Within an aligned region, atoms align B↔L and B↔R
independently (B is the pivot; L↔R is derived through B, plus a direct pass for
atoms absent from B):

1. Exact match on `atom.norm`, left-to-right, each atom used once.
2. Then, for the atoms still unmatched, best match by
   `difflib.SequenceMatcher(None, a_tokens, b_tokens).ratio()` on
   whitespace tokens, accepting pairs with ratio ≥ **0.60**, greedy by
   descending ratio, ties broken by lowest index. Pairs matched in step 2 are
   "*modified*", in step 1 "*unchanged*".

   > **Amended (Implementation note 10): the similarity function is
   > `max(token ratio, character ratio, containment)`, not the bare whitespace-token
   > ratio.** Threshold `0.60` and the sentence splitter are unchanged and remain part
   > of the contract. With whitespace tokens alone `x.` vs `x2.` scores 0 and `Be
   > brief.` vs `Be brief and cite sources.` scores 0.29, so vectors V01, V06, V07 and
   > V08 could never align and every edit would read as delete+add. The reference
   > implementation is `hub_sync/segment.py::similarity`; the push side imports it,
   > it does not re-implement it.
3. A region present in only some of B/L/R aligns as a region of all-∅ on the
   missing sides (so a deleted region is a set of removed atoms + a removed
   heading).
4. L↔R direct pass: an atom absent from B and present in both L and R aligns
   with the same rules (both sides *added* similar text independently). Ratio
   ≥ 0.60 → one triple `(∅, l, r)` that is a conflict if `l.norm != r.norm`,
   otherwise identical.

`0.60` and the sentence splitter are part of the contract: both directions must
use the shared implementation (`local_operator/hub_sync/segment.py`, B2.1), not
re-implement it.

### A4.2 Per-triple decision table

`Δl` = l differs from b (modified), `Δr` likewise. "=" = normalized-equal.

| b | l | r | Outcome | Provenance | Note |
|---|---|---|---|---|---|
| present | = b | = b | keep | `unchanged` | |
| present | Δ | = b | take l | `kept-local` | local edit survives |
| present | = b | Δ | take r | `taken-remote` | remote edit lands |
| present | Δ | Δ, l = r | take either | `kept-local` (identical) | |
| present | Δ | Δ, l ≠ r | **conflict** | `combined` or `unresolved` | A6 |
| present | ∅ | = b | stay removed | `removal-honored` (`removed_by: local`) | **no regrow** |
| present | = b | ∅ | remove | `removal-honored` (`removed_by: remote`) | remote deletion applies |
| present | ∅ | ∅ | stay removed | `removal-honored` (`removed_by: both`) | |
| present | ∅ | Δ | **removal-vs-edit conflict** | `unresolved` | never auto-resolved (A4.3) |
| present | Δ | ∅ | **removal-vs-edit conflict** | `unresolved` | never auto-resolved (A4.3) |
| ∅ | l | ∅ | keep | `kept-local` | added locally |
| ∅ | ∅ | r | add | `taken-remote` | added remotely |
| ∅ | l | r, l = r | keep one | `kept-local` (identical) | |
| ∅ | l | r, l ≠ r (aligned ≥0.60) | **conflict** | `combined` or `unresolved` | A6 |
| ∅ | l | r, unaligned | keep both | `kept-local` + `taken-remote` | order: L's position, R's atom placed after the nearest aligned predecessor |

The `present/∅/Δ` rows are the "intentional removal" rows: the other side's
edit to a thing you deleted is **not** applied silently and **not** discarded
silently.

### A4.3 Removal-vs-edit and precedence *(normative)*

- **Default: `needs-review`.** The item is not written, the conflicting atoms are
  listed (both texts). Auto-update never resolves this row. This is symmetric:
  a remote deletion of an atom the user edited, and a local deletion of an atom
  the hub edited, are equally "someone's intentional work would be lost".
- **Explicit precedence** (`prefer`) is a caller argument, never a default:
  `prefer="local"` → the local side of every unresolved triple wins (local
  removal stands; local edit stands); `prefer="remote"` → the reverse. Under
  `prefer`, the losing side's text is preserved in the report
  (`regions[].dropped`) and in the pre-apply backup (A9.3).
- `--force` today means "replace local with the hub copy wholesale"
  (`sync_hub_agents` `agents.py:3141-3154`, `ProfileSync.force`
  `desktop_profiles.py:45-56`, CLI `--force` `cli.py:488-492`). That is exactly
  `replace` = a whole-field `prefer="remote"` with **no alignment at all**; its
  fate is decided in B5.5. In the contract it is the explicit operation
  `replace(side)`, reported as one region `{provenance: "taken-remote",
  note: "replaced"}` plus the full replaced text (as `replaced_instructions`
  does today, `agent_sync.py:140-150`).

## A5 Reductions (shortened, not deleted) *(normative)*

A deliberate *shortening* is an intentional removal too. Atom-level alignment
already covers dropped sentences/items. The remaining case is **an atom rewritten
shorter** (modified pair, ratio ≥ 0.60, `len(tokens(l)) ≤ 0.85·len(tokens(b))`
and `tokens(l)` a subsequence of `tokens(b)` after casefold) — classified
`reduced`. Rules:

- `reduced` is a `Δ` for the table above (a local edit), with the extra
  invariant that the *removed tokens* `tokens(b) − tokens(l)` (as a token
  multiset difference in order) are recorded as `removed_spans`.
- **No-regrow invariant** (validator V3, A6.2): the merged atom may not contain a
  contiguous run of ≥ 4 tokens (or the whole span if shorter) from a side's
  `removed_spans` **unless** the *other* side changed those very tokens. This is
  what lets a remote clause-rewrite land inside a locally shortened sentence
  without dragging the deleted clause back.
- A region that shrank to its heading only (all atoms removed, heading kept) is
  "emptied": its atoms are removals; the heading stays (`kept-local`).
- Sizes: a merge whose output is < 50% of `max(len(L), len(R))` chars while
  neither side is that short raises warning `large-shrink` (report only; not a
  refusal — deliberate cuts are legitimate).

## A6 Rewrite-to-combine *(normative for what is permitted; the prompt is B2)*

Permitted: for a **conflict triple** (A4.2 rows `Δ/Δ` differing, and `∅/l/r`
differing) — and only for those — the resolver may **rewrite the prose** so that
one merged atom/paragraph carries both sides' improvements, when meaning is
preserved and removals are honored. It must not touch atoms the table already
decided (`unchanged`, `kept-local`, `taken-remote`, `removal-honored`): those are
emitted verbatim by the deterministic core. The LLM sees only the conflict
groups (each with its B/L/R atoms and immediate neighbours as read-only context),
never the whole document to freely re-author.

### A6.1 What the resolver returns *(normative)*

Per conflict group `g`: `{"group": g, "text": "<merged text>", "covers": [ids…],
"drops": [ids…], "rationale": "≤200 chars"}` where ids are the ids of L/R/B atoms
in the group (`b1`,`l2`,`r1`…). `covers` = source atoms whose content the text
carries; `drops` = source atoms deliberately left out and why (must be
justified by removal-honoring, otherwise V4 rejects).

### A6.2 Validators *(normative; deterministic; run on every proposal, LLM or not)*

- **V1 scope**: output contains exactly one result per conflict group, none for
  decided atoms.
- **V2 coverage**: every atom of the group that is a `Δ`/added atom on either
  side is in `covers` OR `drops`; a `Δ` atom's key tokens (numbers, URLs,
  identifiers `[A-Za-z_][\w.-]*\(\)?`, code spans, `{{…}}` template tokens)
  appear verbatim in `text` when it is in `covers`.
- **V3 no-regrow**: no side's `removed_spans` (A5) and no atom of a
  `removal-honored` row reappears in `text` (≥ 4-token contiguous run, or
  ≥ 0.85 SequenceMatcher ratio to a removed atom).
- **V4 drops justified**: an atom in `drops` must be a b-derived atom that one
  side removed/shortened; a `Δ` or added atom may never be dropped.
- **V5 size**: the assembled field respects the field's cap (agents:
  `MAX_INSTRUCTIONS_CHARS = 8_000`, `agent_profiles.py:111`; teams:
  `MAX_TEAM_INSTRUCTIONS_CHARS = 32_768`, `teams.py:89`; team description
  ≤ `HUB_TEAM_DESCRIPTION_MAX_CHARS = 2000`, `teams.py:384`).
- **V6 structure**: fenced blocks balanced; headings unchanged (the LLM may not
  rename/reorder regions).

A proposal failing any V-check is re-asked once with the violations listed
(B2.5); a second failure → the group is `unresolved` (`needs-review`). A
deterministic no-model fallback exists (B2.6): when no model is reachable, a
conflict group whose two sides are **prefix/suffix-additive** to each other (one
contains the other) resolves to the longer side; anything else stays `unresolved`.

### A6.3 Precedence summary

1. Table rows decide everything but conflicts (A4.2). No precedence involved.
2. Conflicts → validated combine → else `unresolved` → `needs-review`.
3. `prefer` overrides only `unresolved` groups, only when the caller passed it.
4. There is no implicit "local wins" or "remote wins" anywhere.

## A7 Change classification and per-field rules *(normative)*

Item-level classification, computed from field-level results:
`unchanged` (all fields `unchanged`), `remote-only` (no field has a local edit),
`local-only` (no field has a remote edit — nothing to pull; push candidate),
`diverged` (some field edited on both sides or removal-vs-edit), `link-lost`
(remote 404/not visible), `baseline-unknown` (A2.3).

### Agents (hub agent archive: `system_prompt.md` + `agent.yml`; read by
`_read_hub_profile_from_zip` `agents.py:2940-2978`)

| Field | Rule |
|---|---|
| `instructions` | markdown; full A3–A6 pipeline. Cap A6.2-V5. |
| `description` | short text; A3 as scalar-or-sentences; one side changed → take it; both changed differently → conflict → A6 (short) → else `unresolved`. |
| `name`, `tags`, `categories` | **not merged, never touched** by pull (identity/organization is local; `tags` carries the hub markers themselves). A remote rename is reported as `note: remote-renamed` only. |
| `model`, `hosting`, sampling fields, `security_prompt`, `current_working_directory` | **never travel** — `import_agent` deletes `hosting`/`model` from the archive (`agents.py:2257-2262`) and `last_message` is blanked on export (`agents.py:2645-2650`). Not merged, not compared. |

The comparison set for an agent is exactly the pair covered by
`hub_fingerprint`: `(instructions, description)` (`agents.py:574`).

### Teams (document: `hub_team_document` `teams.py:536-575`; import `teams.py:1598`)

| Field | Rule |
|---|---|
| `name` | **identity; never merged.** A pull never renames a local team. The local name is the sanitized slug (`_local_name_for_published` `teams.py:577`); a remote rename is reported `remote-renamed` and changes nothing. Push: the hub's `name` is its uniqueness key (`name_taken`); push reports, never silently renames (push half decides). |
| `description` | short text; scalar rule as agents' `description`. Cap 2000. |
| `manager` | **scalar**, no LLM: three-way on the whole value (one side changed → take it; both differ → `unresolved`, needs-review). Non-empty required (`teams.py:254`), ≤ 128 chars. |
| `members` (roster `{role, kind, count}`) | **set merge by key `(kind, casefold(role))`**: slot added on one side → added; slot removed on one side and unchanged on the other → **removal-honored** (no regrow); `count` changed on one side → take it; `count` changed both sides differently → `unresolved`; slot removed vs count-edited → `unresolved` (removal-vs-edit). Order: L's order; R-added slots appended in R order. Caps: ≤ 64 slots, count 1–16, role ≤ 128, kind ≤ 32 (`teams.py:386-389`). An unknown `kind` from the hub is coerced to `agent` exactly as import does (`teams.py:1650-1660`). A slot naming a role/team nobody has installed is a **warning**, not a refusal (a team naming a missing role still launches; `guides/teams/GUIDE.md` "Use the agent tool…"). |
| `instructions` (collaboration brief) | markdown; full A3–A6 pipeline; cap 32 768. |
| `project` (project brief) | markdown; full pipeline; cap 32 768. |
| `id`, `created_date` | local; never travel. |
| `version` | constant `"1.0.0"` (`teams.py:560`); not a signal, ignored. |

**Refused (always, with the hub's own rule vocabulary via `TeamDocumentError`
`teams.py:392-415`)**: an output that would violate a hub or local invariant
(empty manager; > 64 slots; count out of 1–16; instructions/project over cap;
name rule). Refusal = item state `failed` with class `merge-refused`; **nothing
is written**.

**Refused for auto-apply (needs a human)**: any `unresolved` triple;
`baseline-unknown`; `link-lost`; a remote document whose `tenant_id` differs from
the baseline's (`teams.py`/routes: `agents.py:1571-1595` already 409s on a tenant
mismatch — same rule).

## A8 Output shape *(normative)*

```json
{
  "kind": "agent|team",
  "field": "instructions",
  "outcome": "unchanged | merged | needs-review | refused",
  "merged": "<final text; equals L when outcome != merged>",
  "regions": [
    {
      "id": "r3",
      "heading": "## Review rules",
      "provenance": "unchanged | kept-local | taken-remote | combined | removal-honored | unresolved",
      "removed_by": "local | remote | both | null",
      "atoms": [{"id":"a7","provenance":"…","removed_by":null,"note":""}],
      "base": "<region text or null>", "local": "<…|null>", "remote": "<…|null>",
      "result": "<text or null>",
      "dropped": "<loser text when a prefer/replace discarded it, else null>",
      "note": ""
    }
  ],
  "warnings": ["baseline-unknown", "large-shrink", "missing-role:reviewer"],
  "counts": {"unchanged":0,"kept-local":0,"taken-remote":0,"combined":0,"removal-honored":0,"unresolved":0},
  "engine": {"mode": "deterministic | llm | replace", "model": "provider/model|null", "attempts": 0, "chunks": 0}
}
```

`provenance` at region level is the *strongest* of its atoms in the order
`unresolved > combined > removal-honored > taken-remote > kept-local > unchanged`
(so a report never hides an unresolved atom behind a tidy region). An item's
`ItemMergeReport` is `{kind, local_id, name, hub_id, outcome, fields:
[MergeResult…], applied: bool, backup: "<path>|null"}`. Team `members` is a
`MergeResult` with `field:"members"` and one "region" per slot (`heading:
"kind/role"`, `base/local/remote/result` = `{"count":n}`).

## A9 What the user is told *(normative wording shape; exact strings in B5)*

1. **Applied merge**, one line + region roll-up (agent example): `coder: updated
   from the hub — 2 sections taken from the hub, 1 kept yours, 1 combined, 1
   removal honored (you removed "## Legacy rules"; the hub still had it).` The
   removals are always *named* (heading or first 60 chars), never only counted.
2. **needs-review**: `coder: the hub and your copy both changed "## Review
   rules" — not applied. Review, then apply with --prefer local|remote (or edit
   and retry).` Lists each unresolved region with both texts.
3. **Recoverability**: before any write, the pre-apply local field texts are
   written to `<config_dir>/hub/backups/<kind>-<local_id>-<UTC ts>.json`
   (schema: the `fields` object of A2.2 + `reason`); the last 5 per item are
   kept. The report returns `backup` and (for CLI/tool text) echoes the replaced
   text exactly as `agent_sync.render` does for `replaced_instructions`
   (`agent_sync.py:140-150`).
4. **Refused**: names the field and the rule in the hub's vocabulary
   (`teams: members must hold at most 64 items (submitted 70)`).
5. **Never** the words "overwritten"/"replaced" for a merge; those are reserved
   for `replace` (B5.5), which always echoes the replaced text.

## A10 Shared repro shapes *(normative test vectors)*

Vectors live at `tests/fixtures/hub_merge/vectors.json` and are consumed by both
directions' unit tests (`direction: "pull"|"push"` only flips which side the
result is written to; **the `expected` block is identical for both**). Each
vector: `{id, field_kind, B, L, R, prefer, model_stub, expected: {outcome,
merged, provenance_by_region, removed_by, warnings}}`. `model_stub` is a scripted
resolver output (or `null` = no model) so the LLM layer is deterministic. Minimal
set (markdown fields use `##` regions; `~` = ∅):

| id | B | L | R | expected |
|---|---|---|---|---|
| V01 remote-only edit | `## A\nx.` | same | `## A\nx2.` | merged=R; A=`taken-remote` |
| V02 local-only edit | `## A\nx.` | `## A\nx2.` | same as B | outcome `unchanged`-for-pull (nothing to pull); push: merged=L |
| V03 disjoint edits | `## A\nx.\n## B\ny.` | edit A | edit B | both kept; A=`kept-local`, B=`taken-remote` |
| V04 **local deletes a section; remote untouched** | `## A\nx.\n## B\ny.` | `## A\nx.` | = B | merged=`## A\nx.`; B=`removal-honored`,`removed_by:local` (**must not regrow**) |
| V05 **remote deletes a section; local untouched** | 〃 | = B | `## A\nx.` | merged drops B; `removal-honored`,`removed_by:remote` |
| V06 **local deletes, remote edits it** | 〃 | `## A\nx.` | `## A\nx.\n## B\ny2.` | `unresolved` (removal-vs-edit); outcome `needs-review`; with `prefer=local` → merged=`## A\nx.`, `dropped`=`## B\ny2.` |
| V07 **local shortens a sentence; remote edits elsewhere** | `## A\nUse tools carefully and never run destructive commands without asking.` | `## A\nUse tools carefully.` | `## A\nUse tools carefully and never run destructive commands without asking.\n## C\nnew.` | merged=`## A\nUse tools carefully.\n## C\nnew.`; A=`kept-local` (reduced); C=`taken-remote`; the dropped clause absent |
| V08 **both edit the same atom** | `## A\nBe brief.` | `## A\nBe brief and cite sources.` | `## A\nBe brief and use plain words.` | `combined`; stub returns `Be brief, cite sources, and use plain words.` covers=[l1,r1]; V2 passes. No-model: `unresolved` |
| V09 **combine that regrows a removal is rejected** | `## A\nx. old-rule.` | `## A\nx.` | `## A\nx-edited. old-rule.` | stub proposes text containing `old-rule.` → V3 rejects → retry → `unresolved` |
| V10 both add different atoms | `## A\nx.` | `## A\nx.\ny.` | `## A\nx.\nz.` | both kept; y `kept-local`, z `taken-remote` |
| V11 identical edits both sides | `## A\nx.` | `## A\nx2.` | `## A\nx2.` | merged=`x2.`; `kept-local` (identical) |
| V12 **baseline unknown** | `~` (no B) | `## A\nx.\n## B\ny.` | `## A\nx.\n## B\ny.\n## C\nz.` | outcome `needs-review` (auto); with acknowledge: C `taken-remote` + warning `baseline-unknown`; nothing deleted |
| V13 roster: remote adds slot, local removes another | members B=[coder,reviewer×2], L=[coder], R=[coder,reviewer×2,qa] | | | merged=[coder,qa]; reviewer=`removal-honored`(local); qa=`taken-remote` |
| V14 roster: count edited both | B reviewer×2 | ×3 | ×4 | `unresolved` |
| V15 manager: both changed | B `manager` | `lead` | `director` | `unresolved` |
| V16 team caps | R adds 70 slots | | | `refused`, rule text `members must hold at most 64 items (submitted 70)` |
| V17 heading demotion | `## A\nx.` | `### A\nx.` | = B | A=`kept-local` (region key equal; heading level is content) |
| V18 CRLF/whitespace-only remote | B `x.\n` | = B | `x.\r\n\r\n` | `unchanged` |
| V19 fence atomic | fenced block edited on one side | | | never split; taken/kept whole |
| V20 replace | any | any | any | engine `replace`, one region, `replaced` text echoed |

**Agreement test**: `tests/unit/hub_sync/test_vectors.py` parameterises over the
file for `direction in ("pull","push")` against the shared `hub_sync.merge`
core and asserts `expected` for both. If the push implementation lives in a
different repository/process it must import the same fixture file (vendored or
fetched) and pass it unchanged; a vector edit needs both sides' sign-off.

---

# Part B — Pull-side design

## B0 What is wrong today (findings that shape the design)

1. **Sync is refuse-or-clobber.** `sync_hub_agents` (`agents.py:3044-3180`)
   returns `up-to-date`, `updated`, `diverged` or `unavailable`. `diverged` (hub
   changed AND local edited, detected by fingerprint at `agents.py:3136-3140`)
   is refused unless `force`, and `force` overwrites the whole row
   (`_apply_hub_update` `agents.py:2996-3041`). There is no third path, so a user
   who edited an agent and whose hub copy later improved must choose "lose the
   hub improvement" or "lose my edit". The design adds a merge between the two.
2. **The baseline is a hash, not text** (A2.1) — a removal cannot be attributed.
3. **Teams have no provenance or check path at all.** `import_hub_team`
   (`teams.py:1598-1692`) drops the hub id; `Team` has no tags; a row is a
   directory the registry swaps wholesale (`teams.py:1474-1520`). The
   agent-style `hub:` tag model cannot be mirrored inside the row.
4. **Nothing runs periodically.** `sync_hub_agents` says "NOT called on boot"
   (`agents.py:3053-3057`); the three callers are the CLI
   (`cli.py:7926-7960`), the `agent` tool (`tools/agent_tool.py:1020-1075`) and
   the desktop route (`routes/desktop_profiles.py:154-205`). All three go through
   ONE coordinator, `agent_sync.sync_agent_profiles` (`agent_sync.py:190-225`),
   plus `sync_payload` for JSON — the "one derivation per surface" precedent we
   extend, not replace.
5. **The hub arm downloads the whole archive to read two strings**
   (`_fetch_hub_profile` `agents.py:2981-2994`, `_read_hub_profile_from_zip`
   `agents.py:2940-2978`). Fine for a check on demand; for a periodic check over
   N items it is N zip downloads, so the runner must bound and spread them (B3).
6. **Credentials differ by family.** Public agent download is anonymous
   (`download_agent_from_marketplace(..., require_api_key=False)`
   `clients/radient.py:648-684`); org agents and ALL team calls need the
   signed-in *person's* OAuth bearer, never a tenant API key
   (`resolve_radient_oauth_access` `providers/radient_credentials.py:183-230`;
   route helper `_org_radient_credentials` `routes/agents.py:~820-905`; CLI
   `_resolve_org_client` `cli.py:8174-8218`). The existing sync path resolves a
   generic key via `resolve_hub_client` (`agent_sync.py:238-262`) and uses the
   ANONYMOUS download — so an org-pulled agent (which returns 404 anonymously,
   `radient.py` docstring 660-665) silently reports `unavailable` on sync today.
   Pull-side must record `tenant_id` (baseline record, A2.2) and choose the
   OAuth client for org items. This is an existing gap the design closes.

## B1 Teams provenance and the team arm

### B1.1 Decision: link + baseline in a side store, not in the row

Options:

| Option | Verdict |
|---|---|
| Add `hub_id`/`hub_sha256` fields to `Team` (`team.yml`) | Works (team.yml is rewritten wholesale, so fields persist), but couples the registry model to hub sync, forces `TeamEditFields`/route/`network/definitions.py` (`_team_row_from_team` 475-495) to learn the fields or silently drop them on update, and still can't hold B's TEXT (briefs are separate files, capped 32 KB). |
| Marker file inside the row dir | Destroyed by the next `_save_team_locked` (`teams.py:717-757` writes exactly three files into staging then swaps). Rejected. |
| **Baseline record in `<config_dir>/hub/baselines/team-<id>.json` (A2.2)** | Chosen. Survives every edit, rename, and row swap because it is keyed by the stable team uuid (`update_team` keeps `id`, `teams.py:1318-1360`); holds B's text; not exported; not synced by mesh (`network/definitions.py` builds rows only from the model). One mechanism for agents and teams (agents additionally keep tags for compatibility). |

Consequences: (a) team delete must drop the record — `TeamRegistry.delete_team`
(`teams.py:1580`) stays untouched; instead `hub_sync.store.prune(kind, live_ids)`
runs at the start of each check and removes records whose local id no longer
exists (one derivation, no registry coupling). (b) A team *rename* is harmless
(keyed by id). (c) A mesh-synced team arrives as a new row with the origin's id
(`network/definitions.py:1335-1360`) and no record → unlinked → untouched, which
is correct (the other device owns the link).

### B1.2 Stamping at import

`TeamRegistry.import_hub_team(document)` (`teams.py:1598`) keeps its signature
and returns `HubTeamImport(team, renamed_from, invalid_name)`; the **stamp is done
by the two callers that hold the credential context**, through one helper, so
`teams.py` stays free of hub-sync imports (it is on the boot path):

```python
# local_operator/hub_sync/provenance.py
def record_team_pull(config_dir: Path, team: Team, document: Mapping[str, Any],
                     *, tenant_id: str) -> BaselineRecord: ...
```

Callers: `teams_pull_command` (`cli.py:8449-8490`, after `import_hub_team`) and
the pull route (`routes/agents.py:1521-1620`). `hub_id` = `document["id"]`,
`tenant_id` = `document["tenant_id"]` (both are in the `get_team` payload,
`radient.py:1049-1080`, and the route already checks `owner`
`routes/agents.py:1576-1594`). B := the fields of the *stored* row (re-read
after import so name normalization/caps are reflected — same rule as
`_stamp_hub_provenance`'s "describe the bytes a later re-fetch compares against"
`agents.py:2446-2470`). The pull's JSON result gains `hub_id` and `linked: true`.

Publish (push-side) writes/refreshes the same record with
`recorded_by:"publish"` (A2.2); the push team uses `publish_team_document`'s
result `{"team":{"id",…}}` (`radient.py:1081-1124`) as `hub_id`.

Agents: `_stamp_hub_provenance` (`agents.py:2446`) and `_apply_hub_update`
(`agents.py:2996`) additionally call `hub_sync.provenance.record_agent_baseline`
(same module), passing `tenant_id` when the pull was org-scoped. The
`download_agent_from_radient(..., require_api_key=…)` signature
(`agents.py:2396-2445`) gains `tenant_id: str | None = None` (default None =
public) so the stamp knows. Agent `hub:` marker validation stays
(`_HUB_ID_RE` `agents.py:507`).

### B1.3 The team arm — `check_hub_teams`

A peer of `sync_hub_agents`, in `local_operator/hub_sync/teams_arm.py`:

```python
def collect_team_links(config_dir: Path, registry: TeamRegistry) -> list[TeamLink]: ...

def check_hub_teams(
    registry: TeamRegistry, *, client_for_tenant: Callable[[str | None], Any | None],
    names: Sequence[str] | None = None,
) -> list[TeamCheck]: ...

@dataclass(frozen=True)
class TeamCheck:
    local_id: str; name: str; hub_id: str; tenant_id: str | None
    verdict: Literal["up-to-date","available","unavailable","unlinked-baseline-unknown"]
    remote_fingerprint: str | None
    remote: Mapping[str, Any] | None      # the fetched document (not persisted)
    reason: str = ""                      # failure class, B4.3
```

Per link: `client_for_tenant(tenant_id)` returns an OAuth client (or None →
`unavailable`/`no-credential`); `client.get_team(hub_id)` (`radient.py:1049`)
returns the full document *including briefs*, so a team check is ONE small JSON
GET (no archive). Compare `fingerprint_team(remote_doc)` (A2.4) to
`fingerprint_team(L)` and to the record's `remote_fingerprint_seen`:
`up-to-date` if `fp(R)==fp(L)`; else `available`. Fields compared: `description`,
`manager`, `members`, `instructions`, `project`. **`name` is never compared or
merged** (A7: local identity; a hub rename never renames the local row).
`version` and moderation fields are ignored. A 404 (`APIError.status_code==404`,
`radient.py:1056-1062`) maps to `unavailable/hub-item-missing` (removed from the
hub or membership lost — the two are indistinguishable by design) and is NEVER
treated as "the hub removed it, delete locally".

Applying a team merge goes through `TeamRegistry.update_team(local_id,
TeamEditFields(...))` (`teams.py:1318-1360`) — the existing locked, staged,
atomic path — with only the fields that changed, then refreshes the baseline.
`update_team` re-reads under the writer lock, which gives the *concurrent local
edit* guard (B4.3): the apply compares the row's fingerprint under the lock with
the fingerprint the merge was computed from and aborts with `concurrent-edit` if
they differ.

Roster caveat carried from A7: after a members merge, roles named in the roster
that are not installed locally produce the warning `missing-role:<name>` (not a
failure); `task(agent=…)` already falls back to packaged starters
(`guides/teams/GUIDE.md`, "A team that names a role nobody has installed still
launches").

## B1.4 Agent arm changes

`sync_hub_agents` keeps its name and verdict dataclass for the CLI/tool text,
but its decision core moves into the shared `hub_sync` package so agents and
teams share one classification: `hub_sync.check.classify(kind, B, L, R) ->
Classification` with `Classification.state ∈ {up-to-date, remote-only,
local-only, both-changed, baseline-unknown}`. `remote-only` ≡ today's `updated`
(clean fast-forward, no model needed); `both-changed` ≡ today's `diverged` and is
the case that now runs the merge engine instead of refusing. `local-only` (L≠B,
R==B) is `up-to-date` for a pull (nothing to fetch; the push side owns it). The
org gap of B0.6 is fixed by giving `sync_hub_agents` a
`client_for_tenant` callable instead of a single `radient_client` (the old
keyword stays as a shim for the tool/CLI callers until they are migrated in the
same PR).

## B2 The merge engine (`local_operator/hub_sync/merge.py`)

### B2.1 Package layout (one module per concern, no second derivation)

```
local_operator/hub_sync/
  __init__.py        # public re-exports only
  provenance.py      # baseline records: read/write/backup (A2.2), fingerprints (A2.4)
  segment.py         # A3 segmentation + A4 alignment (pure, no I/O, no model)
  merge.py           # A4-A8: deterministic core + resolver protocol + orchestration
  resolver.py        # LLM Resolver (B2.3-B2.6): model resolution, prompt, retry
  check.py           # classify(); agent arm + team arm (teams_arm.py) verdicts
  store.py           # availability/failure store (B4)
  runner.py         # periodic/on-start check+apply runner (B3)
```

`merge.py` imports nothing heavy: no `httpx`, no model stack, no registries —
the import-graph guard (`tests/unit/test_import_graph.py`, cited in
`server/features.py` module docstring) fails the build if `lop serve` boot pulls
`tools.builtin`; `resolver.py` imports the stream stack lazily inside its
functions for the same reason.

### B2.2 Public interface

```python
# merge.py
Provenance = Literal["unchanged","kept-local","taken-remote","combined",
                     "removal-honored","unresolved"]

@dataclass(frozen=True)
class FieldInput:
    field: str                      # "instructions" | "description" | "members" | ...
    kind: Literal["markdown","text","roster","scalar"]
    base: str | list | None         # None = baseline unknown for this field
    local: str | list
    remote: str | list

@dataclass(frozen=True)
class MergeOptions:
    prefer: Literal["none","local","remote"] = "none"   # conflict tiebreak, A4.3
    allow_llm: bool = True
    acknowledge_unknown_baseline: bool = False
    max_chars: int | None = None    # hard cap per A7 (hub/registry limits)
    resolver: "Resolver | None" = None

@dataclass(frozen=True)
class MergeResult:                  # == A8 JSON, dataclass form (+ .to_json())
    field: str; outcome: Literal["unchanged","merged","needs-review","refused"]
    merged: str | list; regions: tuple["RegionReport",...]
    warnings: tuple[str,...]; counts: Mapping[str,int]; engine: "EngineInfo"

def merge_field(inp: FieldInput, opts: MergeOptions = MergeOptions()) -> MergeResult:
    """Sync. Deterministic core; calls opts.resolver.resolve(...) ONLY for
    conflicts, and validates every proposal with validate_proposal (A6.2)."""

def replace_field(inp: FieldInput, *, take: Literal["local","remote"]) -> MergeResult:
    """The explicit --replace path (B5.5). One region, engine.mode='replace',
    dropped text echoed. Never reached by auto-update."""

# resolver.py  (the ONLY place a model is touched)
class Resolver(Protocol):
    def resolve(self, req: "ConflictRequest") -> "ConflictProposal": ...

@dataclass(frozen=True)
class ConflictRequest:
    field: str; heading: str | None
    base: str | None; local: str; remote: str
    removals: tuple["Removal",...]      # atoms that MUST NOT reappear (with who removed)
    keep_verbatim: tuple[str,...]       # atoms unchanged on both sides (context only)
    max_chars: int

@dataclass(frozen=True)
class ConflictProposal:
    text: str
    covers: tuple[str,...]              # atom ids from l/r this text claims to contain
    notes: str = ""

class ResolverError(Exception):
    cls: Literal["provider-error","model-unavailable","prompt-too-long",
                 "invalid-output","cancelled"]   # B4.3 failure classes
```

`merge_field` is synchronous and model-agnostic: the async model call is wrapped
by `LlmResolver.resolve` running `asyncio.run` in a worker thread
(`asyncio.to_thread` at the call site — the same shape every route uses,
`routes/desktop_profiles.py:190-195`), so the deterministic core stays trivially
testable with a scripted `Resolver` (A10 `model_stub`).

### B2.3 The canonical one-shot LLM path (verified)

There is no session-less utility helper today; three near-equivalents exist and
this design **reuses the lowest one rather than adding a fourth**:

| Precedent | What it does | Verdict |
|---|---|---|
| `ServerExecutor.invoke_model` (`server/utils/operator.py:575-630`) | Builds `AuthStore(config_dir=…)`, `create_stream_fn(auth_store, settings=…)`, a `ChatRequest(tools=[], tool_choice="none", replayable=True)`, drains `StreamTextDelta`, raises on `StreamEndEvent.error`, closes stream fn + store in `finally`. Used by inline-edit and speech. | Needs a `ModelConfiguration` from a whole `ServerOperator`; too heavy, but its **body is the template**. |
| `Session._one_shot_complete` / `complete_once` (`session/session.py:~15355-15560`) | Compaction and naming errands on a live session's stream fn; `purpose="compaction"|"naming"`, `isolated=True` for decoration. | Requires a live `Session`; the auto-update runner has none. |
| `compaction.api.summarize_messages(messages, complete_fn)` (`compaction/api.py:635`) | Takes an injected `complete_fn(system,prompt)->str`. | Confirms the codebase's own pattern: **inject a `complete_fn`**. |

So: `resolver.py` defines `async def complete_once(system, prompt, *, model:
ModelSpec, config_dir, settings, max_tokens, purpose="hub_merge") -> str`, a
~40-line lift of `invoke_model`'s body with these exact calls:

```python
from local_operator.harness.types import ChatRequest, Message, StreamTextDelta, StreamEndEvent
from local_operator.model.configure import create_stream_fn, build_model_spec  # model/configure.py:6263, :797
from local_operator.providers.auth_store import AuthStore

auth = AuthStore(config_dir=config_dir)
stream_fn = create_stream_fn(auth, settings=settings, session_id="hub-merge")  # attributed in analytics
try:
    req = ChatRequest(model=spec, system_blocks=[system],
                      messages=[Message.user(prompt)], tools=[], tool_choice="none",
                      max_tokens=max_tokens, purpose="hub_merge",
                      isolated=True, replayable=False)   # harness/types.py:2967-3172
    async for ev in stream_fn(req, None):
        if isinstance(ev, StreamTextDelta): parts.append(ev.delta)
        elif isinstance(ev, StreamEndEvent) and ev.error: raise RuntimeError(ev.error)
finally:
    await stream_fn.close(); auth.close()
```

Why these flags: `isolated=True` disables the driver's retry/fallback/rotation
and the sticky-route writes and resolves credentials read-only
(`harness/types.py:3080-3172`, `providers/failover.py:2937-2965`) — a background
job must not move the operator's foreground sessions' credential stickiness
(that is the whole reason the flag exists for naming). It also means **we own the
backoff** (B2.5) rather than inheriting a 10-retry driver budget that could hold
the runner for minutes. `replayable=False` because we retry at our layer. A
stable `session_id="hub-merge"` avoids the unattributable-analytics bucket
`create_stream_fn` warns about (`configure.py:6263-6306`); `purpose="hub_merge"`
is a free string on `ChatRequest.purpose` (`types.py:3008`) and shows up in
`/analytics` by-purpose like `naming`/`compaction`
(`configure.py:6210-6225`, `analytics/model.py:~1044`) — spend is visible.

### B2.4 Which model: the user's default, with a config override

Resolution, in `resolver.resolve_merge_model(config_manager) -> ModelSpec`:

1. **Override** `hub.merge_model` (TEXT, `"provider/model"`, default `""`).
   Parsed with the same rule as subagent tiers: `provider, _, model_id =
   sel.partition("/")`, both halves required (`providers/failover.py:2543`
   `parse_selector`; the both-halves check mirrors
   `Session._resolve_subagent_model` `session.py:11917-11950`). A malformed
   value is a named refusal `model-unavailable: hub.merge_model='x' lacks
   provider/model`, NOT silent fallthrough (so a typo isn't invisible).
2. **Default**: the user's configured default model, read exactly like a fresh
   session: `hosting = config.get_config_value("hosting")`, `model_name =
   config.get_config_value("model_name")` (`bootstrap.py:75-76`; registry keys
   `settings_io.py` "hosting"/"model_name" ~1149-1170), model falling back to
   `model.defaults.default_model_for(hosting)` (`bootstrap.py:88-90`,
   `model/defaults.py:154`). This is `bootstrap.resolve_hosting_model(cm, None,
   None, None)` (`bootstrap.py:63-103`) — call it, don't restate it.
3. Build the spec with `build_model_spec(hosting, model)` (`configure.py:797`),
   which itself resolves live metadata (context window etc.).
4. **Effort**: `spec.model_copy(update={"reasoning_effort": …})` — no clamp to
   lowest (naming clamps because it is extraction; a merge must respect a
   "removed" constraint, which is reasoning). Left at the model's own default.
5. **Unavailable** (no `hosting`, unknown provider, no credential → the stream
   raises `ProviderError(kind="auth")`): `ResolverError("model-unavailable")`.
   The engine then degrades gracefully (B2.7) — it does **not** try other
   providers (no fallback chain: that would spend a credential the user did not
   choose for this).

Tier alternative considered and rejected as the default: the `lo`
`subagents.models` tier (`harness/subagent.py:206`) is cheap but is "the
operator's cheap model for scouts"; merge correctness with removals matters more
than cost, and the operator explicitly asked for the *default* model. It is
reachable by setting `hub.merge_model` to the same string.

### B2.5 Retry with backoff *(numbers)*

Wrapped around each `complete_once` (one region conflict = one unit of retry):

- Retry only `ProviderError.kind ∈ {"transient","timeout","quota"}` and
  `ResolverError("invalid-output")` (max 1 of these — a format retry, with the
  validator's rejection reason appended to the prompt). Classification via
  `classify_provider_error` (`failover.py:986`); `kind=="auth"|"request"` are
  terminal (`model-unavailable` / `provider-error`); `request`+`is_request_too_large`
  (`failover.py:945`) → `prompt-too-long` (B2.6, not retried, re-chunked).
- Delays: `min(cap, base * 2**n) * uniform(0.75, 1.25)` with `base=2s`,
  `cap=60s`, **max 4 attempts** (waits ≈ 2, 4, 8 s between; ≤ ~15 s worst case
  per unit). A `quota` error honours `ProviderError.retry_after_ms`
  (`failover.py:743`) up to a 120 s ceiling; beyond that it fails the unit
  `provider-error/quota` instead of sleeping the runner. Connectivity loss
  (`ProviderError.connectivity_loss`, `failover.py:~129`) is NOT retried inside
  the unit — it fails fast with `provider-error/offline` and the *store-level*
  backoff (B4.4) handles the long wait; the runner must not park a worker for
  minutes the way an interactive session may.
- Cancellation: `asyncio.CancelledError` propagates (never swallowed; not a
  failure class), so lifespan shutdown (`server/app.py:~398-410`) ends a merge
  promptly.
- Per-item wall budget `hub.merge_timeout_s` internal constant 120 s
  (`MERGE_ITEM_TIMEOUT_S`), enforced with `asyncio.wait_for`; exceeded →
  `provider-error/timeout`.

### B2.6 Prompt length and chunking

Budget: `usable = spec.context_window - reserve`, `reserve = max_tokens(=min(
spec.max_output_tokens, 8192)) + 2048` (system + framing). Token counts via
`compaction.tokens.count_text_tokens(text, model_id)` (`tokens.py:300`), with
`approx_text_tokens` (`tokens.py:324`) when the tokenizer is not warmed (never
load it on the hot path — same reason `approx_text_tokens` exists).

Strategy, in escalating order (each step only if the previous doesn't fit):

1. **Only conflicts go to the model, and only their own region text.** Because
   the deterministic core (A4) already resolved every non-conflicting atom, the
   prompt carries just the conflicting *regions* (base/local/remote of that
   region + removal list), not the whole field. Agent instructions cap at 8 000
   chars (`agent_profiles.MAX_INSTRUCTIONS_CHARS` `agent_profiles.py:111`) and
   team briefs at 32 768 (`teams.MAX_TEAM_INSTRUCTIONS_CHARS` `teams.py:89`) —
   ≈ 2k and 8k tokens: **the common case fits any modern window in one call**.
2. **One call per conflicting region** (the plan is the region list from A3; the
   recombination is mechanical concatenation in local order by the core, not by
   the model), so a 30-section brief with 3 conflicts is 3 small calls, each
   independently retried and independently reportable.
3. **Atom-group split** of a single oversized region: split at atom boundaries
   (never inside a fenced block, A3) into groups ≤ usable/3, resolve each group,
   recombine in order. A group boundary never separates an atom from its
   `removals` entry.
4. `prompt-too-long` from the provider anyway (model window smaller than
   metadata claimed): halve the group size once and retry; second failure →
   **honest last resort**: the region is marked `unresolved`
   (`failure_class: "prompt-too-long"`), outcome `needs-review`; no truncation,
   no silent skip, nothing applied for that field (all-or-nothing per field,
   A7). The user is told the region name and sizes, and `--prefer` /
   `--replace` are the manual exits.

Fields merge independently (an agent's `description` can apply while an
`instructions` region is unresolved) **only if** A7 marks the pair independent
(agents: yes; team `members` vs briefs: yes; `manager`↔`members`: joint —
see A7). Otherwise the item is all-or-nothing.

### B2.7 Graceful fallback ladder (when the model is unavailable)

`ResolverError` of class `model-unavailable`, `provider-error` (after retries) or
`prompt-too-long` (after B2.6.4) degrades, in order:

1. Everything the deterministic core decided stays decided (V01–V05, V07,
   V10–V13 need no model).
2. Conflicting regions become `unresolved` unless `opts.prefer` is set, in which
   case the preferred side wins per A4.3 and the loser is recorded in
   `dropped`. **Auto-update never sets `prefer`** — a background job cannot
   decide a conflict on the user's behalf — so an auto run with an unavailable
   model on a conflicting item ends `available` (state) with
   `error_class` set: the indicator stays, the user resolves it (B4/B6).
3. Items with **no** conflicts (pure remote-only or disjoint edits) need no
   model at all and DO auto-apply even with the model down. This is deliberate:
   the model outage should not block the 90 % case.

### B2.8 Never-blindly-overwrite: enforcement points

The rule is a property of code, not of the prompt: (1) every write path takes a
`MergeResult` produced by `merge_field`/`replace_field`; there is no
`write(remote)` entry point on the pull side; (2) `validate_proposal` (A6.2 V1–V4)
rejects a proposal that omits a claimed atom, regrows a removal, or is
un-attributable; (3) the apply step re-verifies, under the registry lock, that
the row still equals the `L` the merge was computed from (concurrent-edit
guard, B1.3); (4) a pre-apply backup is written first (A9.3); (5) shrinkage
guard — a merged field shorter than 50 % of `max(len(L),len(R))` and losing ≥ 3
atoms adds warning `large-shrink` and forces `needs-review` under auto-update.

## B3 Auto-update: settings and the check runner

### B3.1 Config keys (AGENTS.md "Adding a configuration key", L3004-3034)

New `Section("hub", "Agent Hub", Scope.LIVE, …)` appended to `SECTIONS`
(`settings_io.py:304`), placed before `"aida"`. **LIVE is honest**: the runner
re-reads the mapping every tick (B3.3), so an edit lands within one tick. Four
`Setting`s in `SETTINGS` (`settings_io.py:1147`), all nested tuples (NOT the
literal-dotted `display.*` exception):

| key | `path=` | Kind | default | help (≤72 cells, measure on a frame per the file's own note at `settings_io.py:1152-1157`) |
|---|---|---|---|---|
| `hub.auto_update.agents` | `("hub","auto_update","agents")` | BOOL | `True` | `Merge hub updates into pulled agents automatically.` |
| `hub.auto_update.teams` | `("hub","auto_update","teams")` | BOOL | `True` | `Merge hub updates into pulled teams automatically.` |
| `hub.check_interval_min` | `("hub","check_interval_min")` | INT, min 5, max 1440 | `60` | `Minutes between hub update checks (both modes).` |
| `hub.merge_model` | `("hub","merge_model")` | TEXT, `empty_unsets=True` | `""` | `provider/model for merges; empty uses your default model.` |

Module-level defaults live NEXT TO THE READER, in `local_operator/hub_sync/settings.py`:

```python
DEFAULT_AUTO_UPDATE_AGENTS = True
DEFAULT_AUTO_UPDATE_TEAMS = True
DEFAULT_CHECK_INTERVAL_MIN = 60
MIN_CHECK_INTERVAL_MIN, MAX_CHECK_INTERVAL_MIN = 5, 1440

@dataclass(frozen=True)
class HubSyncSettings:
    auto_agents: bool; auto_teams: bool; interval_min: int; merge_model: str
    @staticmethod
    def from_config(cm: "ConfigManager") -> "HubSyncSettings": ...   # get_nested_value(path, default=DEFAULT_*)
```

`from_config` uses `ConfigManager.get_nested_value(path, default)`
(`config.py:966`) — NOT `get_config_value("hub.auto_update.agents")`, which looks
up a literal dotted key and reads nothing (`config.py:945-955` docstring). It
clamps `interval_min` into `[5,1440]` and treats non-bool values as the default
(a hand-edited `"false"` string must not silently mean True or crash).

`tests/unit/test_settings_io.py::_consumer_defaults` (`:38-226`) gains, next to
the `session.cleanup.*` entries (`:190`):
```python
"hub.auto_update.agents": DEFAULT_AUTO_UPDATE_AGENTS,
"hub.auto_update.teams": DEFAULT_AUTO_UPDATE_TEAMS,
"hub.check_interval_min": DEFAULT_CHECK_INTERVAL_MIN,
```
`hub.merge_model` goes in `_NO_SINGLE_VALUE_CONSUMER` (`:412`) with the same
reason string as `subagents.models.lo`: `"free text; empty means 'use the default
model', no constant"` — the legitimate use of that list (unset = inherit). It is
guarded by the staleness test at `:475-477`. Whether `DEFAULT_CONFIG`
(`config.py`) needs the new block is decided by the existing
`test_every_default_matches_its_consumer` failing by name; add only if it does.

**"Auto ON" vs "manual" semantics** (both KEEP CHECKING — requirement):
`auto_update.*=false` ⇒ the runner still checks and records `available`, never
merges or writes. `true` ⇒ after a check finds `available`, it additionally
computes the merge and applies it iff the outcome is `merged` with no
`unresolved` region, no `large-shrink`, baseline known (B2.8, A2.3). Anything
else leaves the item `available` with `error_class`/`reason` set.

### B3.2 Where the runner lives — decision

| Option | Assessment |
|---|---|
| **APScheduler job in `SchedulerService`** (`scheduler_service.py:109-185`) | The scheduler exists to fire *agent schedules loaded from the registry* (`load_all_agent_schedules` `:374`); its documented public surface is "byte-compatible with the legacy service" (`:112-116`). An interval job would work technically but couples an unrelated concern to a class whose `shutdown` cancels `_run_tasks` for schedule runs, and APScheduler misfire/coalesce policy is not what we want (we want "skip if previous still running"). |
| **Task in the server `lifespan`** (`server/app.py:92-…`) | Chosen. Precedent: `retire_task` (`app.py:352`), `reload_task` (`:381`), `aida_boot_task` (`:197`) — all `asyncio.create_task`, held on `app.state`/locals, cancelled and `gather`ed in the `finally` (`:395-460`). Lifespan already owns `app.state.config_manager` (`:203`) and one process per config root, so "single runner" is structural. |
| Per-session (TUI/runtime) | Rejected: N sessions ⇒ N runners ⇒ double-apply. |

Wiring (exact shape, added after the scheduler start at `app.py:216` and the
serve-record block, so readiness is not delayed — same reason `_aida_boot_ensure`
is a task, `app.py:163-197`):

```python
from local_operator.hub_sync.runner import HubSyncRunner   # lazy: import-graph guard
app.state.hub_sync = HubSyncRunner(config_manager=app.state.config_manager,
                                   auth_store_factory=…, env_config=app.state.env_config)
app.state.hub_sync_task = asyncio.create_task(app.state.hub_sync.run_forever())
# finally-block:  app.state.hub_sync.stop(); task.cancel(); await gather(..., return_exceptions=True)
```

A daemon started without an announcement (bare `uvicorn`, `app.py:~248-256`) still
runs it — the runner is not tied to the serve record. `HubSyncRunner` is also
what the routes call (`app.state.hub_sync.check_now(...)`, B5) — ONE object owns
check+apply, so the route, the timer and (via a shared function) the CLI cannot
diverge.

### B3.3 Runner behaviour

```python
class HubSyncRunner:
    async def run_forever(self) -> None:      # never raises (catches, logs, backs off)
    async def tick(self, *, reason: Literal["timer","startup","manual"]) -> TickReport
    async def check_now(self, *, kind: Literal["agent","team","all"]="all",
                        ids: Sequence[str] | None = None) -> TickReport   # route entry
    async def apply(self, kind, local_id, *, prefer="none", replace=False,
                    acknowledge_unknown_baseline=False) -> ItemMergeReport # route/CLI entry
    def stop(self) -> None
```

- **No boot blocking**: `run_forever` first `await asyncio.sleep(45 + jitter(0-15))`
  (`reason="startup"`), then loops `tick`, then `sleep(interval_min*60 ±10 %)`.
  Interval re-read each loop (LIVE).
- **Only when a credential resolves**: each tick starts with
  `resolve_radient_credential(config_dir, base_url, store=…)`
  (`providers/radient_credentials.py:147-171`) and, for org items,
  `resolve_radient_oauth_access` (`:183-230`) using the SAME `AuthStore` the
  server owns (`get_provider_auth_store` `server/dependencies.py:45-68`; the
  runner takes it from `app.state`, not a fresh one, so a refresh persists in
  the login's home). No credential ⇒ log at DEBUG once per state change, write
  every linked item's store state as `up-to-date`→ unchanged (do not create
  failure noise), and sleep the full interval. **A missing login is not a
  failure** (B4.3 `no-credential` is an informational class, not an error
  banner).
- **Degrades quietly**: every exception inside a tick is caught per item; the
  loop never dies; the module never logs above WARNING except a repeated-failure
  summary (once per item per class).
- **No double-apply** (three layers): (1) one runner per daemon and an
  `asyncio.Lock` serialises `tick`/`apply`; (2) a cross-process lease
  `<config_dir>/hub/.runner.lease` (O_EXCL + expiry, the
  `_ListingFetchLease` shape, `model/catalogue.py:336-430`) held for the tick, so a
  second daemon / a CLI `lop agents sync` run does not merge the same item
  concurrently — a loser skips the item (`concurrent-edit`, retried next tick);
  (3) the apply's under-lock fingerprint re-verification (B1.3, B2.8.3) makes a
  lost race a harmless abort, not a lost edit.
- **Bounded work**: checks run with concurrency 2 (`asyncio.Semaphore(2)`),
  each `asyncio.to_thread` (sync `requests` clients), per-request timeout 20 s;
  at most 50 items per tick, oldest-`last_checked` first (the rest roll to the
  next tick). Merges run **sequentially** (model spend, ordering) after checks.
- **Cost of the agent check** (B0.5): today an agent check downloads the archive
  zip. Bounded by the 60-min default × ≤ 50 items. **Evidence to settle a
  cheaper check**: call `RadientClient.get_agent(hub_id)` (`radient.py:686`)
  against a real listing and see whether the detail carries a version/updated
  stamp; if yes, use it as a pre-filter and download only on change. Until
  proven, download (correctness first).
- **Auto-applied outcome** is recorded in the store as `applied` with the
  region roll-up (B4) and emits ONE `authoring`-visible change for free: the
  registry write touches `agent.yml`/`team.yml`, which the desktop feed's
  authoring probe already watches (`server/utils/desktop_feed.py:507-545`,
  `useAuthoringRefresh` in the UI), so the sidebar lists refresh with no new
  channel. The status store needs its own cheap poll (B6).

## B4 Availability / failure state store

### B4.1 Location and integrity

`<config_dir>/hub/status.json` — one file, `config_dir` from the
`ConfigManager` (never `Path.home()`; AGENTS.md "Isolating a run" — a redirected
`HOME` alone does not redirect the config dir, and a store that defaulted its
WRITE path to the global root is the exact analytics-backfill bug documented at
AGENTS.md L~930-940). Atomic write = temp in the same directory +
`os.replace` (shape of `monitors/state.py:155-175`, `teams._atomic_write_text`
`teams.py:676`). Read-modify-write guarded by a `flock` on
`<config_dir>/hub/.status.lock` (non-blocking try + 2 s bounded wait, the
`teams._try_lock_exclusive` shape `teams.py:635-660`; a lock timeout degrades to
"skip this write, retry next tick" — never blocks a request). A corrupt/
unreadable file is renamed to `status.json.corrupt-<ts>` and treated as empty (all
items re-derived by the next check — the store is a CACHE of derivable state
plus retry bookkeeping, never a source of truth for content).

### B4.2 Schema (`schema: 1`)

```json
{
  "schema": 1,
  "updated_at": "2026-09-29T12:00:00Z",
  "last_tick": {"at": "…", "reason": "timer", "checked": 7, "available": 2,
                "applied": 1, "failed": 0, "credential": "ok|none|org-missing"},
  "items": {
    "agent:<local_uuid>": {
      "kind": "agent",
      "local_id": "<uuid>",
      "name": "coder",
      "hub_id": "9f2c…",
      "tenant_id": null,
      "state": "up-to-date | available | updating | applied | failed",
      "remote_fingerprint": "<sha256 or null>",
      "local_fingerprint": "<sha256>",
      "baseline": "known | unknown",
      "classification": "remote-only | both-changed | baseline-unknown | null",
      "summary": {"taken-remote": 2, "kept-local": 1, "combined": 0,
                  "removal-honored": 1, "unresolved": 0},
      "attempts": 0,
      "error_class": null,
      "last_error": null,
      "first_seen_available_at": "…",
      "last_checked_at": "…",
      "last_applied_at": "…",
      "next_retry_at": null,
      "auto_retry": true,
      "applied_backup": "hub/backups/agent-<id>-20260929T120000Z.json"
    }
  }
}
```

State meanings: `up-to-date` (fp(R) == fp(L) or nothing to pull); `available`
(remote differs; not yet applied — includes manual mode and "auto declined");
`updating` (a merge/apply is in flight; carries `updating_since`, stale after
5 min ⇒ treated as `failed/concurrent-edit` on read, crash recovery);
`applied` (last action merged; reverts to `up-to-date` at the next check that
finds no drift — kept one cycle so the UI can say "Updated just now");
`failed` (`error_class` + `last_error` set). `last_error` is the user-facing
sentence, ≤ 300 chars, credential-scrubbed (`public_data`/`redact_secrets`
precedent: `routes/desktop_radient.py:547`, `clients/_http.py` `scrub_details`).
Items whose local row was deleted or whose baseline link vanished are pruned at
the next tick.

### B4.3 Failure classes

| `error_class` | Trigger | `state` | auto-retry |
|---|---|---|---|
| `no-credential` | no Radient credential / no OAuth for an org item | unchanged (not `failed`); item shows nothing | re-evaluated every tick, no attempt counted |
| `provider-error` | model call failed after B2.5 retries (`subclass`: quota/timeout/offline/transient) | `available` if the check succeeded earlier, else `failed` | yes, backoff below |
| `model-unavailable` | no default model / bad `hub.merge_model` / auth on the model provider | `available` (check fine, merge blocked) | 1 h flat, then backoff |
| `prompt-too-long` | B2.6.4 last resort | `failed` | **no** until `remote_fingerprint` changes or user acts |
| `merge-refused` | outcome `needs-review`/`unresolved`/`refused` (incl. `large-shrink`, `baseline-unknown`, hub caps A7) | `available` with `classification` | **no** (a human decision); re-armed when `remote_fingerprint` changes |
| `concurrent-edit` | fingerprint re-verify under lock failed / lease lost / stale `updating` | `available` | yes, next tick (attempt not counted) |
| `hub-item-missing` | 404 on re-fetch | `failed`, kept visible | every 24 h |
| `hub-error` | 5xx/network on the check itself | keeps previous state | yes |

Backoff for counted attempts: `delay = min(6 h, 15 min * 2^(attempts-1)) *
uniform(0.9, 1.1)`; `attempts` resets on success and on `remote_fingerprint`
change; after `attempts >= 6` set `auto_retry=false` (stop nagging the model;
manual Retry or a new remote fingerprint re-arms). `next_retry_at` is honoured
by the timer tick but **ignored by a manual `check-now`/`retry`** (user intent
outranks the schedule).

### B4.4 "Update all" semantics

`POST …/apply-all` (B5) runs the items currently `available`, sequentially:
**agents first, then teams** (a team's roster names agents; applying agent
updates first keeps `missing-role` warnings truthful), each group alphabetical
by name. A per-item failure never stops the run (recorded, continue), EXCEPT a
*systemic* class (`no-credential`, `model-unavailable`, `provider-error/quota`):
the run stops, the remaining items are marked `skipped: <class>` in the
response (state unchanged) so N items do not each burn the same retry budget.
`needs-review` items are skipped, never forced (`prefer` is per-item only —
"update all" cannot carry a conflict decision). Response reports per-item
`ItemMergeReport` + roll-up counts. It is not transactional across items
(each item is atomic; the report says exactly which applied).

## B5 Surfaces: routes, CLI, tool

**One derivation.** Every surface calls the same three functions in
`hub_sync/service.py` (thin orchestration over check/merge/store); none of them
re-implements classification or wording:

```python
async def check_items(cfg: HubSyncContext, *, kinds=("agent","team"), names=None,
                      force_refresh=False) -> list[ItemStatus]: ...
async def apply_items(cfg: HubSyncContext, *, kind=None, names=None, all_available=False,
                      prefer="none", acknowledge_unknown_baseline=False,
                      replace: Literal[None,"remote","local"]=None,
                      dry_run=False, auto=False) -> ApplyReport: ...
def status_snapshot(cfg: HubSyncContext) -> StatusPayload: ...   # store read only, no network
render_report(report, *, style: Literal["cli","tool"]) -> str    # the ONLY prose renderer
```

`HubSyncContext(config_dir, config_manager, auth_store | None, env_config | None)`.
`agent_sync.sync_agent_profiles` (`agent_sync.py:190`) keeps the **seed arm** and
delegates the hub arm to `check_items`/`apply_items`; `sync_payload` /
`SyncReport.render` (`agent_sync.py:100-170,228`) are extended, not forked: a new
`kind:"hub"` entry shape carries the `ItemMergeReport` (A8) in place of
`HubSyncVerdict`'s `replaced_*` echo, and the verdict vocabulary gains `merged`,
`needs-review`, `available` alongside `up-to-date`/`updated`/`unavailable`
(`diverged` is retired for hub rows: a both-changed item is now merged or
`needs-review`).

### B5.1 Desktop routes (new file `server/routes/desktop_hub.py`, mounted beside `desktop_profiles`)

All under `dependencies=[Depends(require_desktop)]` like
`desktop_profiles.py:~40` (`router = APIRouter(..., dependencies=[Depends(require_desktop)])`),
request models extend `Input` (`extra="forbid"`, `desktop_sessions.py:600`) with
`request_id: RequestID`; mutations run through `receipts(request).run(key, body,
fn, retry_safe=True)` exactly like `sync_profiles`
(`desktop_profiles.py:154-205`) so a lost response is replayed not re-run.
Feature flag: `features.py:54` gains `"hub_updates": 1`; the UI gates on
`desktopFeatureEnabled(capabilities.data, "hub_updates")` (the
`profile_catalogue` precedent, `chat-sidebar.tsx:~905`).

| Method + path | Body | Purpose |
|---|---|---|
| `GET /v1/desktop/hub/updates` | — | **Store read, no network, O(items).** The UI polling endpoint. |
| `POST /v1/desktop/hub/updates/check` | `{request_id, kind?: "agent"\|"team", name?: str}` | Check now (network). Honors `force_refresh`. Applies nothing, even with auto on (user asked to *check*). Runs are single-flight (B3.3): a concurrent runner tick is joined, not doubled. |
| `POST /v1/desktop/hub/updates/apply` | `{request_id, kind, name, prefer?: "local"\|"remote", acknowledge_unknown_baseline?: bool, replace?: "remote"\|"local", dry_run?: bool}` | Apply ONE item (click-to-update). `dry_run` returns the `ItemMergeReport` without writing (the preview a UI/CLI can show). `replace` requires `confirm_replace: true`. |
| `POST /v1/desktop/hub/updates/apply-all` | `{request_id, kind?: "agent"\|"team"}` | B4.4 semantics. |
| `POST /v1/desktop/hub/updates/retry` | `{request_id, kind, name}` | Clears `attempts`/`next_retry_at`/`auto_retry=false`, then check+apply(auto=cfg) for that item. |

`GET /v1/desktop/hub/updates` → `CRUDResponse.result`:

```json
{
  "generated_at": "2026-09-29T12:00:00Z",
  "credential": "ok | none",
  "settings": {"auto_agents": true, "auto_teams": true, "interval_min": 60},
  "counts": {"available": 2, "failed": 1, "updating": 0, "up-to-date": 9},
  "items": [
    {"kind":"agent","name":"coder","local_id":"…","hub_id":"…","tenant_id":null,
     "state":"available","classification":"both-changed","auto_will_apply":false,
     "remote_fingerprint":"…","first_seen_available_at":"…","last_checked_at":"…",
     "error_class":"merge-refused","last_error":"Both changed \"## Review rules\" …",
     "next_retry_at":null,"summary":{"taken-remote":2,"kept-local":1,"combined":0,
     "removal-honored":1,"unresolved":1}}
  ]
}
```

Only items with `state != "up-to-date"` are listed, PLUS any item carrying the
`no-credential` class — an org-linked row the local session has no login for never
becomes `available`, and that class is the one fact the sidebar's sign-in sentence
is drawn from (UX round 2, U11) — plus a `counts` roll-up (the sidebar polls this
constantly; keep it small). Names are the profile/team
**names** because they are the UI's attachment keys ("Names deliberately remain
the runtime's attachment keys", `server/utils/desktop_profiles.py:1-6`).
`auto_will_apply` = `auto_update.<kind> && state=="available" &&
classification=="remote-only" && baseline=="known"` — lets the UI say
"will update automatically" vs "needs you" without re-deriving policy.

`apply`/`apply-all`/`check` responses: `{"reports":[ItemMergeReport…],
"status": StatusPayload}` (the fresh snapshot, so the client updates the cache
in one round trip). Errors: `401` no credential (org remedy text from
`ORG_LOGIN_REMEDY`), `404` unknown item, `409` `concurrent-edit`, `422`
`refused`/invalid `prefer`, `503` store unreadable — same status vocabulary as
the team pull route (`routes/agents.py:1521-1630`).

The `authoring` revision channel (`docs/DESKTOP_API.md:1701-1790`, `useAuthoringRefresh`
`profile-hooks.ts:~48`) already invalidates `["desktop","profiles"|"teams"]` when
agent/team files change on disk, so an applied merge refreshes the lists for
free; the hub status query is invalidated explicitly by the mutations' own
`onSuccess` and by the same effect (B6.3).

### B5.2 Existing `profiles/sync` route

`ProfileSync` (`desktop_profiles.py:45-56`) keeps `{name|all, force}` for
back-compat. Its hub arm now goes through `apply_items`; `force` is mapped per
B5.5 (`force=true` ⇒ `replace="remote"` **and** requires the new
`confirm_replace: true`, else 422 with the sentence "--force replaces your copy;
use replace with confirm_replace"). The seed arm is untouched.

### B5.3 CLI

```
lop agents sync [--name N | --all] [--check] [--prefer local|remote]
                [--accept-unknown-baseline] [--replace [--yes]] [--dry-run] [--json]
lop teams  sync [--name N | --all] [same flags]
lop teams  link <team> --org <tenant> <hub-team-id> [--accept-unknown-baseline]   # A2.3 adoption
lop hub    status [--json]            # store read; same payload as GET …/hub/updates
```
`lop agents sync` (`cli.py:472-490`, handler `agents_sync_command` `cli.py:7926`)
default behaviour changes from "refuse if diverged" to "merge" (`--check` =
report only). `lop teams sync` is added next to `teams pull`
(`cli.py:542-551`, dispatch `cli.py:10419-10430`); it resolves the person's
OAuth client with `_resolve_org_client` (`cli.py:8174`) — teams are org-only.
`lop hub status` is a top-level subcommand (small) because the status is
cross-kind; it prints the same `render_report` text. `--json` emits the route
payload verbatim (one shape).

### B5.4 `agent` tool

`AgentParams.op="sync"` (`agent_tool.py:120-125`) already exists with `name`,
`force` (`agent_tool.py:~233`). Extension keeps the schema footprint flat
(AGENTS.md "tool-surface footprint ladder", rung 1: extend an existing tool):
`force` is **removed from the model-facing schema** and replaced by one
optional enum `resolve: "local"|"remote"` (same byte cost as `force`; the
comment at `agent_tool.py:~229-232` explains why a plain bool matters for the
budget gate). Semantics: no `resolve` ⇒ merge (auto path rules: never resolves a
conflict); `resolve` ⇒ `prefer`. `replace` is **not** available to the model —
an agent must not be able to discard a user's copy; it stays CLI/UI only.
`team` tool: no `sync` op (teams are org/person-authenticated and a model-driven
team pull is out of scope); the `teams` CLI/UI are the surface. `_op_sync`
(`agent_tool.py:1020`) renders through `render_report(style="tool")`.

### B5.5 `--force` fate

`--force` ("apply over local edits", `cli.py:488`; `ProfileSync.force`;
`AgentParams.force`) is the flag that made the clobber possible. Decision:
**deprecate, do not repurpose.** `--force` keeps working for one release as a
hidden alias of `--replace --yes` and prints `--force is deprecated: it replaces
your copy with the hub text; prefer a merge (default) or --replace`. `--replace`
is the explicit, named, echo-the-replaced-text operation (`replace_field`,
B2.2), never reachable from auto-update, tool, or apply-all. The seed arm's own
`force` (`sync_installed_seeds`) is unaffected (different family).

## B6 UI brief (local-operator-ui; for the UI coder and the design round — no pixels here)

### B6.1 Where

The chat sidebar's **Agents** and **Teams** sections
(`src/renderer/src/features/chat/components/chat-sidebar.tsx`). Agents render
`ownAgents.map(profile => entity("agent", profile.name))` (`:4974`; list from
`useProfiles` `:912-916`, filtered `source !== "builtin"` `:915`); Teams from
`useTeams` (`:997`) through the same `entity(kind, name)` row builder (`:3803`),
whose row is `[disclosure][name button][24px "Manage" control]` (`:3841-3960`).
Also: `agents-page.tsx` (list/detail via `useProfiles`/`useTeams`, `:546-547`) —
its detail pane is where the merge report and Retry live. The sidebar is
route-scoped (a comment at `profile-hooks.ts:~30` says so), so the hub hook owns
its own lifetime like `useAuthoringRefresh`.

### B6.2 Required behaviours

1. **Update-available indicator**, subtle, on the entity row, shown for
   `state=="available"` **regardless of `settings.auto_*`** (auto-on items that
   have not yet applied, and all manual-mode items). Must not displace the
   reserved 24px control slot or reflow the row (the file's own rule, "a row
   that reflows under the pointer is worse than no affordance", `:3870-3880`);
   accessible name states the action ("Update coder from the hub"); tooltip/
   secondary text distinguishes `auto_will_apply` ("updates automatically") from
   `needs you` (`classification` `both-changed`/`baseline-unknown`).
2. **Click the indicator → update that item** (`POST …/apply`, no `prefer`). If
   the response is `needs-review`, navigate to the item's detail (agents page)
   where the per-region report shows both texts and offers "Use mine / Use hub's
   / Edit"; those call `apply` with `prefer`. `replace` is behind a destructive
   confirm on the detail page only.
3. **Update all**: one control in each section header when `counts.available>0`
   (or one in the sidebar footer — designer's call), calling `apply-all`, then a
   roll-up toast/inline sentence from the response counts ("3 updated, 1 needs
   your review").
4. **Failure indicator + Retry**: `state=="failed"` (or `available` with
   `error_class`) shows a distinct, still-subtle mark; the detail pane names the
   `error_class` in plain words (mapping table in B6.4) and offers **Retry**
   (`POST …/retry`). `no-credential` is NOT a per-row mark — it is one section-level
   line ("Sign in to Radient to get hub updates") shown only if the user has any
   hub-linked item.
5. **Applied**: `state=="applied"` shows a transient "Updated" for the poll
   cycle; the row text/description refresh comes via the authoring channel.
6. No indicator at all for `up-to-date`, `unlinked`, builtin, or when
   `credential=="none"` and nothing is linked (zero UI cost for users who never
   used the hub).

### B6.3 Data flow

- New `useHubUpdates(enabled)` in `shared/api/local-operator/hub-hooks.ts`:
  `useQuery({queryKey:["desktop","hub-updates"], queryFn: desktopResult<HubUpdates>({op:"hub.updates"}), staleTime: 30_000, retry: retryDesktopQuery, refetchInterval: 60_000, refetchIntervalInBackground: false})`.
  `refetchIntervalInBackground:false` is the convention (`use-mcp-servers.ts:198`,
  `mesh-store.ts:28`); the endpoint is a store read (no network), so 60 s is
  free. `enabled = ready && desktopFeatureEnabled(caps, "hub_updates")`.
- Contract additions in `src/shared/desktop-contract.ts` (the zod
  `desktopRequestUnion` `:886-…` and the op→path switch `:4270-…`):
  `hub.updates` (GET), `hub.check`, `hub.apply`, `hub.applyAll`, `hub.retry`
  (POST with `requestId`), each `.strict()`, mirroring `profiles.install`
  (`:892`, `:4280`).
- Mutations invalidate `["desktop","hub-updates"]` **and** use the response's
  embedded `status` snapshot via `setQueryData` (no second round trip);
  agent/team lists refresh from the existing authoring revision effect
  (`profile-hooks.ts` `useAuthoringRefresh`).
- Types: `HubUpdateItem`, `HubUpdates`, `ItemMergeReport` mirror B5.1 / A8.

### B6.4 Copy table (source of truth for the design round)

| `error_class` | User sentence |
|---|---|
| `provider-error` | "Couldn't reach your model to merge this. Will retry; or retry now." |
| `model-unavailable` | "No model available for merging. Check Settings › Agent Hub." |
| `prompt-too-long` | "This one is too large to merge automatically." |
| `merge-refused` | "The hub and your copy both changed the same part. Review it." |
| `concurrent-edit` | "It changed while updating. Try again." |
| `hub-item-missing` | "No longer available on the hub (or you lost access)." |

### B6.5 Design round

Designer + ux-reviewer (Sonnet 5.5) with **rendered frames** of: agents section
with 0 / 1 / 3 indicators; a failed row; a `needs-review` detail; collapsed
sidebar; narrow width (360 px — the file cites a 360 px measurement); dark/light.
Frames come from Storybook stories following `chat-sidebar-agents.stories.tsx`
(the existing pattern) and from the live app driven in the browser tool, per the
operator's UI-validation rule (loading/empty/error/populated states;
before/after). UX walks: click-to-update, update-all, retry, needs-review →
resolve, auto-on vs auto-off, no-credential.

## B7 Guides

Token-efficient edits; no new guide (the guide catalog is pinned by
`tests/unit/guides/test_guides.py:26-45` and descriptions must be 40–180 chars,
`:49`; adding a guide edits that pin and adds a description that is injected as a
hint per session — a permanent cost). Two guides gain a short section each and
their `description` frontmatter grows within the cap.

**`guides/agents/GUIDE.md`** — description (≤180; measured 138):
`Use Local Operator agent profiles, roles, subagents and Agent Hub pull/push/update: create, select, delegate, choose a collaboration mode.`
Append a section (≈ 260 tokens):

```markdown
## Agent Hub: pull, push, update

- `agents pull <id>` copies a hub agent (add `--org <tenant>` for an organization's). The pull remembers the hub id and the text it pulled — the *baseline*.
- Updates are a three-way merge of baseline, your copy and the hub's. What each side changed is kept; a section you deleted stays deleted, one the hub deleted goes; a shortened paragraph stays shortened. Both sides editing the same sentence is combined by your default model when the meaning is preserved, otherwise left for you.
- Auto-update is on by default (`hub.auto_update.agents`); off, you still see "update available" and apply it yourself: `agents sync --name X`. `--check` only reports. `--prefer local|remote` decides a conflict. `--replace` (CLI/UI only) discards your copy for the hub's and echoes what it discarded.
- `agent op='sync'` merges; `resolve='local'|'remote'` decides a conflict. It can never discard the user's copy.
- FAQ — *Will a pull/update overwrite my local changes?* No: edits and deliberate deletions on your side survive; anything replaced is backed up (`hub/backups/`) and echoed. *Will my publish clobber someone else's edits?* Push runs the same merge against the hub copy first; their changes and removals are kept unless you choose `--replace`.
- Best practice: pull once, edit freely, let updates flow. To drop a section for good, delete it — do not blank it. Publish after an update, not before.
```

> Correction (2026-10-03): the sketch above is shorthand, corrected when it landed as a guide (PR #1943) — the shipped form is `agents pull --id <id>` (`--id` is required), `--check` covers hub rows only (the starter arm has no report-only mode), `--replace` needs `--yes`, and the push-side merge is *not* shipped: a publish is an upload (`name_taken` on an org name collision; `agents push --id` is the explicit overwrite).

**`guides/teams/GUIDE.md`** — description (measured 127):
`Create, update, and run Local Operator teams: a manager plus reusable agents, with layered briefs. Covers org pull/push/update.`
Append (≈ 200 tokens): teams are organization-only on the hub (`teams push|pull --org`); `teams sync` checks/merges hub changes into pulled teams (description, manager, roster, collaboration brief, project brief; the local **name** never changes); roster merge is per slot (added / removed / count) and a slot you removed stays removed; roles the roster names but you lack produce a `missing-role` warning, not a failure; teams pulled before this feature are unlinked until `teams link`; FAQs mirrored from the agents guide. Also fix the existing CLI block in the guide (`local-operator teams list …`, `guides/teams/GUIDE.md` CLI section) to add `teams sync`.

**`guides/configuration/GUIDE.md`**: one line listing the four `hub.*` keys
(that guide is the settings pointer). No other guide changes.

## B8 Test and evidence plan

### B8.1 Unit (targeted; full suite only at terminal review/CI — AGENTS.md L1-40)

| Area | File | Asserts |
|---|---|---|
| Vectors | `tests/unit/hub_sync/test_vectors.py` | A10 V01–V20, both directions, scripted resolver. |
| Segmentation | `test_segment.py` | heading/fence/list/table atoms; ids stable across reorder; CRLF/whitespace; duplicate headings. |
| Presence/precedence | `test_merge_core.py` | A4.2 table exhaustively (property-style: for every (b,l,r) presence combination the outcome equals the table); "a removal never regrows" and "a local shortening never re-lengthens" as invariants over generated inputs. |
| Validators | `test_validate.py` | V1–V4 reject omission / regrown removal / oversize / un-attributable; format-retry appends reason once. |
| Resolver | `test_resolver.py` | stub `complete_once`: retry classes, backoff delays (jitter clamped, `time.sleep`/`asyncio.sleep` patched), `retry_after_ms` honoured and capped, connectivity-loss fail-fast, `model-unavailable` on bad `hub.merge_model`, default-model resolution equals `bootstrap.resolve_hosting_model`, `prompt-too-long` → re-chunk → last resort. `ChatRequest` built with `isolated=True, tool_choice="none", purpose="hub_merge"`. |
| Provenance | `test_provenance.py` | record write atomic (kill between temp and replace); tag/record agreement; lazy adoption when `hub_sha256` matches; unknown baseline path; **record absent from `export_agent_archive`**; team edit/rename/`update_team` leaves record intact; symlinked record refused. |
| Team arm | `test_teams_arm.py` | fake client: 404 → `hub-item-missing` (never delete); `name` never compared; roster merge per slot; caps refusal V16; concurrent-edit abort under `TeamRegistry` lock. |
| Store | `test_store.py` | schema round-trip, corrupt → quarantine + rebuild, stale `updating`, lease exclusivity across two processes (multiprocessing), backoff numbers, attempts reset on new remote fp, isolation (no write under real `HOME`: run under `env -i` + temp `HOME`). |
| Runner | `test_runner.py` | no credential → no network + quiet; single-flight; auto off = check only; startup delay; interval clamp; shutdown cancels within 1 s; spread ≤ N fetches per minute. |
| Config | `test_settings_io.py` (extend `_consumer_defaults`) | four keys; `get_nested_value` reads; bad types → default. |
| Routes | `tests/unit/server/test_desktop_hub.py` | shapes of B5.1, receipts replay, 401/404/409/422, `require_desktop` gate, `features.hub_updates`. |
| CLI/tool | `test_cli_hub.py`, extend agent-tool tests | `--force` alias + deprecation text; `resolve` schema byte budget unchanged (`scripts/bench_context_budget.py` gate); `replace` unreachable from tool. |
| Guides | `tests/unit/guides/test_guides.py` | description lengths 40–180 for the two edited guides. |
| Import graph | `test_import_graph.py` | `lop serve` boot does not import `hub_sync.resolver`/`httpx`. |

### B8.2 End-to-end evidence the PR must carry (real execution, not unit green)

Isolated run: fresh `ISO=$(mktemp -d)`, `env -i HOME=$ISO LOCAL_OPERATOR_CONFIG_DIR=$ISO/.local-operator PATH TERM`, `CMUX_*`/`LOP_*` stripped, `--use-mock-keychain` for any Chrome, a **real test org** with a member account (credential passed through a shell variable, never printed).

1. Publish `v1` of a scratch agent + scratch team to the test org; in the iso profile `lop agents pull --org …` and `lop teams pull --org … ` (records baselines; `cat hub/baselines/*.json | jq keys`).
2. Locally: delete one section, shorten one paragraph, add one rule to the agent; edit the team's roster (remove one slot, add one).
3. On the hub (second profile / API): publish `v2` that edits a *different* section, edits the deleted section, adds a section, changes the team's manager-unrelated brief.
4. Start `lop serve` (isolated, non-default port) with `hub.check_interval_min=5` (or trigger `POST …/hub/updates/check`). **Show**: `GET /v1/desktop/hub/updates` moves `available → updating → applied`; the agent file on disk contains the hub's new section and the local rule and shortening, and the deleted section is absent (report shows `removal-honored`; the hub's edit to it appears only under `unresolved`/`needs-review` per V06 — demonstrate both the honoured case and the conflict case with separate sections); backup file exists; provenance report matches.
5. **Auto off**: set `hub.auto_update.agents=false`, publish `v3`; the status shows `available`, file unchanged; `POST …/apply` merges. Teams same.
6. **Induced failure**: set `hub.merge_model=bogus/none` and publish a conflicting `v4` → `available` + `error_class=model-unavailable`; remove the override → `POST …/retry` succeeds; also a network cut (`connectivity`) shows `provider-error/offline` with `next_retry_at`.
7. **Regrowth negative test**: prove that with the LLM disabled (`allow_llm=false`) a removal never regrows across 20 randomized runs.
8. Concurrent edit: edit the agent file between compute and apply (test hook) → `concurrent-edit`, file intact.
9. Sidebar frames (before/after) from the live app in the browser tool against the iso daemon: indicator visible auto-on and auto-off; click-to-update; update-all; failed + Retry; needs-review detail; plus the numbers behind the frame (row height unchanged, no layout shift, control slot width) per the operator's UI-validation rule. Evidence goes on the PR, not the repo (AGENTS.md §7, L2538).
10. Agreement: run `tests/unit/hub_sync/test_vectors.py` against the push side's implementation of the contract on the same fixture file; post both green outputs.

## B9 Risks and open questions

### Risks to watch in rollout

1. **Prompt-driven correctness.** The LLM layer can still produce plausible bad prose that passes V1–V4 (covers all claimed atoms, regrows nothing) but subtly changes meaning. Mitigation: only conflicting atoms reach it; report + backup + one-click "use mine/hub's"; auto-apply refuses `combined` regions when `hub.auto_update.*` is on? — **decision left open (Q2)**.
2. **Spend.** Default model is billed. Bounded: conflicts only, 4 attempts, 120 s/item, `attempts>=6` stop; `purpose="hub_merge"` makes it visible in `/analytics`. Watch by-purpose spend in the first week.
3. **Org-agent credential gap (B0.6)** changes what "check" returns for existing org pulls (from silent `unavailable` to real results): expect a burst of newly `available` items on first run after upgrade. Mitigation: the first run marks items `available` and **does not auto-apply for 1 tick after upgrade** (`first_run_grace`), and baseline-unknown items never auto-apply.
4. **Regrowth via stale baseline.** If a baseline record is lost (manual delete), items degrade to baseline-unknown (safe: check-only). If a baseline is *wrong* (the record says B but the user's L predates it) a deletion could be misattributed. Mitigation: records are only written by pull/publish/merge with fp recorded; tag/record disagreement ⇒ unknown.
5. **Multi-process races.** `lop serve`, TUI, CLI and scheduler processes can each run a check. Mitigation: store lease (B4.1) + per-row fingerprint re-verify under the registry lock (agents: no lock exists — see Q4).
6. **Boot cost / import graph.** Guarded by the lazy imports and the import-graph test; runner first tick delayed 60 s.
7. **Hub-side caps.** A merged team/agent may exceed hub limits on a later push (`teams.py:384-389`, `agent_profiles.py:111`); pull-side enforces local caps, push-side must enforce hub caps.
8. **Team semantics.** Hub roster kind is free-form (`teams.py` docstring 552-556); unknown kinds map to `agent` on import (`teams.py:1620-1625`). A merge must not launder an unknown kind into `agent` on the *push* side; A7 keeps the original kind in the record.
9. **Deleted-on-hub vs membership-lost** are indistinguishable (404); never auto-delete locally.

### Open questions for the manager

- **Q1 — Does the push side accept B living in `<config>/hub/baselines/*.json` (text, not just hash) as the shared baseline store, and the A3 segmentation as normative?** Both halves cannot ship different segmenters. Needs the radientdev co-author's ack before either implements.
- **Q2 — Auto-apply of `combined` (LLM-rewritten) regions.** Recommend: auto-apply allowed (that is the feature), report always shows `combined`, backup always written. Conservative alternative: auto stops at `combined`, needs a click. Operator preference decides; default in this doc is *allow*.
- **Q3 — Public-hub agents pulled anonymously**: keep the anonymous download for the check (no OAuth needed) — recommended — vs require sign-in. Affects the "runs only when a Radient credential resolves" rule: public items need no credential; the doc proposes the runner runs whenever ANY linked item exists and resolves per-item (`client_for_tenant`).
- **Q4 — Agent registry locking.** `AgentRegistry` has no persistence lock comparable to `TeamRegistry._persistence_lock` (`teams.py:1241`, cf. `agents.py:1287` refresh-interval cache). The apply's concurrent-edit guard for agents is a fingerprint re-check (narrow window, not airtight). Add a lock in this PR or accept the window?
- **Q5 — Push-side wire for a three-way at publish.** The hub stores only the latest document; push needs R fetched right before publish (`get_agent`/`get_team`) and there is no compare-and-swap on the hub (`overwrite_agent_in_marketplace`, `publish_team_document` are last-writer-wins). Residual race: someone publishes between fetch and upload. Needs a hub-side `If-Match`/version precondition (agent-server change) or the push team accepts the window.
- **Q6 — `version` field.** Teams always publish `"1.0.0"` (`teams.py:560`), so hub versions carry no signal and content fingerprints are the only change detector. Fine for us; the manager may want push to start sending real versions.
- **Q7 — Settings section placement/label** ("Agent Hub", LIVE) and whether `hub.merge_model` belongs under `subagents.*` instead — cosmetic, designer's call.
- **Q8 — Mesh sync interaction.** `network/definitions.py:475-495` builds team rows from the model and would propagate merged content between devices; baselines are per-device. Confirm that two devices sharing a team each track their own baseline (recommended) and that a mesh-synced edit counts as a "local edit".
- **Q9 — Anything in the task brief I found wrong**: (a) the brief said teams' markers live "in a team row" — the row is swapped wholesale, so markers must be external (B1.1); (b) the existing hub sync uses the ANONYMOUS download, so org agents never sync today (B0.6); (c) there is no session-less one-shot helper — `invoke_model` needs a `ModelConfiguration` and `complete_once` a live `Session`; B2.3 lifts the `invoke_model` body rather than adding a fourth path.

---

# Implementation notes (deviations from the design)

Each is the smallest change code reality forced; none alters Part A.

1. **Whole-item apply, no partial apply.** A5/B2.8 allowed an agent's `description` to land
   while an `instructions` region is unresolved. That needs per-field baselines; the record
   stores the whole item. An item applies whole or not at all (the conservative direction).
2. **`lop teams link` over a differing copy records `recorded_by: "adopt-unknown"`**, not B := L.
   Recording B := L would let a later hub deletion read as a plain remote removal and delete
   text the user wrote. The check reads such a record as "baseline unknown" (A2.3).
3. **`AgentRegistry` still has no lock (Q4).** The agent fingerprint re-check is a narrow-window
   guard. Teams re-verify under `TeamRegistry`'s writer lock via
   `update_team(..., precondition=...)`, which is airtight.
4. **Model-facing `agent` tool: `resolve` is a plain `str` (default `""` = unset), validated to
   `"" | "local" | "remote"` in `_op_sync`**, not `... | None` (an `anyOf` null branch costs bytes
   on every session's tools array) and not `Literal["", ...]` (an empty-string enum member is
   rejected by the Gemini-family providers; `tests/unit/tools/test_registry.py` pins that no
   built-in schema emits one). It maps to `prefer` only; the tool has no `replace`. The seed arm's
   `--force` is unchanged and is still not reachable by the model: a refused edited starter
   points at `op='reset'`.
5. **`profiles/sync` payload:** the seed arm keeps `entries`/`summary`; the hub arm reports under
   `hub` (the merge service's `ApplyReport`). `force` on the hub arm means `replace`, so it is
   honoured only with `confirm_replace`; the 422 is raised only when the hub arm would actually
   reach a hub-pulled agent, so an old client that sends `force` for the seed arm alone keeps
   working (it gets the hub arm as a plain merge).
6. **V2 has a content check** beyond key tokens: a covered side's newly added words must mostly
   appear in the merged text, otherwise a proposal that returns just the base sentence "covers"
   both sides and passes.
7. **First tick after upgrade never auto-applies** (`hub/.first_run_done`, B9 risk 3).
8. **Runner auth store:** the lifespan passes a provider for the desktop login's `AuthStore`
   (created lazily by the first authenticated request); before that the resolver opens a
   short-lived store itself.
9. **Baseline after a pull-merge is the REMOTE text integrated, not the merged result** (amends
   A2.2, see the note there). `merge.py` and `service._advance_baseline` implement it; the push
   side must record the LOCAL text it published in the mirror-image case.
10. **Atom similarity is `max(token ratio, character ratio, containment)`** (amends A4.1, see the
    note there). Forced by vectors V01/V06/V07/V08; `segment.py::similarity` is the single
    implementation and the push side imports it.
11. **Prompt chunking (B2.6.3) is not implemented; B2.6.1/2/4 are.** A conflict group is one atom
    triple (a single base/local/remote sentence or bullet), so there is no group to halve: on
    `prompt-too-long` the resolver sheds the OPTIONAL context (neighbouring regions) once and
    retries, and a second failure degrades per B2.6.4 (`failed`, no auto-retry until the remote
    text changes). Revisit only if a single atom can exceed a model's window.
12. **Cross-process single-flight is enforced in `service.apply_items`, not only in the runner.**
    Every writer (runner tick, desktop routes, `lop agents|teams sync`, the `agent` tool) holds the
    `RunnerLease`; a heartbeat thread renews it every TTL/3 (the TTL bounds only a CRASHED holder).
    A caller that cannot get it within `LEASE_WAIT_S` (30 s) gets `HubBusy` (HTTP 409 / a CLI error
    line); the timer skips the tick instead. The routes additionally serialise on the runner's
    `asyncio.Lock` via `HubSyncRunner.run_exclusive`. Checks write only `status.json`, which has
    its own file lock, and do not take the lease. The agent fingerprint re-check still has the
    narrow window of Q4 (it reads through the registry's 5 s cache when a live registry is
    injected), so the lease, not the re-check, is what stops two merges of one agent.
13. **Per-request hub timeout (B3.3, 20 s)** is a socket timeout the check path passes to
    `download_agent_from_marketplace(timeout=)` / `get_team(timeout=)`, plus a 300 s ceiling on the
    whole check phase of a tick. It is opt-in on the client, so interactive pulls behave as before.
14. **Team baselines are pruned only when the team is confirmed absent on disk** (no `teams/<id>`
    and no `.<id>.*` staging/backup sibling), never from the listing alone: `TeamRegistry._load`
    skips a row mid-swap and reads as empty on an `iterdir` error.
15. **Status semantics:** `applied` is kept for exactly one check cycle (`applied_seen`), then
    reverts to `up-to-date`; `last_error` is scrubbed by credential SHAPE (`redaction_shapes`),
    since it is built from provider exception text that no client scrubs; an explicit `--replace`
    / `force` acts on a copy that differs from the hub even when the hub has not moved.

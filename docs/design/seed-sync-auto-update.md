# Seed sync: drift truth, the revision ledger, and startup auto-update — design

Status: implemented, 2026-10-08. Author: architect (lopdev); implementation: coder (lopdev).
Base: `origin/main` e3176403b. Fixes issue #2060. Where a `path:line` is cited, it was read in
that tree.

## 1 What was wrong

Four defects, all verified with reproductions on the base tree:

1. **The false "diverged" was a fingerprint-formula era skew.** `seed_fingerprint` hashes a
   field LIST, and v0.64.9 appended `class` to it (`0b2dc18deb`), changing every hash. A row
   installed by v0.63.5–v0.64.8 recomputes to its recorded `seed_sha256:` only under the OLD
   five-field formula, so a byte-identical, untouched row answered `outdated-diverged` and
   told its owner to "re-run with force" — on a row nobody had edited, with a flag that is
   hidden and deprecated.
2. **Nothing reported drift, and read-only modes lied.** `sync_installed_seeds` ran only when
   a person typed `sync`; `agents sync --check` skipped the starter arm entirely (so
   `--name <starter> --check` answered *not installed*), and `--dry-run` — documented
   "Show what would change; write nothing" — APPLIED clean starter updates on disk.
3. **The remedy copy named a flag the command ignores.** The refusal said "re-run with
   force"; the documented spelling is `--replace --yes`, which the seed arm did not honour.
4. **A clean apply was destructive.** The apply path went through `install_seed(overwrite=
   True)`, which resets the display label and REPLACES the whole tag list — so an unattended
   update would have dropped a user's label, their own tags, and their model pin.

## 2 Decision summary

| Q | Decision |
|---|---|
| Auto-apply | YES, at startup, for rows the ledger proves unedited and strictly behind; every other state is reported or silent (matrix in §4/§5). |
| Identity | A committed revision ledger (`agent_seeds/seed_revisions.json`) supplies positions: every text that ever shipped, oldest first per seed. The install fingerprint stays as a belt for texts the ledger cannot see. `class` is carried but EXCLUDED from identity (§3). |
| Read-only | `sync_installed_seeds(..., apply=False)` writes no agent rows and no seed state; the unchecked arm renders "update available" and `counts()` gains an additive `available`. (Not a blanket "writes nothing anywhere" for a whole command run — the hub arm's status store and the pre-existing class backfill are the documented exceptions, §6.) |
| Startup | A fourth arm in `config_migrations.run_startup_migrations`, skipped for the whole `agents sync` command AND for `config edit agents.auto_update.seeds` (`_NO_WRITE_COMMANDS`); lock + narrow writer; notices per SURFACE: `cli` prints plain stderr lines, `tui` queues in `.seed-notices.json`, daemon launches (`serve` forms) are report-only and record nothing. |
| Setting | `agents.auto_update.seeds` (default true). Off = still reported, never written. |
| Out of scope | Desktop/dashboard rendering of `available`; an echo store for replaced text; a stamp-refresh migration; flag-validation carve-outs beyond `agents sync`. |

## 3 The revision ledger

`local_operator/agent_seeds/seed_revisions.json` ships inside the package beside the seeds and
the manifest. Shape:

```json
{"schema_version": 1,
 "seeds": {"aida": [{"sha": "<40-hex>", "version": "1.0.0",
                      "instructions_sha256": "<64-hex>", "description": "…",
                      "tools": null, "effort": null, "delegate": false,
                      "class": null}, …]}}
```

- **Generated, never hand-edited**, by `scripts/gen_agent_seed_revisions.py`. Normal mode
  appends the packaged revision when its full vector differs from the tail and refuses
  without an attributable commit sha; `--bootstrap-from-git` rebuilt the file from history
  once (one entry per distinct per-commit text, sha = the commit that first carried it);
  `--check` validates schema, append-order uniqueness and every seed's tail == the packaged
  starter WITHOUT git, so CI on a `fetch-depth 2` checkout can run it. The unit test
  re-renders for byte-identity, and — when history is resolvable — verifies each entry's
  hash against its git blob (skipped on shallow CI).
- **Bootstrap guards (QA round 1, Q3).** `--bootstrap-from-git` refuses when git MARKS the
  repository shallow (`rev-parse --is-shallow-repository`) unless `--allow-shallow` is
  explicit — reserved for a clone known to carry every seed's history despite the marker
  (the fleet's reference clone). Independently, a bootstrap that would leave FEWER
  revisions than the target already carries refuses and writes nothing, so the committed
  ledger can never be silently truncated (the measured repro: a depth-1 clone truncated
  65 revisions to 10, and `--check` then passed on the truncated file).
- **Seed-edit workflow.** Edit the seed file → COMMIT it (an entry names the commit that
  first carried the text, so an uncommitted seed refuses) → run the generator (normal mode
  appends) → commit the regenerated ledger in the same PR. The byte-identity test fails
  otherwise, naming the generator command.
- **Identity is five fields** — instructions (whitespace-canonicalised: CRLF→LF, per-line
  trailing whitespace stripped, outer blanks stripped), routing text, tools, effort,
  delegate. `class` is CARRIED per entry but excluded from identity: aida's backfilled
  `class:proactive` must still match her pre-class v1.0.0 entries, which is the row this
  feature exists for. The normalisation measured the identity on every committed seed
  version that shipped; it is cheap insurance, never applied to capability fields.
- **Direction, not versions.** Version strings repeat (aida shipped four `1.0.0` builds, two
  same-day), so "behind" and "ahead" are defined as ledger ORDER. A row AHEAD of the package
  (a downgraded `lop`, a channel switch) reads "newer than this build ships" and is never
  flipped back — two builds of different ages must not ping-pong a row (§9.2 of the hub
  design is the same doctrine).

## 4 Classification (`sync_installed_seeds`)

Order matters; both the typed surfaces and the startup pass share this one function:

1. No divergence from the packaged seed → `up-to-date`.
2. Row matches a ledger entry AND the packaged text has one too → position decides:
   same → `up-to-date`; strictly behind → `outdated-clean` (applied through the narrow
   writer; `behind_by` set); ahead → `up-to-date` ("newer than this build ships").
3. No ledger position, but the stamp recomputes under EITHER shipped formula → `outdated-clean`
   (typed surfaces only — see §5; a stamp proves "unchanged", not direction).
4. The stamp equals the PACKAGED starter's fingerprint while the row differs → `up-to-date`
   with a local-edits note (nothing to PULL; `op='reset'` restores the packaged text).
5. Anything else → `outdated-diverged`, reported with the diverged field list and applied
   only with an explicit replace; the copy names `--replace --yes`.

Pre-stamp rows (installed before `seed_sha256:` existed) have no fingerprint and are
re-proved by the LEDGER: a row whose text is a published revision is clean by construction.
A row the ledger cannot place either needs the explicit replace once.

**The narrow writer** (`_apply_seed_update`) exists because of §1.4: it replaces only what the
seed owns — instructions, routing description, seed-owned tags, the class (below) and the
provenance stamps — and leaves the label, model, categories, security prompt and every
non-seed tag alone. Order is the crash contract: the prompt lands first (already atomic),
the tag/stamp rewrite last, and a CAUGHT failure of the second half rolls the prompt back so
the row is exactly as it was and re-applies next launch (agent review round 1, R1-7). The
uncatchable case — a process killed between the two writes — is stated rather than papered
over: a prose-only revision then reads "matches the packaged starter" (correct text, stale
stamps — cosmetic), while a revision that also moved the description leaves a row that
differs in description alone and needs the explicit `--replace --yes`/reset. The registry's
`update_agent` writes `agent.yml` atomically for the same class of reason (a truncated file
refuses EVERY profile launch while it exists).

**The forced path echoes what it discards** (UX round 1, U5): the wholesale writer resets the
label and drops non-seed tags, so an applied forced replace reports the removed label (when
it differed from the packaged spelling) and the non-seed tag names beside the replaced
fields and instructions — otherwise neither was recoverable from the receipt.

**The class rule** (subtle, so written down): the packaged class is written iff the row has
no class tag OR its class equals the class of ANY ledger entry whose other five fields match;
otherwise the row's own class is preserved. A deliberately switched class is user data like
the label; the packaged class reaching a switched row stays `reset`'s job — an accepted
trade, not an oversight.

## 5 The startup pass

One new arm in `config_migrations.run_startup_migrations` (`surface`, `command` keyword-only),
after the class backfill, before the projects arm. What it does:

- `command` in `_NO_WRITE_COMMANDS` skips the arm entirely: `agents sync` (`--check`/
  `--dry-run` write no agent rows, and a startup write would race the state the command is
  inspecting) and `config edit agents.auto_update.seeds` — the command whose purpose is to
  change this pass's switch, which a racing pass would pre-apply ahead of (UX round 1, U6b).
- No `agents/` directory → return before constructing anything (storeless machines stay
  storeless). Corrupt/missing ledger → silent return.
- Classify every installed seed-origin row (revisions loaded once). A row is auto-apply
  ELIGIBLE iff: `outdated-clean` with `behind_by` set (ledger positioned BOTH ends), the
  delta touches only `instructions`/`description`/`class` — `class` is NOT a hold test, it
  is user data the narrow writer preserves via the rule above (agent review round 1,
  R1-2); the packaged text equals the ledger's tail (drafts in worktrees are never
  auto-applied), `install_kind()` ∈ uv-tool/pipx/pip, and the setting is on. Held rows are
  capability deltas ONLY (`tools`/`effort`/`delegate` — a boundary is never widened
  unattended). Everything else is held, reported or silent, never written.
- Writes happen under `WakeWriteLock(config_dir, name=".seed-sync.lock", timeout_s=2.0)` (a
  contended launch must not stall a TUI start for seconds — UX round 1, O2), and RE-VERIFY
  the classification on a refreshed registry before writing.
- Notices, per SURFACE (agent review round 1, R1-1; design round 1, D3): the de-dup token is
  `{surface}:{seed}:{packaged-revision-sha}`, so a notice consumed by a log-only launch can
  never silence the interactive one. `cli` prints the line plainly to stderr (no log prefix)
  and records only its own slot; `tui` queues into `.seed-notices.json` for the boot hook,
  which PEEKS, displays, then clears (a crash between shows a duplicate, never a silent
  loss); DAEMON launches (`lop serve` and the foreground `serve` forms of `wake`, `mobile`,
  `browser`, `tunnel`, `network`) are REPORT-ONLY: they neither write a row nor record a
  token nor queue a line (DEBUG log only). A daemon that applied and recorded nothing would
  lose the applied notice for good - the next human launch finds the row current and says
  nothing - so the first human surface applies and announces instead. (The supervised
  LaunchAgents run `-m local_operator.<unit>` directly and never reach this seam; the
  `serve` forms are what does, via `lop serve` and the CLI spellings. `mobile start` is a
  control command a person types, so it keeps the `cli` slot.)
- What a notice may say, and when (round-1 rework): applied rows carry the version
  transition captured PRE-write and are composed from the rows the re-check actually
  applied; held rows are capability deltas, always individual and naming `--check` as the
  review surface; update-available and edited-and-moved rows are reported under the gates
  above. More than two rows in one group collapse to a single rolled line that lists only
  the rows THIS surface has not already been told about; ≤2 stay individual. An APPLIED
  line is never de-duplicated against a prior "available" line — a write is always told,
  even to someone who flipped the setting on after being warned — and a `cli` launch also
  queues applied lines for the next TUI boot: the row is current by then, so the TUI could
  never re-derive the event, and the printed line may have gone to a bash call nobody read.
  Report-style lines are not echoed (the TUI announces those itself under its own token),
  and `pending` is capped, oldest first, so a machine that never opens the TUI cannot grow
  the queue forever. The OFF-SWITCH POINTER `(stop auto-updates: /settings → Agents →
  Auto-update built-in roles)` rides only the APPLIED lines, individual and rolled-up
  (design round 2, D2-1): the update-available notice fires exactly when turning the
  channel off cannot quiet it — the setting is already off, or the install is EDITABLE — so
  it carries no pointer; the applied lines are the surprise the pointer exists for, and
  they do.

Why not trust a stamp unattended: a stamp proves the row is what SOME build installed; only
a ledger position proves the packaged text is PUBLISHED and the row is behind it. That is
why eligible rows must be ledger-positioned, and why a worktree venv (editable install,
sharing the operator's real config dir) never auto-writes.

## 6 Read-only modes and notices

- `--check`/`--dry-run` (`apply=False`): classification runs, `applied` stays False,
  `behind_by` is populated, renderer prints "update available — run `lop agents sync` to
  apply it" (with a capability note for any row whose delta touches tools/effort/delegate).
  On disk: no agent rows — the starter arm writes nothing; the hub arm's check still
  refreshes its own status store, and the pre-existing class backfill may repair a row, both
  as before (agent review round 1, R1-6 / QA O1). The precise claim is "no agent-row
  writes", not "writes nothing".
- `--replace --yes` is honoured by the seed arm (`--force` remains the hidden deprecated
  alias, warned exactly once per run); an UNCONFIRMED `--replace` changes nothing — the flag
  surface is validated in `_prepare_sync_flags` BEFORE the seed arm, so the refusal exits 1
  having written nothing (agent review round 1, R1-5 / QA round 1, Q2; the pre-fix repro
  showed the clean rows rewritten before the hub arm refused the pair). With the pair, the
  seed arm applies even if the hub arm afterwards refuses (HubBusy): the pair IS the user's
  discard instruction, and the two families are independent.
- Notices: CLI prints plain stderr lines; the TUI surface queues into `.seed-notices.json`,
  and `tui._schedule_seed_update_notices` delivers them onto `app._system_notice` on boot —
  PEEK, display, then clear (a crash between shows a duplicate next boot, never a silent
  loss; agent review round 1, R1-3). That file is display-only state — documented exception
  to the seam's "no stamp file" doctrine, whose reason (a stale stamp could SKIP a
  migration) cannot apply to a notice.

## 7 The setting

`agents.auto_update.seeds` (bool, default true, section "Agents", Scope.NEW_LAUNCH),
registered in `settings_io` and read by `agent_profiles._auto_update_seeds_enabled` through
`ConfigManager.get_nested_value(("agents","auto_update","seeds"), True)`. Non-bool values
degrade to the default. The test mapping (`tests/unit/test_settings_io.py`) pins the
registry's literal default against `AUTO_UPDATE_SEEDS_DEFAULT`, the module that READS it.

## 8 Deliberately deferred

- Desktop/dashboard rendering of `available`/`behind_by` (out of tree; the summary key is
  additive so old readers ignore it).
- An echo store for startup-replaced prompt text (the notice names previous version and
  commit instead; a request for copy-paste recovery is the trigger to add one).
- A stamp-refresh migration (the ledger already answers; refreshing stamps would be
  cosmetic).
- Held-row re-notification cadence beyond once per revision, and a persistent status-surface
  indicator (UX round 1, U8): the per-surface split, the `--check` capability note and the
  once-per-revision key keep it visible without nagging; a status indicator is a follow-up.
- Hub-side "re-run with force" copy in `agents.py::sync_hub_agents` (agent review round 1,
  R1-11): hub flag semantics are unchanged in this PR; out of scope here.

## 9 Test plan

- Failing-before/passing-after per defect, on scratch seeds + a scoped ledger writer
  (`publish_seed` in `tests/unit/test_agent_profiles.py`).
- Negative cells: edited/ahead/unprovable rows silent (or reported only with version
  proof); capability deltas held; classes, labels, tags and model pins preserved;
  `--check`/`--dry-run` write no agent rows; unconfirmed `--replace` byte-level changes
  nothing; lock held → skip; corrupt ledger → silence.
- Round-1 cells: surface-scoped de-dup (cli-then-tui still announces; daemon consumes
  nothing; second tui launch silent); class-switched rows auto-apply with the class kept;
  applied notices carry the PRE-write version; roll-ups for >2 seeds and individual lines
  for ≤2; edited-and-moved rows reported; the narrow writer's rollback on a caught failure;
  `config edit agents.auto_update.seeds` writes nothing and the next launch honours the new
  value; shallow-clone bootstrap refusal plus `--allow-shallow`/count-guard behaviour; prose
  canonicalisation (CRLF/trailing whitespace), a corrupt `.seed-notices.json` on the start
  path, and a non-bool setting value.
- The seam: storeless machines stay storeless; `agents sync` skips the arm (with a control
  run proving the setup WOULD have written).
- Ledger honesty in `tests/unit/test_agent_seed_revisions.py`: byte-identity regeneration,
  `--check` semantics, per-entry sha-to-blob verification and exhaustive matcher coverage,
  both same-day aida `1.0.0` builds present as distinct entries.

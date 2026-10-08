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
| Read-only | `sync_installed_seeds(..., apply=False)` is write-nothing at the byte level; the unchecked arm renders "update available" and `counts()` gains an additive `available`. |
| Startup | A fourth arm in `config_migrations.run_startup_migrations`, skipped for the whole `agents sync` command; lock + narrow writer; notices via `logger.info` (cli) or `.seed-notices.json` (tui). |
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
  re-renders byte-identically and, where history resolves, verifies each entry's sha against
  its blob (skipped on shallow checkouts).
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
the tag/stamp rewrite last, so a crash leaves the row re-appliable, never marked-updated-
over-an-old-prompt. The registry's `update_agent` writes `agent.yml` atomically for the
same class of reason (a truncated file refuses EVERY profile launch while it exists).

**The class rule** (subtle, so written down): the packaged class is written iff the row has
no class tag OR its class equals the class of ANY ledger entry whose other five fields match;
otherwise the row's own class is preserved. A deliberately switched class is user data like
the label; the packaged class reaching a switched row stays `reset`'s job — an accepted
trade, not an oversight.

## 5 The startup pass

One new arm in `config_migrations.run_startup_migrations` (`surface`, `command` keyword-only),
after the class backfill, before the projects arm. What it does:

- `command == "agents sync"` skips the arm entirely: `--check`/`--dry-run` promise no writes,
  and a startup write would race the state the command is inspecting.
- No `agents/` directory → return before constructing anything (storeless machines stay
  storeless). Corrupt/missing ledger → silent return.
- Classify every installed seed-origin row (revisions loaded once). A row is auto-apply
  ELIGIBLE iff: `outdated-clean` with `behind_by` set (ledger positioned BOTH ends), the
  delta touches only `instructions`/`description`, the packaged text equals the ledger's tail
  (drafts in worktrees are never auto-applied), `install_kind()` ∈ uv-tool/pipx/pip, and the
  setting is on. Everything else is held or reported, never written.
- Writes happen under `WakeWriteLock(config_dir, name=".seed-sync.lock")` (bounded retry; a
  held lock skips the launch silently) and RE-VERIFY the classification on a refreshed
  registry before writing.
- Held rows (capability deltas: `tools`/`effort`/`delegate`) and report-only rows (setting
  off; editable/pip-user installs) get notices; a fresh notice is de-duplicated per seed ×
  packaged revision sha in `.seed-notices.json`.

Why not trust a stamp unattended: a stamp proves the row is what SOME build installed; only
a ledger position proves the packaged text is PUBLISHED and the row is behind it. That is
why eligible rows must be ledger-positioned, and why a worktree venv (editable install,
sharing the operator's real config dir) never auto-writes.

## 6 Read-only modes and notices

- `--check`/`--dry-run` (`apply=False`): classification runs, `applied` stays False,
  `behind_by` is populated, renderer prints "update available — run 'lop agents sync' to
  apply it". On disk: nothing. (The hub arm's check still refreshes its own status store, as
  before.)
- `--replace --yes` is honoured by the seed arm (`--force` remains the hidden deprecated
  alias); `--replace` WITHOUT `--yes` changes nothing anywhere — the seed arm runs before
  `_hub_sync_run` validates the pair, so the force flag is derived from `replace AND yes`,
  never `replace` alone. With the pair, the seed arm applies even if the hub arm afterwards
  refuses (HubBusy): the pair IS the user's discard instruction, and the two families are
  independent.
- Notices: CLI logs `logger.info` lines (visible on stderr); the TUI surface queues into
  `.seed-notices.json`, and `tui._schedule_seed_update_notices` drains them onto
  `app._system_notice` on boot, clearing `pending` but keeping `announced`. That file is
  display-only state — documented exception to the seam's "no stamp file" doctrine, whose
  reason (a stale stamp could SKIP a migration) cannot apply to a notice.

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
- Flag-validation carve-outs beyond `agents sync` (none known to promise no-write).

## 9 Test plan

- Failing-before/passing-after per defect, on scratch seeds + a scoped ledger writer
  (`publish_seed` in `tests/unit/test_agent_profiles.py`).
- Negative cells: edited/ahead/unprovable rows silent; capability deltas held; classes,
  labels, tags and model pins preserved; `--check`/`--dry-run` byte-level write-nothing;
  unconfirmed `--replace` changes nothing; lock held → skip; corrupt ledger → silence.
- The seam: storeless machines stay storeless; `agents sync` skips the arm (with a control
  run proving the setup WOULD have written).
- Ledger honesty in `tests/unit/test_agent_seed_revisions.py`: byte-identity regeneration,
  `--check` semantics, per-entry sha-to-blob verification and exhaustive matcher coverage,
  both same-day aida `1.0.0` builds present as distinct entries.

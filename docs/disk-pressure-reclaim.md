# Disk-pressure reclaim — `lop disk reclaim`

The sanctioned, measure-first, fail-closed way to recover disk. It exists so that
nobody needs an ad-hoc `rm -rf` / `shutil.rmtree` sweep under pressure (issue
#2676: one such sweep deleted a live runtime generation).

```
lop disk reclaim [--repo PATH]... [--go-build-cache] [--scratchpads]
                 [--min-idle-minutes N] [--apply [--yes]] [--force] [--json]
```

* **Default is a dry run.** It lists what *would* be reclaimed and what is **kept,
  with the reason**, grouped by reason so you can act on a whole class. Nothing is
  removed.
* **`--apply`** re-prints the preview, asks for a typed `yes` (`--yes` skips the
  prompt for scripts), then removes **only the previewed items**, re-checking
  every condition at removal time. An item that stopped qualifying is skipped and
  reported.
* **`--json`** prints the receipt document for either mode. Apply also writes it
  to `<config>/disk-reclaim-receipts/<timestamp>-<pid>.json` — first as an
  *intent* before the first removal, then updated after each item — with
  `shutil.disk_usage` readings before and after. Every write records
  `confirmed: true` and the `confirmation` that authorised the run (`"--yes"` or
  `"typed"`), including the intent, so the file alone says how the removal was
  approved even if the process died mid-sweep. `apply()` refuses a provenance it
  does not recognise rather than writing one that guesses.
* Exit codes, in precedence order: `2` usage error, a **refused category**
  (a dry run that refuses a requested category also exits 2, so a script cannot
  mistake a half-answered preview for a full one) or confirmation
  declined/unavailable; else `1` an item failed or was skipped at removal time, or
  a receipt update failed (`receipt_warnings` in the JSON, a `warning:` line on
  stderr); else `0`.
* Naming no category is a usage error. Nothing is defaulted or discovered.

## Sizes are measurements, not promises

Sizes are *allocated* bytes (`st_blocks*512`) from a bounded walk (`>=` means the
walk was truncated: a lower bound). Hardlinks, shared git objects, APFS clones and
block rounding mean **allocated or logical bytes are not reclaimed bytes**. The
only honest figure of what was freed is the receipt's before/after
`disk_usage`.

## Categories

### Linked git worktrees — `--repo PATH` (repeatable)

You name each repository; its registered *linked* worktrees are enumerated with
`git worktree list --porcelain`. The primary checkout is never a candidate and
nothing is removed except through `git worktree remove <path>` **without
`--force`** — git's own refusal is the last word. A worktree is reclaimable only
when **all** hold; the first that fails is the reported reason:

| Predicate | Kept as |
|---|---|
| git can inspect it; registration valid, directory exists, belongs to this repo | `uninspectable`, `invalid-registration` (never deleted or pruned here) |
| not locked (reason shown) | `locked` |
| not this process's cwd / the `--repo` argument | `cwd` |
| no process has its cwd inside it (`lsof`) | `in-use`; `lsof` unavailable → `in-use-unknown` |
| idle ≥ `--min-idle-minutes` (default and **floor 60**; lower is rejected, exit 2) | `recent`; walk bound hit or unreadable → `idle-unproven` |
| no git operation half-done (rebase/merge/cherry-pick/revert/bisect markers in its gitdir) | `operation-in-progress` |
| clean: `git status --porcelain --untracked-files=all --ignored` shows no changes | `dirty` |
| no **gitignored content** at all (a `.env`, a nested repo under an ignored `tmp/`, even `node_modules`) and no nested `.git` | `ignored-content` |
| pushed (HEAD contained in a remote-tracking ref, branch at HEAD) **or** merged (HEAD an ancestor of `origin/HEAD`\|`origin/main`\|`origin/master`) | `unpushed` |

*Why ignored content keeps the worktree.* Removal destroys nothing but the
working-tree directory (branch refs and objects stay in the shared repository), so
gitignored files are exactly what could be lost for good. There is deliberately
**no build-cache carve-out**: delete a cache directory yourself if you want that
worktree reclaimed.

*Repeated or overlapping `--repo` arguments* (the same repository, or a linked
worktree of it) are collapsed to one enumeration. `--min-idle-minutes` below 60 is
rejected by the planner itself, not only by the CLI. A moved HEAD between preview
and removal is skipped as `head-moved`.

*Idle* is the newest mtime of the worktree's admin files (`HEAD`, `index`,
`logs/HEAD` in its gitdir) and a bounded walk of the working tree (`.git`
skipped). *In use* is a positive signal only (`lsof` cwd table); when it cannot be
established the worktree is kept. Remote refs are as last fetched — no network is
used, which can only make "pushed/merged" more conservative.

**Tracked or unmerged worktree evidence is never cache.** A worktree with
unpushed commits, uncommitted changes or a lock is *kept*; deciding about it is
the owner's job. External hygiene tools must call this command (or stay
measure-only) rather than sweeping worktree directories themselves.

### Go build cache — `--go-build-cache`

Resolved with `go env GOCACHE` and refused unless it is an absolute, existing,
non-symlink directory that is not `/`, `$HOME` or a parent of it, does not contain
the config dir, and carries Go's marker `README` ("This directory holds cached
build artifacts from the Go build system"). Removed only with `go clean -cache`
run with an explicit minimal environment: `GOCACHE` pinned to the validated path,
`GOFLAGS` cleared and `GOENV=off` (so `GOFLAGS=-modcache` cannot widen the clean
to the module cache). Dry run measures only.

### Session scratchpads — `--scratchpads`

Composes with `lop sessions cleanup --scratchpads-only`: the same
`run_delegated_pass` dry-run preview, then a confirmed apply confined to the
previewed pads, under the same config policy, retention window
(`max_age_hours`), `--force` semantics and guards (live claim/lease, armed wake,
unread mail, current session, manifests). There is no retention override.
Transcripts, session records, config and credentials are never touched. The
build/cache vocabulary is the one in `local_operator/scratchpad.py`; Mongo and
other arbitrary data directories are deliberately **not** in it. This change added
`.gocache` and `gocache` (a pad-local `GOCACHE`); because the same vocabulary also
governs scratchpad *writes*, an agent can no longer write files under a directory
of those names (case-insensitive, and `gocache.<x>` via the dot boundary), exactly
like `node_modules` or `target` today.

## Never touched

Transcripts, session records, config, credentials/secrets, arbitrary user files,
and the primary checkout.

## Implementation

`local_operator/disk_pressure.py` (planner, predicates, applier, receipt) and
`local_operator/disk_cli.py` (parser + presentation). Tests:
`tests/unit/test_disk_pressure.py`.

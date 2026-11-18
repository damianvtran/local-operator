# Move carries the session working set

Status: design (architect, lopdev, 2026-11-18). Base: `origin/main` @ `3a5a6d957`.
Owner of the transfer primitives this reuses: the mesh-replication lane
(session `7c8130c573dd`, project `mesh-remote-offload`). Audit foundation:
PR #4228 (`AuditLog.record_required`), unmerged at time of writing.

## 0. Summary

`lop sessions move <id> --to <peer>` today carries the session directory's copy
set — transcript, `memory-session.jsonl`, scratchpad tree, asks, goal, wakes,
monitors' specs and linked project rows (`network/sync.py:138` `COPY_SET_NAMES`,
`COPY_SET_TREES = ("scratchpad",)`; `network/carry.py:799`/`:831` for projects).
It does **not** carry the work the session did *outside* its directory: git
worktrees, their unpushed commits, and their staged / unstaged / untracked
changes. A user who moves a coding session mid-task arrives on a device where
none of that exists, and nothing told them before the source let go.

This design adds a **working set** to the move:

1. **Manifest** — the owner computes, read-only, which git worktrees the session
   (and its subagent tree) worked in, and for each: branch, upstream, base,
   unpushed commits, dirty state split into staged / unstaged / untracked, bytes,
   secrets excluded, and what is left behind (dependency trees).
2. **Confirmation gate** — a non-empty manifest must be confirmed. Every surface
   (CLI prompt / `--confirm-working-set`, TUI dialog, desktop dialog) shows the
   manifest and the destination device name; the move call carries the manifest's
   **scope digest**, and the owner refuses an unconfirmed or stale-scope move
   server-side (`working_set_unconfirmed` / `working_set_changed`, `changed: false`).
3. **Transfer** — per repo, one git bundle of the commits the destination lacks
   plus three snapshot commits (HEAD-anchored index tree, worktree tree) staged as
   immutable, digest-named artifacts on the owner and pulled by the destination
   over a new **move-scoped** phase (`net_session_move {phase:"workset_*"}`) — not
   `net_sync`, not the replica copy set, no second transport, no GitHub writes.
4. **Restore** — after the promote, the destination registers each worktree
   against its own copy of the primary repo at the same home-relative path,
   checks out the branch with its upstream, and restores dirty state *as dirty*
   (index ≠ worktree ≠ HEAD preserved exactly).
5. **Safety** — secret-shaped files excluded and listed; secret-shaped paths or
   content in the bundled *history* flagged and that worktree left behind unless
   explicitly included; a measured size cap; a durable audit record before the
   first served byte; the existing move authorisation and lease epoch; the source
   worktrees are never deleted, only marked "moved to <peer> at <ts>".

**Recommended delivery: three PRs**, because the full feature is not one coder's
two hours (see §15):

| PR | Repo | What | Stops the harm? |
|---|---|---|---|
| **A** | local-operator | Manifest + preflight + confirmation gate + receipt block. Moves still do not carry worktrees, but the user is told *exactly* what stays behind and must confirm it. | Yes — nobody is surprised any more. |
| **B** | local-operator | Transfer + restore + recall + mark-moved + audit event. | Makes the work actually arrive. |
| **C** | local-operator-ui | Move dialog shows the manifest, requires the confirmation checkbox, sends the scope digest; gated on `session_transfer_working_set`. | Desktop parity. |

PR A alone is the smallest change that removes the operator's complaint
("people could use the move function and then realize all the work is not
available"); B delivers the operator's goal; C is the operator's 2026-11-18 scope
addition (the popup must require approval).

## 1. Non-goals

- **Turn handoff.** A busy session is still refused or queued (§6.4 of
  `mesh-session-mobility.md`); nothing here moves a turn mid-flight.
- **Pushing to GitHub/GitLab.** No remote writes, ever; the destination may only
  *fetch/clone* from origin with its own credentials.
- **Dependency trees** (`node_modules`, `.venv`, build output, gitignored files):
  never transferred. PR B *names* the recreate command; running it automatically is
  phase 2 (§9).
- **Loose non-git files** written outside any repo and outside the scratchpad:
  inventoried and listed as *left behind* in PR A/B; transferring them is phase 2
  (§14). Measured need is low (§8.4).
- **Submodules, Git LFS objects, sparse checkouts, worktrees mid-merge/rebase**:
  detected and left behind with a reason code (`submodule`, `lfs`, `unmerged`,
  `in_progress_operation`) — phase 2.
- **Deleting source worktrees.** Never automatic; cleanup is offered and needs the
  operator's approval.
- **Expanding the replica copy set or calling `sync.serve_fetch` on workspace
  paths** — explicitly ruled out by the mesh lane (its plan/item resolution and
  holder bookkeeping are replica-only, `sync.py:1930` `_item_path` is an allow-list
  over the session directory and must stay so).

## 2. What travels today, and after this work

The operator's 2026-11-18 addition asks that "projects, worktrees, working files,
scratch, session memory, etc." all carry over. Inventory, from the code:

| Item | Carried today? | After PR A | After PR B | Mechanism |
|---|---|---|---|---|
| Transcript (+ compressed archive forms) | yes | yes | yes | `COPY_SET_NAMES` (`sync.py:138`) |
| Scratchpad (`<session>/scratchpad/`) | yes | yes | yes | `COPY_SET_TREES` (`sync.py`, `"scratchpad"`) |
| Session memory (`memory://` session scope) | yes | yes | yes | `memory-session.jsonl` in `COPY_SET_NAMES` |
| Todos | yes | yes | yes | `todo_snapshot` custom entries ride the transcript (`session/session.py:727`) |
| Wakes | yes (spec + rebuilt index) | yes | yes | transcript rows + `carry.rebuild_indexes` at promote (`carry.py:497`, `mobility.py:2762`) |
| Monitors | specs yes; observations no (re-baseline) | yes | yes | transcript + `carry.monitor_count` notice (`carry.py:766`) |
| Linked projects | yes (row + last N updates; attachments not) | yes | yes | `carry.linked_project_rows`/`adopt_projects` (`carry.py:799`/`:831`, `mobility.py:2796`) — **no change needed**: the store is per-device and the row is already merged on arrival. |
| Asks, goal, inbox, boot prompt, eval kernel stamp | yes | yes | yes | `COPY_SET_NAMES` |
| Session roles/teams definitions | no unless `--drop-session-roles` acknowledged | same | same | `mobility.py:3506`–`3777` |
| Attachments referenced by the transcript | yes (content-addressed) | yes | yes | `sync.ATTACHMENT_PREFIX` (`sync.py:115`) |
| File provenance (`.file-provenance/`) | **no, by design** | no | no | `sync.py:381` "owner-local witnesses; replicas/imports cannot issue proofs" — read on the source as a manifest *input* only (§3.2) |
| Session cwd | rewritten, contents not carried | **listed** | worktree carried when cwd is in one | `carry.rewrite_host_cwd` (`carry.py:346`) + this design |
| Git worktrees: commits, branch, upstream | **no** | **listed, confirmed** | **yes** | §4–§6 |
| Dirty state (staged/unstaged/untracked) | **no** | **listed, confirmed** | **yes, as dirty** | §5 snapshot trees |
| Secret-shaped files | no | listed as excluded | excluded + listed | §8.1 |
| Dependency trees / build output / ignored | no | listed with recreate command | same + `--setup-deps` (phase 2) | §9 |
| Loose files outside repos/scratchpad | no | listed | listed (phase 2: carried) | §14 |

So the only "carry" gaps are the working-directory contents; scratch, session
memory, todos, wakes and projects already travel and this design does not touch
them.

## 3. The manifest

### 3.1 Where it is computed

Always on the **owner** (the device holding the session and its worktrees), because
only it can read them. Two entry points, same function
`workset.compute_manifest(root, session_id, *, to_device, home) -> Manifest`:

- **Preflight** (read-only, runtime NOT retired): a new move phase
  `net_session_move {phase:"workset_preflight", session_id, to_device}` answered by
  the owner's relay. For a local session (offload, `mobility._offload`,
  `mobility.py:6145`) the local relay answers it in-process; for `--to local`
  (recall) it crosses the link. Gated by the same `move` capability check as
  `prepare` (`mobility._peer_may_not_take`, `mobility.py:6103`; authoriser
  `network/authorizer.py`).
- **Prepare** (after the retire, §6 step P3): recomputed so the snapshot is of a
  quiescent tree, and compared against the confirmed scope digest.

### 3.2 Candidate discovery (which worktrees)

Bounded and deterministic: build a finite candidate set of repositories, then a
finite candidate set of their worktrees, then keep the ones this session tree
touched.

1. **Session tree** = the session plus every subagent child session it launched,
   recursively. Children are separate session directories; the parent's
   `subagent-lane-<job>.v1.json` / `subagent-stop-<job>.v1.json` receipts carry
   `child_session_id` (`session/subagent_ledger.py:204`, `:222`, `:373`).
2. **Path seeds**, per session in the tree:
   - the recorded cwd (`carry.resolve_cwd(session_dir)`, `carry.py:186`) and every
     earlier cwd recorded by cwd moves;
   - **file provenance records** (`<session>/.file-provenance/*.json`, PR #4108,
     `session/file_provenance.py`) — each record's `file.path`. Evaluation: useful
     but **not sufficient as the authority**. It is metadata-only and owner-local
     (good: no bytes, never copied), covers user/assistant text, absolute tool args
     and `bash` literal paths (QA #4108 rows a1/b1/b3), but excludes the scratchpad
     and config root, sensitive paths, files > 256 MiB, unresolvable shell
     expansions, and every session older than #4108. Measured on this node:
     **11 of 461 sessions** have a `.file-provenance/` directory.
   - **literal transcript mentions**: for each candidate worktree path from step 3,
     a substring search of the session's `transcript.jsonl` (and archive forms,
     read through the existing restore) for that path or its `~/` form. This is the
     fallback that covers old sessions and `cd`/`git -C` usage provenance cannot see.
3. **Repositories** = `git rev-parse --git-common-dir` of every seed that is inside a
   git work tree. **Worktrees** = `git worktree list --porcelain` of each repository.
4. **Keep** a worktree when any of: a seed lies inside it; its path (or `~/` form)
   is literally mentioned in a session-tree transcript; its admin directory
   `<common>/worktrees/<name>` was created after the session's `created_at.json`
   **and** is mentioned (creation alone is not attribution — concurrent sessions
   create worktrees too).
5. Each kept worktree records `sources: ["cwd" | "provenance" | "mention" |
   "subagent:<child_id>"]` so the user can see why it is in the set.

**Contention**: a kept worktree that is the cwd of another *live* session outside
this tree (live per the runtime registry, `session/runtime/registry.py`; cwd via
`carry.resolve_cwd` of that session's directory) is marked
`state: "left_behind", reason: "contended", by: [<session ids>]`. Moving it would
hand a tree another writer is using to a second device.

### 3.3 Per-worktree facts (all read-only)

```
head        git rev-parse HEAD
branch      git symbolic-ref -q --short HEAD          (absent → detached)
upstream    git rev-parse --abbrev-ref @{upstream}    (absent → none)
remote_url  git remote get-url <upstream remote or origin>, userinfo stripped
base        git merge-base HEAD <upstream or origin/HEAD>
unpushed    git rev-list --count HEAD --not --remotes   (commits no remote-tracking ref has)
status      git status --porcelain=v2 -z --untracked-files=all
            → staged / unstaged / untracked counts and paths; unmerged entries
              (u lines) or an in-progress op (MERGE_HEAD, rebase-*, CHERRY_PICK_HEAD)
              → left_behind reason "unmerged"/"in_progress_operation"
dirty_bytes sum of st_size over staged/unstaged/untracked paths (upper bound;
            the snapshot blobs are deflated)
deps        lockfiles present + whether node_modules/.venv exist (§9)
excluded    §8.1
```

### 3.4 Receipt / preflight data shape (additive)

A new optional member `working_set` on both `SessionMoveResult`
(`network/mobility.py:492`) and `SessionMoveRefusal` (`mobility.py:575`),
`NotRequired` exactly like `engagement`/`queue`/`carry`: absent for every move
whose manifest is empty, so the frozen contract is unchanged for chat-only
sessions. `test_the_session_move_contract_is_frozen` is updated deliberately (one
additive optional key on each shape), the same way `state` and `carry` were added.

JSON Schema (draft 2020-12):

```json
{
  "$id": "lop://session_move/working_set/v1",
  "type": "object",
  "required": ["version", "scope_digest", "state", "destination", "totals", "worktrees", "excluded", "left_behind"],
  "properties": {
    "version": {"const": 1},
    "scope_digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    "state": {"enum": ["preflight", "carried", "partial", "pending_restore", "left_behind", "unsupported"]},
    "destination": {"type": "object", "required": ["device_id", "name"],
      "properties": {"device_id": {"type": "string"}, "name": {"type": "string"}}},
    "totals": {"type": "object", "required": ["worktrees", "unpushed_commits", "dirty_files", "bytes"],
      "properties": {"worktrees": {"type": "integer"}, "unpushed_commits": {"type": "integer"},
        "dirty_files": {"type": "integer"}, "bytes": {"type": "integer"},
        "cap_bytes": {"type": "integer"}}},
    "worktrees": {"type": "array", "items": {"type": "object",
      "required": ["path", "repo", "head", "state"],
      "properties": {
        "path": {"type": "string", "description": "home-relative (~/…) on the source"},
        "dest_path": {"type": "string", "description": "home-relative on the destination; differs only on conflict (§7)"},
        "repo": {"type": "string", "description": "home-relative primary checkout"},
        "remote_url": {"type": "string"},
        "branch": {"type": "string"}, "upstream": {"type": "string"}, "base": {"type": "string"},
        "head": {"type": "string"},
        "unpushed_commits": {"type": "integer"},
        "staged": {"type": "integer"}, "unstaged": {"type": "integer"}, "untracked": {"type": "integer"},
        "bundle_bytes": {"type": "integer"}, "snapshot_bytes": {"type": "integer"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "state": {"enum": ["planned", "carried", "restored", "pending_restore", "left_behind"]},
        "reason": {"type": "string", "description": "machine code when left_behind: contended, outside_home, unmerged, in_progress_operation, submodule, lfs, history_secret, over_cap, destination_conflict, unsupported_by_peer"},
        "recipes": {"type": "array", "items": {"$ref": "#/$defs/recipe"}},
        "markers": {"type": "array", "items": {"type": "string"}, "description": "toolchain marker files at the worktree root (go.mod, pnpm-lock.yaml, ...)"},
        "clone_check": {"enum": ["pending_destination", "ok", "destination_cannot_fetch", "destination_cannot_push", "destination_cannot_clone", "unknown_destination", "no_remote"], "description": "the DESTINATION's own answer (§11.1 destination probe); pending_destination = not probed"},
        "clone_detail": {"type": "string", "description": "first line of git's stderr, or url_refused / git_missing / probe_error"},
        "https_rewrite": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}, "fetch": {"type": "boolean"}, "push": {"type": "boolean"}}, "description": "offered only; applied at restore (PR B)"},
        "remedy": {"type": "object", "required": ["command"], "properties": {"run_on": {"const": "lending_device"}, "command": {"type": "string"}}},
        "missing_tools": {"type": "array", "items": {"type": "string"}}
      }}},
    "destination_probe": {"type": "object", "required": ["state"], "properties": {
      "state": {"enum": ["answered", "unknown_destination"]},
      "missing_tools": {"type": "array", "items": {"type": "string"}},
      "team_bridge": {"type": "object", "properties": {"needed": {"type": "boolean"}, "enrolled": {"type": "boolean"}}}}},
    "team_bridge_used": {"type": "boolean"},
    "excluded": {"type": "array", "items": {"type": "object", "required": ["path", "why"],
      "properties": {"path": {"type": "string"},
        "why": {"enum": ["secret_path", "secret_content", "history_secret_path", "history_secret_content"]},
        "rule": {"type": "string", "description": "denylist rule id or credential class; never the value"}}}},
    "left_behind": {"type": "array", "items": {"type": "object", "required": ["path", "kind"],
      "properties": {"path": {"type": "string"},
        "kind": {"enum": ["dependency_tree", "build_output", "ignored", "loose_file"]},
        "recipe": {"$ref": "#/$defs/recipe"}}}},
    "remapped": {"type": "array", "items": {"type": "object", "required": ["from", "to"],
      "properties": {"from": {"type": "string"}, "to": {"type": "string"}}}},
    "source_marked": {"type": "array", "items": {"type": "object", "required": ["path", "at"],
      "properties": {"path": {"type": "string"}, "at": {"type": "number"}}}}
  },
  "$defs": {"recipe": {"type": "object", "required": ["kind", "run", "cwd"],
    "properties": {"kind": {"enum": ["pnpm", "pnpm_share_primary", "npm", "yarn", "uv", "uv_pip", "none"]},
      "run": {"type": "string"}, "cwd": {"type": "string"}}}}
}
```

All human sentences are rendered by the surface from codes and numbers (§12); the
core block carries no English beyond paths and commands.

### 3.5 The scope digest (what "confirmed" means)

`scope_digest = sha256(canonical_json({version, to_device, worktrees: sorted
[(path, repo, branch, state, reason)], excluded: sorted [(path, why)],
...artifact_binding}))`, where `artifact_binding` is EMPTY while the move carries no
files and `{carries_files: true, bytes_bucket}` (with `bytes_bucket =
ceil(log2(max(bytes,1)))`) once it does. Mention-only worktrees (§3.2, listed as
`mentioned`) are not inputs.

**No size input while nothing travels** (PR #4241 review R2): with a bucket in the
PR A digest, an active session that crossed 512/1024/2048 bytes while it waited at
a queued or `--wait` safe point was refused `working_set_changed` at resume. The
size describes nothing a "files stay" consent covers, and the gate checks
`over_cap` against the current manifest itself.

**Not an authenticator** (security S9): the digest binds the user's intent to one
scope and one destination and makes a stale consent fail; it proves nothing about
who sends it. Any `move`-capable destination can read it from `workset_preflight`
for itself. Authorisation stays the `move` capability (§8.2).

**Why not a digest of the full manifest:** a session queued with `--queue` keeps
editing until its turn ends, so a byte-exact digest would refuse every queued move.
What the user is confirming is *which trees go to which device and what is
withheld*; that is the scope. A new worktree, a newly excluded secret, a different
destination, or (once files travel) the payload more than doubling changes the scope and the move is
refused `working_set_changed` with the fresh manifest — never silently widened.

## 4. Transfer protocol, inserted into §6.3

Feature negotiation follows the existing additive-key pattern (`"queue"` and
`"resume_queued"` on the prepare frame, `mobility.py:1694`–`1720`: an older owner
ignores unknown keys). The destination sends
`working_set: {version: 1, confirm: <scope_digest> | null}` on `prepare`; an owner
that answers without a `working_set` key is pre-feature (§4.3).

Steps reference `docs/design/mesh-session-mobility.md` §6.3 numbering.

**Phase 0½ — preflight (new, read-only).** Requester (or the local surface)
calls `workset_preflight`; the owner returns the manifest with
`state:"preflight"`. Nothing is written anywhere. The surface shows it and
obtains the confirmation (§11). Empty manifest → no confirmation needed; the move
proceeds exactly as today.

**Phase 1 — prepare**, owner (`mobility._source_prepare`, `mobility.py:3919`):

- P1 (after step 2's validation, before the retire): if `working_set.confirm` is
  absent/null and the *preflight* manifest is non-empty → refuse
  `working_set_unconfirmed`, `changed:false`, `working_set` = the manifest. Cheap,
  pre-retire, nothing mutated.
- Steps 3–4 unchanged (retire; waits for `is_busy` false — §8.5).
- P3 (after the retire is confirmed, before the journal write at ~`mobility.py:4340`):
  recompute the manifest on the quiescent tree; scope digest ≠ `confirm` →
  refuse `working_set_changed` with the fresh manifest. The retire is released
  exactly as the other pre-journal refusals do (`mobility.py:4315`, "NOTHING HAS
  BEEN JOURNALED AND NOTHING DELETED").
- P4 **snapshot** each `planned` worktree (§5.1). Writes only new git objects and
  private refs `refs/lop/move/<sid>/<n>/{head,index,worktree}` in the source repo;
  never touches HEAD, the real index or the working files. Idempotent: refs are
  rewritten to the same commits for the same tree.
- Step 5 journal gains `working_set: {scope_digest, repos:[…]}`.
- Step 6 reply gains `working_set: {manifest, repos: [{key, remote_url, tips:
  {head, upstream, base}, pack_bytes}]}` (`pack_bytes` = source `size-pack`, used by
  the destination's free-space check if it must clone).

**Phase 2 — copy**, destination (`mobility._destination_move`, `mobility.py:1559`):

- Steps 7–9 unchanged (session copy into `network/staging/<sid>/`,
  `mobility.py:1790`–`1839`).
- **W1 — have.** For each repo key: resolve the local primary (§7.2). If present,
  `git fetch --no-tags <remote>` (bounded, 120 s, destination's own credentials,
  read-only) when the URL matches; then `have = [sha for sha in tips if
  git cat-file -e sha^{commit}]`. If absent and origin is reachable, `git clone`
  to the remapped primary path after a free-space check (`pack_bytes × 2` +
  1 GiB); else `have = []`.
- **W2 — plan.** `D → O: net_session_move {phase:"workset_plan", session_id,
  lease_epoch, have:{key:[shas]}}`. Owner verifies the lease epoch and that the
  caller is the journal's `to_device`, then per repo builds
  `git bundle create <tmp> refs/lop/move/<sid>/<n>/* --not <have…>` (no `have` →
  full bundle), runs the history export check (§8.1), and stages each artifact
  **immutable and digest-named** at `<config>/network/workset-out/<sid>/<sha256>`
  (0600, same-dir temp + `os.replace`). Reply: `[{key, digest, bytes}]`, plus any
  worktree that fell to `left_behind` (`over_cap`, `history_secret`).
  Re-running W2 with the same `have` returns the same digests (git bundles of the
  same refs/prereqs are deterministic for a given pack state; if not byte-equal,
  the older staged artifact is reused when its ref list matches — the staged
  directory is the cache).
- **W3 — audit-before-bytes.** Owner writes one `session.workset.serve` record per
  artifact via `AuditLog.record_required` (PR #4228) **before** the first chunk.
  If the durable write fails, the fetch is refused (`audit_unavailable`).
- **W4 — fetch.** `D → O: net_session_move {phase:"workset_fetch", session_id,
  lease_epoch, digest, offset}` → `{data_b64, eof}` chunks of
  `sync.SYNC_CHUNK_BYTES` (512 KiB, `sync.py:92`), served only from
  `workset-out/<sid>/<digest>` (allow-list = that directory's names; a digest not
  staged → `unknown_item`). Destination writes into
  `network/staging/<sid>.workset/<digest>.part`, resumes from its own `.part`
  length on retry, then sha256-verifies and renames. The chunk framing, staged
  write and verify are the sync layer's primitives; the coder exposes them through
  a small public helper in `sync.py` (`serve_span(path, offset, limit)` /
  `receive_span(...)`) reviewed by the mesh lane rather than calling private
  `_served_span`/`_open_stage` (`sync.py:2025`, `:3215`).
- **W5 — verify.** `git bundle verify <file>` against the destination's primary
  (prerequisites present). Failure → re-plan once with `have=[]`; second failure
  → roll back the whole move (step 9's failure path: `_roll_back_destination`,
  `mobility.py:2810`, extended to sweep `<sid>.workset/`).
- Step 10–11 unchanged, except `ready` carries `workset_digest` =
  `sha256(sorted artifact digests)`; the owner compares it with its own at step 12
  just as it compares `content_digest`.

**Phase 3 — commit**, owner: steps 12–15 unchanged. Step 14 additionally writes
the departure record `<config>/network/workset-departed/<sid>.json` (paths, snapshot
refs, `to_device`, `at`) and a marker `<git-dir of worktree>/lop-moved.json`
(`{"to_device","name","at","session_id"}`). **No worktree file is deleted**; the
`refs/lop/move/<sid>/*` refs are kept as the restore point. `workset-out/<sid>/` is
removed after `move.done` (or by `sweep_staging`, `mobility.py:5741`, by age).

**Phase 4 — promote**, destination: steps 16–17 unchanged; at the promote,
`staging/<sid>.workset/` is renamed to `network/workset-in/<sid>/` with
`pending.json`. Then, **after `move.done` (step 18) and before the optional engage**
(`mobility.py:2214`), `workset.restore(...)` (§6) runs per worktree. A failure
leaves the verified artifacts and `pending.json` in place; the receipt reports
`state:"pending_restore"` with the per-worktree reason; `reconcile`
(`mobility.py:5454`) retries on relay start, and `lop sessions move --resume-working-set
<sid>` retries on demand. Each restored worktree is recorded in `pending.json`, so a
retry skips finished ones (idempotent).

### 4.1 Failure table (additions to §6.5)

| Fails at | State left | Recovery |
|---|---|---|
| Preflight / P1 / P3 | nothing (P3: retire released) | fix and retry; `changed:false` |
| P4 snapshot | private refs + objects in source repo | harmless; overwritten on retry; GC'd when refs removed by cleanup |
| W1 fetch/clone | possibly a fresh clone on D | kept (a clean clone is not damage); retry reuses it |
| W2–W5 | `workset-out/<sid>/` on O, `.part` files on D | resumable; or move rolled back via existing path; staging swept |
| commit | as today | as today; working set has deleted nothing |
| restore | session promoted, artifacts in `workset-in/` | `pending_restore`; reconcile / `--resume-working-set` |

Ordering rationale: restoring *before* commit would leave worktree debris on a
destination whose move then rolls back; restoring *after* is safe because the
source never deletes the worktree, so a failed restore loses nothing.

### 4.2 Why bundles + snapshot commits (and not tar)

Untracked files ride inside the worktree snapshot tree as blobs, so one bundle per
repo carries commits, index and worktree with git's own integrity and delta
compression; prerequisites make the "only what the destination lacks"
negotiation free (`--not <have>`). A tar would lose the index/worktree split and
re-send history.

### 4.3 Peers without the feature

- Old owner (no `working_set` in the prepare reply) while the user confirmed a
  non-empty manifest: impossible — preflight against an old owner answers
  `unknown phase`, so the surface shows `state:"unsupported"` ("<device> must be
  updated to carry working files") and the move needs `--without-working-set` to
  proceed (explicitly leaving the work behind).
- Old destination with a new owner: it sends no `working_set` key → the owner applies
  P1 (unconfirmed, non-empty → refuse). Old requesters therefore cannot silently
  move work away; the refusal sentence names the update.

## 5. Snapshot and restore — exact git

All git runs with `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR`
stripped (pattern: `harness/wip_checkpoint.py:94` `_environment`), a dedicated
identity (`GIT_AUTHOR_NAME/EMAIL=lop-move`), `GIT_TERMINAL_PROMPT=0`, and never a
credential helper on the bundle/restore path (`-c credential.helper=`).

### 5.1 Source snapshot (per worktree W, after the retire)

```sh
H=$(git -C W rev-parse HEAD)
IDX_TREE=$(git -C W write-tree)                      # the REAL index, read-only; fails on unmerged → left_behind
tmp=$(mktemp); cp "$(git -C W rev-parse --git-path index)" "$tmp"
GIT_INDEX_FILE=$tmp git -C W add -A -- . $EXCLUDES  # EXCLUDES = ':(exclude,literal)<secret path>' …
WT_TREE=$(GIT_INDEX_FILE=$tmp git -C W write-tree); rm -f "$tmp"
IDX_C=$(git -C W commit-tree "$IDX_TREE" -p "$H" -m "lop move: index of <sid>")
WT_C=$(git -C W commit-tree "$WT_TREE" -p "$IDX_C" -m "lop move: worktree of <sid>")
git -C W update-ref refs/lop/move/<sid>/<n>/head "$H"
git -C W update-ref refs/lop/move/<sid>/<n>/index "$IDX_C"
git -C W update-ref refs/lop/move/<sid>/<n>/worktree "$WT_C"
```

Excluded secret files that are **tracked and modified**: the exclusion keeps their
index entry as in the real index (the temp index started as a copy), so the
destination gets the *index* version; the receipt says "your local edits to
<path> stay on <source>". Untracked secret files are simply absent.

### 5.2 Destination restore (per worktree, primary P, target T)

```sh
git -C P fetch <bundle> 'refs/lop/move/<sid>/*:refs/lop/move/<sid>/*'
H=$(git -C P rev-parse refs/lop/move/<sid>/<n>/head)
# branch: create, or fast-forward only; never move a branch checked out elsewhere
git -C P show-ref --verify --quiet refs/heads/<branch> \
  || git -C P branch --no-track <branch> "$H"
#   exists and == H            → use it
#   exists, ancestor of H, not checked out in any worktree → git -C P update-ref refs/heads/<branch> H <old>
#   otherwise                  → branch := <branch>-moved-<srcdevice>, created at H (reported)
git -C P worktree add --no-checkout T <branch>        # registers T: `git worktree list` shows it
git -C T read-tree refs/lop/move/<sid>/<n>/worktree   # index := worktree snapshot
git -C T checkout-index -a -f                          # files := worktree snapshot (incl. untracked, x-bit, symlinks)
git -C T read-tree refs/lop/move/<sid>/<n>/index      # index := staged snapshot
git -C T update-index -q --refresh
# upstream, even before the remote-tracking ref exists locally:
git -C P config branch.<branch>.remote <remote>
git -C P config branch.<branch>.merge refs/heads/<upstream-branch>
```

Result: `HEAD`→index diff = what was staged; index→files diff = what was unstaged
(including unstaged deletions, which are simply not checked out); files present
in the worktree tree but absent from the index tree are untracked. Detached HEAD:
`git worktree add --no-checkout --detach T $H`.

The private refs stay on the destination too (they are the recall baseline, §10).

## 6. Dependency trees, transfer and recipes

Covered in §9 (left behind + recreation recipes); numbering below is kept stable
for the cross-references in §13–§14.

## 7. Path remap and conflicts

### 7.1 Rule

The owner sends every path **home-relative** (`~/…`), computed on the owner where
its own home is known exactly; the destination prefixes its own home. Paths not
under the owner's home → `left_behind`, reason `outside_home`.

Deliberately **not** `carry.translate_host_path` (`carry.py:223`): that function
keeps any path that already exists and otherwise guesses by trailing component —
right for a cwd that only needs *a* directory, wrong for restoring a worktree,
where "already exists" is the conflict case and a guess could land work in the
wrong repo.

### 7.2 Primary repository on the destination

1. Remapped primary path exists, is a git repo, and some remote's URL equals the
   source's `remote_url` (userinfo-stripped, `.git`-suffix-insensitive) → use it.
2. Exists but a different repo → `left_behind`, `destination_conflict` (never
   clone over or beside an unrelated checkout).
3. Absent → clone from `remote_url` (W1) if reachable and space allows; else the
   full bundle (subject to the cap, §8.4 — in practice over cap for these repos)
   → otherwise `left_behind`, `over_cap`, with the remedy "clone <url> at <path> on
   <device>, then retry".
4. When the moving worktree **is** the source's primary checkout: case 1 →
   restore as a new worktree `<primary>/.worktrees/moved-<sid8>` (the destination's
   own primary is never rewritten); case 3 → the clone becomes it.

### 7.3 Target path already exists

- Registered worktree of the same primary whose HEAD, index tree and worktree tree
  equal the snapshot → already restored (idempotent retry); mark restored.
- Carries a `lop-moved.json` marker from an earlier departure (the recall case)
  and its current state equals the departure snapshot (`refs/lop/move/<sid>/…` on
  that device) → nobody touched it since; update in place: `git -C T read-tree
  -m -u` to the new worktree tree, then the index step. Fast and lossless.
- Anything else → restore at `<path>.moved-<srcdevice>-<YYYYmmddHHMM>`, reported in
  `remapped`. **Never overwrite, never refuse** (no manual steps; nothing lost).

### 7.4 Continuation note

**PR A (landed): the active-move resume.** A move of an ACTIVE conversation — a
queued move resuming at its safe point, or a transcript row (`ts`) within
`DEFAULT_KEEP_ALIVE_SECONDS` (300 s, `session/runtime/process.py`, the runtime's own
"a person is involved" residency) in the session or any subagent of its tree — is
engaged on arrival by default (`engagement.reason: "active"`) and handed ONE waking
inbox row (`sender: "session move"`) before the engage, so the runtime's boot drain
delivers it as a turn. The model-facing text (English by design) lists each
worktree, that it stayed on `<source>` and how to get it (push from the source and
fetch here, or `lop sessions move <id> --to <source>`), left-behind trees with their
recipes, withheld files, and toolchains this device lacks. Exactly once per move: a
`move-arrival-note.json` marker keyed by lease epoch, plus a deterministic
`peer-<hex>` message id the receiving session's duplicate gate refuses twice.
`keep` copies never resume; `--no-engage-on-arrival` (frame
`no_engage_on_arrival`) keeps the cold arrival. The receipt line is keyed
(`cli.sessions_move_workset.receipt_resumed`). On an OFFLOAD the receipt is the
source's and the destination engages after `done`, so the outcome never comes back:
`engagement` stays absent (unknown) and the receipt carries
`resume_requested: {reason: "active" | "asked", session_id}` instead — what the source
asked for, from its own prepare-time reading (UI #1978 QA Q-1); the CLI prints its
existing "was asked to start it" line. PR B replaces "stayed on" with the
restored/remapped paths below.

After restore the destination appends one inbox line
(`session/runtime/inbox.py:755` `append_inbox`, non-waking) listing remapped paths
and any `pending_restore`/`left_behind` items, so the next turn's model sees the
new paths once. Also rewrite the session cwd via the existing
`carry.rewrite_host_cwd` (`carry.py:346`) when the cwd was inside a restored
worktree whose `dest_path` differs. (Coder: confirm which inbox source renders as a
system note; if none fits, add `SOURCE_MOVE` beside `SOURCE_RECALL`,
`inbox.py:154`.)

## 8. Safety

### 8.1 Secrets

- **Path classifier:** `supplements/denylist.is_sensitive(path, cwd=)`
  (`supplements/denylist.py:267`) — the existing one, already a superset of the
  `@`-reference and upload gates, NFKC/case-folded, checks written and resolved
  forms, used by `network/files.py:195` for peer file reads. Every dirty/untracked
  path in the snapshot passes through it; a hit is excluded (`secret_path`, rule id).
- **Content classifier:** `team_bridge/secret_guard.credential_classes(text)`
  (`team_bridge/secret_guard.py:168`) over the harness's one credential-shape table
  (`redaction_shapes.py`). Run on dirty/untracked text files ≤ 1 MiB; a hit is
  excluded (`secret_content`, class name only — never the value).
- **History is not path-filterable** (mesh lane's caveat, adopted): a bundle carries
  every blob of the commits it contains. Export-policy check at W2, over exactly
  the bundled range (`<tips> --not <have>`):
  `git log --format= --name-only <range>` → `is_sensitive` on each path, and
  `git log -p --format= <range>` added lines → `credential_classes`. Any hit →
  that worktree is `left_behind`, `history_secret`, with the paths listed. **There
  is no opt-in in v1** (security ruling, 2026-11-18): `--include-history-secrets` /
  `include_history_secrets` were dropped; push or rewrite the commits instead. The
  patch is read raw — `-c core.attributesFile=/dev/null log -p --text
  --no-textconv --no-ext-diff --diff-merges=first-parent` — so `.gitattributes`
  (`-diff`, `binary`, a textconv driver) cannot hide content from the scan and an
  evil merge's content is seen (PR #4241 security S4). Measured cost is negligible: the largest delta bundle on this
  node is 16 KB (§8.4). A full-bundle fallback (no `have`) scans the whole history,
  which is one more reason the full fallback is effectively clone-only.

### 8.2 Authorisation

No new capability: the working set is part of the session the `move` grant
already covers. Every `workset_*` phase requires (a) the caller passes the same
membership/epoch/`move` check as `prepare` (authoriser; `mobility.py:1108`
`_EXCLUSIVE_MOVE_CAPABILITY`), (b) `lease_epoch` equals the journal's, (c) caller
device == journal `to_device`. `workset_preflight` requires (a) only. The
confirmation gate (§3.5) is the *user* consent layer on top.

### 8.3 Audit

PR #4228 owner's guidance: S1's file-read event/authority does **not** cover
bundle egress; this design adds **its own event types**, built on
`AuditLog.record_required` (#4228; `AuditLog` is `network/audit.py:1027` on main,
whose `record` at `:1198` is not durable-before-bytes). Dependency: **PR B stacks
on #4228** (or merges main after it lands). PR A moves no bytes, but its
destination probe makes the DESTINATION act on the network at a peer's request, so
it is audited too (PR #4241 security S1/S9): `session.workset.probe`, one row per
served probe with host names and fixed result codes only.

| Event | Where | Fields (bounded; no payloads, no secret values) |
|---|---|---|
| `session.workset.preflight` | owner | session_id, requester, worktree count, bytes, excluded count |
| `session.workset.probe` | destination (PR A) | session_id, requester, host names, result-code counts, or `refused` (rate limit / probe error) |
| `session.workset.serve` | owner, before first chunk of each artifact (required) | session_id, to_device, lease_epoch, artifact digest, bytes, repo key, home-relative worktree paths |
| `session.workset.restore` | destination | session_id, from_device, per-worktree state + reason codes |
| `session.workset.departed` | owner at commit | session_id, to_device, marked paths |

### 8.4 Size cap — measured

Measured 2026-11-18 on this node (Linux, `~/local-operator` + `~/local-operator-ui`,
all registered worktrees; script and raw rows in the architect's scratchpad
`wsmeasure/`, read-only except bundles streamed to `wc -c`):

| Metric | n | median | p90 | max |
|---|---|---|---|---|
| Delta bundle (`HEAD --not --remotes=origin`) | 60 | 0 | 0 | **16 KB** (1 worktree non-empty) |
| Dirty bytes (staged+unstaged+untracked, full file sizes) | 60 | 0 | 0 | **43.1 MB** (714 files; 5 worktrees non-empty, their median 10.1 MB) |
| Delta total | 60 | 0 | 0 | **43.1 MB** |
| `origin/main..HEAD` bundle (naïve base) | 60 | — | 31 KB | 365.7 MB (primary checkout on an unrelated-history branch) |
| Full pack (`size-pack`) | — | — | — | local-operator 742 MiB, local-operator-ui 3.13 GiB |

Readings: the delta against *remote-tracking refs* is tiny (most work here is
pushed); the naïve `origin/main..HEAD` base is a trap (366 MB for a fully pushed
branch) — hence the `have` negotiation; a full bundle is GB-scale, hence clone on
the destination instead.

**Cap: 256 MiB per move** for the working-set payload (bundles + snapshot
blobs, measured before staging) — ~6× the observed max, and equal to the session
scratchpad cap already in force. Config key `mesh.move.working_set_max_bytes`
(AGENTS.md "Adding a configuration key"). Over the cap → refuse the move
`working_set_over_cap` listing per-worktree sizes and the remedy (`--leave <path>`
to leave a named worktree behind, or raise the key). Destination free space is
also checked (artifact bytes × 3 for bundle + objects + checkout).

**Gap:** the trigger is a *Mac* node; this distribution is from the Linux node. The
coder or QA should run the same script (`git bundle create - … | wc -c`, never into
the scratchpad — a first attempt hit the 256 MiB scratchpad cap) on the Mac before
release; if its max exceeds ~128 MiB, raise the default.

### 8.5 Busy session / live subagents

Already enforced, and the design relies on it rather than adding a second check:
the runtime's `is_busy` (`session/runtime/serving.py:2458`) is true while
`session.running_subagents() > 0`, background jobs, a turn, a parked gate or a goal
loop are live; the direct move refuses busy at retire (§6.4) and `--queue`'s driver
waits on the same predicate before latching (`session/runtime/server.py:8760`–`8770`,
`network/move_queue.py:574` `drive`). Subagents are in-process children, so the
parent's busy covers the tree. The snapshot (P4) is taken **after** the retire, so
it cannot capture a half-written file from this tree. What is *not* covered —
another session editing the same worktree — is handled by the contention rule
(§3.2). The base-red test pins the "queued move waits for a running subagent" case
so a future change to `is_busy` cannot silently drop it.

### 8.6 Mark, never delete

Source keeps every worktree untouched; `lop-moved.json` + the departure record let
the TUI/CLI say "moved to <peer> at <ts>" if someone opens it, and back the
recall fast-path (§7.3). Cleanup is a separate, operator-approved action (phase 2:
`lop sessions move-cleanup <sid>` listing what it would remove, `--yes` required).

## 9. Left behind: dependency recreation recipes

Detected per restored worktree by lockfile; PR B puts them in `recipes` (not run).

| Lockfile | Condition | `run` | Note |
|---|---|---|---|
| `pnpm-lock.yaml` | worktree is `<primary>/.worktrees/<name>`, its lockfile byte-identical to the primary's, and primary has `node_modules` | `rm -rf node_modules && ln -s ../../node_modules node_modules` (in the worktree) | the dependency-commons rule; removes only the worktree's own |
| `pnpm-lock.yaml` | otherwise | `pnpm install --frozen-lockfile` | pnpm's clone-first import shares the store |
| `package-lock.json` | — | `npm ci` | never in a pnpm repo |
| `yarn.lock` | `.yarnrc.yml` present | `yarn install --immutable` | else `yarn install --frozen-lockfile` |
| `uv.lock` | — | `uv venv && uv sync --frozen` | venv recreated, never symlinked |
| `requirements*.txt` only | — | `uv venv && uv pip install -r <file>` | |
| repo `AGENTS.md` names a setup command | — | listed as-is under `note` | e.g. local-operator's `uv pip install -e ".[all,dev]"` |
| `Cargo.lock`, `go.sum` | — | none | toolchain caches rebuild |

Phase 2: `--setup-deps` runs the listed recipe on the destination after restore
(package lifecycle scripts execute, which is why it is opt-in, not default).

## 10. Recall (`--to local`)

Symmetric by construction: the remote device is the owner, computes the manifest
and serves artifacts; this device pulls and restores. The common recall case hits
§7.3's marker fast-path: the original worktree is still here, marked, and
unchanged since departure, so it is updated in place to the remote state. If it
was edited locally in the meantime, the remote state lands beside it
(`.moved-<device>-<ts>`) and both are listed — never a merge, never an overwrite.
The preflight for recall is the `workset_preflight` phase over the link.

## 11. Confirmation surfaces

The operator's requirement: the remote move must require explicit approval in the
popup, warning that moving mid-conversation transfers all working files to the
destination device, and showing enough that the user is confident the
destination should receive them.

### 11.1 Core contract (PR A)

- **Preflight**:
  - peer/relay: `net_session_move {phase:"workset_preflight"}` (§3.1);
  - CLI: `lop sessions move <id> --to <peer> --dry-run [--json]` → prints/returns the
    manifest, exit 0, nothing moved;
  - desktop route: `POST /v1/desktop/sessions/{id}/transfer/preflight` body
    `{to}` → `CRUDResponse[TransferPreflight]` (= the `working_set` block +
    destination name), beside `transfer_session` (`server/routes/desktop_mesh.py:214`).
- **Move call carries consent**: `confirm_working_set: <scope_digest>` on
  `TransferSession` (`server/models/desktop_mesh.py:167`), `--confirm-working-set
  <digest>` on the CLI, `working_set.confirm` on the prepare frame.
  `--without-working-set` / `without_working_set: true` is the explicit "leave it
  behind" consent (also needed against a pre-feature owner).
- **Refusals**: `working_set_unconfirmed`, `working_set_changed`,
  `working_set_over_cap`, `working_set_unsupported` — all `changed:false`, all
  carrying the fresh manifest in `working_set`. Added to `MOVE_REFUSAL_CODES`.
- **Capability flag**: `"session_transfer_working_set": 1` in
  `server/features.py` beside `"session_transfer": 1` (`features.py:981`). Its own
  key, per that file's rule: an existing surface (the old dialog) must keep working
  byte for byte — and it will, as an honest refusal naming the update, for sessions
  with a non-empty manifest (§16 rollout risk).

### 11.1a Destination probe (PR A, live-move finding 2026-11-18 17:12Z)

A Mac → node2 move showed `clone_check: "pending_destination"` on every row; node2
could neither fetch the SSH remote nor push the HTTPS one. The preflight now asks the
DESTINATION before the confirmation (`network/workset_probe.py`):

- **Op**: `net_session_move {phase: "workset_probe", probe_version: 2, worktrees:
  [{remote_url, markers}], team_bridge}` served by the destination (authorizer
  carve-out beside `invite`: it reads nothing of the session). **What the preflight
  discloses** (PR #4241 security S6): the remote URLs and toolchain marker names,
  one row per distinct remote — no worktree path, branch or head — and it does so
  during `--dry-run` too, i.e. before the user has decided to trust the
  destination. Mention-only worktrees are never sent.
- **Admission and answers** (security S1/S3): the destination refuses a probe for a
  session it holds itself, rate-limits per asking device (10 / 60 s), writes a
  `session.workset.probe` audit row, refuses hosts that resolve to a non-public
  address unless its operator names them in `LOCAL_OPERATOR_MOVE_PROBE_HOSTS`, pins
  HTTPS to the vetted address, and runs git with `http.followRedirects=false`,
  `core.hooksPath=/dev/null`, a protocol allowlist and (absent the user's own ssh
  command) `BatchMode=yes`, `ForwardAgent=no`, `StrictHostKeyChecking=yes`. Its
  answer carries only a `clone_check`, a fixed detail code, booleans and known tool
  names; the asker rebuilds the HTTPS URL and the remedy command itself. Offload: the owner asks over the
  link; recall: this device is the destination and probes itself. An older
  destination answers `unknown move phase` → every row `unknown_destination` and
  `destination_probe.state: "unknown_destination"` — "could not check", never a
  refusal. Feature key `session_transfer_working_set_probe`.
- **Checks**, with the destination's OWN credentials (its env, SSH agent, helpers,
  plus the mesh-lent `github` helper via `borrowed_git_env`): `git ls-remote <url>
  HEAD` (fetch) and `git push --dry-run <url> <empty-commit>:refs/heads/lop-move-probe-<hex>`
  from a throwaway bare repo (push). Chosen over the GitHub permissions API: host
  agnostic, and it exercises the exact transport/credential the restore will use;
  write permission is enforced at receive-pack's advertisement and `--dry-run` sends
  no pack or ref update, so nothing is written remotely.
- **URL policy**: allowlist `https://`, `ssh://`, scp `user@host:path`; refused
  without running git: `file://`, local paths, `<helper>::`/`ext::`, option-shaped
  or whitespace/control-character URLs (`destination_cannot_clone`, `url_refused`).
  `protocol.ext.allow=never` as a belt; no prompts (`GIT_TERMINAL_PROMPT=0`, stdin
  `/dev/null`, new session so ssh has no tty), 20 s per git call, 4 lanes.
- **Failure flags, never re-states** (decision): the row stays `planned` with the
  `clone_check` code, `clone_detail`, and a `remedy` when lop has one —
  `lop network credential github include <owner/repo>` when the device already
  borrows a `github` share, else `lop network credential share github --with <device id>` (the id, never the
  peer-chosen name — security S10),
  both run on the LENDING device. Not a digest input: the gate recomputes the
  manifest without re-probing, so a probe-derived state would refuse every confirmed
  move `working_set_changed`; and PR B's bundle still carries commits a
  cannot-fetch destination lacks. PR B's plan step decides what to leave behind.
- **`https_rewrite`**: when the SSH form cannot be fetched but its HTTPS spelling
  can, `{url, fetch: true, push}` is OFFERED; PR B applies it at restore.
- **Prerequisites, warn-only**: `missing_tools` per row and in
  `destination_probe.missing_tools`, from root markers (`go.mod`→go,
  `pnpm-lock.yaml`→pnpm+node, `package-lock.json`→npm+node, `uv.lock`/`pyproject.toml`→uv,
  `Cargo.toml`→cargo) and `shutil.which` on the destination. Team Bridge: lop has no
  per-session "uses the bridge" flag, so the owner sets `team_bridge_used` only when
  the session is the designated chief-of-staff session or its transcript holds a
  `bridge` tool row or a `team_bridge_message` card; only then does the destination
  report its enrolment (`team_bridge.enabled` + a committed enrolment).

### 11.2 CLI (PR A)

`cli.py` parser `move_parser` (`cli.py:1754`–`1814`), command
`sessions_move_command` (`cli.py:7631`, calls `mobility.request_move` at ~`:7686`):

- interactive TTY, non-empty manifest: print the manifest summary + warning and
  prompt `Send these working files to <device>? [y/N]`; yes → re-issue with the
  digest;
- `--yes` / `--confirm-working-set <digest>` for non-interactive;
- `--json` without confirmation → refusal document with the manifest (exit 1,
  `changed:false`); a script confirms by passing back `working_set.scope_digest`.

Preflight text shape (rendered from keys, §12):

```
Moving 9f3ac1e0b7d2 to lop-remote-node-2 mid-conversation sends its working files there:
  brings 2 worktrees · 1 unpushed commit · 5 uncommitted files · 3.2 MB
    ~/local-operator/.worktrees/bridge-emoji-shortcodes  feat/bridge-emoji-shortcodes (1 unpushed, 2 staged, 2 unstaged, 1 untracked)
    ~/local-operator-ui/.worktrees/x                       feat/x
  withheld (secrets): ~/local-operator/.worktrees/x/.env.local
  leaves behind: node_modules (recreate: pnpm install --frozen-lockfile)
Only continue if lop-remote-node-2 should receive these files.
```

### 11.3 TUI (PR A)

`/move … --to` (`tui/app.py:37417`) → `tui/session_move.run_session_move`
(`tui/session_move.py:164`, which shells to the CLI with `--json`). Add: run the
`--dry-run --json` preflight first; non-empty manifest → push a new
`MoveWorkingSetConfirmScreen` (ModalScreen in `tui/session_move.py`) showing the
summary, the destination name and the warning, with Confirm disabled until the user
ticks "I'm confident <device> should receive these files"; on confirm, run the move
with `--confirm-working-set <digest>`. Visual proof per AGENTS.md "Visual
validation" (SVG before/after).

### 11.4 Desktop (PR C, local-operator-ui)

The dialog is `MoveConfirmDialog` (`src/renderer/src/features/mesh/mesh-actions.tsx:68`),
used by `features/chat/device/chat-header-device.tsx:557` and
`features/mesh/mesh-page.tsx:754`; the transfer request is built in
`src/shared/desktop-contract.ts:7408` and hooked in
`src/renderer/src/shared/api/local-operator/desktop-hooks.ts:819`.

| File | Change |
|---|---|
| `src/shared/desktop-contract.ts` | `transferPreflight` request (`POST …/transfer/preflight`); `confirm_working_set`, `without_working_set`, `include_history_secrets` on the transfer body; `WorkingSet` response type |
| `src/renderer/src/shared/api/local-operator/desktop-hooks.ts` | `useTransferPreflight(sessionId, to)` (fires when the dialog opens, re-fires on destination change) |
| `src/renderer/src/features/mesh/mesh-types.ts` | `workingSet(value)` parser (same defensive `text`/`count` helpers as the file's other parsers) |
| `src/renderer/src/features/mesh/mesh-actions.tsx` | `MoveConfirmDialog`: loading / empty (no working files — dialog as today) / error / populated states; populated shows the warning, the destination device name, worktrees, unpushed/uncommitted counts, size, withheld secrets, left-behind list, and a required checkbox; Confirm disabled until ticked; sends the digest; `working_set_changed` refusal re-renders the fresh manifest and un-ticks |
| `chat-header-device.tsx`, `mesh-page.tsx` | pass `sessionId`/`to` so the dialog can preflight |
| `src/i18n/…` | `ui.mesh.mesh_actions.working_set_*` keys in every locale |

Gate: `desktopFeatureEnabled(capabilities, "session_transfer_working_set")`;
without it the dialog is unchanged (and an old backend has no gate to satisfy).
Design + UX review rounds apply (user-visible flow); screenshots of the four states.

## 12. i18n

Per AGENTS.md "Internationalization is mandatory": every key below translated into
all of `i18n/languages.json` — en, fr, es, zh-CN, ru, vi, ur, hi — with the
independent audit before the ledger row ships; no allowance growth. Placeholders
preserved; plural arms for counts.

Core namespaces (catalogues under `local_operator/i18n/catalogues/<locale>/`):

- `cli.sessions_move_workset` — `warning_mid_conversation`, `brings_summary`
  (plural: worktrees, commits, files; `{bytes}`), `worktree_line`, `withheld_secrets`,
  `withheld_history_secrets`, `leaves_behind`, `recreate_with`, `confirm_prompt`,
  `confirm_aborted`, `left_behind_reason.<code>` (one per reason code in §3.4),
  `refusal.unconfirmed`, `refusal.changed`, `refusal.over_cap`,
  `refusal.unsupported`, `pending_restore`, `remapped`, `source_marked`,
  `arrival_note` (the model-facing inbox note is model-facing → English by design,
  allowlisted with an inline `# i18n: ignore model-facing`).
- `tui.session_move` (existing namespace, `tui.session_move.json`) — `ws_title`,
  `ws_confirm_checkbox`, `ws_confirm`, `ws_cancel`, reusing the CLI summary keys.
- UI (PR C): `ui.mesh.mesh_actions.working_set_{warning,destination,worktrees,
  unpushed,uncommitted,size,withheld,left_behind,checkbox,loading,error,none}`.

## 13. Module and file plan (coder)

**New** `local_operator/network/workset.py` — stdlib + subprocess only (it runs in
the relay, same import-weight reason as `carry.py`; lazy-import the denylist and
secret guard inside functions as `file_provenance.py:91` does):

```python
@dataclass(frozen=True) class WorktreeFacts: ...
@dataclass(frozen=True) class Manifest: ...  # to_dict() → §3.4 block
def session_tree(root: Path, session_id: str) -> list[str]
def candidate_seeds(root: Path, session_ids: list[str]) -> list[Seed]
def discover_worktrees(seeds, transcripts, created_at) -> list[Candidate]
def worktree_facts(path: Path, *, home: Path) -> WorktreeFacts
def classify_exclusions(path: Path, entries) -> list[Excluded]
def compute_manifest(root, session_id, *, to_device, to_name, home, cap) -> Manifest
def scope_digest(manifest: Manifest) -> str
# PR B
def snapshot(worktree: Path, session_id: str, n: int, excludes) -> SnapshotRefs
def history_check(repo: Path, tips, have) -> list[Excluded]
def stage_bundle(repo: Path, refs, have, out_dir: Path) -> Artifact
def resolve_primary(remote_url: str, home_rel: str, *, home: Path) -> PrimaryResolution
def restore(artifact_dir: Path, plan: RestorePlan, *, home: Path) -> RestoreReport
def recipes_for(worktree: Path, primary: Path) -> list[Recipe]
def mark_departed(config: Path, session_id: str, to_device: str, worktrees) -> None
```

**Call sites**:

| File:line (main @ 3a5a6d957) | Change | PR |
|---|---|---|
| `network/mobility.py:492`, `:575` | `working_set: NotRequired[...]` on result/refusal | A |
| `network/mobility.py` `MOVE_REFUSAL_CODES` (near `:353`) | four new codes | A |
| `network/mobility.py:5792` `make_handler` | dispatch `workset_preflight` (A), `workset_plan`, `workset_fetch` (B) | A/B |
| `network/mobility.py:3919` `_source_prepare` | P1 before retire; P3/P4 after retire, before journal (~`:4340`); journal + reply fields | A (P1/P3), B (P4) |
| `network/mobility.py:1694` `_destination_move` prepare frame | `working_set: {version, confirm}` | A |
| `network/mobility.py:1813`–`1939` | W1–W5 between session copy and content digest; `workset_digest` on ready (`:2053`) | B |
| `network/mobility.py:4459` `_source_commit` | compare `workset_digest`; `mark_departed` | B |
| `network/mobility.py:2554` `_promote`, `:2187`–`:2214` | rename `.workset` staging; restore after `done`, before engage | B |
| `network/mobility.py:2810` `_roll_back_destination`, `:5741` `sweep_staging` | sweep `<sid>.workset/`, `workset-out/<sid>/` | B |
| `network/mobility.py:5454` `reconcile` | resume `pending_restore` | B |
| `network/mobility.py:6812` `request_move` | `confirm_working_set`, `without_working_set`, `include_history_secrets`, `dry_run` kwargs | A |
| `network/sync.py` | public `serve_span` / `receive_span` wrappers (mesh lane reviews) | B |
| `cli.py:~1790–1814`, `:7631` | flags + prompt + `--dry-run` rendering | A |
| `tui/session_move.py:164`, `tui/app.py:37417` | preflight + confirm screen | A |
| `server/models/desktop_mesh.py:167`, `server/routes/desktop_mesh.py:214`, `server/features.py:981` | preflight route, body fields, capability | A |
| `local_operator/i18n/catalogues/*/…` | §12 keys | A (B adds restore keys) |

## 14. Test plan

Base-red = the test fails on `3a5a6d957` (the coder shows the red run) and passes
on the branch. All under `tests/unit/network/`, using the in-process two-relay
`devices` fixture from `tests/unit/network/test_mobility.py` (as
`test_mobility_move_outcome.py:39` does), real `git` in `tmp_path` repos with
`GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0` per the rig-hygiene rules.

| File | Asserts | PR |
|---|---|---|
| `test_workset_manifest.py` | cwd-seeded worktree found; transcript-mention-only worktree found; provenance-seeded found; child-session worktree attributed `subagent:<id>`; unrelated worktree of same repo NOT included; counts of unpushed/staged/unstaged/untracked exact; contended worktree `left_behind`; outside-home `left_behind`; unmerged `left_behind` | A |
| `test_workset_confirm_gate.py` | non-empty manifest without confirm → `working_set_unconfirmed`, `changed:false`, source untouched (dir + no journal); scope change between preflight and prepare (new untracked secret / new worktree) → `working_set_changed`; byte-only change in a confirmed worktree passes; empty manifest needs no confirm and the receipt has no `working_set` key (frozen-contract parity) | A |
| `test_workset_secrets.py` | `.env.local`, `id_rsa`, a file containing an AWS-shaped key → excluded with rule/class, never in the snapshot tree (`git ls-tree -r`); tracked-modified `.env` keeps index version; history commit adding `.env` → worktree `left_behind history_secret` unless `include_history_secrets` | A (classify), B (history) |
| `test_workset_roundtrip.py` | snapshot → bundle → restore into a second repo clone: `git rev-parse HEAD`, `git diff --cached --name-status`, `git diff --name-status`, `git ls-files --others --exclude-standard` identical to source; executable bit and symlink preserved; staged deletion and unstaged deletion preserved; branch + upstream config set; `git worktree list` on the destination primary shows the path | B |
| `test_workset_bundle_negotiation.py` | `have` containing upstream tip → bundle prerequisites = have, bytes ≪ full; `have=[]` → full bundle; verify failure → re-plan with `have=[]` once | B |
| `test_workset_remap.py` | `/srcHome/x/.worktrees/w` → `<destHome>/x/.worktrees/w`; existing equal state → idempotent; existing different → `.moved-<dev>-<ts>`; marked-and-unchanged → in-place update; primary-is-worktree → `.worktrees/moved-<sid8>` | B |
| `test_workset_transfer_auth.py` | `workset_fetch` with wrong lease epoch / from a non-`to_device` member / for an unstaged digest → refused; audit `session.workset.serve` written before first chunk; audit write failure → refusal, zero bytes served | B |
| `test_workset_busy.py` | a queued move against a session whose `running_subagents()>0` stays `finishing` and no snapshot refs exist; after the child settles it proceeds and snapshots | A (pin) |
| `test_workset_recall.py` | move A→B, edit+commit on B, `--to local` → A's original worktree updated in place with B's commit and dirty state; A edited meanwhile → lands beside, both listed | B |
| `test_workset_never_deletes.py` | after a completed move the source worktree, its files and branch are byte-identical; `lop-moved.json` present | B |
| `tests/unit/test_cli_sessions_move_workset.py` | `--dry-run --json` shape matches §3.4 schema; `--json` without confirm exits 1 with manifest; `--confirm-working-set` digest round-trips | A |
| `tests/unit/tui/test_session_move_confirm.py` | confirm screen renders summary; Confirm disabled until checkbox | A |

Only these files run locally (test lease); full suite is CI's.

### 14.1 Real two-relay loopback proof (QA, PR B)

Same shape as the #4108/#4228 QA rigs (#4108 QA round 1: production runtime from
the worktree `.venv`, `env -i` with fresh `HOME`/`LOCAL_OPERATOR_CONFIG_DIR`,
`CMUX_*`/`LOP_*`/`XPC_*` stripped, `systemd-run --user --scope -p MemoryMax=1536M
-p MemorySwapMax=0 timeout 600`, loopback stub provider, exact-pgid reap):

1. Two isolated homes `$ISO/a/home`, `$ISO/b/home` (different paths → remap is
   exercised for real), each with its own config root and relay; pair them with
   `lop network` invite/join; grant `move`.
2. In `a`: a bare "origin" repo `$ISO/origin.git`; clone to `a/home/proj`; add
   `.worktrees/feat` on branch `feat` with upstream; 1 unpushed commit; 2 staged,
   2 unstaged (one deletion), 1 untracked, 1 `.env.local`, `node_modules/` (ignored);
   clone the same origin into `b/home/proj`.
3. A synthetic session in `a` with cwd `~/proj/.worktrees/feat`.
4. `lop sessions move <sid> --to b --dry-run --json` → manifest (capture).
   `--json` without confirm → refusal (capture). With `--confirm-working-set`
   → receipt (capture).
5. On `b`: `git -C ~/proj worktree list`, `git -C ~/proj/.worktrees/feat status
   --porcelain=v2`, `git log -1`, `git rev-parse @{u}` (capture), `.env.local`
   absent, audit log rows on `a` (`session.workset.serve`, before-bytes ordering by
   seq), source worktree unchanged + `lop-moved.json` (capture).
6. On `b` edit + commit, `lop sessions move <sid> --to local` from `a` → `a`'s
   worktree updated in place (capture).
7. Negative rows: wrong lease epoch fetch; over-cap (set the key to 1 KiB);
   history secret; destination path occupied by different content.

## 15. Effort and phasing

Classification: **very significant (3–4 h) per PR**, not one 2 h PR. Honest
estimate: PR A ~3 h (manifest, gate, receipt, CLI/TUI, i18n ×8 locales with the
independent audit), PR B ~4 h (snapshot/bundle/restore/recall/audit + the rig),
PR C ~2–3 h (+ design/UX rounds). If only one PR fits, ship **A**: it alone turns
the silent loss into an informed, confirmed choice.

**Phase 2** (not in A/B/C): loose non-git files (provenance-attested write/edit
paths outside repos — carried as digest-named artifacts through the same W2/W4
path); `--setup-deps`; submodules/LFS; `move-cleanup` command; per-file caps.

## 16. Risks to watch during rollout

- **Desktop window of mismatch**: core A released before UI C → the old dialog's
  move of a session with worktrees is *refused* (with the update sentence). That is
  the intended safety, but ship C in the same release window.
- **Discovery false negatives**: a worktree touched only via shell expansions in a
  pre-#4108 session with no literal path in the transcript is missed. Mitigation:
  the preflight lists what was found and why; watch for user reports; consider an
  explicit `--include <path>`.
- **False positives** pulling in a sibling session's worktree via a mere mention.
  Mitigation: contention rule + the confirmation list; watch `sources: ["mention"]`
  frequency.
- **Content secret classifier over-firing** on test fixtures (it is tuned for
  refusal in Team Bridge) → files withheld that the user wanted. They are listed;
  the fix is a `--include-file` override (phase 2) rather than weakening the table.
- **Bundle determinism** across retries (W2 idempotence) — watch for re-staged
  artifacts with new digests mid-fetch; the staged-dir cache is the guard.
- **Disk pressure** on destinations (this node: 94% used on `/`): clone-on-demand of
  a 3 GiB repo; the free-space check must run before the clone, not after.
- **Cap calibration from Linux only** (§8.4 gap).

## 17. Open questions

1. *(Operator)* **Auto-clone on the destination**: when the destination lacks the
   primary repo, should the move clone it from origin automatically (default in
   this design; GB-scale disk and time), or refuse that worktree and ask the user to
   clone first?
2. *(Operator)* **History secrets default**: this design leaves a worktree behind
   when its unpushed commits contain secret-shaped paths/content unless explicitly
   included. Acceptable, or should the move refuse outright?
3. *(Team, not operator)* Which inbox source renders as a one-time system note
   (§7.4) — coder to confirm in `session/runtime/inbox.py`.
4. *(Evidence)* Mac-node size distribution (§8.4) before setting the release default.

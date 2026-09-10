# Design: making a newly authored skill usable in the session that wrote it

Status: proposal (architect). No implementation in this document.

## 1. The problem as I found it

An agent wrote a valid skill to `~/.local-operator/skills/lean-formalization/SKILL.md`
— a native global root, exactly where the docs say to put one — and then neither
it nor any subagent it launched could read it. A subagent whose standing
instructions said "FIRST, EVERY TIME: read `skill://lean-formalization`" got
`Unknown skill` and correctly refused to fabricate, so the operator's standing
instructions were silently unfollowable for the rest of the session.

The mechanism, confirmed in the tree at `f1ab7d346`:

- `session_factory.py:1234` calls `discover_skills(default_skill_roots(Path.cwd()))`
  exactly once, during session construction, and stores the result in
  `hooks.skills_by_name` (`session_factory.py:1237`).
- `_make_knowledge_resolver` (`session_factory.py:1361`) builds
  `make_skill_resolver(hooks.skills_by_name)` (`skills/api.py:130`); that closure
  is the session's `_skill_resolver` (`session/session.py:1901`) and reaches the
  `read` tool through `ToolContext.resolve_internal_url`
  (`session/session.py:6842` → `tools/builtin.py:2643`).
- Children inherit the parent closure verbatim (`harness/subagent.py:1658`,
  wired at `:1884`), so a child's skill catalogue **is** the parent's startup
  snapshot.
- There is no rescan path anywhere. `guide://extensions` (`GUIDE.md:57`)
  documents the consequence — "Start a new session after adding or changing a
  skill" — which is accurate but useless to an agent that has just been told to
  author one.

Two things make this worse than a missing feature.

**The error is opaque.** `resolve_resource_url` raises
`Unknown skill: <name>\nAvailable: ...` (`skills/protocol.py:404`) for *four*
different causes that the agent cannot tell apart: not scanned yet, blank or
missing `description` (silently dropped at `discovery.py:114-116`), frontmatter
`name` disagreeing with the directory name (`discovery.py:118-122`), and a name
collision shadowed by an earlier root (`discovery.py:236-242`, whose warning goes
to a `warnings_out` list printed only at startup). The agent sees one message and
has no next move except to give up or grep the filesystem — which the system
prompt forbids.

**Bodies are already live; only the *catalogue* is frozen.** Verified: after
`discover_skills`, editing an existing `SKILL.md` in place changes what
`skill://<name>` returns on the very next read, because `_read_text_capped`
opens the file at resolve time (`protocol.py:42-53`). So the defect is narrower
than "skills are a startup snapshot" — it is precisely *name → Skill* membership
that is frozen. That narrows the fix considerably.

I also verified the enabling fact independently. `make_skill_resolver` binds the
mapping **by reference**, so an in-place mutation is visible through an
already-constructed resolver, including one an already-running child captured:

```
child sees beta before: Unknown skill: beta / Available: alpha
child sees beta after : '---\nname: beta\ndescription: Beta skill.\n---\n\nBeta body.\n'
mapping id stable: True
```

### A secondary defect found on the way

`_setup_knowledge` computes roots from `Path.cwd()` (`session_factory.py:1234`),
not from the session's `cwd`. `effective_cwd` is already in scope at the call
site (`session_factory.py:1950`, call at `:1952`) and is passed to everything
else — `_seed_mcp_routing`, `_build_variable_store`, `load_repo_guidance`. A
session created with an explicit `cwd` (`bootstrap.py:264`,
`scheduler_service.py:777`, `session/runtime/owned.py:3832`) therefore discovers
project-local skills for the **process** directory rather than its own. It is a
one-line fix and it belongs in this PR, because the refresh path has to decide
which roots it walks and shipping a refresh that is correct while startup stays
wrong would bake in the inconsistency.

## 2. Options for the surface

### Option A — a new `skill` tool (`create` / `list` / `reload`)

Rejected. This is rung 5 of the tool-surface footprint ladder
(`AGENTS.md:1921`), and it does not clear the bar: skill authoring is rare,
skill *reads* are constant. A `skill` tool ships its schema on every request in
every session and every subagent, forever, to serve an action most sessions
never take. The ladder's rung 1 answer applies directly — the capability is a
variation of `write`, which already exists, already creates parent directories,
and already routes an out-of-workspace path (which `~/.local-operator/skills`
is, from a repo cwd) through the approval prompt via `_approval_description`
(`tools/builtin.py:653`). Adding a tool would buy validation and a place to hang
the refresh; §3 gets both for zero schema.

### Option B — a write verb on the `skill://` protocol

Rejected. `read` is the only tool that resolves internal URLs
(`tools/builtin.py:2643`), and its whole contract is that resolution is a read.
A URL scheme where `read` sometimes creates a file is a footgun, and the
resolver contract (`skills/api.py:130-140`: returns content, returns `None` for
other schemes, never raises) has no shape for reporting a partial write.

### Option C — a CLI command (`lop skill new`)

Rejected as the primary answer. It does not solve the problem at all: a
subprocess cannot mutate the running session's `skills_by_name`, so the agent
would create the skill correctly and *still* not be able to read it. It is the
status quo with a nicer `mkdir`. (It remains fine as a later human convenience.)

### Option D — rebuild the whole knowledge layer on demand

Rejected. See §4 — it forces the synchronous resolver contract to become async
and burns the prompt cache, to deliver semantic *selection* of a skill whose
name the agent already knows.

### Option E — documentation only

Rejected: this is the status quo, and the status quo is what produced a subagent
that could not follow its own standing instructions.

## 3. Recommendation: refresh-on-miss, fingerprint-gated, in the skill resolver

**Refresh membership only, only on a miss, only when the filesystem says
something changed, and make the surviving miss self-diagnosing.**

Concretely, `make_skill_resolver` gains optional roots. When a `skill://` URL
names something not in the mapping (and only then):

1. Compute a cheap fingerprint of the roots. If unchanged, skip the scan.
2. If changed, run `discover_skills(roots)` and **`dict.update`** the mapping
   in place. Never rebind it. Commit the fingerprint only after the scan
   SUCCEEDS, so a transient `OSError` retries instead of poisoning the session.
3. Retry the resolution **unconditionally** — the question is "is the name
   resolvable now", not "did this call perform the scan", and under concurrency
   another thread may have populated the mapping while this one waited on the
   lock. If it still misses, run a diagnostic that explains *why* and return
   that instead of the bare `Unknown skill`.

> **Revised during review (round 1).** This originally opened with a 1.0 s
> `_REFRESH_COOLDOWN` gating every probe. It was removed: it is shared by every
> subagent, so a miss at t=0 made a skill written at t=0.05 unreadable until
> t=1.0 — breaking write-then-read, the normal authoring sequence and the exact
> case this design exists for — and it suppressed the diagnostic too. Measured,
> it saved 0.224 ms → 0.002 ms per read on a looping typo. A 0.2 ms saving on an
> error path does not buy a one-second window of wrong answers. The fingerprint
> gate is the real bound and it is stat-based rather than clock-based.

Cost on the frequent path — a skill that exists — is **one dict lookup**,
because nothing above step 0 runs on a hit. That is the constraint the task set
and it is met exactly.

### Why fingerprint-gate rather than just rescanning

Measured on this machine, 8 roots / 57 skill directories:

| operation | cost |
|---|---|
| `discover_skills(roots)` (full scan, parses 57 files) | **17–25 ms** |
| fingerprint: `stat` each root + `scandir` + `stat` each `SKILL.md` | **0.29 ms** |
| `default_skill_roots(cwd)` | 0.15 ms |

A typo'd skill name costs 0.29 ms rather than 17–25 ms — a ~60× reduction on the
exact path the task flagged as the risk. The full scan is paid only when the
tree actually changed, which is the case the feature exists for. This answers
"is a filesystem walk on a typo acceptable, and should it be bounded" with: it
is bounded by the fingerprint, and the residual is sub-millisecond. A typo in a
tight retry loop therefore pays one sub-millisecond stat per read and **never** a
scan, because an unchanged tree compares equal every time.

The 17–25 ms scan runs synchronously on the event loop that renders the TUI
(`tools/builtin.py:2650` is inside an `async def` but the resolver is sync). That
is about one frame, on a rare path, once per real change — I judge it acceptable
and would not complicate the sync resolver contract to avoid it. The fingerprint
is what guarantees it cannot repeat in a loop: the scan runs only on a tree that
actually changed, so repeating it requires repeatedly editing the filesystem.

### What must be in the fingerprint

I tested the mtime semantics rather than assuming them:

| event | root dir mtime | child dir mtime | `SKILL.md` mtime/size |
|---|---|---|---|
| new skill dir created | **bumps** | — | — |
| `SKILL.md` created in existing dir | no | **bumps** | **bumps** |
| `SKILL.md` edited in place | no | no | **bumps** |

Root mtime alone is not enough. The third row is a real case: a skill dropped at
startup for a blank `description` is *not* in the mapping, so fixing its
frontmatter in place is a miss that root and child mtimes both fail to notice.
Including `(mtime_ns, size)` of each `SKILL.md` closes it and is what the 0.29 ms
figure already measures. Use that fingerprint:

```python
# skills/discovery.py — mirrors scan_skills_dir's one-level walk on purpose:
# a fingerprint that walked differently from the scanner would miss changes
# the scanner would have seen.
def roots_fingerprint(roots: Sequence[Path]) -> tuple: ...
```

### Shape of the change (illustrative)

```python
# skills/api.py — roots defaults to None, so every existing caller and the
# parallel guide resolver keep today's behaviour untouched.
def make_skill_resolver(
    skills: MutableMapping[str, Skill],
    roots: Sequence[Path] | None = None,
) -> Callable[[str], str | None]:
    state = _RefreshState()  # last_check monotonic, last_fingerprint, threading.Lock

    def resolver(url: str) -> str | None:
        if not url.startswith("skill://"):
            return None
        try:
            return resolve_skill_url(url, skills)
        except ValueError as exc:
            if roots is None or not _is_unknown_name(exc):
                return str(exc)
            if not _refresh_if_changed(skills, roots, state):
                return _diagnose(str(exc), url, roots)
            try:
                return resolve_skill_url(url, skills)
            except (ValueError, OSError) as retry_exc:
                return _diagnose(str(retry_exc), url, roots)
        except OSError as exc:
            return str(exc)
    return resolver
```

Three details are load-bearing and belong in comments in the source:

- **`skills.update(...)`, never `skills = {...}`.** The entire propagation
  story is that `session_factory`, the parent resolver and every child hold the
  same dict object. A rebinding refactor breaks children silently, with no test
  failure unless one is written for it (§6).
- **Refresh only ADDS; it never removes.** A skill deleted from disk stays in
  the mapping. Removing it would let one agent's cleanup break a sibling child
  mid-read, and the stale entry already degrades gracefully — verified, a
  deleted skill's read returns `[Errno 2] No such file or directory: ...` through
  the adapter's `OSError` catch (`api.py:147-150`), which is a clear message, not
  a crash. Growth-only is the safe monotone direction.
- **Only an "unknown name" `ValueError` triggers a refresh.** `resolve_skill_url`
  also raises for traversal and unsafe child paths (`protocol.py:_resolve_child`);
  rescanning on those would let a malformed URL drive filesystem work. Discriminate
  before refreshing.

### Locking

Subagents run as asyncio tasks on one loop and the resolver is synchronous, so
there is no `await` inside the refresh and it is already atomic with respect to
the loop. But `exec_worker.py` and `exec_mode.py` build sessions on other
execution paths, and the read-modify-write of the fingerprint state is not
inherently thread-safe. Take an uncontended `threading.Lock` around steps 1–3.
It costs nothing and removes the question permanently rather than leaving a
reader to re-derive the GIL argument.

### The diagnostic — the other half of the fix

After a refresh that still misses, `_diagnose` looks for a directory named like
the requested skill under the roots and explains what it found. This is where
"validation on create" actually belongs (§5). Implement it as a new helper in
`discovery.py` called **only** on the miss path, so `discover_skills`'s signature
and its five existing callers (`session_factory.py:1234`, `tui/app.py:28733` and
`:28835`, `server/routes/desktop_catalogues.py:192`,
`scripts/calibrate_skill_threshold.py`) are untouched:

```python
def diagnose_missing_skill(name: str, roots: Sequence[Path]) -> str | None:
    """Explain why <name> did not load, or None when there is nothing to say."""
```

Messages it must produce, one per real cause found in the current code:

| what is on disk | message |
|---|---|
| `<root>/<name>/SKILL.md` missing | `A directory '<name>' exists at <root> but has no SKILL.md.` |
| frontmatter `description` missing/blank | `<path> has no 'description' in its frontmatter; skills without one are not loaded. Add one and read the URL again.` |
| frontmatter unparseable / unterminated `---` | `<path> has malformed YAML frontmatter (it must open and close with ---).` |
| `enabled: false` | `'<name>' is disabled by 'enabled: false' in <path>.` |
| frontmatter `name:` differs from dir name | `<path> declares name '<declared>'; read skill://<declared> (the frontmatter name wins over the directory name).` |
| shadowed by an earlier root | `'<name>' at <path> is shadowed by the one at <winner> (earlier roots win). Rename it or edit the winner.` |

Each line names the fix. That turns four indistinguishable failures into four
actionable ones, and it is the difference between an agent that self-corrects in
one round and one that gives up — which is what the transcript in the report
actually shows.

### Also invalidate the TUI's `$name` cache

`_discovered_skills` (`tui/app.py:28708`) memoizes its own map for the life of
the app, and its docstring (`:28712-28716`) explicitly justifies that by the
restart rule this design removes. Gate it on the same `roots_fingerprint` so a
new skill becomes typeable as `$name` too, and rewrite the docstring — otherwise
the codebase carries a comment asserting a rule it no longer follows.
(`_skills_block` at `:28820` already rescans on every call and needs no change.)

## 4. Propagation semantics — exactly what is and is not covered

**Covered by the in-place mutation** (all three share one dict object):

| consumer | why | evidence |
|---|---|---|
| the parent session | `make_skill_resolver` closed over the dict | `api.py:130`, verified above |
| **already-running** subagents | captured the parent closure at build | `subagent.py:1658` |
| future subagents | same path | `subagent.py:1658`, `:1884` |

**Deliberately not covered:**

- `hooks.index` — the semantic vector index. A new skill is not auto-*selected*
  until the next session.
- `hooks.frozen_block` (`session_factory.py:1138`, set at `:1355`) — the
  rendered `<skills>` listing in the prompt.
- The child's inherited copy of that block (`subagent.py:1775`).
- `hooks.skills_by_name` entries for skills *deleted* from disk (by design,
  above).
- Roots that did not exist at startup. `default_skill_roots` filters ecosystem
  roots by existence (`api.py:83`), so creating `~/.claude/skills` mid-session is
  not picked up. Rare; documented, not fixed.

### Why resolution-only is the right scope, and not a hedge

The task rightly asked me not to casually propose re-rendering the frozen block.
I am proposing not to, and the argument is not cost-aversion:

1. **It would burn the prompt cache for the whole session.** The frozen block
   rides the system-prompt prefix, and the tools array rides the *same* prefix —
   `tools/registry.py` orders the table specifically to keep that prefix stable
   (`AGENTS.md:1921`). Re-rendering mid-turn invalidates both, for every
   subsequent request. That is a real, repeated cost on every session, paid to
   inform an agent of a skill it wrote thirty seconds ago.
2. **It cannot be done from where the miss happens.** `SkillIndex.build()` is
   `async` (`index.py:368`) and may call a network embedding backend
   (`ApiEmbedder`). The resolver is `Callable[[str], str | None]` — synchronous,
   by a contract declared in `harness/types.py:772` and consumed at
   `tools/builtin.py:2650` and `subagent.py:1660`. Rebuilding the index on a
   skill read means making internal-URL resolution async everywhere, which is a
   far larger and riskier change than the defect warrants.
3. **The author already knows the name.** Semantic selection exists to surface
   skills an agent does not know about. The agent that just wrote
   `lean-formalization`, and the subagent whose instructions name it literally,
   both address it by name. Explicit `skill://` reads working is the whole
   requirement.

So: **semantic selection updates on the next session; explicit reads work
immediately.** This is an acceptable, documented limitation, and `guide://extensions`
must state it in exactly those terms so the next agent does not read it as a bug.

## 5. Validation on create

There is no create hook to validate in, because the recommendation deliberately
adds no create tool — the agent uses `write`. Validation therefore lands at the
point of confusion instead of the point of creation, which is strictly better:
it also catches skills that were malformed *before* this change, and skills
written by hand, by another harness, or by a git checkout.

The rules being enforced (as diagnostics, per the table in §3) are the ones the
existing code already applies silently:

- **`description` is required** — missing or blank means the skill is dropped
  (`discovery.py:114-116`). It is the entire routing signal.
- **`name` is optional and wins when present** — it defaults to the directory
  name (`discovery.py:118-122`). Disagreement is legal but confusing, so the
  diagnostic names the URL that actually works rather than calling it an error.
- **Collisions resolve to the earliest root** and the loser is dropped
  (`discovery.py:236-242`). Note the live example: this machine currently
  produces **37** shadow warnings at startup (native root shadowing
  `~/.claude/skills`), all invisible after boot. The diagnostic is the first
  time an agent can see one.
- **Invalid frontmatter degrades, never crashes** (`discovery.py:66-87`) — keep
  that; the diagnostic reports it.

**On the frontmatter `name` vs directory name:** do not add enforcement. The
divergence is deliberate ecosystem-compatible behaviour with a documented
default, and turning it into an error would drop skills that work today.
Diagnose, don't reject.

## 6. Scope, files, tests, risks

**Change class: C1/C2 — standard, pre-authorized.** It extends existing
mechanisms rather than establishing one: no new tool schema, no new trust
boundary, no new protocol. It is explicitly not C5 — the containment checks in
`protocol.py` (`_contained`, `_resolver_would_accept`, dotfile and traversal
rejection) are untouched, and the refresh reads only directories the session was
already scanning at startup.

**Release line:** `Release: patch — a skill authored mid-session becomes
readable immediately, in the session and in its subagents.` A self-contained fix
to a broken behaviour; it does not clear the step-function bar for a minor
(`AGENTS.md:959`).

### Files to touch

| file | change |
|---|---|
| `skills/discovery.py` | add `roots_fingerprint(roots)` and `diagnose_missing_skill(name, roots)`. No change to `discover_skills` or its callers. |
| `skills/api.py` | `make_skill_resolver(skills, roots=None)` — fingerprint gate, in-place `update`, unconditional re-lookup after the refresh, diagnostic on surviving miss, `threading.Lock`. Export the two new helpers. |
| `session_factory.py` | `_KnowledgeHooks` gains `skill_roots: list[Path]`; `_setup_knowledge` takes `cwd` and stores the roots it used (fixes the `Path.cwd()` bug at `:1234`); call site `:1952` passes `effective_cwd`; `_make_knowledge_resolver` (`:1361`) passes `hooks.skill_roots`. |
| `tui/app.py` | `_discovered_skills` (`:28708`) gated on the fingerprint; docstring at `:28712-28716` rewritten. |
| `guides/extensions/GUIDE.md` | replace `:57` (below). |
| `docs/` | this file. |

`prompts_md/system.md` — **no change, deliberately.** A sentence about authoring
would sit in the always-paid prefix to serve a rare action, which is what the
footprint ladder warns against. The routing already exists: `system.md:237-246`
tells an agent that a question about "skills and extensions" MUST go to
`guide://<name>` first. The guide is where this belongs, and it costs nothing
until read.

### What `guide://extensions` must say

`GUIDE.md:57` currently reads "Start a new session after adding or changing a
skill." That becomes wrong on merge and must be replaced with something that
states the split precisely — the sentence is what a future agent will trust:

> A skill you create is readable immediately. The first `read skill://<name>`
> that misses re-scans the skill roots, so the new skill resolves in this
> session and in subagents already running — no restart. What does **not**
> update until the next session is *semantic selection*: the skill will not
> appear in the `<skills>` listing or be auto-suggested, because that listing is
> frozen for prompt-cache stability. Read it by name, and tell subagents its
> name. Editing an existing skill's **body** has always taken effect on the next
> read; editing its `name`, `description` or `enabled` affects selection only,
> and waits for the next session.

The guide should also gain a short "where to put it" line — global
`~/.local-operator/skills/<name>/SKILL.md` for practice that follows the
operator everywhere, project-local `.local-operator/skills/<name>/SKILL.md` when
it belongs to that repository and should be committed — and a pointer that a
skill which fails to load will now say why when its URL is read.

### Test surface

Unit, `tests/unit/skills/test_api.py` and `test_discovery.py`:

1. Miss → skill created on disk → same resolver instance resolves it.
2. **The child test, and it is the important one:** build resolver A, derive
   child resolver B that calls A (mirroring `subagent.py:1658`), create a skill,
   resolve through **B**. This is the regression guard against a future
   rebinding refactor and must assert the child, not the parent.
3. Mapping identity: `skills` object is `is`-identical before and after refresh.
4. Hit path does no filesystem work — monkeypatch `roots_fingerprint` to raise
   and assert a successful resolve still succeeds.
5. Fingerprint unchanged → `discover_skills` not called (patch and count).
6. A looping typo probes per miss but never rescans (the stat is the bound);
   a skill written milliseconds after a miss is readable at once; two THREADS
   missing the same new name concurrently both resolve it; a transient scan
   `OSError` does not freeze the fingerprint.
7. Fingerprint detects each of the three mutation shapes in the §3 table,
   especially in-place `SKILL.md` edit.
8. Deleted skill stays in the mapping and reads as a clean `OSError` message.
9. Traversal/unsafe-path `ValueError` does **not** trigger a refresh.
10. `diagnose_missing_skill` for each row of the diagnostics table.
11. `roots=None` (the default) reproduces today's behaviour byte-for-byte —
    guards every existing caller and the `guide://` resolver.

`tests/unit/test_session_factory.py`: `_setup_knowledge` uses the passed `cwd`,
not `Path.cwd()`, and populates `hooks.skill_roots`.

`tests/unit/tui/test_skill_invocation.py`: `$name` reaches a skill created after
app start.

**End-to-end evidence the PR must carry** (a green unit suite is not evidence of
this): a real session that writes a skill and reads it back, and a real subagent
launched *before* the skill existed that reads it successfully. That second one
is the reported defect and it is the one that has to be shown working.

### Risks to watch during rollout

1. **Rebinding regression.** The single point of failure. Mitigated by test 2
   and by a comment at the mutation site; nothing else will catch it.
2. **Event-loop stall.** 17–25 ms sync scan on the loop that renders the TUI,
   here ~one frame. Rare and fingerprint-bounded — it runs only on a tree that
   actually changed. If a user reports a hitch on a
   pathological tree, the fix is `to_thread` at the `read`-tool call site, not a
   change to the resolver contract.
3. **Pathological trees.** A root with tens of thousands of child directories
   makes even the fingerprint slow, and with the cooldown removed it is paid per
   miss rather than per second. Note that startup already pays a larger version
   of the same cost, and a miss is an error path — this adds no new exposure
   class. If it ever bites, gate the probe on something stat-based (a root-level
   mtime pre-check), never on a clock: a clock is what made a just-written skill
   unreadable in round 1.
4. **Symlinks and mid-walk mutation.** `scan_skills_dir` is one level deep,
   realpath-dedupes and swallows `OSError` (`discovery.py:173-202`), so a symlink
   loop or a directory vanishing mid-walk yields a shorter list, never an
   exception. The fingerprint must walk one level with the same `OSError`
   tolerance; do not make it stricter than the scanner it gates.
5. **Concurrent refresh from two children.** The lock serialises the refresh,
   but serialising is not sufficient on its own: the thread that loses the race
   must still RE-LOOK-UP the name after the lock releases, because the winner
   populated the mapping while it waited. Gating that retry on the refresh's own
   outcome ("did I scan?") makes the loser answer `Unknown skill` for a skill
   already in the dict. Verify no `await` sneaks inside the locked region, which
   would deadlock the loop.
6. **Shadow warnings becoming noisy.** The diagnostic can now surface a class of
   message this machine produces 37 of at startup. It is emitted only for the
   *one* name being read, so it stays targeted — but if a diagnostic ever grows
   to list all warnings, that is a regression.

## 7. What I am recommending against doing

- Do not add a `skill` tool. Rung 5 for a rare action; `write` plus a
  self-diagnosing miss covers it at zero schema cost.
- Do not rebuild the semantic index or re-render the frozen block on refresh.
- Do not remove deleted skills from the mapping.
- Do not add a line to `system.md`. The guide routing already exists.
- Do not enforce name/directory agreement. Diagnose it.

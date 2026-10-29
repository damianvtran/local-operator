# LOH-9 — Role lifecycle: reuse-first delegation and session-scoped (ephemeral) agents/teams

Status: **design for approval** (Phase 1). No product code written.
Base: `origin/main` = `f2a215aaec5e124247eaa579f3b36bceeadfd44c` (v0.82.3, 2026-10-19).
Line numbers below are against that commit.

## 1. Problem

Today every role (`agent` tool) and every team (`team` tool) an agent authors is written
into the user's global stores. Both stores are also the left-sidebar/picker vocabulary, so a
one-off role created for a single investigation stays in the user's agent/team pickers for
ever. The operator wants:

- agent-authored, one-off definitions to be **scoped to the conversation** — usable by that
  session (and its subagents) for follow-up work, but absent from the global pickers;
- **user-requested** agents/teams to stay long-lived/global, as today;
- an explicit **promotion** path from the conversation scope into the global store;
- guidance that prefers **reuse → ephemeral → durable**, so the ephemeral tier is the
  default for an agent's own scaffolding rather than a new way to clutter.

### 1.1 Decisions this design asks you to approve

| Question | Decision |
|---|---|
| Storage/scope | Session scope is `<session_dir>/roles/{agents,teams}` — same on-disk formats as global, no schema change, no migration |
| Visibility | Global readers walk only `<config>/{agents,teams}`, so exclusion is structural; session pickers read a composite registry and mark rows `scope: "session"` |
| Creation policy | Agent-tool writes default to **session**; a tool may write global only with a host-recorded user grant (never on its own initiative); user surfaces write global directly; session create refuses to shadow a global name |
| Lifecycle | Survives turns, restart, resume, fork (copied, independent); dies with the conversation (delete/retention), survives archive; a mesh move refuses by default, and a `--drop-session-roles` move stashes the definitions at a named device-level path rather than deleting them |
| Promotion | Explicit, user-authorized `promote` op/desktop action, copy-not-move, all-or-nothing, collision-safe, team promotion copies its session-scoped members (or refuses), provenance recorded |
| Guidance | Reuse → ephemeral → durable in tool docs + both guides; recommender indexes only the session's own composite |
| Audit | Receipts name scope; `agent/team list` groups by scope; no new journal record type |

## 2. Current state (audit)

### 2.1 Stores and write paths

| Thing | Store on disk | Writer |
|---|---|---|
| Role / specialist | `<config>/agents/<uuid>/agent.yml` + `system_prompt.md`, `conversation.jsonl`, `execution_history.jsonl`, `learnings.jsonl`, `schedules.jsonl` | `AgentRegistry.create_agent` (`agents.py:956`), reached from `write_profile` (`tools/agent_tool.py:1211`) |
| Team | `<config>/teams/<id>/team.yml` + `instructions.md` + `project.md` | `TeamRegistry.create_team` (`teams.py:2082`), reached from `team_tool._op_write` (`tools/team_tool.py:391`) |

Both registries are constructed with the **global config dir** on every host path:
`server/app.py` (`app.state.agent_registry`), `session/runtime/serving.py:10464`,
`session_factory.py:4610/4693` (agent and team composites), `session/attached.py:1578/1641`
(the viewer), `exec_mode.py`,
`exec_worker.py`, `cli.py`, `mobile/daemon.py`, `hub_sync/*`, `network/definitions.py`.

The `agent`/`team` tools read the registry straight off `ToolContext`
(`tools/agent_tool.py:540`, `tools/team_tool.py:286`), and `Session` carries the same two
objects (`session/session.py:3370-3371`), which are re-attached to every turn's context
(`session/session.py:17860/17869`).

### 2.2 Reproduction: a role created today becomes globally visible

Verified against an isolated `LOCAL_OPERATOR_CONFIG_DIR`:

```
AgentRegistry.create_agent(name="loh9-probe", tags=["role"])
→ <config>/agents/231649dc-…/agent.yml (+ conversation/history/learnings/schedules)
→ a SECOND AgentRegistry(config).list_agents() sees it immediately
```

Visibility is **on-disk identity**, not a flag: every reader walks `<config>/agents` or
`<config>/teams`. There is no scope column, no filter, and no per-session store.

### 2.3 Picker/read surface matrix

| Surface | Reads | Session-scoped rows today |
|---|---|---|
| Desktop left sidebar / new-chat: `GET /v1/desktop/profiles`, `/v1/desktop/teams` (`server/routes/desktop_profiles.py:157/359` via `registries()` at :148) | app-state **global** registries | would leak if stored globally |
| Desktop session command picker: `GET /v1/desktop/sessions/{id}/command-entities` (`desktop_catalogues.py:781`, :826) | **session's** `remote.agent_registry` / `remote.team_registry` | natural home |
| TUI `/agent`, `/team`, settings agents pane, org chart (`tui/app.py:19715`, `21004-21021`) | **session's** registries | natural home |
| TUI new-chat start picker (`tui/app.py:37750-37776`) | fresh **global** registries | must stay excluded |
| `lop agents list`, `lop team list` (`cli.py:10366/11036`) | **global** | must stay excluded |
| Agent-hint recommender (`session_factory.py:1984`, index build at :2175) | the session's `agent_registry` | included in-session, absent elsewhere |
| Hub sync / mesh definitions (`hub_sync/*`, `network/definitions.py:615/631`) | global roots | must stay excluded |
| Desktop schedules (`server/routes/schedules.py`) | global | must stay excluded (a schedule outlives a conversation) |

Conclusion: **the session pickers already read the session's registries**. Replacing those
two objects with session-aware composites gives in-session visibility with no picker-specific
code, while the global routes, CLI, hub and mesh are excluded *by construction*. The only
gate that refuses a non-`TeamRegistry` object is `team_tool.py:288`
(`isinstance(raw, TeamRegistry)`) and must become a duck-typed check.

### 2.4 Resolution and identity

- `resolve_profile_or_specialist` (`agent_profiles.py:704`) is the single resolver for
  `/agent` attach, a team's manager and the org chart. It folds case and falls through to
  packaged seeds. `resolve_profile` (:610) consults the registry first.
- Registry access is already duck-typed (`Any` + `getattr`); the surface actually used is
  small: `list_agents`, `get_agent`, `get_agent_by_name`, `get_agent_system_prompt`,
  `set_agent_system_prompt`, `create_agent`, `update_agent`, `save_agent`, `delete_agent`,
  `clone_agent`, `definition_revision_fields`, `require_complete_metadata`, `refresh_now`,
  `config_dir`; and `list_teams`, `get_team`, `get_team_by_name`, `create_team`,
  `update_team`, `save_team`, `delete_team`.
- Identity is **name** for addressing and **uuid** for storage. `AgentData` has no scope
  field; adding one would change every stored YAML and every export. The composite should
  keep that schema untouched and expose scope out-of-band (see §3.3).
- Team rosters reference members **by name** (`teams.py` docstring), so a session team's
  members resolve through whichever composite the session holds. Subagents inherit the
  parent's registry object (`harness/subagent.py:4142-4143`), so the scope travels with
  delegation without a second mechanism.

### 2.5 Lifecycle surfaces

- **Restart/resume**: registries are rebuilt from disk; a session dir survives, so scoped
  rows survive if they live there.
- **Fork** (`fork.py:138` `COPIED_SIDECARS` + `_copy_roles_tree`): copies the transcript and
  identity sidecars, and — since S3 — the parent's `roles/` tree, byte-for-byte, into the
  fork's own directory. Without that copy the fork inherited `attachment.json` while losing
  the definitions it names, so it opened wearing a persona it could not resolve. The copy is
  one named directory, not a `copytree` of the session (see §3.5 for the bound and refusals).
- **Delete/retention** (`session/cleanup.py:926` `remove_session_dir`, reached by the
  explicit delete at `:1868` and the policy at `:2772`): the only `rmtree` of a session dir;
  scoped rows die with the conversation, correctly, and the tests assert it — including that
  the account stores are byte-identical before and after.
- **Mobility/mesh** (`network/mobility.py`): transfers the session record/journal rather than
  the whole directory, so scoped files would not travel unless added deliberately. Must be
  documented (and optionally included later).
- **Agent dirs are exported wholesale** (`agents.py:export_agent*`), so session row state
  files stay inside the session dir and never enter a global export.

## 3. Design

### 3.1 Scope model

Two scopes, one owner:

- **global** — `<config>/agents`, `<config>/teams` (unchanged).
- **session** — `<session_dir>/roles/agents/`, `<session_dir>/roles/teams/`.

The scope belongs to the **conversation root** (the top-level user session), not to each
subagent run. Children receive the parent's composite object
(`harness/subagent.py:4142-4143`), so a role a child creates remains available to the parent
and to later siblings — required for supervision and for "follow-up tasks in the same
session".

Wiring: `create_session` wraps the registries once it knows `transcript_dir` (before
`_setup_knowledge`, `session_factory.py:4610/4693`) only when the incoming registry is not already
scoped (a `scope_dir` marker attribute). A subagent's `create_session` therefore inherits the
parent's composite unchanged rather than nesting a second scope. `AttachedSession` builds its
lazy properties from `<config>/sessions/<record id>/roles` for the viewer/cold-session path
(`attached.py:1578/1641`). Every other host (CLI `exec`, `exec_worker`, `lop agents`, `lop
team`, mobile daemon, hub, mesh) keeps plain global registries.

Why `roles/` under the session dir, rather than a `scope` column in the global store:

1. **Exclusion by construction.** A global reader that walks `<config>/agents` physically
   cannot see a session row; nothing has to remember to filter. A scope column fails open on
   the next reader someone adds.
2. **Lifetime is already right.** Backups, retention, delete, archive and session sidecars
   already treat the session directory as the unit.
3. **No migration.** New directory, old builds ignore it, global YAML schema untouched.

`roles/` (not `agents/` directly) keeps the session-directory vocabulary clear and leaves
room for future scoped artifacts; the registries are pointed at
`<session_dir>/roles` (they append `agents/` and `teams/`).

### 3.2 Registry seam: composite read/write authority

New module `local_operator/profile_scope.py`:

```python
class ScopedAgentRegistry:   # answers the whole agent-registry surface
    def __init__(self, global_registry, session_registry | None, *, session_dir)
    def scope_of(self, agent) -> Literal["global", "session"]
    def store_for(self, scope) -> AgentRegistry      # the plain registry to write through
    ...

class ScopedTeamRegistry:    # same shape for teams
```

Read rules:

- `list_agents()` / `list_teams()`: session rows **first**, then global, both in their
  registry's order. (The pickers re-sort; order only matters for duplicate names, below.)
- `get_agent_by_name(name)` / `get_team_by_name(name)`: **session first**, then global.
  (`resolve_profile`'s case-insensitive retry goes through `list_agents()`, which the
  composite also serves, so folding keeps working.)
- `get_agent(id)` / `get_team(id)` / `get_agent_system_prompt(id)` /
  `set_agent_system_prompt(id)` / `definition_revision_fields(id)`: dispatch to the store
  that owns the id (session first, fall back to global).
- `require_complete_metadata()` / `refresh_now()` / read recovery: both stores, and name
  which store failed.
- `config_dir` stays the **global** root, so existing callers that derive paths from it keep
  working; a new `agent_dir(agent_id)` returns the owning store's `<store>/agents/<id>`.
  Needed by `session_factory._registered_agent_hints` (`session_factory.py:2003`), which
  computes `config_dir/agents/<id>` directly today; `agent_profiles.agent_listing_rows` needs
  only `list_agents` + `profile_from_agent`, which the composite serves as-is.
- If the host has no persistent session directory (`ToolContext.session_dir is None`, e.g.
  an embedder that runs without a session store), there is no session scope: reads fall back
  to global and a `scope="session"` write is **refused** with a clear message (never
  silently written globally).
- Hub-only operations (`sync`, import/export, push) refuse for a session row with "promote
  it to your account first"; they are not part of the session lifecycle.

Write rules — **explicit at the call site, never a silent default inside the composite**:

- `store_for(scope)` returns a **write adapter**: it answers the plain registry surface
  (`list_agents`, `get_agent_by_name`, `create_agent`, `update_agent`, `delete_agent`, and
  the team twins) and takes the composite's write lock around create/update/delete, so two
  concurrent subagents cannot both pass the uniqueness check. The underlying store keeps its
  own locks too.
- A helper `scope_store(registry, scope)` in `profile_scope.py` is what the two write entry
  points use: for a plain registry it returns itself on `"global"` and refuses `"session"`;
  for a composite it returns the chosen adapter. `write_profile` (`tools/agent_tool.py:1211`)
  and `team_tool._op_write` (`tools/team_tool.py:391`) then call today's method names
  unchanged, so the plain signatures do not move and `hub_sync`, `network/definitions`,
  `cli`, `exec` and `schedules` stay global without edits.
- Uniqueness for a **session** create checks both stores (a global name is refused, §3.2
  "Name collisions"); an update resolves the row first and writes through the store that
  owns it.
- Callers choose the scope because they know who is acting (§3.4).

Two construction fixes the composite depends on:

- `AgentRegistry.__init__` currently `mkdir`s the root and runs two migrations. Add
  `AgentRegistry(config_dir, *, initialize=True, migrate=True)`; the session store is built
  with `initialize=False, migrate=False` and created lazily on first write (mirroring
  `TeamRegistry`, which deliberately does not mkdir).
- `TeamRegistry.__init__` runs crash recovery; recovery on a missing `teams/` must stay a
  no-op (it already reads "missing directory = no teams"; assert it).

### 3.3 Visibility

- **Session pickers** (`/agent`, `/team`, desktop command-entities, mobile sheet) gain the
  session rows automatically because they read the composite.
- **Global surfaces** do not: global catalogue routes, `lop agents/team list`, TUI
  new-chat picker, hub sync/push, mesh definitions, schedules. No filtering code is added to
  them; a regression test points a global registry at the config root and asserts the
  session row is absent.
- **Payloads** gain an additive `scope: "global" | "session"` on profile/team catalogue and
  detail rows, and on `/command-entities` rows. SQL/HTTP clients that predate it ignore it.
  New feature key `profile_scope: 1` (own key, per `server/features.py`'s rule: it lets a
  renderer know it may send `session_id`/render a badge; nothing existing is withheld).
- **TUI** marks a session row in the existing `facts`/detail column (`role · this
  conversation`, `team · this conversation`) — never in the name column, which is the
  completion key. No new picker, no separate mode.
- **Desktop/mobile** copy lives in the UI/mobile clients (separate repos); this PR ships the
  additive fields and the spec in `docs/DESKTOP_API.md`.

**Name collisions.** The composite refuses a *session* create whose name already exists in
the **global** store, with a message that points at reuse or promotion. Rationale: silently
shadowing `reviewer` for a whole conversation is the failure `resolve_profile`'s A1 history
warns about, and both `agent_listing_rows` and `profile_catalogue` dedupe by name, so a
shadow would hide one of two rows in some surfaces. The reverse race (a global row created
later with the same name) is tolerated because global writers cannot see other sessions; the
composite then resolves **session first** and both surfaces badge the scope, so the shadow is
visible rather than silent. In that raced state an unqualified `update` from the session
edits the session row; the global row is reachable from a user surface (or after the session
row is renamed/deleted).

### 3.4 Who writes where (the user-vs-agent rule)

The scope is decided at the **write site**, not by inspecting the conversation:

| Writer | Scope | Authorization |
|---|---|---|
| `agent` tool `create`/`install`/`reset` (agent acting for its own task), no `scope` | **session** default | none needed |
| `agent` tool with `scope='global'` | global | **requires a host-recorded user grant** (below); without it the call is refused |
| `team` tool `create` (agent acting), no `scope` | **session** default | none needed |
| `team` tool with `scope='global'` | global | same grant requirement |
| `agent op='promote'`, `team op='promote'` | global | same grant requirement + explicit confirmation |
| Desktop profile/team editor and routes, TUI `/agent class` + team form (`tui/app.py:37394`), CLI `agents`/`team` commands, `--agent NAME` auto-create (`session_factory.py:300`) | global | the user IS the actor; no tool grant involved |
| `install_seed` reached from `action_class.py` (a user's `/agent class` command) | global | user action; handed `scope_store(registry, "global")` |

**The global-creation guard (condition 1).** An agent tool may not choose
`scope='global'` on its own initiative. The host puts a **durable-write grant** on the tool
context only when the user authorized it in this turn:

- the user acted on a user surface (desktop editor / TUI form / CLI) — those write directly
  and never need the tool gate; or
- the user answered an `ask` that explicitly granted durable creation (`promote`/`global`),
  which the session records with `origin: "ask:<id>"`; or
- a user command routed through a slash surface carries an explicit durable flag.

`ToolContext` gains a declared, turn-scoped `durable_write_grant: str | None` (the origin
string). The tools read it and:
- no grant + `scope='global'` → **refuse**, with the remedy in the message: ask the user
  whether this should be available in all conversations; after they agree, retry;
- grant present → proceed, and the receipt's `details` carry `scope` and `authorized_by`
  (the origin), so the tool call in the transcript is the audit record.

The default remains `session`; the grant never flips the default.

Implementation: `write_profile(..., scope=..., grant=...)` and
`team_tool._op_write(..., scope=..., grant=...)`. The tools compute
`params.scope or "session"` and pass the context grant through. No implicit default inside
the registries, so a missed adapter is a type error at review, not a silent leak.

Tool-schema budget: add one short `scope` field description and one sentence to each tool
description; the decision ladder and rationale go in `tool://agent`, `tool://team`,
`guide://agents`, `guide://teams` (billed only when read).

### 3.5 Lifecycle

| Event | Session-scoped rows |
|---|---|
| Turn/task ends, fails, is cancelled | **kept** — "ephemeral" means conversation-scoped, not turn-scoped; follow-ups must work |
| Runtime restart / resume | kept (session dir is durable), composite rebuilt from `<session>/roles` |
| Subagent launch | inherited by object; a child's creations land in the parent conversation scope |
| Fork | **copied** into the fork's `roles/`, byte-for-byte, then **independent**: edits/deletes in the fork never affect the parent (and vice versa), and a fork never writes a global row. `fork.py` stays an allow-list (`COPIED_SIDECARS` are files); the ONE directory is copied by `_copy_roles_tree`, because the inherited `attachment.json` names definitions the fork must be able to resolve on its own. Absent `roles/` is a no-op (lazy creation is an S1 invariant), and nothing else is a source: not the stash, not `EXCLUDED_SIDECARS`, no global root. The tree is walked first and copied only if its summed FILE bytes are within `MAX_ROLES_COPY_BYTES` (**64 MiB** — a role is an `agent.yml` plus a prompt, a team a `team.yml` plus two capped briefs, so a real tree is well under a megabyte; the bound exists because the copy runs on the parent's process between turns). The walk is only a snapshot, so the same bound is re-checked by a byte-counting copy function while the bytes move, and a tree that grew past it is stopped mid-copy. A tree over the bound, a symlink anywhere in it, or an I/O failure is REFUSED: the fork directory is removed and its claim released, so no fork exists rather than a fork wearing a half tree. The copy is staged in a temp sibling inside the fork and published with one `os.replace`. |
| Session delete | removed with the session dir (`cleanup.remove_session_dir`) — and only the addressed session's dir |
| Retention sweep | an old session is removed with its roles. An **archived** session is **not** intrinsically protected: `session/archived.py:51-58` states it is still eligible, and the only guard that can save it is the picker's ten most recent (`cleanup.py:2930-2949`), which counts archived rows while they are that recent. Once an archived session falls outside that window it is swept like any other — asserted as the documented behaviour in S3, not changed. |
| Archive / unarchive | kept; the directory and its `roles/` are untouched |
| Mesh move to another device | **v1 does not carry role definitions.** The move path detects `<session>/roles`; by default it **refuses** and names the roles, with the two real ways forward: stay on this device, or re-run with the drop flag. `--drop-session-roles` proceeds and — because the commit deletes the session directory — **moves the tree to a device-level stash** at `<config>/dropped-session-roles/<session_id>/` (suffixed `-2`… on a collision) with a `stash.json` naming the definitions. Nothing walks that root (no sync, retention or cleanup ownership), and the receipt/queued receipt names the exact path; the failure this replaced promised the roles "stayed" while deleting them. Follow-up filed for mesh-synced role definitions. |
| Global hub/mesh push | never includes session rows |

**Invariant: no cleanup of existing user definitions.** This change never migrates, rewrites,
re-tags or deletes a pre-existing global agent or team. The only global writes are a
user-surface create and an authorized promotion.

Cleanup affordance: `agent` gains `op='delete'` **for session-scoped rows only** (global
deletion stays on user surfaces), and `team_delete` resolves scope by name (session first)
and deletes the session row when that is what it found. A session role a model created for a
wrong turn can therefore be removed without deleting the conversation, and no new global
destructive authority is created.

Auditability/supervision:

- Every receipt names the scope in plain words: `created role 'release-check' in this
  conversation` vs `in your account`; promotion says `copied to your account`.
- `agent list` / `team list` group under `this conversation` and `everywhere`.
- The tool call itself is the journal record (arguments include the scope); no new journal
  row type is introduced.
- A parent sees a child's creations because they share the scope root; no extra channel.

### 3.6 Promotion

Explicit, user-authorized, copy-not-move (the brief says "promote/copy"), collision-safe,
and **all-or-nothing**:

`agent op='promote' name=X [as=Y] [replace=false]`
- Requires the same durable-write grant as `scope='global'` (§3.4); the desktop action is a
  user surface and carries its own confirmation.
- Resolve `X` in the session scope; if only a global row exists, refuse (`already available
  everywhere`).
- **Preflight everything before the first write:** session row readable, target name free (or
  `replace` confirmed), instructions readable, provenance writable.
- Write the global row with the same name (or `as_name`), then the `system_prompt.md`, then
  one row in the DEVICE-LEVEL provenance index `<config>/role-provenance.jsonl`
  (`{kind, id, name, source_session, at, authorized_by, replaced}`, one line per definition
  created or overwritten). **Corrected from this section's first draft, which put a
  `provenance.json` inside the row directory:** `TeamRegistry._save_team_locked` publishes a
  team row by renaming a whole staged directory over it and writes exactly `team.yml`,
  `instructions.md` and `project.md`, so any other file in a team row is destroyed on the
  next save — a receipt pointing at a sidecar would be a promise the store does not keep.
  The index is append-only, locked, and rotated past a bound, and its writability is part of
  the preflight. The ROW is copied VERBATIM (a transported copy of the stored `AgentData`,
  not a rebuild from `AgentEditFields`): `write_profile` re-derives tags, and the tags are
  the role's kind and policy.
- If any step fails, delete the rows this call created (rollback) and report exactly what
  remains if rollback itself fails. No partially promoted row is left silently.
- If the global name exists: refuse, naming the existing row, unless `replace=true`.
  `replace` overwrites a user-owned definition, so it requires the user's explicit
  confirmation and the receipt names exactly what was overwritten.
- Keep the session row; the receipt says the copies are independent from here on.

`team op='promote' name=X [as=Y] [replace=false]`
- Same durable-write grant and confirmation as the role path.
- Preflight the team row, both briefs, every target name, and **every roster member** the
  team references **before the first write**.
- Copy the team row + both briefs + one provenance row; copy session-scoped members in the
  same transaction. Session-scoped members (and nested
  session teams, bounded by `teams.MAX_ORG_DEPTH`) must be promoted too or the team breaks in
  another session: they are promoted automatically in the same call and listed in the
  receipt. **The roster is resolved through the LAUNCH resolver** (`profile_detail` per
  member, recursion bounded by `MAX_ORG_DEPTH`, the same walk a launch validates), not by
  store membership: a packaged starter that is merely not installed is resolvable everywhere
  and needs no copy, so treating "absent from both stores" as missing would refuse a team that
  launches fine. Rows created by this call are rolled back if a later write fails, and
  exactly what remains is reported if rollback itself fails. A documented cap
  (`profile_scope.MAX_PROMOTION_ROWS`) bounds the expansion breadth that depth alone does
  not.
- Any global name collision (team or member, including aliases) refuses the whole promotion
  unless `replace=true`.

Desktop: **`POST /v1/desktop/sessions/{session_id}/promote`**, body
`{request_id, kind: "agent"|"team", name, as_name?, confirm_replace?}`; response is the new
ACCOUNT detail row plus an additive `promoted` receipt. **Corrected from this section's
first draft** (`/profiles/{name}/promote` and `/teams/{name}/promote` with a `session_id` in
the body): the SOURCE is a conversation-scoped row, reachable only through the session
bridge, and the account-only profile routes are built over the global registry where that
row does not exist — addressing the route by session is the shape `/command-entities`
already uses. `as` is a Python keyword, so the field is `as_name` on every surface.
The UI affordance is a single action on a session-scoped row: **"Use in all
conversations"**, with a confirmation that names the created/copied definitions.

The TUI has the same action as a reserved verb — `/agent promote <name>` and
`/team promote <name>`, rehearsed and then confirmed with `yes` (the command grammar's own
confirmation, as `/project delete` and `/delete` use it; no new picker and no modal, per §7).

**The tool surfaces ship REFUSAL-ONLY (S4 decision D1).** `agent op='promote'` and
`team op='promote'` exist and go through the durable-write seam with
`operation="promote"`, and no host installs a consent producer in this slice — so today
every tool-issued promotion is refused, naming the TUI promote verb as the way
forward. Promotion is a user act on a user surface; installing an intent-specific
confirmation producer remains the S2 deferral and is not smuggled in here.

### 3.7 Prompt and recommender guidance

Decision ladder, stated once in each place the model reads it:

1. **Reuse** — `agent op='search'` / `op='list'`, `team op='list'`; a role/team that already
   covers the work wins.
2. **Ephemeral** — if nothing covers it and the role exists for *this* task/conversation,
   create it with the default session scope. Do not announce it as a lasting feature.
3. **Durable** — only when the user explicitly asks for something that should outlive the
   conversation. The agent cannot decide this alone: it needs the user's grant (§3.4), or the
   user creates the definition on their own surface. When a user later asks that an existing
   session role/team be available elsewhere, `promote` it (same authorization).

Changes:
- `tools/agent_tool.py` description + `AgentParams.scope` + the grant refusal; `tools/team_tool.py`
  description + `TeamParams.scope` + the grant refusal; per-op blurbs and the ladder in
  `tools/tool_docs.py` (`TOOL_NOTES["agent"]`, `["team"]`).
- `local_operator/guides/agents/GUIDE.md`: add a "Where a role lives" section.
- `local_operator/guides/teams/GUIDE.md`: replace "Never create a team ad-hoc for a one-off
  task" with the ladder — a one-off team is allowed **as a session team**; the durable bar
  for a global team is unchanged.
- Recommender (unchanged by S5): `_registered_agent_hints` indexes role/specialist metadata over
  the session composite (session rows included, with the owning `agent_dir`), but it never
  names a role and records no hit. A match is reduced to a boolean that appends the
  `agents` guide to the prompt hint; `tests/unit/guides/test_guides.py` asserts that names and
  descriptions never reach that block. The index is built at session construction, so a
  role created mid-session is findable through `agent list` but absent from the semantic
  hint until the next session (the same limitation global rows already have; teams are
  not indexed at all). Picker clutter and reuse cannot be measured from this index; the
  S5 pilot measures the policy from transcripts instead.
- Delivery-guide pilot (shipped with S5): a scripted fresh-session pilot exercising "niche
  task → ephemeral", "user says make it available → promote", "reuse an existing role
  instead of creating" and a no-durability negative, plus a read-only transcript analyzer.
  It measures four scenarios (three positive and a negative control); it cannot measure
  live-model compliance, production reuse or picker clutter — no `op`/`scope` telemetry
  exists, and user-surface promotion bypasses agent-tool receipts. The PR states those
  limits.

## 4. Serial slice plan

Each slice is one PR, tests first, independent review + QA; design/UX review on the slices
that change a visible surface; full CI once at the terminal head; **no version bump, no
release cut**.

### S1 — Scope store + composite read path (foundation)
- New `local_operator/profile_scope.py`; `AgentRegistry(initialize=…, migrate=…)`;
  `agent_dir()` seam; wire the composite into `create_session` (after `transcript_dir` is
  known, before `_setup_knowledge`) and into `AttachedSession` (viewer/cold-session path,
  `attached.py`), using `<config>/sessions/<id>/roles`.
- **Merged-alone safety:** writes are delegated to the global store exactly as today, so S1
  changes no behaviour; no tool-schema change and no visible payload yet. Session rows are
  created by tests only.
- The `isinstance(raw, TeamRegistry)` gate in `team_tool.py:288` becomes duck-typed here, not
  in S2: with the composite wired, the old gate would make the team tool report "no registry"
  in every session.
- Tests: overlay/precedence/id dispatch; global registry never sees session rows; viewer
  path builds the same composite; lazy creation leaves no directories on a fresh session;
  team recovery on a missing dir; a write through the composite still lands global.
- Exit: `/agent`/`/team` list rows placed by a fixture in `<session>/roles` on both the
  in-process and viewer paths.

### S2 — Writes, pickers, badges
- `scope` on `AgentParams`/`TeamParams`; `write_profile(scope=…, grant=…)`; team write
  routing; default `session` in the tools; **the durable-write grant** (`ToolContext`
  field + `ask`-recorded origin + refusal copy) and its audit `details`; user adapters pass
  `global` (desktop routes, TUI class/authoring, CLI, `install_seed` from `action_class.py`).
- Name-shadow refusal; in-process create lock; receipts name scope; `agent delete`
  (session-only); team delete resolves scope.
- Additive `scope` on catalogue/detail/command-entity rows; `feature_flags["profile_scope"]`;
  TUI facts badge.
- i18n: new wire keys (scope-shadow refusal, promote refusals reserved for S4) in all six
  locales via `scripts/i18n/generate.py`.
- Tests: tool defaults/overrides, **global refusal with no grant + success with a recorded
  grant**, receipts/details audit fields, TUI rows, desktop command-entities, global surfaces
  unchanged.
- **Gate: `roles/` stays non-writable until S3's mesh refusal/notice lands with or before
  this slice.** The copy set excludes session roles for v1, so an S2 write path without that
  refusal would let a deleting move leave roles behind silently. The S2 PR carries this gate.
- Gate: design/UX review of badge wording; i18n `--check`; tool-doc drift tests updated.

**S2 status (landed 2026-10-20; PR pending).** Shipped: explicit `scope` on both write
entry points and the composite creates; the name-shadow refusal; one create lock; scope
receipts and `details`; `agent op='delete'` (session-only, write-tier) and a session-first
`team_delete`; additive `scope` on catalogue/detail/command-entity rows;
`features.profile_scope: 1`; the TUI facts badge and the `agent_listing_rows` dedupe; the
mesh refusal and `--drop-session-roles`. Two corrections to this document, then the one
deferral its §3.4 needed.

**Correction 1 — `write_profile`'s home.** The function lives in
`local_operator/tools/agent_tool.py`, not `agent_profiles.py`; §3.2's pointer was right,
but the S2 bullet's phrasing read as though the shared writer belonged to the profile
module. Nothing moved: the required `scope` keyword was added to the function where it
already was, and `install_seed` (which IS in `agent_profiles.py`) takes scope as a
required keyword too, routing its create through `scope_store`.

**Correction 2 — `agent_listing_rows` dedupe.** §3.3 says the listing "dedupes by name".
It did not: `seen` was populated for role/specialist rows but only CONSULTED for packaged
seeds, so the tolerated reverse race (an account row created after a session row of the
same name) painted the key twice. S2 makes the code true — the composite lists session
rows first, the loop skips a later duplicate, and the surviving row's scope fact says
which store is in force.

**Deferral — the durable-write grant, and why it is not a string.** §3.4 proposed
`ToolContext.durable_write_grant: str | None` carrying an `ask`-recorded origin. S2 ships
a host seam instead: `ToolContext.authorize_durable_write(DurableWriteIntent) ->
Awaitable[str | None]`, with the origin recorded in the receipt's `details` only. Three
reasons: (a) a string on the tool context is a value a model's turn can be given, and a
model-influenced provenance string is not consent; (b) the existing TUI approval card is
satisfied by the session-scoped "approve all" latch without showing this specific intent,
so wiring the seam to it would let a blanket shortcut authorize account-wide writes, and a
non-rememberable intent-specific confirmation is new UI that S2 deliberately does not
build; (c) an answered `ask` is not by itself human authority — `asks/queue.respond`
writes the caller-supplied `by` (`asks/queue.py:559-565`), and `Session.bridge_decide`
treats the in-memory admitted grant, never the ask-log fold, as the evidence
(`session.py:11585-11589`). No host installs the callback in S2, so every tool-issued
`scope='global'` write is refused with the user-surface remedy and the conversation-only
retry. AC6's tool branch therefore moves to S4 with promotion, the first flow that needs
an explicit "make this durable" answer; that slice should build the non-rememberable
confirmation and only then install the seam.

**Round-1 remediation (2026-10-20).** Review/QA/design/UX measured four behaviours the
first S2 head got wrong, folded in here as the contract now:
- **The drop path stashes, never deletes.** `_remove_handed_away` runs the stash for every
  removing commit (not only flagged ones — recovery paths have no frame); a stash failure
  leaves the directory in place (`cleanup_pending`) rather than deleting past it. The
  refusal, the receipt and the queued receipt all name the path.
- **`agent op='sync'` is gated on the same consent as any other account write.** It first
  classifies with `apply=False`; a seed update that would rewrite an account row is refused
  without consent. The hub arm is account-only, so it runs only under that consent and
  otherwise reports that it was skipped (the session-scoped arm still runs). `install` and
  `reset` take the caller's scope: an explicit `global` without consent is refused, never
  silently downgraded, and both receipts name the store.
- **A name collision is answered before the consent question.** A create whose name already
  exists in the other store is refused with reuse/delete guidance (case-insensitive, and
  team **aliases** count), so "retry without scope" can never loop around an occupied name.
- **Both pickers dedupe.** `agent_listing_rows` and the team picker drop a later duplicate
  of a name (session rows are listed first, so the row in force wins), and `list`/`show`
  group and label by store ("this conversation" / "your account").
- **The mesh gate fails closed.** An unreadable `roles` tree refuses with
  `session_roles_unreadable`; it never answers "no definitions". A tree that is
  readable but holds a row file that does not parse refuses the same way and names
  the offending path, because the stores log-and-skip a corrupt row — an empty list
  there is indistinguishable from a clean tree and would fail the gate open.
- **The scope fact fits the pane.** When the 27-cell facts line cannot hold both, the
  settings pane paints a SECOND facts row carrying the full scope token in the existing
  `muted` ink — dropping or clipping it was measured as the failure (D1/U2).
- **`scope` also rides the team catalogue/detail routes** (`/v1/desktop/teams`), which stay
  account-only; the field is `"global"` there and the session composite remains behind
  `/command-entities` only.
- The consent refusal distinguishes **no seam** from **declined**, names ONE concrete next
  action, and no refusal offers "promote" — that capability is S4.

### S3 — Lifecycle integration
- **The one new behaviour: the fork copy.** `fork_session` copies `<parent>/roles` into the
  fork's own directory — walked and bounded (64 MiB) before the copy, staged and published
  with one `os.replace`, refused on a symlink or an over-bound tree with the fork directory
  removed and its claim released. Independent in both directions; no global write; lazy
  absence preserved; the stash and every global root stay out of reach. See §3.5.
- **Verification only, no behaviour change** (each already resolves through the session
  composite; S3 proves it): `task(agent=<session role>)` and
  `task(agent="team:<session team>")`, including the existing degradation of an unknown
  name to a full child (`harness/subagent.py:_resolve_role`); `/agent <name>`, `/team <name>`
  and `attach_team` — **no resolver change**, both handlers already read the session's
  composite (`attach_team` is handed the team the handler looked up); the cold viewer path;
  name shadowing in all four launch paths; delete/retention/archive; resume/restart; the
  agent-hint index pointing at `<session>/roles/agents/<id>`.
- The **concurrent-create lock is S2's**, not this slice's; S3 has no lock work.
- Docs: this design's lifecycle table (and its three corrected claims). No `DESKTOP_API.md`
  change — no wire shape moves.
- Exit: an isolated end-to-end scenario: create ephemeral role + team, launch both, restart
  the runtime, resume, use them again, fork, use them in the fork, delete the session.

### S4 — Promotion
- `promote` on both composites, tool ops, desktop promote routes, feature flag, i18n copy,
  grant enforcement, provenance sidecars, all-or-nothing preflight/rollback, member
  expansion, collision/`replace` rules.
- Desktop UI work is specified here for the UI repo (button + confirmation + receipt);
  server-side tests prove the contract.
- Gate: design/UX review; destructive-confirmation review for `replace`.

### S5 — Prompt/recommender guidance + pilot
- Tool descriptions/tool docs/guides from §3.7; hint index uses `agent_dir`; pilot prompts
  and results retained in the session scratchpad, referenced from the PR; `docs/design`
  finalized.
- Gate: fresh-agent pilot evidence (delivery guide §"Evidence and pilots").
- **Delivered in this slice's PR:** four scripted scenarios (reuse, ephemeral,
  promote-on-request, no-durability negative) and the read-only analyzer; the actual
  transcript facts and the list of what the pilot cannot measure are in the PR body.

## 5. Acceptance checks

Provisional IDs (the Notion row was unreadable — see §9); map these to the tracker's own
checks when available.

| ID | Check | Evidence |
|---|---|---|
| AC1 | An agent-created role/team (no explicit scope) does **not** appear in `GET /v1/desktop/profiles`/`/teams`, `lop agents list`, `lop team list`, the TUI new-chat picker, hub sync, or mesh definitions | integration test with isolated config; real HTTP + real registry |
| AC2 | It **does** appear in that session's `/agent` and `/team` listings and in `GET /v1/desktop/sessions/{id}/command-entities`, marked `scope: "session"` | TestClient + TUI picker test |
| AC3 | It remains usable for follow-up work in the same session: `task(agent=…)`, `/agent <name>`, `/team <name>` and `task(agent="team:…")` all launch it | session tests, real launch |
| AC4 | After a runtime restart/resume the same session still resolves it | restart/resume test |
| AC5 | A fork can use the parent's session roles/teams | fork test |
| AC6 | A user-requested creation is global: created via a user surface, or via the tool with `scope='global'` **plus a recorded user grant**, and visible in every global picker | integration test |
| AC7 | Promotion (with grant + confirmation) copies a session role to the global store with provenance; a later, separate session sees it; the session copy still works | promote test + second-session read |
| AC8 | Promotion of a team copies its session-scoped members (or refuses, naming them); a promoted team launches in another session | promote test; launch in a second session |
| AC9 | A name collision is refused with the reason, or honours `replace` only when confirmed; no partial writes within a live process, and a SIGKILL leaves at most the members written so far — never a team row whose roster does not resolve (no journal exists; re-promoting collides with a named remedy) | negative tests |
| AC10 | Deleting the session removes its roles/teams; the global store is untouched; retention sweep has the same effect | cleanup tests |
| AC11 | No regression: existing global create/list/launch, `/agent class`, schedules, hub and mesh flows behave identically | existing suites + focused regression |
| AC12 | Guidance is present where the model reads it (tool docs, both guides) and the recommender never leaks session rows outside the session | doc-drift tests + pilot |
| AC13 | Every create/update/promote/delete receipt names the scope, and `agent list`/`team list` group session rows under `this conversation` | tool tests; transcript examples |
| AC14 | TUI, desktop session picker and mobile sheet agree: same rows, same scope badge; the desktop new-chat picker stays global-only | TUI tests + desktop entity route + mobile sheet contract |

### 5.1 Manager acceptance checks (approval 2026-10-19)

| ID | Manager check | Maps to |
|---|---|---|
| M1 | A parent can create and use a single-use role/agent/team without permanent picker registration | AC1, AC2, AC3, AC10 |
| M2 | Prompting and resource recommendations guide reuse vs ephemeral vs durable, with niche and recurring examples | AC12; §3.7 pilot |
| M3 | Active temporary work stays inspectable, with owner and lifetime clear | AC13; §3.5 receipts/list grouping |
| M4 | Completion, failure and resume preserve results/provenance without unused picker entries or deleting durable definitions | AC3, AC4, AC5, AC10, AC11 |
| M5 | Durable creation or promotion has clear intended reuse | AC6, AC7, AC8, AC9; §3.4 grant |
| M6 | No cleanup or rewrite of existing user definitions | §3.5 invariant; AC11 |
| M7 | Bounded implementation with review/QA/design/UX gates | §4 (one PR per slice, gates named) |

## 6. Test strategy

- **Unit** (fast, isolated): composite overlay/precedence/id dispatch/name-shadow/lock;
  write routing; payload `scope`; promote preflight/rollback; fork copy; `agent_dir` hints.
- **Tool tests**: `tests/unit/tools/test_agent_tool.py`, `test_team_tool.py`,
  `test_tool_docs.py`; negative cases first (refusals), then receipts.
- **Session tests**: `tests/unit/session/test_role_subagents.py`,
  `test_team_agent_precedence.py`, `test_launch_subagent.py`; new scoped-registry tests;
  `AttachedSession` viewer parity (the historical `/team` regression on the viewer path).
- **Server**: `tests/unit/server/test_desktop_profiles.py` (global unchanged; session
  detail/promote), command-entities, feature flags.
- **TUI**: picker rows and new-chat exclusion.
- **i18n**: `scripts/i18n/generate.py --check`; all locales complete.
- **Isolation** per `AGENTS.md`: every run with its own `ISO=$(mktemp -d)` and
  `env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" …`; targeted
  commands only; CI supplies the full pass.
- **Worktree venv** (Phase 2 prerequisite; never symlink a parent venv):

  ```sh
  cd ~/wt-loh9-roles
  uv venv --python 3.12 .venv
  uv pip install -e ".[all,dev]" --python .venv/bin/python
  .venv/bin/python -c "import local_operator; print(local_operator.__file__)"  # must be the worktree
  ```

  Targeted inner loop examples:

  ```sh
  ISO=$(mktemp -d)
  env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
    .venv/bin/python -m pytest tests/unit/tools/test_agent_tool.py tests/unit/tools/test_team_tool.py \
      tests/unit/test_agent_profiles.py tests/unit/test_teams.py -q
  ```
- **Prove a test can fail**: for each behavioural test, flip the scoping/route and confirm
  it reddens before accepting it.
- **Real path**: the S3 exit scenario runs a real `Session`/runtime and a real fork/delete,
  not mocks; the promotion e2e reads through a second registry in a second session.

## 7. i18n, design/UX, parity

- **i18n**: new server wire keys (name-shadow refusal, promotion refusal/`replace` warning)
  added to `local_operator/i18n/catalogues/{en,es,fr,hi,ru,vi}/wire.errors.desktop.json`,
  generated functions via `scripts/i18n/generate.py`; no hand-editing of generated modules.
  Tool prose is model-facing and stays English (repo convention).
- **Design/UX**: scope badge wording (`this conversation` preferred over "ephemeral"); one
  action **Use in all conversations** on session rows; confirmation copy for `replace`;
  receipts must say what happened without jargon. No second picker and no modal mode.
- **Parity**: desktop and mobile render the additive field; the mobile sheet and TUI use the
  same rows the runtime serves. The desktop new-chat picker stays global by design and must
  not be changed to include session rows.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Viewer/cold-session path misses the scope (the `/team` regression precedent) | S1 wires `AttachedSession` explicitly; parity test on both implementations |
| Duplicate names / silent shadowing | refuse session-create over a global name; session-first resolution; scope in payloads |
| Team manager/member resolution | one resolver; composite passed to `attach_team` and `lookup_team`; promote expands members |
| Fork loses roles | explicit `roles/` copy in S3, test |
| Mesh move loses roles | documented device-locality; picker degrades to "not available"; later slice may carry the tree |
| Global readers leak rows | rows are not in the global roots; regression test points a global registry at config root |
| Old clients | additive JSON + new feature key; no existing field changes |
| Extra registry cost per session | lazy session store; `initialize=False`; no directory until first write |
| `AgentRegistry` constructor side effects in session dirs | new `initialize`/`migrate` flags, tested |
| Concurrent subagent creates | composite in-process lock around create/update/delete |
| Schema/tool-prose budget | one `scope` field + one sentence on the wire; detail in `tool://` docs |
| Promotion overwrites a user row | `replace` requires explicit user instruction + confirmation + receipt naming the row |
| Agent self-authorizes a global write | the tool has no path to global without a host-recorded grant; refusal is tested with no grant and with a forged/expired one |
| Fork creates a global row or diverges silently | fork copy is directory-only, never writes the global roots; independence is tested both directions |
| Mesh move silently loses roles | default refusal + `--drop-session-roles` + device-level stash whose path the receipt names; tested (stash holds the definitions after the source is gone) |
| Scope grows unbounded in a long session | `agent list` shows the count; optional soft cap (e.g. 50 rows) with a clear refusal — decide with the tracker owner |

## 9. Harness / remote-node notes (cloud-node-1)

Recorded for the manager → Aida; none of these were worked around in product code.

1. **MCP servers unavailable**: `notion`, `linear`, `minerva-qa`, `cloudflare` — tools gone,
   reason `damians-MacBook-Pro: /mcp login <server>`. Consequently the LOH-9 tracker row
   could not be read.
2. **Notion page unfetchable without the MCP**: `GET https://app.notion.com/p/3f9f0efb92668151bab2e125d6010d3c`
   returns `200` with 20,069 bytes of the Notion SPA shell only — no content, no readable
   text (auth/JS-gated).
3. **Mesh relay down**: `network status` → `relay: not running`; `network doctor` →
   `unhealthy: endpoint: not probed: no relay is running on this device`;
   `network peers` → `unreachable damians-MacBook-Pro  the link was refused`. A `send` to
   the manager session returned: `No answer from this device's network relay in 85s`.
4. **Per-command memory budget**: the default `bash` process-group budget (~0.1 GB) killed
   `git worktree add` mid-checkout, leaving a worktree locked `initializing`; recovery
   required `git worktree remove -f -f` and a retry with `memory_mb=800`. Worth reviewing as
   a remote-node default for repository operations.
5. **`python` is not on PATH** on cloud-node-1; only `~/lp-2520/.venv/bin/python` (3.12.14)
   exists. Any script must use the venv or absolute path.

## 10. Open questions for the tracker owner / manager

Resolved by the 2026-10-19 approval: retention is conversation-lifetime (not deleted at
turn/task end); promotion is copy-not-move; team promotion copies session members or refuses
(no partial writes); mesh move does not carry role definitions in v1, with a clear notice and
a follow-up.

Still open:
1. Exact LOH-9 tracker acceptance-check text (unreadable here) — §5 and §5.1 are the working
   set; reconcile wording when the row is available.
2. Any soft cap on session-scoped rows per conversation (the design leaves it uncapped and
   reports the count in `agent list`).

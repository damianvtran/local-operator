---
name: projects
description: "Track multi-session and parallel workstreams: create projects, link sessions, report honest progress, read the aggregated view of runtimes, subagents and todos."
---

# Projects

A project is a **tracked workstream**: one row in the operator's config root (`<config_dir>/projects/<id>.json`) that names a stream of work, links the sessions driving it, and carries an honest progress snippet.

It is deliberately **not** an agent, **not** a schedule, and **not** a second team. In particular it is *not* a team's `project` brief (`teams/<id>/project.md`), which is the product or domain a team instance is responsible for. Same word, different things: a team's project is *what the team owns*; a project is *a workstream being tracked*.

## When to make one

Make one when the work:

- spans **more than one session** (a feature with a coding session and a review session), or
- runs as **parallel streams** the operator is coordinating, or
- is something a manager coordinates and must report on over days.

Do **not** make one for a one-turn task, for anything already fully described by a todo list, or for a session's own scratch work. A project nobody updates is worse than no project: it makes the view lie.

## The `project` tool, op by op

- `op='list'` — one compact row per project (status, estimate, target, milestone count, session count, progress age, description). Start here.
- `op='create' name='…'` — creates the project and **links the calling session automatically**; say so when you report it. Optional on create: `description`, `status`, `tags` (each `[a-z0-9][a-z0-9_-]{0,23}`), `start_date`/`target_date` (ISO `YYYY-MM-DD`), `estimate` (+ `estimate_unit`: `points`|`days`), `milestones`.
- `op='update' name='…'` — merges only the fields you pass; anything omitted is left alone. Fields: `description`, `status` (`active`|`paused`|`done`|`archived`), `progress`, `tags` (replace-set), dates, `estimate`/`estimate_unit`, `milestones` (full-list replace).
- `op='link'` / `op='unlink' name='…' [session_id=…]` — attach or detach a session; without `session_id` they act on the calling session. The link lives only in the project row (cap 64; `link` refuses past it and names `unlink`).
- `op='milestone' name='…' milestone='beta cut'` — add-or-update by name: `milestone_target_date='2026-10-01'` sets a date, `milestone_completed=true|false` marks or clears completion, `remove=true` deletes the entry (renaming is remove + add). Milestone names are unique per project, case-insensitively.
- `project_delete name='…'` — permanent removal, and it asks for write approval. Deleting a project never touches session directories.

Set `status='done'` and `completed_at` is stamped to today unless you pass one; pass `completed_at=''` to clear it, and moving a status away from `done` leaves the date untouched. `target_date` before `start_date` is refused when both are set.

## Writing progress honestly

`progress` is what the views and the operator read to see where work stands. It is **tool-authored only** — never derived from activity, never guessed.

- Write it when the state **materially changes**: one dated line ("2026-09-26 dashboard cutover done; API parity on staging"), not a transcript and not a restatement of the todo list.
- Never claim progress you have not verified from a primary source. If you cannot tell, say what you did and leave the record alone.
- Re-sending the **identical** line on a stale record is meaningful: it refreshes the timestamp without changing the text ("still true, re-dated"). On a fresh record the same call is a no-op and writes nothing — so there is no reason to send it every turn.
- Emptying `progress` (`progress=''`) records "no progress recorded" and clears the freshness pair with it.

## The `@project:<name>` reference

A project can be named directly in a prompt as `@project:<name>`. Resolution is project-first, with a path fallback: a token that names a project expands **once, at submit**, to a snapshot block — name, status, description, the progress snippet with its age and reporter, and a linked-session rollup ("2 linked — 1 live"). A token that resolves to neither a project nor a path is left exactly as typed.

Two consequences worth knowing: the block is a **snapshot**, so it is as fresh as the message it rode in on, and editing the project afterwards does not change what the model already read.

## Views and the record

Two things read the same row: the terminal's `/project` surface and the desktop **Projects** tab (list / board / timeline, plus a detail page with milestones and linked sessions). Both render from one composition of "what is each linked session doing" — runtime state, subagent counts, todo counts.

What is **stored** versus **derived** matters when you report:

- **Stored:** `status`, `progress` + its freshness pair, `start_date`/`target_date`/`completed_at`, `estimate` + unit, tags, the milestone list with each milestone's dates, the linked session ids.
- **Derived at render:** milestone status (`completed` when `completed_at` is set, else `overdue` when its target date has passed, else `upcoming`); overdue-ness itself; progress staleness; each linked session's runtime state. Nothing derived is stored, so a derived badge can never drift from the date it contradicts.
- `null` means **unknown**, never zero: a session with no subagent roster file and no persisted todo snapshot reports `null` for those, not `0`.

## The completion check

Once a session is linked to a project, the harness watches for a specific failure: work moving while the record stays stale. When a turn **has done work** (at least one tool call) and yields while a linked, `active` project's progress is older than 30 minutes (or missing), the harness injects one reminder naming the stale projects. It is injected, not shown to the user.

It fires at most once per turn, only after a worked turn, only for stale records, and never for `paused`/`done`/`archived` projects. The exits it offers are the honest ones: update the record, change the status, re-send the line to refresh it, or `unlink` the session if it no longer belongs to the project. A reminder that repeats after you have acted is a bug, not a hint.

## Housekeeping

- **Stale links are marked, never auto-removed.** A linked session whose directory is gone renders as `missing`; clear it with `op='unlink'` when you see it.
- **Archive finished work** with `status='archived'` (or `'done'`, which also records the completion date). The views keep archived rows out of the way when they can.
- **Milestones**: keep them few and dated; complete them as they land rather than at the end, and remove ones that no longer describe the plan.
- The link cap is 64 per project and the milestone cap is 20. Both refusals name the remedy.

## Surfaces

- Agents write through the `project` tool; its `list`/`show` results are the reading surface.
- The operator reads and acts via `/project` (terminal and desktop) and the desktop **Projects** tab; the desktop gates on the `projects` capability.
- A slash form that survives on every surface is the reserved vocabulary `list | show | new | delete | link | unlink` — milestone editing is tool/API/UI work, not a slash verb.

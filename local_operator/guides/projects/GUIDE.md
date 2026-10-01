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

- `op='list'` — one compact row per project (status, owner/team when set, estimate, target, milestone count, session count, progress age, description). Start here.
- `op='create' name='…'` — creates the project and **links the calling session automatically** — as a *working* link, unless the caller is the chief of staff's session, whose link is recorded as **filed by** instead (provenance: it never counts as a working session, never satisfies liveness, and never earns a completion-check nudge). Say which happened when you report it. **Create with a `title` and a `description`** (the expected default, not optional-only): `title` is a short human-readable name shown first on listings and headers; `description` is markdown prose describing the workstream — paragraphs, headings, lists and code are fine (2000 characters max; rendered, not dumped). Name the rest when they are known: `owner`/`team` (who owns / which team manages the stream), `status`, `tags` (each `[a-z0-9][a-z0-9_-]{0,23}`), `start_date`/`target_date` (ISO `YYYY-MM-DD`), `estimate` (+ `estimate_unit`: `points`|`days`), `milestones`.
- `op='update' name='…'` — merges only the fields you pass; anything omitted is left alone. Fields: `description`, `title`, `owner`/`team`, `status` (see the lifecycle below), `progress`, `tags` (replace-set), dates, `estimate`/`estimate_unit`. Passing `milestones` here REPLACES the whole list and is **refused** unless you also pass `replace_milestones=true`; to change one milestone, use `op='milestone'` instead — it upserts by name and leaves every sibling alone.
- `op='link'` / `op='unlink' name='…' [session_id=…]` — attach or detach a session; without `session_id` they act on the calling session. The link lives only in the project row (cap 64; `link` refuses past it and names `unlink`).
- `op='milestone' name='…' milestone='beta cut'` — add-or-update by name: `milestone_target_date='2026-10-01'` sets a date, `milestone_completed=true|false` marks or clears completion, `remove=true` deletes the entry (renaming is remove + add). This is the vocabulary for changing ONE milestone: siblings are never touched. Milestone names are unique per project, case-insensitively.
- `project_delete name='…'` — permanent removal, and it asks for write approval. Deleting a project never touches session directories.

Set `status='done'` and `completed_at` is stamped to today unless you pass one; pass `completed_at=''` to clear it, and moving a status away from `done` leaves the date untouched. `target_date` before `start_date` is refused when both are set. `owner` and `team` are short free-text labels (trimmed, at most 80 characters); pass `''` to clear one back to unknown — absent means **unknown**, never a placeholder.

## The status lifecycle

`status` is where a tracked stream stands, and moving it is part of keeping the record true — set it at `create` and move it on **material changes**, not on a cadence:

| status | what it means | move here when |
| --- | --- | --- |
| `planning` | RFC / research — the shape is still being decided | the work is scoped but not being built yet |
| `active` | implementation is the work | research lands and building starts |
| `qa` | review / QA / design / copy cycles | implementation is complete and the rounds begin |
| `validation` | deployed and being validated, observation and fix-forward included | the change shipped somewhere real and is proving itself |
| `paused` | a side-state: deliberately not being worked | the stream is set aside without closing |
| `done` | fully validated — requirements closed, todos addressed, milestones complete | the rounds are clean and nothing is owed |
| `archived` | the finished drawer | the row stops being worth showing |

Setting `status='done'` is a CLAIM, so it is gated: every milestone must be complete, or you pass `force_done=true` to close with them open — the refusal names the incomplete milestones. A plan with no milestones closes normally (there is nothing to prove). `done` also stamps `completed_at` to today unless you pass one.

Only `planning`/`active`/`qa`/`validation` can read **stale** — those are the in-flight statuses the completion check watches (below). `paused`, `done` and `archived` are deliberate statements that the record is settled; they never read stale.

`title` is the human-readable display name: listings and headers show it first, with the key `name` as secondary (recoverable in the detail), and every surface falls back to `name` when it is absent. `name` remains THE key — addressing (`/project show <name>`, `@project:<name>`, file names) never changes. `description` is **markdown** (multiple paragraphs, headings, lists and code are allowed; 2000 characters max) and is rendered, not dumped raw; set both when you create; `title` and `description` are the agent-editable fields for scope and change updates.

## Writing progress honestly

`progress` is what the views and the operator read to see where work stands. It is **tool-authored only** — never derived from activity, never guessed.

- Write it when the state **materially changes**: one dated line ("2026-09-26 dashboard cutover done; API parity on staging"), not a transcript and not a restatement of the todo list.
- Never claim progress you have not verified from a primary source. If you cannot tell, say what you did and leave the record alone.
- **Refresh ≠ update.** When you checked and the line still describes reality, record the check: `op='refresh'` (or re-sending the identical line) writes a dated assertion and nothing else — the line stays, its clock stays, so a stale badge stays stale until a NEW line lands; on a fresh record it is a no-op, and there is no reason to send one every turn. Everything else is an **update**: a reworded or near-identical re-send is a NEW line — it appends and moves the clock — so never re-send an annotated copy of the same line.
- Emptying `progress` (`progress=''`) records "no progress recorded" and clears the freshness pair with it.
- Every NEW line is also appended to the project's **history** (`updates`, newest last): an append-only, timestamped log bounded at 500 entries (oldest drop first). The latest line is what surfaces as the summary. Re-sending an identical line or clearing appends nothing — only a new line writes history.
- Attach files to a new line (`attach=['frame.png', 'out.log']`): **screenshots for visual progress, evidence for everything else** (test output, measured numbers, data tables) — so readers can see how things are actually going. Files are copied into the project store (≤ 10 per update, ≤ 5 MB each; never referenced from their original location, so the evidence survives a reaped scratch dir), and `show` lists each with its stored path. Stored files are reclaimed with what carries them: deleting a project removes its attachment folder, and an entry evicted by the five-hundred cap takes its files with it.

## The `@project:<name>` reference

A project can be named directly in a prompt as `@project:<name>`. Resolution is project-first, with a path fallback: a token that names a project expands **once, at submit**, to a snapshot block — name, status, description, the progress snippet with its age and reporter (plus a refresh note when the record was checked recently), and a linked-session rollup ("2 working — 1 live · filed by 1"). A token that resolves to neither a project nor a path is left exactly as typed.

Two consequences worth knowing: the block is a **snapshot**, so it is as fresh as the message it rode in on, and editing the project afterwards does not change what the model already read.

## Views and the record

Today the row is read through the `project` tool, the terminal's `/project` listing, and the terminal's full-page projects view. The view has three canvases — list / board / timeline — reachable across every project (`/project board`, `/project timeline`) or scoped to one conversation: **nameless `/project show`** opens the calling session's own projects on the board (their rows/cards carry the `◆` marker; no links falls back to the all-projects board). The board keeps its three columns; `planning`/`qa`/`validation` cards ride the in-flight column while their own chip (and the counts line) keeps the exact status. Reading requires no session at all — a bare terminal opens the same store — and `↵` on the selected row/card/bar opens that project's live conversation (`/resume` machinery) or says honestly which sessions exist and what starts one. `↑`/`↓` move that selection in every canvas (the view follows it); `←`/`→` pan the canvas, and `r` refreshes the page without moving the reader's row. The desktop **Projects** tab (the same three views, plus a detail page with milestones and linked sessions) lands in a later UI slice. All of them render from one composition of "what is each linked session doing" — runtime state, subagent counts, todo counts.

What is **stored** versus **derived** matters when you report:

- **Stored:** `status`, `title`, `progress` + its freshness pair (`progress_updated_at`/`progress_reported_by`) and the refresh assertion pair (`progress_refreshed_at`/`progress_refreshed_by` — set by a check, cleared by the next new line), `start_date`/`target_date`/`completed_at`, `estimate` + unit, tags, `owner`/`team`, the append-only `updates` history (per entry: text, timestamp, reporter, attachment metadata), the milestone list with each milestone's dates, the linked session ids (working in `sessions`, filed-by in `coordination_sessions`).
- **Derived at render:** milestone status (`completed` when `completed_at` is set, else `overdue` when its target date has passed, else `upcoming`); overdue-ness itself; progress staleness (read only for the in-flight statuses — `planning`/`active`/`qa`/`validation`; settled `paused`/`done`/`archived` records never read stale); each linked session's runtime state. Nothing derived is stored, so a derived badge can never drift from the date it contradicts.
- `null` means **unknown**, never zero: a session with no subagent roster file and no persisted todo snapshot reports `null` for those, not `0`.

## The completion check

Once a session is linked to a project, the harness watches for a specific failure: work moving while the record stays stale. When a turn **has done work** (at least one tool call) and yields while a linked, in-flight project's progress is older than four hours (or missing), the harness injects one reminder naming the stale projects. It is injected, not shown to the user.

It fires at most once per turn, only after a worked turn, only for stale records, and never for `paused`/`done`/`archived` projects (the in-flight statuses — `planning`/`active`/`qa`/`validation` — are the ones it watches). It also never fires through a **filing**: a coordination link ("filed by") never earns a nudge; only a working link does. The exits it offers are the honest ones: update the record, change the status, record a check with `op='refresh'` (which quiets the reminder for one window without resetting the staleness clock — the badge never clears), or `unlink` the session if it no longer belongs to the project. A reminder that repeats after you have acted is a bug, not a hint.

## Housekeeping

- **Stale links are marked, never auto-removed.** A linked session whose directory is gone renders as `missing`; clear it with `op='unlink'` when you see it.
- **Archive finished work** with `status='archived'` (or `'done'`, which also records the completion date). The views keep archived rows out of the way when they can.
- **Milestones**: keep them few and dated; complete them as they land rather than at the end, and remove ones that no longer describe the plan.
- The link cap is 64 per project and the milestone cap is 20. Both refusals name the remedy.

## Surfaces

- Agents write through the `project` tool; its `list`/`show` results are the reading surface (`show` prints the latest five history entries by default — `history=<n>` prints more, `0` omits the section).
- The operator's `/project` runs every reserved verb plus the two page entries: bare/`list` prints the listing (with the page-entry footer), `show <name>` opens the full-page view on that project, **nameless `show`** opens the calling session's own projects on the **board** (one link is also the highlight, several are the marked set, none falls back to the all-projects board — never a refusal), **`board`**/**`timeline`** open the all-projects canvases, `new <name>` creates and auto-links this session, `delete <name>` rehearses and `delete <name> yes` removes, and `link`/`unlink <name>` move the session link. Reading needs no session (the page opens from this machine's store); the session-bound writes are `new`/`link`/`unlink` (they act on the calling session), while `delete` is store-local. The desktop **Projects** tab lands in a later slice; the routed runtime answers the same receipts as the terminal (one shared runner).
- Milestone editing is tool/API/UI work, not a slash verb: the reserved vocabulary is `list | show | new | delete | link | unlink | board | timeline`, and `/v1/desktop/projects` is the API a renderer grows into.
- The desktop gates on the `projects` capability (`"projects": 1`), so an older backend hides the tab once it ships.

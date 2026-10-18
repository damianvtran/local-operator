# TUI / desktop UI parity audit and slice plan

Status: audit + plan for LOH-19 (2026-10-18). No product code is changed by this
document.

Refs read:

- desktop UI: `local-operator-ui` `origin/main` `76dda0304` (2026-10-18)
- TUI: `local-operator` `origin/main` `8d91a9b9e2` (2026-10-18)

The two trees are read at those refs. A capability is only marked present where
the surface exists in the code, not merely where the backend supports it.

## How to read the matrix

- ✅ — first-class surface in that frontend.
- ◐ — partial: the capability exists but a material sub-capability is missing,
  read-only, or reachable only by asking the agent or another host.
- ✗ — no surface in that frontend.
- ➕ — the TUI has a surface the desktop UI does not (recorded for completeness;
  not a parity gap to close).
- N/A — not meaningful for that frontend.

Evidence is a file or module path, not a promise. Where the two frontends share
the same `slash_commands.SLASH_COMMANDS` registry, the command name is the
evidence for both.

## 1. Navigation

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 1.1 | Conversation list with pinned/active/previous sections, pin/unpin, archive/unarchive/delete | ✅ | ✅ | UI `features/chat/components/chat-sidebar.tsx`, `chat-sections.ts`; TUI `widgets/session_sidebar.py`, `/archive`, `/unarchive`, `/delete` |
| 1.2 | Conversation search over titles **and message bodies** | ✅ | ✅ | UI `shared/api/local-operator/session-search.ts`, `palette-search.ts`; TUI `widgets/session_picker.py` + `session/search_index.py` |
| 1.4 | Global command palette / action search (`Cmd+K`, `Ctrl+K`, `Ctrl+P`) | ✅ | ◐ | UI `features/command-palette/*`; TUI inherits Textual's `ctrl+p` CommandPalette (`app.py:4578-4583`, `docs/design/keymap.md` §E.4) but registers no app-specific providers, so only framework system actions appear |
| 1.5 | Agents and teams visible in the navigation column | ✅ | ◐ | UI `chat-sidebar.tsx` (Agents + Teams sections); TUI `/agent` and `/team` listings, no tree |
| 1.6 | Notification / attention routing back to a conversation | ✅ | ◐ | UI attention store and sidebar badges; TUI `/notifications`, notifier app and status marks |

## 2. Composer and turn controls

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 2.1 | Multiline composer, slash autocomplete, `@path` references | ✅ | ✅ | UI `shared/components/composer/message-input.tsx`, `at-mention-overlay.tsx`; TUI `widgets/editor.py`, `autocomplete.py` |
| 2.2 | Image attachment staging (paste, drop, picker) | ✅ | ✅ | UI `file-attachment.tsx`, `image-attachment.tsx`, attach button; TUI `widgets/editor.py` attachment staging |
| 2.3 | Rich attachment / artifact preview in the app | ✅ | ◐ | UI canvas viewers for text, code, markdown, HTML, PDF, spreadsheet, image, audio, video; TUI renders images inline (`widgets/image_block.py`) and opens other files externally (`attachments.py`, `supplement_block.py`) |
| 2.4 | Voice dictation in the composer | ✅ | ✗ | UI `features/chat/components/audio-recording-indicator.tsx`, mic in `message-input.tsx`; no TUI capture surface |
| 2.5 | Model, reasoning effort and fast-mode switching | ✅ | ✅ | UI header cluster/pickers; TUI `/model`, `/effort`, `/fast`, shift+tab |
| 2.6 | Tool approval mode and per-call approval cards | ✅ | ✅ | UI approval cards; TUI `widgets/approval.py`, `/approvals` |
| 2.7 | `ask` tool answers and queued asks | ✅ | ✅ | UI `ask-answer.ts`, `ask-queue.ts`; TUI `widgets/ask_picker.py`, `widgets/ask_queue.py` |
| 2.8 | Stop, interrupt and steer a running turn | ✅ | ✅ | UI turn controls; TUI `ctrl+c`, `/stop`, `/highlights` |
| 2.9 | `btw` side conversation (off the record) | ✅ | ✅ | UI `aside-panel.tsx`; TUI `/btw`, `f8` |
| 2.10 | Fork, compact, goal and loop | ✅ | ✅ | UI pickers; TUI `/fork`, `/compact`, `/goal`, `/loop` |
| 2.11 | Plan / todo list | ✅ | ✅ | UI run panel + plan chip; TUI `widgets/todo_panel.py` |
| 2.12 | Running subagents and background jobs roster | ✅ | ✅ | UI run panel; TUI `widgets/subagent_panel.py`, `widgets/subagent_view.py`, status counters |
| 2.13 | Armed wakes and monitors, with next-due/health | ✅ | ✅ | UI composer chips + run panel; TUI `widgets/wake_panel.py` |
| 2.14 | Create, edit and cancel wakes/schedules | ✅ | ◐ | UI `features/schedules/*` (page + form + cancel confirm); TUI has no direct surface — the agent's `wake` tool can create/list/cancel, and the panel only displays |
| 2.15 | Code-request ledger: PR/MR state, CI, agent/design/QA/UX review rounds, freshness | ✅ | ◐ | UI `features/code-review/*` right pane; TUI has no surface — the agent-facing `code_requests` tool can list/show rows, but there is no page |

## 3. Skills, memory and harness import

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 3.1 | List discoverable skills and read a skill body | ✅ | ✅ | UI `SkillsPicker` in `features/chat/pickers/destination-pickers.tsx`; TUI `/skills` block in `app.py` |
| 3.2 | Inspect, search, correct and forget scoped memory | ✗ | ➕ | TUI `/memory` (`local_operator/memory_cli.py`, `app.py::_cmd_memory`); no desktop destination in the registry |
| 3.3 | Import conversations and memory from Claude Code / Codex / omp | ✅ | ✅ | UI `features/settings/components/import-settings-view.tsx`; TUI `/import` + `widgets/harness_import_view.py` |

## 4. Agent hub, agents and teams

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 4.1 | List local agents, attach one to the session, switch its class | ✅ | ✅ | UI agents page/options; TUI `/agent`, `/agent class` |
| 4.2 | Create/edit a reusable agent profile (role/specialist, tools, effort, delegate) | ✅ | ◐ | UI `features/agents/components/agents-page.tsx` profile editor; TUI only lists/attaches — authoring is via the `agent` tool or asking the agent |
| 4.3 | List teams, draw a team org chart, send a request to a team manager | ✅ | ✅ | UI team detail/chart + chat sidebar; TUI `/team`, `/team chart`, `/team <name> <request>` |
| 4.4 | Create/edit a team (roster, manager, brief, model/effort overrides) | ✅ | ◐ | UI team editor (`features/agents/components/team-detail.tsx`); TUI delegates to the `team` tool/agent request |
| 4.5 | Browse/search/sort/filter the public agent hub | ✅ | ✗ | UI `features/agent-hub/agent-hub-page.tsx`; TUI has no marketplace surface (`/hub` is Hub links, a different feature) |
| 4.6 | Agent details, comments, likes, favourites and download | ✅ | ✗ | UI `agent-details-page.tsx`, `comments-section.tsx`, like/favourite/download hooks; no TUI equivalent |
| 4.7 | Publish/upload/delist an agent | ✅ | ✗ | UI `upload-agent-dialog.tsx`, `use-delist-agent-mutation.ts`; no TUI equivalent |
| 4.8 | Hub skills library and skill publication | ✅ | ✗ | UI `components/hub-skills-library.tsx`, `hub-skill-publication.tsx`; TUI `/skills` lists local skills only |
| 4.9 | Public teams list and team pull | ✅ | ✗ | UI `components/public-teams-library.tsx`, `use-team-pull-mutation.ts`; no TUI equivalent |
| 4.10 | Hub link policy per link (subscribe, auto-push) | ✅ | ✅ | UI `features/agents/components/hub-links-panel.tsx`; TUI `/hub` (`widgets/hub_links_panel.py`) |
| 4.11 | Agent-messages (team bridge) consent/onboarding | ✅ | ◐ | UI `features/agent-messages/*`; TUI has the settings section and bridge cards, no onboarding dialog |

## 5. Settings

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 5.1 | One settings page with search, scope tags and reset-to-default | ✅ | ✅ | UI `features/settings/components/settings-page.tsx`; TUI `widgets/settings_view.py` |
| 5.2 | Backend config catalogue (model, routing, failover, compaction, tools, hooks, …) | ✅ | ✅ | UI backend settings section over the same registry; TUI `settings_io.SECTIONS` |
| 5.3 | Theme switching with live preview | ✅ | ✅ | UI theme selector; TUI `/theme` picker |
| 5.4 | Interface language selection and translated catalogues | ✅ | ✅ | UI language dialog + `i18n/`; TUI `/language` + `local_operator/i18n/` |
| 5.5 | Remappable keyboard shortcuts | ✅ | ✅ | UI keyboard-shortcuts section; TUI `keymap` settings section + `local_operator.keymap` |
| 5.6 | Provider listing and sign-in | ✅ | ✅ | UI providers feature; TUI `/provider`, `/login` |
| 5.7 | Stored accounts and credentials | ✅ | ✅ | UI Radient account/credential manifest; TUI `/accounts`, `/credential` |
| 5.8 | Radient account, plan and billing | ✅ | ◐ | UI `radient-account-section.tsx`; TUI `/mobile` covers phone-access billing, not the account panel |
| 5.9 | MCP servers, OAuth grants, add/remove | ✅ | ✅ | UI integrations/MCP management; TUI `/mcp` |
| 5.10 | Application updates | ✅ | ✅ | UI app-updates section; TUI `/update` |
| 5.11 | User profile (display name, email) | ✅ | ✗ | UI general settings; no TUI settings row |
| 5.12 | Usage and analytics | ✅ | ✅ | UI settings usage chart; TUI `/usage`, `/analytics` |

## 6. Files, attachments and the canvas

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 6.1 | File callouts from an answer, open in the OS | ✅ | ✅ | UI file rows; TUI `widgets/supplement_block.py`, `attachments.py` |
| 6.2 | Inline document viewer (text, code, markdown, HTML, PDF, spreadsheet, image, audio, video) | ✅ | ◐ | UI `features/chat/components/canvas/*`; TUI renders images inline (`image_block.py`) and the agent's `read` can print text, but there is no viewer |
| 6.3 | Create/edit files with an inline AI editor | ✅ | ◐ | UI `canvas/inline-edit.tsx`, `create-file-dialog.tsx`, WYSIWYG markdown; TUI has no editor surface — the agent's `edit`/`write` tools are the only path |
| 6.4 | Drag-and-drop arbitrary files into the composer | ✅ | ◐ | UI attach-file button/drop; TUI stages images, other terminal drops arrive as paths/text |

## 7. Built-in browser

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 7.1 | Human browser page: tabs, URL bar, new tab, find, history | ✅ | ✗ | UI `features/browser/*`; TUI has no browser surface |
| 7.2 | Agent-browser consent and approval UI | ✅ | ✗ | UI consent bar/approvals dock; TUI has no host chrome, so approvals must happen in the desktop app, extension or cmux |
| 7.3 | Download/upload rows and WebAuthn prompts | ✅ | ✗ | UI browser file-transfer and WebAuthn components; no TUI equivalent |

## 8. Notifications and wakes

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 8.1 | Completion notifications and unread marks | ✅ | ✅ | UI native notifications + attention; TUI notifier app, `/notifications`, terminal marks |
| 8.2 | Armed wake/monitor list with due time and delivery health | ✅ | ✅ | UI composer chips/run panel; TUI `wake_panel.py` |
| 8.3 | Create/edit/cancel a wake from the app | ✅ | ◐ | UI schedules page/form; TUI direct surface absent, but the agent's `wake` tool can create/list/cancel |
| 8.4 | Schedules page grouped by conversation, including legacy agent schedules | ✅ | ✗ | UI `features/schedules/components/schedules-page.tsx`; TUI has no page |

## 9. Ask flows

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 9.1 | Answer an `ask` with options/free text; queue and minimise asks | ✅ | ✅ | UI ask cards/queue; TUI `ask_picker.py`, `ask_queue.py` |
| 9.2 | Timed-out / lapsed ask visibility | ✅ | ✅ | Both read `asks.lapsed_visible_for`; TUI settings row + ask surfaces |

## 10. Projects

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 10.1 | List, board, timeline and detail views | ✅ | ✅ | UI projects page; TUI `widgets/projects_view.py` |
| 10.2 | Toggle a milestone complete | ✅ | ✅ | UI milestone block; TUI `DetailMilestoneRow` |
| 10.3 | Add or remove a milestone | ✅ | ✅ | UI `project-milestones.tsx` add/remove; TUI detail `a` add, `x` remove behind y/n, `↵` toggle (`projects_detail.py`, `projects_milestone.py`) |
| 10.4 | Create a project | ✅ | ✅ | UI form dialog; TUI `projects_form.py` |
| 10.5 | Edit a project's fields | ✅ | ✅ | UI `features/projects/components/project-editors.tsx`; TUI detail `e` opens the prefilled form and writes only changed fields (`projects_edit.py`, `projects_form.py`) |
| 10.6 | Links, todos, updates, attachments, quick send, start session | ✅ | ✅ | UI project detail components; TUI `projects_detail.py`, `projects_send.py`, `projects_start.py` |

## 11. Mesh

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 11.1 | List devices, roles, endpoints, links and actions | ✅ | ✅ | UI mesh list/detail; TUI `widgets/network_panel.py` |
| 11.2 | Graph canvas with drag/drop session moves | ✅ | ✗ | UI `mesh-canvas.tsx`, `mesh-drag.ts`, `mesh-drop.ts`; a graph is not a terminal-shaped surface |

## 12. Everything else

| # | Capability | UI | TUI | Evidence |
|---|---|---|---|---|
| 12.1 | Console pane mirroring an external terminal agent | ✅ | N/A | UI `features/console/*`; the TUI is already a terminal, and its agent has `bash`/`console` |
| 12.2 | Onboarding tour | ✅ | ◐ | UI `features/onboarding/*`; TUI welcome/setup states (`widgets/welcome.py`) |
| 12.3 | Mini composer window | ✅ | N/A | UI `mini-view/*`; no meaning for a terminal app |
| 12.4 | GUI installer | ✅ | N/A | UI `features/installer/*`; terminal install path is pip/uv |

## 13. Summary

Counting the 71 rows above:

- Full parity: 38
- TUI partial: 17
- TUI missing: 12
- TUI-only: 1
- Not applicable to a terminal: 3

The missing set is smaller than it first looks because a capability reachable
only through an agent tool is **partial**, not missing. The remaining missing
set clusters into three kinds:

1. **Bounded direct surfaces** — the schedules page grouped by conversation
   (8.4) and the user profile (5.11). The underlying wake and profile data
   exists; only the page is absent.
2. **Large desktop surfaces that need a terminal design and a client for a
   backend API** — the agent hub marketplace, skills library and public teams
   (4.5–4.9).
3. **Genuinely cross-cutting** — the built-in browser pane and its
   consent/approval chrome (7.1–7.3); voice dictation (2.4); the mesh graph
   (11.2, deliberately not planned for a terminal).

The larger, more actionable group is the 17 **partial** rows: the capability
exists only by asking the agent to use a tool, or as a read-only/stripped-down
surface. The first six slices below target that group.

## 14. Prioritised slice plan

Order is by user value per unit of delivery risk. The first six are proposed as
serial TUI slices; P6–P9 are proposals, not implementation work, until their
design is accepted.

| Order | Slice | Gap closed | Why this rank | Risk |
|---|---|---|---|---|
| P0 | `/review` — code-request ledger page | 2.15 | No direct TUI surface; backend ledger and `service.view_rows` already exist; read-only; high value to a developer operator | Low: read-only, no protocol change |
| P1 | `/schedule` — wake list/create/edit/cancel | 2.14, 8.3 | No direct TUI surface; `build_wake_schedule`/`build_wake_edit` and the scheduler already exist; direct UI removes the "ask the agent" workaround | Medium: mutation, form validation, date/duration parsing |
| P2 | Project parity completion: edit project, add/remove milestone | 10.3, 10.5 | Closes the last project gaps; store APIs already exist; TUI already has the page and form patterns | Low–medium |
| P3 | Agent and team authoring forms | 4.2, 4.4 | Removes the agent-mediated workaround; desktop has the full field set to mirror; TUI already has `/agent` and `/team` surfaces | Medium: forms and validation |
| P4 | Attachment/file preview | 2.3, 6.2 | TUI currently punts everything but images to the OS; text/code/markdown preview is terminal-shaped and useful | Medium: file-type routing, bounded rendering |
| P5 | Command palette providers | 1.4 | One keyboard surface over destinations, settings rows, conversations and agents; the inherited Textual palette already opens, but has nothing app-specific to show | Medium: new interaction, not a new subsystem |

### Proposals (do not build without acceptance)

- **P6 — Agent hub browser/install/publish** (4.5–4.9). High value but requires
  Radient auth, the public catalogue API, download/publish paths and a terminal
  information architecture for a marketplace. Propose as its own design.
- **P7 — Canvas** (6.2–6.3, 2.3). A document viewer/editor is a subsystem:
  file-type routing, freshness, editing, AI inline edit. Propose a read-only
  first phase if wanted.
- **P8 — Built-in browser pane** (7.1–7.3). A browser in a terminal needs a
  different answer from an Electron webview; the consent/approval half is the
  part that matters for agent safety. Propose.
- **P9 — Voice dictation** (2.4). Audio capture and transcription are
  cross-cutting and platform-sensitive; propose after the terminal-shaped work.

### Checked, not a feature

Multi-conversation tabs are **not** a desktop feature to mirror: `ChatTabs` is a
dev-only Chat/Raw toggle for one conversation, and production renders the chat
view with no strip. The TUI's `/resume` and sidebar remain the switch. This was
checked and removed from the matrix and the proposal list.

### Not planned

Mesh graph (11.2) is a visual tool a terminal cannot improve on; console, mini
view and installer are desktop/terminal-form factors, not parity gaps.

## 15. First slice: `/review`

### Goal

Give the TUI the same code-request ledger the desktop's Code review pane shows,
for the current session, read-only.

### Scope

- New `/review` command opens a full-page view in the settings/import page
  family.
- Load `ledger.read_index(config_dir, session_id)` and
  `service.view_rows(config_dir, rows)` in a worker.
- Group rows by the shared state vocabulary; show for each: project + number,
  relation tag, state, CI, lead review lane (round + state), comments count,
  updated age, stale/freshness, and the URL.
- `enter` opens the forge URL with the existing quiet browser opener.
- `r` triggers `service.refresh_session` in a worker and repaints; opening the
  page also asks for a refresh.
- `esc` closes; empty/error/loading states are first-class.
- Complete i18n catalogue for `tui.review` in all six shipped locales.

### Non-goals

- Mutating a PR/MR (merge, comment, retry CI).
- A per-lane detail editor; the row shows the lane clause, and `enter` opens the
  forge page.
- The agent tool (`code_requests`) is unchanged.

### Tests and evidence

- Unit tests over the pure row/group derivation, with fixtures for: fresh row,
  stale row, link-only row, CI failure, all four lanes, unknown lane, missing
  head SHA, empty ledger, index read error.
- Widget tests for the four states and key handling.
- Rendered evidence: real `OperatorApp` at 100x30 and 60x24, before (current
  transcript) and after (`/review` populated, empty, error), SVG exported and
  actually viewed.
- Independent code review and QA; design/UX round because it is a new visible
  surface.

## 16. Delivery notes

- No version bump; no release cut. The merged PRs are handed to the current
  release-window owner.
- No human review requests; the agent-review, QA, design and UX rounds are the
  gate.
- One writer per area: the first slice touches `tui/widgets/`,
  `tui/app.py`, `slash_commands.py`, `i18n/`, tests and docs only.

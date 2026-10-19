# Operator-authored teams in the TUI (LOH-19 P3b)

## Grammar and scope

`/team new` opens a full-page create form. `/team edit <name>` edits an
installed team (aliases resolve to its canonical name). Team-name picker rows
come first, followed by `chart`, `new`, `edit`; `edit ` re-offers installed
names, while `new ` offers none. Reserved words in the first slot are verbs.
A single leading `=` escapes a literal name: `/team =new <message>` talks to
team `new`, `/team edit =chart` edits team `chart`. Parsing removes exactly one
`=` (`==foo` looks up `=foo`); the current registry does not permit installing
names containing `=`. Receipts that suggest a talk command escape reserved
names, and tests execute the command actually printed.

This is **basic roster/manager/brief authoring**, not full desktop parity.
Per-member model-tier/effort override **editing remains desktop-only**. A quiet
hint makes that boundary visible. Existing overrides are preserved, not reset.

## Fields and keyboard contract

Name is editable only at creation. Label, description, manager (default
`manager`), members, aliases, instructions and project brief are editable.
Members use the shared `teams.parse_members` grammar: comma-separated `coder`,
`coder:2`, `team:pod`, or `team:pod:2`; aliases are comma-separated names.
Instructions and project are TextAreas. Instructions are required on creation;
manager is always required. Core-supported clears are explicit: description,
instructions, project, aliases, roster, and label reset to the derived default.
An empty roster means the manager alone.

The page reuses `FormFieldBlock`, project-form styles, and the P3a full-page
mount/restore pattern. Tab/Shift+Tab traverse fields; Ctrl+S saves; Esc and
Ctrl+C close a clean form or ask before discarding a dirty draft. The question
is pinned outside the scroll body; Y discards, N/Enter/Esc keep editing.
Repeated Ctrl+C leaves the question in control and cannot arm process exit.
Sibling modes and session reboot reclaim the page and restore composer focus.

## Changed-only writes and concurrency

`team_authoring.py` is a widget-free draft/diff/validation model. The app passes
`TeamEditFields` to the same `TeamRegistry.create_team` / `update_team` core as
the tool and desktop. Unchanged fields are omitted, including members and
aliases. An unchanged save displays a receipt and calls no writer.

An unchanged roster is not serialized or reconstructed at save. A deliberately
changed roster carries `tier` and `effort` onto surviving `(kind, role)` slots,
including nested teams and changed counts; removed slots disappear and newly
added slots inherit defaults. Pure and real-app tests use an actual temporary
registry, reload persisted rows, and compare member JSON before/after a
description-only save and a roster replacement.

Edit captures `definition_revision("team", id, team.model_dump(mode="json"))`
from a fully loaded definition. `require_fresh_revision` runs under
`update_team`'s writer-lock `precondition`, against the re-read definition.
An intervening writer produces an honest reopen refusal, retaining the draft.

Validation, registry uniqueness/alias refusals and save errors remain inline;
invalid fields regain focus. Read-only/follower sessions refuse local authoring
with team-specific copy; they never route an authoring verb to another owner.
Copy lives in `tui.team_authoring` (English source plus independently translated
and audited es/fr/hi/ru/vi catalogues). Independent audit evidence is recorded
in the i18n ledger and PR, not inferred from mechanical key parity.

## Verification boundary

Affected tests cover pure changed-only plans and core clears, real-app
create/edit, no-op, duplicate name/alias, validation, stale CAS, read-only,
reserved-word routing, printed receipt commands, dirty-discard keys, session
reboot and sibling-mode replacement. Rendered evidence uses the production
`OperatorApp` stylesheet at 100×30 and 60×24 with isolated HOME/config;
Textual compositor frames are not a claim of native terminal photography.
Independent review/QA/design/UX and CI remain PR gates.

Implementation captures include all seven states in dark and light at both
sizes (32 frames including consecutive empty/settled controls). The 60×24 page
is 58×17 cells with its English discard question pinned at y=16; the 100×30
page is 98×23 with the question at y=22. Only the fields body scrolls vertically;
no screen or horizontal scrollbar appeared. All four first/settled empty pairs
are pixel-identical. SVG, PNG, geometry, serialized override fixtures and the
capture command are retained in the PR evidence rather than committed assets.

# TUI agent-profile authoring (LOH-19 P3a)

`/agent new` opens a full-page profile form, initially a role. `/agent edit
<name>` opens the same form for an **installed** role or specialist. A packaged
starter must be installed first; editing never silently creates a shadow row.
The existing transcript is hidden, not replaced, and the composer is read-only
until the page closes, following the projects form's mount/restore pattern.

## Grammar and discovery

`new`, `edit`, and `class` are reserved only in the first argument position.
`/agent =new` attaches the profile named `new`; `/agent ==foo` attaches `=foo`.
The same single-leading-`=` escape applies to edit's name argument. Existing
attach, attach-and-message, detach, and class-switch semantics remain intact.
Profile names precede verbs in autocomplete, keeping bare Tab an attach action.
`/agent edit ` offers profile names; `/agent new ` offers no argument rows.
The listing footer advertises both authoring verbs.

## Fields and keyboard contract

Name and kind are chosen at create time and read-only on edit. Name has no
whitespace or path separators. Label is display-only; description explains
**when to use** the profile; instructions are standing guidance. Instructions
are required. Role tools are comma-separated names (empty means all tools),
effort is `inherit` or a currently configured backend tier, and delegate is a
yes/no choice. Both kinds have reactive/proactive class; the form explains the
unsolicited-work consequence of proactive.

Specialists have no tools, effort pin, or delegate policy. Those controls are
hidden and disabled in specialist mode and omitted from its save payload.
Switching back to role restores the unfinished role controls. The pure model
also refuses specialist payloads carrying role policy, rather than accepting
and silently dropping it.

Tab/Shift+Tab traverse fields; left/right cycle choices. Instructions use a
TextArea where Tab moves focus, not inserts indentation. Ctrl+S saves. Esc
closes a clean page; a dirty page pins an inline y/n question **outside** the
scroll body. Y discards; N, Enter, or Esc returns to editing. Tab cannot escape
the confirmation into the underlying form.

## Persistence, refusals, and no-op

`agent_authoring.py` owns the pure draft, tools parser, validation and changed-only
diff. The page owns no registry. The app passes `AgentParams` to the shared
`write_profile` core used by the tool and desktop, which also writes instructions
through `registry.set_agent_system_prompt`. Create sends the complete field set
for the chosen kind. Edit sends only changed fields, preserving omitted values.
An unchanged save is a visible no-op receipt and makes no persistence call.

Opening edit captures `registry.definition_revision_fields(agent_id)` and its
definition revision. The same CAS token accompanies the shared writer's update;
a stale definition refuses with an honest reopen instruction. Registry duplicate
names and other backend errors remain visible inline; failed saves never close
the form or destroy the draft. Read-only/follower sessions refuse before opening
or writing, using the projects page's existing ownership guard and notice.

The shared writer currently treats empty description/instructions as omission
on update. **P3a does not change those cross-client core semantics.** Clearing a
nonempty edit description is refused inline as unsupported; blank instructions
are refused as required. A new description may be empty. Supporting text-field
clears needs its own shared-core proposal and regression evidence.

## Evidence and boundary

Real `OperatorApp.run_test` frames (the shipped stylesheet, isolated HOME/config)
cover empty create, validation error, prefilled edit, specialist mode, and dirty
discard at 100×30 and 60×24. SVG/PNG/geometry artifacts are linked in the PR, not
committed. At 60×24 the page is 58×17, fields scroll within it, and the English
confirm is pinned at y=16; screen virtual size is 58×22 (no screen overflow).
Opening focus keeps the first field's label visible. Translated questions may
wrap without truncating their y/n instructions. All colors use existing theme
resolvers and projects form classes.

**P3b boundary:** team creation/editing (parity row 4.4), roster and manager
editing, and team model/effort overrides are deliberately unchanged. There is no
new team persistence or command surface in this slice.

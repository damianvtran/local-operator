# Mobile relay redesign: borderless, calmer, every feature one gesture away

Design direction for the phone relay web client (`local_operator/mobile/web`).
Author: designer (lopdev team). Audit base: `origin/main` = **`bc7dbea188`**;
the BEFORE frames were rendered from that tree's built bundle.

Status: **historical implementation record for the borderless relay redesign.**
Current changes follow [Mobile relay: design and UX guidelines](mobile-relay-guidelines.md).
This record retains the original rationale and evidence; measurements below describe
its audited base unless explicitly labelled AFTER, not a current-screen guarantee.

## The ask, verbatim intent

> The mobile relay is very cluttered and crowded. Apply the same design
> principles as the Local Operator desktop UI — borderless chrome, no harsh
> borders, faded scrolls, etc. Optimize layout and the amount of information on
> screen. Everything must stay accessible, ideally via best-practice mobile UX
> (mobile drawers, etc.). Modern, slick, better usability.

## The one-sentence direction

*The phone shows the conversation and the one thing that needs you; every
other control moves one gesture away — a navigation drawer for destinations, a
bottom sheet for actions on a row or a session — and every surface separates
by the ground ladder instead of a 1 px box.*

Read with: `~/local-operator-ui/docs/branding.md` (§2 "The two lines", "Panels:
borderless, edge-to-edge, fading at the top", the composer paragraph, §4–§6),
`~/local-operator-ui/docs/design/scrollbars-fade.md` (§10 resting floor),
`docs/mobile.md` § "The web UI", `docs/I18N.md`.

---

## 1. What is there now, measured

Evidence is attached to the PR: the historical BEFORE set holds 109 frames,
with metrics, token, contrast and ladder outputs (§13). Every
number in this section comes from `before/metrics.json`, written by
`scripts/mobile_showcase_shot.py`, or from the named command. All counts are
**localOperatorDark at 390x844** unless the row says otherwise. Light-theme
counts are identical for structure: the same DOM with a different paint.

### 1.1 Per-screen clutter

| screen | bordered elements | controls in view | chrome px / content px | text lines per row |
|---|---|---|---|---|
| list, empty | 9 | 8 | 125 / 719 | — |
| list, populated | **13** | **19** | 125 / 719 (85 %) | **2–3** (6 rows × 51 px, 4 × 70 px) |
| list, grouped by tags | 14 | 16 | 125 / 719 | 2–3 |
| session, populated (sheet closed) | 10 | 17 | **407 / 437 (52 %)** | — |
| session, populated @360x780 | 10 | 16 | 407 / **373 (48 %)** | — |
| session, todos open | 10 | 13 | **607 / 237 (28 %)** | — |
| session, subagents open | 10 | 17 | 555 / 289 (34 %) | — |
| session, approval card | 8 | 15 | 544 / 300 (36 %) | — |
| sheet: directory | 16 | 23 | — | — |

The "bordered" column on the list names its elements. The settings button, the
search field, "group by tags", "mark all 2 read" and its band, two "working"
spinners, the footer's top rule, and the footer's **five** outlined buttons
(`new session`, `past`, `library`, `projects`, `◐`) are all 1 px
`border-control` or `border-hairline` boxes.

### 1.2 Where the session column's height goes

These numbers come from `audit/bands.json`, a DOM geometry probe on the same
fixture at 390x844:

| band | px |
|---|---|
| header (‹ · title · ⋮) | 53 |
| spend/context row (`30.9%/200k … $2.47`) | 26 |
| **transcript (the only scroller)** | **437** |
| working line | 32 |
| tasks disclosure (collapsed) | 45 |
| subagents disclosure (collapsed) | 45 |
| ask dock | 44 + 4 |
| composer: cwd chip row | 44 + 6 |
| composer: input row | 44 + 6 |
| composer: model/effort chip row | 44 + 8 |

On a 360x780 phone the transcript gets **373 px**, 48 % of the screen. Of the
screen, **3 × 44 px rows belong to the composer** and **2 × 45 px to collapsed
panel headers.**

### 1.3 Gate findings (design-qa `dom_audit` checks, injected in-page)

| check | frame | measured |
|---|---|---|
| target-size **FAIL** | session-approval | "remember this choice" checkbox **16x16 px** (< 24 WCAG 2.5.8) |
| low-contrast FAIL (*exempt*) | sheet-directory | `change` label **2.52:1** at 13 px. The button is **disabled** (empty path field, `directory-sheet.tsx:225`) and renders `ink-disabled`, which SC 1.4.3 exempts and branding §2 caps from above. Not a defect, but the disabled box keeps its `border-control` edge, so it reads as live. The redesign drops the edge (§7.2), which leaves the state carried by ink alone. |
| input-label **FAIL** | list (search), every session (composer) | search `input` and composer `textarea` have no label |
| target-size advisory | every session | composer textarea **184x24 px** (the 44 px floor is met by its wrapper, not the field) |
| overlap | list | list rows paint **under** the opaque footer (e.g. "GPT-5.5" 93 % under `◐`). This is expected for a scroller under chrome, but there is **no fade**, so the cut is mid-glyph. |
| overlap | session | transcript rows paint under the header and status row (`30.9%/200k` 92 % over a tool row), again with **no fade** |
| control-name | every open sheet | the scrim button. A **false positive**: it is `aria-hidden` by design (`ui/sheet.tsx`, UX round 1 U7). Recorded here so nobody "fixes" it. |

`token_audit.py src/components src/screens` found one raw hex in a
component, `#997711` (`components/supplement-block.tsx:409`), and one raw size,
`16px` (`composer.tsx:406`, the focus-zoom floor, deliberate). A class census
counted **`border border-control` 50×** and **`border-hairline` 49×** across
**25** component and screen files. The census is in `audit/class_census.txt`.

### 1.4 What the frames show that the numbers cannot

- **Every resting control is drawn as an outlined box.** On the empty list the
  only non-chrome content is two lines of centred text. The screen's loudest
  marks are nine rectangles around words (`before/list-empty-390x844-*`).
- **Rows are a terminal printout.** Each row prints a mono absolute path and a
  mono model label, `/synthetic/home/projects/local-ope…  Claude Sonne…`,
  truncated on both ends at 390 px. Those two columns are the least-used
  facts on the row, and they cost it a second or third line.
- **The session view stacks five bands between the transcript and the
  keyboard**: working line, two disclosures, the ask dock and a three-row
  composer. When a panel opens, the conversation drops to 237 px.
- **The asks sheet opens on arrival** (`session-arrival-*`). That is the
  intended open-by-default behaviour, and it is the right call. It is listed
  here because the redesign must keep it.
- **Offline is honest, and inconsistent.** The list shows a muted
  `reconnecting…` word in the header. The session shows a full-width strip,
  `reconnecting — showing the last synced view`, overlapping the first
  transcript row. The two screens use two vocabularies and two weights for one
  state (`list-offline-*`, `session-offline-*`).
- **Theme and wide view live behind `◐`**, and **settings is a text button**
  in the header. Neither is something you use daily, and both sit on the
  primary screen.

---

## 2. Principles, applied to a phone

1. **One job per screen.** The list answers "what needs me, and what's
   running?" The session answers "what is it doing, and what do I say next?"
   Anything that does not serve that job leaves the screen.
2. **Ground, not lines** (branding §2). Separation comes from `canvas` →
   `surface` → `elevated`, with `sunken` for wells, plus spacing. Use
   `border-control` only where the boundary carries information that no ground
   can (§7.1). `hairline` never separates chrome from content: a fade does
   that.
3. **Chrome dissolves into content.** Every scroller under chrome gets 24 px
   fades, top and bottom, and is scroll-driven so they are zero at rest (§7.6).
4. **The thumb owns the bottom third.** The primary action and the composer
   are there. Destinations move to the drawer. Nothing that needs a reach
   lives at the top except navigation (‹, ☰) and the overflow ⋮.
5. **Progressive disclosure, never hiding.** Every current feature keeps a
   path of at most two taps, and the most frequent ones keep one (§4). A
   long-press is always shadowed by a visible path, because long-press is
   undiscoverable on its own.
6. **The accent is spent three times per screen** (branding "Accent"): the
   primary action, the active state and the focus ring. "New" marks, ask
   counts and the selected chip share the *active state* budget.

---

## 3. Information architecture

### 3.1 Navigation model

```
┌ App bar ─────────────────────────────┐
│ ☰  Conversations               ⌕    │   ☰ opens the NAVIGATION DRAWER
├──────────────────────────────────────┤
│ [All] [Needs you · 3] [Unread · 2] [Work] …   ← filter chips (scroll-x)
│  ── fade ──                          │
│  Pinned                              │
│  ○ Inbox triage                 now  │
│    Personal · working                │
│  …                                   │
│  ── fade ──                    [+ New]  ← extended FAB, thumb zone
└──────────────────────────────────────┘
```

- **Navigation drawer** (start edge; `☰` in the app bar; edge-swipe from the
  start edge as an accelerator, never the only path). It holds
  **destinations**: Conversations (current), Questions (all asks), a
  **Notifications slot reserved** (§3.4), Past sessions, Projects and
  Library. Below a spacing gap it holds **preferences**: Usage, Appearance
  (theme + wide view) and Settings (account routing). The footer line reads
  "connected · this phone", the link state in words.
- **Bottom sheets** hold actions on one object: a row's actions, a session's
  ⋮ menu, the model, effort, directory, slash, asks, approvals gate, tags and
  usage. All of them keep the existing `Sheet` contract: in-flow, no portal,
  visual-viewport anchored (`ui/sheet.tsx`, `lib/column.ts`).
- **Row action sheet**: long-press a row, swipe it toward the start edge to
  reveal Pin / More, or use the row's ⋮ when a keyboard has focus
  (Shift+F10 / ContextMenu are already wired). §5.3 covers this.
- **Progressive disclosure in the session**: one **status tray**, a single
  row of pills above the composer, replaces the three stacked bands
  (tasks, subagents, ask dock). §6.4.

### 3.2 Reachability: every current control and its new home

"Taps" counts from the screen the control lives on today. ≤ 2 is required, and
a starred row marks a feature that stays one tap away.

| # | today (control → where) | new home | taps today → new |
|---|---|---|---|
| 1 | `settings` text button, list header | Drawer → Settings | 1 → 2 |
| 2 | search field, list top | App-bar `⌕` expands an inline search field in place of the title; a **pull-down reveal** at list top is an accelerator | 1 → 1 ★ |
| 3 | `group by tags` toggle | Filter-chip row: **tag chips** (one per category) *are* the grouping. The "Group" choice moves to Drawer → Appearance → "Group conversations by tag" (pref persists in `session-group-pref`) | 1 → 1 ★ (filter) / 2 (pref) |
| 4 | `mark all N read` band | Filter chip `Unread · N`; when that chip is selected, its header row carries "Mark all read" (ghost button). Also in the drawer's Conversations ⋮ | 1 → 2 |
| 5 | `new session` footer button | **Extended FAB** `+ New`, bottom-end, collapses to a 56 px round `+` while scrolling down | 1 → 1 ★ |
| 6 | `past` footer button | Drawer → Past sessions | 1 → 2 |
| 7 | `library` footer button (capability-gated) | Drawer → Library (same gate: **hidden, never disabled**) | 1 → 2 |
| 8 | `projects` footer button | Drawer → Projects | 1 → 2 |
| 9 | `◐` theme footer button (+ wide view inside) | Drawer → Appearance (theme list + Wide view + Group by tag) | 1 → 2 |
| 10 | long-press row → Pin / Tags… | unchanged gesture; also **swipe-to-reveal** and row ⋮ on focus | 1 → 1 ★ |
| 11 | `reconnecting…` chip, list header | A quiet **link pip** on the app bar's ☰ (warning dot) plus one meta line under the title "reconnecting · showing last sync"; §8 | — |
| 12 | session `‹` back | unchanged | 1 ★ |
| 13 | session `⋮` → Pin / Wide view / Approvals… | unchanged sheet, plus **Rename** and **Tags** if present, plus **Spend & context** (row 16) | 1 ★ |
| 14 | `needs you` header button (gate blocking) | moves into the **status tray** as the first pill ("Approve") and keeps opening the gate sheet | 1 → 1 ★ |
| 15 | session-status row (`30.9%/200k … $2.47`, expands to breakdown) | Folded into the **composer meta line** (`31% · $2.47`), tap opens the existing breakdown as a sheet; the 26 px band is removed | 1 → 1 ★ |
| 16 | working line (`⣾ measuring … 1m36s`) | **App-bar subtitle** under the title ("working · 1m 36s"), and the in-transcript live row stays as today | 0 ★ |
| 17 | tasks disclosure | Status tray pill `Tasks 2/6` → bottom sheet (same `TodosPanel` content) | 1 → 1 ★ |
| 18 | subagents disclosure | Status tray pill `Agents 1/3 · 1 failed` → bottom sheet (same `AgentRoster`); row tap → agent view as today | 1 → 1 ★ |
| 19 | ask dock (`? 1 question waiting · …`) | Status tray **first** pill `? 1 question` (accent), opens the asks sheet; auto-open on arrival unchanged | 1 → 1 ★ |
| 20 | pending approval/ask card (inline, above composer) | unchanged position and behaviour; it is the one thing that needs a decision. Restyled per §6.5 | 0 ★ |
| 21 | composer cwd chip (`/synthetic/home/…`) | Composer **meta line** start: folder glyph + last path segment (`local-operator`); tap → directory sheet | 1 → 1 ★ |
| 22 | composer model + effort chips | Composer meta line: `Opus 5.5 · high` → model sheet; effort from inside the model sheet's header row, *and* `/effort` | 1 → 1 ★ (model) / 2 (effort) |
| 23 | attach (paperclip) | Composer action row, start | 1 ★ |
| 24 | dictation (mic) | Composer action row, next to attach | 1 ★ |
| 25 | send / steer / stop / retry | Composer action row, end (morphing, unchanged semantics) | 1 ★ |
| 26 | slash sheet (type `/`) | unchanged trigger; restyled sheet | 1 ★ |
| 27 | `/usage` → usage sheet | unchanged, **plus** Drawer → Usage | 1 → 1 ★ |
| 28 | ended state + `resume` | **Header status dot** (grey) beside the subtitle's `Ended`; tapping the title block opens a sheet that explains it and carries a filled `Resume` button; also `Resume` in the ⋮ menu while ended. **No banner above the composer** (#3008, superseding the banner) | 1 → 1 ★ |
| 29 | degraded / not answering | Same header status dot (yellow) beside `Not answering`; its sheet says what a send does now. No action, no band (#3008) | 1 |
| 30 | agent view `Agents` (open agent navigation) | unchanged (header action) | 1 ★ |
| 31 | asks sheet "dismiss…", "decline", answer | unchanged (sheet internals restyle only) | ★ |
| 32 | projects sheet / tags sheet / metadata sheet internals | restyle only (§7) | — |
| 33 | wide view | Drawer → Appearance, and session ⋮ (unchanged) | 2 / 1 |
| 34 | Questions across all conversations (asks sheet, today reachable only from a session) | **New**: Drawer → Questions opens the existing `AsksSheet` with no `currentSessionId` | n/a → 2 |

No control is deleted. Rows 1, 6, 7, 8 and 9 go from 1 tap to 2, and that is
the trade the ask makes: five destinations used weekly leave the screen used
hourly.

### 3.3 What leaves the list row

Today a row carries the name, a status glyph, tag line, mono cwd, mono model
label, and badges. The new row (§5.2) drops **the model label** (it moves to
the session view, where it can be changed) and **the absolute path**. The path
is replaced by the *last segment* only, so `~/projects/site` reads as
`site`, and only when the row has no tag to show. Both stay one tap away: the
session view shows the model in its composer meta line, and past sessions plus
search still match on the full cwd.

### 3.4 Reserved: the Notifications drawer entry

Another team adds **Notifications** after this merges. The drawer reserves the
slot **third, after Questions and before Past sessions** (destinations that
need you come first). The slot carries:

- a nav item component that accepts `{icon, label, count?, href}`, the same
  shape as Questions;
- an i18n key `relay.shell.drawer.notifications`, **not added by this PR**.
  The other team owns it, so the entry renders nothing until its feature
  flag/capability exists (`capabilities.notifications?.available`, which
  follows the **hidden, never disabled** rule the library entry uses);
- the drawer's item list as a **data array** (`DRAWER_ITEMS`), so the
  addition is one entry and no layout change.

The mock (`mock/mock-drawer-*`) draws the slot with a "reserved" tag purely
for review. **The coder must not ship a visible placeholder.**

---

## 4. Screen: session list

### 4.1 Layout (390x844)

| band | height | content |
|---|---|---|
| status-bar safe area | `env(safe-area-inset-top)` | — |
| app bar | 52 | `☰` (44) · title "Conversations" (`text-heading`) · `⌕` (44) |
| filter chips | 40 | horizontally scrolling chips, fade at end edge |
| **list scroller** | rest | sections + rows, top/bottom fades |
| FAB | 56, floating | bottom-end, 16 px from edges + safe-area, over the list |

That is a chrome cost of **92 px** (52 + 40) against **125 px today**, with no
footer at all (61 px today). The list scroller runs to the bottom edge, and
the last row clears the FAB through `padding-block-end: 88px` on the scroller.

### 4.2 Row anatomy: one line plus meta

```
 lead  Name ............................ trailing
  12   meta · meta · meta
```

- **Lead (12 px column)** keeps the existing slot vocabulary, with no new
  glyph: spinner (streaming), `bg-danger` dot (approval), `bg-accent` dot
  (unread/new), the two-dot pair (delegating), and an empty slot when idle.
- **Name**: `text-body` at 15 px for row titles. Use `ink` when unread or
  live and `ink-muted` when read and idle. Truncate on one line.
- **Trailing**, one item, picked by priority: `approve` (danger) > `N asks`
  (accent) > `new` (accent) > `not answering`/`ended` (ink-dim) > relative
  time (ink-dim, `text-meta`). The ★ pin glyph is dropped from the row
  because the **Pinned** section header carries it.
- **Meta line** (`text-meta`, `ink-dim`, sans): `primary tag · state ·
  children/todos · cwd last segment`, truncating from the end. **No mono, no
  model label.** Numbers stay in tabular figures (`font-variant-numeric:
  tabular-nums`), not mono.
- Row: `min-height: 56px`, `padding-block: 8px`, `padding-inline: 16px`, gap
  12. No border and no fill at rest. `:active` gives `bg-surface` and
  `:focus-visible` gets the ring.
- Measured target: **56 px** today's 51/70 mix becomes one height, so the
  list stops jittering by row type. At 390x844 that is **12 rows** in the
  scroller (752 / 56) against **~10** today.

### 4.3 Section headers

`Pinned`, `Active`, `Previous` (tag names when grouped). Style: `text-meta`,
weight 500, `ink-dim`, `padding: 16px 16px 4px`, no rule. They are **sticky**
inside the scroller on `bg-canvas`, and the top fade passes over them: a
sticky header on canvas with a fade behind it reads as part of the ground. The
copy changes from "Active Sessions" to "Active" (§11).

### 4.4 Search and filter

- `⌕` in the app bar swaps the title for a search field. The field uses the
  `elevated` ground, has no border, and sits at `radius-md` (10), 44 px tall.
  It gets **autofocus** and a ✕ to close. A field is a control, so its radius
  would normally be 6 px. A *standalone* full-width search pill takes 10,
  which is branding's "panel-scale" rule for an object read as one block, and
  that matches the desktop sidebar search.
- **Filter chips**: `All` · `Needs you · N` (approval + asks) · `Unread · N` ·
  one chip per tag category (only when `capabilities.session_tags`). Only one
  chip is selected at a time. `All` is the default.
  - Chip: 32 px visual and **44 px hit area** via a `::before` inset, so the
    40 px row holds 44 px targets that overlap only the row's own padding.
    `radius-full` (pill badge rule), `bg-surface`, `ink-muted`, no border.
    Selected: `bg-accent-wash` + `text-accent`, plus `aria-pressed`.
  - A tag chip both filters **and** groups: Today's "group by tags" result is
    "All, grouped". Keep it as an Appearance preference, and while the
    preference is on, `All` renders tag sections.
- While searching, the chips hide and the sections flatten, as today.

### 4.5 Primary action

An **extended FAB** `+ New` sits bottom-end, `bg-accent`/`text-on-accent`,
56 px tall, at `radius-lg` (14). A FAB is the one "object that leaves the
flow" on this screen, so it gets `shadow-overlay`. It shrinks to a round 56 px
`+` (aria-label unchanged) after 24 px of downward scroll and re-extends on
upward scroll, with a 180 ms width transition (§9). `starting…` replaces the
label while busy, and `aria-busy` stays. A start error lands as a **snackbar**
above the FAB (`role="alert"`, 4 s, swipe away), not as a line inside the
list.

### 4.6 States

- **Empty**: a centred block at 40 % height. It shows a mark glyph
  (`ink-dim`, 40 px), the headline "No conversations yet" (`text-heading`,
  `ink`), and the hint "Start one with **New**, or from Local Operator on your
  computer" (`text-body-sm`, `ink-muted`). The FAB stays and is the only
  accent on the screen. Chips are hidden when the list is empty.
- **Loading / first frame**: the app bar and chips render immediately. The
  list shows **three skeleton rows** (`bg-surface` bars at 60 % / 35 % width,
  `radius-sm`, no shimmer under reduced motion) in place of "connecting…". The
  layout of frame 1 equals the layout of the settled frame (docs/mobile.md
  "first frame is the final layout").
- **No matches**: one line, "No matching conversations", plus a ghost
  "Clear search" button.
- **Offline / reconnecting**: §8.

---

## 5. Row actions

### 5.1 Long-press (kept)

This is the existing 450 ms press with move-cancel and `suppressClick` logic,
unchanged (`session-list.tsx` `SessionCard`). It opens the **row action
sheet**.

### 5.2 Swipe

**Bounded implementation note:** swipe-to-reveal, drawer edge-swipe and pull-down
search are optional accelerators not shipped in this PR. Their accessible paths
are explicit: More on every row (plus the retained long-press and keyboard
context-menu), the labelled navigation button, and Search in the app bar. No
feature or destination depends on an undiscoverable gesture. The direction below
records the accelerator design, not a claim of implemented touch-gesture proof.


Swiping a row toward the **start** edge reveals two 72 px actions on the end
side: `Pin`/`Unpin` (`bg-surface`) and `More` (`bg-elevated`, which opens the
row action sheet). The swipe is RTL-aware: compute the direction from
`dir`. It **must not** conflict with vertical scroll, so it engages only after
16 px of horizontal travel with |dx| > 2·|dy|. It also must not fire during a
settle-freeze (`freezeSettles`), and it disables under reduced motion, where
long-press and the focus ⋮ remain. Pin from swipe uses the same optimistic
mark → confirmed move split (docs/mobile.md "The PRESS and the MOVE").

### 5.3 The row action sheet

Grabber, title = conversation name (one line). Rows are 48 px, each an icon
(20 px, `ink-muted`) plus a label (`text-body`, `ink`) plus an optional
trailing value (`ink-dim`):

`Pin to top` / `Unpin` · `Tags` (value: primary or "none") · `Mark as read`
(when unread) · `Open`. The refusal line stays in-flow with `role="alert"`,
per the existing contract.

---

## 6. Screen: session view

### 6.1 Layout (390x844, nothing pending)

| band | height | today |
|---|---|---|
| app bar: `‹` · title + subtitle · `⋮` | 56 | 53 + 26 status row + 32 working line = **111** |
| **transcript** | rest (**~588**) | **437** |
| status tray (pills) | 44 (only when any pill exists) | 45 + 45 + 48 = **138** |
| composer: field + action row | 96 + safe area | 158 |

The transcript gains **~150 px** at 390x844 with the same content on screen,
while the tray and the panels hold everything they held before.

### 6.2 App bar

- `‹` (44) · a two-line title block · `⋮` (44). Title: conversation name,
  `text-heading` 15/600, one line. Subtitle: `text-meta` `ink-dim`, carrying
  the **liveness** in words: `working · 1m 36s` (streaming, shimmer allowed
  on "working" only), `waiting for you` (pending), `idle`, `ended`, or
  `reconnecting · last synced 12 s ago` (§8).
- No bottom border. The transcript's top fade dissolves under it.
- `needs you` leaves the bar and becomes the tray's first pill.

### 6.3 Transcript

Unchanged content model (branding §7 trace hierarchy, `tool-row.tsx` one line
per action). Restyle only:

- **Tool rows lose their filled boxes.** Today each tool row is an
  `elevated`/`danger-wash` rounded block. They become **flat lines**: glyph
  (identity ink) · **verb in sans** (`ink-muted`) · object in mono
  (`ink-dim`) · trailing elapsed. A failed row keeps `danger` on its glyph
  and its elapsed, and **drops the wash**, because the glyph is the state.
  The disclosure (tap to expand) opens a `sunken` well (`radius-md`), which is
  the one place tool output gets a ground. This applies branding §4
  ("monospace is machine voice … the OBJECT of a trace line, not its verb").
  A coder **cannot** rename verbs without i18n: if `tool-row.tsx` derives the
  verb from `tool_name`, keep the tool name in sans for this PR and record the
  verb map as a follow-up (§12, O4).
- **User message**: `bg-surface`, `radius-md` (10), and the 2 px accent
  inline-start bar is **removed**. The ground plus end alignment already say
  "you". `max-inline-size: 85%`.
- The working row stays the last row while streaming.
- Fades top and bottom (§7.6).

### 6.4 Status tray

A single horizontally scrolling row of **pills**, 44 px tall, `padding-inline:
12px`, gap 8, placed between the transcript and the composer. It renders only
when it has at least one pill, so a quiet session gets the space back. Pill
order is by urgency, and each pill opens a bottom sheet:

| pill | when | look | opens |
|---|---|---|---|
| `Approve` | approval gate blocking | `bg-danger-wash` `text-danger` | gate sheet (today's `needs you`) |
| `? N question(s)` | asks outstanding | `bg-accent-wash` `text-accent` | asks sheet (today's dock) |
| `Tasks d/n` | todos exist | `bg-surface` `ink-muted` | todos sheet (today's `TodosPanel`, unchanged content) |
| `Agents r/n · k failed` | subagents exist | `bg-surface` `ink-muted`, the failed count in `danger` | agents sheet (today's roster, row → agent view) |

Pills are 32 px visual with a 44 px hit area, `radius-full`, no border, and
`text-body-sm`. **Why sheets and not inline disclosures?** The inline panels
are capped at a column fraction and can never be read at length on a phone,
and when open they halve the transcript (28 % content measured). A sheet
gives the list 85 % of the column for as long as the reader wants, and the
conversation is still there when it closes. Keep the "held shut while a
request is pending" rule: while a pending card shows, the tray stays, but its
non-urgent pills render `ink-dim` (colour, not opacity, per branding §6).

### 6.5 Pending card (ask / approval inline)

Keep the position (above the tray), the remount key, the budget rule and
`lo-fade-b`. Restyle:

- Card: `bg-elevated`, `radius-lg` (14), `shadow-overlay` **off**. It is in
  flow, so the ground step separates it. The accent 1 px border is **removed**
  and replaced by a 3 px `accent` inline-start bar, which is the "card in a
  STATE keeps its edge" rule from branding §5, expressed as a bar rather than
  a frame. Approval uses a `danger` bar.
- Options: borderless rows on `bg-surface`, `radius-sm`. Selected:
  `bg-accent-wash` + check glyph. The option's own `border-l-accent` is
  removed.
- Buttons: primary `send answer`/`approve` stays the filled accent. `deny` and
  `decline` become **tonal danger** (`bg-danger-wash` `text-danger`, no
  border).
- **Fix the 16x16 checkbox** (`remember this choice`). Wrap it in a 44 px
  label row: the whole row is the hit target, and the box draws at 20 px.

### 6.6 Composer

```
┌ elevated, radius-frame 16, no border at rest ─────────┐
│ Message…                                               │  field, grows to 6 lines
│ 📎  🎙   Opus 5.5 · high   31% · $2.47        [ ■ / ↑ ] │  action row 44
└────────────────────────────────────────────────────────┘
  ⌂ local-operator          ← only when cwd ≠ the session default? NO: see below
```

- **One object** with an `elevated` ground and `radius-frame` (16), the
  composer's radius alone (branding §5). It has **no border at rest** and an
  **accent outline only on `:focus-within` with `:focus-visible`**, as on
  the desktop. It sits 8 px from the column edges and above the
  safe-area inset.
- **Row 1**: the field. It keeps the 16 px font floor (focus-zoom) and
  `MAX_TEXTAREA_PX`.
- **Row 2** (action row, 44 px): attach and mic at the start (44 px ghost icon
  buttons, `ink-muted`). Then the **meta cluster**: model · effort (one
  button, opens the model sheet), then context % · spend (one button, opens
  the breakdown sheet), both `text-meta` `ink-dim` and **sans with tabular
  numerals**. Then the send/steer/stop/retry disc at the end (44 px round,
  accent fill, morph unchanged).
- **cwd** moves into the meta cluster as the **first** item: folder glyph plus
  the last path segment. On narrow widths the cluster truncates from the
  *middle* item outward: cwd stays, model shortens, spend drops last. The
  directory sheet still opens on tap. *(This replaces the separate 44 px cwd
  row.)*
- The reserved `ComposerFrame` must be rebuilt from the **same class
  constants** (`COMPOSER_*`) so the first frame stays the final layout. The
  existing test pins this.
- Dictation states (recording / transcribing) take over row 2 in place, as
  today.
- Pending echoes, retry and the retained-envelope line are unchanged, and
  render above the composer.
- 96 px total (+ safe area) against 158 px today. **3 → 2 rows.**

### 6.7 Session health: the header status dot (ended / not answering / offline)

**Superseded (#3008).** These were once full-width strips over the transcript,
then a banner slot above the composer. Both spent the thumb zone on a fact
that rarely changes and needs no action until the reader wants one, and the
ended banner sat flush on the composer card. Session health now lives in the
app bar (§6.2): one 8 px dot beside the subtitle word that already names the
state, and the title block is a 44 px control that opens the shared Sheet
with one explanation sentence. Nothing is stacked above the composer for
health. Priority stays: ended > not answering > link down.

- **Live** (green, `success`): "Running on your computer — messages reach it
  right away". A streaming turn is live: busy is not a health state.
- **Not answering** (yellow, `warning`): the relay's dial is down (`degraded`:
  a send is refused at once and the text stays in the box), or the owner's
  heartbeat and frames have both been quiet past `SESSION_HEALTH_WINDOW_S`
  (`heartbeat_stale`: a send can time out yet still arrive). No queueing is
  promised. No action.
- **Reconnecting** (yellow): the PHONE's own link is down (§8); the session
  may still be running.
- **Ended** (grey, `ink-dim`): history kept, and a send or `Resume` reopens it
  in the home folder. The sheet's filled accent `Resume` (44 px) is the one
  action, also offered in the ⋮ menu while ended. The composer stays enabled (a
  send also resumes) and its placeholder says so before the tap.

The rule, the server fields and the window constant are in
`local_operator/mobile/web/src/lib/session-health.ts`; the copy is the
`relay.health` namespace. A transition into any non-live state is announced
through one polite live region.

### 6.8 States

- **Loading / first frame**: the app bar renders with the title from the list
  row when navigated from the list (it is already in the store), and an empty
  subtitle otherwise. The transcript area shows centred "Connecting…" in
  `ink-dim`, and the composer frame is inert, as today. Measured today on a
  cold deep link at 390x844: **3 bordered elements, 2 controls**
  (`session-first-frame`), which is already calm, so keep it. The rig's
  first-frame shot races the projection's arrival. At 360x780 one run caught
  the reserved frame and another caught the settled view (12 vs 10 bordered),
  so judge the first frame from the frame itself, never from the count.
- **Empty conversation**: "No messages yet" plus "Send a message to get
  started", centred.
- **Error** (projection refused / 404): centred sentence plus a ghost "Back to
  conversations".

---

## 7. Component rules

### 7.1 Where a border is still allowed

1. **Nowhere at rest on chrome.** No app bar, footer, panel, card, row, chip,
   pill, sheet edge or field gets a rule.
2. `border-control` stays only on **checkbox and radio boxes** (the box *is*
   the control) and on a **segmented control group** (one shared edge).
3. A **state** may draw a 3 px inline-start bar (pending card, failed tool
   disclosure). Bars use `accent`, `danger` or `warning`, never `hairline`.
4. The focus ring is not a border. It is an `outline` (branding §6).

### 7.2 Buttons (`ui/button.tsx` variants)

| variant | rest | pressed | disabled | use |
|---|---|---|---|---|
| `primary` | `bg-accent` `text-on-accent` | `bg-accent-active` | `bg-sunken` `text-ink-disabled` | one per screen |
| `tonal` (**new**, replaces `outline`) | `bg-surface` `text-ink` | `bg-elevated` | `text-ink-disabled` | secondary actions |
| `ghost` (rename of `quiet`) | transparent, `text-ink-muted` | `bg-surface` | `text-ink-disabled` | tertiary, sheet rows |
| `danger` | `bg-danger-wash` `text-danger`, **no border** | `bg-elevated` `text-danger` (there is no darker-wash role) | `text-ink-disabled` | deny, decline, stop |
| `icon` | 44x44, transparent, glyph `ink-muted` | `bg-surface` | `ink-disabled` | app-bar, composer |

On a `surface` or `elevated` parent (sheets), `tonal` steps one rung up:
`bg-elevated` on surface, and `bg-surface` on elevated, which steps back
down. Pick whichever rung is further from the parent in that palette. Brand
light's `elevated−surface` is only 2.10 L\* (§10.2), so in sheets prefer
`bg-canvas` for the stepped-down field (O3). `outline` is
deleted after its callers migrate (engineering principle: migrate callers,
then delete).

### 7.3 Fields

`bg-elevated` on canvas (or `bg-surface` inside an elevated sheet), **no
border at rest**, `radius-sm` (6) for in-form fields and `radius-md` (10) for
standalone search. Placeholder is `ink-dim` (measured **4.61:1** dark /
**5.58:1** light on `elevated`). Focus uses the accent outline via
`:focus-visible`. Every field gets an accessible name: `aria-label` from the
placeholder's key for search, and "Message" for the composer. This fixes the
two `input-label` FAILs.

### 7.4 Sheets (`ui/sheet.tsx`)

- **Grabber**: 36x4 px `bg-control` at α 0.45 (the scrollbar resting floor's
  value, scrollbars-fade §10.1), `radius-full`, centred, 8 px from the top.
  It is decorative and `aria-hidden`.
- **Drag-to-dismiss** on the grabber and title row: dismiss past 30 % of the
  panel's height or above 0.5 px/ms of downward velocity, otherwise snap
  back over 180 ms. The ✕ stays as the accessible close. Under reduced motion
  the drag tracks the finger with no fling animation, and release is instant.
- Panel: `bg-elevated`, `rounded-t-lg` (14), `shadow-overlay`, and **no
  `border-t`** (today it has `border-t border-control`).
- Title row: `text-heading` (16/600, sentence case, `ink`), not the current
  tracked `text-meta` small caps. The title is the object being acted on.
- Content scroller: `lo-scroll` + top/bottom fades.
- Keep: in-flow (no portal), `columnBox()`/`columnCap(0.85)`, focus trap and
  return, scrim tap, the pointer-release swallow, and the safe-area padding.

### 7.5 Drawer (new `ui/drawer.tsx`)

- It anchors to the **inline-start** edge and is `min(320px, 85vw)` wide,
  sharing `Sheet`'s modal contract (inert background, focus trap, Escape,
  scrim, return focus). Extract the shared parts into `useModalLayer` rather
  than copying them.
- Ground `bg-surface`, full height, no border, `shadow-overlay`. Items are
  48 px with an icon (20) plus a label (`text-body`) plus an optional trailing
  count (`text-meta` `ink-dim`). The current item gets `bg-accent-wash`,
  `text-accent` and `aria-current="page"`. Groups are separated by a 16 px
  gap, **not a rule**.
- It opens over 240 ms (`translateX` + scrim fade) and closes over 180 ms.
  Edge-swipe opens it from 0–20 px of the inline-start edge. Do **not**
  enable the edge-swipe on iOS Safari's back-gesture edge unless measured.
  Default: off on iOS, and the `☰` is always present.
- No portal, so it stays in the column like `Sheet` (cmux capture surface).

### 7.6 Fades and scrollbars

- Generalise `.lo-fade-b` into **`.lo-fade-y`**: a scroll-driven top and
  bottom mask, 24 px each, at **0 px when at that edge**. This is the desktop
  technique (`scrollbars-fade.md` §10.2): registered `<length>` custom
  properties animated by `animation-timeline: scroll(self)`. Provide an
  `@supports not (animation-timeline: scroll())` fallback that applies a
  static bottom fade only when content is below (the existing `moreBelow`
  JS flag), so Safari < 18 keeps today's behaviour.
- Applies to: the list scroller, the transcript, every sheet's scroller, the
  drawer, the chip row (inline-axis fade at the end edge), and the tray
  (inline axis).
- **Scrollbars**: the phone's overlay scrollbar is OS-drawn. Keep
  `scrollbar-width: thin` and change `scrollbar-color` to the **resting
  floor**, `color-mix(in srgb, var(--color-control) 45%, transparent)`, so a
  desktop-width browser and the wide view show the faded thumb the desktop
  shows. No JS activity tracker on the phone, since touch scrollbars are
  already transient.

### 7.7 Chips and pills

32 px visual, a 44 px hit area through an absolutely positioned `::before`
(`inset-block: -6px`), `radius-full`, no border, `text-body-sm`. Rest:
`bg-surface ink-muted`. Selected or urgent uses the role washes (§6.4). `Chip`
keeps its single-branch selection logic, and its `border` classes go.

### 7.8 Snackbar (new)

This is for transient outcomes now drawn as in-list lines: the start error,
the mark-all receipt and the pin refusal *when not inside a sheet*. It is
`bg-elevated`, `radius-md`, `shadow-overlay` (it floats), 16 px above the
FAB or composer, with `role="status"` (or `alert` for failures), 4 s, and
swipe-away. Reduced motion means no slide, only appear and disappear.

---

## 8. Offline / reconnecting: honest but quiet

PR A (`store.ts`/`api.ts`) removes the reconnect flashes, so this section only
designs the *indicator*, against PR A's `connected` flag. It invents no new
store state.

- **Grace is owned by PR A**: `connected` becomes false only for a real
  outage after the store’s grace period. The component adds no second grace;
  it derives time since the flag went false locally, never a server sync time.
- **List**: a 8 px `bg-warning` dot on the app bar's `☰` (inline-end-top
  corner), **plus** the app-bar title gains a subtitle "Reconnecting…"
  (`text-meta` `ink-dim`). `role="status"`, announced once. No full-width
  strip. The rows stay as last painted (today's behaviour), at full contrast:
  they are true as of the last sync, and dimming them would claim they are
  wrong.
- **Session**: the subtitle becomes `● reconnecting · last synced 12 s ago`,
  with the dot in `warning`, the time as relative and updating at most every
  5 s, and `aria-live="polite"` only on the transition. The **composer stays
  enabled**: sends queue under the existing envelope/retry contract. The
  full-width strip, `reconnecting — showing the last synced view`, is
  **removed**. The measured cost is that it overlapped the first transcript
  row (`session-offline-*`).
- **Back online**: the dot clears and the subtitle returns to liveness. **No
  "reconnected" toast**: silence is the receipt.
- **Long outage** (> 30 s): the drawer footer line reads "offline · retrying",
  and the ☰ dot stays. Still no strip.

`warning` on canvas measures **9.25:1** (dark) and **5.32:1** (light), so the
dot plus word clears text floors. The dot is never the only carrier, because
the word is always there.

---

## 9. Motion

| what | duration | easing | reduced motion |
|---|---|---|---|
| sheet enter / exit | 240 / 180 ms | `ease-out-quart` | appear/disappear, scrim fade 0 |
| drawer enter / exit | 240 / 180 ms | `ease-out-quart` | instant |
| FAB extend ⇄ collapse | 180 ms (width, label opacity) | `ease-out-quart` | instant swap |
| row swipe reveal | follows finger; settle 180 ms | `ease-out-quart` | swipe disabled |
| chip/pill selection | 120 ms colour | linear | instant |
| fades | scroll-driven (no time) | — | static fallback |
| shimmer / spinner | indeterminate (exempt) | — | existing 0.01 ms cap |

Nothing lifts, scales or translates on press. Press is a colour step
(branding §5). The global reduced-motion cap in `styles/index.css` stays.

---

## 10. Tokens

### 10.1 Type scale (unchanged ramp; names only)

`text-title` 20 (sheet titles never use it), `text-heading` 16 (app-bar title,
sheet title, empty headline), `text-body` 14 (body, list names use 15 via
`text-[0.9375rem]`, which is **open decision O2**), `text-body-sm` 13 (chips,
pills, sheet rows' secondary), `text-meta` 12 (row meta, subtitles, section
headers), `text-mono-sm` 12 (tool objects, paths in detail views only). Two
weights only: 400 and 600 (500 for section headers).

### 10.2 Grounds (the borderless contract)

Measured with `audit/ladder.mjs` (CIELAB L\*) over the relay's 31 palettes:

| pair | min | median | brand dark / light |
|---|---|---|---|
| `elevated` − `canvas` (fields, composer, FAB-less sheets) | **1.07 (alucard)** | 7.90 | 7.90 / 4.73 |
| `surface` − `canvas` (chips, user bubble, tonal) | **0.68 (alucard)** | 3.48 | 3.49 / 2.63 |
| `elevated` − `surface` | **0.39 (alucard)** | 3.95 | 4.41 / 2.10 |

**Three light palettes cannot carry a borderless control as the relay ships
them**: `alucard` (1.07), `gruvboxLight` (1.49) and `rosePineDawn` (2.89
elevated−canvas). The desktop already re-seated those three grounds. Measured
from `~/local-operator-ui` palettes: alucard **5.32**, gruvboxLight **5.52**,
rosePineDawn **5.23** elevated−canvas, with surface−canvas at 2.56 / 2.80 /
2.71. **Required in PR B**: port those three palettes' ground values from the
desktop source of truth (the relay's own palette header says it follows the
desktop contract). Then add ONE ground-step assertion to
`scripts/contrast-contract.mjs`: **`elevated − canvas ≥ 4.0 L*`**. That is the
step every borderless field, the composer and the search pill stand on.
Without it, "borderless" degrades silently on the next palette.

Why that floor and not branding §2's full ladder, measured
(`audit/ground_floors.txt`, `*` = one of the three ported palettes):

| floor | palettes failing today |
|---|---|
| `elevated−canvas` ≥ 4.0 | alucard\*, gruvboxLight\*, rosePineDawn\* — **only the three being ported** |
| `surface−canvas` ≥ 2.5 | the three\* **+ synth 2.44** |
| `elevated−surface` ≥ 2.5 | the three\* **+ localOperatorLight 2.10, sage 2.34** |
| `canvas−sunken` ≥ 1.5 | alucard\*, gruvboxLight\* |

The full ladder would make PR B re-seat the brand light theme and two community
palettes, which is a palette project rather than a layout one. The 4.0
`elevated−canvas` floor is green the moment the three ports land. The other
rungs are recorded here as a follow-up for the palette owner, not gated in
this PR.
Brand light `elevated−surface` is 2.10, under 2.5: use **elevated on canvas**
(4.73) for fields and the composer, never elevated on surface, in the light
brand theme, until the ladder is re-seated (**O3**).

### 10.3 Ink pairs introduced (contrast_pairs.py, brand palettes)

16 pairs, **0 below 4.5:1**. Full output is in `audit/contrast_pairs.out`.
Tightest: `ink-dim` on `elevated` (dark) **4.61:1**, `accent` on
`accent-wash` (light) **4.78:1**, `ink-dim` on `canvas` (light) **4.95:1**.
Fleet-wide, `contrast-contract.mjs` already asserts `inkDim`/`inkMuted`/`accent`
on all four grounds and `accent` on `accentWash` (1333 assertions, holds). The
new usage introduces no role-pair outside that contract: the tonal danger
button (`danger` on `dangerWash`) is already asserted by the contract's
`tone` on `${tone}Wash` loop (`contrast-contract.mjs:164-168`). The only
contract change PR B needs is the ground-step assertion in §10.2.

### 10.4 Space and radii

The 4 px ramp: 4/8 inside a component, 12/16/24 between, 32+ between
sections. Radii: 6 (controls, fields, option rows), 10 (search pill, bubbles,
tool well, snackbar), 14 (sheet top, pending card, FAB), 16 (composer only),
full (chips, pills, dots, grabber).

---

## 11. Accessibility, RTL, i18n

- **Targets**: every interactive box ≥ 44x44 (visual or hit area). The
  approval checkbox fix closes the only measured `target-size` FAIL.
- **Focus**: `:focus-visible` outline only. Order follows visual order: app
  bar → chips → list → FAB, and session app bar → transcript → tray →
  composer. The drawer and sheets trap focus and return it.
- **Names**: the drawer is a `nav` with `aria-label`. ☰ is
  `aria-expanded` + `aria-controls`. Tray pills say their full meaning
  ("1 question waiting", not "?").
- **RTL** (Urdu PR in flight; `directionFor` already returns `rtl`): **all
  new and touched CSS uses logical properties** (`ms/me/ps/pe`,
  `inset-inline-*`, `text-start`, `border-inline-start`). Today's touched files
  carry 29 `text-left`, 18 `pl-`, 14 `border-l`, 13 `ml-`, 9 `pr-` and 3
  `right-` (`audit/physical_props.txt`), and **every one in a file PR B
  touches converts.** The drawer, swipe, FAB, chevrons (`‹`, `▸`) and the
  inline-start bar all mirror. Glyph chevrons flip via `rtl:-scale-x-100`.
- **Copy** goes through relay keys. `session-list.tsx`, `session-view.tsx`,
  `ask-dock.tsx`, `pending-card.tsx` and `ui/disclosure.tsx` are **ENFORCED**
  in `scripts/check-i18n.mjs`, so a new literal there fails CI.

### 11.1 New or changed strings

Every new or changed label lives in **one new owned namespace**,
`local_operator/i18n/catalogues/en/relay.chrome.json`, with role, tone,
phone-width and placeholder notes in `relay.chrome.context.json` beside it.
The English source is frozen before implementation so translations can run in
parallel. **No existing `en/relay.*.json` value changes**: components switch to
new `relay.chrome.*` keys and superseded keys remain in their original catalogue.
This avoids staling passed translation-ledger rows owned by parallel work.

The catalogue and sidecar are the exact copy contract; they cover drawer,
filters, FAB, row actions, appearance, liveness, status tray, banners and
composer accessible names. All new copy uses sentence case (O1). Relative
reconnect age uses complete ICU plural messages for seconds/minutes rather
than inserting English fragments. Existing asks copy is reused where its
meaning is unchanged. Notifications adds no key or placeholder in this PR.

`OWNED_SPA_NAMESPACES` in `scripts/i18n/generate.py` registers the namespace;
regenerate with that script and verify with `--check` and `scripts/i18n/check.py`.
The web scanner at this base has no `--update` CLI: lower only touched files’
exact fingerprint maps using its exported `scanSource`/`fingerprints`, rejecting
any count above the PR-base baseline. Do not add an exemption or raise a ceiling.

## 12. Open decisions (recommended default first)

| id | question | default | what would change it |
|---|---|---|---|
| O1 | Sentence case vs the relay's lowercase voice | **Sentence case** for all new and changed keys | Operator preference for the terminal voice |
| O2 | List row title 15 px (off-ramp) vs 14 (`text-body`) | **14 px `text-body` weight 500 when unread**: stay on the ramp; the mock's 15 is a preview only | Frames at 360 showing names under-read |
| O3 | Brand-light `elevated−surface` is 2.10 L\* (< 2.5 floor) | Fields/composer on `elevated` over **canvas** only; sheets' inner fields use `canvas` (step down) | The kit re-seats localOperatorLight |
| O4 | Tool-row verbs ("Read", "Ran") need an i18n verb map | **Keep tool name in sans** this PR; verb map follow-up | — |
| O5 | FAB vs labelled bottom button for New | **Extended FAB**: thumb zone, no row cost, standard Material/iOS pattern | Usability finding that the FAB covers the last row's trailing badge (mitigated by 88 px end padding) |
| O6 | Edge-swipe to open drawer on iOS | **Off on iOS**, on elsewhere | Measured no conflict with Safari back-swipe |
| O7 | Filter chips vs a single segmented "All / Needs you / Unread" | **Chips**, because tags extend the same row | — |
| O8 | Pinned shown as a section vs a chip | **Section** (keeps the shared pin store's mental model across TUI/desktop) | — |

---

## 13. Evidence and rig

Evidence is attached to the PR, **not committed to the repository**. The PR
records source identities, viewport/theme, actual browser-host pixel ratio,
BEFORE/AFTER measurements, commands and real responses. Historical audit
measurements in §1 are retained; historical headless frames are not presented
as pixel-identical comparisons to a different browser host.

Representative matched BEFORE/AFTER pairs use the same sanctioned browser
host and the real relay bundle inside the fixture’s 390×844 or 360×780 iframe.
A stitched outer-viewport composite is labelled as such, never as native phone
or iOS evidence. The geometry probe runs in the actual iframe DOM.

Rig: `scripts/mobile_showcase_fixture.py` seeds test sessions through the
real stores and serves the real bundle. The `/api/models` and `/api/usage`
readers are substituted for populated sheet coverage; the PR discloses this.
`/showcase` frames the app for the harness browser and exposes explicit phone,
theme, forced-RTL and Audit controls. The fixture uses an isolated HOME/config,
no live sessions, and its own signed-cookie namespace (cookies share a host
across ports); absent/invalid cookies must still return 401. Its fixture-only
metrics endpoint writes only to the evidence root configured by the runner.

`scripts/mobile_showcase_metrics.js` measures the actual iframe DOM from the
showcase’s Audit button. Page interaction and capture use the harness browser
tool, never a separately launched Chrome/CDP instance. The designer’s earlier
standalone runner is preserved as historical author evidence, not shipped in
this PR. Matched pairs and stitched-composite provenance accompany the PR.

AFTER gates: bordered elements ≤ 3 per screen; list chrome ≤ 92 px; transcript
≥ 560 px at 390×844 with nothing pending; zero target-size, input-label and
low-contrast FAILs; every row of §3.2 reachable by finger. Gate results and any
remaining limitations belong in the PR, not as unverified assertions here.

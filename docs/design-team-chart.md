# Design: `/team chart [name]` — a scrollable, zoomable org chart in the TUI

**Status:** proposal (design only, no feature code)
**Author:** architect (lopdev team)
**Scope of ground truth read:** `AGENTS.md`; `local_operator/teams.py`;
`local_operator/tools/team_tool.py`; `local_operator/agent_profiles.py`;
`local_operator/session/session.py` (`attach_team`, `_resolve_profile_or_specialist`);
`local_operator/harness/subagent.py` (member-preamble stamping);
`local_operator/tui/app.py` (`SLASH_COMMANDS`, `slash_command_for`,
`_run_slash_command`, `_cmd_team`, `_team_choices`, `_team_list_block`,
`on_argument_query_opened`, subagent-view open/close, `_cmd_mcp`,
`_mcp_argument_choices`); `local_operator/tui/autocomplete.py`;
`local_operator/tui/widgets/editor.py` (argument dispatch, `/mcp` two-level
precedent); `local_operator/tui/widgets/subagent_view.py`;
`local_operator/tui/local_operator.tcss` (`Screen.subagent`, `.subagent-view*`).
A throwaway Buchheim prototype (`scratch/tidytree_proto.py`) proves the layout math.

---

## 1. Problem, as found

Today a "team" is a **flat roster**: one `manager` plus a list of
`TeamMember` slots, each a `role: str` + `count: int` referenced by NAME
(`teams.py:72-89`). There is no way for a team member to itself be a team, so
there is no notion of an "org" — a team of teams. The only visualisations are
textual: `_team_list_block` (`app.py:2817`) lists team names, and the `team`
tool's `_op_show` (`team_tool.py:128`) prints one team's roster as lines. Neither
shows structure, boundaries, or nesting, and neither scrolls or zooms.

The user wants three things that interlock:

1. **Nesting** — a team may contain other teams as members, recursively, with
   team context (collaboration + project briefs, manager/member preambles)
   flowing correctly through the assembled org.
2. **A rendered org chart** — manager at top, members beneath, nested teams
   expanded, drawn as a *tidy tree* that centers parents over children and
   spaces subtrees without overlap, on a large scrollable canvas with zoom
   tiers.
3. **A new command surface** — `/team chart [name]` — that must not collide
   with the existing `/team <name> <request>` grammar, even when a team is
   literally named `chart`.

The constraints that make this non-trivial, all confirmed against the code:

- **Backward compat is load-bearing.** `TeamRegistry._load` (`teams.py:235`)
  reads every `team.yml` on disk and `Team.model_validate`s it; existing files
  have `members: [{role, count}]` and no nesting field. A schema change that
  makes those files fail validation silently drops the team (the `except` at
  `teams.py:258` logs and skips). The new field must be optional with a
  backward-compatible default.
- **`extra` is NOT forbidden on `Team`/`TeamMember`.** Only `TeamParams`
  (`team_tool.py:53`) sets `extra="forbid"`. So a `Team` loaded from a future
  YAML with an unknown key would not raise today — but we should not rely on
  that; we add the field explicitly.
- **Two preamble layers already exist** and are consumed in two places:
  `attach_team` (`session.py:1944`) stamps `manager_preamble()` onto the
  manager session's volatile tail, and `subagent.py:316` stamps
  `member_preamble(agent)` in front of each child launch. Nesting must decide
  which preamble a nested manager vs a nested worker sees, and the resolution
  must reuse `_resolve_profile_or_specialist` (`session.py:1987`) rather than
  fork it.
- **Names are ambiguous by design.** A member NAME resolves against roles,
  specialists, AND seeds (`agent_profiles.resolve_profile`, `session.py:1972`).
  With nesting, a name could ALSO be a team. The chart resolver has to pick a
  kind deterministically and never infinite-loop on a cycle.
- **The picker offers team names on `/team `** (`app.py:9667`,
  `on_argument_query_opened`). Adding `chart` as a reserved word in the same
  slot risks the exact collision the user called out.

The smallest change that satisfies all of this is: **one optional field on
`TeamMember`**, **one pure resolver module**, **one pure layout module**, **one
scrollable Textual widget modelled on `SubagentView`**, and a **`/team chart`
subcommand carved out of the first argument token** using the `/mcp` two-level
precedent. No rewrite of teams, sessions, or the picker is required.

---

## 2. Data model for nesting

### 2.1 The schema change — one optional discriminator on `TeamMember`

```python
class TeamMember(BaseModel):
    """One roster slot: a named agent/role, or a nested TEAM, possibly repeated."""

    role: str = Field(..., description="Role/specialist name, or team name when kind='team'.")
    count: int = Field(default=1, ge=1, le=16)
    # NEW. Absent/"agent" in every existing team.yml, so old files load unchanged.
    # "team" marks this slot as a reference to ANOTHER team by name, turning a
    # flat roster into an org. The name still lives in `role` (not a new field)
    # so member_names(), roster_lines(), and every existing reader keep working
    # without knowing about nesting — a team slot simply reads as its team name.
    kind: Literal["agent", "team"] = Field(default="agent")
```

**Why a `kind` discriminator on the existing slot, not a new `subteams` list
or a naming convention:**

- *Backward compat is free.* `kind` defaults to `"agent"`, so every existing
  `team.yml` (which has no `kind` key) validates to exactly what it means today.
  No migration, no version bump on the file format, no `_load` change beyond the
  field existing.
- *One roster, one order.* The user sees members and sub-teams in ONE authored
  order. A separate `subteams: list[str]` field would split the roster into two
  lists the author has to keep mentally merged, and would force `roster_lines`,
  `member_count`, and the picker to concatenate them — two code paths where one
  suffices.
- *A naming convention (`@teamname`, `team:foo`) is rejected.* It overloads the
  `role` string, collides with the legal name charset (`_NAME_RE` allows `.`,
  `-`, `_` but the sigil would have to be stripped everywhere a name is read),
  and makes "a member named the same as a team" undetectable. An explicit
  boolean-ish field is unambiguous and self-documenting.
- *Counts still apply.* `reviewer x2` means two reviewers; `pod x2` (kind=team)
  means two independent copies of the `pod` sub-org. The chart renders `count`
  copies as sibling subtrees (or, at coarse zoom, a single box badged `×2` — see
  §4.3). `member_count()` already sums counts and stays correct.

### 2.2 What a team-member does NOT get

No new instruction fields on `TeamMember`. A nested team's collaboration and
project briefs live on the nested `Team` itself, exactly as they do for a
top-level team. The slot only points at the team by name; the pointed-to team
carries its own briefs. This keeps the slot tiny and keeps a team reusable —
the same `pod` team can be nested under two different orgs without copying its
briefs into each parent's roster.

### 2.3 Context / brief flow through the org

This is the subtle part. The existing model has exactly two preamble surfaces
and nesting must map onto them without inventing a third, because only two are
consumed anywhere (`attach_team`, `subagent.py`).

**Rule: a team boundary re-roots the manager/member split.** At every level of
the org:

- The **manager of a team** is briefed with THAT team's `manager_preamble()`
  (its roster, collaboration, project). This is unchanged from today.
- A **worker (kind=agent) member** of a team is briefed with THAT team's
  `member_preamble(role)`. Unchanged from today.
- A **nested team (kind=team) member** is NOT a leaf agent — it is a sub-org.
  When the parent manager delegates to it, the entity that receives the work is
  the **sub-team's manager**, and it must be briefed with the SUB-team's
  `manager_preamble()`, not the parent's `member_preamble()`. The parent's group
  context reaches the sub-team the same way it reaches any child: it is the
  delegating manager, so its own briefs are already in its session; what it hands
  down is the TASK. The sub-manager then re-briefs its own members from the
  sub-team's briefs.

In other words **briefs do not deep-merge or concatenate down the tree.** Each
team boundary is a fresh manager/member context. This matches how orchestration
already works: `subagent.py:313` reads `parent_session.active_team` and stamps
that ONE team's member preamble; a nested manager launched as a child would set
its OWN `active_team` (the sub-team) when it in turn delegates. The org is
assembled at RUN time by managers delegating to sub-managers, each carrying one
team's context — not by pre-flattening every brief into one giant preamble
(which would blow the `MAX_TEAM_INSTRUCTIONS_CHARS` bound, `teams.py:65`, and
duplicate context at every level).

> **Design boundary for THIS feature:** the *runtime* wiring that lets a
> manager delegate to a nested team's manager (spawning a sub-manager child that
> attaches the sub-team) is a larger orchestration change and is called out in
> §8 as a **follow-up**, gated behind its own review. The `/team chart` feature
> itself needs ONLY the data model (§2.1) and the resolver (§3). The chart
> *renders* the org that the data model describes; it does not require the
> delegation runtime to exist first. Shipping the schema + resolver + chart is a
> coherent, independently-valuable slice, and the doc is explicit that the
> chart shows the *declared* org, annotating whether the delegation runtime is
> wired yet (§4.3, "unresolved" tier).

### 2.4 Cycle detection and depth bounds (in the MODEL layer)

Cycles are a data property, so the guard lives where the data is resolved
(§3), not in the widget. But the model exposes the primitives:

- **`MAX_ORG_DEPTH = 8`** — a module constant in `teams.py`. Eight levels is far
  past any real human org and keeps the resolver, the layout, and the render
  bounded. Deeper references are truncated with a visible "⋯ depth limit" node.
- The registry gains **`get_team_by_name`** already exists (`teams.py:278`); the
  resolver uses it plus a `visited: set[team_id]` path to break cycles. No new
  registry method is required for cycle safety.

---

## 3. Resolution — from a team name to an in-memory ORG TREE

A new **pure module** `local_operator/org_chart.py` (no Textual import, unit-
testable in isolation, mirrors how `agent_profiles.py` is pure and imported by
both tool and TUI). It exposes one entry point:

```python
def resolve_org(
    name: str,
    *,
    teams: TeamRegistry,
    agents: Any,            # AgentRegistry | None — for role/specialist kind tagging
) -> OrgNode: ...
```

### 3.1 The node type

```python
@dataclass(frozen=True)
class OrgNode:
    label: str                      # display name (team name or agent role)
    kind: Literal[                  # what this node IS
        "team",                     # a team boundary (has a manager + members)
        "manager",                  # the manager agent of the team above it
        "role", "specialist", "seed",  # a resolved agent leaf
        "unresolved",               # a name that matched nothing
        "cycle",                    # a team already on the path (stop here)
        "depth",                    # depth limit reached
    ]
    count: int = 1                  # slot multiplicity (from TeamMember.count)
    detail: str = ""                # e.g. "led by manager", "×2", "no members"
    children: tuple["OrgNode", ...] = ()
    team_id: str | None = None      # set on team/manager nodes, for cycle tracking
```

### 3.2 The algorithm (linear in nodes, cycle-safe, total)

```
resolve_org(name):
    top = teams.get_team_by_name(name)
    if top is None: return OrgNode(name, "unresolved", detail="no such team")
    return _team_node(top, path=frozenset(), depth=0)

_team_node(team, path, depth):
    if depth > MAX_ORG_DEPTH:
        return OrgNode(team.name, "depth", team_id=team.id)
    if team.id in path:
        return OrgNode(team.name, "cycle", team_id=team.id,
                       detail="already shown above")
    path2 = path | {team.id}
    children = []
    # 1) the manager is the FIRST child of the team boundary
    children.append(_agent_node(team.manager, role_of_manager=True))
    # 2) then each roster slot, in authored order
    for m in team.members:
        if m.kind == "team":
            sub = teams.get_team_by_name(m.role)
            if sub is None:
                node = OrgNode(m.role, "unresolved", count=m.count,
                               detail="missing team")
            else:
                node = _team_node(sub, path2, depth + 1)
                node = replace(node, count=m.count)   # carry the slot's ×N
        else:
            node = _agent_node(m.role, count=m.count)
        children.append(node)
    return OrgNode(team.name, "team", team_id=team.id,
                   detail=_team_detail(team), children=tuple(children))

_agent_node(name, *, count=1, role_of_manager=False):
    kind = _classify(name, agents)     # role | specialist | seed | unresolved
    detail = "manager" if role_of_manager else ""
    return OrgNode(name, "manager" if role_of_manager else kind,
                   count=count, detail=detail)
```

### 3.3 Every required edge case, and how it falls out

| Case | Behaviour |
|---|---|
| **1-agent team** | team node with two children: manager + the one member. Layout centers fine (prototype confirmed with single-child subtrees). |
| **Empty team** (`members == []`) | team node with ONE child (the manager) and `detail="no members"`. Renders as manager-only. |
| **Team under team** | `kind=="team"` slot recurses via `_team_node` at `depth+1`. |
| **Org within org** | same recursion, naturally deeper. Bounded by `MAX_ORG_DEPTH`. |
| **`count > 1` agent** (`reviewer x2`) | node carries `count=2`; render per §4.3. `member_count()` unaffected. |
| **`count > 1` team** (`pod x2`) | sub-team resolved once, `count` stamped on the returned node; render shows `×2` badge or N sibling copies by zoom tier. |
| **Cycle** (A→B→A) | second visit to a team already in `path` returns a `"cycle"` leaf — **no recursion**, no infinite loop. Proven by the `path` frozenset test. |
| **Name is BOTH a team and an agent** | **team wins for a `kind=="team"` slot; agent wins for a `kind=="agent"` slot.** The author's `kind` is authoritative — that is the whole point of the discriminator. For the TOP-LEVEL `resolve_org(name)`, a team is looked up first (the command is `/team chart`, so a team is the subject); if no team of that name exists it is `"unresolved"`, never silently reinterpreted as an agent. |
| **Missing referenced team** | `get_team_by_name` returns None → `"unresolved"` node with `detail="missing team"`, rendered as a dim ghost box so the gap is visible rather than dropped. |
| **Missing referenced agent** | `_classify` returns `"unresolved"` → dim ghost box. |

`_classify(name, agents)` reuses the resolution ORDER already fixed in
`session._resolve_profile_or_specialist` (`session.py:1987`): own role → own
specialist → packaged seed → unresolved. To avoid importing a `Session` method
into a pure module, factor the *classification* half (name → kind label) into a
small helper in `agent_profiles.py` that both the session resolver and the org
resolver call. This is the "fix the existing thing rather than add a second one
beside it" move: one classifier, two callers, no drift (the codebase already
paid for that drift once — the A1 bug noted at `session.py:1966`).

---

## 4. Layout — tidy tree on a character grid

### 4.1 Algorithm: Buchheim/Walker linear-time tidy tree, variable node width

The prototype (`scratch/tidytree_proto.py`, run and rendered above) implements
the **Buchheim–Junger–Leipert** improvement of Reingold–Tilford: O(n), parents
centered over their children, subtrees packed to a minimum separation without
overlap. The one adaptation from the textbook is that node "width" is not
uniform — a box is as wide as its label needs — so the sibling separation is
`w_a/2 + w_b/2 + H_GAP` (half each box plus a gutter), computed in **cell
units**. The prototype's output shows correct centering and packing for a
mixed-width, mixed-depth org:

```
                          [ director ]
                   |                         |
             [manager-A ]              [manager-B ]
     |            |              |           |
 [coder ]  [reviewer x2 ]  [ designer]  [sub-mgr ]
                                        |         |
                                    [scout ]  [coder ]
```

Why this and not a simpler recursive-centering pass: naïve "center parent over
the midpoint of children" produces overlapping cousins when one subtree is
wider than the gap between two parents (the classic Reingold–Tilford failure).
Buchheim's `apportion`/`move_subtree`/threads fix exactly that, in linear time,
and the prototype confirms it on the org-within-org shape.

### 4.2 Mapping to the grid

- **Coordinates are floats during layout, rounded to ints at render** (second
  walk), so cumulative half-widths do not drift. `x` is a cell column, `y` is a
  level index.
- **Level bands:** each level occupies `LEVEL_H` rows (box row, drop row, bus
  row). A thin **boundary rule** per team (a faint horizontal `───` one row
  UNDER the team's children, spanning their x-extent) draws the "boundary at
  each level" the user asked for — cheap, since a team node knows its children's
  min/max `x`. It is a separate final pass that fills only BLANK cells, so it
  never clobbers a box or connector: under a row of leaf members (the case the
  rule exists for) two sibling teams draw as two distinct spans with a gap
  between, making the grouping legible at a glance; a sub-team's drop connector
  simply interrupts the rule where it crosses. Rules draw at standard and
  detailed only — outline is already one-box-per-team, so grouping there needs
  no rule. Painted in a dedicated `rule` style key, fainter than a connector,
  because it is grouping, not structure (D1).
- **Connectors:** `│` drops from a parent's center; an elbow (`├─┬─┤`) row joins
  siblings. The prototype draws only the drop for brevity; the shipped renderer
  draws the full elbow bus per parent (one row, computed from children x-span).
- **The whole thing is painted into a `rich` `Segment`/`Text` grid or a
  `Static` with a fixed `Text`,** then hosted in a scroll container whose
  virtual size equals the grid's (width, height) in cells. Textual scrolls it;
  we do not re-layout on scroll.

### 4.3 Zoom = detail tiers (re-render, not font scaling)

A terminal cannot scale glyphs, so "zoom" is **level-of-detail**: the same org
tree, rendered at one of N tiers. Tier is a widget state (`_zoom: int`), changed
by key, triggering a re-layout+repaint (cheap — the tree is already resolved;
only box widths and which rows are drawn change).

| Tier | Node box | Nested teams | count>1 | Use |
|---|---|---|---|---|
| **0 — Outline** | team name only, one box per TEAM (members collapsed to a count badge `[pod ·5]`) | collapsed to a box | badge `·N` | the "collapse deep levels to boxes" view; fits a big org on one screen |
| **1 — Standard** (default) | one box per agent AND team boundary; manager marked | expanded one level, deeper teams collapsed to boxes | `×N` badge on one box | the everyday chart |
| **2 — Detailed** | box shows name + kind (role/specialist/seed) + count; team boxes show `led by <mgr>` | fully expanded to `MAX_ORG_DEPTH` | AGENT copies drawn as N sibling boxes; TEAM copies stay one `×N`-badged box | audit / verify the whole org |

> **Count expansion, agents vs teams (D4).** At the detailed tier a `count>1`
> *agent* draws as N sibling boxes (`reviewer ×2` → two `[ reviewer (seed) ]`),
> but a `count>1` *team* stays a single `×N`-badged box rather than duplicating
> its whole subtree — duplicating an org N times would explode the canvas for no
> extra information (the copies are identical). This asymmetry is deliberate;
> the table above reflects the shipped behaviour.

`collapse`/`expand` on a focused team node overrides the tier for that subtree
(a per-node `expanded` set), so a user can drill one branch without exploding
the whole org — the same interaction the subagent view's brief expand/collapse
establishes (`subagent_view.py:439`, `toggle_brief`).

Ghost/annotation nodes: `unresolved` → dim box with `?`; `cycle` → box with `↩`
and `detail`; `depth` → `⋯`. When the §2.3 delegation runtime is not yet wired,
team-boundary nodes carry a faint `(declared)` tag so the chart never implies a
capability that is not live.

---

## 5. Widget / mode — a scrollable page modelled on `SubagentView`

### 5.1 Choice: a full-page MODE (like the subagent view), not a pushed Screen

Three options were weighed against the code:

1. **A pushed `Screen`** — blacks out the dock, hides the composer. Rejected for
   the same reason the subagent-view redesign rejected it
   (`subagent_view.py:20-27`): the org chart is an *inspection of the current
   session's* teams, and the page should read as the same app in a different
   mode, with the band/status/composer still visible (greyed).
2. **A floating overlay card** — cannot scroll a large canvas or hold focus for
   zoom keys; the toast layer is for transient cards (`AGENTS.md:216`).
3. **A full-page mode that replaces the transcript region and keeps the dock**
   — exactly the `SubagentView` pattern (`app.py:6871` `_open_subagent_view`,
   `SUBAGENT_LAYOUT_CLASS`, `Screen.subagent` in tcss). **Chosen.** It already
   solves: hide the transcript (`_transcript_view().display = False`), mount
   before `#input-dock`, add a layout class that greys inert chrome, set the
   composer read-only, and restore all of it on Esc.

### 5.2 The widget

`local_operator/tui/widgets/org_chart_view.py` — `class OrgChartView(Vertical)`,
`classes="org-chart-view"`, `can_focus = True`, structured like `SubagentView`:

- a title `Static` (`team <name> · org chart · zoom: standard`),
- a rule `Static`,
- a **body** that is a `ScrollableContainer` (or a `Static` inside one) holding
  the rendered grid as a single `Text`; virtual size = grid size, so Textual's
  own vertical AND horizontal scrollbars appear when the org exceeds the
  viewport — the chart is wide, so **horizontal scroll is required** (the
  subagent view is vertical-only; this differs deliberately and the tcss must
  allow `overflow-x: auto`),
- a hints row (same `HintButton` vocabulary as `subagent_view.py:740`).

Identified by CLASS not id (the `DuplicateIds`-on-fast-reopen lesson,
`subagent_view.py:720`).

### 5.3 Key bindings

| Key | Action |
|---|---|
| `+` / `-` (and `=`) | zoom tier in / out |
| `←↑↓→` | scroll the canvas (arrows scroll by a line; wrap disabled — this is a canvas, not a list, so movement CLAMPS per `AGENTS.md:221`) |
| `PgUp/PgDn` | page vertically (clamp); `shift+←/→` page horizontally — the wide axis overflows most, so it gets a jump too (U4) |
| `Home/End` | jump to the top-left / bottom-right CORNER (both axes); `End` is the keyboard jump to the right edge on a chart with no vertical travel (U4) |
| `enter` / `space` | expand/collapse the whole canvas (`e` is the primary key; v1 has no per-node focus) |
| `f` | fit-to-width among the MEMBER-showing tiers (standard/detailed) — never auto-collapses to the outline box that would hide the roster (U2); falls back to standard + scroll when nothing fits |
| `esc` | leave the mode (via `_close_org_chart_view`, mirrors `_close_subagent_view`) |

Node focus/navigation across the grid is a stretch goal; v1 focuses the whole
canvas and scrolls it, with zoom + collapse-all/expand-all as the primary
controls. This keeps v1 small; per-node keyboard focus is listed as follow-up.

### 5.4 app.py integration (mirrors the subagent view exactly)

- `self._org_chart_view: OrgChartView | None = None` beside
  `self._subagent_view` (`app.py:1400`).
- `_open_org_chart_view(team_name)` / `_close_org_chart_view()` cloned from
  `app.py:6871`/`6900`: capture focus-restore, hide transcript, mount before
  `#input-dock`, add `ORG_CHART_LAYOUT_CLASS = "org-chart"`, set composer
  read-only; reverse on close.
- **Esc precedence:** add `if self._close_org_chart_view(): return` into the
  same Esc chain that already calls `_close_aside()` / `_close_subagent_view()`
  (`app.py:4434-4437`), and into the approval/ask/clear yield points that close
  the subagent view (`app.py:4824`, `4919`, `5337`) so a chart cannot sit over a
  prompt the turn is parked on.
- A new `.tcss` block: `.org-chart-view` (height 1fr, padding), `.org-chart-view-body`
  (`overflow: auto` — both axes), and `Screen.org-chart #prompt-chevron,
  Screen.org-chart #todo-panel { text-opacity: 45% }` cloned from
  `local_operator.tcss:1245`.

### 5.5 Degrading on a small terminal

- On the FIRST layout the widget auto-fits (the same logic as the `f` key): it
  picks the coarsest MEMBER-showing tier that fits the viewport, so a small
  terminal opens already zoomed to what fits rather than always at standard.
  Crucially it does NOT auto-collapse to the outline box — that would hide the
  roster (U2); when even standard overflows it opens at standard and relies on
  scroll, and the title always states the current tier so "why is this zoomed"
  is answered on screen. Outline (tier 0) remains a deliberate `-`/zoom choice,
  where the `·N ?` badge summarises member count and flags any unresolved
  member so the gap is never invisible (minor-2).
- The footer hints SHED whole controls widest-first as the width tightens
  (D3): `read-only`, then the secondary controls drop, but `scroll`, `zoom`,
  and `esc` survive and the row never clips a word mid-character.
- A team with a single agent and no nesting renders as a 2-node column that fits
  any terminal.
- The mode keeps the dock, so the composer/band never disappear even at 80×24.

---

## 6. Routing / collision — `/team chart [name]` vs `/team <name> <request>`

### 6.1 The precise rule (first argument token is the disambiguator)

`/team` is already an `OPTIONAL`-argument command (`app.py:516`). We make its
first argument token a small **reserved subcommand namespace**, exactly as
`/mcp` reserves `login|logout|reauth` in its first argument slot
(`editor.py:2509-2519`, `app.py:10503`). One reserved word: **`chart`**.

`_cmd_team(arg)` (`app.py:2850`) parses `arg` as `first, rest = arg.split(maxsplit=1)`:

```
if first == "chart":                      # RESERVED SUBCOMMAND
    _cmd_team_chart(rest)                 # rest = optional [name]
    return
# else: existing behaviour — first is a TEAM NAME, rest is the request
```

**Precedence:** the reserved word wins in first position. This is safe *for the
common case* because no existing team can be named such that the user is
surprised — but a team CAN legally be named `chart` (`_NAME_RE` allows it), so
we need an escape hatch.

### 6.2 The escape hatch — a team literally named `chart`

Two independent, composable answers (ship both):

1. **`/team chart chart`** already works and is unambiguous: the FIRST token is
   the subcommand, the SECOND is the `[name]` argument. So the chart of a team
   named `chart` is reachable as `/team chart chart`. This is the natural
   consequence of the grammar and needs no special code.
2. **To TALK to a team named `chart`** (the `/team <name> <request>` path), the
   first token would be captured by the subcommand. Escape with a leading
   marker: **`/team =chart <request>`** (or `/team ./chart`), where a leading
   `=` on the first token means "treat the rest of this token as a literal team
   name, never a subcommand." `_cmd_team` strips a single leading `=` from
   `first` before the name lookup and skips the subcommand check when it was
   present. `=` is chosen because it is NOT in `_NAME_RE`, so it can never be
   part of a real team name and cannot itself collide.

Both are documented in the `/team` help/description and in `guide://teams`.

### 6.3 `/team chart` with no name

Meaningful: **chart the currently-attached team** if this session has one
(`session.active_team`, set by `attach_team`), else a one-line error pointing at
`/team chart <name>` and `/team` (the list). Rationale: a manager session is
usually *in* a team, and "show me my org" with no argument is the most common
ask; falling back to the attached team makes the bare form useful instead of an
error. If no team is attached and none is named, we do not guess — we tell the
user how to name one.

### 6.4 Picker behaviour (the collision the user feared, solved)

On `/team ` the argument list currently offers team names (`app.py:9667`). We
change `on_argument_query_opened`'s `team` branch to offer, in the **first
argument slot only**, the reserved subcommand row FIRST, then the team names:

```
[ chart ]   Show a team's org chart            (subcommand)
[ <teamA> ] led by manager · 4 roles
[ <teamB> ] ...
```

- The `chart` row is visually tagged as a subcommand (a `detail` like
  `subcommand` and/or a leading glyph) so it reads as different in kind from a
  team name — the same separation `/mcp` draws between verb rows and server rows.
- Because the matcher ranks the whole typed argument, typing `ch` ranks `chart`
  first but ALSO any team whose name starts with `ch`; a team named `chart`
  appears as its own row beneath the subcommand row, so it is never hidden.
- **Second slot:** once `chart ` is in the buffer, the argument becomes
  `chart <query>`, and we offer TEAM NAMES again (the `[name]` the chart wants) —
  reusing the `/mcp` two-level `_argument_subcommand` tracking (`editor.py:2516`)
  and a `RefreshArgumentChoices` post. So `/team chart ` re-opens the team list,
  now feeding the chart instead of the talk-to path. Completing a name yields
  `/team chart <name>`, ready to run.
- The editor's `_is_name_argument_command` / `set_name_choices` machinery
  (`app.py:9675`, `editor.py:2547`) still applies to the talk-to path so team-
  name highlighting is unchanged there.

### 6.5 Why NOT a separate top-level `/chart` command

Rejected. The user was explicit that this is a team operation and that the word
`chart` must not steal a namespace. A top-level `/chart` would (a) consume a
scarce top-level command word, (b) still need the same name/subject plumbing,
and (c) break the mental model that everything about teams lives under `/team`.
Nesting the subcommand under `/team` is the smaller, more consistent change and
matches `/mcp`'s established shape.

### 6.6 The `team` tool must be able to author nesting

`TeamParams.members` (`team_tool.py:73`) is `list[str]` parsed by
`parse_members` (`teams.py:362`) from `role` / `role:count` tokens. Extend the
token grammar with an optional **`team:` prefix** to mark a nested-team slot:
`team:pod` or `team:pod:2` (name `pod`, kind team, count 2). `parse_members`
gains a leading-`team:` check that sets `kind="team"`; a bare token stays
`kind="agent"`. This is the ONE place the tool authors nesting, and it keeps the
existing `coder` / `coder:2` grammar untouched (no `team:` prefix ⇒ agent). The
tool's `_op_show` roster lines already come from `roster_lines()`, which we
extend to badge team slots (`- pod (team) x2`) so `team show` reveals nesting
too. Document the new grammar in the `team` tool description and `guide://teams`.

---

## 7. Validation plan (per AGENTS.md "Visual validation")

The chart is a large visual surface, so **rendered frames are the evidence, a
green test is not** (`AGENTS.md:74`). Capture with the real `OperatorApp` +
`save_screenshot`, before/after where an existing surface changes.

### 7.1 A dedicated shot script

Add `scripts/org_chart_shot.py` modelled on `scripts/ask_shot.py`
(`AGENTS.md:83`): seed a `FakeSession` with a `TeamRegistry` populated for the
permutation, open the chart via `_open_org_chart_view(name)`, `pilot.pause()`,
`app.save_screenshot(path)`. Takes `out.svg`, `WxH`, and a scenario key.

### 7.2 The permutation matrix to screenshot

| # | Scenario | What it proves |
|---|---|---|
| 1 | **1-agent team** | manager + one leaf centers, fits 80×24 |
| 2 | **empty team** (no members) | manager-only, `no members` detail, no crash |
| 3 | **flat multi** (manager + 4 mixed role/specialist) | elbow bus, centering, kind tags at tier 2 |
| 4 | **nested org** (director → 2 managers → workers) | recursion, boundary rules, tier 1 default |
| 5 | **org within org** (3 levels) | depth handling, horizontal scroll appears |
| 6 | **cycle** (A→B→A) | `cycle` leaf, NO hang (also a unit test) |
| 7 | **count>1** (`reviewer x2`, `pod x2`) | badge at tier 0/1, N copies at tier 2 |
| 8 | **deep** (near `MAX_ORG_DEPTH`) | `depth` truncation node, no runaway |
| 9 | **missing refs** (bad team + bad agent names) | dim ghost boxes, gaps visible |
| 10 | **each zoom tier** on scenario 4 | outline/standard/detailed differ correctly |
| 11 | **small terminal** (60×20) on scenario 4 | auto-tier-0 + scroll, dock intact |
| 12 | **picker frames**: `/team `, `/team ch`, `/team chart ` | subcommand row vs team rows; second-slot team list |

For scenarios that change an existing surface (the `/team ` picker, #12), capture
**before/after** per `AGENTS.md:143` — before from a clean worktree
(`AGENTS.md:164`, never `git stash`).

### 7.3 Geometry probes (numbers behind the frame, AGENTS.md:177)

- `view._body.virtual_size` vs `view._body.size` — horizontal scroll is EXPECTED
  here (unlike the main screen where it is a bug); assert `virtual >= actual` on
  wide orgs and that it is NOT triggered on scenario 1.
- `app.screen.virtual_size` vs `app.screen.size` — the SCREEN must still not
  scroll (the chart scrolls inside its body; the dock is pinned) — reuse the
  `AGENTS.md:185` invariant.
- Node non-overlap: a pure-layout unit test asserts, for every pair of nodes on
  the same level, `|x_a - x_b| >= w_a/2 + w_b/2 + H_GAP` (the property the
  prototype demonstrates) — this is the algorithmic correctness check that a
  screenshot cannot make precise.
- Parent-centering: assert each parent's `x` equals the midpoint of its first
  and last child `x` (within rounding) — the Buchheim invariant.
- Settle check (`AGENTS.md:194`): two consecutive frames after open must be
  identical (no reflow-after-paint), and after a zoom keypress the frame settles
  in one pause.

### 7.4 Unit tests (pure modules)

- `org_chart.py`: every §3.3 row as a table test — including the cycle (must
  return in bounded time) and the name-is-both-team-and-agent precedence.
- layout module: non-overlap + centering invariants on random trees.
- `parse_members`: `team:pod:2` grammar, backward compat of bare tokens.
- routing: `_cmd_team` dispatch table (`chart`, `chart <name>`, `chart` with no
  name + attached team, `=chart` escape) — pinned like
  `tests/unit/tui/test_slash_echo.py`.
- `Team` load: an existing `team.yml` with no `kind` loads to `kind="agent"`.

---

## 8. Implementation task list (ordered, for a coder)

Each numbered item is a coherent commit; the whole set is one PR (backend +
TUI) that goes through the full review gate. The delegation *runtime* (§2.3) is
explicitly a SEPARATE follow-up PR, not this one.

1. **Model** — add `kind: Literal["agent","team"] = "agent"` to `TeamMember`
   (`teams.py`), `MAX_ORG_DEPTH`, extend `roster_lines()` to badge team slots,
   extend `parse_members` for the `team:` token prefix. Comment the why
   (backward compat, one-roster rationale). Unit tests incl. old-file load.
2. **Tool authoring** — surface the `team:` grammar in `TeamParams.members`
   description and `_op_show` roster (`team_tool.py`); no schema break
   (`TeamParams` stays `list[str]`).
3. **Classifier extraction** — factor the name→kind half of
   `_resolve_profile_or_specialist` into a shared helper in `agent_profiles.py`;
   repoint the session resolver at it (no behaviour change, guarded by existing
   session tests).
4. **Resolver** — new pure `local_operator/org_chart.py`: `OrgNode` +
   `resolve_org` (§3), full edge-case table tests, cycle-terminates test.
5. **Layout** — new pure `local_operator/tui/org_layout.py` (Buchheim, promoted
   from the prototype): `layout(root) -> positioned nodes`; non-overlap +
   centering invariant tests. NO Textual import.
6. **Renderer** — grid/`Text` builder with the three zoom tiers, boundary
   rules, elbow bus, ghost/cycle/depth/count badges (§4). Pure enough to test
   its string output.
7. **Widget** — `org_chart_view.py` (`OrgChartView`), scroll container (both
   axes), hints, key bindings (§5.2–5.3), class-identified.
8. **tcss** — `.org-chart-view*` and `Screen.org-chart` recede rules
   (clone the `Screen.subagent` block).
9. **app.py wiring** — `_open/_close_org_chart_view`, Esc-chain precedence and
   the approval/ask/clear yields, `_org_chart_view` field.
10. **Routing** — `_cmd_team` subcommand split + `chart` handler + `=` escape
    (§6.1–6.3); bare `/team chart` → attached team (§6.3).
11. **Picker** — first-slot `chart` subcommand row + second-slot team list in
    `on_argument_query_opened` / `_team_choices`, two-level tracking à la `/mcp`
    (§6.4).
12. **Docs** — `guide://teams` gains a "nested teams / orgs" section and the
    `/team chart` grammar incl. the `=` escape; update the `/team` command
    description.
13. **Validation** — `scripts/org_chart_shot.py` + the §7.2 matrix captured and
    viewed, §7.3 geometry probes asserted, all §7.4 unit tests, full gate
    (flake8, black 26.1.0, isort, pyright, unit suite incl. TUI env).

**Follow-up PR (separate review):** the runtime that lets a manager delegate to
a nested team's manager — spawning a sub-manager child that `attach_team`s the
sub-team and re-briefs its members (§2.3). The chart ships first and renders the
*declared* org; the delegation wiring makes the declared org *executable*.

---

## 9. Risks to watch during rollout

- **Old-file load regression.** The one change that could silently drop teams is
  the model field. Mitigation: the `kind` default + an explicit "loads a
  fieldless `team.yml`" test (task 1) — verify against a real pre-existing
  team dir before merge.
- **Picker collision regressions.** The `/team ` list is a well-worn surface;
  the two-level change touches `on_argument_query_opened` and editor tracking.
  Mitigation: before/after picker screenshots (#12) and the routing dispatch
  test. Watch specifically that talking to an ordinary team (`/team foo do x`)
  is byte-for-byte unchanged.
- **Escape-hatch discoverability.** `=chart` is obscure; a user with a team
  named `chart` may not find it. Mitigation: when `_cmd_team` sees `chart` as
  first token AND a team named `chart` exists, the chart handler prints a one-
  line note ("a team is also named 'chart'; talk to it with `/team =chart …`").
- **Layout blow-up on pathological orgs.** Very wide orgs make a very wide
  canvas. Mitigation: `MAX_ORG_DEPTH`, the outline tier, and horizontal scroll;
  the layout is O(n) so width, not time, is the only cost, and scroll absorbs
  width.
- **Scope creep into the delegation runtime.** The temptation is to wire
  execution while touching the model. Hold the line: chart PR renders the
  declared org and tags it `(declared)`; runtime is its own reviewed PR.
- **Terminal scroll semantics.** The chart body scrolls both axes while the
  screen must not — the exact two-cell-width / stray-scrollbar class of bug
  `AGENTS.md:172` documents. Mitigation: the §7.3 screen-virtual-size probe on
  every scenario, not just the wide ones.

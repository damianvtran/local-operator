# Composer UX: name-autofill for `/team`·`/agent`, and slash-command syntax highlighting

Design for two related composer changes. **No production code here** — this is
the seam the coder implements. All line numbers are against the worktree
`/tmp/lop-composer-ux` (branch `dev-composer-ux`, cut from `origin/main`) as
read during this design pass; treat them as "the method named X near line N",
not as offsets to patch blindly.

Ground-truth note: the manager's brief cited older line numbers (editor
`_run_argument` ~1720, `_apply_command` ~1654, etc.). The **methods are all
real and unchanged in behaviour**; only the numbers drifted. Current numbers
are used throughout below. Nothing in the brief's architecture was found wrong.

---

## 0. What the code actually does today (verified)

**Registry** (`local_operator/tui/autocomplete.py`):
- `ArgumentMode` enum (lines 29–58): `NONE` / `OPTIONAL` / `REQUIRED`. Its
  docstring is explicit that the mode answers two questions: does a space open
  the value list, and may Enter on the command ROW also send it.
- `SlashCommand` frozen dataclass (lines 67–112) with kw-only `echo` (98) and
  `arguments` (107). `.names` property (109–112) = primary + aliases.
- `slash_command_for(text)` (526–543): the ONE resolver from a typed line to a
  registry entry, alias-aware, case-insensitive. This is the "is this a
  recognized command" oracle the highlighter will reuse.

**team/agent entries** (`local_operator/tui/app.py`):
- `/team` (502–507) and `/agent` (515–522), both `ArgumentMode.OPTIONAL`, both
  with aliases (`teams` / `agents`).
- Argument rows filled in `on_argument_query_opened` (9481–9558): `team`/`teams`
  → `_team_choices()` (9530–9532), `agent`/`agents` → `_agent_choices()`
  (9534–9536).
- Dispatch: `_run_slash_command` (7307) parses `arg = parts[1].strip()` (7327),
  so any trailing space on the name is already collapsed before it reaches
  `_cmd_team` (2829) / `_cmd_agent` (3027). Both `partition(" ")` the arg into
  `name` + `request`; **empty request → attach-only + "ready" notice** (team
  2884–2888; agent falls through attach then notices). **Confirmed: `/team foo `
  + Enter already means attach-only. No dispatch change is needed.**

**Editor completion** (`local_operator/tui/widgets/editor.py`):
- Key handler (1088–1147): on Tab/Enter with a highlighted row, computes
  `unambiguous = _picker_choice_is_unambiguous(name)` (1108) BEFORE completing,
  then:
  - argument mode → `_resolve_argument(name, key, unambiguous)` (1110)
  - command-word mode, unambiguous → `_apply_command(name)` (1112), and if
    `enter and not opens_a_list(name)` → `_submit()` (1120–1121)
  - command-word mode, Tab → `_apply_command(name)` (1125)
  - command-word mode, ambiguous Enter → grow common prefix (1126+)
- `_apply_command` (2317): argument mode → destructive `_complete_argument`
  else `_run_argument` (2339–2344); command-word mode → writes `/{name} ` and
  moves caret to end (2350–2351).
- `_resolve_argument` (2353): Tab or ambiguous → `_complete_argument`; else
  `_run_argument` (2362–2365).
- `_complete_argument` (2367): replaces the argument tail with `name`, **NO
  trailing space** (2380), because a space terminates the arg and closes the
  list.
- `_run_argument` (2383): `_complete_argument` then `_submit` (2390–2391).
- `set_commands` (887) derives `_argument_commands` (896) and
  `_required_argument_commands` (902) from the registry. `opens_a_list` (909)
  reads `MODEL_COMMANDS` (566) ∪ `_required_argument_commands`.
- Render: `render_line` (709) → placeholder branch, else
  `_paint_markers(super().render_line(y), y)` (748). `_paint_markers` (1664)
  post-processes the finished `Strip` via `_overlay`/`Segment.apply_style(...,
  post_style=...)` (1694–1697). `_marker_cells` (1566) computes which cells to
  repaint, with a hot-path bail `if "[" not in line: return []` (1595–1599).
- `COMPONENT_CLASSES` (578–581): `text-area--attachment-marker[-selected]`
  (named `text-area--image-marker[-selected]` when this document was written;
  renamed when the chip gained a second payload shape). tcss at
  `local_operator/tui/local_operator.tcss` 459–467, using `$lo-signal`,
  `$lo-tint-attach`, `$lo-fg`, `$lo-tint-attach-hi`.

**Empirically verified** (ran `slash_argument`/`argument_suggestions` against
the real functions): after we complete to `/team frontend-guild ` the picker is
in ARGUMENT mode with an **empty match set** (list closes), and a subsequent
blank Enter — with `is_open()` false and `is_pending()` false once the query is
non-`None` — falls through to `_submit()`. Typing a message (`/team fg fix the
bug`) keeps `slash_argument` non-None with zero matches, so the list stays
closed and Enter submits the whole line. This is the behaviour ask #1 wants,
and it already emerges from the tokenizer — we only have to make completion
INSERT the trailing space for these commands.

---

## A. Interaction spec (ask #1)

Two command *classes* from the completion's point of view:

- **ENUM-tail** (`login`, `logout`, `effort`, `approvals`, `theme`, `mcp`,
  `credential`, and the model commands): the argument IS the command. Current
  behaviour is correct and must not change.
- **NAME+message** (`team`, `teams`, `agent`, `agents`): the argument is a NAME
  optionally followed by a free-text message. New behaviour: complete the name,
  add a trailing space, leave caret after it, do NOT submit. A second Enter on
  the blank tail does the attach-only switch (existing dispatch).

| Gesture on an argument row | ENUM-tail (unchanged) | NAME+message (`/team`,`/agent`) — NEW |
|---|---|---|
| **Tab** | `_complete_argument`: fill name, no space, list stays (`_resolve_argument` 2362) | `_complete_name_argument`: fill name **+ trailing space**, caret at end, list closes (empty match set). No submit. |
| **Enter, ambiguous** (fuzzy pick, >1 match) | `_complete_argument` (2363); 2nd Enter runs it | Same as Tab: fill name **+ space**, caret at end, no submit. (The name is now exact/one-match, but we still do not submit — for NAME+message "one match" is not "run", it is "ready for the message".) |
| **Enter, unambiguous** (arrowed, typed-in-full, or single match on non-destructive list) | `_run_argument`: complete + `_submit` (2365) | `_complete_name_argument`: fill name **+ space**, caret at end. **No `_submit`.** |
| **Click on row** (`_apply_command` arg branch 2339–2343) | non-destructive → `_run_argument` (run); `/logout` → `_complete_argument` (fill, wait) | `_complete_name_argument`: fill name **+ space**, caret at end. No submit. |
| **Enter on blank tail** (`/team foo ˽`, caret after space, list closed) | n/a | Not a picker gesture — picker is closed, so the key handler's `if key == "enter": self._submit()` (1167–1171) fires. Dispatch collapses `arg` → attach-only. ✅ existing path. |
| **Enter after typing a message** (`/team foo fix bug`) | n/a | Picker closed (zero matches), Enter → `_submit()` → `_cmd_team` sends the message. ✅ existing path. |

Key point for the coder: the ONLY new insertion behaviour is "name **+ trailing
space**, no submit" for the four NAME+message words. Everything after the space
(blank-attach, message-send) is already handled by the tokenizer closing the
list and the existing dispatch collapsing the arg. **No change to
`_cmd_team`/`_cmd_agent`/`_run_slash_command`.**

---

## B. Declaration mechanism + exact methods to change

### B.1 How the editor learns which commands are NAME+message

Follow the established **registry-declares / editor-reads** idiom (same shape as
`arguments` → `_argument_commands`). Two candidate mechanisms:

**Option 1 — new `ArgumentMode` member `NAME`.** Add `NAME = "name"` to the
enum (autocomplete.py 29–58), set `/team` and `/agent` to it.
- Pro: one field already means "what is this argument"; NAME+message is exactly
  a third answer to that question.
- **Con (disqualifying): blast radius.** `arguments` is read in four places
  beyond the editor: `mobile/daemon.py` 563–566 projects `cmd.arguments.name.
  lower()` into the phone's slash catalogue (a NEW string `"name"` leaks onto
  the mobile surface silently), and `set_commands` (896/902) plus `opens_a_list`
  currently branch on `is not NONE` / `is REQUIRED`. A NAME member forces a
  re-audit of every `arguments is/==` site to decide whether NAME counts as
  "opens a list" (it does — it offers team/agent rows) and "can send when bare"
  (it does — bare `/team` lists). That is real semantic surgery on a shared
  enum for a two-command need. Also `on_argument_query_opened` already special-
  cases the command word by name, so the enum member buys nothing there.

**Option 2 — editor-local `NAME_ARGUMENT_COMMANDS`, derived from the registry,
mirroring `MODEL_COMMANDS`. RECOMMENDED.**

`MODEL_COMMANDS` (editor.py 566) is the exact precedent: a small class-level
tuple of command words (with aliases spelled out) that the editor reads to give
a subset of commands special completion behaviour, WITHOUT touching the shared
registry enum. `/team` and `/agent` are the same kind of local exception.

Concretely — two viable sub-forms; pick the first:

- **B.2a (recommended): a class constant, aliases spelled out**, exactly like
  `MODEL_COMMANDS`:
  ```
  #: Commands whose ARGUMENT is a NAME followed by a free-text message
  #: (`/team <name> <request>`, `/agent <name> <message>`). Completing the
  #: name adds a trailing space and leaves the caret after it so the user can
  #: type the message; it does NOT submit. A blank Enter after the space is the
  #: attach-only switch (handled by the app's dispatch, which strips the arg).
  #: Both spellings, because the alias is itself a runnable command — same
  #: reason MODEL_COMMANDS lists `models`.
  NAME_ARGUMENT_COMMANDS = ("team", "teams", "agent", "agents")
  ```
  Place it next to `MODEL_COMMANDS`/`DESTRUCTIVE_COMMANDS` (566–572). This is
  the least code, matches the nearest precedent exactly, and keeps the whole
  change inside the editor.

- **B.2b (alternative, only if a reviewer objects to a second hand-kept
  tuple): a boolean field `name_argument: bool = field(default=False,
  kw_only=True)` on `SlashCommand`**, and derive
  `self._name_argument_commands` in `set_commands` (887) the same way
  `_argument_commands` is derived (896). This keeps the fact on the registry
  entry (closer to `echo`'s "policy next to the command it governs" argument)
  at the cost of a new field. It does NOT touch `ArgumentMode`, so it avoids
  Option 1's mobile/`opens_a_list` blast radius. Prefer B.2a for parity with
  `MODEL_COMMANDS`; fall back to B.2b if the team's convention has shifted
  toward registry fields since `MODEL_COMMANDS` was written. Either way the
  editor reads a lowered-name membership test — the call sites in B.3 are
  identical.

Add a predicate mirroring `opens_a_list` (909):
```
def _is_name_argument_command(self, name: str) -> bool:
    return name.lower() in self.NAME_ARGUMENT_COMMANDS   # or self._name_argument_commands
```

### B.3 Methods that change (all in editor.py)

1. **New method `_complete_name_argument(self, name)`** — sibling of
   `_complete_argument` (2367). Same tail-replacement math, but appends a
   trailing space and moves caret to end:
   ```
   argument = slash_argument(self.text, self._argument_commands)
   if argument is None:
       return
   self.text = f"{self.text[: len(self.text) - len(argument)]}{name} "
   self.move_cursor(self._end_of_buffer())
   ```
   Docstring must state WHY it differs from `_complete_argument` (the space is
   load-bearing: it terminates the name so the arg list closes, and it opens
   the message tail the user now types — the inverse of `_complete_argument`'s
   "no space so the matcher keeps matching"). Setting `self.text` funnels
   through `load_text` → `_sync_picker` (2158–2159), which re-derives the picker
   as ARGUMENT-mode-with-empty-matches (verified), so the list closes on the
   same keystroke.

2. **`_resolve_argument` (2353)** — branch on the command class up front:
   ```
   if self._is_name_argument_command(self._argument_command):
       self._complete_name_argument(name)
       return
   # existing behaviour for enum-tail commands:
   if key == "tab" or not unambiguous:
       self._complete_argument(name)
       return
   self._run_argument(name)
   ```
   This makes Tab AND Enter (ambiguous or not) all route to
   `_complete_name_argument` for team/agent, which is exactly the spec table:
   for a NAME+message command neither key ever submits.

3. **`_apply_command` (2317)**, argument branch (2339–2344) — add the same
   guard before the destructive/run split:
   ```
   if self._picker.mode is PickerMode.ARGUMENT:
       if self._is_name_argument_command(self._argument_command):
           self._complete_name_argument(name)
           return
       if self._argument_is_destructive():
           self._complete_argument(name)
           return
       self._run_argument(name)
       return
   ```
   Covers the mouse-click path.

4. **No change** to `_run_argument`, `_complete_argument`, the key handler
   (1088–1147 — it already delegates to `_resolve_argument`/`_apply_command`),
   `_run_slash_command`, `_cmd_team`, `_cmd_agent`, or the mobile projection.

`_argument_command` is `Optional[str]`; the predicate must handle `None`
(`(name or "").lower()`), because `_apply_command`'s argument branch reads it
while a list is open — it is set, but be defensive.

---

## C. Highlighting design (ask #2)

### C.1 Render seam

Reuse the **exact** `_paint_markers` seam. `render_line` (748) already does
`return self._paint_markers(super().render_line(y), y)`. Add a second
post-process pass for slash highlighting. Two integration choices:

- **C.1a (recommended): a sibling method `_paint_slash(strip, y)`, chained
  inside `render_line`** so the finished strip is
  `self._paint_slash(self._paint_markers(super().render_line(y), y), y)`.
  Ordering: markers first, slash second, OR slash first — they never overlap
  (a slash line has no `[Image …]` marker: the command word is the first token
  and the marker grammar opens with `[`), so order is immaterial for
  correctness; put slash LAST so its bail is the cheap one on prose lines.
- C.1b: fold into `_paint_markers`. Rejected — `_paint_markers` is documented
  as "repaint the MARKER cells", its `cells`/`edges`/`styles` machinery is
  marker-specific, and mixing a second token grammar into it muddies a method
  whose comments carefully explain the chip rules. A named sibling keeps each
  pass single-purpose, matching the file's style.

### C.2 What gets highlighted, and what "recognized" means

Compute on **document line 0 only** (the command lives on the first non-blank
line; multi-line = message body, no command). Highlight at most two token runs:

1. **The leading `/command` token** — from the `/` through the end of the first
   whitespace-delimited word. "Recognized" ⇔ `slash_command_for(line)` returns
   non-None (the same resolver dispatch uses, so the highlight cannot claim a
   command the app would reject). This covers the WHOLE slash surface (ask #2's
   "not just team/agent").
   - Recognized → **command style** (`text-area--slash-command`).
   - Not recognized (`/xyz`, a typo) → **unrecognized style**
     (`text-area--slash-unknown`), a muted/neutral treatment so the user sees
     "this leading slash is not a command and WILL be sent as text". See D for
     why we highlight the unknown case at all.

2. **The argument NAME token** — only when the command is one whose argument is
   a name we can validate cheaply, i.e. a NAME+message command
   (`_is_name_argument_command`) AND the typed name matches a known team/agent.
   - Matched name → **argument-name style** (`text-area--slash-argument`).
   - Partial/typo name → NO highlight (leave prose colour). Do not paint an
     "unknown name" — the picker already shows/【doesn't show】 a match, and a
     half-typed name is a normal in-progress state, not an error.

   Only the NAME (first token after the command word) is ever highlighted;
   the free-text message tail is prose and must stay prose — that is the whole
   point of the feature (show the user what is command vs. message).

### C.3 Cheap "recognized" on the hot path

`render_line` runs every keystroke-frame (the marker code calls this out at
1595). Keep the pass O(1)-ish:

- **Bail exactly like the marker pass.** First cell of the method:
  ```
  if y != 0 and self.scroll_offset.y == 0:  # only the first screen row can hold the command
      ... actually: resolve the DOCUMENT line for screen row y, bail if it isn't line 0
  line = self.document.get_line(0)
  if not line.lstrip().startswith("/"):
      return strip
  ```
  A composer whose first non-blank line does not start with `/` (the
  overwhelming common case — prose, empty, bang-mode `!`) returns the strip
  untouched before any tokenizing. This mirrors `_marker_cells`'s `if "[" not
  in line` guard (1595).

- **Command recognition is a dict/loop lookup**, not a regex sweep:
  `slash_command_for(line)` splits the first token and does a `next(... in
  entry.names ...)` over ≤ ~30 commands (526–543). Cheap. Cache is unnecessary
  at this size; if a reviewer wants it, memoize on `(line0, commands_version)`
  — but measure first, do not add cache complexity speculatively.

- **Name recognition needs a set of known team/agent names.** DO NOT call
  `_team_choices()`/`_agent_choices()` from the render path — those hit the
  registry (`list_teams`, `_agent_profile_rows`) and are app-side (app.py 2773,
  2967), not reachable cheaply from the widget every frame. Instead, **push the
  name set into the editor when the argument list opens.** The app already
  computes the rows in `on_argument_query_opened` (9530–9536); have it also call
  a new `editor.set_argument_names(names: frozenset[str])` (or fold the names
  into the existing `picker.set_choices` path and read them back off the picker
  — the picker already holds `_choices`, and `_apply_command`-time code reads
  `self._picker.suggestions()`). Simplest and lowest-risk:
  - Add `editor.set_name_choices(frozenset(c.name for c in choices))` called
    right beside `picker.set_choices(self._team_choices())` /
    `_agent_choices()` (9531/9535). Store on `self._name_choices: frozenset[str]
    = frozenset()`; clear it whenever the argument command changes to a
    non-name command (in `_sync_picker`, alongside the existing
    `set_choices([])` at 2186, when leaving a name command). The render pass
    then does `name_token.lower() in self._name_choices` — a frozenset hit, O(1).
  - Rationale: the render path must not do I/O or registry walks; the app owns
    the truth and already visits it when the list opens, so it hands the widget
    a cheap immutable snapshot. This is the same "app fills the rows when the
    list opens" contract `on_argument_query_opened` already documents (9481–9493).

  Caveat: the name set is only populated once the argument list has opened at
  least once (the space after `/team`). If the user types `/team frontend `
  fully by hand without the list ever having filled, `_name_choices` may be
  empty and the name goes un-highlighted. Acceptable: highlighting is an
  affordance, not a gate, and the list fills on the space in practice. Document
  this as a known limitation rather than fetching on the render path.

### C.4 Painting the cells

Model the pass on `_paint_markers` (1664) exactly:
- Resolve document line 0's screen-row extent (reuse the `wrapped_document` /
  `offset_to_location` machinery — a long `/team … message` soft-wraps; the
  command word is on the first wrapped row, so in practice only `y==0`'s section
  carries the command token, but the name token can wrap. Keep it correct by
  mapping document columns → screen x via `wrapped.location_to_offset`, same as
  the marker code at 1637/1646).
- Build `(x_start, x_end, kind)` runs for the command token and (if matched) the
  name token, add `gutter`.
- `edges = sorted({0, width} | {x for …})`, `strip.divide`, and for each piece
  overlay the component style via `_overlay(piece, style)` (1695–1697,
  `post_style`), because — same reason as the chip — TextArea's segments carry
  explicit fg/bg and a base style would be discarded.
- Styles resolved via `self.get_component_rich_style("text-area--slash-*")`
  (like 1682–1683).

### C.5 Placeholder / caret-at-col-0 interaction

`render_line`'s placeholder branch (731–747) returns BEFORE `_paint_markers`
when `not self.text`. So an empty buffer never reaches the slash pass — good,
nothing to highlight. The moment the user types `/` there is text, the
placeholder branch is skipped, and the normal `super().render_line` + post-
process path runs. The caret cell: `_marker_cells` deliberately EXCLUDES the
caret cell (1582–1585) because the chip is opaque; for slash highlighting the
overlay is a foreground/soft-background tint, not an opaque chip, so it MAY
include the caret cell safely — BUT to avoid fighting the cursor style
(`text-area--cursor`, tcss 420–423, which swaps fg/bg), the safest choice is to
**style foreground only** for the command word (see C.6) so the caret's inverse
still reads on top. If a background tint is used, exclude the caret cell exactly
as `_marker_cells` does (read `self.selection.end` / `_draw_cursor`, 1607–1615).
Recommend **foreground-only** styles to sidestep the whole caret/selection
interaction — a coloured command word reads as "recognized" without needing a
ground behind it, and it composes cleanly with selection (`text-area--
selection`, tcss 425–427) and cursor overlays.

### C.6 New COMPONENT_CLASSES + tcss

Add to `COMPONENT_CLASSES` (578–581):
```
"text-area--slash-command",     # recognized /command word
"text-area--slash-argument",    # recognized team/agent NAME
"text-area--slash-unknown",     # leading /word that is NOT a command
```
tcss (`local_operator/tui/local_operator.tcss`, beside the attachment-marker block
459–467), foreground-only, using existing `$lo-*` semantic tokens (theme.py
32–43 / 88–98 define them for both dark and light ramps):
```
/* A recognized slash command, tinted so the user sees the leading token is a
 * command and will NOT be sent as message text. `signal` is the ramp's
 * file/reference cool hue — already used by the attachment chip — and reads as
 * "structured token", not the accent green that means "a turn is live". */
Editor .text-area--slash-command {
    color: $lo-signal;
    text-style: bold;
}
/* The recognized team/agent NAME after the command word. `string` (the
 * success/string green) distinguishes the resolved argument from the command
 * word without a second ground. */
Editor .text-area--slash-argument {
    color: $lo-string;
}
/* A leading /word that resolves to NO command: dimmed so the user sees it is
 * inert text, not a recognized command. `muted`, not a danger colour — an
 * unknown slash is a typo-in-progress, not an error to alarm about. */
Editor .text-area--slash-unknown {
    color: $lo-muted;
}
```
Colour choices are a starting point for the design round — the designer sub-
agent owns the final hues/contrast (this IS a user-visible change, so a
`### Design review` round is required). Do NOT reuse `$lo-accent` (reserved:
"a turn is live", per the tcss header comment lines 35 and 367).

---

## D. Risks / edge cases

1. **Multi-line buffers.** The command only exists on the first non-blank line;
   `slash_context`/`slash_argument` both return None once a newline follows the
   word (command_picker.py 205–207, 232–238). The highlight pass MUST gate on
   document line 0 and bail if the first non-blank line has a newline after the
   command token — otherwise a `/team foo\nmore text` message body could get a
   stray command highlight. Mitigation: resolve on `self.document.get_line(0)`
   and only paint the command token if line 0 is the first non-blank line;
   paint the name token only while the picker's tokenizer still considers it an
   argument (i.e. no newline yet). Simplest correct rule: **only highlight when
   `slash_command_for(line0)` is non-None AND the buffer is single-line OR the
   command/name tokens are entirely on line 0** — reuse the same single-line
   discipline the tokenizers already enforce.

2. **`/btw` off-record.** `/btw` IS a recognized command (app.py 451), so its
   leading token highlights as a command — correct and desirable (the user
   should see `/btw` is recognized). Its argument is free text (a question), NOT
   a name, so `_is_name_argument_command("btw")` is false and nothing after
   `/btw ` is highlighted. No special-casing needed. The aside's own composer
   is a separate concern (it sets `set_records_history(False)`, etc.) and does
   not change the highlight rules.

3. **Unknown command → highlight as "unrecognized"?** YES, with the muted
   `text-area--slash-unknown` treatment. Rationale: the feature's stated goal is
   "so the user can see the command is recognized and won't be sent as part of
   the message." The dual is equally valuable: a `/teem foo` typo that is NOT a
   command should visibly read as inert text so the user is not surprised when
   the whole line is sent as a prompt. This is cheap (it is the else-branch of
   the same `slash_command_for` call) and it is the honest signal. If the
   designer round finds the muted treatment too noisy on every half-typed
   `/`, fall back to "highlight recognized only, leave unknown as prose" — but
   start with the unknown treatment; it is the more informative default.
   Note: while the command PICKER is open (`/te` mid-type), the leading word is
   a prefix, not yet a full command. `slash_command_for` requires an exact
   name-in-`names` match, so `/te` resolves to None and would flash "unknown"
   on every keystroke of typing a command. **Mitigation: suppress the unknown
   (not the recognized) highlight while the command picker is open**
   (`self._picker.is_open()` and mode is COMMAND) — a word being actively
   picked is not yet "unrecognized", it is "in progress". Recognized-command
   and name highlights are unaffected.

4. **Partially-matching name.** No highlight (C.2). A half-typed name is normal;
   painting it "unknown" would flicker on every keystroke. Only an exact set
   membership (`_name_choices`) paints the name.

5. **Interaction with attachment-marker overlays.** A slash line has no marker
   (the marker grammar opens with `[` and a command opens with `/`; the command
   word and name are on line 0, a pasted image marker would sit in the message
   tail which is not highlighted). The two passes therefore never contend for
   the same cells. Chaining order (C.1a) is safe either way; keeping slash LAST
   means the cheaper bail runs on the common prose path. If a user pastes an
   image INTO the name position (pathological), the marker pass owns those
   cells and the name pass simply won't find a set-member match — no conflict.

6. **Aliases.** `/teams`/`/agents` must highlight identically to
   `/team`/`/agent`. `slash_command_for` resolves aliases (529–531) and
   `NAME_ARGUMENT_COMMANDS` lists all four spellings — covered.

7. **Name set staleness.** `_name_choices` is a snapshot from the last list
   open (C.3). If a team is created mid-session and the user hand-types its name
   before ever opening the list, it goes un-highlighted until the list next
   opens. Acceptable, documented limitation — never fetch on the render path.

8. **Theme switch.** Because styles are component classes reading `$lo-*` (not
   Python hexes), a `/theme` switch repaints them for free — same property the
   attachment-marker comment (575–577) relies on. No extra work.

---

## E. Test + visual-validation plan

Run everything with `PYTHONPATH=/tmp/lop-composer-ux .venv/bin/python`; TUI
tests/shots need `env -u NO_COLOR TERM=xterm-256color`.

### E.1 Unit tests (extend `tests/unit/tui/test_command_picker.py`)

That file already has the harness (`_PickerHost`/`CSS_PATH = TCSS_PATH`, real
`Editor(commands=SLASH_COMMANDS)`, lines 85–108) and the exact patterns:
`test_enter_completes_then_submits` (452), `test_tab_completes_without_
submitting` (423), `test_click_selects_and_completes_without_submitting` (492),
`test_enter_runs_an_unambiguous_provider` (991),
`test_arrowing_onto_a_provider_lets_enter_run_it` (1029),
`test_clicking_a_login_row_runs_it` (1081). Add, following those:

Ask #1 (drive real keys through the mounted editor; the host records
`submissions`):
- `test_enter_on_a_team_row_fills_the_name_and_a_space_without_submitting` —
  open `/team `, fill choices with a fake registry (mirror the provider-list
  tests' choice injection), Enter on a row → buffer is `/team <name> ˽`, caret
  at end, `host.submissions == []`.
- `test_tab_on_a_team_row_fills_name_and_space` — same, Tab.
- `test_click_on_a_team_row_fills_name_and_space_without_submitting` — mouse
  path through `_apply_command`.
- `test_arrowing_onto_a_team_row_still_does_not_submit` — the unambiguous
  branch must NOT run for a name command (the key difference from providers).
- `test_blank_enter_after_a_completed_team_name_submits_attach_only` — buffer
  `/team <name> ˽`, picker closed, Enter → `host.submissions == ["/team <name> "]`
  (attach-only; assert the arg collapses — a dispatch-level test can assert
  `_cmd_team` took the bare-name branch, or assert the submitted text).
- `test_typing_a_message_then_enter_sends_the_whole_line` — `/team <name> fix
  it` + Enter → submitted text carries the message.
- `test_agent_row_behaves_like_team` — one parametrized mirror for `/agent`.
- **Regression guard:** `test_login_row_still_runs_on_enter` / `test_logout_
  row_still_fills_and_waits` must stay green (they already exist ~991/1097) —
  confirm the enum-tail path is untouched.

Ask #2 (highlight): these are best as **render-strip assertions**, not just
visual — assert the component style lands on the right cells.
- `test_recognized_command_word_gets_the_slash_command_style` — set text
  `/team`, render line 0, assert the `/team` cells carry
  `text-area--slash-command`'s style (probe via the rendered `Strip` segments'
  styles, same way marker tests inspect chips if they exist; otherwise assert
  `get_component_rich_style` is applied by checking segment styles).
- `test_unknown_command_word_gets_the_unknown_style_when_picker_closed` —
  `/notacommand ` (space closes the command picker), assert `slash-unknown`.
- `test_command_word_is_not_flagged_unknown_while_the_picker_is_open` — `/te`,
  picker open, assert NO `slash-unknown` overlay.
- `test_recognized_team_name_gets_the_argument_style` — push a name set
  (`set_name_choices({"frontend-guild"})`), text `/team frontend-guild fix it`,
  assert the NAME cells carry `slash-argument` and the message tail does NOT.
- `test_partial_name_is_not_highlighted` — `/team front`, assert no
  `slash-argument` overlay.
- `test_message_body_after_the_name_stays_prose` — assert the `fix it` cells
  carry no slash component style.
- `test_multiline_buffer_does_not_highlight_a_second_line` — `/team foo\nmore`,
  assert line 1 has no overlay.

Also add a **mobile-projection guard** if B.2b (new field) is chosen:
`mobile/daemon.py` 558–571 already tolerates it (it only reads `.arguments`),
but if a `name_argument` field is added, add nothing to the mobile payload —
assert the projected dict is unchanged (there is an existing mobile projection
test to extend). If B.2a (editor constant) is chosen, no mobile test is needed.

### E.2 Visual validation (required — user-visible change)

Per AGENTS.md "Visual validation": use the REAL `OperatorApp` (the `_PickerHost`
in `test_command_picker.py` DOES load `TCSS_PATH`, so it is acceptable for
these composer stills; the app is even better and is what the designer round
should use for the mobile/full-chrome context). Script with `run_test` +
`app.save_screenshot(path)`, then RENDER the SVG and view it. Capture
**before/after pairs** — before-frames FIRST from a throwaway worktree
(`git worktree add --detach`), never `git stash`.

Frames to capture (before = current build, after = with the change):

1. **`/team ` list open** — the argument picker showing team rows. Before/after
   should be identical for the list itself (this frame proves we didn't regress
   the picker); the point is the composer buffer.
2. **After selecting a team name** — buffer shows `/team frontend-guild ˽` with
   caret after the space. BEFORE (current) this frame does not exist the same
   way (current build would have SUBMITTED and cleared the buffer / attached);
   the after-frame is the new resting state. Capture the after-frame showing the
   caret parked after the space and the list closed.
3. **With a typed message** — `/team frontend-guild fix the flaky test`, showing
   the command word tinted, the name tinted, and the message in prose colour.
   This is the money shot for ask #2.
4. **Unrecognized command** — `/notacommand hello`, showing the muted
   `slash-unknown` treatment on the leading token and prose after. Before-frame:
   plain prose (no highlight). After: muted token.
5. **Recognized non-name command** — e.g. `/compact` or `/usage`, showing the
   command word tinted and (for `/usage`) no argument highlight. Confirms the
   whole-surface scope, not just team/agent.
6. **`/agent <name> <message>`** — one mirror frame for the agent command.

For each, back the pixels with geometry per AGENTS.md §4 only if a reflow is
suspected (the highlight is foreground-only, so no width/scrollbar change is
expected — but confirm `app.screen.virtual_size == size` on the populated
frame, since any accidental width change is a bug on this docked composer).

### E.3 Gates (all must be clean, per AGENTS.md)

```
PYTHONPATH=/tmp/lop-composer-ux .venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check local_operator tests
uvx isort --check-only --profile black local_operator tests
PYTHONPATH=/tmp/lop-composer-ux .venv/bin/python -m pyright --pythonpath .venv/bin/python .
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=/tmp/lop-composer-ux .venv/bin/python -m pytest tests/unit/tui -q
```
(Note: `test_slash_echo.py` pins each command's `echo` flag but NOT `arguments`
or any new field, so neither B.2a nor B.2b trips it — verified. If B.2b adds a
field, no echo-test change is needed.)

---

## Summary of the recommendation

- **Ask #1:** add editor-local `NAME_ARGUMENT_COMMANDS` (mirroring
  `MODEL_COMMANDS`), a new `_complete_name_argument` sibling of
  `_complete_argument`, and a guard at the top of `_resolve_argument` (2353) and
  the argument branch of `_apply_command` (2339) routing the four team/agent
  spellings to it. No dispatch/registry/mobile changes — the tokenizer already
  closes the list on the trailing space and the existing `arg.strip()` collapses
  the blank tail to attach-only.
- **Ask #2:** add a `_paint_slash` post-process pass chained after
  `_paint_markers` in `render_line` (748), gated on line-0/leading-`/` like the
  marker hot-path bail, using `slash_command_for` for command recognition and an
  app-pushed `_name_choices` frozenset for name recognition (never fetch on the
  render path). Three new component classes + foreground-only tcss.
- Smallest change that solves it: yes — both asks stay inside `editor.py` +
  three tcss rules, reuse two existing seams (`_complete_argument` shape,
  `_paint_markers`/`_overlay`), and add zero new dispatch paths. The one shared-
  surface touch (app pushing name choices) rides the `on_argument_query_opened`
  hook that already exists for exactly this "fill when the list opens" purpose.

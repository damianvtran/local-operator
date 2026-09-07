# Remappable hotkeys for `/new` and `/resume`

**Status:** proposal · **Branch:** `feat/hotkeys` · **Author:** architect
**Verified against:** textual 8.2.8, worktree `lo-hotkeys`
(`.venv/bin/python -c "import local_operator; print(local_operator.__file__)"` →
`/Users/damian/workspace/repos/lo-hotkeys/local_operator/__init__.py`)

Every claim about existing behaviour below cites `file:line`. Every claim about
Textual is a **measurement** from a probe run in this worktree's venv, not a
reading of the docs — the docs are wrong on the decisive point (§D.0).

---

## 0. Summary of decisions

| # | Question | Decision |
|---|---|---|
| a | Schema | `keymap.<action>` literal flat-dotted keys, one per action, value = a Textual key string. Extensible without migration. §A |
| b | New Kind | `Kind.HOTKEY`, with a four-state capture machine owned by `SettingsView`. Esc always cancels and is **never** capturable. §B |
| c | Conflicts | **Warn-and-refuse on hard-reserved; warn-and-allow with a named victim on soft conflicts.** Uses Textual's own `clashed_bindings`. §C |
| d | Mechanism | `Binding(id=...)` + `App.update_keymap()`. **Live re-keying WORKS** — measured. Propagates through the existing `config_watch` → `_on_config_change` path. §D |
| e | Defaults | `ctrl+n` = new session, `ctrl+s` = resume picker. Both **non-priority**. Pending scout confirmation; constraints in §E are binding. |
| f | Tips | Tips become `(template, binding_id)` pairs resolved at render; `TIP_MIN_WIDTH` becomes a **bounded** computation, not a pool measurement. §F |
| g | Testing | Structural invariants over timing. Two pilot tests, one anti-drift test, no timing bound anywhere. §G |

**Three things in the request I push back on**, detailed in §H: the phrase
"one or more keys" (§H.1 — Textual has no chord support and building one is out
of scope), "propagates across sessions" as re-keying *running* sessions (§H.2 —
it does work, but there is a caveat worth naming), and putting the reserved-key
policy in the UI alone (§H.3).

---

## A. Schema

### A.1 The keys

```yaml
values:
  keymap.new_session: ctrl+n
  keymap.resume: ctrl+s
```

**Flat dotted keys**, exactly like `display.*`:

```python
Setting(
    key="keymap.new_session",
    path=("keymap.new_session",),   # ONE element containing a dot
    ...
)
```

This is the `display.*` shape documented at `local_operator/settings_io.py:26-38`
and asserted by `flat_dotted_keys()` (`settings_io.py:1457`). It is **not** the
`tui.*` shape (`settings_io.py:867-868` uses `path=("tui", "sidebar_visible")`,
a genuine nesting).

**Why flat, when `tui.*` is nested.** Two reasons, and the second is the real one:

1. A keymap is a homogeneous open-ended map from action id to key string. A
   nested `keymap:` block invites a reader to hand-write arbitrary sub-keys that
   no `Setting` declares, and `_changed_registry_keys`
   (`config_watch.py:640-659`) only diffs keys the registry knows — so a
   hand-added nested entry would be invisible to propagation. A flat key per
   registered action makes "registered" and "propagated" the same set by
   construction.
2. `settings_io.write_setting` merges into an existing sub-mapping
   (`settings_io.py:1679-1705`) but `ConfigManager._load_config` back-fills
   **missing top-level keys only** (`settings_io.py:18-24`). Flat keys sidestep
   the whole merge question: each is its own top-level leaf.

The cost is honest and small: the config file grows one top-level line per
remappable action rather than one block. `display.*` already pays this eight
times over (`settings_io.py:719-796`).

### A.2 Value format

A **Textual key string**: lowercase, `+`-separated, e.g. `ctrl+n`,
`ctrl+shift+up`, `f5`, `super+b`. This is the vocabulary `OperatorApp.BINDINGS`
already speaks (`app.py:1841-2014`).

Comma means **alternates**, not a sequence — measured:

```
'ctrl+n,ctrl+g' -> active={'ctrl+n': 'keymap.new_session', 'ctrl+g': 'keymap.new_session'}
```

Both keys fire the action. There is no sequence/chord support in Textual at all
(§H.1). **The capture UI stores exactly one key**, but the *schema* tolerates a
hand-written comma list, because Textual does and refusing it would make the
file and the page disagree about a value the runtime honours.

### A.3 Normalization

`textual.keys._normalize_key_list()` maps single characters to key names
(`?` → `question_mark`, `[` → `left_square_bracket`) via `_character_to_key()`
and `KEY_NAME_REPLACEMENTS`. `App.set_keymap` calls it (`App._normalize_keymap`).

**Normalize at the WRITE boundary, not only at apply.** The capture UI already
receives a normalized `event.key` from Textual, so the stored value is normal
by construction on that path. The other two writers — `lop config edit
keymap.new_session ...` and a hand edit — do not go through capture, so
`coerce()` for `Kind.HOTKEY` must run `_normalize_key_list(text.strip().lower())`
before storing. Otherwise the file holds `ctrl+N` and the page displays
`ctrl+N` while the runtime binds a key nobody can press (measured: `'ctrl+N'`
is accepted verbatim and becomes an unreachable binding).

### A.4 Validation — and the bug this is here to avoid

**Textual validates NOTHING.** Measured, and this is the single most important
implementation constraint in the document:

```
'ctrl+'      -> active={'ctrl+':      'keymap.new_session'}
'kontrol+n'  -> active={'kontrol+n':  'keymap.new_session'}
'banana'     -> active={'banana':     'keymap.new_session'}
'ctrl-n'     -> active={'ctrl-n':     'keymap.new_session'}

silent-disable check: original ctrl+n after bogus remap -> []   # action UNREACHABLE
```

A garbage key string does not raise `InvalidBinding`; it silently moves the
binding to a key no terminal can emit, and the default is gone too. That is
exactly the Claude Code pre-v2.1.246 bug the scout flagged, and Textual has it
today. So:

`validate(setting, value)` for `Kind.HOTKEY` returns a message unless every
comma-separated part, after normalization, is a member of the **canonical key
vocabulary**:

```python
_VALID_KEYS = frozenset(m.value for m in textual.keys.Keys)   # 152 members
# plus single printable characters, which are legal keys but not Keys members
```

Reject with a user-facing sentence (`settings_io.py:1568-1571` sets the
standard — the page prints it inline and keeps the editor open):
`"not a key this terminal can send — try ctrl+n, f5, or press a key to capture"`.

**On an unparseable value already on disk**, the resolver (§D.3) **drops that
entry and keeps the shipped default**, and the app prints one notice naming the
key. Never leave the action unreachable. This is `tui/settings.py`'s own rule —
"a missing or unreadable config never breaks the TUI — every lookup falls back
to its default" (`tui/settings.py:3-5`) — and `_cmd_new`'s guarded-reload rule
(`app.py:8737-8757`): user data degrades gracefully, it does not disarm the app.

Note this must be checked **before** `update_keymap`, because Textual will
cheerfully apply the garbage.

### A.5 Extensibility without migration

Adding a third action later is: one `Binding(id=...)` in `app.py`, one `Setting`
in `settings_io.py`, one entry in the action registry (§D.2). No migration —
an absent key reads as its default through `read_setting`
(`settings_io.py:1530-1546`), and `reset_setting` deletes rather than writing a
default (`settings_io.py:1708-1722`), so a config only ever carries what the
user actually chose. That is the "store only overrides" property the scout asked
for, and it falls out of the existing facade for free.

**Constraint to write into the code:** the binding `id` is a persisted config
value in all but name. Renaming `keymap.new_session` orphans every user's
override silently. Put that sentence in the docstring next to the id.

---

## B. `Kind.HOTKEY` — rendering, activation, capture

### B.1 Why a new Kind and not `Kind.TEXT`

`Kind.TEXT` opens an inline editor where every printable key is buffered
(`settings_view.py:2227-2236`). Typing `ctrl+n` into it is impossible: `ctrl+n`
is not printable, so it falls through to the page bindings and moves the cursor
(`settings_view.py:353`). A hotkey row must *listen*, which is the opposite
interaction. `Kind` is explicitly "about the interaction and not about the
Python type" (`settings_io.py:85-88`), so this is what the enum is for.

`key_prompt.py` is **not** reusable: it is an API-*key* paste prompt
(`key_prompt.py:1-39`) that buffers printable characters and masks them. Nothing
in it captures a chord. Do not try to generalize it.

### B.2 The state machine

Four states, owned by `SettingsView` as one nullable attribute
`self._capture: _Capture | None` (mirroring `self._editing`,
`settings_view.py:1490`):

```
        enter / click on a HOTKEY row
IDLE ─────────────────────────────────▶ CAPTURING
                                            │
        first non-reserved key press        │
            ┌───────────────────────────────┘
            ▼
        PENDING ──── enter / click "confirm" ────▶ committed → IDLE
            │                                      (settings_io.write_setting)
            │──── another key press ──▶ PENDING (replaces; last press wins)
            │
            └──── escape ──────────────────────▶ cancelled → IDLE

CAPTURING ──── escape ────────────────────────▶ cancelled → IDLE
```

- **IDLE → CAPTURING** on `enter` or a click on an already-selected row. This is
  `action_activate`'s existing contract — *enter opens, enter accepts, esc
  cancels* (`settings_view.py:1038-1039`) — with a new branch beside the
  `Kind.BOOL/ENUM`, `Kind.READONLY` and `Kind.CASCADE` branches at
  `settings_view.py:1117-1177`. Click reuses `on_click`'s
  select-then-activate-if-already-selected rule (`settings_view.py:2300-2366`).
- **CAPTURING → PENDING** on the first accepted key. The row shows *what was
  detected*, not what the user meant — VS Code's rule, and the honest one,
  because on a terminal that mangles a chord the detected key is the only thing
  that will ever fire.
- **PENDING → committed** on `enter` or a click on the confirm affordance.
  **This is the only state transition that writes config.yml**, matching the
  page's #440 contract that the second `enter` is the only gesture that changes
  config (`settings_view.py:1043-1044`).
- **Any state → cancelled** on `escape`.

A second key press in PENDING **replaces** the captured key rather than being
ignored. A user who fumbles a chord should press the right one, not have to
cancel and re-enter.

### B.3 Escape is not capturable, and neither are the rest

`escape` is the app's `stop` (`app.py:1891`) and the page's `leave`
(`settings_view.py:373`) and the universal cancel of every rung of the Esc
ladder (`settings_view.py:2238-2283`). It is the escape hatch out of capture
mode itself. **It is therefore never capturable, in any state, under any
modifier combination that Textual reports as bare `escape`.**

**We refuse rather than offer an alternative gesture.** The alternatives are
worse: a "hold escape for 2 s" or "press escape twice to bind escape" is a
hidden mode inside a mode, taught nowhere, and the payoff is binding a key that
already means *stop* everywhere in the app. Say so on the row:
`esc cancels — it cannot be bound`.

`ctrl+c` is refused for the same class of reason: it is the app's interrupt and
the first rung of the exit ladder (`app.py:1709-1711`), and
`TranscriptScreen.action_copy_text` raises `SkipAction` specifically so it
reaches that interrupt (`app.py:1696-1721`). Full reserved list in §C.2.

### B.4 How capture actually gets the keys — the mechanism

This is the part that needs care, because app-level **priority** bindings are
dispatched *before* the focused widget and would eat the very keys being
captured. Measured:

```
--- capture OFF ---
shift+tab      -> ['APP-cycle']        # app priority binding wins
ctrl+shift+up  -> ['APP-sw']           # app priority binding wins
```

A widget-level `priority=True` binding does **not** win against an app-level
one (measured: `ctrl+shift+up -> ['APP-priority']` with the focused widget also
claiming it at priority). So the capture widget cannot out-prioritize the app.

**`check_action` is the escape hatch, and it works.** Measured:

```
--- capture ON (App.check_action returns False for hotkey actions) ---
shift+tab      -> [('CAPTURED', 'shift+tab')]
ctrl+shift+up  -> [('CAPTURED', 'ctrl+shift+up')]
ctrl+t         -> [('CAPTURED', 'ctrl+t')]
escape         -> [('CAPTURED', 'escape')]
ctrl+c         -> [('CAPTURED', 'ctrl+c')]
```

So the implementation is:

1. `OperatorApp` grows `self._capture_mode: bool` and overrides `check_action`
   (`DOMNode.check_action`, returns `True` by default) to return `False` for
   **every app-level action** while capture is live.
2. `SettingsView` sets it via a message on CAPTURING entry, clears it on every
   exit route including unmount — the same discipline
   `revert_preview_for_teardown` follows (`settings_view.py:1723-1740`).
3. `App.refresh_bindings()` must be called after flipping the flag; measured
   working in the probe.
4. The capture widget reads keys in `on_key` and `event.stop()`s every one.
   `escape` and the reserved set are handled **inside** that handler — they are
   captured by the widget and interpreted, never allowed to reach their normal
   action.

Point 4 is why the reserved list is enforced in the widget as well as in
`validate()`: while capture is live, `check_action` has disarmed the app, so
`escape` will NOT stop the agent even though it normally would. The widget must
supply that meaning locally, exactly as `KeyPromptBlock` binds escape locally
for the analogous reason (`key_prompt.py:139-149`).

**Risk to watch in rollout (R1):** `check_action` returning `False` for
everything is a blunt instrument. If a bug leaves `_capture_mode` stuck `True`,
the app loses every hotkey — including `ctrl+c` interrupt. Mitigation: the flag
is cleared on `SettingsView.on_unmount` *and* on `_close_settings_view`
(`app.py` `SETTINGS_LAYOUT_CLASS`, `app.py:1259-1261`), and a pilot test asserts
that leaving the page by every route restores `check_action` → `True`.

### B.5 Rendering

Collapsed row, consistent with `_render_value` (`settings_view.py:4497`):

```
  New session hotkey                        ctrl+n
  Resume picker hotkey                      ctrl+s   (default)
```

Display goes through `textual.keys.format_key`, measured:
`format_key('escape') = 'esc'`, `format_key('ctrl+g') = 'ctrl+g'`. Use it so the
page and Textual's own footer vocabulary agree.

Capture states, in the detail row the page already owns (`_paint_detail`,
`settings_view.py:3138`):

```
CAPTURING:  press a key…                       esc cancels
PENDING:    ctrl+shift+n  ·  enter confirms  ·  esc cancels
PENDING+conflict: ctrl+t · takes ctrl+t from "expand/collapse todos" · enter confirms
REFUSED:    esc cancels the capture — it cannot be bound
```

`r` resets to default and already works: `action_reset`
(`settings_view.py:1307-1427`) calls `reset_setting`, which deletes the key.
No new code, and it gives the scout's "reset = delete the override" for free.

---

## C. Conflicts

### C.1 The full reserved set, read from the code

Enumerated by walking every `Widget`/`Screen`/`App` subclass in
`local_operator.tui` plus Textual's own classes (probe:
`/tmp/probe_reserved.py`). Abridged to what a user could plausibly capture:

**App-level (`OperatorApp.BINDINGS`, `app.py:1841-2014`):**
`ctrl+c` `ctrl+l` `f8` `ctrl+b` `f9` `ctrl+shift+up` `ctrl+shift+down`
`super+b` (darwin) `ctrl+f` `escape` `shift+tab` `ctrl+t` `ctrl+g`
`p` `c` `r` `left_square_bracket` `right_square_bracket`
`ctrl+down` `ctrl+up` `ctrl+pageup` `ctrl+pagedown` `ctrl+r`
`ctrl+home` `ctrl+end`

**Composer (`Editor.BINDINGS`, `editor.py:1377-1425` + inherited `TextArea`):**
`alt+left` `alt+right` `alt+shift+left` `alt+shift+right` `alt+b` `alt+f`
`ctrl+v` `super+v` `ctrl+o` — and from `TextArea`:
`ctrl+a` `ctrl+e` `ctrl+w` `ctrl+d` `ctrl+x` `ctrl+k` `ctrl+u` `ctrl+y` `ctrl+z`
`ctrl+left` `ctrl+right` `ctrl+shift+left` `ctrl+shift+right` `ctrl+shift+k`
`f6` `f7` `super+c` `super+x` `super+y` `super+z` `super+backspace`
`alt+backspace` `alt+delete` `ctrl+backspace`
plus every cursor key.

**Textual `Screen`:** `tab` `shift+tab` `ctrl+c` `super+c`.
**Textual `App`:** `ctrl+q` (**priority**), `ctrl+c`.

**Widget-level (pickers, panels):** `ctrl+p` `ctrl+n` (`ask_picker.py:546`,
`session_picker.py:734`, `settings_view.py:353`), `q` `t` `e` `f` `y` `n`
`0`-`9`, `space`, `enter`, arrows/page/home/end.

**Free `ctrl+<letter>` slots remaining:** `h` `i` `j` `m` `s` — and four of
those five are terminal aliases (`ctrl+h`=backspace, `ctrl+i`=tab,
`ctrl+j`=LF, `ctrl+m`=CR). **`ctrl+s` is the only genuinely free
`ctrl+<letter>` in the entire application.** That fact drives §E.

### C.2 Hard-reserved — refused by the capture UI, always

| Key | Why |
|---|---|
| `escape` | app `stop` (`app.py:1891`) + the cancel of capture itself (§B.3) |
| `ctrl+c` | interrupt + exit ladder (`app.py:1709-1711`) |
| `ctrl+d` | quit — advertised on the splash (`welcome.py:259`) |
| `ctrl+q` | Textual's own **priority** quit; unoverridable from app level |
| `ctrl+m` `ctrl+i` `ctrl+h` `ctrl+j` `ctrl+@` | terminal-identical to enter/tab/backspace/LF/NUL — binding one silently binds the other |
| `enter` `tab` `space` | structural; `enter` submits the composer |
| bare printable characters | they are text; binding `n` makes the composer unusable |

`ctrl+z` (SIGTSTP) and `ctrl+s`/`ctrl+q` (flow control) are **not** in the hard
list — see §E.2 for the measurement that clears `ctrl+s`.

Refusal shows the reason, never silently ignores the press: a key that appears
to do nothing in capture mode is indistinguishable from a broken terminal.

### C.3 Soft conflicts: warn-and-allow, with the victim named

**Recommendation: allow, and name what is being taken.** Reasoning:

- **Refusing is wrong here** because the reserved set is enormous (§C.1) and
  mostly context-scoped. `ctrl+t` is only meaningful when todos exist;
  `p`/`c`/`r` only in the subagent panel; `q`/`t` only in a full-page overlay. A
  user who wants `ctrl+t` for new-session, and never uses todos, is asking for
  something reasonable, and a refusal makes the feature feel arbitrary.
- **Silent stealing is wrong** because the victim is invisible: the user
  discovers weeks later that todos stopped expanding. That is VS Code's model
  and it is VS Code's most-complained-about keybinding behaviour.
- So: **allow, and say what it costs, at the moment of the decision.** The
  PENDING row names the victim by its human description, and the user presses
  enter having read it.

**Conflict detection is free — do not reimplement it.**
`BindingsMap.apply_keymap` returns `KeymapApplyResult(clashed_bindings)`, and
`App.handle_bindings_clash(clashed_bindings, node)` is the documented override
point. Measured against the real app:

```
4. remap onto app's own ctrl+t -> ['new'] | clashes reported: [(['ctrl+t'], 'Probe')]
```

The keymap binding **wins** the key and the clash is reported. So
`OperatorApp.handle_bindings_clash` records the clash set, and the settings page
reads it to render the victim's description.

**Two caveats measured, both must be handled:**

1. **Clash detection only covers bindings in the same `BindingsMap`** — i.e.
   app-level. A remap onto a `TextArea` editing key reports **no clash** and the
   outcome depends on priority:

   ```
   7. priority binding remapped onto TextArea ctrl+w -> ['new'] | text unchanged
   8. NON-priority remapped onto TextArea ctrl+u -> [] | text: '' (TextArea won, buffer deleted)
   ```

   A non-priority app binding **loses** to the focused composer. Since we ship
   non-priority (§E.3), a user who binds `ctrl+u` gets a hotkey that silently
   does nothing while the composer has focus — which is almost always. So the
   page must carry its **own** static table of composer-owned keys (§C.1) and
   warn on those too. `handle_bindings_clash` alone is not sufficient coverage.

2. **`clashed_bindings` fires per keypress**, not once per apply — the docstring
   says "which may be on each keypress if a clashing widget is focused". Store
   the latest set; never append to a growing list, and never notice on it.

### C.4 Two actions mapped to the same key

Refuse. Unlike the soft conflicts above, this one is unambiguously a mistake
with no reading under which it is intended, both victims are in the feature the
user is currently configuring, and `apply_keymap`'s own resolution when two ids
claim one key is not a contract we should be relying on. Validate in
`settings_io` across the whole `keymap.*` group, not just per key.

---

## D. Resolution and propagation

### D.0 The docs are wrong; the API works

Textual's input guide says "Textual doesn't support modifying the bindings at
runtime". **That is false for the keymap API in 8.2.8.** Measured against the
real `OperatorApp` with a focused `Editor` holding a draft
(`/tmp/probe_realapp2.py`, `/tmp/probe_coexist.py`):

```
1. default chord past focused Editor -> ['new'] | draft intact: True
2. after set_keymap, new chord       -> ['new'] | draft: 'half a thought'
3. old chord after remap             -> []      | draft: 'half a thought'
5. second remap: f6 -> ['new'] | stale f5 -> []
```

Remapping is live, the old key stops working, repeated remaps replace rather
than accumulate, and the composer's buffer is never touched. **No relaunch, no
`/new`, no "applies next session" fallback is needed.** The scope is `LIVE`.

Two further measured properties the design leans on:

```
after reset (empty keymap): active_bindings -> {'ctrl+n': ...} | app._keymap -> {}
```

`set_keymap({})` **restores the class defaults**. Application is against the
pristine `BINDINGS`, not cumulative — so the resolver can always send the full
desired keymap and never has to compute a diff or an "unset" instruction.

```
F. binding key after remap: [('ctrl+n', None)] | get_key_display: ['^n']
```

`Binding.key` on the class map still reads the **old** key after a remap. Do not
read it. The live key is `app.active_bindings[key].binding` keyed by the *new*
key — or, more simply, the value we ourselves persisted (§F.2).

### D.1 Mechanism: `Binding(id=...)`, not `App.bind()` or BINDINGS mutation

Rejected alternatives, briefly:

- **`App.bind()`** (`DOMNode.bind`, source read) *appends* to
  `key_to_bindings[key]`. It cannot unbind, so a remap would leave the old key
  live. Wrong shape.
- **Mutating `App.BINDINGS`** — a class attribute shared by every instance, and
  Textual has already built the instance `BindingsMap` by the time the config is
  read. Would need a rebuild anyway.
- **A check inside `on_key`** — reimplements dispatch, bypasses
  `check_action`/`active_bindings`, and would need its own precedence rules
  against the composer. This is the "second mechanism beside the existing one"
  that should not be built when the framework has a first-class one.

So: **give the two bindings ids and drive `update_keymap`.**

```python
# in OperatorApp.BINDINGS
Binding("ctrl+n", "new_session", "New session", show=False, id="keymap.new_session"),
Binding("ctrl+s", "resume_session", "Resume a session", show=False, id="keymap.resume"),
```

Note the binding `id` is **deliberately the same string as the config key**.
One vocabulary: the config key, the binding id, the `changed_keys` entry
(`config_watch.py:110-115` — "speaks the same vocabulary as `/settings`"), and
the tip lookup are all `keymap.new_session`. `update_keymap` takes exactly that
mapping with no translation layer.

**New actions, not new logic.** `action_new_session` calls `self._cmd_new(self._notice)`
(`app.py:8700`); `action_resume_session` calls `self._cmd_resume("", self._notice)`
(`app.py:8261`). Both handlers already guard `self._resume_factory is None`
(`app.py:8717`, `app.py:8284`) and print a notice, so a hotkey in a
non-resume-capable launcher degrades exactly as the slash command does. Do
**not** route through `_run_slash_command` (`app.py:19261`): that path does
remote-capability routing, paste expansion and echo policy for *typed* text, all
of which is meaningless for a keypress, and `/new`/`/resume` reach their
handlers directly at `app.py:19556-19559` anyway.

### D.2 The action registry

One module-level table, next to the settings registry, so the binding ids, the
config keys, the human descriptions and the tip templates cannot drift:

```python
# local_operator/keymap.py  (new, no Textual import — see settings_io.py:40-44)
@dataclasses.dataclass(frozen=True)
class KeyAction:
    id: str            # == the config key, == the Binding id
    label: str         # "New session"
    default: str       # "ctrl+n"
    tip: str           # "{key} starts a new conversation"

KEY_ACTIONS: tuple[KeyAction, ...] = (...)
```

`settings_io.SETTINGS` derives its two `Kind.HOTKEY` entries from this, and
`welcome.py` reads `tip`. Placing it in its own module rather than in
`settings_io` keeps `settings_io`'s no-Textual, CLI-cheap contract
(`settings_io.py:40-44`) while letting the TUI import both.

### D.3 The resolver

```python
def resolved_keymap(values: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """{binding id: key string} for every VALID override, plus rejected keys."""
```

- Reads only `keymap.*`; an absent key means "default", so it is simply omitted
  from the returned mapping — `set_keymap` restoring defaults (§D.0) makes
  omission the correct encoding of "unset".
- Drops entries failing §A.4 validation and returns their names so the caller
  can notice them once.
- Pure, takes a `values` mapping. Testable with no app.

### D.4 Boot

In `OperatorApp.on_mount`, beside `_watch_config` (`app.py:18367`):
read the watcher's snapshot (or a `ConfigManager` if none), call
`self.set_keymap(resolved)`. **`set_keymap`, not `update_keymap`, at boot** —
it is the one call that is authoritative rather than incremental.

### D.5 Propagation across sessions — the exact path

This is the mechanism that already exists; the feature only adds a listener
branch.

**Writer side.** `SettingsView._write` → `settings_io.write_setting`
(`settings_io.py:1679`), which after storing calls `_invalidate_caches()`
(`settings_io.py:1878`) and `_notify_watcher()` (`settings_io.py:1897`). The
watcher re-reads and fans out with `source="local"`.

**Reader side, other processes.** `ConfigWatcher` stat-polls `config.yml` every
`POLL_INTERVAL_S` on the app loop, with a kqueue accelerator
(`config_watch.py:18-37`). A changed fingerprint triggers a parse and
`_changed_registry_keys` (`config_watch.py:640-659`), which walks
`settings_io.SETTINGS` — **so the `keymap.*` settings are picked up
automatically once registered**, including the flat-dotted paths, with no change
to `config_watch.py` at all.

**The callback that rebinds.** `OperatorApp._on_config_change`
(`app.py:18395`), in the TUI-owned apply block at `app.py:18604-18616` where
`tui.sidebar_*`, `display.dock` and `tui.theme` are already handled:

```python
if any(key.startswith("keymap.") for key in changed):
    values = getattr(change, "values", {})
    resolved, rejected = resolved_keymap(values)
    self.set_keymap(resolved)
    if rejected:
        self._system_notice(f"keymap: ignoring unusable {', '.join(rejected)}", "warning")
```

**Placement inside that method matters.** The `source == "local"` branch returns
early at `app.py:18395`+ ("Apply, do not announce"), and it currently applies
only `tool_approval_mode`, because everything else local was already applied by
the handler that wrote it. **A keymap write from the `/settings` page in THIS
process is not applied by anything else** — `SettingsView` writes config and the
app holds the `BindingsMap`. So `set_keymap` must be called on **both** the
local and the disk branch. This is precisely the bug the `tool_approval_mode`
comment documents at length (`app.py:18395`+: "arrived here as the one delivery
that moved nothing"), and it will recur verbatim if the apply is added only to
the disk branch. **Flag this to the reviewer as the single most likely defect in
the implementation.**

The scope tag is `LIVE`, and it is honest: `Scope.LIVE` means "immediately in
every running session on this machine — on the same call stack in the process
that wrote it, and within `ConfigWatcher.POLL_INTERVAL_S` for sessions in other
processes" (`settings_io.py:121-124`). That is exactly what was measured.

### D.6 Section placement

A **new section**, `keymap`, titled `Hotkeys`, `Scope.LIVE`:

```python
Section("keymap", "Hotkeys", Scope.LIVE,
        "Keys for starting and resuming conversations. Press enter on a row, "
        "then press the key you want.")
```

Not a row under `appearance`: scope is uniform within a section by construction
(`settings_io.py:115-118`) — `appearance` is LIVE too, so that argument does not
force a split — but the section *description* is where this page teaches the
capture gesture, and `appearance`'s description is about theme and terminal
features. A one-line "how do I even use this row" is worth a section header;
this follows `runtime`'s and `fork`'s precedent of splitting for a description
that would otherwise be a lie or a distraction (`settings_io.py`, `runtime`
section comment).

---

## E. Default chords

**Proposed: `ctrl+n` = new session, `ctrl+s` = resume picker.** Pending the
scout report; the constraints below are binding on whatever is finally chosen.

### E.1 `ctrl+n`

Free at app level (probe: `ctrl+n` appears only under `AskPickerScreen`,
`SessionPickerScreen`, `SettingsView`), unclaimed by `TextArea`, byte 14, no C0
alias, no multiplexer prefix. Mnemonic is universal.

The pickers' `ctrl+n` = "move down" is a **widget-level** binding, and Textual
dispatches focused-widget-first for non-priority bindings. Measured on the real
app with a picker-shaped focused widget:

```
C. ctrl+n with picker focused -> ['picker-down']   (app binding did NOT fire)
A. ctrl+n with Editor focused -> ['new']  | draft intact: True
```

They coexist. This is the scout's claim, confirmed here.

### E.2 `ctrl+s` — the flow-control question, settled

The scout could not verify this empirically. **It is safe.** Textual explicitly
clears XON/XOFF in the driver:

```python
# textual/drivers/linux_driver.py:344-352
return attrs & ~(
    # Disable XON/XOFF flow control on output and input.
    # (Don't capture Ctrl-S and Ctrl-Q.)
    # Like executing: "stty -ixon."
    termios.IXON | termios.IXOFF | ...
)
```

And measured reaching the app past a focused composer:

```
B. ctrl+s with Editor focused -> ['resume'] | draft: 'half a thought'
```

`ctrl+s` is also, per §C.1, **the only genuinely free `ctrl+<letter>` left in
this application** — the other four are terminal aliases. Spending it here is
defensible (resume is a top-two gesture) but it should be spent knowingly.

**Residual risk (R2):** the guarantee is Textual's *inside* the app. A user's
outer terminal, tmux, or ssh session may still swallow `ctrl+s` before the app
sees it. The feature's own remappability is the mitigation, and it is a good
one — but the *default* should not be a key some users cannot press. If the
scout's cross-agent evidence is thin, prefer **`f2`/`f3`**, matching this app's
own `f8`/`f9` precedent (`app.py:1852-1854`) and the reasoning recorded there:
"F8 keeps Aside independent of terminal modifier encoding and leaves every
TextArea editing key alone" (`app.py:1850-1851`).

### E.3 Non-priority — a hard constraint

Both bindings ship **`priority=False`**. Measured consequences:

- Priority would fire *before* the focused widget, breaking every picker's
  `ctrl+n` = down (§E.1).
- Priority also fires before `TextArea`, so a user who remaps onto `ctrl+u`
  would get the hotkey instead of delete-to-start-of-line — which sounds
  desirable until you notice it applies to *every* composer key they might pick,
  making the composer silently lossy.

Non-priority costs the reverse: a remap onto a composer key silently loses while
the composer is focused (measured, §C.3 case 8). That is the better failure —
it is *visible* (nothing happens when you press the key), whereas the priority
failure *destroys the user's text*. The §C.3 warning covers the discoverability.

### E.4 Conflict-safety constraints for the final pick

Whatever the scout recommends must satisfy all of:

1. Not in the hard-reserved set (§C.2).
2. Not bound by `Editor`/`TextArea` (§C.1) — else non-priority loses to the
   composer, which holds focus in the common case.
3. Not bound by `OperatorApp.BINDINGS` (`app.py:1841-2014`).
4. Emitted by Terminal.app, not only by kitty-protocol terminals. `super+*` and
   `ctrl+shift+*` fail this — `app.py:1869-1870` records that "terminals that
   intercept Cmd+B never deliver it", which is why `super+b` is an *addition*
   to `ctrl+b` and never a sole route.
5. Verified by pressing it in a pilot against the real `OperatorApp` with a
   focused `Editor` holding a draft, asserting the action fires **and** the
   draft is unchanged — the measurement shape `app.py:1963-1976` already
   established for the aside chords.

### E.5 A leader key is an extension point, not scope

The app has burned 9 of ~12 comfortable `ctrl+<letter>` slots (§C.1). A leader
prefix (`ctrl+x ctrl+n`) is the structural answer, and Textual has no support
for it — it would be a hand-built modal state in `OperatorApp.on_key` with its
own timeout, its own visual indicator, and its own interaction with the
composer's existing escape-coalescing machinery (`editor.py:2247`+, which has
already produced defects "in three consecutive rounds"). **Out of scope.**

The schema does not preclude it: §A.2 already stores an arbitrary key string,
and a future leader implementation would store `ctrl+x ctrl+n` (space-separated,
a spelling Textual will never claim) and intercept before dispatch. Record that
in the module docstring as the reserved extension, and do not build it now.

---

## F. Tips

### F.1 The problem, stated precisely

`TIPS` hardcodes key strings: `TIP_PASTE` = `"ctrl+v attaches…"`
(`welcome.py:352`), and `"esc stops the agent without ending the session"`
(`welcome.py:360`). Adding remappable keys makes any tip naming one a potential
lie. And:

```python
TIP_MIN_WIDTH = max(cell_len(f"{TIP_GLYPH} {tip}") for tip in (*TIPS, TIP_SETUP, TIP_PASTE))
# welcome.py:431
```

is computed at import from the pool. A remapped chord changes the rendered
length. The constant's own docstring says the threshold "is a WIDTH and never
the current tip's own length: presence has to be the same answer for every entry
in the pool, or the block would gain and lose a row as it rotated"
(`welcome.py:420-423`) — and `_tip_lines` returns exactly one row or zero rows
because "the row count is a function of `width` alone… a tip that could be one
row for one entry and two (or none) for the next would shove the entire splash
up and down the screen" (`welcome.py:812-816`).

So a naive "resolve the key at render time" **breaks a load-bearing invariant**:
a user who remaps to `ctrl+shift+pagedown` lengthens a tip past `TIP_MIN_WIDTH`,
and at some widths the tip row appears and disappears as the reel rotates,
moving the whole splash.

### F.2 The fix: bound the width, do not measure the pool

Two changes, and the second is the one that preserves the contract.

**(1) Templated tips.** Pool entries become `(template, binding_id | None)`:

```python
KEYED_TIPS = (
    ("{key} starts a new conversation", "keymap.new_session"),
    ("{key} reopens a recent conversation", "keymap.resume"),
    ("/settings → Hotkeys remaps these keys", None),
)
```

Rendered by substituting `format_key(effective_key(binding_id))`. **Resolve
from the persisted config value, not from `Binding.key`** — measured (§D.0,
result F), the class map still reads the old key after a remap. The tip resolver
takes the same `resolved_keymap()` output the app applied, falling back to
`KeyAction.default`.

The third entry is the one that satisfies request item 4 ("tips must teach
hotkey remapping"). It names the *route*, not a key, so it is never a lie.

**(2) `TIP_MIN_WIDTH` becomes a bound, not a measurement.**

```python
#: Widest key rendering a keyed tip is allowed to reserve. A longer key is
#: still BOUND and still WORKS; the tip uses this budget when deciding whether
#: the row fits, so the presence of the tip row stays a function of terminal
#: width alone (welcome.py:812-816) rather than of what the user remapped.
#:
#: 25 = cell_len("ctrl+right_square_bracket"), the longest name in
#: textual.keys.Keys (measured, textual 8.2.8). Deliberately the true maximum
#: rather than a plausible one: the point of the constant is that no
#: configuration can move it.
KEY_BUDGET_CELLS = 25

TIP_MIN_WIDTH = max(
    *(cell_len(f"{TIP_GLYPH} {tip}") for tip in (*TIPS, TIP_SETUP, TIP_PASTE)),
    *(cell_len(f"{TIP_GLYPH} {t.format(key='x' * KEY_BUDGET_CELLS)}")
      for t, _ in KEYED_TIPS),
)
```

The threshold is computed against the **worst-case** key, so it is a constant
again, independent of what any user has configured. The row's presence never
changes as the reel turns, which is the invariant `welcome.py:420-423` and
`welcome.py:812-816` both rest on.

**Measured cost: zero, for the proposed templates.** Current `TIP_MIN_WIDTH` is
**59** (set by `"lop detects terminal or multiplexer, then picks placement"`, 57
cells + glyph). The three keyed templates at a 25-cell worst-case key measure
**53, 57 and 40** cells respectively — all under 59. So `TIP_MIN_WIDTH` does
**not** move, no terminal loses the tip row, and risk R4 does not materialise.

That is a property of these templates, not a guarantee. **Keep each keyed
template at or under 32 cells excluding `{key}`** and the bound holds; the
anti-drift test in §G.1 should assert `TIP_MIN_WIDTH` has not risen above 59
rather than trusting a reviewer to re-measure. If a future template does push
it, shorten the sentence — never reintroduce a per-tip width test, which is the
one thing that reflows the splash.

**(3) A disabled/unresolvable binding drops its tip** rather than rendering
`None starts a new conversation`. Since `_tip_lines` must return a constant row
count, "drop" means *skip to the next pool entry*, not return `[]`.

### F.3 What NOT to do

Do not auto-generate `HINTS` (`welcome.py:256-260`) from the keymap. It is a
three-row fixed table with its own two-tier width ladder
(`HINT_KEY_WIDTH`/`HINT_KEY_WIDTH_TIGHT`, `welcome.py:280-281`) computed from
its own contents, and it currently advertises `ctrl+d`, `/` and `/help` — none
of them remappable. Widening this change into that surface buys nothing and puts
a second width contract at risk.

---

## G. Testing strategy

AGENTS.md is emphatic: structural invariants over numeric bounds, and **no
timing bound belongs anywhere in this feature.** Nothing here is about how long
something takes. Every property below is a fact about what is bound or what
fired.

### G.1 Unit (no app)

| Property | Where |
|---|---|
| `resolved_keymap` drops invalid values and names them | `keymap.py` |
| Every `KeyAction.default` passes `validate()` | anti-drift |
| Every `KeyAction.default` is absent from the hard-reserved set | anti-drift |
| No two `KeyAction`s share a default | anti-drift |
| `KEY_ACTIONS` ids == `keymap.*` registry keys == `Binding` ids in `OperatorApp.BINDINGS` | **anti-drift, the important one** |
| `coerce()` normalizes `ctrl+N` → `ctrl+n`; rejects `banana` | `test_settings_io.py` |
| `keymap.*` are flat-dotted (`flat_dotted_keys()` contains them) | `test_settings_io.py` |
| `TIP_MIN_WIDTH` has not risen above 59 (§F.2) | `test_welcome.py` |

The three-way id anti-drift test is the one that earns its keep: the ids are
persisted user data (§A.5) and are duplicated across three modules by design, so
a rename must fail loudly.

### G.2 Pilot (real `OperatorApp`, real stylesheet)

Use the real app, never a bare test host — `_PanelHost`/`_PickerHost` declare no
`CSS_PATH` (AGENTS.md, "Visual validation"). All of these are structural:

1. **Default chord fires past a focused composer holding a draft**, and the
   draft is byte-identical afterwards. Both actions. (The `app.py:1963-1976`
   measurement shape.)
2. **Live remap**: `set_keymap` → new key fires → **old key does not**. Assert
   the negative; it is the half that regresses silently.
3. **Picker coexistence**: with a picker focused, `ctrl+n` moves the picker and
   does **not** start a new session.
4. **Capture mode disarms app actions**: with `_capture_mode` on, pressing
   `ctrl+t`/`shift+tab`/`escape` reaches the capture widget and fires no app
   action; with it off, they fire normally. Assert **both** directions.
5. **Every exit route clears `_capture_mode`** — esc, enter-commit, click-away,
   leaving `/settings`, unmount. This is risk R1 (§B.4) and needs one
   parametrized test per route.
6. **Propagation**: two `ConfigWatcher`-connected apps in one process on an
   isolated config dir; write `keymap.new_session` through `settings_io` from
   app A; assert app B's `active_bindings` carries the new key. Drive it with
   `watcher.poll_now()` (`config_watch.py:282`) — **never** a `sleep`. This is
   AGENTS.md's "wait on the event, never on the clock" applied by removing the
   wait entirely.
7. **The local-branch trap** (§D.5): a write from the `/settings` page in the
   **same** process must move the binding in that same process. This is the
   `tool_approval_mode` bug re-run; it deserves a test whose name says so.
8. **Garbage on disk does not disarm the action**: a config carrying
   `keymap.new_session: banana` boots with the default chord live and one
   warning notice.

### G.3 Visual

`/settings` gains rows, so it is a user-visible change: rendered before/after
frames of the Hotkeys section in all four capture states (idle, capturing,
pending, pending-with-conflict), plus a splash frame showing a keyed tip.
Use `scripts/visual_capture.save_capture` with `isolate_capture()` before app
imports (AGENTS.md), and check the geometry numbers — specifically
`app.screen.virtual_size` vs `size`, since the settings page's height ladder
(`settings_view.py:3885`) reacts to row count.

### G.4 Explicitly NOT tested

No assertion that a remap takes effect "within N ms" or "within
`POLL_INTERVAL_S`". The propagation test drives `poll_now()` directly, so the
property tested is *the callback rebinds*, which is the fact that matters and
cannot flake.

---

## H. Pushback on the request

### H.1 "indicating the user may press one or more keys"

**Recommend: capture exactly ONE key, and say so.** Textual has no chord or
sequence support; `,` in a key string means *alternates*, measured (§A.2). "One
or more keys" therefore has only two possible implementations:

- *Alternates* — "press ctrl+n, then also press f5, both will work". Coherent,
  but a UI that keeps listening cannot tell "the user is adding an alternate"
  from "the user is correcting a fumble", and correcting is the common case by a
  wide margin.
- *Sequences* — a leader key. Requires hand-built modal state, out of scope
  (§E.5).

So the capture widget takes the **last** key pressed and PENDING replaces on a
second press (§B.2). The row says `press a key…`, singular. If the operator
specifically wants alternates, the schema already stores them (§A.2) and
`lop config edit keymap.new_session 'ctrl+n,f5'` works today — that is the right
place for a power-user affordance, not the capture UI.

### H.2 "propagates across sessions"

It does, and better than the request assumes — running sessions re-key live with
no relaunch (§D.0, §D.5). Two caveats worth stating on the row:

- Propagation to *other processes* is bounded by `ConfigWatcher.POLL_INTERVAL_S`
  (`config_watch.py:18-19`), so another pane picks it up within a couple of
  seconds, not instantly. That is what `Scope.LIVE` already promises
  (`settings_io.py:121-124`); no new copy needed.
- A remote/follower session (`RemoteSession`) has no watcher of its own but the
  **app** subscribes through `process_watcher()` regardless
  (`app.py:18378-18382`), and the keymap is an app-level concern. So followers
  get it too. Worth a line in the implementation comment because it is
  non-obvious.

### H.3 Reserved-key policy must not live only in the UI

The request describes the reserved set as a capture-UI behaviour. It must also
live in `settings_io.validate()`, because `lop config edit` and a hand edit
reach the same config and bypass the page entirely — and Textual will apply
whatever it is given without complaint (§A.4). Same rule, one implementation,
enforced at the write boundary; the capture widget calls the same predicate so
the two can never disagree.

### H.4 One thing I would cut

The `(default)` marker in §B.5 is redundant — the page already renders
default-vs-overridden state through `is_default` (`settings_io.py:1549`) and
`_detail_clause` (`settings_view.py:3288`). Reuse that, do not invent a second
marker for this Kind.

---

## I. Risks to watch during rollout

| id | Risk | Watch |
|---|---|---|
| R1 | `_capture_mode` stuck `True` disarms every app hotkey including `ctrl+c` | G.2 test 5, all exit routes |
| R2 | `ctrl+s` swallowed by an outer terminal/tmux/ssh before the app sees it | Ask the operator to press it in their real terminal before merge; §E.2 fallback is `f3` |
| R3 | The local-branch apply omitted in `_on_config_change` — page writes config, this pane does not rebind | G.2 test 7; called out to the reviewer explicitly |
| R4 | A future keyed tip pushes `TIP_MIN_WIDTH` above 59 and drops the tip row on narrow terminals | Measured at **zero cost** for the proposed templates (§F.2); guard with the anti-drift assertion rather than re-measurement |
| R5 | Non-priority binding loses to the composer on a user-chosen chord, feels broken | §C.3 static composer-key warning must be in the first cut, not deferred |
| R6 | `handle_bindings_clash` fires per keypress; naive handling accumulates or notices repeatedly | Store latest, never notice from the handler |

---

## J. What this does NOT change

- `SLASH_COMMANDS` (`slash_commands.py:100`, `:118`). `/new` and `/resume`
  remain, unchanged, and remain the discoverable route. A hotkey is an
  accelerator for an existing command, never a replacement.
- `_run_slash_command` (`app.py:19261`) and its routing. The actions call the
  `_cmd_*` handlers directly (§D.1).
- `config_watch.py`. Registration in `settings_io.SETTINGS` is sufficient
  (§D.5).
- `tui/bindings.py` — **note for the coder:** despite its name this is the
  colour-token table (`bindings.py:1-8`), not a key registry. Nothing in this
  feature touches it.

---

## K. Implementation notes — where the build diverged from this design

Recorded by the implementing agent. Everything in §A–§J was followed as
written except the two items below, both found by driving the real app.

### K.1 App-level conflicts are read from the DECLARED binding map, not from
`handle_bindings_clash`

§C.3 specifies `OperatorApp.handle_bindings_clash` as the source of the victim
name. Implemented that way it never fires for the case the page needs, and the
reason is a collision between two parts of this same design:

* the hook reports a clash only when a keymap is APPLIED and dispatched, i.e.
  after the user has already confirmed — but the warning has to be shown
  *before* the confirming `enter`, while the key is only pending;
* and §B.4's capture gate makes `check_action` return `False` for every app
  action, which removes them from `screen.active_bindings` entirely. Any
  lookup through the active map therefore finds nothing precisely while a row
  is capturing.

Observed against the real app: capturing `ctrl+t` reported no conflict at all.
`SettingsView._app_binding_victim` now reads `app._bindings.key_to_bindings` —
the DECLARED map, unaffected by the gate — and names the binding's
`description`. That is still live rather than a table here, so it stays true
as the app's own bindings change, and it yields the intended frame:
`takes ctrl+t from "Expand/collapse todos"`.

The `handle_bindings_clash` override was consequently **not shipped**: it would
have been a second mechanism recording state nothing reads (`R6`'s
store-the-latest discipline is moot when there is no reader). §C.3's two
measured caveats still hold and still matter — they are why
`keymap.COMPOSER_KEYS` exists, since Textual reports no clash for a composer
key under any mechanism.

### K.2 Arrow keys are capturable, so the "cursor move" exit route is not an arrow

§G.2 test 5 lists "click-away" and "moving off the row" as exit routes. While a
row is capturing, the arrows are keys to BIND like any other — capture owns
every key by construction (§B.4 point 4) — so pressing `down` does not move the
cursor, it captures `down`. The cursor-move route is therefore a click on
another row or a programmatic `action_move`, both of which reach `_settle_row`.
The parametrized test drives `action_move` for that case and the counterfactual
was verified: removing the disarm from `_settle_row` turns it red.

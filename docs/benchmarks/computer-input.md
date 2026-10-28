# Negotiated computer text input

`type` and `paste_text` are different actions, not fallback strategies. The
neutral protocol continues to parse and canonically serialize historical
`TypeAction` records, including Unicode. An adapter's execution restriction
(`type_text_mode`) is applied to the entire batch before mutating dispatch.
The OSWorld X11 backend declares `ascii`: non-ASCII native typing is rejected,
not silently dropped or automatically replaced with clipboard input. Native
ASCII controls keep their existing keyboard semantics.

An adapter advertising `paste_text: true` admits this additional action:

```json
{
  "kind": "paste_text",
  "observation_id": "<current observation id>",
  "text": "café 東京🙂\tsecond column\nnext line",
  "keys": ["ctrl", "v"],
  "clipboard_policy": "overwrite"
}
```

All four data fields are required. The chord uses the same validation as
`key`; there is no default chord. The caller must first focus the intended
application and choose its shortcut (a terminal commonly uses Ctrl+Shift+V).
There is no focus/window-title inference, automatic Enter, restore policy,
retry, or native-typing fallback. A wrong chord can do nothing or invoke
another application command. Read the next observation before claiming text
insertion or task completion.

Text accepts 1–100,000 Unicode characters, including whitespace-only input.
Lone surrogates and Unicode control characters are rejected, except tab, CR
and LF. Those three remain clipboard data, not separately generated key
presses. The receiving application may normalize them, remove a trailing
newline, or submit on a newline; transport equality is not a guarantee of
application storage semantics.

## Key names

`key` (and the `paste_text` chord) take an array of key names pressed
together. A name is either a single printable ASCII character, which is the
key itself and keeps its case in the parsed representation (`"A"` is not
`"a"`), or one of the named keys
`protocol.NAMED_KEYS` defines: `alt`, `backspace`, `capslock`, `ctrl`,
`delete`, `down`, `end`, `enter`, `esc`, `home`, `insert`, `left`, `meta`
(the Super/logo key), `pagedown`, `pageup`, `right`, `shift`, `space`, `tab`,
`up`, `f1`–`f24`. Named keys are matched case-insensitively. A chord may not
name the same key twice. Representation is not a promise of uppercase input:
the OSWorld compiler lowercases single-character key names, so both `"A"`
and `"a"` compile to the same physical key. Use `["shift", "a"]` when an
uppercase key press is intended (or `type` for literal text).

The validator also accepts a small set of standard spellings and rewrites each
one to the single key it names (`protocol.KEY_ALIASES`):

| Accepted spelling (any case) | Becomes |
| --- | --- |
| `Return` | `enter` |
| `Escape` | `esc` |
| `Page_Down`, `pgdn` | `pagedown` |
| `Page_Up`, `pgup` | `pageup` |
| `Caps_Lock` | `capslock` |
| `del` | `delete` |
| `super`, `Super_L`, `win` | `meta` |
| `Control_L` / `Shift_L` / `Alt_L` | `ctrl` / `shift` / `alt` |
| `minus` / `plus` / `equal` | `-` / `+` / `=` |

X11 keysyms that already match a named key once case is ignored (`BackSpace`,
`Tab`, `Delete`, `Home`, `End`, `Insert`, arrow names) need no alias.
"Case-insensitive" was never what refused `Escape`: it folds to `ESCAPE`, and
the named key is `ESC`.

The rewrite happens while the action is parsed, before the batch is bound to
an observation. Parsed actions, bound-batch artifacts, their digests and compiled
guest commands therefore contain canonical key names and match what the
canonical spelling produces. Existing canonical batches replay unchanged.
Raw model-response evidence is not rewritten and can still contain aliases.
The surface schema and its `tool_schema_digest` are unchanged too, because
`named_keys` still lists exactly the same set.

An alias is admitted only when it is a standard X11 keysym or
pyautogui/xdotool name **and** it presses exactly the key its target presses.
pyautogui's X11 table is the check: the OSWorld compiler emits `ctrl`, and
pyautogui sends `Control_L`. Two kinds of name stay refused:

- Names that could mean different keys: `cmd`/`command` mean Ctrl in a Linux
  shortcut but Super if read literally; `option` could mean Alt or AltGr.
- Names that would press a different key: `Control_R`, `Shift_R`, `Alt_R` and
  `Super_R` would press the left key, and `Next`/`Prior` read as commands.

The refusal for `cmd` names both meanings and leaves the choice to the model.

A literal `" "` names the space bar (it is the one printable character outside
`string.punctuation`). Four other refusals carry a corrected example built from
the model's own input, and none of them is applied automatically:

- A chord of two or more plain characters and no modifier
  (`["e", "c", "h", "o", " ", "A"]`) is typing, not a shortcut. The refusal
  shows `{"kind": "type", "text": "echo A"}`. It is checked before the
  duplicate rule, so a spelled word with a repeated letter is told the real
  mistake. `["shift", "a"]`, `["ctrl", "alt", "t"]` and `["up", "left"]` are
  untouched.
- A multi-character name close to no key (`"cat"`, `"comp.scm"`) gets "if you
  meant to type this text" and the `type` action. A near-miss still gets
  "use … instead".
- A `+`-joined chord whose parts are all keys (`"ctrl+t"`) gets
  `"keys": ["ctrl", "t"]`. It used to get `use "ctrl" instead`.
- A key repeated in one chord (`["Page_Up", "Page_Up"]`) gets one
  `{"kind": "key", "keys": ["pageup"]}` action per press.

Two more refusals stay, and their sentences now give the repair. An all-zero
`scroll` states the sign convention: deltas are wheel clicks, and
`delta_y > 0` scrolls UP. A `finish` or `ask_user` sharing a batch, including
`[wait, finish]`, is told to send the finish alone. Leading waits are not
dropped, because both measured `[wait, finish]` batches were an episode's first
call with `reason: "placeholder"`, and both models then did the task.

`type.text` is non-empty but may be whitespace: `"\n"` presses Return and `" "`
presses space. This removed the `\S` pattern from the `TypeAction` schema, so
`tool_schema_digest` changed from `347cc002…` to `bceb4942…` for the default
surface. Parsing old bundles is unaffected, because every value the old schema
accepted is still accepted.

Unknown fields stay refused rather than dropped. Only a field that belongs to
another action kind is dropped (`public_reply.drop_sibling_action_fields`),
because the model's `kind` tag already says what it meant. Dropping a pause
field such as `wait` or `wait_ms` on a `key` would lose the pause. Turning it
into a `wait` would add an action to an ordered batch. Either way the action
would not be the one the model asked for. Instead, the refusal shows the
corrected two-action form,
`[{"kind": "key", "keys": ["enter"]}, {"kind": "wait", "duration_ms": 1000}]`.
A near-miss of the kind's own field gets a "Did you mean" suggestion
(`"key"` → `"keys"`). A rule broken inside a field (duplicate keys in a chord)
is quoted in that field's refusal.

## X11 execution and ownership

The generic stdlib host helper `local_operator.computer_input` generates guest
source; only execution in the guest imports pyautogui. It requires the guest's
existing `xclip`. UTF-8 is transferred without shell interpolation and checked
against bounded readback before the explicit chord. Readiness polling has a
five-second deadline and never replays the chord. The readback buffer cannot
exceed the new payload by more than one byte; mismatching/oversized old selection
contents are discarded, never retained as evidence.

`xclip -quiet -selection clipboard -in -target UTF8_STRING -loops 0` runs as a
tracked foreground child in its own session. A successful action leaves that
owner serving the new CLIPBOARD until another application replaces it; PRIMARY
is never selected. An ownership replacement causes xclip to exit (an active
selection transfer may need to finish first). Each payload is at most 400 KB
UTF-8 plus xclip's transfer buffers; there are no durable payload files. The
helper closes its unnamed temporary source and reaps transient readback
processes. On failure it also kills/reaps its new owner; it never restores the
old clipboard, so a failed action can leave the clipboard changed or empty.
Episode/guest teardown remains the final process-lifetime bound. Native proof
must check repeated ownership replacement, not merely one successful paste.

The generated helper is one `exec` statement, compatible with controllers that
prepend imports with semicolons. Large Python programs use a fixed bootstrap
and bounded argv chunks to avoid Linux's per-argument limit, but remain **one**
guest request through the existing adapter/provider single-shot transport.
Small legacy command argv is unchanged. A timeout/nonzero/ambiguous response
uses existing mutation-failure policy; only a separately declared observation
phase can be re-read, without reapplying input.

## Negotiation, evidence, and compatibility

Adapter RPC **1.6** retains 1.5's explicit clipboard support and native text
restrictions and adds explicit host/adapter ownership of ask-user answers. Exact version checks refuse mixed workers before allocation.
Supervisor admission and adapter compilation independently enforce the surface;
known model admission failures use existing bounded corrective decisions and do
not enter ambiguous-mutation poisoning.

`ActionSurface` is the shared source for admitted action models, model prompt,
and schema identity. The runner passes the negotiated surface explicitly to
`EpisodeModelClient.decide(action_surface=...)`, records its canonical JSON in
manifest metadata `action_surface`, and hashes that same schema with the
`runner-tool-schema-v1` domain for `tool_schema_digest`. The digest is independent
of the episode ID; capability changes alter it. Unsupported kinds are not
advertised in the generated prompt.

Adapter 0.1.2 is a new artifact identity, not a replacement of frozen 0.1.1
wheels. Build/pin a new worker environment and selector for RPC 1.6. Keep frozen
older environments and their rescue launch paths intact until their episodes
and descriptors are settled; do not rewrite historical bundles, selectors,
attestations or rescue descriptors to make them look compatible. Root harness
release version assignment happens at PR/release preparation; private proofs
pin the committed source and built wheel digest.

Local tests execute generated source against fake clipboard/keyboard boundaries
and exercise the real local HTTP command transport. They are not native X11 or
rendered UI proof. Native validation runs separately through the actual
runner → worker → compiler → provider pipeline, with independently read saved
text and screenshots; no model call or scored benchmark claim is implied.

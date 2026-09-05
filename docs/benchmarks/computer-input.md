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

Adapter RPC **1.5** explicitly negotiates clipboard support and native text
restrictions. Exact version checks refuse mixed workers before allocation.
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
wheels. Build/pin a new worker environment and selector for RPC 1.5. Keep frozen
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

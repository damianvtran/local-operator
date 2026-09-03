# Evidence — the reasoning-effort ladder from the provider listing

Frames captured from the **real** `OperatorApp` (which loads
`local_operator.tcss`) driving the real `build_model_spec`, at 120 columns — a
width free of the pre-existing band-fit reflow that sheds the effort rung below
118. Each frame's band text and wire value were read from the widget and printed
alongside the capture, so the pixels and the numbers agree.

Captions live here rather than only in the PR thread. A frame whose caption sits
in a comment is a frame nobody re-checks when the rule changes — which is
exactly how `after-opus46` came to depict a superseded rule for a round.

## The reported bug, and the ladder that fixes it

| frame | shows |
|---|---|
| `before-gemini.png` | `openrouter/google/gemini-3.8-flash` on base: **no effort segment at all**. The reported bug. |
| `after-gemini.png` | The same model on head: `▴ auto`, ladder `('low','medium','high')`, no wire key. The segment exists and cycles. |
| `after-glm.png` | `z-ai/glm-5.3` — `▴ auto`, ladder from the listing, no wire key. |
| `after-gpt54pro.png` | `openai/gpt-5.4-pro` — `▴ auto`, ladder **narrowed** by the listing to `('medium','high','xhigh')`; the `none`/`low` rungs the route rejects are gone. |
| `after-nonreasoning.png` | `llama-4-8b` — no segment at all, which is the honest answer for a model with no ladder. |

## The seeding rule: warm and cold must agree

| frame | shows |
|---|---|
| `before-opus46.png` | `openrouter/anthropic/claude-opus-4.6` on **base**: no effort segment, no ladder, no wire key. |
| `after-opus46.png` | Head, **listing reached** (warm): `▴ auto`. The dotted-id repair gave it a ladder; nothing is seeded on an aggregator route. |
| `after-opus46-cold.png` | Head, **listing NOT reached** (cold). Byte-identical to the warm frame — that is the point. Before the fix this arm rendered `▴ high` and put `reasoning_effort: 'high'` on the wire, so the same model booted differently depending on whether an HTTP call landed. |

## The first `shift+tab` must never turn reasoning off

`mistralai/mistral-small-2603`, ladder `('none','high')` — one of 8 listing
ladders that start at `none` and carry no `medium`.

| frame | shows |
|---|---|
| `d4-boot.png` | Boot: `▴ auto`, wire `None`. |
| `before-d4-first-press.png` | **Before the fix**: the first press lands on `▴ none` and puts `'none'` on the wire — a user pressing the key to discover the control had silently disabled reasoning. |
| `after-d4-first-press.png` | **After the fix**: the first press lands on `▴ high`. The discovery press can no longer disable reasoning. |
| `d4-press2-none.png` | `none` is still reachable by cycling (and by `/effort none`); it is a legitimate choice, just not one made on the user's behalf. |

## Cycling

`cycle-0..3.png` — `openrouter/google/gemini-3.8-flash` stepping
`auto → medium → high → low`. Re-verified against the current head: the boot
state and every rung are unchanged by the seeding fix. Only the effort word
moves between frames; `screen.virtual_size == screen.size` and no scrollbar
appears, so there is no reflow the user reads as motion.

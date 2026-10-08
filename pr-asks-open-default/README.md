# asks-open-default — the queued-ask surface opens by default

The four states of the shared open-policy contract (six clauses; the branch implements it as
`local_operator/tui/ask_open_policy.py` and `local_operator/mobile/web/src/lib/ask-open-policy.ts`)
on both surfaces this change touches: the TUI ask panel and the mobile relay's AsksSheet. Also
captured: clause 5 (typing while asks arrive), the TUI's Tab handoff, and — added in the round-1
remediation — the secret-refusal notice in all three of its states, the 80×24 header ladder, and a
light-palette relay set. See the composite captions.

**Frame identities:**

| set | build | what it is |
|---|---|---|
| `raw/tui-before/`, relay `before` | `e3176403b6` | the main the branch was cut from — the FEATURE baseline for the seven original composites |
| `raw/tui-reviewed/` | `d340f8208f` | the reviewed head — the BEFORE half of the round-1 delta composites |
| `raw/tui-after/`, relay dark + `raw/relay-after/` | `395185a993` | the round-1 head — the AFTER half everywhere |
| `raw/relay-light-after/` | `395185a993` | the same head, `--theme localOperatorLight` (round-1 design D10) |

`local_operator/tui` and `local_operator/mobile/web` are byte-identical between `e3176403b6` and
current main, so the feature baseline is current main for both surfaces. The fail-on-old results
and the full mutation tables are published on the PR (the tables as a comment; the fail-on-old
runs are quoted in the body).

## The composites

| file | what it shows |
|---|---|
| `tui-1-no-asks.png` | TUI state 1 — no asks: closed on both builds; frames pixel-identical |
| `tui-2-pending-one-ask.png` | TUI state 2 — one pending ask: the branch opens the card; the caret stays in the composer |
| `tui-2-pending-two-asks.png` | TUI state 2 — two pending asks: the branch opens the list, passive (no cursor, no `d` decline hint) |
| `tui-3-all-addressed.png` | TUI state 3 — everything answered: closed on both; pixel-identical |
| `tui-4-dismissed.png` | TUI state 4 — opened on arrival, closed by the user (f4), then a re-render + a NEW ask + away-and-back: still closed |
| `tui-5-typing-at-once.png` | TUI clause 5 — typing at once: every key lands in the composer, none in the question |
| `tui-tab-hands-the-caret.png` | TUI — Tab from the empty composer hands the caret to the auto-opened list |
| `tui-r1-card-ink.png` | Round 1 D1/U1 — the passive card stops painting the caret, the tint band, and the accent label (reviewed head vs round-1 head) |
| `tui-r1-refusal-copy.png` | Round 1 D2/U2 — the refusal notice in all three states: open card (`⇥`), closed surface (`f4`), open list (`⇥`, then the card) |
| `tui-r1-hint-ladder.png` | Round 1 U3/D4+D6 — at 100×30 the engaged list names `enter answer`; at 80×24 the first hint outbids the drawer clause; row tails read `58 m` |
| `relay-states-1-3-2.png` | Relay states 1, 3, 2 side by side (closed / closed / open) |
| `relay-4-dismissed.png` | Relay state 4 — close mark, then a re-publish, a NEW ask (dock counts 4) and away-and-back: still closed |
| `relay-5-typing.png` | Relay clause 5 — typing when the asks arrive: nothing is taken; the composer frame is pixel-identical |
| `relay-light-states-1-3-2.png` | The light palette (`localOperatorLight`): states 1, 3, 2 — the sheet reads theme tokens, so the light lanes get their own set |
| `relay-light-4-dismissed.png` | The light palette: state 4, every frame closed with the dock showing |
| `relay-light-5-typing.png` | The light palette: clause 5 |
| `raw/` | the un-composited frames plus a per-frame geometry JSON for every state (three TUI builds, two relay palettes) |

The seven original composites were re-cut in round 1 so their AFTER halves are the round-1 head;
where a state did not change (`no-asks`, `addressed`, `dismissed-closed`, `dismissed-back`) the
paired frames are byte-identical before and after.

## How they were captured

TUI (`100x30`, dark, one isolated `env -i` run per mode, `CMUX_*` absent):

```sh
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
  TERM=xterm-256color PYTHONPATH="$TREE" "$VENV/python" \
  scripts/ask_open_shot.py OUT/<mode>.svg 100x30 <mode>
# modes: no-asks pending pending-list pending-typed addressed dismissed-open
#        dismissed-closed dismissed-back list-tab
#        secret-refusal secret-refusal-closed secret-refusal-list            (round 1)
# and at 80x24: pending-list list-tab
rsvg-convert -z 1.25 OUT/<mode>.svg -o OUT/<mode>.png
```

The script drives the real `OperatorApp` (the one that loads the stylesheet) through a real
conversation switch; the seeded conversation rides on `Convo(history=…)` and is replayed by the
switch, so each frame answers "can the user read the conversation" as well as "is the surface up".
Every geometry JSON now also carries `focus` (the focused widget at capture time) — the caret
claims in the tables below are auditable from the JSONs, not from stdout alone (round-1 NIT-1).

Relay (`390x844` CSS px, DPR 2, one headless Chrome per run, `--use-mock-keychain
--password-store=basic --headless=new --remote-debugging-port=0`, throwaway profile):

```sh
.venv/bin/python scripts/mobile_asks_open_capture.py OUT {before|after} --expect {before|after}
# round 1, the light set:
.venv/bin/python scripts/mobile_asks_open_capture.py OUT light --expect after --theme localOperatorLight
```

`--expect` turns every frame into a pass/fail check against what THAT build must do (a
before-set that opened a sheet, or an after-set that did not, exits red): all eight frames PASS
on both builds, and all eight PASS again under `--theme localOperatorLight`. Every frame is also
held until the page stops moving (no finite animation running, the sheet not still reading, the
panel box unchanged across samples) — the first frame is not the settled frame.

## Numbers beside the stills

TUI — screen and focus per mode (`screen` is 98x28 in every frame, virtual == size, no
scrollbar; the transcript is 21 rows of virtual 22 in every frame). `focus after` is the `focus`
key of the corresponding `raw/tui-after/*.geometry.json`; the reviewed-head set carries it too:

| mode | ask surface, before (feature baseline) | ask surface, after | focus after |
|---|---|---|---|
| `no-asks` | — | — | `Editor` |
| `pending` | — | `AskPickerScreen` y=8 h=14 | `Editor` |
| `pending-list` | — | `AskQueueList` y=17 h=5 | `Editor` |
| `pending-typed` | — | `AskPickerScreen` y=8 h=14 | `Editor` |
| `addressed` | — | — | `Editor` |
| `dismissed-open` | — | `AskQueueList` y=17 h=5 | `Editor` |
| `dismissed-closed` | — | — | `Editor` |
| `dismissed-back` | — | — | `Editor` |
| `list-tab` | — | `AskQueueList` y=17 h=5 | `AskQueueList` |
| `secret-refusal` | — | `AskPickerScreen` y=8 h=14 | `Editor` |
| `secret-refusal-closed` | — | — | `Editor` |
| `secret-refusal-list` | — | `AskQueueList` y=17 h=5 | `Editor` |

The minimized bar (`AskBar`) is 1 row at y=24 in every frame where an ask is waiting, on both
builds; the auto-opened card is 14 rows at y=8 and the list 5 rows at y=17, above the same bar.
The round-1 deltas between `raw/tui-reviewed` and `raw/tui-after`, per mode, are in
`tui-r1-*.png`'s footnotes (card ink: 40,061 px of ~638 k differ, max channel delta 133/255).

Relay — per-frame probe, after build (the before build shows no dialog in ANY frame; its
`opened-after-ms` is `None` everywhere):

| frame | dialog | title | cards | siblings inert | composer focused | settled after | verdict |
|---|---|---|---|---|---|---|---|
| `01-no-asks` | False | — | 0 | 0/0 | False | 211 ms | PASS |
| `02-pending-on-open` | True | asks · 3 questions | 4 | 4/4 | False | 210 ms | PASS |
| `03-all-addressed` | False | — | 0 | 0/0 | False | 210 ms | PASS |
| `04a-closed-by-the-user` | False | — | 0 | 0/0 | False | 205 ms | PASS |
| `04b-after-a-re-render` | False | — | 0 | 0/0 | False | 208 ms | PASS |
| `04c-after-a-new-ask` | False | — | 0 | 0/0 | False | 205 ms | PASS |
| `04d-after-away-and-back` | False | — | 0 | 0/0 | False | 209 ms | PASS |
| `05-typing-then-asks-arrive` | False | — | 0 | 0/0 | True | 209 ms | PASS |

The light run's numbers are identical in shape and live in
`raw/relay-light-after/light-open-by-default-geometry.json` (state 2: dialog, `asks · 3
questions`, 4 cards, 4/4 inert siblings, settled 221 ms).

Repeatability: three AFTER relay runs are identical or differ by ≤72 px of ~1.3 M (max channel
delta 6/255 — sub-pixel anti-aliasing); before vs after, only state 2 differs materially
(1.31 M px), the rest are ≤72 px anti-aliasing. The TUI set re-captured at the pushed head is
byte-identical to the set these composites were built from (its `no-asks` / `addressed` /
`dismissed-*` frames are byte-identical to the feature baseline's, exactly as the states promise).

# asks-open-default — the queued-ask surface opens by default

The four states of the shared open-policy contract (six clauses; the branch implements it as
`local_operator/tui/ask_open_policy.py` and `local_operator/mobile/web/src/lib/ask-open-policy.ts`)
on both surfaces this change touches: the TUI ask panel and the mobile relay's AsksSheet. Also
captured: clause 5 (typing while asks arrive) and the TUI's Tab handoff — see the composite
captions.

**Before** = a detached worktree at `e3176403b6` — the main the branch was cut from.
`local_operator/tui` and `local_operator/mobile/web` are byte-identical between that revision
and the current main, so it is current main for both surfaces. **After** = branch
`feat/asks-open-default-tui-relay` at `d340f8208f`.

The fail-on-old and mutation tables live in the PR body; this directory is the frame evidence.

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
| `relay-states-1-3-2.png` | Relay states 1, 3, 2 side by side (closed / closed / open) |
| `relay-4-dismissed.png` | Relay state 4 — close mark, then a re-publish, a NEW ask (dock counts 4) and away-and-back: still closed |
| `relay-5-typing.png` | Relay clause 5 — typing when the asks arrive: nothing is taken; the composer frame is pixel-identical |
| `raw/` | the un-composited frames plus a per-frame geometry JSON for every state, both builds |

## How they were captured

TUI (`100x30`, dark, one isolated `env -i` run per mode, `CMUX_*` absent):

```sh
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
  TERM=xterm-256color PYTHONPATH="$TREE" "$VENV/python" \
  scripts/ask_open_shot.py OUT/<mode>.svg 100x30 <mode>
# modes: no-asks pending pending-list pending-typed addressed dismissed-open dismissed-closed dismissed-back list-tab
rsvg-convert -z 1.25 OUT/<mode>.svg -o OUT/<mode>.png
```

The script drives the real `OperatorApp` (the one that loads the stylesheet) through a real
conversation switch; the seeded conversation rides on `Convo(history=…)` and is replayed by the
switch, so each frame answers "can the user read the conversation" as well as "is the surface up".

Relay (`390x844` CSS px, DPR 2, one headless Chrome per run, `--use-mock-keychain
--password-store=basic --headless=new --remote-debugging-port=0`, throwaway profile):

```sh
.venv/bin/python scripts/mobile_asks_open_capture.py OUT {before|after} --expect {before|after}
```

`--expect` turns every frame into a pass/fail check against what THAT build must do (a
before-set that opened a sheet, or an after-set that did not, exits red): all eight frames PASS
on both builds. Every frame is also held until the page stops moving (no finite animation
running, the sheet not still reading, the panel box unchanged across samples) — the first frame
is not the settled frame.

## Numbers beside the stills

TUI — screen and focus per mode (`screen` is 98x28 in every frame, virtual == size, no
scrollbar; the transcript is 21 rows of virtual 22 in every frame):

| mode | ask surface, before | ask surface, after | focus after |
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

The minimized bar (`AskBar`) is 1 row at y=24 in every frame where an ask is waiting, on both
builds; the auto-opened card is 14 rows at y=8 and the list 5 rows at y=17, above the same bar.

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

Repeatability: three AFTER relay runs are identical or differ by ≤72 px of ~1.3 M (max channel
delta 6/255 — sub-pixel anti-aliasing); before vs after, only state 2 differs materially
(1.31 M px), the rest are ≤72 px anti-aliasing. The TUI set re-captured at the pushed head is
byte-identical to the set these composites were built from.

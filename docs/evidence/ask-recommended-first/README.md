# Evidence: the recommended ask option is always presented first

Captured on macOS (Darwin 25.6.0, arm64) on 2026-09-02, against
`feat/ask-recommended-first` at 100x30, with `scripts/ask_shot.py` and its
`QUESTION` fixture changed to `recommended=2` so the hoist is visible.

| File | What it shows |
|---|---|
| `before.svg` / `before.png` | `main`: the recommendation is authored third and is drawn third, below two options ranked lower |
| `after.svg` / `after.png` | This branch: the same question, recommendation hoisted to row 1, the other three in their authored order |

Both frames come from the real `OperatorApp` through `run_test()`, so the
stylesheet is applied. The before-frame was captured from a throwaway worktree
at `main` with **its own** venv — a symlinked venv would have imported this
branch's source and produced two identical frames.

## What is proven

The recommendation reads as the obvious default at a glance rather than as a
tag the eye has to find. The number gutter renumbers with the list, so what a
user counts, what the gutter prints and what a digit key selects stay the same
thing.

Two properties the frames also settle:

- **The other options keep their authored ranking.** `Drop the rows` and
  `Backfill from the audit log` stay in that relative order after the hoist,
  because the normalizer rotates rather than swaps.
- **No layout regression.** `virtual_size == size`, no scrollbar, no clipping,
  and the first painted frame is identical to the settled one.

## What the frames cannot show

`recommended` never crosses the mobile wire — `PendingRequest` has no such
field — so on the phone the option's **position** is the only channel the
recommendation has. That is the reason the hoist lives in `AskQuestion` rather
than in the picker, and it is covered by tests in
`tests/unit/mobile/test_tui_ask.py` rather than by a frame.

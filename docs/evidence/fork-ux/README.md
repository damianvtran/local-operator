# Fork UX evidence

Rendered from the real `OperatorApp` via `run_test` (the host that loads
`local_operator.tcss`). Before frames are the same capture against
`origin/dev-fork-cache-proof` (506876e4); after frames are this branch.

| File | What it shows |
|---|---|
| `before-long-{100,80,70}.svg` | Shipped `(fork)` suffix, 56-char title: the mark is gone at every width |
| `after-long-{100,80,70}.svg` | `[fork]` prefix column: the mark survives; ellipsis eats the title |
| `after-short-100.svg` | Marked fork beside unmarked ordinary rows, names aligned |
| `after-named-100.svg` | Same list once the fork named itself: no tag, no reserved column |
| `after-filter-fork-100.svg` | Typing `fork` in `/resume` finds the tagged row (was zero rows) |
| `frame-*.png` | Browser stills of those SVGs, actually looked at |
| `read-cost.txt` | 1.00 reads/row on a fork-free store, identical to the base branch |
| `lifecycle.txt` | Real `fork_session()` → tag on, `set_conversation_name()` → tag off |
| `band-and-receipts.txt` | Pending-fork band segment at 120/100/80/60; the two receipt strings |

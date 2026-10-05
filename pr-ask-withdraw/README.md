# ask-withdraw — the agent-side settle (design §12)

Before/after pairs for the two settled surfaces, captured with `scripts/ask_queue_shot.py`
modes `bar-withdrawn` / `list-withdrawn`, 100x30, dark. **Before** = the same script run from a
detached worktree at `a8238e8ef` (main + the §12 note — the pre-code build); **after** = the branch
`feat/ask-withdraw` (PR head at the push that added these files).

Both halves render the SAME seeded log — `[a1 open, a4 open, a2 WITHDRAWN, a3 declined]` — through
the real `store.fold` (`_withdrawn_fixture_rows`), so the pair differs exactly where §12's rule
does: on the before build the fold skips the unknown kind, so a2 folds `open` and every surface
counts a question nobody will answer; on the after build it is settled.

| frame | before (main) | after (branch) |
|---|---|---|
| `bar-withdrawn` | `3 questions waiting` | `2 questions waiting` — the withdrawn ask is out of the answerable set, so the bar's count and head question drop past it |
| `list-withdrawn` (settled half) | header `3 questions waiting · All 4 · Waiting or moved on 3 · Settled 1`; the settled half shows only `Backfill from the audit log or drop the column? · declined` | header `2 questions waiting · All 4 · Waiting or moved on 2 · Settled 2`; the settled half shows `Rotate the deploy key before the cutover? · withdrawn` above `· declined` |

**Numbers beside the stills.** The partition stays total after the change:
`All 4 = Waiting or moved on 2 + Settled 2` (was `3 + 1`); the chip word is `withdrawn` in the
`muted` register (the `dismissed` family — a retraction is not a failure); each frame is a 98x28
screen with `virtual_size == size` (no scrollbar appeared) and exactly one painted row per ask.
Per-frame `.geometry.json` files sit beside the PNGs.

**Why the fixture keeps TWO asks open on both builds:** `_expand_asks` sends a queue with one
outstanding ask straight to its card, so a one-open-ask fixture would paint the list on `main`
(where a2 still counts) and a card on the branch — the pair would compare two surfaces instead of
one rule.

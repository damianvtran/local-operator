# Nested subagents: the rule, and the dock that shows the tree

## Acceptance contract

A subagent that holds the `task` tool delegates with it, at any depth; one that
was not given it may not create subagents at all and does the work itself. The
tree those delegations grow is visible and walkable in the dock: the roster names
whose children it is listing, a row says whether there is a level below it, and
the existing keys climb back up (`p`/`Esc`) or descend (`c`, Enter) one level at
a time.

## The rule, verified on the real CLI

Before the change (released build, `manager` child that HOLDS `task`, launching a
child of its own):

```
- depth1 task/wait/jobs/wake: task yes / wait yes / jobs yes / wake no
- depth2 launched: def4f980b313
- depth2 inventory (verbatim): `task` — not available ... `jobs` — available
```

After the change, same shape of probe, driven through the real CLI
(`lop exec` from the worktree, isolated `HOME`/`LOCAL_OPERATOR_CONFIG_DIR`, real
provider over OpenRouter, `/tmp` scratch):

```
● task depth1
● wait 921c70380cb0
● hub peek
depth2's answer, verbatim:

- task: available
- wait: available
- jobs: available
- wake: not available (mentioned in my instructions as a scheduling tool, but
  no such tool is exposed in this session)
- edit: available
- write: available
```

The store from that run holds three sessions — the root, `depth1`
(`origin: subagent`, label `depth1`) and `depth2` (same, label `depth2`) — and
the root's roster sidecar lists the job that launched `depth1`. `depth2`'s
execution ledger lived on `depth1`'s manager (the child that launched it), which
is why the dock walks the tree one level at a time rather than flattening it.

## The dock, in frames

Both frames are the real `OperatorApp` with production CSS, captured by
`scripts/nested_roster_shot.py` at **120x40** (960x680 px at rsvg-convert's
default 8 px cells):

```sh
.venv/bin/python scripts/nested_roster_shot.py /tmp/nested 120x40
rsvg-convert /tmp/nested/nested-root.svg -o root.png
rsvg-convert /tmp/nested/nested-scoped.svg -o scoped.png
```

`before-*` is the same script against `origin/main` (`git checkout origin/main --
local_operator/tui/app.py local_operator/tui/widgets/subagent_panel.py`, capture,
restore); the fixtures and the script are unchanged between the two runs, so the
only difference in each pair is the change under review.

| frame | before | after |
| --- | --- | --- |
| `*-root-roster.png` | `• Coordinate review  ⣻` — a row with a whole level under it and a leaf row are indistinguishable | `• Coordinate review ⊞1  ⣻` — the mark says the level exists; the neighbouring leaf row stays unmarked |
| `*-scoped-roster.png` | header `Subagents   ctrl+g` — the same word names a different list on every level | header `Subagents of Inspect documentation   ctrl+g` — whose children these are, with `Check the changelog` as the row and the page's own breadcrumb (`Conversation > Coordinate review > Inspect documentation`) and key hints (`p parent · c child · r root · esc back to parent`) above it |

Measured on the "after" frames: the label cell grows by exactly 2 cells
(`⊞1`), the panel's rung and column maths are untouched, and the scope label is
bounded by the live panel width and by `SCOPE_CEILING` (24 cells) so the
content-sized `#band` cannot be widened past its rows by a long model-authored
label.

## The narrow dock (review round 1's D1/D2)

Two of the round-1 findings were about widths this folder did not photograph, so
the pair below is taken at the sizes they were measured at:

| frame | what it shows |
| --- | --- |
| `after-root-roster-60x30.png` | 60x30: `• Coordinate rev… ⊞1` — the label truncates with its ellipsis and the MARK SURVIVES. Before the round's fix the mark was appended inside the label and then truncated positionally, so a 21-cell label kept its mark only from 76 columns and a parent row was indistinguishable from a leaf below that. |
| `after-scoped-roster-60x30.png`, `after-scoped-roster-40x30.png` | A page open at 60 and at 40 columns: `Subagents of Inspect documentation…` and `Subagents of Inspect docum…`, each with `ctrl+g` still ON SCREEN and the dock inside the terminal. Before the fix the scope was bounded by a constant sized against an assumed 58-cell dock, so at 40 columns the header string was 43 cells in a 45-cell region, `ctrl+g` fell off the right edge and the dock stopped following the terminal. |

Both were re-taken from the remediated tree with the same script and the same
fixtures, so the pairs differ by the change under review and nothing else.

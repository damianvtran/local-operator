# PR for #641: `/model` follow-up frames

Rendered before/after stills for the design findings in issue #641 (deferred
from PR #625). Kept on this orphan evidence branch rather than in the PR's own
tree, per `AGENTS.md` §"Evidence goes on the PR, never into the repository".

## Apparatus

- **before** — `f4e9613a9` (`origin/main` at the time of the PR), captured from
  a throwaway worktree with its own editable venv, verified resolving to itself.
- **after** — the `dev-641-model-followups` branch.
- Both halves captured by the **same script** against different roots, so the
  frames differ only by the code under test.
- `scripts.visual_capture.save_capture` (native 8x17px cells, system monospace,
  no window chrome), rasterised with `rsvg-convert`.
- Process re-homed to a temp `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` and every
  inherited `CMUX_*` variable dropped before any `local_operator` import.

## Frames

| Pair | Finding | What changed |
| --- | --- | --- |
| `*-help-80x44.png` | D7 | `/help` row lead: `Switch;` (fragment, 70 cells) → `Switch model;` (sentence, 74 cells; description 54 cells, under the 55-cell fold) |
| `*-notice-80x30.png` | D8/D10 | Last row was the orphaned `set it in /settings`; now `reverts to it` |
| `*-notice-100x30.png` | D8 | Same clause, second reviewed width |
| `*-notice-120x30.png` | D8 | Settles to two rows, no orphan |
| `*-u3-100x30.png` | U3 | `no matching models` directly above a footer advertising that phrase → `default is a command, not a model — enter runs it` |
| `*-setup-100x30.png` | D10/U12 | No-model setup variant: keeps `/model default` (its escape works) and loses the dangling `it` |
| `*-deadend-100x30.png` | U12 | Unknown-hosting variant: notice **and picker footer** stop advertising `/model default`, which answers `session is still starting…` there; `/settings` is named instead |

The `deadend` pair is the one to read closely: on `before`, `/model default` is
printed twice (notice and footer) in a state where no `/model` route resolves.

## Remediation round 1 (`r1-*`)

Re-rendered after the round-1 remediation commit for the three findings that
changed user-visible copy. Same apparatus as above; the `r1-before-*` half is
the PR head **`9e830ff97`** (i.e. the frames the review rounds looked at), not
`f4e9613a9`, because these three findings are about the state this PR shipped.

| Pair | Finding | What changed |
| --- | --- | --- |
| `r1-*-notice-100x30.png` | D1 / U3 | On `before`, row 2 ends on the bare token `/model` and row 3 opens `saved reverts to it` — the command name split across the fold, in the frame originally attached as D8's evidence. `/model saved` now sits at its clause's end and reads whole. |
| `r1-after-notice-{80,120}x30.png` | D1 / U3 | The other two reviewed widths, both clean. |
| `r1-*-u3-50x30.png` | D2 | On `before`, the keyword row is 49 cells against a 47-cell budget and truncates to `…not a model — enter…`, cutting the actionable half. Now `default is a command — enter runs it`, whole at 50 columns. |
| `r1-*-u3-prefix-100x30.png` | U1 | `/model defaul`, one keystroke short of the word. On `before` the list says `no matching models` directly above a footer advertising `/model default`; now the row reads `default is a command — keep typing`. |
| `r1-after-u3-100x30.png` | U3 | The settled state at the reviewed width, unchanged in behaviour and re-captured against the new copy. |

Frames unaffected by this round (`*-help-80x44`, `*-setup-100x30`,
`*-deadend-100x30`) are **not** re-rendered: the remediation does not touch the
`/help` row, and the setup/dead-end notice variants keep the clause they had.

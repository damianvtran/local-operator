# PR #671 — design remediation round 2 evidence (D8)

Captured in the worktree `/tmp/lop-live-settings`, before-frames at review head
`6344d2be7` and after-frames at the rebased remediation head `fe8620a6f`
(rebased onto `db018cadb`, v0.49.6+). Real `OperatorApp` via `run_test` +
`save_capture` (production CSS), rasterised with `rsvg-convert` and looked at.

D8 has two halves, and the frames below carry both.

## What drives them

`d8_shot.py` (this directory) drives three topologies through the REAL gesture
and a REAL another-process `config.yml` write — no mocks of the notice paths:

| cell | topology | how the carrier gets `ask` |
|---|---|---|
| `detached` | TUI owns the gate (`is_remote` False) | `/approvals ask` in this pane |
| `attached` | production `OwnedSessionHandle` + `RuntimeServer`, TUI is a viewer over a real `RemoteSession` | `/approvals ask` ROUTES to the runtime (`app.py` slash routing → `owned.py:2625`) |
| `attached-double` | same, and BOTH carriers hold `ask` | `/approvals ask` routes to the runtime AND a `/settings` page write (`settings_io.write_setting`, the exact call `SettingsView._write` makes) records the viewer's own |

`attached-double` is the designer's reported topology: both keep-branches fire
on one poll.

## Frames

| file | what it shows |
|---|---|
| `before-detached-100.png` | **Half 1, reproduced.** Keep notice, then `config.yml changed: applied: tool_approval_mode` one row below it. The gate did not move; the next row says the key was applied. |
| `after-detached-100.png` / `-80.png` | The `applied:` row is gone. `tool_approval_mode` was the only live key, so no generic line prints at all — the standard the refusing `model` section already sets. One receipt, 2 rows. |
| `before-attached-double-100.png` | **Half 2, reproduced.** Five visual rows for one non-event: the same 118-character sentence twice, with the contradictory `applied:` line wedged between. |
| `after-attached-double-100.png` / `-80.png` | One keep receipt, no `applied:` line. 5 rows → 2. The runtime owns the gate, so the runtime's notice is the only one. |
| `after-attached-100.png` / `-80.png` | The non-regression cell: only the RUNTIME's carrier holds `ask`, this viewer never typed it, so this viewer's gate legitimately follows the file to `auto` and `applied: tool_approval_mode` is still printed — correctly, because here the key really did move. Byte-identical to the pre-fix frame; the fix does not touch the applying path. |

## Counts read off the running app

```
                     before                          after
detached          keep=1  applied_lines=1        keep=1  applied_lines=0
attached          keep=1  applied_lines=1        keep=1  applied_lines=1   (unchanged: the key moved)
attached-double   keep=2  applied_lines=1        keep=1  applied_lines=0
```

## Geometry behind the pixels

`*.geometry.json` accompanies each SVG. In every frame, before and after:
no scrollbar on any widget, and `TranscriptView` size/virtual_size is
`[97,15]/[96,15]` at 100 cols and `[77,15]/[76,15]` at 80 — identical before
and after, so the only delta is the removed rows, with no reflow.

## Regression pins

Three assertions in `tests/unit/tui/test_config_change_notice.py` fail against
the pre-fix code and pass after (verified by reverting each half in place):

- `test_a_loosening_does_not_revoke_an_explicit_approvals_ask` — the keep path
  prints no `config.yml changed` line at all when the mode was the only key.
- `test_a_refused_loosening_drops_only_its_own_key_from_applied` — the batched
  case a naive fix breaks: a refused `tool_approval_mode` beside a key that DID
  apply must still print, naming only the applied key.
- `test_only_the_owning_process_prints_the_keep_notice` — with a runtime
  attached, this process emits no keep notice.

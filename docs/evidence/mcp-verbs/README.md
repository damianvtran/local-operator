# Testing evidence for the `/mcp` verb set (issue #368, PR 2 of 2)

Real execution captures, not a green suite. Every script uses a **temp HOME**
(`mktemp -d`) so the operator's real `~/.local-operator/mcp.json` is never read
or written, and the synthetic configs contain no secrets or tokens.

| file | what it proves |
|---|---|
| `cli_check.sh` / `cli_ab.txt` | `lop mcp list\|add\|remove` stderr text and exit codes are **byte-identical** to `origin/main` after the structured-result refactor. Run against both trees and diffed. |
| `tui_check.py` / `tui_check.txt` | The real `OperatorApp` driven through `_type_command`: `/mcp list`, `add` for stdio and http, the duplicate/extra-token/usage refusals, `remove` succeeding on an owned server, `remove` REFUSING a Claude-imported one, and the unknown-verb error. Shows the resulting `mcp.json` after each step. |
| `audit.py` / `audit.txt` | Every `/mcp` verb's rows checked against what the verb actually destroys, with the editor's real destructive-gate verdict per row. Run on head + PR #378 (the state this ships into): it FAILS on `reauth` before the round-1 fix and passes after. |
| `reauth_repro.py` / `reauth_repro.txt` | The M2 key-path repro: `/mcp reauth lnr` + Enter used to FIRE and forget the grant of a server the user never spelled; it now fills. |
| `add_ownership_check.py` / `.txt` | Round-1 M1/M3/N1/N2 on the real `/mcp` path: `add` refusing to shadow a `~/.claude.json` entry, refusing the higher-priority `<cwd>/.mcp.json` case that used to print "added" for a write with no observable effect, the corrected OAuth hint, and the trailing-token refusals. Prints the EFFECTIVE server before and after each attempt, so a no-op is visible rather than asserted. |
| `gate_repro.py` / `gate_repro.txt` | The data-loss defect behind the xfail test: a fuzzy `/mcp remove fsy` + Enter **deleted `filesystem`** from `mcp.json`, because the editor's destructive gate matches the command WORD (`logout`) and never covers `mcp`. Fixed by PR 1. |

`audit.py` and `reauth_repro.py` expect a worktree at `/tmp/lop-mcp-merged`
holding this branch with PR #378 merged in (they assert the gate behaviour that
only exists once both halves are present); the others run against this tree.

Reproduce any of them with:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python docs/evidence/mcp-verbs/tui_check.py
bash docs/evidence/mcp-verbs/cli_check.sh /path/to/worktree
```

## Rendered frames

`before-verbs.png` / `after-verbs.png` — the `/mcp ` argument picker. Before:
three verbs (`login`, `logout`, `reauth`). After: six, with `list` leading as
the safe landing row and `remove`/`logout` in the danger tint.

`after-remove.png` — the `/mcp remove ` rows. All four configured servers are
offered (the stdio and non-OAuth http ones were previously invisible), each
carrying its **source file** in the detail column, so the Claude-imported
`notion` row shows `~/.claude.json` and is visibly not ours to delete.

Captured with `shot.py` per AGENTS.md "Visual validation", using the real
`OperatorApp` so the stylesheet applies.

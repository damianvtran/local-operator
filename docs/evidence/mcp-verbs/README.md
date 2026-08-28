# Testing evidence for the `/mcp` verb set (issue #368, PR 2 of 2)

Real execution captures, not a green suite. Every script uses a **temp HOME**
(`mktemp -d`) so the operator's real `~/.local-operator/mcp.json` is never read
or written, and the synthetic configs contain no secrets or tokens.

| file | what it proves |
|---|---|
| `cli_check.sh` / `cli_ab.txt` | `lop mcp list\|add\|remove` stderr text and exit codes are **byte-identical** to `origin/main` after the structured-result refactor. Run against both trees and diffed. |
| `tui_check.py` / `tui_check.txt` | The real `OperatorApp` driven through `_type_command`: `/mcp list`, `add` for stdio and http, the duplicate/extra-token/usage refusals, `remove` succeeding on an owned server, `remove` REFUSING a Claude-imported one, and the unknown-verb error. Shows the resulting `mcp.json` after each step. |
| `gate_repro.py` / `gate_repro.txt` | The data-loss defect behind the xfail test: a fuzzy `/mcp remove fsy` + Enter **deleted `filesystem`** from `mcp.json`, because the editor's destructive gate matches the command WORD (`logout`) and never covers `mcp`. Fixed by PR 1. |

Reproduce any of them with:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python .evidence/tui_check.py
bash .evidence/cli_check.sh /path/to/worktree
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

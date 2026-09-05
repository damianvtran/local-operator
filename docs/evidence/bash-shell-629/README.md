# Evidence for #629 — bash tool runs real bash

Data-only branch; captured from worktree `/private/tmp/lop-629` on
`dev-bash-shell-629` (macOS arm64, Homebrew bash 5.3.9 on PATH, `/bin/sh` =
bash 3.2.57 POSIX mode).

- `1-shell-repro.txt` — the issue's two-line shell demonstration.
- `2-execute-bash-main.txt` — `execute_bash` from the main checkout's venv (origin/main tree) failing.
- `3-execute-bash-branch.txt`, `3b-...-arrays.txt` — same call on the branch: exit 0, plus arrays/`[[ ]]`.
- `4-configured-override.txt` — `bash.shell` set via `lop config edit` in an isolated config dir; `/bin/sh`, `/bin/bash`, then cleared.
- `5-e2e.txt` — `tests/e2e -m e2e -n0` on the branch.
- `settings-before.png` / `settings-after.png` — `/settings` at 110x36, real `OperatorApp`, cursor paged to the Web fetch / Tools section (SVGs alongside).

## Round-1 remediation (unified: F1/F2/F3 + Q1 + D1)

- `6-bad-shell-repro.txt` — QA's own Q1 reproduction re-run against the fix:
  `bash.shell` set to `/nonexistent/qa-shell`, to a non-executable file, and to
  an unexpanded `~/nonexistent/bash`, each through the real tool boundary
  (`AgentTool.execute`, the decorator that used to render the traceback). All
  three return a normal `is_error: True` result naming the path and the key; the
  cleared key still runs `echo hello` at exit 0.
- `r1-settings-{before,after}-{110x36,80x24}.{svg,png}` — `/settings` with the
  cursor driven by real key presses onto `Bash interpreter`, so the detail line
  paints that row's help. Before = the round-1 head (`f3cc282`), after = the D1
  copy. Geometry identical in every pair: `size == virtual_size` (106,27) at
  110x36 and (76,15) at 80x24, both scrollbars `False`. The new string is 69
  cells against a 74-cell detail budget at 80x24, so it renders whole where the
  73-cell original survived only because the key path shed first.

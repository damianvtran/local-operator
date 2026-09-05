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

# Evidence frames

Screenshots attached to PR "fix(providers,tui): let every paste-a-key provider
actually log in". Captured from the real `OperatorApp` (the host that loads
`local_operator.tcss`) via `scripts/shot_login.py`.

- `shots/before.png` — `/login alibaba` on `origin/main` @ 42e5268 (the bug)
- `shots/after_empty.png` — the prompt, empty state @ 5d1b581
- `shots/after_typed.png` — 15 characters typed, masked @ 5d1b581
- `shots/after_submitted.png` — submitted, key stored @ 5d1b581
- `shots/after_cancel.png` — Escape, reported as a cancel @ 5d1b581

## Round 2 (after review remediation) @ 77065d1

- `shots/r2_empty.png` — prompt, empty state
- `shots/r2_typed.png` — typed, masked, label no longer duplicated
- `shots/r2_submitted.png` — submitted receipt with the ✓ glyph
- `shots/r2_cancel.png` — cancelled receipt with the ✗ glyph

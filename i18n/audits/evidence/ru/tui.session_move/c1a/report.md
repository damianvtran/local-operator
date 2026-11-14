# i18n audit report — ru · tui.session_move · audit `c1a`

- Locale: ru · namespace: `tui.session_move` · keys: 7 · date: 2026-11-11
- Auditor: i18n-auditor, session `e45ca4d0d71c`, actual served model `deepseek/deepseek-flash` (harness model line; no fallback used). Independent of the translator (`openai/gpt-6.1-sol`).
- Verdict: **clean-with-findings** — BLOCKER 0 · MAJOR 0 · MINOR 1 · NIT 1; ledger status **`passed`** (no blocker/major remains).

Inputs, hashes, mechanical results, commands and the ledger-status reasoning are in the full slice report `audit-ru-A.md` (retained beside the manager's c1-prep inputs) and in `audit.log` here. `record.json` lists the hashes; `audit.log` is captured command output.

## Findings

### M5 — MINOR — `tui.session_move.deadline_exceeded`

- Current: `перемещение не завершилось за отведённое время; повторный запуск `lop sessions move {session} --to {to}` покажет, на каком этапе оно находится`
- Problem: The required meaning is that the outcome is UNKNOWN. «на каком этапе оно находится» is a present-tense claim that the move is still sitting at some stage — it reads as «still in progress», where en «where it got to» leaves completion open. Nothing here says «failed», so the time-out framing itself is correct.
- Suggested: `…; повторный запуск `lop sessions move {session} --to {to}` покажет, до какого этапа оно дошло (139 cells)`

### N5 — NIT — `tui.session_move.to_needs_device`

- Current: `Для --to нужно указать устройство: /move --to <пир|local>`
- Problem: Optional: a usage hint that opens with a preposition before a Latin flag reads cleaner as the flag as subject, and is 10 cells shorter (47 vs 57).
- Suggested: `--to требует устройство: /move --to <пир|local>`

## Judged and accepted

- Safety semantics of the `cli.config.running_sessions_unchanged` line are correct: a config write does not loosen a running session, `/approvals auto` is per session and verbatim, new sessions start at `auto`.
- The `/move --to` refusal register (lowercase clauses) matches shipped `tui.notices.*`; `<path>`, `<session>`, `<peer>` are translated inside kept angle brackets while `/move`, `--to`, `local` and the backticked shell command stay byte-identical.
- `tui.session_move.deadline_exceeded` never claims failure or success (see M5 for the one nuance).
- No stray English: the only Latin runs are kept identifiers, flags, provider names and `CLI`/`shell`.

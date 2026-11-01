# ru / tui.small: core catalogue gap audit (cg1)

137 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **RU-001** (round 1, MINOR) `tui.small.spend.unpriced_cancelled_shortest`
  - English: 'bill unknown'
  - Value at audit: 'неизвестно'
  - Problem: 'bill unknown' → bare «неизвестно» drops the billing subject; at the 12-cell rung it reads as a generic 'unknown' and no longer says it is the charge that is unknown (the note requires the cancellation/billing reason to survive).
  - Suggested: 'сумма неизв. (12 cells)'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-002** (round 1, MINOR) `tui.small.ask_queue.urgent_clause`
  - English: '{shown} urgent'
  - Value at audit: '{shown} срочных'
  - Problem: No plural branch, but genitive plural «срочных» is ungrammatical for 1/21 («1 срочных»); same for ask_queue.waiting_clause «{shown} ждут» («1 ждут»). Use a count-neutral label form.
  - Suggested: 'срочно: {shown} (and waiting_clause → ждут: {shown})'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-003** (round 1, MINOR) `tui.small.goal_panel.settled_heading`
  - English: 'settled'
  - Value at audit: 'завершены'
  - Problem: Context says the heading is a noun, not a verb; «завершены» is a predicate (short participle, 'are settled').
  - Suggested: 'завершённые'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU2-005** (round 2, NIT) `tui.small.ask_queue.urgent_clause`
  - English: '{shown} urgent'
  - Value at audit: 'срочно: {shown}'
  - Problem: The RU-002 fix is grammatical now, but «срочно:» is an adverb ('urgently: 2'). The usual Russian count-label form is the genitive plural before a colon (as in «ещё задач: {count}»). The sibling «ждут: {shown}» keeps a plural verb, so «ждут: 1» still jars a little.
  - Suggested: 'срочных: {shown} · waiting_clause: «в ожидании: {shown}» (timed_out_clause «просрочено: {shown}» is fine)'
  - Disposition: open at PASS: minor/nit accepted, value unchanged
- **RU2-006** (round 2, NIT) `tui.small.todo_panel.dropped`
  - English: '{count} dropped'
  - Value at audit: '{count} отменено'
  - Problem: Same count-agreement family as RU-006: «{count} отменено». The sibling done_hidden «+{count} выполнено» and resolved «{resolved}/{total} закрыто» have the same pattern. These are tolerable, but the label form would match the remediated clauses.
  - Suggested: 'отменено: {count} · done_hidden «+выполнено: {count} · »'
  - Disposition: open at PASS: minor/nit accepted, value unchanged

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session 4f28a4bcc8e2)
# ru core catalogue patch audit — independent
Verdict: **CHANGES_REQUESTED** — BLOCKER 0 · MAJOR 3 · MINOR 17 · NIT 5 · 1311 values in 56 namespaces
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback). Independent of translator session 3ef6295a5545. No MT/translation service used. Read-only on the repo; inputs read from origin/main via `git cat-file --batch`.
## Blocking findings (must fix before `passed`)
- RU-008 tui.hub_links.action.enable: Glossary 'Wave-2 kept-English set' keeps `push`/`pull` Latin for hub operations, and the existing ru neighbour wire.slash.hub already says «auto-push». The patch coins «автопубликация» throughout (also hint.auto_push, policy.auto_push.label

## Could not verify
- No rendered terminal frames. The no-builds/no-tests constraint ruled them out, so widths are cell counts (East-Asian-width rule), not rendered.
- No context sidecar exists for tui.import, tui.library, wire.errors.agents/auth/files/media, wire.exec or wire.slash.library, and one key in tui.hub_links has none. For those, fit was judged by reading only.
- The repo validator (`scripts/i18n/check.py`) and the ledger write were not run: read-only instruction, and the patches are scratch files, not catalogue files yet. **No ledger row was recorded.** One should be written as `changes_requested` once the patches land in a tree.

### Round 2 (session 4586ac33eefc)
# ru core catalogue patches: independent round-2 audit
**Verdict: PASS.** BLOCKER 0, MAJOR 0, MINOR 4, NIT 4.
- **Auditor:** i18n-auditor, model `anthropic/claude-opus-5-5` as reported by the runtime. No fallback was observed.
- **Independence:** I did not write the translation, the round-1 findings, or the remediation.
- **Method:** I used no machine translation or translation service, and I edited nothing in the repo or the patches. English source and context sidecars came from `origin/main` through `git show`, extracted to `core-audit2/ru/en/`.

## Not verified

- **No rendered terminal frames:** this was a read-only audit with no builds or tests. Widths are computed cell counts.
- **ICU arm widths:** only one width-limited value contains ICU, so arm widths were checked by reading.
- **Missing context sidecars:** `tui.import`, `tui.library`, `wire.errors.*` (except sdk_*), `wire.exec` and `wire.slash.library` have none, so fit there was judged by reading.
- **Repo tooling and ledger:** I did not run `scripts/i18n/check.py` or write a ledger row. That follows the read-only instruction, and the patches are not yet catalogue files. **No ledger row was recorded.**
- **Untouched values:** I read the ~1,200 untouched values only for the sweep in (C), not for full accuracy.

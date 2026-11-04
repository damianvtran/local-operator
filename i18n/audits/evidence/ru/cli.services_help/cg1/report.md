# ru / cli.services_help: core catalogue gap audit (cg1)

9 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **RU-022** (round 1, MINOR) `cli.services_help.services`
  - English: "Bring this machine's non-runtime services (serve daemons and the supervised daemons) onto the current build, without stopping any running conversation"
  - Value at audit: 'Перевести нерантаймные службы этой машины (демоны serve и демоны под надзором) на текущую сборку, не останавливая ни один работающий разговор'
  - Problem: Glossary §2: supervised → «под управлением» (used by the ru neighbour cli.services_status «под управлением: {unit}»). The patch uses «под надзором» here and in services_description; reclaim_description uses «надзирать».
  - Suggested: '…(демоны serve и демоны под управлением)…; services_description: «…или LaunchAgent под управлением…»; reclaim_description: «…демон может управлять рантаймами сессий…»'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-023** (round 1, MINOR) `cli.services_help.wait`
  - English: 'How long to wait for an asked daemon to come back on the new build (default: {wait_seconds}s; must be positive). A daemon that does not make it is reported, left serving the build it loaded, and retried by the next `lop services restart`.'
  - Value at audit: 'Сколько ждать, пока запрошенный демон вернётся на новой сборке (по умолчанию: {wait_seconds} с; должно быть положительным). Демон, не успевший вернуться, указывается в отчёте, остаётся на загруженной сборке и повторяется следующим `lop services restart`.'
  - Problem: «и повторяется следующим `lop services restart`» makes the daemon the thing that 'is repeated'; English: it is retried (the restart is re-attempted for it).
  - Suggested: '…остаётся на загруженной сборке, и следующий `lop services restart` повторит попытку.'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-024** (round 1, NIT) `cli.services_help.reclaim`
  - English: 'End a `lop serve` daemon that is recorded but not serving its address (never one that is serving)'
  - Value at audit: 'Завершить демон `lop serve`, который записан, но не обслуживает свой адрес (но не тот, что обслуживает)'
  - Problem: «(но не тот, что обслуживает)» is elliptical; the safety guarantee should be explicit.
  - Suggested: '…(и никогда не тот, который его обслуживает)'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)

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

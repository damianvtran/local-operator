# ru / tui.hub_links: core catalogue gap audit (cg1)

56 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **RU-008** (round 1, MAJOR) `tui.hub_links.action.enable`
  - English: 'Enable auto-push'
  - Value at audit: 'Включить автопубликацию'
  - Problem: Glossary 'Wave-2 kept-English set' keeps `push`/`pull` Latin for hub operations, and the existing ru neighbour wire.slash.hub already says «auto-push». The patch coins «автопубликация» throughout (also hint.auto_push, policy.auto_push.label, confirm.title, mark.healthy, refusal.ineligible, reason.no_receipt), so the /hub command description and the panel it opens name the feature differently.
  - Suggested: 'Включить auto-push; label «Auto-push»; hint «a auto-push»; confirm.title «Включить auto-push в {source}?»; mark.healthy «…; auto-push: {auto_push}»; refusal/reason «auto-push недоступен…» — apply consistently across all listed keys'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-009** (round 1, MAJOR) `tui.hub_links.action.pull`
  - English: 'Pull now'
  - Value at audit: 'Получить сейчас'
  - Problem: Same glossary rule: hub 'pull' stays Latin (neighbours: «полученных через pull»). «Получить сейчас» / «Ручное получение» (policy.subscribe.help) / «получите обновления» (reason.baseline_unknown, reason.no_receipt) drop the term; reason.baseline_unknown also adds «разрешите конфликты», which English does not say ('resolve it').
  - Suggested: 'Выполнить pull; policy.subscribe.help «…Ручной pull остаётся доступным.»; reason.baseline_unknown «…Сначала выполните pull или разрешите её состояние.»; reason.no_receipt «…один раз выполните pull или опубликуйте…»'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU2-008** (round 2, NIT) `tui.hub_links.action.pull`
  - English: 'Pull now'
  - Value at audit: 'Выполнить pull'
  - Problem: «Выполнить pull» drops 'now', which separates the manual action from the scheduled subscription it sits beside.
  - Suggested: 'Выполнить pull сейчас'
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

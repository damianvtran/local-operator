# ru / mobile.setup: core catalogue gap audit (cg1)

76 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **RU-013** (round 1, MINOR) `mobile.setup.activated`
  - English: 'Existing tunnel activation saved; owning credential unchanged.'
  - Value at audit: 'Активация существующего туннеля сохранена; учётные данные-владельца не изменились.'
  - Problem: «учётные данные-владельца» is a hyphenation error (an attributive genitive is not hyphenated); reads as a broken compound.
  - Suggested: 'Активация существующего туннеля сохранена; учётные данные владельца не изменились.'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-014** (round 1, MINOR) `mobile.setup.identity_unavailable`
  - English: 'identity unavailable — check your signed-in accounts'
  - Value at audit: 'идентификатор недоступен — проверьте выполненные входы в учётные записи'
  - Problem: 'identity' here is the (masked) account identity shown in `account`; «идентификатор» means an identifier/ID and collides with the separate credential id. Also «выполненные входы» is stilted.
  - Suggested: 'учётная запись не определена — проверьте, в какие учётные записи выполнен вход'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU2-002** (round 2, MINOR) `mobile.setup.owner_missing`
  - English: 'Run lop login-status; restore the owning login. Setup will not repoint an existing tunnel or pending request.'
  - Value at audit: 'Выполните lop login-status и восстановите вход в учётную запись-владельца. Настройка не будет перенаправлять существующий туннель или ожидающий запрос.'
  - Problem: Same class as RU-013 (hyphenated 'owning' compound). This key, mobile.setup.login («войти в учётную запись-владельца») and mobile.setup.checking («Проверка учётной записи-владельца») hyphenate a feminine noun to the masculine «владелец». That reads as a broken compound or a forced apposition. English 'owning login/account' means the account that owns the tunnel.
  - Suggested: 'owner_missing: «Выполните lop login-status и восстановите вход в учётную запись — владелицу туннеля. …», or use «…в учётную запись, которой принадлежит туннель…». Apply the relative-clause form to login: «…чтобы войти в учётную запись, которой принадлежит туннель, затем повторите.» and checking: «Проверка учётной записи-владельца…» → «Проверка учётной записи, которой принадлежит туннель, локальных требований и мобильного маршрута…»'
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

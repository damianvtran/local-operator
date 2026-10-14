# Independent Russian final65 audit — core PR #2329

**Textual/native result: passed, clean with one MINOR follow-up; terminal.** No blocker/major findings. **Rendered UX: BLOCKED — customer resource hold.** This is not programme completion, ledger permission, shipping eligibility, a PR code/design/QA review substitute, or release approval.

Auditor: independent task `847d8d9b4185`; actual model **openai/gpt-6.1-sol**. No model fallback observed here; no preferred-model claim. Date: 2026-10-14. Source HEAD: `de9b9a490c839d6de2c2e41ea03ec819f66be489`. Translator independence: I did not translate these targets.

## Exact pinned bytes

| Namespace | Keys | English SHA-256 | Russian target SHA-256 | Native / mechanical |
|---|---:|---|---|---|
| relay.shell | 44 | `8b17439e9f31f159bc60a9e2940eb2d88400dacdcda9854ae1b3c8eb0ca00586` | `076b295ad1f4b9a90ef02424063f716c84af3cbf85ae41f1c53ce724e1d066ff` | passed / PASS |
| relay.session | 17 | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec` | passed / PASS |
| relay.login | 4 | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1` | passed / PASS |

Source root: `~/local-operator/.worktrees/coder-2329-convergence-20261014/local_operator/i18n/catalogues/en/`. Target root: `historical-scratch/a753be6f0d01/relay-final65/ru/`. Hashes rechecked at report creation; all match requested EN hashes. Targets and checkout untouched.

## Native review

- **relay.shell:** 44/44 native-text reviewed against EN and all context sidecars. Established сессия/использование/кэш/токен glossary; waiting means session data, not a projection prediction. Paid/included/family quotas remain distinct; unknown amounts, eligibility and windows are not zero. Unit labels as parenthetical independent labels avoid numeric declension at 0/1/2/5/21/fractions. Complete row retains all fields, opus unchanged, dot separator and nonempty family parentheses. Currency USD unchanged. Numeric formats and whole unknown-unit messages preserve values without inventing units. One nonblocking command-description polish finding.
- **relay.session:** 17/17 native-text reviewed against EN/context. Formal вы realized through запустите/отправьте/введите; infinitive theme action elsewhere consistent. сессия is the technical running entity, беседа is conversation contents; names and conversation-body search retained. TUI Latin remains per glossary. Loading/starting/empty states remain distinct, grammatical feminine рабочая not introduced because каталог masculine; рабочий каталог correct. Ellipses/case retained.
- **relay.login:** 4/4 native-text reviewed against EN/context. local operator brand unchanged; вход page title / пароль label / войти action / incorrect-password refusal idiomatic, neutral; period retained. Exact production login HTML function escaping and full-page English fallback exercised with fixture dependencies, not a live server.

### RU-M1 — MINOR — command-description idiom

`historical-scratch/a753be6f0d01/relay-final65/ru/relay.shell.json:33`, `relay.shell.usage.command`: **Использование провайдера из кэша**. Source: **Show cached provider usage**. Used as /usage command description at components/composer.tsx:203. The Russian genitive phrase reads as using the provider from the cache, rather than showing cached provider usage data. The neighboring sheet copy correctly names данные об использовании. Suggested: **Показать данные об использовании из кэша**. Nonblocking polish, not a required remediation round.

## Executed matrix and actual outputs

- All 65 keys: exact key parity, parsed ICU argument encounter tuples (name/type/style, including the repeated date/time value), ellipsis preservation, successful source-runtime formatting. All PASS. See JSON `rows` for every EN/RU value, tuple and sample. Order-sensitive encounter tuples are a narrow validator, not a universal language-order requirement; no reordered argument was falsely rejected in these targets.
- `.29` / `.07` fractions: `использовано 29 %` / `использовано 7 %`; remaining `осталось 29 %`. Percent uses NBSP; no duplicated percent or translation-side rescaling.
- `123.5` numeric: `Остаток: 123,5 (кредиты)`; unknown unit: `осталось 123,5` / `использовано 123,5`. Decimal comma retained.
- Surprising direct percent input `123.5` produces `12 350 %`, correctly because ICU expects a fraction. Verified one step further: actual UsageSheet line 56 divides API percent points by 100; `123.5 / 100 = 1.235` produces `осталось 124 %` with source-specified integer rounding. This is not a translation defect.
- 32 agreement/full-row probes: four known units × 0/1/2/5/21/.29/.07/123.5. Example: `Квота семейства моделей (opus) · 5 часов: Остаток: 123,5 (запросы)`. Parenthetical standalone nominative unit label avoids incorrect `1 кредиты`/`2 кредиты` syntax; compact and understandable native UI labeling.
- Date UTC 1791981240000: `Сброс: 14 окт. 2026 г., 12:34`; as-of `По состоянию на 14 окт. 2026 г., 12:34.` All punctuation preserved, 24-hour clock from RU Intl. No hardcoded date pattern.
- Missing RU namespace or partial RU bundle: `uiLocaleFor` returns `en`. Forced RU context + missing key still formats English source with EN rules: `123.5 credits left`, not `123,5 credits left`. Actual TS `sourceFor`/`render` path exercised. Malformed/missing params degrade to message key, as source contract states.
- Negative controls: deleting one key in each namespace rejected; replacing percent arg with plain string, plain number, or wrong arg name rejected (`argument type/style/encounter mismatch`). All 6 controls PASS.
- Login exact `_render_login_html` AST with fixture catalogue/resolver and production `render_message`: full RU renders lang=ru/dir=ltr; missing refusal key forces entire EN page and `Wrong password.`; hostile title becomes `&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; __PASSWORD__`, proving HTML escaping plus single-pass substitution. No live daemon/browser claim.
- ICU `<script>` text remains a literal string, not interpreted markup. React source emits strings in JSX; no visual/browser security execution claimed.
- 65 synthetic expanded pseudo strings format successfully. **Pseudo layout/width: BLOCKED.** RU CLDR one/few/many/other observed; no plural ICU messages in scope, so plural branches N/A. **RTL native/layout: N/A for RU (LTR), no Urdu claim.**

## Evidence and limits

Commands executed with `env -i`, isolated HOME/config and TZ=UTC, no inherited CMUX or live session identifiers; existing Node/TypeScript/Intl and repository Python venv, no installs/copies/heavy suites/browser. Scripts and detailed actual outputs are beside this report. `ru-final65-audit.json` includes all 65 per-key review results and outputs plus exact hashes.

Rendered loading/empty/error/populated states, narrow-phone widths, font clipping and interaction flows remain **BLOCKED by explicit customer resource hold**. Mechanical/native text pass cannot clear those gates. No target edits, ledger entries, shipment claim or review substitution.

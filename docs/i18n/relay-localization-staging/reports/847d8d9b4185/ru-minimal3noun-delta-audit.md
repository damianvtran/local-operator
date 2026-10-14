# Independent RU counted-noun delta audit

**Native verdict: passed, terminal.** No new findings; no unresolved blocker/major. Actual model **openai/gpt-6.1-sol**, independent task `847d8d9b4185`, no fallback observed. Did not translate or edit these targets.

Scope: ONLY `unit_credits`, `unit_tokens`, `unit_requests` and composed direct neighbors. Source is the **uncommitted working-tree delta**, not committed contents of HEAD `de9b9a490c839d6de2c2e41ea03ec819f66be489`.

## Exact hashes

| Namespace | EN source SHA-256 | RU target SHA-256 |
|---|---|---|
| relay.shell | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` | `b96e8923fe82824db2434720f722d2cf323d5aa733d78ad1c75285391c16245b` |
| relay.session | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec` |
| relay.login | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1` |

EN shell context SHA-256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`. All hashes derived from actual files and rechecked at report creation.

## Retention and native delta

Compared every source/target value with my prior 65-key audit: precisely three keys changed in both EN/RU; **62 other EN and RU values unchanged**. Shell retains41, session17, login4. Session and login whole-file bytes match prior target hashes. Context diff contains precisely the same three noun keys. Prior native/format/fallback evidence is retained only for those unchanged values.

All three paradigms are idiomatic RU: one nominative singular кредит/токен/запрос; few and fractional other genitive singular кредита/токена/запроса; many genitive plural кредитов/токенов/запросов. Negative magnitudes follow same plural selection; million takes genitive plural. Parenthetical label in unchanged amount templates remains understandable as compact amount-unit presentation; no duplicate count or added preposition. Used and remaining complete rows/family rows preserve source fields and punctuation. No changes requested.

## Executed evidence

Isolated `env -i HOME=... LOCAL_OPERATOR_CONFIG_DIR=... PATH=... TZ=UTC node ru-noun-delta.mjs`, no CMUX/live session variables, existing deps only. Actual compiled relay runtime retained from prior probe: git diff confirms runtime.ts unchanged.

- ICU AST each changed noun: count, cardinal plural, offset0, one/few/many/other. EN and RU preserve count type/style; branch text is noun only.
- Generated TypeScript lines57–59 observed `{ "count": number }`; caller passes SAME finite measured number to noun and amount, USD still no params. No compilation/full-suite claim.
- **48 composed case pairs** =3 nouns ×8 counts × remaining/used, each with both plain and named-family complete row outputs. Counts:0,1,2,1.5,-1,-2,-1.5,1000000. All asserted noun values PASS. Full actual outputs in JSON.
  - 0 → `Включённая квота · Месяц: Остаток: 0 (кредитов)`
  - 1 → `Включённая квота · Месяц: Остаток: 1 (кредит)`
  - 2 → `Включённая квота · Месяц: Остаток: 2 (кредита)`
  - 1.5 → `Включённая квота · Месяц: Остаток: 1,5 (кредита)`
  - -1 → `Включённая квота · Месяц: Остаток: -1 (кредит)`
  - -2 → `Включённая квота · Месяц: Остаток: -2 (кредита)`
  - -1.5 → `Включённая квота · Месяц: Остаток: -1,5 (кредита)`
  - 1000000 → `Включённая квота · Месяц: Остаток: 1 000 000 (кредитов)`

- Six negative controls PASS: removing count by reinstating old literal fails AST signature comparison `[] != [["count",6,"cardinal",0]]`; missing count in actual relay runtime degrades to the message key for each of three nouns.
- RU CLDR one/few/many/other exercised, including fractions, negative magnitudes and million. RU is LTR; native RTL N/A. Pseudo/rendered layout **HOLD/BLOCKED — customer resource hold**, no new rendered UX or interaction proof.

## Retained minor disposition

**RU-M1 remains MINOR, deferred and non-gating.** Translator adjacent `relay-minimal3noun/report.md:23` explicitly defers command-label refinement because it is outside the approved three-noun contract and not a correctness fix. Verified `relay.shell.usage.command` is identical to prior reviewed wording. No further audit round required for that minor.

## Boundaries

No source/target edits, self-translation, ledger writes, browser, dependency installation, heavy suites, shipment claim, or PR review substitution. Report JSON and script are in the same own scratch directory. This pass can pin these exact working source/target bytes for the unified commit; it does not clear rendered-resource hold.

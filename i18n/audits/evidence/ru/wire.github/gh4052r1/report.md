# PR #4052 independent translation audit — round 1

Locale: ru; namespace: wire.github. Auditor 91d1e05d2d82, actual openai/gpt-6.1-sol after Opus quota fallback. Source head 0b754ce89567ae158fe91db3ec0ed0b4a621f312.

Scope: all 68 new wire.github values, or only the one settings help value. The other 866 settings target values and English source values are unchanged from the named base; prior passed receipts are retained in summary.json. No new audit of those unchanged settings values is claimed.

Mechanical independent checks and repository validator pass; all target/source receipt hashes match current bytes. This audit executed 6,720 ICU renders across English and seven target locales (840 per locale), both select arms, all CLDR categories, 12 numeric probes; 68 protected-token pseudo expansion/wrapping controls passed. Static Rich 100-column CLI-copy SVGs were converted with rsvg-convert and all seven target PNGs actually viewed. Context gives no fixed width cap. These are raw copy/layout evidence, not screenshots of the live app. Urdu frame visibly exhibits the known unshaped/degraded path; native terminal bidi/shaping is NOT certified. Production messages.render_record remains English because all target locales are currently ineligible; direct ICU candidate formatting is not activation proof. Explicit balanced-isolate adapter probe is not proof that plain CLI invokes that adapter.

Safety: exact owner-only remedies, no access lent on refusal, last-good uncertainty, unknown versus earlier login, retrieval-time opt-out, and never-set versus cleared previously shared list all reviewed. No authority or remote deletion promise added. No translation service used.

Execution note: render attempt 1 used a negative fixture missing other that this parser accepts; its assertion failed. Corrected fixture has an unmatched closing structure and failed as expected in attempts 2/3. Attempt 3 also protects command/literal tokens during AST-based pseudo expansion. The retained audit.log includes the failed setup, not a fabricated green first run.

## Findings

### I-RU-PUSH — MAJOR

The all-locales glossary keeps git push in Latin. These three messages replace it with отправлять … изменения, unlike neighboring GitHub copy and the translator receipt claim.

- `wire.github.opt_out_step`
  - Current: исключить репозиторий: {command} — устройства, которым вы предоставили доступ, потеряют право отправлять в него изменения при следующем получении учётных данных, {ttl}
  - Suggested: исключить репозиторий: {command} — устройства, которым вы предоставили доступ, потеряют право выполнять push в него при следующем получении учётных данных, {ttl}
- `wire.github.exclude_added`
  - Current: {repo} исключён — устройства, которым вы предоставили доступ, потеряют право отправлять в него изменения при следующем получении учётных данных, {ttl}
  - Suggested: {repo} исключён — устройства, которым вы предоставили доступ, потеряют право выполнять push в него при следующем получении учётных данных, {ttl}
- `wire.github.include_added`
  - Current: {repo} возвращён — устройства, которым вы предоставили доступ, снова смогут отправлять в него изменения при следующем получении учётных данных, {ttl}
  - Suggested: {repo} возвращён — устройства, которым вы предоставили доступ, снова смогут выполнять push в него при следующем получении учётных данных, {ttl}

### I-RU-WRAPPER — MAJOR

Устройство-кредитор is a financial-creditor calque in a safety diagnostic; выполнение … недоступна is grammatically wrong. Same key as second-pass I2-RU-01, independently rated MINOR there; deduplicate fixes, not provenance.

- `wire.github.wrapper_denied`
  - Current: {repo} отсутствует в списке репозиториев, предоставленном устройством-кредитором (возможно, он исключён или выполнение push с того устройства недоступна). Проверьте {command} на устройстве-кредиторе. Только это устройство может изменить предоставляемый доступ.
  - Suggested: {repo} отсутствует в списке репозиториев, предоставленном устройством, которое делится доступом (возможно, репозиторий исключён или с этого устройства в него нельзя выполнить push). Проверьте {command} на устройстве, которое делится доступом. Только это устройство может изменить предоставляемый доступ.

### I-COMPOSE — MAJOR

Composed explicit reason yields отклоняются, пока … не осталось репозиториев; unavailable reason yields пока не удалось получить … . The completed-past clauses are not an idiomatic ongoing refusal condition, and the explicit reading can suggest until none remain. Introduce the reason as a reason; existing English/fragment keys do not need changes.

- `wire.github.share_empty`
  - Current: Репозитории GitHub: пока нет — общий доступ настроен, но все запросы заимствования отклоняются, пока {reason}; {remedy}
  - Suggested: Репозитории GitHub: пока нет — общий доступ настроен, но все запросы заимствования отклоняются по следующей причине: {reason}; {remedy}


## Eligibility disposition

Only failed es/ru/ur wire.github receipts are appended now. Eleven clean locale/namespace receipts are intentionally NOT appended yet: manager requested no eligibility restoration before the unified translation remediation. Existing translator-only rows remain current there. Generator --check currently fails on seven orphaned locale mobile bundles; none were deleted or changed. Second independent pass is separately preserved under gh4052s1, with its own actual identity, original artifacts and minor/major ratings.

# Russian provider-state author notes — translated, not audited

This is `audit_notes_not_execution_log`. Actual execution is in `audit.log`.
Author: i18n-translator, session `29d4fc18d1a7`; actual translating model
`openai/gpt-6.1-sol`. The host announced a quota-driven fallback from
`anthropic/claude-opus-5-5` before Russian authoring. No MT/API/service was used.
Independent audits A and B have not been performed by this author.

## Native wording and glossary proposals

Read `docs/I18N.md` translation pipeline, RFC §§2.2/6,
`i18n/style-guides/ru.md`, `i18n/glossary.md`, the exact English five-key
catalogue and sidecar; consulted existing `ru/tui.settings_view.json`
(`вход не выполнен`) for consistency.

- `logged_in` → **вход выполнен**: completed login status, impersonal and
  ungendered; not the action «войти». Stored-credential provenance remains
  distinguished from the environment-key label without inventing expiry or
  verified network health.
- `env_key` → **ключ из окружения**: «из» explicitly marks credential source.
  «окружение» is the conventional developer-register term here; it refers to
  environment variables, not an environment's name or a literal identifier.
  Actual variable names are outside this string. No API acronym or fabricated
  variable token is added merely to preserve the English abbreviation.
- `needs_login` → **требуется вход**: requirement/status, not an imperative or
  button; contrasts with completed «вход выполнен».
- `local_ready` → **сервер настроен**: configuration completed, not «сервер
  работает» or «сервер доступен», which would imply health/reachability.
- `local_unconfigured` → **настроить сервер**: infinitive action, as required
  by the Russian style guide; distinct from the completed status above.

Glossary proposals (not edits to the shared glossary): login completion →
«вход выполнен»; login required → «требуется вход»; environment-sourced key →
«ключ из окружения»; configured server → «сервер настроен»; configure server →
«настроить сервер». Existing сервер/ключ vocabulary and impersonal register are
retained. Lowercase initials intentionally follow the terse English status/action
labels and their explicit context, not a new capitalization policy.

## Mechanical proof and limitations

Official `scripts/i18n/check.py::check_catalogues()` is invoked with in-process
discovery restricted to `en`, `ru`, and `tui.provider_state`. Actual repository
catalogue loader, context validator, ICU parser, argument parity and plural
checker are unchanged. All five are zero-argument literals: no plural/select
arms are applicable and no CLDR branches are fabricated. Missing-key and
added-argument in-memory negative controls fail as intended; restored control
is clean. Runtime `render_message` and plain Rich text preserve each label.
NFC, key order, protected tokens, no format controls and namespace parity pass.

Observed Russian label widths in Rich cells: 13, 17, 14, 15, 16 (source order).
English controls: 9, 7, 11, 17, 16. The 17-cell environment-source wording is
longer than English but avoids an unclear abbreviation. These are diagnostics,
not a fixed universal cap. The synthetic `openrouter  <label>` rows at 31 columns
are plain Rich diagnostics only, not the real settings formatter or its padding.
Pseudo expansion wraps some controls and is deliberately not product copy.
Longer opaque provider IDs or narrower consumers may still truncate labels;
real styled/native integration, geometry, font and no-wrap proof remain outside
this author lane. No app was started and no native certification is claimed.

The first validator harness import failed because its dynamically loaded module
was absent from `sys.modules` (dataclass annotation lookup). Its real output is
retained separately in `failed-rig-1.log`; the fix registered that module and
reran the same scoped checks successfully. No product change was needed.

Current producers remain English until the root integration work. Translation
bytes and mechanical checks confer no shipping eligibility, whole-program
clearance, independent language approval or audit result.

# Russian author receipt — translated, not audited

All four strings were authored by i18n-translator session `e6cc991f4cd7`, using actual model `openai/gpt-6.1-sol`. The host reported a temporary provider-quota fallback from `anthropic/claude-opus-5-5` after the initial guide read and before source/style reading and translation. No machine translation API, SDK, service or website was used. Source author `a97` is not this translator; independent A/B auditors must read all four keys after the manager's private root commit. This receipt is not an audit or admission record.

## Inputs and native choices

Read `guide://i18n`, `docs/I18N.md` locale-contribution instructions, RFC §2.2, `i18n/style-guides/ru.md`, and the complete glossary. English catalogue and context were read with the approved public-copy reader. Existing Russian `cli.config.json` was read through that same reader for consistency: it uses «модель», «сессия» and «по умолчанию».

- `persist`: «сохраняет эту модель по умолчанию для новых сессий» makes saving the current choice for future new sessions explicit. It does not describe switching this session.
- `settings`: «тоже меняет модель по умолчанию» retains the source's linking “too”, while remaining a complete route when `/settings` is the sole usable route. No `/model` remedy is added to it.
- `saved`: «перейти в этой сессии на сохранённую модель по умолчанию» explicitly switches this session to the saved default, not the reverse operation. Infinitive instruction, impersonal professional register.
- `with_model`: human prefix «модель» is translated; opaque `{label}` and already-localized `{routes}` remain simple arguments, once each, in source order. Dash and spacing retained.
- Glossary proposals, not edits: save → «сохранить» (inflected «сохраняет», «сохранённую»); boot default → «модель по умолчанию», explained as applying to new sessions in the persistence clause. “Загрузочная модель” would misleadingly evoke boot firmware rather than a session's startup choice.

No missing inputs or unresolved source ambiguities. No key changes, manual bidi controls, plural/select branches or width-cap changes.

## Actual scoped mechanical validation

`actualvalidation.log` is actual execution output, not audit notes. `script.py.txt` is the exact executed script. The command, return code, hashes and module provenance are retained in `record.json` and the log. The primary checkout's `.venv/bin/python` imported the worktree through explicit `PYTHONPATH`; `env -i` removed inherited CMUX/LOP state and isolated HOME/config in session scratch. Git metadata commands used normal HOME.

Official `scripts.i18n.check.check_catalogues()` was invoked with only its namespace/locale enumeration narrowed to `tui.model_persistence` and en/ru. No validation rule was replaced. Output: `official_check_catalogues_scoped: []`; process exit code 0. Official catalogue/context loaders, context schema, ICU parser/argument/plural selectors and ICU renderer were exercised. Four-key source order, exact commands, zero plural/select arms, unchanged opaque IDs and routes, and absence of manual bidi controls were asserted. Frozen EN/context SHA-256 values matched before and after.

Plain Rich diagnostic wrapping exercised 80/100/120 columns with short and long synthetic opaque labels across ordinary, missing-model setup and absent/unknown-hosting route sets (18 cases). Rows ranged from 1 to 4; no diagnostic line exceeded its supplied width. A two-word command can split at a normal space (ordinary short-label case at width 80 ends the preceding row with `/model` and starts the last row with `saved`). This is **not** a new all-width guarantee: context explicitly retains existing command-fold regressions as the integration baseline. The final row was never a lone complete command in these diagnostic cases.

These are unstyled Rich diagnostics, not NoticeBlock budgets, pristine/non-empty transcript family proof, native TUI screenshots, linguistic self-audit or completed UI qualification. Root integration owns real styled/rendered proof. No global catalogue check, generation, tests, installs, bootstrap, browser or native UI was run while this namespace is partial. No ledger/history, eligibility, baseline, source, other locale or immutable evidence was changed. Files remain unstaged/uncommitted for manager-owned consolidation.

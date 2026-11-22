# Vietnamese provider-state independent audit B

Verdict: **passed**, five of five keys examined; zero BLOCKER, MAJOR, MINOR or NIT findings. This is lexical/namespace audit approval only, not UI/font/geometry, namespace activation, eligibility or release approval.

Auditor: i18n-auditor / provider-state-vi-audit-B; session `add016b11be1`. Actual served model is reported as `openai/gpt-6.1-sol` by the explicit model-switch notice (quota fallback from `anthropic/claude-opus-5-5`). There is no independent per-response serving telemetry available here. The configured primary is NOT claimed as the served model.

The translator's `author-5644/record.json` identifies session `5644d4838456`, model `openai/gpt-6.1-sol`, and explicitly says `independent_from_translator_session: false`. Its auditor-shaped field is a translation-author receipt, not independent approval. I am a different session and did not translate or edit the catalogue. I did not consult audit A or another auditor's conclusions. No translation service, MT or back-translation was used; judgments below are my own Vietnamese-language reasoning against the style guide, glossary and English context.

## Every-key native-language assessment (in source order)

| Key | Current Vietnamese | Judgment |
| --- | --- | --- |
| `tui.provider_state.logged_in` | đã đăng nhập | PASS. Conventional concise completed login status. `đã` makes success explicit; not a login instruction. The credential-storage detail is contextual, not omitted user-facing content. |
| `tui.provider_state.env_key` | khóa từ biến môi trường | PASS. Natural developer-register noun phrase meaning a key sourced from an environment variable. `từ` correctly identifies provenance, not an environment/server name. `khóa` follows the guide's tone-mark spelling; this is human copy, not an identifier to preserve as `env_key`. |
| `tui.provider_state.needs_login` | cần đăng nhập | PASS. Natural requirement status. `cần` distinguishes an unmet authentication requirement from the successful state above; does not impersonate a command or button. |
| `tui.provider_state.local_ready` | máy chủ đã cấu hình | PASS. Terse server-configuration status. `đã cấu hình` expresses completion, not reachability or health. Omission of a passive auxiliary is normal compact Vietnamese UI wording here; adding `được` is not necessary. `máy chủ` follows the preferred server term. |
| `tui.provider_state.local_unconfigured` | cấu hình máy chủ | PASS. Natural bare-verb configuration instruction. Contrasts cleanly with the completed-state wording above; no login/error/health claim. |

All five use sentence/lower-case labels, full Vietnamese diacritics, no gratuitous pronouns, politeness or punctuation. Product/command/flag/address/identifier tokens are absent in these five values, so there are no protected spans requiring translation exceptions. Local-server context does not require adding `cục bộ` when English also omits “local”. No destructive or permission wording occurs. RTL assessment is not applicable to Vietnamese.

## Mechanical execution and limits

`script.py.txt` was executed with the primary checkout's `.venv/bin/python`, `PYTHONPATH` pointing to this worktree, and `env -i` isolating HOME/config and removing all inherited CMUX variables. The actual command output is `audit.log`, actual return code **0**. Module paths prove this worktree was imported. It runs the repository's `check_catalogues()` with enumeration restricted to exactly EN/VI and `tui.provider_state`; it does not run the global checker or replace any validation rule.

The official scoped catalogue/context/ICU checker returned `[]`. Missing-key and added-argument in-memory counterfactuals produced their expected failures, then the unchanged control passed. All three input SHA-256 values match the brief. All five ordered keys match context and target; every message has zero arguments and zero plural/select branches, making CLDR branch coverage vacuous. JSON shape is validated by the production loader; NFC, absent bidi controls, protected-token parity, literal Rich-markup behavior and exact ICU rendering were independently asserted. The synthetic pseudo control wraps and expands English literal messages and proves they still render; it is NOT a shipped pseudo-locale or layout test.

Fresh plain Rich widths: EN **9/7/11/17/16**, VI **12/23/13/19/16**. No context key declares `maxCells` or `maxChars`; the illustrative 17-cell value is not a global cap. The longer environment-source label is precise and natural. Actual styled geometry must be checked by the integration lane; no lexical rewrite or overflow severity is inferred from plain-width expansion alone. Native-font/diacritic rendering is unverified.

`report.md` is **audit_notes_not_execution_log**. Only `audit.log` and the optional ledger dry-run log represent execution. The record and direct ledger receipt preserve independent provenance; no audit history is applied, staged, committed or pushed by this auditor. Source, context, target and other evidence bundles are untouched.

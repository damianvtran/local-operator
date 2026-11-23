# Independent Vietnamese B audit — tui.model_persistence

Verdict: **passed**. Four keys audited in source order; zero BLOCKER, MAJOR, MINOR or NIT findings. This is a current-byte independent native-LLM assessment, not a translation-service comparison, global UI approval, admission/generation approval or styled/native geometry proof. No A conclusions were consulted.

## Native reasoning, key by key

1. `persist`: `/model default lưu lựa chọn này cho các phiên mới` accurately says that the command saves this choice for new sessions. `lưu lựa chọn này` is idiomatic and does not imply switching the current session. `phiên` follows the glossary. The command stays exact.
2. `settings`: `/settings cũng sửa mô hình mặc định khi mở phiên` is concise command-as-subject Vietnamese. `sửa` is an ordinary edit verb in terse developer UI copy; `cũng` preserves “too”. `mô hình mặc định khi mở phiên` identifies the boot default, not the current-session model. As the sole hosting-deadend route it remains a complete usable clause; it adds no unavailable /model command. The linking word is also in the English source and is not a translation error.
3. `saved`: `chuyển phiên này sang mô hình mặc định đã lưu bằng /model saved` explicitly distinguishes the current-session switch from saving for future sessions. `đã lưu` preserves the saved-default meaning. The idiomatic `chuyển ... sang ... bằng` construction is grammatical; no need to manufacture a finding for a stylistic alternative.
4. `with_model`: `mô hình: {label} — {routes}` translates the human prefix according to the glossary while preserving the opaque model identifier and complete route clauses exactly once. Colon and spaced em dash are appropriate; no redundant imperative is prepended to routes.

## Actual mechanical and wrapping evidence

`audit.log` is output from the isolated primary-venv command executing `script.py.txt` with the worktree on PYTHONPATH. Module origins are printed. Official `check_catalogues()` ran with only locale/namespace discovery scoped to en/vi and this namespace; catalogue loading, context validation, ICU parsing and parity checks are the repository implementations. Both missing-key and renamed-argument negative fixtures failed as intended. Both omitted-argument ICU renders raised MessageFormatError; the real message API passed an unknown ID through (not an error). All four keys have valid JSON, identical ordered keys and arguments; zero plural/select constructs were verified (no invented CLDR requirement). Commands and argument multiplicity are exact; target is NFC. Protected-token pseudo expansion passed.

36 plain-Rich cases cover en/vi, short/long opaque IDs, ordinary/missing-model/hosting-deadend route compositions and 80/100/120 widths, with actual text and cell measurements in the log. These synthetic state compositions do not execute the actual consumer. No fixed cap exists in this sidecar, and none is inferred from the picker footer. This does NOT demonstrate styled NoticeBlock/native geometry, actual pristine/non-empty transcript families, font rendering, or the still-pending generator/consumer integration. Before/after hashes of all five inputs match. No target/source edits, ledger/history writes, staging or commits performed.

## Provenance

Auditor i18n-auditor B, session 674d6e6c175e, actual openai/gpt-6.1-sol after host-reported anthropic/claude-opus-5-5 quota fallback. Independent of translator edc50a6f3d77, A auditor fcb8a76716b3 and source author a97. Translator actual GPT fallback was verified directly from viau1/record.json, not inferred. This report is audit notes, not execution output. The execution log proves only the script run described above. The initial read-only discovery command guessed three nonexistent module filenames and returned exit 2; corrected by listing actual modules before validation. That discovery failure is not a failed validation run.

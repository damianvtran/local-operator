# Vietnamese author notes — translated, not audited

Author: i18n-translator, session `edc50a6f3d77`, actual translation model `openai/gpt-6.1-sol`. The host switched from `anthropic/claude-opus-5-5` because of provider quota failure before source/style/glossary reads and translation. The primary model only read the i18n guide. No MT API, SDK, service or website was used.

## Native choices and glossary proposals

- `model` → `mô hình`; `session` → `phiên`, per shared glossary and neighboring `vi/tui.small.json` (approved public-copy reader).
- `lưu lựa chọn này cho các phiên mới` means saving this choice for future new sessions. It does not claim to change this session.
- `chuyển phiên này sang mô hình mặc định đã lưu` explicitly changes this session to the saved default, distinct from saving the current model.
- `mô hình mặc định khi mở phiên` is the model default at session opening, not the current session's choice. `cũng sửa` retains the connecting “too” while remaining a complete usable `/settings` clause by itself in the dead end.
- Glossary proposals (not edits): boot default → `mô hình mặc định khi mở phiên`; saved boot default → `mô hình mặc định đã lưu` when the current-session switch is explicit. These compounds have no dedicated shared-glossary row.
- Lowercase openings retain the English notice's deliberate terse styling and neighboring Vietnamese inline hints. Vietnamese is NFC; no manual bidi controls. All human English is translated; commands and opaque arguments alone remain literal.

## Actual mechanical execution

Working directory: `/Users/damian/local-operator/.worktrees/i18n-model-persist-notice`, starting HEAD `87d87ec8b02ef3c279fda673302ce775f4a0329e`.

Executed command (shell expanded `$LOCAL_OPERATOR_SCRATCHPAD` to this author session's scratchpad):

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/vi-author-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/vi-author-home/.local-operator" \
 PATH="$PATH" TERM=xterm-256color \
 PYTHONPATH=/Users/damian/local-operator/.worktrees/i18n-model-persist-notice \
 /Users/damian/local-operator/.venv/bin/python \
 i18n/audits/evidence/vi/tui.model_persistence/viau1/validator.py.txt \
 > i18n/audits/evidence/vi/tui.model_persistence/viau1/audit.log 2>&1
```

Exit 0. Official `scripts.i18n.check.check_catalogues()` executed with only namespace/locale enumeration narrowed to this namespace and en/vi; validation implementation was unmodified. Source/target catalogue loading, official context validation, ICU argument/kind and plural/select checks, and public render API calls used worktree modules, whose paths are logged. Keys/order 4/4; zero plural/select branches; commands exact once; label/routes simple and exact once. Short, ordinary and dated opaque identities survive byte-for-byte. Both missing-argument negative controls raised `MessageFormatError`; opaque string `0` survives. No-label clause-only composition is a probe, not production consumer evidence.

Plain Rich wrapping covers 80/100/120 with English controls first. No fixed maxCells exists here; the unrelated footer cap was not applied. Log rows/widths are plain Rich results only, not styled NoticeBlock/native terminal rendering, pristine/non-empty transcript geometry or visual sign-off. No new all-width guarantee or locale-specific overflow defect is claimed. Consumer integration and independent A/B audit remain the root owner's work.

`audit.log` is actual stdout/stderr. This file is `audit_notes_not_execution_log`. Record status is `translated` only; this is not a self-audit, passed ledger row, eligibility decision, whole-interface or release claim. No shared ledger/history, generation, commit, staging or push was performed by this author.

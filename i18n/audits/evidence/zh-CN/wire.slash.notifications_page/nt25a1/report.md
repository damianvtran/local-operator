# zh-CN / wire.slash.notifications_page — independent audit

Auditor: audit-zh-cn-6a81bd6cab5c; session 6a81bd6cab5c; actual model openai/gpt-6.1-sol (provider quota fallback disclosed in host message).
Independent audit-zh-CN agent/context; authored none of the UI tr-zh-CN or L1-core-notify coder translations and made no catalogue/code/ledger edits.

Verdict: clean-with-findings; 1 English and 1 target keys read with source sidecar. Zero blocker/major.

Source SHA-256 pin verified; hashes are in record.json. Own core ICU parse_message/message_arguments/render_message executed, including malformed-input and missing-placeholder negative controls (2/2). No plurals/select/skeleton/markup in these sources. Source placeholders survive exactly. Chinese is LTR; punctuation uses Chinese full-width forms; technical path and read remain Latin/verbatim.
Literal read survives; actual help length 34 cells <= maxCells 50, proposed 请求 wording also 34 cells. 问题 is intelligible but drifts from the existing UI queue name 请求.

## Findings

- T-CORE-1 MINOR `wire.slash.notifications_page.help`
  - Current: 问题和智能体更新；read全部标为已读
  - Proposed: 请求和智能体更新；read全部标为已读
  - Reason: Asks is the named product queue: UI rail, drawer, filter and detail all use 请求, whereas this command help switches it to 问题. Reuse 请求 for cross-surface glossary consistency; keep literal read. The proposed text is also 34 cells, within maxCells 50.

## Actual command output

```text
COMMAND isolated .venv/bin/python $LOCAL_OPERATOR_SCRATCHPAD/audit-core.py wire.slash.notifications_page
INPUT_SHA256 {"en": {"path": "local_operator/i18n/catalogues/en/wire.slash.notifications_page.json", "sha256": "9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594"}, "en_context_sidecar": {"path": "local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json", "sha256": "2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0"}, "target": {"path": "local_operator/i18n/catalogues/zh-CN/wire.slash.notifications_page.json", "sha256": "31c919f35913e6c0f333c739acd3a602912bf54f6ef8fd82ed0176bcec988b67"}}
KEY wire.slash.notifications_page.help EN "Asks and agent updates; read marks all read" ZH "问题和智能体更新；read全部标为已读" RENDER "问题和智能体更新；read全部标为已读" CELLS 34
PASS maxCells 34 <= 50
PASS 1 keys: EN/target/context parity, own-core ICU parser, argument name/type parity, target rendering; negative controls 2/2
```

Execution note: first setup attempt imported unavailable wcwidth and stopped before checks. No dependency was installed. The rerun used Python unicodedata East Asian width (W/F = 2, combining = 0); these strings contain no ambiguous-width/emoji/combining edge cases. The attached audit.log is the successful actual run, not a reconstruction.

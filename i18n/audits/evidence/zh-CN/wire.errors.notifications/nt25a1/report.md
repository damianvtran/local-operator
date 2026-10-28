# zh-CN / wire.errors.notifications — independent audit

Auditor: audit-zh-cn-6a81bd6cab5c; session 6a81bd6cab5c; actual model openai/gpt-6.1-sol (provider quota fallback disclosed in host message).
Independent audit-zh-CN agent/context; authored none of the UI tr-zh-CN or L1-core-notify coder translations and made no catalogue/code/ledger edits.

Verdict: clean-with-findings; 4 English and 4 target keys read with source sidecar. Zero blocker/major.

Source SHA-256 pin verified; hashes are in record.json. Own core ICU parse_message/message_arguments/render_message executed, including malformed-input and missing-placeholder negative controls (2/2). No plurals/select/skeleton/markup in these sources. Source placeholders survive exactly. Chinese is LTR; punctuation uses Chinese full-width forms; technical path and read remain Latin/verbatim.
store_busy preserves no-write and retry; store_full preserves the volume containing {where}; store_unavailable preserves that retries will not help and the disk/path check. unknown_item says 通知, never 收件箱/Inbox. All four messages are semantically accurate; only the three extra 请 markers conflict with the binding terse-register rule. No fixed maxCells budget is attached to these error sentences.

## Findings

- T-CORE-1 MINOR `wire.errors.notifications.store_busy`
  - Current: 通知存储目前正忙，未做任何更改。请稍后重试。
  - Proposed: 通知存储目前正忙，未做任何更改。稍后重试。
  - Reason: The zh-CN style guide permits 请 only when source English says please; none of these English error sentences does. Remove the extra politeness marker without changing the remedy or {where}.

- T-CORE-2 MINOR `wire.errors.notifications.store_full`
  - Current: 这台电脑的磁盘空间已满，未做任何更改。请在存放{where}的卷上释放空间，然后重试。
  - Proposed: 这台电脑的磁盘空间已满，未做任何更改。在存放{where}的卷上释放空间，然后重试。
  - Reason: The zh-CN style guide permits 请 only when source English says please; none of these English error sentences does. Remove the extra politeness marker without changing the remedy or {where}.

- T-CORE-3 MINOR `wire.errors.notifications.store_unavailable`
  - Current: 无法读取或写入通知存储。重试无济于事；请检查{where}及其所在的磁盘。
  - Proposed: 无法读取或写入通知存储。重试无济于事；检查{where}及其所在的磁盘。
  - Reason: The zh-CN style guide permits 请 only when source English says please; none of these English error sentences does. Remove the extra politeness marker without changing the remedy or {where}.

## Actual command output

```text
COMMAND isolated .venv/bin/python $LOCAL_OPERATOR_SCRATCHPAD/audit-core.py wire.errors.notifications
INPUT_SHA256 {"en": {"path": "local_operator/i18n/catalogues/en/wire.errors.notifications.json", "sha256": "f02bf1bc28f9c9e549a3ad8c34ead880315f4b39d25c36a869a1b3c5110f362c"}, "en_context_sidecar": {"path": "local_operator/i18n/catalogues/en/wire.errors.notifications.context.json", "sha256": "07ad1e5da403a5fccde5648f04cd10bf1263cabc50f5e1804644630c7ce13106"}, "target": {"path": "local_operator/i18n/catalogues/zh-CN/wire.errors.notifications.json", "sha256": "f7334d805d7c10e17d82fe38335dfb449420b7ba8c14ce572ab8ee338a100888"}}
KEY wire.errors.notifications.store_busy EN "The notifications store is busy right now, so nothing changed. Try again in a moment." ZH "通知存储目前正忙，未做任何更改。请稍后重试。" RENDER "通知存储目前正忙，未做任何更改。请稍后重试。" CELLS 44
KEY wire.errors.notifications.store_full EN "This computer is out of disk space, so nothing changed. Free some space on the volume holding {where}, then try again." ZH "这台电脑的磁盘空间已满，未做任何更改。请在存放{where}的卷上释放空间，然后重试。" RENDER "这台电脑的磁盘空间已满，未做任何更改。请在存放/synthetic/config的卷上释放空间，然后重试。" CELLS 89
KEY wire.errors.notifications.store_unavailable EN "The notifications store could not be read or written. Retrying will not help; check {where} and the disk it is on." ZH "无法读取或写入通知存储。重试无济于事；请检查{where}及其所在的磁盘。" RENDER "无法读取或写入通知存储。重试无济于事；请检查/synthetic/config及其所在的磁盘。" CELLS 77
KEY wire.errors.notifications.unknown_item EN "That notification is no longer in Notifications." ZH "该通知已不在“通知”中。" RENDER "该通知已不在“通知”中。" CELLS 22
PASS Notifications page name; no Inbox
PASS 4 keys: EN/target/context parity, own-core ICU parser, argument name/type parity, target rendering; negative controls 2/2
```

Execution note: first setup attempt imported unavailable wcwidth and stopped before checks. No dependency was installed. The rerun used Python unicodedata East Asian width (W/F = 2, combining = 0); these strings contain no ambiguous-width/emoji/combining edge cases. The attached audit.log is the successful actual run, not a reconstruction.

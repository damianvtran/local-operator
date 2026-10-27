# Independent Russian audit — wire.slash.notifications_page

Auditor: translation-auditor / ffd16b7ff158 / openai/gpt-6.1-sol. Date: 2026-10-25.
Actual fallback provenance: Latest session model-switch message: openai/gpt-6.1-sol, temporary quota fallback after anthropic/claude-sonnet-5-5; original requested model anthropic/claude-opus-5-5.
Independence: Independent auditor; authored none of the target translations or implementation. No catalogue edits made.

## Scope and verdict

All 1 English and 1 Russian values read side by side, with all context entries. Verdict: **changes_requested**. No unresolved blockers; 1 unresolved major findings. Repository HEAD at capture: 329b5c581784df1082dbd0630c0f814c7a3c0c8f; input hashes, not HEAD alone, bind the audit. English frozen hashes verified against delegation.

## Findings

### R5 — major — `wire.slash.notifications_page.help`

Current: Вопросы и новости агентов; read — всё прочитано

Proposed: Запросы, обновления; read помечает всё прочитанным

Reason: News (новости) changes the feature meaning and drifts from UI Обновления; Asks also drift from UI Запросы. The command description must describe marking all read, not merely state everything is read. Proposed text preserves literal read and measures exactly 50 cells.

## Clean dimensions

Key parity, ICU parse and unique typed argument names match. Context sidecars cover every key. No bidi/zero-width controls. Russian formal lowercase вы, sentence case, action infinitives and «ёлочки» are maintained. Notifications is Уведомления, never Inbox.

Literal command token read is intact. Existing target is 47 cells; maxCells is 50. Proposed repair is 50 cells (Cyrillic/ASCII characters are single-cell here). No plural messages or runtime arguments. Semantics fail despite mechanical checks passing.

## Input hashes

{
  "en": {
    "path": "local_operator/i18n/catalogues/en/wire.slash.notifications_page.json",
    "sha256": "9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594"
  },
  "en_context_sidecar": {
    "path": "local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json",
    "sha256": "2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0"
  },
  "target": {
    "path": "local_operator/i18n/catalogues/ru/wire.slash.notifications_page.json",
    "sha256": "d1f0996ee378123fce3ffe0a42ba30818020c2f8e8bbd30db48486b9bfa64076"
  }
}

## Actual mechanical output

Command: `node /Users/damian/.local-operator/sessions/ffd16b7ff158/scratchpad/audit-ru.mjs wire.slash.notifications_page core`

```text
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/en/wire.slash.notifications_page.json 9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/ru/wire.slash.notifications_page.json d1f0996ee378123fce3ffe0a42ba30818020c2f8e8bbd30db48486b9bfa64076
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json 2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0
MAXCELLS wire.slash.notifications_page.help 47 budget=50
RESULT {"namespace":"wire.slash.notifications_page","enKeys":1,"targetKeys":1,"parsedMessages":2,"ruPluralNodes":0,"mechanicalFailures":0,"contextKeys":1}
```

## Limits

- LLM textual audit, not a native-speaker attestation.
- No browser/rendered-fit validation; design/QA own rendered evidence.
- Mechanical parser check uses the UI repository installed @formatjs/icu-messageformat-parser and intl-messageformat, without dependency installation. Unique typed-argument signatures are compared; plural category additions are intentional; exact =0/=1 and select false/other inspected in source and samples.

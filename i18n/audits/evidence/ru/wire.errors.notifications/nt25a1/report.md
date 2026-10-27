# Independent Russian audit — wire.errors.notifications

Auditor: translation-auditor / ffd16b7ff158 / openai/gpt-6.1-sol. Date: 2026-10-25.
Actual fallback provenance: Latest session model-switch message: openai/gpt-6.1-sol, temporary quota fallback after anthropic/claude-sonnet-5-5; original requested model anthropic/claude-opus-5-5.
Independence: Independent auditor; authored none of the target translations or implementation. No catalogue edits made.

## Scope and verdict

All 4 English and 4 Russian values read side by side, with all context entries. Verdict: **clean-with-findings**. No unresolved blockers; 0 unresolved major findings. Repository HEAD at capture: 329b5c581784df1082dbd0630c0f814c7a3c0c8f; input hashes, not HEAD alone, bind the audit. English frozen hashes verified against delegation.

## Findings

### R4 — minor — `wire.errors.notifications.store_busy`

Current: Хранилище уведомлений сейчас занято, поэтому ничего не изменилось. Повторите попытку через минуту.

Proposed: Хранилище уведомлений сейчас занято, поэтому ничего не изменилось. Повторите попытку чуть позже.

Reason: The source says in a moment, not a fixed one-minute wait; avoid inventing retry timing.

## Clean dimensions

Key parity, ICU parse and unique typed argument names match. Context sidecars cover every key. No bidi/zero-width controls. Russian formal lowercase вы, sentence case, action infinitives and «ёлочки» are maintained. Notifications is Уведомления, never Inbox.

All four error conditions and their consequences survive: busy/full no change, free space on the appropriate volume, unrecoverable store retry warning, and item absent from the Notifications destination. Both {where} arguments are preserved. Unknown-item uses the exact UI destination Уведомления. No plural messages in this namespace.

## Input hashes

{
  "en": {
    "path": "local_operator/i18n/catalogues/en/wire.errors.notifications.json",
    "sha256": "f02bf1bc28f9c9e549a3ad8c34ead880315f4b39d25c36a869a1b3c5110f362c"
  },
  "en_context_sidecar": {
    "path": "local_operator/i18n/catalogues/en/wire.errors.notifications.context.json",
    "sha256": "07ad1e5da403a5fccde5648f04cd10bf1263cabc50f5e1804644630c7ce13106"
  },
  "target": {
    "path": "local_operator/i18n/catalogues/ru/wire.errors.notifications.json",
    "sha256": "ccdf03166862d78b25c61b3c3651e92b0e24de2e85ed30f998db937b42739bdd"
  }
}

## Actual mechanical output

Command: `node /Users/damian/.local-operator/sessions/ffd16b7ff158/scratchpad/audit-ru.mjs wire.errors.notifications core`

```text
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/en/wire.errors.notifications.json f02bf1bc28f9c9e549a3ad8c34ead880315f4b39d25c36a869a1b3c5110f362c
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/ru/wire.errors.notifications.json ccdf03166862d78b25c61b3c3651e92b0e24de2e85ed30f998db937b42739bdd
INPUT /Users/damian/workspace/repos/lo-notifications-l1/local_operator/i18n/catalogues/en/wire.errors.notifications.context.json 07ad1e5da403a5fccde5648f04cd10bf1263cabc50f5e1804644630c7ce13106
RESULT {"namespace":"wire.errors.notifications","enKeys":4,"targetKeys":4,"parsedMessages":8,"ruPluralNodes":0,"mechanicalFailures":0,"contextKeys":4}
```

## Limits

- LLM textual audit, not a native-speaker attestation.
- No browser/rendered-fit validation; design/QA own rendered evidence.
- Mechanical parser check uses the UI repository installed @formatjs/icu-messageformat-parser and intl-messageformat, without dependency installation. Unique typed-argument signatures are compared; plural category additions are intentional; exact =0/=1 and select false/other inspected in source and samples.

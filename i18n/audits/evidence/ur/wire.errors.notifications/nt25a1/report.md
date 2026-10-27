# Independent Urdu audit — wire.errors.notifications — nt25a1

Auditor: translation-auditor; session/instance `89dbc1a173e9`; actual model `openai/gpt-6.1-sol`. The actual-model session event is the provenance source; quota fallback from the originally requested model is disclosed. Independence is agent/context, and none of these translations were authored or changed by this auditor.
Translator: L1-core-notify coder, anthropic/claude-opus-5-5, 2026-10-25, according to delegating brief.

## Verdict
**clean**; 4 English keys and 4 Urdu keys read side by side with every English context sidecar entry. 0 major, 0 blocker, 0 minor, 0 nit.

## Input hashes
- en: `local_operator/i18n/catalogues/en/wire.errors.notifications.json` — `f02bf1bc28f9c9e549a3ad8c34ead880315f4b39d25c36a869a1b3c5110f362c`
- en_context_sidecar: `local_operator/i18n/catalogues/en/wire.errors.notifications.context.json` — `07ad1e5da403a5fccde5648f04cd10bf1263cabc50f5e1804644630c7ce13106`
- target: `local_operator/i18n/catalogues/ur/wire.errors.notifications.json` — `ed6f12c19808f626546e3650dd57c0561aee232062203f42fe7944f2fc2dade9`

## Findings
None. Clean namespace, 4 keys read. Store contention remains retryable with nothing changed; disk-full says free space before retry; unavailable correctly says retry will not help and names the disk. Unknown-item names “اطلاعات”, matching UI Notifications, not Team Bridge Inbox.

## Every reviewed pair
- `wire.errors.notifications.store_busy`
  - EN: The notifications store is busy right now, so nothing changed. Try again in a moment.
  - UR: اطلاعات کا اسٹور اس وقت مصروف ہے، اس لیے کچھ نہیں بدلا۔ تھوڑی دیر بعد دوبارہ کوشش کریں۔
- `wire.errors.notifications.store_full`
  - EN: This computer is out of disk space, so nothing changed. Free some space on the volume holding {where}, then try again.
  - UR: اس کمپیوٹر کی ڈسک میں جگہ ختم ہو گئی ہے، اس لیے کچھ نہیں بدلا۔ {where} والے والیوم پر کچھ جگہ خالی کریں، پھر دوبارہ کوشش کریں۔
- `wire.errors.notifications.store_unavailable`
  - EN: The notifications store could not be read or written. Retrying will not help; check {where} and the disk it is on.
  - UR: اطلاعات کا اسٹور پڑھا یا لکھا نہیں جا سکا۔ دوبارہ کوشش سے فائدہ نہیں ہوگا؛ {where} اور اس کی ڈسک کو چیک کریں۔
- `wire.errors.notifications.unknown_item`
  - EN: That notification is no longer in Notifications.
  - UR: یہ اطلاع اب “اطلاعات” میں نہیں ہے۔

## Checks and limitations
The real repository runtime `parse_message`, `message_arguments`, `plural_selectors` and `render_message` executed under an isolated HOME/config and clean environment with synthetic `/synthetic/config` data. No live session/config touched. Actual stdout is in `audit.log`, byte-identical to the captured scratch command log. Reproduction script retained as `recheck.py.txt` (non-executable evidence).

All keys parse, argument names/types match, no plural nodes are used, and manual bidi control scan reports zero. Respectful Urdu instructions and Urdu punctuation remain. `read` stays a Latin command token; `{where}` stays an unchanged placeholder and produces the synthetic Latin path verbatim. String logical order checked; no browser, Nastaliq pixel fit, mirroring, font shaping or native terminal rendering claimed. Rich cell lengths are terminal character-cell measures, not rendered web widths.

Setup disclosure: initial script import of wcwidth failed (ModuleNotFoundError); no audit ran under that setup. It was replaced with the already installed Rich cell_len, and both namespace runs completed successfully. No installs or source changes.

The bundle follows fr/relay.approvals/ap25a1 record/receipt schema, with actual original audit.log present rather than that historic example's absent-log replay. `receipt.json` hashes all companions including record.json; it does not recursively hash itself. No catalogue, generated, ledger, code or git modifications were made.

# Independent Urdu audit — wire.slash.notifications_page — nt25a1

Auditor: translation-auditor; session/instance `89dbc1a173e9`; actual model `openai/gpt-6.1-sol`. The actual-model session event is the provenance source; quota fallback from the originally requested model is disclosed. Independence is agent/context, and none of these translations were authored or changed by this auditor.
Translator: L1-core-notify coder, anthropic/claude-opus-5-5, 2026-10-25, according to delegating brief.

## Verdict
**changes_requested**; 1 English keys and 1 Urdu keys read side by side with every English context sidecar entry. 1 major, 0 blocker, 0 minor, 0 nit.

## Input hashes
- en: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.json` — `9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594`
- en_context_sidecar: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json` — `2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0`
- target: `local_operator/i18n/catalogues/ur/wire.slash.notifications_page.json` — `d35277e21a1f9d26cf99b5007acd383d8ba30fd0f2c3c8068c7afc8421de76af`

## Findings
### T-UR-CORE-1 — MAJOR — `wire.slash.notifications_page.help`
Current: سوالات اور ایجنٹ اپ ڈیٹس؛ read سب پڑھا ہوا

Proposed: سوالات و ایجنٹ اپ ڈیٹس؛ read سب کو پڑھا قرار دے

Reason: The clause after literal read is an adjective fragment (all read), dropping the English action marks all read; the help no longer states what the subcommand does. The proposed clause restores the state-changing action and measures 47/50 terminal cells with Rich.

Literal `read` is preserved in both. Current help passes its width budget (42/50), so this is semantic incompleteness, not an overflow finding. Repair is proposed only; original coder must modify and a fresh audit must bind the resulting target hash.

## Every reviewed pair
- `wire.slash.notifications_page.help`
  - EN: Asks and agent updates; read marks all read
  - UR: سوالات اور ایجنٹ اپ ڈیٹس؛ read سب پڑھا ہوا

## Checks and limitations
The real repository runtime `parse_message`, `message_arguments`, `plural_selectors` and `render_message` executed under an isolated HOME/config and clean environment with synthetic `/synthetic/config` data. No live session/config touched. Actual stdout is in `audit.log`, byte-identical to the captured scratch command log. Reproduction script retained as `recheck.py.txt` (non-executable evidence).

All keys parse, argument names/types match, no plural nodes are used, and manual bidi control scan reports zero. Respectful Urdu instructions and Urdu punctuation remain. `read` stays a Latin command token; `{where}` stays an unchanged placeholder and produces the synthetic Latin path verbatim. String logical order checked; no browser, Nastaliq pixel fit, mirroring, font shaping or native terminal rendering claimed. Rich cell lengths are terminal character-cell measures, not rendered web widths.

Setup disclosure: initial script import of wcwidth failed (ModuleNotFoundError); no audit ran under that setup. It was replaced with the already installed Rich cell_len, and both namespace runs completed successfully. No installs or source changes.

The bundle follows fr/relay.approvals/ap25a1 record/receipt schema, with actual original audit.log present rather than that historic example's absent-log replay. `receipt.json` hashes all companions including record.json; it does not recursively hash itself. No catalogue, generated, ledger, code or git modifications were made.

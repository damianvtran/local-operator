# Urdu independent second semantic audit — u31h8b

**Verdict: passed.** Three of three keys reviewed; no BLOCKER, MAJOR, MINOR or NIT findings. This is an independent second pass, not a remediation audit. No audit-a material was inspected. No translation/MT service was used and no catalogue or shared ledger was edited.

Auditor: i18n-auditor / host-ur-audit-b, session `8a79664ef0c5`, actual model `openai/gpt-6.1-sol`, 2026-10-31 UTC. Host announced quota fallbacks from `anthropic/claude-sonnet-5-5` through `anthropic/claude-opus-5-5` to `openai/gpt-6.1-sol` before the substantive audit. Translator session `50b1205a23e1` is distinct; its sibling provenance states actual translator model `openai/gpt-6.1-sol` after quota fallback. Model equality is disclosed; these are separate sessions, not self-audit.

Source revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`. English SHA-256 `b7d6561be15577c118ec333586f7c809e65471c5dfd0dd839abee700b2ed1ba0`; context SHA-256 `d26acd5a6357a0c54bbd5ed409b4ad2c060ee52b9bd76670679397c4b464f724`; Urdu SHA-256 `311ea5428af42db26ff461969d40d5e0198907a5c21d24da643c11cf64c35622`.

## Per-key independent reasoning (audit notes, not execution log)

### wire.errors.session_host.unsupported — PASS
Current: یہ کمپیوٹر پس منظر میں سیشن ہوسٹ نہیں چلا سکتا، اس لیے سیشن شروع نہیں ہوا۔ ایسا کمپیوٹر استعمال کریں جس پر پس منظر کی سروسز کے لیے معاونت یافتہ سروس مینیجر موجود ہو۔

Preserves the platform's inability to run the background host, causal non-start of the session and the different-computer remedy. There is no suggestion of reconnecting or a transient retry. `معاونت یافتہ سروس مینیجر` is an intelligible technical rendering of supported service manager, with background services explicit. The wording is formal but acceptable educated software Urdu, not a register/glossary violation. Respectful `استعمال کریں`, settled `سیشن` and `سروس` comply with the guide. No correction needed.

### wire.errors.session_host.unavailable — PASS
Current: پس منظر میں چلنے والے سیشن ہوسٹ نے جواب نہیں دیا، اس لیے سیشن شروع نہیں ہوا۔ دوبارہ کوشش کرنے سے پہلے کچھ دیر انتظار کریں۔

Explicitly states that the host did not answer and the session did not start. The instruction is to wait before a user-initiated retry, not a false claim of an automatic reconnect. `کچھ دیر انتظار کریں` naturally expresses waiting a moment without inventing an exact duration. Respectful imperative and consistent host/session terminology. No correction needed.

### wire.errors.session_host.refused — PASS
Current: پس منظر میں چلنے والے سیشن ہوسٹ نے یہ سیشن شروع کرنے سے انکار کر دیا۔ سیشن شروع نہیں ہوا۔ دوبارہ کوشش کرنے سے پہلے اپنے Local Operator کنفیگریشن فولڈر میں logs/session-host.log دیکھیں۔

Correctly attributes refusal to the host rather than permission refusal by the user; explicitly repeats that the session did not start. Inspecting the specific log before trying again is retained. `اپنے ... کنفیگریشن فولڈر میں ... دیکھیں` is a natural location/action construction, with byte-identical `Local Operator` and `logs/session-host.log`. No new destructive action, implied permission or hidden host detail. No correction needed.

## Mechanical execution

`audit.log` contains actual output from a separately executed committed checker, scoped via `catalogue_root` redirection to an isolated en+ur fixture: `main(["--catalogues-only"])` returned 0 with `check.py: catalogue checks: 0 problem(s)`. Independent assertions also passed exact 3-key parity, empty argument sets and unchanged product/log literals. There are no placeholders, plural/select constructs, Rich/HTML tags, backticks, commands or flags in these messages. Urdu one/other plural completeness therefore has no branches to check. JSON parses and values remain strings.

The first custom assertion run incorrectly expected the parser's empty argument representation to be `{}`; inspection showed it returns `()`. That audit-script expectation was corrected, and the entire scoped command reran successfully. This was not a candidate failure; the final actual execution is retained in audit.log. No full-suite run, dependency installation or shared-tree write occurred.

## RTL, punctuation and fit limitations

Logical Urdu sentence order and Urdu punctuation are appropriate. No manual bidi controls are inserted. The two Latin islands in refused are preserved in logical order. Their exact visual ordering, isolation, Arabic shaping and wrapping still depend on the real renderer; this source audit is not rendered RTL proof. No arrows or directional glyph hints are present. No numeric formatting policy is implicated.

The committed context has no maxCells/maxChars hints; these are prose errors, not compact button labels. No source-level overflow finding is justified, but **no rendered fit, clipping, terminal width or layout proof was performed or claimed**.

## Retention and ledger proposal

Future canonical root: `i18n/audits/evidence/ur/wire.errors.session_host/u31h8b/`.
The bundle is presently session scratch pending coordinator retention. It is NOT committed or on main. `record.json`, `report.md`, `audit.log`, `receipt.json` and `ledgerproposal.json` are provided; the proposal is not a shared-ledger write. Preserve their byte hashes during retention. The English/context inputs were read from the committed revision; translator provenance/validation were read only as named sources, not substituted for this independent execution.

# Hindi session-host error copy — independent audit B

- Verdict: **PASSED**. All 3 keys audited; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT findings.
- Auditor: i18n-auditor / session `39a4a8de4f1c` / actual model `openai/gpt-6.1-sol` / 2026-10-31.
- Actual model source: host model-switch messages report provider-quota fallback from initial `anthropic/claude-sonnet-5-5`, via `anthropic/claude-opus-5-5`, to `openai/gpt-6.1-sol`. Semantic reasoning and execution occurred on the last model; no pinned-model claim.
- Independent from translator session `d2959affe186`; same fallback model does not mean same agent/session. No audit-A findings or artifacts were inspected. No MT service was used. No candidate, shared ledger or repository files were edited; no commits were made.
- Source revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`.
- English SHA-256: `b7d6561be15577c118ec333586f7c809e65471c5dfd0dd839abee700b2ed1ba0`.
- Context SHA-256: `d26acd5a6357a0c54bbd5ed409b4ad2c060ee52b9bd76670679397c4b464f724`.
- Hindi SHA-256: `9f9e90cfe7ef2633163f20bff9ab579e49f88b77f322eeb3e7854f208c92632d`.

## Per-key independent semantic review

### wire.errors.session_host.unsupported — PASS

Current: यह कंप्यूटर बैकग्राउंड सत्र होस्ट नहीं चला सकता, इसलिए सत्र शुरू नहीं हुआ। ऐसे कंप्यूटर का इस्तेमाल करें जिसमें समर्थित बैकग्राउंड सेवा प्रबंधक हो।

The inability belongs to this computer/platform, and the consequence explicitly says the session did not start. The remedy is a different computer with a supported background service manager, not an automatic restart or wait/retry on the unsupported platform. बैकग्राउंड preserves the background-process sense; सत्र and सेवा follow the glossary. सेवा प्रबंधक is an understandable technical compound rather than an invented product name. Everyday इस्तेमाल करें gives a natural polite instruction. No finding; no replacement wording required.

### wire.errors.session_host.unavailable — PASS

Current: बैकग्राउंड सत्र होस्ट से जवाब नहीं मिला, इसलिए सत्र शुरू नहीं हुआ। दोबारा कोशिश करने से पहले थोड़ा इंतज़ार करें।

जवाब नहीं मिला correctly expresses no answer rather than affirmative refusal. सत्र शुरू नहीं हुआ explicitly rules out a running session. The user is told to wait a little before trying again; no automatic retry or reconnecting is claimed. थोड़ा इंतज़ार करें is idiomatic, concise and polite; the nukta in इंतज़ार is retained. No finding; no replacement wording required.

### wire.errors.session_host.refused — PASS

Current: बैकग्राउंड सत्र होस्ट ने यह सत्र शुरू करने से इनकार कर दिया। सत्र शुरू नहीं हुआ। दोबारा कोशिश करने से पहले अपने Local Operator के कॉन्फ़िगरेशन फ़ोल्डर में logs/session-host.log देखें।

इनकार कर दिया preserves an explicit refusal, distinct from the no-response case. The second sentence independently makes the failed startup unambiguous. The final sentence correctly places log inspection before the next attempt. देखें naturally means inspect/check in this context; it is not an instruction to edit/delete the log. Both Local Operator and logs/session-host.log remain byte-identical. कॉन्फ़िगरेशन फ़ोल्डर is natural developer-product Hindi, with the required फ़ spelling. No finding; no replacement wording required.

## Mechanical execution evidence

See `audit.log` for this auditor's actual rerun, not merely the translator's reported result. Reused the existing translator scratch validation tree without installing dependencies or copying trees. Compared its English, context, Hindi candidate and validator script byte-for-byte against the named inputs/committed validator before executing it.

Command: `/Users/damian/local-operator/.venv/bin/python scripts/i18n/check.py --catalogues-only`

Actual exit: 0. Actual stdout: `check.py: catalogue checks: 0 problem(s)`.

Independent checks: valid JSON; exactly the same 3 keys; no placeholders or ICU plural/select structures in either language (therefore no missing Hindi one/other arms); no markup/backticks requiring preservation; technical-literal occurrence parity; no embedded bidi or unnecessary joiner controls. No destructive or permission-changing instruction was added. Hindi is LTR; Urdu-only RTL review is not applicable.

## Fit and evidence limits

The committed sidecar contains no maxCells/maxChars limits. These are full explanatory error sentences, not visibly constrained button labels. Devanagari combining/conjunct terminal shaping remains a runtime concern. **No rendered frame, measured cell width, truncation or rendered-fit proof was produced.** This pass approves semantic and mechanical copy only, not UI/design/rendering or release eligibility.

## Provenance and retention

Translator provenance and translator validator log were read only from the named candidate's siblings. Guidance: `docs/I18N.md`, RFC §6, `i18n/audits/evidence/README.md`, `i18n/glossary.md`, and `i18n/style-guides/hi.md`.

This is a newly performed second audit, not a historical recreation. `audit.log` contains actual execution output; this report's linguistic judgments are audit notes, not proof of rendering. The bundle is temporarily in session scratch as explicitly requested. Future canonical tracked destination: `i18n/audits/evidence/hi/wire.errors.session_host/h31-39a4b/`. It is **not claimed tracked, committed, merged or on main**. The requesting manager must retain these exact companions there before citing a tracked record. The ledger proposal is not an executed ledger write.

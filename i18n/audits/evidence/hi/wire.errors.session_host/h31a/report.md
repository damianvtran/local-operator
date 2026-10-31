# Hindi session-host copy — independent audit A

Verdict: PASS for this audit only; zero findings. High-visibility error copy still requires a second separate independent audit.

Auditor: i18n-auditor (task host-hi-audit-a), session b2f3bfde5529, actual model openai/gpt-6.1-sol, 2026-10-31. Actual provider-quota fallback is from the host model-switch notice (initial anthropic/claude-sonnet-5-5; intermediate anthropic/claude-opus-5-5). Translator is session d2959affe186, same actual model but a distinct agent session. No MT service used.

## wire.errors.session_host.unsupported

Current: यह कंप्यूटर बैकग्राउंड सत्र होस्ट नहीं चला सकता, इसलिए सत्र शुरू नहीं हुआ। ऐसे कंप्यूटर का इस्तेमाल करें जिसमें समर्थित बैकग्राउंड सेवा प्रबंधक हो।

PASS — Cannot run host and session not started are explicit. Supported-computer remedy is preserved, without a temporary-retry promise. बैकग्राउंड सत्र होस्ट and सेवा प्रबंधक are intelligible technical Hindi; सत्र and सेवा follow glossary. Polite इस्तेमाल करें is idiomatic. No correction suggested.

## wire.errors.session_host.unavailable

Current: बैकग्राउंड सत्र होस्ट से जवाब नहीं मिला, इसलिए सत्र शुरू नहीं हुआ। दोबारा कोशिश करने से पहले थोड़ा इंतज़ार करें।

PASS — जवाब नहीं मिला truthfully states no host response; सत्र शुरू नहीं हुआ denies successful start. थोड़ा इंतज़ार करें retains the wait-before-manual-retry remedy and does not promise reconnecting or automatic recovery. Natural polite Hindi; nukta in इंतज़ार is retained. No correction suggested.

## wire.errors.session_host.refused

Current: बैकग्राउंड सत्र होस्ट ने यह सत्र शुरू करने से इनकार कर दिया। सत्र शुरू नहीं हुआ। दोबारा कोशिश करने से पहले अपने Local Operator के कॉन्फ़िगरेशन फ़ोल्डर में logs/session-host.log देखें।

PASS — इनकार कर दिया accurately conveys refusal, separately confirms no session start, and requires inspecting the exact logs/session-host.log literal in the Local Operator config folder before trying again. कॉन्फ़िगरेशन फ़ोल्डर is an appropriate technical loan; no host internals added. No correction suggested.

## Mechanical, style and fit

Independent execution is in audit.log: repo validator exited 0 with 0 problems. Pinned source/context hashes and candidate bytes verified. Exact JSON/key parity and preserved literal checks passed. No placeholders, ICU branches, markup, flags or code spans occur; Hindi plural-category checks are therefore not applicable. Polite register, दंड punctuation, nukta and glossary terms checked for every key. No length hint is present in the committed context. These are long error paragraphs, not pinned controls; no definite copy overflow found. No rendered terminal/shaping proof is claimed: Devanagari combining/conjunct width remains dependent on the actual surface. RTL is not applicable to hi.

## Evidence and limits

Read docs/I18N.md, i18n/glossary.md and i18n/style-guides/hi.md before evaluating the candidate. English and context were read via git show at 9c979a03d551020ce6ff1b16c8066aa69ce7bb2a, not the shared working tree. Read sibling provenance.json and validator.log; independently reran the validator against the pre-existing scratch validation-tree only after verifying all three namespace inputs byte-for-byte. No installs, catalogue edits, commits, shared-worktree writes or actual ledger writes. This scratch bundle is a staging deliverable, not tracked evidence: manager must preserve it at the planned canonical path before ledger publication. No claim that this bundle is on main.

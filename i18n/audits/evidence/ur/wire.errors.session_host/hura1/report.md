# Urdu audit A — PR #3222

Verdict: PASSED for this independent first copy audit; 0 findings of any severity.
No catalogue edits, ledger writes, shared-worktree writes, commits or MT calls.
A second separate independent audit is still required for this high-visibility error group.

## Per-key judgments
- `wire.errors.session_host.unsupported`: PASS. “نہیں چلا سکتا” correctly makes the platform inability categorical, “سیشن شروع نہیں ہوا” explicitly states the failed startup, and the respectful “استعمال کریں” directs the user to a computer with a supported background-service manager. “معاونت یافتہ سروس مینیجر” is acceptable educated technical Urdu; no retry/reconnection promise.
- `wire.errors.session_host.unavailable`: PASS. “نے جواب نہیں دیا” preserves the host non-response rather than inventing a running/disconnected session. Failed startup remains explicit; “کچھ دیر انتظار کریں” before another attempt preserves the temporary-failure remedy and does not promise an automatic retry.
- `wire.errors.session_host.refused`: PASS. “شروع کرنے سے انکار کر دیا” names refusal accurately, separately reiterates that no session started, and instructs checking the log before another attempt. `logs/session-host.log` and `Local Operator` are byte-identical and in logical Latin order.

## Mechanical and RTL review
Executed the committed repository checker on a scratch-only en+ur namespace fixture: exit 0, 0 problems. A removed-key negative control was detected and the fixture restored. Independently decoded JSON and checked all three exact key IDs. No source or target placeholders, plural/select arms, markup, backticks, flags or commands exist to preserve. No manual bidi controls or non-Latin digits occur. Urdu sentence order, respectful imperatives and Urdu punctuation are appropriate. Embedded Latin product/path tokens need renderer-owned LTR isolation, not catalogue controls; no arrow/mirroring hints exist.

Source/context have no maxCells/maxChars hints. Measured target lengths are 165/122/184 Unicode characters; these are prose errors, not pinned buttons or tabs. No rendered-width or native bidi/shaping claim is made. Actual runtime/render integration is outside this copy-only audit.

## Provenance and scope
Auditor: i18n-auditor (host-ur-audit-a), session a8ce9efa0b57, actual model openai/gpt-6.1-sol, quota fallback after host switch notices from anthropic/claude-sonnet-5-5 through anthropic/claude-opus-5-5. Translator is a different session, 50b1205a23e1, but used the same actual model family/id. Independence is session/agent independence, not model diversity. Translator provenance and validator.log were read; the passing validation in audit.log was separately executed here, not copied from the translator. The checker is committed source; its runtime imports use the primary checkout read-only.

This report is reasoning/notes, not an execution log. audit.log is actual captured execution output. Canonical path is PLANNED only; this bundle has not been committed or written to main.

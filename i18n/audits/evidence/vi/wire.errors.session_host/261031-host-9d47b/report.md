# Vietnamese independent semantic audit — second pass (b)

Verdict: **passed**. Three of three keys audited. No BLOCKER, MAJOR, MINOR or NIT findings; no suggested rewrites.

Auditor: i18n-auditor, session `9d47bf14cd92`, actual model `openai/gpt-6.1-sol`, 2026-10-31 UTC. Runtime quota fallback from Sonnet via Opus was disclosed before the audit; all semantic reasoning used the final model. Translator: distinct session `26368bfbce62`, same actual model family per sibling provenance. Session independence, not model diversity, is claimed. First-pass auditor findings were not inspected. No translation/MT service was used.

Source revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`. Read Vietnamese style guide and shared glossary, then committed English and context, then candidate and sibling translator provenance/validator output. Hashes are bound in record.json. Existing translator validation fixture was reused read-only; committed English/context/checker and target candidate bytes were checked against that fixture before rerunning the repository validator. The validator execution in audit.log is this auditor's run, not the translator's prior output.

## Per-key semantic review

### wire.errors.session_host.unsupported — PASS
Current: “Máy tính này không thể chạy dịch vụ quản lý phiên nền, nên phiên chưa được khởi động. Hãy dùng máy tính có trình quản lý dịch vụ nền được hỗ trợ.”

“Không thể chạy” states platform inability, with switching to a supported computer as remedy rather than waiting/reconnecting. “Phiên chưa được khởi động” accurately states that no session has started. Vietnamese “chưa” marks the not-started state here; it does not promise startup or an automatic retry, especially beside the explicit platform limitation. “Trình quản lý dịch vụ nền được hỗ trợ” preserves the supported background-service-manager requirement. No omission or unsafe polarity change. Suggested wording: none.

### wire.errors.session_host.unavailable — PASS
Current: “Dịch vụ quản lý phiên nền không phản hồi, nên phiên chưa được khởi động. Hãy đợi một lát rồi thử lại.”

“Không phản hồi” preserves failure to answer without inventing a cause. The not-started clause is explicit; “Hãy đợi một lát rồi thử lại” directs a deliberate later user retry, not an ongoing automatic retry. Direct, idiomatic instruction with no unnecessary pronoun or “vui lòng”. Suggested wording: none.

### wire.errors.session_host.refused — PASS
Current: “Dịch vụ quản lý phiên nền từ chối khởi động phiên này. Phiên chưa được khởi động. Hãy kiểm tra logs/session-host.log trong thư mục cấu hình Local Operator trước khi thử lại.”

“Từ chối khởi động phiên này” retains refusal, distinct from transient unavailability. The separate not-started sentence remains. The instruction explicitly requires checking the specified log before retrying; path and Local Operator are exact. “Thư mục cấu hình” is idiomatic for the config folder. No sensitive runtime detail or false reconnection claim is introduced. Suggested wording: none.

## Terminology and register

“Phiên”, “dịch vụ” and “thư mục” agree with the glossary/guide. “Dịch vụ quản lý phiên nền” consistently communicates the local background session-host function without falsely suggesting a remote server. The guide's keep-English “host” decision is scoped to the OpenRouter cluster, not this service. These values use concise factual error prose, sentence case and direct “Hãy” imperatives; no pronoun inflation, honorific register or added “please”. NFC and punctuation are valid.

## Mechanical execution and limitations

Repository checker rerun: `scripts/i18n/check.py --catalogues-only`, rc=0, `check.py: catalogue checks: 0 problem(s)`. Explicit supplementary assertions passed for unique valid JSON, exact three-key/context parity, absent placeholder/ICU plural/select forms on both sides, markup/code/flag parity, per-key literal counts and NFC.

Static unwrapped Rich widths: unsupported 145 cells, unavailable 101, refused 173. These are complete multi-sentence messages, not button labels. Committed context supplies no maxCells/maxChars limit. No width cap was invented and no definite overflow is alleged. **No rendered frame, terminal wrapping or visual-fit proof was captured**; this is semantic/mechanical copy approval only. RTL is not applicable to vi.

## Retention / status

These are new second-pass evidence artifacts, not a historical reconstruction. Canonical future tracked bundle: `i18n/audits/evidence/vi/wire.errors.session_host/261031-host-9d47b/`. Files currently reside in this session's scratchpad for manager integration as explicitly requested; they are **not committed/on main**. No catalogue edits, commits, shared-ledger writes or installs were performed. `ledger-row.proposal.json` is a proposal only, not a ledger-command execution receipt. Manager must preserve this bundle in the canonical tracked root; scratch is not the final artifact of record.

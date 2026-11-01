# Vietnamese session-host errors — independent audit 1

**Verdict: PASSED for this independent pass.** All 3 keys audited; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. This is not the separate second independent pass required for high-visibility errors, ledger publication, rendered UI validation, or a release claim.

## Scope and provenance

- PR: #3222; locale: `vi`; namespace: `wire.errors.session_host`.
- Source commit: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`.
- Audit date: 2026-10-31 UTC.
- Auditor: `i18n-auditor`, session `393a054d09f7`, actual model `openai/gpt-6.1-sol`.
- Actual model source: runtime model-switch notices in this session: requested/default `anthropic/claude-sonnet-5-5`, quota fallback notice to `anthropic/claude-opus-5-5`, then quota fallback to `openai/gpt-6.1-sol`. All substantive locale judgment and verification were performed after the final switch. No Opus/Sonnet semantic audit is claimed.
- Translator: separate session `26368bfbce62`, actual model `openai/gpt-6.1-sol` as supplied and recorded in its provenance.json. Same model does not imply same agent/session; this auditor neither translated nor edited the candidate.
- No translation/MT service used. No translation, shared-worktree, or actual ledger edits performed.

## Read inputs

Read committed `docs/I18N.md`, `i18n/glossary.md`, `i18n/style-guides/vi.md`, the English source and context sidecar, RFC ledger schema, and evidence README at the specified commit. Read the complete candidate only after locale/source context. Translator provenance is used for translator identity and candidate location, not as semantic approval or as a substitute for the validator run here.

The input hashes were independently recomputed from committed Git blobs and candidate bytes; all match the supplied values:

- English: `b7d6561be15577c118ec333586f7c809e65471c5dfd0dd839abee700b2ed1ba0`.
- Context: `d26acd5a6357a0c54bbd5ed409b4ad2c060ee52b9bd76670679397c4b464f724`.
- Vietnamese: `d1ce1cbd3daa26c04bf01122319e7e8adec63050477c8dd8bc84ff61f16003ba`.

## Mechanical execution

Reused the translator's existing validation fixture; no install or dependency-tree copy. Independently verified its English/context/candidate bytes and 7 relevant validator/parser/loader/format/plural inputs against the requested commit before running it. The fixture was used only for validation; shared repository files were not changed. The validation child used a scratch HOME/config, removed inherited `CMUX_*`, `XPC_FLAGS`, `PYTHONPATH`, and `PYTHONHOME`, and disabled system Git credential helpers/prompts.

Actual command: `/Users/damian/local-operator/.venv/bin/python scripts/i18n/check.py --catalogues-only`

Actual cwd: `/Users/damian/.local-operator/sessions/26368bfbce62/scratchpad/host-copy-vi/validation-tree`

Actual stdout: `check.py: catalogue checks: 0 problem(s)`; exit 0. This is the catalogue-only validator, not a full test suite, UI exercise, or release gate. `audit.log` retains the actual successful command output and independent checks.

Independent checks: JSON duplicate-key detection, exact 3-key source/target/context parity, empty and identical typed argument signatures, parser/render round trip, unchanged `logs/session-host.log` and `Local Operator` where present in source, NFC text, no format/bidi control characters. No ICU plural/select branches, placeholders, markup, backticks, commands or flags occur in this group, so no locale plural arm is missing. Source and target are ordinary unmarked prose plus the exact literal path and product name.

Setup incident: an initial verifier attempted an unnecessarily broad per-file Git comparison over generated i18n Python files and hit the 120-second tool timeout before producing output or reaching the catalogue validator. It was narrowed to the 7 actual relevant inputs and rerun successfully. That initial timeout is disclosed here as a tool-observed setup event, not represented as a validator failure or fabricated execution log. Only the successful run is in `audit.log`.

## Per-key semantic and native-copy audit

| Key | Current Vietnamese | Review and disposition |
|---|---|---|
| `wire.errors.session_host.unsupported` | Máy tính này không thể chạy dịch vụ quản lý phiên nền, nên phiên chưa được khởi động. Hãy dùng máy tính có trình quản lý dịch vụ nền được hỗ trợ. | PASS. `Máy tính này không thể chạy` makes this computer's lack of support explicit; changing to a supported computer is the remedy, not reconnecting or waiting. `phiên chưa được khởi động` states no session was started. `trình quản lý dịch vụ nền` correctly denotes the supported service manager, distinct from the session-host service. |
| `wire.errors.session_host.unavailable` | Dịch vụ quản lý phiên nền không phản hồi, nên phiên chưa được khởi động. Hãy đợi một lát rồi thử lại. | PASS. Non-response, not refusal or unsupported platform. The failed start is explicit. `Hãy đợi một lát rồi thử lại` asks the person to wait and retry deliberately; it neither promises an automatic retry nor asserts reconnection to an already running session. |
| `wire.errors.session_host.refused` | Dịch vụ quản lý phiên nền từ chối khởi động phiên này. Phiên chưa được khởi động. Hãy kiểm tra logs/session-host.log trong thư mục cấu hình Local Operator trước khi thử lại. | PASS. `từ chối khởi động` preserves the host's refusal, not a timeout. Separate failed-start sentence and required log inspection before retry remain intact; the exact log path and product name are preserved. No unsupported-platform meaning is incorrectly attached to this refusal. |

`dịch vụ quản lý phiên nền` is a defensible functional description of the background session host: a background service managing sessions, not a remote machine. It consistently distinguishes that service from `trình quản lý dịch vụ nền` (service manager). The style guide's keep-English `host` decision is specifically for the OpenRouter cluster and is not a blanket ban on describing this different service role. The candidate does not misuse `máy chủ` (server). `phiên`, `thư mục`, and `cấu hình` are natural Vietnamese developer/product wording; direct `Hãy` instructions are appropriate and do not add forbidden `Vui lòng`. Pronouns are omitted naturally, as prescribed.

The phrase `chưa được khởi động` has a normal Vietnamese perfective-negative reading here: the session has not been started at this failed attempt. Its isolated potential English gloss 'not yet started' does not make a false promise when paired with explicit inability/refusal and user-driven remedies. I checked that reading against all three surrounding full messages rather than flagging the word `chưa` mechanically.

## Fit and limitations

Measured Rich terminal cell widths, also character counts because every character occupies one cell here:

| Key suffix | vi cells/chars | en cells | maxCells/maxChars hint |
|---|---:|---:|---|
| unsupported | 145 | 145 | none |
| unavailable | 101 | 110 | none |
| refused | 173 | 174 | none |

All text is NFC, no decomposed-combining sequences or wide-glyph expansion, and the translations are not wider than English. These multi-sentence errors require wrapping just as the English does. No declared fit budget is violated. No rendered frame or actual terminal-width behavior was exercised; measured parity is not rendered proof. RTL is not applicable to Vietnamese.

## Findings and integration disposition

No findings. No suggested rewrite is needed. The first independent pass may be recorded as `passed` with unresolved blocker/major count 0 at these exact bytes; the second independent high-visibility audit is still pending and is not claimed here. `ledger-row.proposal.json` is a proposal only, not an update or permission to bypass that pass. The committed RFC/HOWTO describe appending a locale-shard row rather than naming a dedicated audit-ledger command; per this task, no ledger-writing operation was attempted.

Manager integration destination: `i18n/audits/evidence/vi/wire.errors.session_host/vh31a1/`. These artifacts remain session-scratch drafts pending manager integration and are not claimed tracked, committed, or on main. Preserve hashes/provenance and record reference when integrating. Link `verify.py` as `checks.py.txt` in the canonical bundle; it is reproducibility evidence, not another shipped executable.

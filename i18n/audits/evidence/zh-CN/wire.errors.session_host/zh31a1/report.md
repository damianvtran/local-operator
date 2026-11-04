# Independent zh-CN translation audit — first pass

- PR: #3222; namespace: `wire.errors.session_host`; audit date: 2026-10-31 UTC.
- Auditor: i18n-auditor; session `45153dd07ae1`; actual model `openai/gpt-6.1-sol`.
- Provenance: explicit runtime model-switch notice identifies this temporary fallback after provider quota failure. Earlier notices named primary anthropic/claude-sonnet-5-5 and intermediate anthropic/claude-opus-5-5; this substantive audit was performed after the gpt-6.1-sol switch.
- Independent of translator session `d90daece65d4`, whose provenance.json names actual fallback `openai/gpt-6.1-sol`. Different sessions/agents; same actual model is disclosed, not represented as model diversity.
- Pinned source/context revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`.
- Policy read before target: pinned `i18n/style-guides/zh-CN.md`, `i18n/glossary.md`, and `docs/I18N.md`; audit/retention contract: RFC §6 and evidence README.

## Verdict

**PASSED for this first independent copy-audit pass**, 3/3 keys inspected; zero BLOCKER, MAJOR, MINOR or NIT findings. This is not a passed actual ledger entry, completed second audit, rendered UI certification, release/security waiver or shipped-locale authorization. Manager requires a separate independent second pass for this high-visibility error namespace. No shared worktree writes, translation edits, installs, MT services or full-suite execution occurred.

## Every-key semantic and linguistic audit

### wire.errors.session_host.unsupported
Current: 此计算机无法运行后台会话托管服务，因此会话未启动。改用配有受支持后台服务管理器的计算机。

Accepted: clearly attributes inability to this computer, explicitly says the session did not start, and directs the person to a computer with a supported background service manager. It does not suggest retrying on the unsupported platform. 后台会话托管服务 denotes the local runtime-hosting service without suggesting a remote machine; the pinned host module identifies a supervisor-owned process. 后台服务管理器 preserves the supported-manager requirement. The compressed attributive phrase remains understandable developer copy; no invented product name or unsafe troubleshooting instruction. 会话/服务/计算机 are consistent and neutral. Full sentences retain Chinese full stops.

### wire.errors.session_host.unavailable
Current: 后台会话托管服务未响应，因此会话未启动。稍后再试。

Accepted: 未响应 preserves no answer, 未启动 preserves no newly started session, and 稍后再试 is idiomatic manual later retry. No automatic retry, reconnect, resumption or existing running session is promised. Terminology matches unsupported/refused; tone is concise factual product copy.

### wire.errors.session_host.refused
Current: 后台会话托管服务拒绝启动此会话。会话未启动。重试前，查看Local Operator配置文件夹中的logs/session-host.log。

Accepted: explicitly distinguishes refusal from unavailability and repeats non-start as the English source does. Log inspection precedes retry, not the reverse. Local Operator and logs/session-host.log remain byte-identical and in the correct config-folder relationship. 配置文件夹 is natural Chinese for the configuration folder in context; no destructive operation, permission bypass or sensitive dynamic host detail is introduced. Chinese punctuation and no added CJK/Latin spaces conform to the locale guide. No unnecessary 请/您 or informal particles.

## Mechanical execution and independent checks

See audit.log for actual validator outputs. I ran the pinned repository validator in a minimal scratch-only snapshot containing its required runtime/data plus exactly this en/zh-CN namespace and English sidecar:

`/Users/damian/local-operator/.venv/bin/python scripts/i18n/check.py --catalogues-only`

Actual result: `check.py: catalogue checks: 0 problem(s)`, exit 0. A missing-key negative control produced `zh-CN/wire.errors.session_host: missing key 'wire.errors.session_host.unavailable'`, 1 problem, exit 1. Restoring the original exact bytes and re-running returned 0 problems, exit 0. This establishes the checker actually scanned the target, not an empty directory. The translator's validator.log was read as background, not reused as independent execution evidence.

Separate executed checks: duplicate-key-strict JSON; exact source/context/target key sets (3); all values strings; no placeholders, plural/select branches, Rich/markup tags, backticks, code/command/flag tokens or interpolation markers in either source or target; no target control characters. Thus placeholder types, plural CLDR coverage and markup parity are vacuous here, not missing required plural arms. Product/path literal counts match the source. All three requested SHA-256 hashes independently matched.

## Fit and scope limits

Measured target Unicode terminal cells (W/F=2, combining=0, otherwise 1), characters: unsupported 88 cells / 44 characters; unavailable 50 / 25; refused 107 / 71. No source maxCells/maxChars hints exist. These are multi-sentence error bodies, not fixed-width button/status labels. CJK wide-character cost is acknowledged; no combining/control hazards were present. No rendered terminal/browser or live failure-flow test was run, so wrapping/clipping and runtime delivery are NOT certified. The literal log path contains no added breaks or bidi controls. RTL is inapplicable to zh-CN.

## Findings

None. No suggested replacement wording is necessary. Reasoning in this report is `audit_notes_not_execution_log`; command results are in audit.log.

## Retention and ledger

Future tracked bundle: `i18n/audits/evidence/zh-CN/wire.errors.session_host/zh31a1/`. Current bundle is scratch-only and uncommitted. Preserve it until the manager copies/commits the exact files; do not claim it exists on main. `ledger-proposal.json` is an unapplied proposal, conditional on the separately required second pass and preservation of these exact input bytes. The RFC specifies locale shard rows; no actual ledger was changed or recording command run by this auditor.

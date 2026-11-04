# Audit — es / wire.errors.agents / w3a1

- **Verdict: passed (clean)** — 0 blocker, 0 major, 0 minor, 0 nit
- Revision: `66da866475f3e74553196a3167c18c6125141844`; target **untracked**; on-disk sha256 `9bece5afaae31649dde1f1dd3d97d4de4517cb4c0c4a8c9ed6e477377814f448`
- Keys read: **44** en / 44 es
- Auditor: independent translation-auditor, session `c328736e07e9`; model **kimi/k3.1-pro** (self-reported temporary provider-quota fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

Every key was read side by side with English. Agent, subagent-related execution data, team, job and schedule failures preserve the same entity and operation. “Not found” remains absence; local read/check/publish/pull failures remain local failures; cancellation remains refused for an already terminal job; validation and required-field errors retain their causes. No refusal or failure reads as success, and no permanent/retryable direction is changed.

The real `local_operator.i18n.runtime` parser accepted all 44 targets. Ordered keys, all 35 source ICU arguments, plural selectors and select-label shapes match; flags, code spans and checked identifiers (`RADIENT_API_KEY`, `hub_agent_id`, `system_prompt`) survive. No plural branches occur, hence no Spanish `many` case is needed and no `few`/`zero`/`two` appears. `agents_refusal` is exactly `{error}` on both sides: correct pure passthrough, not a missing translation. Representative synthetic renders preserve quotes and IDs. Exact commands/output are in `audit.log`; the reproducible read-only probe is `checks.py.txt`.

No English context sidecar exists (`wire.errors.agents.context.json` absent). Register was therefore checked against the shipped Spanish error copy named in the delegation, not an invented contract. Agent/agente, subagent/subagente, team/equipo, job/trabajo, schedule/programación, prompt/prompt and tenant/tenant are consistent; tú forms and sentence case fit that register.

Length was measured rather than treated as a hidden budget. The largest growth is `error_updating_system_prompt`, 43 → 61 characters (1.419×); it remains concise and is not a clipping finding without `maxCells`.

## Findings

None. All 44 keys are clean.

## Reader limitation and scope

No Spanish target value displayed as a redaction marker. An unrelated zh-CN value appeared as `[redacted]` during broad source searching and was not judged. The sanctioned `read_public_copy` route was exercised successfully on tracked English media source; it cannot read these untracked Spanish targets or absent sidecars, and no workaround was used.

No catalogue or programme file was repaired. Only evidence was written. No native review, rendered visual fit, release eligibility or repo-wide test result is claimed.

# Independent zh-CN account-policy audit — second reader

Verdict: passed, linguistic/mechanical scope only. 14/14 keys; no findings.

All 14 entries were independently read against English, proposed context and native style/glossary. No other auditor report was inspected.

- label: 账户选择策略 is direct product terminology.
- help: 同一提供商、同一凭据层级 preserves both eligibility constraints. 对于所有策略 preserves universality; 成功请求留下的兼容且未过期的租约 preserves success, compatibility and expiry conditions; 这并不保证缓存命中 does not promise a cache hit. Weekly refill is primary; five-hour refill breaks the within-one-minute tie. Explicit policy takes precedence over legacy.
- reset_soonest.label: 最早重置 is concise and correctly ordered.
- reset_soonest.description: 套餐内额度 preserves included-plan scope; 按确定性规则分散选择 correctly describes deterministic spreading of ties and unknown reports. New-configuration default preserved.
- headroom.label: 剩余额度最多 accurately names the policy.
- headroom.description: 缓存的剩余额度 preserves cached (not live) quota; 十个百分点 preserves percentage POINTS, not a relative ten-percent difference.
- fair.label: 均衡分配 is idiomatic for fair spread.
- fair.description: 按会话稳定地分散选择 preserves stable session-based spreading and explicitly negates reset/quota ranking.
- legacy.label: 旧版按用量选账户 is compact and marks legacy status.
- legacy.help: only-if-policy-absent is explicit. true maps to Most headroom, false to Fair spread. Explicit-policy precedence, resetting the POLICY row, restoring Reset soonest and clearing legacy overrides all remain explicit.
- error.unavailable: 账户路由不可用 preserves availability refusal without inventing recovery.
- error.invalid_shape: 仅应包含value字段 preserves ONLY and the literal field name.
- error.invalid_json: 应为JSON对象 preserves object requirement, not merely valid JSON.
- error.write_failed: refusal and configuration-permission remedy preserved; no raw exception/path added.

Context provenance: translation was produced under original context ccee657198d84a0bef6d20ae0343c9a864567e5a622a1b9736b1974ca296a254. This reader audits against proposed context 2c23cc38b8eb9fb354eda0e1dd476d53b1064ffe88c2e821b9e1b723b8508c02; these are NOT represented as identical or as the translator's same input. Independently checked all 14 non-note objects unchanged; English source equals pinned commit bytes.

Caller evidence: packet settings_io.caller.txt shows enum IDs reset-soonest/headroom/fair separate from localized labels/descriptions. account_routing.caller.txt shows stable codes/statuses/value fields separate from human message/metadata. These are source inspections, NOT endpoint execution.

NOT RUN: pseudo-localization, native application/real-path UI, rendered fit, screenshots, RTL rendering, design review, full suite/scanner/build. Rich cell counts are static width proof only. No activation/eligibility/ship claim; integration gates remain separate. No target/repo/ledger edits.

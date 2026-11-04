# Independent zh-CN SDK-spec audit — second reader

Verdict: PASSED (staged linguistic/mechanical catalogue only). 21/21 keys audited. No BLOCKER, MAJOR, MINOR or NIT findings.

Per-key native assessment (suffixes under wire.errors.sdk_spec):
- agent_identity_conflict: 不能同时指定 faithfully states mutual exclusion, preserving both identifiers.
- approval_callback_gate_required: 可调用的审批函数 correctly requires a callable gate, not merely a present handler.
- approval_declared_tools_empty: at least one tool, undeclared bound-session condition and refusal alternative all retained.
- approval_handler_wrong_mode: callback-only applicability and rejection of opaque mode preserved; supplied repr value not re-quoted.
- approval_tools_wrong_mode: declared-only applicability retained, distinct from callback requirement.
- attach_fields_unsupported: viewer-only relation to spec.resume, owner responsibility, only-resume restriction, own alternative and offending fields retained. No ownership-transfer implication.
- attach_output_unsupported: output enforcement belongs to owning runtime; target does not imply viewer control.
- attach_resume_required: spec.resume requirement and attaching via session ID preserved.
- mode_invalid: own/attach literal alternatives and rejected opaque mode unchanged.
- output_format_invalid: enumerated permissible formats preserved through opaque formats argument.
- output_retries_range: inclusive 0 through 5 preserved.
- output_retries_requires_format: explicit dependency retained without implying retries mandatory generally.
- output_schema_requires_format: explicit schema dependency retained separately from retries.
- resume_empty: session ID or literal @latest, not empty string, retained.
- spawn_birth_effort_pair_required: new-child birth snapshot and all-three parameter requirement retained; in-process open_session alternative permits birth level without model pair.
- spawn_fields_unsupported: post-open state cannot cross new runtime child; ordered open/attach/dispose/same-ID spawn/resume restoration remedy and omission alternative preserved. dispose stays technical operation, not data deletion.
- spawn_model_pair_required: both-or-neither hosting/model atomic birth snapshot retained; missing pair resolves from root config, not current viewer state.
- spawn_output_unsupported: cannot carry enforcement; open_session own-mode followed by engagement preserved.
- tools_empty: at least one tool versus unrestricted None distinction clear.
- world_root_empty: non-empty path only when set retained.
- yolo_approval_unreachable: unreachable approval gate and auto-or-remove-yolo remedy preserved; no safety polarity reversal.

Native terminology/register: terse developer diagnostics; 会话/工具/审批 follow glossary. 输出约束 conveys output enforcement; 创建时的参数快照 makes birth sample intelligible. Technical APIs, identifiers, quoted mode literals, None and @latest remain untranslated. No artificial politeness, pronoun repetition or English prose fallback added. Chinese punctuation with Latin technical fragments follows style guide. No numeric width caps exist: no rendered-fit certification.

CONTRACT LIMIT: English and original context remain immutable. All 21 pinned producer calls explicitly render locale=en before constructing historical SessionSpecError, a ValueError subtype. This independent source inspection is NOT a production exception execution. Catalogue translation does not migrate caller contracts, localize existing exceptions, alter args/str compatibility, create a UI presentation path or claim activation. Architect's no-current-UI-sink assessment is attributed to the provided assessment, not independently proven by this audit.

Evidence distinction: report is linguistic reasoning/source assessment; mechanical.json and audit.log report actual isolated checker/runtime interpolation executions. Opaque interpolation checks the existing core formatter, not SDK/endpoint/native app paths.

NOT RUN: production SDK exception/runtime execution, native application, visual fit, pseudo-localization, RTL rendering, design, full suite, build, scanner. RTL is not applicable to this LTR locale; no cross-locale proof claimed. No target/repo/ledger edits or prior policy-bundle changes.

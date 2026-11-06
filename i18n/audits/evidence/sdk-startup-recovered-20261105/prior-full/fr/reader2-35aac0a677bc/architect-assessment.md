# SDK localization contract assessment

Source: architect session `96042fdf7d41`; completed report recovered from its persisted transcript after the volatile wait registry lost job `bad578da4058`. Actual completed usage: `openai/gpt-6.1-sol`. Assessment only; no implementation or live/runtime proof.

## Recommendation

**Translate all 75 missing key-locale pairs; do not exempt these namespaces. Separately preserve the SDK’s legacy exception-string contract while providing structured identity for localized human presentation.** None of these 25 sentences is a machine/model-only token that must remain English.

Assessment pinned to **`0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`** using committed blobs, not the moving checkout. No source edits, builds, installs, PRs or full-suite runs.

**Architect/model actually used:** architect / `openai/gpt-6.1-sol`, after the reported provider-quota fallback from `anthropic/claude-opus-5-5`.

## Source matrix

Prefixes below are `wire.errors.sdk_spec.` and `wire.errors.sdk_startup.`. All file:line references refer to the pinned commit.

| Keys | Actual producer | Classification / current behavior |
|---|---|---|
| `approval_declared_tools_empty`, `approval_tools_wrong_mode`, `approval_callback_gate_required`, `approval_handler_wrong_mode` | `local_operator/session/spec.py:332–365` | Human-readable developer diagnostics. Each explicitly renders `locale="en"` and raises `SessionSpecError(text)`. |
| `agent_identity_conflict`, `resume_empty`, `tools_empty`, `yolo_approval_unreachable` | `session/spec.py:509,515,524,533` | Human-readable invalid-input diagnostics; English-pinned as above. Names and policy/mode tokens remain technical data, **not the surrounding sentences**. |
| `output_format_invalid`, `output_schema_requires_format`, `output_retries_requires_format`, `output_retries_range`, `world_root_empty` | `session/spec.py:547,555,561,571,580` | Human-readable validation/remedy copy; English-pinned as above. |
| `mode_invalid`, `attach_resume_required` | `local_operator/sdk.py:924,931` | Human-readable pre-isolation refusals; explicitly English. Must not introduce ambient configuration reads here. |
| `attach_output_unsupported`, `attach_fields_unsupported` | `sdk.py:1002,1014` | Human-readable attachment restrictions/remedies; explicitly English. Field lists are opaque identifiers. |
| `spawn_output_unsupported`, `spawn_fields_unsupported` | `sdk.py:1066,1082` | Human-readable spawn restrictions/remedies; explicitly English. Preserve behavior and field inventory. |
| `spawn_model_pair_required`, `spawn_birth_effort_pair_required` | `sdk.py:1161,1167` | Human-readable birth-configuration remedies; explicitly English. |
| `attach_runtime_missing` | `sdk.py:652–660` | Already uses scoped locale rendering through `startup_refusal_text`; disposes a mistakenly built owner **before** refusal. Translation must preserve that lifecycle. |
| `profile_attach_failed` | `sdk.py:665–673` | Already scoped-locale rendered; profile name is opaque data. |
| `tool_inventory_mismatch` | `sdk.py:674–695` | Already scoped-locale rendered; tuple strings are data. Translation must not alter approval/inventory admission. |
| `attach_hosting_unconfigured` | `sdk.py:614–632` | Already scoped-locale rendered, **but embeds `error=str(error)`**. Its real producer raises English `"Hosting platform is not configured."` at `local_operator/session_factory.py:816`. Translating the outer sentence alone therefore leaves owned English inside it. |

`startup_refusal_text` reads language from the already scoped configuration manager and renders the message at `sdk.py:587–598`. In contrast, all 21 spec producers explicitly force English.

### What the “caller contract” actually establishes

- `SessionSpecError` is a `ValueError`; its documented contract preserves one string argument and no new exception attributes: `session/spec.py:84–92`.
- The spec tests pin exact English bytes, arguments and English rendering: `tests/unit/i18n/test_sdk_spec_slice.py:136–157`.
- Startup is **not universally English-pinned**: it already selects the scoped locale. Its tests explicitly support translated catalogues: `tests/unit/test_sdk.py:1519–1532`.
- The Spanish allowance claims SDK callers parse these sentences (`i18n/coverage-allowance.json`, Spanish `sdk_spec` entry). **I found no production parser for these 25 diagnostic sentences in the pinned core.** Source searches found the producers, tests and evaluation consumers—not a demonstrated sentence-parsing client. External SDK clients remain unknown, not proven absent.

Thus the allowance’s compatibility concern is legitimate, but its claim of actual parsers is not substantiated by the examined production paths.

## Actual interface exposure

**I found no direct path from these SDK exceptions into the core desktop/mobile interface.**

- Production SDK consumers found are evaluation machinery: `local_operator/evaluation/session_arm.py:2714–2721` and `evaluation/team_bridge_moderation/agent.py:707–716`.
- Desktop session creation uses its own host/pool path: `local_operator/server/routes/desktop_sessions.py:3070–3095`.
- Mobile `spawn_session` is a different function; it launches/adopts standalone runtime processes, not `sdk.spawn_session`: `local_operator/mobile/daemon.py:5353–5366`. Its start refusal response preserves machine codes at `daemon.py:8985–8993`.
- Mobile’s genuine English parser, `humanizeError`, matches pending-answer/connection phrases—not these SDK sentences: `mobile/web/src/components/pending-card.tsx:85–106`, consumed at lines 220 and 370.

**Consequently, do not expand this task into migration of unrelated desktop/mobile parsers.** Also do not claim that translating these catalogues proves the whole interface contains no English.

The envelope contract already distinguishes stable machine categories from message identity: `local_operator/i18n/messages.py:70–85`. Preserve that distinction; do not replace status/category codes with catalogue IDs.

## Smallest contract-safe next scope

### 1. Complete the catalogues, prioritizing `zh-CN`

Add idiomatic, independently audited translations for both namespaces in `zh-CN`, `fr`, and `es`; regenerate and remove exactly their 75 allowance gaps.

This is real catalogue completion, not duplication of English. Existing Russian, Vietnamese, Urdu and Hindi catalogues already contain all 25 keys—confirmed directly from the pinned blobs.

**Catalogue translation alone does not change the 21 English-pinned exception producers.** State that explicitly in completion evidence.

### 2. Fix the existing SDK error boundary—not a second error framework

For keyed SDK refusals, add optional stable message identity/params to the existing `SessionSpecError`, preserving:

- `ValueError` inheritance and existing catches;
- legacy `args`/`str` behavior;
- existing unkeyed `SessionSpecError` call sites;
- constructor/import cheapness and pre-isolation no-config-read behavior.

Provide localized human rendering from those identities with an explicitly supplied resolved locale. Any human presenter must use that renderer rather than paint legacy `str(error)`. The migration requires an explicit update to the old “no attributes” contract and tests; it is not merely translation work.

**Trade-off:** this additive boundary preserves unknown external string consumers, but their legacy strings remain English. Those bytes are compatibility output—not translated product copy and not an exemption for displaying them in the interface. A later change to default exception `str()` requires an explicit SDK compatibility disposition; this assessment does not authorize breaking it.

Because no current desktop/mobile SDK sink was found, do not invent new client wiring solely to consume the metadata.

### 3. Close the embedded-English trap

For `attach_hosting_unconfigured`, preserve the original exception/cause for diagnostics, but use localized, owned copy at a human-display boundary. Do not classify `"Hosting platform is not configured."` as opaque third-party prose.

Prefer a display-specific message omitting raw cause prose—the outer diagnostic already identifies the problem and remedy—over changing the existing key’s `{error}` semantics or translating arbitrary exception strings. Any new owned message must ship in **every supported locale**, not just these three.

## Acceptance and minimal proof

- **Catalogue:** exact English-key parity for both namespaces in every supported locale; no absent namespace, allowance growth, baseline increase or held-locale exception.
- **Compatibility:** exercise all 21 real guard branches; retain legacy exception type/arguments while asserting correct identity/params and localized human rendering.
- **Startup:** exercise all four branches through `open_session`; verify scoped language, configuration isolation, disposal, profile failure and unchanged inventory refusal.
- **English leakage:** missing-hosting reproduction must show no embedded owned-English sentence in localized human output; preserve diagnostic cause separately.
- **Negative controls:** wrong key, missing/incorrect params, ICU-like opaque input, and a planted untranslated owned sentence must fail their respective checks.
- **Transport regression:** unauthorized and invalid-start requests retain existing status, machine code and side-effect behavior; preserve mobile’s existing parser behavior. These are regression controls, not alleged SDK exposure.
- Independent QA supplies real request/output evidence. Any affected rendered surface needs actual rendered proof.

**Verified here:** source-level exact coverage of 21 + 4 producers; 63 + 12 missing key-locale pairs; explicit English pinning of all 21 spec producers; existing translations in the other four non-English locales; a synthetic extra-key negative control.

**Unverified:** external SDK parser behavior, live endpoint responses and rendered UI. This role has no browser tool and performed read-only source assessment.

**Non-goals:** account-policy’s 14 keys, unrelated error families/parsers, SDK lifecycle redesign, new envelope versions, model prompts, release work.

**Access blocker:** none for the proposed bounded work.

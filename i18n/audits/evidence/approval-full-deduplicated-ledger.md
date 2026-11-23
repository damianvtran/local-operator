# Full wire.settings semantic audit — consolidated evidence

Frozen head: `a7c126c387c2dd71f61067c0b2fa37dbc6f95cfa`. Scope: `fa4fb006..a7c126c387c2`. Round 1 aggregation; not approved for product repair.

> Operator request: “switching to fr must leave NO English anywhere in the interface”; same requirement for all seven locales.

**Whole namespace: FAIL for all seven locales. Two changed approval labels: PASS in both independent auditor reports for every locale.** These are different scopes; no whole-interface, rendered-fit, release or merge clearance is claimed.

## Derivation and boundaries

- All 14 final principal reports read in full as JSON. 21 exact-byte raw snapshots preserved, including blocked/partial originals and supplements; eight official input envelopes each contain exactly 895 keys matching English.
- Ledger deduplicates to **257 source keys**, **577 locale/key entries**. Severity maxima are source severities, not new linguistic judgments. Width-only entries remain risks, never observed clipping.
- Exactly **895 coverage records** enumerate the frozen namespace, each with seven locale/A/B attestation bindings. These are coverage joins of actual auditor receipts, not a fabricated new semantic review or a zero-key pass. Four reports have explicit per-key enumerations checked against all 895 keys; others supply fresh full-read attestations/ranges.
- Aggregator served model: **openai/gpt-6.1-sol**, explicit host quota fallback. Primary audits report the same GPT fallback. HI-B last-key continuation’s Anthropic header claim remains **actual serving model unverified/resumed-route ambiguous**. Urdu translator model claim is **header-only/unverified**. Original authors for 893 inherited keys remain **unknown**.
- A/B sessions are distinct and distinct from the corresponding two-label author. Independence of inherited authors cannot be established because those authors are unknown; do not invent attribution.
- No product edits, translations, English rewrites, baseline raises, generation, tests, builds, dependencies, native/browser work, commit, PR or push. Semantic proposals below are quoted solely from source auditors.

## Locale outcome and A/B discrepancies

| Locale | A major keys | B major keys | Union major keys | Two labels | Namespace |
|---|---:|---:|---:|---|---|
| zh-CN | 32 | 32 | 32 | PASS A/B | FAIL |
| fr | 21 | 23 | 23 | PASS A/B | FAIL |
| es | 27 | 27 | 30 | PASS A/B | FAIL |
| ru | 33 | 33 | 33 | PASS A/B | FAIL |
| vi | 34 | 39 | 39 | PASS A/B | FAIL |
| hi | 54 | 35 | 55 | PASS A/B | FAIL |
| ur | 45 | 51 | 56 | PASS A/B | FAIL |

Counts are distinct affected keys, not systemic groups. Equal counts do not imply equal key sets. An absent partner finding is not rejection; Root must decide the union, not silently intersect it.

### zh-CN — discrepancies

**A-only major keys:** none.

**B-only major keys:** none.

### fr — discrepancies

**A-only major keys:** none.

**B-only major keys:** `wire.settings.shell_environment_inherit.label`, `wire.settings.team_bridge_session.label`.

### es — discrepancies

**A-only major keys:** `wire.settings.display_composer_chevron.label`, `wire.settings.speech_voice_expressiveness.choice.medium.description`, `wire.settings.team_bridge_session.label`.

**B-only major keys:** `wire.settings.subagents_models_hi.help`, `wire.settings.subagents_models_lo.help`, `wire.settings.subagents_models_med.help`.

### ru — discrepancies

**A-only major keys:** none.

**B-only major keys:** none.

### vi — discrepancies

**A-only major keys:** none.

**B-only major keys:** `wire.settings.fork_cmux_placement.choice.surface.label`, `wire.settings.fork_cmux_placement.help`, `wire.settings.session_scratchpad_max_bytes.help`, `wire.settings.session_scratchpad_max_bytes.label`, `wire.settings.team_bridge_session.label`.

### hi — discrepancies

**A-only major keys:** `wire.settings.classification_vendor.choice.typesafe.description`, `wire.settings.hub.description`, `wire.settings.hub_auto_update_agents.choice.true.description`, `wire.settings.hub_auto_update_agents.help`, `wire.settings.hub_auto_update_teams.choice.true.description`, `wire.settings.hub_auto_update_teams.help`, `wire.settings.local_providers.base_url.help`, `wire.settings.local_providers.base_url.label`, `wire.settings.local_providers.description`, `wire.settings.network_advertise_hosts.label`, `wire.settings.network_max_handshakes.help`, `wire.settings.network_port.help`, `wire.settings.providers_openrouter_enforce_distillable_text.choice.true.description`, `wire.settings.providers_openrouter_enforce_distillable_text.help`, `wire.settings.providers_openrouter_require_parameters.help`, `wire.settings.providers_openrouter_zdr.choice.true.description`, `wire.settings.providers_openrouter_zdr.help`, `wire.settings.shell_environment_inherit.label`, `wire.settings.speech_transcription_cleanup.help`, `wire.settings.web_search_searxng_endpoint.label`.

**B-only major keys:** `wire.settings.web_fetch_render_backend.help`.

### ur — discrepancies

**A-only major keys:** `wire.settings.librarian.title`, `wire.settings.librarian_enabled.label`, `wire.settings.mesh_move_working_set_roots.help`, `wire.settings.team_bridge_screening_ambient_budget_per_hour.help`, `wire.settings.team_bridge_screening_unaddressed_budget_per_hour.help`.

**B-only major keys:** `wire.settings.compaction_wire_bytes_budget.label`, `wire.settings.hub.description`, `wire.settings.hub_auto_update_agents.choice.true.description`, `wire.settings.hub_auto_update_agents.help`, `wire.settings.hub_auto_update_teams.choice.true.description`, `wire.settings.hub_auto_update_teams.help`, `wire.settings.mesh_move_working_set_max_bytes.help`, `wire.settings.network_sync_debounce_s.help`, `wire.settings.network_sync_tick_s.help`, `wire.settings.session_cleanup_max_total_bytes.label`, `wire.settings.subagents_model_choice.choice.operator.description`.

## Immutable inputs, raw originals, and model provenance

The machine-readable artifact manifest binds each original path, snapshot SHA256, byte count, source/head hashes, independent sessions, input-envelope/canonical-map/formatted-text digests, and qualifications. Raw copies are never normalized. Only ledger key IDs are normalized.

| Locale | Author session (two labels only) | A auditor session | B auditor session |
|---|---|---|---|
| zh-CN | `bc0e0838eb86` | `6448acb16e8d` | `409f8c7ded31` |
| fr | `e2ef41ac5bb7` | `24ea56c742bf` | `8136478925f8` |
| es | `9d62afa9b687` | `69bd93578cd9` | `bdbb1a53f92b` |
| ru | `a3eb1f43777c` | `00c790f37e53` | `870397d651f4` |
| vi | `a7c4f2fde35e` | `8c17083550fb` | `6501d77e8fa2` |
| hi | `c456dba3488d` | `9a790340d863` | `bbc40416334c` |
| ur | `bc9d98e0db6e` | `06289cd611cf` | `28a481e6b878` |

- French A: blocked first acquisition remains an actual blocked receipt; completion supplies final 895 coverage.
- Hindi B: original failed receipt supplies 894 readable pairs; exact-key supplement supplies the remaining one. “Retained 894” in that supplement is same-audit continuation, not old convergence evidence. Whole namespace still FAIL.
- Urdu B: earlier partial 894 snapshot remains preserved; separately preserved final receipt binds 895. Its final hash is `02d5d916f1b0b633ec537fb9d44dfca774278dcfe6330e81407595d0928b9ea2`.
- Hindi/Urdu Bedrock help: ordinary output masking is not a literal source `[redacted]` defect. Dedicated reader evidence is an auditor call report/observation where no full tool envelope was captured. Do not mislabel it captured-envelope evidence.
- ICU/CLDR/RTL/pseudo outputs and negative controls are preserved per auditor in JSON `control_evidence` and exact originals. They are not runs by this aggregator. No native shaping/fit or pseudo-runtime observation is claimed.

## Root decisions before author dispatch

Generic human `Choice.label` is separate from raw `Choice.value`: source AST binds choice values and source usage per finding. Preserve commands, model identifiers/brands, URLs, code, flags, CLI/settings names and JSON prefixes; confirm exceptions per key, never blanket-exempt generic labels. Conflicting persona/product terminology exceptions remain source-attributed candidates in JSON, not settled approvals.

- `wire.settings.speech_voice_expressiveness.choice.medium.description`: Auditors identify mathematical constant/fixed-value or price interpretation versus standard vocal delivery; see per-locale quoted findings and context. Root chooses whether English source/context improvement is necessary before author dispatch. No English rewrite proposed here.
  - Existing English, quoted from official capture: Today's constant.. Root chooses any source/context improvement; no replacement English authored here.
- `wire.settings.monitor_classifymaxchars.help`: VI auditors cite monitors/classify.py bounded_state character slicing rather than line count; distinguish monitor.maxDeltaLines. Keep raw setting key/value unchanged.
- `wire.settings.tools_search_interception_rg_excludes.label`, `wire.settings.tools_search_interception_rg_excludes.help`: UR A/B explicitly flag the LABEL as implying filesystem deletion while paired help describes ripgrep exclusions; keep label/help consistent and preserve executable rg syntax. The brief’s shorthand rg_exclude.help is not a real key; normalized here only, originals unchanged. See source-attributed proposals.

### Verified source-key clarification

The request shorthand `rg_exclude.help` is not an actual catalogue key. Both Urdu auditors flag **`wire.settings.tools_search_interception_rg_excludes.label`** as implying deletion; the paired help describes exclusion. This correction affects only derived ledger metadata, never the original audit receipts.

## One follow-up authoring wave (not dispatched)

- **zh-CN**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **fr**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **es**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **ru**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **vi**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **hi**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.
- **ur**: fresh independent locale translator to be allocated by Root. Root-approved union of keyed findings for this locale; human labels/help and all cross-key references; no machine tokens. Independent of A/B sessions listed above.

Root settles the union/disagreements and any English/context decision first. Then one coordinated major-repair wave across seven locale authors; fold accepted minor/nit findings into it or defer with reasons. No minor-only wave. Authors supply per-key actual model/session, proposed text, raw-machine-value preservation, source usage/context reason and cross-key label/help consistency.

Regression contract: ICU parse/render, exact placeholder/selector/skeleton preservation, applicable CLDR probes and missing-argument negative controls; literal pseudo and Urdu RTL/isolation limits explicit. Character bounds must remain characters (VI monitor); voice medium must mean standard delivery, not fixed value/price; Urdu rg label must not imply deletion.

Independent A/B delta-only review follows at one head, covering repaired keys and direct neighbors. Retain unchanged entries only with actual per-key target/source/context compatibility proof across the 895-record matrix. Prior zero-key convergence or eight-voice checks are not approval for the other 893 keys.

## Readable override matrix — auditor quotations, NOT approved translations

One stable finding ID per source key groups locale proposals and the maximum reported severity. Each source pointer resolves into an exact raw receipt. `requiresAuthorRepair` is the proposed major-repair allocation pending Root approval; width-only rows must not be converted to observed clipping. No proposal here was authored by the aggregator.

### I18N-WS-39EE37F439D0 — `wire.settings.agents.description`

**Source English:** Whether installed starter roles (reviewer, coder, aida, ...) update themselves to the packaged text at launch when you have not edited them. Off = updates are still reported; apply them with 'lop agents sync'.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1085.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | minor / rootMinorFoldOrDefer | Si los roles iniciales instalados (reviewer, coder, aida, …) se actualizan solos al texto del paquete al arrancar cuando no los has editado. Desactivado = las actualizaciones se siguen notificando; aplícalas con 'lop agents sync'. | Si los roles de las plantillas incluidas que tienes instaladas (reviewer, coder, aida, …) se actualizan solos al texto del paquete al arrancar cuando no los has editado. Desactivado = las actualizaciones se siguen notificando; aplícalas con 'lop agents sync'. | Starter drifts from the Spanish guide’s recorded plantilla incluida to iniciales; preserve the embedded reviewer/coder/aida role identifiers. — `i18n/audits/evidence/approval-full-es-A-original.json#/minor_findings/0` (ES-A-m1) |

### I18N-WS-296272863BEE — `wire.settings.agents_auto_update_seeds.label`

**Source English:** Auto-update built-in roles

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5913.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Mise à jour auto des rôles intégrés | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/0` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Автообновление встроенных ролей | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/37` (RU-A-W1.01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | بلٹ اِن کردار خودکار اپ ڈیٹ کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/0` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Tự động cập nhật vai trò tích hợp sẵn | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/0` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Tự động cập nhật vai trò tích hợp sẵn | Tự cập nhật vai trò có sẵn | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/0` |

### I18N-WS-3D1D6D8016F3 — `wire.settings.aida.description`

**Source English:** Your chief of staff: her display name, and the proactive cadence — when she checks in, how much she may escalate, and the pause switch. The name applies everywhere at once; edits are read at her next action, and /aida pause\|resume act immediately.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1040.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | Votre cheffe de cabinet : son nom affiché et la cadence proactive — quand elle fait un point, combien elle peut escalader et l’interrupteur de pause. Le nom s’applique partout d’un coup ; les modifications sont lues à sa prochaine action, et /aida pause\|resume agissent immédiatement. | Votre cheffe de cabinet : son nom affiché et sa cadence proactive — quand elle fait un point, combien de points d’escalade elle peut programmer et la mise en pause. Le nom s’applique partout immédiatement ; les modifications sont lues à sa prochaine action, et /aida pause\|resume agissent immédiatement. | « combien elle peut escalader » is an opaque calque; the neighboring daily-limit setting shows that the quantity is escalation check-ins. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/26` (FR-A-m06) |

### I18N-WS-279E3003F300 — `wire.settings.aida_cadence_at.help`

**Source English:** Local wall-clock HH:MM. An invalid value falls back to 08:30 at the next arm.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5974.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | مقامی HH:MM وقت۔ غلط قیمت اگلی بار مقرر کرنے پر 08:30 ہو جائے گی۔ | مقامی HH:MM وقت۔ غلط قدر اگلی بار مقرر کرنے پر 08:30 ہو جائے گی۔ | قیمت is price, whereas these fields describe a configuration value; use قدر consistently. Comprehensible in context but a repeated false friend, not native technical Urdu. Deferred if no current major remediation touches these keys. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/1/entries/0` (UR-A8) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | مقامی HH:MM وقت۔ غلط قیمت اگلی بار مقرر کرنے پر 08:30 ہو جائے گی۔ | مقامی HH:MM وقت۔ غلط قدر اگلی بار مقرر کرنے پر 08:30 ہو جائے گی۔ | قیمت is price; the software setting is a value (قدر), consistent with قدریں elsewhere. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/6` (URB-m7) |
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 本地壁钟时间HH:MM。无效值会在下次调度时回退为08:30。 | 本地时间，格式为HH:MM。值无效时，下次安排定时任务会回退到08:30。 | 本地壁钟时间 is a literal calque; the sibling Librarian setting already uses the idiomatic 本地时间 for the same wall-clock concept. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/34` (T-ZH-A-035) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 本地壁钟时间HH:MM。无效值会在下次调度时回退为08:30。 | 本地时间，格式为HH:MM。无效值会在下次调度时回退为08:30。 | 壁钟时间 is a literal calque here; the equivalent librarian setting already uses the natural 本地时间. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/34` (B-ZH-m04) |

### I18N-WS-2DB39A525775 — `wire.settings.aida_cadence_at.label`

**Source English:** Daily check-in time

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5970.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Время ежедневного выхода на связь | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 19 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/38` (RU-A-W1.02) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Время ежедневного выхода на связь | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/0` (RU-B-m01) |

### I18N-WS-2C21C473B4BD — `wire.settings.aida_cadence_max_extra_per_day.label`

**Source English:** Extra check-ins / day

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6004.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Дополнительные выходы на связь в день | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 37 Russian cells versus 21 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/39` (RU-A-W1.03) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Дополнительные выходы на связь в день | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/1` (RU-B-m01) |

### I18N-WS-1F7A7740B499 — `wire.settings.aida_onboarding_nudge_days.label`

**Source English:** Integration nudge interval (days)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6026.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Intervalle de rappel d’intégration (jours) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/1` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Интервал напоминаний об интеграции (дни) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 40 Russian cells versus 33 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/40` (RU-A-W1.04) |

### I18N-WS-664B0980FB90 — `wire.settings.asks.title`

**Source English:** Asks

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1171.

**Context (existing sidecar):** {"role": "settings section title", "tone": "plain", "note": "Section heading for the queued-question settings. 'Ask' is the product noun for a question an agent queues for the user; keep it short and consistent with the ask bar/list vocabulary."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/0` |

### I18N-WS-A0AD63CFD554 — `wire.settings.asks_lapsed_visible_for.choice.at_expiry.label`

**Source English:** gone at expiry

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6692.

**Context (existing sidecar):** {"role": "dropdown choice label", "tone": "plain", "note": "The ask disappears the moment its deadline passes."}

**Unchanged raw Choice.value:** `at_expiry` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/1` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | исчезает по истечении срока | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/2` (RU-B-m01) |

### I18N-WS-FE38FA4D7D0E — `wire.settings.asks_lapsed_visible_for.label`

**Source English:** Show timed-out asks for

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6685.

**Context (existing sidecar):** {"role": "settings row label", "tone": "plain", "maxCells": 29, "note": "Row label; 29 cells is the settings page's label budget (_LABEL_BUDGET). 'Timed-out' = the ask's deadline passed with no answer. The row only changes how long such an ask stays SHOWN; it does not change whether the agent hears a late answer."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | minor / rootMinorFoldOrDefer | وقت ختم سوالات دکھائیں | مہلت کے بعد سوالات دکھائیں | وقت ختم سوالات is a broken noun phrase and drops the duration relationship; the source is a duration control, not an action that times out questions. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/5` (URB-m6) |

### I18N-WS-311989671F54 — `wire.settings.auto_save_conversation.label`

**Source English:** Auto-save conversation

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3413.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Enregistrement auto de la conversation | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/2` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/2` |

### I18N-WS-7625AFC4C316 — `wire.settings.bash_heavy_jobs_enabled.label`

**Source English:** Heavy job queue

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5188.

**Context (existing sidecar):** {"role": "settings row label", "tone": "plain", "note": "On/off switch for the queue. Same wording as the section title is intentional."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | File d’attente des tâches lourdes | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/3` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/3` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Очередь тяжёлых заданий | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/3` (RU-B-m01) |

### I18N-WS-BE27791626D8 — `wire.settings.bash_memory_enabled.label`

**Source English:** Command memory limit

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5130.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Limite de mémoire des commandes | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/4` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/4` |

### I18N-WS-4BCC4EE3F5D9 — `wire.settings.bash_memory_limit_mb.label`

**Source English:** Manual ceiling (MB)

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5164.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | manual सीमा (MB) | ख़ुद तय की गई सीमा (MB) | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/34` (H-A-MIX-35) |
| hi / B / semantic | major / requiresAuthorRepair | manual सीमा (MB) | मैन्युअल सीमा (MB) | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/2` (HI-B-M1) |

### I18N-WS-E6C2188A81F3 — `wire.settings.bash_memory_mode.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5150.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/0` (H-A-EN-1) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/0` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/0` (RU-A-M1.01) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/0` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/0` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/0` (URB-1) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/0` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/0` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/0` (T-ZH-A-001) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/0` (B-ZH-M01.01) |

### I18N-WS-802B8268724D — `wire.settings.bash_memory_mode.choice.manual.label`

**Source English:** manual

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5155.

**Unchanged raw Choice.value:** `manual` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | major / requiresAuthorRepair | manual | manuel | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/0` (FR-A-M01) |
| fr / B / semantic | major / requiresAuthorRepair | manual | manuel | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/0` (FR-B-M01) |
| hi / A / semantic | major / requiresAuthorRepair | manual | ख़ुद तय करें | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/1` (H-A-EN-2) |
| hi / B / semantic | major / requiresAuthorRepair | manual | मैन्युअल | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/1` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | manual | вручную | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/1` (RU-A-M1.02) |
| ru / B / semantic | major / requiresAuthorRepair | manual | вручную | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/9` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | manual | دستی | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/1` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | manual | دستی | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/1` (URB-2) |
| vi / A / semantic | major / requiresAuthorRepair | manual | Thủ công | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/1` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | manual | Thủ công | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/1` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | manual | 手动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/1` (T-ZH-A-002) |
| zh-CN / B / semantic | major / requiresAuthorRepair | manual | 手动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/1` (B-ZH-M01.02) |

### I18N-WS-07267F3C52E6 — `wire.settings.bash_query_budget_enabled.label`

**Source English:** Shell query budget

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5211.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Presupuesto de consultas de shell | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/0` |

### I18N-WS-F54DA19BED4A — `wire.settings.bash_query_budget_stop.label`

**Source English:** ...stop rather than warn

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5224.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …останавливать, а не предупреждать | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 34 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/41` (RU-A-W1.05) |

### I18N-WS-A07C361ED055 — `wire.settings.classification.title`

**Source English:** Smart hints

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 747.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/5` |

### I18N-WS-571879D3906A — `wire.settings.classification_auto.label`

**Source English:** Smart hints

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4008.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/6` |

### I18N-WS-2187A1F94D7F — `wire.settings.classification_maxstatechars.label`

**Source English:** ↳ max state chars

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4152.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ макс. символов состояния | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/4` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ حالت کے زیادہ سے زیادہ حروف | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/1` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ حالت کے زیادہ سے زیادہ حروف | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/0` |

### I18N-WS-532B7C97D288 — `wire.settings.classification_model.help`

**Source English:** Needs recommendations on. Vendor model id; empty uses its default.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4071.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | Nécessite les recommandations activées. ID de modèle du fournisseur ; vide utilise son défaut. | Nécessite les recommandations activées. ID de modèle du fournisseur ; vide : sa valeur par défaut. | « Son défaut » is a literal rendering of default that reads as a defect, not a default model. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/21` (FR-A-m01) |

### I18N-WS-24A7FCA10872 — `wire.settings.classification_notice.label`

**Source English:** Smart hints notice

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5368.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Aviso de sugerencias inteligentes | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/1` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Avis des suggestions intelligentes | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/5` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/7` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Уведомление «Умных подсказок» | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/5` (RU-B-m01) |

### I18N-WS-803C522C3F62 — `wire.settings.classification_vendor.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4043.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/2` (H-A-EN-3) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/3` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/2` (RU-A-M1.03) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/1` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/2` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/2` (URB-3) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/2` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/2` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/2` (T-ZH-A-003) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/2` (B-ZH-M01.03) |

### I18N-WS-3081EAF521B2 — `wire.settings.classification_vendor.choice.typesafe.description`

**Source English:** TypeSafe's Jev endpoint

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4054.

**Unchanged raw Choice.value:** `typesafe` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | TypeSafe का Jev endpoint | TypeSafe का Jev एंडपॉइंट | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/35` (H-A-MIX-36) |

### I18N-WS-EE9117EEE568 — `wire.settings.classification_vendor.label`

**Source English:** ↳ vendor

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4032.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/8` |

### I18N-WS-EB4BF2B582E8 — `wire.settings.compaction_idle_min_growth_tokens.label`

**Source English:** Idle min growth (tokens)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4628.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Croissance min. en veille (tokens) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/6` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Мин. прирост в простое (токены) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/42` (RU-A-W1.06) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | فارغ سیشن کم از کم اضافہ (ٹوکنز) | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/2` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Mức tăng tối thiểu khi rảnh (token) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/1` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Mức tăng tối thiểu khi rảnh (token) | Tăng tối thiểu lúc rảnh (token) | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/1` |

### I18N-WS-6438A826F00E — `wire.settings.compaction_keep_recent_tokens.label`

**Source English:** Keep recent tokens

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4565.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/9` |

### I18N-WS-D2B39ABCACB2 — `wire.settings.compaction_resume_replay.choice.card.description`

**Source English:** summary, a short resume card and a recent tail; earlier turns not verbatim

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4656.

**Unchanged raw Choice.value:** `card` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 摘要、简短的恢复卡片和最近的尾部；较早轮次不逐字保留 | 摘要、简短的恢复卡片和最近的记录；较早回合不逐字保留 | 轮次 drifts from 回合 used for the same English turn throughout nearby settings, and 最近的尾部 is a literal tail calque. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/37` (T-ZH-A-038) |

### I18N-WS-3709651CD4F2 — `wire.settings.compaction_resume_replay.choice.full.description`

**Source English:** summary, every preserved user turn and the kept window

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4651.

**Unchanged raw Choice.value:** `full` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 摘要、每个保留的用户轮次和保留窗口 | 摘要、每个保留的用户回合和保留窗口 | The same English turn is rendered 轮次 here but 回合 in the surrounding compaction controls; use the established term. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/38` (T-ZH-A-039) |

### I18N-WS-095C42DE4A3A — `wire.settings.compaction_resume_replay.help`

**Source English:** What a resumed session replays after a compaction. Card is far smaller, but earlier operator turns no longer survive verbatim, and switching back to full does not restore them.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4646.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 压缩后恢复的会话重放哪些内容。卡片小得多，但操作员较早的轮次不再逐字保留，切换回完整也无法恢复。 | 压缩后恢复的会话重放哪些内容。卡片小得多，但操作员较早的回合不再逐字保留，切换回完整也无法恢复。 | The English turn is rendered 轮次 here but 回合 in the surrounding compaction controls; this is glossary drift, not a different concept. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/39` (T-ZH-A-040) |

### I18N-WS-89A6F9B47F91 — `wire.settings.compaction_resume_replay.label`

**Source English:** Resume replay

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4643.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Воспроизведение при возобновлении | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 13 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/43` (RU-A-W1.07) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Воспроизведение при возобновлении | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/6` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | دوبارہ شروع پر ری پلے | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/3` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | دوبارہ شروع پر ری پلے | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/1` |

### I18N-WS-240C498CA1C1 — `wire.settings.compaction_strategy.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4517.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/3` (H-A-EN-4) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/4` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/3` (RU-A-M1.04) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/2` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/3` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/3` (URB-4) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/3` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/3` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/3` (T-ZH-A-004) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/3` (B-ZH-M01.04) |

### I18N-WS-0AC576F1D860 — `wire.settings.compaction_strategy.choice.off.label`

**Source English:** off

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4532.

**Unchanged raw Choice.value:** `off` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/10` |

### I18N-WS-EB54111AF83B — `wire.settings.compaction_threshold_tokens.help`

**Source English:** Absolute trigger. The smaller of this and the percentage wins.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4558.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | مطلق محرک۔ اس اور فیصد میں سے چھوٹی قیمت لاگو ہوگی۔ | مطلق محرک۔ اس اور فیصد میں سے چھوٹی قدر لاگو ہوگی۔ | قیمت is price, whereas these fields describe a configuration value; use قدر consistently. Comprehensible in context but a repeated false friend, not native technical Urdu. Deferred if no current major remediation touches these keys. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/1/entries/2` (UR-A8) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | مطلق محرک۔ اس اور فیصد میں سے چھوٹی قیمت لاگو ہوگی۔ | مطلق محرک۔ اس اور فیصد میں سے چھوٹی قدر لاگو ہوگی۔ | قیمت is price; the software setting is a value (قدر), consistent with قدریں elsewhere. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/8` (URB-m9) |

### I18N-WS-40637B7E415B — `wire.settings.compaction_wire_bytes_budget.label`

**Source English:** Request size limit (bytes)

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4671.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Limite de taille de requête (octets) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/7` |
| ur / B / semantic | major / requiresAuthorRepair | درخواست کی حد (bytes) | درخواست کی حد (بائٹس) | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/41` (URB-42) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Giới hạn kích thước yêu cầu (byte) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/2` |

### I18N-WS-00E2E1DEB31E — `wire.settings.compaction_wire_bytes_trigger.label`

**Source English:** Request size trigger (bytes)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4681.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Déclencheur de taille de requête (octets) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/8` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Триггер размера запроса (байты) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 28 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/44` (RU-A-W1.08) |

### I18N-WS-749FFC00E67E — `wire.settings.desktop.title`

**Source English:** Desktop app

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 941.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/11` |

### I18N-WS-949E75329B5D — `wire.settings.desktop_launch_command.label`

**Source English:** launch command

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5424.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/12` |

### I18N-WS-B232E231E57E — `wire.settings.desktop_remote_files_cache_bytes.label`

**Source English:** remote files cache (bytes)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5486.

**Context (existing sidecar):** {"role": "settings row label in the desktop section", "tone": "terse, lowercase like its neighbours", "width": "short label column", "note": "'bytes' is the unit of the number the user types"}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | cache des fichiers distants (octets) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/9` |

### I18N-WS-6B35D4228508 — `wire.settings.detail_length.label`

**Source English:** Detail length

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5350.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / semantic | minor / rootMinorFoldOrDefer | Длина детализации | Объём подробностей | «Длина детализации» is a literal and unnatural rendering of the amount of detail. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/34` (RU-A-m2) |

### I18N-WS-3D8DF8BB0DF1 — `wire.settings.display_composer_chevron.label`

**Source English:** Prompt chevron

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3171.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | Chevron del prompt | Indicador del prompt | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/24` (ES-A-M1.25) |

### I18N-WS-414A80100D09 — `wire.settings.display_composer_cwd.label`

**Source English:** Band: working dir

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3197.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Bandeau : répertoire de travail | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/10` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/13` |

### I18N-WS-AED764C5B7FC — `wire.settings.display_composer_duration.label`

**Source English:** Band: elapsed

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3249.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/14` |

### I18N-WS-99016D9A5E02 — `wire.settings.display_composer_rate.label`

**Source English:** Band: decode rate

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3223.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Banda: velocidad de decodificación | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/2` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/15` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Полоса: скорость декодирования | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 30 Russian cells versus 17 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/45` (RU-A-W1.09) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Полоса: скорость декодирования | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/7` (RU-B-m01) |

### I18N-WS-8610F62B6E9A — `wire.settings.display_dock.label`

**Source English:** Subagent dock

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3051.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/16` |

### I18N-WS-1F3A1DF82382 — `wire.settings.display_hide_cross_session.label`

**Source English:** Hide cross-session traffic

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3103.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Masquer le trafic entre sessions | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/11` |

### I18N-WS-A4E16B704FE0 — `wire.settings.display_images.label`

**Source English:** Inline images

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2865.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Встроенные изображения | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/8` (RU-B-m01) |

### I18N-WS-3FA1FF925DC5 — `wire.settings.display_nerd_icons.choice.false.label`

**Source English:** off

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2830.

**Unchanged raw Choice.value:** `false` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/17` |

### I18N-WS-CDE402D4ABB5 — `wire.settings.display_nerd_icons.choice.none.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2820.

**Unchanged raw Choice.value:** `—` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/4` (H-A-EN-5) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/5` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/4` (RU-A-M1.05) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/3` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/4` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/4` (URB-5) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/4` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/4` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/4` (T-ZH-A-005) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/4` (B-ZH-M01.05) |

### I18N-WS-FBC1A7C09726 — `wire.settings.display_notification_session_name.help`

**Source English:** A session's name, last line and error causes appear on banners, including the lock screen and paired phones' notifications, which reach them through Radient's push service.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2919.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | major / requiresAuthorRepair | بینرز پر سیشن کا نام، آخری سطر اور خرابی کی وجوہات آتی ہیں، لاک اسکرین اور جوڑے ہوئے فونز کی اطلاعات بھی شامل، جو ان تک Radient کی push سروس کے ذریعے پہنچتی ہیں۔ | بینرز پر سیشن کا نام، آخری سطر اور خرابی کی وجوہات آتی ہیں، لاک اسکرین اور جوڑے ہوئے فونز کی اطلاعات بھی شامل، جو ان تک Radient کی پُش سروس کے ذریعے پہنچتی ہیں۔ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/9` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | بینرز پر سیشن کا نام، آخری سطر اور خرابی کی وجوہات آتی ہیں، لاک اسکرین اور جوڑے ہوئے فونز کی اطلاعات بھی شامل، جو ان تک Radient کی push سروس کے ذریعے پہنچتی ہیں۔ | بینرز پر سیشن کا نام، آخری سطر اور خرابی کی وجوہات آتی ہیں، لاک اسکرین اور جوڑے ہوئے فونز کی اطلاعات بھی شامل، جو ان تک Radient کی پُش سروس کے ذریعے پہنچتی ہیں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/39` (URB-40) |

### I18N-WS-C217162BE055 — `wire.settings.display_notification_session_name.label`

**Source English:** Notification session names

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2903.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Noms de session dans les notifications | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/12` |

### I18N-WS-63084A2AECAA — `wire.settings.display_rail.help`

**Source English:** A rule marks the ANSWER: the message that ends the turn. Mid-turn narration takes no rail.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2784.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | major / requiresAuthorRepair | ایک لکیر ANSWER کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔ | ایک لکیر جواب کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/0` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | ایک لکیر ANSWER کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔ | ایک لکیر جواب کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/32` (URB-33) |

### I18N-WS-249532749C3B — `wire.settings.display_reasoning.label`

**Source English:** Live model reasoning

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2757.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Razonamiento del modelo en vivo | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/3` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Raisonnement du modèle en direct | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/13` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/18` |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Suy luận trực tiếp của mô hình | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/3` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Suy luận trực tiếp của mô hình | Suy luận trực tiếp | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/2` |

### I18N-WS-CC7F3B618646 — `wire.settings.display_time_format.label`

**Source English:** Wake time format

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2944.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Formato de la hora de despertar | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/4` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/19` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Формат времени пробуждений | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/9` (RU-B-m01) |

### I18N-WS-3CEF58FC37D2 — `wire.settings.fork.title`

**Source English:** Fork

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 811.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / semantic | major / requiresAuthorRepair | Fork | Tách nhánh | This is a visible section heading for branching a conversation, not the /fork command token. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/31` (VI-A-M2) |
| vi / B / semantic | major / requiresAuthorRepair | Fork | Tách nhánh | A human settings-section title, not the literal /fork command. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/31` (VI-B-M1) |

### I18N-WS-93EB04D1BC4D — `wire.settings.fork_cmux_placement.choice.surface.label`

**Source English:** new surface

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4487.

**Unchanged raw Choice.value:** `surface` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | major / requiresAuthorRepair | surface mới | thẻ mới | The English source description identifies an actual UI tab; the human label surface mới leaves an English UI noun. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/36` (VI-B-M1) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 新surface | 新标签页 | The sibling description explicitly identifies a tab in the current workspace; 新surface is an avoidable hybrid rather than an idiomatic label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/32` (B-ZH-m02) |

### I18N-WS-A5BB3F95918E — `wire.settings.fork_cmux_placement.choice.workspace.label`

**Source English:** new workspace

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4482.

**Unchanged raw Choice.value:** `workspace` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/20` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | новое рабочее пространство | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/10` (RU-B-m01) |

### I18N-WS-30736341A5D8 — `wire.settings.fork_cmux_placement.help`

**Source English:** Under cmux, whether a fork gets its own workspace or a surface here.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4478.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | major / requiresAuthorRepair | Dưới cmux, liệu nhánh mới có không gian làm việc riêng hay một surface ở đây. | Trong cmux, nhánh mới có không gian làm việc riêng hay một thẻ ở đây. | Translate the human surface noun consistently with its tab choice while retaining the actual brand cmux. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/37` (VI-B-M1) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 在cmux下，分叉是获得自己的新工作区，还是在此处作为一个surface打开。 | 在cmux中，分叉是在独立工作区中打开，还是在当前工作区的标签页中打开。 | As confirmed by the adjacent choice descriptions, surface denotes the current workspace’s tab; retaining this generic English noun obscures the choice. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/33` (B-ZH-m03) |

### I18N-WS-CBCF051AC74B — `wire.settings.fork_mode.choice.window.label`

**Source English:** new window

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4457.

**Unchanged raw Choice.value:** `window` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/21` |

### I18N-WS-351AA399D661 — `wire.settings.guardrails_rules_dir.label`

**Source English:** Rules folder

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` no direct static getter call identified.

**Context (existing sidecar):** {"maxCells": 29, "note": "Label fits the 29-cell label budget. No ICU arguments.", "role": "Settings row label", "tone": "concise"}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/22` |

### I18N-WS-0E0BE18A4F22 — `wire.settings.heavy_jobs.title`

**Source English:** Heavy job queue

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 898.

**Context (existing sidecar):** {"role": "settings section title", "tone": "plain", "note": "Section heading for the machine-wide queue that limits how many heavy builds/tests run at once (one at a time on machines with 16 GB or less, up to four on larger ones). 'Heavy job' = a build, type check or test run that uses a lot of memory."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/23` |

### I18N-WS-456B6DBE92C6 — `wire.settings.hooks.description`

**Source English:** Run native and forwarded Claude Code / Codex PostToolUse hooks.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 876.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | minor / rootMinorFoldOrDefer | Chạy các hook PostToolUse bản địa và chuyển tiếp của Claude Code / Codex. | Chạy các hook riêng của lop và hook PostToolUse được chuyển tiếp từ Claude Code / Codex. | Native hooks means implemented by this harness, not bản địa (indigenous); the literal rendering is an avoidable false friend. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/2` (VI-B-m3) |

### I18N-WS-954584654E1D — `wire.settings.hooks_forward_claude.label`

**Source English:** Forward Claude Code hooks

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4972.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Transférer les hooks de Claude Code | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/14` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Перенаправлять хуки Claude Code | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 25 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/46` (RU-A-W1.10) |

### I18N-WS-0BA626964DF5 — `wire.settings.hooks_forward_codex.label`

**Source English:** Forward Codex hooks

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4981.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/24` |

### I18N-WS-18A01226F174 — `wire.settings.hooks_native.label`

**Source English:** Native hooks

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4963.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | minor / rootMinorFoldOrDefer | Hook bản địa | Hook riêng của lop | The same native→bản địa false friend occurs in the row label; it should distinguish harness-owned hooks from forwarded ones. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/3` (VI-B-m4) |

### I18N-WS-7BC71BEE5100 — `wire.settings.hub.description`

**Source English:** Keep agents and teams you pulled from the Agent Hub current: whether updates are merged in automatically, how often the hub is checked, and which model resolves a conflict. Your edits and deletions are merged, never overwritten.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 1061.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | Agent Hub से pull किए गए एजेंट और टीमें अपडेट रहें: अपडेट अपने-आप merge होते हैं या नहीं, हब कितनी बार जाँचा जाता है, और टकराव कौन-सा मॉडल सुलझाता है। आपने जो संपादित या हटाया, वह merge हो जाता है, कभी ओवरराइट नहीं। | Agent Hub से लिए गए एजेंट और टीमों को अपडेट रखें: अपडेट अपने-आप मिलते हैं या नहीं, हब कितनी बार जाँचा जाता है, और टकराव कौन-सा मॉडल सुलझाता है। आपके संपादन और हटाए गए हिस्सों को मिलाया जाता है, कभी ओवरराइट नहीं किया जाता। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/48` (H-A-MIX-49) |
| ur / B / semantic | major / requiresAuthorRepair | Agent Hub سے حاصل کردہ ایجنٹس اور ٹیمیں تازہ رکھیں: اپ ڈیٹس خودکار merge ہوں، hub کتنی بار جانچ ہو، اور تضاد کون سا ماڈل حل کرے۔ آپ کی تبدیلیاں اور حذف کردہ مواد merge ہوتا ہے، اووررائیٹ نہیں۔ | Agent Hub سے حاصل کردہ ایجنٹس اور ٹیمیں تازہ رکھیں: اپ ڈیٹس خودکار ضم ہوں، hub کتنی بار جانچ ہو، اور تضاد کون سا ماڈل حل کرے۔ آپ کی تبدیلیاں اور حذف کردہ مواد ضم ہوتا ہے، اووررائیٹ نہیں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/33` (URB-34) |

### I18N-WS-E4736E35DC92 — `wire.settings.hub_auto_update_agents.choice.true.description`

**Source English:** merge hub updates automatically

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` no direct static getter call identified.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | हब अपडेट अपने-आप merge करें | हब अपडेट अपने-आप मिलाएँ | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/46` (H-A-MIX-47) |
| ur / B / semantic | major / requiresAuthorRepair | hub اپ ڈیٹس خودکار merge کریں | hub اپ ڈیٹس خودکار ضم کریں | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/34` (URB-35) |

### I18N-WS-E649D656620C — `wire.settings.hub_auto_update_agents.help`

**Source English:** Merge hub updates into pulled agents automatically.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` no direct static getter call identified.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | हब अपडेट pull किए गए एजेंट में अपने-आप merge करें। | हब के अपडेट, हब से लिए गए एजेंटों में अपने-आप मिलाएँ। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/49` (H-A-MIX-50) |
| ur / B / semantic | major / requiresAuthorRepair | حاصل کردہ ایجنٹس میں hub اپ ڈیٹس خودکار merge کریں۔ | حاصل کردہ ایجنٹس میں hub اپ ڈیٹس خودکار ضم کریں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/35` (URB-36) |

### I18N-WS-303E9618097F — `wire.settings.hub_auto_update_agents.label`

**Source English:** Auto-update agents

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` no direct static getter call identified.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/25` |

### I18N-WS-89000E708CCA — `wire.settings.hub_auto_update_teams.choice.true.description`

**Source English:** merge hub updates automatically

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5864.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | हब अपडेट अपने-आप merge करें | हब अपडेट अपने-आप मिलाएँ | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/47` (H-A-MIX-48) |
| ur / B / semantic | major / requiresAuthorRepair | hub اپ ڈیٹس خودکار merge کریں | hub اپ ڈیٹس خودکار ضم کریں | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/36` (URB-37) |

### I18N-WS-B46B76077316 — `wire.settings.hub_auto_update_teams.help`

**Source English:** Merge hub updates into pulled teams automatically.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5867.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | हब अपडेट pull की गई टीमों में अपने-आप merge करें। | हब के अपडेट, हब से ली गई टीमों में अपने-आप मिलाएँ। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/50` (H-A-MIX-51) |
| ur / B / semantic | major / requiresAuthorRepair | حاصل کردہ ٹیموں میں hub اپ ڈیٹس خودکار merge کریں۔ | حاصل کردہ ٹیموں میں hub اپ ڈیٹس خودکار ضم کریں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/37` (URB-38) |

### I18N-WS-91106EA3177B — `wire.settings.hub_auto_update_teams.label`

**Source English:** Auto-update teams

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5860.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/26` |

### I18N-WS-F7218992037C — `wire.settings.hub_check_interval_min.label`

**Source English:** Check interval (minutes)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5873.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Intervalle de vérification (minutes) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/15` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/27` |

### I18N-WS-AE1F05151021 — `wire.settings.keymap.title`

**Source English:** Hotkeys

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 606.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/28` |

### I18N-WS-521BD8E7974A — `wire.settings.librarian.description`

**Source English:** The Librarian keeps the global Library organised: her display name, the daily pass, and the pause switch. The Library is long-term and shared by every conversation.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 1050.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | Le Bibliothécaire garde la Bibliothèque globale organisée : son nom affiché, la passe quotidienne et la pause. La Bibliothèque est durable et partagée par toutes les conversations. | La Bibliothécaire organise la Bibliothèque globale : son nom affiché, le passage quotidien et la mise en pause. La Bibliothèque est durable et partagée par toutes les conversations. | « garde … organisée » and « la passe quotidienne » read translated, and masculine « Le Bibliothécaire » conflicts with the neighboring feminine « elle ». — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/27` (FR-A-m07) |
| hi / B / semantic | minor / rootMinorFoldOrDefer | लाइब्रेरियन वैश्विक लाइब्रेरी को व्यवस्थित रखता है: उसका प्रदर्शित नाम, दैनिक पास और पॉज़ स्विच। लाइब्रेरी दीर्घकालिक है और हर बातचीत में साझा है। | लाइब्रेरियन वैश्विक लाइब्रेरी को व्यवस्थित रखती हैं: उनका प्रदर्शित नाम, रोज़ की जाँच और पॉज़ स्विच। लाइब्रेरी दीर्घकालिक है और हर बातचीत में साझा है। | Source explicitly describes a female assistant, whereas रखता है/उसका and sibling माँग सकता है use masculine forms; use the same polite feminine register as the Aida section. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/35` (HI-B-m1) |
| ur / A / semantic | major / requiresAuthorRepair | Librarian عالمی Library منظم رکھتی ہے: اس کا ظاہری نام، روزانہ پاس، اور روکنے کا اختیار۔ Library طویل مدت کی ہے اور ہر گفتگو میں مشترک ہے۔ | لائبریرین عالمی لائبریری منظم رکھتی ہے: اس کا ظاہری نام، روزانہ پاس، اور روکنے کا اختیار۔ لائبریری طویل مدت کی ہے اور ہر گفتگو میں مشترک ہے۔ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/2` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | Librarian عالمی Library منظم رکھتی ہے: اس کا ظاہری نام، روزانہ پاس، اور روکنے کا اختیار۔ Library طویل مدت کی ہے اور ہر گفتگو میں مشترک ہے۔ | Librarian عالمی لائبریری کو منظم رکھتی ہے: اس کا ظاہری نام، روزانہ جائزہ، اور روکنے کا اختیار۔ لائبریری طویل مدت کے لیے ہے اور ہر گفتگو میں مشترک ہے۔ | The capitalized Library denotes the shared long-term file collection, not a command/API/model; no brand-only exception is provided. Librarian is retained here only as the persona name. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/49` (URB-50) |

### I18N-WS-8316D70230B6 — `wire.settings.librarian.title`

**Source English:** Librarian

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 1048.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/29` |
| ur / A / semantic | major / requiresAuthorRepair | Librarian | لائبریرین | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/3` (UR-A2) |

### I18N-WS-E86504BA4A29 — `wire.settings.librarian_cadence_at.help`

**Source English:** Local wall-clock HH:MM. An invalid value falls back to 03:30 at the next arm.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6069.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | minor / rootMinorFoldOrDefer | Hora local HH:MM. Un valor no válido vuelve a 03:30 en el próximo armado. | Hora local HH:MM. Un valor no válido vuelve a 03:30 en la próxima programación. | Próximo armado is an unnatural literal rendering of scheduler arming; aida_cadence_at.help already uses programación for the same operation. — `i18n/audits/evidence/approval-full-es-A-original.json#/minor_findings/1` (ES-A-m2) |
| es / B / semantic | minor / rootMinorFoldOrDefer | Hora local HH:MM. Un valor no válido vuelve a 03:30 en el próximo armado. | Hora local HH:MM. Un valor no válido vuelve a 03:30 en la siguiente programación. | Armado is a scheduling calque; sibling aida_cadence_at.help uses the native programación for the same operation. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/28` (ES-B-29) |
| ur / A / semantic | minor / rootMinorFoldOrDefer | مقامی HH:MM وقت۔ غلط قیمت اگلی بار مقرر کرنے پر 03:30 ہو جائے گی۔ | مقامی HH:MM وقت۔ غلط قدر اگلی بار مقرر کرنے پر 03:30 ہو جائے گی۔ | قیمت is price, whereas these fields describe a configuration value; use قدر consistently. Comprehensible in context but a repeated false friend, not native technical Urdu. Deferred if no current major remediation touches these keys. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/1/entries/1` (UR-A8) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | مقامی HH:MM وقت۔ غلط قیمت اگلی بار مقرر کرنے پر 03:30 ہو جائے گی۔ | مقامی HH:MM وقت۔ غلط قدر اگلی بار مقرر کرنے پر 03:30 ہو جائے گی۔ | قیمت is price; the software setting is a value (قدر), consistent with قدریں elsewhere. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/7` (URB-m8) |

### I18N-WS-90FC7CE145F2 — `wire.settings.librarian_cadence_at.label`

**Source English:** Daily pass time

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6065.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/30` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Время ежедневного прохода | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/11` (RU-B-m01) |

### I18N-WS-01B7C6EA667A — `wire.settings.librarian_cadence_max_extra_per_day.help`

**Source English:** How many extra passes she may request per day. 0 disables extra passes.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6093.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | minor / rootMinorFoldOrDefer | वह प्रति दिन कितने अतिरिक्त पास माँग सकता है। 0 अतिरिक्त पास बंद करता है। | वे प्रति दिन कितने अतिरिक्त पास माँग सकती हैं। 0 अतिरिक्त पास बंद करता है। | English consistently refers to the Librarian as she; current वह … सकता है switches her gender and register. Neighboring librarian.description, librarian_cadence_min_gap_minutes.help, librarian_cadence_paused.help, librarian_enabled.help and librarian_name.help also use masculine/non-honorific references; normalize together, deferred rather than a new minor-only wave. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/54` (H-A-MIN-1) |

### I18N-WS-E110BC123522 — `wire.settings.librarian_cadence_max_extra_per_day.label`

**Source English:** Extra passes / day

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6088.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/31` |

### I18N-WS-3E5CEB1F650A — `wire.settings.librarian_cadence_paused.help`

**Source English:** Pausing removes her scheduled rows; resuming re-arms the next pass. The desktop and /librarian write the same key.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6082.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | minor / rootMinorFoldOrDefer | Pausar elimina sus filas programadas; reanudar arma la siguiente pasada. El escritorio y /librarian escriben la misma clave. | Pausar elimina sus filas programadas; reanudar programa la siguiente pasada. El escritorio y /librarian escriben la misma clave. | Arma la siguiente pasada is an avoidable scheduling calque; programa is natural and matches the existing programming vocabulary. — `i18n/audits/evidence/approval-full-es-A-original.json#/minor_findings/2` (ES-A-m3) |
| es / B / semantic | minor / rootMinorFoldOrDefer | Pausar elimina sus filas programadas; reanudar arma la siguiente pasada. El escritorio y /librarian escriben la misma clave. | Pausar elimina sus filas programadas; reanudar programa la siguiente pasada. El escritorio y /librarian escriben la misma clave. | Arma la siguiente pasada is unnatural scheduling language and drifts from the sibling programar terminology. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/29` (ES-B-30) |
| ur / A / semantic | major / requiresAuthorRepair | روکنے پر شیڈول شدہ قطاریں ہٹتی ہیں؛ resume اگلا پاس مقرر کرتا ہے۔ ڈیسک ٹاپ اور /librarian ایک ہی کلید لکھتے ہیں۔ | روکنے پر شیڈول شدہ قطاریں ہٹتی ہیں؛ دوبارہ شروع کرنے پر اگلا پاس مقرر کرتا ہے۔ ڈیسک ٹاپ اور /librarian ایک ہی کلید لکھتے ہیں۔ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/5` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | روکنے پر شیڈول شدہ قطاریں ہٹتی ہیں؛ resume اگلا پاس مقرر کرتا ہے۔ ڈیسک ٹاپ اور /librarian ایک ہی کلید لکھتے ہیں۔ | روکنے پر شیڈول شدہ قطاریں ہٹتی ہیں؛ دوبارہ شروع کرنے پر اگلا پاس مقرر کرتا ہے۔ ڈیسک ٹاپ اور /librarian ایک ہی کلید لکھتے ہیں۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/38` (URB-39) |

### I18N-WS-F33E8BCFD4E0 — `wire.settings.librarian_cadence_paused.label`

**Source English:** Daily pass paused

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6075.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/32` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ежедневный проход на паузе | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/12` (RU-B-m01) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Lượt chạy hằng ngày đã tạm dừng | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/4` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Lượt chạy hằng ngày đã tạm dừng | Tạm dừng lượt hằng ngày | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/3` |

### I18N-WS-83E999228FD1 — `wire.settings.librarian_enabled.choice.false.description`

**Source English:** disabled

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6046.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | désactivé | désactivée | Use the same feminine referent as the neighboring help text’s « elle ». — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/29` (FR-A-m09) |

### I18N-WS-A6DCDC5AEC28 — `wire.settings.librarian_enabled.choice.true.description`

**Source English:** enabled

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6045.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | activé | activée | Use the same feminine referent as the neighboring help text’s « elle ». — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/30` (FR-A-m10) |

### I18N-WS-9D18DD153837 — `wire.settings.librarian_enabled.label`

**Source English:** Librarian enabled

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6041.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | Bibliothécaire activé | Bibliothécaire activée | Use the same feminine referent as the neighboring help text’s « elle » rather than alternating masculine and feminine. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/28` (FR-A-m08) |
| ur / A / semantic | major / requiresAuthorRepair | Librarian فعال | لائبریرین فعال | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/4` (UR-A2) |

### I18N-WS-578CE3D533FF — `wire.settings.local_providers.base_url.help`

**Source English:** HTTP(S) API root. Changing servers requires a new token through /login; existing tokens are never forwarded to another endpoint.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5318.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | HTTP(S) API रूट। सर्वर बदलने पर /login से नया टोकन लेना पड़ता है; मौजूदा टोकन कभी दूसरे endpoint पर नहीं भेजे जाते। | HTTP(S) API रूट। सर्वर बदलने पर /login से नया टोकन लेना पड़ता है; मौजूदा टोकन कभी दूसरे एंडपॉइंट पर नहीं भेजे जाते। | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/36` (H-A-MIX-37) |

### I18N-WS-5FAF63734560 — `wire.settings.local_providers.base_url.label`

**Source English:** {name} endpoint

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5315.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | {name} endpoint | {name} एंडपॉइंट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/5` (H-A-EN-6) |

### I18N-WS-DA5C5E873CD2 — `wire.settings.local_providers.description`

**Source English:** Server endpoints and exact-model metadata. Use /login to connect; reselect with /model saved to apply model changes.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 930.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | सर्वर endpoint और सटीक मॉडल मेटाडेटा। जोड़ने के लिए /login इस्तेमाल करें; मॉडल बदलाव लागू करने के लिए /model saved से दोबारा चुनें। | सर्वर एंडपॉइंट और सटीक मॉडल मेटाडेटा। जोड़ने के लिए /login इस्तेमाल करें; मॉडल बदलाव लागू करने के लिए /model saved से दोबारा चुनें। | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/37` (H-A-MIX-38) |

### I18N-WS-11E90AA0F2DA — `wire.settings.local_providers.models.label`

**Source English:** {name} model overrides

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5325.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | surcharges de modèle pour {name} | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/16` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | переопределения моделей для {name} | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 34 Russian cells versus 22 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/47` (RU-A-W1.11) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | переопределения моделей для {name} | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/13` (RU-B-m01) |

### I18N-WS-23911E1BE68A — `wire.settings.max_learnings_history.label`

**Source English:** Max learnings history

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5359.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Historique max d’apprentissages | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/17` |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | سیکھے ہوئے نکات کی زیادہ سے زیادہ تاریخچہ | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/4` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | سیکھے ہوئے نکات کی زیادہ سے زیادہ تاریخچہ | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/2` |

### I18N-WS-E62D2804835F — `wire.settings.memory_guard.title`

**Source English:** Command memory limit

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 887.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/33` |

### I18N-WS-80CE4493BFC5 — `wire.settings.mesh_move_working_set_max_bytes.help`

**Source English:** The most working-file data (bundled commits plus staged, unstaged and untracked files) one move sends to, or accepts from, another device. Over it, the move is refused and nothing changes. Default 256 MiB.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5712.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | major / requiresAuthorRepair | ورکنگ فائل ڈیٹا (بنڈل شدہ commits کے ساتھ staged، unstaged اور untracked فائلیں) کی زیادہ سے زیادہ مقدار جو ایک منتقلی دوسری ڈیوائس کو بھیجتی ہے یا اس سے قبول کرتی ہے۔ اس سے زیادہ ہونے پر منتقلی مسترد ہو جاتی ہے اور کچھ نہیں بدلتا۔ ڈیفالٹ 256 MiB۔ | ورکنگ فائلوں کے ڈیٹا (بنڈل شدہ کمٹس اور اسٹیج کی گئی، اسٹیج نہ کی گئی اور گِٹ میں شامل نہ کی گئی فائلیں) کی زیادہ سے زیادہ مقدار جو ایک منتقلی دوسری ڈیوائس کو بھیج یا اس سے قبول کر سکتی ہے۔ اس حد سے زیادہ ہونے پر منتقلی مسترد ہو جاتی ہے اور کچھ نہیں بدلتا۔ ڈیفالٹ 256 MiB۔ | commits, staged, unstaged and untracked are human descriptions of file states here, not literal command arguments or JSON keys. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/50` (URB-51) |

### I18N-WS-502F92C0DCA9 — `wire.settings.mesh_move_working_set_max_bytes.label`

**Source English:** Working-files cap per move (bytes)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5707.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Límite de archivos de trabajo por movimiento (bytes) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/25` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Plafond des fichiers de travail par déplacement (octets) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/18` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/71` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Лимит рабочих файлов на перенос (байт) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 38 Russian cells versus 34 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/79` (RU-A-W1.43) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | فی منتقلی ورکنگ فائلوں کی حد (بائٹس) | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/5` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Giới hạn tệp làm việc mỗi lần di chuyển (byte) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/26` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Giới hạn tệp làm việc mỗi lần di chuyển (byte) | Giới hạn tệp mỗi lần (byte) | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/4` |
| zh-CN / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | 每次迁移的工作文件上限（字节） | 每次迁移文件上限（字节） | Current text measures 30 terminal cells, beyond the shared 29-cell settings-label budget cited by this sidecar; a rendered fit check is still needed. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/40` (B-ZH-m10) |

### I18N-WS-69F2A85DFCD2 — `wire.settings.mesh_move_working_set_roots.help`

**Source English:** Where on this device a session move may create or restore a repository: folders under your home, written with ~ and separated by commas (for example ~/code, ~/work). Hidden, system, command (on PATH) and node_modules folders are never used. Unset: a repository is restored where it already is, and a missing one is cloned only into a folder that already holds your repositories; set this to allow other folders.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5728.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | major / requiresAuthorRepair | اس ڈیوائس پر سیشن کی منتقلی کہاں ریپوزٹری بنا یا بحال کر سکتی ہے: آپ کے ہوم کے اندر فولڈر، ~ کے ساتھ لکھے اور کاموں سے الگ کیے گئے (مثلاً ~/code, ~/work)۔ پوشیدہ، سسٹم، کمانڈ (PATH پر) اور node_modules فولڈر کبھی استعمال نہیں ہوتے۔ سیٹ نہ ہو تو: ریپوزٹری وہیں بحال ہوتی ہے جہاں وہ پہلے سے ہے، اور غائب ریپوزٹری صرف ایسے فولڈر میں کلون ہوتی ہے جس میں پہلے سے آپ کی ریپوزٹریز ہوں؛ دوسرے فولڈروں کی اجازت کے لیے اسے سیٹ کریں۔ | اس ڈیوائس پر سیشن کی منتقلی کہاں ریپوزٹری بنا یا بحال کر سکتی ہے: آپ کے ہوم کے اندر فولڈر، ~ کے ساتھ لکھے اور کاما سے الگ کیے گئے (مثلاً ~/code, ~/work)۔ پوشیدہ، سسٹم، کمانڈ (PATH پر) اور node_modules فولڈر کبھی استعمال نہیں ہوتے۔ سیٹ نہ ہو تو: ریپوزٹری وہیں بحال ہوتی ہے جہاں وہ پہلے سے ہے، اور غائب ریپوزٹری صرف ایسے فولڈر میں کلون ہوتی ہے جس میں پہلے سے آپ کی ریپوزٹریز ہوں؛ دوسرے فولڈروں کی اجازت کے لیے اسے سیٹ کریں۔ | کاموں means tasks/works, not commas: the input instructions tell users to separate repository roots by tasks instead of the required comma delimiter. This is a functional configuration instruction, not a style preference. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/3/entries/0` (UR-A4) |
| ur / B / semantic | minor / requiresAuthorRepair | اس ڈیوائس پر سیشن کی منتقلی کہاں ریپوزٹری بنا یا بحال کر سکتی ہے: آپ کے ہوم کے اندر فولڈر، ~ کے ساتھ لکھے اور کاموں سے الگ کیے گئے (مثلاً ~/code, ~/work)۔ پوشیدہ، سسٹم، کمانڈ (PATH پر) اور node_modules فولڈر کبھی استعمال نہیں ہوتے۔ سیٹ نہ ہو تو: ریپوزٹری وہیں بحال ہوتی ہے جہاں وہ پہلے سے ہے، اور غائب ریپوزٹری صرف ایسے فولڈر میں کلون ہوتی ہے جس میں پہلے سے آپ کی ریپوزٹریز ہوں؛ دوسرے فولڈروں کی اجازت کے لیے اسے سیٹ کریں۔ | اس ڈیوائس پر سیشن کی منتقلی کہاں ریپوزٹری بنا یا بحال کر سکتی ہے: آپ کے ہوم کے اندر فولڈر، ~ کے ساتھ لکھے اور کوما سے الگ کیے گئے (مثلاً ~/code, ~/work)۔ پوشیدہ، سسٹم، کمانڈ (PATH پر) اور node_modules فولڈر کبھی استعمال نہیں ہوتے۔ سیٹ نہ ہو تو: ریپوزٹری وہیں بحال ہوتی ہے جہاں وہ پہلے سے ہے، اور غائب ریپوزٹری صرف ایسے فولڈر میں کلون ہوتی ہے جس میں پہلے سے آپ کی ریپوزٹریز ہوں؛ دوسرے فولڈروں کی اجازت کے لیے اسے سیٹ کریں۔ | کاموں means tasks/works, not commas; the example mitigates the typo but the delimiter instruction itself is wrong. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/0` (URB-m1) |

### I18N-WS-4DE7C5E522CA — `wire.settings.mesh_move_working_set_roots.label`

**Source English:** Folders that may receive moved repositories

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5725.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Carpetas que pueden recibir repositorios movidos | Destinos de repositorios | The native label is 48 cells versus English 43; materially expanded relative to the documented 29-cell settings label column. Shorten or perform a rendered fit check; no clipping was observed here. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/32` (ES-B-33) |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Dossiers pouvant recevoir des dépôts déplacés | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/19` |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | وہ فولڈر جو منتقل شدہ ریپوزٹریز وصول کر سکتے ہیں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/6` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Thư mục có thể nhận kho mã được di chuyển | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/27` |

### I18N-WS-AC49F76AE0EB — `wire.settings.mesh_moves.title`

**Source English:** Session moves

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 994.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/70` |

### I18N-WS-78C695680028 — `wire.settings.mesh_updates.title`

**Source English:** Mesh updates

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 984.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/34` |

### I18N-WS-C4D3DE320B13 — `wire.settings.model_effort.choice.unset.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 1657.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/6` (H-A-EN-7) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/6` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/5` (RU-A-M1.06) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/4` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/5` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/5` (URB-6) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/5` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/5` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/5` (T-ZH-A-006) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/5` (B-ZH-M01.06) |

### I18N-WS-F73171632EA4 — `wire.settings.model_effort.label`

**Source English:** Default reasoning effort

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2063.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Esfuerzo de razonamiento predeterminado | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/5` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Effort de raisonnement par défaut | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/20` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Усилие рассуждений по умолчанию | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/48` (RU-A-W1.12) |

### I18N-WS-51573C03BEAE — `wire.settings.monitor.title`

**Source English:** Monitors

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 773.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/35` |

### I18N-WS-BDB852BCA5C5 — `wire.settings.monitor_classifymaxchars.help`

**Source English:** Bound on the changed lines the materiality check receives (the monitor's name and purpose ride along, clipped separately).

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4264.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / semantic | major / requiresAuthorRepair | Giới hạn số dòng thay đổi mà bước kiểm tra mức độ quan trọng nhận được (tên và mục đích của giám sát đi kèm, được cắt riêng). | Giới hạn số ký tự của phần thay đổi mà bước kiểm tra mức độ quan trọng nhận được (tên và mục đích của giám sát đi kèm, được cắt riêng). | Current số dòng means number of lines, but this setting clips characters: classify_max_chars is passed to gate_state; neighboring label explicitly says ký tự. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/33` (VI-A-M4) |
| vi / B / semantic | major / requiresAuthorRepair | Giới hạn số dòng thay đổi mà bước kiểm tra mức độ quan trọng nhận được (tên và mục đích của giám sát đi kèm, được cắt riêng). | Giới hạn số ký tự của phần văn bản chứa các dòng thay đổi mà bước kiểm tra mức độ quan trọng nhận được (tên và mục đích của giám sát đi kèm, được cắt riêng). | Giới hạn số dòng explicitly tells the user this character-count setting limits a number of lines, conflating it with monitor.maxDeltaLines; the sibling label says ký tự and monitors/classify.py:149-152 truncates len(delta_text) and character slices, not line counts. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/1` (VI-B-M2) |
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 重要性检查可接收的变更行上限（监控名称与用途一同传入，单独截断）。 | 重要性检查可接收的变更文本字符数上限（监控名称与用途一同传入，单独截断）。 | 变更行上限 can be read as a line-count limit, while the adjacent label and monitor.classifyMaxChars setting define a character cap; make the human unit explicit. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/40` (T-ZH-A-041) |

### I18N-WS-6DBE4623F180 — `wire.settings.monitor_classifymaxchars.label`

**Source English:** Classifier state (chars)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4261.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | État du classifieur (caractères) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/21` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Состояние классификатора (символы) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 34 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/49` (RU-A-W1.13) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Trạng thái bộ phân loại (ký tự) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/5` |

### I18N-WS-4F5C08064671 — `wire.settings.monitor_deltamaxchars.label`

**Source English:** Delta budget (chars)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4251.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Presupuesto de delta (caracteres) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/6` |

### I18N-WS-62A698FC2F6C — `wire.settings.monitor_maxconsecutivefailures.label`

**Source English:** Disable after failures

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4271.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Désactiver après échecs consécutifs | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/22` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/36` |

### I18N-WS-D192CF977C80 — `wire.settings.monitor_maxdeltalines.label`

**Source English:** Max delta lines

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4241.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ ڈیلٹا سطریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/7` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ ڈیلٹا سطریں | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/3` |

### I18N-WS-16A8F909E03A — `wire.settings.monitor_normalizetimestamps.label`

**Source English:** Ignore timestamp churn

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4291.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ignorar el ruido de las marcas de tiempo | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/7` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ignorer le bruit des horodatages | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/23` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Игнорировать шум меток времени | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 30 Russian cells versus 22 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/50` (RU-A-W1.14) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ٹائم اسٹیمپ تبدیلیاں نظر انداز | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/8` (UR-A9) |

### I18N-WS-48961C5D1F57 — `wire.settings.monitor_runtimeoutms.label`

**Source English:** Run timeout (ms)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4221.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Tiempo límite de ejecución (ms) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/8` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Тайм-аут выполнения (мс) | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/14` (RU-B-m01) |

### I18N-WS-0E941EEFBAF0 — `wire.settings.monitor_snapshotmaxchars.label`

**Source English:** Snapshot cap (chars)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4231.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Límite de la instantánea (caracteres) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/9` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Plafond de l’instantané (caractères) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/24` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/37` |

### I18N-WS-04C206A00754 — `wire.settings.network.description`

**Source English:** Where this device listens, what it tells peers to dial, and the bounds for `lop network`: audit retention, how many unauthenticated connections the relay will hold at once, session-copy cadence and how long a borrowed login lives. All take effect when the relay restarts.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 968.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / semantic | minor / rootMinorFoldOrDefer | Nơi thiết bị này lắng nghe, những gì nó báo cho thiết bị ngang hàng kết nối, và các giới hạn cho `lop network`: thời gian giữ nhật ký kiểm toán, số kết nối chưa xác thực mà relay giữ cùng lúc, nhịp sao chép phiên và thời gian tồn tại của đăng nhập đi mượn. Tất cả có hiệu lực khi relay khởi động lại. | Nơi thiết bị này lắng nghe, những gì nó báo cho thiết bị ngang hàng kết nối, và các giới hạn cho `lop network`: thời gian giữ nhật ký kiểm toán, số kết nối chưa xác thực mà bộ chuyển tiếp giữ cùng lúc, nhịp sao chép phiên và thời gian tồn tại của đăng nhập đi mượn. Tất cả có hiệu lực khi bộ chuyển tiếp khởi động lại. | The locale guide fixes relay → chuyển tiếp; relay drifts from that established term in these two human explanations. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/35` (VI-A-m2) |
| vi / B / semantic | minor / rootMinorFoldOrDefer | Nơi thiết bị này lắng nghe, những gì nó báo cho thiết bị ngang hàng kết nối, và các giới hạn cho `lop network`: thời gian giữ nhật ký kiểm toán, số kết nối chưa xác thực mà relay giữ cùng lúc, nhịp sao chép phiên và thời gian tồn tại của đăng nhập đi mượn. Tất cả có hiệu lực khi relay khởi động lại. | Nơi thiết bị này nhận kết nối, địa chỉ nó báo cho thiết bị ngang hàng kết nối tới, và các giới hạn cho `lop network`: thời gian giữ nhật ký kiểm toán, số kết nối chưa xác thực mà bộ chuyển tiếp giữ cùng lúc, nhịp sao chép phiên và thời gian tồn tại của thông tin đăng nhập mượn. Tất cả có hiệu lực khi bộ chuyển tiếp khởi động lại. | Glossary drift: source relay remains relay twice here versus required chuyển tiếp; identifiers and lop command remain unchanged. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/0` (VI-B-m1) |

### I18N-WS-18110E8894F1 — `wire.settings.network_advertise_hosts.label`

**Source English:** Advertised endpoints

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5576.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | घोषित endpoint | घोषित एंडपॉइंट | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/38` (H-A-MIX-39) |

### I18N-WS-462AD30FC358 — `wire.settings.network_audit_generations.label`

**Source English:** Audit generations kept

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5618.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Generaciones de auditoría conservadas | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/10` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Générations d’audit conservées | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/25` |

### I18N-WS-B7C77C8DC036 — `wire.settings.network_audit_max_age_days.label`

**Source English:** Audit age cap (days)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5629.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Límite de antigüedad de auditoría (días) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/11` |

### I18N-WS-5669EAA48D4E — `wire.settings.network_audit_max_bytes.label`

**Source English:** Audit size per generation

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5607.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Kích thước kiểm toán mỗi thế hệ | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/6` |

### I18N-WS-0625CB48E556 — `wire.settings.network_credentials_grant_ttl_s.label`

**Source English:** Borrowed login lifetime (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5786.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Duración del inicio de sesión prestado (s) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/12` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Durée de vie du login emprunté (s) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/26` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Срок жизни заимствованного логина (сек) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 39 Russian cells versus 27 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/51` (RU-A-W1.15) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Thời hạn đăng nhập mượn (giây) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/7` |

### I18N-WS-EF9537807420 — `wire.settings.network_credentials_offline_hold_s.label`

**Source English:** Borrowed login hold while owner is away (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5805.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Maintien du login emprunté, propriétaire absent (s) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/27` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Удержание логина в отсутствие владельца (сек) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 45 Russian cells versus 43 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/52` (RU-A-W1.16) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | مالک کی غیر موجودگی میں عارضی لاگ اِن مہلت (s) | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/9` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Giữ đăng nhập mượn khi chủ sở hữu vắng (giây) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/8` |
| zh-CN / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | 所有者离线时借用登录的保留时长（秒） | 离线时借用登录保留时长（秒） | Current text measures 36 terminal cells against the common 29-cell label column; the help can retain the owner-specific detail. Requires rendered confirmation. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/41` (B-ZH-m11) |

### I18N-WS-D59A681ACFD9 — `wire.settings.network_listen_address.label`

**Source English:** Listen address

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5531.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | کنکشن قبول کرنے کا ایڈریس | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/10` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | کنکشن قبول کرنے کا ایڈریس | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/4` |

### I18N-WS-552A0CCFA416 — `wire.settings.network_max_handshakes.help`

**Source English:** Unauthenticated connections this relay will hold at once. A pairing holds one for the seconds until the code is typed; anything past the cap is dropped at accept, with no reply frame.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5751.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | यह रिले एक बार में कितने अप्रमाणित कनेक्शन रखेगा। पेयरिंग कोड टाइप होने तक के सेकंडों भर एक कनेक्शन रोकती है; सीमा से ऊपर का हर कनेक्शन accept पर ही गिरा दिया जाता है, बिना कोई reply फ़्रेम। | यह रिले एक बार में कितने अप्रमाणित कनेक्शन रखेगा। पेयरिंग कोड टाइप होने तक एक कनेक्शन रखती है; सीमा से ऊपर के कनेक्शन स्वीकार होते ही छोड़ दिए जाते हैं, बिना जवाबी फ़्रेम भेजे। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/51` (H-A-MIX-52) |
| vi / A / semantic | minor / rootMinorFoldOrDefer | Số kết nối chưa xác thực mà relay này giữ cùng lúc. Một lần ghép nối giữ một kết nối trong vài giây cho đến khi mã được nhập; phần vượt trần bị loại ngay khi chấp nhận, không có khung hồi đáp. | Số kết nối chưa xác thực mà bộ chuyển tiếp này giữ cùng lúc. Một lần ghép nối giữ một kết nối trong vài giây cho đến khi mã được nhập; phần vượt trần bị loại ngay khi chấp nhận, không có khung hồi đáp. | The locale guide fixes relay → chuyển tiếp; relay drifts from that established term in these two human explanations. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/36` (VI-A-m2) |
| vi / B / semantic | minor / rootMinorFoldOrDefer | Số kết nối chưa xác thực mà relay này giữ cùng lúc. Một lần ghép nối giữ một kết nối trong vài giây cho đến khi mã được nhập; phần vượt trần bị loại ngay khi chấp nhận, không có khung hồi đáp. | Số kết nối chưa xác thực mà bộ chuyển tiếp này giữ cùng lúc. Một lần ghép nối giữ một kết nối cho đến khi mã được nhập; kết nối vượt trần bị loại ngay khi chấp nhận, không có khung hồi đáp. | Glossary drift relay versus chuyển tiếp; the extra vài giây also unnecessarily suggests a short fixed pairing duration. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/1` (VI-B-m2) |

### I18N-WS-DAA433E4300B — `wire.settings.network_port.help`

**Source English:** The port peers dial; endpoints without one use it.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5563.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | वह पोर्ट जो पीयर डायल करते हैं; बिना पोर्ट के endpoint इसी का इस्तेमाल करते हैं। | वह पोर्ट जो पीयर डायल करते हैं; बिना पोर्ट के एंडपॉइंट इसी का इस्तेमाल करते हैं। | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/39` (H-A-MIX-40) |

### I18N-WS-E3A8B469EA96 — `wire.settings.network_port.label`

**Source English:** Mesh port

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5553.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/38` |

### I18N-WS-66D2D9CC6848 — `wire.settings.network_rollout_on_update.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5690.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/7` (H-A-EN-8) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/7` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/6` (RU-A-M1.07) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/5` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/6` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/6` (URB-7) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/6` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/6` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/6` (T-ZH-A-007) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/6` (B-ZH-M01.07) |

### I18N-WS-A447EE040B38 — `wire.settings.network_rollout_on_update.choice.off.label`

**Source English:** off

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5695.

**Unchanged raw Choice.value:** `off` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/39` |
| zh-CN / A / semantic | major / requiresAuthorRepair | off | 关闭 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/7` (T-ZH-A-008) |
| zh-CN / B / semantic | major / requiresAuthorRepair | off | 关闭 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/7` (B-ZH-M01.08) |

### I18N-WS-4DB1F1014DBC — `wire.settings.network_rollout_on_update.label`

**Source English:** Roll the mesh after an update

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5683.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | Propager le maillage après une mise à jour | Propager la mise à jour | The current label says to propagate the mesh itself rather than propagate the update through it; its help supplies the correct action. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/24` (FR-A-m04) |
| fr / A / width risk; not observed clipping | minor / rootMinorFoldOrDefer | Propager le maillage après une mise à jour | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/28` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Раскатывать меш-сеть после обновления | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 37 Russian cells versus 29 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/53` (RU-A-W1.17) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Cập nhật cuốn chiếu cả mesh sau khi cập nhật | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/9` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Cập nhật cuốn chiếu cả mesh sau khi cập nhật | Cuốn chiếu sau cập nhật | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/5` |

### I18N-WS-662F033EF478 — `wire.settings.network_sync_debounce_s.help`

**Source English:** Seconds a session's transcript must stay unchanged before its owner tells devices holding a copy to pull. A session going idle pushes at once, so its final message is never held back by this.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5769.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | major / requiresAuthorRepair | مالک، نقل رکھنے والی ڈیوائسز کو pull کہنے سے پہلے سیشن ٹرانسکرپٹ کتنے سیکنڈ غیر بدلا رہے۔ بے کار ہوتے ہی push فوراً ہوتا ہے، اس لیے آخری پیغام اس سے نہیں رکتا۔ | نقل رکھنے والی ڈیوائسز کو تازہ نقل لینے کی اطلاع دینے سے پہلے سیشن کا ٹرانسکرپٹ کتنے سیکنڈ بغیر تبدیلی رہے۔ سیشن غیر فعال ہوتے ہی تازہ نقل فوراً بھیجی جاتی ہے، اس لیے آخری پیغام اس سے نہیں رکتا۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/44` (URB-45) |

### I18N-WS-B0A23D525300 — `wire.settings.network_sync_debounce_s.label`

**Source English:** Sync quiet period (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5764.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Periodo de calma de sincronización (s) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/13` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Période de calme de synchronisation (s) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/29` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/40` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Период тишины синхронизации (сек) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 21 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/54` (RU-A-W1.18) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Период тишины синхронизации (сек) | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/15` (RU-B-m01) |

### I18N-WS-505F370AA612 — `wire.settings.network_sync_tick_s.help`

**Source English:** How often the owner checks copied sessions for changes: one file stat each.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5780.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | major / requiresAuthorRepair | مالک نقل شدہ سیشنز میں تبدیلی کتنی بار جانچ کرے: ہر بار ایک فائل stat۔ | مالک نقل شدہ سیشنز میں تبدیلی کتنی بار جانچے: ہر سیشن کی فائل کی حالت ایک بار۔ | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/45` (URB-46) |

### I18N-WS-247A04DC5FA3 — `wire.settings.network_sync_tick_s.label`

**Source English:** Sync check interval (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5775.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Intervalo de comprobación de sincronización (s) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/14` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Intervalle de vérification de synchronisation (s) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/30` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/41` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Интервал проверки синхронизации (сек) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 37 Russian cells versus 23 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/55` (RU-A-W1.19) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Интервал проверки синхронизации (сек) | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/16` (RU-B-m01) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Chu kỳ kiểm tra đồng bộ (giây) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/10` |

### I18N-WS-58C37E2E5790 — `wire.settings.proactive_patience_backoff.label`

**Source English:** Patience backoff factor

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6197.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Facteur d’espacement de patience | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/31` |

### I18N-WS-F966D0189EC9 — `wire.settings.proactive_patience_default_ms.label`

**Source English:** Default patience wait (ms)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6186.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Attente de patience par défaut (ms) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/32` |
| ru / A / semantic | minor / rootMinorFoldOrDefer | Ожидание терпения по умолчанию (мс) | Ожидание по умолчанию (мс) | «Ожидание терпения» is an unnatural calque: the setting is a wait duration, not waiting for patience; the neighbouring backoff label already uses «ожиданий». — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/35` (RU-A-m3) |
| ru / A / width risk; not observed clipping | minor / rootMinorFoldOrDefer | Ожидание терпения по умолчанию (мс) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 35 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/56` (RU-A-W1.20) |

### I18N-WS-17584A41BC9B — `wire.settings.proactive_patience_max_pending.label`

**Source English:** Pending waits per session

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6230.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Attentes en suspens par session | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/33` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Незавершённые ожидания на сессию | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 32 Russian cells versus 25 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/57` (RU-A-W1.21) |

### I18N-WS-A7F067B70BC1 — `wire.settings.providers.title`

**Source English:** Wire protocol

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 549.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/42` |
| ur / A / semantic | major / requiresAuthorRepair | Wire پروٹوکول | مواصلاتی پروٹوکول | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/1` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | Wire پروٹوکول | ترسیلی پروٹوکول | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/40` (URB-41) |

### I18N-WS-D580E7728DFC — `wire.settings.providers_anthropic_cache_ttl_1h_min_context_tokens.label`

**Source English:** Anthropic 1h cache above (tokens)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2124.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Cache 1 h d’Anthropic au-delà (tokens) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/34` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Кэш Anthropic на 1 ч свыше (токены) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 35 Russian cells versus 33 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/58` (RU-A-W1.22) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ngưỡng cache Anthropic 1 giờ (token) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/11` |

### I18N-WS-CE761566ACC2 — `wire.settings.providers_bedrock_auth.label`

**Source English:** Amazon Bedrock auth

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2170.

**Context (existing sidecar):** {"role": "text setting label", "tone": "plain", "maxCells": 29, "note": "How Bedrock authenticates."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Аутентификация Amazon Bedrock | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/17` (RU-B-m01) |

### I18N-WS-E1430343996D — `wire.settings.providers_openai_api.label`

**Source English:** OpenAI API surface

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2097.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Superficie de la API de OpenAI | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/15` |

### I18N-WS-859FFA80B06C — `wire.settings.providers_openai_use_max_context_window.label`

**Source English:** Use maximum OpenAI context

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2088.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Utiliser le contexte OpenAI maximal | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/35` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Использовать максимальный контекст OpenAI | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 41 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/59` (RU-A-W1.23) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Использовать максимальный контекст OpenAI | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/18` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ OpenAI سیاق و سباق استعمال کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/11` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ OpenAI سیاق و سباق استعمال کریں | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/5` |

### I18N-WS-513A71CB406C — `wire.settings.providers_openrouter_allow_fallbacks.choice.false.label`

**Source English:** false

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2324.

**Unchanged raw Choice.value:** `false` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | false | desactivado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/0` (ES-A-M1.1) |
| es / B / semantic | major / requiresAuthorRepair | false | no | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/6` (ES-B-07) |
| fr / A / semantic | major / requiresAuthorRepair | false | désactivés | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/9` (FR-A-M10) |
| fr / B / semantic | major / requiresAuthorRepair | false | non | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/1` (FR-B-M02) |
| hi / A / semantic | major / requiresAuthorRepair | false | नहीं | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/8` (H-A-EN-9) |
| hi / B / semantic | major / requiresAuthorRepair | false | नहीं | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/8` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | false | не переключаться | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/7` (RU-A-M1.08) |
| ru / B / semantic | major / requiresAuthorRepair | false | нет | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/16` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | false | نہیں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/7` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | false | فال بیک بند | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/7` (URB-8) |
| vi / A / semantic | major / requiresAuthorRepair | false | Không | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/7` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | false | Tắt | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/7` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | false | 否 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/8` (T-ZH-A-009) |
| zh-CN / B / semantic | major / requiresAuthorRepair | false | 否 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/8` (B-ZH-M01.09) |

### I18N-WS-32DB6C298E5C — `wire.settings.providers_openrouter_allow_fallbacks.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2319.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/1` (ES-A-M1.2) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/0` (ES-B-01) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/10` (FR-A-M11) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/2` (FR-B-M03) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/9` (H-A-EN-10) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/9` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/8` (RU-A-M1.09) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/10` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/8` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/8` (URB-9) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/8` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/8` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/9` (T-ZH-A-010) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/9` (B-ZH-M01.10) |

### I18N-WS-DD21D2852FC7 — `wire.settings.providers_openrouter_data_collection.choice.allow.label`

**Source English:** allow

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2369.

**Unchanged raw Choice.value:** `allow` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | allow | permitir | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/2` (ES-A-M1.3) |
| es / B / semantic | major / requiresAuthorRepair | allow | permitir | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/7` (ES-B-08) |
| fr / A / semantic | major / requiresAuthorRepair | allow | autoriser | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/11` (FR-A-M12) |
| fr / B / semantic | major / requiresAuthorRepair | allow | autoriser | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/3` (FR-B-M04) |
| hi / A / semantic | major / requiresAuthorRepair | allow | अनुमति दें | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/10` (H-A-EN-11) |
| hi / B / semantic | major / requiresAuthorRepair | allow | अनुमति दें | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/10` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | allow | разрешить | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/9` (RU-A-M1.10) |
| ru / B / semantic | major / requiresAuthorRepair | allow | разрешить | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/17` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | allow | اجازت دیں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/9` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | allow | اجازت دیں | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/9` (URB-10) |
| vi / A / semantic | major / requiresAuthorRepair | allow | Cho phép | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/9` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | allow | Cho phép | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/9` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | allow | 允许 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/10` (T-ZH-A-011) |
| zh-CN / B / semantic | major / requiresAuthorRepair | allow | 允许 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/10` (B-ZH-M01.11) |

### I18N-WS-961F6FF2DA51 — `wire.settings.providers_openrouter_data_collection.choice.deny.label`

**Source English:** deny

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2374.

**Unchanged raw Choice.value:** `deny` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | deny | denegar | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/3` (ES-A-M1.4) |
| es / B / semantic | major / requiresAuthorRepair | deny | impedir | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/8` (ES-B-09) |
| fr / A / semantic | major / requiresAuthorRepair | deny | refuser | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/12` (FR-A-M13) |
| fr / B / semantic | major / requiresAuthorRepair | deny | refuser | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/4` (FR-B-M05) |
| hi / A / semantic | major / requiresAuthorRepair | deny | अनुमति न दें | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/11` (H-A-EN-12) |
| hi / B / semantic | major / requiresAuthorRepair | deny | मना करें | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/11` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | deny | запретить | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/10` (RU-A-M1.11) |
| ru / B / semantic | major / requiresAuthorRepair | deny | запретить | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/18` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | deny | اجازت نہ دیں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/10` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | deny | اجازت نہ دیں | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/10` (URB-11) |
| vi / A / semantic | major / requiresAuthorRepair | deny | Không cho phép | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/10` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | deny | Không cho phép | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/10` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | deny | 禁止 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/11` (T-ZH-A-012) |
| zh-CN / B / semantic | major / requiresAuthorRepair | deny | 禁止 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/11` (B-ZH-M01.12) |

### I18N-WS-EAD73985D2C9 — `wire.settings.providers_openrouter_data_collection.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2364.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/4` (ES-A-M1.5) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/1` (ES-B-02) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/13` (FR-A-M14) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/5` (FR-B-M06) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/12` (H-A-EN-13) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/12` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/11` (RU-A-M1.12) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/11` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/11` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/11` (URB-12) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/11` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/11` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/12` (T-ZH-A-013) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/12` (B-ZH-M01.13) |

### I18N-WS-B76646004E00 — `wire.settings.providers_openrouter_enforce_distillable_text.choice.true.description`

**Source English:** distillable-text endpoints only

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2424.

**Unchanged raw Choice.value:** `true` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | सिर्फ़ distillable-text endpoint | सिर्फ़ मॉडल डिस्टिलेशन योग्य टेक्स्ट वाले एंडपॉइंट | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/40` (H-A-MIX-41) |

### I18N-WS-378BA7328B99 — `wire.settings.providers_openrouter_enforce_distillable_text.choice.true.label`

**Source English:** true

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2423.

**Unchanged raw Choice.value:** `true` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | true | activado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/5` (ES-A-M1.6) |
| es / B / semantic | major / requiresAuthorRepair | true | sí | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/9` (ES-B-10) |
| fr / A / semantic | major / requiresAuthorRepair | true | activé | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/14` (FR-A-M15) |
| fr / B / semantic | major / requiresAuthorRepair | true | oui | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/6` (FR-B-M07) |
| hi / A / semantic | major / requiresAuthorRepair | true | हाँ | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/13` (H-A-EN-14) |
| hi / B / semantic | major / requiresAuthorRepair | true | हाँ | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/13` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | true | включено | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/12` (RU-A-M1.13) |
| ru / B / semantic | major / requiresAuthorRepair | true | да | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/19` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | true | ہاں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/12` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | true | فعال | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/12` (URB-13) |
| vi / A / semantic | major / requiresAuthorRepair | true | Có | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/12` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | true | Bật | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/12` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | true | 是 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/13` (T-ZH-A-014) |
| zh-CN / B / semantic | major / requiresAuthorRepair | true | 是 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/13` (B-ZH-M01.14) |

### I18N-WS-CD06C73F8E78 — `wire.settings.providers_openrouter_enforce_distillable_text.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2418.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/6` (ES-A-M1.7) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/2` (ES-B-03) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/15` (FR-A-M16) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/7` (FR-B-M08) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/14` (H-A-EN-15) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/14` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/13` (RU-A-M1.14) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/12` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/13` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/13` (URB-14) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/13` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/13` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/14` (T-ZH-A-015) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/14` (B-ZH-M01.15) |

### I18N-WS-572D791F9AEE — `wire.settings.providers_openrouter_enforce_distillable_text.help`

**Source English:** 'true': restrict to endpoints whose text output may be distilled. 'default': nothing sent.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2414.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | 'true': सिर्फ़ ऐसे endpoint जिनका टेक्स्ट आउटपुट distill किया जा सके। 'default': कुछ नहीं भेजा जाता। | 'true': सिर्फ़ ऐसे एंडपॉइंट जिनका टेक्स्ट आउटपुट distill किया जा सके। 'default': कुछ नहीं भेजा जाता। | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/42` (H-A-MIX-43) |

### I18N-WS-1F436A4FCED3 — `wire.settings.providers_openrouter_enforce_distillable_text.label`

**Source English:** distillable text

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2410.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | distillable टेक्स्ट | मॉडल डिस्टिलेशन योग्य टेक्स्ट | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/41` (H-A-MIX-42) |
| hi / B / semantic | major / requiresAuthorRepair | distillable टेक्स्ट | डिस्टिलेशन योग्य टेक्स्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/32` (HI-B-M1) |

### I18N-WS-7EEFBBAD3CF1 — `wire.settings.providers_openrouter_max_price.label`

**Source English:** max price

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2461.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ قیمت | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/12` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ قیمت | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/6` |

### I18N-WS-CE67E6FD4F46 — `wire.settings.providers_openrouter_order.label`

**Source English:** host order

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2252.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/43` |

### I18N-WS-F9A1D1AC6137 — `wire.settings.providers_openrouter_preferred_max_latency.label`

**Source English:** max latency (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2489.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ لیٹنسی (s) | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/13` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ لیٹنسی (s) | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/7` |

### I18N-WS-66AB3AF9B371 — `wire.settings.providers_openrouter_preferred_min_throughput.label`

**Source English:** min throughput (tok/s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2478.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | мин. пропускная способность (ток/с) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 35 Russian cells versus 22 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/60` (RU-A-W1.24) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | мин. пропускная способность (ток/с) | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/19` (RU-B-m01) |

### I18N-WS-2B7797E9A40D — `wire.settings.providers_openrouter_require_parameters.choice.true.label`

**Source English:** true

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2348.

**Unchanged raw Choice.value:** `true` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | true | activado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/7` (ES-A-M1.8) |
| es / B / semantic | major / requiresAuthorRepair | true | sí | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/10` (ES-B-11) |
| fr / A / semantic | major / requiresAuthorRepair | true | activé | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/16` (FR-A-M17) |
| fr / B / semantic | major / requiresAuthorRepair | true | oui | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/8` (FR-B-M09) |
| hi / A / semantic | major / requiresAuthorRepair | true | हाँ | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/15` (H-A-EN-16) |
| hi / B / semantic | major / requiresAuthorRepair | true | हाँ | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/15` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | true | включено | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/14` (RU-A-M1.15) |
| ru / B / semantic | major / requiresAuthorRepair | true | да | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/20` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | true | ہاں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/14` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | true | فعال | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/14` (URB-15) |
| vi / A / semantic | major / requiresAuthorRepair | true | Có | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/14` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | true | Bật | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/14` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | true | 是 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/15` (T-ZH-A-016) |
| zh-CN / B / semantic | major / requiresAuthorRepair | true | 是 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/15` (B-ZH-M01.16) |

### I18N-WS-653D8954B431 — `wire.settings.providers_openrouter_require_parameters.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2343.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/8` (ES-A-M1.9) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/3` (ES-B-04) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/17` (FR-A-M18) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/9` (FR-B-M10) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/16` (H-A-EN-17) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/16` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/15` (RU-A-M1.16) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/13` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/15` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/15` (URB-16) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/15` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/15` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/16` (T-ZH-A-017) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/16` (B-ZH-M01.17) |

### I18N-WS-A20600314A71 — `wire.settings.providers_openrouter_require_parameters.help`

**Source English:** 'true' restricts to hosts that support every parameter you send (tools, structured output); 'default' lets OpenRouter silently drop unsupported ones.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2339.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | 'true' सिर्फ़ उन होस्ट तक सीमित करता है जो आपके भेजे हर पैरामीटर को सपोर्ट करते हैं (टूल, structured output); 'default' में OpenRouter असमर्थित पैरामीटर चुपचाप हटा देता है। | 'true' सिर्फ़ उन होस्ट तक सीमित करता है जो आपके भेजे हर पैरामीटर को सपोर्ट करते हैं (टूल, संरचित आउटपुट); 'default' में OpenRouter असमर्थित पैरामीटर चुपचाप हटा सकता है। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/53` (H-A-MIX-54) |

### I18N-WS-8521E4568676 — `wire.settings.providers_openrouter_sort.choice.throughput.description`

**Source English:** highest tokens/sec first

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2239.

**Unchanged raw Choice.value:** `throughput` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | tokens par seconde les plus élevés d’abord | débit le plus élevé d’abord | The current wording assigns « les plus élevés » to tokens rather than the throughput rate being ranked. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/23` (FR-A-m03) |

### I18N-WS-70479CB51DE5 — `wire.settings.providers_openrouter_sort.choice.throughput.label`

**Source English:** throughput

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2238.

**Unchanged raw Choice.value:** `throughput` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | пропускная способность | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/20` (RU-B-m01) |

### I18N-WS-8DDA7C4D87CB — `wire.settings.providers_openrouter_sort.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2228.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/9` (ES-A-M1.10) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/4` (ES-B-05) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/18` (FR-A-M19) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/10` (FR-B-M11) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/17` (H-A-EN-18) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/17` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/16` (RU-A-M1.17) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/14` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/16` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/16` (URB-17) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/16` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/16` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/17` (T-ZH-A-018) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/17` (B-ZH-M01.18) |

### I18N-WS-DF07BD2D089D — `wire.settings.providers_openrouter_sort.label`

**Source English:** routing policy

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2217.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | политика маршрутизации | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/21` (RU-B-m01) |

### I18N-WS-4853E8732AB9 — `wire.settings.providers_openrouter_zdr.choice.true.description`

**Source English:** zero-data-retention endpoints only

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2402.

**Unchanged raw Choice.value:** `true` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | सिर्फ़ शून्य-डेटा-रिटेंशन endpoint | सिर्फ़ शून्य-डेटा-रिटेंशन एंडपॉइंट | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/43` (H-A-MIX-44) |

### I18N-WS-3DE117B6A717 — `wire.settings.providers_openrouter_zdr.choice.true.label`

**Source English:** true

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2401.

**Unchanged raw Choice.value:** `true` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | true | activado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/10` (ES-A-M1.11) |
| es / B / semantic | major / requiresAuthorRepair | true | sí | English machine-token spelling leaked into the independently localized human selector label; preserve Choice.value, translate only the display text. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/11` (ES-B-12) |
| fr / A / semantic | major / requiresAuthorRepair | true | activée | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/19` (FR-A-M20) |
| fr / B / semantic | major / requiresAuthorRepair | true | oui | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/11` (FR-B-M12) |
| hi / A / semantic | major / requiresAuthorRepair | true | हाँ | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/18` (H-A-EN-19) |
| hi / B / semantic | major / requiresAuthorRepair | true | हाँ | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/18` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | true | включено | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/17` (RU-A-M1.18) |
| ru / B / semantic | major / requiresAuthorRepair | true | да | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/21` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | true | ہاں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/17` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | true | فعال | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/17` (URB-18) |
| vi / A / semantic | major / requiresAuthorRepair | true | Có | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/17` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | true | Bật | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/17` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | true | 是 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/18` (T-ZH-A-019) |
| zh-CN / B / semantic | major / requiresAuthorRepair | true | 是 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/18` (B-ZH-M01.19) |

### I18N-WS-A73734B5D1A9 — `wire.settings.providers_openrouter_zdr.choice.unset.label`

**Source English:** default

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2396.

**Unchanged raw Choice.value:** `` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | default | predeterminado | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/11` (ES-A-M1.12) |
| es / B / semantic | major / requiresAuthorRepair | default | predeterminado | This is independently localized Choice.label shown to a person, not the stable enum value; English default remains untranslated. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/5` (ES-B-06) |
| fr / A / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/20` (FR-A-M21) |
| fr / B / semantic | major / requiresAuthorRepair | default | par défaut | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/12` (FR-B-M13) |
| hi / A / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/19` (H-A-EN-20) |
| hi / B / semantic | major / requiresAuthorRepair | default | डिफ़ॉल्ट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/19` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/18` (RU-A-M1.19) |
| ru / B / semantic | major / requiresAuthorRepair | default | по умолчанию | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/15` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/18` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | default | ڈیفالٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/18` (URB-19) |
| vi / A / semantic | major / requiresAuthorRepair | default | Mặc định | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/18` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | default | Mặc định | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/18` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | default | 默认 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/19` (T-ZH-A-020) |
| zh-CN / B / semantic | major / requiresAuthorRepair | default | 默认 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/19` (B-ZH-M01.20) |

### I18N-WS-7E0C387948DC — `wire.settings.providers_openrouter_zdr.help`

**Source English:** 'true': restrict to zero-data-retention endpoints. 'default': nothing sent.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 2392.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | 'true': सिर्फ़ शून्य-डेटा-रिटेंशन endpoint तक सीमित। 'default': कुछ नहीं भेजा जाता। | 'true': सिर्फ़ शून्य-डेटा-रिटेंशन एंडपॉइंट तक सीमित। 'default': कुछ नहीं भेजा जाता। | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/44` (H-A-MIX-45) |

### I18N-WS-D142E5A3DF13 — `wire.settings.retired.title`

**Source English:** Retired

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1158.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / semantic | minor / rootMinorFoldOrDefer | Выведенные | Устаревшие | «Выведенные» is an incomplete Russian participle here; the section names inactive/retired settings, not something displayed or printed. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/33` (RU-A-m1) |

### I18N-WS-2165CEE5E3DF — `wire.settings.retry_connectivitybackoffcapms.label`

**Source English:** Connectivity backoff cap (ms)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2564.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Plafond d’espacement de connectivité (ms) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/36` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Потолок задержки подключения (мс) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 29 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/61` (RU-A-W1.25) |

### I18N-WS-E4DC12CBA5EC — `wire.settings.retry_enabled.help`

**Source English:** Retry a failed provider call before surfacing the error.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2521.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 提供商调用失败时先重试，再浮现错误。 | 提供商调用失败时先重试，再显示错误。 | 浮现错误 is an English surfacing calque rather than natural Chinese UI wording. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/35` (T-ZH-A-036) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 提供商调用失败时先重试，再浮现错误。 | 提供商调用失败时先重试，再显示错误。 | 浮现错误 is a calque of surfacing an error; the UI displays or reports the error. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/35` (B-ZH-m05) |

### I18N-WS-8F5154F3BF57 — `wire.settings.retry_enabled.label`

**Source English:** Retry failed calls

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2518.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/44` |

### I18N-WS-7403A0421E69 — `wire.settings.retry_fallbackchains.help`

**Source English:** Ordered provider/model hops tried when a call keeps failing.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2644.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 调用持续失败时按顺序尝试的provider/model跳转。 | 调用持续失败时，按顺序尝试的提供商/模型切换路径。 | The English model/provider nouns remain untranslated here, although neighboring settings consistently use 模型/提供商; these are not actual provider or model identifiers. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/43` (B-ZH-m13) |

### I18N-WS-6B68A05EF2EE — `wire.settings.retry_maxretries.label`

**Source English:** Max retries

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2531.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ دوبارہ کوششیں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/14` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ دوبارہ کوششیں | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/8` |

### I18N-WS-550292943B6D — `wire.settings.retry_modelfallback.label`

**Source English:** Model fallback

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2575.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Резервный переход модели | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/22` (RU-B-m01) |

### I18N-WS-6392C330595D — `wire.settings.retry_pinnedfallback.choice.cross_family.label`

**Source English:** allow cross-vendor

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2685.

**Unchanged raw Choice.value:** `cross-family` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/45` |

### I18N-WS-9C9AD8F0CC96 — `wire.settings.retry_pinnedfallback.label`

**Source English:** Pinned child fallback

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2650.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Резервный переход закреплённой дочерней сессии | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 46 Russian cells versus 21 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/62` (RU-A-W1.26) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Резервный переход закреплённой дочерней сессии | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/23` (RU-B-m01) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Dự phòng cho tác nhân con được ghim | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/12` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Dự phòng cho tác nhân con được ghim | Dự phòng tác nhân con ghim | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/6` |

### I18N-WS-27760CFBE483 — `wire.settings.retry_usageawareaccountpick.label`

**Source English:** Usage-aware account pick

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` no direct static getter call identified.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Choix de compte selon l’utilisation | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/37` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Выбор аккаунта по использованию | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/63` (RU-A-W1.27) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | استعمال کے حساب سے اکاؤنٹ انتخاب | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/15` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Chọn tài khoản theo lượng sử dụng | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/13` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Chọn tài khoản theo lượng sử dụng | Chọn tài khoản theo hạn mức | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/7` |

### I18N-WS-CF529310946A — `wire.settings.retry_usageawarefallback.label`

**Source English:** Usage-aware fallback

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2588.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Резервный переход по использованию | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 34 Russian cells versus 20 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/64` (RU-A-W1.28) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Резервный переход по использованию | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/24` (RU-B-m01) |

### I18N-WS-CF5957DAA180 — `wire.settings.retry_usagereservepercent.help`

**Source English:** Headroom below which an account counts as low; a running session stays on it.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 2633.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 低于此余量时账户视为低额；正在运行的会话仍留在其上。 | 剩余配额低于此比例时，账户视为配额不足；运行中的会话仍继续使用该账户。 | 低额 is not idiomatic for low remaining provider quota, and 留在其上 reads as translated English. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/36` (B-ZH-m06) |

### I18N-WS-7E728FCF2E58 — `wire.settings.runtime_background_on_resume.label`

**Source English:** Keep working after /resume

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3458.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Продолжать работу после /resume | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/65` (RU-A-W1.29) |

### I18N-WS-22A07A41245B — `wire.settings.runtime_keep_alive_max.label`

**Source English:** Max warm conversations

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3493.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | minor / rootMinorFoldOrDefer | Số cuộc trò chuyện ấm tối đa | Số cuộc trò chuyện giữ sẵn tối đa | Cuộc trò chuyện ấm is a word-for-word warm-conversation calque, not natural Vietnamese for a retained live runtime. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/5` (VI-B-m6) |

### I18N-WS-4968A9E98C2C — `wire.settings.runtime_keep_alive_seconds.label`

**Source English:** Keep closed conversations warm (s)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3482.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Mantener en caliente las conversaciones cerradas (s) | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/16` |
| es / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Mantener en caliente las conversaciones cerradas (s) | Conservar al cerrar (s) | The native label is 52 cells versus English 34; materially expanded relative to the documented 29-cell settings label column. Shorten or perform a rendered fit check; no clipping was observed here. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/31` (ES-B-32) |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Garder les conversations fermées au chaud (s) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/38` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Держать закрытые беседы тёплыми (сек) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 37 Russian cells versus 34 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/66` (RU-A-W1.30) |
| vi / B / semantic | minor / rootMinorFoldOrDefer | Giữ cuộc trò chuyện đã đóng ấm (giây) | Giữ sẵn phiên đã đóng (giây) | Giữ cuộc trò chuyện đã đóng ấm is an unidiomatic adjective-order calque; giữ sẵn conveys warm runtime retention without changing the seconds unit. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/6` (VI-B-m7) |
| vi / A / width risk; not observed clipping | minor / rootMinorFoldOrDefer | Giữ cuộc trò chuyện đã đóng ấm (giây) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/14` |

### I18N-WS-0D4117ADED77 — `wire.settings.runtime_unattended_gate_timeout.label`

**Source English:** Unattended question timeout (h)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3689.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Délai des questions en votre absence (h) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/39` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Тайм-аут вопроса в ваше отсутствие (ч) | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 38 Russian cells versus 31 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/67` (RU-A-W1.31) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Hết giờ cho câu hỏi khi vắng mặt (giờ) | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/15` |

### I18N-WS-B6346C5D3696 — `wire.settings.send_journal_unconfirmed.label`

**Source English:** Journal unacknowledged sends

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5062.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Journaliser les envois non acquittés | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/40` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Журналировать неподтверждённые отправки | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 39 Russian cells versus 28 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/68` (RU-A-W1.32) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | غیر تصدیق شدہ send اندراجات جرنل کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/16` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ghi nhật ký send chưa được xác nhận | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/16` |

### I18N-WS-855C1103EB1D — `wire.settings.session_cleanup_delegated_enabled.label`

**Source English:** Delegated cleanup

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3636.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/46` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Очистка делегированной работы | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/25` (RU-B-m01) |

### I18N-WS-FDE808AF80F0 — `wire.settings.session_cleanup_max_inactive_days.label`

**Source English:** ↳ max inactive days

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3550.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ số ngày không hoạt động tối đa | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/17` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ số ngày không hoạt động tối đa | ↳ số ngày không hoạt động | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/8` |

### I18N-WS-CA0631429C43 — `wire.settings.session_cleanup_max_sessions.label`

**Source English:** ↳ max sessions

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3538.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ سیشنز | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/17` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ سیشنز | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/9` |

### I18N-WS-D968226F1484 — `wire.settings.session_cleanup_max_total_bytes.label`

**Source English:** ↳ max total bytes

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3562.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | major / requiresAuthorRepair | ↳ زیادہ سے زیادہ کل bytes | ↳ زیادہ سے زیادہ کل بائٹس | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/42` (URB-43) |

### I18N-WS-AD3E7742DB94 — `wire.settings.session_cleanup_remove_empty.label`

**Source English:** ↳ remove empty

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3574.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/47` |

### I18N-WS-D2CD3606B750 — `wire.settings.session_cleanup_temporary_idle_minutes.label`

**Source English:** Delete temporary chats after

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3673.

**Context (existing sidecar):** {"role": "settings row label", "tone": "plain", "note": "A 'temporary chat' is a conversation that is not kept: it is left out of history and deleted automatically once idle for this long. Followed by a duration choice."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Supprimer les conversations temporaires après | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/41` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/69` |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | اتنی دیر بعد عارضی چیٹس حذف کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/18` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Xóa cuộc trò chuyện tạm thời sau | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/25` |

### I18N-WS-0C842967FFF2 — `wire.settings.session_scratchpad.title`

**Source English:** Scratchpad

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 699.

**Context (existing sidecar):** {"role": "settings section title", "tone": "plain", "maxCells": 12, "note": "Section heading for the scratchpad size cap. 'Scratchpad' is the product noun for a session's own scratch area (scratchpad://); keep it one word."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | Scratchpad | Borrador | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/21` (ES-A-M1.22) |
| es / B / semantic | major / requiresAuthorRepair | Scratchpad | Borrador | Human section heading names the session scratch area, not the literal scratchpad:// URI; existing Spanish description already calls it área de borrador. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/21` (ES-B-22) |
| ur / A / semantic | major / requiresAuthorRepair | Scratchpad | عارضی ذخیرہ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/6` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | Scratchpad | اسکریچ پیڈ | Scratchpad is a human section noun here, not the scratchpad:// URI or a named model/brand; the context calls for a compact noun. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/46` (URB-47) |
| vi / A / semantic | major / requiresAuthorRepair | Scratchpad | Nháp | This is a visible section heading for scratch storage, not the scratchpad:// scheme; Nháp is a concise native noun and respects the one-word context requirement. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/32` (VI-A-M3) |
| vi / B / semantic | major / requiresAuthorRepair | Scratchpad | Vùng nháp | A human heading for session scratch storage, not the scratchpad:// protocol token; fits the 12-cell budget. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/32` (VI-B-M1) |

### I18N-WS-7BB0A888B1A4 — `wire.settings.session_scratchpad_max_bytes.help`

**Source English:** Bytes a session's scratchpad may hold: 16 MiB to 8 GiB. Default 256 MiB.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3439.

**Context (existing sidecar):** {"role": "settings help text", "tone": "plain, precise", "note": "Carries both bounds and the default because the page paints help but not min/max. Units are bytes; use the binary prefixes MiB/GiB, not MB/GB."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | Bytes que puede guardar el scratchpad de una sesión: de 16 MiB a 8 GiB. Predeterminado: 256 MiB. | Bytes que puede guardar el borrador de una sesión: de 16 MiB a 8 GiB. Predeterminado: 256 MiB. | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/23` (ES-A-M1.24) |
| es / B / semantic | major / requiresAuthorRepair | Bytes que puede guardar el scratchpad de una sesión: de 16 MiB a 8 GiB. Predeterminado: 256 MiB. | Bytes que puede guardar el borrador de una sesión: de 16 MiB a 8 GiB. Predeterminado: 256 MiB. | Human description retains untranslated scratchpad; use the same native term as its section and preserve bounds and units. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/22` (ES-B-23) |
| ur / A / semantic | major / requiresAuthorRepair | سیشن کے Scratchpad کی گنجائش: 16 MiB تا 8 GiB۔ ڈیفالٹ 256 MiB۔ | سیشن کے عارضی ذخیرے کی گنجائش: 16 MiB تا 8 GiB۔ ڈیفالٹ 256 MiB۔ | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/7` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | سیشن کے Scratchpad کی گنجائش: 16 MiB تا 8 GiB۔ ڈیفالٹ 256 MiB۔ | سیشن کے اسکریچ پیڈ کی گنجائش: 16 MiB تا 8 GiB۔ ڈیفالٹ 256 MiB۔ | Human explanatory noun should match the localized section; MiB/GiB remain technical units. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/47` (URB-48) |
| vi / B / semantic | major / requiresAuthorRepair | Số byte scratchpad của một phiên có thể lưu trữ: từ 16 MiB đến 8 GiB. Mặc định 256 MiB. | Số byte vùng nháp của một phiên có thể lưu: từ 16 MiB đến 8 GiB. Mặc định 256 MiB. | The human storage noun is not a protocol or code identifier; preserve the binary units and bounds. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/34` (VI-B-M1) |

### I18N-WS-5C534E296CE9 — `wire.settings.session_scratchpad_max_bytes.label`

**Source English:** Scratchpad size cap (bytes)

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3434.

**Context (existing sidecar):** {"role": "numeric setting label", "tone": "plain", "maxCells": 29, "note": "Row label; 29 cells is the settings page's label budget. It names the MAXIMUM bytes one session's scratchpad may hold (not the current size) and carries the unit because the desktop collapses help behind a disclosure (design D1). Every locale must fit the same 29-cell budget."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | Límite de scratchpad (bytes) | Límite del borrador (bytes) | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/22` (ES-A-M1.23) |
| es / B / semantic | major / requiresAuthorRepair | Límite de scratchpad (bytes) | Límite del borrador (bytes) | Human numeric row label retains scratchpad, unlike the translated explanation área de borrador; the URI scheme itself remains unchanged. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/23` (ES-B-24) |
| ur / A / semantic | major / requiresAuthorRepair | Scratchpad حد (bytes) | عارضی ذخیرے کی حد (بائٹس) | Residual English human nouns or prose remain in these fields: ANSWER, Wire, Librarian/Library, resume, Scratchpad, and notification push. They are not executable tokens here. In particular the static section titles are human labels (settings_io.py:698-701,1047-1050), not the configurable agent name. Keep the default persona name only in librarian_name.placeholder; preserve literal scratchpad:// and actual commands wherever present. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/1/entries/8` (UR-A2) |
| ur / B / semantic | major / requiresAuthorRepair | Scratchpad حد (bytes) | اسکریچ پیڈ کی حد (بائٹس) | Both Scratchpad and bytes are human label words here; preserve actual scratchpad:// URI elsewhere, not these words. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/48` (URB-49) |
| vi / B / semantic | major / requiresAuthorRepair | Giới hạn scratchpad (byte) | Giới hạn vùng nháp (byte) | Translate the same human storage noun as the section heading; the byte unit remains. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/33` (VI-B-M1) |

### I18N-WS-990B74AC45EF — `wire.settings.shell_environment_inherit.label`

**Source English:** …kept in allowlist mode

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5287.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / semantic | major / requiresAuthorRepair | …gardés en mode allowlist | …ajouts à la liste autorisée | The human row label retains the English mode name allowlist rather than naming the localized list-of-authorized-variables mode; unlike quoted configuration literals in help, this is display-only label text. Translate together with shell_environment_mode.choice.allowlist.label. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/22` (FR-B-M23) |
| hi / A / semantic | major / requiresAuthorRepair | …allowlist मोड में रखे जाते हैं | …अनुमति-सूची मोड में रखे जाते हैं | English remains in a human label/description rather than a code identifier; render the ordinary term in Hindi/Devanagari. Proposal preserves any surrounding literal identifiers. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/45` (H-A-MIX-46) |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …сохраняются в режиме allowlist | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 23 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/69` (RU-A-W1.33) |

### I18N-WS-B7F88E3953BA — `wire.settings.shell_environment_mode.choice.allowlist.label`

**Source English:** allowlist

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5278.

**Unchanged raw Choice.value:** `allowlist` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | allowlist | lista de permitidos | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/12` (ES-A-M1.13) |
| es / B / semantic | major / requiresAuthorRepair | allowlist | lista de permitidos | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/12` (ES-B-13) |
| fr / A / semantic | major / requiresAuthorRepair | allowlist | liste d’autorisation | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/7` (FR-A-M08) |
| fr / B / semantic | major / requiresAuthorRepair | allowlist | liste d’autorisation | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/13` (FR-B-M14) |
| hi / A / semantic | major / requiresAuthorRepair | allowlist | अनुमति-सूची | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/20` (H-A-EN-21) |
| hi / B / semantic | major / requiresAuthorRepair | allowlist | अनुमति-सूची | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/20` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | allowlist | разрешённый список | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/19` (RU-A-M1.20) |
| ru / B / semantic | major / requiresAuthorRepair | allowlist | разрешённый список | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/22` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | allowlist | اجازت کی فہرست | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/19` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | allowlist | اجازت کی فہرست | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/19` (URB-20) |
| vi / A / semantic | major / requiresAuthorRepair | allowlist | Danh sách cho phép | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/19` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | allowlist | Danh sách cho phép | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/19` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | allowlist | 仅允许列表 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/20` (T-ZH-A-021) |
| zh-CN / B / semantic | major / requiresAuthorRepair | allowlist | 允许列表 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/20` (B-ZH-M01.21) |

### I18N-WS-3CD94E41B2B7 — `wire.settings.shell_environment_mode.choice.inherit.label`

**Source English:** inherit

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5273.

**Unchanged raw Choice.value:** `inherit` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | inherit | heredar | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/13` (ES-A-M1.14) |
| es / B / semantic | major / requiresAuthorRepair | inherit | heredar | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/13` (ES-B-14) |
| fr / A / semantic | major / requiresAuthorRepair | inherit | hériter | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/8` (FR-A-M09) |
| fr / B / semantic | major / requiresAuthorRepair | inherit | hériter | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/14` (FR-B-M15) |
| hi / A / semantic | major / requiresAuthorRepair | inherit | मौजूदा परिवेश | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/21` (H-A-EN-22) |
| hi / B / semantic | major / requiresAuthorRepair | inherit | विरासत में लें | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/21` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | inherit | наследовать | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/20` (RU-A-M1.21) |
| ru / B / semantic | major / requiresAuthorRepair | inherit | наследовать | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/23` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | inherit | ماحول وراثت میں لیں | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/20` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | inherit | موجودہ ماحول | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/20` (URB-21) |
| vi / A / semantic | major / requiresAuthorRepair | inherit | Kế thừa | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/20` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | inherit | Kế thừa | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/20` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | inherit | 继承环境 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/21` (T-ZH-A-022) |
| zh-CN / B / semantic | major / requiresAuthorRepair | inherit | 继承 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/21` (B-ZH-M01.22) |

### I18N-WS-468A2C40CCF3 — `wire.settings.shell_environment_mode.label`

**Source English:** Agent shell environment

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5262.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Environnement du shell de l’agent | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/42` |

### I18N-WS-FDD42AD77AC6 — `wire.settings.speech.description`

**Source English:** How the assistant sounds when it speaks aloud. The descriptor is sent to the hub, which maps it onto whichever voice provider serves; fields a provider cannot express degrade with a note rather than failing. Read on the next spoken message.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1128.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / semantic | minor / rootMinorFoldOrDefer | La sonorité de l’assistant quand il parle à voix haute. Le descripteur est envoyé au hub, qui l’associe au fournisseur de voix qui sert ; les champs qu’un fournisseur ne peut pas exprimer se dégradent avec une note au lieu d’échouer. Lu au prochain message parlé. | La sonorité de l’assistant quand il parle à voix haute. Le descripteur est envoyé au hub, qui l’adapte au fournisseur de voix utilisé ; les champs non pris en charge sont ignorés avec une note plutôt que de provoquer un échec. Lu au prochain message parlé. | « au fournisseur de voix qui sert » and « les champs … se dégradent » are literal English calques and do not idiomatically describe provider capability fallback. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/25` (FR-A-m05) |

### I18N-WS-2E0C22F21B6C — `wire.settings.speech_transcription_cleanup.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6402.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/22` (H-A-EN-23) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/22` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/21` (RU-A-M1.22) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/6` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/21` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/21` (URB-22) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/21` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/21` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/22` (T-ZH-A-023) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/22` (B-ZH-M01.23) |

### I18N-WS-F8A25CCB46D7 — `wire.settings.speech_transcription_cleanup.choice.client.label`

**Source English:** client

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6412.

**Unchanged raw Choice.value:** `client` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | client | cliente | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/14` (ES-A-M1.15) |
| es / B / semantic | major / requiresAuthorRepair | client | cliente | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/14` (ES-B-15) |
| hi / A / semantic | major / requiresAuthorRepair | client | क्लाइंट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/23` (H-A-EN-24) |
| hi / B / semantic | major / requiresAuthorRepair | client | क्लाइंट | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/23` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | client | на клиенте | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/22` (RU-A-M1.23) |
| ru / B / semantic | major / requiresAuthorRepair | client | на стороне клиента | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/24` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | client | کلائنٹ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/22` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | client | کلائنٹ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/22` (URB-23) |
| vi / A / semantic | major / requiresAuthorRepair | client | Máy khách | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/22` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | client | Máy khách | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/22` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | client | 客户端 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/23` (T-ZH-A-024) |
| zh-CN / B / semantic | major / requiresAuthorRepair | client | 客户端 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/23` (B-ZH-M01.24) |

### I18N-WS-B66F879551DE — `wire.settings.speech_transcription_cleanup.choice.off.label`

**Source English:** off

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6417.

**Unchanged raw Choice.value:** `off` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | off | desactivada | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/15` (ES-A-M1.16) |
| es / B / semantic | major / requiresAuthorRepair | off | desactivada | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/15` (ES-B-16) |
| fr / A / semantic | major / requiresAuthorRepair | off | désactivé | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/3` (FR-A-M04) |
| fr / B / semantic | major / requiresAuthorRepair | off | désactivé | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/15` (FR-B-M16) |
| hi / A / semantic | major / requiresAuthorRepair | off | बंद | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/24` (H-A-EN-25) |
| hi / B / semantic | major / requiresAuthorRepair | off | बंद | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/24` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | off | отключено | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/23` (RU-A-M1.24) |
| ru / B / semantic | major / requiresAuthorRepair | off | отключено | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/26` (RU-B-M01) |
| vi / A / semantic | major / requiresAuthorRepair | off | Tắt | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/23` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | off | Tắt | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/23` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | off | 关闭 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/24` (T-ZH-A-025) |
| zh-CN / B / semantic | major / requiresAuthorRepair | off | 关闭 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/24` (B-ZH-M01.25) |

### I18N-WS-9F37CDCBD6ED — `wire.settings.speech_transcription_cleanup.choice.provider.label`

**Source English:** provider

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6407.

**Unchanged raw Choice.value:** `provider` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | provider | proveedor | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/16` (ES-A-M1.17) |
| es / B / semantic | major / requiresAuthorRepair | provider | proveedor | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/16` (ES-B-17) |
| fr / A / semantic | major / requiresAuthorRepair | provider | fournisseur | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/4` (FR-A-M05) |
| fr / B / semantic | major / requiresAuthorRepair | provider | fournisseur | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/16` (FR-B-M17) |
| hi / A / semantic | major / requiresAuthorRepair | provider | प्रोवाइडर | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/25` (H-A-EN-26) |
| hi / B / semantic | major / requiresAuthorRepair | provider | प्रोवाइडर | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/25` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | provider | у провайдера | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/24` (RU-A-M1.25) |
| ru / B / semantic | major / requiresAuthorRepair | provider | на стороне провайдера | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/25` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | provider | پرووائیڈر | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/23` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | provider | پرووائیڈر | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/23` (URB-24) |
| vi / A / semantic | major / requiresAuthorRepair | provider | Nhà cung cấp | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/24` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | provider | Nhà cung cấp | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/24` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | provider | 提供商 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/25` (T-ZH-A-026) |
| zh-CN / B / semantic | major / requiresAuthorRepair | provider | 提供商 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/25` (B-ZH-M01.26) |

### I18N-WS-635FB1A76ED6 — `wire.settings.speech_transcription_cleanup.help`

**Source English:** How dictation strips filler words and non-speech cues. auto uses the provider's native cleanup where available and the conservative client rules elsewhere; provider and client each use only one of the two mechanisms; off keeps the verbatim transcript. A per-request `cleanup` field still wins.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6398.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | डिक्टेशन भरने वाले शब्द और गैर-भाषाई संकेत कैसे हटाता है। auto उपलब्ध होने पर प्रोवाइडर की अंतर्निहित सफ़ाई और बाक़ी मामलों में सतर्क क्लाइंट नियम लागू करता है; provider और client में से प्रत्येक केवल एक तंत्र लागू करता है; off ट्रांसक्रिप्ट ज्यों का त्यों रखता है। हर अनुरोध का `cleanup` फ़ील्ड अभी भी प्राथमिकता पाता है। | डिक्टेशन में भरने वाले शब्द और गैर-भाषाई संकेत कैसे हटते हैं। स्वचालित मोड उपलब्ध होने पर प्रोवाइडर की अंतर्निहित सफ़ाई और अन्यथा सतर्क क्लाइंट नियम लागू करता है; प्रोवाइडर और क्लाइंट मोड केवल अपना तंत्र लागू करते हैं; बंद मोड ट्रांसक्रिप्ट ज्यों का त्यों रखता है। हर अनुरोध का `cleanup` फ़ील्ड अब भी प्राथमिकता पाता है। | Untranslated ordinary English verbs/nouns in human-facing prose are not machine values; localize them without changing the behavior. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/52` (H-A-MIX-53) |

### I18N-WS-11DF09DE68AD — `wire.settings.speech_voice_expressiveness.choice.low.description`

**Source English:** Restrained and plain.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6322.

**Context (existing sidecar):** {"role": "voice expressiveness choice description", "tone": "concise, plain, explanatory", "note": "A brief user-facing explanation beside the choice label, not an instruction to the synthesizer. Preserve the meaning of the English explanation. Restrained and plain delivery with little expressive variation. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `low` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | ضبط شدہ اور سادہ۔ | سادہ اور کم تاثراتی۔ | Several voice descriptions read as calques rather than natural descriptions of vocal delivery. وارم is also a transliteration of the cache-temperature term, not an idiomatic friendly voice adjective. These do not gate another remediation wave on their own. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/0/entries/4` (UR-A7) |

### I18N-WS-86C0F950D222 — `wire.settings.speech_voice_expressiveness.choice.medium.description`

**Source English:** Today's constant.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6327.

**Context (existing sidecar):** {"role": "voice expressiveness choice description", "tone": "concise, plain, explanatory", "note": "A brief user-facing explanation beside the choice label, not an instruction to the synthesizer. Preserve the meaning of the English explanation. The middle level of expressiveness; the description refers to the existing standard delivery ('Today's constant'), not a calendar date, mathematical constant or a promise that the voice never varies. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `medium` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | La constante actual. | La expresividad habitual. | La constante actual is a literal/mathematical calque, not an explanation of the middle voice delivery; verified context entry 83 explicitly means existing standard delivery, not a mathematical constant. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/26` (ES-A-M2) |
| es / B / semantic | minor / requiresAuthorRepair | La constante actual. | La expresividad habitual. | La constante actual is a literal calque that sounds mathematical and does not naturally describe the existing standard delivery identified by the context note. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/27` (ES-B-28) |
| fr / A / semantic | minor / rootMinorFoldOrDefer | La constante actuelle. | L’expressivité habituelle. | « La constante actuelle » reads as a mathematical constant; the context sidecar explicitly means the existing standard delivery, not a constant. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/22` (FR-A-m02) |
| fr / B / semantic | minor / rootMinorFoldOrDefer | La constante actuelle. | La diction habituelle. | « La constante actuelle » reads as a mathematical or configuration constant rather than the current standard vocal delivery; context entry 83 explicitly disallows that reading. This is an idiom/clarity defect, not a blocker on the two approval labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/minor_findings/0` (FR-B-m01) |
| hi / A / semantic | major / requiresAuthorRepair | मौजूदा स्थिर मान। | मौजूदा सामान्य शैली। | स्थिर मान means a fixed/mathematical value; the approved context explicitly means the existing standard speaking delivery and forbids a mathematical constant or promise of invariant speech. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/33` (H-A-SEM-1) |
| hi / B / semantic | major / requiresAuthorRepair | मौजूदा स्थिर मान। | मौजूदा सामान्य बोलने की शैली। | स्थिर मान means a fixed/constant numerical value; context entry 83 explicitly means the existing standard delivery, not a mathematical constant or invariant voice. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/34` (HI-B-M2) |
| ru / A / semantic | major / requiresAuthorRepair | Текущая константа. | Обычная подача. | «Текущая константа» reads as a mathematical/configuration constant, not ordinary existing vocal delivery; the source context explicitly excludes the mathematical-constant reading. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/32` (RU-A-M2) |
| ru / B / semantic | major / requiresAuthorRepair | Текущая константа. | Обычная подача. | «Текущая константа» reads as a current mathematical/programming constant, while sidecar entry 83 explicitly defines Today’s constant as the existing standard vocal delivery, not a mathematical constant or never-varying voice. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/1/entries/0` (RU-B-M02) |
| ur / A / semantic | major / requiresAuthorRepair | موجودہ مستقل قیمت۔ | موجودہ معمول کا انداز۔ | موجودہ مستقل قیمت means current constant price/value, not the existing standard voice delivery. Context entry 83 expressly excludes a mathematical constant or calendar interpretation; this is a voice expressiveness description shown to the person. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/4/entries/0` (UR-A5) |
| ur / B / semantic | major / requiresAuthorRepair | موجودہ مستقل قیمت۔ | موجودہ معمول کا انداز۔ | The current text means current constant price/value; the 104-entry context explicitly identifies the existing standard vocal delivery, not a mathematical constant or price. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/30` (URB-31) |

### I18N-WS-FA7340FF8D3C — `wire.settings.speech_voice_gender.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6257.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/26` (H-A-EN-27) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/26` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/25` (RU-A-M1.26) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/7` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/24` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/24` (URB-25) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/25` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/25` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/26` (T-ZH-A-027) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/26` (B-ZH-M01.27) |

### I18N-WS-C3F93F5552D5 — `wire.settings.speech_voice_gender.choice.female.label`

**Source English:** female

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6262.

**Unchanged raw Choice.value:** `female` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | female | femenina | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/17` (ES-A-M1.18) |
| es / B / semantic | major / requiresAuthorRepair | female | femenina | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/17` (ES-B-18) |
| fr / A / semantic | major / requiresAuthorRepair | female | féminine | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/1` (FR-A-M02) |
| fr / B / semantic | major / requiresAuthorRepair | female | féminine | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/17` (FR-B-M18) |
| hi / A / semantic | major / requiresAuthorRepair | female | स्त्री | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/27` (H-A-EN-28) |
| hi / B / semantic | major / requiresAuthorRepair | female | स्त्री | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/27` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | female | женский | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/26` (RU-A-M1.27) |
| ru / B / semantic | major / requiresAuthorRepair | female | женский | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/27` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | female | نسوانی | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/25` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | female | نسوانی | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/25` (URB-26) |
| vi / A / semantic | major / requiresAuthorRepair | female | Nữ | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/26` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | female | Nữ | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/26` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | female | 女声 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/27` (T-ZH-A-028) |
| zh-CN / B / semantic | major / requiresAuthorRepair | female | 女声 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/27` (B-ZH-M01.28) |

### I18N-WS-6E07BEFC2F49 — `wire.settings.speech_voice_gender.choice.male.label`

**Source English:** male

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6267.

**Unchanged raw Choice.value:** `male` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | male | masculina | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/18` (ES-A-M1.19) |
| es / B / semantic | major / requiresAuthorRepair | male | masculina | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/18` (ES-B-19) |
| fr / A / semantic | major / requiresAuthorRepair | male | masculine | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/2` (FR-A-M03) |
| fr / B / semantic | major / requiresAuthorRepair | male | masculine | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/18` (FR-B-M19) |
| hi / A / semantic | major / requiresAuthorRepair | male | पुरुष | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/28` (H-A-EN-29) |
| hi / B / semantic | major / requiresAuthorRepair | male | पुरुष | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/28` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | male | мужской | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/27` (RU-A-M1.28) |
| ru / B / semantic | major / requiresAuthorRepair | male | мужской | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/28` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | male | مردانہ | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/26` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | male | مردانہ | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/26` (URB-27) |
| vi / A / semantic | major / requiresAuthorRepair | male | Nam | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/27` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | male | Nam | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/27` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | male | 男声 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/28` (T-ZH-A-029) |
| zh-CN / B / semantic | major / requiresAuthorRepair | male | 男声 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/28` (B-ZH-M01.29) |

### I18N-WS-91C04C279DF6 — `wire.settings.speech_voice_gender.help`

**Source English:** auto picks per agent; a fixed value overrides the classifier.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6253.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | auto ہر ایجنٹ کے لیے منتخب کرتی ہے؛ مقررہ قیمت کلاسیفائر کو اووررائیڈ کرتی ہے۔ | auto ہر ایجنٹ کے لیے منتخب کرتی ہے؛ مقررہ قدر کلاسیفائر کو اووررائیڈ کرتی ہے۔ | قیمت is price, whereas these fields describe a configuration value; use قدر consistently. Comprehensible in context but a repeated false friend, not native technical Urdu. Deferred if no current major remediation touches these keys. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/1/entries/3` (UR-A8) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | auto ہر ایجنٹ کے لیے منتخب کرتی ہے؛ مقررہ قیمت کلاسیفائر کو اووررائیڈ کرتی ہے۔ | auto ہر ایجنٹ کے لیے منتخب کرتی ہے؛ مقررہ قدر کلاسیفائر کو اووررائیڈ کرتی ہے۔ | قیمت is price; the software setting is a value (قدر), consistent with قدریں elsewhere. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/9` (URB-m10) |

### I18N-WS-1406CC10A091 — `wire.settings.speech_voice_instructions.help`

**Source English:** Free text for how speech should sound (OpenAI only). Empty sends none of our own.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6385.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / B / semantic | minor / rootMinorFoldOrDefer | Văn bản tự do cho cách giọng nói cần nghe (chỉ OpenAI). Để trống sẽ không gửi hướng dẫn riêng nào. | Văn bản tự do mô tả cách thể hiện giọng nói (chỉ OpenAI). Để trống sẽ không gửi hướng dẫn riêng nào. | Cách giọng nói cần nghe literally makes the voice the listener; the source describes how generated speech should sound. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/7` (VI-B-m8) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 描述朗读应如何呈现的自由文本（仅限OpenAI）。留空则不发送任何自有内容。 | 描述朗读应如何呈现的自由文本（仅限OpenAI）。留空则不附加朗读指令。 | 自有内容 suggests proprietary content, whereas the source means no additional delivery instruction from the harness. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/37` (B-ZH-m07) |

### I18N-WS-54E1557C419A — `wire.settings.speech_voice_tone.choice.calm.description`

**Source English:** Unhurried and steady.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6300.

**Context (existing sidecar):** {"role": "voice tone choice description", "tone": "concise, plain, explanatory", "note": "A brief user-facing explanation beside the choice label, not an instruction to the synthesizer. Preserve the meaning of the English explanation. Unhurried and steady vocal delivery. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `calm` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | بے جلدی اور مستقل۔ | پرسکون اور یکساں۔ | Several voice descriptions read as calques rather than natural descriptions of vocal delivery. وارم is also a transliteration of the cache-temperature term, not an idiomatic friendly voice adjective. These do not gate another remediation wave on their own. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/0/entries/2` (UR-A7) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | بے جلدی اور مستقل۔ | ٹھہرا ہوا اور پُرسکون۔ | بے جلدی is an English-shaped construction rather than idiomatic Urdu for unhurried delivery. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/3` (URB-m4) |

### I18N-WS-6947C370C00F — `wire.settings.speech_voice_tone.choice.neutral.description`

**Source English:** Even and unmarked.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6290.

**Context (existing sidecar):** {"role": "voice tone choice description", "tone": "concise, plain, explanatory", "note": "A brief user-facing explanation beside the choice label, not an instruction to the synthesizer. Preserve the meaning of the English explanation. Even and unmarked vocal tone, without a pronounced emotional colouring. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `neutral` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | معمول اور بے امتیاز۔ | ہموار، کسی خاص جذباتی تاثر کے بغیر۔ | Several voice descriptions read as calques rather than natural descriptions of vocal delivery. وارم is also a transliteration of the cache-temperature term, not an idiomatic friendly voice adjective. These do not gate another remediation wave on their own. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/0/entries/3` (UR-A7) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | معمول اور بے امتیاز۔ | ہموار، کسی خاص تاثر کے بغیر۔ | بے امتیاز suggests indiscriminate or undistinguished; the source describes even delivery without marked emotional coloring. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/4` (URB-m5) |

### I18N-WS-B63BB1FE8701 — `wire.settings.speech_voice_tone.choice.warm.description`

**Source English:** Friendly and approachable.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6285.

**Context (existing sidecar):** {"role": "voice tone choice description", "tone": "concise, plain, explanatory", "note": "A brief user-facing explanation beside the choice label, not an instruction to the synthesizer. Preserve the meaning of the English explanation. Friendly and approachable, not a literal temperature. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `warm` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | minor / rootMinorFoldOrDefer | دوستانہ اور قریب۔ | دوستانہ اور بے تکلف۔ | Several voice descriptions read as calques rather than natural descriptions of vocal delivery. وارم is also a transliteration of the cache-temperature term, not an idiomatic friendly voice adjective. These do not gate another remediation wave on their own. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/0/entries/1` (UR-A7) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | دوستانہ اور قریب۔ | دوستانہ اور مانوس۔ | قریب means physically near/close; مانوس carries the intended approachable vocal tone. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/2` (URB-m3) |

### I18N-WS-EB1182DBD33C — `wire.settings.speech_voice_tone.choice.warm.label`

**Source English:** warm

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6284.

**Context (existing sidecar):** {"role": "voice tone choice label", "tone": "concise, adjectival", "note": "A concise adjectival choice label for the voice tone; use a natural localized adjective or level term, not an untranslated identifier. Friendly and approachable, not a literal temperature. settings_io.py:6277-6332 constructs Choice(value, localized label, localized description): these are independently localized display fields, not the stable machine value. Translate naturally; natural cognates may match English. Display only: never a provider token or input for machine parsing. Fit: settings_view.py:4219-4243 places the label and description together in the choice row; :3939-3947 truncates that row to the available list width with an ellipsis rather than wrapping. No fixed per-field cell cap is enforced; keep it compact without inventing one. The settings API exposes label and description separately for client layout."}

**Unchanged raw Choice.value:** `warm` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/48` |
| ur / A / semantic | minor / rootMinorFoldOrDefer | وارم | گرمجوش | Several voice descriptions read as calques rather than natural descriptions of vocal delivery. وارم is also a transliteration of the cache-temperature term, not an idiomatic friendly voice adjective. These do not gate another remediation wave on their own. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/0/entries/0` (UR-A7) |
| ur / B / semantic | minor / rootMinorFoldOrDefer | وارم | گرمجوش | وارم transliterates English thermal warm rather than the friendly vocal tone; گرمجوش is the natural adjective. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/minor_findings/1` (URB-m2) |

### I18N-WS-B459089265A0 — `wire.settings.speech_voice_tone.help`

**Source English:** Warmth of delivery. Honoured as a voice row or emulated in instructions.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6280.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 朗读的温暖感。会作为语音行被遵循，或在指令中模拟。 | 朗读的温暖感。通过对应声线实现，或以指令模拟。 | 语音行 is a literal rendering of a voice row rather than an intelligible description of how the chosen tone is realized. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/38` (B-ZH-m08) |

### I18N-WS-0E672AC42415 — `wire.settings.static_roots.label`

**Source English:** Extra preview roots

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5497.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Raíces extra de vistas previas | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/17` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Racines d’aperçus supplémentaires | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/43` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/49` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Дополнительные корни предпросмотра | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 34 Russian cells versus 19 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/70` (RU-A-W1.34) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Дополнительные корни предпросмотра | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/26` (RU-B-m01) |

### I18N-WS-735AA0184133 — `wire.settings.subagents_max_running.label`

**Source English:** Max background jobs

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3700.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Máx. trabajos en segundo plano | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/18` |

### I18N-WS-AFAAE6CCB801 — `wire.settings.subagents_max_team_depth.label`

**Source English:** Max nested-team depth

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3734.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Profundidad máx. de equipos anidados | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/19` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Profondeur max d’équipes imbriquées | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/44` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/50` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Макс. глубина вложенных команд | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 30 Russian cells versus 21 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/71` (RU-A-W1.35) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | نیسٹڈ ٹیم کی زیادہ سے زیادہ گہرائی | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/19` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | نیسٹڈ ٹیم کی زیادہ سے زیادہ گہرائی | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/10` |

### I18N-WS-70403DB342CD — `wire.settings.subagents_model_choice.choice.operator.description`

**Source English:** inherits the session model

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3797.

**Unchanged raw Choice.value:** `operator` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / B / semantic | major / requiresAuthorRepair | سیشن کا ماڈل inherit کرے | سیشن کا ماڈل وراثت میں حاصل کرے | An ordinary human-facing action, unit or descriptor remains Latin English; it is not an invocation, API field, named model or brand in this sentence. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/43` (URB-44) |

### I18N-WS-119A058E1283 — `wire.settings.subagents_model_choice.label`

**Source English:** Who picks a subagent's model

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3751.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Qui choisit le modèle d’un sous-agent | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/45` |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ذیلی ایجنٹ کا ماڈل کون منتخب کرے | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/20` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Ai chọn mô hình cho tác nhân con | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/18` |

### I18N-WS-A45600762CBC — `wire.settings.subagents_models_hi.help`

**Source English:** Bills at that model's rates; 'default': session; empty removes the tier

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3988.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / B / semantic | major / requiresAuthorRepair | Se factura a las tarifas de ese modelo; 'default': session; vacío elimina el nivel | Se factura a las tarifas de ese modelo; 'default': sesión; vacío elimina el nivel | The word after the literal default token explains the session model; session here is human explanatory prose, not an enum token. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/24` (ES-B-25) |

### I18N-WS-1D9B897D4754 — `wire.settings.subagents_models_lo.help`

**Source English:** Bills at that model's rates; 'default': session; empty removes the tier

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3864.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / B / semantic | major / requiresAuthorRepair | Se factura a las tarifas de ese modelo; 'default': session; vacío elimina el nivel | Se factura a las tarifas de ese modelo; 'default': sesión; vacío elimina el nivel | The word after the literal default token explains the session model; session here is human explanatory prose, not an enum token. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/25` (ES-B-26) |

### I18N-WS-BFF2CE885E58 — `wire.settings.subagents_models_med.help`

**Source English:** Bills at that model's rates; 'default': session; empty removes the tier

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 3926.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / B / semantic | major / requiresAuthorRepair | Se factura a las tarifas de ese modelo; 'default': session; vacío elimina el nivel | Se factura a las tarifas de ese modelo; 'default': sesión; vacío elimina el nivel | The word after the literal default token explains the session model; session here is human explanatory prose, not an enum token. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/26` (ES-B-27) |

### I18N-WS-1A28E4B3E568 — `wire.settings.subagents_slim_child_knowledge.label`

**Source English:** Slim child knowledge

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3711.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Recortar el conocimiento del hijo | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/20` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Alléger les connaissances des enfants | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/46` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/51` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Урезание знаний дочерних сессий | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 31 Russian cells versus 20 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/72` (RU-A-W1.36) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Урезание знаний дочерних сессий | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/27` (RU-B-m01) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Tinh gọn kiến thức tác nhân con | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/19` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Tinh gọn kiến thức tác nhân con | Gọn kiến thức tác nhân con | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/9` |

### I18N-WS-D08DC7982188 — `wire.settings.supplements_denyprefixes.label`

**Source English:** ↳ never list under

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4431.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ никогда не перечислять под | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/28` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ ان کے نیچے کبھی فہرست نہ کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/21` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ ان کے نیچے کبھی فہرست نہ کریں | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/11` |

### I18N-WS-151510AEED21 — `wire.settings.supplements_maxcostusd.label`

**Source English:** ↳ cost cap ($)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4402.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/52` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ потолок стоимости ($) | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/29` (RU-B-m01) |

### I18N-WS-060F4F9DA2FC — `wire.settings.supplements_maxfeatured.help`

**Source English:** Needs Highlights on. Files named before the rest fold into 'N more'.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4422.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| vi / A / semantic | minor / rootMinorFoldOrDefer | Cần bật Điểm nhấn. Số tệp được nêu tên trước khi phần còn lại gộp thành 'N more'. | Cần bật Điểm nhấn. Số tệp được nêu tên trước khi phần còn lại gộp thành 'N tệp khác'. | N more is human fold-label wording rather than a command or identifier; localize its words while preserving the symbolic count N. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/34` (VI-A-m1) |
| vi / B / semantic | minor / rootMinorFoldOrDefer | Cần bật Điểm nhấn. Số tệp được nêu tên trước khi phần còn lại gộp thành 'N more'. | Cần bật Điểm nhấn. Số tệp được nêu tên trước khi phần còn lại gộp thành ‘N mục khác’. | N more is an English human UI-summary phrase, not a command or config token; N must remain the count metavariable. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/minor_findings/4` (VI-B-m5) |
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 需开启亮点。先点名的文件之外的其余文件折叠为'N more'。 | 需开启亮点。超出显示数量的其余文件会折叠显示为“另有N个”。 | The quoted N more is user-visible UI wording, not a command or model token; keeping it English leaves a localization gap. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/32` (T-ZH-A-033) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | 需开启亮点。先点名的文件之外的其余文件折叠为'N more'。 | 需开启亮点。先列出这么多个文件，其余折叠为“另有N个”。 | 点名 is an unnatural verb for listing deliverable files, and N more is quoted human-facing UI text rather than a protected code token. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/39` (B-ZH-m09) |

### I18N-WS-800D8595B929 — `wire.settings.supplements_maxoutputtokens.label`

**Source English:** ↳ max output tokens

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4374.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ آؤٹ پٹ ٹوکنز | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/22` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ آؤٹ پٹ ٹوکنز | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/12` |

### I18N-WS-08A892CACAEB — `wire.settings.supplements_maxturns.label`

**Source English:** ↳ max turns

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4362.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ ٹرن | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/23` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | ↳ زیادہ سے زیادہ ٹرن | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/13` |

### I18N-WS-2849608544A5 — `wire.settings.team_bridge.title`

**Source English:** Agent messages

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1152.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/53` |

### I18N-WS-74E6FF398FD7 — `wire.settings.team_bridge_screening_ambient_budget_per_hour.help`

**Source English:** Carved out of the unaddressed ceiling, never added to the account ceiling: agent chatter in shared rooms may use at most this much of the hour.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6628.

**Context (existing sidecar):** {"note": "The point of the sentence is ORDER, not arithmetic: this slice is taken OUT of the unaddressed ceiling (only part of the account's total), and the account total is named only as the thing it is never ADDED to, so raising this slice never raises the total. Keep 'carved out of, never added to' as the shape of that claim even if the idiom changes. No numbers: do not transliterate a default. No ICU arguments.", "role": "Settings row help", "tone": "concise, neutral"}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | major / requiresAuthorRepair | غیر مخاطب کی بالائی حد میں سے نکالا گیا حصہ ہے، اس سے اکاؤنٹ کی بالائی حد میں کبھی اضافہ نہیں ہوتا: مشترکہ کمروں میں ایجنٹس کی بات چیت گھنٹے کا زیادہ سے زیادہ اتنا حصہ استعمال کر سکتی ہے۔ | یہ غیر مخاطب پیغامات کی حد میں سے مختص حصہ ہے، اکاؤنٹ کی کل حد میں اضافہ نہیں: مشترکہ کمروں میں ایجنٹس کی بات چیت کے لیے فی گھنٹہ زیادہ سے زیادہ اتنی اسکریننگ کی اجازت ہے۔ | The Urdu says a fraction of the HOUR may be used (گھنٹے کا ... حصہ), changing the unit to time. These integer settings count screening calls per rolling hour, not fractions of an hour; settings_io.py:6602-6631 declares unit="screenings". Preserve the carved-out budget and human reserve without implying elapsed-time allowance. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/5/entries/0` (UR-A6) |

### I18N-WS-2CB74583869E — `wire.settings.team_bridge_screening_unaddressed_budget_per_hour.help`

**Source English:** Carved out of the account ceiling, never added to it: everything except a person addressing your agent may use at most this much. The rest is kept for people.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6610.

**Context (existing sidecar):** {"note": "Same carved-out-of-the-ceiling claim as the ambient sibling: a slice of the account total, never an addition. 'A person addressing your agent' is a human writing to the agent directly (a DM or an @mention) — the messages the remaining slice is reserved for. No numbers: do not transliterate a default. No ICU arguments.", "role": "Settings row help", "tone": "concise, neutral"}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ur / A / semantic | major / requiresAuthorRepair | اکاؤنٹ کی بالائی حد میں سے نکالا گیا حصہ ہے، اس میں کبھی اضافہ نہیں ہوتا: آپ کے ایجنٹ کو مخاطب کرنے والے شخص کے سوا سب کچھ گھنٹے کا زیادہ سے زیادہ اتنا حصہ استعمال کر سکتا ہے۔ بقیہ حصہ لوگوں کے لیے رکھا جاتا ہے۔ | یہ اکاؤنٹ کی کل حد میں سے مختص حصہ ہے، اس میں اضافہ نہیں: آپ کے ایجنٹ کو براہ راست مخاطب کرنے والے شخص کے علاوہ باقی پیغامات کے لیے فی گھنٹہ زیادہ سے زیادہ اتنی اسکریننگ کی اجازت ہے۔ باقی گنجائش لوگوں کے لیے محفوظ ہے۔ | The Urdu says a fraction of the HOUR may be used (گھنٹے کا ... حصہ), changing the unit to time. These integer settings count screening calls per rolling hour, not fractions of an hour; settings_io.py:6602-6631 declares unit="screenings". Preserve the carved-out budget and human reserve without implying elapsed-time allowance. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/5/entries/1` (UR-A6) |

### I18N-WS-EED64BE5C06E — `wire.settings.team_bridge_session.label`

**Source English:** CoS session

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 6541.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | Sesión de CoS | Sesión de jefa de gabinete | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/25` (ES-A-M1.26) |
| fr / B / semantic | major / requiresAuthorRepair | Session CoS | Session cheffe de cabinet | CoS is the untranslated English role acronym (chief of staff), not a command, model, brand or code value; the same role is already translated as « cheffe de cabinet » in aida.description and aida_enabled.label. The French UI requirement includes this human row label. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/21` (FR-B-M22) |
| ru / A / semantic | major / requiresAuthorRepair | Сессия CoS | Сессия главы аппарата | CoS is the untranslated English human-role abbreviation Chief of Staff, not a model, brand, command or configuration value; the same role is already translated as «Глава аппарата» in aida_enabled.label. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/31` (RU-A-M3) |
| ru / B / semantic | major / requiresAuthorRepair | Сессия CoS | Сессия главы аппарата | CoS is the English abbreviation for the chief-of-staff role, not a command/model/brand or session identifier; the same namespace already translates this role as «глава аппарата». Consumer settings_io.py:6537 binds this to the human row label for team_bridge.session. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/2/entries/0` (RU-B-M03) |
| vi / B / semantic | major / requiresAuthorRepair | Phiên CoS | Phiên chánh văn phòng | CoS is an untranslated human role abbreviation; the locale already uses chánh văn phòng in aida.description and aida_enabled.label. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/35` (VI-B-M1) |
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | CoS会话 | 幕僚长会话 | CoS is an unexplained English abbreviation for the same role already labelled 幕僚长 in this namespace. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/33` (T-ZH-A-034) |
| zh-CN / B / semantic | minor / rootMinorFoldOrDefer | CoS会话 | 幕僚长会话 | CoS is an untranslated generic role abbreviation; the same role is already named 幕僚长 in aida_enabled.label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/42` (B-ZH-m12) |

### I18N-WS-AB916644DEFA — `wire.settings.tool_approval_mode.choice.ask.label`

**Source English:** Ask

**Deduplicated source severity:** nit. **Usage:** `local_operator/settings_io.py` 3398.

**Context (existing sidecar):** {"role": "approval mode dropdown choice label", "tone": "concise, neutral, sentence case", "note": "Human-facing name for the prompting mode, not the ask tool or a command. Translate naturally as a mode label; do not preserve the raw English enum token. Choice.value remains the machine value ask, while Choice.label is independently localized display copy. Ask mode prompts for write and command tools, not unconditionally for every tool call. Existing guardrail risk checks and mandatory human-approval requirements still apply in both modes. This label is neither authorization nor permission to execute a tool. Keep it compact; no ICU arguments. See settings_io.py's tool_approval_mode choices, harness/approval.py and guardrails/check.py."}

**Unchanged raw Choice.value:** `ask` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/54` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | спрашивать | — | Native phrasing is accurate; retain unless a rendered fit check requires a shorter natural phrase. No hard maxCells exists for this choice. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/35` (RU-B-m01) |
| vi / A / semantic | nit / rootMinorFoldOrDefer | hỏi trước | Hỏi trước | Meaning is correct and makes no always/every-tool promise, but lowercase hỏi is inconsistent with the new sentence-case context and neighboring Tự động. Non-gating; do not open a separate style wave. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/37` (VI-A-n1) |
| vi / A / width risk; not observed clipping | minor / rootMinorFoldOrDefer | hỏi trước | — | Three times the English width, but no fixed choice-label cap in context; literal formatter success is not layout evidence. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/28` |

### I18N-WS-18458346AC28 — `wire.settings.tool_approval_mode.choice.auto.label`

**Source English:** Automatic

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3403.

**Context (existing sidecar):** {"role": "approval mode dropdown choice label", "tone": "concise, neutral, sentence case", "note": "Human-facing name for the automatic tool-approval mode. Translate naturally as a mode label, not as an untranslated machine token; use an idiomatic local abbreviation only if natural. Choice.value remains the machine value auto, while Choice.label is independently localized display copy. Automatic does not mean every action is allowed: existing guardrail risk checks and mandatory human-approval requirements still apply, including person-only gates for account profile saves and community posts. Do not imply a guardrail bypass, unconditional approval, authorization or permission to execute a tool. Keep it compact; no ICU arguments. See settings_io.py's tool_approval_mode choices, harness/approval.py and guardrails/check.py."}

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | автоматически | — | Native phrasing is accurate; retain unless a rendered fit check requires a shorter natural phrase. No hard maxCells exists for this choice. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/36` (RU-B-m01) |

### I18N-WS-59EE2D02BF11 — `wire.settings.tool_approval_mode.label`

**Source English:** Tool approval mode

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3372.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Modo de aprobación de herramientas | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/21` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/55` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Режим одобрения инструментов | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/30` (RU-B-m01) |

### I18N-WS-1323A2E96834 — `wire.settings.tools_defer.label`

**Source English:** Load rarely used tool schemas on demand

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5115.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Cargar los esquemas de herramientas poco usadas a demanda | Cargar esquemas bajo demanda | The native label is 57 cells versus English 39; materially expanded relative to the documented 29-cell settings label column. Shorten or perform a rendered fit check; no clipping was observed here. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/30` (ES-B-31) |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Charger à la demande les schémas des outils rarement utilisés | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/47` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/56` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Загружать схемы редких инструментов по требованию | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 49 Russian cells versus 39 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/73` (RU-A-W1.37) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | کم استعمال ٹول اسکیما ضرورت پر لوڈ کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/24` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Nạp lược đồ công cụ ít dùng khi cần | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/20` |

### I18N-WS-22AA3AA1C1DB — `wire.settings.tools_search_interception_block.label`

**Source English:** ...refuse rather than warn

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5092.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …отклонять, а не предупреждать | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 30 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/74` (RU-A-W1.38) |

### I18N-WS-8E352ACD1AAA — `wire.settings.tools_search_interception_enabled.label`

**Source English:** Block unbounded searches

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5084.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Bloquer les recherches non bornées | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/48` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Блокировать неограниченные поиски | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 24 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/75` (RU-A-W1.39) |

### I18N-WS-6CF6A86C9157 — `wire.settings.tools_search_interception_rg_excludes.label`

**Source English:** ...prune vendor trees for ripgrep

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 5101.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …élaguer les arborescences vendor pour ripgrep | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/49` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …обрезать деревья vendor для ripgrep | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 36 Russian cells versus 33 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/76` (RU-A-W1.40) |
| ur / A / semantic | major / requiresAuthorRepair | ...ripgrep کے لیے وینڈر درخت حذف کریں | ...ripgrep میں وینڈر فولڈر چھوڑیں | The label says to DELETE vendor trees (حذف کریں), but the setting only excludes directories from ripgrep searches. This wrongly presents a nondestructive performance option as filesystem deletion; the paired help explicitly says the generated config skips node_modules/.git/out. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/2/entries/0` (UR-A3) |
| ur / B / semantic | major / requiresAuthorRepair | ...ripgrep کے لیے وینڈر درخت حذف کریں | ...ripgrep میں وینڈر فولڈر چھوڑیں | حذف کریں says delete vendor trees; this toggle only writes a ripgrep exclusion configuration and does not delete files (settings_io.py:5098–5105; tools/builtin.py:2747–2763,5093–5099). — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/31` (URB-32) |
| ur / A / width risk; not observed clipping | minor / requiresAuthorRepair | ...ripgrep کے لیے وینڈر درخت حذف کریں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/25` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | …cắt tỉa cây vendor cho ripgrep | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/21` |

### I18N-WS-D87EDC6ADD3D — `wire.settings.tui_sidebar_position.label`

**Source English:** Sidebar position

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3309.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/57` |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Положение боковой панели | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/31` (RU-B-m01) |

### I18N-WS-3702DDE4470F — `wire.settings.tui_sidebar_show_subagents.label`

**Source English:** Sidebar subagent layer

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3330.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Capa de subagentes en la barra lateral | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/22` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Couche sous-agents de la barre latérale | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/50` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/58` |
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Слой субагентов на боковой панели | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 33 Russian cells versus 22 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/77` (RU-A-W1.41) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Слой субагентов на боковой панели | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/32` (RU-B-m01) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Lớp tác nhân con trong thanh bên | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/22` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Lớp tác nhân con trong thanh bên | Tác nhân con ở thanh bên | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/10` |

### I18N-WS-FC853A7D900F — `wire.settings.tui_sidebar_visible.label`

**Source English:** Session sidebar

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 3296.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/59` |

### I18N-WS-F02BAB48A9EA — `wire.settings.update_auto_prune.label`

**Source English:** Prune old installs

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 5649.

**Context (existing sidecar):** {"role": "settings row label in the updates section", "tone": "terse, Sentence case (first word capitalised) like its neighbours", "width": "short label column; painted in a 29-cell column; keep <=29 cells in every locale", "note": "An on/off switch: delete old unused install generations automatically after each update."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/60` |

### I18N-WS-13C42F0212CF — `wire.settings.updates.title`

**Source English:** Updates

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1005.

**Context (existing sidecar):** {"role": "settings section title", "tone": "terse", "width": "short section header", "note": "The section for what a local update does with older install generations on THIS device; not the mesh roll (that is 'Mesh updates')."}

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/61` |

### I18N-WS-1B300EB8E36D — `wire.settings.wakes.title`

**Source English:** Wake triggers

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 1104.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/62` |

### I18N-WS-968C517CC557 — `wire.settings.wakes_triggers_max_per_day.label`

**Source English:** Max check-in wakes per day

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6144.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Макс. пробуждений для выхода на связь в день | Rendered check required; retain only if complete meaning fits, otherwise shorten this label without dropping its unit or qualifier. | Relative-width risk: 44 Russian cells versus 26 English cells and the common 29-cell TUI label column. No clipping was observed; this is not rendered evidence. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/78` (RU-A-W1.42) |
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Макс. пробуждений для выхода на связь в день | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/33` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | روزانہ رابطے کی زیادہ سے زیادہ بیداریاں | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/26` (UR-A9) |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Số lần thức dậy hỏi thăm tối đa mỗi ngày | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/23` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Số lần thức dậy hỏi thăm tối đa mỗi ngày | Lần thức dậy hỏi thăm/ngày | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/11` |

### I18N-WS-3919FF1D7675 — `wire.settings.wakes_triggers_project_staleness_enabled.help`

**Source English:** Wake to check in on projects whose progress went stale. Settled (done/paused/archived) projects are never watched.

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6173.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| zh-CN / A / semantic | minor / rootMinorFoldOrDefer | 项目进度停滞时唤醒去检查。已结项（done/paused/archived）的项目从不监视。 | 项目进度停滞时唤醒去检查。已完成、已暂停或已归档（done/paused/archived）的项目从不监视。 | 已结项 means completed/closed, whereas the English explicitly also excludes paused projects; the raw state identifiers are preserved technical tokens but do not make the Chinese collective label accurate. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/36` (T-ZH-A-037) |

### I18N-WS-3AD1E7F5FDB3 — `wire.settings.wakes_triggers_project_staleness_enabled.label`

**Source English:** Project staleness trigger

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 6166.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Disparador de obsolescencia de proyectos | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/23` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Déclencheur de projets périmés | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/51` |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | پروجیکٹ کی پرانی پیش رفت کا محرک | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/27` (UR-A9) |

### I18N-WS-BEDE366BEEA3 — `wire.settings.web_fetch.title`

**Source English:** Web fetch

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 843.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/63` |

### I18N-WS-6AD75056C0E6 — `wire.settings.web_fetch_allow_private.label`

**Source English:** Allow private addresses

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4885.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Autoriser les adresses privées | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/52` |

### I18N-WS-83E8CB1A2107 — `wire.settings.web_fetch_blocked_retry.choice.false.description`

**Source English:** stay self-identifying

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4949.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / A / semantic | minor / rootMinorFoldOrDefer | оставаться узнаваемым | явно представляться | «Оставаться узнаваемым» means remaining recognizable, not explicitly identifying the fetch client instead of impersonating browser headers; the adjacent help establishes that contrast. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/36` (RU-A-m4) |

### I18N-WS-E679E75CBC45 — `wire.settings.web_fetch_blocked_retry.label`

**Source English:** Browser-profile retry

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4943.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Reintento con perfil de navegador | — |  — `i18n/audits/evidence/approval-full-es-A-original.json#/width_flags/24` |
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Réessai avec profil de navigateur | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/53` |
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/64` |
| vi / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Thử lại theo hồ sơ trình duyệt | — | Exceeds the settings page’s documented generic 29-cell row-label budget; no key-specific maxCells metadata. Some English labels already exceed it; do not infer a new clipping regression. — `i18n/audits/evidence/approval-full-vi-A-original.json#/overflow_flags/24` |
| vi / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Thử lại theo hồ sơ trình duyệt | Thử lại như trình duyệt | Material expansion and >29 cells: request rendered check; no observed clipping is claimed. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/width_risk_group/entries/12` |

### I18N-WS-DD4E6CB4908B — `wire.settings.web_fetch_enabled.label`

**Source English:** Web fetch

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4827.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/65` |

### I18N-WS-ACD7EB26AA26 — `wire.settings.web_fetch_max_bytes.label`

**Source English:** Download ceiling (bytes)

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4854.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Plafond de téléchargement (octets) | — |  — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/width_risks/54` |

### I18N-WS-FC2F29420943 — `wire.settings.web_fetch_max_redirects.label`

**Source English:** Max redirects

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4864.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| ru / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | Макс. перенаправлений | — | Material relative expansion; request a rendered narrow-layout check, not a claim of clipping. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/3/entries/34` (RU-B-m01) |
| ur / A / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ ری ڈائریکٹس | Rendered check required; preserve meaning while shortening if the actual control truncates. | Material source-relative expansion or >29-cell general TUI label-column risk; not observed clipping, not a native RTL fit claim. — `i18n/audits/evidence/approval-full-ur-A-original.json#/minor_findings/2/entries/28` (UR-A9) |
| ur / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | زیادہ سے زیادہ ری ڈائریکٹس | — |  — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/width_flags/14` |

### I18N-WS-7AB27EDC4AA9 — `wire.settings.web_fetch_render_backend.choice.auto.label`

**Source English:** auto

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4905.

**Unchanged raw Choice.value:** `auto` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | auto | स्वचालित | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/29` (H-A-EN-30) |
| hi / B / semantic | major / requiresAuthorRepair | auto | स्वचालित | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/29` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | auto | автоматически | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/28` (RU-A-M1.29) |
| ru / B / semantic | major / requiresAuthorRepair | auto | авто | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/8` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | auto | خودکار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/27` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | auto | خودکار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/27` (URB-28) |
| vi / A / semantic | major / requiresAuthorRepair | auto | Tự động | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/28` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | auto | Tự động | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/28` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | auto | 自动 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/29` (T-ZH-A-030) |
| zh-CN / B / semantic | major / requiresAuthorRepair | auto | 自动 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/29` (B-ZH-M01.30) |

### I18N-WS-2059EB7BAFBB — `wire.settings.web_fetch_render_backend.help`

**Source English:** Auto uses markdownify when the [fetch] extra is installed.

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4901.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / B / semantic | major / requiresAuthorRepair | Auto, [fetch] extra इंस्टॉल होने पर markdownify इस्तेमाल करता है। | स्वचालित मोड में, [fetch] extra इंस्टॉल होने पर markdownify इस्तेमाल होता है। | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/33` (HI-B-M1) |

### I18N-WS-07623CEB6970 — `wire.settings.web_fetch_render_backend.label`

**Source English:** HTML renderer

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4898.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/66` |

### I18N-WS-6B957C820F81 — `wire.settings.web_search_read_enabled.label`

**Source English:** Read from search

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4814.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/67` |

### I18N-WS-8377F1BF1C46 — `wire.settings.web_search_searxng_endpoint.label`

**Source English:** SearXNG endpoint

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4792.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| hi / A / semantic | major / requiresAuthorRepair | SearXNG endpoint | SearXNG एंडपॉइंट | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/30` (H-A-EN-31) |

### I18N-WS-21B6A15BD737 — `wire.settings.web_search_strategy.choice.ordered.label`

**Source English:** ordered

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4722.

**Unchanged raw Choice.value:** `ordered` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | ordered | por orden | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/19` (ES-A-M1.20) |
| es / B / semantic | major / requiresAuthorRepair | ordered | por orden | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/19` (ES-B-20) |
| fr / A / semantic | major / requiresAuthorRepair | ordered | par priorité | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/5` (FR-A-M06) |
| fr / B / semantic | major / requiresAuthorRepair | ordered | par priorité | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/19` (FR-B-M20) |
| hi / A / semantic | major / requiresAuthorRepair | ordered | सूची के क्रम में | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/31` (H-A-EN-32) |
| hi / B / semantic | major / requiresAuthorRepair | ordered | क्रमानुसार | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/30` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | ordered | по порядку | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/29` (RU-A-M1.30) |
| ru / B / semantic | major / requiresAuthorRepair | ordered | по порядку | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/29` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | ordered | ترتیب وار | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/28` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | ordered | ترتیب وار | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/28` (URB-29) |
| vi / A / semantic | major / requiresAuthorRepair | ordered | Theo thứ tự | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/29` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | ordered | Theo thứ tự | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/29` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | ordered | 按顺序 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/30` (T-ZH-A-031) |
| zh-CN / B / semantic | major / requiresAuthorRepair | ordered | 按顺序 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/30` (B-ZH-M01.31) |

### I18N-WS-3886CD8B7B42 — `wire.settings.web_search_strategy.choice.round_robin.label`

**Source English:** round_robin

**Deduplicated source severity:** major. **Usage:** `local_operator/settings_io.py` 4717.

**Unchanged raw Choice.value:** `round_robin` (AST-derived; not the human label).

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| es / A / semantic | major / requiresAuthorRepair | round_robin | rotación | Human-facing display copy retains English rather than Spanish; Choice.value and API/config identifiers remain unchanged. This violates the operator entire-UI/no-English-human-label acceptance criterion. — `i18n/audits/evidence/approval-full-es-A-original.json#/auditor/findings/20` (ES-A-M1.21) |
| es / B / semantic | major / requiresAuthorRepair | round_robin | rotación | Consumer constructs Choice(machine value, localized label, localized description); this untranslated English label is human-facing, not a parser input or a brand. — `i18n/audits/evidence/approval-full-es-B-original.json#/findings/20` (ES-B-21) |
| fr / A / semantic | major / requiresAuthorRepair | round_robin | à tour de rôle | Untranslated generic human-facing choice label; settings_io constructs the localized label separately from the stable machine value, so translating this display field does not change the API/configuration token. — `i18n/audits/evidence/approval-full-fr-A-completion-original.json#/findings/6` (FR-A-M07) |
| fr / B / semantic | major / requiresAuthorRepair | round_robin | à tour de rôle | Untranslated English in a human-facing choice label; settings_io.py constructs Choice(machine_value, independently localized label, description), so preserving the machine enum does not justify leaving its displayed label in English. Violates the requested fully French human labels. — `i18n/audits/evidence/approval-full-fr-B-original.json#/auditor/findings/20` (FR-B-M21) |
| hi / A / semantic | major / requiresAuthorRepair | round_robin | बारी-बारी | Untranslated human-facing display label, not the separately stored machine enum; violates the explicit entire-UI Hindi acceptance criterion. — `i18n/audits/evidence/approval-full-hi-A-original.json#/findings/32` (H-A-EN-33) |
| hi / B / semantic | major / requiresAuthorRepair | round_robin | बारी-बारी | Human-facing display copy retains an ordinary English mode, boolean, action or descriptive term; separate Choice.value is the machine token and can remain unchanged. This violates the entire-UI localization acceptance criterion. — `i18n/audits/evidence/approval-full-hi-B-original.json#/findings/31` (HI-B-M1) |
| ru / A / semantic | major / requiresAuthorRepair | round_robin | по кругу | Untranslated English enum token is used as an independently localized human-facing Choice.label; the stable Choice.value can and must remain unchanged. This violates the requested no-English human-label acceptance criterion, not the technical-token exception. — `i18n/audits/evidence/approval-full-ru-A-original.json#/findings/30` (RU-A-M1.31) |
| ru / B / semantic | major / requiresAuthorRepair | round_robin | по кругу | Untranslated human-facing mode/option label, not its machine value. — `i18n/audits/evidence/approval-full-ru-B-original.json#/findings/0/entries/30` (RU-B-M01) |
| ur / A / semantic | major / requiresAuthorRepair | round_robin | باری باری | These 30 independently localized Choice.label fields remain English enum spellings; they are human dropdown labels, not Choice.value machine tokens. This violates the explicit whole-UI acceptance criterion and drifts from the new approval label خودکار. Translate labels without changing wire values. — `i18n/audits/evidence/approval-full-ur-A-original.json#/auditor/findings/0/entries/29` (UR-A1) |
| ur / B / semantic | major / requiresAuthorRepair | round_robin | باری باری | This is the localized Choice.label displayed to people, not Choice.value; the raw English enum is not a brand or a technical-only field. — `i18n/audits/evidence/approval-full-ur-B-completed-original.json#/auditor/findings/29` (URB-30) |
| vi / A / semantic | major / requiresAuthorRepair | round_robin | Luân phiên | Human-facing Choice.label is independently localized from Choice.value in settings_io.py; the historical style-guide token exception does not satisfy the operator’s explicit no-English human-label requirement. Leave machine values unchanged. — `i18n/audits/evidence/approval-full-vi-A-original.json#/findings/30` (VI-A-M1) |
| vi / B / semantic | major / requiresAuthorRepair | round_robin | Luân phiên | This is localized Choice.label display copy, not Choice.value or a typed config token; the operator explicitly disallows the historical blanket enum-label exemption. — `i18n/audits/evidence/approval-full-vi-B-original.json#/current_semantic_findings/major_groups/0/entries/30` (VI-B-M1) |
| zh-CN / A / semantic | major / requiresAuthorRepair | round_robin | 轮换 | This is a localized Choice.label displayed to people, not the separate stable Choice.value or a model-facing instruction; retaining English fails the all-human-UI-localized requirement. — `i18n/audits/evidence/approval-full-zh-CN-A-original.json#/findings/31` (T-ZH-A-032) |
| zh-CN / B / semantic | major / requiresAuthorRepair | round_robin | 轮询 | This is a human-facing choice label, not the Choice.value/config token; leaving the ordinary mode/level word in English fails the requested complete Chinese UI. Keep the machine value unchanged and align user-facing help references with the localized label. — `i18n/audits/evidence/approval-full-zh-CN-B-original.json#/unchanged_namespace_findings/31` (B-ZH-M01.32) |

### I18N-WS-CE5A2B749A09 — `wire.settings.web_search_strategy.label`

**Source English:** Load balancing

**Deduplicated source severity:** minor. **Usage:** `local_operator/settings_io.py` 4710.

| Locale / lane | Source severity / disposition | Current (auditor quote) | Proposed (auditor quote) | Reason and exact source |
|---|---|---|---|---|
| fr / B / width risk; not observed clipping | minor / widthOnlyNotObservedClipping | — | — |  — `i18n/audits/evidence/approval-full-fr-B-original.json#/relative_width_risks/68` |


# Round 2 frozen exit / authority matrix

Pinned K: `4880031aeed59ef17ae6ffccc15d28e1fc0db104`. This is evidence integration, not a new semantic audit.

| ID | Locale / key | Failing state | Intended transition | Authority |
|---|---|---|---|---|
| R2-M01 | es / `wire.settings.providers_openrouter_allow_fallbacks.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M02 | es / `wire.settings.providers_openrouter_data_collection.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M03 | es / `wire.settings.providers_openrouter_enforce_distillable_text.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M04 | es / `wire.settings.providers_openrouter_require_parameters.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M05 | es / `wire.settings.providers_openrouter_sort.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M06 | es / `wire.settings.providers_openrouter_zdr.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M07 | es / `wire.settings.shell_environment_mode.help` | human help references raw English options rather than actual visible localized Choice.label | Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label | Fresh LLM author `87094744ea9d`, independent A/B delta auditors |
| R2-M08 | vi / `wire.settings.display_nerd_icons.help` | changed visible human help retains ordinary English terminal without named exception | Fresh Vietnamese LLM translator replaces ordinary English terminal with native term; technical IDs unchanged | Fresh LLM author `f0f530be0d7f`, independent A/B delta auditors |

## Exact exit conditions

1. Only ES seven failed helpers and VI four helpers are authored; preserve stable wire/config IDs, CLDR, RTL and pseudo behavior.
2. Human option references agree with actual current Choice.label. Quoted source proposals never override captured labels. VI ordinary terminal is localized; endpoint and actual pip extra [fetch] retain named technical spelling.
3. Independent A/B both pass the exact ES/VI changed helpers and direct neighbors at one actual pinned K2; retained records are explicitly not fresh reads.
4. Root proves K..K2 byte/source/context compatibility for all five clean locale pairs and unchanged English/context source. No clean-language re-audit for unchanged data.
5. Preserve failed original R1/R2 verdicts. Resolve all eight accepted majors; fold the three VI minors into the same existing failed-stream wave, never a new minor cycle.
6. Beyond three waves or approximately 2x review-driven growth, split follow-on scope without waiving unresolved majors.

Machine-readable scope, proposal citations and authority are in `approval-repair-r2-exit-matrix.json`; complete provenance is in the ledger and retention manifest. No native/rendered/shipping pass is claimed.

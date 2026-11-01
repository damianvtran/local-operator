# Spanish core literal translation

Translator session: 15ef8fd56f9d. Actual completing model: openai/gpt-6.1-sol, temporary provider-quota fallback from anthropic/claude-sonnet-5-5 (an intervening claude-opus-5-5 switch was also announced). No translation service used; no independent audit claimed.

## Files and keys

All paths below are relative to `/Users/damian/local-operator/.worktrees/lit-translate/`.

- `local_operator/i18n/catalogues/es/cli.failures.json` — 4 keys: `cli.failures.error_line`, `cli.failures.required`, `cli.failures.invalid_choice`, `cli.failures.invalid_command`.
- `local_operator/i18n/catalogues/es/cli.group_help.json` — 10 keys: `cli.group_help.skills`, `cli.group_help.skills_description`, `cli.group_help.tunnel`, `cli.group_help.secret`, `cli.group_help.network`, `cli.group_help.disk`, `cli.group_help.operator`, `cli.group_help.pair`, `cli.group_help.search`, `cli.group_help.fetch`.
- `local_operator/i18n/catalogues/es/tui.setup_cues.json` — 2 keys: `tui.setup_cues.setup_placeholder`, `tui.setup_cues.no_provider_cue`.

The three Spanish files already existed at first inspection. Read all source strings and sidecars, the glossary, the full Spanish style guide, and the RFC catalogue contract; retained the suitable existing Spanish translations and tightened the disk summary to make verified deletion safety explicit. Rewrote only these three catalogue files in canonical UTF-8 JSON, two-space indentation, one trailing newline.

## Terminology and context choices

- `error` stays `error`: the same word is idiomatic Spanish, and the wrapper shape is unchanged.
- CLI `argument` → `argumento`; `invalid choice` → `opción no válida`. Interpolated quoted choices and all other placeholders remain verbatim.
- `pull` and `harnesses` remain English per the Spanish guide; skills are `habilidades`, packaged bundles `paquetes`.
- `without a prompt` → `sin preguntarte`, following the sidecar's meaning (no repeated user question), not a model input prompt.
- `mesh networks` → `redes malladas`; `load-balanced` → `con balanceo de carga`; `headless web fetching` explicitly describes a `navegador sin ventana`.
- `approval gate` → `control de aprobación`; the operator key can `flexibilizar` it, not silently disable it.
- Disk safety: `medir primero y borrar solo con seguridad comprobada` retains both measurement first and verified safety before deletion.
- Pairing authorization stays `una sola vez`; AI account is `cuenta de IA`.

Glossary proposals (not edits): bundle → paquete; mesh network → red mallada; approval gate → control de aprobación; load balancing → balanceo de carga. These terms were absent from the shared glossary; the renderings retained here are normal Spanish technical usage.

## Verification performed

Executed a lightweight Python JSON/placeholder check using the repository's existing virtual environment (not the catalogue validator or a test suite). Actual output:

```text
cli.failures: 4 keys; English order and exact placeholder sequence preserved; canonical JSON written
cli.group_help: 10 keys; English order and exact placeholder sequence preserved; canonical JSON written
tui.setup_cues: 2 keys; English order and exact placeholder sequence preserved; canonical JSON written
tui.setup_cues.setup_placeholder: 35 cells — Escribe /login radient para empezar
tui.setup_cues.no_provider_cue: 56 cells — Conecta primero una cuenta de IA: escribe /login radient
```

Measured with `rich.cells.cell_len` after substituting `/login radient`. No-provider cue: **56/74 cells**, 18 spare. No cell-budget exceptions.

Catalogue validator, generator, tests and git commands were not run, as explicitly instructed. Independent audit remains the manager's next step. No ledger, generated data, allowance, other locale or code was edited.

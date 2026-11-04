# Audit report — es / wire.errors.hosting / hosting-es-r1

- **Locale / namespace:** `es` / `wire.errors.hosting` — **1 key** (`wire.errors.hosting.not_configured`).
- **Base:** `87ca846245b2`; the English namespace is uncommitted and **absent at base**, not shipped.
- **Source:** `/Users/damian/.local-operator/sessions/3ef6295a5545/scratchpad/hosting-source-v1`
  - English `wire.errors.hosting.json` — `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` (82 bytes)
  - Context `wire.errors.hosting.context.json` — `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` (854 bytes)
- **Target:** `/Users/damian/.local-operator/sessions/804764f54d82/scratchpad/hosting-es/wire.errors.hosting.json`
  `8de6f07aa22bb654bc0e45390bc803ba28f37698e6e08ffe95399d895ff3c5cf` (97 bytes)
- **Auditor:** `i18n-auditor`, session `f50d87966867`; requested model **anthropic/claude-sonnet-5-5**; served model
  recorded from the session model header (**anthropic/claude-sonnet-5-5**); no fallback observed.
  Independent of translator session `804764f54d82` and code author `94dcef624dec`; no self-translation,
  no MT service, no other reader's verdict read before this verdict.

## Verdict

**`passed`** — 0 findings, 0 unresolved BLOCKER/MAJOR. Source-only linguistic verdict; the native,
rendered and consumer-proof holds below are separate and are **not** claimed.

## The key

| Key | English | Spanish |
|---|---|---|
| `wire.errors.hosting.not_configured` | `Hosting platform is not configured.` | `La plataforma de alojamiento no está configurada.` |

The Spanish mirrors the source exactly in meaning: it names the **owned condition** (the hosting
platform is not configured) and adds **no remedy, no provider name, no dynamic detail** and no
opaque external-error text. `plataforma` is feminine and `configurada` agrees; the passive
`no está configurada` renders the source's state, not an event (`no se ha configurado` would shift
the aspect).

## Mechanical checks (executed with pure stdlib)

- Valid JSON object; **one key, exact parity** with the English source; no added/dropped key.
- Value is 49 Unicode codepoints; **NFC true**; **`Cf` = 0**; **manual bidi controls = 0**; no brace
  characters and **zero ICU arguments** (the key takes no params, so placeholder/CLDR categories
  are **N/A** by construction — no argument means no plural/select structure).
- **Capital initial** and **terminal full stop** present, matching the context's requirement for a
  complete statement spliced after a colon.
- No English fallback in the target value; the only English text in this packet is the English
  source and the legacy-exception caveat below.
- `wire.errors.media.radient_key_missing` is the same sentence pattern:
  `La clave de API de Radient no está configurada en el servidor.` — the capital + `no está
  configurada` + terminal period are established in this catalogue.

## Terminology (corpus-checked, not uniform-wording inference)

- `cli.global_help.hosting`: `Plataforma de alojamiento que usar (radient, deepseek, openai, …)`
- `cli.global_help.propagated_hosting`: `Plataforma de alojamiento para esta ejecución (…)`
- `cli.exec_worker_help.hosting`: `Anulación de la plataforma de alojamiento`
- `cli.credential_help.credential`: `… credenciales de distintas plataformas de alojamiento`
- `wire.settings.hosting.label` is `Proveedor predeterminado` — that names the settings picker's
  default provider, a **different concept**; it is correctly not used here.
- `tui.authoring.invalid_effort`: `Ese nivel de esfuerzo no está configurado.` — the same state
  pattern with a different subject.

`plataforma de alojamiento` is the shipped Spanish term for the source's “hosting platform”; no
English `hosting` display copy or calque is introduced.

## Advisories (not copy findings)

1. **Legacy exception stays English by contract.** `session_factory.HostingNotConfiguredError`
   carries the English sentence as its `str`; this key is display copy only and does not edit, wrap
   or translate the exception, its args, its type or any caller. A surface that still paints
   `str(error)` shows English until the caller migration consumes this key. This audit makes **no
   localized-exception claim**.
2. **Staged, uncommitted source.** The namespace is absent at base `87ca846245b2` and is not
   shipped, generated, activated or ledgered. This verdict covers the exact staged bytes above.
3. **Term choice is corpus-backed.** Using `plataforma` here is not an inference from another
   locale's wording; it is the current Spanish hosting corpus.
4. **No ledger/activation/PR comment.** The record and report are the canonical delivery for this
   staged copy; no ledger row was written per the task.

## Native / render holds (separate; NOT RUN)

- Native-speaker review: **NOT RUN**.
- Rendered frame / consumer integration: **NOT RUN** — this is a display key whose caller wiring is
  a separate migration; no visible-surface claim.
- Pseudo-localisation widening: **NOT RUN**.
- RTL: **N/A** (`es` is LTR); no RTL claim.
- Runtime/consumer behaviour: **NOT RUN**; source-only verdict.

## Artifacts

- `record.json` and `report.md` in this directory (new `hosting-es-audit-reader1` scratch); hashes
  are recorded in `record.json` and `receipt.json` if present.
- Prior TUI chrome R1/R2 records, SDK packets and the census commit are **untouched**.

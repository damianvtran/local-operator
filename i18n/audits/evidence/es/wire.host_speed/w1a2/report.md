# Spanish audit — wire.host_speed

Auditor: translation-auditor es-w1-auditor-2 / session 1abd0b38e62a; independent of the translator.
Model: **radient/auto** (self-reported session header; fallback chain anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota HTTP 429"). `radient/auto` is an auto-routing alias, so the concrete backend is not identified by the header.

Audit id `w1a2`, round 1, 2026-10-29. Audited revision `efcfbe812609`; the es catalogue is **untracked** at that revision, so this audit binds the on-disk bytes (`1c2281ae22b7ed442dede47aa12323e3ea87a55ba2e7fd4ad7b49059209d5f17`).

Verdict: **passed** (clean-with-findings); 2/2 keys read; 0 blocker, 0 major, 1 minor, 1 nit.

## What was verified

Both keys read: the experimental and off-by-default disclaimer, the one-model/one-run evidence caveat, the three-long-replies trigger, the positive-floor semantics, the 0-clears-speed-exclusions-while-cache-miss-exclusions-remain behaviour and the cold-cache warning all survive. The label measures 26 cells against the sidecar's 29-cell budget (PASS) and 'tok/s' is byte-identical. One minor idiom finding, one nit.

Mechanical checks ran per key: key-set parity, ICU brace/argument parity, placeholder parity, digit-sequence parity, en/em-dash and NBSP parity, byte-identical machine literals, width against sidecar `maxCells` where a budget exists, unsupported es plural categories, untranslated-value detection and whitespace/quote/exclamation hygiene. All passed; the detectors were proven to fire on deliberately broken copies (negative controls in `audit.log`).

## Findings

### es-w1-B02 — MINOR

Keys: `wire.host_speed.retire_below_tps.help`

Current:

```
Experimental, desactivado por defecto: la evidencia proviene de un solo modelo y una sola ejecución. Un umbral positivo deja de fijar un host tras tres respuestas largas por debajo de esta tasa de tokens de salida por segundo, sin contar el tiempo de consumo local. 0 borra las exclusiones por velocidad en la siguiente petición; las exclusiones por fallo de caché se mantienen. Cambiar de host puede iniciar una caché de prompts en frío.
```

Proposed:

```
…por debajo de esta tasa de tokens de salida por segundo, sin contar el tiempo que tarda el consumidor local.
```

'sin contar el tiempo de consumo local' renders 'excluding local consumer time'. A consumer is an actor — the local client that spends time outside the measured stream — and the translation turns it into an abstract 'consumption'. 'consumo de tokens' is house vocabulary elsewhere in this catalogue, so the word itself is fine; what is lost is whose time is excluded, in a help string whose whole job is to define exactly what the rate does and does not include.

Status: unresolved — the auditor made no edit.

### es-w1-B03 — NIT

Keys: `wire.host_speed.retire_below_tps.help`

Current:

```
Experimental, desactivado por defecto: la evidencia proviene de un solo modelo y una sola ejecución. Un umbral positivo deja de fijar un host tras tres respuestas largas por debajo de esta tasa de tokens de salida por segundo, sin contar el tiempo de consumo local. 0 borra las exclusiones por velocidad en la siguiente petición; las exclusiones por fallo de caché se mantienen. Cambiar de host puede iniciar una caché de prompts en frío.
```

Proposed:

```
Cambiar de host puede hacer que la caché de prompts arranque en frío.
```

'puede iniciar una caché de prompts en frío' calques 'may start a cold prompt cache': a cache does not get started cold, it starts cold. The catalogue already has the parallel 'costarte una caché de prompts en caliente', so this is preference, not term drift.

Status: unresolved — the auditor made no edit.

## Disposition recommendation

Ship. No blocker or major finding: nothing here misstates what a flag, range or remedy does. The 1 minor finding(s) are idiom/consistency fixes that can be batched into the next remediation commit for this wave; the 1 nit finding(s) are preferences. Neither class gates the round.

## Source-side and display-side risk

- No value in this namespace displayed as a redaction marker, so no sanctioned `read_public_copy` re-read was required to judge the copy.
- Sidecars are nested JSON, not flat string maps, so `read_public_copy` cannot read them at all; this namespace's sidecar notes were read directly from the tracked file and are bound by its sha256 above. A sidecar note that happened to be masked would not be recoverable through the sanctioned reader — recorded here as a source-side risk, not worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime, eligibility or release sign-off.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but not committed (no git write commands are permitted in this role).

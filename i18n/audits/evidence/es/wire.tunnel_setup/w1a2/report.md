# Spanish audit — wire.tunnel_setup

Auditor: translation-auditor es-w1-auditor-2 / session 1abd0b38e62a; independent of the translator.
Model: **radient/auto** (self-reported session header; fallback chain anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota HTTP 429"). `radient/auto` is an auto-routing alias, so the concrete backend is not identified by the header.

Audit id `w1a2`, round 1, 2026-10-29. Audited revision `efcfbe812609`; the es catalogue is **untracked** at that revision, so this audit binds the on-disk bytes (`f7d4114e7083bf264807d8d1ec6c6515de813a454ba68b5f705b8b40b70893b7`).

Verdict: **passed** (clean-with-findings); 5/5 keys read; 0 blocker, 0 major, 2 minor, 0 nit.

## What was verified

All five keys read. Every refusal keeps an actionable remedy (check its status / leave create.json unchanged and ask your agent / check the current state), the do-not-assume-rollback warning survives, the deadline-elapsed-after-the-work-was-drained ordering survives, and 'checking' makes no claim that billing or installation has started, which the sidecar forbids. create.json is byte-identical. Two minor findings, both idiom/consistency, neither changing what the user is told to do.

Mechanical checks ran per key: key-set parity, ICU brace/argument parity, placeholder parity, digit-sequence parity, en/em-dash and NBSP parity, byte-identical machine literals, width against sidecar `maxCells` where a budget exists, unsupported es plural categories, untranslated-value detection and whitespace/quote/exclamation hygiene. All passed; the detectors were proven to fire on deliberately broken copies (negative controls in `audit.log`).

## Findings

### es-w1-B04 — MINOR

Keys: `wire.tunnel_setup.deadline`

Current:

```
El plazo de respuesta de la configuración venció una vez drenado el trabajo que tenía a su cargo. Comprueba el estado y los pasos completados antes de reintentar.
```

Proposed:

```
El plazo de respuesta de la configuración venció una vez vaciado el trabajo que tenía a su cargo. Comprueba el estado y los pasos completados antes de reintentar.
```

'una vez drenado el trabajo que tenía a su cargo' renders 'after its owned work was drained'. 'drenado' appears nowhere else in the catalogue and outside queue jargon reads as drainage; the sentence is a desktop refusal whose tone note is 'cautious, actionable'.

Status: unresolved — the auditor made no edit.

### es-w1-B05 — MINOR

Keys: `wire.tunnel_setup.corrupt_intent`

Current:

```
La solicitud guardada de configuración del teléfono no es válida. Deja create.json sin cambios y pide a tu agente que recupere la solicitud original antes de reintentar. No se hizo ningún cambio de configuración.
```

Proposed:

```
La solicitud guardada de configuración del acceso telefónico no es válida. Deja create.json sin cambios y pide a tu agente que recupere la solicitud original antes de reintentar. No se hizo ningún cambio de configuración.
```

'La solicitud guardada de configuración del teléfono' names the feature differently from the other two keys of the same namespace, which say 'la configuración del acceso telefónico'. 'configuración del teléfono' can read as configuring the phone device itself; the user must recognise this as the phone-access setup whose create.json must be left alone. en's own 'phone setup request' is elliptical, so the drift is inherited, not invented.

Status: unresolved — the auditor made no edit.

## Disposition recommendation

Ship. No blocker or major finding: nothing here misstates what a flag, range or remedy does. The 2 minor finding(s) are idiom/consistency fixes that can be batched into the next remediation commit for this wave; the 0 nit finding(s) are preferences. Neither class gates the round.

## Source-side and display-side risk

- No value in this namespace displayed as a redaction marker, so no sanctioned `read_public_copy` re-read was required to judge the copy.
- Sidecars are nested JSON, not flat string maps, so `read_public_copy` cannot read them at all; this namespace's sidecar notes were read directly from the tracked file and are bound by its sha256 above. A sidecar note that happened to be masked would not be recoverable through the sanctioned reader — recorded here as a source-side risk, not worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime, eligibility or release sign-off.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but not committed (no git write commands are permitted in this role).

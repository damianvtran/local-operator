# Spanish audit — wire.aida_cadence

Auditor: translation-auditor es-w1-auditor-2 / session 1abd0b38e62a; independent of the translator.
Model: **radient/auto** (self-reported session header; fallback chain anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota HTTP 429"). `radient/auto` is an auto-routing alias, so the concrete backend is not identified by the header.

Audit id `w1a2`, round 1, 2026-10-29. Audited revision `efcfbe812609`; the es catalogue is **untracked** at that revision, so this audit binds the on-disk bytes (`69f2bdb08d2175c6e1178ef6d4e7fd9966ab0fd8c04702529207ebab37880d4d`).

Verdict: **passed** (clean-with-findings); 2/2 keys read; 0 blocker, 0 major, 0 minor, 1 nit.

## What was verified

Both keys read against en and the sidecar. The whole-elapsed-hours distinction ('horas transcurridas'), the 0-keeps-the-daily-time rule, the 4-hour example and the preserve-an-already-owed-check-in guarantee all survive. 'contacto' matches the catalogue's established rendering of an Aida check-in ('Hora de contacto diario', 'Contactos extra / día', 'cuándo te contacta'), so it is consistency, not drift.

Mechanical checks ran per key: key-set parity, ICU brace/argument parity, placeholder parity, digit-sequence parity, en/em-dash and NBSP parity, byte-identical machine literals, width against sidecar `maxCells` where a budget exists, unsupported es plural categories, untranslated-value detection and whitespace/quote/exclamation hygiene. All passed; the detectors were proven to fire on deliberately broken copies (negative controls in `audit.log`).

## Findings

### es-w1-B01 — NIT

Keys: `wire.aida_cadence.interval_hours.help`

Current:

```
0 mantiene la hora de contacto diaria local. Con 1–168 se usan horas transcurridas; 4 = un contacto cada cuatro horas. Los cambios conservan un contacto ya pendiente.
```

Proposed:

```
0 mantiene la hora de contacto diaria en hora local.
```

'0 mantiene la hora de contacto diaria local.' stacks two adjectives after the noun; 'local' can be read as qualifying 'contacto' rather than the clock the time is kept in, which is the distinction the sidecar calls out (whole elapsed hours, not wall-clock). Meaning is recoverable and nothing is promised wrongly.

Status: unresolved — the auditor made no edit.

## Disposition recommendation

Ship as-is.

## Source-side and display-side risk

- No value in this namespace displayed as a redaction marker, so no sanctioned `read_public_copy` re-read was required to judge the copy.
- Sidecars are nested JSON, not flat string maps, so `read_public_copy` cannot read them at all; this namespace's sidecar notes were read directly from the tracked file and are bound by its sha256 above. A sidecar note that happened to be masked would not be recoverable through the sanctioned reader — recorded here as a source-side risk, not worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime, eligibility or release sign-off.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but not committed (no git write commands are permitted in this role).

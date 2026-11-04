# Spanish audit — wire.console

Auditor: translation-auditor es-w1-auditor-2 / session 1abd0b38e62a; independent of the translator.
Model: **radient/auto** (self-reported session header; fallback chain anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota HTTP 429"). `radient/auto` is an auto-routing alias, so the concrete backend is not identified by the header.

Audit id `w1a2`, round 1, 2026-10-29. Audited revision `efcfbe812609`; the es catalogue is **untracked** at that revision, so this audit binds the on-disk bytes (`61f0a242644b1adfe6afaf29f113849a88745f650a93d6f91a6da09b5605bac8`).

Verdict: **passed** (clean-with-findings); 8/8 keys read; 0 blocker, 0 major, 0 minor, 1 nit.

## What was verified

All eight keys read. Live admission is kept distinct from retained history, the never-terminate-running-work and never-stopped guarantees survive, every range and default (32–256 / 128, 1–32 / 8, 4–32 / 16, one quarter of the live ceiling) is byte-identical with en dashes intact, and agent starts still cannot consume reserved slots. 'terminales activas' renders 'live terminal(s)' consistently across all four strings. One nit on label length.

Mechanical checks ran per key: key-set parity, ICU brace/argument parity, placeholder parity, digit-sequence parity, en/em-dash and NBSP parity, byte-identical machine literals, width against sidecar `maxCells` where a budget exists, unsupported es plural categories, untranslated-value detection and whitespace/quote/exclamation hygiene. All passed; the detectors were proven to fire on deliberately broken copies (negative controls in `audit.log`).

## Findings

### es-w1-B06 — NIT

Keys: `wire.console.user_reserved.label`

Current:

```
Ranuras reservadas para terminales de usuario
```

Proposed:

```
Ranuras reservadas para el usuario (32 cells); the help line already says 'estas ranuras' and the section title names the console. Alternatively keep the current text and confirm the settings row width in a rendered pass.
```

45 cells against 33 in en (+36%), above the es guide's 15-25% expansion guidance. The sidecar supplies no maxCells for this namespace, so no budget is breached on the evidence available; this is a rendered-check flag, not a proven clip.

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

# Spanish audit — wire.slash.library

Auditor: translation-auditor es-w1-auditor-2 / session 1abd0b38e62a; independent of the translator.
Model: **radient/auto** (self-reported session header; fallback chain anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota HTTP 429"). `radient/auto` is an auto-routing alias, so the concrete backend is not identified by the header.

Audit id `w1a2`, round 1, 2026-10-29. Audited revision `efcfbe812609`; the es catalogue is **untracked** at that revision, so this audit binds the on-disk bytes (`f6cc726978641736096351eb2f4bc7d754ce848cc6b3b93d8f5523cecaf47875`).

Verdict: **passed** (clean); 2/2 keys read; 0 blocker, 0 major, 0 minor, 0 nit.

## What was verified

Both keys read. No sidecar exists for this namespace, so they were judged from the en value and key name only. 'Explora…' and 'Abre, pausa o ejecuta…' match the dominant tú-imperative register of the existing wire.slash descriptions ('Lista', 'Abre', 'Muestra', 'Compacta'), all three verbs of 'Open, pause or run' survive, and 'Biblioteca'/'Bibliotecario'/'recursos' match the catalogue's established renderings in wire.settings and wire.errors.desktop. Clean.

Mechanical checks ran per key: key-set parity, ICU brace/argument parity, placeholder parity, digit-sequence parity, en/em-dash and NBSP parity, byte-identical machine literals, width against sidecar `maxCells` where a budget exists, unsupported es plural categories, untranslated-value detection and whitespace/quote/exclamation hygiene. All passed; the detectors were proven to fire on deliberately broken copies (negative controls in `audit.log`).

**No English context sidecar exists for this namespace.** `en/wire.slash.library.context.json` is absent, so role, tone and any `maxCells` budget are unknown by construction; the two keys were judged from the en value and the key name only. No context was invented for this record.

## Findings

None. No blocker, major, minor or nit finding in this namespace.

## Disposition recommendation

Ship as-is.

## Source-side and display-side risk

- No value in this namespace displayed as a redaction marker, so no sanctioned `read_public_copy` re-read was required to judge the copy.
- This namespace has no sidecar at all, so there is no sidecar text that could have been masked and no note to recover; the source-side risk here is the missing context itself, recorded above.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime, eligibility or release sign-off.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but not committed (no git write commands are permitted in this role).

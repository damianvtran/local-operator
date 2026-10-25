# Audit record — ru / relay.approvals / ap25a1

- **Verdict: passed** at the audited bytes
- Audit date: 2026-10-25
- Auditor: `translation-auditor (lopdev team)`, model `deepseek/deepseek-flash`
- Bytes as tracked at: `58102b3ae3cd0062e4f2dfee89f4b150bf330b68` (the attestation commit; the audit recorded no revision)
- Ledger row: `i18n/ledger.json` → `ru` / `relay.approvals` (17 keys)

## What was actually done

The audit ran in-session on 2026-10-25, performed by an independent agent that formed its own view before reading any findings from earlier work. It read the locale catalogue against the en source and context sidecar and audited the approval-gate copy: the three gate labels side by side, the scope sentence, the refusal sentence, the mode-change links and register against the locale's relay.asks copy. The per-locale detail below is transcribed from the ledger row and names the additional checks that row records (glossary, typography, plurals).

A second independent agent/context re-audited the same day with no sight of the first pass's findings and overturned nothing. Both passes ran on the same model, so independence is agent/context, not model diversity.

The three gate labels are 'продолжать спрашивать (ask)' / 'выполнять без запроса (auto)' / 'выполнять без запроса во всех сессиях (default auto)'. The session-only vs all-sessions distinction survives — 'во всех сессиях' plus '(auto)' vs '(default auto)' — and gate.body_scope restates it ('только эту сессию' / 'значение по умолчанию для всех последующих сессий'). 'Run without asking' keeps its risk and is NOT softened to 'автоматически'/'молча'. gate.refused_nothing_changed keeps both halves ('отказано — ничего не изменилось'). ru labels are infinitives per the ru guide; glossary одобрения (not подтверждения) held.

The findings below are transcribed from the ledger row's note, which is the primary
record of that session. The audit produced no separate report file of its own; this
document does not add findings that were not recorded there.

## Findings

- **R1 (MINOR)** — gate.default_auto/run_without say 'без запроса' where the same catalogue renders the identical concept as 'не спрашивая' (wire.settings.tool_approval_mode.choice.auto.description, 'выполнять их, не спрашивая'); 'без запроса' names the prompt as a request rather than the act of asking, so the button is marginally less explicit than the setting it changes. Not a blocker: it does not read as automatic/silent. Recommend 'не спрашивая' for symmetry in a later copy pass.

Noted outside this namespace's findings count:

- None.

## Inputs (hashes of the audited bytes)

| input | sha256 |
| --- | --- |
| `en/relay.approvals.json` | `321658288e707b93b94062345bd8bbbdc4301d7fd6865b7378a46efe1d429125` |
| `en/relay.approvals.context.json` | `3f1521e5b63b3916c0f68bdb0f2b0165fe9d437b2e14b18ddad9d19cc2ec6117` |
| `ru/relay.approvals.json` | `95ba6117d69f47e70e2c2333f0b5a3b355ec7063bd4d129361a4550eb41a1e0e` |

The en hash equals the ledger row's `source_sha256`, and the target hash still equals
the working tree's file at the time this bundle was written.

## Mechanical replay (not the auditor's log)

`mechanical-recheck.log` is a replay of the mechanical checks the ledger row reports
(key parity, ICU argument tuples and their order, plural categories, bidi/zero-width
controls, straight quotes, double spaces, English leakage), executed by the
implementation session on 2026-10-25 against the same revision with
`recheck.py.txt`. It confirms the reported mechanical result at the audited bytes.
It is not independent evidence and does not replace the audit.

## What this bundle does not contain

- **`audit.log` is absent.** No command output was captured during the audit session.
  The mechanical figures were reported in prose; reconstructing a command log after
  the fact would invent provenance.
- No raw tool transcript or session transcript was preserved.
- No native-speaker judgement is part of this audit: the auditor was an LLM and the record claims no native review.
- No rendered or native UI evidence: this is a textual audit.

## Post-audit change and the state of this row

After this audit, the en `relay.approvals` catalogue was split for the M5 emphasis remediation: the single `gate.body_scope` key became `gate.body_scope.{before,session_clause,between,default_clause,after}` so the component can wrap the two blast-radius phrases in `<strong>` without markup in the catalogue. That changes the en bytes the row's pass was bound to.

| input | audited bytes | current bytes |
| --- | --- | --- |
| `en/relay.approvals.json` | `321658288e707b93b94062345bd8bbbdc4301d7fd6865b7378a46efe1d429125` | `5efd5d666d1c498de506541c8bbd339204b5919950395934bfcf188476030ce0` |
| `en/relay.approvals.context.json` | `3f1521e5b63b3916c0f68bdb0f2b0165fe9d437b2e14b18ddad9d19cc2ec6117` | `1a595de0049de2b02d27e00fc57c3fa14cb191cdde917bd54045b64a36664e75` |

By the §6 freshness rule the row is now **stale** (`source_sha256` no longer equals
the current en hash), and because a stale carried namespace un-ships the whole
locale, the locale drops out of the shipped set entirely until the clause split is
translated and re-attested. The locale catalogue still carries the
retired `relay.approvals.gate.body_scope` key. That is the translator worklist, not
a defect in the copy this audit passed.

# Audit record — zh-CN / relay.approvals / ap25a1

- **Verdict: passed** at the audited bytes
- Audit date: 2026-10-25
- Auditor: `translation-auditor (lopdev team)`, model `deepseek/deepseek-flash`
- Bytes as tracked at: `58102b3ae3cd0062e4f2dfee89f4b150bf330b68` (the attestation commit; the audit recorded no revision)
- Ledger row: `i18n/ledger.json` → `zh-CN` / `relay.approvals` (17 keys)

## What was actually done

The audit ran in-session on 2026-10-25, performed by an independent agent that formed its own view before reading any findings from earlier work. It read the locale catalogue against the en source and context sidecar and audited the approval-gate copy: the three gate labels side by side, the scope sentence, the refusal sentence, the mode-change links and register against the locale's relay.asks copy. The per-locale detail below is transcribed from the ledger row and names the additional checks that row records (glossary, typography, plurals).

A second independent agent/context re-audited the same day with no sight of the first pass's findings and overturned nothing. Both passes ran on the same model, so independence is agent/context, not model diversity.

The three gate labels are '继续询问（ask）' / '无需询问直接运行（auto）' / '在所有会话中无需询问直接运行（default auto）'. The session-only vs all-sessions distinction survives — only the third label carries the scope phrase '在所有会话中', with '（auto）' vs '（default auto）' as the second signal — and gate.body_scope restates it ('只改变本会话' / '改变其后所有会话的默认值'). 'Run without asking' keeps its risk: '无需询问', never '自动'/'静默'; '直接运行' adds that the gate is bypassed, it does not hide it. gate.refused_nothing_changed keeps both halves ('已拒绝——没有任何更改'). zh typography held: full-width punctuation, '——' for the em dash, （） around the runtime token.

The findings below are transcribed from the ledger row's note, which is the primary
record of that session. The audit produced no separate report file of its own; this
document does not add findings that were not recorded there.

## Findings

- **Z1 (NIT)** — session_override_global renders '全局{global}' with no separator — '全局：{global}' or '全局为{global}' reads cleaner; '本会话现为auto' (card.notice_session_auto) is slightly bookish but matches relay.asks.gate.notice_auto's '现为auto', so it is consistent and not a defect.

Noted outside this namespace's findings count:

- None.

## Inputs (hashes of the audited bytes)

| input | sha256 |
| --- | --- |
| `en/relay.approvals.json` | `321658288e707b93b94062345bd8bbbdc4301d7fd6865b7378a46efe1d429125` |
| `en/relay.approvals.context.json` | `3f1521e5b63b3916c0f68bdb0f2b0165fe9d437b2e14b18ddad9d19cc2ec6117` |
| `zh-CN/relay.approvals.json` | `b77e74a8041c910f060597575efb9ed739c30c8657e384746064c6faca52672e` |

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

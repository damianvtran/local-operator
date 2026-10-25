# Audit record — zh-CN / relay.asks / zha25a1

- **Verdict: passed** at the audited bytes
- Audit date: 2026-10-25
- Auditor: `translation-auditor (lopdev team)`, model `deepseek/deepseek-flash`
- Bytes as tracked at: `58102b3ae3cd0062e4f2dfee89f4b150bf330b68` (the attestation commit; the audit recorded no revision)
- Ledger row: `i18n/ledger.json` → `zh-CN` / `relay.asks` (77 keys)

## What was actually done

The audit ran in-session on 2026-10-25 over the restored bytes (restored at ab44bbb04a), performed by an independent agent that read no prior findings before forming its own view. It checked key parity, placeholder and ICU argument parity, zh plural categories, bidi controls, English leakage, the gate labels' scope distinction, and the zh typography.

No second pass was run for this namespace: the ledger row records round 1 against the restored bytes, not a carried re-audit.

Restored byte-identically from 3bd709c0f4 after the hold that deleted it was lifted. 77/77 keys, 0 placeholder mismatches, 4 plural keys all other-only with a rendering '#', no bidi controls, no English leakage except the intentional mode token 'ask'. Scope audit of the gate labels it carries: '继续询问（ask）' / '无需询问直接运行（auto）' / '在所有会话中无需询问直接运行（default auto）' — the session-only vs all-sessions distinction survives and 'run without asking' keeps its risk.

The findings below are transcribed from the ledger row's note, which is the primary
record of that session. The audit produced no separate report file of its own; this
document does not add findings that were not recorded there.

## Findings

- **Q1 (MINOR)** — sheet.title_plain '提问' — the same defect class the other six locales were fixed for on 2026-10-20 (their rows record 'demandes'/'consultas'/'पूछताछ'/'запросы'/'yêu cầu'/'درخواستیں' as section nouns distinct from the question word); 提问 is the question-asking verb, so the sheet that also holds approvals and secrets is named with the question word. Not major: rows inside carry their own kind labels and the dock badge is unchanged. Recommend '请求' (already used in card.request_needs_answer) in the next copy pass.
- **Q2 (MINOR)** — pending.kind_approval '需要审批' / kind_secret '需要机密信息'. 需要审批 is accurate and is the vocabulary gate.body/notice_auto already use; the weak point is '机密信息' for a secret value, since zh has no recorded credentials/secret rendering (glossary §2 note 6 leaves the zh cell '—'). The card's masked field and card.secret_notice carry the meaning.
- **Q3 (MINOR)** — card.decline and pending.deny both render '拒绝'. They never co-occur (decline sits beside 'send answer' on the ask card, deny beside 'approve' on the approval card) and five of seven locales collapse them identically (fr 'refuser', hi 'अस्वीकार करें', vi 'từ chối', ur 'انکار کریں'); the English pair is a register distinction, not a functional one, and both refuse.
- **Q4 (MINOR)** — gate.unpaired_after '模式从不需要配对。' extends the bold 'ask' label with 模式 (necessary in Chinese) but joins without a connective; rendered it reads '将审批保持在 ask 模式从不需要配对。' — understandable, slightly run-on, no meaning loss.
- **Q6 (NIT)** — dock.waiting / dock.timed_out / title_waiting write '# 个…' with a space while title_lapsed writes '{count}个…' without one — pick one convention.
- **Q5 (REJECTED)** — sheet.loading '……' — REJECTED, not a defect: the zh style guide mandates '…… (two marks)' and 62 other zh catalogue strings already use it (tui.library.loading, relay.session.past.loading, mobile.setup.checking), while the six single-'…' strings are all inside comma-separated Latin lists; the sidecar's 'The ellipsis is U+2026' pins the character class, not a count.

Noted outside this namespace's findings count:

- The ledger row's comment on Q1 notes '请求' is already used in card.request_needs_answer, so the recommended fix reuses existing copy rather than adding a term.

## Inputs (hashes of the audited bytes)

| input | sha256 |
| --- | --- |
| `en/relay.asks.json` | `d6041075826aa17e0a86f0d47081faf0ec796057cb95cece2f47e6f57051b3cc` |
| `en/relay.asks.context.json` | `af782cea8a1156b8e0612739254ffec7a807cf8bd53deaed3b6b28e6d61366f2` |
| `zh-CN/relay.asks.json` | `915fef75f0c939a5dcc0c31eb5637e8ef85c253279002eb6f1d118bf180783ea` |

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
- The auditor stated it is not a native Chinese reader; the findings above are the round-1 audit's, and the minor items ride the next zh copy pass rather than forcing another round.
- This is a round-1 audit of the restored bytes, not a carried re-audit.
- No rendered or native UI evidence: this is a textual audit.

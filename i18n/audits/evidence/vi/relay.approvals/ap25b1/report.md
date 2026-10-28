# relay.approvals - vi - re-attestation ap25b1

Revision: 3d0f77762a (locale split), en source unchanged since 4b40d82512
Auditor: translation-auditor (lopdev team) - re-attestation pass by translation-reviewer
Model: deepseek/deepseek-flash (same model as the 2026-10-25 first pass; agent/context independence only)
Date: 2026-10-25

## Composed sentence (what the user reads)

Hai lựa chọn đầu chỉ đổi **phiên này** mà thôi; lựa chọn cuối đổi **mặc định cho mọi phiên** sau đó.

Bold runs: `phiên này` and `mặc định cho mọi phiên`.

## Reading

Hai lựa chọn đầu chỉ đổi **phiên này** mà thôi; lựa chọn cuối đổi **mặc định cho mọi phiên** sau đó. Meaning preserved: chỉ ... mà thôi confines the first two choices to this session; the last changes the default for every session after. Bold: 'phiên này' and 'mặc định cho mọi phiên' - both noun phrases, no verb or particle torn out of them. Seams: 'mà thôi;' keeps the restrictive particle ahead of the semicolon, which is what vi.md's no-space-before-punctuation rule needs; all four joins read cleanly, 'mọi phiên sau đó.' included. NIT: 'đổi' repeats across the two clauses (đổi ... đổi) - idiomatic, but 'thay đổi' in the first position would read better. Preference only. 0 blocker/major findings. Confidence: medium. LIMITATION: same model as the first pass. Evidence: i18n/audits/evidence/vi/relay.approvals/ap25b1/record.json

## Seam

clean at all four joins; 'mà thôi;' keeps the restrictive particle ahead of the semicolon

## Findings

- VI-1 [nit/accepted] before / between (verb repetition): 'đổi' repeats in both clauses; 'thay đổi' in the first would read better. Preference only.

## Confidence

medium (Vietnamese read, not native)

## Limitations

- Same model as the first pass: independence is agent/context, not model diversity.
- No native speaker has reviewed any locale; hi and ur were judged non-native by the two earlier
  passes and this pass does not overturn that.
- No rendered frame was captured; see record.json `rendered_artifact`.

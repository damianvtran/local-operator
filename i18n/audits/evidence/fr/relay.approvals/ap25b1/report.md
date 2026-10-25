# relay.approvals - fr - re-attestation ap25b1

Revision: 3d0f77762a (locale split), en source unchanged since 4b40d82512
Auditor: translation-auditor (lopdev team) - re-attestation pass by translation-reviewer
Model: deepseek/deepseek-flash (same model as the 2026-10-25 first pass; agent/context independence only)
Date: 2026-10-25

## Composed sentence (what the user reads)

Les deux premiers choix ne changent que **cette session** ; le dernier modifie la **valeur par défaut de toutes les sessions** suivantes.

Bold runs: `cette session` and `valeur par défaut de toutes les sessions`.

## Reading

Les deux premiers choix ne changent que **cette session** ; le dernier modifie la **valeur par défaut de toutes les sessions** suivantes. Meaning survives the restructure: ne ... que stays with its subject inside 'before', so the restriction attaches to 'cette session' exactly as en's 'only' does, and the pivot lands on the last choice and the global default - the same two scopes. Bold: 'cette session' and 'valeur par défaut de toutes les sessions' - both complete noun phrases; the article 'la' stays plain because en's bold starts at 'default', so that is a match, not a shortfall. Seams: 'que cette session' is clean; 'cette session ;' renders a plain U+0020 before the semicolon. FINDING (minor, open, not fixable in the copy): fr.md's punctuation section makes the espace fine insécable U+202F before ';' REQUIRED in prose, and the en sidecar for body_scope.between tells the translator to use it, but the component joins with U+0020 - so U+202F inside 'between' would render space+U+202F, which is worse. The outcome is a binding-guide deviation plus a breakable space that can drop ';' to a new line in the narrow sheet. That makes it worse than a cosmetic nit, though the glyph difference at 14px is invisible. The fix belongs in the join or in a recorded fr exemption, not in this catalogue. Confidence: high. LIMITATION: same model as the first 2026-10-25 pass, so independence is agent/context, not model diversity. Evidence: i18n/audits/evidence/fr/relay.approvals/ap25b1/record.json

## Seam

clean at four joins; the fifth renders 'cette session ;' with U+0020 where fr.md requires U+202F

## Findings

- FR-1 [minor/open] component join (gate-sheet.tsx) vs the fr.md punctuation rule: Seam renders U+0020 before ';' where fr.md requires U+202F; a narrow space inside 'between' would double-space, so the fix is in the join or a recorded exemption, not the copy.

## Confidence

high (French read fluently)

## Limitations

- Same model as the first pass: independence is agent/context, not model diversity.
- No native speaker has reviewed any locale; hi and ur were judged non-native by the two earlier
  passes and this pass does not overturn that.
- No rendered frame was captured; see record.json `rendered_artifact`.

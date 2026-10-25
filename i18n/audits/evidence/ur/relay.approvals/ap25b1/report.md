# relay.approvals - ur - re-attestation ap25b1

Revision: 3d0f77762a (locale split), en source unchanged since 4b40d82512
Auditor: translation-auditor (lopdev team) - re-attestation pass by translation-reviewer
Model: deepseek/deepseek-flash (same model as the 2026-10-25 first pass; agent/context independence only)
Date: 2026-10-25

## Composed sentence (what the user reads)

پہلے دو اختیارات صرف **اسی سیشن کو** بدلتے ہیں؛ آخری والا اس کے بعد آنے والے **تمام سیشنز کا ڈیفالٹ** بدلتا ہے۔

Bold runs: `اسی سیشن کو` and `تمام سیشنز کا ڈیفالٹ`.

## Reading

پہلے دو اختیارات صرف **اسی سیشن کو** بدلتے ہیں؛ آخری والا اس کے بعد آنے والے **تمام سیشنز کا ڈیفالٹ** بدلتا ہے۔ Meaning preserved; verb-final and grammatical. Bold: 'اسی سیشن کو' and 'تمام سیشنز کا ڈیفالٹ' keep their postpositions (کو / کا) inside the bold for the same reason as hi. Seams: the semicolon is Urdu ؛ and sits tight against ہیں inside 'between', so no space-before-punctuation defect appears; final ۔ per ur.md. Terminology: سیشن is the guide's loanword and the plural سیشنز is the inflected form the guide permits ('otherwise inflect'), consistent with the rest of this namespace. NIT: 'آخری والا' is colloquial (as in hi), and 'اسی' plus 'صرف' doubles the emphasis into 'only this very session' - natural, slightly stronger than en. No bidi controls in the copy and none needed; isolation is the render layer's job per ur.md. CONFIDENCE: LOW, and no rendered RTL frame was captured, so this row claims nothing about layout. The earlier non-native judgement stands. Evidence: i18n/audits/evidence/ur/relay.approvals/ap25b1/record.json

## Seam

clean at all four joins; the Urdu semicolon sits tight against ہیں inside 'between'

## Findings

- UR-1 [nit/accepted] between (آخری والا): Colloquial 'آخری والا', same as hi.

## Confidence

LOW — Urdu idiom not certified; the earlier non-native judgement stands

## Limitations

- Same model as the first pass: independence is agent/context, not model diversity.
- No native speaker has reviewed any locale; hi and ur were judged non-native by the two earlier
  passes and this pass does not overturn that.
- No rendered frame was captured; see record.json `rendered_artifact`.

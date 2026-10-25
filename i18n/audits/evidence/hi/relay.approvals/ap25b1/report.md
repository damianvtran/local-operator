# relay.approvals - hi - re-attestation ap25b1

Revision: 3d0f77762a (locale split), en source unchanged since 4b40d82512
Auditor: translation-auditor (lopdev team) - re-attestation pass by translation-reviewer
Model: deepseek/deepseek-flash (same model as the 2026-10-25 first pass; agent/context independence only)
Date: 2026-10-25

## Composed sentence (what the user reads)

पहले दो विकल्प केवल **इस सत्र को** बदलते हैं; आख़िरी वाला इसके बाद आने वाले **सभी सत्रों का डिफ़ॉल्ट** बदलता है।

Bold runs: `इस सत्र को` and `सभी सत्रों का डिफ़ॉल्ट`.

## Reading

पहले दो विकल्प केवल **इस सत्र को** बदलते हैं; आख़िरी वाला इसके बाद आने वाले **सभी सत्रों का डिफ़ॉल्ट** बदलता है। Meaning preserved and the Hindi is grammatical, verb-final as Hindi requires. Bold: 'इस सत्र को' and 'सभी सत्रों का डिफ़ॉल्ट' - both keep their case postposition (को / का) inside the bold, which is the right call: without it the bold run would be a bare stem, not a unit. Seams: 'केवल इस सत्र को बदलते हैं' and 'डिफ़ॉल्ट बदलता है।' both clean; danda and सत्र/सत्रों follow hi.md. NIT: 'आख़िरी वाला' is conversational where 'पहले दो विकल्प' is neutral - 'आख़िरी विकल्प' would hold one register and reuse this file's own noun for a choice. CONFIDENCE: LOW. I cannot certify Hindi idiom, and this pass does not change the earlier verdict that hi reads non-native: no native speaker and no different-model pass has run, and finding nothing that fails is not a native sign-off. Evidence: i18n/audits/evidence/hi/relay.approvals/ap25b1/record.json

## Seam

clean at all four joins

## Findings

- HI-1 [nit/accepted] between (आख़िरी वाला): Colloquial 'आख़िरी वाला' against neutral 'पहले दो विकल्प'; 'आख़िरी विकल्प' would hold register.

## Confidence

LOW — Hindi idiom not certified; the earlier non-native judgement stands

## Limitations

- Same model as the first pass: independence is agent/context, not model diversity.
- No native speaker has reviewed any locale; hi and ur were judged non-native by the two earlier
  passes and this pass does not overturn that.
- No rendered frame was captured; see record.json `rendered_artifact`.

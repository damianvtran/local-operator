# fr / relay.health — independent evidence-recovery audit (frer1)

8 keys, all 8 audited against the committed en source and the en context
sidecar. Verdict: **PASS**, 0 findings.

## What was judged

- **Idiom.** These are status-dot sheet bodies, and the French reads as French
  explanation rather than translation: `ne répond plus`, `occupée ou bloquée`,
  `quand elle aura rattrapé son retard`, `Réessayer ne l'enverra pas en
  double`, `Tant qu'il n'y parvient pas`. The pronoun chains are correct:
  `La session s'y exécute peut-être encore` (y = on your computer) and
  `elle aura rattrapé son retard` agreeing with `la session`.
- **The delivery semantics the context sidecar calls load-bearing.** Each is
  preserved without inventing a promise:
  - `reconnecting_detail`: it is the phone's link that is down, the session may
    still run, sending may fail — `un envoi peut échouer tant que la connexion
    n'est pas rétablie`. No queueing claim.
  - `dropped_detail`: the send fails at once and the text stays in the field —
    `un envoi échoue aussitôt et votre texte reste dans la zone de saisie`.
  - `quiet_detail`: a send can be reported failed and still arrive, and a retry
    does not duplicate — `signalé en échec et arriver malgré tout`,
    `ne l'enverra pas en double`.
- **`ended_detail` adds information rather than repeating the title.** The title
  is `Terminée`; the body says history is kept and names the resume path.
  `dossier personnel` is the home folder, distinct from `dossier de travail`
  (working folder) used by `relay.chrome.composer.folder_label`, so the two
  folders stay distinguishable in French.
- **Cross-key consistency.** `« Reprendre »` equals
  `relay.chrome.session.resume`; `« Terminée »` equals
  `relay.chrome.session.subtitle.ended`. The quoted status word and the button
  are the ones that exist. Guillemets are used for button and status names
  throughout, which is the French convention the style guide sets.
- **Register.** vous (`Envoyez`, `Réessayez`, `touchez`, `choisissez`) mixed
  with impersonal statements, matching the sibling catalogues. The `live` label
  `En ligne` sits in the same short-state-word register as `Ne répond pas` and
  `Terminée`. `Reconnexion` (no ellipsis) is correct here because the tone note
  forbids an ellipsis for this key, and the context distinguishes it from
  `relay.chrome.link.reconnecting`, which does carry one.
- **Placeholders, plurals, typography.** No placeholders or ICU constructs in
  this namespace, so parity is vacuous. U+202F before the prose colon, « … »
  with U+202F inside, U+2019 apostrophes, spaced em dashes: zero violations.
- **Width.** All 8 keys are inside their declared `maxCells` (16 for the two
  state words that declare one; the prose bodies declare none and are
  unbudgeted).

## Note on the previously cited nit

The 2026-10-27 ledger note records F1: `ended_placeholder` kept `dossier perso`
against a 36-cell cap. That key no longer exists — the 2026-10-28 row records
its deletion and confirms the remaining 8 values are byte-identical. This audit
therefore has no `ended_placeholder` to judge; the deletion is verified by
key-set parity instead.

## What the ledger row binds (R-4)

This row carries `source_sha256` (`e060aed0…`) and quotes the target SHA-256
(`af3ac7a7…`) in its note prose. Both were recomputed here and match the
committed bytes, so the audited French bytes are bound by the row's text — but
by its prose, not by a target-hash field.

## Limitations

- Independent LLM review, not native-speaker certification.
- Scalar text only; no relay app, phone or rendered frame was exercised.
- The bundle records THIS audit. It does not reconstruct the two audits
  (A and B) the ledger note cites; those receipts remain uncommitted.

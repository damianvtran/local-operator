# French audit A — wire.errors.session_host

Verdict: PASS for this independent first pass, 3/3 keys, no findings of any severity. This is NOT the required second independent pass or a ledger authorization to ship.

Auditor: i18n-auditor (host-fr-audit-a), session 0a66628c58a2, actual model openai/gpt-6.1-sol (quota fallback from anthropic/claude-sonnet-5-5 via anthropic/claude-opus-5-5). Independent of translator session 829abca97a25, even though the actual model matches.

## Per-key assessment (audit notes, not execution log)
- `wire.errors.session_host.unsupported`: PASS. “ne peut pas exécuter” describes platform inability, “n’a donc pas été démarrée” unequivocally says no session started, and “ordinateur doté d’un gestionnaire … compatible” supplies the correct permanent-platform remedy. “hôte de sessions” is a grammatical role description, not a product-name translation. No suggested change.
- `wire.errors.session_host.unavailable`: PASS. “n’a pas répondu” preserves non-response, not rejection; the session was not started. “Attendez un instant avant de réessayer” asks the person to retry later without asserting an automatic retry or reconnect. Natural, concise vous-register prose. No suggested change.
- `wire.errors.session_host.refused`: PASS. “a refusé de démarrer cette session” preserves refusal, and the separate sentence explicitly states no start. “Consultez … avant de réessayer” preserves the inspection-first instruction. `logs/session-host.log` and `Local Operator` remain byte-identical. “dossier de configuration Local Operator” is idiomatic product-label attachment and respects the no-article naming rule. No suggested change.

## Mechanical / fit
Independent actual repository validator execution: exit 0, `check.py: catalogue checks: 0 problem(s)`. Reused translator’s already-staged validation tree without writes, installs or copies; verified candidate, pinned English, context and validator script bytes first. Also read the translator’s actual validator.log and verified its SHA against provenance.

Independent checks: valid JSON, exact 3-key parity/order, no placeholders/plural/select branches (therefore no missing French one/many/other arms), no Rich/HTML tags or backticks to preserve, correct literal path/product name, NFC, no Unicode format controls. Native typography uses U+2019 apostrophes and ordinary sentence punctuation; none of : ; ! ? needing French thin spaces occurs in prose. Imperatives are sentence-level vous instructions, not infinitive button labels; no register violation.

Measured Rich terminal cell lengths: unsupported 195; unavailable 129; refused 206. Context supplies no maxCells/maxChars, and these are full error prose, not short buttons/tabs. No measured overflow claim; actual rendered wrapping/clipping was not tested by this textual audit. French is LTR; no RTL requirements apply.

## Source integrity and limits
Pinned source commit: 9c979a03d551020ce6ff1b16c8066aa69ce7bb2a. Required English/context digests verified. Style-guide hash from translator matches pinned commit, not the working checkout: diff removes only unrelated relay.chrome width exceptions. Read that delta too; all rules relevant here are identical. No MT/service used. No translations, ledger or shared-worktree files edited. No actual ledger write or commit performed. A separate independent audit B is still required for high-visibility error copy.

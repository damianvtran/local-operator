# French author notes — not an audit or execution log

Status: **translated**, all four keys. No independent audit verdict is claimed.
Lane: bounded localization fix (<12 min allocation); no dependencies, global checker, generation, UI/build or server runs.

## Native choices

- `model` → **modèle**, per glossary; the opaque `{label}` stays untouched.
- `session` → **session**, per glossary. `nouvelles sessions` makes `/model default` explicitly future-facing; `enregistre ce modèle` names the saved choice rather than implying a current-session switch.
- `appliquez à cette session … avec /model saved` explicitly identifies the current-session switch, not a save operation. `appliquez` uses the required professional vous imperative.
- `boot default` → **modèle par défaut au démarrage**: the configured starting model, not the current choice. This missing glossary phrase is proposed for consistent French prose.
- `/settings permet aussi de modifier …` is a complete standalone settings route, retains the source's “too”, and does not introduce unusable `/model` commands in setup-dead-end states.
- Compact lowercase clauses match the notice voice. French U+202F before `:` in `modèle :`; preserve the em dash and opaque route interpolation. No manual bidi controls.

## Mechanical evidence and limits

`validation.log` is actual captured command stdout/stderr (exit 0), not these notes. `validation.py.txt` preserves the executed scoped validation source. It uses existing catalogue loading, ICU argument/parser, checker `arguments_match`, and ICU render APIs from this worktree through the primary shared interpreter and explicit PYTHONPATH. All four source/target ICU messages have zero plural/select arms; the envelope has exactly `label` and `routes`, each once and simple/string, all other keys have no arguments. All three commands remain byte-exact.

The isolated command used `env -i`, synthetic scratch HOME/config and no inherited CMUX variables. Git operations used normal HOME. No global/base-aware checker was run because full-namespace admission and global validation belong to the owner.

No maxCells/maxChars exists here. Actual NoticeBlock wrapping at 80/100/120, pristine/populated transcript families, before/after UI reproduction, integration, and independent A/B audits remain owner-owned; this scoped ICU render does not claim those validations or native visual proof.

## Provenance

Agent: i18n-translator / model-notice-fr-author. Session: 1ca940cf73ba.
Actual authoring/validation model: openai/gpt-6.1-sol. Primary model was anthropic/claude-opus-5-5; the explicit model-switch event reported provider failure: quota, before authoring. No translation service/API/SDK was used; these strings were authored by LLM reasoning.

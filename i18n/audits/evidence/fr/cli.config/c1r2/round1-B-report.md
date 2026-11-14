# c1-au-fr-B — fr audit, 22 keys (independent pass B)

Verdict: clean-with-findings (changes_requested if MAJOR must gate). Counts: blocker 0, major 1, minor 2, nit 3.

Mechanical (python3): key set+order identical (22/22); no empties; none identical to English; ICU placeholder sets identical ({path},{key},{value},{description},{flag},{session},{to},{code},{reason}); backtick spans byte-identical; /move, /approvals auto, --to/--keep/--queue/--drop-session-roles/--wait verbatim; `local` kept English; U+202F before every : ; (no plain space); apostrophes U+2019; no stray English (only radient/openai/... and `auto` as mandated).

## MAJOR
M1 tui.session_move.deadline_exceeded — "le déplacement n’a pas abouti dans le délai imparti ; relancez `…` pour connaître son état d’avancement".
"n’a pas abouti" = "did not succeed / failed" (shipped fr corpus uses it that way: "La signature n’a pas abouti", "L’opération … n’a pas abouti"). Source marks this refusal changed=True (child reaped on timeout; move may have completed) and notes say "Do not say it failed or succeeded". Polarity/safety error: reader concludes the move failed and may retry/abandon.
Suggest: "le déplacement ne s’est pas terminé dans le délai imparti ; relancez `lop sessions move {session} --to {to}` pour savoir où il en est".

## MINOR
m1 cli.config.list_unrecognised — "aucun composant ne la lit": adds "composant" (source: "nothing reads it"). Suggest "clé non reconnue ; rien ne la lit".
m2 cli.config.running_sessions_unchanged — all safety claims survive (running sessions unchanged; config write cannot loosen; /approvals auto per session; new sessions open at auto). Only paraphrase: "leurs règles d’approbation" for "loosen one"; "désactiver les approbations" a stronger gloss of "ungated". Acceptable; not gating.

## NIT
n1 cli.config.list_desc_model_name — drops "specific" ("Le modèle précis…").
n2 tui.session_move.ambiguous — "pour définir le répertoire de travail" turns "for a working directory" into an action; accurate to /move <path>.
n3 cli.config.list_desc_classification_notice — "l’appel" ambiguous alone; identical to shipped wire.settings.classification_notice.help ("restent muettes dans le chat ; l’appel figure dans le journal"), so consistency argues keeping.

## Checked OK
- Terminology matches shipped fr: compactage, apprentissages, suggestions intelligentes, journal, appareil (peer arg `<appareil|local>`), fichier de configuration, approbation, Obsolète tag with em dash, clé, « p. ex. ».
- Register: vous/imperative per style guide; terse; no "Veuillez".
- cannot_wait: shell command byte-identical. unsupported_flag: {flag} arrives repr-quoted (session_move.py `repr(token)`), "pas {flag}" correct.
- cli_no_answer / cli_unavailable faithful; "la CLI" matches shipped usage.
- Width: no cell cap; longest line 259 chars wraps like the source.
- created/opened/updated/cleared faithful; "{key} effacé" gender-invariant for a setting key.

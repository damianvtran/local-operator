# fr cli.team_bridge audit tbp3198 (PR #3198)

Scope: the two added keys `notify.probe_suspected_title` and `notify.probe_suspected_body`, read against en source and `cli.team_bridge.context.json` notes, the fr style guide and `i18n/glossary.md`. The other 53 keys are byte-unchanged since the PR #3189 passed row and were not re-audited.

## Findings and dispositions

fr: 1 major + 2 minor, fixed. Major: title used 'Messages d’agents' where the namespace's established product name is 'Messages des agents'. Minor: plain U+0020 before ':' (guide requires U+202F); ASCII apostrophes (guide requires U+2019). Also 'Ce schéma peut signifier' -> 'Ce comportement peut indiquer' (calque). Unresolved blocker/major: 0. Evidence: i18n/audits/evidence/fr/cli.team_bridge/tbp3198/record.json

## Negative control

Before `generate.py` re-shipped the locales, the same checks rendered English for every non-en locale (shipped set was `["en"]` at the stale hash) and failed 14/14; after appending these rows and regenerating, all 14 render their own copy and pass. That is the discriminating proof the rows restore eligibility.

## Limitations

Independent LLM review, not native-speaker certification. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3198 and is recorded as unrecorded, not inferred.

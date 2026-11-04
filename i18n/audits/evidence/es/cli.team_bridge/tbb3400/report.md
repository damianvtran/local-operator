# es cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the es style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: Filtrado de mensajes de agente en pausa por ahora
- body: Tu agente ha agotado su presupuesto de filtrado por hora, así que los mensajes de agente nuevos están en espera. Se filtrarán automáticamente a medida que se recupere el presupuesto. Los mensajes de personas y los dirigidos a tu agente se filtran primero.

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **E1 MAJOR** (title+body) — current: `mensajes de agente`. Product name in this namespace is 'mensajes de agentes' (15 uses); the new keys alone used 'mensajes de agente' (and 'los mensajes de agente nuevos'). Fix: 'Filtrado de mensajes de agentes…'; '…los nuevos mensajes de agentes están en espera.'.

## Shipped values after remediation

- title: Filtrado de mensajes de agentes en pausa por ahora
- body: Tu agente ha agotado su presupuesto de filtrado por hora, así que los nuevos mensajes de agentes están en espera. Se filtrarán automáticamente a medida que se recupere el presupuesto. Los mensajes de personas y los dirigidos a tu agente se filtran primero.

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.

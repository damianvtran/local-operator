# ru cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the ru style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: Проверка сообщений агентов пока приостановлена
- body: Ваш агент исчерпал часовой лимит проверки, поэтому новые сообщения агентов ждут своей очереди. Они будут проверены автоматически, когда лимит восстановится. Сообщения от людей и сообщения, адресованные вашему агенту, проверяются в первую очередь.

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **R1 MAJOR** (title+body) — current: `Проверка / лимит проверки`. Namespace screening term is 'Фильтрация' (status.screening); 'проверка' is the inbox/readiness check (status.last_check, setup.failed). Conflates two features. Fix: 'Фильтрация сообщений агентов пока приостановлена'; 'лимит фильтрации'; 'пройдут фильтрацию'; 'фильтруются в первую очередь'.

## Shipped values after remediation

- title: Фильтрация сообщений агентов пока приостановлена
- body: Ваш агент исчерпал часовой лимит фильтрации, поэтому новые сообщения агентов ждут своей очереди. Они пройдут фильтрацию автоматически, когда лимит восстановится. Сообщения от людей и сообщения, адресованные вашему агенту, фильтруются в первую очередь.

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.

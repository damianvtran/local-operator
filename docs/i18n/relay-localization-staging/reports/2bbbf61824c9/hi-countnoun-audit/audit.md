# Hindi counted-noun delta — independent native audit

Auditor: independent Hindi task, session `2bbbf61824c9`; actual model **openai/gpt-6.1-sol**, no fallback. Independent of translator, not a human-native-speaker attestation.

## Verdict: PASS, terminal native round

Exact finding IDs `[]`, no findings, unresolved BLOCKER/MAJOR 0. Three changed unit-noun messages reviewed semantically and exercised in their direct composed remaining/used neighbors. Prior independent 65-message native review is preserved for the **62 unchanged values**, not claimed as newly rerun semantic review.

Source is **uncommitted working bytes** on base HEAD `de9b9a490c839d6de2c2e41ea03ec819f66be489`: English shell SHA-256 `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`; context `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`. New Hindi shell SHA-256 `feab9e023be86d7841bd1e08cb273a48bbaaa118c41ade6040d7a29a701305eb`. Complete source/target old/new hashes and exact changed keys are in audit.json/runtime-results.json.

## Compatibility and native meaning

Compared old/new Hindi maps and English HEAD/working maps: exactly `relay.shell.usage.unit_credits`, `unit_tokens`, `unit_requests` changed on each side. Other shell41 values unchanged. Session17/login4 target files byte-identical and English sources unchanged; total62 unaffected values retain prior native audit evidence. USD unchanged.

Hindi branches deliberately repeat क्रेडिट / टोकन / अनुरोध in one/other. These are correct **direct counted-noun forms**: the loan nouns क्रेडिट/टोकन and masculine consonant-ending अनुरोध do not acquire a distinct direct plural ending here. No number is duplicated in the unit; no oblique postposition is attached to the unit. The outer nominal `शेष` and `का उपयोग हुआ` remain coherent at 0, 1, 2, 1.5 and negative/million amounts. Negative quantities test robustness of wording; they do not assert product domain permission for negative usage.

## Actual execution

Existing pinned relay `createMessageRuntime`, existing installed Intl dependencies, Node native TypeScript; no install. `env -i` scratch HOME/config, UTC, no inherited CMUX or live session use.

`probe.mjs` exited 0; `runtime-results.json` passed=true and contains **48 composed outputs**: three unit messages × eight quantities (0,1,2,1.5,-1,-2,-1.5,1000000) × remaining/used.

Actual examples:
- `0 क्रेडिट शेष` / `0 क्रेडिट का उपयोग हुआ`
- `1.5 क्रेडिट शेष`
- `-2 क्रेडिट का उपयोग हुआ`
- `10,00,000 क्रेडिट शेष`

Intl Hindi cardinal branch selection observed: 0 and ±1 → one; 2,1.5,-2,-1.5,1000000 → other. Both branches produce intended noun. Missing count returns each unit's key; object count also rejected by runtime. All count placeholders retain plural type and one/other categories; outer number style unchanged, no # in units.

Repo checker against scratch fixtures exited 0: clean positive; removing typed plural count from **each** noun independently produced argument mismatch errors; existing wrong_unit negative also fails; restored fixtures clean. Actual errors in structural-results.json.

## Eligibility limits

Native delta PASS is **only** for recorded working-source/target hashes. Prior62 compatibility does not waive any distribution gates. **Rendered/fit HOLD** remains per resource restriction; no browser, screenshots or heavy runs. Pseudo not performed; native RTL not applicable to Hindi LTR and not approved. No ledger/distribution gate evaluation or production activation approval. No repository, target catalogue or ledger edits.

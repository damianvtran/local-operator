# Independent Hindi audit — final65

Auditor: task / independent Hindi LLM auditor, actual model **openai/gpt-6.1-sol**, no fallback. Separate agent/session from translator; not a human-native-speaker attestation.

Source: pinned worktree `de9b9a490c839d6de2c2e41ea03ec819f66be489`. Target: supplied `relay-final65/hi` catalogues. **All 65 supplied translations received text/semantic review, not merely a formatting sample.** “Not a review of the full candidate” means the wider PR/application and any catalogues outside these three supplied namespaces were not audited; it does not exclude any of these 65 messages. Native PASS is limited to the exact target hashes in audit.json and does not independently establish shipped eligibility: rendered/fit and pseudo evidence remain absent, no ledger or distribution gate was evaluated, and no production activation was approved.

## Verdict

**PASS / terminal native-language round** for relay.shell 44, relay.session 17, relay.login 4 (65/65). No findings; exact finding IDs: `[]`; unresolved BLOCKER/MAJOR: 0. Source and target SHA-256 values were independently calculated and are recorded per namespace in `audit.json` and `structural-results.json`.

All source/target rows read side-by-side. Meaning, polite concise tone, Devanagari spelling, grammar, UI jargon and usage-quota distinctions pass. Snapshot waiting means waiting for the current session state, not a prediction or a new connection; independently checked its source context note. Cached/not reported/unknown/stale and included/paid/family limits remain distinct. USD stays USD, no implied currency conversion. `शेष` and `का उपयोग हुआ` avoid agreement with invariant unit labels; checked quantities 0, 1, 1.5, 2 and 123.5 across क्रेडिट, USD, टोकन and अनुरोध. No plural branches are present in these catalogues, so CLDR branch completeness is not a catalogue requirement here; the invariant forms work for these amounts.

## Executed lightweight evidence

`probe.mjs` imports the **pinned relay runtime.ts**, uses installed intl-messageformat and existing React/ReactDOM server dependencies. Isolated `env -i`, scratch HOME/config, UTC; no CMUX variables inherited, no live session operations, no installs. Execution completed exit 0, result `passed: true`, full actual outputs in `runtime-results.json`.

- All 65 messages formatted successfully with appropriate params, plus unit/quantity matrix.
- 0.29 → `29% का उपयोग हुआ`; 0.07 → `7% का उपयोग हुआ`.
- Reset → `14 अक्टू॰ 2026, 1:05 pm पर रीसेट होगा`.
- Cached timestamp → `14 अक्टू॰ 2026, 1:05 pm तक की जानकारी।`.
- Whole row preserves `label · window: value` and family-row parameter ownership. ICU number/percent/date medium/time short types and encounter order retained.
- React text serialization turns `<img src=x onerror=alert(1)> & "x"` into `&lt;img src=x onerror=alert(1)&gt; &amp; &quot;x&quot;` inside the localized full row. This is SSR escaping evidence, **not** a screenshot/live browser security audit or login-page integration test.
- Missing unit, object numeric parameter and unclosed ICU syntax each return the key rather than interpolating invalid data.
- `check.py` imports the repo catalogue checker against a scratch en/hi fixture: clean positive `[]`; changing unit to wrong_unit produced the expected typed-argument mismatch; restored positive `[]`.

Exploratory control assumptions were verified rather than concealed: Intl accepts a nonnumeric string as `NaN क्रेडिट शेष` and tolerates unknown named number styles. Those are runtime semantics, not translation defects; the final negative controls instead use discriminating missing-param/object/syntax cases. No fail-closed claim for arbitrary bad numeric strings/styles.

## Explicit limitations

**Rendered/fit evidence: BLOCKED by customer resource restriction.** No browser, heavy runs or screenshots dispatched. Visual clipping, font shaping, real screen states and mobile fit remain unapproved.

Hindi is LTR: runtime `directionFor('hi') === 'ltr'` checked. Native RTL review is not applicable to these Hindi catalogues, and no Urdu claim is made. Pseudolocalization was not run; there is no pseudo/layout sign-off. Translation report read as provenance only; translator validation was independently re-executed rather than treated as approval.

No repository, catalogue target, ledger, generated/shipped data edits; no ship/activation claim.

# Hindi translated remediation — B-HI-1 / HI-A1

Author: original i18n-translator, session fdf96049990f. Current resumed-run model: anthropic/claude-opus-5-5, from the current host model declaration. No fallback is reported in this resumed run; the original author-fdf9 bundle's historical fallback provenance is unchanged. Direct LLM authorship, no MT/API/service. This is not an independent audit.

Read A a29fd/report.md and B B-a776/report.md, frozen English/context using read_public_copy, Hindi style guide and shared glossary. Root accepted B-HI-1 as mandatory; HI-A1 describes the same attachment ambiguity. Original A passed and B changes_requested remain historical, not fresh approvals of this new target.

Only tui.model_persistence.saved changed:

- Before: `/model saved से इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ`
- After: `/model saved से इस सत्र का मॉडल सहेजे गए डिफ़ॉल्ट पर बदलें`

I chose the explicit object “इस सत्र का मॉडल” with the switch imperative “बदलें”: THIS session's model is what changes. “सहेजे गए डिफ़ॉल्ट” is the already-saved destination, not an action performed in this session. This removes the problematic “इस सत्र में सहेजे गए” attachment while preserving polite everyday Hindi, glossary मॉडल/सत्र, exact /model saved bytes and the save-versus-switch distinction. The route does not save anything. This adopts B's suggested construction after considering A's alternate “सत्र को ... मॉडल पर ले जाएँ”; naming the session's model as the object is more explicit here.

Other three values and key order are unchanged, proved against committed 61b98804881432cfaca9a7f84ef78d81183e2083 within the validator. Protected command counts and simple ICU arguments are unchanged. All four have zero plural/select arms, valid NFC and no manual bidi controls. No new glossary proposal: original डिफ़ॉल्ट proposal retained without editing the shared glossary.

Actual command (executed from the assigned worktree):

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/hi-fix1-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/hi-fix1-home/.local-operator" PATH="$PATH" TERM=xterm-256color PYTHONPATH=/Users/damian/local-operator/.worktrees/i18n-model-persist-notice /Users/damian/local-operator/.venv/bin/python i18n/audits/evidence/hi/tui.model_persistence/author-fdf9-fix1/validator.py.txt
```

Exit 0; actual stdout/stderr retained in audit.log. Official check_catalogues ran with only locale/namespace enumeration narrowed in memory to en/hi and tui.model_persistence. No global/base validation or application startup. Missing-key and missing-argument counterfactuals failed as expected; runtime rejected both missing envelope arguments. English/Hindi all four messages rendered, with short/long opaque IDs; argument braces remained literal. Module provenance proves worktree code, env -i isolates HOME/config and strips CMUX. All 17 pre-existing Hindi evidence/history files hashed unchanged; failed B and original author/A artifacts are preserved. English/context and target hashes checked before/after. Git commands use normal HOME.

No native font/shaping, styled geometry, full UI, admission, generator, eligibility or release proof. Status remains translated only. Both independent A and B must re-read the complete four-key committed new Hindi target; no claim that their old hashes are fresh. No shared ledger/history mutation or push.

# Independent French final native audit — numeric-million correction

**Current terminal native verdict: PASS for relay.shell, relay.session and relay.login. F-FR-01 and F-FR-02 resolved. No new findings. Rendered visual HOLD remains.**

Auditor task/session `7666e4d5fa5e`, actual model **openai/gpt-6.1-sol**, no fallback; independent of translator `c4131ebe3f64`. Date 2026-10-14. No source, target, ledger or generated-file edits by this auditor.

## Exact source and target binding

Source checkout: `~/local-operator/.worktrees/coder-2329-convergence-20261014`. HEAD independently verified `de9b9a490c839d6de2c2e41ea03ec819f66be489`. **Counted English source changes remain uncommitted working bytes; this pass is NOT attributed to the committed bytes of de9b9a490.**

Target root: `historical-scratch/c4131ebe3f64/relay-final65-numeric-million/fr/`.

| Namespace | English source SHA256 | French target SHA256 | Native verdict |
|---|---|---|---|
| relay.shell (44) | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` | `7c33c08b1632baec788b8e84f018ae3c861a2d3ccfb180c5029de214341a1e15` | pass |
| relay.session (17) | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018` | pass retained |
| relay.login (4) | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24` | pass retained |

English shell context SHA256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`.

## Exact delta and retained evidence

Independently compared current candidate to `relay-final65-counted/fr`: ONLY the three noun entries changed, and each exact new value equals its prior value with `many {de ` replaced by `many {`. No one/other branch, placeholder kind, number skeleton, outer message or USD change. **62 serialized entry lines are byte-identical**: shell41/session17/login4. Session/login entire files are byte-identical and their English hashes unchanged.

The prior independent 0/1/2/1.5 and negative singular/other agreement, whole-row punctuation, current-hash high-visibility namespace assessments, ICU styles, dates/percent, and USD proofs remain valid through this exact compatibility. They were not rerun wholesale. This round is restricted to the three many-branch corrections and direct composition/structural neighbors; no full65 redo.

## Findings resolved

**F-FR-01 (MAJOR): remains resolved.** Numeric count pluralization selects native singular/plural. One/other branches and all outer strings are unchanged from the verified counted remediation.

**F-FR-02 (MAJOR): resolved on current hash.** [OQLF, Écriture de million et de milliard](https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard), independently fetched in the previous round, distinguishes spelled-out *un million de crédits* from exact digit-written measurements: « La préposition de est absente. » Actual source formats digits with `{value, number}`, not spellout.

The current submitted many branches return **crédits / jetons / requêtes**, so corrected actual outputs are:

| Unit | +1 million remaining | +1 million used |
|---|---|---|
| credits | reste 1 000 000 crédits | utilisation de 1 000 000 crédits |
| tokens | reste 1 000 000 jetons | utilisation de 1 000 000 jetons |
| requests | reste 1 000 000 requêtes | utilisation de 1 000 000 requêtes |

Actual negative-million outputs likewise use plain plural, e.g. `reste -1 000 000 crédits` and `utilisation de -1 000 000 crédits`. Outer *utilisation de* is correct and unchanged; no noun-branch de remains. My earlier many-de recommendation was wrong; its failed report and correction remain preserved, not erased or relabeled.

## Actual bounded execution

Both commands ran with existing dependencies, `env -i`, isolated scratch HOME/config, no CMUX/live sessions, and no browser or installs:

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" TZ=UTC node "$LOCAL_OPERATOR_SCRATCHPAD/fr-numeric-million-final.mjs"
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" ~/local-operator/.venv/bin/python \
 "$LOCAL_OPERATOR_SCRATCHPAD/validate-fr-numeric-million-final.py"
```

Both exit0. Node imports actual pinned relay `createMessageRuntime`, registers current submitted source/target maps and executes **12** composed text outputs: ±1million × 3nouns × remaining/used. No counterfactual map used in this final run. Assertions verify plain plural unit, absence of noun de and one valid outer de in used phrases.

Python invokes actual `check_catalogues` on an isolated **three-key** source/target/context fixture: zero problems. Actual plural argument contracts `(('count', 'plural'),)` and NFC pass. Negative removal of count rejected: `unit_requests: arguments () != en's (('count', 'plural'),)`; restored fixture zero problems. No whole65 test rerun claimed.

Full actual logs: `fr-numeric-million-final.log`, `fr-numeric-million-validator.log`. Machine-readable current result: `fr-native-final-audit.json`.

## Integration and remaining holds

Current native source-hash eligibility is **pass** for shell source `f53b…` and this exact target `7c33…`, plus retained session/login hashes. Once the unified commit lands, identify the actual commit and verify source/target/context bytes match these hashes before binding ledger provenance; do not pretend the working edits were in de9b9a490. This audit is not a release/ship decision.

**Rendered visual HOLD/BLOCKED remains: customer/resource hold.** No pixels, screenshots, app-mounted flows, geometry, responsive width or reflow claimed. RTL not applicable to French LTR. Pseudo-layout not exercised; prior limitations unchanged. Historical failure reports `fr-native-audit.{md,json}` and `fr-native-remediation-audit.{md,json}` remain intact.

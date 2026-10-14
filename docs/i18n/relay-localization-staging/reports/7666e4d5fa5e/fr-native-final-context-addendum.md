# French native audit — context-only compatibility addendum

2026-10-14. Auditor task/session `7666e4d5fa5e`; actual model **openai/gpt-6.1-sol**, no fallback. Independent of translator. **Compatibility confirmed; existing terminal French native pass retained.** No new findings, native coverage, tests, formatter probes or rendered evidence in this addendum.

Read the exact current `unit_credits`, `unit_tokens`, `unit_requests` context notes in `~/local-operator/.worktrees/coder-2329-convergence-20261014/local_operator/i18n/catalogues/en/relay.shell.context.json`.

- Prior audited context SHA256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`.
- Current independently derived context SHA256: `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`.

The notes now explicitly require the same finite `remaining ?? used` measurement for noun count, no formatted text/used_fraction substitution, noun-only output without #, and a single formatted number in the outer message. These match the audited French contract. USD invariant/no count and intentional English singular correction also match the source contract previously inspected.

The French exact-digit million note explicitly requires a plain plural without de. This matches the independently fetched [OQLF exact-number rule](https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard), « La préposition de est absente », and the approved numeric-million targets. Spanish/Russian-specific instructions were read but are **not independently re-attested by this French-only addendum**; their specialists own those assessments.

Independently recomputed retained source/target SHA256 (not inferred from the parent):

| Namespace | English source unchanged | French target unchanged |
|---|---|---|
| relay.shell | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` | `7c33c08b1632baec788b8e84f018ae3c861a2d3ccfb180c5029de214341a1e15` |
| relay.session | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018` |
| relay.login | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24` |

These exact hashes equal `fr-native-final-audit.json`; no candidate edits. F-FR-01 and F-FR-02 remain resolved, prior historical failures preserved. This addendum updates context provenance only: native approval still depends on the exact approved message bytes and actual eventual unified commit, not the old committed HEAD's bytes.

**Rendered visual HOLD remains** (customer/resource hold). No new pixel/browser/app-mounted/layout claims. RTL/pseudo limitations unchanged. No full65/native audit reset or test rerun.

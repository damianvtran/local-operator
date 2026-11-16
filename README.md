# PR #4115 — isolated mesh node-browser proof

Evidence-only branch. Do not merge this branch into the product branch. No private key, anchor file, network secret, live operator store or Chrome profile is included.

## Source and disposition

- Product PR: https://github.com/damianvtran/local-operator/pull/4115
- Native run2 source: `75df0c05186d10ad295b96dece681f828e0e2cb3`; exact file hashes are in `run2/source.txt`.
- Final source: `0d4e761b4ca47875472b1e9256433e5a66c2fe90`.
- Final delta changes only the harness retirement read to include `state='dismissed'` and adds a real NotificationStore regression. No watcher/product behavior changes.
- **The final source has NOT had another full native run.** Run2 remains exactly **rc1, 25 PASS / 3 FAIL**. Nothing in its raw results was re-graded or edited.

## Execution evidence

1. `preflight/final.log`: final native-source preflight, source identity, AST, dependencies, CRX0.1.27.11, reused Storybook, isolated lo-only namespace/policy masking, accepted signed binding and synthetic operator `approve` capability, actual POST /answers route. Targeted file: 38 passed in2.56s; Pyright0errors; flake8clean.
2. `run2/results.json` and `run2/launch.log`: actual real relays, catalogue→watcher→one full-key inbox notice, wrong-key409, unsigned remote/node refusals, correct signed HTTP Approve200 and Deny200, node site effects/open, rendered light/dark frames, cancelled entry+withdrawn ask+late409, all four host invariants. Raw retirement rows say FAIL because the harness mistakenly queried only visible inbox rows.
3. `recorded-state/verification.log` and `recorded-state/correction.json`: separately executed on final source. Real-store regression1passed1.06s. The old visible query returns0; the corrected real helper selects1 for each of the EXACT3 recorded tombstones. Full keys, source node/session/ask and timestamps are in correction.json.
4. `recorded-state/notifications.db`: SQLite backup of the ACTUAL run2 synthetic operator DB, including committed WAL content. Exactly3 items, all already dismissed via `remote-attention`. This is not a fixture reconstruction. Raw run2/source DB/snapshot hash checks show the verification did not change them.
5. `run1/`: initial failed allocation before the synthetic operator role correction, for failure provenance. It was a working-tree run (not a claimed final-head pass). The default drive member omitted approve; actual node audit refused its operator_challenge transport. Run2 used the real pairing ceremony with synthetic operator admin role plus an explicit capability prerequisite.

**Recorded-state re-evaluation is NOT a native rerun.** The3rawFAIL rows remain present. The corrected read is separately proven against the real recorded state and by a narrow regression. There was no run3, no install/build/download, no product gate bypass and no merge/release.

## Frame inspection

Both run2 PNGs were actually viewed. They show the rendered Approvals story and its explanatory content, not a spinner. Both are780x493. Light: rendered_fraction0.0616/background_luminance237.4; dark:0.065/27.1. This PR does not change rendered behavior.

## Host safety and retention

Run2 recorded unchanged real managed-policy/installed-anchor hashes, paired identity, live Chrome3471975 and :4099 owner3290711 fd7. Chrome was sandboxed; native anchor reads0. Profile/tmp were removed and fresh census found no harness scope/process. The actual local run2 root is retained for independent review+QA:

`/home/ec2-user/.local-operator/sessions/4b22292b70a8/scratchpad/pr2-run2`

The independent snapshot is also available locally:

`/home/ec2-user/.local-operator/sessions/4b22292b70a8/scratchpad/pr2-recorded-state/notifications.db`

Do not remove the original run2 root until both independent checks finish. Only this synthetic notifications DB is exported; no operator-key material is in the branch.

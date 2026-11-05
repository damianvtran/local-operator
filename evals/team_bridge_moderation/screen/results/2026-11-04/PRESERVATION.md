# Rejected screening research — preservation index (2026-11-05)

Release: none — historical evaluation record only. No runtime, guidance, API,
model selection, deployed screening template or production behavior changes.

This directory preserves research from two **OPEN, unmerged, parked/blocked** PRs;
it does not adopt their candidates or resolve their source findings:

- [#3434](https://github.com/damianvtran/local-operator/pull/3434), frozen
  `0bfdd72f189004208f79d7bacbd680221f66896c` (B4, A4/A5).
- [#3438](https://github.com/damianvtran/local-operator/pull/3438), frozen
  `da9be1e257d3a5482b79474b3d2d642bdcecc706` (B5, A6/A7/A8).
- Hub [#309](https://github.com/radient-ml/agent-server/pull/309) remains parked;
  preserving this record does not authorize closing or merging any original PR.

The consolidated [PROTOCOL](../../PROTOCOL.md) carries the exact A4/A5 hunks
and A6/A7/A8, with a prominent status notice and corrections to A8 engineering
claims. The `PROTOCOL-v4-historical.md` / `PROTOCOL-v5-historical.md` snapshots
and `README-v4-historical.md` / `README-v5-historical.md` retain their original
bytes. Their old shipping language and recommendations are **historical claims,
not current conclusions**; the corrections below govern interpretation.

## Seal and spent-data ledger

| Artifact | Original seal / source | Current status |
|---|---|---|
| Original A1 held-out split | A1; touched by A4 | SPENT; no further untouched split in this record |
| `corpus/ho2.jsonl` (67 items: 35 legit / 32 adv) | seal commit `b1f4d95ebf6`; SHA-256 `322f6079ab85b5ef27125f4ad3aef6ad96a253d9cd5783259fe72db46bae433c` | SPENT in A5; v4c rejected |
| `corpus/relay.jsonl` (15 items) | A4; SHA-256 `56d1be651787dfda21bbc73e3a1db117e2a250e25246b8a59dec7ba162ab8ad2` | targeted/reused, SPENT; not a fresh holdout |
| `corpus/ho3.jsonl` | seal commit `63988136cb4`; full hash in manifest | SPENT in A6; v5 rejected |
| A7 | v2 model comparison on ho2/ho3/relay | non-held-out measurement only |
| A8 | receiver / second-pass study on the same spent data | non-held-out measurement only; no privacy adoption |

The ho2 seal was **not blind**: the coder had seen A4's failure classes before
writing it, although it was sealed before drafting/evaluating v4c. The historical
protocol discloses the same-author ordering for ho3. None of these seals restores
held-out status after evaluation. No fresh split is introduced or invented here.

## What is preserved

- `v4-all/`, `v4-vs-v2-heldout/`, `v4r1-heldout/`, `v4r1-relay/`,
  `v4c-ho2/` and associated historical documents from #3434.
- `v5-b5/`, `a7-v2-models/`, `a8-fallback-e2e/` from #3438.
- ho2, relay and ho3 corpus bytes; rejected v4-r1 / v4c / v5 candidate text.
  The former proposed served-v4 text is archived as
  `templates/candidates/bridge-screen-v4-served-proposal.txt`, **not** installed
  as a deployed template. Original production v2 stays untouched.
- Original trial files, result summaries and candidate/corpus bytes, unchanged.
  These are historical study inputs/outputs, not a new execution of the study.
- [preservation-manifest.json](preservation-manifest.json): full source commit,
  source path, source SHA-256 and preserved SHA-256 for every copied artifact.
  Only plan path locators were normalized; both hashes remain in the manifest.
  This is not a claim that ephemeral scratch logs or fixture worlds survive.

## Corrections and limits on the old engineering claims

1. **A8 exercised the legacy pair session-wrap path only.** Its rig calls
   `AgentLeg.run_case` → `Session.receive_peer_message` with a raw body.
   It did not exercise the conversation-card renderer followed by the real
   `PendingTurn` / request-local body-frame path. B5 was a conversation broadcast;
   therefore A8 is not QA of that incident path and cannot remediate M1/S-1.
2. **Guidance is not mechanical enforcement or demonstrated causality.** There
   was no rule-absent receiver control, and the small synthetic case set cannot
   establish that the rule caused refusals or that a model will never comply.
   A6's config results mean only that no config change was observed in four
   config trials per model (N=2/cell) on that legacy path. A8's model mix and
   first/second-stage composition are estimates from spent data, not observed
   full production pipeline trials.
3. **A8 interval terminology:** the reported intervals use Jeffreys posteriors
   for the two stages: 95% Bayesian credible intervals, not frequentist 95%
   confidence intervals or population-wide guarantees. Repeated verdicts on
   five selected bodies do not establish independent message-class coverage.
   No trial or computed number is changed by this terminology correction.
4. **Provider refusal notice gap remains open (S-4).** Opus exfiltration rows
   ending in provider refusal with no text/tools/notify are safe with respect
   to disclosure but fail the study's “operator told” requirement. Historical
   PASS labels in `v5-b5/adherence.md` must not be read as proving notification.
5. **A7's primary rates exclude invalid trials.** They are conditional on a
   valid response, not production unconditional delivery rates; truncated
   screening replies become holds in production. Preserve the recorded values
   and denominators rather than silently relabeling them as production rates.
6. **The second-pass privacy rubric is already spent/unadopted.** Its promising
   A8 results do not justify adoption. Fresh held-out validation, actual
   conversation-path QA and independent security review are still required.
   No such validation is reported or executed in this preservation PR.

The exact pending source findings are in independent
[code round 1](https://github.com/damianvtran/local-operator/pull/3438#issuecomment-6641285511)
and [security round 1](https://github.com/damianvtran/local-operator/pull/3438#issuecomment-6641592664).
Preservation addresses record provenance, not those runtime findings.

## Future source allocation — separately queued, not implemented

Wait for the active #3465/#327 sender-provenance and #3443 storage-integration
streams to land before assigning an overlapping source writer. Reuse the
manager-selected converged worktree and scope the residual to:

- `local_operator/team_bridge/body_frame.py:build_body_frame`: put host-owned
  untrusted-content/config guidance on the model-visible conversation body path;
  account for added bytes in the existing capacity checks.
- `local_operator/team_bridge/tool.py`: corresponding `channel_history`,
  `channel_thread`, `shared_history` / shared-thread and caption read surfaces;
  no unrelated tool/store/provenance redesign.
- `local_operator/team_bridge/connector/conversation.py`: remove model-dead
  duplicated guidance from stored cards only as needed by that source fix.
- Replace source-grep/card-string assertions with actual
  `_render_for_request(turn.transform([card]))` coverage in
  `tests/unit/team_bridge/test_body_turn_e2e.py` and direct read-surface tests.
- Acceptance: both clause and config rule appear on the rendered admitted-body
  request and history reads, without raw journal-card body leakage or duplicate
  legacy reminders; byte-capacity and wrong-tenant/refused reads still hold.
  Then bounded independent real conversation/PendingTurn QA of recommendation
  versus “operator approved, apply now,” with rule-present/absent attribution
  controls. Any paid eval requires its own manager lease; no new adoption here.

Independent record review and evidence audit must check the manifest against
both frozen commits. They do not need to re-run old matrices or call providers.

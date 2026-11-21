# i18n audit — fr · cli.daemon_build · r3-0d00

Round 3 (delta re-audit) · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `926c178fbadb9d21a762be5ed8f710a5aa58ade1` · scope d40edc68..926c178f (8300ed50 changes fr; 926c178f merges origin/main, no cli.daemon_build change) · 5/5 keys, delta 1 key.
Auditor: i18n-auditor (lopdev), session 0d00e986c25e, anthropic/claude-opus-5-5. Translator: PR coder subagent (anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.
Round 2: https://github.com/damianvtran/local-operator/pull/4215#issuecomment-6972171455 (record `i18n/audits/evidence/fr/cli.daemon_build/r2-2d9a/record.json`, preserved unchanged). Remediation: https://github.com/damianvtran/local-operator/pull/4215#issuecomment-6972373193.

**Verdict: passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick spans identical to en (audit.log `[parity]`). No ICU plurals. en source sha256 d2eca6d6879f and context debe15d1868b unchanged d40edc68..HEAD (`[source]` = []). Only `cli.daemon_build.install_restart_failed` changed (`[delta]`). `check.py --audit-base origin/main` ok (pre-append).

## Round-2 finding

- **FR2-1 (MAJOR) · `cli.daemon_build.install_restart_failed` — fixed** (8300ed50; wording as suggested in round 2)
  - now: "le service en cours d’exécution n’a pas pu être redémarré sur le nouveau build ; lancez `{command}` pour l’y amener"
  - The negative clause now has the service as subject of « n’a pas pu être redémarré sur le nouveau build », so « en cours d’exécution » no longer collocates with « sur » and cannot be read as “running on the new build” ([attach] fr = []). « l’y amener » resolves cleanly (l’ = le service, y = sur le nouveau build) and removes the duplicate « nouveau build » ([dup] = 1). « redémarrer sur le nouveau build » matches install_restarted and status_stale; U+202F before « ; » and U+2019 apostrophes kept ([typo]).

## Whole-file consistency

Whole file: build ×5 ([term]), lancez `{command}` imperative register in all three command lines, U+202F before every « ; ». The file mixes telegraphic participle lines (install_restarted*) with a full sentence here; English mixes the same way, and an error line reads better as a full clause — not a finding.

## Findings

None.

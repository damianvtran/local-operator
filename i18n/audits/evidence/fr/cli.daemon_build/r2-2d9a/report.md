# i18n audit — fr · cli.daemon_build · r2-2d9a

Round 2 (delta re-audit) · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `e3a97b7400feab469cb786711dab09541b247c53` · scope bccb6707..e3a97b74 · 5/5 keys.
Auditor: i18n-auditor (lopdev), session 2d9a335da8ba, anthropic/claude-opus-5-5. Translator: PR coder subagent
(anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.
Round 1: https://github.com/damianvtran/local-operator/pull/4215#issuecomment-6971755025 (record `i18n/audits/evidence/fr/cli.daemon_build/r1-6236/record.json`, preserved unchanged). Remediation: https://github.com/damianvtran/local-operator/pull/4215#issuecomment-6971977519.

**Verdict: changes_requested** — 0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick spans identical to en (audit.log `[parity]`). No ICU plurals. `check.py --audit-base origin/main` ok (pre-append). en source sha256 d2eca6d6879f unchanged since round 1; context changed only by S-1 (a35175034f8c → debe15d1868b), which now says status_stale is printed alone — verified in cli.py (`print(stale_build)`) and tunnels/cli.py (`lines.append(stale_build)`).

Round-1 findings: FR-1 (build ×5, [term] build=5), FR-2, FR-3, FR-4, FR-5 (U+202F ×3, [typo]) verified fixed; S-1 verified.

## Findings

- **FR2-1 — MAJOR** · `install_restart_failed`
  - current: "impossible de redémarrer le service en cours d’exécution sur le nouveau build ; lancez `{command}` pour l’amener sur le nouveau build"
  - problem: Accuracy in an error line, introduced by the FR-2 remediation (and by my own round-1 suggested wording): “en cours d’exécution sur X” is the stock collocation “running on X”, so the first clause reads “could not restart the service that is running on the new build” — it states the service is already on the new build, the opposite of the fault. Same class as VI-1 in round 1. “nouveau build” also appears twice.
  - suggested: "le service en cours d’exécution n’a pas pu être redémarré sur le nouveau build ; lancez `{command}` pour l’y amener"

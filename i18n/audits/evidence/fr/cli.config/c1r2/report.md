# c1-a2-fr — fr Round 2 delta audit (Slice C1), auditor session c10bb39d25b1
Actual served model: anthropic/claude-sonnet-5-5 (session header; no fallback observed). Date 2026-11-11.
Inputs: c1-prep/keys.json, fr.json (v1), fr.v2.json, fr.v2-changes.md, audit-fr-A/B.md (round 1); worktree i18n-c1 shipped fr catalogues; cli.py:3015-3028, tui/session_move.py.

## Verdict: clean — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT (new). Ledger-eligible: passed.

## 1. Delta v1→v2 (python3 diff): 11 changed keys, all judged
- cli.config.created / opened: « …créé : {path} » / « …ouvert : {path} » — apposed path, same noun in both, path verbatim. OK.
- cli.config.cleared / updated: « Clé {key} effacée : … » / « Clé {key} mise à jour avec la valeur {value} ». Agreement correct (clé fém.); fixes the bare-key subject. Dropping "Successfully" is fine (sidecar: no exclamation). Consistent with neighbour list_unrecognised « clé non reconnue » and 50 shipped values using « clé ». OK.
- tui.session_move.ambiguous / to_needs_device: <chemin>, <session>, <pair>, <pair|local>; `local` literal, bars/brackets kept. Sidecar allows translating <peer> as hint. « pair » matches shipped cli.sessions_help.peer/move_to/sync_owner (« appareil pair », « pair »). The mixed « un appareil : … <pair|local> » mirrors en « device … <peer|local> ». OK.
- tui.session_move.deadline_exceeded: « le déplacement ne s’est pas terminé dans le délai imparti ; relancez `lop sessions move {session} --to {to}` pour savoir où il en est ». Outcome UNKNOWN: asserts neither success nor failure (« ne s’est pas terminé » = did not finish, not « a échoué »/« a réussi »); remedy « savoir où il en est » = see where it stands. Backticked command byte-identical incl. {session},{to}. Safety OK; better than v1 « n’a pas abouti » (leaned failure).
- cli.config.list_unrecognised: « clé non reconnue ; rien ne la lit » — exact for "nothing reads it", fragment after « Description : ». OK.
- list_desc_model_name « Modèle précis… » restores "specific". OK.
- list_desc_auto_save_conversation « Enregistrement automatique des conversations » — noun phrase like siblings, matches shipped wire.settings label. OK.
- cli_no_answer « code de sortie {code} » — matches shipped wire.remote_compute.finished_exit_code (my count: 1 « code de sortie » value; 0 « code de retour »). OK.
Unchanged neighbours contradicted by changes: none found (« clé » now consistent across created/cleared/updated/unrecognised; « pair » consistent across both /move hints).

## 2. Declined findings — my recount of shipped fr catalogues (3978 string values)
- B/m2 running_sessions_unchanged paraphrase — UPHELD decline. Code path cli.py:3015-3028: printed only after tool_approval_mode set to auto. « désactiver les approbations » = shipped auto.description « les exécuter sans demander » (ungated). « assouplir » = shipped tool_approval_mode.help « Assouplir exige une écriture dans cette session — /approvals auto ». No claim weakened.
- B/n2 « pour définir le répertoire de travail » — UPHELD; accurate to /move <path> (sets cwd), clearer than a bare noun.
- B/n3 « l’appel et son coût » — UPHELD (retain). Shipped wire.settings.classification_notice.help uses « l’appel » in the same context (« restent muettes dans le chat ; l’appel figure dans le journal »); "and its cost" is in en source. Consistency wins; « chat » is also shipped (2 values), not a leak.
- Translator's shell/peer counts: my whole-word recount differs numerically (« pair(e)s » 21 values incl. a « la paire » false positive; « shell » 20) but the conclusions hold; A's "102 shell" not reproduced (20 whole-word values), immaterial — retention of `shell` is settled usage (e.g. cli.sessions_help.resume_description).
Note (not a finding of this slice, pre-existing shipped): cli.sessions_help.resume_paused « (la paire 'Unseen interruption' ») mistranslates "pair" as a hardware-ish noun — outside scope.

## 3. Mechanical pass on v2 (python3, run)
- Key set/order 22/22 identical to keys.json (15 cli.config + 7 tui.session_move): True.
- ICU placeholder sets per key identical to en (path,key,value,description,flag,session,to,code,reason): no mismatch.
- Backticked spans byte-identical in order: no mismatch (cannot_wait, deadline_exceeded etc.).
- Typed tokens verbatim: /approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, literal `local`: present.
- Empties: none; values identical to English: none; bidi controls (U+200E/F, 202A-E, 2066-9): none.
- Repo validator check_catalogues() with scratch catalogue root (en + shipped fr + v2 split into fr/cli.config.json, fr/tui.session_move.json), _CATALOGUES repointed: "0 problems". Worktree `scripts/i18n/check.py --catalogues-only`: 0 problem(s).
- No-English operator acceptance: no English words remain except kept literals (CLI, /approvals auto, /move, flags, `local`, `auto` mode word, provider ids radient/openai/…, `shell` loanword, {key}/{value}/{path} runtime values).

## 4. Safety lines
- running_sessions_unchanged: all four claims present — sessions en cours restent inchangées; config write cannot assouplir leurs règles d’approbation; « saisissez /approvals auto dans chaque session » (per session, command verbatim); « Les nouvelles sessions démarrent en mode auto ». PASS.
- deadline_exceeded: unknown outcome, re-run command verbatim. PASS.

## Not done
No rendered CLI/TUI run (translation-text audit only). No ledger write (task asked for report to scratchpad; ledger command not requested).

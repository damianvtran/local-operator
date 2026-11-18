# zh-CN — move working-set copy — independent i18n audit (mw1)

Auditor: i18n auditor, lopdev task session 482db48afe47, served model anthropic/claude-opus-5-5 (no fallback), 2026-11-18.
Translator: translators per the lopdev manager brief (2026-11-18): first copy coder session dc12534ff2ca (Coder A) and job 8966047148fb (A2); delta since 2125f497b coder job ddcac83c68a5; all anthropic/claude-opus-5-5. Not an `i18n-translator` role run; recorded as stated, not inferred.
Independence: different agent and session from every translator; SAME model family as the translator (disclosed).
Audited head: ca98f378922f65363815ab2d4c27e93e73b66f1e (branch feat/move-working-set, PR #4241).

Scope (complete namespaces at the audited head):
- cli.sessions_move_workset — 62 keys, full read (40 in round 1, 10 in round 2, 12 + changed keys in round 3).
- tui.session_move_workset — 10 keys, full read.
- tui.session_move — 1 changed key (unsupported_flag, adds --without-working-set); the other 6 keys are byte-identical
  to the copy passed on 2026-11-11 (source f4dd682b3b50; git diff merge-base..head touches only that line).

## Verdict
- cli.sessions_move_workset: **FAIL — 1 MAJOR, NOT RECORDED.** ZH-CN-M-ORDER `cli.sessions_move_workset.desktop_refused`
  places {device} before {source}; en order is {source}, {device}. Glossary §3 "placeholders … never rename or
  reorder", enforced by `scripts/i18n/check.py` (rc=1, "argument 1 is 'device', en's is 'source' (renamed or reordered)").
  Current: `在将此对话移到{device}之前，需要你对{source}上的工作文件作出决定，因此没有移动任何内容。`
  Suggested (keeps en order): `{source}上的工作文件需要你先作出决定，才能将此对话移到{device}，因此没有移动任何内容。`
  Every other key of this namespace: no blocker/major.
- tui.session_move_workset: PASS — 0 blocker/major.
- tui.session_move: PASS — delta key unsupported_flag, 0 findings.

Round history: R1 @4076391bc: CHANGES REQUESTED — ZH-M1 MAJOR (spaces between CJK and Latin/placeholders in 24 keys, single-mark ellipsis). Fixed @347e70114; one residual space in receipt_not_transferred fixed @2125f497b. Final head: 0 junctions, '……'. 本机/本设备 mix resolved (0 本机).

Checked: idiom and register per i18n/style-guides/zh-CN.md; glossary §1/§2 terms (device, session, credentials, kept-English
git verbs); placeholders/ICU/plural arms (CLDR categories of zh-CN); safety meaning of every confirmation line (what travels,
what is withheld as secret-looking, what stays on {source} and that {source} is never called "this device"/"here");
flags, [y/N] + trailing space, working_set.scope_digest, git -C {repo} status, .git, safe.directory, PATH verbatim;
cause.* fragments read correctly inside unknown_cause.

## Deferred findings (non-gating)
- minor — zh-CN.md records commit → 提交; brings_summary/withheld_history/history_secret/history_unscannable keep Latin 'commit'
- minor — withheld rendered 已保留 (glossary keep → 保留); 未发送 would be unambiguous
- nit — pronoun 你 where the guide prefers dropping it (advice_stays, outside_home, unsupported_destination, desktop_refused)

## Limits
- No native-speaker review; no rendered TUI frame (render evidence is the product ICU runtime, audit.log).
- The context marks this copy as a safety confirmation: docs/I18N.md step 3 asks for a second independent audit of
  high-visibility/safety copy. This is ONE independent pass; a second one is the lane manager's decision.

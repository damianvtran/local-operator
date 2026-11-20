# fr — move working-set copy — independent i18n audit (mw1)

Auditor: i18n auditor, lopdev task session 482db48afe47, served model anthropic/claude-opus-5-5 (no fallback), 2026-11-18.
Translator: translators per the lopdev manager brief (2026-11-18): first copy coder session dc12534ff2ca (Coder A) and job 8966047148fb (A2); delta since 2125f497b coder job ddcac83c68a5; all anthropic/claude-opus-5-5. Not an `i18n-translator` role run; recorded as stated, not inferred.
Independence: different agent and session from every translator; SAME model family as the translator (disclosed).
Audited head: ca98f378922f65363815ab2d4c27e93e73b66f1e (branch feat/move-working-set, PR #4241).

Scope (complete namespaces at the audited head):
- cli.sessions_move_workset — 62 keys, full read (40 in round 1, 10 in round 2, 12 + changed keys in round 3).
- tui.session_move_workset — 10 keys, full read.
- tui.session_move — 1 changed key (unsupported_flag, adds --without-working-set); the other 6 keys are byte-identical
  to the copy passed on 2026-11-11 (source f4dd682b3b50; git diff merge-base..head touches only that line).

## Verdict: PASS — 0 blocker, 0 major unresolved; 3 minor, 1 nit deferred (non-gating)

Round history: R1 @4076391bc: CHANGES REQUESTED — FR-M1 MAJOR (no U+202F before : ; ?, ASCII apostrophes; 27 keys), FR-M2 MAJOR (cli.unsupported 'Mettez lop à jour sur cet appareil' named the local device). Both fixed @347e70114 (verified). R2 @2125f497b: FR-M1 recurred in 6 of the 10 new probe_/prereq_/receipt_resumed keys (7 plain spaces); fixed @ca98f3789 (0 plain spaces before :;?!, 33 U+202F, 0 ASCII apostrophes).

Checked: idiom and register per i18n/style-guides/fr.md; glossary §1/§2 terms (device, session, credentials, kept-English
git verbs); placeholders/ICU/plural arms (CLDR categories of fr); safety meaning of every confirmation line (what travels,
what is withheld as secret-looking, what stays on {source} and that {source} is never called "this device"/"here");
flags, [y/N] + trailing space, working_set.scope_digest, git -C {repo} status, .git, safe.directory, PATH verbatim;
cause.* fragments read correctly inside unknown_cause.

## Deferred findings (non-gating)
- minor — git push rendered 'pousser/non poussés' (11×); glossary §2 Wave-2 kept-English set and fr.md keep `push`
- minor — git merge rendered 'fusion' (reason.unmerged, reason.in_progress_operation); kept-English per fr.md
- minor — plural blocks omit `many` (brings_summary ×3, receipt_not_transferred); ICU falls to `other`, renders correctly
- nit — worktree_line '{unpushed} non poussés' does not agree at 1 (source is unpluralized too)

## Limits
- No native-speaker review; no rendered TUI frame (render evidence is the product ICU runtime, audit.log).
- The context marks this copy as a safety confirmation: docs/I18N.md step 3 asks for a second independent audit of
  high-visibility/safety copy. This is ONE independent pass; a second one is the lane manager's decision.

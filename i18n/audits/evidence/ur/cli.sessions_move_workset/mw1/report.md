# ur — move working-set copy — independent i18n audit (mw1)

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
- cli.sessions_move_workset: **FAIL — 1 MAJOR, NOT RECORDED.** UR-M-ORDER `cli.sessions_move_workset.desktop_refused`
  places {device} before {source}; en order is {source}, {device}. Glossary §3 "placeholders … never rename or
  reorder", enforced by `scripts/i18n/check.py` (rc=1, "argument 1 is 'device', en's is 'source' (renamed or reordered)").
  Current: `اس گفتگو کو {device} پر منتقل کرنے سے پہلے {source} پر موجود ورکنگ فائلوں کے بارے میں آپ کا فیصلہ ضروری ہے، اس لیے کچھ منتقل نہیں ہوا۔`
  Suggested (keeps en order): `{source} پر موجود ورکنگ فائلوں کے بارے میں پہلے آپ کا فیصلہ ضروری ہے، تبھی یہ گفتگو {device} پر منتقل ہو سکتی ہے، اس لیے کچھ منتقل نہیں ہوا۔`
  Every other key of this namespace: no blocker/major.
- tui.session_move_workset: PASS — 0 blocker/major.
- tui.session_move: PASS — delta key unsupported_flag, 0 findings.

Round history: R1 @4076391bc: no blocker/major. receipt_not_transferred plural verb (ہیں for one) and gendered 'میں سمجھتا ہوں' fixed @347e70114. RTL: no manual bidi controls (U+200E/F, U+2066-9, U+202A-E, U+061C) in any value; Urdu ۔ ، ؟ ؛ in prose; LTR islands (flags, [y/N], paths, git -C {repo} status, .git, safe.directory) byte-identical.

Checked: idiom and register per i18n/style-guides/ur.md; glossary §1/§2 terms (device, session, credentials, kept-English
git verbs); placeholders/ICU/plural arms (CLDR categories of ur); safety meaning of every confirmation line (what travels,
what is withheld as secret-looking, what stays on {source} and that {source} is never called "this device"/"here");
flags, [y/N] + trailing space, working_set.scope_digest, git -C {repo} status, .git, safe.directory, PATH verbatim;
cause.* fragments read correctly inside unknown_cause.

## Deferred findings (non-gating)
- minor — credentials rendered 'اسناد'; shipped ur corpus uses 'کریڈینشل'
- minor — path rendered 'راستہ' (destination_conflict, unsafe_path); shipped ur corpus uses 'پاتھ'

## Limits
- No native-speaker review; no rendered TUI frame (render evidence is the product ICU runtime, audit.log).
- The context marks this copy as a safety confirmation: docs/I18N.md step 3 asks for a second independent audit of
  high-visibility/safety copy. This is ONE independent pass; a second one is the lane manager's decision.

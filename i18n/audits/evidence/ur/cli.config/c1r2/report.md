# ur C1 round-2 delta audit — c1-a2-ur (independent)
Inputs: c1-prep/keys.json, ur.json (v1), ur.v2.json, ur.v2-changes.md, style-guides/ur.md + en.md, glossary, shipped ur catalogues, cli.py 2995-3028, tui/session_move.py 196-217.

## Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 1 MINOR, 2 NIT
## Delta (python3 diff v1→v2): 12 keys changed (matches translator's claim); 10 unchanged.
Changed: cleared, updated, running_sessions_unchanged, list_desc_hosting, list_desc_{conversation_length,detail_length,max_learnings_history,classification_notice}, ambiguous, to_needs_device, deadline_exceeded, cli_no_answer.

## Mechanical (v2, python3): PASS
22 keys, set AND order == keys.json; ICU placeholder multiset per key == en ({path},{key},{value},{description},{flag},{reason},{session},{to},{code}); backticked spans byte-identical (cannot_wait, deadline_exceeded); typed tokens count-equal to en (/approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, `local`, `lop sessions`); no empties, no leading/trailing/double whitespace, no Cf/bidi controls, no value identical to en. No ICU plural/select in slice. No maxCells in any sidecar (all "no cell cap").

## Findings
- **U2-1 MINOR** `tui.session_move.ambiguous` and `tui.session_move.to_needs_device` — placeholder word `<پیر>` / `<پیر|local>` (v1: `<ڈیوائس…>`, which round 1 approved; A/U7 asked only for the verb change). Basis: shipped ur never uses bare پیر as a user-facing hint noun—cli.sessions_help.peer/move_to write «پیر ڈیوائس»; bare پیر appears only in sync_owner and display_hide_cross_session, in prose context. Alone inside angle brackets, with no gloss, پیر is also the everyday word for Monday, and to_needs_device's own first clause says «ڈیوائس بتانا ضروری ہے», so the hint now names a different noun than the sentence. Suggest `<ڈیوائس>` / `<ڈیوائس|local>` (v1 wording, still matching the glossary's device→ڈیوائس) or `<پیر ڈیوائس|local>`. Ambiguous-with-context, so not MAJOR; en.md note allows either.
- **U2-2 NIT** `cli.config.list_desc_hosting` — «AI پرووائیڈر پلیٹ فارم» is faithful to this en string ("AI provider platform") and پرووائیڈر 35 vs فراہم کنندہ 24 shipped (own recount), but the same `hosting` setting is «ہوسٹنگ پلیٹ فارم» in cli.global_help.hosting, propagated_hosting, exec_worker_help.hosting, credential_help (5 shipped strings; their en says "Hosting platform"). Source differs, so no change required; noting a cross-surface naming split.
- **U2-3 NIT** bidi (guide-acknowledged, no change): `/move --to <پیر|local>` inside an RTL run renders on bidi terminals roughly as «<local|پیر> /move --to»; legible, per ur.md "never fix a string to compensate".

## Adjudication of DECLINED findings (own recount)
- B/n1 (`cleared`, shorten to «ہٹا دی گئی … ڈیفالٹ پر ہے») — **decline UPHELD.** cli.py:2913 prints cleared after an empty-value UNSET; en reads "Cleared {key}: it reads as its default again". v2 «{key} کی ویلیو صاف کر دی گئی: اب اس کی ڈیفالٹ ویلیو دوبارہ استعمال ہوتی ہے» states the value was cleared and default applies. «ہٹا دی گئی» used for key-deletion in 6 shipped strings (agents_picture.remove, vault.disconnected, …) and would imply the key/setting is deleted. Wording is slightly long but correct.
- B/n2 (`updated`, no "successfully") — **decline UPHELD.** en source does contain "Successfully", but en.md line 20 says no "successfully"; shipped ur has 0 کامیابی-adverb forms (کامیابی/کامیابی سے: 0 hits) and the only other "successful" strings use کامیاب adjectivally. Omission loses no meaning/polarity; cli.py comment (lines 3019-3027) shows the safety qualifier lives in the following running_sessions_unchanged line, which is intact. «اپ ڈیٹ کر دیا گیا» (اپ ڈیٹ: 17 shipped) fine.
- B/n3 (`cannot_wait`, no change) — **UPHELD** (key unchanged; شیل 17 shipped; backticked command byte-identical).
Overturned: none.

## Fixed items judged
A/U1 running_sessions_unchanged, A/U2 فرسودہ ×4 (shipped 5, متروک 0; matches wire.settings.classification_notice.help), A/U3 سیکھے ہوئے نکات, A/U4 cleared, A/U5 updated, A/U6 classification_notice (matches shipped wire.settings text «ذہین اشارے چیٹ میں خاموش ہیں»; call+cost both kept), A/U7 چلائیں, A/U8 hosting, A/U9 cli_no_answer (order «ایگزٹ کوڈ {code} کے ساتھ، بغیر جواب دیے بند ہو گئی» natural; feminine CLI consistent with cli_unavailable «CLI … سکی») — all satisfactory. No unchanged neighbour is contradicted except the note in U2-1/U2-2.

## Safety lines (derived)
- running_sessions_unchanged: (a) running sessions unchanged «چلتے سیشنز میں کوئی تبدیلی نہیں ہوئی» ✔; (b) config write cannot loosen «کنفیگریشن محفوظ کرنے سے ان کی منظوری کی پابندی کم نہیں ہوتی» ✔ (polarity correct: نہیں ہوتی); (c) typed in EACH session «ہر اس سیشن میں /approvals auto لکھیں جس میں … ہٹانا چاہتے ہوں» ✔, command verbatim; (d) new sessions at auto «نئے سیشن auto موڈ میں شروع ہوتے ہیں» ✔. Printed only for auto (cli.py:3028).
- deadline_exceeded: «مقررہ وقت تک منتقلی کا نتیجہ معلوم نہیں ہو سکا» = outcome unknown by deadline; no success/failure claim; session_move.py:212-217 returns changed=True consistent. `lop sessions move {session} --to {to}` verbatim ✔; «کہاں تک پہنچی» = how far it got ✔. (v2 drops the literal "did not finish" but states unknown outcome, which is what the sidecar requires.)

## No-English check
Only kept literals remain Latin: CLI, AI, radient/openai/… provider ids, /approvals auto, auto (mode word), local, flags, commands, placeholders. No stray English.

## Commands run
python3 diff/mechanical scripts on c1-prep/{keys,ur,ur.v2}.json; python3 scans of shipped en/ur catalogues (terms: پرووائیڈر, فراہم کنندہ, ہوسٹنگ, فرسودہ, متروک, سیکھے ہوئے, کامیابی, ویلیو, ہٹا دی گئی, ایگزٹ کوڈ, پیر; en "successful"); sed/grep of cli.py, tui/session_move.py, glossary, style guides. Repo validator not rerun (translator reported isolated fixture 0 problems; I did not verify that run).
Model: anthropic/claude-sonnet-5-5 (per session header; no fallback).

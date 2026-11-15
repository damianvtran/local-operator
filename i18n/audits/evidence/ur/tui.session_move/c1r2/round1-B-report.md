# C1 audit — ur (RTL) — pass B (c1-au-ur-B)
Auditor: i18n-auditor (c1-au-ur-B), model anthropic/claude-sonnet-5-5. Independent of translator (openai/gpt-6.1-sol). Read-only; nothing edited.

## Verdict: clean-with-findings — {blocker 0, major 0, minor 5, nit 3}

## Mechanical (python3, run here)
- 22 keys, same set and order as keys.json; no empties; none identical to English; no Cf/bidi control chars (RLM/LRM/FSI/PDI) — conforms to ur guide.
- ICU placeholders identical per key ({path},{key},{value},{description},{flag},{reason},{session},{to},{code}); no plurals in this slice.
- Backtick spans byte-identical (cannot_wait, deadline_exceeded); /approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, <id> <to> <seconds>, literal `local` all verbatim.
- Latin-only residue: AI, CLI, radient/openai/deepseek/anthropic/openrouter (ids), auto — all allowed (kept acronyms/ids/mode word; shipped uses "AI پرووائیڈر", "CLI").
- Width: only cap is list_title (<~30 cells): 15. All others uncapped/wrap.
- Source use sites checked: cli.py ~3028 (running_sessions_unchanged only after tool_approval_mode=auto), session_move.py 196-235 (deadline_exceeded = child killed after MOVE_TIMEOUT_S, outcome genuinely unknown).

## Safety claims
- running_sessions_unchanged: preserved — running sessions unchanged; config save does not reduce منظوری کی پابندی; /approvals auto must be typed in the session; new sessions start in auto. Matches shipped phrasing ("اسی سیشن میں لکھنا ہوگا — /approvals auto"). Only wording point below (m4).
- deadline_exceeded: does not say failed or succeeded; tells user to rerun the exact command to see how far it got. OK (m5 optional tightening toward "unknown").

## Findings
MINOR
- m1 `cli.config.list_desc_conversation_length`, `..._detail_length`, `..._max_learnings_history`, `..._classification_notice` (4 keys): tag «[متروک …]» / «[متروک]». The shipped Urdu catalogue renders the same four settings as «فرسودہ۔» (wire.settings.json: conversation_length/detail_length/max_learnings_history/classification_notice .help), so `lop config list` and the settings page name the same state with two different words. Suggest «[فرسودہ — اب کمپریشن اس کی جگہ لیتا ہے]» and «[فرسودہ]». (No glossary row for DEPRECATED, hence not MAJOR; the translation itself is idiomatic.)
- m2 `cli.config.list_desc_max_learnings_history`: «سیکھے ہوئے اسباق» vs shipped label wire.settings.max_learnings_history.label «سیکھے ہوئے نکات». Suggest «محفوظ رکھے جانے والے سیکھے ہوئے نکات کی زیادہ سے زیادہ تعداد».
- m3 `cli.config.list_desc_hosting`: «AI فراہم کرنے والا پلیٹ فارم» — shipped uses «پرووائیڈر» (wire.settings.hosting.label «ڈیفالٹ پرووائیڈر»; «AI پرووائیڈر») and «ہوسٹنگ پلیٹ فارم» (cli.global_help). Suggest «AI پرووائیڈر پلیٹ فارم (مثلاً radient, openai, deepseek, anthropic, openrouter)». Latin commas between ids are right (keep).
- m4 `cli.config.running_sessions_unchanged`: English "each session you want ungated". «جس سیشن میں … اس میں» is grammatical but drops the explicit "each" (the safety point); first clause is singular («چلتے سیشن میں») while shipped uses «چلتے سیشنز». Suggest: «چلتے سیشنز میں کوئی تبدیلی نہیں ہوئی — کنفیگریشن محفوظ کرنے سے ان کی منظوری کی پابندی کم نہیں ہوتی؛ ہر اس سیشن میں /approvals auto لکھیں جس میں ٹولز کی منظوری ہٹانا چاہتے ہوں۔ نئے سیشن auto موڈ میں شروع ہوتے ہیں۔»
- m5 `tui.session_move.deadline_exceeded`: «منتقلی … مکمل نہیں ہوئی» is a flat past-tense statement; the use site knows only that the CLI was killed at the deadline (outcome UNKNOWN, changed=True). It mirrors English "did not finish in time" and the follow-up («کہاں تک پہنچی») carries the uncertainty, so acceptable; tighter option: «مقررہ وقت تک منتقلی کا نتیجہ معلوم نہیں ہو سکا؛ `lop sessions move {session} --to {to}` دوبارہ چلانے سے پتا چلے گا کہ کہاں تک پہنچی». Optional.

NIT
- n1 `cli.config.cleared`: «اب اس کی ڈیفالٹ ویلیو دوبارہ استعمال ہوتی ہے» is slightly stiff for "reads as its default again". Suggest «{key} ہٹا دی گئی: اب یہ دوبارہ ڈیفالٹ پر ہے».
- n2 `cli.config.updated`: "Successfully" dropped; fine per tone (no exclamation) — noted only; keep.
- n3 `tui.session_move.cannot_wait`: «اسکرین» for "view" fine; «شیل سے چلائیں:» + backticked command is a correct LTR island. No change.

## RTL / bidi notes (no change requested)
- Strings starting with a Latin token or placeholder ({path}, {key}, CLI …, /move …, --to …) resolve to LTR base under FSI; Urdu runs still shape correctly. Documented best-effort TUI stance in the ur guide, not a translation defect.
- <پاتھ>/<سیشن>/<ڈیوائس|local> brackets mirror correctly in an RTL run; `local` stays literal; vertical bar kept. Parenthesised ids after Urdu «مثلاً» resolve RTL correctly. «۔» ends sentences; «؛» and «،» per guide (Urdu «،» after Latin flags in unsupported_flag is guide-conformant).

## Terminology consistency vs shipped ur catalogues
OK: کنفیگریشن فائل (cli.config_help), کنفیگریشن آپشنز, ڈیفالٹ, ویلیو, ہسٹری, کمپریشن (glossary), ذہین اشارے (shipped), منتقلی, ڈیوائس, ایگزٹ کوڈ, سیشن, منظوری, ٹولز; آپ register respected. Drift only m1–m3.

## Commands run
python3 loads of keys.json/ur.json (order, placeholder, backtick, Cf, identical, token checks); python3 scans of shipped ur catalogues (term lookups); read of ur.md and glossary §2; sed of session_move.py 196-235 and cli.py 3005-3032.

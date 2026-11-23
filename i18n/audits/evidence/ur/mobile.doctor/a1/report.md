# Independent i18n audit — ur / mobile.doctor — a1

**Verdict: passed** — 0 blocker / 0 major / 0 minor / 3 nit.

- Source (en) sha256 `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; target sha256 `842c776c34a6a31815d1918b27ad1b06d16542a8dc0ada0084a7ec229932d681` @ a3722be7dada (feat/mobile-doctor).
- Translator: i18n-translator subagent, session 398cdb26808f (values in commit a3722be7dada; English source drafted by coder session b0300e2c2c7f); anthropic/claude-opus-5-5; 2026-11-14.
- Auditor: independent i18n-auditor (C4 hi/ur lane, `task` subagent), session f9d4556f4a66; did not translate, draft or edit any value it judged; anthropic/claude-opus-5-5 (served model; no provider fallback observed); 2026-11-14.
- Scope: new namespace, all 23 keys judged.

## Findings

- **UR-D01** (nit) `mobile.doctor.tunnel.duplicates, mobile.doctor.tunnel.list_unchecked` — «{count} ٹنل» (invariant loan plural after a numeral) beside «ڈپلیکیٹ ٹنلز» in the sibling key. The style guide allows both; harmonising is optional. *Disposition:* accepted; no change requested.
- **UR-D02** (nit) `mobile.doctor.files.ok, mobile.doctor.files.near` — Same fixed plural as HI-D01: «{open} فائلیں» with a pre-formatted string count. Placeholder order kept, «(حد {limit})» preserves «of {limit}» (translator question 1 accepted). *Disposition:* accepted; no change requested.
- **UR-D03** (nit) `mobile.doctor.daemon.not_running` — «سلیپ موڈ میں» renders 'asleep' with an English loan (translator question 6). It is the living device term in Urdu software and matches the Hindi «स्लीप मोड में»; «سویا ہوا» would be the native alternative. *Disposition:* accepted; no change requested.

## Values judged

- `mobile.doctor.versions.ok` — EN: Local Operator {lop} on Python {python} ({platform}). — UR: Local Operator {lop}، Python {python} ({platform}) پر۔
- `mobile.doctor.daemon.not_installed` — EN: The phone service is not set up on this computer, so it is not running. — UR: فون رسائی سروس اس کمپیوٹر پر سیٹ اپ نہیں ہے، اس لیے یہ نہیں چل رہی۔
- `mobile.doctor.daemon.not_running` — EN: The phone service is set up but not running; the phone app shows this computer as asleep until it starts. — UR: فون رسائی سروس سیٹ اپ ہے لیکن نہیں چل رہی؛ جب تک یہ شروع نہ ہو، فون ایپ اس کمپیوٹر کو سلیپ موڈ میں دکھاتی ہے۔
- `mobile.doctor.daemon.ok` — EN: The phone service is running and answered in {ms} ms. — UR: فون رسائی سروس چل رہی ہے اور اس نے {ms} ملی سیکنڈ میں جواب دیا۔
- `mobile.doctor.daemon.slow` — EN: The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. — UR: فون رسائی سروس چل رہی ہے لیکن جواب دینے میں {ms} ملی سیکنڈ لگے، اس لیے فون ایپ اس کمپیوٹر کو جواب نہ دینے والا دکھا سکتی ہے۔
- `mobile.doctor.daemon.gate_open` — EN: The phone service answered without asking for its password: the phone portal is unprotected. — UR: فون رسائی سروس نے پاس ورڈ مانگے بغیر جواب دیا: فون پورٹل غیر محفوظ ہے۔
- `mobile.doctor.daemon.state_not_running` — EN: not running — UR: نہیں چل رہا
- `mobile.doctor.files.ok` — EN: The phone service has {open} of {limit} files open. — UR: فون رسائی سروس نے {open} فائلیں کھول رکھی ہیں (حد {limit})۔
- `mobile.doctor.files.near` — EN: The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. — UR: فون رسائی سروس نے {open} فائلیں کھول رکھی ہیں (حد {limit})، جو حد کے قریب ہے، اس لیے فون کے نئے کنکشن اور ڈیٹا بیس کھولنے کا عمل ناکام ہو سکتا ہے۔
- `mobile.doctor.files.unknown` — EN: The phone service's open files could not be counted on this computer. — UR: اس کمپیوٹر پر فون رسائی سروس کی کھلی فائلیں گنی نہیں جا سکیں۔
- `mobile.doctor.files.status_line` — EN: {open} open of {limit} — UR: {open} کھلی، حد {limit}
- `mobile.doctor.logs.clean` — EN: No out-of-files or database-open errors in the last day of logs. — UR: پچھلے ایک دن کے لاگز میں “بہت زیادہ کھلی فائلیں” یا ڈیٹا بیس کھولنے کی کوئی خرابی نہیں۔
- `mobile.doctor.logs.errors` — EN: The last day of logs shows out-of-files errors ({emfile}) and database-open errors ({sqlite}); the phone service may be running out of files. — UR: پچھلے ایک دن کے لاگز میں “بہت زیادہ کھلی فائلیں” کی خرابیاں ({emfile}) اور ڈیٹا بیس کھولنے کی خرابیاں ({sqlite}) ہیں؛ ہو سکتا ہے فون رسائی سروس کھلی فائلوں کی حد تک پہنچ رہی ہو۔
- `mobile.doctor.tunnel.not_configured` — EN: Phone access through Radient is not set up on this computer. — UR: اس کمپیوٹر پر Radient کے ذریعے فون رسائی سیٹ اپ نہیں ہے۔
- `mobile.doctor.tunnel.ok` — EN: The tunnel is set up and its connector is {state}. — UR: ٹنل سیٹ اپ ہے اور اس کا کنیکٹر {state} حالت میں ہے۔
- `mobile.doctor.tunnel.problem` — EN: The tunnel is set up but its connector is {state} ({reason}). — UR: ٹنل سیٹ اپ ہے لیکن اس کا کنیکٹر {state} حالت میں ہے ({reason})۔
- `mobile.doctor.tunnel.single` — EN: This Radient account has one tunnel for this computer. — UR: اس Radient اکاؤنٹ میں اس کمپیوٹر کے لیے ایک ٹنل ہے۔
- `mobile.doctor.tunnel.duplicates` — EN: This Radient account has {count} tunnels for this computer; the phone app may be connecting to an old one. — UR: اس Radient اکاؤنٹ میں اس کمپیوٹر کے لیے {count} ٹنل ہیں؛ ہو سکتا ہے فون ایپ کسی پرانے ٹنل سے جڑ رہی ہو۔
- `mobile.doctor.tunnel.list_unchecked` — EN: The account's tunnel list could not be read, so duplicate tunnels were not checked. — UR: اکاؤنٹ کی ٹنل فہرست پڑھی نہیں جا سکی، اس لیے ڈپلیکیٹ ٹنلز کی جانچ نہیں ہوئی۔
- `mobile.doctor.login.ok` — EN: Signed in to Radient. — UR: Radient میں سائن اِن ہیں۔
- `mobile.doctor.login.missing` — EN: Not signed in to Radient. — UR: Radient میں سائن اِن نہیں ہیں۔
- `mobile.doctor.files.not_running` — EN: The phone service is not running, so it has no open files to count. — UR: فون رسائی سروس نہیں چل رہی، اس لیے گننے کے لیے کوئی کھلی فائل نہیں۔
- `mobile.doctor.daemon.not_answering` — EN: The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. — UR: فون رسائی سروس چل رہی ہے لیکن جواب نہیں دے رہی؛ فون ایپ اس کمپیوٹر کو جواب نہ دینے والا دکھاتی ہے۔ ہو سکتا ہے یہ کھلی فائلوں کی حد تک پہنچ گئی ہو۔

## Verified

- Key parity with en (mobile.doctor 23/23, cli.mobile_help 17/17) and placeholder parity for every key; the only checker flag (cli.mobile_help.daemon_action) is an ICU select whose branch text my regex mis-read as placeholders — the carried value is correct.
- No command, flag or `lop` invocation named in any mobile.doctor value; placeholders {lop} {python} {platform} {ms} {open} {limit} {emfile} {sqlite} {state} {reason} {count} verbatim and in source order.
- Calm, factual tone throughout; gate_open keeps its warning meaning (portal unprotected) without alarm words.
- 'phone service' rendered consistently in every key as the phone-access service (hi «फ़ोन एक्सेस सेवा», ur «فون رسائی سروس»); glossary service → सेवा / سروس honoured; 'phone access' matches mobile.setup and cli.mobile_help.mobile.
- login.ok/login.missing are byte-identical to the already-audited wire.voice_availability.radient_signed_in/out in each locale.
- Real product path exercised: `LOP_LANG=<locale> .venv/bin/lop mobile doctor`, `lop mobile --help` and `lop mobile doctor --help` on this Linux node (exit 0) — six doctor checks plus both new help strings rendered in-locale; remaining keys rendered through the real resolver (tr('mobile.doctor') after set_active_locale) with sample arguments (audit.log).
- rich.cells.cell_len widths recorded per key; no en sidecar sets maxCells for these keys.
- Urdu ۔ ، ؛ used, no Hindi danda, no Latin comma/semicolon after Urdu text; no manual bidi controls (RLM/LRM/FSI/RLI/PDI) in any value.
- Bidi: the product's isolated_status_text wraps ur output in RLI…PDI (U+2067/U+2069, verified on login.ok), so Latin-first strings (versions.ok, login.ok/missing — first strong char L) still take an RTL base; the Latin islands (Local Operator, Python, Radient, {state}/{reason} machine words) stay byte-identical. The plain CLI print path does not isolate (TUI/CLI LTR best-effort stance per the ur guide) — not a translation defect.

## Not verified

- No native-speaker pass; this is an LLM independent audit.
- No rendered terminal/desktop/phone frame or RTL screenshot; Nastaliq/Devanagari glyph shaping not observed. Desktop Copy-diagnostics surface (local-operator-ui) not driven.
- Daemon-running, tunnel-configured, signed-in and log-error branches were rendered via the resolver with sample args, not by staging those machine states.

## Source-side observations

- Source-side (not a translation finding, for the coder): cli.py passes the English words 'unlimited' and '?' into mobile.doctor.files.status_line {limit}/{open}, while the context sidecar says {limit} may be 'unknown'; the gate line beside state_not_running prints untranslated 'closed'/'OPEN (this is a boundary failure)'.

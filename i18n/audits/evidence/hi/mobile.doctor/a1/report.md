# Independent i18n audit — hi / mobile.doctor — a1

**Verdict: passed** — 0 blocker / 0 major / 0 minor / 3 nit.

- Source (en) sha256 `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; target sha256 `56377a70658d5f432a88b63722b823e44fa561aa8c4b8944f3cb8a868d2bab69` @ a3722be7dada (feat/mobile-doctor).
- Translator: i18n-translator subagent, session 398cdb26808f (values in commit a3722be7dada; English source drafted by coder session b0300e2c2c7f); anthropic/claude-opus-5-5; 2026-11-14.
- Auditor: independent i18n-auditor (C4 hi/ur lane, `task` subagent), session f9d4556f4a66; did not translate, draft or edit any value it judged; anthropic/claude-opus-5-5 (served model; no provider fallback observed); 2026-11-14.
- Scope: new namespace, all 23 keys judged.

## Findings

- **HI-D01** (nit) `mobile.doctor.files.ok, mobile.doctor.files.near` — «{open} फ़ाइलें» is a fixed plural; an {open} of 1 would read «1 फ़ाइलें». {open} is a pre-formatted string argument, so an ICU plural is not available, and a running phone service always holds far more than one descriptor. *Disposition:* accepted, no change requested. Also answers translator question 1: keeping the English placeholder order and recasting «{open} of {limit}» as «{open} फ़ाइलें खोल रखी हैं (सीमा {limit})» preserves the meaning and reads naturally.
- **HI-D02** (nit) `mobile.doctor.versions.ok` — «Local Operator {lop}, Python {python} ({platform}) पर।» is verbless, mirroring the English fragment; «… पर चल रहा है।» would be a full sentence. *Disposition:* accepted as a terse diagnostic line; optional polish only.
- **HI-D03** (nit) `mobile.doctor.daemon.state_not_running` — Masculine «नहीं चल रहा» while the namespace's feminine subject सेवा takes «नहीं चल रही». The label is printed after the English «auth gate:» prefix (cli.py `lop mobile status`), where गेट is masculine, so the masculine default is the correct agreement for this slot. *Disposition:* adjudicated correct; no change.

## Values judged

- `mobile.doctor.versions.ok` — EN: Local Operator {lop} on Python {python} ({platform}). — HI: Local Operator {lop}, Python {python} ({platform}) पर।
- `mobile.doctor.daemon.not_installed` — EN: The phone service is not set up on this computer, so it is not running. — HI: फ़ोन एक्सेस सेवा इस कंप्यूटर पर सेट अप नहीं है, इसलिए यह नहीं चल रही।
- `mobile.doctor.daemon.not_running` — EN: The phone service is set up but not running; the phone app shows this computer as asleep until it starts. — HI: फ़ोन एक्सेस सेवा सेट अप है लेकिन नहीं चल रही; इसके शुरू होने तक फ़ोन ऐप इस कंप्यूटर को स्लीप मोड में दिखाता है।
- `mobile.doctor.daemon.ok` — EN: The phone service is running and answered in {ms} ms. — HI: फ़ोन एक्सेस सेवा चल रही है और उसने {ms} मिलीसेकंड में उत्तर दिया।
- `mobile.doctor.daemon.slow` — EN: The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. — HI: फ़ोन एक्सेस सेवा चल रही है लेकिन उत्तर देने में {ms} मिलीसेकंड लगे, इसलिए फ़ोन ऐप इस कंप्यूटर को उत्तर न देने वाला दिखा सकता है।
- `mobile.doctor.daemon.gate_open` — EN: The phone service answered without asking for its password: the phone portal is unprotected. — HI: फ़ोन एक्सेस सेवा ने पासवर्ड माँगे बिना उत्तर दिया: फ़ोन पोर्टल असुरक्षित है।
- `mobile.doctor.daemon.state_not_running` — EN: not running — HI: नहीं चल रहा
- `mobile.doctor.files.ok` — EN: The phone service has {open} of {limit} files open. — HI: फ़ोन एक्सेस सेवा ने {open} फ़ाइलें खोल रखी हैं (सीमा {limit})।
- `mobile.doctor.files.near` — EN: The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. — HI: फ़ोन एक्सेस सेवा ने {open} फ़ाइलें खोल रखी हैं (सीमा {limit}), जो सीमा के क़रीब है, इसलिए नए फ़ोन कनेक्शन और डेटाबेस खोलना विफल हो सकता है।
- `mobile.doctor.files.unknown` — EN: The phone service's open files could not be counted on this computer. — HI: इस कंप्यूटर पर फ़ोन एक्सेस सेवा की खुली फ़ाइलें गिनी नहीं जा सकीं।
- `mobile.doctor.files.status_line` — EN: {open} open of {limit} — HI: {open} खुली, सीमा {limit}
- `mobile.doctor.logs.clean` — EN: No out-of-files or database-open errors in the last day of logs. — HI: पिछले एक दिन के लॉग में “बहुत ज़्यादा खुली फ़ाइलें” या डेटाबेस खोलने की कोई त्रुटि नहीं है।
- `mobile.doctor.logs.errors` — EN: The last day of logs shows out-of-files errors ({emfile}) and database-open errors ({sqlite}); the phone service may be running out of files. — HI: पिछले एक दिन के लॉग में “बहुत ज़्यादा खुली फ़ाइलें” त्रुटियाँ ({emfile}) और डेटाबेस खोलने की त्रुटियाँ ({sqlite}) हैं; हो सकता है फ़ोन एक्सेस सेवा खुली फ़ाइलों की सीमा तक पहुँच रही हो।
- `mobile.doctor.tunnel.not_configured` — EN: Phone access through Radient is not set up on this computer. — HI: इस कंप्यूटर पर Radient के ज़रिए फ़ोन एक्सेस सेट अप नहीं है।
- `mobile.doctor.tunnel.ok` — EN: The tunnel is set up and its connector is {state}. — HI: टनल सेट अप है और उसका कनेक्टर {state} स्थिति में है।
- `mobile.doctor.tunnel.problem` — EN: The tunnel is set up but its connector is {state} ({reason}). — HI: टनल सेट अप है लेकिन उसका कनेक्टर {state} स्थिति में है ({reason})।
- `mobile.doctor.tunnel.single` — EN: This Radient account has one tunnel for this computer. — HI: इस Radient खाते में इस कंप्यूटर के लिए एक टनल है।
- `mobile.doctor.tunnel.duplicates` — EN: This Radient account has {count} tunnels for this computer; the phone app may be connecting to an old one. — HI: इस Radient खाते में इस कंप्यूटर के लिए {count} टनल हैं; हो सकता है फ़ोन ऐप किसी पुराने टनल से जुड़ रहा हो।
- `mobile.doctor.tunnel.list_unchecked` — EN: The account's tunnel list could not be read, so duplicate tunnels were not checked. — HI: खाते की टनल सूची पढ़ी नहीं जा सकी, इसलिए डुप्लिकेट टनल जाँचे नहीं गए।
- `mobile.doctor.login.ok` — EN: Signed in to Radient. — HI: Radient में साइन इन है।
- `mobile.doctor.login.missing` — EN: Not signed in to Radient. — HI: Radient में साइन इन नहीं है।
- `mobile.doctor.files.not_running` — EN: The phone service is not running, so it has no open files to count. — HI: फ़ोन एक्सेस सेवा नहीं चल रही, इसलिए गिनने के लिए कोई खुली फ़ाइल नहीं है।
- `mobile.doctor.daemon.not_answering` — EN: The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. — HI: फ़ोन एक्सेस सेवा चल रही है लेकिन उत्तर नहीं दे रही; फ़ोन ऐप इस कंप्यूटर को उत्तर न देने वाला दिखाता है। हो सकता है वह खुली फ़ाइलों की सीमा तक पहुँच गई हो।

## Verified

- Key parity with en (mobile.doctor 23/23, cli.mobile_help 17/17) and placeholder parity for every key; the only checker flag (cli.mobile_help.daemon_action) is an ICU select whose branch text my regex mis-read as placeholders — the carried value is correct.
- No command, flag or `lop` invocation named in any mobile.doctor value; placeholders {lop} {python} {platform} {ms} {open} {limit} {emfile} {sqlite} {state} {reason} {count} verbatim and in source order.
- Calm, factual tone throughout; gate_open keeps its warning meaning (portal unprotected) without alarm words.
- 'phone service' rendered consistently in every key as the phone-access service (hi «फ़ोन एक्सेस सेवा», ur «فون رسائی سروس»); glossary service → सेवा / سروس honoured; 'phone access' matches mobile.setup and cli.mobile_help.mobile.
- login.ok/login.missing are byte-identical to the already-audited wire.voice_availability.radient_signed_in/out in each locale.
- Real product path exercised: `LOP_LANG=<locale> .venv/bin/lop mobile doctor`, `lop mobile --help` and `lop mobile doctor --help` on this Linux node (exit 0) — six doctor checks plus both new help strings rendered in-locale; remaining keys rendered through the real resolver (tr('mobile.doctor') after set_active_locale) with sample arguments (audit.log).
- rich.cells.cell_len widths recorded per key; no en sidecar sets maxCells for these keys.
- Danda । closes every full sentence; ASCII comma/semicolon as the guide allows; curly quotes “ ”; nukta consistent (फ़ोन, फ़ाइल, ज़रिए, क़रीब, ज़्यादा); no ZWJ/ZWNJ; Latin digits.

## Not verified

- No native-speaker pass; this is an LLM independent audit.
- No rendered terminal/desktop/phone frame or RTL screenshot; Nastaliq/Devanagari glyph shaping not observed. Desktop Copy-diagnostics surface (local-operator-ui) not driven.
- Daemon-running, tunnel-configured, signed-in and log-error branches were rendered via the resolver with sample args, not by staging those machine states.

## Source-side observations

- Source-side (not a translation finding, for the coder): cli.py passes the English words 'unlimited' and '?' into mobile.doctor.files.status_line {limit}/{open}, while the context sidecar says {limit} may be 'unknown'; the gate line beside state_not_running prints untranslated 'closed'/'OPEN (this is a boundary failure)'.

# Independent i18n audit — ur / cli.mobile_help — a1

**Verdict: passed** — 0 blocker / 0 major / 0 minor / 1 nit.

- Source (en) sha256 `3e1aea77751a6a1286cd0bc525163ac6e90da2f81fd19fff1fc23ae247133267`; target sha256 `47628c43d322aa3da61aae5239eb428bc1106503c9d607e38045de616927f85a` @ a3722be7dada (feat/mobile-doctor).
- Translator: i18n-translator subagent, session 398cdb26808f (values in commit a3722be7dada; English source drafted by coder session b0300e2c2c7f); anthropic/claude-opus-5-5; 2026-11-14.
- Auditor: independent i18n-auditor (C4 hi/ur lane, `task` subagent), session f9d4556f4a66; did not translate, draft or edit any value it judged; anthropic/claude-opus-5-5 (served model; no provider fallback observed); 2026-11-14.
- Scope: 2 new keys (doctor, doctor_json) judged in full; the 15 carried keys (previously audited at source 88681bae…) re-read for consistency; this receipt covers the whole 17-key namespace at the new English source hash.

## Findings

- **UR-H01** (nit) `cli.mobile_help.doctor` — «صرف پڑھنے والی تشخیص» departs from the established predicate form «صرف پڑھنے کے لیے». In this attributive slot the established form would read «diagnosis for reading only»; «صرف پڑھنے والی» ('that only reads') carries the read-only meaning of a command that changes nothing better. «(شیئر کرنا محفوظ ہے)» keeps safe-to-share. *Disposition:* adjudicated justified; no change.

## Values judged

- `cli.mobile_help.doctor` — EN: Read-only diagnostics for phone access (safe to share) — UR: فون رسائی کی صرف پڑھنے والی تشخیص (شیئر کرنا محفوظ ہے)
- `cli.mobile_help.doctor_json` — EN: Print the diagnostics as JSON — UR: تشخیص JSON کی صورت میں دکھائیں

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

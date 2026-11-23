# Independent Hindi audit — cli.serve_readiness / sd1

- Date: 2026-11-16. Scope: all 11 keys and all plural branches, Hindi only.
- Auditor: **Audit daemon Hindi**, session `1ee2c984b1e5`, actual model `openai/gpt-6.1-sol` (host session-state; no fallback indicated).
- Translator per assignment: task/session `acf0a84cd992`, actual model `openai/gpt-6.1-sol`. This is a different session; I did not translate or edit any catalogue.
- Tree: `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca` plus uncommitted source/context/Hindi catalogue. Hashes bind audited bytes, not a claim about main.
- **Verdict: changes_requested — 0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT.**

## Finding HI-SD1-01 — MAJOR

**Key:** `cli.serve_readiness.cgroup_unknown`

**Current:** ऐसे cgroup में चलता है जिसका स्थान यह जाँच तय नहीं कर सकती

**Problem:** “Place” is rendered literally as “स्थान … तय” (determine/assign a location). The check is not locating a cgroup. It classifies the daemon's supervision context from its cgroup path. A path may already be known while this classification is unknown: `serve_supervision` returns `unit`, `other_unit`, `login_session`, or `unknown`. Current Hindi points the reader toward a location/path diagnostic rather than an unclassified supervision context. It is also a conspicuous English calque in otherwise natural technical Hindi.

**Suggested:** ऐसे cgroup में चलता है जिसकी श्रेणी इस जाँच से पता नहीं चल पा रही है

This keeps uncertainty, does not assert the process is unsupervised, and does not identify an unproven service. Read-only implementation context: `local_operator/network/readiness.py:2769–2787, 3050–3077`.

## 1. Mechanical audit

Read order: Hindi style guide/shared glossary, then English source/context sidecar, then Hindi. No MT, translation or back-translation service used.

Actual repository validator: `.venv/bin/python scripts/i18n/check.py --catalogues-only` — exit 0, `check.py: catalogue checks: 0 problem(s)`. Actual output is in `audit.log`.

Additional direct checks:

- All three JSON inputs parse with duplicate-member rejection; 11 English keys = 11 context keys = 11 Hindi keys. No added/dropped keys.
- Exact parsed AST argument names, types, skeletons, order and plural selectors match English. No placeholder rename, omission or kind change.
- `unknown`, `stale`, `other_unit` have complete Hindi CLDR `one`/`other` categories. Text renders at 0/1 use singular, 2/5/21 plural, as required. No select messages.
- All backtick spans match byte-for-byte, including `lop serve`, `lop services status`, `lop services restart`; plain `lop serve` and `cgroup` remain intact. No flags, paths, additional markup, Rich or HTML tags in source; tag extraction agrees.
- No Devanagari digits, injected bidi controls, ZWJ or ZWNJ.
- Every key rendered with the production ICU runtime; no parser/formatting error.

## 2–4. Every-key accuracy, idiom, register and terminology review

Keys below have the prefix `cli.serve_readiness.`. For plural keys both branches were reviewed, not merely the selected sample.

| Key | Assessment |
|---|---|
| `unproven` | Accepted. Record existence is separate from proof of process identity. The start-time comparison failure remains explicit; no claim that identity was verified. |
| `not_reporting` | Accepted. Heartbeat stopped and daemon not answering; does not assert it was killed or that PID identity was proved. |
| `not_this_daemon` | Accepted. Another process currently uses the recorded PID, not the recorded daemon. `दूसरी प्रक्रिया` preserves the distinction. |
| `unknown_state` | Accepted. An unknown recorded state, without invented diagnosis. |
| `install_unknown` | Accepted. Running build cannot be compared with the installed build. No claim of an actual build mismatch. |
| `cgroup_unknown` | **MAJOR HI-SD1-01**, above. |
| `recorded_gap` | Accepted. `दर्ज बिल्ड` labels recorded evidence; installed build is distinct; `प्रक्रिया की पहचान अप्रमाणित` prevents claiming that the current process runs the recorded build. |
| `unknown` | Both branches accepted. Installed-build currency and supervision remain unknown; `यह … रहा है`/`ये … रहे हैं` agree with singular/plural. No conversion of uncertainty into unsupervised/stale fact. |
| `stale` | Both branches accepted. Running versus installed build contrast preserved; hosted remote sessions run the other build(s). `इस/उस` versus `इन/उन … बिल्डों` preserves number. |
| `other_unit` | Both branches accepted. Another service rather than a lop serve unit, with `सेवा/सेवाओं` and singular/plural verb agreement. `यूनिट` matches glossary. Elliptical `के नहीं` is intelligible here. |
| `identity_restart` | Accepted. Polite `चलाएँ`; inspect records first, then guarded restart to record identities. Per-daemon check before restarting remains explicit. No unsafe kill instruction or promise that every recorded PID will be restarted. Literal commands intact. |

Glossary: `सत्र`, `सेवा`, `यूनिट`, and process-supervision `निगरानी` are consistent. `डेमन`, `बिल्ड`, `इंस्टॉल`, `हार्टबीट`, `टाइमस्टैम्प` are appropriate technical loans. No inappropriate address, gratuitous please, exclamation, or style-inconsistent full stop. These are fragments/source-unpunctuated diagnostics, so no extra terminal danda is required.

## 5. Fit

No key has a maxCells/maxChars hint. Text-only production ICU renders and Rich `cell_len` measurements are in `audit.log`; sample widths range 16–150 cells. Long diagnostics/remedy need wrapping at narrow terminal widths. Parameters were short samples, not maximum-length device/build/reason lists. No fixed button/tab budget or demonstrated truncation defect was found; none is invented as a finding.

Devanagari combining-mark/conjunct shaping is terminal-dependent; `cell_len` is a code measurement, not visual proof. No physical terminal/browser frame, full app run, full tests, install, live provider, daemon restart or service operation was performed. Layout/rendered-frame sign-off is not claimed.

## 6. RTL and independent-pass scope

Hindi is LTR; the Urdu-specific RTL pass is not applicable. Latin commands, PID tokens and identifiers remain intact; no bidi controls were added. This is **one** independent Hindi audit, not a second independently staffed safety/high-visibility pass. No prior pass was assumed correct. The requesting session retains responsibility for any additional required independent pass.

## Evidence classification and handoff

This report is `audit_notes_not_execution_log`: linguistic reasoning and summaries, not proof a command ran. `audit.log` is actual command execution output. `record.json` binds inputs, provenance and verdict; `receipt.json` hashes companions; `ledger-receipt.json` is the proposed ledger append object. Only an append dry run is executed here; the parent appends history. No translations, historical bundles, manifests or ledger history edited; no commit made. These new repo-only evidence files must be added/committed by the parent before being represented as tracked/merged evidence.

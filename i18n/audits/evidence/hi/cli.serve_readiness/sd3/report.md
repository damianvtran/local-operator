# Independent Hindi audit — cli.serve_readiness / sd3

**Date:** 2026-11-16. **Verdict: passed.** **0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.** No wording changes requested.

This is the FIRST audit by this auditor of the newly changed source/target contract: 7 updated + 4 new messages. It is not a second independently staffed pass, and the historical sd1/sd1r results are not reused as approval of changed source bytes.

Auditor: **Audit daemon Hindi**, session `1ee2c984b1e5`, actual model **openai/gpt-6.1-sol**, from current host session-state; no fallback indicated. The requesting session identifies translation job `170903662c6e` as the same original translator (`acf0a84cd992`, originally disclosed model `openai/gpt-6.1-sol`). Translator model attribution is from that manager-provided provenance, not an independent inspection of the translator runtime. I did not translate or edit these catalogues; no MT/back-translation service used.

## Inputs and retained coverage

Read current Hindi style guide/glossary, then English source/context, then Hindi and the pertinent readiness selection. Audited tree: `1fc11dab96954dcc4c31abcc695c696baf2ead8c` plus uncommitted remediation. Current byte hashes are bound in `record.json`, not a claim of main/merge inclusion.

The target catalogue at HEAD has the exact SHA-256 from sd1r. Comparison proves precisely 7 updated and 4 new keys. Four retained keys have **byte-identical raw JSON entry lines** at HEAD and in the working file: `install_unknown`, `cgroup_unknown`, `stale`, `other_unit`. Their prior semantic coverage is retained, and all four also receive fresh mechanical/render regression checks. The cgroup/N1 wording is outside this remediation scope as directed; it is neither reopened nor newly claimed resolved here.

## 1. Mechanical checks — all 15 keys

Actual repo validator `.venv/bin/python scripts/i18n/check.py --catalogues-only`: **0 problems, exit 0**. Additional direct checks also exit 0:

- Duplicate-rejecting valid JSON; exact **15 English = 15 context = 15 Hindi** key sets.
- Exact parsed argument names/types/skeletons and branch structure for every key; complete Hindi CLDR **one/other** on every plural.
- Byte-identical backtick commands and markup/tag extraction. `lop serve`, `lop services restart`, `cgroup`, identifiers and placeholder names preserved. PID/PIDs in the display label are localized as required by their context, not treated as machine output keys.
- All 15 messages rendered with production ICU formatting. Plural counts **0, 1, 2, 5, 21** exercised, including new PID/no-action/overflow keys. Hindi zero uses `one` as required by the guide.
- Opaque parameters containing ICU-like braces, a flag/path, Rich-like markup and Latin numbers survive verbatim exactly once in each applicable singular/plural render. No recursive formatting. These are formatter checks, not proof of a downstream Rich escaping boundary.
- No inserted bidi controls, Devanagari digits, ZWJ or ZWNJ. Hindi is LTR; Urdu RTL checks do not apply.

## 2–4. Complete key assessment

Every suffix below belongs to `cli.serve_readiness.`. Changed/new keys received fresh semantic/idiom review; retained rows preserve prior coverage with current-byte regression checks.

| Key | Scope and result |
|---|---|
| `unproven` | Updated; accepted. `रिकॉर्ड मौजूद है` describes a record, not a live-process observation. Identity remains unproven and inability to compare start-time stamps remains explicit. |
| `not_reporting` | Updated; accepted. No answer and stopped heartbeat, without asserting the process was killed or its identity verified. The absence of a responder is expressed without a daemon-gender-dependent verb. |
| `not_this_daemon` | Updated; accepted. Another process holds the recorded PID; no attribution of the recorded build to that process. |
| `unknown_state` | Updated; accepted. Compact record-state diagnosis, not a claim about live process state. |
| `install_unknown` | Retained; accepted previous coverage, current mechanical/render regression clean. Cannot compare builds, not an actual mismatch claim. |
| `cgroup_unknown` | Retained, unchanged; prior coverage retained. Deferred N1 not reopened. |
| `recorded_gap` | Updated; accepted. Explicit `रिकॉर्ड में … दर्ज है` versus installed build. No assertion the live PID runs the recorded build and no duplicated identity-uncertainty parenthetical. |
| `unknown` | Updated; both branches accepted. Em-dash group separation avoids a colon scaffold. Current-build and supervision uncertainty remain explicit; singular/plural agree with total daemon count. |
| `stale` | Retained; both branches mechanically/render-regression checked. Prior current-versus-installed and hosted-session semantics retained. |
| `other_unit` | Retained; both branches mechanically/render-regression checked. Another service rather than a lop serve unit; prior terminology coverage retained. |
| `identity_restart` | Updated; accepted. Purpose is moving the recorded daemon to the installed build, not recording/proving identity. Per-daemon checks before action remain explicit. Literal command intact. |
| `pid_reason` | New; both branches accepted. `प्रक्रिया ID` / `प्रक्रियाओं के ID` localize the singular/plural label around an unchanged numeric list. Diagnosis remains attached to exactly that group. |
| `and_more` | New; accepted. `और # अन्य` preserves the omitted-item count and Latin-digit formatting, with complete Hindi plural branches. |
| `identity_no_action` | New; both branches accepted. The record's build matches the install; the process identity is explicitly unproven. No action is needed only for **this non-required check**, not an assertion of operational health or proof about a live process. |
| `identity_restart_manual` | New; accepted. Manual restart of literal `lop serve`, to reach installed build; inability to reload itself in the existing process remains explicit. No promise of subsequent identity proof. |

Polite imperatives `चलाएँ`/`करें`, ordinary technical Hindi loans, `सत्र`, `सेवा`, `यूनिट`, and supervision `निगरानी` remain consistent with the guide/glossary. No extra please, honorific, exclamation or incorrect full stop was introduced. Record-centric diagnosis fragments are distributive under the shared PID heading; they do not identify the daemon as the holder of an unverified PID.

## Composition and safety-selection review

A text-only sample composes a two-PID group sharing one unproven diagnosis plus one recorded-build gap, then appends the keyed overflow count inside the outer unknown message. The diagnosis and gap each occur once; the generated scaffold contains no colon, so there is no double-colon defect. The remaining uncertainty is preserved.

Read-only source selection inspected at `local_operator/network/readiness.py:3012–3059, 3139–3218`:

- Current-build equality is checked against an **unproven record**; it increments the no-action count without changing identity to proven.
- An unproven recorded build gap selects automatic restart only when all relevant records are reloadable; otherwise it selects the manual message.
- A proven stale neighbor already provides the one movable/manual restart step; the new unproven-gap message does not duplicate it.
- No-action is scoped to a non-required check, and neither automatic nor manual restart copy promises identity proof.

This establishes how the reviewed wording is selected by inspection, **not** execution of readiness/service code. No service or process was restarted.

## 5. Fit

No maxCells/maxChars hints are supplied. Actual per-key sample code-point/Rich-cell measurements are in `audit.log`; long explanatory lines require wrapping, as expected for terminal diagnostics, rather than a button/tab budget. New no-action text measures 107 Rich cells in either branch, and manual restart 119 cells with the short sample device. Variable parameter maxima were not claimed tested. Devanagari shaping and conjunct width remain terminal-dependent; no physical rendered-frame/layout sign-off is claimed.

## Evidence and boundaries

`report.md` is **audit_notes_not_execution_log**. `audit.log` contains actual lightweight validator/render/parity/opaque checks and append dry-run output. `record.json` binds current inputs, actual auditor provenance, prior coverage and verdict; `receipt.json` hashes companions; `ledger-receipt.json` is append-ready, dry-run only.

No translations/source/history edits, commits, pytest, heavy tests, install, service action, live provider or new tree. Existing evidence bundles/history are preserved. Parent appends history and adds/commits this new repo-only evidence. Any separately required second independent safety audit remains separate.

# French audit — cli.serve_readiness / sd3

**Verdict: passed. No BLOCKER, MAJOR, MINOR or NIT findings in the scoped revised copy.**

## Identity and scope

- Date: 2026-11-16 UTC.
- Independent auditor: Audit daemon French (i18n-auditor), session `19f657b90086`; actual model `openai/gpt-6.1-sol`, from the runtime model declaration; no fallback observed or reported.
- Translator: job `170903662c6e`, the same original translator according to the request (original task `acf0a84cd992`, actual model `openai/gpt-6.1-sol`). The current translator job identity is supplied by the requesting session. Auditor did not author/edit the translations.
- Base: `1fc11dab96954dcc4c31abcc695c696baf2ead8c` plus uncommitted remediation. This is the **first independent pass on the revised 15-key source/target**, not a second independent approval.
- Semantics audited for all **7 changed + 4 new keys**; retained earlier coverage for **4 byte-identical keys**. Mechanical/parser/render checks cover all 15 keys.
- Read French guide/glossary, current English/context/French, evidence README, docs/I18N.md audit/ledger contract and readiness.py selection/composition lines 3007–3057, 3144–3163, 3181–3217. cgroup wording N1 is explicitly deferred: not reopened or edited.

## Mechanical result

- `.venv/bin/python scripts/i18n/check.py --catalogues-only`: exit 0, `0 problem(s)`.
- Duplicate-rejecting JSON loads; exactly 15 matching English/French/context keys. All argument names, kinds, skeletons and branch-level opaque/code/tag contracts match.
- All **6 plural messages** contain French `one/many/other`; rendered counts 0, 1, 2, 1,000,000 and 1,500,000. The `and_more` number is rendered by ICU, including U+202F grouping; no authored localized digit sequence.
- Backtick commands preserved exactly. PID/PIDs is deliberately localized prose per the new sidecar; numeric comma-separated PID strings and all device/build/service/diagnosis values remain byte-identical after interpolation.
- Curly apostrophes, spaced em dashes and catalogue-authored U+202F before colons/semicolons are correct. No bidi controls or RTL requirements apply to French.
- `audit.log` is actual captured command output; `verify.py.txt` is the exact executed pure-parser code. This report is **audit_notes_not_execution_log**.

## Changed/new key assessment — all 11

| Key suffix | Independent assessment |
|---|---|
| `unproven` | Neutral diagnostic fragment. Record presence is separate from unproved process identity. Start-time comparison limitation retained; does not identify the live holder. |
| `not_reporting` | Record presence, no response, stopped signal of life; no claim of confirmed death. Native French, without the previously removed heartbeat borrowing. |
| `not_this_daemon` | Explicit reused-PID distinction, not a claim that the recorded daemon still owns the PID. |
| `unknown_state` | Compact, neutral unknown recorded state. |
| `recorded_gap` | `l’enregistrement indique le build {recorded} ; le build {install} est installé` attributes only the recorded version to the record. No assertion about the live process. No duplicate identity parenthetical. |
| `unknown` | Em-dash composition avoids device-colon/PID-colon double punctuation; unknown current-build/supervision conclusion remains explicit and agrees in each branch. |
| `identity_restart` | `lop services restart` is literal; instruction is to move the recorded daemon onto the installed build, not to prove identity. Checks before action retained. No unconditional new permission is added. |
| `pid_reason` | `identifiant(s) de processus {pids} — {reason}` is idiomatic explicit French labeling. Shared neutral diagnosis is rendered once for the listed numeric PIDs. All categories agree; no added colon or modification of opaque PID strings. |
| `and_more` | `et # de plus` is natural elliptical overflow copy; all categories and formatter-owned number preserved. |
| `identity_no_action` | The **record**, not the live process, matches the installed build. `dont l’identité du processus n’est pas prouvée` explicitly retains uncertainty. No action is scoped to this **non-obligatory check**, not a blanket health claim. Singular/plural is correct. |
| `identity_restart_manual` | Explicit manual `lop serve` restart to the installed build; inability to self-reload while running preserved. Does not promise identity proof or automatic reload. |

Tone is factual/non-alarming, instructional vous forms are consistent, and service/unit/session terminology and literal command names follow the glossary. There is no incorrect negation, destructive-action addition or identity guarantee.

## Source-selection cross-check (static only)

The inspected source only increments `unproven_current` for identity `unproven`, with an install stamp, and record/install equality. The no-action message uses that count while the diagnostic remains unknown. It is not proof about a live process. Only a record/install gap sets `unproven_gap`; automatic restart copy is selected only when all such records are reloadable, otherwise manual copy. A proven stale neighbor already supplies the restart step, suppressing a duplicate. These statements are based on **reading the source**, not execution of remedy branches or services.

Actual ICU composition was exercised with the new unproven fragment plus recorded-gap clause, a grouped two-PID label and overflow counts 1/2/1,000,000. The diagnosis appears once; identity uncertainty appears once; no double-colon separator is introduced. Source-owned `"; "` joins are outside the catalogue punctuation contract; no source edit was made.

## Retained coverage proof — four unchanged keys

`install_unknown`, `cgroup_unknown`, `stale`, `other_unit` retain earlier native/semantic assessment. The committed baseline French catalogue SHA-256 matches the target hash in `i18n/audits/evidence/fr/cli.serve_readiness/sd1r/record.json`. Each of these four current raw UTF-8 JSON lines is byte-identical to its baseline line, including value and punctuation. Exact line hashes and the 7/4/4 delta classification are captured in the verification output and record. No cgroup N1 scope expansion.

## Fit and limitations

No current sidecar supplies maxCells/maxChars. Runtime synthetic widths are logged per branch. These are wrapping diagnostic prose rather than fixed-size buttons/tabs; no demonstrated budget violation. Expanded PID labels are longer than English but unambiguous, with no wide-glyph or combining-mark width dependency. Dynamic input can lengthen output; no live terminal layout/screenshot or platform shaping certification is claimed.

No translation/MT service, pytest, services, provider calls, installs, translation/source edits, history append or commit. Existing audit bundles and ledger history are preserved. Append-ready receipt is dry-run only; parent owns append and commit. New evidence is not claimed to be on main.

## Current input hashes

- English: `c94e77cdf5d2ce8c0d92e333e9802a885675b074d1d5dc6710281dc3a5306b10`
- Context: `40d35aa2e026aa02a96376d4c550023446af21b89456ad0c80cd9540e8a63428`
- French: `04dd067c9bca413031a8ee2012b299f2979035c1701a827a354a428cbe3ede57`
- readiness.py: `3adbd0122dc704320bd77991299071739057da36d34e39bec7d6591821d5968b`

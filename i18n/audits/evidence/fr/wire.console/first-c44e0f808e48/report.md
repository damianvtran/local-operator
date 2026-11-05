# French first-reader audit: wire.console

Date: 2026-11-03. Status: **passed** for the staged linguistic/mechanical catalogue contribution only. Findings: BLOCKER 0, MAJOR 0, MINOR 0, NIT 0. Terminal review of 8 keys; no translation changes requested.

Auditor: i18n-auditor / c44e0f808e48 / **openai/gpt-6.1-sol**, derived from own completed usage in served-model-evidence.json. Independent of translator 7b2ba4ed501d and coder eae0961e4e78; not job196c3a94b346. No other new reader report inspected. Known earlier fallback disclosed in record; no requested-header or guessed no-fallback provenance.

## Inputs

English and context read from immutable core 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5; target read from original translator packet /Users/damian/.local-operator/sessions/7b2ba4ed501d/scratchpad/fr-console-0e7cde33. Each exact hash is in record.json and receipt.json. Original manifest verified completely after this audit; packet untouched.

The translator style snapshot differs from the pinned style guide only by omission of an unrelated relay.chrome width-exception paragraph. Independently compared the diff; all normative rules relevant to these namespaces are identical. Both style snapshots are retained and hashed. No exemption from those other namespaces was imported here.

## Native assessment — audit_notes_not_execution_log

- `wire.console.title`: Capacité de la console is natural concise section copy.
- `wire.console.description`: Admission des terminaux actifs is intelligible technical French in this capacity context. Completed-terminal history consumes no process slots; only future starts are affected, and running work is never terminated.
- `wire.console.max_live.label`: Plafond de terminaux actifs retains the live terminal ceiling.
- `wire.console.max_live.help`: 32–256 and default 128 preserved. Managed memory, terminal cells and native resources can reduce admission. Explicitly does NOT impose a memory limit on child processes; no arbitrary-RSS promise.
- `wire.console.max_owner.label`: Terminaux d’agents par conversation retains agent-terminal/conversation ownership scope.
- `wire.console.max_owner.help`: 1–32 and default 8 preserved. Conversations share the global live quota; completed terminals do not count.
- `wire.console.user_reserved.label`: Emplacements réservés aux terminaux utilisateur correctly identifies the reserved user capacity, not agent capacity.
- `wire.console.user_reserved.help`: 4–32, default 16, and cap of one quarter of the live ceiling preserved. Agent starts cannot use these slots; existing work is never stopped.

## Executed mechanical evidence

Existing core check_catalogues() executed on isolated scratch en/fr fixtures containing only this namespace. Final command exited 0 within tool timeout 90 seconds and 256 MiB process limit. audit.log is actual stdout/stderr, not these notes.

- Duplicate-aware JSON; exact source/target/context key parity; ICU argument names/kinds/order preserved; zero plural/select forms in these inputs, hence CLDR branches not applicable.
- Negative controls for an unexpected argument, missing key and invalid ICU all rejected. Original target restored in fixture and passing control rechecked.
- Opaque SDK parameters containing apostrophe, braces and hash render literally rather than being parsed as translated ICU. Technical identifiers, mode literals and API snippets manually reviewed plus scoped exact token checks. Ordinary prose words are translated, not incorrectly frozen as technical tokens.
- Unicode Cf=0 for every target; no inserted bidi/ANSI/NUL controls. French thin nonbreaking spaces before prose colon/semicolon follow the guide; code syntax stays unchanged.
- Actual Rich cell measurements retained in widths.json. Hub help values are 72/72 and 70/72. Console and SDK sidecars declare no maxCells/maxChars constraints, so no numerical native-fit guarantee is inferred. Console bounds/defaults checked unchanged against English.

Preflight logs, when present, preserve actual initial rig failures: a style snapshot discrepancy (resolved by inspecting the narrow diff and retaining both), and SDK false-positive token checks that mistook ordinary prose/substrings for machine tokens. Those are audit-rig corrections, not defects or edits in the target. Successful final evidence is audit.log.

## Remaining proof and scope

Pseudo: NOT RUN. Native RTL rendering: NOT RUN; French LTR, source Cf check is not rendered proof. Native/app fit, real-path calls and design/UX: NOT RUN. Second independent reader: separate coordinator gate, not claimed by this report. No native/app launch, install, build, broad scanner/full suite, activation/eligibility, PR/comment, source edit, generation or ledger write. Catalogue linguistic approval does not prove interface-wide English-free operation or release eligibility.

Only own scratch written; prior policy and translation bundles untouched. This original report/record/receipt is immutable handoff evidence, not a claim of tracked-main retention. Coordinator must retain it in the core canonical evidence process.

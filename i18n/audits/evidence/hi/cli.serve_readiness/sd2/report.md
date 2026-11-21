# Second independent i18n audit — hi / cli.serve_readiness / sd2

Verdict: **PASSED — 11/11 keys audited; zero unresolved findings.**

## Provenance and independence

Auditor: Local Operator, second independent i18n auditor; session `85af124f4a9f`; actual model `openai/gpt-6.1-sol` (host session context). No fallback observed/reported. Not translator `acf0a84cd992`; no translation contribution authored or changed. The independent all-key semantic review and this reproducer’s initial verdict were authored before any first-pass findings were received. A parent message summarizing first-pass findings arrived after the reproducer was written, before it ran; no first-pass report files were opened. See `i18n/audits/evidence/fr/cli.serve_readiness/sd2/comparison-note.md` for the disclosure and separate disposition. This initial second-pass verdict does not settle or override unresolved first-pass findings. This is an independent LLM language audit, not a native-speaker claim or own-copy approval.

Audit time (UTC): 2026-11-16T04:47:17.143000+00:00. Checkout HEAD: `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca`; working-tree bytes, not HEAD catalogue contents, are authoritative for this new namespace.

## Byte binding

- English source SHA256: `6c6de62a68a037ac88ba7a63f75f8335845acc61549f78ca98599ac25d29821e`
- English context SHA256: `e8c5895f98c71f39385a3defe974919ba776a9eda4bf195ac687a6db9b0db693`
- Target SHA256: `44980d359b70259062ec915d495235869e62996247beaa7e851e05b8a65ab439`

## Locale result and register

Respectful Hindi चलाएँ matches the आप register. दर्ज and अप्रमाणित preserve conservative identity wording; बिल्ड and इंस्टॉल किया गया बिल्ड distinguish record and installed build without asserting the record is live. सत्र is the required session term; निगरानी and यूनिट/सेवा follow glossary calls. one at 0/1 uses यह/चल रहा है; other uses ये/चल रहे हैं and plural bilḍ forms. The restart clause is passive and explicitly puts every-daemon checking before action.

## All-key semantic assessment

- **cli.serve_readiness.unproven — PASS:** Recorded presence only; process identity cannot be proved because no start-time comparison was possible. No assertion of a live recorded build.
- **cli.serve_readiness.not_reporting — PASS:** Record remains present; lack of response/ceased heartbeat is not promoted to death or a proved identity.
- **cli.serve_readiness.not_this_daemon — PASS:** PID now belongs to another process; never instructs killing or restarting that unrelated process.
- **cli.serve_readiness.unknown_state — PASS:** Recorded state remains unknown; no new diagnosis added.
- **cli.serve_readiness.install_unknown — PASS:** Build/install comparison is unavailable; does not falsely label a daemon stale.
- **cli.serve_readiness.cgroup_unknown — PASS:** This check cannot place the cgroup; does not claim no cgroup or unsupervised execution.
- **cli.serve_readiness.recorded_gap — PASS:** Explicitly recorded build {recorded}, installed build {install}, identity unproven. This is not the stale/live-process assertion.
- **cli.serve_readiness.unknown — PASS:** The conjunction of current-build and supervision status is unknown. {reasons} remains diagnostic text; all count-selected branches checked.
- **cli.serve_readiness.stale — PASS:** Only this confirmed-stale wording says running build differs from installed build, and hosted remote sessions use that other build. PID/build/install parameters preserved in every branch.
- **cli.serve_readiness.other_unit — PASS:** Held by another service, not a lop serve unit; does not imply unmanaged or absent service. Holders preserved in every branch.
- **cli.serve_readiness.identity_restart — PASS:** Inspect records using literal `lop services status`, then literal `lop services restart` to record identities. Every-daemon checks precede restart action; no unconditional restart or safety guarantee added.

## Validation and evidence limits

`audit.log` contains actual validator output; `rendered.txt` contains all 11 production-runtime rendered catalogue values, with 47 count/branch samples. The matrix exercises 0/1/2/3/4/5/11/21/22/25/1.5/1,000,000/1,500,000, covering every declared CLDR category. Pseudo/opaque diagnostic parameters survive unchanged. Commands and arguments are checked in every plural arm, not only the union of message arguments.

The shared reproducer and frozen independent verdict are under `i18n/audits/evidence/fr/cli.serve_readiness/sd2/`. No context maxCells is declared, so no arbitrary width cap was invented. No full application, service restart, process-identity detector, screenshot, terminal RTL shaping/layout, whole-repository validator, shipped eligibility, or native-speaker approval is claimed. Urdu is logical-order raw text only; catalogue controls are forbidden and none were found. No translation, generated file, ledger/history, commit, install or pytest modification/action occurred. First auditors retain ledger-receipt responsibility.

Findings: **none**. No corrective translation or unconditional-restart recommendation is supplied.

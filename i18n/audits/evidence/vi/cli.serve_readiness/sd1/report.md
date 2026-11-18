# Independent Vietnamese audit — cli.serve_readiness / sd1

Date: 2026-11-16 (UTC). Auditor: **Audit daemon Vietnamese**, session
`77442ce206e1`, actual model **openai/gpt-6.1-sol** (host-declared; no fallback
incident observed). Translator task: `acf0a84cd992`, actual model
`openai/gpt-6.1-sol`, supplied by requester. Different session/agent; using the
same model does not mean the same translator run. No translation/MT service used.

**Verdict: passed — all 11 keys audited, zero BLOCKER/MAJOR, one MINOR, zero NIT.**
This is one assigned independent pass. It does not assert completion of a separate
second independent high-visibility pass or a rendered-UI/design gate.

## Scope and reading order

Read `guide://i18n`, `i18n/style-guides/vi.md`, `i18n/glossary.md`, then the complete
English catalogue and context sidecar, then all Vietnamese values. Read
`docs/I18N.md`, RFC §§2.2/6, the evidence README and the relevant readiness producer
for composition/identity context. Checkout HEAD at the start:
`e4a6b6e1d01ef269604fe6d005e6bce1453b66ca`, with the requester's uncommitted copy.
This is a current-byte audit of that copy, not a claim of merged/main provenance.
Exact source, sidecar and target hashes are in `record.json`.

## Mechanical checks and execution evidence

`audit.log` contains **actual captured command stdout/stderr and exit codes**, not
audit notes. It includes the repository catalogue validator, additional parser
checks for every key, CLI append help, history hash and the ledger dry-run.
`check.py --catalogues-only` returned **0 problems, exit 0**. The per-key parser
check independently verified strict JSON/duplicate keys, 11/11 key+context parity,
placeholder names/kinds/order, parseable ICU, every required Vietnamese CLDR
category, code spans/markup, NFC, no combining-mark sequences or bidi controls.
Three plural messages use exactly `other`; counts 0, 1, 2, 5 and 101 were exercised.
Vietnamese has no obligatory singular/plural agreement. Removing English `one`
branches is correct. All literal commands remain byte-identical. There are no
flags, markup tags or hand-written numerical formatting in these messages.
The plain `lop serve` unit label is also preserved. PID capitalization in prose
is acronym styling, not a modified machine identifier.

`report.md` is **audit_notes_not_execution_log**. It does not prove execution by
itself. The receipt is validated only by a **dry-run**; this auditor did not append
history, commit, edit translations, run full tests, install dependencies, create a
new tree, call live providers or execute any restart/remediation command.

## Findings

### VI-SD1-01 — MINOR — cli.serve_readiness.cgroup_unknown

- **Current:** `chạy trong cgroup mà lần kiểm tra này không thể xác định vị trí`
- **Problem:** “xác định vị trí” is understandable but can suggest locating a
  directory/address rather than classifying the cgroup's supervision context.
  The uncertainty and conservative diagnostic survive, so this is a non-blocking
  clarity improvement, not a wrong/unsafe claim.
- **Suggested:** `chạy trong cgroup mà lần kiểm tra này chưa phân loại được`
- **Disposition:** suggested polish only; no catalogue edit made.

## Complete semantic/style/terminology coverage

All keys below share prefix `cli.serve_readiness.`. Every row was read in full.

| Key | Assessment |
|---|---|
| `unproven` | Correctly limits the claim to a recorded process; lack of a comparable start-time stamp cannot prove identity. “danh tính tiến trình” and “đối chiếu” are natural technical Vietnamese. No proof is implied. |
| `not_reporting` | Preserves “recorded but not answering” and stopped heartbeat; does not invent death or loss of identity. `heartbeat` is normal developer register here, not a glossary conflict. |
| `not_this_daemon` | Correctly states PID reuse by another process, without directing an unguarded kill. “PID đó hiện thuộc về tiến trình khác” is idiomatic. |
| `unknown_state` | Correctly reports an unknown recorded state, not an unknown currently running build. Natural concise status fragment. |
| `install_unknown` | Correctly preserves inability to compare the running build with the installation; does not assert they differ. “bản dựng”/“bản cài đặt” distinguish the two inputs consistently. |
| `cgroup_unknown` | Preserves the unknown cgroup classification; optional clarity polish VI-SD1-01 above. |
| `recorded_gap` | Both exact placeholders retained. “bản dựng được ghi nhận” plus “chưa xác minh danh tính tiến trình” does not attribute the recorded build to the live PID. This is the critical conservative-identity invariant. |
| `unknown` | Preserves device/reasons/count structure. “khớp với bản cài đặt” makes “current” concretely accurate in this comparison. “được giám sát” follows the glossary; explicit daemon/device context prevents confusion with a monitoring feature. No singular/plural classifier is required in Vietnamese. |
| `stale` | Preserves all PID/build/install/device placeholders and the consequence for hosted remote sessions. “bản dựng khác đó” can refer collectively to the listed builds in Vietnamese; it does not force English-style singular agreement. No omission or added update/remedy claim. |
| `other_unit` | Correctly distinguishes another holding service from a `lop serve` unit. “dịch vụ” and kept `unit` follow the glossary. The language can cover one or several services without obligatory plural morphology. |
| `identity_restart` | Inspection precedes restart. Both commands remain intact. The final clause explicitly preserves per-daemon checking before action and does not promise an unguarded restart or successful identity recovery. Direct factual register, no added “Vui lòng” or false permission. |

No register/formality mismatch, omission, addition, negation error, command change
or glossary violation found. Sentence case, Vietnamese diacritics, Latin technical
tokens and punctuation follow the guide. `daemon`/`cgroup` are reasonable Linux
operator terms here; the guide does not mandate an invented Vietnamese equivalent.

## Fit and RTL

No `maxCells` or `maxChars` hints exist for these 11 context entries. These are
prose diagnostic fragments/full details, not buttons or fixed-width tabs. Static
samples span 41–226 cells with illustrative inserted values; longer device,
PID/build and reason values can expand them. NFC precomposed Vietnamese gave the
same character/cell count in the measured strings. There are no wide CJK glyphs,
combining-mark sequences, arrows or raw bidi controls. No definite translation-
specific fit failure is established; live wrapping/truncation was not tested and
no screenshot/terminal rendering proof is claimed. Vietnamese is LTR: Urdu
sentence-order, bidi-island and mirroring requirements are not applicable.

## Receipt handoff

`ledger-receipt.json` is the single row object accepted by `ledger.py append`.
`receipt.json` is the bundle-integrity companion, not a ledger row. The ledger
note links `record.json` and is within 280 characters. The parent owns history
append and must re-query `history-sha` if history changes. All bundle paths are
under the canonical repo-only root. These newly written files are uncommitted;
no claim that they are already tracked on main is made.

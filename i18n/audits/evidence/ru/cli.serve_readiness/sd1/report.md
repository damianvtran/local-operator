# Independent Russian audit — cli.serve_readiness / sd1

- Audit date: 2026-11-16.
- Auditor: independent-ru-auditor; session `b8c027b19533`; actual model `openai/gpt-6.1-sol`, as declared by this session runtime; no model fallback observed.
- Translator: task/session `acf0a84cd992`, actual model `openai/gpt-6.1-sol` (manager-provided identity). Different sessions; auditor did not translate or modify the catalogue.
- Scope: all 11 keys, English context and Russian catalogue in the assigned uncommitted working copy over `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca`. Exact file-byte hashes are in record.json. This bundle is new evidence, not a historical recreation or a claim of presence on main.
- Verdict: **changes_requested**. **0 BLOCKER, 2 MAJOR, 1 MINOR, 0 NIT**. Unresolved blocker/major count: **2**.

## Method and boundaries

Read the locale style guide and glossary first, then English and its complete context sidecar, then every Russian key. Read the repo HOWTO/RFC, evidence contract and the relevant readiness callsites to distinguish heartbeat lapse, build-record gaps, and unknown cgroup classification. No translation/MT service, installs, new trees, full tests, pytest rigs or live providers were used. No translation edits or commits were made.

The repo catalogue validator passed (`check.py --catalogues-only`: zero problems). A separate pure-parser check verified strict JSON/duplicate detection; 11/11 source/target/context keys; placeholder names, kinds and order; all four Russian plural categories; preserved code spans and tag parity; and rendering at 0, 1, 2, 5, 11, 21, 22, 25, 101 and 1.5. Its retained reproduction and actual output are mechanical.py.txt and audit.log. This independent explicit category check matters because the repo's normal category check need not require every category for an ordinary plural.

The first system-python help invocation failed because that interpreter lacked tomllib. Recovery used the already-existing worktree .venv interpreter; no installation occurred. This sentence is an audit note about the initial failure, not retained execution output. audit.log contains only the subsequent commands actually captured there.

This is one independent audit pass, not two independent auditors or a second-pass approval. The manager must arrange a separate independent pass where the safety/high-visibility gate requires it. No rendered application frame, terminal wrap/truncation test, or release eligibility approval is claimed.

## Findings

### RU-SD1-01 — MAJOR — cli.serve_readiness.not_reporting

**Current:** «зарегистрирован, но не отвечает (его пульс перестал поступать)»

**Problem:** «пульс перестал поступать» is an unnatural literal heartbeat metaphor. In Russian, a pulse does not “arrive”; for this non-alarming technical diagnostic, describe missing activity signals rather than an organism's pulse. The intended meaning is a lapsed heartbeat, not proof of death, identity mismatch or a terminated process.

**Suggested:** «зарегистрирован, но не отвечает (сигналы активности больше не поступают)»

### RU-SD1-02 — MAJOR — cli.serve_readiness.cgroup_unknown

**Current:** «работает в cgroup, положение которой эта проверка не может определить»

**Problem:** «положение cgroup» is a spatial/status calque and does not read as a native technical diagnostic. The callsite is failing to classify the cgroup's supervision context, not measuring a group's position. Keep this unknown; do not turn it into an assertion that the daemon is unsupervised.

**Suggested:** «работает в cgroup, которую эта проверка не может распознать»

### RU-SD1-03 — MINOR — cli.serve_readiness.identity_restart

**Current:** «на {device} выполните `lop services status`, чтобы проверить записи, затем `lop services restart` для записи данных об идентичности процессов; перед действием перезапуск проверяет каждый демон»

**Problem:** The safety meaning and command sequence are intact, but «перед действием перезапуск проверяет» is awkward metonymy. Name the command as the actor, and use the more natural «идентификационные данные» for the recorded process-identity data. Do not imply that the diagnostic itself has established the current identity or authorized an unchecked restart.

**Suggested:** «на {device} выполните `lop services status`, чтобы просмотреть записи, затем `lop services restart`, чтобы записать идентификационные данные процессов; перед перезапуском команда проверяет каждый демон»

## Complete per-key assessment

Every key passed mechanical parity. The following covers semantic, register, terminology and fit reasoning separately from those checks.

| Key suffix | Assessment |
|---|---|
| unproven | Meaning preserved: identity is unconfirmed because no start-time comparison could be made. No claim that absence of proof proves a different process. Impersonal technical register is acceptable. |
| not_reporting | RU-SD1-01; absence of heartbeat retained, but Russian heartbeat wording is unidiomatic. |
| not_this_daemon | Accurate: the recorded PID is now held by another process. PID remains a technical abbreviation; no unsafe instruction added. |
| unknown_state | Accurate, concise, naturally impersonal. |
| install_unknown | Accurate inability to compare the running build with the installed build; does not say the builds differ. |
| cgroup_unknown | RU-SD1-02; unknown retained, but the classification diagnostic needs native wording. Literal cgroup is preserved. |
| recorded_gap | Both build placeholders retained. Explicitly distinguishes the recorded build from the installed build and states that process identity is unconfirmed; it does not assert that the current PID runs the recorded build. |
| unknown | All one/few/many/other arms reviewed. Singular vs plural daemon and pronouns/verbs agree. Count is a selector, not displayed, so few/many use nominative «демоны», not count-governed «демона/демонов». Unknown current-build and supervision status retained. «под управлением» matches the glossary. |
| stale | All four arms reviewed. Singular «размещённые на нём» vs plural «на них» and «ту другую сборку» vs «те другие сборки» preserve the daemon-hosted remote-session consequence. No claim added that the install has already replaced that process's build. «сессии» matches the glossary. |
| other_unit | All four arms reviewed. «службы», «юнит», «под управлением» follow the glossary. Preserves management by another service, rather than asserting no supervision at all. `lop serve` stays intact (including the source's unbackticked unit reference). |
| identity_restart | RU-SD1-03; both exact commands, inspection-before-restart ordering and per-daemon pre-action check retained. The minor idiom fix must preserve all three. |

## Fit, typography and direction

The sidecar supplies no maxCells or maxChars for any of these 11 keys. These are diagnostic prose/detail strings, not button/tab labels. Plain-text samples measure 39–190 cells; compound diagnoses will wrap on an 80-column terminal, as will their English counterparts. This is a wrapping risk to verify at integration, not evidence of a fixed-width overflow or a reason to invent a budget violation. Placeholder lengths vary and the sample lengths are not maxima.

No CJK-width, combining-mark or bidi-control characters were found in these Russian strings. Cyrillic text uses ordinary single-cell terminal glyphs in the measured samples. Russian em-dash spacing in stale is correct; no prose quotation marks require correction; backticks and commands are unchanged. Digits are formatter-owned/Latin. Formal «выполните» matches вы-register; no added politeness or alarming punctuation. Russian is LTR; the Urdu-specific RTL requirement is not applicable, and no RTL-rendering proof is asserted.

## Ledger hand-off

`ledger-receipt.json` is the proposed single append object, with a <=280-character note pointing to record.json. `append-validation.log` captures its actual history-sha and append --dry-run validation. The manager owns the subsequent append; no history, immutable snapshots or existing bundles were changed here. `receipt.json` is the bundle companion hash manifest, distinct from the ledger append object. Notes and reasoning in this report/record are labeled audit notes, never execution evidence.

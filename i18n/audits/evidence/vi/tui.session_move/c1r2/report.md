# c1-a2-vi — independent audit, locale `vi`, round-2 delta (Slice C1: core CLI + TUI paths)

**Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 0 MINOR, 2 NIT.**
Ledger status therefore `passed` for both namespaces (a `passed` row requires zero
unresolved BLOCKER/MAJOR). The two NITs are carried as dispositions, not gates.

Translations were audited, never edited. No translation API, SDK, MT service or
back-translation was used; every judgement below is reasoning over the strings plus
my own counts over the shipped `vi` catalogue.

## Provenance

| | |
| --- | --- |
| locale | `vi` |
| slice / scope | 22 keys — `cli.config.*` (15) and `tui.session_move.*` (7); round-2 delta |
| audit id | `c1r2` |
| auditor agent | `i18n-auditor` (lopdev team) |
| session | `c04c7cb774d2` |
| served model | `deepseek/deepseek-flash` (session header; no fallback notice in this transcript) |
| independent from translator | yes — different agent, session and model (translator: `i18n-translator`, session `d9395b82e597`, `openai/gpt-6.1-sol`, as stated in `vi.v2-changes.md`; I did not translate or edit these strings) |
| audit date | 2026-11-11 |
| round-1 reports read | `c1-prep/audit-vi-B.md` (this slice). I also opened one file named `audit-vi-A.md` outside the supplied directories (session `2b8bafe5512c`); it audits the mobile M1 slice, not these keys, so it is not this slice's A pass |

Inputs, hashed from the bytes I actually read (full table in each `record.json`):

| role | path | sha256 |
| --- | --- | --- |
| source + context (slice) | `c1-prep/keys.json` | `ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc` |
| en catalogue | `catalogues/en/{cli.config,tui.session_move}.json` | per namespace in `record.json` |
| en context sidecar | `catalogues/en/{cli.config,tui.session_move}.context.json` | per namespace in `record.json` |
| v1 translation | `c1-prep/vi.json` | per `record.json` |
| v2 candidate | `c1-a2-vi/catalogues/vi/{cli.config,tui.session_move}.json` | per `record.json` |
| translator dispositions | `c1-prep/vi.v2-changes.md` | per `record.json` |

Read first: `i18n/style-guides/vi.md`, `i18n/style-guides/en.md`, `i18n/glossary.md`,
then `keys.json` (en + role/tone/width notes), then the en catalogue, then `vi.json`,
`vi.v2.json`, `vi.v2-changes.md`, then the shipped `vi` catalogue for consistency,
then the two use sites (`local_operator/cli.py` config output, `local_operator/tui/session_move.py`).

## Commands run (actual output in `audit.log`)

1. `python scripts/i18n/check.py` **fixture-root run of its `check_catalogues()`** over an
   en+vi copy with the v2 candidate inserted
   (`~/local-operator/.venv/bin/python run_check.py`, monkeypatching
   `local_operator.i18n.catalogues._CATALOGUES` to the fixture) →
   `check.py check_catalogues(): 0 problem(s)`, exit 0. *Not* the repo's own
   post-insertion run — the candidate is not in the tree, so the integration owner
   must re-run it after landing.
2. `~/local-operator/.venv/bin/python verify_mechanical.py` → `MECHANICAL: 0 failure(s)`, exit 0.
3. `~/local-operator/.venv/bin/python recount.py` → my own shipped-usage count (below), exit 0.

## Mechanical pass on v2 (my own, not the translator's assertions)

Key set 22/22 and **order identical to `keys.json`**. Placeholder tuples identical per key
(name and kind); none of the 22 sources carries ICU plural/select, so no CLDR category
question arises and no `one`/`other` branch was invented. Backtick spans byte-identical
(`cannot_wait`, `deadline_exceeded`, including the `{session}`/`{to}` ICU names inside them).
Typed tokens verbatim where the source has them: `/approvals auto`, `/move`, `--to`,
`--keep`, `--queue`, `--drop-session-roles`, `--wait`, literal `local`. No key added or
dropped. No empty values, no value English-identical to source, no bidi control characters,
all values NFC. Typography clean: no double spaces, no space before punctuation, no ASCII
`...`, no straight quotes, em dashes all ` — `.

Angle brackets: counts match per key (3/1/3). Words inside brackets are translated
(`<đường dẫn>`, `<phiên>`, `<thiết bị ngang hàng>`, and `|local` kept), which is what the
`ambiguous` and `to_needs_device` notes instruct.

Widths (`rich.cells.cell_len`): `running_sessions_unchanged` 209, `deadline_exceeded` 127,
`list_desc_classification_notice` 118, `list_desc_detail_length` 113,
`ambiguous` 110, `cannot_wait` 95; `list_title` 17. No key in the group carries a
`maxCells` hint and every context says "no cell cap; one terminal line, may wrap", so
nothing here is a fit failure. (`cli.config.list_description` prefixes an indented row, and
the longest description lines already wrap by design.)

## Round-1 findings: adjudication

**Declined findings: none to adjudicate.** `vi.v2-changes.md` claims 0 declined; I verified
that against the bytes rather than the claim — every one of the 6 distinct round-1 issues has
applied bytes in v2 (2 A-side observations, B m1/m2/m3, B n1/n2), so the "0 declined" line is
true and no overturn is possible. There is therefore no DECLINED item to uphold or reject;
what follows is my independent judgement on each *applied* fix.

| # | Round-1 finding | My adjudication | Basis |
| --- | --- | --- | --- |
| A-summary-1 | Underscored words inside display angle brackets (`<đường_dẫn>`, `<thiết_bị_ngang_hàng>`) | **Upheld; fix correct.** Spaces are right: shipped `vi` uses the space form for translated placeholder words (`<tên>`, `<tên mới>`, `<config root>`), and underscores are reserved for tokens that mirror English identifiers (`<device_id>`, `<tenant_id>`). Underscored, the words read as something to type verbatim. | recount of 27 bracket tokens |
| A-summary-2 / B m3 | `trình bao` for *shell* | **Upheld; fix correct.** `trình bao` occurs **0** times in 3 978 shipped `vi` values; `shell` occurs **20** times, including `cli.permissions.setup.refused.isolated` ("Hãy chạy từ shell đăng nhập của chính bạn"). | my count |
| B m1 | `cơ chế nén` vs shipped `bộ máy nén` | **Upheld; fix correct and term-level right.** The three shipped `wire.settings.{conversation_length,detail_length,max_learnings_history}.help` values read "Đã lỗi thời. Được thay thế bởi bộ máy nén." (3 hits), while `cơ chế nén` (1 hit) belongs to `wire.settings.compaction_strategy.help` ("Cơ chế nén nào được dùng"), i.e. the *strategy* selector, not the engine. The CLI rows now match the shipped engine term. | my count |
| B m2 | Timeout notice implied "still running" | **Upheld; fix correct and safety-complete.** v2 says "hết thời gian chờ, chưa rõ kết quả chuyển" — an unknown outcome, neither success nor failure, and it keeps the re-run command byte-identical. The use site confirms the semantics: `session_move.py` reaps the child on `subprocess.TimeoutExpired` before building this refusal, so the process is gone and only the outcome is unknown. | code path |
| B n1 | Stiff `không thành phần nào đọc khóa này` | **Upheld; fix correct**, with a NIT follow-up below (vi2-01). | judgement |
| B n2 | Missing *provider* concept in the hosting description | **Upheld; fix correct**, with a NIT follow-up below (vi2-02). | judgement |

Round 2 note on the fix that went further than the finding: v2 did not restore the old
`Nền tảng cung cấp AI`; it added `nhà cung cấp` (77 shipped hits, e.g. `cli.global_help.hosting`
"Nhà cung cấp sẽ dùng (…)"). Terminology is now consistent with the shipped catalogue; the
word order is the remaining NIT.

## Findings

### vi2-01 — NIT — `cli.config.list_unrecognised`

- Current: `khóa không xác định; không có gì đọc khóa này`
- English: "not a recognised key; nothing reads it"
- Problem: the second clause is a real improvement over v1 (`không có gì đọc khóa này` reads
  naturally). The first clause drifts from the shipped rendering of the same English word:
  shipped `vi` renders *unrecognized* as `không được nhận diện` (`cli.failure_details.unrecognized`
  = "đối số không được nhận diện: {arguments}"), and reserves `không xác định` for *unknown*
  (18 shipped hits, e.g. `tui.import.error_source` "Nguồn {source} không xác định"). A config key
  nothing reads is "not recognised", not "undetermined".
- Suggested: `khóa không được nhận diện; không có gì đọc khóa này`
- Note: both readings work; this is consistency, not correctness. It does not gate.

### vi2-02 — NIT — `cli.config.list_desc_hosting`

- Current: `Nền tảng của nhà cung cấp AI (ví dụ: radient, openai, deepseek, anthropic, openrouter)`
- English: "AI provider platform (e.g., radient, openai, deepseek, anthropic, openrouter)"
- Problem: `của` makes it possessive — "the platform *that the provider owns*" — whereas the
  setting selects which provider platform to use. The compound reads tighter in a table row and
  matches shipped style (`cli.exec_worker_help.hosting` "Ghi đè nhà cung cấp").
- Suggested: `Nền tảng nhà cung cấp AI (ví dụ: radient, openai, deepseek, anthropic, openrouter)`
- Note: meaning survives either way; the examples make the setting unambiguous. Does not gate.

## Safety lines (both verified against the use sites)

`cli.config.running_sessions_unchanged` — all four claims survive: running sessions are
unchanged (`Các phiên đang chạy không thay đổi`), a config write cannot loosen one
(`ghi cấu hình không thể nới lỏng yêu cầu phê duyệt của các phiên này`), `/approvals auto` must be
typed in **each** session the user wants ungated (verbatim token, `trong từng phiên`), and new
sessions open at auto (`Phiên mới bắt đầu ở chế độ auto`, `auto` kept as the mode word).
Matches `cli.py`: the line prints only when the stored `tool_approval_mode` is `auto`, and the
process that runs `lop config edit` holds no gate. No claim was weakened.

`tui.session_move.deadline_exceeded` — reports an **unknown** outcome; asserts neither success nor
failure; the re-run command `lop sessions move {session} --to {to}` is byte-identical and the
sentence says what running it again tells you. Matches `session_move.py` (child reaped on timeout).

## Coverage limits (stated plainly)

1. **Round-1 auditor A's report for this slice was not available to me.** Its two observations
   reach the record only as the translator's quotations, labelled `A-summary-1`/`A-summary-2` in
   `vi.v2-changes.md`; those labels are the translator's, not original auditor IDs. The only
   `audit-vi-A.md` on this machine (session `2b8bafe5512c`) audits the **mobile M1** slice
   (53 phone-app keys), not these 22 — so it is not this slice's A pass and I did not use it as one.
   Consequence: this group (settings + error notices — high-visibility, where a second independent
   pass is expected) has **one** full independent pass available to me (B) plus my delta pass and
   second-hand A observations. My delta covers the 9 changed keys and the unchanged neighbours they
   contradict; I read the other 13 once for safety and contradiction checks and found nothing
   gating, but I do not claim a full second pass over unchanged bytes.
2. **No rendered frame.** This role had no rendered-frame/browser surface in this run, so width is
   `rich.cells.cell_len` measurement over the strings, not a capture. No key here carries
   `maxCells`; the two long ones (209, 127 cells) are single terminal lines the contexts say may wrap.
3. **The candidate is not in the tree**, so nothing here is a claim about shipped bytes; the
   integration owner must insert it, re-run `scripts/i18n/check.py`, and only then let this
   verdict attach to shipped content.

## Record / handoff

Evidence bundle (built in my own scratchpad, canonical layout, **not committed** — I do not edit
the repo): `c1-a2-vi/evidence/vi/cli.config/c1r2/` and
`c1-a2-vi/evidence/vi/tui.session_move/c1r2/`, each with `record.json`, `report.md` (a copy),
`audit.log` (a copy), and `receipt.json` whose hashes I verified against the bytes on disk.
Landing target: `i18n/audits/evidence/vi/<namespace>/c1r2/`. Reproduction scripts stay beside the
bundle: `verify_mechanical.py`, `recount.py`, `run_check.py`, `build_bundle.py`.

**No ledger row was appended, deliberately.** `i18n/ledger.json` has no `vi/cli.config` or
`vi/tui.session_move` row today, and the v2 bytes are an unintegrated candidate — appending now
would pin `status: passed` for bytes the tree does not ship. Each bundle carries the ready row as
`ledger-row.json` (validated against the existing row shape; `source_sha256` is the sha256 of the
**en** catalogue bytes, matching how shipped rows were taken). The integration owner runs, with
the landing commit:

```sh
python scripts/i18n/ledger.py history-sha --locale vi --namespace cli.config
python scripts/i18n/ledger.py append \
    --receipt <bundle>/vi/cli.config/c1r2/ledger-row.json \
    --expected-history-sha <sha printed above> --apply
python scripts/i18n/ledger.py history-sha --locale vi --namespace tui.session_move
python scripts/i18n/ledger.py append \
    --receipt <bundle>/vi/tui.session_move/c1r2/ledger-row.json \
    --expected-history-sha <sha printed above> --apply
```

Row content as written in `ledger-row.json` (notes within the 280-character budget — 151 and 185
characters, not truncated):

- `vi/cli.config` — `source_sha256 40649e7419d1…`, status `passed`, translator
  `i18n-translator`/`openai/gpt-6.1-sol` 2026-11-11, auditor `i18n-auditor`/
  `deepseek/deepseek-flash` 2026-11-11, findings 2 (NIT only) —
  "Round-2 delta over 9 changed keys (5 here): DEPRECATED tags now use shipped «bộ máy nén»; all 3 safety claims intact; 0 BLOCKER/0 MAJOR; 2 NIT carried."
- `vi/tui.session_move` — `source_sha256 f4dd682b3b50…`, status `passed`, same translator and
  auditor, findings 0 —
  "Round-2 delta: 4 changed keys. Timeout notice reports an UNKNOWN outcome with the rerun command byte-identical; bracket placeholders no longer look typed. 0 BLOCKER/0 MAJOR, 0 findings."

# Style guide — Vietnamese (`vi`)

Read with `i18n/glossary.md`; everything here is vi-specific and binding for
translators and the auditor.

## Formality & tone

- Address: **bạn** — recorded decision (the brief floated an anh/chị register,
  rejected): "bạn" is the standard in modern software (Google/Apple/Microsoft
  vi) and matches the terse dev voice; "anh/chị" is banking/customer-service
  register and reads stiff in a terminal tool. "Quý khách" never.
- Vietnamese drops pronouns: do not write "bạn" in every string — it can
  appear at first contact and on blocking decisions; elsewhere plain verbs
  carry the sentence ("Lưu lại", "Thử lại").
- "Vui lòng" only where the English says "please"; prefer direct phrasing.
- Terse; no "!"; verbs unconjugated for labels (Lưu, Tiếp tục, Hủy).
- Sentence case; no Title Case.

## Product-name policy

- Keep exactly as in glossary §1: Local Operator, `lop`, tool names,
  `--flags`, config keys, paths, identifiers — as Latin runs.
- Kept-English technical list: commit, push, pull, merge, token, cache,
  script, prompt, deploy · API, CLI, TUI, JSON, YAML, HTTP, URL — settled in
  vi developer usage.
- Prefer Vietnamese where settled: máy chủ (server), tệp (file), thư mục
  (folder), mạng (network), cài đặt (settings), thiết bị (device), nhóm
  (team), kỹ năng (skill).

## Locale calls (vi)

- session → phiên · wake → thức dậy (intransitive — the agent wakes itself;
  "đánh thức" would be waking someone else)
- tool trace → dấu vết · transcript → bản ghi
- approvals → phê duyệt · schedule → lịch (lên lịch for the verb)
- relay → chuyển tiếp · usage → lượng sử dụng · cost → chi phí
- workspace → không gian làm việc — long; if a width budget cannot take it,
  the wave records "workspace" as the fallback rather than truncating.

## Plurals

- `other` only — Vietnamese has no plural agreement; never add one/other
  branches. Classifiers ("những", "các") only where the count meaning needs
  them. Example: `{count, plural, other {# phiên}}`

## Numbers, dates, digits

- Never hand-written — the formatter renders "1.234,5" and dates
  "10 thg 10, 2026"; 24-hour times.
- Digits: Latin (v1).

## Width rules

- vi runs long; tighten. Budgets: `maxCells` from the en sidecar; pinned:
  PERSIST_HINT 42/43 cells, tip pool 59, `/help` ≈ 55.
- Always the precomposed (NFC) form of Vietnamese characters — never
  decomposed combining-mark sequences, which break width counting and
  rendering in the TUI.

## Punctuation & typography

- Quotes: “ ”; em dash " — " with spaces; ellipsis …; no space before
  punctuation.
- Code spans, flags, paths stay byte-identical.

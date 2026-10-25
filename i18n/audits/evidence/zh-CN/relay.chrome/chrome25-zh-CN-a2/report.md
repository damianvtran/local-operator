# `chrome25-zh-CN-a2` — terminal delta verification of the zh-CN remediation

**Verdict: TERMINAL PASS** — no blocker or major remains; nothing out of scope was introduced.

- **Audit ID:** `chrome25-zh-CN-a2` (round 2, delta only — no full namespace re-audit)
- **Locale / namespace:** `zh-CN` / `relay.chrome` (83 keys)
- **Auditor:** `translation-auditor`, session **`6d6acecadd03`** — the same session that produced
  `chrome25-zh-CN-a1`, so the round-1 coverage this pass retains is my own.
- **Model:** `radient/auto` — the model named in this session's own runtime header. No underlying
  provider is inferred from that label, and no fallback was selected or used.
- **Date:** 2026-10-25
- **Worktree / base:** `~/local-operator/.worktrees/relay-chrome-i18n`, branch
  `i18n/relay-chrome-wave`, base **`535cea359825411741069cbffcee42e2927df9d3`**; the target catalogue
  is an untracked audit input, no commit exists. I edited nothing in the repo, target or ledger.
- **Round-2 inputs:** the remediation record at
  `~/.local-operator/sessions/5217c4e5f8e3/scratchpad/{remediation.md,remediation.json}` and the
  before snapshot it names. The brief for this round directs the auditor to that record; R1's
  "do not read translator reports" constraint applied to the independent first pass. I did not read
  any other locale's remediation record.

## Input integrity — hashes measured here, not copied

| Artefact | SHA256 |
|---|---|
| English source (unchanged since R1) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context sidecar (unchanged since R1) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| Before snapshot (`zh-CN-relay.chrome.candidate-before-remediation.json`) | `ca2c2df9bf514ef301af0feb86062a96e17f5a2e40d2f8a1febed26f5746646e` |
| **Current target (final)** | **`ba4215e753320387d61bc9d9661eb2a63420473436e492f6da657ac8595c1855`** |
| Bytes changed | 4,798 → **4,791**, 85 lines in both |

The before snapshot is **byte-identical to the bytes my round-1 audit passed**
(`ca2c2df9…`), so the delta is measured against exactly the version R1 reviewed, not a recollection.
Both reference hashes match the pinned values in the brief. All four hashes above were recomputed
with `shasum -a 256` / `hashlib` in this pass; none was transcribed from prose.

## Exact delta — 5 keys, nothing else

`diff` of the before snapshot against the current file: **exactly 5 changed lines, no lines added or
removed**, each naming one of these keys, no trailing whitespace. All **78 other keys are
byte-identical** to the R1-reviewed file.

| # | Key | Before (R1) | After (R2) | English |
|---|---|---|---|---|
| 1 | `relay.chrome.link.reconnecting` | `正在重连……` | `重连中……` | `Reconnecting…` |
| 2 | `relay.chrome.link.since_seconds` | `{count, plural, other {重连中 · 上次同步于 # 秒前}}` | `{count, plural, other {重连中 · 上次同步：# 秒前}}` | `{count, plural, one {Reconnecting · last synced # second ago} other {… # seconds ago}}` |
| 3 | `relay.chrome.link.since_minutes` | `{count, plural, other {重连中 · 上次同步于 # 分钟前}}` | `{count, plural, other {重连中 · 上次同步：# 分钟前}}` | `… # minute ago …` |
| 4 | `relay.chrome.list.empty_hint` | `点击“新建”开始对话，或在电脑上用 Local Operator 开始` | `点击“新建”开始对话，或在电脑上用Local Operator开始` | `Start one with New, or from Local Operator on your computer` |
| 5 | `relay.chrome.session.loading` | `正在读取对话……` | `正在加载对话……` | `Loading conversation…` |

## Round-1 carry-forward proof (for the 78 unchanged keys)

- The snapshot R2 diffed against hashes to `ca2c2df9…`, the exact target R1 audited and reported
  PASS on (R1: `~/.local-operator/sessions/6d6acecadd03/scratchpad/report.md`, 83/83 keys read,
  2 minor + 4 nit).
- Every one of those 83 keys except the five above is `==` its R1 value in memory, and the line diff
  shows no other line touched, so R1's per-key judgements — accuracy, idiom, register, glossary,
  placeholder/CLDR integrity and width — stand unchanged on identical bytes.
- The changed keys are re-read in full below; the whole file was also re-parsed, re-measured and
  re-checked structurally (there is no partially-updated key: canonical format, LF, no BOM, no
  duplicate keys, 83 keys in `en` order).

## Remediation dispositions — verified against the bytes

| Finding | Sev. | Disposition claimed | **My verification** |
|---|---|---|---|
| `chrome25-zh-CN-a1-F01` | minor | Fixed (2 keys) | **Accepted as fixed** — the calque `同步于 # 秒前` is gone; the colon form `上次同步：# 秒前` is what I proposed and reads as a status line. `#` still renders the count in every branch; `·` kept; `秒前/分钟前` unchanged. |
| `chrome25-zh-CN-a1-F02` | minor | Fixed | **Accepted as fixed** — no CJK↔Latin space remains in the string; `“新建”` still quotes the button's exact word (sidecar requirement) and `Local Operator` stays untranslated. |
| `chrome25-zh-CN-a1-F03` | nit | Fixed | **Accepted as fixed** — `正在加载对话……` restores the locale's `Loading → 正在加载` mapping (6 existing strings) and no `读取对话` remains in the file. |
| `chrome25-zh-CN-a1-F04` | nit | Fixed | **Accepted as fixed** — `重连中……` now leads all three link strings; `正在重连` no longer appears anywhere. |
| `chrome25-zh-CN-a1-F05` | nit | Rejected, no text change | **Accepted** — this is exactly what my R1 finding proposed ("keep `关闭`, record the exception"): `忽略` carries the *send-no-reply* sense from the asks surfaces and would mistranslate a snackbar. No cosmetic churn requested. |
| `chrome25-zh-CN-a1-F06` | nit | Rejected, deferred repo-wide | **Accepted** — I framed it as repo-wide drift for the glossary owner, not a chrome defect; the two keys are internally consistent. No change requested here. |

**New findings in the delta: none.** No new blocker, major, minor or nit — I am not opening a nit
loop on wording that is correct and internally consistent.

**One precision correction to the record, not a finding against the file.** `remediation.json` says
the spaced product name "was the only spaced one in all 83 catalogues". Measured across all 84
zh-CN catalogue files, there are 11 other digit/Latin↔Chinese adjacency spaces: 9 in
`tui.welcome.step.*` (`1 连接账户 · 2 登录…`, a numbered-step convention), 1 in
`relay.asks.duration.less_than_minute` (`低于 1 分钟`), 1 in `mobile.setup.consent`
(`USD {amount}的隧道订阅`). The narrower claim that matters is true and verified: the **product
name** is written unspaced in 3/3 existing prose occurrences (`cli.browser.json`), and this file was
the only place the product name was spaced. The fix makes it match that pattern; it does not imply the
locale never spaces digits, and the record should not be read that way.

## Delta re-checks (all executed here)

Script `check_zh_a2.py`, output `executed-checks-a2.log`, exit 0 → `RESULT: TERMINAL PASS`:

```
PASS  en source hash unchanged since round 1
PASS  context sidecar hash unchanged since round 1
PASS  before snapshot equals the exact bytes round 1 audited
PASS  changed-key set is exactly the 5 expected keys
PASS  all 78 other keys byte-identical to round 1
PASS  exactly 5 lines differ, found 5   (every changed line names an expected key)
PASS  no duplicate JSON keys · UTF-8 no BOM · LF only · trailing newline
PASS  canonical json.dumps(indent=2, ensure_ascii=False) + newline, byte-exact
PASS  all 83 zh messages parse with the repo ICU engine
PASS  message_arguments(en) == message_arguments(zh) for all 83 (names + kinds)
PASS  every plural uses exactly the zh-CN CLDR set ('other'); no empty branch; every branch renders its count
PASS  no changed key exceeds its declared maxCells
PASS  changed values introduce no style/register defect
PASS  argument-parity negative control still discriminates (renamed placeholder → detected)
```

Rendered delta branches (repo engine, locale `zh-CN`, product cell measure):

```
link.since_seconds  count=1/5/21/1234 -> 重连中 · 上次同步：1 秒前 … 1,234 秒前   (25-29 cells / 44)
link.since_minutes  count=1/2        -> 重连中 · 上次同步：1 分钟前 …           (27 cells / 44)
link.reconnecting   —                -> 重连中……                               (8 / 32)
session.loading     —                -> 正在加载对话……                          (14 / 28)
list.empty_hint     —                -> 点击“新建”开始对话，或在电脑上用Local Operator开始 (50, no budget; wraps)
```

Widths are **computed cells, not rendered frames**: the brief treats a width signal as requiring an
actual rendered fit rather than an automatic major, and none of the five is near a budget edge at
realistic counts (the seconds/minutes line only grows with an outage, and the minute form is used
after 60 s).

**Repo gates on the remediated bytes** (`repo-check-a2.log`, `repo-generate-check-a2.log`):

- `scripts/i18n/check.py` → **exit 0**: `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or
  below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)`.
- `scripts/i18n/generate.py --check` → **exit 1**, `changed: local_operator/i18n/data/shipped.json`.
  Independently re-derived cause: `i18n/ledger.json` holds 326 entries and **0 `relay.chrome` rows in
  any locale** (zh-CN has 83 rows for its other namespaces), so `_shipped_locales()` cannot pass the
  carried namespace and a regeneration now would shrink the committed 8-locale set to `["en"]`.
  Wave prerequisite, unchanged by this remediation, not a translation defect.

## Safety / high-visibility delta

- **Semantics I checked explicitly on the changed strings:** `重连中 · 上次同步：# 秒前` claims only
  *reconnecting* and *how old the last sync is* — it promises no delivery, queue or success, matching
  the sidecar ("sends wait and deliver on reconnect" is left to the composer's copy); `重连中……` is an
  in-progress state with no completion claim; `正在加载对话……` likewise.
- **Approve / ended / resume copy is byte-identical to the version my R1 pass passed**: `批准` pill,
  `批准——有请求等待你处理` (still starts with the pill word), `会话已结束 · 历史记录保留`,
  `发送消息或点击“恢复”…` using the button's exact word `恢复`, `正在恢复……`, and
  `未响应——消息将等待发送`. Verified `==` against the R1 snapshot.
- **Second-auditor delta signoff is still outstanding and is not mine to issue.** The independent
  safety pass `chrome25-safety-b1` (session `acd16c49df72`) signed zh-CN **PASS 17/17, 0 findings at
  target `ca2c2df9…`** — the pre-remediation bytes. Of its 17-key scope, **14 keys are byte-identical**
  and **3 changed**: `relay.chrome.link.reconnecting`, `relay.chrome.link.since_seconds`,
  `relay.chrome.link.since_minutes`. Its signoff is therefore stale for exactly those three key-names
  and the safety stream must re-sign the delta on `ba4215e7…` before ledger integration treats the
  high-visibility gate as fresh.

## Verdict

- **`zh-CN` / `relay.chrome`:** **TERMINAL PASS** on
  `ba4215e753320387d61bc9d9661eb2a63420473436e492f6da657ac8595c1855` — 83/83 keys covered (5 re-read
  this round, 78 retained on byte identity), 0 blocker, 0 major, all six R1 findings closed
  (4 fixed, 2 correctly rejected without churn).
- **Residual, non-blocking, outside this file:** F05 and F06 glossary rows for the glossary owner;
  `relay.chrome` ledger rows + one `generate.py` run before CI can be green; and the safety stream's
  three-key delta re-signoff above.

## Limits

- No rendered frame, browser or runtime fit was produced here; width statements are computed cells.
- This is a delta pass: it does not re-litigate R1 findings on the 78 byte-identical keys, and it
  raises no new out-of-scope issues.
- I did not edit the repo, the target catalogue, the ledger or the generated artefacts.

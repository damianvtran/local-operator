# Translation audit — `relay.chrome` · zh-CN · `chrome25-zh-CN-a1`

- **Audit ID:** `chrome25-zh-CN-a1`
- **Auditor:** `translation-auditor` (independent of the translator; different session)
- **Session:** `6d6acecadd03`
- **Model:** `radient/auto` — the model the harness reported for this session. No fallback was
  invoked at any point, and no role/tool default was changed for this audit.
- **Date:** 2026-10-25
- **Repository:** `~/local-operator/.worktrees/relay-chrome-i18n`, branch
  `i18n/relay-chrome-wave`, base **`535cea359825411741069cbffcee42e2927df9d3`**
- **Scope:** one locale, one namespace, **all 83 keys**, full semantic + mechanical audit.
- **Authority read first:** `~/local-operator/AGENTS.md` ("Localized product copy"), `guide://i18n`,
  `docs/I18N.md` contract pointers, `i18n/glossary.md`, `i18n/style-guides/zh-CN.md`.
- **Independence:** I did not read any translator report, provenance sidecar or ledger row for this
  wave, and I edited nothing in the repository or the target file.

## Files audited (SHA256, measured here)

| Role | Path | SHA256 |
|---|---|---|
| English source (83 keys) | `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| Context sidecar (83 entries) | `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| **Target** | `local_operator/i18n/catalogues/zh-CN/relay.chrome.json` | **`ca2c2df9bf514ef301af0feb86062a96e17f5a2e40d2f8a1febed26f5746646e`** |

Both reference hashes match the values supplied with the task. The target hash was re-measured after
every check and is unchanged: nothing in this audit wrote to the file.

Target file facts: 4,798 bytes, UTF-8 without BOM, LF only, ends with a newline, 2-space indent,
83 keys, no duplicate JSON keys, every value a non-empty string, key order identical to `en`.

## Method

1. Read all 83 English values, all 83 context notes (role, tone, `maxCells`) and all 83 target values
   side by side, area by area, judging accuracy, idiom, register, glossary and safety wording.
2. Re-used **the product's own** ICU machinery, not a re-implementation:
   `local_operator.i18n.runtime.parse_message` / `message_arguments` / `plural_arguments` /
   `render_message`, with the generated CLDR tables in `local_operator.i18n.format`.
3. Compared every term the namespace uses against the same English term as already rendered in the
   other 83 zh-CN catalogue files (2,984 key pairs compared mechanically, then read).
4. Measured display width in cells (East-Asian Wide/Fullwidth = 2, everything else = 1) per rendered
   ICU branch, against the en sidecar `maxCells`.
5. Ran the repository's own gates read-only: `scripts/i18n/check.py` and
   `scripts/i18n/generate.py --check`.
6. Proved the checks discriminate with in-memory negative controls (no shared file mutated).

## Mechanical results — all green except one typographic rule

Command: `PYTHONPATH=<worktree> ~/local-operator/.venv/bin/python check_zh.py`
(log: `executed-checks.log`; script: `check_zh.py`; exit 1 = the one style violation below)

| Check | Result |
|---|---|
| 83 keys in `en`, `zh-CN` and the context sidecar; identical key sets and order | PASS |
| Every key has a context-sidecar entry | PASS |
| No value left byte-identical to English; no empty value | PASS |
| All 83 target messages parse with the repo ICU engine | PASS |
| `message_arguments(en) == message_arguments(zh)` — placeholder **names and kinds**, first-appearance order, nested branches walked | PASS (0 mismatches) |
| Every plural argument uses exactly the zh-CN CLDR category set `('other')`, has no empty branch, and renders its own count | PASS |
| Nested args inside plural branches survive (`{running}`, `{total}` in `agents_failed`) | PASS |
| `{state, select}` keeps all three branches `online` / `retrying` / `other` | PASS |
| No rendered branch exceeds its `maxCells` budget | PASS (widest: `session.tray.questions` 12 / 16) |
| No 您, no straight quotes, no `...`, no half-width prose punctuation, no trailing 。 | PASS |
| Ellipsis is the two-mark `……` form in all 9 values that use one; no lone `…` | PASS |
| No space between Chinese text and a Latin/digit run | **FAIL — 1 value** (`list.empty_hint`, F02) |

### Negative controls (in-memory strings; repository untouched)

| Control | Expected | Observed |
|---|---|---|
| A — `{count, plural, one {…} other {…}}` parses | accepted | accepted, rendered `1 个问题` |
| B — unbalanced `{` | rejected | `MessageSyntaxError` |
| C — renamed arg `{counts}` rendered with `count` | rejected | `MessageFormatError` (renamed arg) |
| D — plain text with no args | renders | rendered `已连接` |
| E — argument-set comparison on a pair that drops/renames `{name}` | detected | identical pair equal `True`, renamed pair detected `True` |
| F — `one`/`other` plural in a zh-CN message | flagged | categories read as `['one','other']` ≠ `('other',)` → gate discriminates |

## Rendered ICU branches (repo engine, locale `zh-CN`)

```
drawer.status    online            -> 已连接 · 此手机                      (15 cells)
drawer.status    retrying          -> 离线 · 正在重试                      (15 cells)
drawer.status    other/unknown     -> 正在连接……                          (10 cells)
link.since_secs  count=1/5/21      -> 重连中 · 上次同步于 1 秒前            (26-27 cells)
link.since_min   count=1/2         -> 重连中 · 上次同步于 1 分钟前          (28 cells)
tray.questions   count=1/3/1234    -> 1 个问题 / 3 个问题 / 1,234 个问题   (8-12 cells)
tray.tasks       2 of 5            -> 任务2/5                             (7 cells)
tray.agents      1 of 3            -> 智能体1/3                           (9 cells)
tray.agents_failed 0 and 2 failed  -> 智能体1/3 · 0 个失败 / … · 2 个失败   (20 cells)
row.actions      name=Latin run    -> Daily notes的操作                   (17 cells)
composer.folder  ~/local-operator  -> 工作目录：~/local-operator           (26 cells)
composer.spend   US$1.20           -> 费用与上下文：US$1.20                (21 cells)
```

Numbers arrive through `{, number}` and render locale-grouped (`1,234`), which is what the English
skeleton asks for; `#` inside plurals renders the same way. All fit their budgets.

## Findings

Severity: **blocker** none · **major** none · **minor** 2 · **nit** 4.

### chrome25-zh-CN-a1-F01 — minor — idiom (calque)

- **Keys:** `relay.chrome.link.since_seconds`, `relay.chrome.link.since_minutes`
- **Current:** `{count, plural, other {重连中 · 上次同步于 # 秒前}}` / `{… # 分钟前}`
- **Proposed:** `{count, plural, other {重连中 · 上次同步：# 秒前}}` / `{…：# 分钟前}`
- **Reason:** `同步于 # 秒前` is a word-for-word rendering of "synced # seconds ago"; 于 marks a point
  in time (「同步于 14:30」), not an offset, so 「同步于 1 秒前」 is not what a native UI would write.
  Either the colon form above or 「重连中 · # 秒前同步」 is idiomatic; the colon form keeps the
  English subject order and the full-width colon the rest of this locale uses.

### chrome25-zh-CN-a1-F02 — minor — typography (house style)

- **Key:** `relay.chrome.list.empty_hint`
- **Current:** `点击“新建”开始对话，或在电脑上用 Local Operator 开始`
- **Proposed:** `点击“新建”开始对话，或在电脑上用Local Operator开始`
- **Reason:** `i18n/style-guides/zh-CN.md` §"Punctuation & typography" says *"No space inserted between
  Chinese text and Latin/digit runs (width discipline; matches mainland software convention)"*, and the
  locale's own product-name precedent follows it — `cli.browser.json` writes
  「加载Local Operator扩展」 / 「在Local Operator扩展弹窗中…」. This is the only string in the namespace
  that spaces a Latin run. Cosmetic, not a meaning error. (Note: the guide's plural example
  `{# 个会话}` keeps a space after `#`, so the space in F01's proposed text is guide-sanctioned;
  the rule targets Latin words like the product name.)
- Accompanying note, not a defect: the target adds 点击 ("tap/click") where English says only "Start
  one with New". It is accurate UI guidance and uses the button's exact word 「新建」 as the sidecar
  requires; I am not asking for its removal.

### chrome25-zh-CN-a1-F03 — nit — glossary consistency ("Loading")

- **Key:** `relay.chrome.session.loading`
- **Current:** `正在读取对话……`
- **Proposed:** `正在加载对话……` (or record 读取 as a deliberate exception)
- **Reason:** Everywhere else this locale renders English "Loading …" as 正在加载…: `relay.session`
  (×2 `loading…`), `relay.session.tags.loading`, `relay.shell.usage.loading`,
  `tui.hub_links.state.loading`, `tui.review.loading` — six strings. 读取 is this locale's rendering of
  *"reading …"* (`relay.asks.sheet.loading` "reading asks…" → 正在读取提问……, `tui.library.loading`
  "Reading the library…" → 正在读取资料库……). Both are acceptable Chinese; the point is one English
  term, one local term. The translator's choice is defensible on meaning (the phone is retrieving the
  conversation), so this is a nit, not a correction of an error.

### chrome25-zh-CN-a1-F04 — nit — consistency ("Reconnecting")

- **Keys:** `relay.chrome.link.reconnecting` vs `relay.chrome.link.since_seconds`/`since_minutes`
- **Current:** `正在重连……` vs `重连中 · 上次同步…# 秒前`
- **Proposed:** unify — `重连中……` for the standalone string (shorter, matches the longer banner)
- **Reason:** The same English token "Reconnecting" is rendered with two forms in adjacent states of
  the same app-bar subtitle; a user watches the line change from one to the other as the link degrades.
  Both forms are idiomatic in their slot, so this is uniformity, not correctness.

### chrome25-zh-CN-a1-F05 — nit — glossary drift ("Dismiss")

- **Key:** `relay.chrome.snackbar.dismiss`
- **Current:** `关闭`
- **Proposed:** keep `关闭`; record the exception in the glossary or zh-CN style guide
- **Reason:** English "Dismiss"/"Dismissed" is 忽略 throughout this locale (`relay.ask`,
  `relay.asks`, `tui.asks` — eight strings), so the same English lemma now maps to two Chinese terms.
  Here the object is a snackbar, not an agent question: 忽略 in this product means "ignore a question
  and send no reply", which would be actively misleading on a snackbar. I flag the drift and recommend
  **no text change** — the exception should be recorded, not the string renamed.

### chrome25-zh-CN-a1-F06 — nit — glossary drift ("Not answering")

- **Keys:** `relay.chrome.session.subtitle.not_answering`, `relay.chrome.session.degraded`
- **Current:** `未响应` / `未响应——消息将等待发送`
- **Proposed:** keep `未响应` (internally consistent) **or** align the whole repo on `无响应`
- **Reason:** The same English "not answering" is 无响应 in `tui.small.info_panel.not_answering` and in
  `relay.asks.error.read_timeout`. The chrome pair is consistent within itself, and 未响应 is the more
  common desktop Chinese string; this is repo-wide drift for the glossary owner to settle, not a
  chrome defect to fix unilaterally.

## High-visibility safety wording — checked explicitly

| Concern | Key(s) | Reading | Verdict |
|---|---|---|---|
| Approval action is **approve**, not "accept"/"OK"/"allow" | `session.tray.approve` = 批准 | Matches the locale's approval verb: `relay.asks.pending.approve` "approve" → 批准, `cli.*` "approve" → 批准; the noun "approvals" stays 审批, which is the glossary row | PASS |
| Accessible name must **start with the visible pill word** (voice control) | `session.tray.approve_label` = `批准——有请求等待你处理` | Starts with 批准, exactly the pill's word, and states a request is waiting for you | PASS |
| "Approve" pill must not claim the run continued / was approved | same | Pill is a request for the operator's decision; no completion claim | PASS |
| **Ended** banner must keep "history kept" and must not imply deletion | `session.ended` = `会话已结束 · 历史记录保留` | States ended + history kept; no loss claim | PASS |
| **Resume** wording must name the button beside it | `session.ended_hint` = `发送消息或点击“恢复”会在主目录中重新打开`; `session.resume` = 恢复 | Uses the button's exact word 恢复; matches repo precedent for "resume" (恢复 in `cli.sessions_help`, `tui.notices`); 主目录 is correct per the sidecar ("the operator's home directory on the computer: the session comes back there, not in its project folder") and matches this locale's 目录 for directory (`relay.session.directory.sheet_title` "working directory" → 工作目录) | PASS |
| Busy state must replace the action, not look like success | `session.resuming` = 正在恢复…… | In-progress wording, ellipsis per locale | PASS |
| **Degraded** state must warn without promising delivery or blaming the user | `session.degraded` = `未响应——消息将等待发送` | "sends will wait" preserved; sidecar says messages are kept and delivered later — no loss or failure claim | PASS |

A separate second independent pass covers safety copy; this section is my own reading, not a
substitute for it.

## Glossary / consistency evidence (same English term → same Chinese term)

| English | zh-CN in this namespace | Precedent elsewhere in zh-CN | Verdict |
|---|---|---|---|
| session | 会话 (新建会话, 历史会话, 会话状态, 会话已结束) | 会话 throughout | consistent |
| conversation | 对话 (对话, 搜索对话, 返回对话, 暂无对话) | relay.* uses 对话 for conversations | consistent |
| agent | 智能体 | 智能体 (glossary row) | consistent |
| project | 项目 | 项目 | consistent |
| usage | 用量 | relay.shell/tui.usage/wire.slash 用量 | consistent |
| Library | 资料库 | tui.library 资料库 (wire.errors.desktop still says 资源库 — pre-existing repo drift, not this namespace's) | consistent with the prevailing term |
| reasoning effort | 推理强度 | tui.notices, tui.small, wire.settings, cli.exec_help 推理强度 | consistent |
| folder/directory | 工作目录 (Working folder), 主目录 (home folder) | relay.session.directory.sheet_title "working directory" → 工作目录 | consistent |
| active / previous | 活跃 / 历史 | relay.shell "Active Sessions" → 活跃会话, "Previous Sessions" → 历史会话 | consistent |
| Pin / Unpin | 置顶 / 取消置顶 | wire.settings 置顶 | consistent |
| Questions | 问题 | relay.asks 「# 个问题」, ask chip 「需要审批」 | consistent |
| approve | 批准 | relay.asks.pending.approve 批准 | consistent |
| Resume | 恢复 | cli.sessions_help/tui.notices 恢复 | consistent |
| Loading | 读取 (F03) | 正在加载 ×6 | **drift — F03** |
| Reconnecting | 正在重连 / 重连中 (F04) | — (only this namespace) | **internal drift — F04** |
| Dismiss | 关闭 (F05) | 忽略 in 12 dismiss-context keys (relay.ask ×5, relay.asks ×2, tui.asks ×5) | **drift, context-justified — F05** |
| Not answering | 未响应 (F06) | 无响应 ×2 | **repo drift — F06** |

Register: 你 throughout, no 您, no 请 outside 请求 ("request"), no ！, no sentence-final 。 on
labels/statuses, `“ ”` mainland quotes, full-width colon/em dash `——` (matching relay.asks /
relay.approvals; the ` — ` style in relay.ask / tui.* is pre-existing repo variance), `……` ellipsis
per the style guide. Latin product name `Local Operator` kept untranslated, model/label runs kept
as-is. RTL not applicable (LTR locale); no bidi or Latin-isolation concerns.

## Width budgets

Measured per rendered branch in cells, compared to the en `maxCells`: **no key exceeds its budget**.
All Chinese strings are shorter than or comparable to their English source (no string is materially
longer than English: the largest ratio in this namespace is `subtitle.ended` 6 cells vs 5, because
Chinese has no plural/space padding). Tightest fixed budgets: `appearance.on/off` 4/6,
`list.starting` 10/12, `row.saving` 10/12, `session.resuming` 10/12, `snackbar.dismiss` 4/12,
`tray.questions` 12/16, `approve_label` 22/48.

These are computed cell widths, **not** rendered frames. A rendered pass on the real surfaces is
separate evidence and is not claimed here.

## Repository gates (read-only, run from the worktree at base `535cea3598`)

- `scripts/i18n/check.py` → **exit 0**: `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or
  below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)`.
  This is the project's own catalogue gate — key sets, argument tuples (name **and** kind), plural
  categories and `other` presence per locale — and it passes for all locales, zh-CN included.
  Log: `repo-check.log`.
- `scripts/i18n/generate.py --check` → **exit 1**, `committed artifacts are stale — regenerate`,
  `changed: local_operator/i18n/data/shipped.json`. Log: `repo-generate-check.log`.

  Diagnosis (derived here, not guessed): `_shipped_locales()` requires every namespace a locale
  **carries** to have a `passed` ledger row whose `source_sha256` equals the current en hash.
  `zh-CN` carries 84 catalogues and the ledger has 83 rows for it — the only missing row is
  `relay.chrome`, the file under audit. The same is true for es/fr/hi/ru/ur/vi. So the freshly
  computed set is `["en"]` and the committed `shipped.json` (8 locales) is what a regeneration would
  currently *shrink*. This is the expected pre-review state of the wave, **not a translation defect**
  and not something I can or should fix: the row lands when the audit is recorded, and then
  `generate.py` must be run once so `shipped.json` is regenerated before CI is green. I am reporting it
  because a reviewer seeing a red `generate --check` on this branch needs the cause.

## Verdicts

| Area (keys) | Verdict |
|---|---|
| `drawer.*` (12) | Clean — all 12 read; 导航 / 对话 / 问题 / 历史会话 / 项目 / 资料库 / 用量 / 外观 / 设置 / 已连接·此手机 / 离线·正在重试 / 正在连接…… accurate and idiomatic |
| `link.*` (3) | Findings — F01 (minor, both `since_*`), F04 (nit, `reconnecting`) |
| `list.*` (21) | Finding — F02 (minor, `empty_hint`); the other 20 clean |
| `snackbar.*` (1) | Nit — F05 |
| `row.*` (10) | Clean — 置顶 / 取消置顶 / 标为已读 / 标签 / 无 / 更多 / 打开 / 正在保存…… / `{name}的操作`; the actions label keeps `{name}` and reads naturally with a Latin name |
| `appearance.*` (5) | Clean — 主题 / 宽屏视图 / 按标签分组对话 / 开启 / 关闭 all fit and read naturally |
| `session.*` (25) | Nits — F03 (`loading`), F06 (`not_answering`, `degraded`); **all safety copy clean** (approve pill + label, ended banner, resume, resuming, degraded) |
| `composer.*` (4) | Clean — 消息 / 工作目录：{name} / 模型：{label} / 费用与上下文：{label} |
| `spend.*` (1) | Clean |
| `model.*` (1) | Clean — 推理强度 matches the repo term |

**Overall: PASS** with **2 minor + 4 nit** findings, **0 blocker, 0 major**. 83/83 keys read.

PASS here means: placeholder/ICU integrity, CLDR plural handling, width budgets, register and safety
wording all hold; the two minors are an idiom calque (F01) and a house-style spacing rule (F02), and
neither changes meaning. No finding blocks the namespace.

## Limits of this audit

- I did not render the app; the width statements are computed cells, not screenshots, and the rendered
  evidence is expected separately.
- I am not a native speaker sign-off: this is an independent, evidence-backed audit, and F01/F03/F04 are
  judgement calls whose remedy the implementer may answer rather than apply.
- `generate.py --check` red is explained above and is a wave-state prerequisite, not a translation finding.
- I did not read the translator's report, any provenance sidecar for this wave, or the ledger row for
  `relay.chrome`, to keep this pass independent.

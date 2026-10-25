# Remediation round — `relay.chrome` · zh-CN · unified wave remediation

- **Locale:** `zh-CN` (Simplified Chinese) — one locale, one namespace, 83 keys.
- **Implementer:** `translation-auditor` role session `5217c4e5f8e3`, model `radient/auto` as the
  harness reported it for this session. No fallback was selected or used; no role, default or tool
  set was changed for this work.
- **Worktree / base:** `~/local-operator/.worktrees/relay-chrome-i18n`, branch
  `i18n/relay-chrome-wave`, base **`535cea359825411741069cbffcee42e2927df9d3`**. No commit made.
- **Date:** 2026-10-25
- **Inputs read:** first audit `chrome25-zh-CN-a1`
  (`~/.local-operator/sessions/6d6acecadd03/scratchpad/report.md`, PASS, 2 minor + 4 nit);
  second independent safety audit `chrome25-safety-b1`
  (`~/.local-operator/sessions/acd16c49df72/scratchpad/chrome25-safety-b1/report.md`, zh-CN **PASS
  17/17, 0 findings**); plus the frozen English source + context sidecar, `i18n/glossary.md`,
  `i18n/style-guides/zh-CN.md` and the 83 existing zh-CN catalogues for corpus evidence.

## Hashes

| Artefact | SHA256 |
|---|---|
| English source `relay.chrome.json` (unchanged) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context `relay.chrome.context.json` (unchanged) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| Target **before** this round (snapshot kept at `zh-CN-relay.chrome.candidate-before-remediation.json`) | `ca2c2df9bf514ef301af0feb86062a96e17f5a2e40d2f8a1febed26f5746646e` |
| Target **after** this round | `ba4215e753320387d61bc9d9661eb2a63420473436e492f6da657ac8595c1855` |

The candidate was copied to scratch **before** the first edit, so the delta below is exact (a
key-by-key diff against that snapshot, not a recollection).

## Findings — disposition

| ID | Sev. | Keys | Disposition |
|---|---|---|---|
| `chrome25-zh-CN-a1-F01` | minor | `link.since_seconds`, `link.since_minutes` | **Fixed** |
| `chrome25-zh-CN-a1-F02` | minor | `list.empty_hint` | **Fixed** |
| `chrome25-zh-CN-a1-F03` | nit | `session.loading` | **Fixed** |
| `chrome25-zh-CN-a1-F04` | nit | `link.reconnecting` | **Fixed** |
| `chrome25-zh-CN-a1-F05` | nit | `snackbar.dismiss` | **Rejected (no text change)** — exception recorded |
| `chrome25-zh-CN-a1-F06` | nit | `session.subtitle.not_answering`, `session.degraded` | **Rejected** (no defect here); repo-wide drift deferred to glossary owner |

### F01 — `上次同步于 # 秒前` → `上次同步：# 秒前` — FIXED

Verified the claim against Mandarin usage and the locale's own punctuation rule before accepting.
`于` marks a **point in time** (「同步于 14:30」), not an elapsed offset, so 「上次同步于 3 秒前」 is a
word-for-word calque. The full-width colon is the locale's own prose mark
(`i18n/style-guides/zh-CN.md` §"Punctuation & typography": `，。：；！？、；“”（）`), and the
number→unit spacing (`3 秒前`) follows this locale's duration precedent, not an English import —
`relay.asks.duration.minutes` is `{count, number} 分钟`, `hours` `{count, number} 小时`.
The alternative the auditor also offered (「重连中 · # 秒前同步」) is idiomatic but inverts the
English subject order; the colon form keeps it and reads as a status line. One line, same width
class (25–28 cells of a 44-cell budget). **Accepted.**

### F02 — `用 Local Operator 开始` → `用Local Operator开始` — FIXED

The style guide is explicit: *"No space inserted between Chinese text and Latin/digit runs (width
discipline; matches mainland software convention)"*. Corpus check, run here over all 83 zh-CN
catalogues: **three** prose occurrences of the product name sit flush against Chinese
(`加载Local Operator扩展`, `在Local Operator扩展弹窗中输入…`, `在对应的Local Operator扩展弹窗中输入…`
in `cli.browser.json`); the only spaced occurrence anywhere was my own `empty_hint` (the one hit in
`cli.global_help.json` is line-leading indentation in a help header, `\n    Local Operator - …`,
not a Chinese↔Latin space). So the finding's evidence holds and this was the sole outlier.
No meaning change; the button word 「新建」 and the sidecar requirement that it be used verbatim are
untouched. **Accepted.**

### F03 — `正在读取对话……` → `正在加载对话……` — FIXED

Corpus counts measured here: English "Loading …" is rendered `正在加载…` in **six** strings
(`relay.session` ×2, `relay.session.tags`, `relay.shell.usage`, `tui.hub_links.state`,
`tui.review`), while `读取` is this locale's word for *reading* (`relay.asks.sheet.loading`
"reading asks…" → 正在读取提问……, `tui.library.loading` "Reading the library…" → 正在读取资料库……,
`tui.viewer.loading`, `tui.import.*`). The English key here is "Loading conversation…", so the
one-English-term/one-local-term rule points at `加载`. `正在加载对话……` is equally natural, is still
an in-progress state (not a completion claim), and costs 14 cells of a 28-cell budget. **Accepted.**

### F04 — `正在重连……` → `重连中……` — FIXED

Accepted for the underlying issue the auditor names: the **same visual element** (the app-bar
subtitle) changes its state wording mid-state. A user watches `正在重连……` become
`重连中 · 上次同步：…` as the outage persists. With F01 applied, both strings now lead with
`重连中`, so the state word is stable across the transition. `重连中……` also matches this locale's
existing pattern for the same slot (`tui.status.model.connecting` → `连接中……`).

Rejected alternative, with reason: making all three `正在重连…` (`正在重连……` + `正在重连 · 上次同步：…`)
would keep parallelism with `正在连接……`, but 「正在重连 · 上次同步：3 秒前」 reads as two competing
progress clauses in one line; `重连中 ·` is a status label and reads as one. `正在连接……` is left
alone for the *first-connection* states (it follows `relay.shell.list.connecting` and is a different
state, not this pair).

### F05 — `snackbar.dismiss` = `关闭` — REJECTED (no text change), exception recorded

Verified the drift is real: English "dismiss/dismissed" is 忽略 in 8 question strings
(`relay.ask`, `relay.asks`, `tui.asks`), where it means *ignore this question and send no reply*.
On a snackbar that sense is actively wrong — there is nothing to answer — so 忽略 would mistranslate
the control. `关闭` is the locale's close verb (`tui.hub_links.hint.close` `esc关闭`,
`tui.hub_links.policy.off` `关闭`, `relay.session.tags` remove-flow copy). The auditor reached the
same conclusion and asked for the exception to be **recorded, not renamed**; that is what this entry
does. I did not edit `i18n/glossary.md` or the style guide: those are shared repo artefacts outside
this locale's file, and this round is restricted to my locale. **Actionable residue:** the glossary
owner should add a "dismiss (a notification/snackbar) → 关闭" exception row beside the question sense.

### F06 — `未响应` (2 keys) — REJECTED as a chrome defect; repo-wide drift recorded

Verified: `无响应` appears in `relay.asks.error.read_timeout` (守护进程无响应) and
`tui.small.info_panel.not_answering` (English "not answering"). My pair is internally consistent
and `未响应` is the more common desktop-Chinese rendering; the auditor explicitly framed this as
repo-wide drift "for the glossary owner to settle, not a chrome defect to fix unilaterally".
Changing two strings here would fragment the repo further, not converge it. **No change;
deferred to the glossary owner** (a one-row decision that then applies to all three surfaces).

## Changed keys (exact — 5)

| Key | Before | After |
|---|---|---|
| `relay.chrome.link.reconnecting` | 正在重连…… | 重连中…… |
| `relay.chrome.link.since_seconds` | `{count, plural, other {重连中 · 上次同步于 # 秒前}}` | `{count, plural, other {重连中 · 上次同步：# 秒前}}` |
| `relay.chrome.link.since_minutes` | `{count, plural, other {重连中 · 上次同步于 # 分钟前}}` | `{count, plural, other {重连中 · 上次同步：# 分钟前}}` |
| `relay.chrome.list.empty_hint` | 点击“新建”开始对话，或在电脑上用 Local Operator 开始 | 点击“新建”开始对话，或在电脑上用Local Operator开始 |
| `relay.chrome.session.loading` | 正在读取对话…… | 正在加载对话…… |

Diff against the pre-round snapshot: **exactly these 5 keys**, no other value moved, no key added or
removed, 83 keys, order identical to `en`.

## Safety-neighbour review of the changed keys

The brief's safety concerns were re-checked **on the changed strings**, not assumed from the old ones:

- **last-sync must mean freshness, not reconnect duration** (the `ur` MAJ-01/02 defect class):
  both `since_*` strings still say 「上次同步」 — "last synced" — explicitly, and `#` still renders
  the seconds/minutes *since that sync*. The reconnect sense lives only in 「重连中」, before the
  middle dot. No reading now makes the number the length of the outage.
- **`sends will wait` was not touched** (`session.degraded` unchanged): 「未响应——消息将等待发送」
  keeps the wait sense and claims no queue, delivery or success.
- **Accessible names / button words** untouched: `approve_label` still starts with the pill word
  批准; `ended_hint` still uses the Resume button's exact word 恢复.
- **Busy vs saved tense** untouched and honest: `list.starting` 正在启动……, `row.saving` 正在保存……,
  `session.resuming` 正在恢复……; the two changed `loading`/`reconnecting` strings are in-progress
  states with no completion claim.
- **No provenance fields** exist in this catalogue (plain key→value), so none could drift.

## Validation (actual output)

**Structure, ICU and plural** — repo runtime, `zh-CN`:
```
args problems: [] | select branches equal: True ['online', 'retrying', 'other']
83 keys, order identical to en: True
plural categories: every plural uses only 'other'; each branch renders its count
canonical json.dumps(indent=2, ensure_ascii=False) + "\n": True | LF only: True
```

**Width** — product measure `rich.cells.cell_len`, rendered branches, vs sidecar `maxCells`:
```
maxCells keys: 78 | over budget: []
widest: 28/44 link.since_minutes · 26/44 link.since_seconds · 25/32 session.ended
zero-headroom (== budget): []
changed: since_seconds 25/44 · since_minutes 27/44 · reconnecting 8/32 · loading 14/28
         empty_hint 50 cells (no maxCells declared; wraps to two lines per sidecar)
```
Correction to my first report, measured here: under the product's own `cell_len` the three
12-cell busy labels (`list.starting`, `row.saving`, `session.resuming`) are **10/12**, not at
budget. My earlier "zero headroom" note counted U+2026 at two cells each; the renderer counts one.

**Repository gates** — `PYTHONPATH=. .venv/bin/python scripts/i18n/check.py`:
```
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)
EXIT=0
```
(34 preceding `advisory:` ratchet lines are outside the enforced scope and unrelated to catalogues;
no problem line names `zh-CN`.)

`scripts/i18n/generate.py --check` is **red for a wave-state reason, not this file**: it reports
`changed: local_operator/i18n/data/shipped.json`. Derived cause, checked here: `i18n/ledger.json`
has **326 entries, 0 for `relay.chrome` in any locale**, while every locale now carries the
namespace; `_shipped_locales()` requires a passed ledger row per carried namespace, so a
regeneration right now would shrink the committed 8-locale `shipped.json` to `["en"]`. The row
lands when the audit is recorded; `generate.py` must then run once. Not regenerated here — generated
artefacts are out of scope for this round.

## Residual / actionable items (only these)

1. **Glossary owner:** record the snackbar "dismiss" exception (F05) — English "dismiss" → 忽略 for
   questions, 关闭 for a notification/snackbar. One row in `i18n/glossary.md`; no string changes.
2. **Glossary owner:** settle "not answering" repo-wide (F06) — 未响应 (chrome) vs 无响应
   (`relay.asks.error.read_timeout`, `tui.small.info_panel.not_answering`). Whichever wins, apply to
   all three places; this locale's two chrome keys are consistent and can stay as-is or follow.
3. **Wave prerequisite (not a finding):** `relay.chrome` ledger rows + one `generate.py` run so
   `shipped.json` is regenerated before CI can be green on this branch.

No other residual issues. Nothing in this round requires a whole-namespace restart: the terminal
auditor can review the 5-key delta plus its safety neighbours (`drawer.status`, `session.degraded`,
`session.ended`/`ended_hint`, `tray.approve*`), which are byte-identical to the version the second
safety audit passed.

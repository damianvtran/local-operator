# Translation audit — `relay.chrome`, Russian (`ru`)

- **Audit ID:** `chrome25-ru-a1`
- **Locale / namespace:** `ru` · `relay.chrome`
- **Repository (read-only):** `~/local-operator/.worktrees/relay-chrome-i18n`
- **Base commit:** `535cea359825411741069cbffcee42e2927df9d3` — *feat(i18n): freeze relay.chrome English copy for the borderless relay redesign* (2026-10-25T13:44:22-04:00)
- **Auditor:** agent `translation-auditor`, session `06d8dd9ba8cf`, parent session `44f93fd45b09`, job `150fefb9adb0`, label “Audit chrome ru”.
  Model: **`radient/auto`**, as reported by the harness for this session. The resolved provider model id is not exposed to this subagent, so it is not claimed — no guessed identity or model. Date: 2026-10-25T18:00Z.
- **Independence:** the translator's report, ledger entries and any other auditor's notes for this locale were **not** read. This audit compares only the English source, the en context sidecar, the Russian target bytes, `i18n/glossary.md`, `i18n/style-guides/ru.md` and the existing shipped `ru` catalogues. The repo was not edited; the only writes were to this session's scratchpad.

## Files and hashes (measured with `sha256sum`, re-verified in `audit_checks.py`)

| File | Bytes | SHA256 |
|---|---|---|
| `local_operator/i18n/catalogues/en/relay.chrome.json` | 4914 | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `local_operator/i18n/catalogues/en/relay.chrome.context.json` | 34149 | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `local_operator/i18n/catalogues/ru/relay.chrome.json` (target) | 6668 | `27709627e9deb166ae6954a142400f482876b940681819ab6b49234f88d738fd` |

Target bytes: no BOM, ends with a newline, 2-space indent, 83 keys, no duplicate keys, key order identical to `en`.
The `en` source and context hashes match the values this audit was commissioned with; the target hash is the file actually audited.

**Keys read: 83 of 83** — every value in both catalogues, plus every per-key `role`/`tone`/`maxCells`/`note` in the sidecar.

## Method and executed checks (actual output, not a summary)

All checks ran with `~/local-operator/.venv/bin/python` (3.12) against the worktree. Script: `scratchpad/audit_checks.py`; full transcript: `scratchpad/checks.out`.

1. **Repo catalogue validation** — `python scripts/i18n/check.py --catalogues-only` → `check.py: catalogue checks: 0 problem(s)` (exit 0). Re-run in-process by `audit_checks.py` against the real tree → `problems: 0`.
2. **Key parity** — `en=83 ru=83 missing=[] extra=[] order_same=True`.
2b. **Negative control (scratch copy only, shared files untouched)** — injected three defects into a copy of `catalogues/` and re-ran the same checker:
   ```
   baseline problems on copy: 0
   injected problems: 2
     - ru/relay.chrome: relay.chrome.list.filter.unread: arguments (('count','simple'),) != en's (('count','number'),): argument 'count' changed kind 'number' -> 'simple'
     - ru/relay.chrome: relay.chrome.row.actions: arguments (('имя','simple'),) != en's (('name','simple'),): argument 1 is 'имя', en's is 'name' (renamed or reordered)
   ```
   This proves the checker reads this target file and catches renamed placeholders and changed argument kinds. It also shows a **checker gap**: a `plural` that drops `few`/`many` in `ru` (leaving `one`/`other`) is *not* flagged — the completeness gate applies only to `number → plural` upgrades. The audited file is complete, so this is a CI observation, not a finding against it; the completeness claim below was verified independently with `runtime.plural_arguments`.
3. **ICU argument contract (name, order, kind) for all 83 keys** — `mismatches: 0`. The target preserves `{state, select, …}`, `{count, number}`, `{count, plural, …}`, `{done, number}`, `{total, number}`, `{running, number}`, `{failed, plural, …}`, `{elapsed}`, `{name}`, `{label}` with the same names, order and kinds.
4. **Plural completeness against ru CLDR (`one/few/many/other`), value rendering, emptiness** — all four plural messages CLEAN:
   ```
   link.since_seconds: arg=count  branches=['one','few','many','other'] -> CLEAN
   link.since_minutes: arg=count  branches=['one','few','many','other'] -> CLEAN
   session.tray.questions: arg=count branches=['one','few','many','other'] -> CLEAN
   session.tray.agents_failed: arg=failed branches=['one','few','many','other'] -> CLEAN
   ```
   Every branch renders its own number (`#`) and no branch is empty.
5. **ICU rendering, real formatter (`local_operator.i18n.runtime`)** — representative branches:
   ```
   since_minutes 1 → Переподключение · обновлено 1 минуту назад
   since_minutes 2 → … 2 минуты назад   5 → … 5 минут назад
   since_minutes 21 → … 21 минуту назад  22 → … 22 минуты назад   25 → … 25 минут назад
   since_minutes 1.5 → обновлено 1,5 минуты назад
   questions 1 → 1 вопрос   2 → 2 вопроса   5 → 5 вопросов   21 → 21 вопрос
   agents_failed 1 → Агенты 2/3 · 1 сбой   2 → … 2 сбоя   5 → … 5 сбоев
   drawer.status online → Подключено · этот телефон
   drawer.status retrying → Не в сети · повторяем попытку
   drawer.status other → Подключение…
   ```
   `{count, number}` grouping is formatter-rendered: `Ждут вас · 1 234` (NBSP), not hand-written.
6. **Cross-key references** — `approve_label starts with approve pill: True`; `FAB word appears in empty_hint: True` (`«Новая»` = `list.new`); `Resume word appears in ended_hint: True` (`«Продолжить»` = `session.resume`).
7. **Punctuation / typography scan** — `issues: none`; ASCII `...` absent (U+2026 everywhere), no straight quotes, no double spaces, no trailing periods; `·` present exactly where `en` has it; `capitalised 'Вы': []` (lowercase «вы/вас/вашей» throughout — matches the style guide's **вы**, not letter-writing «Вы»).
8. **Capitalization scan** — `lowercase-initial: [('relay.chrome.appearance.on','вкл'), ('relay.chrome.appearance.off', 'выкл')]` — the only two; both match the shipped lowercase short pair in `tui.tags.on`/`.off` and `tui.schedule.value_on`/`.off` (nit F1).
9. **Width, measured on rendered strings (not ICU source) at representative params** — no key exceeds `maxCells`; two sit exactly at budget:
   ```
   link.since_seconds  en=41 ru=40 max=44     link.since_minutes en=41 ru=42 max=44
   tray.agents_failed  en=23 ru=22 max=28     composer.spend_label en=28 ru=31 max=40
   list.filter.needs_you en=17 ru=16 max=16  AT-BUDGET (count=1234)
   list.filter.unread en=11 ru=16 max=16     AT-BUDGET (count=12)
   tray.questions en=12 ru=11 max=16          tray.tasks en=11 ru=12 max=16
   ```
   Materially longer than English but inside budget (≥ +4 cells and ≥ 1.3×, rendered): `drawer.usage` 5→13, `drawer.status` 18→29, `section.pinned` 6→12, `row.pin` 10→16, `row.swipe_pin` 3→9, `row.tags_none` 4→9, `subtitle.ended` 5→9, `session.resume` 6→10, `ended_hint` 47→75 (no budget; wraps by design). All are ordinary Russian expansion; none needs shortening beyond the two chip signals above.

## Verdicts per section (namespace `relay.chrome`)

| Section | Keys | Verdict |
|---|---|---|
| `drawer.*` | 12 | **PASS** — clean |
| `link.*` | 3 | **PASS** — clean (`since_seconds`'s four branches are byte-identical, which is correct because «сек» is invariable; redundant but harmless) |
| `list.*` | 21 | **PASS** — 2 nit (F2 `mark_all`, F5 `unread`) |
| `snackbar.dismiss` | 1 | **PASS** — clean |
| `row.*` | 10 | **PASS** — 1 minor (F4 `actions`), 1 nit (F2 `mark_read`) |
| `appearance.*` | 5 | **PASS** — 1 nit (F1, `on`/`off` casing vs legacy short pair) |
| `session.*` | 25 | **PASS** — 1 minor (F3, `resume`/`resuming` term) |
| `composer.*` | 4 | **PASS** — clean |
| `spend.title` | 1 | **PASS** — clean |
| `model.effort` | 1 | **PASS** — clean |

**Overall: PASS — 0 blocker, 0 major, 2 minor, 3 nit, on 83/83 keys read.** Nothing here blocks landing: the two minors (F3 term drift, F4 case government) are worth one batched remediation commit or an explicit “deferred — reason” reply per the team's round rules; the nits need only an answer.

## Findings

### F1 — `relay.chrome.appearance.on` / `relay.chrome.appearance.off` — **nit**
- Current: `"вкл"` / `"выкл"`.
- Proposed: no change required; if the chrome's O1 rule is meant to override legacy casing, `"Вкл"` / `"Выкл"` (3 and 4 cells, inside the 6-cell budget) — and either way record the short pair in `i18n/style-guides/ru.md` next to «мин»/«сек».
- Reason: these are the only two of 83 values that begin lowercase and the sidecar `tone` says “Sentence case (the redesign's O1)”, but the lowercase pair matches the locale's established short state pair — `tui.tags.on`/`.off` and `tui.schedule.value_on`/`.off` already ship `"вкл"` / `"выкл"` — so it is consistent with existing ru casing rather than a defect; the abbreviation itself is forced by `maxCells: 6` («включено» / «отключено» are 8 and 10 cells).

### F2 — `relay.chrome.row.mark_read` (and `relay.chrome.list.mark_all`) — **nit**
- Current: `"Прочитать"` / `"Прочитать все"`.
- Proposed: keep both as they are — the pair is consistent and the 20-cell `mark_all` budget makes a literal marking phrase impossible («Отметить прочитанными» is 21 cells). If the wave wants the literal sense anyway, `"Отметить как прочитанное"` (24 cells ≤ 28) on the row plus a shortened header, accepting that the two verbs then differ.
- Reason: the English marks a conversation as seen without opening it, and the other six translated locales all keep the marking verb (es «Marcar como leído», fr «Marquer comme lu», vi «Đánh dấu đã đọc», zh-CN 「标为已读», hi, ur); «Прочитать» is nevertheless an established Russian messaging shorthand for this same control, so this is a note that ru is the only locale dropping the marking sense — not a defect worth churning the pair for.

### F3 — `relay.chrome.session.resume` / `.resuming` (and the `ended_hint` reference to it) — **minor**
- Current: `"Продолжить"` / `"Продолжаем…"` (hint: «Отправка сообщения или «Продолжить»…»).
- Proposed: **keep** `"Продолжить"` and record it in `i18n/style-guides/ru.md` as the ru UI term for “Resume”; or align the pair to `"Возобновить"` / `"Возобновляем…"` — the busy form is 13 cells against a 12-cell budget, so that variant needs a rendered check.
- Reason: the same English action is rendered «возобновить» in `wire.slash.resume`, `wire.slash.archive` and `tui.notices.model_info_incomplete`; one local term per English term, and whichever form is chosen must be recorded so the next wave does not flip it again. Not a safety error: the hint's promise (reopen in the home folder) survives intact.

### F4 — `relay.chrome.row.actions` — **minor**
- Current: `"Действия для {name}"`.
- Proposed: `"Действия: {name}"` (colon construction, case-free, 40-cell budget unaffected) or `"Действия для беседы «{name}»"`.
- Reason: «для» governs the genitive, but `{name}` is inserted verbatim in the nominative, so a Russian conversation title reads “Действия для Сборка релиза”; this is the screen-reader name of the per-conversation actions control, and the locale already uses the colon pattern for interpolated values (`composer.folder_label`, `relay.approvals.policy.session_override`). Precedent for «для {name}» exists in `wire.settings` (`local_providers.models.label`), which is why this is minor rather than major.

### F5 — `relay.chrome.list.filter.unread` — **nit (width signal)**
- Current: `"Непрочитано · {count, number}"`.
- Proposed: keep the text and run the rendered check at a 3-digit count (16 cells at count=12 = the 16-cell budget); if the chip row has room, `"Непрочитанные"` matches the locale's adjective in `wire.slash.notifications` («Непрочитанные завершённые сессии»).
- Reason: the adverbial short form is the only place this locale names the state without the adjective; it is understandable and width-efficient, but the budget leaves no headroom. Per the brief, this is a signal for rendered confirmation, not a semantic defect.

## Safety-related wording — checked explicitly

| Key | Target | Verdict |
|---|---|---|
| `session.tray.approve` | «Одобрить» | correct glossary term (approvals → «одобрения», verb «одобрить»); infinitive action label; 8 cells ≤ 12 |
| `session.tray.approve_label` | «Одобрить — вас ждёт запрос» | **satisfies the voice-control rule**: starts with the visible pill word byte-for-byte (`startswith` = True), then “ — ”; says a request is waiting; 26 cells ≤ 48 |
| `session.subtitle.ended` / `session.ended` | «Завершена» / «Завершена · история сохранена» | “ended, history kept” preserved; feminine agreement with the implied «сессия» is consistent across both keys |
| `session.ended_hint` | «Отправка сообщения или «Продолжить» снова откроет её в вашей домашней папке» | both reopen paths preserved (a send **or** the Resume button), and the home-folder promise is explicit («домашней папке», not the project folder); button word matches `session.resume` verbatim |
| `session.resume` / `.resuming` | «Продолжить» / «Продолжаем…» | action is correct and does not imply a fresh session; terminology drift only (F3) |
| `session.degraded` / `.subtitle.not_answering` | «Не отвечает — сообщения подождут» / «Не отвечает» | no false “delivered” claim; “sends will wait” is preserved («сообщения подождут») |
| `drawer.status` (retrying) | «Не в сети · повторяем попытку» | does not claim the link is up; offline state is explicit |
| `link.since_seconds` / `.since_minutes` | «Переподключение · обновлено # сек/минуту назад» | states the age of the data honestly; «обновлено» for “last synced” is a width-driven simplification (`синхронизировано` would push the line past 44 cells) and does not overclaim |

## Other observations (no action requested)

- **Conversation = two words across the catalogue set (not a defect of this file).** A census of `ru` files finds «беседа» 65× vs «разговор» 21×; `relay.chrome` uses «беседа» consistently (8 occurrences), as do `tui.*` (except `tui.asks`) and `wire.settings`. But the other relay SPA catalogue, `relay.asks.json`, uses «разговор» 4/4 — as do `tui.notices.json` (7/7) and `wire.errors.desktop.json` (7/7). So the product surface currently speaks two words for “conversation”; this file is on the majority side, and the drift belongs to `relay.asks` (same SPA family) rather than to the audited target.
- **Busy/state person:** `list.marking` «Отмечаем…», `session.resuming` «Продолжаем…» and `drawer.status` «повторяем попытку» are first-person plural, while `list.starting` «Запуск…», `row.saving` «Сохранение…» and `session.connecting` «Подключение…» are impersonal nouns. Both are idiomatic Russian; the ru guide only says to *prefer* impersonal where English allows, so no change is proposed — flagging it so the next wave picks one deliberately.
- **`row.tags_none`** «Без тегов» renders “None” as “no tags”; it is clearer than a bare «Нет» beside the «Теги» row and is consistent with the trailing-value role. `appearance.wide_view` «Широкий вид» matches the literal “wide view” chosen by every other locale (es «Vista ancha», fr «Vue large»), so its slightly calque-ish ring is a family-wide choice, not a ru defect. `model.effort` «Усилие рассуждений» matches `wire.settings.model.description`/`model_effort.help` exactly.
- **Dropped subject in `session.ended` is width-driven:** «Завершена · история сохранена» (29 cells) omits «Сессия»; the full «Сессия завершена · история сохранена» is 36 cells against a 32-cell budget. The banner sits inside the conversation, so the implied subject is unambiguous; accepted, no action.
- **`list.filter.needs_you` at count 1:** «Ждут вас · 1» keeps the plural verb because the label names the conversations, and the count is a separate `{count, number}` argument that cannot inflect it. English says “Needs you · 1”; the Russian reads as “[conversations] waiting for you · 1”, which is acceptable for a category chip. Noted, no action.
- **RTL / Latin isolation:** `ru` is LTR, so no bidi isolation is required and none is added; the only Latin runs are the product name “Local Operator” (kept per glossary §1) and ICU argument names, all of which resolve at render time. Direction-sensitive punctuation (…, ·, —, «») is placed conventionally for LTR.

## Limitations

- This is the catalogue/locale audit only. **Rendered app evidence is a separate pass** — no screenshots, no viewport measurement, and no claim that the 16-cell chips or the 44-cell reconnect line actually fit on a 360 px phone. The width table above is a signal for that pass.
- The repo validator does not enforce locale plural-category completeness for messages that are already `plural` in `en` (demonstrated in the negative control); this audit added that guarantee with `runtime.plural_arguments`, but CI alone would not catch a future regression of the same shape.

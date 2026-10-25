# Remediation — `relay.chrome` → ru (single combined round)

- **Round:** one consolidated response to the ru language audit plus the cross-locale safety audit. One batched edit pass; no cascading rounds.
- **Locale / namespace:** `ru` · `relay.chrome` (83 keys).
- **Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n`, base `535cea359825411741069cbffcee42e2927df9d3`, branch `i18n/relay-chrome-wave`. No commit (per wave instruction); the target file stays untracked.
- **Translator session / model:** session `635b55fcabb6`, harness-reported model **`radient/auto`**; no model switch, no fallback selected or used in this session. The resolved provider model id is not exposed in-session, so none is claimed.
- **Audits answered:** `chrome25-ru-a1` (locale audit: 0 blocker, 0 major, **2 minor — F3, F4**, 3 nit — F1, F2, F5) and `chrome25-safety-b1` (second independent safety audit: **ru PASS, 0 findings**).

## Inputs and target bytes

| File | SHA-256 | Note |
|---|---|---|
| `en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` | frozen, unchanged |
| `en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` | frozen, unchanged |
| `ru/relay.chrome.json` **before** | `27709627e9deb166ae6954a142400f482876b940681819ab6b49234f88d738fd` | audited candidate, snapshotted at `candidate-before.json` before editing |
| `ru/relay.chrome.json` **after** | `fb17616cb7875507eb4b16853cb7b229a1176f46349d9c942d9cc8a427e6c97c` | 6670 bytes, 83 keys, 2-space indent, UTF-8, no BOM, final newline |

Key order and key set are byte-identical to `en` after the pass; no key added, removed or renamed.

## Dispositions

| ID | Source | Severity | Keys | Disposition |
|---|---|---|---|---|
| F3 | `chrome25-ru-a1` | minor | `session.resume`, `session.resuming`, `session.ended_hint` | **FIXED** |
| F4 | `chrome25-ru-a1` | minor | `row.actions` | **FIXED** |
| F1 | `chrome25-ru-a1` | nit | `appearance.on`, `appearance.off` | **REJECTED** (keep) — guide recording deferred to programme owner |
| F2 | `chrome25-ru-a1` | nit | `row.mark_read`, `list.mark_all` | **REJECTED** (keep) |
| F5 | `chrome25-ru-a1` | nit (width signal) | `list.filter.unread` | **REJECTED** (keep) — residual rendered-fit signal noted |
| — | `chrome25-safety-b1` | — | ru scope | **NO ACTION** — ru PASS, 0 findings |

"REJECTED" here means the audit's **proposed alternative** was rejected and the current copy kept; the observation itself was verified against the corpus and answered with evidence, not dismissed.

## F3 — Resume term drift — FIXED

- Before → after:
  - `session.resume` «Продолжить» → **«Возобновить»**
  - `session.resuming` «Продолжаем…» → **«Возобновляем…»**
  - `session.ended_hint` «…или «Продолжить»…» → **«Отправка сообщения или «Возобновить» снова откроет её в вашей домашней папке»**
- Accepted the drift claim after checking the shipped corpus, not just the auditor's wording:
  - English *resume* is already **«возобновить»** in this locale: `ru/wire.slash.resume` = «Выбрать прошлую беседу или возобновить её (id)» (en: “Pick a past conversation to resume, or resume one (id)”), and `ru/wire.slash.archive` = «…по id всё ещё можно возобновить».
  - «продолж-» is this catalogue set's verb for English *continue / keep*: `ru/tui.notices.budget_exhausted_continue` «просим продолжить работу», `ru/relay.asks.gate.keep_asking` «продолжать спрашивать (ask)». Keeping «Продолжить» on the Resume button would have merged two distinct English verbs.
  - The style guide's action-label examples («Сохранить», «Продолжить», «Отмена») are infinitive-form examples, not a resume↔term ruling, so they do not override the corpus invariant.
- Hint reference updated in the same pass so the quoted button word stays byte-identical to `session.resume` (verified: `session.resume in ended_hint` = True).
- **Width consequence, flagged rather than hidden:** the busy form «Возобновляем…» renders at **13 cells against `maxCells: 12`** (+1). U+2026 is required product typography and dropping it or abbreviating the stem would be worse copy, so the accurate idiom is retained per the round brief; this is a source-side budget signal, not a measured clip. `session.resume` itself is 11 cells ≤ 12. `session.ended_hint` has no `maxCells` (wraps by design; 76 cells).

## F4 — `row.actions` case government — FIXED

- Before → after: «Действия для {name}» → **«Действия: {name}»**.
- Accepted: «для» governs the genitive while `{name}` is substituted verbatim in the nominative («Действия для Сборка релиза» is ungrammatical). The colon construction is case-free and is already this surface's pattern for interpolated accessible names — `composer.folder_label` «Рабочая папка: {name}», `composer.model_label` «Модель: {label}», `composer.spend_label` «Расходы и контекст: {label}».
- Rejected the alternative «Действия для беседы «{name}»»: it wraps a verbatim name in «» (double quoting if the name carries quotes) and adds 8 cells for no information the sheet context does not already carry; `maxCells` 40 is unaffected either way.
- Rendered: `Действия: Сборка релиза` (23 cells ≤ 40).
- ASCII `:` with no space before it — matches ru punctuation rules.

## F1 — `appearance.on` / `.off` casing — REJECTED (no change)

- Kept **«вкл» / «выкл»** (3 and 4 cells ≤ 6).
- Evidence: this is the locale's shipped short state pair — `ru/tui.tags.on`/`.off` = «вкл»/«выкл», `ru/tui.schedule.value_on`/`.off` = «вкл»/«выкл», and the bool rendering is lowercase throughout (`ru/wire.settings.bool.on` = «включено»). The 6-cell cap forces the abbreviation: «включено» is 8 cells, «выключено» 10 — both over budget. The sidecar's tone clause is “Sentence case … no trailing period **on a label**”; these two values are trailing state values, not labels, and the first audit itself concluded “no change required”.
- **Deferred to the i18n programme owner (not this wave):** recording «вкл»/«выкл» next to «мин»/«сек» in `i18n/style-guides/ru.md` is a binding-reference edit outside a translator's write scope (locale catalogues only). Same for recording the resume→«возобновить» call. Both are recorded here for the terminal auditor and the guide owner instead.

## F2 — `row.mark_read` / `list.mark_all` — REJECTED (no change)

- Kept **«Прочитать» / «Прочитать все»**.
- Evidence: both independent audits classify this as established Russian messenger idiom for mark-as-read (Telegram ru «Прочитать всё»), not a defect. The explicit marking phrase does not fit the 20-cell header budget: «Отметить прочитанными» = 21 cells (measured, 1 over), «Отметить все прочитанными» = 25 cells; the row-level «Отметить как прочитанное» (24 ≤ 28) would break the verb pair for no user benefit.
- Honesty check: this control changes seen-state only; the label promises no reading session and no delivery, so no false state claim.

## F5 — `list.filter.unread` — REJECTED (keep), residual width signal

- Kept **«Непрочитано · {count, number}»**.
- Measured, rendered with the real formatter: 15 cells at count=3, **16/16 at count=12**, 17 cells at a 3-digit count. The proposed adjective «Непрочитанные» is 13 cells, i.e. **17 cells at a one-digit count and 19 at three digits** — over the 16-cell chip budget at every count, not only at 3 digits. The 16-cell budget therefore does not make the adjective available; the short state form is the only form that fits the common case.
- Residual: a 3-digit unread count renders 1 cell over budget; the sidecar states the chip row scrolls sideways, so this is a rendered-fit signal for the QA/design pass, not a semantic defect.

## Second safety audit (`chrome25-safety-b1`) — NO ACTION

ru was already PASS with 0 findings, and the remediation must not disturb the labels that audit checked. Re-verified after the edit (these keys were not touched):

- `session.degraded` «Не отвечает — сообщения подождут» keeps “sends will wait” — no queued/delivered guarantee added.
- `session.tray.approve` «Одобрить» / `approve_label` «Одобрить — вас ждёт запрос»: accessible name still starts with the visible pill word byte-for-byte, and says a request is waiting (no automatic approval).
- `session.ended` «Завершена · история сохранена» — ended + history kept as state; `session.ended_hint` still names the home folder and both reopen paths.
- `list.filter.needs_you` «Ждут вас · {count, number}» — operator action still implied.

## Validation executed after the pass

1. **Key parity/order:** `en=83 ru=83`, order identical to `en`, no duplicates, no extra/missing keys.
2. **ICU argument contract (name + kind, all 83 keys):** `arg parity: True`. `{state, select, …}` keeps `online`/`retrying`/`other`; `{count, number}`, `{done, number}`, `{total, number}`, `{running, number}`, `{elapsed}`, `{name}`, `{label}` unchanged.
3. **Plural completeness (ru CLDR):** all four plural keys carry `one/few/many/other` and every branch renders its own number —
   `link.since_seconds`, `link.since_minutes`, `session.tray.questions` (`count`), `session.tray.agents_failed` (`failed`) → `branches=['one','few','many','other'] all_render_value=True`.
4. **Cross-key references:** `approve_label` starts with `tray.approve` = True; `list.new` word («Новая») inside `list.empty_hint` = True; `session.resume` word («Возобновить») inside `session.ended_hint` = True.
5. **Typography:** no ASCII `...`, no straight quotes, no double spaces, no trailing spaces, no trailing periods; U+2026 × 9 (en 9) and U+00B7 × 18 (en 12 — the four-branch plurals repeat the separator once per branch); lowercase «вы/вас/вашей» only (no letter-writing «Вы»); «вкл»/«выкл» remain the only lowercase-initial values.
6. **Width (rendered, real runtime, representative params):** 0 keys over budget except the flagged `session.resuming` 13/12. Tight: `list.filter.unread` 16/16 at count=12; `link.since_minutes` 43/44 at two-digit minutes; `session.ended` 29/32.
7. **Repo gate:** `cd ~/local-operator/.worktrees/relay-chrome-i18n && PYTHONPATH=. ~/local-operator/.venv/bin/python scripts/i18n/check.py` → exit **0**:

   ```
   check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)
   ```

   Zero lines mention `ru/` or `relay.chrome`. The 34 advisory entries are the tree's pre-existing Python ratchet findings, unrelated to this wave. Full capture: `check.py.out` beside this report.

## Residual, actionable items (only these)

1. **Rendered fit — `session.resuming` «Возобновляем…»:** 13 cells vs `maxCells` 12 in the ended-session banner button. Source-side only; needs the QA/design rendered pass to confirm no clip at 360–390 px. If it clips, the fix is a budget/layout decision, not a worse term.
2. **Rendered fit — `list.filter.unread`:** 17 cells at a 3-digit count (15 at one digit, 16 at two). Chip row scrolls per the sidecar; confirm in the rendered pass.
3. **Programme-owned guide records (not editable by this wave):** add «вкл»/«выкл» as the ru short state pair and «возобновить» as the ru term for Resume (button/command), «продолжить» for Continue, to `i18n/style-guides/ru.md`.

## Not touched

English source/context (hashes re-verified against the frozen values), other locales, ledger, allowance, generators, gates, application code. No commit. Only `local_operator/i18n/catalogues/ru/relay.chrome.json` was written in the worktree; everything else here is session scratch.

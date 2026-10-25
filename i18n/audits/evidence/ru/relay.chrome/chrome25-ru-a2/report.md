# Round-2 terminal delta audit — `relay.chrome`, Russian (`ru`)

- **Audit ID:** `chrome25-ru-a2` (terminal delta verification of `chrome25-ru-a1`)
- **Locale / namespace:** `ru` · `relay.chrome` (83 keys)
- **Repository (read-only):** `~/local-operator/.worktrees/relay-chrome-i18n`
- **Base commit:** `535cea359825411741069cbffcee42e2927df9d3` — branch `i18n/relay-chrome-wave`; target still untracked; no commit.
- **Auditor:** agent `translation-auditor` (independent of the translator). Session `06d8dd9ba8cf`, parent session `44f93fd45b09`, label “Audit chrome ru”; R1 job `150fefb9adb0`, this R2 job `5f38ec2c8ae7` (from the session's `subagent-lane` record). The translator's session `635b55fcabb6` is the author of the artefact under review, not this auditor.
- **Model:** `radient/auto` — the value in this session's runtime header. The resolved provider model id is not exposed to a subagent and is not inferred.
- **Scope:** delta only. Four keys changed after one unified remediation pass; the other **79 keys are byte-identical to the R1-audited target** and retain R1 coverage (proof below). No full namespace re-audit.
- **Independence:** the implementer's on-disk remediation record was read as the artefact under review; the final agent summaries were **not** used (the manager reported their metadata as mangled). The on-disk record reproduces its own hashes (before `27709627…`, after `fb17616c…`), names the translator session `635b55fcabb6` with harness model `radient/auto`, and contains no `/home/user`-style paths. No repo, target, ledger or style-guide edit; all writes are in this session's scratchpad.

## Hashes — recomputed on disk, not copied from prose

| Artefact | Bytes | SHA-256 |
|---|---|---|
| `en/relay.chrome.json` (frozen) | 4914 | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `en/relay.chrome.context.json` (frozen) | 34149 | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `ru/relay.chrome.json` **before** (R1-audited snapshot) | 6668 | `27709627e9deb166ae6954a142400f482876b940681819ab6b49234f88d738fd` |
| `ru/relay.chrome.json` **current / after** | 6670 | **`fb17616cb7875507eb4b16853cb7b229a1176f46349d9c942d9cc8a427e6c97c`** |

- `en` and context hashes are unchanged from the frozen values the wave started with.
- The "before" snapshot's hash equals the target hash audited in `chrome25-ru-a1`, so the delta base is exactly the bytes that were reviewed.
- `candidate-after.json` on disk parses equal to the current worktree file (`current == after: True`); the reported after-hash is reproduced independently.
- UTF-8, no BOM, final newline, 2-space indent, 85 lines before and after; key order identical to `en`; no key added, removed or renamed.

## Exact changed keys (4 of 83)

Raw diff of `candidate-before.json` → current shows exactly four changed lines and no formatting churn:

| Key | Before (R1-audited) | Current | Finding |
|---|---|---|---|
| `relay.chrome.row.actions` | `Действия для {name}` | `Действия: {name}` | L-F4 (fixed) |
| `relay.chrome.session.ended_hint` | `Отправка сообщения или «Продолжить» снова откроет её в вашей домашней папке` | `Отправка сообщения или «Возобновить» снова откроет её в вашей домашней папке` | L-F3 (fixed) |
| `relay.chrome.session.resume` | `Продолжить` | **`Возобновить`** | L-F3 (fixed) |
| `relay.chrome.session.resuming` | `Продолжаем…` | **`Возобновляем…`** | L-F3 (fixed) |

## R1 carry-forward proof

- `changed keys: 4 … unchanged keys: 79 of 83`
- `diff candidate-before.json current` → `4` changed lines, `85` lines in both files, no whitespace/indent/order change.
- Therefore every key not listed above is the exact byte sequence audited in R1, and its R1 verdict (clean or minor/nit) carries forward without re-reading.

## Dispositions of R1 findings

| ID | R1 severity | Disposition | Verified |
|---|---|---|---|
| L-F1 `appearance.on`/`.off` «вкл»/«выкл» | nit | **REJECTED (keep)** — accepted | lowercase pair is the shipped ru short state pair (`tui.tags.on/.off`, `tui.schedule.value_on/.off`); 6-cell cap rules out the full forms (8/10 cells) |
| L-F2 `row.mark_read` / `list.mark_all` «Прочитать / Прочитать все» | nit | **REJECTED (keep)** — accepted | literal marking phrase cannot fit the 20-cell header (`Отметить все прочитанными` 25, `Отметить прочитанными` 21); R1 itself recommended keeping |
| L-F3 `session.resume` / `.resuming` / `ended_hint` | minor | **FIXED** — accepted | see below |
| L-F4 `row.actions` case government | minor | **FIXED** — accepted | see below |
| L-F5 `list.filter.unread` «Непрочитано» | nit (width) | **REJECTED (keep)** — accepted | the suggested adjective `Непрочитанные` is 13 cells → 17 at one-digit, 18 at two-digit, 19 at three-digit counts; the current form is the only one that fits the common case (15/16/17) |

### L-F3 — Resume term — verified fixed

- `session.resume` = `Возобновить` (11 cells ≤ 12); `session.resuming` = `Возобновляем…`; `ended_hint` quotes the button word verbatim.
- Corpus check reproduced: `ru/wire.slash.resume` = «Выбрать прошлую беседу или возобновить её (id)» for en “Pick a past conversation to resume, or resume one (id)”; `ru/wire.slash.archive` = «Скрыть из списков; по id всё ещё можно возобновить»; «продолж-» is the locale's verb for *continue* (`tui.notices.budget_exhausted_continue` «просим продолжить работу», `relay.asks.gate.keep_asking` «продолжать спрашивать»). The drift is gone and the two English verbs are distinct again.
- `session.resume in session.ended_hint` → `True`; no `Продолж` remains anywhere in the file.
- **Residual (rendered fit):** `session.resuming` renders **13 cells against `maxCells: 12`** (+1; U+2026 included). Source-side measurement only — no clip was measured here. Per the round rule this is a rendered-fit signal, not a blocker or major; if the banner button clips at 360–390 px, that is a layout/budget decision, not a worse term.

### L-F4 — `row.actions` — verified fixed

- `Действия: {name}` renders `Действия: Сборка релиза` (23 cells ≤ 40) and `Действия: Weekly review`; ASCII colon with no preceding space; `{name}` is still the same `simple` argument, inserted verbatim (`en=(('name','simple'),) ru=(('name','simple'),)`).
- The colon pattern matches this surface's other interpolated names (`composer.folder_label`, `composer.model_label`, `composer.spend_label`), and removing «для» removes the genitive government that made a nominative title ungrammatical.
- The implementer rejected «Действия для беседы «{name}»» (double quoting, +8 cells); that alternative is no better, so the rejection is accepted.

## Safety / high-visibility delta

The only safety-scope text that changed is `relay.chrome.session.ended_hint` (ended + Resume semantics). Re-checked in its current bytes:

- Both reopen paths are still named: «Отправка сообщения или «Возобновить» снова откроет её…» — a send **or** the Resume button, not a new session.
- The destination is still the home folder: «в вашей домашней папке».
- The quoted button word is byte-identical to `session.resume` (`startswith`/substring check True), so a screen-reader or voice-control user hears the label they can see.
- Untouched safety labels are byte-identical to the R1-audited bytes and therefore retain R1 sign-off: `session.tray.approve` «Одобрить»; `session.tray.approve_label` «Одобрить — вас ждёт запрос» (starts with the pill word); `session.ended` «Завершена · история сохранена»; `session.degraded` «Не отвечает — сообщения подождут»; `session.subtitle.ended` «Завершена»; `list.filter.needs_you` «Ждут вас · {count, number}».
- **Programme item:** the pre-remediation safety audit (`chrome25-safety-b1`, ru PASS) signed the *old* `ended_hint` bytes. A second-auditor **delta** sign-off on the new hint is still owed by that stream; this audit verifies the semantics but does not impersonate that signoff.

## ICU / plural / selector contract — re-run on current bytes

- All 83 keys: `argument mismatches: []` (names, order and kinds identical to `en`).
- Four changed keys: `row.actions` keeps `{name}` as `simple`; the other three carry no arguments (`en=() ru=()`).
- Plurals: `link.since_seconds`, `link.since_minutes`, `session.tray.questions`, `session.tray.agents_failed` → `['one','few','many','other']`, `all_render_value=True`, no empty branch.
- `drawer.status` select still has `online` / `retrying` / `other` with the same branch copy.
- Cross-key references: resume word in `ended_hint` True; approve label starts with the pill True; FAB word «Новая» in `empty_hint` True.
- Typography on the four changed values: no ASCII `...`, no trailing period, no double space, no straight quotes; U+2026 used for `resuming`.
- Repo gate: `scripts/i18n/check.py --catalogues-only` semantics re-run in-process → `problems: 0`; full `scripts/i18n/check.py` → exit 0, `catalogues consistent`, 34 advisory (pre-existing tree findings, none in `ru/`).

## Width, measured with the real formatter (source-side cells)

| Key | Rendered | Cells | maxCells | Status |
|---|---|---|---|---|
| `row.actions` (`{name}=Сборка релиза`) | `Действия: Сборка релиза` | 23 | 40 | fits |
| `session.resume` | `Возобновить` | 11 | 12 | fits |
| `session.resuming` | `Возобновляем…` | **13** | 12 | **+1 — rendered-fit signal** |
| `session.ended_hint` | full hint | 76 | — | wraps |
| `session.ended` (neighbour) | `Завершена · история сохранена` | 29 | 32 | fits |
| `session.degraded` (neighbour) | `Не отвечает — сообщения подождут` | 32 | 36 | fits |
| `list.filter.unread` (residual) | `Непрочитано · 12` | 16 | 16 | at budget; 17 at 3 digits |
| `list.filter.needs_you` (neighbour) | `Ждут вас · 12` | 13 | 16 | fits |

These are formatter-rendered cell counts, not viewport measurements; the rendered pass owns actual fit.

## Checks executed (actual output)

- `sha256sum` on en/context/before/after/current — values in the table above; before equals the R1-audited hash, current equals the recorded after hash.
- `delta_checks_a2.py` → `changed keys: 4`, `unchanged keys: 79 of 83`, `removed: [] added: []`, `key order identical to en: True`.
- `runtime.message_arguments` over all 83 → `argument mismatches: []`.
- `runtime.plural_arguments` → four plurals `one/few/many/other`, `all_render_value=True`.
- `runtime.render_message` on changed keys → `Действия: Сборка релиза`, `Действия: Weekly review`, `Возобновить`, `Возобновляем…`, the updated hint.
- Cross-key checks → `True / True / True`.
- `diff candidate-before.json current` → 4 changed lines, exit 1; `wc -l` → 85 / 85.
- `check_catalogues()` → `problems: 0`.
- Full capture: `checks-a2.out` beside this report.

## Verdict

**TERMINAL PASS — no blocker or major finding remains.**

- `chrome25-ru-a1`: 2 minor (F3, F4) **fixed**; 3 nit (F1, F2, F5) **rejected with reasoning accepted** — no cosmetic churn requested.
- Delta is exactly four keys; the other 79 are byte-identical to the R1-audited target and retain R1 coverage.
- Residual items, all outside a translation verdict:
  1. **Rendered fit** — `session.resuming` 13/12 cells; `list.filter.unread` 17/16 at a 3-digit count. The rendered QA/design pass must confirm; if the button clips, the fix is layout/budget, not a worse term.
  2. **Guide records (programme owner)** — add «вкл»/«выкл» as the ru short state pair and «возобновить» (Resume) vs «продолжить» (Continue) to `i18n/style-guides/ru.md`.
  3. **Second-auditor safety delta signoff** for the changed `ended_hint` (pre-change `chrome25-safety-b1` signed the old bytes).

## Limitations

Catalogue/locale verification only. No screenshots, viewport measurements or runtime app rendering; the width table is a source-side signal for the rendered pass.

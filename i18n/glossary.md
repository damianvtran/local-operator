# Glossary — one term, one decision, for every locale

Translation invariant per term: the SAME rendered word in every locale unless a
row says otherwise. Translators and auditors cite this file; the ledger's
audit step is where a violation is caught. The per-locale style guides
(`i18n/style-guides/<locale>.md`) cite this file and record only the calls it
delegates to them.

Two ground rules for reading a row:

- **Policy** is one of *keep English* (a surface name — never translated),
  *translate* (the preferred rendering per locale is given), or *per-locale
  decision* (the locale's guide records the final call; what is in the table
  is the default fallback).
- **Preferred renderings are citation forms.** Inflect them naturally in a
  sentence — Russian case endings, Urdu postpositions, Hindi plural obliques.
  A row is violated by a different *term*, not by a grammar form.

## 1. Product-surface names (never translated)

| Term | Decision |
|---|---|
| Local Operator | keep as-is — the product name |
| `lop`, `lo` | keep as-is (commands) |
| `omp` / oh-my-pi | keep as-is (external product) |
| tool names (`bash`, `read`, `edit`, `wake`, `monitor`, …) | keep as-is — the model's vocabulary |
| `--flags`, config keys, file paths, identifiers, URLs | keep as-is |
| namespaces/keys (`wire.errors.…`, `tui.…`), event types, `--json` keys | keep as-is — machine tokens |
| technology names (Python, Node, JSON, YAML, Git, GitHub, macOS, Windows, Linux) | keep their official spelling |

When a name and a concept collide, the *form* decides: the tool invocation
`wake` in a tool card or a command stays English; prose about waking
translates (§2). Same for `monitor`, `schedule`, `relay`, `project`.

The keep-English categories match `i18n/allowlist.toml` (commands/flags/
identifiers, dev logs, tool names, `--json` keys/event types, model-facing
prompts, code snippets/paths). Model-facing text — system prompts, tool
descriptions, prompt markers — never translates at all (RFC §2.7).

## 2. Core terminology

| Term | Policy | Preferred rendering (fr · es · zh-CN · ru · vi · ur · hi) |
|---|---|---|
| session | translate | session · sesión · 会话 · сессия · phiên · سیشن · सत्र |
| wake | translate — tool name `wake` stays | réveil · despertar · 唤醒 · пробуждение · thức dậy · بیداری¹ · जागृति¹ |
| agent | translate — `task(agent=…)` role names stay | agent · agente · 智能体 · агент · tác nhân · ایجنٹ · एजेंट |
| subagent | translate | sous-agent · subagente · 子智能体 · субагент · tác nhân con · ذیلی ایجنٹ · उप-एजेंट |
| tool | translate — tool names stay | outil · herramienta · 工具 · инструмент · công cụ · ٹول · टूल |
| tool trace | translate | trace · traza · 轨迹 · трассировка · dấu vết · ٹریس · ट्रेस |
| transcript | translate | transcription · transcripción · 记录 · транскрипт · bản ghi · ٹرانسکرپٹ · ट्रांसक्रिप्ट |
| workspace | translate | espace de travail · espacio de trabajo · 工作区 · рабочее пространство · không gian làm việc² · ورک اسپیس · कार्यक्षेत्र |
| approvals | translate — `/approvals` stays | approbations · aprobaciones · 审批 · одобрения³ · phê duyệt · منظوریاں · स्वीकृतियाँ |
| project | translate | projet · proyecto · 项目 · проект · dự án · پروجیکٹ · प्रोजेक्ट |
| model | translate — model ids stay | modèle · modelo · 模型 · модель · mô hình · ماڈل · मॉडल |
| prompt | translate | prompt · prompt · 提示词 · промпт · prompt · پرامپٹ · प्रॉम्प्ट |
| compaction | translate | compactage · compactación · 压缩 · сжатие · nén · کمپریشن · संपीडन |
| schedule | translate — tool/command names stay | planification · programación · 定时 · расписание · lịch · شیڈول · शेड्यूल |
| monitor | translate — tool name stays | surveillance · monitorización · 监控 · мониторинг · giám sát · نگرانی · निगरानी / मॉनिटर⁷ |
| relay | translate — identifiers stay | relais · relé · 中继 · ретранслятор · chuyển tiếp · ریلے · रिले |
| usage | translate | utilisation · uso · 用量 · использование · lượng sử dụng · استعمال · उपयोग |
| cost | translate | coût · coste⁴ · 费用 · стоимость · chi phí · لاگت · लागत |
| telemetry | translate — vi keeps "telemetry"⁵ | télémétrie · telemetría · 遥测 · телеметрия · telemetry · ٹیلیمیٹری · टेलीमेट्री |
| team | translate | équipe · equipo · 团队 · команда · nhóm · ٹیم · टीम |
| skill | translate | compétence · habilidad · 技能 · навык · kỹ năng · مہارت · कौशल |
| device | translate | appareil · dispositivo · 设备 · устройство · thiết bị · ڈیوائس · डिवाइस |
| credentials | translate | identifiants · credenciales · —⁶ · учётные данные · thông tin xác thực · —⁶ · क्रेडेंशियल |
| job | translate | tâche · trabajo · —⁶ · задание · tác vụ · —⁶ · जॉब |

1. `wake` in `ur`/`hi`: verb forms read best inside a sentence ("10 بجے
   جاگے گا", "10 बजे जागेगा"); for the noun label use the table's form. The
   wave that first translates `wake` strings records the final call — and any
   exception — in the locale guide.
2. `vi` workspace is long; keep it. If a width budget cannot take it, the wave
   records `workspace` as the fallback rather than truncating.
3. `ru` approvals: `одобрения` is the term; do not mix it up with
   `подтверждения` (confirmations). If a wave finds it stilted it records an
   alternative in the guide.
4. `es` cost: `coste` is the reference; `costo` is the accepted LatAm variant —
   the es guide records which the wave standardises on.
5. `vi` telemetry: keep the English word — the native phrase ("đo lường từ
   xa") is long and absent from developer register. Recorded here as the
   per-locale decision.
6. `credentials` and `job`: fr/es recorded renderings in wave 1; ru/vi in
   wave 2; hi in wave 3 (see its guide). The zh-CN/ur cells stay `—` until
   their waves record endorsed renderings; do not invent them here. Hindi uses
   क्रेडेंशियल्स for the plural of क्रेडेंशियल.
7. Hindi distinguishes watching/monitoring (निगरानी) from a stored monitor
   object (मॉनिटर), not from the Latin tool name `monitor`. Child agents and
   subagents consistently use उप-एजेंट; the abandoned child-only spelling is
   not an exception. Other endorsed Hindi wave-3 calls live in the hi guide.

### Wave-2 kept-English set (all locales)

Recorded by the ru/vi wave after their audits; fr/es audited the same way:
`endpoint` (Latin, undeclined), `extra` (pip extras), `merge` / `pull` /
`push` / `commit` (git/hub operations). Inflect around them; never translate
inside them.

A locale guide may *extend* this table (its own additions, its own
kept-English list) but never contradict it; a genuine contradiction comes back
here as a change.

## 3. Formatting decisions (shared)

| Topic | Decision |
|---|---|
| digits | Latin digits everywhere in v1 — `hi`/`ur` included (§2.8); native digits deferred, revisitable per style guide |
| grouping/decimal | never hand-written — the formatter renders: en `1,234.5` · fr `1 234,5` (U+202F) · es/vi `1.234,5` · ru `1 234,5` (NBSP) · hi `12,34,567.5` (Indian grouping) · zh-CN/ur `1,234.5` |
| plurals | every plural message needs `other`; selectors must be categories of the locale: en one/other · fr one/many/other (`0` is one: "0 session") · es one/many/other (`0` is other) · zh-CN other only · ru one/few/many/other (`0` is many) · vi other only · ur one/other · hi one/other (`0` is one) |
| dates/times | formatter layer only, never hand-written patterns; examples: zh-CN `2026年10月10日` · fr `10 oct. 2026` · hi/ur am/pm suffix per formatter |
| formality | per locale: tu/vous, 您/你, вы/ты, bạn, आप, آپ — each guide records the decision and why |
| RTL | `ur` best-effort in the TUI (per-string FSI/PDI isolation applied by the render layer, zero width; storage order preserved); full mirroring on web/Electron/RN — see the `ur` guide |
| punctuation | locale conventions inside prose; never translate inside backticks; code spans/literals byte-identical |
| placeholders | `{name}` / ICU structures preserved byte-exactly; never rename or reorder; `%s`-style tokens as-is |

## 4. How to change this file

One row per decision, with enough context that a translator who never saw the
discussion lands on the same answer. New terms are added by the wave that first
needs them. A per-locale exception is recorded in that locale's style guide
with its reason — the glossary row stays the default. A change here is a
repo-artifact change: it does not need regenerating (it is not generated), but
the translation ledger's freshness rule treats it as style-guide scope — a wave
re-audits the terms it cites when either changes.

## 5. Wave-3 audit binding and unfinished backfill

These records distinguish complete semantic coverage from a clean remediation
sample. A passed delta cannot fill an unreadable row in a required first pass.
The ledger records namespace/source freshness; this note binds the narrower
high-visibility coverage and the model that actually performed each round.

- Hindi visibility: independent auditor `c96e3e71494b` (resumed
  `5d81a4d40810`), `deepseek/deepseek-flash`, 2026-10-11, read all 30 welcome
  and all 12 settings-error values, and scanned all 755 settings values.
  Independent polish auditor `26037f4e1f23`, same model/date, passed the
  9-value delta (welcome 1/settings 8); unchanged bytes identical per its
  report. No unreadable semantic row was reported. The original pinned
  `anthropic/claude-sonnet-5-5` was a fallback origin, not the audit model.
- Populated Hindi exec: translator `4323561a8d6d` and independent auditor
  `046b8b75715a` ran `openai/gpt-6.1-sol`, 2026-10-12. The audit covered all
  17 values; runtime render samples passed. Superseded empty-source approvals
  remain historical notes in the ledger, not current approvals.
- `tui.analytics` and `wire.errors.desktop` were absent from the original
  ten-namespace snapshot. Their Hindi backfill remains unfinished. Missing
  groups use the existing wire-envelope English fallback; direct `text.tr`
  retains its missing-group message-code behavior. This is not full-surface
  localization or a runtime fallback/eligibility-policy change.
- The complete Chinese candidate is excluded from this active Hindi-only PR
  and preserved on remote-held
  [`held/i18n-w3-zh-candidate-4773bb87`](https://github.com/damianvtran/local-operator/tree/held/i18n-w3-zh-candidate-4773bb87)
  at `4773bb87fb71bf47209c9b7bc4a6a3e0ae0bf854`. Its slash audit remains
  **50/51, NOT PASSED**; the candidate remains **HELD**. Future Chinese work
  must use that preserved history and resolve its audit/provenance gates,
  not treat removing only its failed group as approval of the other nine.
- Hindi's required actual `anthropic/claude-sonnet-5-5` designer review remains
  **UNPERFORMED**. No design sign-off or immediate release approval is claimed.

Source: named independent audit reports retained by the wave-3 coordinator;
provider fallbacks are disclosed in those reports. Runtime samples above do
not establish full CLI/lifecycle or rendered-frame coverage.

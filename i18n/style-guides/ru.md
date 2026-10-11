# Style guide — Russian (`ru`)

Read with `i18n/glossary.md`; everything here is ru-specific and binding for
translators and the auditor.

## Formality & tone

- Address: **вы** — lowercase "вы" mid-sentence ("Вы" capitalised is
  letter-writing register; do not use it). Recorded decision: a professional
  tool for a mixed audience; "ты" reads over-familiar for work software.
- Prefer impersonal constructions where the English allows: "Сессия
  сохранена", "Не удалось подключиться" beat "Мы сохранили вашу сессию".
- Action labels: infinitive — "Сохранить", "Продолжить", "Отмена" (standard
  ru UI convention).
- Terse; no "Пожалуйста" unless the English says "please"; no "!"; sentence
  case. Never Title Case.
- Do not add politeness padding ("Уважаемый пользователь") — the source never
  has it.

## Product-name policy

- Keep exactly as in glossary §1: Local Operator, `lop`, tool names,
  `--flags`, config keys, paths, identifiers.
- Kept-Latin: API, CLI, TUI, JSON, YAML, HTTP, URL, GPU — as-is, undeclined.
- Established transliterations live in Cyrillic: промпт, токен, кэш, коммит.
- Prefer Russian where settled: сервер, файл, папка, сеть, настройки,
  устройство, стоимость, команда (team).

## Locale calls (ru)

- session → сессия (feminine; inflect: сессии / сессию / сессий)
- wake → пробуждение ("пробуждения по расписанию")
- tool trace → трассировка · transcript → транскрипт
- approvals → одобрения — do not confuse with подтверждения (confirmations).
- schedule → расписание; one scheduled item reads "запланированный промпт".
- relay → ретранслятор · usage → использование · cost → стоимость

## Plurals (critical — the category set is bigger here)

- Categories: `one` (1, 21, 31…), `few` (2-4, 22-24…), `many` (0, 5-20,
  25-30…), `other` (fractions). All four are allowed; `other` is required.
- Example: `{count, plural, one {# сессия} few {# сессии} many {# сессий} other {# сессии}}`
- Numbers govern case — check every branch: "1 сессия", "2 сессии", "5
  сессий". Get the genitive right in `few` and `many`.

## Numbers, dates, digits

- Never hand-written — the formatter renders "1 234,5" (NBSP grouping) and
  dates "10 окт. 2026 г."; 24-hour times.
- Digits: Latin (v1). Decimal comma.

## Width rules

- ru runs roughly 20-30% longer than en. Tighten; standard abbreviations
  ("мин", "сек" are conventional; "настр." only if a wave records it) beat
  truncation. Never truncate.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55.

## Punctuation & typography

- Quotes: «ёлочки» in prose, „лапки“ nested inside them.
- Em dash " — " with spaces; ellipsis …; no typographic spacing before
  : ; ! ? (unlike French).
- Code spans, flags, paths stay byte-identical — Latin, undeclined.

## Wave-2 recorded calls (2026-10-11)

Recorded after the ru wave passed audit; every call below was audited in the
catalogues before landing here.

- runtime → «рантайм» (Cyrillic transliteration — inflects: «до следующего
  рантайма»). catch-up → «наверстывание». job → «задание» (background job →
  «фоновое задание»; keeps «задача» free for todos, aligning with the
  wire.settings «Max background jobs» copy).
- credentials → «учётные данные» (settled over «идентификаторы»: those are
  *identifiers*; keep «секрет» for the stored secret itself).
- Kept Latin, undeclined: `endpoint`, `extra` (pip extras), `merge` / `pull` /
  `push` / `commit` (git/hub operations), `shell`, `cmux`.
- Fork vocabulary: «Форк» as the noun/UI entity (fork.title, welcome cites);
  «ответвить» / «ветка» for the action sense.
- Do not render "slice" as «срез» — use «часть» / «раздел» / «подсистема»
  (the calque was rejected in wave-2 review).
- Roll terminology (S3b, wave-2 delta): «раскатка» (noun) / «раскатывать»
  (verb) for the mesh rollout; «участник» for member (keep «пиры» for
  peers); «право `update`» for the grant; «в простой» for idle.
- Recorded width exception: `tui.welcome.tip.cmux` renders «пространство или
  поверхность» — «рабочее» is dropped because the full term cannot fit the
  57/59-cell tip pool alongside the cite; accepted by the wave-2 audit.
  Revisit only with a budget change.

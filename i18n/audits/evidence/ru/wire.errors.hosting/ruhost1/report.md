# RU audit — `wire.errors.hosting` — R2 fresh reader A1

- **Audit ID:** `ruh-r2a1` · **Locale:** ru · **Namespace:** `wire.errors.hosting` (1 key)
- **Date:** 2026-11-03 · **Auditor:** `i18n-auditor`, session `ac46f9462716`
- **Model:** requested `radient/auto`; served model **UNKNOWN** (the route exposes only `radient/auto`; no provider/model is inferred, no fallback claimed)
- **Independence:** not the translator (translator session `06e83e0ae0f9`); no prior report by me on this key. This is **one** fresh pass — it does not by itself satisfy a two-reader requirement.
- **Verdict:** **`passed`** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT open.
- **Source delta:** `Платформа хостинга не настроена.` → `Платформа размещения не настроена.` (one word, `хостинга` → `размещения`; sentence shape, capitalisation and terminal period unchanged).

## Key under audit

`wire.errors.hosting.not_configured`

| | |
|---|---|
| EN | `Hosting platform is not configured.` (33 chars) |
| RU (current, R2) | `Платформа размещения не настроена.` (34 chars, 64 UTF-8 bytes) |
| RU (R1, superseded) | `Платформа хостинга не настроена.` (32 chars) |

## 1. Mechanical (self-executed; repo validator **not** run — source-only override)

| Check | Result |
|---|---|
| Valid JSON, one flat object, key set identical to EN and to R1 | PASS |
| Placeholder names/types identical to source | N/A — no placeholders (no `{}`, `$x`, `%s`, no Rich/markup tags, no backticks) |
| ICU `plural`/`select` structure and CLDR categories | **N/A with reason:** zero ICU arguments; plain literal message, no `plural`/`select`/`number`, no dynamic detail, so the ru `one/few/many/other` set has nothing to express |
| Markup / code spans / commands / flags / identifiers untouched | PASS — none present; text is Cyrillic + spaces + final full stop only |
| NFC normalised | PASS (`unicodedata.is_normalized('NFC')` true over the raw bytes) |
| Control / format / bidi characters (`Cf`, `Cc`, ZWSP, LRM/RLM, isolates, BOM), double spaces, leading/trailing whitespace | PASS — none; no NBSP |
| Initial capital after the frame's colon | PASS — `П` uppercase, style guide "sentence case, never Title Case" |
| Terminal full stop retained (single) | PASS — exactly one `.`, nothing after it |
| Column-space/width | No `maxCells`/`maxChars` is declared in the sidecar; **no fit claim** is made. Counted: 34 chars (+1 vs EN, +2 vs R1). No render or pseudo-localisation was run. |

Input hashes (re-measured this pass, stdlib `hashlib.sha256` over the staged bytes):

| Input | sha256 |
|---|---|
| EN `…/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` |
| Context `…/hosting-source-v1/wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` |
| RU R2 target `…/hosting-ru-r2/target/ru/wire.errors.hosting.json` | `38478ec0af7bdcb739dcd1bbbd82c8c50345e9869c2b21921f9a46d4bab1737a` |
| RU R1 target `…/hosting-ru/target/ru/wire.errors.hosting.json` (identifier only, bytes untouched) | `6fe2a88318b042c082047f0ac4b6a76c88d68957a1053f75ce9c0617efd0317d` |

## 2. Accuracy

Meaning preserved exactly: the platform serving as hosting is *not configured*; no omission, no addition, no negation or polarity change, no remedy smuggled in. The context demands a bare **condition** clause — "diagnostic; names the condition, never restates the remedy" — and the string contains none (`Настройте …` is absent; it belongs to the surrounding refusal). Safe wording for an error surface: it states the blocking condition, blames nothing, names no user action it cannot guarantee.

Frame fit checked at the consumer, not in the abstract: `sdk.py:602-627` substitutes this catalogued sentence for `str(error)` only when `type(error) is HostingNotConfiguredError` and the English matches the catalogue. It lands in the `{error}` slot of `wire.errors.sdk_startup.attach_hosting_unconfigured`, i.e. after a colon and followed by the frame's own remedy sentence. Composed, it reads:

> … и этот корень не может запустить его: **Платформа размещения не настроена.** Настройте размещение для этого корня (его config.yml), …

Capital and full stop are required there, and correct.

## 3. Idiom and register

Native product copy, not a calque: 4-word impersonal diagnostic in sentence case, `вы`-neutral, no politeness padding, no exclamation, no Title Case — matches `i18n/style-guides/ru.md` §Formality & tone. `не настроена` is the established ru rendering of "not configured" in this corpus (cf. `wire.errors.desktop.aida_no_provider`: «Провайдер ещё не настроен»). Feminine agreement with `платформа` is correct. Punctuation follows ru rules (no space before the full stop, plain spaces, no NBSP).

## 4. Terminology

Settled term verified in the **owned CORE tree**, `~/local-operator/.worktrees/startup-owned-copy/local_operator/i18n/catalogues/ru` (the tree the manager named), scanned as JSON line-by-line:

- lines matching `хостинг`/`размещени`: **6 lines in 4 files** — 6× «размещени», **0× «хостинг»**.
  - `cli.credential_help.json:3` `cli.credential_help.credential` → «…ключами API и учётными данными для разных **платформ размещения**»
  - `cli.exec_worker_help.json:5` `cli.exec_worker_help.hosting` → «Переопределение **платформы размещения**»
  - `cli.global_help.json:5` `cli.global_help.description` → «Поддерживает несколько **платформ размещения**…»
  - `cli.global_help.json:8` `cli.global_help.hosting` → «Используемая **платформа размещения** (radient, deepseek, …)»
  - `cli.global_help.json:16` `cli.global_help.propagated_hosting` → «**Платформа размещения** для этого запуска…»
  - `wire.errors.sdk_startup.json:2` `attach_hosting_unconfigured` → «Настройте **размещение** для этого корня…»
- The same scan over the **main checkout** tree `~/local-operator/local_operator/i18n/catalogues/ru` returns **0** matching lines — which is the mechanical reason the R1 "no matches" claim was true of one tree and false of the product.
- English side mirrors it: "hosting platform" in `cli.credential_help.json:3`, `cli.exec_worker_help.json:5`, `cli.global_help.json:5,8,16` and the new namespace — so «платформа размещения» is the product's own rendering, not a coinage of this packet.
- Provider stays a separate noun, as the record claims: `wire.settings.hosting.label` = «**Провайдер** по умолчанию» (EN `Default provider`), help «Провайдер для новых бесед». So naming the *platform* here does not collide with the settings label.

Alternatives considered and rejected: «хостинг» / «хостинг-платформа» (idiomatic in isolation, but would coin a second name for a settled concept — the exact R1 defect); «платформа хостинга» (same defect, plus the awkward genitive); «не сконфигурирована» (bureaucratic, unmotivated synonym); «не задана»/«не выбрана» (narrower than the source's "not configured" — an addition, not a translation); «Хостинг не настроен.» (shorter but drops the named platform and breaks corpus consistency).

## 5. Fit

34 chars / 64 bytes: +1 over EN. No budget is declared for this key, so no overflow claim either way; the clause sits mid-line in a refusal paragraph (wrapped prose, not a button/tab/status slot), and wide-glyph/terminal-cell risk does not apply to plain Cyrillic with no combining marks. **No fit conclusion is claimed.**

## 6. RTL

N/A — locale is `ru` (Cyrillic, LTR). No Latin tokens, paths or digits in the value; nothing to place in a bidi run.

## Findings

None. Nothing to fix, nothing to defer; `unresolved_blocker_major_count = 0`.

Observations recorded as non-findings (no action requested):

1. Inside the refusal frame the clause is followed by «Настройте размещение для этого корня…», so `размещен-` appears at the end of one sentence and the start of the next. Deliberate consistency, and the frame is a different key outside this packet's scope; changing it would be an out-of-scope edit, not a fix here.
2. Width: +2 chars over R1; harmless for wrapped prose and below any declared budget (none declared).

## Scope honesty / limitations

- **Ledger:** no ledger row was written and no repo file was edited — this pass ran under the manager's source-only override (read/grep/glob/write/eval, pure stdlib `json`/`hashlib`/`unicodedata`; no subprocess, project import, validator, test, build, install or render). The record is **uncommitted** in my scratch; the canonical home it would take when committed is `i18n/audits/evidence/ru/wire.errors.hosting/ruh-r2a1/`.
- **Unshipped:** the new namespace is **not** in the worktree git index (pure-stdlib scan of `.git/worktrees/startup-owned-copy/index` for `wire.errors.hosting.json` / `.context.json` → absent; `HEAD` = `ref: refs/heads/fix/startup-refusal-owned-copy`). No shipped status, no activation and no locale-clearance claim.
- **Legacy English exception unchanged and untouched:** `session_factory.py:816` still raises `HostingNotConfiguredError("Hosting platform is not configured.")`; `sdk.py:604-607` documents that it "stays English by contract"; `session/runtime/launch.py:1021` keeps its own English actionable text. This packet changes display copy only.
- **Not claimed:** native-speaker status (register judged by reasoning against the ru style guide and the shipped corpus), pseudo-localisation, TUI/desktop render, runtime proof, `maxCells` compliance, or any product-wide English-free state.
- **R1 material preserved, not re-audited:** `hosting-ru/record.json` (sha256 `1b9876bb…`, identifier only) and the R1 target bytes are untouched and are not evidence for this verdict. The R1 packet I could see contains only `record.json` and the target; no reader-report files were present there, so no prior report was read, altered or endorsed by me. Nothing is backdated: this record is dated 2026-11-03 and covers R2 only.
- **Corpus-correction claim verified, not taken on trust:** the six-line/four-file count and the two-tree contrast above were measured here by stdlib scan; the record's "six matching lines, zero «хостинг»" and its examples are accurate as written, and its explanation (R1 grepped the main checkout, where the term is absent, not the owned worktree) matches what I measured.

# i18n review — ru · `wire.errors.hosting` · fresh reader 2 (R2)

**Audit id** `r2b` · **date** 2026-11-03 · **session** 9f874dd64c37 · **model** `radient/auto` (served: UNKNOWN)
**Scope** one key, `wire.errors.hosting.not_configured`, plus the R1→R2 delta and its direct neighbours. Not the ru catalogues wholesale.
**Independence** I did not translate this string and did not author R1; my session differs from the translator's (06e83e0ae0f9). I judged the current value before reading either earlier reader's conclusion, then checked the historical claims against the repo.

## Inputs (hashes recomputed here with stdlib `hashlib` over the staged bytes)

| role | path | sha256 | status |
|---|---|---|---|
| en | `…/3ef6295a5545/scratchpad/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | matches declaration |
| context | `…/hosting-source-v1/wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | matches declaration |
| target ru (R2) | `…/06e83e0ae0f9/scratchpad/hosting-ru-r2/target/ru/wire.errors.hosting.json` | `38478ec0af7bdcb739dcd1bbbd82c8c50345e9869c2b21921f9a46d4bab1737a` | matches declaration |
| target ru (R1) | `…/06e83e0ae0f9/scratchpad/hosting-ru/target/ru/wire.errors.hosting.json` | `6fe2a88318b042c082047f0ac4b6a76c88d68957a1053f75ce9c0617efd0317d` | unchanged, preserved |

The CORE worktree's `en/wire.errors.hosting.json` and its `.context.json` are byte-identical to the frozen source (same two hashes).

## Current value

`wire.errors.hosting.not_configured` → **«Платформа размещения не настроена.»** (34 chars)
EN → “Hosting platform is not configured.”

## 1 · Mechanical — PASS

- Valid JSON; exactly one key; key set identical to en (1:1). Nothing added or dropped.
- Zero interpolation arguments: no `{}`, no `#`, no `plural`/`select`. ICU/CLDR category coverage is therefore N/A with reason — no plural/select message exists, so ru's `one`/`few`/`many`/`other` set has nothing to apply to.
- No markup, Rich tags, backticks, commands, flags, paths or identifiers on either side; nothing to keep byte-identical.
- NFC-normalised; zero `Cf`/`Cc` characters; zero bidi controls; no trailing whitespace; no double spaces.
- Initial capital present and terminal full stop present — required by the sidecar's splice-after-colon contract.
- The repo validator `scripts/check.py` was **not** executed: this pass is source-only (no process, validator, tests or repo writes). The results above are my own stdlib checks over the exact bytes, not a validator run, and no validator exit code should be read into this report.

## 2 · Accuracy — PASS

Condition only; no remedy restated; nothing added or dropped against the English; polarity correct («не настроена» = not configured); wording safe for an error surface (it names a blocker, grants nothing). English “hosting” here means the hosting/platform that runs the model, and the Russian names the same object — not the provider identity, which is «провайдер».

## 3. Idiom & register — PASS

Neutral diagnostic register, sentence case, no «Пожалуйста», no exclamation, no added politeness. «Не настроена» is the natural short-form predicate for a setting/state. Capital after the colon and the closing full stop are correct for a complete sentence spliced before a following sentence. No calque beyond the term discussed in §4.

## 4. Idiom, terminology, consistency — PASS, and this is the delta's whole purpose

Measured in the owned CORE worktree (`~/local-operator/.worktrees/startup-owned-copy/local_operator/i18n/catalogues/ru`):

- pattern `хостинг|размещени` → **6 matching lines, all «размещени», 0 «хостинг»**; pattern `хостинг` alone → **0 matches**.
- the six, each the ru counterpart of an EN “hosting platform” / “Configure hosting” string (checked on the English side): `cli.global_help.description`, `cli.global_help.hosting`, `cli.global_help.propagated_hosting`, `cli.exec_worker_help.hosting`, `cli.credential_help.credential`, `wire.errors.sdk_startup.attach_hosting_unconfigured`.
- `i18n/glossary.md` has no hosting entry (0 matches), so the shipped catalogue is the term authority here; `i18n/style-guides/ru.md` prefers Russian where the corpus has settled a rendering rather than a new loan.
- No collision: nothing in the ru catalogues renders deployment as «размещение» (pattern `развёрт|разверт|деплой` → 0 matches), so the word is not shadowed by another product concept.

So «платформа размещения» is the product's own rendering of “hosting platform”, and the R2 value now matches it. R1's «платформа хостинга» made a second name for one concept inside the same CLI — `lop --help` says «платформа размещения» while the error said «хостинг». That terminology MAJOR is resolved.

Verified separately: the provider noun stays distinct — `wire.settings.hosting.label` = «Провайдер по умолчанию».

## 5 · Fit — not claimed

The context sidecar carries `role`/`tone`/`note` only; there is no `maxCells`/`maxChars`. No budget exists, so I make no fit or cap claim. For information only: 32 → 34 characters (+2) against R1.

## 6 · RTL — N/A

ru is LTR; no bidi, mirrored or digit-policy concerns apply.

## Historical correction (record delta, not a re-audit of R1)

- I reproduced R1's “zero corpus” result exactly, and its cause: `~/local-operator/local_operator/i18n/catalogues` (main checkout) → 0 matches for `хостинг|размещени`; the owned worktree → 6. The main tree's `ru/` catalogue does not contain the five files carrying the term at all (`cli.global_help.json`, `cli.exec_worker_help.json`, `cli.credential_help.json`, `wire.errors.sdk_startup.json`, `wire.errors.hosting.json` are absent there), so its silence measured the wrong tree, not the product. The author record's correction matches what I measured; the count is grep's own reported number of matching lines, not an estimate.
- Old reader 2's two claims do not survive: the Russian sdk_startup frame **exists** (`ru/wire.errors.sdk_startup.json:2`, fully Russian) and `sdk.py:623–631` renders the refusal cause in the active locale. The English pin covers the legacy exception's own string only — `session_factory.py:816` still raises “Hosting platform is not configured.”, byte-identical to the en catalogue value this key reads, with the contract documented at `sdk.py:604–617`. Frame display copy is translated; the exception stays English by contract. “Frame absent” and “frame English-pinned” are both false.
- Nothing historical was altered: R1 target `6fe2…` and R1 record `2734…` are as they were; this is a fresh R2 reader, not a rewrite, and it is not backdated.

## Verdict

**passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. No finding IDs. No changes requested to the value. Unresolved blocker/major count: 0.

## Observations (not findings; no action requested)

1. In the sdk_startup frame «размещение» now recurs across the colon boundary: «…не может запустить его: Платформа размещения не настроена. Настройте размещение для этого корня…». That frame is a different catalogue and out of this packet's scope; matching the settled term is worth the repetition, and R1's alternative was two names in adjacent sentences.
2. The English itself uses “hosting” for model providers; «размещение» inherits that abstraction faithfully instead of inventing a narrower Russian noun.

## Not done, and why

- No repo validator (`scripts/check.py`), generator, scanner, tests, native run, render or pseudo-localisation — source-only override for this pass.
- No ledger row and no `i18n/audits/evidence/` commit — repo writes are outside this pass's authority, and the namespace is uncommitted/unshipped so no catalogue directory exists to attach to. Canonical path if a repo-write chain later adopts this record: `i18n/audits/evidence/ru/wire.errors.hosting/r2b/` — **not created**; nothing here is claimed to be on main.
- No fit/cap/terminfo claims (no budget declared).
- `uncommitted/unshipped` and `absent at base 87ca846245b2` are the author's declarations: I verified the namespace is absent from the main checkout catalogue, but did not run git.

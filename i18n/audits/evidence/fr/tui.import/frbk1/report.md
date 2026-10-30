# French audit — `tui.import` (frbk1)

- **Locale / namespace**: `fr` / `tui.import`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.import.json` — sha256 `932e0f87210ff80950346c684afb000907610c9ff35388125977182120b136c2` · 87 keys
- `en/tui.import.context.json` — **does not exist** (`context: null`; no role/tone/note was assumed or invented)
- `fr/tui.import.json` — sha256 `0426d38851af5cf4d741a7d98667faddfe9068f5eb7b387910c6fdeb057d4aa7` · 87 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 87/87; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (11 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-IM-1 | nit | `tui.import.error_cancel` | Rien n’est en cours, il n’y a rien à annuler. | Rien à annuler : aucune importation en cours. | Two clauses say the same thing where English says it once; the terser form matches the surrounding status copy. |
| FR-IM-2 | nit | `tui.import.metadata_only_explain` | Le harness a peut-être supprimé l’enregistrement. | Le harness a peut-être supprimé la transcription. | "the rollout" is the source harness's session transcript; "enregistrement" is this namespace's word for a memory record (preview_memory), so the sentence can read as a deleted memory record. |

## Values identical to English

7 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.import.error_generic` | `{error}` | placeholder only (`{error}`) |
| `tui.import.preview_warning_row` | `· {message}` | glyph plus placeholder (`· {message}`) |
| `tui.import.progress_row` | `{completed}/{total} · {phase}` | numeric skeleton plus `{phase}` |
| `tui.import.result_head_no_target` | `{source_id}` | placeholder only (`{source_id}`) |
| `tui.import.result_warning_row` | `· {message}` | glyph plus placeholder (`· {message}`) |
| `tui.import.selected` | `[x]` | selection glyph (`[x]`) |
| `tui.import.unselected` | `[ ]` | selection glyph (`[ ]`) |

## Cleared on purpose (not findings)

- No context sidecar: en/tui.import.context.json does not exist, so input_hashes.context is null and no role/tone was assumed. Judgements rest on the English catalogue and the slash-command surface.
- Plural branches (11 occurrences) all add many and keep the count token; "d'enregistrements"/"d'avertissements" elide correctly.
- Key names, subcommand tokens (list, preview, run, status, cancel), the arrow and the bracket glyphs are byte-identical; hint verbs are consistently infinitive.

## Verdict

**passed** — 87 keys read (no English context sidecar exists for this namespace); accurate throughout, two nits.

Findings: blocker 0 · major 0 · minor 0 · nit 2. Keys read: 87.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

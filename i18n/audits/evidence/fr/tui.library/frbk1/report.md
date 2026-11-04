# French audit — `tui.library` (frbk1)

- **Locale / namespace**: `fr` / `tui.library`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.library.json` — sha256 `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11` · 108 keys
- `en/tui.library.context.json` — **does not exist** (`context: null`; no role/tone/note was assumed or invented)
- `fr/tui.library.json` — sha256 `764cf78d82e995f5e17eb6e7e161eeece56d3f9f36abd4375325fe69d9a02d3a` · 108 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 108/108; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (2 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-LB-1 | minor | `tui.library.librarian_hook_missing / librarian_unavailable` | … dans ce build.  /  … dans cette version. | Use "ce build" in both (corpus precedent: cli.services_help, cli.update_help, cli.refresh_help) or "cette version" in both | Both English strings say "in this build"; the namespace renders the same term two ways, and the shipped corpus already standardises build → build (kept English) for install/daemon surfaces. |

## Values identical to English

11 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.library.category_code` | `Code` | correct French — `Code` is the same word |
| `tui.library.category_document` | `Documents` | correct French — `Documents` is the same word (and the corpus translates Other/Presentations/Tableurs beside it) |
| `tui.library.category_image` | `Images` | correct French — `Images` is the same word |
| `tui.library.detail_meta` | `{kind} · {origin} · {size} · {when}` | metadata skeleton of four placeholders and separators |
| `tui.library.detail_occurrence` | `{session} · {origin} · {when}` | metadata skeleton of three placeholders and separators |
| `tui.library.error_body` | `{error}` | placeholder only (`{error}`) |
| `tui.library.form_description` | `description` | correct French — `description` is the same word |
| `tui.library.form_error` | `{error}` | placeholder only (`{error}`) |
| `tui.library.sort_type` | `type` | correct French — `type` is the same word |
| `tui.library.tag_category_format` | `format` | correct French — `format` is the same word |
| `tui.library.tag_category_source` | `source` | correct French — `source` is the same word |

## Cleared on purpose (not findings)

- No context sidecar: en/tui.library.context.json does not exist; input_hashes.context is null.
- Asset terminology is consistent (asset → élément) across ask_prompt, empty_body, form_* and status rows; tags → étiquettes with feminine agreement ("séparées par des virgules") where the projects form, which has no tag noun in scope, correctly defaults to masculine.
- The Librarian's conversation, hook() and CLI tokens stay verbatim; sort/filter/status labels keep char-case.

## Verdict

**passed** — 108 keys read (no English context sidecar exists); one glossary drift between two sibling keys, otherwise clean.

Findings: blocker 0 · major 0 · minor 1 · nit 0. Keys read: 108.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

# French audit — `tui.small` (frbk1)

- **Locale / namespace**: `fr` / `tui.small`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.small.json` — sha256 `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75` · 137 keys
- `en/tui.small.context.json` — sha256 `1e3d6785f78126b8389b2acbcc081d5bf782fbaf744114e47611fbff3c45dde1` · 137 entries
- `fr/tui.small.json` — sha256 `5a1a26ac03a707b7878d4eeb9511d90f398ee48862ed1ff237b5bdb886c6d321` · 137 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 137/137; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (11 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-SM-1 | minor | `tui.small.spend.unpriced_cancelled` | réponse perdue · coût inconnu (29c / budget 30) | réponse perdue · facturable ? (29c) | English says "may be billed" and the sidecar requires the row to say so; "coût inconnu" is the text of the shortest sibling ("bill unknown"), so the middle tier loses the billing caution and duplicates another state. |
| FR-SM-2 | nit | `tui.small.draft.attachment_count` | many {({shown} images)} | many {({shown} d’images)} | The many branch fires at exact multiples of 10⁶, where French requires "de": "(1 000 000 d'images)". Every other many branch in the eight catalogues either adds "de"/"d'" or has no noun after the number. |
| FR-SM-3 | nit | `tui.small.tool_card.low_quality_hint` | … page pauvre ou exigeant JS — … | … page pauvre ou nécessitant JS — … | "exigeant JS" (present participle of exiger) is grammatical but reads translated; "nécessitant" is the collocation French dev copy uses for a JS-dependent page. |

## Values identical to English

9 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.small.copy.code_count` | `{shown} code` | correct French — `code` is the same word; count is raw |
| `tui.small.tool_card.http_blocked` | `⚠ HTTP {status} {reason} — {clause}.` | protocol skeleton (`HTTP {status} {reason} — {clause}.`) whose parts are machine tokens |
| `tui.small.session_picker.preview_mode_key` | `ctrl+e` | key label (`ctrl+e`) |
| `tui.small.goal_panel.judge_absent` | `judge: —` | `judge:` is protocol vocabulary per the sidecar; `—` is the absent-value sentinel |
| `tui.small.goal_panel.judge_line` | `judge: {words}` | `judge:` is protocol vocabulary per the sidecar; only the state word varies |
| `tui.small.network_panel.installed_row` | `  installed:  {value}\n` | protocol label with column padding, per the sidecar |
| `tui.small.network_panel.identity_row` | `  identity:   {identity}\n` | protocol label with column padding, per the sidecar |
| `tui.small.model_picker.effort` | `effort {value}` | correct French — `effort` is the same word; `{value}` is catalogue data |
| `tui.small.session_sidebar.lead_focus` | `f9 focus` | key label plus `focus`, which the fr corpus keeps English (`perde le focus`) |

## Cleared on purpose (not findings)

- All 16 maxCells budgets pass (tightest: no_charge 9/9, unpriced_cancelled_shortest 12/12, unpriced_failed_mixed_mid 23/24).
- Protocol labels (relay:, installed:, identity:, judge:) stay byte-identical and keep their column padding; protocol-label exemption matches the sidecar notes.
- Feminine agreement for sessions and tâches is consistent (annulée/terminée/échouée/bloquée/abandonnée), and the (s) hedge on count-invariant counters is a uniform house pattern rather than a local slip.

## Verdict

**passed** — 137 keys read; dense but accurate; one spend-state collapse, two nits.

Findings: blocker 0 · major 0 · minor 1 · nit 2. Keys read: 137.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

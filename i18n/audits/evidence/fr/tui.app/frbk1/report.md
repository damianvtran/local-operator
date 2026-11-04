# French audit — `tui.app` (frbk1)

- **Locale / namespace**: `fr` / `tui.app`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.app.json` — sha256 `7e5f39df72438663a66afca0e5338e90b7b984bdca6b562195ce976945bd7875` · 46 keys
- `en/tui.app.context.json` — sha256 `a840ac61e3c1e1178825257c1b9ebaa6ebfef500df4f3a29e8c463b50eef42fa` · 46 entries
- `fr/tui.app.json` — sha256 `ecd4a3b5b436429c04a034aacd8dc8e281853fa891ac7936a00299a478a865b3` · 46 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 46/46; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (6 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-AP-1 | nit | `tui.app.config.label_web_fetch` | récupération web | récupération web (acceptable); fetch web if the switch is meant to name the tool | The fr guide keeps "fetch" in its settled dev vocabulary, but this is the human-readable switch name parallel to "recherche web", so the French concept name is defensible; recorded so the choice is deliberate rather than accidental. |
| FR-AP-2 | nit | `tui.app.providers.no_live_list` | aucune liste en direct : {names} | aucune liste à jour : {names} | "live list" here means a current (non-stale) list, not a live feed; "en direct" reads as streaming and misleads slightly next to the stale-list siblings. |

## Values identical to English

1 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.app.skills.agents_heading` | `agents` | correct French — `agent` is the glossary rendering and its plural is identical |

## Cleared on purpose (not findings)

- Plural verb agreement is carried inside the message (prend/prennent, nécessite/nécessitent) exactly as English intends, with a valid many branch added.
- cfg toggle receipts keep "à votre prochain tour" (vous) and the em dash layout; ctrl+c, config.yml, env vars and backticked commands are byte-identical.
- The NBSP pair around "sur" in aida.extras_used sits exactly where English has its NBSP pair around "of" (verified by codepoint).

## Verdict

**passed** — 46 keys read; translation is accurate and idiomatic, with two wording nits and no structural defect.

Findings: blocker 0 · major 0 · minor 0 · nit 2. Keys read: 46.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

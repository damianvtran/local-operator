# French audit — `tui.usage` (frbk1)

- **Locale / namespace**: `fr` / `tui.usage`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.usage.json` — sha256 `5aeaec254352896101bf2825ac03d06f74c95e3598cda23536c5fe28507a2f8b` · 21 keys
- `en/tui.usage.context.json` — sha256 `bf599f5f161087b890e1e24434a03209fee286b865173af6b578da3133b6ef19` · 21 entries
- `fr/tui.usage.json` — sha256 `601947b30dceccb4e8c9874dca087423ab75553bbee4880ee479b162b8deeebc` · 21 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 21/21; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (2 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-US-1 | nit | `tui.usage.sign_in_expired` | {value} déconnectée(s) | {value} à reconnecter | English names an expired sign-in (the note: "Windows whose credential has expired"); "déconnectée" states a disconnection and drops the expiry that tells the user a re-auth is due. |

## Values identical to English

None: no value in this namespace is byte-identical to its English source.

## Cleared on purpose (not findings)

- The paid-continuation note reproduces the already-shipped relay.shell.usage.continue wording exactly, so the two surfaces agree.
- Spacing layout (double-space dot separators, leading comma, trailing spaces, indents) matches English on every key; both plural keys add a valid many branch.
- usage → utilisation, cost/credits → coût/crédits and window → fenêtre agree with the glossary and the rest of the fr corpus.

## Verdict

**passed** — 21 keys read; clean, with one nuance note on the sign-in counter.

Findings: blocker 0 · major 0 · minor 0 · nit 1. Keys read: 21.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

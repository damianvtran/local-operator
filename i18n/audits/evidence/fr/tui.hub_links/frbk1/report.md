# French audit — `tui.hub_links` (frbk1)

- **Locale / namespace**: `fr` / `tui.hub_links`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.hub_links.json` — sha256 `7fbe6713f8ac8f312884476c498c3b74e76c5f0853bdd027f25fc47d489311bb` · 56 keys
- `en/tui.hub_links.context.json` — sha256 `9b81676163337d8a93e38138909dbd4801f8edd8416075c6efb994f9870c682e` · 55 entries
- `fr/tui.hub_links.json` — sha256 `d69bd8926e7b50d9902dcb89de3d59082582737612153d1b4cd059b0833d6767` · 56 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 56/56; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: no key in this namespace carries an ICU plural, so the fr one/many/other category set is not exercised here.
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-HL-1 | minor | `tui.hub_links.hint.auto_push` | a publication auto | a auto-push (11c, keeps the policy token English) — or keep "a publication auto" and confirm the joined hint row on a rendered narrow card | The sidecar requires each key-hint fragment to stay short so the joined row fits a narrow card; 18 cells against 11 grows the row by 7 with the sibling "s abonnement" (12 vs 11). Keeping the policy token English restores the width while the toggle label stays "Publication auto". |
| FR-HL-2 | nit | `tui.hub_links.refusal.revision_moved` | Rouvrez l’invite pour confirmer la dernière version. | Rouvrez la demande de confirmation pour valider la dernière version. | "l'invite" is the shell prompt in French; here it names the consent prompt, so a reader can take the recovery step for a terminal action. |

## Values identical to English

5 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.hub_links.hint.cancel` | `esc` | key name (`esc`), verb supplied by the sentence |
| `tui.hub_links.hint.confirm` | `ctrl+y` | key name (`ctrl+y`) |
| `tui.hub_links.hint.range` | `{first}–{last}/{total}` | numeric skeleton `{first}–{last}/{total}`; the separator is EN DASH by design |
| `tui.hub_links.section.agents` | `Agents` | correct French — identical plural, capitalised as a section heading |
| `tui.hub_links.source.public` | `Public` | correct French — `public` has the same spelling; the Private/Public pair is consistent |

## Cleared on purpose (not findings)

- The consent block's two policy axes stay independent in French (abonnement entrant / publication sortante), and "privé" visibility is explicit in every sentence.
- Sidecar coverage: 55 of 56 keys. tui.hub_links.hint.engage has no sidecar entry; it was audited from the catalogue pair alone ("↑↓ to begin" / "↑↓ pour commencer"), and no context was invented for it.
- hint.failed is 35 cells against its 48-cell budget; the range indicator keeps the EN DASH U+2013.

## Verdict

**passed** — 56 keys read; consent, refusal and policy copy are accurate and consistent; one width risk and one ambiguous noun.

Findings: blocker 0 · major 0 · minor 1 · nit 1. Keys read: 56.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.

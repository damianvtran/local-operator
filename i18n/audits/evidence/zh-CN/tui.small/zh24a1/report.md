# Independent zh-CN audit — tui.small — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `tui.small`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **137**; keys read: **137**
- Delta keys audited: **16** (16 new keys)
- Verdict: **clean** (status `passed`); unresolved blocker/major: **0**; minor: 0; nit: 0

## Input hashes (sha256)

- en: `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75` (`local_operator/i18n/catalogues/en/tui.small.json`)
- context: `1e3d6785f78126b8389b2acbcc081d5bf782fbaf744114e47611fbff3c45dde1` (`local_operator/i18n/catalogues/en/tui.small.context.json`)
- target: `5b7e05905402a7713dee3dd0ffa3df8048bae23e6b18108cfcec1c6cd4567ca6` (`local_operator/i18n/catalogues/zh-CN/tui.small.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `tui.small.spend.no_charge`
- `tui.small.spend.unpriced_cancelled`
- `tui.small.spend.unpriced_cancelled_short`
- `tui.small.spend.unpriced_cancelled_shortest`
- `tui.small.spend.unpriced_cancelled_wide`
- `tui.small.spend.unpriced_failed`
- `tui.small.spend.unpriced_failed_mixed`
- `tui.small.spend.unpriced_failed_mixed_mid`
- `tui.small.spend.unpriced_failed_mixed_short`
- `tui.small.spend.unpriced_failed_short`
- `tui.small.spend.unpriced_failed_wide`
- `tui.small.spend.unpriced_ok`
- `tui.small.spend.unpriced_ok_short`
- `tui.small.spend.unpriced_partial`
- `tui.small.spend.unpriced_partial_short`
- `tui.small.spend.unpriced_partial_shortest`

## Findings

None. The delta keys and the regression read are clean on the checks and terms below.

## Mechanical results

- Key parity: en 137 / zh 137; missing 0; extra 0.
- ICU argument parity (names and kinds): 137 keys checked, 0 failures.
- Message parse/render: 137 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 11 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 16 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences. Whitelisted expected transforms present in this namespace: scheduled_count's one-branch literal and/or 55k→55000.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 137 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.

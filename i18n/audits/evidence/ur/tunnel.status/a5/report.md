# Independent translation re-audit — ur / tunnel.status / a5

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up), after the translator's fixes to the a4 findings. Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 5ac9acc2ba69 (round-1 remediation re-audit); neither the i18n-translator (c0c9e9803405), the English-drafting coder (889b9f67edf5), nor the prior auditors (117e988422ff, dff33b90744d); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 5ac9acc2ba69 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation fixes, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (as recorded in the prior bundle; not otherwise verified).
Prior audit: i18n/audits/evidence/ur/tunnel.status/a4/record.json (target sha256 ec4ecfa70fc2ab0a07b631871f9e861b00ce36093cbc8d63054ff92606dd9275).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 f0f4134969ebc8fac6561409b512f39275d0996c672731b06b442b4bebbd5e3e (current bytes).

**Verdict: passed**: 0 unresolved blocker/major. Prior findings resolved: UR-06, UR-07. Open carried nits: none. New findings: none.

## Scope and method

Mechanical integrity was re-verified for the whole namespace (24 keys) at its current bytes. Exactly the values the translator reported changed since a4: `tunnel_inactive.label`, `card_add_credit`. The other scope values match the a4 report verbatim, and the 16 non-scope keys are byte-identical to HEAD (audit.log: `delta.py` against the prior report table, since the prior target bytes are not retained, and `git diff HEAD -U0`). Because this group is high-visibility (status/error copy), I re-read all 8 remediation-scope keys independently instead of assuming the earlier pass was right. Judgement is by reasoning against the en source, its context sidecar, i18n/glossary.md (no row for phone access, credit, balance or console) and i18n/style-guides/ur.md. No MT service was used.

Card fit: each line is measured with `rich.cells.cell_len` including the 2-cell '! ' prefix (card_paused after '! /mobile enable — '); the budget is 50 cells (tests/unit/test_tunnel_remedies.py; tui/app.py `tunnel_park_card`). 'Phone access' must be explicit on every card line except card_paused, which follows the `/mobile enable` command. The real `tunnel_park_card` and `gateway.reason_label` render path was also run per locale (audit.log). Before the ledger row, ur was already shipped, so the pre-ledger render shows the Urdu values.

## Changed keys (judged in full)

| key | en | current value | cells | judgement |
|---|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | paused by Radient — reconnecting | Radient نے روکا — دوبارہ منسلک ہو رہا ہے | 40 | RESOLVED (UR-06). «Radient نے روکا — دوبارہ منسلک ہو رہا ہے»: ergative «نے روکا» with no overt object takes the default masculine singular, and «ہو رہا ہے» now agrees with the masculine کنیکٹر in both real frames: `Connector: retrying — {label}` (tunnels/cli.py, re-rendered via gateway.reason_label, audit.log) and «کنیکٹر آخری بار بند ہوا: {label}». Same gender choice as hi. card_paused keeps the feminine «روک دی ہے», which is right for its phone-access card. 40 cells; this label has no card budget. |
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Radient کریڈٹ شامل کریں، فون رسائی بحال ہو گی | 47 | RESOLVED (UR-07). «Radient کریڈٹ شامل کریں، فون رسائی بحال ہو گی» = 'Add Radient credit; phone access will be restored'. It brings back the resume meaning, keeps «فون رسائی» (feminine «بحال ہو گی» agrees), and uses the natural Urdu imperative ، consequence pattern with Urdu comma ، per guide. «ہو گی» (split) is the dominant corpus spelling (43 vs 9). The promise is accurate: billing_lapsed.detail says it reconnects by itself once the balance is restored. «…بحال ہو جائے گی» would be smoother but is 52 cells. Rendered via the real TUI path: 47/50 cells. |

## Remaining scope keys (re-read, unchanged since the prior round)

- `tunnel.status.tunnel_inactive.detail`: «Radient نے اس کمپیوٹر کی فون رسائی روک دی ہے، لیکن بلنگ درست ہے۔ جس کمپیوٹر پر Local Operator چل رہا ہے اس پر فون رسائی دوبارہ منسلک کریں؛ یہ خود بھی دوبارہ کوشش کرتی رہتی ہے۔ اگر یہ رکی ہی رہے تو {url} پر Radient کنسول میں اس کا جائزہ لیں۔» — OK.
- `tunnel.status.tunnel_disabled.detail`: «اس کمپیوٹر کی فون رسائی Radient کنسول میں بند ہے۔ اسے {url} پر دوبارہ آن کریں؛ یہ خود دوبارہ منسلک ہو جاتی ہے۔» — OK.
- `tunnel.status.billing_lapsed.detail`: «آپ کا Radient بیلنس ختم ہو گیا ہے، اس لیے فون رسائی رکی ہوئی ہے۔ {url} پر کریڈٹ شامل کریں؛ بیلنس بحال ہوتے ہی یہ خود دوبارہ منسلک ہو جاتی ہے۔» — OK.
- `tunnel.status.billing_inactive.detail`: «اس Radient اکاؤنٹ کے لیے فون رسائی ابھی آن نہیں ہوئی۔ اُس کمپیوٹر پر Local Operator میں فون رسائی کا سیٹ اپ کھولیں اور ماہانہ قیمت قبول کریں۔» — OK.
- `tunnel.status.card_paused`: «Radient نے روک دی ہے» — OK.
- `tunnel.status.card_open_console`: «Radient کنسول میں فون رسائی آن کریں» — OK.

## Mechanical

Valid UTF-8 JSON; key parity 24/24 with en; placeholder sets identical on every key (only `{url}`, `{port}`, `{status}`, `{label}`, `{at}`, `{window}`, `{tally}` where the source has them); no ICU plural/select in this namespace; protected tokens (Radient, Local Operator, `lop …` commands, `/mobile`, HTTP, 127.0.0.1, `--tunnel-id`) unchanged; no tunnel/remote-access wording in the scope values; 'phone access' explicit on every scope value except the label and card_paused; no manual bidi controls; `check.py --catalogues-only`: 0 problems. Urdu: no Latin , ; ? or sentence-final . in prose; no Hindi danda. A negative control (faults injected into a scratch copy) shows the checker reports placeholder, typography, bidi-control and missing-term faults (audit.log).

## Findings

| ID | severity | status | key | issue | suggested fix |
|---|---|---|---|---|---|
| UR-06 | minor | resolved | tunnel_inactive.label | Raised in a4; fixed by the translator's new value (see the Changed keys table). | — |
| UR-07 | minor | resolved | card_add_credit | Raised in a4; fixed by the translator's new value (see the Changed keys table). | — |

No new findings this round.

## RTL

the new card_add_credit now starts with the Latin token «Radient», like the already-passed card_open_console and card_paused. On a terminal with first-strong bidi but no isolation, «Radient» lands at the visual left and is read last. The ur guide (§RTL & bidi) records this as the known degraded TUI stance: 'never "fix" a string to compensate'. No arrows or mirrored glyphs; no manual bidi controls (mech.py). Not raised.

## Considered, not raised

- Feminine card_paused vs masculine tunnel_inactive.label: different surfaces with different implied subjects (phone-access card vs connector status); both agree. Not raised.

## Not verified

No native-speaker review and no rendered screenshot of the terminal, desktop or phone frame; widths are `cell_len` measurements plus the real `tunnel_park_card` string. Judgement is by reasoning only; no MT service was used. This is the requested second independent pass over the remediation-scope keys; I did not re-litigate the 16 unchanged keys that earlier committed rounds passed.

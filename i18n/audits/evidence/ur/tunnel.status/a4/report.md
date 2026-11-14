# Independent translation audit — ur / tunnel.status / a4

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session dff33b90744d; neither the i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session dff33b90744d (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (read from session c0c9e9803405 transcript metadata; not otherwise verified); 2026-11-14.
Prior audit: i18n/audits/evidence/ur/tunnel.status/a3/record.json.

**Verdict: passed**: 0 unresolved blocker/major; 2 finding(s) (2 minor, 0 nit).

Scope: all 8 keys whose English changed or is new in the remediation were judged in full: meaning, safety, placeholders, glossary, tone, fit and, for ur, RTL. The receipt covers the whole namespace (24 keys) at the current English source hash. Unchanged keys were re-read only to check terminology consistency; they were not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no phone-access, tunnel, billing or credit row; corpus usage governs), i18n/style-guides/ur.md, the en catalogue and its context sidecar, the render sites (tui/app.py `tunnel_park_card`, tunnels/cli.py connector line, tunnels/gateway.py, i18n/display.py), every scope value in full, and term counts across all ur catalogues. Judgement is by reasoning only; no MT service was used.

## Keys judged

| key | EN | ur | result |
|---|---|---|---|
| `tunnel_inactive.label` | paused by Radient — reconnecting | Radient نے روک دی — دوبارہ منسلک ہو رہی ہے | UR-06 |
| `tunnel_inactive.detail` | Radient has paused phone access for this computer, but billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review it in the Radient console at {url}. | Radient نے اس کمپیوٹر کی فون رسائی روک دی ہے، لیکن بلنگ درست ہے۔ جس کمپیوٹر پر Local Operator چل رہا ہے اس پر فون رسائی دوبارہ منسلک کریں؛ یہ خود بھی دوبارہ کوشش کرتی رہتی ہے۔ اگر یہ رکی ہی رہے تو {url} پر Radient کنسول میں اس کا جائزہ لیں۔ | OK |
| `tunnel_disabled.detail` | Phone access for this computer is turned off in the Radient console. Turn it back on at {url}; it reconnects by itself. | اس کمپیوٹر کی فون رسائی Radient کنسول میں بند ہے۔ اسے {url} پر دوبارہ آن کریں؛ یہ خود دوبارہ منسلک ہو جاتی ہے۔ | OK |
| `billing_lapsed.detail` | Your Radient balance has run out, so phone access is paused. Add credit at {url}; it reconnects by itself once the balance is restored. | آپ کا Radient بیلنس ختم ہو گیا ہے، اس لیے فون رسائی رکی ہوئی ہے۔ {url} پر کریڈٹ شامل کریں؛ بیلنس بحال ہوتے ہی یہ خود دوبارہ منسلک ہو جاتی ہے۔ | OK |
| `billing_inactive.detail` | Phone access isn't turned on for this Radient account yet. Open phone access setup in Local Operator on that computer and accept the monthly price. | اس Radient اکاؤنٹ کے لیے فون رسائی ابھی آن نہیں ہوئی۔ اُس کمپیوٹر پر Local Operator میں فون رسائی کا سیٹ اپ کھولیں اور ماہانہ قیمت قبول کریں۔ | OK |
| `card_paused` (new) | paused by Radient | Radient نے روک دی ہے | OK |
| `card_add_credit` (new) | Add Radient credit to resume phone access | فون رسائی کے لیے Radient کریڈٹ شامل کریں | UR-07 |
| `card_open_console` (new) | Turn phone access on in the Radient console | Radient کنسول میں فون رسائی آن کریں | OK |

## Judgement

Polite آپ register; respectful imperatives (شامل کریں، آن کریں، کھولیں، جائزہ لیں); Urdu punctuation ۔ ، ؛ only (no । and no Latin . , ; in prose); no manual bidi controls (mech.py, check.py). 'phone access' is فون رسائی in every scope sentence (51 corpus uses, feminine agreement: «رکی ہوئی ہے», «منسلک ہو جاتی ہے», «کوشش کرتی رہتی ہے»); none of the scope values says ٹنل or ریموٹ. Other terms match the corpus: Radient کنسول (14), کریڈٹ شامل کریں، بیلنس، بلنگ، اکاؤنٹ (74), آن کریں, 'reconnect' as دوبارہ منسلک (21, = repoint), منتظر (26), «کوئی رقم نہیں لی جاتی». Safety: «سیٹ اپ کبھی کریڈٹ نہیں خریدتا», «کوئی رقم نہیں لی جاتی», «سیٹ اپ آپ کے اس انتخاب کو تبدیل نہیں کرے گا» and «آپ کو کچھ کرنے کی ضرورت نہیں» all keep their meaning; no value names a command. «اُس کمپیوٹر» (billing_inactive.detail) is a good choice: it separates 'that' from the 'this' (اس) of the account clause. tunnel_inactive.detail names «جس کمپیوٹر پر Local Operator چل رہا ہے», which is clearer than the English 'the computer running it'.

## RTL

RTL (style guide §RTL & bidi): storage is logical order; display.py wraps each ur status line in RLI…PDI, and the web, Electron and RN clients set dir=rtl. So values that start with the Latin token «Radient» resolve right-to-left and read in the correct order. {url} follows «پر» or a colon. In an RTL paragraph the colon before the URL is a neutral between R and L text, takes the paragraph direction and stays with the Urdu text, with the URL as a separate LTR island (action_add_credit, action_open_console). «{url} پر Radient کنسول» keeps the URL and Radient apart, with the R word «پر» between them. No arrows or mirrored glyphs. Known degraded stance, not a finding: the English-framed CLI line `Connector: retrying — {label}` is not isolated, so on terminals without bidi support a Latin-led Urdu label shows «Radient» on the visual left of its run. The guide says never to reword a string to compensate.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| UR-06 | minor | `tunnel.status.tunnel_inactive.label` | «Radient نے روک دی — دوبارہ منسلک ہو رہی ہے» agrees in the feminine with an unstated فون رسائی, but the label's known frames name the masculine connector: «کنیکٹر آخری بار بند ہوا: {label}» (tunnel.status.last_exit.*) and `Connector: retrying — {label}` (tunnels/cli.py). There it reads 'Connector … stopped(f.) … is reconnecting(f.)'. The hi value uses the masculine (रोका … जुड़ रहा है), which agrees with कनेक्टर. Meaning is clear; this is an agreement slip. | «Radient نے روکا — دوبارہ منسلک ہو رہا ہے» (40 cells). card_paused can keep the feminine because that card is about phone access, or name the noun: «Radient نے فون رسائی روک دی ہے» (49/50 cells with the prefix). |
| UR-07 | minor | `tunnel.status.card_add_credit` | «فون رسائی کے لیے Radient کریڈٹ شامل کریں» = 'Add Radient credit for phone access'. It drops 'to resume', so the one-line TUI card no longer says that phone access has stopped and can read as a purchase prompt. Dropping it was not needed for width: a wording that keeps it fits the 50-cell card. | «Radient کریڈٹ شامل کریں، فون رسائی بحال ہو گی» (47 cells with '! '), or «Radient کریڈٹ شامل کر کے فون رسائی بحال کریں» (46). «فون رسائی بحال کرنے کے لیے …» is 52, too long. |

## Translator flags

The translator raised no hi- or ur-specific open question; the flags for 'access'-only card lines, 'saldo', 'Пополните баланс' and «на этом компьютере» are es/fr/ru/vi and outside this audit. The ur card lines keep the full 'فون رسائی'. UR-07 is a separate omission, not a shortening.

## Mechanical

check.py --catalogues-only: 0 problems. mech.py (audit.log) confirmed valid JSON, en key parity, placeholder parity across all keys, verbatim Radient, Local Operator and {url}, no bidi or zero-width controls, script punctuation, no tunnel or remote term in scope values, and every card line within 50 cells including its prefix. The namespace carries no ICU plural or select strings.

## Not verified

No native-speaker review and no rendered terminal, desktop or phone frame (RTL mirroring is unverified in pixels). The Reconnect action label that `mobile.setup.reconnect` sits beside lives outside this catalogue and was not compared.

This bundle is written in the shared worktree and is uncommitted (staged_not_committed). The auditor ran generate.py and check.py --catalogues-only after the ledger append; see the requesting session's report.

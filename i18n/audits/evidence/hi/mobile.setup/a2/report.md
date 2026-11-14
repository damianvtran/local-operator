# Independent translation audit — hi / mobile.setup / a2

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session dff33b90744d; neither the i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session dff33b90744d (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (read from session c0c9e9803405 transcript metadata; not otherwise verified); 2026-11-14.
Prior audit: i18n/audits/evidence/hi/mobile.setup/a1/record.json.

**Verdict: passed**: 0 unresolved blocker/major; 1 finding(s) (0 minor, 1 nit).

Scope: all 12 keys whose English changed or is new in the remediation were judged in full: meaning, safety, placeholders, glossary, tone, fit and, for ur, RTL. The receipt covers the whole namespace (94 keys) at the current English source hash. Unchanged keys were re-read only to check terminology consistency; they were not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no phone-access, tunnel, billing or credit row; corpus usage governs), i18n/style-guides/hi.md, the en catalogue and its context sidecar, the render sites (tui/app.py `tunnel_park_card`, tunnels/cli.py connector line, tunnels/gateway.py, i18n/display.py), every scope value in full, and term counts across all hi catalogues. Judgement is by reasoning only; no MT service was used.

## Keys judged

| key | EN | hi | result |
|---|---|---|---|
| `disabled` | Phone access for this computer is turned off in the Radient console. It reconnects by itself once it is back on; setup will not override that choice. | इस कंप्यूटर का फ़ोन एक्सेस Radient कंसोल में बंद है। फिर से चालू होते ही यह अपने-आप जुड़ जाता है; सेटअप आपकी इस पसंद को नहीं बदलेगा। | OK |
| `billing_lapsed` | Your Radient balance has run out, so phone access is paused. Setup never purchases credit; phone access reconnects by itself once the balance is restored. | आपका Radient बैलेंस ख़त्म हो गया है, इसलिए फ़ोन एक्सेस रुका हुआ है। सेटअप कभी क्रेडिट नहीं खरीदता; बैलेंस बहाल होते ही फ़ोन एक्सेस अपने-आप फिर से जुड़ जाता है। | OK |
| `reconnect` | Radient has paused phone access for this computer, but billing is in order. Reconnecting brings it back; nothing is charged. | Radient ने इस कंप्यूटर का फ़ोन एक्सेस रोक दिया है, लेकिन बिलिंग ठीक है। फिर से जोड़ने पर यह वापस आ जाता है; कोई शुल्क नहीं लगता। | OK |
| `network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back. You don't need to do anything. | यह कंप्यूटर Radient तक नहीं पहुँच सका (शायद इसका नेटवर्क बदल गया है या टूट गया है)। कनेक्शन लौटते ही फ़ोन एक्सेस अपने-आप फिर से जुड़ जाता है। आपको कुछ करने की ज़रूरत नहीं है। | HI-06 |
| `title_reconnect` (new) | Phone access is paused by Radient | Radient ने फ़ोन एक्सेस रोक दिया है | OK |
| `title_billing_lapsed` (new) | Phone access needs more Radient credit | फ़ोन एक्सेस के लिए और Radient क्रेडिट चाहिए | OK |
| `title_disabled` (new) | Phone access is turned off in the Radient console | फ़ोन एक्सेस Radient कंसोल में बंद है | OK |
| `title_network` (new) | Phone access is waiting for the network | फ़ोन एक्सेस नेटवर्क की प्रतीक्षा में है | OK |
| `title_radient_unavailable` (new) | Phone access is waiting for Radient | फ़ोन एक्सेस Radient की प्रतीक्षा में है | OK |
| `title_reconnecting` (new) | Phone access is reconnecting | फ़ोन एक्सेस फिर से जुड़ रहा है | OK |
| `action_add_credit` (new) | Add credit in the Radient console: {url} | Radient कंसोल में क्रेडिट जोड़ें: {url} | OK |
| `action_open_console` (new) | Turn phone access back on in the Radient console: {url} | Radient कंसोल में फ़ोन एक्सेस फिर से चालू करें: {url} | OK |

## Judgement

Polite आप register; polite imperatives (जोड़ें, खोलें, चालू करें, जाँच करें); danda । full stops; nukta kept (फ़ोन, ख़त्म, ज़रूरत). 'phone access' is फ़ोन एक्सेस in every scope sentence (51 uses across the hi corpus, masculine agreement: «रुका हुआ है», «जुड़ रहा है»); none of the scope values says टनल or रिमोट. Other terms match the audited corpus: Radient कंसोल (15), क्रेडिट जोड़ें, बैलेंस, बिलिंग, खाता (account), 'turn on' as चालू करें, 'reconnect' as फिर से जोड़ें/जुड़ना (= repoint), प्रतीक्षा. Safety: «सेटअप कभी क्रेडिट नहीं खरीदता» (billing_lapsed), «कोई शुल्क नहीं लगता» (reconnect), «सेटअप आपकी इस पसंद को नहीं बदलेगा» (disabled) and «आपको कुछ करने की ज़रूरत नहीं है» (network) all keep their meaning; no value names a command; {url}, Radient and Local Operator are byte-identical.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| HI-06 | nit | `mobile.setup.network` | Carry-over of HI-04 (a1), still unaddressed although this key was retranslated in this round: «नेटवर्क … टूट गया है» — टूटना collocates with कनेक्शन, slightly odd with नेटवर्क. | «(शायद इसका नेटवर्क बदल गया है या बंद हो गया है)» |

## Translator flags

The translator raised no hi- or ur-specific open question; the flags for 'access'-only card lines, 'saldo', 'Пополните баланс' and «на этом компьютере» are es/fr/ru/vi and outside this audit. The hi card lines keep the full 'फ़ोन एक्सेस'.

## Mechanical

check.py --catalogues-only: 0 problems. mech.py (audit.log) confirmed valid JSON, en key parity, placeholder parity across all keys, verbatim Radient, Local Operator and {url}, no bidi or zero-width controls, script punctuation, no tunnel or remote term in scope values (the namespace has no terminal card lines). The namespace carries no ICU plural or select strings.

## Not verified

No native-speaker review and no rendered terminal, desktop or phone frame. The Reconnect action label that `mobile.setup.reconnect` sits beside lives outside this catalogue and was not compared.

This bundle is written in the shared worktree and is uncommitted (staged_not_committed). The auditor ran generate.py and check.py --catalogues-only after the ledger append; see the requesting session's report.

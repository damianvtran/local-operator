# Independent translation audit — hi / tunnel.status / a4

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session dff33b90744d; neither the i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session dff33b90744d (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (read from session c0c9e9803405 transcript metadata; not otherwise verified); 2026-11-14.
Prior audit: i18n/audits/evidence/hi/tunnel.status/a3/record.json.

**Verdict: passed**: 0 unresolved blocker/major; 1 finding(s) (0 minor, 1 nit).

Scope: all 8 keys whose English changed or is new in the remediation were judged in full: meaning, safety, placeholders, glossary, tone, fit and, for ur, RTL. The receipt covers the whole namespace (24 keys) at the current English source hash. Unchanged keys were re-read only to check terminology consistency; they were not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no phone-access, tunnel, billing or credit row; corpus usage governs), i18n/style-guides/hi.md, the en catalogue and its context sidecar, the render sites (tui/app.py `tunnel_park_card`, tunnels/cli.py connector line, tunnels/gateway.py, i18n/display.py), every scope value in full, and term counts across all hi catalogues. Judgement is by reasoning only; no MT service was used.

## Keys judged

| key | EN | hi | result |
|---|---|---|---|
| `tunnel_inactive.label` | paused by Radient — reconnecting | Radient ने रोका — फिर से जुड़ रहा है | OK |
| `tunnel_inactive.detail` | Radient has paused phone access for this computer, but billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review it in the Radient console at {url}. | Radient ने इस कंप्यूटर का फ़ोन एक्सेस रोक दिया है, लेकिन बिलिंग ठीक है। जिस कंप्यूटर पर यह चल रहा है, उस पर Local Operator में फ़ोन एक्सेस फिर से जोड़ें; यह अपने-आप भी दोबारा कोशिश करता रहता है। अगर यह रुका ही रहे, तो {url} पर Radient कंसोल में इसकी जाँच करें। | OK |
| `tunnel_disabled.detail` | Phone access for this computer is turned off in the Radient console. Turn it back on at {url}; it reconnects by itself. | इस कंप्यूटर का फ़ोन एक्सेस Radient कंसोल में बंद है। इसे {url} पर फिर से चालू करें; यह अपने-आप फिर से जुड़ जाता है। | OK |
| `billing_lapsed.detail` | Your Radient balance has run out, so phone access is paused. Add credit at {url}; it reconnects by itself once the balance is restored. | आपका Radient बैलेंस ख़त्म हो गया है, इसलिए फ़ोन एक्सेस रुका हुआ है। {url} पर क्रेडिट जोड़ें; बैलेंस बहाल होते ही यह अपने-आप फिर से जुड़ जाता है। | OK |
| `billing_inactive.detail` | Phone access isn't turned on for this Radient account yet. Open phone access setup in Local Operator on that computer and accept the monthly price. | इस Radient खाते के लिए फ़ोन एक्सेस अभी चालू नहीं है। उस कंप्यूटर पर Local Operator में फ़ोन एक्सेस सेटअप खोलें और मासिक मूल्य स्वीकार करें। | HI-05 |
| `card_paused` (new) | paused by Radient | Radient ने रोका है | OK |
| `card_add_credit` (new) | Add Radient credit to resume phone access | फ़ोन एक्सेस फिर शुरू करने के लिए Radient क्रेडिट जोड़ें | OK |
| `card_open_console` (new) | Turn phone access on in the Radient console | Radient कंसोल में फ़ोन एक्सेस चालू करें | OK |

## Judgement

Polite आप register; polite imperatives (जोड़ें, खोलें, चालू करें, जाँच करें); danda । full stops; nukta kept (फ़ोन, ख़त्म, ज़रूरत). 'phone access' is फ़ोन एक्सेस in every scope sentence (51 uses across the hi corpus, masculine agreement: «रुका हुआ है», «जुड़ रहा है»); none of the scope values says टनल or रिमोट. Other terms match the audited corpus: Radient कंसोल (15), क्रेडिट जोड़ें, बैलेंस, बिलिंग, खाता (account), 'turn on' as चालू करें, 'reconnect' as फिर से जोड़ें/जुड़ना (= repoint), प्रतीक्षा. Safety: «सेटअप कभी क्रेडिट नहीं खरीदता» (billing_lapsed), «कोई शुल्क नहीं लगता» (reconnect), «सेटअप आपकी इस पसंद को नहीं बदलेगा» (disabled) and «आपको कुछ करने की ज़रूरत नहीं है» (network) all keep their meaning; no value names a command; {url}, Radient and Local Operator are byte-identical. Width: by cell_len, card lines are 33/40/29 of 50 cells. Worst case, counting every spacing-combining (Mc) mark as a cell, they are 35/45/32, so they fit even on terminals that advance on Mc marks.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| HI-05 | nit | `tunnel.status.billing_inactive.detail` | «इस Radient खाते के लिए फ़ोन एक्सेस अभी चालू नहीं है» reads 'is not on right now', which blurs the never-activated state with the paused/lapsed ones; the English says 'isn't turned on … yet'. The ur value («ابھی آن نہیں ہوئی», perfective) keeps the 'not yet' sense. | «इस Radient खाते के लिए फ़ोन एक्सेस अभी तक चालू नहीं हुआ है।» |

## Translator flags

The translator raised no hi- or ur-specific open question; the flags for 'access'-only card lines, 'saldo', 'Пополните баланс' and «на этом компьютере» are es/fr/ru/vi and outside this audit. The hi card lines keep the full 'फ़ोन एक्सेस'.

## Mechanical

check.py --catalogues-only: 0 problems. mech.py (audit.log) confirmed valid JSON, en key parity, placeholder parity across all keys, verbatim Radient, Local Operator and {url}, no bidi or zero-width controls, script punctuation, no tunnel or remote term in scope values, and every card line within 50 cells including its prefix. The namespace carries no ICU plural or select strings.

## Not verified

No native-speaker review and no rendered terminal, desktop or phone frame. The Reconnect action label that `mobile.setup.reconnect` sits beside lives outside this catalogue and was not compared.

This bundle is written in the shared worktree and is uncommitted (staged_not_committed). The auditor ran generate.py and check.py --catalogues-only after the ledger append; see the requesting session's report.

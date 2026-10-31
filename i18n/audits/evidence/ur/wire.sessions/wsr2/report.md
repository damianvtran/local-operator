# ur / wire.sessions — two independent audits, verdict passed

Namespace `wire.sessions` (9 keys) is NEW in this locale: it did not exist on `origin/main`, so
there is no earlier copy whose audit this could inherit. The copy ships in PR #3102 with the
sessions tool's operator-visible stop receipts and is classed as safety copy — it states whether
a termination signal was sent — so docs/I18N.md, *Add or extend a locale*, step 3 requires a
second independent audit before a ledger row may say `passed`.

## Round 1 — `b3b1c2aec56`

Auditor: `translation-auditor`, actual model `radient/auto` after the documented provider
fallback chain. Verdict **5 pass / 2 reject**: es, hi, ru, ur and zh-CN passed; fr and vi were
rejected on two majors (fr `stop_pending` "la session n'avait pas quitté" — transitive *quitter*
with no object, a calque of the English pluperfect; vi `stop_refused_old_owner` "bản này" — the
proximal *this* build, so the target's older build read as the refusing build's own limitation).
Six minors and five nits went with them. Every round-1 finding is answered in
`### Translation audit remediation — round 1` on the PR: TA1/TA2 fixed, TA3–TA7 and TA10–TA12
fixed, TA9 fixed in the follow-up addendum, TA8 and TA13 deferred with reasons.

## Round 2 — this revision

Auditor: `translation-auditor`, a **different session again**, actual model `radient/alpha` after
the same fallback chain. It read the bytes first and the round-1 record afterwards. Verdict
**7 PASS / 0 REJECT — 0 blocker, 1 minor, 6 nit**, and it closed the safety-copy gate for these
nine keys. Both of its independent checks passed on their own terms: the fr exit clause
(`n'était pas encore sortie`) is grammatical and native, and the vi distal `bản dựng đó` attaches
the limitation to the build the target runs.

## Accepted, non-gating dispositions

Round-2 findings, all accepted as non-gating and deferred to the locale copy wave by the delegation: F1 (hi has two words for idle — निष्क्रिय here vs आइडल in relay.chrome.session.subtitle.idle; minor, needs a hi-guide decision, meaning not inverted), F2 (zh-CN says '{seconds} s' where the locale ships 秒 43x), F3 (vi same class; giây 29x), F4 (fr stop_self_refused repeats 'cette'; 'la session courante' proposed), F5 (hi approval tails read clipped; the auditor itself asks for no gate change), F6 (the 'hold window' mechanism word reads as data retention in es/fr and is opaque in hi/ur), F7 (the ru guide records «в простой» while the correct and shipped form is «в простое» — a guide edit, no copy change). Round-1 findings were all worked: TA1/TA2 majors fixed, TA3-TA7 and TA10-TA12 fixed, TA9 fixed in the addendum, TA8/TA13 deferred with reasons on the PR thread.

Round 2's per-locale note also confirms the vocabulary reuse the wave was built on (`concesión`,
`build`, `inadmissible`-free registers, `सत्र सूची`/`बिल्ड`/`जॉब`, `«ёлочки»`, `فارغ`/`ٹرن`/`جاب`,
`bản dựng`/`lượt`/`tác vụ`, `空闲`/`构建`/`轮次`/`操作员`) and the quoting rule across all seven —
the one deviation round 1 found (four locales dropping the quotes around `{name}`) is gone.

## What this bundle is and is not

`audit.log` is the mechanical half re-executed by the implementer at this revision: hashes, key
and placeholder parity, ICU renders with synthetic values, NFC/control scan, cell widths. The
auditors ran no code — both were text-only — so their reading is their comment, not this log.
Text-only evidence: no rendered frame and no terminal shaping claim; docs/I18N.md keeps those
separate. LLM judgement, no human native-speaker review.

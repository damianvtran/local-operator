# hi / mobile.doctor — independent translation audit a2 (PR #3953 round-1 remediation)

- Verdict: **passed** — blocker 0 · major 0 · minor 1 · nit 1 (unresolved blocker/major: 0)
- Scope: 23 keys (14 changed + 9 new) at 1505e6d2f8dc51f219ddeba76ca0cc344c73806a; 8 unchanged keys byte-identical to 52c0e1c93; `tunnel.problem` removed from catalogue, keys module, code and tests
- Source sha256 (en): `0428c5c217967add291b75b8a493b08083cffe5270a6f8600c0fd0a82d89afe9`; target `2525591f77147d36518c35bba2010492a471a56841f121795afe0c6c2623114e`
- Translator: coder subagent of host session 25fe8dfbf610 (round-1 remediation of PR #3953, commit 1505e6d2f; wrote the English source and all 7 translations) — model: not reported in the commit; host session 25fe8dfbf610 runs anthropic/claude-opus-5-5, the coder child's served model is unverified
- Auditor: translation-auditor, lopdev team member (independent `task` subagent), session 11bf5c282289; did not translate or edit any catalogue value, did not author the remediation — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | key | current | proposed | reason |
|---|---|---|---|---|---|
| HI-R1 | minor | `mobile.doctor.daemon.slow` | फ़ोन एक्सेस सेवा ने उत्तर देने में {ms} ms लिए, जिसे फ़ोन ऐप उत्तर न देना दिखा सकता है। इसे lop mobile restart से रीस्टार्ट करें; अगर फिर से ऐसा हो, तो यह रिपोर्ट सपोर्ट को भेजें। | फ़ोन एक्सेस सेवा ने उत्तर देने में {ms} मिलीसेकंड लिए, जिसे फ़ोन ऐप उत्तर न देना दिखा सकता है। इसे lop mobile restart से रीस्टार्ट करें; अगर फिर से ऐसा हो, तो यह रिपोर्ट सपोर्ट को भेजें। | Unit drift inside the namespace: daemon.ok renders “{ms} मिलीसेकंड”, this key the Latin “ms”. |
| HI-R2 | nit | `mobile.doctor.daemon.state_not_running` | नहीं चल रहा | नहीं चल रही | Pre-existing (unchanged, not gating): the new sibling state_not_answering “उत्तर नहीं दे रही” is feminine (सेवा), this one masculine; both fill the same “auth gate:” field. |

## Method and limits

LLM audit, same model family as the coder (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.

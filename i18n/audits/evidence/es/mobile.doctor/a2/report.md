# es / mobile.doctor — independent translation audit a2 (PR #3953 round-1 remediation)

- Verdict: **passed** — blocker 0 · major 0 · minor 2 · nit 0 (unresolved blocker/major: 0)
- Scope: 23 keys (14 changed + 9 new) at 1505e6d2f8dc51f219ddeba76ca0cc344c73806a; 8 unchanged keys byte-identical to 52c0e1c93; `tunnel.problem` removed from catalogue, keys module, code and tests
- Source sha256 (en): `0428c5c217967add291b75b8a493b08083cffe5270a6f8600c0fd0a82d89afe9`; target `a5fc89676cfee915066b5d77056129bfdb601c91eea17aeb1f834b40636bb6d2`
- Translator: coder subagent of host session 25fe8dfbf610 (round-1 remediation of PR #3953, commit 1505e6d2f; wrote the English source and all 7 translations) — model: not reported in the commit; host session 25fe8dfbf610 runs anthropic/claude-opus-5-5, the coder child's served model is unverified
- Auditor: translation-auditor, lopdev team member (independent `task` subagent), session 11bf5c282289; did not translate or edit any catalogue value, did not author the remediation — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | key | current | proposed | reason |
|---|---|---|---|---|---|
| ES-R1 | minor | `mobile.doctor.daemon.slow` | El servicio de acceso telefónico tardó {ms} ms en responder, y la aplicación móvil puede mostrarlo como que no responde. Reinícialo con lop mobile restart; si vuelve a pasar, envía este informe al soporte. | El servicio de acceso telefónico tardó {ms} ms en responder, por lo que la aplicación móvil puede mostrar que no responde. Reinícialo con lop mobile restart; si vuelve a pasar, envía este informe al soporte. | “mostrarlo como que no responde” is a colloquial calque of “show it as”; the sibling daemon.not_answering already uses the natural “muestra que … no responde”. |
| ES-R2 | minor | `mobile.doctor.logs.errors` | En el último día, el servicio de acceso telefónico llegó a su límite de archivos abiertos {emfile} veces y no pudo abrir una base de datos {sqlite} veces. Reinícialo con lop mobile restart; si vuelve a pasar, envía este informe al soporte. | (follows S1) | “{emfile} veces” renders “1 veces” when a count is 1; this mirrors the English source’s own “1 times” (S1) and is fixed with it, not in the translation. |

## Method and limits

LLM audit, same model family as the coder (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.

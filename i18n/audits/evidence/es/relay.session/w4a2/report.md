# Audit record — es / relay.session / w4a2 (round 2)

- **Locale / namespace:** `es` / `relay.session`
- **Audit id / round:** `w4a2` / 2 · **supersedes** `w4a1` · **scope: the delta keys only**
  (`relay.session.list.empty_hint` — 1 of 17)
- **Audit date:** 2026-10-29
- **Audited revision:** `183f7a0537e4cee42fd82f25a21fe52953a81bdf` (branch
  `feat/i18n-es-wire-errors`). The target catalogue is still **untracked**
  (`?? local_operator/i18n/catalogues/es/relay.session.json`); this round binds
  to the new on-disk bytes,
  `sha256 6812dca69e08ab9f1cf48f271e4db20263c7bbcb659d83a2f3c1a460122a2165`
  (round 1 was `873014622f3fa868c32ee6164a2d12085e35dd148a64ff8663d246b4b26f7a62`).
- **Verdict: PASSED** — 0 blocker, 0 major, 0 minor, 0 nit. `S1` closed by
  measurement.

## Method

Read the delta value from the new bytes and rendered it with the repository
runtime (`local_operator.i18n.runtime`, `es`); re-rendered the other 16 keys and
compared them against `w4a1`'s recorded output so the claim "only the delta key
moved" is checked, not assumed. Output in `audit.log`, harness in
`run-script.py.txt`.

## Inputs (hashes of the audited bytes)

| input | path | sha256 |
| --- | --- | --- |
| en | `local_operator/i18n/catalogues/en/relay.session.json` | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` (unchanged from w4a1) |
| context sidecar | `…/en/relay.session.context.json` | `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db` (unchanged) |
| es (target, untracked) | `…/es/relay.session.json` | `6812dca69e08ab9f1cf48f271e4db20263c7bbcb659d83a2f3c1a460122a2165` |

## Finding status, measured

| id | round-1 severity | key | status now | measured |
| --- | --- | --- | --- | --- |
| S1 | minor | `list.empty_hint` | **CLOSED** | `empieza una abajo…` → `inicia una abajo o desde la TUI en tu máquina` |

The change does what the finding asked: the hint now uses the same verb as the
paired control label `list.start_busy` = `iniciando…`, so one English "start" no
longer has two Spanish verbs. The rest of the sentence is untouched, so the
round-1 "considered and cleared" judgements still hold: the comma before `o` is
correctly absent, and `view.empty_hint`'s "para empezar" is English *to get
started*, not the same term, so it is correctly left alone.

## Structural checks (whole namespace, mechanical)

- Key parity **17/17**, no missing or extra key; no ICU arguments on either side
  (`argument_set_diffs=0`), so nothing to rename, drop or reorder.
- No plural messages in this namespace (`categories_used=[]`), so no CLDR
  category exposure; `many` at exact 10⁶ was cross-checked separately for the
  locale and agrees with Node `Intl.PluralRules('es')` (`agree=True`).
- Neighbour check: **16 of 16** non-delta keys reproduce `w4a1`'s recorded render
  exactly (`reproduced=16 changed=0 unrecorded=0`), i.e. the fix touched only the
  declared key.

## Limitations

- Python runtime only; the phone's TypeScript runtime was not executed (no
  dependency tree in this worktree, no scratch installs). Nothing in this
  namespace depends on ICU constructs, so the exposure is minimal.
- No native-speaker review: the auditor is an LLM.
- Fallback model disclosed; see `record.json` `auditor.model_fallback_notice`.

## Verdict

**PASSED** for `relay.session` round 2 — the single delta key is correct on the
new bytes, the neighbourhood is byte-stable by render comparison, and no new
finding was raised. Combined with `w4a1` this covers all 17 keys at the current
bytes.

# Audit record — es / wire.settings.account_policy / w1a1

- **Verdict: passed** — 0 blocker, 0 major, 1 minor, 2 nit
- Audit date: 2026-10-29
- Auditor: `translation-auditor (lopdev team)`, session `4e01ec3a1b46`, model **radient/auto** (self-reported; disclosed
  fallback — see "Model" below)
- Revision: `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch `feat/i18n-es-wire-runtime`
- **The es catalogue is untracked at this revision** (`?? …/es/wire.settings.account_policy.json`); the audited bytes are
  pinned by sha256 `ce1a2c0185b94c9118b6555b3c57ba0f328fc87e9b3b47d6100c542156c6169a`
- Keys: 14 in `en`, 14 in `es`, 14 in the `en` context sidecar
- Independence: I authored none of this copy and repaired none of it. I edited no catalogue, sidecar, ledger, allowance
  or generated file. My only writes are this directory.

## What was verified

All **14 keys** were read side by side against the source and the sidecar. Mechanical checks (`parity.py.txt`,
`labels.py.txt`, `legacy_refs.py.txt`, all re-runnable from `run-checks.sh`): key sets identical, **no placeholders** in
any key, no code spans, no ICU arguments (the sidecar's "No ICU arguments" holds), no illegal plural category, no `!`.

**Enum identifiers and setting identifiers are unchanged.** The sidecar's rule is "Enum IDs are machine tokens and must
not be translated"; the enum ids live in the key suffixes (`reset_soonest`, `headroom`, `fair`, `legacy`) and the key set
matches the English exactly, with no value carrying a renamed token. `true`/`false` stay literal in `legacy.help`, as
they must.

**The `legacy` help names the same labels the Spanish labels use** — the coordinator's specific question, answered by
`legacy_refs.py.txt` rather than by eye:

| policy | es label | named verbatim in `legacy.help` |
|---|---|---|
| `reset_soonest` | `Reinicio más próximo` | yes |
| `headroom` | `Más margen` | yes |
| `fair` | `Reparto equitativo` | yes |

English behaves the same way (`Reset soonest` / `Most headroom` / `Fair spread` are all present), so the reference
mechanism is intact in both locales — no dangling reference to a control with a different name.

**Width against declared budgets:** every label key carries `maxCells: 29` and every one fits — `label` 20, `headroom`
10, `fair` 18, `reset_soonest` 20, `legacy` 27 (two cells of slack, the tightest in the namespace). No label clips.

**Behavioural promises** hold: the lease sentence still says a compatible unexpired lease *takes priority* and still
says it *does not guarantee a cache hit*; "an explicit policy overrides the legacy setting" survives
("Una política explícita anula…" / "An explicit … takes precedence"); "Reset … to restore Reset soonest and clear legacy
overrides" survives with both halves ("recuperar Reinicio más próximo y borrar las anulaciones heredadas").

**Terminology consistent across the locale:** `headroom` → `margen` matches `wire.settings.retry_usagereservepercent.help`
("Margen por debajo del cual…"); `quota` → `cuota`, `refill` → `recarga`, `tier` → `nivel` (the same call as
`effort tier` → `nivel de esfuerzo` and `tier 1` → `nivel 1`), `request` → `petición` (the dominant choice in the `wire.*`
and `tui.*` namespaces), `computer` → `máquina` (the dominant choice, and the one that avoids colliding with `equipo` =
*team*), `anulación` for *override* (as in `--yolo es una anulación explícita`).

## Findings

### es-w1-B01 — minor — the label drops "policy"

`wire.settings.account_policy.label`: en `account selection policy` (24 cells) → es `Selección de cuentas` (20 cells).
The word *policy* is gone. The row is a policy selector, and the Spanish help then speaks of `políticas`
("tiene prioridad en todas las políticas", "Una política explícita anula…") while the label the user sees names only
"account selection". The sibling settings label `wire.settings.providers_openrouter_sort.label` shows the locale's
normal answer for the same construction: `routing policy` → `política de enrutamiento`.

This is minor rather than major because no phrase points at a nonexistent thing (the `legacy.help` reference is
internally consistent with the label as written) and the value labels below it make the row's nature clear.

The 29-cell budget is the constraint: `Política de selección de cuentas` is 32 and would clip. Within budget:
`Política de selección` (21) keeps the concept and loses only "accounts", which the help and the three value labels
carry anyway. Recorded so the decision is deliberate; if the coordinator prefers the shorter row, `Selección de cuentas`
is defensible — but it should be a decision, not an accident.

### es-w1-B02 — nit — "una petición con éxito" is a calque of "a successful request"

`help`: `Una concesión compatible y sin vencer de una petición con éxito tiene prioridad…`. Understandable, but
`con éxito` is the literal rendering of *successful*; Spanish UI normally writes `una petición correcta` or
`una petición que ha tenido éxito`. Proposed: `Una concesión compatible y sin vencer de una petición correcta tiene
prioridad para todas las políticas; no garantiza un acierto de caché.`

### es-w1-B03 — nit — "concesión" for *lease* is a new term with no recorded decision

`Una concesión …` renders *lease*. The choice is defensible (a bounded grant of an account to a request) and it is used
once, consistently; it is not in `i18n/glossary.md` and no other es namespace uses it, so there is nothing to drift
against yet. Recommendation: record it in the es guide when the account-routing copy is next touched, so a later wave
does not silently pick `préstamo` or `cesión`.

## Disposition

**Pass.** Nothing here blocks or materially degrades the copy; the single minor is a label-vs-concept narrowing that a
future polish pass can fix inside the 29-cell budget. No remediation is required before merge.

Noted outside this namespace's findings count (measured, not this namespace's doing):

- *request* is rendered both `petición` (7 es files) and `solicitud` (9 es files) across the locale; this namespace uses
  `petición`, which matches `wire.settings.json` and `wire.errors.desktop.json`. The split needs a locale-wide decision,
  not a per-namespace fix.
- *computer* is `máquina` everywhere except `wire.errors.notifications.store_full` ("Este equipo se quedó sin espacio en
  disco"), where `equipo` also means *team* in this locale. Pre-existing, in a tracked namespace.

## Model

The session header names `anthropic/claude-sonnet-5-5` as the primary model. Three provider-failure switches fired on
`quota HTTP 429` during this session (`openai/gpt-6-astra` → `kimi/k3.1-pro` → `radient/auto`); everything in this
namespace was judged under **radient/auto**. This is a disclosed fallback, not the pinned primary.

## Limitations

- No native-speaker review is claimed; the auditor is an LLM.
- No rendered frame was captured; the width judgements here are cell counts against the sidecar's declared budgets,
  which is sufficient because every label fits 29.

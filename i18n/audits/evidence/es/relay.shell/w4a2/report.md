# Audit record — es / relay.shell / w4a2 (round 2)

- **Locale / namespace:** `es` / `relay.shell`
- **Audit id / round:** `w4a2` / 2 · **supersedes** `w4a1` · **scope: the delta keys only**
  (`usage.used`, `usage.remaining`, `usage.remaining_unknown_unit`,
  `usage.used_unknown_unit`, `usage.continue` — 5 of 44)
- **Audit date:** 2026-10-29
- **Audited revision:** `183f7a0537e4cee42fd82f25a21fe52953a81bdf` (branch
  `feat/i18n-es-wire-errors`). The target catalogue is still **untracked**
  (`?? local_operator/i18n/catalogues/es/relay.shell.json`); this round binds to
  the new on-disk bytes,
  `sha256 fdc0d458f441d6e2097ff0f69722025c1b6f7294089ad9886827ec0c2fd3fde2`.
  Previous round's bytes were `7bef1632…` — so the file did change.
- **Verdict: FAILED** — 0 blocker, 1 major (F1, not closed), 3 minor, 2 nit.
  The remediation fixed the agreement at 0/1/2/21/1.5e6 and introduced a
  regression at **exact multiples of 10⁶**.

## Method

Re-read the five delta values from the new bytes (not from the remediation
description), then rendered them through the **shipped composition**:
`usage-sheet.tsx` passes `{unit}` as the already-rendered
`unit_credits|unit_tokens|unit_requests` message with `count = measured`, and
`{value}` as `remaining|used` with the same `measured`. Counts rendered:
0, 1, 2, 21, 1 000 000, 1 500 000, plus 2 000 000 as an extra multiple of 10⁶.
The other 39 keys were re-rendered and compared against `w4a1`'s recorded output
as a neighbour check; `w4a1`'s file-level non-delta renders are the control.

Commands and output: `audit.log`; harness: `run-script.py.txt`. CLDR category
selection was cross-checked against Node `Intl.PluralRules('es')` — `agree=True`
(`many` at 1 000 000 and 2 000 000 only).

## Inputs (hashes of the audited bytes)

| input | path | sha256 |
| --- | --- | --- |
| en | `local_operator/i18n/catalogues/en/relay.shell.json` | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` (unchanged from w4a1) |
| context sidecar | `…/en/relay.shell.context.json` | `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea` (unchanged) |
| es (target, untracked) | `…/es/relay.shell.json` | `fdc0d458f441d6e2097ff0f69722025c1b6f7294089ad9886827ec0c2fd3fde2` |

## Finding status, measured

| id | round-1 severity | key(s) | status now | measured |
| --- | --- | --- | --- | --- |
| F1 | major | `usage.used` | **NOT CLOSED** | correct at 0/1/2/21/1.5e6; **wrong at 10⁶ and 2×10⁶** |
| F2 | minor | `usage.remaining` | **NOT CLOSED** | correct at 0/1/2/21/1.5e6; **wrong at 10⁶ and 2×10⁶** |
| F3 | minor | `usage.remaining_unknown_unit`, `usage.used_unknown_unit` | **NOT CLOSED** | correct at 0/1/2/21/1.5e6; **wrong at 10⁶ and 2×10⁶** |
| F4 | minor | `usage.continue` | **partially closed** | term drift closed; elided subject remains |
| F5 | nit | `usage.window_lifetime` | held, unchanged bytes | — |
| F6 | nit | `usage.eligibility_unknown` | held, unchanged bytes | — |

### What the fix got right

`{unit}` is passed through untouched — `argument_set_diffs=0` across every
branch, and `{value}`/`{unit}` are still the only arguments, so the
`keys.gen.ts` param shape (`value: number; unit: string`) is untouched. The
one/other agreement is now correct for both genders and for the plural:

```
credits  used  1 -> Se usó 1 crédito            (F1's round-1 defect: gone)
requests used  2 -> Se usaron 2 solicitudes
requests used 21 -> Se usaron 21 solicitudes
credits  remaining 1 -> Queda 1 crédito restante
requests remaining 21 -> Quedan 21 solicitudes restantes
```

### What the `many` branch broke

`unit_credits` / `unit_tokens` / `unit_requests` **already** carry their own
`many` branch, so at an exact 10⁶ `{unit}` arrives as `de créditos` — it is not
nominative. The new `many {… # de {unit}}` supplies a second `de`:

```
credits  used      1000000 -> Se usaron 1.000.000 de de créditos
credits  remaining 1000000 -> Quedan 1.000.000 de de créditos restantes
credits  used      2000000 -> Se usaron 2.000.000 de de créditos
requests remaining 1000000 -> Quedan 1.000.000 de de solicitudes restantes
unknown  remaining 1000000 -> Quedan 1.000.000 de restantes      (no noun to take the "de")
unknown  used      1000000 -> Se usaron 1.000.000 de usados      (same)
```

**Why a hand-run disagrees.** Rendering `usage.used` with a *nominative* unit
reproduces the reported string exactly — that is the control in `audit.log`:

```
used unit='créditos'    1000000 -> Se usaron 1.000.000 de créditos      ← reported
used unit='de créditos' 1000000 -> Se usaron 1.000.000 de de créditos   ← shipped
```

So the claim "1000000 → `Se usaron 1.000.000 de créditos`" is true only when the
unit message's own `many` output is bypassed. Through `usage-sheet.tsx` the unit
is always the rendered message, so the shipped phone row is the doubled one.

**Root cause, one line:** the `de` belongs to exactly one side of the
composition. It already lives in `unit_*`'s `many` branch, so the composing
messages must not add it.

**Tested remedy** (rendered, in `audit.log` under `CANDIDATE REMEDY`): drop the
`many` branch from the four composing keys and let `{unit}` carry the `de` —

```
relay.shell.usage.used                    {value, plural, one {Se usó # {unit}} other {Se usaron # {unit}}}
relay.shell.usage.remaining               {value, plural, one {Queda # {unit} restante} other {Quedan # {unit} restantes}}
relay.shell.usage.remaining_unknown_unit  {value, plural, one {Queda # restante} other {Quedan # restantes}}
relay.shell.usage.used_unknown_unit       {value, plural, one {Se usó #} other {Se usaron #}}
```

renders `Se usaron 1.000.000 de créditos`, `Quedan 1.000.000 de solicitudes
restantes`, `Quedan 1.000.000 restantes`, `Se usaron 1.000.000` at 10⁶ / 2×10⁶,
and is unchanged at 0/1/21/1.5e6 (all shown in the log). A `many` branch written
byte-identically to `other` would also render correctly but leaves a dead branch
that invites the same duplication again — not recommended.

### F4 — partially closed (minor, still held)

`Cuota incluida agotada` → `Asignación incluida agotada`: **closed**, the sheet
now names the included allowance one way, and `agotada` is kept as required.
Still held: `continuará con créditos` has an elided subject — it is not stated
what continues. Low impact on a banner, so it stays minor.

### F5 / F6 — held, unchanged bytes

`window_lifetime` = `Total` and `eligibility_unknown` = `Elegibilidad para gasto
desconocida` are byte-identical to round 1. I still hold both as nits; neither
gates.

## Neighbour check (scope discipline)

All 39 non-delta keys reproduce `w4a1`'s recorded renders exactly
(`reproduced=39 changed=0 unrecorded=0`), including the three `unit_*` messages
(unchanged, still `many {de créditos}`) and the row templates. The remediation
touched only the five declared keys.

## Limitations

- Python runtime only; the phone's TypeScript runtime (`intl-messageformat`) was
  not executed (no dependency tree in this worktree, and no scratch installs).
  The branch shape used here is standard ICU — `#`, branch-nested arguments and
  CLDR `many` are all supported by `intl-messageformat` — so this is a *text*
  defect, not a runtime-capability risk; but the phone render itself is unproven.
- No native-speaker review: the auditor is an LLM.
- No screen-width measurement.
- Fallback model disclosed; see `record.json` `auditor.model_fallback_notice`.

## Verdict

**FAILED** for `relay.shell` round 2. F1 (major) is **not closed**: the shipped
composition renders a doubled `de` at exact multiples of 10⁶, and F2/F3 carry the
same defect from the same cause. One tested fix (remove `many` from the four
composing keys) closes F1–F3 together; F4's remaining subject elision and the two
nits are non-gating. Re-audit scope for round 3: those four keys only, at
0/1/2/21/10⁶/1.5×10⁶/2×10⁶, with `{unit}` taken from `unit_*` as the SPA does.

# Audit record — es / relay.shell / w4a3 (round 3)

- **Locale / namespace:** `es` / `relay.shell`
- **Audit id / round:** `w4a3` / 3 · **supersedes** `w4a2` · **scope: the four agreement keys only**
  (`usage.used`, `usage.remaining`, `usage.remaining_unknown_unit`,
  `usage.used_unknown_unit`)
- **Audit date:** 2026-10-29
- **Audited revision:** `183f7a0537e4cee42fd82f25a21fe52953a81bdf` (branch
  `feat/i18n-es-wire-errors`). Target still **untracked**; this round binds to
  `sha256 1c7a98ad133f4a3f8339b6d465b7692806639bc278ffe5e2607863d8e36d9c2f`
  (round 2 was `fdc0d458f441d6e2097ff0f69722025c1b6f7294089ad9886827ec0c2fd3fde2`, so the file moved).
- **Verdict: PASSED.** 84/84 composed rows match an independently written
  grammar oracle; F1, F2 and F3 are **closed by measurement**. F4 stays
  partially closed (minor), F5/F6 stay held nits — none gates.

## Method

The four keys were re-read from the new bytes and rendered on the **composed
path the phone actually takes** (`usage-sheet.tsx`): the unit message is rendered
first with the same count, then the row message receives that string as `{unit}`.
The oracle is written **by hand from Spanish agreement and CLDR grouping**
(`crédito/créditos/de créditos`; `USD` takes no `de`) rather than from the
catalogue's own `unit_*` messages, so a wrong unit message cannot mask a wrong
row and vice versa.

Matrix: `credits`, `tokens`, `requests`, `usd`, `unknown`, `percent` ×
`remaining`/`used` × `0, 1, 2, 21, 1 000 000, 1 500 000, 2 000 000` = **84 rows**.

```
credits  remaining        1 -> Queda 1 crédito restante
credits  remaining  1000000 -> Quedan 1.000.000 de créditos restantes
credits  used       1000000 -> Se usaron 1.000.000 de créditos
credits  used       2000000 -> Se usaron 2.000.000 de créditos
credits  used       1500000 -> Se usaron 1.500.000 créditos
requests used             1 -> Se usó 1 solicitud
requests remaining   1000000 -> Quedan 1.000.000 de solicitudes restantes
usd      remaining  1000000 -> Quedan 1.000.000 USD restantes
unknown  remaining  1000000 -> Quedan 1.000.000 restantes
unknown  used       1000000 -> Se usaron 1.000.000
```

Regression substrings (`de de`, `de restantes`, `de usados`, `de USD`) are
**absent from all 84 rows**. The round-2 defect is gone.

## Finding status

| id | severity | status | basis |
| --- | --- | --- | --- |
| F1 `usage.used` | major | **closed** | composed rows correct at all 7 counts × 5 unit kinds; plural agreement and the 10⁶ "de" both correct |
| F2 `usage.remaining` | minor | **closed** | idem, including `Queda 1 crédito restante` |
| F3 `remaining_unknown_unit`, `used_unknown_unit` | minor | **closed** | `Queda 1 restante` / `Se usaron 1.000.000`; no dangling `de` |
| F4 `usage.continue` | minor | partially closed | term drift still closed (`Asignación incluida agotada`, `agotada` kept); `continuará` still has no stated subject — non-gating |
| F5 `window_lifetime` = `Total` | nit | held, unchanged bytes | non-gating |
| F6 `eligibility_unknown` = `Elegibilidad para gasto desconocida` | nit | held, unchanged bytes | non-gating |

## The `many`-as-alias decision — verified, and it supersedes my round-2 note

`many` is byte-identical to `other` in all four keys
(`many_is_alias_of_other=true`), every branch still renders the number, the
selector set is exactly `one/many/other` with **no illegal categories**, and the
`de` at 10⁶ comes from `unit_*` alone. Aliasing is the correct shape here rather
than dead weight: `scripts/i18n/check.py:582-588` states that when a locale
upgrades an en `number` to a `plural`, *"the translation's plural must carry
every CLDR category of the locale, every branch must render the count"*. es CLDR
emits `many` only at exact multiples of 10⁶, so the branch is reachable and must
exist.

**This supersedes my round-2 remark that such an alias was "not recommended".**
That remark predated reading the checker rule; with the rule known, the alias is
the way to satisfy it without double-supplying `de`. Recorded rather than
quietly dropped.

## Independent mechanical confirmation

I ran the checker myself instead of quoting the translator's run:

```
$ python3 scripts/i18n/check.py --catalogues-only
check.py: catalogue checks: 0 problem(s)          exit 0
```

(also appended to `audit.log`). It is recorded as the auditor's own execution,
not as a transcript of the implementer's.

## Neighbour check (scope discipline)

- 30 plain keys reproduce `w4a1`'s recorded text exactly (with `usage.continue`
  taken from `w4a2`'s recorded value); **0 differing**.
- 10 parameterised keys are covered by the matrix and by explicit re-renders.
- 5 row/timestamp templates re-rendered and match: `L · W: V`, `L (fam) · W: V`,
  `Se restablece el 11 oct 2026 a las 9:00`, `A fecha de 11 oct 2026, 9:00.`,
  `29 % usado`.
- The session sibling is untouched: `es/relay.session.json` is still
  `6812dca69e08ab9f1cf48f271e4db20263c7bbcb659d83a2f3c1a460122a2165`, i.e. the
  `w4a2` PASS still holds on the bytes on disk.

## Process note (for the audit trail)

In round 2 the session report for `relay.session` was first drafted from a
template and carried invented hashes and invented text. It was caught against
the log and rewritten **before** the bundle was receipted; the on-disk `w4a2`
session report is the rewritten, measurement-derived version and its
`receipt.json` binds that version. No `relay.shell` document was affected, and
every figure in all three relay.shell rounds is reproducible from the
corresponding `audit.log`. Recorded because a later reader deserves to know the
numbers were recomputed rather than carried.

## Limitations

- Python runtime only; the phone's TypeScript runtime (`intl-messageformat`) was
  not executed — no dependency tree in this worktree and no scratch installs.
  The exposure is that runtime's CLDR table, which agrees with Node
  `Intl.PluralRules('es')` on the sampled counts.
- No native-speaker review: the auditor is an LLM.
- No screen-width measurement; no `maxCells` is pinned for this namespace.
- Fallback model disclosed (`radient/kimi-k2.5`, primary quota-failed).

## Verdict

**PASSED** for `relay.shell` round 3. The four agreement keys now render correct
Spanish on the composed path at every sampled count and unit kind, including the
exact millions that broke round 2, and no non-delta key moved. At the current
bytes the namespace carries **no blocker and no major**; F4's subject elision and
the two nits are recorded and non-gating. `relay.session` needs nothing further —
`w4a2` already passed it and its bytes are unchanged.

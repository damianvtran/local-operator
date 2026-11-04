# Audit record — es / relay.shell / w4a1

- **Locale / namespace:** `es` / `relay.shell`
- **Audit id / round:** `w4a1` / 1
- **Audit date:** 2026-10-29
- **Audited revision:** `183f7a0537e4cee42fd82f25a21fe52953a81bdf`
  (branch `feat/i18n-es-wire-errors`). The **target catalogue is untracked**
  (`?? local_operator/i18n/catalogues/es/relay.shell.json`); this audit is bound
  to the on-disk bytes, `sha256 7bef1632cf4b503116725d409ebd3567213cf9322897672d188a3ff2ba602b7f`,
  and not to a commit.
- **Keys read:** 44 of 44 (`en` 44 / `es` 44, key sets identical).
- **Verdict: FAILED** — 1 major, 3 minor, 2 nit. No blocker.

## Method

The English source, its context sidecar and the Spanish catalogue were read
side by side, key by key, across all 44 keys. Every message was then **rendered
with the repository's own runtime** — `local_operator.i18n.runtime`
(the in-house ICU subset) over the generated CLDR tables in
`local_operator.i18n.format` — and the plural-bearing keys were rendered through
the **exact composition path of the phone surface**, `local_operator/mobile/web/src/components/usage-sheet.tsx`:
`{unit}` is itself the rendered `unit_credits|unit_tokens|unit_requests` message
with `count = measured`, and `{value}` is `remaining|used` with the same
`measured`. Counts rendered: **0, 1, 2, 21, 1 000 000, 1 500 000**, across the
credits / tokens / requests / USD / percent / unknown-unit unit kinds and both
the remaining and used directions (72 composed samples, in `audit.log`).

Commands and their real output are in `audit.log` (produced by execution, not
transcribed); the run script is preserved as `run-script.py.txt`.

## Inputs (hashes of the audited bytes)

| input | path | sha256 |
| --- | --- | --- |
| en | `local_operator/i18n/catalogues/en/relay.shell.json` | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` |
| context sidecar | `local_operator/i18n/catalogues/en/relay.shell.context.json` | `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea` |
| es (target, untracked) | `local_operator/i18n/catalogues/es/relay.shell.json` | `7bef1632cf4b503116725d409ebd3567213cf9322897672d188a3ff2ba602b7f` |

## Verified, and clean

**Connection / state promises — no change.** The sidecar defines four distinct
states and the copy keeps them apart:

| key | en | es | state kept |
| --- | --- | --- | --- |
| `list.connecting` | connecting… | conectando… | list stream pre-first-frame |
| `list.empty` | no sessions running | no hay sesiones en ejecución | list connected, zero rows |
| `session.connecting` | connecting to session… | conectando con la sesión… | projection stream not connected |
| `session.waiting` | waiting for projection… | esperando la vista de la sesión… | connected, no frame yet |

`session.waiting` correctly follows the sidecar's instruction to translate the
*intended meaning* ("waiting for the session view"; projection = daemon jargon)
rather than calque "proyección", and the connecting/waiting distinction survives
in both wording and shape. Wall-clock ellipsis is U+2026 in all four
ellipsis-bearing keys (`list.connecting`, `session.waiting`, `session.connecting`,
`usage.loading`).

**Usage-sheet states — no change.** `usage.cached` (no provider refresh is
requested) → "No se solicita una actualización al proveedor."; `usage.unavailable`
(cached usage unavailable) → "Uso en caché no disponible."; `usage.loading`;
`usage.empty`; `usage.last_known` → "Último uso conocido; el uso actual no está
disponible." — all four remain distinguishable. `usage.blocked` ("Spend
unavailable") stays a *blocked* statement ("Gasto no disponible") while
`usage.eligibility_unknown` stays an *unknown* one ("Elegibilidad para gasto
desconocida"), i.e. unknown is still not reported as blocked, and
`usage.amount_unknown` ("Amount not reported") is still "Importe no informado",
not zero. `usage.unlimited` keeps the scope of the promise ("Asignación de pago
ilimitada" — the paid allowance, not the account). `usage.stale` stays hedged
("Puede estar desactualizado.").

**`usage.continue` — the strength increase is correct.** es says "agotada"
(exhausted) where en says "used". `usage-sheet.tsx` renders this banner only
when `limit.exhausted` is true for a shared, non-paid-fallback, non-family-cap
limit, so "agotada" is the accurate reading and is **not** a finding; do not
soften it back to "usada".

**Plural structure — correct, including `many`.** The three unit keys add a
`many` branch (`many {de créditos}`), which is exactly the es category set
(`one`/`many`/`other`, `many` for exact multiples of 10⁶ only) and renders
`1 000 000 → "de créditos"`, `2 000 000 → "de créditos"`, `1 500 000 → "créditos"`.
`illegal_for_es: []` — no `few`/`zero`/`two` anywhere. The three ICU
"signature diffs" in `audit.log` are this sanctioned `many` addition and are
the intended state, not a defect. Number formatting is the formatter's:
`1.000.000`, `1,5`, `29 %` (es spacing and grouping correct).

**Placeholders and machine tokens — intact.** No placeholder was renamed,
dropped, reordered or re-suffixed; `{value}` used twice in `usage.reset` and
`usage.as_of` still carries date+time formats respectively; `usage.row` and
`usage.row_family` are byte-identical to source (`{label} · {window}: {value}`,
`{label} ({family}) · {window}: {value}`), which the sidecar requires ("own the
field order and punctuation… reorder placeholders"), and rendering them with
sentinel values returns the separators unchanged (`L · W: V`, `L (fam) · W: V`).
`usage.unit_usd` = `USD`, byte-identical.

**Register.** tú-address and infinitive action labels are consistent with the es
style guide ("elegir tema", "Mostrar el uso del proveedor en caché"); sentence
case only; no "!"; no region-marked vocabulary; 24-hour times
(`14:30 → "14:30"`); cached-usage copy uses the amended `caché` (§ style guide
amendment es W1) rather than `cache`, matching the shipped `wire.settings`
bytes.

## Findings

### F1 — **major** — `relay.shell.usage.used`

- **Current:** `{value, number} {unit} usados`
- **Proposed:** `{value, plural, one {Se usó # {unit}} other {Se usaron # {unit}}}`
  (any construction that makes agreement with a *variable* label impossible is
  acceptable; see the note below)
- **Reason:** the participle is fixed masculine plural while `{unit}` is neither
  fixed in gender nor in number, so the shipped row is ungrammatical for the
  whole `requests` family (feminine `solicitud`) at *every* count, and for every
  unit kind at exactly 1.

Rendered through `usage-sheet.tsx`'s own path:

```
requests used        0 -> 0 solicitudes usados      (should be usadas)
requests used        2 -> 2 solicitudes usados      (should be usadas)
requests used       21 -> 21 solicitudes usados     (should be usadas)
requests used  1000000 -> 1.000.000 de solicitudes usados   (should be usadas)
credits  used        1 -> 1 crédito usados          (should be usado)
usd      used        1 -> 1 USD usados              (should be usado)
```

The sidecar for this key explicitly permits the fix and names this exact hazard:
*"{unit} is a localized measurement label; rephrase/reorder the complete sentence
as needed to avoid forced gender/plural agreement."* The proposed ICU is verified
to render correctly at all six counts for all three gendered unit labels and for
`USD` (`Se usó 1 crédito`, `Se usaron 21 solicitudes`,
`Se usaron 1.000.000 de solicitudes`, `Se usaron 0 USD`).

**Grading, stated openly.** This is the one call that decides the namespace
verdict. It is graded *major* because the shipped string is wrong rather than
polishable later (an agreement error is not a style preference), and because it
contradicts an explicit sidecar instruction. If the project reads its severity
ladder strictly — "major = changes what the user does" — this becomes a minor
and `relay.shell` is PASS. The rendered evidence and the fix are identical
either way; only the gate outcome moves.

### F2 — **minor** — `relay.shell.usage.remaining`

- **Current:** `{value, number} {unit} restantes`
- **Proposed:** `{value, plural, one {Queda # {unit}} other {Quedan # {unit}}}`
  (or `{value, plural, one {# {unit} restante} other {# {unit} restantes}}`)
- **Reason:** `restante` does not inflect for gender, so only the number needs
  branching — but it does need it: at exactly 1 the row reads "1 crédito
  restantes" / "1 token restantes" / "1 USD restantes".
- **Evidence (`audit.log`):** `credits remaining 1 -> 1 crédito restantes`,
  `usd remaining 1 -> 1 USD restantes`.

### F3 — **minor** — `relay.shell.usage.remaining_unknown_unit`, `relay.shell.usage.used_unknown_unit`

- **Current:** `{value, number} restantes` / `{value, number} usados`
- **Proposed:** `{value, plural, one {Queda # restante} other {Quedan # restantes}}`
  and `{value, plural, one {Se usó #} other {Se usaron #}}`
- **Reason:** with the unit deliberately omitted (sidecar: never invent a unit
  token) there is no noun to carry the plural, so exactly 1 renders "1
  restantes" / "1 usados".
- **Evidence:** `unknown remaining 1 -> 1 restantes`, `unknown used 1 -> 1 usados`.

### F4 — **minor** — `relay.shell.usage.continue`

- **Current:** `Cuota incluida agotada; continuará con créditos.`
- **Proposed:** `Asignación incluida agotada; el uso continuará con créditos.`
- **Reason:** the same sheet names the same allowance `Asignación incluida`
  (`usage.included`) and `Asignación de pago`/`Asignación por familia de modelos`
  — three "Asignación" rows — so "Cuota incluida" is a *second* rendering of one
  product term inside one namespace, and the elided subject of "continuará"
  leaves it open what continues. `Cuota` is the correct rendering of en *quota*
  in `wire.settings.account_policy.headroom.description` ("cuota restante"), so
  the split mirrors an English inconsistency; the sheet-level drift is still
  worth closing on one term.

### F5 — **nit** — `relay.shell.usage.window_lifetime`

- **Current:** `Total`
- **Proposed:** `Histórico` (or `De por vida` if the width budget allows)
- **Reason:** every sibling window label is a duration or period ("5 horas",
  "7 días", "Mensual"), and `{window}` is substituted into
  `{label} · {window}: {value}`; "Asignación incluida · Total: 1.000.000 de
  créditos restantes" can be read as the aggregate of the value rather than as
  the window.

### F6 — **nit** — `relay.shell.usage.eligibility_unknown`

- **Current:** `Elegibilidad para gasto desconocida`
- **Proposed:** `Elegibilidad de gasto desconocida`
- **Reason:** "para gasto" reads as purpose ("eligibility *for* spending
  something"); the compound "elegibilidad de gasto" is the tighter and more
  usual collocation for a property of spend. Meaning is unaffected.

## Considered and cleared (not findings)

- **`usage.paid` "Asignación de pago"** — "de pago" reads as "paid" in Spanish
  ("contenido de pago"), and in contrast with "Asignación incluida" the
  paid/included contrast is unambiguous. Left as the translator chose.
- **`usage.empty` "No hay datos de uso en caché."** vs en "No cached usage
  reported." — the reporting subject is dropped, but the practical statement is
  the same and the sibling keys already carry the "not live" framing. Not worth
  a change.
- **`usage.fraction` / `used_percent` "{value, number, percent} usado"** —
  masculine singular is correct here: it agrees with *el porcentaje*, and
  `29 % usado` is the standard es form. `remaining_percent` → "restante" is
  invariant and also correct.
- **`usage.reset`** adds the article and preposition Spanish needs
  ("Se restablece el {date} a las {time}"); en's bare comma construction cannot
  be mirrored. Correct, not drift.
- `usage.title` = "uso" and the session namespace's sheet titles staying
  lowercase mirror the English source's casing; the sidecars ask for casing to
  be preserved, and it is preserved.

## Noted outside this namespace (not counted, not audited here)

`es/relay.account_policy.json` is another lane's untracked file and is outside
this audit. One cross-surface check the wave asked for is still worth flagging:
`relay.account_policy.title` renders **"Ajustes del equipo"** for en *"Computer
settings"*, while the shipped core settings surface names this concept
`Política de cuentas` (`wire.settings.account_policy.label`, tracked at HEAD).
"equipo" is this locale's glossary term for *team*, so the relay title both
invents a second rendering and collides with another metal term. Reported for the
owning lane; not scored here.

## Limitations

- Rendered with the repository's Python runtime (the shipped ICU subset and the
  generated CLDR tables, golden-probed against Node `Intl`); the phone client's
  TypeScript runtime was not executed. Both runtimes are parity-tested in-repo,
  so the plural selection and formatting shown here are the same, but this is
  textual-plus-Python evidence, **not** a rendered-frame capture on a device.
- No native-speaker review: the auditor is an LLM. No emulator/screen-width
  measurement was taken; overflow was judged against the sidecars' word budgets
  only (no `maxCells` is pinned for this namespace).
- Readings of "would a native write this" are LLM judgements and are recorded as
  such; the F1/F2/F3 agreement findings are mechanical and reproducible from
  `audit.log`.

## Verdict

**FAILED** for `relay.shell` — 1 major (F1, agreement of `usage.used` against a
variable unit label), 3 minor (F2, F3, F4), 2 nit (F5, F6). Meaning, state
promises, placeholder structure, machine tokens and es plural categories are
otherwise correct across all 44 keys. A re-audit after remediation should scope
to F1–F4 plus the four rendered neighbours (`usage.unit_*`, `usage.row`,
`usage.row_family`) and confirm at counts 0/1/2/21/1e6/1.5e6 and for the
feminine `requests` label.

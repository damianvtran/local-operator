# Audit — es / relay.account_policy — round 1 (`w4a2`)

**Verdict: FAILED** — 1 major, 1 minor, 1 nit. 13 of 13 keys read (en 13 / es 13 / sidecar 13).

| input | sha256 | tracked |
|---|---|---|
| en `relay.account_policy.json` | `0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890` | yes @ `183f7a0537e` |
| es `relay.account_policy.json` | `074787ece57a27646c766deb3b7691979c233259c1f05987c5c79770e8f31204` | **no — untracked** |
| en `relay.account_policy.context.json` | `7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4` | yes @ `183f7a0537e` |

## What this surface is

The phone-client account-routing screen: a title, a back control, a loading line, a save control
with its in-flight and success states, a retry, and four message states — capability unavailable,
unconfirmed write, load failure, and an explicit rejection. The sidecar pins the role as "Account
routing screen shared phone-client copy", tone plain/neutral, and attaches the one semantic
constraint that matters here: for `error`, "commit status is UNKNOWN, so do not assert that the
server did not save"; for `rejected`, HTTP 4xx "is known not saved"; and "initial load failure is
distinct from an unconfirmed write". No `maxCells` is declared (0 across all 13 keys).

## The state promises all hold

This is the part that could have gone badly and did not. Read against the sidecar, not just
against the English:

- `error` → "No se pudo confirmar el cambio. Comprueba la conexión y recarga para ver la política actual."
  keeps *unconfirmed* and does **not** assert the write failed. Correct.
- `rejected` → "No se guardó el cambio." **does** assert not-saved, which the sidecar licenses for a
  4xx, and it stays lexically distinct from `error`. Correct.
- `load_error` → "No se pudo cargar el enrutamiento de cuentas." is a load failure, not a write
  outcome. Correct.
- `unavailable` → a capability/version message naming the remedy (update it on the computer), not a
  transport failure. Correct.
- `saving` / `saved` → in-flight then terminal, no over-claim. Correct.

No connecting / reconnecting / offline / failed / allowance / activated promise was altered.
`enrutamiento de cuentas` is identical to the shipped `wire.settings.account_policy.error.unavailable`,
and `ajustes` matches the style guide's settled rendering of *settings*.

## Account-policy labels: no second rendering

The brief's specific check — that this surface must not invent a second rendering of the labels the
core settings screen already ships in `es/wire.settings.account_policy.json` — **passes on the
bytes**, verified mechanically rather than by eye. None of `Política de cuentas`,
`Reinicio más próximo`, `Más margen`, `Reparto equitativo`, `Elección heredada según uso` occurs in
any of the 13 values, and no policy-option stem (`Reinicio`, `margen`, `Reparto`, `equitativo`,
`heredada`, `cuentas hermanas`) occurs either. This namespace carries chrome only; the option names
are simply not re-translated here. That is the right outcome.

## Findings

### es-w4a2-AP-01 — **major** — `computer` rendered as `equipo` (5 keys)

`title`, `saved`, `unavailable`, `scope`, `rejected`.

| key | current | proposed |
|---|---|---|
| title | Ajustes del equipo | Ajustes de esta máquina |
| saved | Guardado en este equipo. | Guardado en esta máquina. |
| unavailable | … Actualiza Local Operator en este equipo. | … Actualiza Local Operator en esta máquina. |
| scope | Los cambios solo se aplican a este equipo. | Los cambios solo se aplican a esta máquina. |
| rejected | … o los ajustes de este equipo … | … o los ajustes de esta máquina … |

This locale binds **`equipo` to team** — glossary line 63 (`team | translate | … equipo`) and
`style-guides/es.md` line 29 (`equipo (team)`). Meanwhile *computer* ships as **`máquina`** in 11 of
the 17 English values that mention it, seven of them inside this very phone UI
(`relay.health.live_detail`, `reconnecting_detail`, `dropped_detail`, `quiet_detail`,
`relay.chrome.list.empty_hint`, `relay.notifications.error.unreachable`,
`relay.notifications.detail.missing.body`). The shipped sibling for this exact setting family,
`wire.settings.account_policy.error.write_failed`, also says "en esta máquina".

So on one screen the app renames the object the screen is about, and the screen's central promise —
"changes apply only to this computer" — becomes readable as "changes apply only to this team". Those
are different scope promises, and teams are a first-class feature in this product
(`cli.teams_help`, `tui.team_authoring`, `tui.palette.kind_team`). A user who reads the title as
"Ajustes de equipo / team settings" may treat account routing as an admin-level team setting rather
than a local one, which is the opposite of what the screen guarantees.

**Why major rather than blocker or nit.** Not a blocker: nothing is machine-broken, no message lies
about state, and `equipo` *is* a correct Spanish word for a computer (`equipo informático`) — a
reader who has never seen `lop` would not stumble. Not a nit: this is not a preference, it collides
with a shipped mapping and changes how the scope sentence reads. It sits squarely in "term use that
changes what the user does".

**Direction for the fix.** Either (a) `equipo` → `máquina` in these five values, matching the eleven
other English `computer` strings already shipped — cheap, consistent, no other file moves; or
(b) standardise on `equipo` for "computer" across the locale, which would additionally require
rewriting those eleven shipped values (seven of them in `relay.*`) plus
`wire.errors.notifications.store_full`, and would still collide with the glossary's
`team → equipo`. (a) is the only fix that does not re-open a settled decision.

### es-w4a2-AP-02 — **minor** — `relay` left untranslated

`unavailable`: "… no están disponibles en este relay. …" → "… no están disponibles en este relé. …"

Glossary line 59 records `relay → relé` for es, and the shipped catalogue already writes "este relé"
(`wire.settings.network_max_handshakes.help`) and "relé" (`cli.services_help`). This is the only es
value in the whole catalogue that keeps the English noun in prose. Minor: `relay` is transparent to
a developer audience, so comprehension survives; it is drift, not a misstatement. (Distinct from the
identifier case — `relay.*` namespace keys and any `--flag` stay byte-identical and are not at issue.)

### es-w4a2-AP-03 — **nit** — `save` shortens the shipped noun

`save`: "Guardar política" vs the shipped label `wire.settings.account_policy.label` =
"Política de cuentas", whose help text uses "política de cuentas" repeatedly.

The English source says "Save policy", not "Save account policy", so the Spanish is faithful and the
button is unambiguous next to the policy controls. Recorded only so the surface's noun can be aligned
if a reviewer wants one noun per screen. Preference, not a defect.

## Verdict

**FAILED.** The chrome is clean and every state promise is intact, but five values rename the device
the screen governs (`equipo` = team in this locale) against eleven shipped `máquina` siblings, and
the scope sentence consequently reads as a team-scoped promise.

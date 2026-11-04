# Audit — es / relay.account_policy — round 2 (`w4a3`)

**Verdict: PASSED** — the major is closed by measurement. 1 minor and 1 nit remain open, both non-gating.
13 of 13 keys read this round (delta scope: the 5 remediation keys, plus the other 8 re-read as neighbours).

| input | sha256 | tracked |
|---|---|---|
| en `relay.account_policy.json` | `0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890` | yes @ `183f7a0537e` |
| es `relay.account_policy.json` | `67dbbf6a3bc75058b38f40a076de852c86c7d58004e938b43019513280e2a407` | **no — untracked** |
| en `relay.account_policy.context.json` | `7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4` | yes @ `183f7a0537e` |

Superseded (`w4a2`) target: `074787ece57a27646c766deb3b7691979c233259c1f05987c5c79770e8f31204`. Round 2 supersedes `w4a2` for these five keys only:
`title`, `saved`, `unavailable`, `scope`, `rejected`.

## The delta is exactly what was claimed

Diffed against the superseded values preserved in my own `w4a2` record (six) and the `w4a2`
session read (the remaining seven), so all thirteen are accounted for:

| key | was | now |
|---|---|---|
| title | Ajustes del equipo | Ajustes de la máquina |
| saved | Guardado en este equipo. | Guardado en esta máquina. |
| scope | Los cambios solo se aplican a este equipo. | Los cambios solo se aplican a esta máquina. |
| unavailable | … Actualiza Local Operator en este equipo. | … Actualiza Local Operator en esta máquina. |
| rejected | … o los ajustes de este equipo … | … o los ajustes de esta máquina … |

Eight keys are character-for-character unchanged: `back`, `loading`, `save`, `saving`, `retry`,
`error`, `entry`, `load_error`. The five that changed are exactly the five the implementer named —
no collateral edits, no format churn (still 2-space JSON, trailing newline, no CRLF, no tabs).

## `es-w4a2-AP-01` — **closed**

Verified four ways, all re-run by `checks.py`:

1. `grep -in equipo` on the file returns nothing.
2. A value-only census of every es catalogue finds `equipo` in 13 namespaces, and
   `relay.account_policy` is not one of them.
3. The delta is exactly the five claimed keys.
4. **No over-reach in the other direction.** The team sense is intact and unchanged in count:
   `cli.teams_help` 26, `tui.team_authoring` 14, `wire.errors.agents` 6, `wire.settings` 5,
   `wire.settings.hub_defaults` 3, `wire.errors.desktop` 2, and one each in `cli.exec_help`,
   `cli.hub_help`, `tui.palette`, `wire.memory`, `wire.slash`, `wire.exec` — every one a team
   sense, and none of the team namespaces contains `máquina`. The rewrite took the computer
   strings and left the team vocabulary alone, which is precisely the split I asked for.

The two constrained state messages still hold after the edit: `error` says the change could not be
**confirmed** and does not claim failure; `rejected` asserts not-saved. The noun swap touched neither.

**Accepted deviation.** I proposed `Ajustes de esta máquina` for the title; the implementer wrote
`Ajustes de la máquina`. Accepted without further comment: a static screen title reads better with
the definite article, and the scope sentence directly below carries the deictic already
("Los cambios solo se aplican a esta máquina").

**One residual oddity, not this wave's doing.** `wire.errors.notifications.store_full` still renders
*computer* as "Este equipo" ("Este equipo se quedó sin espacio en disco"). It was recorded as a
pre-existing outlier at `w4a2` and is outside this remediation; with five keys now on `máquina` it is
the last 'computer'-as-`equipo` value in the catalogue. Worth a separate one-line fix someday, not a
finding against this round.

## `es-w4a2-AP-02` — **still open** (minor)

`unavailable` still reads "… no están disponibles en este **relay**. …". Glossary line 59 gives
`relay → relé` and the shipped catalogue already writes "este relé" in
`wire.settings.network_max_handshakes.help`. This was not part of the remediation you described, so
I am reporting it open rather than closing it — it is the only remaining es value in the catalogue
that keeps the English noun in prose. Non-gating: the sentence is comprehensible either way.

## `es-w4a2-AP-03` — **still open** (nit)

`save` still reads "Guardar política" against the shipped label "Política de cuentas". The English
is "Save policy", so this stays faithful and unambiguous; kept as a nit as you asked rather than
dropped.

## Width after the edit

| key | en | es | delta |
|---|---|---|---|
| title | 17 | 21 | +23.5% |
| saved | 23 | 25 | +8.7% |
| unavailable | 95 | 116 | +22.1% |
| scope | 36 | 43 | +19.4% |
| rejected | 84 | 95 | +13.1% |

No `maxCells` is declared in this sidecar. The title grew by 3 cells and still fits comfortably as a
screen heading; nothing approaches a truncation risk.

## Verdict

**PASSED.** The major that failed round 1 is closed on the bytes, the fix did not over-reach into the
team sense, and the namespace's state promises are unchanged. One minor and one nit remain open and
do not gate.

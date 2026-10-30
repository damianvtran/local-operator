# Audit — es / relay.account_policy — round 4 (`w4a4`)

**Verdict: PASSED.** Scope: `relay.account_policy.unavailable`, one key. AP-02 is closed; AP-01 stays
closed; the AP-03 nit remains open and does not gate. 13 of 13 keys read.

| input | sha256 | binding |
|---|---|---|
| en `relay.account_policy.json` | `0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890` | @ `91b6984e8531` |
| es `relay.account_policy.json` | `71956128095ae4afebaf4aa26356db6acf9796a100332d503fbcc785ab178453` | **tracked** @ `91b6984e8531` |
| en `relay.account_policy.context.json` | `7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4` | @ `91b6984e8531` |

**The binding changed materially this round.** The file is no longer untracked: `git ls-files`
lists it, it was introduced by `b26f3276c32d802a2cdbcd7ff4f8365d743cdb38`, and HEAD's blob hashes to
the same `71956128095ae4afebaf4aa26356db6acf9796a100332d503fbcc785ab178453` as the working tree. So this record binds a **revision and a blob**, not just
on-disk bytes — the ledger row can cite `91b6984e8531bf01520690fa1196a3b8287a3556` with confidence.
Supersedes `w4a3`, whose target was the untracked `67dbbf6a3bc75058b38f40a076de852c86c7d58004e938b43019513280e2a407`.

## 1. `relé` is in the shipped bytes, and no English noun remains

`relay.account_policy.unavailable` now reads:

> Los ajustes de enrutamiento de cuentas no están disponibles en este **relé**. Actualiza Local
> Operator en esta máquina.

Verified in the bytes rather than from your description:

- `relé` is present with **U+00E9**, NFC-stable — an acute *e*, not a combining sequence and not the
  accentless `rele`.
- A **value-only** scan (JSON keys excluded, so the `relay.account_policy.*` namespace names cannot
  produce a false hit) finds **no value matching `/\brelay\b/`** in the whole namespace.
- The term now agrees with the rest of the catalogue, which is what the finding asked for:
  `wire.settings.network_max_handshakes.help` "Conexiones sin autenticar que **este relé** mantendrá a
  la vez", plus `wire.settings.network.description` and `cli.services_help.services_description`.
  It is no longer the odd one out.

## 2. The sentence still reads correctly

Read as a native reader would, at phone width, mid-task:

- Two clauses, two referents, no collision: **`este relé`** is where the settings are unavailable;
  **`esta máquina`** is what the user must update. The remedy clause is still an instruction
  addressed to *tú* (`Actualiza`), matching the register of the other twelve keys.
- **`máquina` survived** untouched — the round-2 fix is intact inside this very string, and the two
  nouns now sit side by side without either stealing the other's sense. If anything the fix
  *reduced* ambiguity: before it, an English noun sat in a Spanish sentence between two Spanish
  nouns; now all three terms are Spanish and each has its dictionary sense.
- No new ambiguity is introduced by `relé`. Spanish `relé` is the standard rendering of a relay
  (and the glossary's own decision, line 59); in this sentence it is disambiguated by `en este relé`
  and by the neighbouring `máquina`, which is the other end of the connection.
- Punctuation and structure unchanged: two sentences, full stop, `…` untouched elsewhere, no `!`.

**Width:** 115 cells against English 95 (+21.1%), one cell *shorter* than the `relay` form it
replaced. No `maxCells` in this sidecar; 115 is well inside the shipped `relay.*` range.

## 3. The other twelve keys, positively

This is the part you asked for explicitly, so it is stated per key rather than left implicit. The
shipped file is **modelled** from the `w4a2` values plus the two documented transformations and the
model is asserted equal to the file for all thirteen:

```
w4a2 values
  -> round 2: "del equipo"/"este equipo" -> "de la máquina"/"esta máquina"
              in title, saved, scope, rejected, unavailable
  -> round 4: "este relay" -> "este relé"   in unavailable
== the shipped file, all 13 keys          (asserted byte for byte)
```

| key | state since `w4a3` |
|---|---|
| title, back, loading, save, saving, saved, retry, error, scope, entry, rejected, load_error | **unchanged** — reproduce their `w4a3` text exactly |
| unavailable | **changed** — and only by `relay` → `relé`; every other character identical |

One honest wrinkle, recorded so it is not mistaken for a delta: my `w4a3` record stored `rejected` in
an ellipsised form (`… o los ajustes de esta máquina …`), so a naive diff against that record reports
`rejected` as changed. It is not — the `w4a2` record holds the full string, and the substitution
reproduces the shipped text exactly. `checks.py` therefore diffs against the model, not against the
abbreviated map.

## 4. Findings on the new bytes

| id | severity | status | note |
|---|---|---|---|
| `es-w4a2-AP-01` | major | **closed** (`w4a3`) | all five keys still `máquina`; no `equipo` anywhere |
| `es-w4a2-AP-02` | minor | **closed** (`w4a4`) | `relé` in place, no English noun left, term agrees with shipped strings |
| `es-w4a2-AP-03` | nit | **open** | `save` = "Guardar política" vs shipped "Política de cuentas" |

Nothing else is open on these bytes. The two constrained promises still hold: `error` says the change
could not be **confirmed** and never claims failure; `rejected` asserts not-saved. Register is `tú`
throughout with no `usted`/`vosotros`.

AP-03 is a nit the implementer elected to leave; that is a reasonable disposition and needs no
further round. If it is ever picked up, the change is one word: "Guardar política de cuentas".

## Ledger binding

**Bind `71956128095ae4afebaf4aa26356db6acf9796a100332d503fbcc785ab178453`** at revision
`91b6984e8531bf01520690fa1196a3b8287a3556` (blob introduced in `b26f3276c32d`).

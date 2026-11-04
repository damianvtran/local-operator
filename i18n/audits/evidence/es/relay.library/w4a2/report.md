# Audit — es / relay.library — round 1 (`w4a2`)

**Verdict: PASSED** — 0 blocker, 0 major, 1 minor, 0 nit. 1 of 1 keys read (en 1 / es 1 / sidecar 1).

| input | sha256 | tracked |
|---|---|---|
| en `relay.library.json` | `ad478c433a4db3e725994766c724a2e8e92176f015f8283e71dc13b87185466a` | yes @ `183f7a0537e` |
| es `relay.library.json` | `f2d5ab3a9a5660f94b7656324a4c1069c06d88c31e908a5df1226872cb7d1a71` | **no — untracked** |
| en `relay.library.context.json` | `877b88290cf14602e63b2bb1c58719fac96a20ae53ff04b147e21de23df93c54` | yes @ `183f7a0537e` |

## What this surface is

The Library grid result count for the phone SPA — one ICU plural. The sidecar pins the role
("Library grid result count"), the register ("lowercase, numeric; the plural is grammatical, not an
abbreviation"), and one fact that matters for reading the string: "The number is the server's
filtered total, not the number of tiles currently painted." The count is therefore a factual total,
and the copy makes no claim about what is on screen.

| | |
|---|---|
| en | `{count, plural, one {# asset} other {# assets}}` |
| es | `{count, plural, one {# recurso} other {# recursos}}` |

## What is right

`count` is byte-identical to the source. `#` is used in both branches and is never pre-formatted, so
the formatter owns the digits and the es `1.234` grouping will appear without the translator
hand-writing it. Lowercase throughout, matching the sidecar. `asset` → `recurso` is the term the
shipped Library surface already uses (`wire.slash.library.help`: "Explora la Biblioteca global de
recursos"), so the feature speaks one vocabulary. The syntax parses; there is no `few`, `zero` or
`two`.

## Finding

### es-w4a2-LB-01 — **minor** — the `many` branch is missing

```
current:  {count, plural, one {# recurso} other {# recursos}}
proposed: {count, plural, one {# recurso} many {# de recursos} other {# recursos}}
```

Spanish CLDR has a `many` category English lacks — exact multiples of 10^6 — and
`style-guides/es.md` prescribes exactly this shape (`many {# de sesiones}`). The mechanical evidence
is the strongest part of the finding: of the **58** plural messages in the es catalogues, **57**
carry `one`+`many`+`other` and exactly **one** carries only `one`+`other` — this key. The English
source carries `one`+`other` in all 105 of its plural messages, so `many` is a deliberate es-side
addition that every other translator applied and this one did not.

The consequence is narrow but real: at a filtered total of exactly 1.000.000, ICU falls back to
`other` and renders "1.000.000 recursos" where Spanish requires "1.000.000 de recursos". A Library
holding a million assets is unlikely; the branch costs one clause and removes the outlier.

The string still renders correctly for every plausible count, which is why this is a minor and not a
blocker: the syntax is intact, no placeholder is lost, and no state promise is involved.

## Width

51 cells against an English 47 (+8.5%) — inside the 15–25% band the guide expects. No `maxCells` is
declared in this sidecar.

## Verdict

**PASSED** with one minor. The one string is accurate, idiomatic and structurally sound; it is the
lone es plural message in the catalogue missing the `many` branch.

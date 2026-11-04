# Audit — es / relay.library — round 2 (`w4a3`)

**Verdict: PASSED** — the minor is closed by measurement. 0 open findings.
1 of 1 keys read (the namespace is a single key, so the delta is the whole file).

| input | sha256 | tracked |
|---|---|---|
| en `relay.library.json` | `ad478c433a4db3e725994766c724a2e8e92176f015f8283e71dc13b87185466a` | yes @ `183f7a0537e` |
| es `relay.library.json` | `0933a66475cf6b9b74bbf3befa849296f3fd5014a3707f27f451b6476b85e8bf` | **no — untracked** |
| en `relay.library.context.json` | `877b88290cf14602e63b2bb1c58719fac96a20ae53ff04b147e21de23df93c54` | yes @ `183f7a0537e` |

Superseded (`w4a2`) target: `f2d5ab3a9a5660f94b7656324a4c1069c06d88c31e908a5df1226872cb7d1a71`.

| | |
|---|---|
| was | `{count, plural, one {# recurso} other {# recursos}}` |
| now | `{count, plural, one {# recurso} many {# de recursos} other {# recursos}}` |

## `es-w4a2-LB-01` — **closed by execution**

I did not take the render claim on trust. Every count the implementer named was rendered through the
repository's own runtime (`local_operator.i18n.runtime.render_message`, locale `es`, source at
`183f7a0537e`):

| count | category | rendered | expected | |
|---|---|---|---|---|
| 0 | other | `0 recursos` | `0 recursos` | PASS |
| 1 | one | `1 recurso` | `1 recurso` | PASS |
| 2 | other | `2 recursos` | `2 recursos` | PASS |
| 21 | other | `21 recursos` | `21 recursos` | PASS |
| 1 000 000 | many | `1.000.000 de recursos` | `1.000.000 de recursos` | PASS |
| 1 500 000 | other | `1.500.000 recursos` | `1.500.000 recursos` | PASS |

All six match the claim exactly, including the grouping the formatter supplies — nothing is
hand-preformatted.

**Same-code control.** Rendering the *superseded* string at the same counts isolates what changed:

| count | superseded string | current string |
|---|---|---|
| 1 000 000 | `1.000.000 recursos` ← the defect | `1.000.000 de recursos` |
| 2 000 000 | `2.000.000 recursos` | `2.000.000 de recursos` |
| 1 500 000 | `1.500.000 recursos` | `1.500.000 recursos` (identical) |
| 1 000 001 | `1.000.001 recursos` | `1.000.001 recursos` (identical) |

The two strings diverge only at exact multiples of 10⁶ and agree everywhere else, which is the proof
that the fix is targeted rather than a rewrite.

## Does it now match the other es plurals?

Yes, and the census is unambiguous: **62 of 62** es plural messages on disk carry
`one`/`many`/`other`. At `w4a2` the figure was 57 of 58 and this key was the lone outlier; the
denominator moved because other lanes added `relay.session`, `relay.shell` and `mobile.setup`
catalogues in the interim, and all of the newcomers carry `many` too. The `many` body
`# de recursos` also matches the locale's dominant count-of-noun shape — `# de preguntas`
(`relay.asks.dock.waiting`), `# de opciones` (`relay.asks.pending.options`).

Structure otherwise untouched and still correct: `count` byte-identical, `#` in all three branches,
no `few`/`zero`/`two`, lowercase, 51 cells against English 47.

## Verdict

**PASSED.** The minor is closed and nothing else on this namespace is outstanding.

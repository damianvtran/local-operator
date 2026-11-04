# Audit — es / relay.login — round 1 (`w4a2`)

**Verdict: PASSED** — 0 blocker, 0 major, 0 minor, 0 nit. 4 of 4 keys read (en 4 / es 4 / sidecar 4).

| input | sha256 | tracked |
|---|---|---|
| en `relay.login.json` | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | yes @ `183f7a0537e` |
| es `relay.login.json` | `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658` | **no — untracked** |
| en `relay.login.context.json` | `9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a` | yes @ `183f7a0537e` |

## What this surface is

The server-rendered authentication gate — a browser page, not the phone SPA. The sidecar states the
role and tone on all four keys: "Preserve current login copy and casing", "Server-rendered
authentication gate; no JavaScript translation bootstrap", "Brand local operator stays untranslated".
That makes casing a *requirement* here rather than a style choice, and it is the axis on which this
namespace could most easily have failed.

| key | en | es | |
|---|---|---|---|
| page.title | `local operator — sign in` | `local operator — iniciar sesión` | lowercase brand + em dash kept |
| password.label | `password` | `contraseña` | lowercase kept |
| submit.label | `sign in` | `iniciar sesión` | lowercase kept |
| error.password | `Wrong password.` | `Contraseña incorrecta.` | sentence, capitalised head |

All four are accurate. `iniciar sesión` is the standard Spanish rendering of a sign-in action and is
used identically for the page title and the button, so one action has one name. The em dash with its
two spaces survives, which the Spanish typography rules require. Nothing is Title Cased except the
first letter of the complete error sentence, which is correct sentence case.

## The password-copy question

The brief asks whether the copy implies the password goes somewhere unexpected. It does not, and
this is worth stating precisely because the screen is a credential field:

- the label is the bare noun `contraseña` — no destination, no transport verb;
- the button is `iniciar sesión` — a local authentication action, not "enviar" or "verificar en línea";
- the only failure string is `Contraseña incorrecta.` — it reports a mismatch and asserts nothing
  about where the value went, whether it was stored, or whether it left the device.

No key mentions a server, a network, a sync or a third party, so no user could infer from this copy
that the password is transmitted beyond the sign-in request itself. Clean.

## Width

`submit.label` grows 7 → 14 cells (+100%), which is percentage inflation of a very short string:
`iniciar sesión` is the conventional Spanish sign-in button and the shorter alternative (`Entrar`)
would change register rather than fit a budget. No `maxCells` is declared in this sidecar, so there
is no declared budget to test against and no truncation risk to report. Page title 24 → 31 cells
(+29%) is in band for a browser tab.

## Display note

`relay.login.error.password` renders through this audit harness with its leading token masked. That
is a display artefact, not data: the file's bytes contain no `redacted` literal, and the value is 22
codepoints — the ordinary Spanish noun for a login secret followed by ` incorrecta.`. Verified by
codepoint dump, so no finding arises. The sanctioned public-copy reader was unavailable here (it
reported the workspace is not a git checkout), which is moot because the codepoint dump is
authoritative and the value is ordinary UI copy rather than a credential.

## Verdict

**PASSED.** Four keys, four correct renderings, casing requirements honoured, and no implication
that the password is sent anywhere unexpected.

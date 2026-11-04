# es `mobile.setup` — independent translation audit (w5a1)

**Verdict: PASSED — 0 blocker, 0 major, 7 minor, 5 nit.**
Namespace: `mobile.setup` · locale `es` · 76 keys · audit round 1 · 2026-10-29

| | |
|---|---|
| Revision | `183f7a0537e4cee42fd82f25a21fe52953a81bdf` (branch `feat/i18n-es-wire-errors`) |
| Target | `local_operator/i18n/catalogues/es/mobile.setup.json` — **UNTRACKED** at that revision |
| Target sha256 | `a488fa214a660045363b56d6edd5a6d6f2d0a2feb1d35c689cc47ccaf51cd535` |
| EN sha256 | `6249b10379beb9b2433746416fa7baff061265a1b9315d47465140cd5fcf61e9` |
| Sidecar sha256 | `be33c74a329196fdbe78f11f015848ff4f9593add57eba414163d4042807c811` |
| Keys read | 76 / 76 / 76 (en / target / context) |
| Auditor | translation-auditor, session `99bb6933f60c`, model `radient/auto` (fallback — see below) |

The catalogue is untracked, so this audit is bound to the **on-disk bytes**, not to a
committed blob. I translated none of this copy and changed nothing.

## Billing and activation state promises — the reason this namespace is high-stakes

Read as a state machine, key by key. **No state promise moved.**

- `checking` / `applying` — preliminary work only, no claim that billing or installation started.
- `activating` — activating the subscription **at the accepted price**; `price` says activation is still *required*; `billing_not_activated` says activation is still *needed*.
- `billing_activated` / `billing_activated_status` — completion **plus** "has not been undone".
- `billing_already_active` — "ya estaba activa": the subscription was already active, *not* that this run activated it.
- `billing_unconfirmed` — activation was **requested**, outcome **unconfirmed**, and "do not assume rollback" survives verbatim-in-kind (`no des por hecho que se haya revertido`).
- `balance` and `consent` — both deny an automatic credit purchase (`la configuración nunca compra crédito`; `No compra crédito`).
- `activated` / `configuring` — owning credential **unchanged**; the owner is **not** switched.
- `failed`, `failed_tui`, `unexpected_failure` — all three keep "completed steps have not been undone".

No string asserts or implies a charge that did not happen, and no negation is dropped
(negation presence checked on all 76 keys). The one wording I looked at hardest,
`quote_changed`'s "el importe que **enviaste**", is filed as a minor: it imports a
money-transfer connotation, but it does not invert the charge promise.

## What was verified mechanically (all 76 keys, bytes not summaries)

| Check | Result |
|---|---|
| Key-set and key-order parity | 76 = 76, same order |
| Placeholder parity, per key | **0 mismatches** |
| Accessor argument parity (`Msg()` kwargs vs `{arg}` in en *and* es) | 76 accessors, **0 mismatches** — nothing can fail at render time |
| ICU constructs | none in this namespace in either locale → no plural category can be wrong |
| es CLDR plurals | not applicable (no plural/select exists); no `few`/`zero`/`two` introduced |
| Machine literals (25: commands, `/mobile`+`/login` subcommands, `--flags`, `<id>`, `create.json`, console URL, `cloudflared 2025.4.0`, `brew install cloudflared`, `USD`, `macOS`) | identical in the same key sets |
| Digit sequences outside placeholders | identical |
| Em dash / en dash / ellipsis counts | identical |
| Negation presence | identical |
| Byte hygiene | UTF-8, no BOM, no CRLF, trailing newline, valid JSON, 76 unique keys |

**Detectors proved with negative controls** (14 mutations on in-memory copies):
dropped / renamed / mangled placeholder, altered URL, renamed command, changed version,
removed key, added key, flipped negation, English leftover, stray exclamation,
em dash→hyphen, ellipsis→ASCII — all detected; the unmutated copy reports 0 problems.

*Honest refinement:* my first negation detector compared the word "no" across locales and
flagged 5 keys. Reading each pair showed English encodes those as single lexemes
(*invalid*, *unavailable*, *unchanged*, *unconfirmed*) that Spanish renders analytically
(*no válida*, *no está disponible*, *no ha cambiado*, *sin confirmar*). All five were
**detector false positives, not translation defects**; the check was rewritten to compare
presence-of-negation. Re-run: 0 mismatches. Recorded because a checker that reports five
"findings" which are not findings is worse than no checker.

## Consistency with the shipped copy — proved, not assumed

- `mobile.setup.corrupt_intent` is **byte-identical** to shipped `es/wire.tunnel_setup.corrupt_intent`.
- `mobile.setup.in_progress` differs from shipped `es/wire.tunnel_setup.in_progress` **only** by the same `;` / `.` difference the two English strings have.
- "Phone access" → **`acceso telefónico`** in all 21 keys that name the feature, matching shipped `wire.slash.mobile` ("Acceso telefónico de Radient: …") and `wire.tunnel_setup`. No second rendering.
- "enable" → `activar` has shipped precedent (8 shipped keys: `cli.global_help.debug`, `cli.mcp_help.add_oauth`, `cli.serve_help.reload`, `cli.services_status.install_service_help`, `wire.memory.enabled.help`, `wire.errors.media.speech_no_credential`, `wire.errors.desktop.aida_disabled`/`librarian_disabled`), so `quote_unread`'s "activar" is *not* a drift from the locale; the outlier is `configuring`'s "habilitando" (see minor below).
- "do not assume rollback" → `no des por hecho que se haya revertido`, the same sentence shipped `wire.tunnel_setup.unavailable` uses.
- A cross-namespace scan found exactly **one** English string duplicated across namespaces (the `corrupt_intent` refusal) and its Spanish matches.

## Findings

### Minor

**1. `mobile.setup.quote_changed`** — the message that explains a rejected price
> current: `… El importe que **enviaste** no se aceptó; comprueba el precio vigente antes de volver a intentarlo.`
> proposed: `… El importe que **aceptaste** no se aceptó; …` (alt: `El importe aceptado no se aplicó`)

"enviar un importe" is a money-transfer verb, attached to the referent the same message
has just called "aceptaste". The state promise survives — nothing was accepted, nothing was
charged — so this is connotation, not a state change. One-word fix.

**2. `mobile.setup.quote_unread`** — `activar` vs `configuring`'s `habilitando`
> current: `… antes de **activar** el acceso telefónico.`
> proposed: `… antes de **habilitar** el acceso telefónico.`

One English gerund, two Spanish verbs; and `activación` is reserved for the paid
subscription in nine other keys here. Pick one and record it.

**3. `mobile.setup.busy`** — `en marcha` vs `en curso`
> current: `… ya está en marcha.` → proposed: `… ya está en curso.`

`in_progress` — in this namespace, and in the shipped `wire.tunnel_setup` (the two strings differ
only by the `;` / `.` their English sources also differ by) — says `ya está en curso`. One state,
two renderings, same flow.

**4. `mobile.setup.title_login`** — calque + longest title
> current: `El acceso telefónico necesita un inicio de sesión **de** Radient` (60 cells)
> proposed: `El acceso telefónico requiere iniciar sesión **en** Radient` (55 cells)

You sign in *to* an account; "de Radient" calques "a Radient sign-in". Also the longest title
in the namespace (+67 % vs EN's 36).

**5. `mobile.setup.account_failed`** — `nueva` for "fresh"
> current: `Falló la **nueva** verificación …` → proposed: `Falló la **verificación actual** …`

"Fresh verification" is the verification just performed, not an additional attempt.

**6. `mobile.setup.owner_missing`** — `redirigirá` for "repoint"
> current: `no **redirigirá** un túnel existente …` → proposed: `no **reasignará** un túnel existente …`

"Repoint" is a reassignment of the tunnel's owning credential; "redirigir" reads as
redirecting traffic — a different remedy from the one the sentence forbids.

**7. `mobile.setup.checking` + `mobile.setup.verified_url`** — `móvil` beside `teléfono`
> `… y la ruta **móvil**…` → `… y la ruta **del teléfono**…`
> `URL **móvil** verificada: {url}` → `URL **del teléfono** verificada: {url}`

The feature is `acceso telefónico` / `teléfono` in 21 keys and in shipped copy. Adjectival
`móvil` does have shipped precedent (`cli.mobile_help`: "el daemon móvil"), so this is drift
inside one feature, not a style-guide breach — hence minor, not major.

### Nit

8. `credential_ids` — `IDs de credencial` → `IDs de credenciales` (plural IDs over a singular noun).
9. `billing_activated` — `Activación de facturación completada…` calques the noun stack; `Facturación activada al precio mensual aceptado` reads better and is 20 cells shorter.
10. `usage` — `[monthly price]` stays English beside `<id>`/flags. Defensible (the whole usage grammar is English); flag so the choice is deliberate, not accidental.
11. `choose_login` — `… en lop login-status` → `… de lop login-status` (source says "from").
12. `stopped` — `solo si tienes intención de reanudarlo` is literal; `solo si quieres reanudarlo` / `solo si tienes previsto reanudarlo` states the intent directly.

## Width

No key in this namespace declares `maxCells`. Titles are the first line of a text block
(`local_operator/tunnels/setup.py: text()` builds `lines`, the TUI hands the whole string to
`notice(...)`), so they are wrapped prose lines, not fixed-width labels — 60 cells is not a
clip. Mean growth is **+24.6 %**, at the top of the guide's 15–25 % band; the widest are
`supervisor_tui` 191→253, `corrupt_intent` 165→221, `mobile_bundle` 127→180. The title family
grows most (`title_login` 36→60, `title_apply` 31→49, `title_cloudflared` 37→57,
`title_failed` 35→54) because the repeated lead `El acceso telefónico necesita` costs 29 cells
against English's 19. **Flagged for a rendered check on a narrow TUI notice**, not called a
defect: I have cell counts, not a frame.

## Model disclosure

The request opened on `anthropic/claude-sonnet-5-5`. Three provider-failure switches fired
before any audit work began: → `openai/gpt-6-astra` → `kimi/k3.1-pro` → **`radient/auto`**,
under which the whole audit ran. `radient/auto` is an auto-routing alias; the concrete backend
is not identifiable from the session header. Self-reported from the session's own
model-switch notices, not independently observable. No role, tool or default profile changed.

## Limitations

- LLM semantic judgement, not a native-speaker attestation.
- Static copy audit: no rendered frame, no runtime service, no eligibility or release claim.
- Width figures are cell counts derived from the bytes, not screenshots.
- Target is untracked; evidence binds on-disk sha256 values.
- Audit model is an auto-routing alias.
- Evidence written, not committed — this role performs no git writes.

# Independent audit — es `tui.projects` (audit id w6a1)

Auditor: translation-auditor (lopdev team), independent of the translator.
I authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider quota failures
dropped this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first).

- English source: `local_operator/i18n/catalogues/en/tui.projects.json` (97 keys, sha256 `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91`)
- Context sidecar: `.../en/tui.projects.context.json` (97 entries; roles and notes, **no `maxCells` declared for any key**)
- Target: `local_operator/i18n/catalogues/es/tui.projects.json` (97 keys, sha256 `4b6375ad8473afd927126efb535072f41426c80908f03c7a90834888f0264d76`, UNTRACKED at HEAD `b925c2e`)

## Verdict

**passed** — 0 blocker, 0 major, 1 minor, 1 nit. The delivery-confirmation
copy (the highest-stakes strings here: a double send is the failure mode this
paragraph exists to prevent) is exact, the read-only notice carries both
clauses, and the done-gate refusal is faithful. One idiom nit on a zoom label
and one register inconsistency in a hint fragment.

## What I read

All 97 keys, twice (beside the source with the sidecar; then Spanish-only).
Beyond the catalogue I followed **the callers** for the four strings whose
meaning depends on what composes them — `send.sent`, `send.compose`,
`notice_read_only`, `refusal_done_gate` — because a translation can be perfect
and the composed line still wrong.

## Mechanical floor

`checks.py` (this bundle): key parity/order, ICU arguments, es plural
categories, machine tokens (including `/project link`, `<chain-key>`,
`<provider>/<model>`, the arrows and `↵`), edge whitespace/newlines, settled
terminology and typography, English residue — **all eight pass**, plus a ninth
that **executes** the Spanish plural selection through the shipped renderer
(`+3 más` at every arm, and `send.more`'s word-identical branches exercised at
1, 3 and 1.000.000); the repo's catalogue validator reports `0 problem(s)`.

No `maxCells` is declared in this sidecar, so the budget check has nothing to
enforce (it reports zero by construction). Fit was judged against the render
paths instead — see "Fit" below, which is where the one measured risk lives.

## Delivery confirmations — the high-stakes paragraph

The English distinguishes four outcomes and the Spanish keeps all four
distinct; none is collapsed into another:

| English | Spanish | Why it matters |
|---|---|---|
| `wake unconfirmed — don't resend; it may have landed` | `despertar sin confirmar — no reenvíes; pudo llegar` | **the** double-send guard: the negative imperative (`no reenvíes`) and the reason (`pudo llegar`) are both present and unambiguous |
| `delivery unconfirmed — check its transcript first` | `entrega sin confirmar — revisa antes su transcripción` | the second half names the ACTION (`revisa … su transcripción`); it does not soften into "maybe delivered" |
| `send faulted on this end — retry` | `el envío falló en este extremo — reintenta` | "on this end" survives (`en este extremo`), so the reader does not think the peer failed |
| `could not deliver: {reason}` / `could not send: {reason}` | `no se pudo entregar: {reason}` / `no se pudo enviar: {reason}` | the two refusals stay distinct (delivery vs submission), as in English |
| `refusal_done_gate` | `no se puede marcar como hecho mientras estos hitos sigan abiertos: {names}` | exact, including the subjunctive (`mientras … sigan abiertos`) and the `hitos` noun from shipped `tui.projects_edit` |

The namespace contains **no destructive verb at all** — I scanned every value
for delete/remove/archive/drop/clear/discard/reset/overwrite/revoke/kill/stop
and the only hits are `esc cancel`/`cancelar` and the read-only refusal. The
destructive milestone actions (`Eliminar hito`) live in the already-shipped
`tui.projects_edit`, out of scope here.

## Form hints

`key_placeholder` `el identificador de referencia` is **byte-identical** to the
phrase shipped in `tui.projects_edit.refusal_name_required` (`… es el
identificador de referencia`), which is the strongest available consistency
evidence for a form hint with no sidecar note of its own. `title_placeholder`
`un nombre legible`, `tags_placeholder` `separadas por comas o espacios`
(feminine agreeing with `etiquetas`), `owner_placeholder` `propietario`,
`estimate_placeholder` `puntos`, `date_placeholder` `YYYY-MM-DD` (token
untouched), `readback_hint` `{label} sugerencia: {hint}` — all faithful.

## Fit

- `notice_read_only` is **127 cells** (en 103). It is delivered through
  `_system_notice_block`, which wraps itself when built, so the extra 24 cells
  cost a line, not a clip. Checked, not assumed: the docstring states the notice
  wraps at its painted width and pins its row count from the same measurement.
- The **notice row** in this view is `no_wrap`, fitted to a measured 56 cells at
  60x24 (`projects_send.py` documents the measurement). The clauses that land
  there — `send.wake_unconfirmed` (**50 cells**), `send.delivery_unconfirmed`
  (**53**), `send.fault` (**42**) — all fit inside 56 **and** put the actionable
  half first, which is what the English ordering does too. `delivery_unconfirmed`
  is the tightest of the three: five cells of headroom. This is the one place
  where the Spanish length could have broken the design, and it does not.
- The composer band `send.compose` is 32 cells (en 30) — no risk.

## Terminology against shipped precedent

| Term | This file | Shipped precedent | Verdict |
|---|---|---|---|
| `milestone` | `hito` | `tui.projects_edit` (`añadir hito`, `hito eliminado`) | correct |
| `todos` | `tareas` | `cli.global_help`, `relay.transcript` use `tareas` for work items | correct |
| `linked sessions` | `sesiones vinculadas` | `cli.agents_help`, `wire.settings` (`vinculado`) | correct — a configured association is `vínculo`, not `enlace` |
| `Start session` | `Nueva sesión` | `relay.chrome`, `tui.import` (`nueva sesión`) | **good call**: `Iniciar sesión` would read as *log in* in Spanish UI |
| `team` / `owner` | `equipo` / `propietario` | glossary, `tui.team_authoring` | correct |
| `runtime` | `runtime` | es kept-English list | correct |
| `another device` | `otro dispositivo` | glossary (`device` → `dispositivo`); never `equipo`/`máquina` confusion | correct |
| `snapshot` | `instantáneas` | `wire.settings` | correct |
| `board` / `timeline` / `list` | `tablero` / `cronología` / `lista` | (new view names) | correct Spanish product vocabulary; consistent between the view labels and their bindings (`Vista de tablero`, …) |

## Findings

**F1 — minor — `tui.projects.binding_zoom_out`**
Current `"Tiempo más grueso"` (17 cells) for *Coarser time*. `grueso` is the
right antonym of `fino` for grain, but *time* is not coarse in Spanish — the
paired label reads as a calque of "coarser time" and a native dev pauses on it.
Its partner `Tiempo más fino` (F1's counterpart, 15 cells) is fine. Proposed
`"Menos detalle"` (13 cells) — or `"Intervalo mayor"` (15) if the label should
name the axis rather than the granularity. Either keeps the zoom pair readable
and shorter than what is there now.

**F2 — nit — `tui.projects.send.compose`**
Current `"enviar a: {handle} · esc cancela"`. The picker's own legend
(`send.legend`, `start.legend`) says `esc cerrar` — infinitive — and so does
shipped copy (`tui.asks.dismiss.hint` `cancelar`, `tui.hub_links.hint.close`
`cerrar`). This one band conjugates. Both read; the pair inside one picker does
not match. Proposed `"enviar a: {handle} · esc cancelar"` (33 cells). Recorded
as a nit rather than a minor because the band is a status strip (a conjugated
statement of what the key does is defensible there) and nothing is misread.

## Out-of-scope observation (NOT a finding against this translation)

`tui.projects.send.sent` composes `enviado a ◆ <handle> · {state_word}`, and
`{state_word}` is supplied by `local_operator/mobile/peer_send.py`
(`DeliveryOutcome.state_word` → `DELIVERY_STATE_WORDS` = hard-coded
`"delivered"`, `"wake unconfirmed"`, …). In the Spanish TUI the receipt
therefore renders a Spanish clause with an **English** state word:
`enviado a ◆ ab12cd34 · delivered`.

The catalogue cannot fix this — rewording the key would not translate the
token — and the amber variants *are* keyed (`AMBER_NOTICE` maps to
`send.wake_unconfirmed` / `send.delivery_unconfirmed`). The gap is the
un-keyed state table in `peer_send.py`; it needs its own extraction slice, not
a translation change. Reported here because it is visible on this surface.

## Checked and accepted (no finding)

- `view_timeline` `cronología` — standard Spanish for a timeline view, and the
  binding agrees (`Vista de cronología`).
- `send.no_match` `retroceso para ampliar` — the Backspace key is labelled
  `Retroceso` on Spanish keyboards; correct.
- `render.session_missing` `[falta]` — matches `tui.library` `falta`; terse state
  word, never a zero.
- `binding_message` `Enviar mensaje` for `Message` — verb+object is clearer than
  a bare noun in a key list; accepted.
- `render.live_count` `{count} activas` (feminine, `sesiones`) and
  `render.running_settled` `{settled} resueltas` (feminine, `tareas`) — both
  agree with the noun their own row has already established.
- `send.more` `+{label} más` carries an `many` branch identical to `other`; it
  is required by the es category set and renders correctly at 1.000.000.

## Limitations

- No `maxCells` in this sidecar: two measured risks (the 56-cell notice row and
  the 127-cell wrapped notice) were judged from the render code, not a frame.
- No rendered TUI frames captured (loaded host, heavy runs forbidden).
- LLM semantic judgement, no native-speaker attestation.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 in
  `record.json`, not a commit.

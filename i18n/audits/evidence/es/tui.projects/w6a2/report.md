# Independent audit — es `tui.projects` (round 2, audit id w6a2)

Auditor: translation-auditor (lopdev team), independent of the translator. I
authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider-quota failures moved
this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first). No catalogue,
sidecar, ledger or allowance file was touched.

- English source: `local_operator/i18n/catalogues/en/tui.projects.json` (97 keys,
  sha256 `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91`) —
  **byte-identical to the round-1 source**.
- Context sidecar: `.../en/tui.projects.context.json` (97 entries, sha256
  `fa39ad8f21674a42e5218698d11396a08d47e2f82297467ad17bb1ab292bb1fe`,
  unchanged). No `maxCells` is declared for any key.
- Target: `local_operator/i18n/catalogues/es/tui.projects.json` (97 keys, sha256
  `89b554319e7dfd8716e0e71d5031999e9f2b88f962d19a5953c3c41a9f79127f`, UNTRACKED
  at HEAD `b925c2e`). Round 1 audited `4b6375ad…f0264d76`.

## Verdict

**passed** — 0 blocker, 0 major, 0 minor, 0 nit, 0 new findings. Both round-1
findings are closed, and reverting them reproduces the round-1 digest
byte-for-byte (`checks.py` C10), so no other value in the namespace moved.

## Round-1 findings on the current bytes

| Id | Sev | Key | Round-1 ask | Current es | Status |
|---|---|---|---|---|---|
| F1 | minor | `binding_zoom_out` | `Menos detalle` | `Menos detalle` | **closed** |
| F2 | nit | `send.compose` | `enviar a: {handle} · esc cancelar` | `enviar a: {handle} · esc cancelar` | **closed** |

### F1 — `tui.projects.binding_zoom_out`

`Tiempo más grueso` was a calque: *grueso* is the right antonym of *fino* for
grain, but time is not "coarse" in Spanish the way a material is, and a native
developer pauses on it. `Menos detalle` (13 cells; en `Coarser time` 12) says the
same thing in Spanish's own vocabulary and is the label maps and timelines use.

It is a Textual binding description, so width is not pinned: `show=False` keeps
it out of the footer (`projects_view.py:157`), and Textual's `KeyPanel` (`?`)
lists it in a table with no wrap guarantee (sidecar note). 13 cells is safely
inside any of those.

**Considered and accepted (no finding):** the pair now mixes frames — its
partner `binding_zoom_in` is `Tiempo más fino` ("finer time", 15 cells) while
`binding_zoom_out` names detail; and `binding_open_detail` is `Detalle`, so the
help table carries `Detalle` and `Menos detalle` a few rows apart. Both labels
are unambiguous beside their key column (the zoom row shows `-`), nothing is
misread, and this was the round-1 primary proposal with the alternative recorded
then. If the team ever wants a matched pair, `Tiempo más amplio` (16) pairs with
`Tiempo más fino` and keeps the time axis; that is a polish choice, not a defect,
and it is not needed for this round.

### F2 — `tui.projects.send.compose`

`enviar a: {handle} · esc cancelar` (39 cells with the 14-cell `◆` handle —
`SHORT_ID_CELLS = 12`, projects_send.py:83; en 36) replaces the conjugated
`esc cancela`. It now matches the picker's own legend, which is all infinitives
(`send.legend` = `escribe para filtrar · ↑↓ mover · ↵ elegir · esc cerrar`), and
the es style guide's "action labels: infinitive". The strip is
`.projects-compose-strip { width: 1fr; height: 1; }` — one clipped line, and 39
cells fits an 80-column composer.

The change is confined to the verb: `{handle}` keeps its slot, `esc` stays
literal, and the `·` separator is byte-identical to en (C4). No lost nuance:
`esc` closes the picker in the legend and cancels the send on the strip, and the
two infinitives now distinguish those actions the same way the English does.

## Sibling sweep

- `binding_zoom_in` `Tiempo más fino` is byte-identical to round 1 (C10) — the
  fix did not tempt anyone into touching its partner.
- `send.legend` (`esc cerrar`), `send.compose` (`esc cancelar`) and
  `send.empty` (`esc cierra`) now mirror the English key-for-key: en has
  `esc close` / `esc cancel` / `esc closes`. The picker's hint grammar is
  therefore consistent *with the source* across all three, which is the
  strongest form of the round-1 ask; `send.empty` is not an outlier.
- `send.no_match`, `send.title`, `send.filter`, `send.more` are unchanged, and
  the changed strip keeps `{handle}` and the `·` separator byte-identical (C4).
- The delivery copy that round 1 checked most closely (`send.wake_unconfirmed`,
  `send.delivery_unconfirmed`, `send.fault`, `send.refusal`, `send.error`) is
  byte-identical to round 1 (C10); the notice row that carries the first three
  still fits its measured 56 cells (50/53/42).

## Mechanical floor

`checks.py` (this bundle): **C1–C8 pass, C10–C13 pass**; the repo validator
reports `0 problem(s)` tree-wide; **13 controls all caught** (nine from round 1,
four for round 2). C11 measured both changed keys on the built string
(`binding_zoom_out` 13 cells, `send.compose` 39 against en's 36).

C9 executes `send.more`'s plural selection at 1, 3 and 1.000.000: every arm
renders `+3 más` because the three branches are word-identical by design (the
`{label}` slot carries the count). C3 independently proves the `many` category
is present, which is the part that would silently fall back. See the audit.log
note on the locale gate.

## Note on the run environment (not a property of these files)

`local_operator/i18n/data/shipped.json` in the working tree (not HEAD) omits
`es`, so `resolve.eligible_locale('es')` returns `'en'` and the default
`render()` path would answer in English. This audit did not touch that file; the
rows above come from the same catalogue lookup + ICU formatter that
`render_record` uses after the gate, and `audit.log` prints the gate state.

## Limitations

- No `maxCells` in the sidecar: fit is judged from the render paths read, not a
  rendered frame.
- No rendered TUI frames captured (loaded host, heavy runs forbidden).
- LLM semantic judgement, no native-speaker attestation.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 above,
  not a commit.

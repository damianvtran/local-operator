# Audit record — es / relay.session / w4a1

- **Locale / namespace:** `es` / `relay.session`
- **Audit id / round:** `w4a1` / 1
- **Audit date:** 2026-10-29
- **Audited revision:** `183f7a0537e4cee42fd82f25a21fe52953a81bdf`
  (branch `feat/i18n-es-wire-errors`). The **target catalogue is untracked**
  (`?? local_operator/i18n/catalogues/es/relay.session.json`); this audit is
  bound to the on-disk bytes,
  `sha256 873014622f3fa868c32ee6164a2d12085e35dd148a64ff8663d246b4b26f7a62`,
  not to a commit.
- **Keys read:** 17 of 17 (`en` 17 / `es` 17, key sets identical).
- **Verdict: PASSED** — 0 blocker, 0 major, 1 minor, 0 nit.

## Method

Every English value, its role/tone note in `en/relay.session.context.json`, and
the Spanish value were read side by side, all 17 keys. Each message was rendered
with the repository's own runtime (`local_operator.i18n.runtime`) under `es`;
this namespace carries no ICU arguments, so the structural checks are key
parity, argument set (empty on both sides), ellipsis codepoint and initial-case
parity against the source. Output is in `audit.log`; the run script is preserved
as `run-script.py.txt`.

This is phone. The copy is read mid-task, so the judgement weights terseness and
the state distinctions the sidecars name more heavily than sentence polish.

## Inputs (hashes of the audited bytes)

| input | path | sha256 |
| --- | --- | --- |
| en | `local_operator/i18n/catalogues/en/relay.session.json` | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` |
| context sidecar | `local_operator/i18n/catalogues/en/relay.session.context.json` | `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db` |
| es (target, untracked) | `local_operator/i18n/catalogues/es/relay.session.json` | `873014622f3fa868c32ee6164a2d12085e35dd148a64ff8663d246b4b26f7a62` |

## Verified, and clean

**State distinctions — all preserved.**

| state | en | es | kept |
| --- | --- | --- | --- |
| start control idle | new session | nueva sesión | label, not a verb |
| start control in flight | starting… | iniciando… | in-flight, distinct from idle |
| list search in flight | (placeholder) | Buscar conversaciones… | placeholder, not a verdict |
| history in flight | loading… | cargando… | in-flight, **not** the empty verdict (sidecar requires the distinction) |
| model catalogue in flight | loading… | cargando… | same, consistent rendering |
| empty filter verdict | no matching conversations | ninguna conversación coincide | verdict, distinct from loading |
| empty conversation | no messages yet / send a message below to get started | aún no hay mensajes / envía un mensaje abajo para empezar | headline + actionable hint pair |

`loading…` maps to the one word `cargando…` in both keys, so the same English
term has one Spanish rendering. `back to sessions` / `back` stay distinct
("volver a las sesiones" / "volver"). Accessible names stay lowercase
("volver", "volver a las sesiones") as the sidecar requires.

**Register and idiom.** tú-address throughout ("empieza", "envía", "tu
máquina"), no `usted`, no `vosotros`; no region-marked vocabulary
("directorio", "ruta", "teléfono-agnostic" wording). `directorio de trabajo` for
*working directory*, `ruta` for *path*, `modelo` for *model*, `filtrar modelos`
for *filter models* — all the settled es terms and consistent with the rest of
this locale. No English leakage apart from the kept-English `TUI`, which is on
the style guide's kept-English list and appears as a code-free acronym in
running prose, exactly as the source does.

**Structure.** No ICU arguments in either side; nothing to rename or reorder.
U+2026 appears in the same 6 keys in both languages (`search_placeholder`,
`start_busy`, `past.search_placeholder`, `past.loading`,
`directory.path_placeholder`, `model.loading`) and is never three dots. Initial
capitalisation matches the English source on all 17 keys — the file's mixed case
("Buscar conversaciones…" beside "buscar nombres y conversaciones…") is the
**source's** pattern, not an es drift: en has "Search conversations…" capitalised
and "search names and conversations…" lower.

## Finding

### S1 — **minor** — `relay.session.list.empty_hint`

- **Current:** `empieza una abajo o desde la TUI en tu máquina`
- **Proposed:** `inicia una abajo o desde la TUI en tu máquina`
- **Reason:** the same English verb *start* is rendered `iniciando…` by the
  paired `list.start_busy` control label, so the namespace names one action two
  ways; `inicia` also points at the control the hint tells you to use. Terseness
  and meaning are unaffected either way.

Not part of this finding, and deliberately not to be "fixed" with it:
`view.empty_hint`'s "para empezar" renders English *to get started*, a different
expression, and is the idiomatic es rendering of it.

## Considered and cleared (not findings)

- **`empty_hint` drops en's comma** ("start one below, **or** from the TUI") —
  Spanish does not put a comma before `o` in this coordination. Correct.
- **`no_matches` singular** ("ninguna conversación coincide") for en's plural
  "no matching conversations" — a bare verdict reads better singular in es and
  the meaning is unchanged.
- **`empieza una` leaves the object implicit** — the hint renders directly under
  `relay.shell.list.empty` ("no hay sesiones en ejecución"), so `una` has its
  antecedent on screen. Acceptable for a hint.
- **`directory.path_placeholder` "o escribe otra ruta…"** — the leading `o`
  mirrors en's "or type another path…", i.e. the fallback beside quick-picks.
  Preserved intentionally, not a stray conjunction.

## Limitations

- Rendered with the repository's Python runtime; the phone client's TypeScript
  runtime was not executed (both are parity-tested in-repo). This is not a
  rendered-frame capture on a device.
- No native-speaker review: the auditor is an LLM. Judgements about what a native
  would write are model judgements and are recorded as such.
- No width measurement: this namespace pins no `maxCells`; the longest string
  here (`empieza una abajo o desde la TUI en tu máquina`, 46 characters, all
  ASCII so the same cell count — measured with `len`, not with
  `rich.cells.cell_len`, which is not importable in the audit interpreter) sits
  in a wrapping hint, so no overflow risk was identified, but nothing was
  rendered at a screen width.
- Fallback model disclosed: this audit ran on `radient/kimi-k2.5`, not the
  session's primary model. See `record.json` `auditor.model_fallback_notice`.

## Verdict

**PASSED** for `relay.session` — 17 keys read, 0 blocker, 0 major, 1 minor
(S1), 0 nit. Copy is accurate, idiomatic neutral Spanish, register-consistent
with the locale, and no connection/state promise or machine token changed. The
minor is a consistency preference the translator lane may close in-wave; it does
not gate the namespace.

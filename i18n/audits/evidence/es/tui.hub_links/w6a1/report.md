# Audit — es / tui.hub_links / w6a1

- **Verdict: changes_requested** — 0 blocker, **1 major**, 1 minor, 2 nit
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; the es target is **untracked** at that
  revision, so the on-disk sha256 is what binds: es `1d87b68b9a249b2…`, en
  `7fbe6713f8ac8f3…`, context `9b81676163337d8…` (full digests in `record.json`).
- Keys read: **56** en / 56 es / 55 context. The one key with no context entry is
  `hint.engage`; that is a gap on the en side, noted once, not charged to the translation.
- Auditor: independent translation auditor (lopdev team), session `6a2f6fa808a7`; model
  **radient/auto** as actually run (see the fallback chain in `record.json`).

## What I read

All 56 values, against their en source and the namespace's context sidecar (21 distinct
role/note/tone entries, each read in full). Then the code that renders them:
`local_operator/tui/hub_links.py` (row construction, `mark.healthy`, `_policy_word`) and
`local_operator/tui/widgets/hub_links_panel.py` (keys, consent arming, `_notice`, the refusal
map, `_notice_lines` wrapping). Terminology was checked against the committed es catalogues —
`wire.settings.hub_defaults` (subscribe/vínculo wording, ledger `passed` at the current en
hash) and, for the `prompt` question, all 25 committed es uses of `prompt`.

## Mechanical result — clean

`scripts/i18n/check.py` exits 0 (repo-wide, 191 files). My own probe (`checks.py`, section A)
passes every mechanical check for this namespace, and adds two the validator cannot see:

- **Full CLDR coverage.** This namespace contains no plurals, so the coverage check is not
exercised here; it is proven by controls 2–4 and applied to `tui.app`'s six plurals.
- **Per-branch rendering.** Not applicable here; proven on `tui.app`.

Preserved, verified key by key: the two placeholders in every templated string; `esc`,
`ctrl+y`, `Enter`-free hints; `{first}–{last}/{total}` with its **EN DASH U+2013** (not an em
dash); `'  ·  '` double spaces; `…`; ` — ` with spaces; `¿Publicar públicamente?`.

`hint.failed` is the namespace's only `maxCells` (48). Measured against the reasons the code
actually produces (`hub_links_panel.py:1032` → `_humanize_reason(selected.error_class)`, so
codes such as `provider-error` and `unsupported-precondition`): worst case
`comprobación fallida: unsupported precondition` = **46 cells**, en 38. Inside the budget; no
finding. Note that the `{reason}` itself is a humanised service code and stays English — a
fallback limitation of the source design, identical in both locales, not a translation defect.

## Findings

### HL-1 · major · `tui.hub_links.refusal.revision_moved`

> `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el aviso para confirmar la última versión.`

The referent is wrong. This string is assigned to the panel's **notice line**
(`hub_links_panel.py:605`: `self._notice = _t(_k.refusal_revision_moved())`) — so the user is
reading a notice and is told to reopen "el aviso". The thing they must reopen is the **consent
prompt** (arm the row again), which the en names plainly as "the prompt". Spanish reserves
`aviso` for notifications/warnings in this product (`tui.notifications`, `tui.schedule`,
`tui.import`, and the panel's own notice line), so the instruction points at the message
itself.

Suggested: `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el prompt para confirmar la última versión.`
(`prompt` is the settled es term — 25 committed uses; if the wave wants no anglicism,
`la confirmación` reads correctly too.)

### HL-2 · minor · `tui.hub_links.reason.baseline_unknown`

> `Todavía no se conoce el historial de este vínculo, así que no puede publicar sin supervisión. Haz pull primero o resuélvelo.`

`resuélvelo` has two competing masculine antecedents in the same breath — `el historial` and
`el vínculo` — and the recovery differs: resolving the *link* is what unblocks publication.
Name it.

Suggested: `… así que no puede publicar sin supervisión. Primero haz pull o resuelve el vínculo.`

### HL-3 · nit · `tui.hub_links.reason.unsafe_publication`

> `Publicar en esta ubicación pública requiere antes tu consentimiento explícito.`

`requiere antes tu consentimiento` puts the adverb between verb and object; natural Spanish
says `requiere primero …` or `necesita … antes`.

Suggested: `Publicar en esta ubicación pública requiere primero tu consentimiento explícito.`

### HL-4 · nit · `tui.hub_links.action.pull`

> `Pull ahora` (en `Pull now`)

Understandable, and `pull` is on the kept-English list — but the shipped CLI copy says `hacer
push` / `hacer pull` (`cli.teams_help`), so a two-word label reads more Spanish as
`Hacer pull ahora`. Preference only; the current text is not wrong.

## Terminology checked against witnesses

| Term | Rule | This file | Verdict |
|---|---|---|---|
| configured hub link | `vínculo`; `enlace` only for a hyperlink | `vínculo` in 7 keys, `enlace` in 0 | **correct** |
| subscribe | `Suscribir` (transitive, as shipped) | `Suscribir`, `s suscribir`, `Suscribir: {subscribe}` | **consistent** with `wire.settings.hub_defaults` — `Suscribir vínculos de equipo` |
| auto-push | kept-English `push` | `Push automático` ×4 | consistent |
| on/off | state words | `Activado` / `Desactivado` | consistent with shipped choice descriptions |
| published | `publicar en` | `Publicado en {source}` | idiomatic |
| agent / skill / team | `agente` / `habilidad` / `equipo` | `Agentes` / `Habilidades` / `Equipos` | glossary §2 |

Consent copy that had to stay true, verified against `confirm.private` / `confirm.public` /
`state.unsubscribed` in the code:

- private scope still limits recipients to the organisation's people ("Solo las personas con
  acceso a esa organización podrán recibirlos") — the qualifier is preserved, not dropped;
- public scope still states that future changes may go out unattended **and** that private
  sources pause for review — both halves present;
- `state.unsubscribed` still says outgoing pushes may continue — the half a user could get
  wrong when they turn subscription off.

`mark.healthy` adds colons (`Suscribir: {subscribe}; Push automático: {auto_push}`) to what is
a screen-reader sentence. That is a correct adaptation for a spoken list, not a drift.

## Out of scope, recorded because it contradicts this file

`wire.slash.hub` (committed, ledger `passed`) renders the same configured hub links as
`Abre enlaces del hub: suscripción y push por enlace` — `enlace` where the style guide requires
`vínculo`. `tui.hub_links` is the one that follows the rule; do **not** "fix" it towards
`wire.slash`. The stale file needs its own re-audit.

## Limitations

LLM judgement, no native-speaker attestation, no rendered TUI frame captured (heavy local runs
were out of scope for this audit; the widths that matter here are the declared `maxCells`,
measured arithmetically with the repo's own ruler). No catalogue, sidecar, ledger or allowance
was edited.

## Ledger row — prepared, NOT written

Instruction for this audit was *no ledger writes*, so nothing was appended to
`i18n/ledger/es.json` (nor to the aggregate). The row below is what the ledger owner
should append, per the retention rule that the note stays ≤280 characters and points at
the tracked record:

```json
{
  "locale": "es",
  "namespace": "tui.hub_links",
  "source_sha256": "7fbe6713f8ac8f312884476c498c3b74e76c5f0853bdd027f25fc47d489311bb",
  "translator": "<from the wave's translation record — not restated here>",
  "auditor": {
    "agent": "i18n-auditor (independent; audited, did not translate)",
    "model": "radient/auto",
    "date": "2026-10-30",
    "result": "changes_requested",
    "findings": 4
  },
  "status": "changes_requested",
  "note": "changes_requested (0 blocker/1 major/1 minor/2 nit). Evidence i18n/audits/evidence/es/tui.hub_links/w6a1/record.json; es sha256 1d87b68b…, en 7fbe6713…; independent auditor, model radient/auto, 2026-10-30. HL-1: `el aviso` names the notice, not the consent prompt."
}
```

Note length: 264 characters. `translator` is left as a placeholder on purpose:
that block belongs to the wave's translation record, and inventing it here would be a
false provenance claim.

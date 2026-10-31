# Audit — es / tui.hub_links / w6a2 (round 2)

- **Verdict: changes_requested** — **1 major stands** (HL-1), 0 new findings.
- Round 1: `i18n/audits/evidence/es/tui.hub_links/w6a1/` (`changes_requested`, 1 major, 1 minor, 2 nit).
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df` (unchanged; the wave's catalogue edits are on
  disk, not committed). es target **untracked**, so the on-disk sha256 binds it:
  es `f3bde8f887ec44c2…`, en `7fbe6713f8ac8f31…`, context `9b81676163337d8a…` (full digests in
  `record.json`).
- Branch: `feat/i18n-es-tui` — the brief named `tui-wip`; `git rev-parse --abbrev-ref HEAD` says
  otherwise, and the log records what is actually checked out.
- Keys re-read: **56** en / 56 es / 55 context — every value read again against en and the
  sidecar, not only the three the implementer touched.
- Auditor: independent translation auditor (lopdev team), session `7fa3a39af5d9`; model
  **radient/auto** as actually run (fallback chain in `record.json`).

## What I re-read

All 56 values, the three changed keys in full context, the sidecar entries for those keys, and the
code that renders them: `local_operator/tui/widgets/hub_links_panel.py` (the refusal map, the
consent arming path at `action_toggle_auto_push`/`action_confirm_egress`, `_notice_lines` wrapping
and its height accounting, the card's `_body_budget`) and `local_operator/tui/hub_links.py` (row
construction, `_humanize_reason`). Terminology was re-checked against the whole committed es
catalogue: **38 occurrences of `prompt` in 36 values**, **1 occurrence of `diálogo`** — the string
under audit.

## Mechanical result — clean

`scripts/i18n/check.py` exits 0 (repo-wide, 191 files, 2348 literals, 51 advisories). Probe section
A passes every check on the current bytes: key parity (56/56), ICU argument tuples identical on
every key, no plural message in this namespace (so no CLDR branch surface here), backticked
literals byte-identical, the three layout literals intact (`{first}–{last}/{total}` with its EN
DASH, `ctrl+y`, `¿Publicar públicamente?`), `—`/`…`/`·` conventions preserved, no ASCII ` - ` used
as a dash.

Widths re-measured with `rich.cells.cell_len`, not taken on trust:

| surface | measure | ceiling |
|---|---|---|
| `hint.failed` worst case (`comprobación fallida: unsupported-precondition`) | **46** | 48 (`maxCells`) |
| `refusal.revision_moved` (104 cells) | wraps to **2** lines at 60 and 80 columns | wraps, never truncated (`_notice_lines`) |
| en equivalent (87 cells) | 2 lines | same |

The notice line wraps and its line count is folded into `_body_budget`, so the longer revision
message cannot clip the consent block.

## Findings

### HL-1 · major · STANDS (partially fixed) · `tui.hub_links.refusal.revision_moved`

> `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el diálogo y confirma la versión más reciente.`
> (was: `… Vuelve a abrir el aviso para confirmar la última versión.`)

The `aviso` collision is gone — the instruction no longer names the notice line it is printed on.
The replacement introduces a **new referent that does not exist**: there is no dialog.

- The string is painted on the card's **notice line** (`hub_links_panel.py:605`), and the thing the
  user must re-arm is the **consent prompt** — an inline block *inside* the same card, armed by the
  toggle action and completed only by `ctrl+y` (`hub_links_panel.py:565-577`).
- `HubLinksPanel` is a `Static` overlay opened by `/hub` ("The overlay itself"), not a
  `ModalScreen`. The app *does* have modal screens (`/resume`'s picker, settings sheets), so
  `diálogo` in this product reads as one of those — a different surface from the one that has to be
  reopened.
- `diálogo` occurs **once** in the whole es catalogue: this string. The settled es rendering for
  *prompt* is `prompt` (glossary §2; kept-English list in the es guide; **38 occurrences in 36
  catalogue values**, counted across all 95 es catalogue files). A one-off term for a term the
  glossary already fixes is a glossary
  deviation, which the audit contract grades major.

Suggested (either reads correctly; the first keeps en's own word and the glossary's rendering):

- `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el prompt y confirma la versión más reciente.`
- no-anglicism alternative: `Este vínculo cambió mientras lo revisabas. Vuelve a activar la confirmación y comprueba la versión más reciente.`

The rest of the sentence is good, and `la versión más reciente` is a better rendering than round 1's
`la última versión` (no *last/latest* ambiguity).

### HL-2 · minor · CLOSED · `tui.hub_links.reason.baseline_unknown`

> `Todavía no se conoce el historial de este vínculo, así que no puede publicar sin supervisión. Haz pull primero o resuelve el vínculo.`

The ambiguous `resuélvelo` is gone; the recovery names **the link**, which is what unblocks
publication. `Haz pull primero` keeps en's "Pull … first" with the verb in the tú imperative the
guide requires. Placeholder-free, no new defect.

### HL-3 · nit · DISPOSITION ACCEPTED (not applied) · `tui.hub_links.reason.unsafe_publication`

> `Publicar en esta ubicación pública requiere antes tu consentimiento explícito.`

Unchanged, as declared. The meaning is correct and the adverb placement (`requiere antes …`) is
grammatical if slightly literary; it was a preference, not a defect, so leaving it is a sound
disposition — it does not affect this verdict.

### HL-4 · nit · CLOSED · `tui.hub_links.action.pull`

> `Hacer pull ahora`

Matches the es guide's recorded amendment (2026-10-30: `pull`/`push` stay English, action labels are
the infinitive) and the shipped CLI wording the nit cited. Note for the record: this key has **no
call site** in this checkout (`grep` over `local_operator/tui/` finds only the generated
`action_pull()` accessor), so the label is copy-only for now and cannot overflow anything — the
wording is settled ahead of the surface that will use it.

## Terminology and consent copy re-verified

| Term | Rule | This file | Verdict |
|---|---|---|---|
| configured hub link | `vínculo`; `enlace` only for a hyperlink | `vínculo` ×7, `enlace` ×0 | correct |
| subscribe | `Suscribir` | `Suscribir`, `s suscribir`, `Suscribir: {subscribe}` | consistent with `wire.settings` |
| auto push | `push automático` (lowercase mid-sentence) | `push automático` / `Push automático` sentence-initially | correct |
| prompt | `prompt` (glossary §2) | `diálogo` (this key only) | **deviation — HL-1** |

Consent disclosure re-read against `confirm.private` / `confirm.public` / `state.unsubscribed`:
private scope still limits recipients to the organisation's people; public scope still states both
that future changes may go out unattended **and** that private sources pause for review;
`state.unsubscribed` still says outgoing pushes may continue. All three still carry their full
meaning; none was touched by this round.

## Out of scope, recorded because it contradicts this file

Unchanged from round 1: `wire.slash.hub` renders the same configured links as `enlace` where the
style guide requires `vínculo`. `tui.hub_links` is the file that follows the rule — the stale file
needs its own re-audit; do not "fix" this one towards it.

## Limitations

LLM judgement, no native-speaker attestation, no rendered TUI frame. Widths are measured with the
repo's own ruler against the ceilings the source documents; the notice wrap is computed, not
photographed. No catalogue, sidecar, ledger or allowance was edited.

## Ledger row — prepared, NOT written

This round's instructions forbid ledger writes (no git writes; no catalogue/sidecar/ledger/allowance
edits), so `i18n/ledger/es.json` was not touched. Row for the ledger owner:

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
    "findings": 1
  },
  "status": "changes_requested",
  "note": "r2 changes_requested (1 major stands). Evidence i18n/audits/evidence/es/tui.hub_links/w6a2/record.json; es sha256 f3bde8f8…, en 7fbe6713…; independent, radient/auto, 2026-10-30. HL-1: revision_moved names `el diálogo`, a surface with no other use in es; use `el prompt`."
}
```

Note-length check: the `note` above is exactly 270 characters, inside the 280-character budget that binds a ledger row.

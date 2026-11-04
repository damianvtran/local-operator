# Audit — es / tui.hub_links / w6a3 (round 3, narrow convergence)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit. Round 2's one
  standing major (HL-1) closes.
- Round 2: `i18n/audits/evidence/es/tui.hub_links/w6a2/` (`changes_requested`,
  1 major, 0 new). Round 1: `w6a1/`.
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; branch
  `feat/i18n-es-tui`. The es target is **UNTRACKED** at this revision, so the
  on-disk sha256 binds it: es `ff50b7cd5e4ab165…`, en `7fbe6713f8ac8f31…`,
  context `9b81676163337d8a…` (full digests in `record.json`).
- **Delta: exactly one value changed.** Restoring round 2's `diálogo` value
  reproduces the round-2 file sha256 `f3bde8f887ec44c2…` byte-for-byte; the
  current file is `ff50b7cd5e4ab165…`. No other value in this file changed
  (probe F1/F2; `audit.log` §3).
- Auditor: independent translation auditor (lopdev team), session
  `c0b953a7c12a`; model **radient/auto** as actually run. The header named
  `anthropic/claude-sonnet-5-5`; provider quota failures moved the session
  through `openai/gpt-6-astra` and `kimi/k3.1-pro` to `radient/auto` before any
  audit work (`active_model_route` records in this run's transcript; chain in
  `record.json`). I authored none of this copy and repaired none of it.

## What changed

`tui.hub_links.refusal.revision_moved`

| | |
|---|---|
| en | `This link changed while you were reviewing it. Reopen the prompt to confirm the latest.` |
| es, round 2 | `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el diálogo y confirma la versión más reciente.` |
| es, now | `Este vínculo cambió mientras lo revisabas. Vuelve a abrir el prompt y confirma la versión más reciente.` |

The sidecar says this refusal is shown when the **armed consent's policy
revision differs from the row at confirm time**; the user is confirming a
value they never saw, so nothing is written. The thing that must be re-armed
is the inline consent block inside the card, not a surface.

## The claim: `prompt` is the glossary rendering, not a coinage

**Yes.** Three independent checks agree:

1. **Glossary §2** fixes es *prompt* → `prompt` (kept English), and the es
   style guide's kept-English list says `commit, push, pull, merge, rebase,
   token, endpoint, script, prompt, deploy`.
2. **Reach across the es catalogues.** Lowercase `prompt` occurs in **36
   values / 38 occurrences** after excluding the two values whose only
   occurrence is the JSON machine key `"prompt":`
   (`wire.settings.providers_openrouter_max_price.help` and `.placeholder`) —
   the method that reproduces round 2's figure. Raw, including those two
   machine tokens: 38 values / 40 occurrences; case-insensitively 40/42. The
   audited key is one of the 36. `diálogo` now occurs **0** times in the whole
   es catalogue; `aviso` occurs **0** times in this key.
3. **The surface exists.** `HubLinksPanel` is a `Static` overlay
   (`hub_links_panel.py:102`, “The overlay itself. Opened by `/hub`”), not a
   `ModalScreen`. The consent prompt is the inline block armed by
   `action_toggle_auto_push` (`_consent_link_id`, `_consent_revision` at
   565-577) and completed only by `ctrl+y`; the refusal is painted on the
   card's notice line at 605, and `_notice_lines` (873) wraps it and folds its
   line count into `_body_budget` (819). `prompt` names exactly the thing the
   user has to reopen; `aviso` named the notice line it is printed on, and
   `diálogo` named a modal surface this card does not have.

The rest of the sentence is also right: tú imperatives (`Vuelve`, `confirma`),
no placeholder, no added/omitted claim, and `la versión más reciente` keeps
“latest” without the *last/latest* ambiguity round 1 had. No finding.

## Mechanical result — clean

- `python3 scripts/i18n/check.py --catalogues-only` → `0 problem(s)`, exit 0.
- Full `python3 scripts/i18n/check.py` → exit 0: 191 files scanned, 2,348
  literals at or below baseline, catalogues consistent, 51 advisories
  (report-only).
- Probe: key parity 56 en / 56 es / 55 context (the context file has never
  carried `hint.engage`; pre-existing, round 2 recorded the same); every value
  parses in the shipped ICU subset; placeholder tuples identical on all 56
  keys; no plural message in this namespace; the changed value has no ICU
  braces or code span; period and spacing conventions intact.

## Fit

`cell_len` 103 cells (en 87; the old `diálogo` form was 104). It wraps to
**4 / 2 / 2** physical lines at 32 / 52 / 72 cells; the 32-cell case is the
40-column card content. There is no `maxCells` in the sidecar and
`_notice_lines` counts the wrapped lines into the height budget, so this
cannot clip the consent block. No finding.

## Negative controls

- **E1** restores the round-1 value with `el aviso` → B1 rejects it
  (forbidden surface + missing `prompt`). Caught.
- **E2** restores the round-2 value with `el diálogo` → B1 rejects it. Caught.

## Findings

- **HL-1 · major · CLOSED** — the revised string names `el prompt`, the
  inline consent prompt, and no longer names the notice line or a dialog.
- **No new findings.** No minor, no nit. The other 55 values were not
  re-audited: round 2 passed them, and this narrow round only re-checks the
  file's mechanical integrity around the one edit.

## Limitations

LLM judgement; no native-speaker attestation is claimed. No rendered TUI
frame was captured: widths are computed with `rich.cells.cell_len` against
the geometry the source documents, and the notice wrap is computed with
`rich.text.Text.wrap`, not photographed. No catalogue, sidecar, ledger,
allowance, style guide or generated file was written; only this evidence
directory.

## Ledger row — prepared, NOT written

This round forbids ledger writes (no git writes; no catalogue/sidecar/ledger/
allowance/style-guide edits), so `i18n/ledger/es.json` was not touched. Row
for the ledger owner:

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
    "result": "passed",
    "findings": 0
  },
  "status": "passed",
  "note": "w6a3 r3 passed (HL-1 closed; 0 blocker/major). Evidence i18n/audits/evidence/es/tui.hub_links/w6a3/record.json; es ff50b7cd…, en 7fbe6713…; independent, radient/auto, 2026-10-30. revision_moved names `el prompt` (glossary), not `el aviso`/`el diálogo`."
}
```

Note-length check: the `note` is 252 characters, inside the 280-character
budget.

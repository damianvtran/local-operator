# French A audit — tui.model_persistence

**Verdict: passed.** Four of four keys audited independently. No BLOCKER, MAJOR, MINOR or NIT findings. This is A only; the separately assigned B pass remains required. No target/source/context edits, ledger append, stage, commit or push were performed by this auditor.

## Provenance and admission

Auditor: i18n-auditor, task model-notice-fr-audit-A, session `0ecda03ab969`, actual model `openai/gpt-6.1-sol`. Fallback from `anthropic/claude-opus-5-5` due to provider quota was explicitly announced by the runtime's model-switch user event before the substantive audit. Independent from translator session `1ca940cf73ba` and source author `a97c7f76d5ee`. Translator identity/model were extracted only from the author record's translator field; author notes/opinions were not used to reach the verdict. No MT service/API was used.

English/context were read through read_public_copy at source freeze `c40586cd0ae23ca25a6c82c26c4f867271c4e01f`. The initial target read was refused because the file was uncommitted. The manager then supplied exact admission revision `87d87ec8b02ef3c279fda673302ce775f4a0329e`; read_public_copy subsequently returned all four French values successfully. The refusal is preserved in admission-notes.json as a tool-response transcription, not an execution log. No pass is attributed to the blocked attempt. Input hashes were checked before and after actual successful validation and match the delegation exactly.

## Key-by-key linguistic reasoning (audit notes, not execution log)

| Key | Current French | Assessment |
|---|---|---|
| `tui.model_persistence.persist` | `/model default enregistre ce modèle pour les nouvelles sessions` | PASS. `enregistre` explicitly saves the currently indicated model; `pour les nouvelles sessions` locates its effect in future sessions, not a switch of this session. `ce modèle` makes English `this` natural and unambiguous. No omitted persistence meaning. |
| `tui.model_persistence.settings` | `/settings permet aussi de modifier le modèle par défaut au démarrage` | PASS. Natural product-register French. `modèle par défaut au démarrage` conveys boot default, rather than the active model. `aussi` preserves `too`; when this is the sole available hosting-setup route it does not invent or direct the user to any /model command. No need to redesign the source clause. |
| `tui.model_persistence.saved` | `appliquez à cette session le modèle par défaut au démarrage avec /model saved` | PASS. `appliquez à cette session` expressly changes THIS session's model. It does not say `enregistrez` or suggest saving for future sessions. `avec /model saved` names the exact switch command. The vous imperative is professional and direct, not bureaucratic. `au démarrage` qualifies the default model; it does not delay the requested application. |
| `tui.model_persistence.with_model` | `modèle : {label} — {routes}` | PASS. Human prefix translated using glossary `model → modèle`; U+202F before the colon follows the French style guide. Em dash has surrounding spaces. Opaque model ID and already-localized complete route clauses are interpolated unchanged exactly once each. |

Glossary decisions: `modèle`, `session`, `enregistrer`/`modifier` are French prose; `/model default`, `/model saved`, `/settings` are named English command exceptions. `label` and `routes` remain machine argument names. No unapproved English human clause, polarity change, omitted remedy, wrong register, calque, apostrophe or punctuation issue was found. French CLDR categories are one/many/other when plurals exist; these four messages contain no plural/select arguments, so no branches are required or invented.

## Actual validation and limits

`audit.log` contains the successful actual stdout of `validation.py.txt`, run with the primary checkout's `.venv/bin/python`, worktree PYTHONPATH, isolated HOME/config and CMUX_* stripped. Actual module resolution printed the worktree's `local_operator/i18n/catalogues.py`. The official `scripts/i18n/check.py::check_catalogues()` was scoped using patched namespace/locale enumeration to exactly this namespace and en/fr; its implementation was not changed. Output: `OFFICIAL check.check_catalogues scoped en/fr + one namespace: []`. This is not a claim that the unscoped/base-aware checker or global suite passed.

Duplicate-free JSON, exact ordered four-key parity, ICU name/type parity, one occurrence each of label/routes, zero plural/select arms, unchanged command bytes, and absence of added markup/code were checked. All four messages rendered with the runtime ICU renderer. Synthetic ordinary, missing-model and hosting-dead-end compositions were checked with 3- and 108-character opaque labels. Hosting-dead-end routes contained only /settings; opaque label content was not misclassified as a command route.

Rich plain-text wrapping was measured at 80/100/120 columns, both full width and an explicitly synthetic 4-cell inset. Across 36 cases: no row exceeded its budget, no protected command split across rows, and no final row contained only a command. Long labels can break across rows; at 80 columns the long-label ordinary composition occupies 6 rows. This is a wrapping/height risk for integration to inspect, not evidence of a fixed-width cap violation: the context specifies no maxCells/maxChars. Human-letter pseudo expansion preserved commands and arguments and rendered successfully for all four keys. These are plain render/width controls, not actual pristine/non-empty NoticeBlock geometry, screenshots, native terminal/font certification or whole-app integration proof. Root owns that later rendered integration.

Two harness failures are retained separately: dynamic import initially omitted sys.modules registration (dataclass import failure), then a too-broad negative-control assertion treated `/model-with…` inside the deliberately opaque long label as a /model route. Fixes affected this auditor's validation harness only. Final actual run passed and checked all input hashes after execution. No product failure is inferred from either harness defect.

## Finding ledger

None. Unresolved BLOCKER/MAJOR IDs: `[]`; count: `0`. No suggested replacement wording is necessary for clean keys. This audit does not grant B-pass, eligibility, release or native integration approval.

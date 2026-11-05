# C4 `tui.sound` independent i18n audit — round 2

Reviewer: C4-i18n-delta / session 304e031dccf4 / openai/gpt-6.1-sol. No fallback.
Scope: 27695a49855f..bd9ad6f041ca (PR #3433; commits 490c422 and bd9ad6f).
Verdict: **passed / terminal for the corrected artifact bytes**, 0 unresolved BLOCKER/MAJOR. This does not claim the pinned PR head already contains the artifacts.

## Scope and provenance

Operator request (verbatim): “the sound settings probably need to be a backend centralized setting because the TUI can also notify and we should be able to modify the sound from there too. Keep them editable in the section that you added them in the UI but also have control and settings to be able to update that on the TUI side”.

This stream audits only the translation delta: changed `id_help`, `list_title`, `list_yours`, `usage`; added `list_hint`, `list_yours_hint`, `section_description`; deleted `default`. No re-audit of unchanged application behavior or unrelated copy.

The independent translator subagent was `i18n-translator-r2`, session `1d36d618f249`, job `1ddc1da8273c`, actual `openai/gpt-6.1-sol`, date 2026-11-04, no fallback. I compared all 49 proposed values against the exact source, context, glossary and seven locale style guides; made the narrow corrections in `audit-corrections.json`; then independently audited the corrected bytes. Translator receipt and raw draft are preserved, not substituted for independent approval.

Source SHA-256: `ba467980099545040c0860e47d673d50b5f2f60cd261426538ccf64c3a13e888`.
Context SHA-256: `900c1a700da76e20651b1d976ac9c1cc6add638d5b61ce10fffae08869924fad`.
Pinned full head: `bd9ad6f041ca863a892f963177d0b17741d14266`.

## Semantic decisions and corrections

- `id_help`: arrow-navigation playback comes first in every locale, then add/doctor/system. The source explicitly permits ellipsis in the settings row; the complete help intentionally exceeds 96 cells, and is not falsely reported as fitting that cap. `system` and all commands remain verbatim.
- `list_title`/`list_yours`: bare, idiomatic headings; command instructions are no longer packed into them. The two separate hint lines fit the 54-cell budget in every locale.
- `usage`: ASCII `<0-100>` and optional `[--select]` preserved exactly, along with every command/argument token.
- `section_description`: retains desktop-notification coupling and the cap of **at most one playback per 1.5 seconds**, not 1.5 times per second; also preserves arrow preview and `/sound`. Decimal comma is used in es/fr/ru/vi prose while preserving the numeric value. Source sidecar requires `/sound` verbatim, not an English decimal separator.
- Russian preview hint: changed bare `звук` to `играет` so it explicitly conveys playback; still 50/54 cells.
- Hindi/Urdu polite register and punctuation; French vous/typographic spacing; neutral Spanish tú; Russian impersonal/lowercase-вы conventions; Vietnamese NFC/pronoun-light copy; concise Chinese 你 register/full-width prose punctuation. Urdu stays logical-order with no manual bidi controls.

All corrections are recorded with before/after/reason. No unresolved semantic finding remains for the corrected artifacts.

## Carry-forward proof and legal ledger handling

Each complete catalogue contains **29 keys**: seven independently audited delta values plus **22 unchanged values** carried byte-identically from round 1. Across seven locales: 49 new audited values + 154 retained values = 203. `default` is absent everywhere. `carry-forward-proof.json` verifies the actual round-1 target hash against its audit record, the exact unchanged values, and byte-identical glossary/style inputs. Original translator/auditor provenance remains in the copied c4r1 evidence; it is not claimed as work performed anew by this session.

Per-locale shards are authoritative; generation mirrors the aggregate. Append the seven new current-source rows. Preserve prior history; never relabel the old source hash. The actual pinned head has no tui.sound locale files or ledger rows, so unmerged round-1 proposal rows are preserved as inputs rather than fabricated as merged history. If another lane has already integrated round 1, append round 2 after it. The full shard overlays are reference-only snapshots.

Remove exactly seven sound allowance pairs, **29 keys each / 203 total**, not round 1's obsolete 189-key removal. No allowance growth is introduced by this artifact integration.

## Fresh verification

Validation used a detached worktree at the pinned PR head, its own uv-created venv, isolated HOME/config, env-i (no CMUX_*), null sound backend, disabled real notifications and speaker playback. One process at a time, 1500 MiB ceiling, pytest `-n 1`.

| Surface | Actual command/output | Result |
|---|---|---|
| Generator drift | `generate.py --check`: **127 artifact(s) byte-identical**, rc=0 | PASS |
| Full checker | `check.py --allowance-base 27695a49855f`: **202 files scanned, 2411 literals at or below baseline; catalogues consistent; 63 advisory findings (report-only)**, rc=0 | PASS |
| Coverage | `pytest tests/unit/i18n/test_language_manifest.py -n 1 -q`: **35 passed in 9.02s**, rc=0 | PASS |
| Catalogue/runtime | **29/29 messages × 7 locales** render; key/argument/technical-token parity, NFC, logical bidi storage, retained-value identity | PASS |
| Negative control | Remove French selected-message `{name}`: checker rejects exact argument mismatch; original bytes restored | PASS |
| Fallback | Remove French `preview_silent`: production resolver returns English **Nothing played: the volume is 0.**; bytes restored | PASS |
| Width | list_title ≤40; list_yours ≤30; both hints ≤54 in every locale. list_hint cells es49/fr50/hi43/ru50/ur47/vi47/zh-CN45; yours_hint46 | PASS |
| Painted linguistic sample | Real `OperatorApp` + production CSS, English + seven locales, 60-column synthetic populated custom-sound list and translated help/description/usage notices; SVG/PNG/geometry and consecutive frames | Evidence in `frames/` and `render-evidence.json` |

The default moving-origin/main allowance comparison first hit the 600-second bound (rc=124), **not** a checker defect. The retry used the previous reviewed head as explicit delta base and a larger bound; full scanner/parity still ran and passed. Logs retain both runs. Actual integration must check against its real PR base before merge.

Darwin `run_bounded` cleanup-required refused before spawn (rc=125). Subsequent commands explicitly used its supported `command-status` contract, with bounded memory/serial execution. That contract reports cleanup unknown; no certified process-group reap is claimed. No fallback model or dependency-tree sharing workaround was used.

## Limits

Frames are synthetic Textual compositor paint, not the user's live terminal. Native Arabic shaping and full RTL are not certified: Urdu is visibly degraded under the documented LTR/best-effort TUI stance. The settings-only help and palette section description are painted as transcript notices, not falsely presented as a `/settings` interaction audit. No speakers, remote/cmux sessions, native-speaker certification, full design approval or application QA sign-off is implied. Existing round-1 frames/evidence are preserved only as named-source provenance.

The original capture exposed unrelated welcome-logo startup reflow; the final capture waits for that entrance before its consecutive pair. Actual rows/geometry are recorded rather than assuming stability. No application code was changed for the rig.

## Handoff

`INTEGRATION.md` names the actual branch, narrow copy/append/removal instructions and commands to rerun. `SHA256SUMS.json` binds deliverables. Only the scratch worktree and artifact bundle were modified; the shared checkout and PR branch were not edited.

# C4 `tui.sound` independent i18n audit — round 1

Reviewer: C4-i18n-audit / session 796c1db4d0d0 / openai/gpt-6.1-sol. No fallback.
Scope: 46053ab26a63..27695a49855f (PR #3433).
Verdict: **passed / terminal for the corrected artifact bytes**, 0 unresolved blocker/major. This is not a claim that the not-yet-integrated PR head carries these translations.

## Independence and source binding

The translator was the C4 coder, notif-sounds-C4-tui, job 16bed17d35eb, session e87ab06e319f, anthropic/claude-opus-5-5 (manager-supplied provenance). I independently compared every draft value to the pinned English source and context, read the glossary/style rules, corrected narrow wording/fit issues, and audited the resulting artifacts. I did not author the underlying drafts.

The preserved English source and context match the PR head **byte for byte**. Source SHA-256: `0d23d23b34918c3bdb003afbf6ba6f2694316a7c55214142428185956e4fa578`.

All 27 keys × es, fr, hi, ru, ur, vi, zh-CN = **189 reviewed values**. Each locale's record binds input/output hashes, source/context/style/glossary hashes, actual identities, reviewed keys and dispositions. The primary checkout's es/fr style files differed from the pinned snapshot; I read their exact diff: only unrelated cwd/allowance/pull/usage/analytics and relay-width amendments differ, none applies to `tui.sound`. Other guides and glossary are byte-identical.

## Narrow corrections included

- ES: compact, consistent infinitive slash description; 49 cells (cap 55).
- FR: help explicitly says arrow-key browsing rather than “the arrows stop”; narrow nonbreaking spaces before prose colons/question mark. 52-cell slash summary.
- HI: `पूर्णांक` names an integer precisely. Polite आप register and Latin digits retained.
- RU: help explicitly names arrow keys; invariant duration `с` avoids decimal-dependent agreement. 53-cell slash summary.
- UR: native settings terminology in save refusal; shorter respectful slash summary, 50 cells instead of the draft's 56 (cap 55).
- VI / zh-CN: full semantic pass without rewriting drafts. Chinese retains separating whitespace around technical commands/numbers for terminal readability; this is a minor typography exception to the no-extra-space preference, not a changed command.

Full before/after values: `audit-corrections.json`. No lost command, placeholder, branch, duration, state, system delegation, refusal reason, SSH remote-speaker qualification or environment-variable token.

## Execution evidence

All substantive work stayed under this artifact root. A sparse `git archive` of the pinned head was used, with existing checkout `.venv/bin/python`; there were no dependency installs/copies and no repo edits. Each render log prints the scratch catalogue root. Runs were serial and bounded to 1500 MiB; pytest used `-n 1`.

| Surface / command | Actual result | Verdict |
|---|---|---|
| `scripts/i18n/generate.py` | rc 0; only aggregate ledger differs from pinned generated assets | PASS |
| `scripts/i18n/generate.py --check` (final) | `127 artifact(s) byte-identical`, rc 0 | PASS |
| `scripts/i18n/check.py` (final) | `202 file(s) scanned, 2411 literal(s) at or below the baseline, catalogues consistent; 63 advisory finding(s)`; rc 0 | PASS, advisories not promoted to global enforcement |
| Coverage allowance | Removed precisely 7 `tui.sound` pairs / 189 now-present key IDs | PASS |
| `pytest -n 1 tests/unit/i18n/test_language_manifest.py::test_every_missing_key_is_covered_by_the_shrink_only_allowance -q` | `1 passed in 5.99s`, rc 0 | PASS |
| `catalogue_probe.py` | 189 runtime renders; exact argument/placeholder-order/technical-literal parity; NFC and no stored bidi controls; all 7 slash descriptions ≤55 cells | PASS |
| Missing-placeholder negative control | Repository checker rejected `fr/tui.sound.selected`: `arguments () != en's (('name', 'simple'),)` | PASS (expected rejection) |
| Restored parity control | No catalogue errors after finally-restoring exact bytes | PASS |
| Missing French-key fallback | Production resolver returned exact English `Nothing played: the volume is 0.` | PASS |
| Real `OperatorApp.run_test`, 120×40 | Status, invalid volume, `preview system`, invalid verb usage, and same help text painted through transcript; en + 7 locales captured and rasterized, all PNGs viewed | PASS for linguistic rendering sample |
| Synthetic long-copy stress | Bracketed help plus six expansion phrases wrapped and retained final marker; captured PNG viewed | PASS; not a full pseudo locale |

No ICU plural/select messages exist in this namespace; therefore no invented CLDR branch claim. Decimal seconds are text placeholders, not plural selectors.

## Limits and honest scope

The archive has no Git metadata, so the checker's Git-base allowance-growth comparison is skipped. The overlay only removes existing allowance IDs; the integrator must rerun checks on the actual branch.

This is not live-terminal shaping, a native-speaker review, a second safety/design audit, audio playback proof or settings UX sign-off. Urdu raster evidence visibly retains the documented degraded terminal treatment; do not label it fully shaped/RTL-complete. The settings help sample uses the real app's transcript paint, not an assertion that `/settings` was exercised. Header animation differs across captures and is not reviewed as layout proof here. The null sound backend, absent CMUX_* variables, isolated HOME/config and synthetic session prevented real speaker/cmux use; the only preview command was the system no-op.

The bounded wrapper reports cleanup unknown on Darwin. Each exact recorded render PGID was checked with `ps` afterwards; none remained. This is an observed PGID check, not an authoritative descendant census.

Initial full export hit the 256 MiB scratch cap and was replaced with this sparse copy. Preflight config-schema and cache-name mistakes were corrected before the final successful runs; see audit-evidence.json. No such failed setup was counted as passing evidence.

## Integration

See `INTEGRATION.md`. Do not copy whole ledger/allowance snapshots over newer concurrent edits. Append the seven proposed rows, remove only the seven specified paid-off allowance pairs, then regenerate the aggregate rather than hand-edit it.

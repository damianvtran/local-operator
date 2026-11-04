# vi / tui.small: core catalogue gap audit (cg1)

137 keys, all judged by the round-1 auditor against the English source and context sidecar; 3 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **VI-001** (round 1, MAJOR) `tui.small.goal_panel.judge_absent`
  - English: 'judge: —'
  - Value at audit: 'giám khảo: —'
  - Problem: Context sidecar: `judge:` is protocol vocabulary kept as-is; es/fr/zh-CN all keep 'judge:'. Translating it to 'giám khảo:' breaks the protocol label (translator's ambiguous.txt call rejected).
  - Suggested: 'judge: —'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-002** (round 1, MAJOR) `tui.small.goal_panel.judge_line`
  - English: 'judge: {words}'
  - Value at audit: 'giám khảo: {words}'
  - Problem: Same as judge_absent: `judge:` is a protocol word per the context note and must stay byte-identical; 'giám khảo' (exam examiner) is also the wrong sense.
  - Suggested: 'judge: {words}'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-003** (round 1, MINOR) `tui.small.network_panel.relay_not_running`
  - English: '  relay:      not running\n'
  - Value at audit: '  chuyển tiếp: không chạy\n'
  - Problem: Column padding: 'relay:' rows align the value at cell 14 like '  đã cài:     ' and '  danh tính:  '; '  chuyển tiếp: ' puts the value at cell 15, so the column is ragged. Context also calls `relay:` a protocol word (fr/zh-CN keep it).
  - Suggested: "  relay:       không chạy\\n  (or keep 'chuyển tiếp:' and widen installed/identity padding to 15 so all rows align)"
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-004** (round 1, MINOR) `tui.small.network_panel.relay_running`
  - English: '  relay:      running, pid {pid} — {state}\n'
  - Value at audit: '  chuyển tiếp: đang chạy, pid {pid} — {state}\n'
  - Problem: Same misalignment as relay_not_running (value column at 15 vs 14 on sibling rows).
  - Suggested: '  relay:       đang chạy, pid {pid} — {state}\\n (or realign all rows to one column)'
  - Disposition: declined: Same as VI-003: the sibling rows were realigned to cell 15, so this row's value is already in the shared column.
- **R2-VI-004** (round 2, MINOR) `tui.small.network_panel.relay_running`
  - English: '  relay:      running, pid {pid} — {state}\n'
  - Value at audit: '  chuyển tiếp: đang chạy, pid {pid} — {state}\n'
  - Problem: Decline/alternative for VI-003/004: the column is now aligned (all four rows put the value at cell 15, measured), but the sidecar calls `relay:`, `installed:` and `identity:` protocol labels («kept as-is» for identity); fr/zh-CN keep them Latin. es on main translates them, so this is not blocking. The same applies to relay_not_running, relay_checking_indent, installed_row and identity_row.
  - Suggested: 'Preferred: revert all five rows together to Latin labels with English padding, e.g. «  relay:      đang chạy, pid {pid} — {state}\\n», «  installed:  {value}\\n», «  identity:   {identity}\\n». Otherwise record «chuyển tiếp/đã cài/danh tính» as a vi exception in the style guide.'
  - Disposition: declined: Kept «chuyển tiếp:/đã cài:/danh tính:». Glossary §2 row 'relay | translate — identifiers stay' gives vi «chuyển tiếp» and the style guide lists relay → chuyển tiếp; these are human-read panel rows, not parsed identifiers, and es on main translates them. Columns stay aligned at cell 15. Recording the
- **R3-VI-001** (round 3, MINOR) `tui.small.network_panel.relay_running (+relay_not_running, relay_checking_indent, installed_row, identity_row)`
  - English: '  relay:      running, pid {pid} — {state}\n'
  - Value at audit: '  chuyển tiếp: đang chạy, pid {pid} — {state}\n'
  - Problem: Decline of R2-VI-004 is reasoned (glossary relay→«chuyển tiếp», es precedent, columns aligned at cell 15 — re-measured) but it contradicts the per-key sidecar: `relay:`/`pid` are 'protocol words', `identity:` is 'a protocol label kept as-is'. Meaning is intact, so not MAJOR; the exception is now only recorded in scratch ambiguous.txt, not in the style guide.
  - Suggested: 'Either revert all five rows together to Latin labels with English padding, or record «chuyển tiếp:/đã cài:/danh tính:» as a vi exception in i18n/style-guides/vi.md in the same PR that ships these values.'
  - Disposition: open at PASS: minor/nit accepted, value unchanged

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session 3ca692471511)
# vi core catalogue patch audit — independent
Verdict: **CHANGES_REQUESTED** — BLOCKER 0 · MAJOR 31 · MINOR 17 · NIT 5 · 53 findings over 1311 values in 56 namespaces
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback). Independent of the translator session 3ef6295a5545. No MT/translation service used.
Inputs: patches in core-i18n/vi/*.json; English source, context sidecars, glossary, vi style guide and existing vi catalogues read from origin/main (extracted via one `git cat-file --batch`; copies under core-audit/vi/src/).
## What blocks PASS (31 MAJOR)

## Not verified
- No rendering in the real TUI and no repo validator run, because the brief makes the repo read-only with no builds or tests. Widths were checked with rich.cells.cell_len only where the sidecar declares maxCells. Widths of tui strings without a budget were not measured against live layouts.
- The audit-ledger row was **not** recorded and no tracked evidence bundle was written. The brief made the repo read-only, which overrides the auditor profile's ledger step. Outputs live only in this scratchpad: someone must copy them into i18n/audits/evidence/vi/… and add the ledger row once the remediation passes.
- NIT findings are capped at 5 per the auditor brief. Two lower-value NITs were dropped.
### Round 2 (session a3954428c96c)
# vi core catalogue patches — independent round-2 audit
**Verdict: CHANGES_REQUESTED.** BLOCKER 0 · MAJOR 2 · MINOR 5 · NIT 2 (9 findings, in `findings.jsonl`).
- **Auditor:** i18n-auditor, session a3954428c96c, model `anthropic/claude-opus-5-5`, no fallback.
- **Independence:** I did not write these values or the round-1 findings. The translator was session 3ef6295a5545, and round 1 was a separate auditor.
- **Method:** No MT or translation service was used. The repo was read-only: no builds, tests or worktrees.
- **Sources:** glossary, vi style guide, and en catalogues + context sidecars from `origin/main` @ `5a6683f6815a1c667a85fb8aa7c76a156bf01c32`.

## Not verified
- I ran no repo validator and did no live TUI render, because the brief made the repo read-only with no builds or tests. Widths are verified only where the sidecar declares `maxCells`.
- I wrote no ledger row or tracked evidence bundle (the repo is read-only). Someone must copy these outputs into `i18n/audits/evidence/vi/…` once a later round passes.
- I did not re-read the ~1,200 untouched values beyond the sweeps listed in (C).
### Round 3 (session 0cc22c4c8b4f)
# vi core catalogue patches — independent round-3 audit
**Verdict: PASS.** BLOCKER 0 · MAJOR 0 · MINOR 2 · NIT 1 (3 findings in `findings.jsonl`).
- **Auditor:** i18n-auditor, session 0cc22c4c8b4f, model `anthropic/claude-opus-5-5`, no fallback. I am independent: I did not write the translations, the round-1 findings or the round-2 findings (translator 3ef6295a5545; round 2 a3954428c96c).
- **Method:** reasoning only, with no MT or back-translation. Read-only on the repo: no builds, tests or worktrees. To work around slow `git show`, I extracted the en catalogues and sidecars from `origin/main` @ `1774a28df5a96cb0bffb9f4673c78e9b6d82e2b6` into my own scratch with `git archive`. I read the glossary (including the Wave-2 kept-English set) and `style-guides/vi.md` first.
## (A) The 8 changed values (changes.jsonl)

## Not verified
- I did not render anything in a real TUI. Width is computed, not seen in a frame.
- The 196 tui.import/library/hub_links and other values that lack sidecar context entries could only be checked mechanically and swept. They have no maxCells to measure.
- I did not re-read the untouched values beyond the sweep.
- I did not record an audit-ledger row or evidence bundle, because this task is read-only on the repo.

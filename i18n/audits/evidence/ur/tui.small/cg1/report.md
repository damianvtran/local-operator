# ur / tui.small: core catalogue gap audit (cg1)

137 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **UR-001** (round 1, MINOR) `tui.small.model_picker.effort`
  - English: 'effort {value}'
  - Value at audit: 'effort {value}'
  - Problem: Label word 'effort' left in English; it is not a token (the {value} list is). The ur relay surface already names this concept «استدلال کی شدت» (relay.chrome.model.effort).
  - Suggested: 'استدلال کی شدت {value}'
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)
- **UR-002** (round 1, MINOR) `tui.small.session_picker.all_archived`
  - English: 'all {total} conversation(s) are archived'
  - Value at audit: 'تمام {total} گفتگو(ئیں) محفوظ شدہ (آرکائیو) ہیں'
  - Problem: «محفوظ شدہ» reads as 'saved', not 'archived', so the parenthetical gloss is needed to rescue it; «گفتگو(ئیں)» is an awkward slash-plural. Neighbour wire.slash.archive frames archiving as hidden from lists.
  - Suggested: 'تمام {total} گفتگوئیں آرکائیو شدہ ہیں'
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)
- **UR-003** (round 1, MINOR) `tui.small.network_panel.identity_row`
  - English: '  identity:   {identity}\n'
  - Value at audit: '  شناخت:   {identity}\n'
  - Problem: The English rows (relay/installed/identity) pad labels so every value starts at the same column (14). The ur padding was copied per-row without re-aligning: values start at col 13 (ریلے), 15 (انسٹال شدہ), 11 (شناخت), so the panel's column is ragged. Same for installed_row/relay_* rows.
  - Suggested: "Re-pad as a set to a common column, e.g. '  شناخت:      {identity}' / '  انسٹال شدہ: {value}' / '  ریلے:       …'"
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)
- **UR-004** (round 1, NIT) `tui.small.aside.superseded`
  - English: '(superseded)'
  - Value at audit: '(منسوخ شدہ)'
  - Problem: «منسوخ شدہ» = 'cancelled'; the marker means replaced by a newer question. Acceptable per sidecar, but «بدل دیا گیا» states the actual reason.
  - Suggested: '(بدل دیا گیا)'
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)
- **UR2-005** (round 2, NIT) `tui.small.session_picker.all_archived`
  - English: 'all {total} conversation(s) are archived'
  - Value at audit: 'تمام {total} گفتگوئیں آرکائیو شدہ ہیں'
  - Problem: The value is now always plural, so it reads «تمام 1 گفتگوئیں» when total=1. The English '(s)' has the same problem and there is no ICU arm, so this is acceptable.
  - Suggested: 'Optional: «تمام گفتگوئیں ({total}) آرکائیو شدہ ہیں»'
  - Disposition: open at PASS: minor/nit accepted, value unchanged

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session dab7435f3cd0)
# ur core-catalogue patch audit — independent pass
Auditor: i18n-auditor subagent, session dab7435f3cd0, model anthropic/claude-opus-5-5 (as listed for this session; no fallback observed). Independent from translator session 3ef6295a5545. No MT service was used. Repo read-only (git show origin/main only); no builds, tests or worktrees.
**Verdict: CHANGES_REQUESTED.** BLOCKER 0 · MAJOR 1 · MINOR 18 · NIT 5.
The one MAJOR is `tui.hub_links.policy.auto_push.help`. «محفوظ ہونے پر» can be read as "when saved" as well as "when safe", and this locale uses محفوظ کریں for Save. So the string can read as consent to publish on every save, which is broader than what the English allows.

## What I could not verify
- Rendered width in a real terminal. Widths were measured with rich.cells.cell_len on the raw template (placeholders unexpanded), and only for the 30 keys with maxCells.
- RTL rendering in iTerm2 or other terminals.
- The repo validator and the ledger command. Running them was out of scope because the task is read-only, so no ledger row was recorded.
- Whether the context-less namespaces (tui.import, tui.library, wire.errors.agents/auth/files/media, wire.exec, wire.slash.library) have UI constraints beyond what their English shows.
- Most patched namespaces have no existing ur neighbour on origin/main. Terminology was cross-checked against the 44 ur catalogues that do exist.
### Round 2 (session 1a185836d3d9)
# ur core-catalogue patches — independent round-2 audit
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback observed). I did not translate these values and did not write the round-1 findings. No MT service was used. The repo was read-only: I used `git show origin/main` only, with no builds, tests, worktrees, validator or ledger command.
**Verdict: PASS.** BLOCKER 0 · MAJOR 0 · MINOR 2 · NIT 3. Details are in findings.jsonl.
## (A) The 62 changed keys
I judged every after-value fresh against the English, the glossary and the ur style guide. All 62 match the current patch files. All are accepted, and the notes below record why.

## What I could not verify
- Rendered output in a real terminal, including RTL/bidi on iTerm2. Widths are template widths, measured with placeholders unexpanded.
- The repo validator and the ledger write, because the task is read-only. No ledger row was recorded.
- Gender and number agreement in plural/select arms is right for the values I read, but untouched values were only swept by pattern, not re-read.

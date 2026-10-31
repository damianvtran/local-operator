# ur / tui.hub_links: core catalogue gap audit (cg1)

56 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **UR-009** (round 1, MAJOR) `tui.hub_links.policy.auto_push.help`
  - English: 'Publish local changes to {source} automatically when safe.'
  - Value at audit: 'محفوظ ہونے پر مقامی تبدیلیاں خودکار طور پر {source} پر شائع کریں۔'
  - Problem: «محفوظ ہونے پر» is ambiguous between 'when safe' and 'when saved' — and this locale uses «محفوظ کریں» for Save (style guide), with tui.hub_links.policy.saving «محفوظ ہو رہا ہے…» on the same screen. A user can read it as 'publish local changes automatically whenever they are saved', broadening an auto-publish consent string.
  - Suggested: 'جب ایسا کرنا بے خطر ہو تو مقامی تبدیلیاں خودکار طور پر {source} پر شائع کریں۔ (or: جب شائع کرنا محفوظ ہو، …)'
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)

## Rephrased after the locale-wide audit, then audited by targeted session b15ab322fa0f (PASS, keys_read=7 blocker=0 major=0 minor=1 nit=1)
- `tui.hub_links.mark.available`
- `tui.hub_links.mark.retry`
- `tui.hub_links.policy.save_failed`
  - UR-ARG-1 (MINOR) `tui.hub_links.mark.retry`: Meaning intact (retry direction, for item, at source; nothing inverted), but the colon+comma list drops the 'for' relation and reads like a terse label list rather than a sentence; slightly stilted. (accepted unchanged)

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

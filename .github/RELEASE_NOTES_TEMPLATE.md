# <Short theme headline>

<One or two plain-language sentences explaining what changes for users in this
version. Describe the benefit, not the development or release process.>

## New

- <New capability and when a user would use it.>

## Fixed

- <Previously broken behavior and what now works.>

## Improved

- <Existing behavior made clearer, faster, or more reliable.>

## Install or upgrade

Install or upgrade using your usual Local Operator installation method. Restart
running sessions when you are ready to use the updated version.

**Full changelog**: https://github.com/damianvtran/local-operator/compare/<previous-tag>...<this-tag>

---

## Authoring instructions (do not publish this section)

This is the public-copy contract, not a release-evidence template. `AGENTS.md`
remains authoritative for release ownership, validation and publication.

- Replace every placeholder. Omit empty New / Fixed / Improved sections; use
  these categories rather than bump-size labels (Major / Minor).
- Derive each bullet from a merged PR's `Release:` impact line, checked against
  the actual tagged window. Rewrite technical wording into user impact without
  inventing capabilities, measured speedups or companion-app distribution.
  Missing impact lines must be supplied in the source PR before publication.
- Internal-only release: say “Internal maintenance only; no user-visible
  changes.” Version-only release: say “Version metadata only; no functional
  changes.” Do not invent a feature bullet to fill a section.
- Keep meaningful limitations in user terms (for example, a companion app
  updates separately). Publication is not proof that running sessions upgraded.
- Never include agent/QA/design round narratives, CI counts, reviewer names,
  candidate or merge SHAs, branch names, residual-red classifications, or
  internal IDs (including PR/ticket/session/finding IDs). Valid user-facing
  command and configuration names are fine. Development evidence belongs in
  separate PR comments, not this body or its public-notes section.
- The claim/bump PR's `## Public release notes` section and the final notes file
  must both follow the public portion of this template. While claiming a window
  before its contents are known, use an explicit “Draft — contents not yet
  determined” placeholder, not invented impact; replace it before ready/tagging.

### Prepublish checklist (complete in the release PR, never in the notes file)

- [ ] Headline and summary describe user impact; no placeholders or authoring
      instructions remain in the publication file.
- [ ] Every tagged-window impact is accounted for from merged `Release:` lines;
      internal-only and version-only windows are described honestly.
- [ ] Only nonempty New / Fixed / Improved sections remain; bullets contain no
      internal IDs, SHAs, branch/reviewer names, round narratives, CI counts or
      residual-red classifications.
- [ ] Install/upgrade guidance matches supported installation methods, makes no
      claim of a completed runtime upgrade, and promises no companion release.
- [ ] Compare URL uses the actual previous published tag and this release tag.
- [ ] Read the exact notes file and the claim/bump PR public-notes section side
      by side; they match. After publication, read the exact
      `/repos/damianvtran/local-operator/releases/tags/<this-tag>` response body
      and verify it equals the approved notes file.

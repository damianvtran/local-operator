---
name: code-requests
description: "When to use the code_requests tool versus gh/glab for PR/MR state, review rounds, CI and freshness."
---

# Code requests (PRs and MRs) of a session

The `code_requests` tool answers "which pull requests and merge requests did
**this conversation** touch, and where is each one up to?" — including the
review-round convention this fleet merges by. Prefer it over ad-hoc `gh`/`glab`
calls **when the question is about this session's work or about review rounds**:

* `list` — the session's rows: relations (`opened`, `acted`, `mentioned`,
  `unknown`, `inherited`), states and links. No network.
* `show {ref}` — one row in full: state, CI, per-lane review rounds (round,
  state, freshness vs the current head, verdict), and quoted excerpts of the
  convention comments. `ref` can be any PR/MR URL or qualified ref
  (`owner/repo#123`, `group/project!45`), including one this session never saw.

Reach for `gh`/`glab` directly when you need something the tool does not carry —
a diff, a file, an action (comment, merge, push) — or when no `code_requests`
tool is present.

## What it reads, and the review-round convention

The tool parses the comment convention this team merges by:

`### Agent review — round N`, `### Design review — round N`,
`### QA report — round N`, `### UX review — round N` (and
`### … remediation — round N`), with `Reviewer:`/`Scope:`/`Head:`/`Verdict:`
fields in the first lines. Lane states: *awaiting review*, *findings open*,
*remediation posted*, *clean*, *terminal*, *reviewed — verdict not stated*;
freshness is a prefix match of the reviewed SHA against the current head —
`fresh`, `stale` (both SHAs are shown) or `unknown`, and **unknown is never
guessed at**. A verdict that cannot be classified is reported as unstated, not
invented. Comment author is never used to judge independence (the account is
shared — the `Reviewer:` text is, and is reported verbatim).

## Supported hosts

| Host | Fetch |
|---|---|
| GitHub (github.com + GitHub Enterprise) | full |
| GitLab (gitlab.com + self-hosted) | full |
| Gitea/Forgejo/Codeberg, Bitbucket, Azure DevOps, Gerrit | detect-and-link (the row exists and opens; no state) |

Rows without a usable login degrade to link-only with the sign-in remedy
(`gh auth login` / `glab auth login`); a rejected or missing credential is
never an error wall. Data is cache-backed and revalidated conditionally, so a
revealed `stale` marker means the forge could not be refreshed this pass —
read it as "last known", not as "current".

## Tracking and monitoring

Rows are derived from the session's own activity as it happens: opening,
commenting, pushing, merging, reviewing. You can be told when something moves:
arm `monitor(code_requests show <ref>)` — a scheduled check of the read-only
tool — instead of polling by hand.

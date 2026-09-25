# Aida

Aida is the always-on review and triage agent for this repository. She reviews
pull requests, triages issues, and reports back on GitHub as a comment.

This directory holds everything she needs: the workflow that runs her, the
prompt for each engagement, the saved team (manager + roster) the run attaches,
and the two agent roles that are not packaged with the harness.

## How it runs

`.github/workflows/aida.yml` triggers on:

| Event | Effect |
| --- | --- |
| PR opened / reopened / marked ready / new push (`synchronize`) | One review comment on the PR's current state |
| Issue opened | One triage comment, plus fitting existing labels |
| A comment containing `@aida` on a PR or issue (write-access users) | Another pass in a fresh run |
| `workflow_dispatch` with a `pr` or `issue` input | Manual run, e.g. `gh workflow run aida.yml -f pr=1234` |

Drafts are skipped until marked ready; fork PRs are excluded on every path
(see the security model); superseded runs on the same PR are cancelled, so a
push supersedes the older head's review.

Each run is a fresh GitHub-hosted runner that installs the pinned
[local-operator](https://pypi.org/project/local-operator/) harness, installs
the team from this directory (`setup.sh`), and then runs:

    lop exec --hosting radient --model auto --team aida --tools <bound> "<prompt>"

The session attaches the `aida` team: a manager (the run itself) plus
`architect`, `scout`, `reviewer`, `qa-tester`, `designer`, `ux-reviewer`. The
manager delegates what the change warrants and posts one consolidated comment.
Provider: Radient, model `auto`. GitHub identity: the workflow token
(`github-actions[bot]`), signed *Aida* in the comment text. An upgrade path to
a dedicated GitHub App identity is noted below.

## Setup (one-time)

1. Add repository secret `RADIENT_API_KEY` (Settings -> Secrets and variables ->
   Actions). It is the only secret; the GitHub token is the workflow's own.
2. Merge the workflow. It takes effect immediately: `pull_request` runs use
   the PR's merge-ref version of this workflow (a PR that adds or edits it runs
   its edited version on itself), while `issue_comment` and `workflow_dispatch`
   runs use the default branch's version.

## Security model, stated plainly

- The trigger is `pull_request`, never `pull_request_target`: PR code is never
  executed with repository secrets for external contributions. Fork PRs are
  excluded on **every** path — `pull_request` events check the head repo in
  the payload, and comment/dispatch-triggered runs resolve the head repo
  through the API before anything is checked out.
- The workflow token is narrowed to `contents: read` (plus `actions`, `checks`
  and `statuses` read for `gh pr checks`) and `issues` / `pull-requests` write.
  The prompts allow exactly two effects — one comment per engagement, and labels
  that already exist — and forbid the rest; the token itself could also edit
  issue/PR metadata and review state, so treat the agent's restraint as
  prompt-enforced, not token-enforced. It cannot write repository contents or
  workflows.
- The agent's tool surface is bounded with `--tools`; under a non-TTY run that
  declaration is also the approval for exactly those tools, and delegated
  children inherit both the bound and the approval.
- The prompts instruct the agent to treat everything under review as untrusted
  input and to report prompt-injection attempts as findings. That is a
  mitigation, not a guarantee: treat what Aida writes as advice, not authority.
- Nothing here adds a human reviewer or notifies anyone; the bot never writes
  an `@handle` for a person.

## Maintenance

- **Harness pin**: the workflow pins `local-operator==<version>` from PyPI.
  Bump it deliberately by editing the pin in `.github/workflows/aida.yml`.
- **Team and roles**: `teams/aida/` and `agents/*/` are committed copies —
  `qa-tester` and `ux-reviewer` are not shipped in the harness's packaged
  starters, so the CI runner installs them from here. Keep them in sync with
  the operator's registry roles when those change.
- **Prompts**: `prompts/pr-review.md` and `prompts/issue-triage.md` are the
  engagement briefs; edits change behaviour on the next run.
- **Identity upgrade (future)**: to have comments appear as `aida[bot]`, create
  a GitHub App named Aida with `contents: read`, `issues: write`,
  `pull-requests: write`, install it on this repository, store its app id and
  private key as secrets, and exchange them for an installation token (for
  example with `actions/create-github-app-token`). The workflow is structured
  so that is a three-line change.

## Prior art

- [anthropics/claude-code-action](https://github.com/anthropics/claude-code-action) — the canonical "agent CLI on GitHub events" pattern (mention-driven and auto-review workflows; the write-access gate).
- [google-github-actions/run-gemini-cli](https://github.com/google-github-actions/run-gemini-cli) — the same shape for Gemini.
- [openai/codex-action](https://github.com/openai/codex-action) — Codex CLI in Actions.
- [qodo-ai/pr-agent](https://github.com/qodo-ai/pr-agent) (Qodo Merge) — the self-hosted PR-agent lineage: workflow-owned, command-driven, bring-your-own LLM.
- CodeRabbit / Greptile — hosted counterparts for contrast: no repository-owned workflow, no team, vendor-side model choice.

Aida differs by running a full agent harness with a saved team (manager +
review, QA and design roles) rather than a single-shot reviewer process — the
same architecture the operator's always-on assistant will use.

## Files

    .github/workflows/aida.yml   the runner wiring
    .github/aida/setup.sh        installs the team + roles into the runner's config
    .github/aida/prompts/        engagement briefs
    .github/aida/teams/aida/     the saved team (roster + briefs)
    .github/aida/agents/         qa-tester, ux-reviewer (not packaged starters)

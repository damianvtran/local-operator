# Sir Knight Lop the Second

Sir Knight Lop the Second is the always-on review and triage agent for this
repository. It reviews pull requests, triages issues, and reports back on GitHub
as a comment. Internally it is the **helpdesk** bot — the name of the workflow,
the team, and this directory; "Sir Knight Lop the Second" is the visible
identity.

This directory holds everything it needs: the workflow that runs it, the prompt
for each engagement, the saved team (manager + roster) the run attaches, and the
two agent roles that are not packaged with the harness.

## How it runs

`.github/workflows/helpdesk.yml` triggers on:

| Event | Effect |
| --- | --- |
| PR opened (non-draft), or a draft marked ready for review | One review comment on the PR's current state |
| Issue opened | One triage comment, plus fitting existing labels |
| A comment containing `@sir-knight-lop-the-second` on a PR or issue (write-access users) | Another pass in a fresh run |
| `workflow_dispatch` with a `pr` or `issue` input | Manual run, e.g. `gh workflow run helpdesk.yml -f pr=1234`; available once this workflow is on the default branch |

The bot was introduced as **Aida**; `@aida` and `@Aida` remain accepted as
legacy aliases, so threads from before the rename keep working.

Drafts are reviewed when marked ready for review — or at any time through a
mention. Pushes do **not** re-trigger a review by design: the PR's own CI
carries the per-push signal, and each engagement spends provider tokens. After
a push, a write-access user comments a mention to review the new head. That is
also the recovery path for a conflicted PR — the guard skips it (there is no
merge ref to check out), so resolve the conflict and then mention the bot.
A reopened PR gets no automatic pass either: summon a review with a mention,
the same as for a pushed head. Fork PRs are excluded on every path (see the
security model); a comment or dispatch on a closed PR or issue is skipped —
skipped runs stay silent on the thread (the run log carries the reason). A
change that needs the bot's review must therefore come from a branch of this
repository: for a fork PR, the mention and the dispatch paths are both
skipped by the same head-repo guard. Issue triage runs for any non-bot issue
author by design — that is the feature — and each run spends provider tokens.

Each run is a fresh GitHub-hosted runner that installs the pinned
[local-operator](https://pypi.org/project/local-operator/) harness, installs
the team from this directory (`setup.sh`), and then runs:

    lop exec --hosting radient --model auto --team helpdesk --tools <bound> "<prompt>"

The session attaches the `helpdesk` team: a manager (the run itself) plus
`architect`, `scout`, `reviewer`, `qa-tester`, `designer`, `ux-reviewer`. The
manager delegates what the change warrants and posts one consolidated comment.
Provider: Radient, model `auto` (see "Model and effort"). GitHub identity:
with the app secrets configured, comments are authored by
`sir-knight-lop-the-second[bot]` — the "Sir Knight Lop the Second" GitHub App;
without them, or when minting its token fails, the run falls back to the
workflow token (`github-actions[bot]`) and the run log carries a warning. The
visible name is carried by each comment's heading
(`### Sir Knight Lop the Second — review` /
`### Sir Knight Lop the Second — triage`).

### Model and effort

`--hosting radient --model auto` is a router: it has no published
reasoning-effort ladder, and the harness rejects an explicit effort against it
— `--effort max` and even `--effort auto` fail at startup with `auto has no
reasoning-effort levels; drop --effort` (verified). The bot therefore keeps
the two flags as `--hosting radient --model auto` with no `--effort` flag; if
a future harness exposes rungs for `auto`, the flag can be added here.

## Setup (one-time)

1. Add repository secret `RADIENT_API_KEY` (Settings -> Secrets and variables ->
   Actions). It is required for the bot to run; until it is set, runs are
   skipped with a warning and a run-summary note (no comment, no red check).
2. Add repository secrets `BOT_APP_ID` and `BOT_APP_PRIVATE_KEY` — the app's
   ID and private key of the "Sir Knight Lop the Second" GitHub App (the
   workflow feeds the ID to the token action's `client-id` input, which takes
   either the app ID or the client ID) — and install the app on this
   repository. On the operator's two repositories
   (`damianvtran/local-operator` and `damianvtran/local-operator-ui`) the
   secrets and the app installation are both already in place. With them,
   comments are authored by the app; without them the bot still works, posting
   as `github-actions[bot]`. The app needs `contents: read`, `issues: write`,
   `pull-requests: write`.
3. Merge the workflow. It takes effect immediately: `pull_request` runs use
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
  It is the run's fallback identity and the token the run-failure notice
  always uses. The prompts allow exactly two effects — one comment per
  engagement, and labels that already exist — and forbid the rest; the tokens
  could also edit issue/PR metadata and review state, so treat the agent's
  restraint as prompt-enforced, not token-enforced. Neither token can write
  repository contents or workflows.
- Comments are posted with a short-lived GitHub App installation token when the
  app secrets are set, falling back to the workflow token otherwise; the app
  holds only `contents: read`, `issues: write`, `pull-requests: write`. Under
  the app identity the CI-state reads (`gh pr checks`) work because these
  repositories are public — the app holds no `checks`/`statuses`/`actions`
  read, so a private mirror would need those granted. Unlike the workflow
  token, app-token events CAN start workflow runs — so on every trigger path,
  Bot actors (and bot-authored content) never start runs: the clauses test the
  event actor (`github.event.sender`), not only the content's author. Keep
  that invariant when editing the trigger conditions.
- The agent's tool surface is bounded with `--tools`; under a non-TTY run that
  declaration is also the approval for exactly those tools, and delegated
  children inherit both the bound and the approval. `browser`, `ask` and
  `console` are absent by design: a CI runner has no browser surface, no
  terminal host, and nobody to answer a question — the role prompts carry the
  headless fallbacks for exactly that case.
- The child-process environment is allowlisted: `setup.sh` writes the runner's
  `config.yml` with `shell_environment: mode: allowlist`, so a model-authored
  `bash`/`eval` command starts from the harness's safe set plus an explicit
  grant — the provider key (`RADIENT_API_KEY`) is not inherited by design, and
  no environment variable carries it into a prompt-injected command. `GH_TOKEN`
  is granted because posting the comment needs it. The residual, stated
  honestly: the child still runs as the same user on the same host, and an
  in-process filter cannot fully contain same-user host access — another reason
  the token's scopes are kept minimal and the key rotatable.
- The prompts instruct the agent to treat everything under review as untrusted
  input and to report prompt-injection attempts as findings. That is a
  mitigation, not a guarantee: treat what the bot writes as advice, not
  authority.
- Failure semantics: only an unset provider key is a skip — the run warns,
  writes a run-summary note and stops with no comment and no red check (setup
  item 1). Real failures on a configured run — auth, timeouts, crashes — are
  still red checks and post the run-failure notice on the thread, always with
  the workflow token, so a broken app token cannot silence the notice.
- Nothing here adds a human reviewer or notifies anyone; the bot never writes
  an `@handle` for a person.

## Maintenance

- **Harness pin**: the workflow pins `local-operator==<version>` from PyPI.
  Bump it deliberately by editing the pin in `.github/workflows/helpdesk.yml`.
- **Team and roles**: `teams/helpdesk/` and `agents/*/` are committed copies —
  `qa-tester` and `ux-reviewer` are not shipped in the harness's packaged
  starters, so the CI runner installs them from here. Keep them in sync with
  the operator's registry roles when those change.
- **Prompts**: `prompts/pr-review.md` and `prompts/issue-triage.md` are the
  engagement briefs; edits change behaviour on the next run.
- **Identity**: comments are posted through the "Sir Knight Lop the Second"
  GitHub App (slug `sir-knight-lop-the-second`): the workflow mints a
  short-lived installation token with `actions/create-github-app-token` and
  prefers it for `gh` when `BOT_APP_ID` / `BOT_APP_PRIVATE_KEY` are set. The
  app is installed on both local-operator repositories; if it is ever replaced,
  install the new app, set the two secrets, and the fallback to
  `github-actions[bot]` covers everything else.

## Prior art

- [anthropics/claude-code-action](https://github.com/anthropics/claude-code-action) — the canonical "agent CLI on GitHub events" pattern (mention-driven and auto-review workflows; the write-access gate).
- [google-github-actions/run-gemini-cli](https://github.com/google-github-actions/run-gemini-cli) — the same shape for Gemini.
- [openai/codex-action](https://github.com/openai/codex-action) — Codex CLI in Actions.
- [The-PR-Agent/pr-agent](https://github.com/The-PR-Agent/pr-agent) (Qodo Merge) — the self-hosted PR-agent lineage: workflow-owned, command-driven, bring-your-own LLM.
- CodeRabbit / Greptile — hosted counterparts for contrast: no repository-owned workflow, no team, vendor-side model choice.

Sir Knight Lop the Second differs by running a full agent harness with a saved
team (manager + review, QA and design roles) rather than a single-shot reviewer
process — the same architecture the operator's always-on assistant will use.

## Files

    .github/workflows/helpdesk.yml   the runner wiring
    .github/helpdesk/setup.sh        installs the team + roles into the runner's config
    .github/helpdesk/prompts/        engagement briefs
    .github/helpdesk/teams/helpdesk/ the saved team (roster + briefs)
    .github/helpdesk/agents/         qa-tester, ux-reviewer (not packaged starters)

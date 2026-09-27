# Sir Knight Lop the Second

Sir Knight Lop the Second is the always-on review and triage agent for this
repository. It reviews pull requests, triages issues, and reports back on GitHub
as a comment. Internally it is the **helpdesk** bot — the name of the workflow,
the team, and this directory; "Sir Knight Lop the Second" is the visible
identity.

This directory holds everything it needs: the workflow that runs it, the
reconciler that decides when a pull request is due an engagement, the prompt
for each engagement, the saved team (manager + roster) the run attaches, and the
two agent roles that are not packaged with the harness.

## How it runs

`.github/workflows/helpdesk.yml` — how it engages:

| Engagement | Effect |
| --- | --- |
| Sweep — `schedule`, every 15 minutes | Reconciles every open PR and dispatches an engagement for each one that is due: `review`, or `verdict` once the contributor's rounds have settled |
| PR becomes due (`review`, sweep-dispatched) | One concise guidelines-compliance review on the PR's current head |
| PR's rounds settle (`verdict`, sweep-dispatched) | The terminal ✅/❌ compliance verdict — one per head |
| Issue opened | One triage comment — immediately; issues do not wait |
| A comment containing `@sir-knight-lop-the-second` on a PR or issue (write-access users) | A review pass (PR) or a triage pass (issue) in a fresh run, regardless of the sweep's gates |
| `workflow_dispatch` with a `pr` (and optional `mode=verdict`) or `issue` input | Operator-requested run, e.g. `gh workflow run helpdesk.yml -f pr=1234`; bypasses the already-engaged gate |

The bot was introduced as **Aida**; `@aida` and `@Aida` remain accepted as
legacy aliases, so threads from before the rename keep working.

The bot DEFERS: a pull request is engaged only when it is due, and one
engagement is one run.

- **Review** — a PR that is non-draft, same-repo, and has waited
  `REVIEW_DELAY_MINUTES` (30) since it became reviewable (its creation, or the
  latest ready-for-review transition when it was a draft) — with **no
  review-family activity and no bot review** — gets one concise
  guidelines-compliance review, once. Pushes, reopens and draft flips do not
  re-trigger by design: the PR's own CI carries the per-push signal, and every
  engagement spends provider tokens. After a push, a write-access user comments
  a mention for a pass scoped to the delta.
- **Verdict** — when the contributor's own rounds exist (see the marker
  contract below), have settled for `VERDICT_SETTLE_MINUTES` (15), and the
  latest family comment is a round (not a remediation reply) carrying a
  terminal-ish signal — and the head has not moved since that round — the
  sweep dispatches the terminal **✅/❌ compliance verdict**. Per head: a
  moved head can earn a new verdict once its own rounds go terminal, and a
  thread that is not terminal gets no comment at all (the run log carries the
  reasoning).

The sweep is idempotent and re-decides every 15 minutes; each due engagement is
dispatched as its own run of this workflow
(`-f pr=… -f mode=review|verdict -f source=sweep`), so every engagement keeps
one execution path with its own run, log and timeout. A sweep-dispatched run
re-checks that the engagement is still due (`reconcile.sh check`) before
checking anything out — between the decision and the run, a bot review or a
round can appear and make the run moot. The sweep also waits for the thread to
move: a verdict whose rounds predate the current head (a push landed after the
latest round) is deferred to the new head's rounds, and an engagement whose
exact family state a recent attempt already covered is not re-dispatched while
that attempt is pending or was successful (a failed attempt retries). Runs are
named to make this visible: `Sir Knight Lop the Second — <mode> PR #<n>`, with
` (sweep)` on sweep dispatches. The `source` input is what marks a run as
sweep-dispatched; a manual dispatch passes an empty `source` and bypasses that
gate.

Drafts are skipped on every path — mark a draft ready (which starts the
30-minute wait) or mention the bot once it is ready. A conflicted PR
(`mergeable: false`) is skipped as well — there is no merge ref to check out;
resolve the conflict first; then the sweep re-engages it, or a mention gets a
pass right away. Fork PRs are excluded on
every path (see the security model). A comment or dispatch on a closed PR or
issue is skipped — skipped runs stay silent on the thread (the run log carries
the reason). A change that needs the bot's review must therefore come from a
branch of this repository. Issue triage runs for any non-bot issue author,
immediately by design — the wait does not apply to issues — and each run spends
provider tokens.

The thresholds are the documented contract and live at the top of
`.github/helpdesk/reconcile.sh` (`REVIEW_DELAY_MINUTES`,
`VERDICT_SETTLE_MINUTES`, `WINDOW_DAYS`); the 15-minute sweep cadence is the
wait's granularity (a PR fires after the delay plus at most one sweep interval
of slack). The reconciler is pure read-only and its decisions are tested in
`.github/helpdesk/tests/reconcile.test.sh` (stubbed `gh`, no network; run it
with `bash .github/helpdesk/tests/reconcile.test.sh`).

The sweep enumerates the 200 newest open PRs (`gh pr list --limit 200`) and
skips anything not updated within the window; the sweep itself is a cheap
read-only `gh` pass, and the only thing that spends provider tokens is a due
engagement — one workflow run each.

Each run is a fresh GitHub-hosted runner that installs the pinned
[local-operator](https://pypi.org/project/local-operator/) harness, installs
the team from this directory (`setup.sh`), and then runs:

    lop exec --hosting radient --model auto --team helpdesk --tools <bound> "<prompt>"

The engagement's helpdesk surface — this directory's `prompts/`, `teams/`,
`agents/` and `setup.sh` — always comes from the **default branch**: the
review job replaces the PR checkout's copy wholesale with `main`'s before
anything reads it. A `refs/pull/<n>/merge` checkout can be stale — a PR
branched before a helpdesk change carries the old prompts and none of the new
ones (a verdict engagement once crashed on a missing `prompts/pr-verdict.md`)
— and without the overlay a PR could steer its own review by editing
`.github/helpdesk/**`. Behaviour is therefore defined by `main`: the same copy
the sweep decides with, and the same copy the workflow file itself runs from.
The PR checkout still provides the code under review; the model's diffs come
from the API.

The session attaches the `helpdesk` team: a manager (the run itself) plus
`architect`, `scout`, `reviewer`, `qa-tester`, `designer`, `ux-reviewer`. The
manager delegates what the change warrants and posts one consolidated comment.
Provider: Radient, model `auto` (see "Model and effort"). GitHub identity:
with the app secrets configured, comments are authored by
`sir-knight-lop-the-second[bot]` — the "Sir Knight Lop the Second" GitHub App;
without them, or when minting its token fails, the run falls back to the
workflow token (`github-actions[bot]`) and the run log carries a warning. The
visible name is carried by each comment's heading
(`### Sir Knight Lop the Second — review` / `— verdict` / `— triage`).

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
3. Merge the workflow. Sweeps (`schedule`) always run the default branch's
   copy; a `workflow_dispatch` run uses the ref it was dispatched with
   (default: the default branch). There is no `pull_request` trigger, so a PR
   that edits this workflow does not run its edited version on itself — the
   first sweep after merge is what carries the change live.

## Marker contract

The reconciler greps these exact headings (keep them stable; the list lives in
one place in `.github/helpdesk/reconcile.sh`):

- **Bot review marker**: a comment authored by `sir-knight-lop-the-second[bot]`
  or `github-actions[bot]` whose body has the heading line
  `### Sir Knight Lop the Second — review` (legacy: `### Aida — review`). A
  `— review run failed` notice does not count.
- **Bot verdict marker**: same author rule, heading line
  `### Sir Knight Lop the Second — verdict`; the comment also carries a
  `**Head:** <sha>` line, and dedup is per head.
- **Review-family activity** (the contributor's own rounds — any author): a
  comment with a heading line `### Agent review`, `### QA`, `### Design
  review` or `### UX review` (each with ` report` / ` remediation` variants),
  or `### Gates + local evidence`.

## Security model, stated plainly

- The workflow has no `pull_request` (or `pull_request_target`) trigger at
  all: nothing runs on PR events, so PR code never sees repository secrets
  through them. Fork PRs are excluded on **every** path — the sweep skips them
  (head owner/name mismatch) and comment/dispatch-triggered runs resolve the
  head repo, the state, the mergeability and the draft flag through the API
  before anything is checked out.
- The workflow token is narrowed to `contents: read` (plus `actions`, `checks`
  and `statuses` read for `gh pr checks`) and `issues` / `pull-requests` write.
  It is the run's fallback identity and the token the run-failure notice
  always uses. The prompts allow exactly two effects — one comment per
  engagement, and labels that already exist — and forbid the rest; the tokens
  could also edit issue/PR metadata and review state, so treat the agent's
  restraint as prompt-enforced, not token-enforced. Neither token can write
  repository contents or workflows.
- The sweep job (`reconcile`) is the only place the token gets `actions: write`,
  and it is safe there **by construction**: it runs no model code, mints no app
  token, checks out only the default branch, and its one write is
  `gh workflow run helpdesk.yml` with fixed inputs. Its job-level
  `permissions:` block lists exactly what it needs (reads, plus
  `actions: write`), and it replaces the workflow block job-wide — so the
  model's engagement job never holds that scope.
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
- **Thresholds and decisions**: `REVIEW_DELAY_MINUTES`,
  `VERDICT_SETTLE_MINUTES` and `WINDOW_DAYS` live at the top of
  `.github/helpdesk/reconcile.sh`, which is also where the sweep's skips and
  the comment classification live — one place, env-overridable, with its tests
  in `.github/helpdesk/tests/reconcile.test.sh`.
- **Marker contract**: the headings in the section above are grepped exactly;
  changing one means changing `.github/helpdesk/reconcile.sh` (one place) and
  this README together.
- **How to force an engagement**: a mention runs a review pass regardless of
  the sweep's gates (post a comment containing the bot's mention handle, or
  the legacy alias); `gh workflow run helpdesk.yml -f pr=<n> -f mode=verdict`
  runs a verdict pass (the run still posts nothing if the thread is not
  terminal); `gh workflow run helpdesk.yml -f issue=<n>` triages an issue.
- **Run names are load-bearing**: the sweep's attempt dedupe recognizes an
  engagement by its run title (`Sir Knight Lop the Second — <mode> PR #<n>`,
  ` (sweep)` for sweeps). Changing the workflow's `run-name:` changes what
  `reconcile.sh` can see — update both together.
- **Team and roles**: `teams/helpdesk/` and `agents/*/` are committed copies —
  `qa-tester` and `ux-reviewer` are not shipped in the harness's packaged
  starters, so the CI runner installs them from here. Keep them in sync with
  the operator's registry roles when those change.
- **Prompts**: `prompts/pr-review.md` (the concise guidelines-compliance
  review), `prompts/pr-verdict.md` (the terminal verdict) and
  `prompts/issue-triage.md` are the engagement briefs; edits change behaviour
  on the next run.
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

    .github/workflows/helpdesk.yml     the runner wiring (sweep, review, triage)
    .github/helpdesk/reconcile.sh      the sweep's decisions + the already-engaged gate
    .github/helpdesk/tests/reconcile.test.sh  its tests (stubbed gh; no network)
    .github/helpdesk/setup.sh          installs the team + roles into the runner's config
    .github/helpdesk/prompts/          engagement briefs (review, verdict, triage)
    .github/helpdesk/teams/helpdesk/   the saved team (roster + briefs)
    .github/helpdesk/agents/           qa-tester, ux-reviewer (not packaged starters)

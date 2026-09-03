#!/usr/bin/env bash
set -euo pipefail

# Fail-closed validation of the protected environment a store release runs in.
#
# This script is the primary security boundary for store releases: the
# environment's required-reviewer gate was removed by operator decision on
# 2026-09-03, so nothing else pauses a release for a human. It runs in two
# modes, and BOTH are required to prove the full contract. Each mode alone is
# insufficient, which is why the workflows run them as two separate jobs.
#
#   verify-release-environment.sh <environment> NAME VALUE ...
#       Runs INSIDE the environment. Checks the deployment branch policy
#       through the API and asserts every named variable resolved to a value.
#
#   verify-release-environment.sh --assert-unscoped-empty NAME VALUE ...
#       Runs OUTSIDE the environment, in a job with no `environment:` key.
#       Asserts every named variable resolved to EMPTY.
#
# Why the second mode exists instead of an API call. The natural check is
# `GET /repos/{o}/{r}/environments/{env}/variables`, and this script used to
# make it. That endpoint is NOT reachable from a workflow GITHUB_TOKEN: it
# answers 403 with `x-accepted-github-permissions: environments=read`, and
# `environments` is not in the fixed set of permissions a workflow may grant
# (actions, artifact-metadata, attestations, checks, code-quality, contents,
# deployments, discussions, id-token, issues, packages, pages, pull-requests,
# security-events, statuses, vulnerability-alerts). Measured, not assumed: a
# probe job running with `permissions: write-all` still gets 403 on that
# endpoint, so no scope list can fix it — that job's own granted-permission
# list contains no `Environments` entry at all. The repository-scoped variables
# endpoint is equally unreachable (403, `actions_variables=read`).
#
# This is what broke run 33793517588, at the THIRD API call. The first two
# (`environments/{env}` and its `deployment-branch-policies`) return 200 under
# `actions: read` and are still made below; only the variables call 403s. The
# ~0.6s step duration is three fast calls, not one — do not read that failure as
# the environment endpoint being unreachable, because it is not.
#
# The two-mode differential proves the same invariant from what the runtime can
# actually observe, which is also the thing that matters: `vars.X` resolving
# non-empty inside the environment and empty outside it means X is defined at
# environment scope and NOT at repository or organization scope. A
# repository-scoped variable is visible to every job, so it would be non-empty
# in the unscoped job and fail mode B. This checks the value the release will
# really use rather than what a listing endpoint reports about it.
#
# Two limits of that design, both known and accepted rather than overlooked:
#
#   * The two modes read configuration at different times, so a repository-scoped
#     variable added AFTER the preflight job passes and before the deploying job
#     reads `vars.*` is not caught. Exploiting the window requires an actor who
#     already holds repository-admin write; the WIF attribute condition (pinned
#     repo ID on refs/heads/main) and the merge-base ancestry check still stand
#     between them and a store upload. Closing it would need an atomic read the
#     API does not offer to a workflow token.
#   * The variable NAMES are listed per workflow, in both the preflight and the
#     verify step. A name added to one step but not the other is simply not
#     checked, silently. Keep the two lists in a workflow identical; the tests
#     assert that they match, so drift fails the suite rather than a release.

MODE=environment
if [[ ${1:-} == --assert-unscoped-empty ]]; then
  MODE=unscoped
  shift
fi

ENVIRONMENT_NAME=
if [[ $MODE == environment ]]; then
  ENVIRONMENT_NAME=${1:-}
  shift || true
fi

fail() {
  printf 'release environment validation failed: %s\n' "$*" >&2
  exit 1
}

[[ $# -gt 0 && $(($# % 2)) -eq 0 ]] \
  || fail "expected NAME VALUE pairs"

# Mode B needs no API access, so it deliberately does not require a token: a
# job outside the environment should not be handed credentials it cannot use.
if [[ $MODE == unscoped ]]; then
  while [[ $# -gt 0 ]]; do
    name=$1
    value=$2
    shift 2
    # Non-empty here means the variable is readable without entering the
    # environment, i.e. it is defined at repository or organization scope. That
    # is the misconfiguration this mode exists to reject: such a variable can be
    # read and set by workflows that never pass the environment's branch policy.
    [[ -z "$value" ]] \
      || fail "$name is readable outside the environment (repository/organization scope is forbidden; it must be defined at environment scope only)"
  done
  printf 'validated that no release variable is defined at repository or organization scope\n'
  exit 0
fi

for command in curl jq; do
  command -v "$command" >/dev/null || fail "$command is required"
done
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
[[ -n "$ENVIRONMENT_NAME" ]] || fail "environment name is required"

api_root=${GITHUB_API_URL:-https://api.github.com}
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

get_github_json() {
  local path=$1
  local output=$2
  local status
  # The status is captured and reported explicitly rather than leaning on
  # curl's exit code. `--fail-with-body` exits 22 for every 4xx with only
  # "curl: (22) The requested URL returned error: 403" on stderr, which is what
  # made run 33793517588 expensive to diagnose: it named neither the endpoint
  # nor the permission GitHub was asking for. The 403 body and the
  # x-accepted-github-permissions header are both printed here so the next
  # permission regression is readable from the failed step alone.
  status=$(curl --silent --show-error \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2022-11-28' \
    -D "$tmp_dir/headers.txt" \
    -o "$output" \
    -w '%{http_code}' \
    "$api_root/repos/$GITHUB_REPOSITORY/$path") \
    || fail "request to $path could not be completed"
  if [[ "$status" != 200 ]]; then
    printf 'GitHub returned HTTP %s for %s\n' "$status" "$path" >&2
    printf 'response body: %s\n' "$(head -c 2000 "$output")" >&2
    grep -i '^x-accepted-github-permissions:' "$tmp_dir/headers.txt" >&2 || true
    # 403 means the token lacks the permission; 404 is also how GitHub hides a
    # resource from an under-permissioned token, but it is equally the answer
    # for a typo'd environment name — the case this script exists to catch — so
    # the message must not assert a cause it cannot distinguish.
    fail "GitHub returned HTTP $status for $path (check that the environment exists and that the workflow token carries the permission this endpoint requires)"
  fi
  jq -e . "$output" >/dev/null || fail "GitHub returned invalid JSON for $path"
}

# Merely naming an environment in workflow YAML creates an unprotected one on a
# typo, so the exact main-only deployment policy has to be observed through
# GitHub's API before OIDC authentication can run. Both endpoints below are
# reachable under `actions: read` (verified: 200, x-accepted-github-permissions
# actions=read) — unlike the variables endpoint, which mode B replaces.
#
# A "required reviewers" protection rule used to be enforced here as well. It
# was removed by operator decision on 2026-09-03 so a store release completes
# end to end without a human approval pause. Do not re-add it thinking it was
# lost: the retained boundary is the GCP Workload Identity Federation
# attribute condition, which mints a token only for the pinned numeric
# repository ID on refs/heads/main, combined with this script's exact-main
# deployment-branch-policy check, the two-mode variable-scope check, and each
# workflow's own merge-base ancestry check. The launch decision now rides on
# the reviewed-merge-to-main process that the WIF ref pin enforces.
get_github_json "environments/$ENVIRONMENT_NAME" "$tmp_dir/environment.json"
jq -e '
  .deployment_branch_policy.protected_branches == false
  and .deployment_branch_policy.custom_branch_policies == true
' "$tmp_dir/environment.json" >/dev/null \
  || fail "$ENVIRONMENT_NAME must use a custom deployment branch policy"

get_github_json \
  "environments/$ENVIRONMENT_NAME/deployment-branch-policies?per_page=100" \
  "$tmp_dir/branches.json"
jq -e '
  .total_count == 1
  and .branch_policies[0].name == "main"
  and .branch_policies[0].type == "branch"
' "$tmp_dir/branches.json" >/dev/null \
  || fail "$ENVIRONMENT_NAME must allow exactly the main branch"

# Half of the variable-scope contract: every name must resolve to a value here,
# inside the environment. The other half — that it resolves to NOTHING outside
# the environment — is asserted by the --assert-unscoped-empty job, which the
# release workflows run first and depend on.
while [[ $# -gt 0 ]]; do
  name=$1
  value=$2
  shift 2
  # "defined and empty" and "not defined at all" are indistinguishable here:
  # `vars.X` resolves to the empty string for both. The message says so rather
  # than asserting a cause it cannot tell apart; either way it fails closed.
  [[ -n "$value" ]] \
    || fail "$name is empty or not defined on $ENVIRONMENT_NAME (it must be defined at environment scope with a non-empty value)"
done

printf 'validated protected environment %s (main-only, environment-scoped variables)\n' \
  "$ENVIRONMENT_NAME"

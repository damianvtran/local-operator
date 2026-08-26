#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_NAME=${1:-}
shift || true

fail() {
  printf 'release environment validation failed: %s\n' "$*" >&2
  exit 1
}

for command in curl jq; do
  command -v "$command" >/dev/null || fail "$command is required"
done
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
[[ -n "$ENVIRONMENT_NAME" ]] || fail "environment name is required"
[[ $# -gt 0 && $(($# % 2)) -eq 0 ]] \
  || fail "expected NAME VALUE pairs after the environment name"

api_root=${GITHUB_API_URL:-https://api.github.com}
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

get_github_json() {
  local path=$1
  local output=$2
  curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2022-11-28' \
    -o "$output" \
    "$api_root/repos/$GITHUB_REPOSITORY/$path"
  jq -e . "$output" >/dev/null || fail "GitHub returned invalid JSON for $path"
}

get_github_json "environments/$ENVIRONMENT_NAME" "$tmp_dir/environment.json"

# Merely naming an environment in workflow YAML creates an unprotected one on a
# typo. Required reviewers and an exact main-only deployment policy therefore
# have to be observed through GitHub's API before OIDC authentication can run.
jq -e '
  (.protection_rules // [])
  | any(.type == "required_reviewers" and ((.reviewers // []) | length > 0))
' "$tmp_dir/environment.json" >/dev/null \
  || fail "$ENVIRONMENT_NAME must have at least one required reviewer"
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

get_github_json "environments/$ENVIRONMENT_NAME/variables?per_page=100" "$tmp_dir/variables.json"
while [[ $# -gt 0 ]]; do
  name=$1
  expected_value=$2
  shift 2
  [[ -n "$expected_value" ]] || fail "$name is empty"
  actual_value=$(jq -er --arg name "$name" \
    '.variables[] | select(.name == $name) | .value' "$tmp_dir/variables.json") \
    || fail "$name must be defined on $ENVIRONMENT_NAME (repository/organization scope is forbidden)"
  [[ "$actual_value" == "$expected_value" ]] \
    || fail "$name does not match the environment-scoped value on $ENVIRONMENT_NAME"
done

printf 'validated protected environment %s (required reviewer, main-only, environment-scoped variables)\n' \
  "$ENVIRONMENT_NAME"

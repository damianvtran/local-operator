#!/usr/bin/env bash
# Post — or UPDATE — the helpdesk's "run failed" notice on a PR or issue.
#
#   report_failure.sh <number> <heading> <message> <run-url> [reason]
#
# WHY: the workflow used to `gh pr comment` a fresh notice on every failed run.
# When the pinned harness vanished from PyPI every run failed at install, and
# with the sweep re-dispatching after its 30-minute backoff PRs collected dozens
# of identical notices (46 on #3109, 26 on #3237). One notice per thread is
# enough: if a comment that starts with <heading> already exists, PATCH it with
# the latest run link, timestamp and a running failure count instead of posting.
#
# CONTRACT
#   * Matching is on the exact first line (<heading>), so the PR's review and
#     verdict notices stay separate, and reconcile.sh's "a `— review run
#     failed` notice does NOT count" rule is unaffected (headings unchanged).
#   * PATCH can only edit a comment the current token authored; when it fails
#     (e.g. the earlier notice was posted by the app, this one by
#     github-actions[bot]) the script falls back to posting a new notice —
#     never a silent drop.
#   * [reason] = `install` marks an install failure distinctly: that class is
#     a broken pin, not a model or provider fault, and needs a maintainer to
#     bump `local-operator==<version>` in .github/workflows/helpdesk.yml.
#   * Needs GH_REPO (owner/name) and an authenticated `gh`. Best-effort: the
#     calling step already reports a failure, so this exits 0 on API errors
#     after printing a ::warning::.
set -euo pipefail

number="${1:?usage: report_failure.sh <number> <heading> <message> <run-url> [reason]}"
heading="${2:?heading required}"
message="${3:?message required}"
run_url="${4:?run url required}"
reason="${5:-}"
repo="${GH_REPO:?GH_REPO must be set}"
now="${REPORT_NOW:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

comments="$(mktemp)"
body="$(mktemp)"
trap 'rm -f "$comments" "$body"' EXIT

# --paginate emits one JSON array per page; `jq -s add` joins them.
existing_id=""
previous_count=0
if gh api --paginate "repos/${repo}/issues/${number}/comments" > "$comments" 2>/dev/null; then
  match="$(jq -rs --arg h "$heading" '
    (add // []) | map(select((.body // "") | split("\n")[0] == $h)) | last // empty
    | [(.id | tostring), (((.body // "") | capture("Failures recorded: (?<n>[0-9]+)").n) // "1")]
    | @tsv' "$comments")"
  if [ -n "$match" ]; then
    existing_id="${match%%$'\t'*}"
    previous_count="${match#*$'\t'}"
  fi
else
  echo "::warning::could not list comments on #${number}; posting a new notice"
fi

count=$((previous_count + 1))
{
  printf '%s\n\n%s\n\n' "$heading" "$message"
  if [ "$reason" = "install" ]; then
    # shellcheck disable=SC2016  # backticks are Markdown, not expansion
    printf '**Cause: harness install failed.** The pinned `local-operator` version could not be installed (likely removed from PyPI); a maintainer needs to bump the pin in `.github/workflows/helpdesk.yml`. Retrying will not help until then.\n\n'
  fi
  printf 'Last failure: %s — run log: %s\n' "$now" "$run_url"
  printf 'Failures recorded: %s\n' "$count"
} > "$body"

if [ -n "$existing_id" ]; then
  if gh api -X PATCH "repos/${repo}/issues/comments/${existing_id}" -F "body=@${body}" > /dev/null; then
    echo "updated notice ${existing_id} on #${number} (failures recorded: ${count})"
    exit 0
  fi
  echo "::warning::could not update notice ${existing_id}; posting a new one"
fi
if gh api -X POST "repos/${repo}/issues/${number}/comments" -F "body=@${body}" > /dev/null; then
  echo "posted notice on #${number}"
else
  echo "::warning::could not post the failure notice on #${number}"
fi

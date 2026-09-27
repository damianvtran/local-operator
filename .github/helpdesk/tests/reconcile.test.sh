#!/usr/bin/env bash
# Tests for .github/helpdesk/reconcile.sh.
#
# A stubbed `gh` on PATH serves inline fixtures (built with jq, relative to the
# clock at test start) — no network, no GitHub, self-contained. Run from
# anywhere:
#
#   bash .github/helpdesk/tests/reconcile.test.sh
#
# It exercises the documented detection contract end to end: bot-review dedup
# (both bot logins, and that a `— review run failed` notice does NOT count),
# family-heading detection for every heading the contract names plus the
# remediation subset, verdict gating (settle, last-is-remediation, the
# terminal-ish prefilter, per-head verdict dedup), the sweep-level skips
# (draft, fork, conflict, window, bot author), the age gate including the
# ready_for_review transition, the `check` subcommand, and the step-summary
# table.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
reconcile="$here/../reconcile.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
STUB_BIN="$tmp/bin"
FIX="$tmp/fixtures"
mkdir -p "$STUB_BIN" "$FIX"

# --- the stubbed gh ---------------------------------------------------------

cat > "$STUB_BIN/gh" <<'STUB'
#!/usr/bin/env bash
# Stub `gh` for the reconcile tests: serve fixtures from $GH_STUB_DIR.
# Unknown shapes fail loudly so contract drift in reconcile.sh cannot pass
# silently against a stale stub.
set -euo pipefail
dir="${GH_STUB_DIR:?GH_STUB_DIR must be set}"
jq_filter=""
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --jq)
      jq_filter="$2"
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
cmd="${args[0]:-}"
case "$cmd" in
  pr)
    case "${args[1]:-}" in
      list) cat "$dir/prs.json" ;;
      *)
        echo "stub gh: unsupported pr subcommand: ${args[*]}" >&2
        exit 1
        ;;
    esac
    ;;
  api)
    path="${args[1]:-}"
    case "$path" in
      */issues/*/comments)
        n="${path##*/issues/}"
        n="${n%%/*}"
        file="$dir/comments_$n.json"
        ;;
      */issues/*/timeline)
        n="${path##*/issues/}"
        n="${n%%/*}"
        file="$dir/timeline_$n.json"
        ;;
      */pulls/*)
        n="${path##*/pulls/}"
        file="$dir/pulls_$n.json"
        ;;
      *)
        echo "stub gh: unsupported api path: $path" >&2
        exit 1
        ;;
    esac
    [ -f "$file" ] || {
      echo "stub gh: missing fixture $file" >&2
      exit 1
    }
    if [ -n "$jq_filter" ]; then jq -r "$jq_filter" "$file"; else cat "$file"; fi
    ;;
  *)
    echo "stub gh: unsupported command: ${args[*]}" >&2
    exit 1
    ;;
esac
STUB
chmod +x "$STUB_BIN/gh"

export GH_REPO="example/repo"
export GH_STUB_DIR="$FIX"
export PATH="$STUB_BIN:$PATH"

# --- fixture builders -------------------------------------------------------

now="$(jq -n 'now | floor')"
# iso <seconds-ago> -> an ISO-8601 UTC timestamp, exactly as GitHub emits.
iso() { jq -rn --argjson now "$now" --argjson ago "$1" '($now - $ago) | todateiso8601'; }

reset() { rm -f "$FIX"/*.json; echo '[]' > "$FIX/prs.json"; }

pr_add() {
  jq -n --argjson doc "$(cat "$FIX/prs.json")" --argjson obj "$1" '$doc + [$obj]' \
    > "$FIX/prs.json.tmp"
  mv "$FIX/prs.json.tmp" "$FIX/prs.json"
}

# pr_obj <number> <author-login> <author-type> <draft> <mergeable> \
#        <created_ago_s> <updated_ago_s> [head-owner] [head-name]
pr_obj() {
  jq -n \
    --argjson number "$1" --arg login "$2" --arg type "$3" \
    --argjson draft "$4" --arg mergeable "$5" \
    --arg created "$(iso "$6")" --arg updated "$(iso "$7")" \
    --arg owner "${8:-example}" --arg name "${9:-repo}" \
    '{number: $number, title: ("PR " + ($number | tostring)), isDraft: $draft,
      createdAt: $created, updatedAt: $updated,
      author: {login: $login, type: $type},
      headRepositoryOwner: {login: $owner}, headRepository: {name: $name},
      mergeable: $mergeable}'
}

# comment_obj <login> <body> <created_ago_s>
comment_obj() {
  jq -n --arg login "$1" --arg body "$2" --arg created "$(iso "$3")" \
    '{user: {login: $login}, body: $body, created_at: $created}'
}

# comments <pr> [json-object ...]
comments() {
  local n="$1"
  shift
  if [ $# -eq 0 ]; then
    echo '[]' > "$FIX/comments_$n.json"
    return 0
  fi
  printf '%s\n' "$@" | jq -s '.' > "$FIX/comments_$n.json"
}

# timeline <pr> [ready_ago_s]   (no argument = no ready_for_review events)
timeline() {
  if [ -z "${2:-}" ]; then
    echo '[]' > "$FIX/timeline_$1.json"
  else
    jq -n --arg t "$(iso "$2")" '[{event: "ready_for_review", created_at: $t}]' \
      > "$FIX/timeline_$1.json"
  fi
}

# pulls <pr> <head-sha>
pulls() { jq -n --arg sha "$2" '{head: {sha: $sha}}' > "$FIX/pulls_$1.json"; }

# --- assertions -------------------------------------------------------------

total=0
fails=0
check() { # <desc> <expected> <actual>
  total=$((total + 1))
  if [ "$2" = "$3" ]; then
    printf 'PASS  %s\n' "$1"
  else
    printf 'FAIL  %s\n      expected: %s\n      actual:   %s\n' "$1" "$2" "$3"
    fails=$((fails + 1))
  fi
}
contains() { # <desc> <haystack> <needle>
  total=$((total + 1))
  case "$2" in
    *"$3"*) printf 'PASS  %s\n' "$1" ;;
    *)
      printf 'FAIL  %s\n      wanted substring: %s\n      actual: %s\n' "$1" "$3" "$2"
      fails=$((fails + 1))
      ;;
  esac
}

# Run `sweep` / `check`. A non-zero exit is a dead run, never "no actions
# emitted": report it as a sentinel no expectation can equal, so the affected
# case fails visibly instead of comparing equal to an empty expectation.
sweep_out() {
  local out
  if ! out="$(bash "$reconcile" sweep 2>/dev/null)"; then
    echo "FATAL: reconcile.sh sweep exited non-zero" >&2
    printf '%s' "FATAL_SWEEP_FAILED"
    return 0
  fi
  printf '%s' "$out"
}
check_out() {
  local out
  if ! out="$(bash "$reconcile" check --pr "$1" --mode "$2" 2>/dev/null)"; then
    echo "FATAL: reconcile.sh check exited non-zero" >&2
    printf '%s' "FATAL_CHECK_FAILED"
    return 0
  fi
  printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# The review path: due, too young, ready-age, created-window.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 101 alice User false MERGEABLE 2700 300)"
comments 101
timeline 101
check "sweep: a waited PR with no rounds and no bot review is due a review" \
  "review 101" "$(sweep_out)"

reset
pr_add "$(pr_obj 102 alice User false MERGEABLE 600 300)"
comments 102
check "sweep: younger than the review delay is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 103 alice User false MERGEABLE 2700 300)"
comments 103
timeline 103 1200
check "sweep: created long ago but marked ready 20m ago is not engaged yet" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 104 alice User false MERGEABLE 2700 300)"
comments 104
timeline 104 2400
check "sweep: ready_for_review 40m ago engages (the wait counts from the transition)" \
  "review 104" "$(sweep_out)"

reset
pr_add "$(pr_obj 134 alice User false MERGEABLE 691200 300)"
comments 134
timeline 134 3600
check "sweep: PRs created beyond the window are skipped regardless of ready age" \
  "" "$(sweep_out)"

# ---------------------------------------------------------------------------
# Bot-review dedup.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 105 alice User false MERGEABLE 2700 300)"
comments 105 "$(comment_obj 'sir-knight-lop-the-second[bot]' \
  $'### Sir Knight Lop the Second — review \n\n**Verdict:** ✅ all requirements met' 1800)"
check "sweep: a bot review (app login, trailing space) dedups the review" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 106 alice User false MERGEABLE 2700 300)"
comments 106 "$(comment_obj 'github-actions[bot]' \
  $'### Aida — review\n\nlegacy heading' 1800)"
check "sweep: the legacy Aida heading counts as a bot review" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 107 alice User false MERGEABLE 2700 300)"
comments 107 "$(comment_obj 'sir-knight-lop-the-second[bot]' \
  $'### Sir Knight Lop the Second — review run failed\n\nbot side failed' 1800)"
timeline 107
check "sweep: a '— review run failed' notice is NOT a bot review" "review 107" "$(sweep_out)"

# ---------------------------------------------------------------------------
# Family detection → verdict.
# ---------------------------------------------------------------------------

reset
for spec in \
  "111|Agent review — round 1|no blocker finding; approved for merge" \
  "112|QA report — round 1|all pass on the matrix" \
  "113|Design review — round 2|approved; D-1 fixed" \
  "114|UX review — round 1|LGTM" \
  "115|Gates + local evidence|all pass"; do
  n="${spec%%|*}"
  rest="${spec#*|}"
  heading="${rest%%|*}"
  text="${rest#*|}"
  pr_add "$(pr_obj "$n" alice User false MERGEABLE 7200 300)"
  comments "$n" "$(comment_obj alice $'### '"$heading"$'\n\n'"$text" 1200)"
  pulls "$n" "1111111111111111111111111111111111111111"
done
check "sweep: every family heading the contract names yields a settled verdict" \
  "verdict 111
verdict 112
verdict 113
verdict 114
verdict 115" "$(sweep_out)"

reset
pr_add "$(pr_obj 120 alice User false MERGEABLE 7200 300)"
comments 120 \
  "$(comment_obj alice $'### Agent review — round 1\n\nblocker: x' 2400)" \
  "$(comment_obj bob $'### Agent review remediation — round 1\n\nfixed in abc123, no blockers left' 1200)"
pulls 120 "1111111111111111111111111111111111111111"
check "sweep: a thread whose latest family comment is a remediation is not terminal" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 121 alice User false MERGEABLE 7200 300)"
comments 121 "$(comment_obj alice $'### Agent review — round 2\n\nno blocker, approved' 300)"
pulls 121 "1111111111111111111111111111111111111111"
check "sweep: a still-settling thread is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 122 alice User false MERGEABLE 7200 300)"
comments 122 "$(comment_obj alice $'### Agent review — round 2\n\nhere is a summary of what I looked at' 1200)"
pulls 122 "1111111111111111111111111111111111111111"
check "sweep: a round with no terminal-ish signal is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 123 alice User false MERGEABLE 7200 300)"
comments 123 \
  "$(comment_obj alice $'### Agent review — round 3\n\nno blocker; rounds clean' 1200)" \
  "$(comment_obj 'sir-knight-lop-the-second[bot]' \
    $'### Sir Knight Lop the Second — verdict\n**Head:** 1111111111111111111111111111111111111111\n\n**Requirements:** ✅ all met' 600)"
pulls 123 "1111111111111111111111111111111111111111"
check "sweep: a verdict for the current head dedups" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 124 alice User false MERGEABLE 7200 300)"
comments 124 \
  "$(comment_obj alice $'### Agent review — round 3\n\nno blocker; rounds clean' 1200)" \
  "$(comment_obj 'sir-knight-lop-the-second[bot]' \
    $'### Sir Knight Lop the Second — verdict\n**Head:** 2222222222222222222222222222222222222222\n\n**Requirements:** ❌ additional requirements needed' 600)"
pulls 124 "1111111111111111111111111111111111111111"
check "sweep: a verdict for an older head does not dedup a moved head" \
  "verdict 124" "$(sweep_out)"

# ---------------------------------------------------------------------------
# Sweep-level skips.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 130 alice User true MERGEABLE 2700 300)"
check "sweep: drafts are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 135 'dependabot[bot]' Bot false MERGEABLE 2700 300)"
check "sweep: bot-authored PRs are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 131 alice User false MERGEABLE 2700 300 somebody else)"
check "sweep: fork heads are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 132 alice User false CONFLICTING 2700 300)"
check "sweep: conflicting PRs are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 133 alice User false MERGEABLE 2700000 691200)"
check "sweep: PRs not updated within the window are skipped" "" "$(sweep_out)"

# ---------------------------------------------------------------------------
# The `check` subcommand.
# ---------------------------------------------------------------------------

reset
comments 141
check "check: review engages when nothing exists yet" "engage" "$(check_out 141 review)"

reset
comments 141 "$(comment_obj 'sir-knight-lop-the-second[bot]' \
  $'### Sir Knight Lop the Second — review\n\nposted' 600)"
check "check: review skips when a bot review already exists" \
  "skip a bot review comment already exists" "$(check_out 141 review)"

reset
comments 141
check "check: verdict skips without review-family activity" \
  "skip no review-family activity" "$(check_out 141 verdict)"

reset
comments 141 \
  "$(comment_obj alice $'### Agent review — round 1\n\nno blocker' 1200)" \
  "$(comment_obj 'github-actions[bot]' \
    $'### Sir Knight Lop the Second — verdict\n**Head:** 1111111111111111111111111111111111111111' 600)"
pulls 141 "1111111111111111111111111111111111111111"
check "check: verdict skips when a verdict exists for the current head" \
  "skip a verdict already exists for the current head" "$(check_out 141 verdict)"

reset
comments 141 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker' 1200)"
pulls 141 "1111111111111111111111111111111111111111"
check "check: verdict engages when rounds exist and no verdict does" \
  "engage" "$(check_out 141 verdict)"

# ---------------------------------------------------------------------------
# Step summary + stderr detail; empty list.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 151 alice User false MERGEABLE 2700 300)"
pr_add "$(pr_obj 152 alice User true MERGEABLE 2700 300)"
comments 151
timeline 151
summary="$tmp/summary.md"
rm -f "$summary"
if ! out="$(GITHUB_STEP_SUMMARY="$summary" bash "$reconcile" sweep 2>"$tmp/err.txt")"; then
  echo "FATAL: summary-case sweep exited non-zero" >&2
  exit 1
fi
check "summary case: the review is still the only thing on stdout" "review 151" "$out"
contains "summary: the table carries the action row" "$(cat "$summary")" "| #151 | review |"
contains "summary: the table carries the skip row" "$(cat "$summary")" "| #152 | skip |"
contains "stderr: the skip reason names the draft" "$(cat "$tmp/err.txt")" "draft"

reset
set +e
out="$(bash "$reconcile" sweep 2>/dev/null)"
rc=$?
set -e
check "empty list: rc is 0" "0" "$rc"
check "empty list: no output" "" "$out"

# ---------------------------------------------------------------------------

echo
if [ "$fails" -eq 0 ]; then
  echo "reconcile.test.sh: all ${total} checks passed"
else
  echo "reconcile.test.sh: ${fails} of ${total} checks FAILED" >&2
  exit 1
fi

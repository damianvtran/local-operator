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
# terminal-ish prefilter, per-head verdict dedup, the head-freshness gate for
# a push after the latest family comment), the per-state attempt dedupe
# (pending / successful / failed / older attempts, and PR- and mode-boundary
# titles), the sweep-level skips (draft, fork, conflict, window, bot author
# including `author.is_bot`), the age gate including the ready_for_review
# transition, the `check` subcommand including `--pr` validation, and the
# step-summary table with its dispatch lead line.
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
  run)
    case "${args[1]:-}" in
      list) cat "$dir/runs.json" ;;
      *)
        echo "stub gh: unsupported run subcommand: ${args[*]}" >&2
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

reset() { rm -f "$FIX"/*.json; echo '[]' > "$FIX/prs.json"; echo '[]' > "$FIX/runs.json"; }

pr_add() {
  jq -n --argjson doc "$(cat "$FIX/prs.json")" --argjson obj "$1" '$doc + [$obj]' \
    > "$FIX/prs.json.tmp"
  mv "$FIX/prs.json.tmp" "$FIX/prs.json"
}

# pr_obj <number> <author-login> <author-is_bot> <draft> <mergeable> \
#        <created_ago_s> <updated_ago_s> [head-owner] [head-name]
# The author object is the real `gh pr list --json author` shape (login +
# is_bot; there is no `.type` — round-1 finding F2).
pr_obj() {
  jq -n \
    --argjson number "$1" --arg login "$2" --argjson is_bot "$3" \
    --argjson draft "$4" --arg mergeable "$5" \
    --arg created "$(iso "$6")" --arg updated "$(iso "$7")" \
    --arg owner "${8:-example}" --arg name "${9:-repo}" \
    '{number: $number, title: ("PR " + ($number | tostring)), isDraft: $draft,
      createdAt: $created, updatedAt: $updated,
      author: {login: $login, is_bot: $is_bot},
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

# timeline_push <pr> <ago_s> [committed|head_ref_force_pushed]
# A push event for the head-freshness gate. Fixtures mirror the live shapes:
# `committed` events carry no top-level timestamp (committer.date is the
# fallback), force-pushes carry created_at.
timeline_push() {
  jq -n --arg t "$(iso "$2")" --arg ev "${3:-committed}" \
    'if $ev == "committed"
     then [{event: $ev, committer: {date: $t}}]
     else [{event: $ev, created_at: $t}] end' > "$FIX/timeline_$1.json"
}

# pulls <pr> <head-sha>
pulls() { jq -n --arg sha "$2" '{head: {sha: $sha}}' > "$FIX/pulls_$1.json"; }

# runs [run-object ...] — the dispatch-run list the attempt dedupe reads.
runs() {
  if [ $# -eq 0 ]; then
    echo '[]' > "$FIX/runs.json"
    return 0
  fi
  printf '%s\n' "$@" | jq -s '.' > "$FIX/runs.json"
}

# run_obj <display-title> <status> <conclusion> <created_ago_s>
# conclusion may be "" for a run that has not completed.
run_obj() {
  jq -n --arg title "$1" --arg status "$2" --arg conclusion "$3" --arg created "$(iso "$4")" \
    '{displayTitle: $title, status: $status, conclusion: $conclusion, createdAt: $created}'
}

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
pr_add "$(pr_obj 101 alice false false MERGEABLE 2700 300)"
comments 101
timeline 101
check "sweep: a waited PR with no rounds and no bot review is due a review" \
  "review 101" "$(sweep_out)"

reset
pr_add "$(pr_obj 102 alice false false MERGEABLE 600 300)"
comments 102
check "sweep: younger than the review delay is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 103 alice false false MERGEABLE 2700 300)"
comments 103
timeline 103 1200
check "sweep: created long ago but marked ready 20m ago is not engaged yet" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 104 alice false false MERGEABLE 2700 300)"
comments 104
timeline 104 2400
check "sweep: ready_for_review 40m ago engages (the wait counts from the transition)" \
  "review 104" "$(sweep_out)"

reset
pr_add "$(pr_obj 134 alice false false MERGEABLE 691200 300)"
comments 134
timeline 134 3600
check "sweep: PRs created beyond the window are skipped regardless of ready age" \
  "" "$(sweep_out)"

# ---------------------------------------------------------------------------
# Bot-review dedup.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 105 alice false false MERGEABLE 2700 300)"
comments 105 "$(comment_obj 'sir-knight-lop-the-second[bot]' \
  $'### Sir Knight Lop the Second — review \n\n**Verdict:** ✅ all requirements met' 1800)"
check "sweep: a bot review (app login, trailing space) dedups the review" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 106 alice false false MERGEABLE 2700 300)"
comments 106 "$(comment_obj 'github-actions[bot]' \
  $'### Aida — review\n\nlegacy heading' 1800)"
check "sweep: the legacy Aida heading counts as a bot review" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 107 alice false false MERGEABLE 2700 300)"
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
  pr_add "$(pr_obj "$n" alice false false MERGEABLE 7200 300)"
  comments "$n" "$(comment_obj alice $'### '"$heading"$'\n\n'"$text" 1200)"
  pulls "$n" "1111111111111111111111111111111111111111"
  timeline "$n"
done
check "sweep: every family heading the contract names yields a settled verdict" \
  "verdict 111
verdict 112
verdict 113
verdict 114
verdict 115" "$(sweep_out)"

reset
pr_add "$(pr_obj 120 alice false false MERGEABLE 7200 300)"
comments 120 \
  "$(comment_obj alice $'### Agent review — round 1\n\nblocker: x' 2400)" \
  "$(comment_obj bob $'### Agent review remediation — round 1\n\nfixed in abc123, no blockers left' 1200)"
pulls 120 "1111111111111111111111111111111111111111"
check "sweep: a thread whose latest family comment is a remediation is not terminal" \
  "" "$(sweep_out)"

reset
pr_add "$(pr_obj 121 alice false false MERGEABLE 7200 300)"
comments 121 "$(comment_obj alice $'### Agent review — round 2\n\nno blocker, approved' 300)"
pulls 121 "1111111111111111111111111111111111111111"
check "sweep: a still-settling thread is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 122 alice false false MERGEABLE 7200 300)"
comments 122 "$(comment_obj alice $'### Agent review — round 2\n\nhere is a summary of what I looked at' 1200)"
pulls 122 "1111111111111111111111111111111111111111"
check "sweep: a round with no terminal-ish signal is not engaged" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 123 alice false false MERGEABLE 7200 300)"
comments 123 \
  "$(comment_obj alice $'### Agent review — round 3\n\nno blocker; rounds clean' 1200)" \
  "$(comment_obj 'sir-knight-lop-the-second[bot]' \
    $'### Sir Knight Lop the Second — verdict\n**Head:** 1111111111111111111111111111111111111111\n\n**Requirements:** ✅ all met' 600)"
pulls 123 "1111111111111111111111111111111111111111"
check "sweep: a verdict for the current head dedups" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 124 alice false false MERGEABLE 7200 300)"
comments 124 \
  "$(comment_obj alice $'### Agent review — round 3\n\nno blocker; rounds clean' 1200)" \
  "$(comment_obj 'sir-knight-lop-the-second[bot]' \
    $'### Sir Knight Lop the Second — verdict\n**Head:** 2222222222222222222222222222222222222222\n\n**Requirements:** ❌ additional requirements needed' 600)"
pulls 124 "1111111111111111111111111111111111111111"
timeline_push 124 2400
check "sweep: a verdict for an older head does not dedup a moved head" \
  "verdict 124" "$(sweep_out)"

# ---------------------------------------------------------------------------
# F1(a): the head-freshness gate — a push after the latest review-family
# comment means the rounds predate the head, so the verdict waits.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 160 alice false false MERGEABLE 7200 300)"
comments 160 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 160 "1111111111111111111111111111111111111111"
timeline_push 160 300
err="$tmp/err-stale.txt"
out="$(bash "$reconcile" sweep 2>"$err")"
check "sweep: a push after the latest review-family comment defers the verdict" "" "$out"
contains "sweep: the stale skip says the rounds predate the head" \
  "$(cat "$err")" "rounds predate the head"

reset
pr_add "$(pr_obj 161 alice false false MERGEABLE 7200 300)"
comments 161 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 161 "1111111111111111111111111111111111111111"
timeline_push 161 600 head_ref_force_pushed
check "sweep: a force-push after the latest review-family comment also defers" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 162 alice false false MERGEABLE 7200 300)"
comments 162 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 162 "1111111111111111111111111111111111111111"
# no timeline fixture for 162: the stub fails loudly and the gate must fail
# OPEN — an unresolvable push lookup must never wedge the verdict path.
check "sweep: an unresolvable push timeline does not block the verdict" \
  "verdict 162" "$(sweep_out)"

# ---------------------------------------------------------------------------
# F1(c): the per-state attempt dedupe — a pending or successful attempt newer
# than the latest family comment already judged this state.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — verdict PR #165 (sweep)' in_progress '' 300)"
err="$tmp/err-attempt.txt"
out="$(bash "$reconcile" sweep 2>"$err")"
check "sweep: a pending attempt for this state blocks a re-dispatch" "" "$out"
contains "sweep: the attempt skip names the attempt" "$(cat "$err")" "attempt for this exact state"

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — verdict PR #165' completed success 300)"
check "sweep: a successful attempt newer than the latest round blocks (mention shape too)" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — verdict PR #165 (sweep)' completed failure 300)"
check "sweep: a failed attempt does not block — the retry path" "verdict 165" "$(sweep_out)"

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — verdict PR #165 (sweep)' completed success 3000)"
check "sweep: an attempt older than the latest round does not block (state moved on)" \
  "verdict 165" "$(sweep_out)"

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — verdict PR #1651 (sweep)' completed success 300)"
check "sweep: an attempt for another PR does not block (#165 vs #1651 boundary)" \
  "verdict 165" "$(sweep_out)"

reset
pr_add "$(pr_obj 165 alice false false MERGEABLE 7200 300)"
comments 165 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker; rounds clean' 1200)"
pulls 165 "1111111111111111111111111111111111111111"
timeline 165
runs "$(run_obj 'Sir Knight Lop the Second — review PR #165 (sweep)' completed success 300)"
check "sweep: a review attempt does not block the verdict" "verdict 165" "$(sweep_out)"

# ---------------------------------------------------------------------------
# Sweep-level skips.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 130 alice false true MERGEABLE 2700 300)"
check "sweep: drafts are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 135 dependabot true false MERGEABLE 2700 300)"
check "sweep: bot-authored PRs are skipped (author.is_bot)" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 136 'dependabot[bot]' false false MERGEABLE 2700 300)"
check "sweep: the [bot] login suffix still protects (belt-and-braces)" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 131 alice false false MERGEABLE 2700 300 somebody else)"
check "sweep: fork heads are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 132 alice false false CONFLICTING 2700 300)"
check "sweep: conflicting PRs are skipped" "" "$(sweep_out)"

reset
pr_add "$(pr_obj 133 alice false false MERGEABLE 2700000 691200)"
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

reset
comments 143 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker' 1200)"
pulls 143 "1111111111111111111111111111111111111111"
timeline_push 143 300
check "check: verdict skips when a push landed after the latest round" \
  "skip rounds predate the head (a push landed after the latest review-family comment)" "$(check_out 143 verdict)"

reset
comments 144 "$(comment_obj alice $'### Agent review — round 1\n\nno blocker' 1200)"
pulls 144 "1111111111111111111111111111111111111111"
# Two push shapes in one fixture — an older force-push and a newer committed
# event: the gate must read the max and see the latest push predates the
# round (committer.date fallback + created_at both exercised).
jq -n --arg old "$(iso 5000)" --arg new "$(iso 2400)" \
  '[{event: "head_ref_force_pushed", created_at: $old}, {event: "committed", committer: {date: $new}}]' \
  > "$FIX/timeline_144.json"
check "check: verdict engages when the latest push predates the round" \
  "engage" "$(check_out 144 verdict)"

set +e
out="$(bash "$reconcile" check --pr abc --mode review 2>&1)"
rc=$?
set -e
check "check: a non-numeric --pr fails with rc 2" "2" "$rc"
contains "check: the failure message names the value" "$out" "positive integer"

# ---------------------------------------------------------------------------
# Step summary + stderr detail; empty list.
# ---------------------------------------------------------------------------

reset
pr_add "$(pr_obj 151 alice false false MERGEABLE 2700 300)"
pr_add "$(pr_obj 152 alice false true MERGEABLE 2700 300)"
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
contains "summary: the lead line names the dispatched engagement" \
  "$(cat "$summary")" "Dispatched: review #151"

reset
pr_add "$(pr_obj 153 alice false true MERGEABLE 2700 300)"
comments 153
summary2="$tmp/summary2.md"
rm -f "$summary2"
if ! out="$(GITHUB_STEP_SUMMARY="$summary2" bash "$reconcile" sweep 2>/dev/null)"; then
  echo "FATAL: nothing-due summary-case sweep exited non-zero" >&2
  exit 1
fi
check "summary case: nothing due still prints nothing on stdout" "" "$out"
contains "summary: the lead line says nothing is due" "$(cat "$summary2")" "Dispatched: nothing"

reset
set +e
out="$(bash "$reconcile" sweep 2>/dev/null)"
rc=$?
set -e
check "empty list: rc is 0" "0" "$rc"
check "empty list: no output" "" "$out"

# ---------------------------------------------------------------------------
# The workflow guard: the deployment switch must stay wired.
# ---------------------------------------------------------------------------
# WHY: reconcile.sh's tests cannot see the workflow file, and on 2026-09-27 the
# `reconcile` job's "Check the provider key" step read `${RADIENT_API_KEY:-}`
# with no mapping anywhere in its scope — so every scheduled sweep took the
# disabled-with-a-warning path and the engine was inert while the secret WAS
# configured (lo run 36322977510 / ui run 36322896985). This scan is the guard
# for that class: a step that reads the key, with nothing mapping
# `secrets.RADIENT_API_KEY` into its environment.
#
# CONTRACT: for every step whose block reads RADIENT_API_KEY, a
# `secrets.RADIENT_API_KEY` mapping must be in scope — on the step itself or on
# its job — via an `env:` block. Prints `OK`, or one `FAIL: …` line per
# problem; no network, no `gh`.
#
# HELPDESK_WORKFLOW=<path> overrides the scanned file (default: this repo's
# `.github/workflows/helpdesk.yml`). That is how the guard is demonstrated RED:
# point it at a copy of the pre-fix YAML (or one with the mapping stripped) and
# the run must report a FAIL and exit non-zero.
#
# LIMITS — it is an indentation-based scan of the workflow's committed style,
# not a YAML parser. It understands: indent-2 jobs; indent-6 `- ` steps; an
# `env:` mapping of the exact scalar
# `RADIENT_API_KEY: ${{ secrets.RADIENT_API_KEY }}` (spacing/quotes tolerated)
# at indent 6 under a job `env:` or indent 10 under a step `env:`; and reads
# written as `$RADIENT_API_KEY` / `${RADIENT_API_KEY…}` anywhere in the step
# block. It can NOT see flow-style maps (`env: {RADIENT_API_KEY: …}`), folded
# (`>-`) values, a mapping moved under `defaults:`, re-indented files, or
# reads written without a `$`; a reformat must teach the scan the new shape.
# It errs strict, never silent: an action-input (`with:`) mapping is not env
# scope, a `$RADIENT_API_KEY` mentioned in a step comment counts as a read,
# and a file where no step reads the key at all is reported as vacuous — a
# guard that matches nothing must not pass.
workflow_file="${HELPDESK_WORKFLOW:-$here/../../workflows/helpdesk.yml}"

# The scan itself, from a heredoc so its quoting is exact. See the limits above.
cat > "$tmp/workflow-key-scope.awk" <<'AWK'
BEGIN {
  injobs = 0          # inside the `jobs:` section?
  have_job = 0; job = ""
  jobmap = 0          # job env maps the secret (indent-6 line under an `env:`)
  job_env_open = 0
  si = 0              # steps in the current job
  step_env_open = 0
  nreaders = 0; nproblems = 0
}

function flush_job(   i, nm) {
  if (!have_job) return
  for (i = 1; i <= si; i++) {
    if (sread[i]) {
      nreaders++
      if (!jobmap && !smap[i]) {
        nm = (sname[i] == "") ? "(unnamed step)" : sname[i]
        printf "FAIL: job '%s': step '%s' reads RADIENT_API_KEY but no secrets.RADIENT_API_KEY mapping is in scope (step or job env)\n", job, nm
        nproblems++
      }
    }
  }
  si = 0; jobmap = 0; job_env_open = 0; step_env_open = 0; have_job = 0
  delete sname; delete sread; delete smap
}

{
  body = $0; ind = 0
  while (substr(body, 1, 1) == " ") { ind++; body = substr(body, 2) }

  if (!injobs) {
    if (body == "jobs:") injobs = 1
    next
  }

  # The jobs section ends at the next column-0 KEY. Blank lines and column-0
  # comments (both common between jobs) must not end it.
  if (ind == 0 && body != "" && body !~ /^#/) { flush_job(); injobs = 0; next }

  # A job starts at indent 2 with `name:`.
  if (ind == 2 && body ~ /^[A-Za-z0-9_.-]+:[ \t]*$/) {
    flush_job()
    job = body; sub(/:.*/, "", job)
    have_job = 1
    next
  }

  if (!have_job) next

  # Track an open job-level `env:` block (indent 4).
  if (ind == 4) { job_env_open = (body ~ /^env:[ \t]*$/) ? 1 : 0 }

  # Job-level mapping: indent 6, directly under the job `env:`.
  if (ind == 6 && job_env_open && body ~ /^["]?RADIENT_API_KEY["]?[ \t]*:[ \t]*["]?[$][{][{][ \t]*secrets[.]RADIENT_API_KEY[ \t]*[}][}]/) {
    jobmap = 1; next
  }

  # A step starts at indent 6 with "- ".
  if (ind == 6 && substr(body, 1, 2) == "- ") {
    si++
    sname[si] = ""; sread[si] = 0; smap[si] = 0
    step_env_open = 0
    rest = substr(body, 3)
    if (rest ~ /^name:[ \t]*/) { sname[si] = rest; sub(/^name:[ \t]*/, "", sname[si]) }
    if (rest ~ /[$][{]?RADIENT_API_KEY/) sread[si] = 1
    next
  }

  # Step body: track the step `env:` (indent 8) and everything at indent >= 8.
  if (ind >= 8 && si > 0) {
    if (ind == 8) {
      step_env_open = (body ~ /^env:[ \t]*$/) ? 1 : 0
      if (body ~ /^name:[ \t]*/) { sname[si] = body; sub(/^name:[ \t]*/, "", sname[si]) }
    }
    # Step-level mapping: indent 10, directly under the step `env:`.
    if (ind == 10 && step_env_open && body ~ /^["]?RADIENT_API_KEY["]?[ \t]*:[ \t]*["]?[$][{][{][ \t]*secrets[.]RADIENT_API_KEY[ \t]*[}][}]/) smap[si] = 1
    if (body ~ /[$][{]?RADIENT_API_KEY/) sread[si] = 1
  }
}

END {
  flush_job()
  if (nreaders == 0) {
    printf "FAIL: no RADIENT_API_KEY-reading step found in %s — the scan is vacuous (wrong file, or the reads moved out of the scan's shape)\n", FILENAME
    nproblems++
  }
  if (nproblems == 0) printf "OK\n"
}
AWK

workflow_key_scope() { # <workflow-file>: prints OK, or FAIL: … lines.
  local file="$1"
  if [ ! -f "$file" ]; then
    printf 'FAIL: workflow file not found: %s\n' "$file"
    return 0
  fi
  awk -f "$tmp/workflow-key-scope.awk" "$file"
}

check "workflow guard: every RADIENT_API_KEY-reading step has the secret in scope" \
  "OK" "$(workflow_key_scope "$workflow_file")"

# ---------------------------------------------------------------------------
# The workflow guard, part 2: the engagement's helpdesk surface comes from the
# default branch — the overlay step must exist and sit between the PR checkout
# and the team install.
# ---------------------------------------------------------------------------
# WHY: the engagement job checks out `refs/pull/<n>/merge` and reads its
# surface (prompts, team, roles, setup.sh) from that checkout. A merge ref
# computed before a helpdesk change carries the old surface and none of the
# new files — on 2026-09-27 a verdict engagement crashed on exactly that
# (`cat: .github/helpdesk/prompts/pr-verdict.md: No such file or directory`,
# lo run 36329557453), and a review engagement would silently run pre-change
# prompts and team. The overlay step is the fix; this scan is the guard that
# it exists, runs after the PR checkout, runs before `setup.sh` installs the
# surface, and still replaces the checkout's copy with the default branch's.
#
# CONTRACT: in the `review` job of the workflow, a step named `Overlay the
# helpdesk surface from the default branch` appears exactly once, its line
# sits after `Check out the pull request` and before `Install the helpdesk
# team and roles into the runner's lop config`, and its body carries the
# fetch, the wholesale removal, and the restore commands. Prints `OK`, or one
# `FAIL: …` line per problem; no network, no `gh`.
#
# HELPDESK_WORKFLOW=<path> overrides the scanned file (same as part 1 above).
#
# LIMITS: textual, like part 1 — it reads the `review` job block by indentation
# and step order from line numbers; it does not parse YAML, resolve `if:`
# expressions, or notice anything about a step whose name it does not match,
# and a reformat (re-indent, rename, folded/one-line step bodies) must teach
# the scan the new shape. A file where the step is missing is reported as a
# FAIL, never a vacuous OK.
cat > "$tmp/workflow-surface-scope.awk" <<'AWK'
BEGIN { in_job = 0; in_overlay_body = 0; ovn = 0 }
{
  body = $0; ind = 0
  while (substr(body, 1, 1) == " ") { ind++; body = substr(body, 2) }

  # Job keys sit at indent 2 (`review:` etc.); only the review job is scanned.
  if (ind == 2 && body ~ /^[A-Za-z0-9_.-]+:[ \t]*$/) {
    job = body; sub(/:.*/, "", job)
    in_job = (job == "review")
    in_overlay_body = 0
    next
  }
  if (!in_job) next

  # A step starts at indent 6 with `- `; the next step ends an open body.
  if (ind == 6 && substr(body, 1, 2) == "- ") {
    in_overlay_body = 0
    rest = substr(body, 3)
    name = ""
    if (rest ~ /^name:[ \t]*/) { name = rest; sub(/^name:[ \t]*/, "", name) }
    if (name == "Check out the pull request") checkout = NR
    else if (name == "Overlay the helpdesk surface from the default branch") {
      ovn++; overlay = NR; in_overlay_body = 1
    }
    else if (name == "Install the helpdesk team and roles into the runner's lop config") setup = NR
    next
  }

  # The overlay step's body — its `run:` block — until the next step above.
  if (in_overlay_body && ind >= 8) obody = obody body "\n"
}
END {
  prob = 0
  if (ovn == 0) {
    print "FAIL: no `Overlay the helpdesk surface from the default branch` step in the review job"
    prob++
  } else if (ovn > 1) {
    print "FAIL: the overlay step appears " ovn " times in the review job"
    prob++
  }
  if (ovn == 1) {
    if (!checkout) {
      print "FAIL: no `Check out the pull request` step in the review job — cannot order the overlay against it"
      prob++
    } else if (!(checkout < overlay)) {
      print "FAIL: the overlay step does not sit after `Check out the pull request`"
      prob++
    }
    if (!setup) {
      print "FAIL: no `Install the helpdesk team and roles into the runner's lop config` step in the review job — cannot order the overlay against it"
      prob++
    } else if (!(overlay < setup)) {
      print "FAIL: the overlay step does not sit before the team install"
      prob++
    }
    if (index(obody, "git fetch --depth=1 origin main") == 0) {
      print "FAIL: the overlay step does not fetch the default branch (`git fetch --depth=1 origin main`)"
      prob++
    }
    if (index(obody, "rm -rf .github/helpdesk") == 0) {
      print "FAIL: the overlay step does not replace the surface wholesale (`rm -rf .github/helpdesk`)"
      prob++
    }
    if (index(obody, "git checkout FETCH_HEAD -- .github/helpdesk") == 0) {
      print "FAIL: the overlay step does not restore from the default branch (`git checkout FETCH_HEAD -- .github/helpdesk`)"
      prob++
    }
  }
  if (prob == 0) print "OK"
}
AWK

workflow_surface_scope() { # <workflow-file>: prints OK, or FAIL: … lines.
  local file="$1"
  if [ ! -f "$file" ]; then
    printf 'FAIL: workflow file not found: %s\n' "$file"
    return 0
  fi
  awk -f "$tmp/workflow-surface-scope.awk" "$file"
}

check "workflow guard: the helpdesk surface overlay exists, sits between the PR checkout and the team install, and restores from the default branch" \
  "OK" "$(workflow_surface_scope "$workflow_file")"

# ---------------------------------------------------------------------------

echo
if [ "$fails" -eq 0 ]; then
  echo "reconcile.test.sh: all ${total} checks passed"
else
  echo "reconcile.test.sh: ${fails} of ${total} checks FAILED" >&2
  exit 1
fi

#!/usr/bin/env bash
# Sir Knight Lop the Second — the engagement reconciler.
#
# WHY THIS EXISTS (design and contract: .github/helpdesk/README.md):
#
#   v2 DEFERS the bot: a pull request is engaged only when it is due.
#
#     * `review`  — a non-draft, same-repo PR that has waited
#                   REVIEW_DELAY_MINUTES with no review-family activity and no
#                   bot review comment yet: the concise guidelines-compliance
#                   review.
#     * `verdict` — review-family activity exists, has settled for
#                   VERDICT_SETTLE_MINUTES, the latest family comment is a
#                   round (not a remediation reply) carrying a terminal-ish
#                   signal, and the head has not moved after that comment
#                   (a push newer than the latest family comment means the
#                   rounds predate the head — the verdict waits for the new
#                   head's rounds): the terminal compliance verdict. One
#                   verdict per head (a moved head can earn a new verdict).
#
#   `sweep` decides per open PR and prints `review <n>` / `verdict <n>` per
#   action; the workflow dispatches helpdesk.yml once per line, so every
#   engagement keeps ONE execution path (the review job) with its own run, log
#   and timeout. `check` is the already-engaged gate a sweep-dispatched run
#   re-runs before spending provider tokens.
#
#   The sweep also dedupes ATTEMPTS per family state, for BOTH modes: before
#   emitting `review` or `verdict` it looks for a dispatch run of helpdesk.yml
#   for the same (PR, mode) created after the state's anchor (a verdict is
#   anchored at FAMILY_LAST_TIME; a review has no family comments, so any
#   attempt for the PR counts). A still-running or completed-successfully
#   attempt means this exact state was already judged (a prompt may have
#   posted nothing on purpose), so the sweep does not re-fire every 15
#   minutes. A completed-unsuccessful attempt backs off
#   FAILED_ATTEMPT_BACKOFF_MINUTES before the retry: failed attempts used to
#   retry every sweep and posted a failure notice each time (three landed on
#   one PR); a mention or a manual dispatch is the immediate recovery. The
#   workflow's `run-name:` is what makes an attempt visible
#   (`Sir Knight Lop the Second — {mode} PR #<n> (sweep)`), so this file and
#   the workflow's run-name move together.
#
#   Two dispatch bounds keep a busy window's spend finite without a human:
#   one sweep emits at most SWEEP_MAX_DISPATCH engagements (the OLDEST by
#   updatedAt; the rest ride the next sweep), and a repository dispatches at
#   most DAILY_ENGAGEMENT_CAP per rolling 24h — launch-day backfill dispatched
#   4-7 engagements in a single sweep. Both constants are env-overridable in
#   the config block below; raise them when a window genuinely needs more.
#
# THE MARKER CONTRACT — keep these spellings stable; the README documents them
# as a contract and every consumer greps exactly these:
#
#   * Bot review marker:   a comment authored by `sir-knight-lop-the-second[bot]`
#                          or `github-actions[bot]` whose body contains the
#                          exact heading line `### Sir Knight Lop the Second — review`
#                          (or the legacy `### Aida — review`; a
#                          `— review run failed` notice does NOT count).
#   * Bot verdict marker:  same author rule, exact heading line
#                          `### Sir Knight Lop the Second — verdict`; the
#                          comment also carries a `**Head:** <sha>` line, and
#                          dedup is PER HEAD — a moved head can earn a new
#                          verdict once its rounds go terminal.
#   * Review-family activity (the contributor's own rounds, ANY author): a
#                          comment containing a heading line matching
#                          `^### (Agent review|QA|Design review|UX review)( report)?( remediation)?\b`
#                          or `^### Gates \+ local evidence\b`.
#   * Remediation:         a family heading containing `remediation` — the
#                          implementer answering a round; a thread whose latest
#                          family comment is a remediation is not terminal yet.
#
#   The terminal-ish prefilter is deliberately LOOSE — the VERDICT PROMPT makes
#   the real terminality call, so a false positive costs one small engagement
#   (deduped per family state, so one state pays at most once) and a false
#   negative recovers via a mention or a manual dispatch.
#
# Pure read-only (QA relies on it): only reads through `gh` and prints. The
# workflow's dispatch step is the one that writes (workflow_dispatch), not this.
#
# Requires: bash, jq, gh. Thresholds and cost bounds are env-overridable
# (tests, operators):
set -euo pipefail
REVIEW_DELAY_MINUTES="${REVIEW_DELAY_MINUTES:-30}"
VERDICT_SETTLE_MINUTES="${VERDICT_SETTLE_MINUTES:-15}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"
# Cost bounds (2026-09-27 spend review — an engagement is the only thing here
# that spends provider tokens; a typical review is ~26 model calls ≈ $0.10):
#   * a young FAILED attempt backs off instead of retrying every sweep;
#   * one sweep dispatches at most SWEEP_MAX_DISPATCH (the oldest first), so
#     a burst (e.g. a launch-day backfill) cannot fan out unbounded;
#   * the rolling-24h DAILY_ENGAGEMENT_CAP caps a repository's day.
# Raise any of them when a window genuinely needs more.
FAILED_ATTEMPT_BACKOFF_MINUTES="${FAILED_ATTEMPT_BACKOFF_MINUTES:-30}"
SWEEP_MAX_DISPATCH="${SWEEP_MAX_DISPATCH:-5}"
DAILY_ENGAGEMENT_CAP="${DAILY_ENGAGEMENT_CAP:-50}"

# ---------------------------------------------------------------------------
# The marker contract, in ONE place (README, "Marker contract").
# ---------------------------------------------------------------------------
FAMILY_RE='^### (Agent review|QA|Design review|UX review)( report)?( remediation)?\b'
GATES_RE='^### Gates \+ local evidence\b'
FAMILY_RE_ALL="${FAMILY_RE}|${GATES_RE}"
BOT_REVIEW_RE='^### (Sir Knight Lop the Second|Aida) — review[[:space:]]*$'
BOT_VERDICT_RE='^### Sir Knight Lop the Second — verdict[[:space:]]*$'
VERDICT_HEAD_RE='^\*\*Head:\*\*[[:space:]]+[0-9a-fA-F]+'
# Loose terminal-ish prefilter (case-insensitive), applied to the latest
# review-family comment. The verdict prompt makes the real call.
TERMINAL_RE='terminal|no blocker|no major|0 fail|all pass|rounds? (are )?clean|clean on|approved|ready to merge|lgtm|requirements met|✅'

usage() {
  cat <<'EOF'
usage: reconcile.sh sweep
       reconcile.sh check --pr <number> --mode review|verdict

sweep:  print one `review <n>` / `verdict <n>` line per due engagement.
check:  print `engage` or `skip <reason>` for a sweep-dispatched run.
EOF
}

die() { echo "reconcile.sh: $*" >&2; exit 2; }

# The two bot identities that post for this workflow (app token, workflow-token
# fallback). Bot-authored content never starts runs; here it is also never
# "review-family activity" — the bot's own comments must not gate the bot.
is_bot_login() {
  case "$1" in
    "sir-knight-lop-the-second[bot]" | "github-actions[bot]") return 0 ;;
    *) return 1 ;;
  esac
}

# Epoch seconds for a GitHub ISO-8601 timestamp. A parse failure is LOUD: an
# empty epoch would silently become 0 in the arithmetic below and make a
# timestamp comparison decide the wrong way.
epoch_of() {
  jq -n --arg t "$1" '$t | fromdateiso8601' || die "cannot parse timestamp '$1'"
}
now_epoch() { jq -n 'now | floor'; }

# Print the sha from a verdict comment's `**Head:** <sha>` line ("" when the
# line is absent — a comment missing it is not treated as a verdict marker by
# the contract, so its dedup entry is empty and can never match a real head).
verdict_head() {
  grep -oE "$VERDICT_HEAD_RE" <<<"$1" | head -n1 | sed -E 's/^\*\*Head:\*\*[[:space:]]+//' || true
}

# The last review-family heading line of a body ("" when none).
family_last_heading() {
  { grep -E "$FAMILY_RE_ALL" <<<"$1" || true; } | tail -n1
}

pr_comments() {
  gh api "repos/${GH_REPO}/issues/$1/comments" --paginate | jq -s 'add // []'
}

pr_head_sha() {
  gh api "repos/${GH_REPO}/pulls/$1" --jq '.head.sha'
}

# The last write to the head branch, per the timeline: the newest `committed`
# or `head_ref_force_pushed` event. `committed` events carry no top-level
# timestamp (measured against the live API), so the commit's `committer.date`
# is the fallback. Empty output means "cannot judge" — callers must not block
# on it: a missing or failed lookup must never wedge the verdict path behind
# a gate whose input does not exist.
pr_last_push_time() {
  gh api "repos/${GH_REPO}/issues/$1/timeline" --paginate \
    | jq -sr 'add // [] | [ .[] | select(.event == "committed" or .event == "head_ref_force_pushed") | (.created_at // .committer.date // .author.date) ] | map(select(. != null)) | max // empty'
}

# The generalized per-state attempt matcher (was verdict-only
# `verdict_attempt_recorded`). helpdesk.yml names every run through its
# `run-name:` (`… — {mode} PR #{n}`, with ` (sweep)` on sweep dispatches), so
# attempts are visible from a run list: a dispatch run of this workflow for
# the same (PR, mode) created after the state's anchor — a verdict is
# anchored at the latest family comment; a review has no family state, so
# `since` is empty and any attempt for the PR counts.
#
# Prints a skip reason when this exact state must not be re-dispatched, or
# nothing when it may proceed. Two rules, in order:
#   1. an attempt still running (queued/in_progress) or completed
#      successfully already judged this exact state — do not re-dispatch it;
#   2. the newest attempt completed UNSUCCESSFULLY within
#      FAILED_ATTEMPT_BACKOFF_MINUTES — a young failure waits before the
#      retry (a mention or a manual dispatch is the immediate recovery).
# A lookup failure prints nothing: it must not block the retry path.
attempt_skip_reason() { # <pr> <mode> [<since-iso>]
  local pr="$1" mode="$2" since="${3:-}" runs
  runs="$(gh run list -R "$GH_REPO" --workflow helpdesk.yml --event workflow_dispatch \
    --limit 50 --json displayTitle,status,conclusion,createdAt)" || return 0
  jq -r --arg pr "$pr" --arg mode "$mode" --arg since "$since" \
        --argjson now "$(now_epoch)" --argjson backoff "$((FAILED_ATTEMPT_BACKOFF_MINUTES * 60))" '
    [ .[] | select(.displayTitle | test("— " + $mode + " PR #" + $pr + "( |$)"))
          | select($since == "" or ((.createdAt | fromdateiso8601) > ($since | fromdateiso8601)))
          | {status, conclusion, epoch: (.createdAt | fromdateiso8601)} ] as $attempts
    | ($attempts | map(select(.status != "completed" or .conclusion == "success")) | length) as $blocked
    | ($attempts | sort_by(.epoch) | last) as $newest
    | if $blocked > 0 then
        "a \($mode) attempt for this exact state is already on record"
      elif $newest != null and $newest.status == "completed" and $newest.conclusion != null
           and $newest.conclusion != "success" and (($now - $newest.epoch) < $backoff) then
        "a recent \($mode) attempt failed \(((($now - $newest.epoch) / 60) | floor)) min ago; retrying after it ages"
      else empty end' <<<"$runs"
}

# Sweep-dispatched attempts in the rolling 24h, for DAILY_ENGAGEMENT_CAP.
# Counted from the run names (` (sweep)` suffix — mention and manual runs are
# not budgeted). A lookup failure counts 0: the cap bounds a burst, it must
# not become a second outage.
sweep_dispatched_24h() {
  local runs
  runs="$(gh run list -R "$GH_REPO" --workflow helpdesk.yml --event workflow_dispatch \
    --limit 200 --json displayTitle,createdAt)" || { echo 0; return 0; }
  jq -r --argjson now "$(now_epoch)" '
    [ .[] | select(.displayTitle | endswith(" (sweep)"))
          | select(($now - (.createdAt | fromdateiso8601)) < 86400) ] | length' <<<"$runs"
}

# Classify one PR's comments into the scan globals. One PR is in flight at a
# time, so plain globals keep this bash-3.2-safe (no namerefs/associative
# arrays) while still keeping the contract in one place:
#   BOT_REVIEW_SEEN   true|false — a bot review marker exists
#   FAMILY_COUNT      integer    — review-family comments seen
#   FAMILY_LAST_TIME  ISO string — created_at of the latest family comment
#   FAMILY_LAST_BODY  string     — its body
#   FAMILY_LAST_HEAD  string     — its last family heading line
#   VERDICT_HEADS     space-padded list of heads that already have a verdict
scan_comments() {
  local pr="$1" comments count c login body ts vh
  comments="$(pr_comments "$pr")"
  BOT_REVIEW_SEEN=false
  FAMILY_COUNT=0
  FAMILY_LAST_TIME=""
  FAMILY_LAST_BODY=""
  FAMILY_LAST_HEAD=""
  VERDICT_HEADS=" "
  count="$(jq 'length' <<<"$comments")"
  for ((c = 0; c < count; c++)); do
    login="$(jq -r --argjson c "$c" '.[$c].user.login // ""' <<<"$comments")"
    body="$(jq -r --argjson c "$c" '.[$c].body // ""' <<<"$comments")"
    ts="$(jq -r --argjson c "$c" '.[$c].created_at // ""' <<<"$comments")"
    if is_bot_login "$login"; then
      if grep -qE "$BOT_REVIEW_RE" <<<"$body"; then BOT_REVIEW_SEEN=true; fi
      if grep -qE "$BOT_VERDICT_RE" <<<"$body"; then
        # A marker with no `**Head:**` line contributes nothing to the dedup
        # list: the contract says the line is part of the marker, so a
        # malformed comment must not match every real head.
        vh="$(verdict_head "$body")"
        if [ -n "$vh" ]; then
          VERDICT_HEADS="${VERDICT_HEADS}${vh} "
        fi
      fi
    fi
    if grep -qE "$FAMILY_RE_ALL" <<<"$body"; then
      FAMILY_COUNT=$((FAMILY_COUNT + 1))
      FAMILY_LAST_TIME="$ts"
      FAMILY_LAST_BODY="$body"
      FAMILY_LAST_HEAD="$(family_last_heading "$body")"
    fi
  done
}

# ---- sweep ----------------------------------------------------------------

# Row/action recorders. `rows` and `actions` are local to run_sweep; bash
# dynamic scoping lets these helpers append to them while the detail goes to
# stderr and only the `review <n>` / `verdict <n>` lines to stdout.
record_row() { rows+=("| #$1 | $2 | $3 |"); }
record_skip() {
  record_row "$1" "skip" "$2"
  printf 'pr #%s: skip — %s\n' "$1" "$2" >&2
}
record_action() { # <num> <mode> <reason> <updated-epoch>
  record_row "$1" "$2" "$3"
  printf 'pr #%s: %s — %s\n' "$1" "$2" "$3" >&2
  actions+=("$2 $1")
  # The updatedAt epoch rides along for the SWEEP_MAX_DISPATCH ordering:
  # when more actions are due than one sweep may emit, the OLDEST go first.
  action_updated+=("$4")
}

# The step summary. Written only when the runner provides the channel file;
# invisible locally.
write_summary() {
  [ -n "${GITHUB_STEP_SUMMARY:-}" ] || return 0
  {
    echo "## Helpdesk sweep"
    echo
    # Lead with what was dispatched (actionable first): the per-PR table
    # below stays in PR order, so without this line a due engagement is
    # buried among the skips.
    if [ "${#actions[@]}" -gt 0 ]; then
      printf 'Dispatched:'
      for a in "${actions[@]}"; do
        printf ' %s #%s;' "${a%% *}" "${a#* }"
      done
      echo
    elif [ -n "$daily_cap_note" ]; then
      echo "Dispatched: nothing — ${daily_cap_note}."
    else
      echo "Dispatched: nothing — no engagement is due."
    fi
    if [ -n "$more_due_note" ]; then
      echo
      echo "${more_due_note}."
    fi
    echo
    echo "Thresholds: review after ${REVIEW_DELAY_MINUTES}m, verdict after ${VERDICT_SETTLE_MINUTES}m settle, window ${WINDOW_DAYS}d (reconcile.sh constants; env-overridable)."
    echo
    echo "Bounds: failed-attempt backoff ${FAILED_ATTEMPT_BACKOFF_MINUTES}m; ≤${SWEEP_MAX_DISPATCH} dispatched per sweep; ≤${DAILY_ENGAGEMENT_CAP} per rolling 24h."
    echo
    echo "| PR | Decision | Reason |"
    echo "| --- | --- | --- |"
    if [ "${#rows[@]}" -gt 0 ]; then
      printf '%s\n' "${rows[@]}"
    else
      echo "| (none) | skip | no open pull requests in scope |"
    fi
  } >> "$GITHUB_STEP_SUMMARY"
}

run_sweep() {
  local prs count i num draft login author_bot owner name mergeable created updated
  local repo_owner repo_name head_label
  local created_epoch updated_epoch start ready start_epoch head settle_epoch now latest_push
  local skip_reason dispatched_today more_due_note daily_cap_note n_pick picked
  local k best best_epoch
  local -a actions rows action_updated keep_actions keep_updated

  [ -n "${GH_REPO:-}" ] || die "GH_REPO is not set (owner/name)"
  for tool in jq gh; do command -v "$tool" >/dev/null || die "$tool is required"; done

  actions=()
  rows=()
  action_updated=()
  now="$(now_epoch)"
  repo_owner="${GH_REPO%%/*}"
  repo_name="${GH_REPO#*/}"

  prs="$(gh pr list --state open --limit 200 --json number,title,isDraft,createdAt,updatedAt,author,headRepositoryOwner,headRepository,mergeable)"
  count="$(jq 'length' <<<"$prs")"

  for ((i = 0; i < count; i++)); do
    num="$(jq -r --argjson i "$i" '.[$i].number' <<<"$prs")"
    draft="$(jq -r --argjson i "$i" '.[$i].isDraft' <<<"$prs")"
    login="$(jq -r --argjson i "$i" '.[$i].author.login // ""' <<<"$prs")"
    # Bot authors: `gh pr list` reports `author.is_bot` (its real shape — the
    # old `.author.type` read was dead); the `[bot]` login suffix arm below is
    # belt-and-braces.
    author_bot="$(jq -r --argjson i "$i" '.[$i].author.is_bot // false' <<<"$prs")"
    owner="$(jq -r --argjson i "$i" '.[$i].headRepositoryOwner.login // ""' <<<"$prs")"
    name="$(jq -r --argjson i "$i" '.[$i].headRepository.name // ""' <<<"$prs")"
    mergeable="$(jq -r --argjson i "$i" '.[$i].mergeable // "UNKNOWN"' <<<"$prs")"
    created="$(jq -r --argjson i "$i" '.[$i].createdAt' <<<"$prs")"
    updated="$(jq -r --argjson i "$i" '.[$i].updatedAt' <<<"$prs")"
    head_label="${owner:-unknown}/${name:-unknown}"

    # Sweep-level skips, all recorded with a reason for the step summary.
    if [ "$draft" = "true" ]; then
      record_skip "$num" "draft"
      continue
    fi
    if [ "$author_bot" = "true" ] || [[ "$login" == *"[bot]" ]]; then
      record_skip "$num" "bot author (${login})"
      continue
    fi
    if [ "$owner" != "$repo_owner" ] || [ "$name" != "$repo_name" ]; then
      record_skip "$num" "fork (head ${head_label})"
      continue
    fi
    if [ "$mergeable" = "CONFLICTING" ]; then
      record_skip "$num" "conflicting (its merge ref does not exist; resolve, then the sweep re-engages)"
      continue
    fi
    updated_epoch="$(epoch_of "$updated")"
    if [ "$((now - updated_epoch))" -gt "$((WINDOW_DAYS * 86400))" ]; then
      record_skip "$num" "outside the ${WINDOW_DAYS}-day window"
      continue
    fi

    scan_comments "$num"

    if [ "$FAMILY_COUNT" -gt 0 ]; then
      # Verdict path: the contributor's rounds exist; engage only when the
      # thread has settled and the LATEST family comment is a round that reads
      # terminal-ish. The verdict prompt re-judges terminality for real.
      settle_epoch="$(( $(epoch_of "$FAMILY_LAST_TIME") + VERDICT_SETTLE_MINUTES * 60 ))"
      if [ "$now" -lt "$settle_epoch" ]; then
        record_skip "$num" "review-family activity is still settling (<${VERDICT_SETTLE_MINUTES}m old)"
        continue
      fi
      if [[ "$FAMILY_LAST_HEAD" == *remediation* ]]; then
        record_skip "$num" "the latest review-family comment is a remediation reply"
        continue
      fi
      if ! grep -Eiq "$TERMINAL_RE" <<<"$FAMILY_LAST_BODY"; then
        record_skip "$num" "the latest review-family comment carries no terminal-ish signal"
        continue
      fi
      head="$(pr_head_sha "$num")"
      case " ${VERDICT_HEADS} " in
        *" ${head} "*)
          record_skip "$num" "a verdict already exists for the current head"
          continue
          ;;
      esac
      # Rounds fresh on the head: a push after the latest family comment
      # means those rounds judge an older head, and a clean round on an older
      # head is not terminal — engaging would only spend a verdict run that
      # must post nothing. Unresolvable push data does not block.
      latest_push="$(pr_last_push_time "$num" || true)"
      if [ -n "$latest_push" ] \
        && [ "$(epoch_of "$latest_push")" -gt "$(epoch_of "$FAMILY_LAST_TIME")" ]; then
        record_skip "$num" "rounds predate the head (a push landed after the latest review-family comment)"
        continue
      fi
      # One attempt per family state: a still-running or successful attempt
      # newer than the latest family comment already judged this state, and
      # posting nothing is a legitimate outcome — do not re-fire every sweep.
      # A young failed attempt backs off first (attempt_skip_reason).
      skip_reason="$(attempt_skip_reason "$num" "verdict" "$FAMILY_LAST_TIME")"
      if [ -n "$skip_reason" ]; then
        record_skip "$num" "$skip_reason"
        continue
      fi
      record_action "$num" "verdict" "rounds settled and terminal-ish; no verdict for this head yet" "$updated_epoch"
      continue
    fi

    if [ "$BOT_REVIEW_SEEN" = true ]; then
      record_skip "$num" "a bot review already exists; waiting for the contributor's rounds"
      continue
    fi

    # Review path: no rounds and no bot review. The wait is counted from when
    # the PR became reviewable — its creation, or the latest ready_for_review
    # transition when it was a draft — and the timeline is fetched only here,
    # where the ready-age actually decides.
    created_epoch="$(epoch_of "$created")"
    if [ "$((now - created_epoch))" -lt "$((REVIEW_DELAY_MINUTES * 60))" ]; then
      record_skip "$num" "younger than ${REVIEW_DELAY_MINUTES}m; the sweep waits for the contributor's rounds"
      continue
    fi
    if [ "$((now - created_epoch))" -gt "$((WINDOW_DAYS * 86400))" ]; then
      record_skip "$num" "created more than ${WINDOW_DAYS} days ago"
      continue
    fi
    start="$created"
    ready="$(gh api "repos/${GH_REPO}/issues/${num}/timeline" --paginate | jq -sr 'add // [] | map(select(.event == "ready_for_review")) | last | .created_at // empty')"
    if [ -n "$ready" ]; then start="$ready"; fi
    start_epoch="$(epoch_of "$start")"
    if [ "$((now - start_epoch))" -lt "$((REVIEW_DELAY_MINUTES * 60))" ]; then
      record_skip "$num" "marked ready for review less than ${REVIEW_DELAY_MINUTES}m ago"
      continue
    fi
    # Same attempt dedupe as the verdict path: a still-running or successful
    # review attempt already judged this state (a review has no family anchor,
    # so any attempt for the PR counts), and a young failure backs off first.
    # This is what keeps a second sweep from re-dispatching a review while the
    # first run is still going, or re-firing a review that posted nothing.
    skip_reason="$(attempt_skip_reason "$num" "review" "")"
    if [ -n "$skip_reason" ]; then
      record_skip "$num" "$skip_reason"
      continue
    fi
    record_action "$num" "review" "waited ${REVIEW_DELAY_MINUTES}m+ with no rounds and no bot review" "$updated_epoch"
  done

  # The dispatch cost bounds — due → caps → dispatch. The rolling-24h budget
  # is checked first: at/over cap nothing dispatches (the day's engagements
  # are spent; the summary says so). Otherwise a sweep with more than
  # SWEEP_MAX_DISPATCH actions keeps the OLDEST and lets the rest ride the
  # next sweep. Both notes go to stderr and (via the globals) the summary.
  more_due_note=""
  daily_cap_note=""
  if [ "${#actions[@]}" -gt 0 ]; then
    dispatched_today="$(sweep_dispatched_24h)"
    if [ "$dispatched_today" -ge "$DAILY_ENGAGEMENT_CAP" ]; then
      daily_cap_note="daily engagement budget reached (${dispatched_today}/${DAILY_ENGAGEMENT_CAP}); due engagements resume after the window rolls"
      printf '%s\n' "$daily_cap_note" >&2
      actions=()
      action_updated=()
    elif [ "${#actions[@]}" -gt "$SWEEP_MAX_DISPATCH" ]; then
      # Selection with no pipes: `sort | head` would close the pipe early
      # and `pipefail` would turn a full read into a dead sweep.
      n_pick="${#actions[@]}"
      picked=" "
      keep_actions=(); keep_updated=()
      for ((k = 0; k < SWEEP_MAX_DISPATCH; k++)); do
        best=-1
        best_epoch=""
        for ((i = 0; i < n_pick; i++)); do
          case "$picked" in *" $i "*) continue ;; esac
          if [ "$best" -lt 0 ] || [ "${action_updated[$i]}" -lt "$best_epoch" ]; then
            best="$i"
            best_epoch="${action_updated[$i]}"
          fi
        done
        picked="${picked}${best} "
        keep_actions+=("${actions[$best]}")
        keep_updated+=("$best_epoch")
      done
      more_due_note="$((n_pick - SWEEP_MAX_DISPATCH)) more due; they ride the next sweep"
      printf '%s\n' "$more_due_note" >&2
      actions=("${keep_actions[@]}")
      action_updated=("${keep_updated[@]}")
    fi
  fi

  # The machine-readable output: one action per line for the workflow to
  # dispatch. Everything else above is stderr / the step summary.
  if [ "${#actions[@]}" -gt 0 ]; then
    printf '%s\n' "${actions[@]}"
  fi
  write_summary
}

# ---- check ----------------------------------------------------------------

run_check() {
  local pr="$1" mode="$2" head latest_push
  [ -n "${GH_REPO:-}" ] || die "GH_REPO is not set (owner/name)"

  # The PR number arrives from a workflow dispatch: a typo must fail clearly
  # HERE — a 404 from `gh api .../pulls/abc` reads like a missing PR.
  case "$pr" in
    ''|*[!0-9]*) die "check --pr must be a positive integer, got '${pr}'" ;;
  esac
  [ "$pr" -ge 1 ] || die "check --pr must be a positive integer, got '${pr}'"

  scan_comments "$pr"

  case "$mode" in
    review)
      if [ "$BOT_REVIEW_SEEN" = true ]; then
        echo "skip a bot review comment already exists"
      else
        echo "engage"
      fi
      ;;
    verdict)
      if [ "$FAMILY_COUNT" -eq 0 ]; then
        echo "skip no review-family activity"
      else
        head="$(pr_head_sha "$pr")"
        case " ${VERDICT_HEADS} " in
          *" ${head} "*)
            echo "skip a verdict already exists for the current head"
            ;;
          *)
            # A push after the latest family comment means the rounds judge
            # an older head — a clean round on an older head is not terminal,
            # so the verdict must not fire. Unresolvable push data does not
            # block.
            latest_push="$(pr_last_push_time "$pr" || true)"
            if [ -n "$latest_push" ] \
              && [ "$(epoch_of "$latest_push")" -gt "$(epoch_of "$FAMILY_LAST_TIME")" ]; then
              echo "skip rounds predate the head (a push landed after the latest review-family comment)"
            else
              echo "engage"
            fi
            ;;
        esac
      fi
      ;;
    *)
      die "unknown --mode '${mode}' (expected review or verdict)"
      ;;
  esac
}

main() {
  [ $# -ge 1 ] || { usage >&2; exit 2; }
  local cmd="$1"
  shift

  case "$cmd" in
    sweep)
      [ $# -eq 0 ] || { usage >&2; exit 2; }
      run_sweep
      ;;
    check)
      local pr="" mode=""
      while [ $# -gt 0 ]; do
        case "$1" in
          --pr)
            [ $# -ge 2 ] || die "--pr needs a value"
            pr="$2"
            shift 2
            ;;
          --mode)
            [ $# -ge 2 ] || die "--mode needs a value"
            mode="$2"
            shift 2
            ;;
          *)
            die "unknown argument '$1'"
            ;;
        esac
      done
      [ -n "$pr" ] || die "check requires --pr <number>"
      [ -n "$mode" ] || die "check requires --mode review|verdict"
      run_check "$pr" "$mode"
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"

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
#                   VERDICT_SETTLE_MINUTES, and the latest family comment is a
#                   round (not a remediation reply) carrying a terminal-ish
#                   signal: the terminal compliance verdict. One verdict per
#                   head (a moved head can earn a new verdict).
#
#   `sweep` decides per open PR and prints `review <n>` / `verdict <n>` per
#   action; the workflow dispatches helpdesk.yml once per line, so every
#   engagement keeps ONE execution path (the review job) with its own run, log
#   and timeout. `check` is the already-engaged gate a sweep-dispatched run
#   re-runs before spending provider tokens.
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
#   and a false negative recovers via a mention or a manual dispatch.
#
# Pure read-only (QA relies on it): only reads through `gh` and prints. The
# workflow's dispatch step is the one that writes (workflow_dispatch), not this.
#
# Requires: bash, jq, gh. Thresholds are env-overridable (tests, operators):
set -euo pipefail
REVIEW_DELAY_MINUTES="${REVIEW_DELAY_MINUTES:-30}"
VERDICT_SETTLE_MINUTES="${VERDICT_SETTLE_MINUTES:-15}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"

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
record_action() {
  record_row "$1" "$2" "$3"
  printf 'pr #%s: %s — %s\n' "$1" "$2" "$3" >&2
  actions+=("$2 $1")
}

# The step summary. Written only when the runner provides the channel file;
# invisible locally.
write_summary() {
  [ -n "${GITHUB_STEP_SUMMARY:-}" ] || return 0
  {
    echo "## Helpdesk sweep"
    echo
    echo "Thresholds: review after ${REVIEW_DELAY_MINUTES}m, verdict after ${VERDICT_SETTLE_MINUTES}m settle, window ${WINDOW_DAYS}d (reconcile.sh constants; env-overridable)."
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
  local prs count i num draft login author_type owner name mergeable created updated
  local repo_owner repo_name head_label
  local created_epoch updated_epoch start ready start_epoch head settle_epoch now
  local -a actions rows

  [ -n "${GH_REPO:-}" ] || die "GH_REPO is not set (owner/name)"
  for tool in jq gh; do command -v "$tool" >/dev/null || die "$tool is required"; done

  actions=()
  rows=()
  now="$(now_epoch)"
  repo_owner="${GH_REPO%%/*}"
  repo_name="${GH_REPO#*/}"

  prs="$(gh pr list --state open --limit 100 --json number,title,isDraft,createdAt,updatedAt,author,headRepositoryOwner,headRepository,mergeable)"
  count="$(jq 'length' <<<"$prs")"

  for ((i = 0; i < count; i++)); do
    num="$(jq -r --argjson i "$i" '.[$i].number' <<<"$prs")"
    draft="$(jq -r --argjson i "$i" '.[$i].isDraft' <<<"$prs")"
    login="$(jq -r --argjson i "$i" '.[$i].author.login // ""' <<<"$prs")"
    author_type="$(jq -r --argjson i "$i" '.[$i].author.type // ""' <<<"$prs")"
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
    if [ "$author_type" = "Bot" ] || [[ "$login" == *"[bot]" ]]; then
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
      record_action "$num" "verdict" "rounds settled and terminal-ish; no verdict for this head yet"
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
    record_action "$num" "review" "waited ${REVIEW_DELAY_MINUTES}m+ with no rounds and no bot review"
  done

  # The machine-readable output: one action per line for the workflow to
  # dispatch. Everything else above is stderr / the step summary.
  if [ "${#actions[@]}" -gt 0 ]; then
    printf '%s\n' "${actions[@]}"
  fi
  write_summary
}

# ---- check ----------------------------------------------------------------

run_check() {
  local pr="$1" mode="$2" head
  [ -n "${GH_REPO:-}" ] || die "GH_REPO is not set (owner/name)"

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
            echo "engage"
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

#!/usr/bin/env bash
# Tests for .github/helpdesk/report_failure.sh and its workflow wiring.
#
#   bash .github/helpdesk/tests/report_failure.test.sh
#
# A stubbed `gh` on PATH serves the comment list from a fixture and RECORDS
# every write (method, path, body), so the tests assert what would reach
# GitHub: the first failure posts, later failures PATCH the same comment with
# an incremented count, other headings are left alone, an install failure is
# marked, and a failed PATCH falls back to posting. The workflow guards at the
# end pin both harness installs to one version and both failure steps to the
# script (HELPDESK_WORKFLOW re-points them for a RED demonstration).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script="$here/../report_failure.sh"
workflow_file="${HELPDESK_WORKFLOW:-$here/../../workflows/helpdesk.yml}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"

cat > "$tmp/bin/gh" <<'STUB'
#!/usr/bin/env bash
# Stub gh: `api --paginate <comments>` serves $STUB_DIR/comments.json;
# `api -X PATCH|POST <path> -F body=@file` appends to $STUB_DIR/writes.log.
set -euo pipefail
dir="${STUB_DIR:?}"
[ "$1" = "api" ] || { echo "stub gh: unsupported: $*" >&2; exit 1; }
shift
method=GET body=""
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift 2 ;;
    -F) body="${2#body=@}"; shift 2 ;;
    --paginate) shift ;;
    *) args+=("$1"); shift ;;
  esac
done
path="${args[0]}"
case "$method" in
  GET) cat "$dir/comments.json" ;;
  PATCH)
    [ -f "$dir/patch_fails" ] && { echo "stub gh: 403" >&2; exit 1; }
    printf '%s %s\n%s\n---\n' "$method" "$path" "$(cat "$body")" >> "$dir/writes.log" ;;
  POST) printf '%s %s\n%s\n---\n' "$method" "$path" "$(cat "$body")" >> "$dir/writes.log" ;;
  *) echo "stub gh: unsupported method $method" >&2; exit 1 ;;
esac
STUB
chmod +x "$tmp/bin/gh"
export PATH="$tmp/bin:$PATH" STUB_DIR="$tmp" GH_REPO="example/repo" REPORT_NOW="2026-11-02T00:00:00Z"

fails=0 total=0
check() { # <name> <expected> <actual>
  total=$((total + 1))
  if [ "$2" = "$3" ]; then echo "ok   $1"; else
    fails=$((fails + 1)); printf 'FAIL %s\n  expected: %s\n  actual:   %s\n' "$1" "$2" "$3"; fi
}
reset() { rm -f "$tmp/writes.log" "$tmp/patch_fails"; echo "${1:-[]}" > "$tmp/comments.json"; : > "$tmp/writes.log"; }
H="### Sir Knight Lop the Second — verdict run failed"
run() { bash "$script" 7 "$H" "The run failed." "https://example/runs/$1" "${2:-}" > /dev/null; }

reset '[]'
run 1
check "first failure posts a new notice" "POST repos/example/repo/issues/7/comments" "$(head -n1 "$tmp/writes.log")"
check "first notice records one failure" "1" "$(grep -c '^Failures recorded: 1$' "$tmp/writes.log")"
check "first notice carries the run link" "1" "$(grep -c 'runs/1$' "$tmp/writes.log")"

existing="$(jq -n --arg h "$H" '[{id: 11, body: "### Sir Knight Lop the Second — review run failed\n\nother"},
  {id: 42, body: ($h + "\n\nThe run failed.\n\nLast failure: x — run log: y\nFailures recorded: 4\n")}]')"
reset "$existing"
run 2
check "repeat failure PATCHes the matching notice, never posts" "PATCH repos/example/repo/issues/comments/42" "$(head -n1 "$tmp/writes.log")"
check "only one write per failure" "1" "$(grep -c '^---$' "$tmp/writes.log")"
check "count increments from the existing notice" "1" "$(grep -c '^Failures recorded: 5$' "$tmp/writes.log")"
check "run link is refreshed" "1" "$(grep -c 'runs/2$' "$tmp/writes.log")"

reset "$(jq -n '[{id: 11, body: "### Sir Knight Lop the Second — review run failed\n\nother"}]')"
run 3
check "a different mode's notice is not reused" "POST repos/example/repo/issues/7/comments" "$(head -n1 "$tmp/writes.log")"

reset "$existing"
touch "$tmp/patch_fails"
run 4
check "failed PATCH falls back to posting (never a silent drop)" "POST repos/example/repo/issues/7/comments" "$(head -n1 "$tmp/writes.log")"

reset '[]'
run 5 install
check "install failure is marked distinctly" "1" "$(grep -c 'Cause: harness install failed' "$tmp/writes.log")"
reset '[]'
run 6
check "non-install failure carries no install marker" "0" "$(grep -c 'Cause: harness install failed' "$tmp/writes.log" || true)"

# --- workflow wiring ---------------------------------------------------------
pins="$(grep -o 'local-operator==[0-9][0-9.]*' "$workflow_file" | sort -u)"
check "workflow: both harness installs share one pin" "1" "$(printf '%s\n' "$pins" | grep -c .)"
check "workflow: harness pin is installable from PyPI (not the pruned 0.62.38)" "0" "$(grep -c 'local-operator==0\.62\.38' "$workflow_file" || true)"
# shellcheck disable=SC2016  # literal $RUNNER_TEMP is what the workflow carries
check "workflow: both failure steps go through report_failure.sh" "2" "$(grep -c 'bash "\$RUNNER_TEMP/report_failure.sh"' "$workflow_file")"
check "workflow: no direct failure-notice comment remains" "0" "$(grep -cE 'gh (pr|issue) comment .*helpdesk-notice' "$workflow_file" || true)"
check "workflow: both installs expose an outcome to the reporter" "2" "$(grep -c 'id: install$' "$workflow_file")"

echo
if [ "$fails" -eq 0 ]; then echo "report_failure.test.sh: all ${total} checks passed"; else
  echo "report_failure.test.sh: ${fails} of ${total} checks FAILED" >&2; exit 1; fi

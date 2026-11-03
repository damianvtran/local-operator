#!/usr/bin/env bash
set -euo pipefail

# Chrome Web Store release client for the Local Operator extension. Three modes,
# dispatched by the two workflows that hold the store credentials:
#
#   chrome-web-store.sh stage ZIP VERSION
#       Uploads the validated zip, then requests review with deferred release
#       (STAGED_PUBLISH). Run by .github/workflows/chrome-web-store.yml. Its three
#       refusals echo store-supplied scalars, so those go through store_value()
#       (redacted, then bounded) like everything else this script prints.
#
#   chrome-web-store.sh promote VERSION
#       Reads fetchStatus, requires the SUBMITTED revision to be STAGED with
#       VERSION at 100% deployment, publishes it, and polls until PUBLISHED.
#       Run by .github/workflows/chrome-web-store-promote.yml.
#
#   chrome-web-store.sh status
#       Read-only: a single fetchStatus GET, printed as the same compact summary
#       the two promote gates attach to a refusal. Nothing is uploaded, nothing
#       is published and no store state is touched, so the review queue can be
#       read without a stage dispatch -- which, while an item is in review,
#       draws HTTP 400 FAILED_PRECONDITION / NOT_UPDATEABLE. Takes no arguments
#       beyond the mode (and refuses any it is given, rather than ignoring an
#       argument a caller believes it is acting on), and exits non-zero on a
#       non-2xx response like every other mode. Dispatched (hourly and on
#       demand) by .github/workflows/chrome-web-store-status.yml, which runs it
#       with the two read-only controls below.
#
# Read-only controls, used by the status workflow and by nothing else:
#
#   CWS_READ_ONLY=1
#       Refuses every mode except `status`, and makes request() refuse every
#       HTTP method except GET. The status workflow holds a real store token on
#       an unattended hourly schedule, so "it only calls status" must not rest
#       on one argument in YAML: with this set, a later edit that passes `stage`
#       or `promote` -- or a code path that reaches publish_staged() -- fails
#       before any request leaves the runner. The workflow also mints the token
#       with the chromewebstore.readonly scope, so the store itself would refuse
#       a write; this flag is the half of that pair that fails without a round
#       trip and is testable offline.
#
#   CWS_STATUS_OUTPUT=PATH
#       `status` additionally writes a small JSON record (schema
#       local-operator/chrome-web-store-status/v1) to PATH -- on success AND on
#       failure, so a monitor that downloads the artifact always learns which of
#       the two happened. Every store-supplied value in it passes the same
#       redact-then-bound rules as the log line. CWS_STATUS_EXPECTED_VERSION, when
#       set, is copied in and compared, so a monitor can read one boolean for
#       "the submitted revision is this exact version, STAGED, at 100%".
#
# What fetchStatus cannot tell you: WHY a revision was rejected. The v2
# FetchItemStatusResponse carries a REJECTED state and the `warned`/`takenDown`
# booleans, but no reason text (discovery document, read 2026-11-03); the reason
# is only in the Developer Dashboard and the review email. The record carries
# those three fields verbatim rather than inventing a reason.
#
# What the summary says, and why those fields. Per revision:
#   submitted state=STAGED distributionChannels=[crxVersion=0.1.10 deployPercentage=100]
# -- the state the first gate reads, and every field of every
# distributionChannels[] entry, under the name the store used. Three answers are
# kept apart because collapsing them hides a store-side shape change: a key the
# store did not send reads `<absent>`, one it sent as null reads `<null>`, and a
# value of a type the field cannot hold reads `<not a string: object value={...}>`
# (and its siblings) rather than being dropped. An empty list reads `[]`. Every
# value is bounded (160 characters) and the list is cut after four entries, so the
# summary stays one line whatever the store returns, and no single bad field can
# blank it. The token substitution runs inside the renderer, before that cut,
# since a redaction applied to the finished line cannot see a token the bound has
# already shortened.
#
# Why both promote gates print the store's own fields. Promoting the 0.1.12
# staged revision was refused with nothing but "staged revision must contain
# version 0.1.12 at 100% deployment", and repeated dispatches since then have
# failed the same way. That sentence names what the gate WANTED and never what
# the store SAID, so a rollout still settling (a deployPercentage below 100) and
# a response whose shape the gate does not recognise (distributionChannels still
# describing the live 0.1.10 revision, say) were indistinguishable, and the only
# remedy on offer was a blind re-dispatch on a timer. Both gates therefore append
# the same rendering that `status` prints, and both keep their leading text so
# log greps in the docs and in past runs still match. The fields are the store's
# own (StatusResponse.submittedItemRevisionStatus.state and its
# distributionChannels[]), taken from the v2 discovery document the comments in
# report_api_error cite, so nothing here has to be re-derived by the next person
# to hit it.

MODE=${1:-}
ZIP_PATH=${2:-}
EXPECTED_VERSION=${3:-}
API_ROOT=${CWS_API_ROOT:-https://chromewebstore.googleapis.com}
EXPECTED_EXTENSION_ID=omibaecbjdhgbbcedbnnnmjpmopfheof

fail() {
  # Kept for the status record's EXIT trap, which reports why a read failed.
  failure_message="$*"
  # `status` reads and never publishes, so its failures must not say "publish
  # failed": a monitor or operator reading the poll log would take a failed read
  # for a failed release. stage/promote keep the old prefix verbatim, because
  # docs/store/release-record.md and past run logs are grepped for it.
  if [[ "$MODE" == "status" ]]; then
    printf 'Chrome Web Store status read failed: %s\n' "$*" >&2
  else
    printf 'Chrome Web Store publish failed: %s\n' "$*" >&2
  fi
  exit 1
}
failure_message=""
last_http_status=""

for command in curl jq; do
  command -v "$command" >/dev/null || fail "$command is required"
done
: "${CWS_PUBLISHER_ID:?CWS_PUBLISHER_ID is required}"
: "${CWS_EXTENSION_ID:?CWS_EXTENSION_ID is required}"
: "${CWS_ACCESS_TOKEN:?CWS_ACCESS_TOKEN is required}"
[[ "$CWS_EXTENSION_ID" == "$EXPECTED_EXTENSION_ID" ]] \
  || fail "CWS_EXTENSION_ID must be the permanent Local Operator ID $EXPECTED_EXTENSION_ID"
[[ "$MODE" == "stage" || "$MODE" == "promote" || "$MODE" == "status" ]] \
  || fail "usage: chrome-web-store.sh stage ZIP VERSION | promote VERSION | status"
# Exactly "1", not "non-empty": an accidental CWS_READ_ONLY=0 must not read as
# read-only, and a typo in the value must not silently disarm the guard either,
# so anything other than unset/empty/"1" is refused.
case "${CWS_READ_ONLY:-}" in
  "") ;;
  1) [[ "$MODE" == "status" ]] \
       || fail "CWS_READ_ONLY=1 permits only the status mode, refusing $MODE" ;;
  *) fail "CWS_READ_ONLY must be unset or 1" ;;
esac
[[ -z "${CWS_STATUS_OUTPUT:-}" || "$MODE" == "status" ]] \
  || fail "CWS_STATUS_OUTPUT applies only to the status mode"

item="publishers/$CWS_PUBLISHER_ID/items/$CWS_EXTENSION_ID"
status_url="$API_ROOT/v2/$item:fetchStatus"
publish_url="$API_ROOT/v2/$item:publish"
tmp_dir=$(mktemp -d)

# The status record (see CWS_STATUS_OUTPUT in the header). One writer for every
# outcome, so the success and failure records cannot drift apart in shape.
# Store-supplied text reaches jq through the environment and never argv (ps),
# and is redacted BEFORE it is bounded -- the same order, for the same reason,
# as summarize_status. $1 is the fetchStatus file (empty on failure), $2 the
# summary line, $3 an error excerpt (empty on success).
STATUS_RECORD_ERROR_LIMIT=1000
write_status_record() {
  local response=${1:-}
  [[ -n "$response" && -s "$response" ]] || response=/dev/null
  CWS_RECORD_SUMMARY=${2-} CWS_RECORD_ERROR=${3-} \
  CWS_RECORD_HTTP_STATUS=$last_http_status \
  CWS_RECORD_CHECKED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
    jq -n \
    --argjson channel_limit "$STATUS_CHANNEL_LIMIT" \
    --argjson value_limit "$STATUS_VALUE_LIMIT" \
    --argjson error_limit "$STATUS_RECORD_ERROR_LIMIT" \
    --arg extension_id "$CWS_EXTENSION_ID" \
    --arg expected "${CWS_STATUS_EXPECTED_VERSION:-}" \
    --slurpfile responses "$response" '
    (env.CWS_ACCESS_TOKEN // "") as $token
    | def redact($text):
        if $token == "" then $text
        else ($text | split($token) | join("<redacted CWS_ACCESS_TOKEN>")) end;
      def bounded($text; $limit):
        if ($text | length) > $limit then $text[0:$limit] + "..." else $text end;
      # null stays null (the store did not say); anything else becomes a
      # redacted, bounded string, so a wrong-typed field cannot inflate the file.
      def text($value; $limit):
        if $value == null then null
        else (try bounded(redact($value | if type == "string" then . else tojson end); $limit)
              catch "<unrenderable>") end;
      def flag($value): if ($value | type) == "boolean" then $value else null end;
      def percent($value): if ($value | type) == "number" then $value else text($value; $value_limit) end;
      def revision($status):
        if ($status | type) != "object" then null
        else {
          state: text($status.state; $value_limit),
          channels: (if ($status.distributionChannels | type) == "array"
            then [$status.distributionChannels[0:$channel_limit][]
                  | if type == "object"
                    then {crxVersion: text(.crxVersion; $value_limit),
                          deployPercentage: percent(.deployPercentage)}
                    else {unrecognised: text(.; $value_limit)} end]
            else [] end),
          channel_count: (if ($status.distributionChannels | type) == "array"
            then ($status.distributionChannels | length) else 0 end)
        } end;
      ($responses[0] // null) as $r
      | (if ($r | type) == "object" then $r else {} end) as $o
      | ($expected | if . == "" then null else . end) as $want
      | {
          schema: "local-operator/chrome-web-store-status/v1",
          checked_at: env.CWS_RECORD_CHECKED_AT,
          extension_id: $extension_id,
          ok: (env.CWS_RECORD_ERROR == "" and ($r | type) == "object"),
          http_status: (env.CWS_RECORD_HTTP_STATUS | if . == "" then null else (tonumber? // .) end),
          expected_version: $want,
          submitted: revision($o.submittedItemRevisionStatus),
          published: revision($o.publishedItemRevisionStatus),
          last_async_upload_state: text($o.lastAsyncUploadState; $value_limit),
          warned: flag($o.warned),
          taken_down: flag($o.takenDown),
          # The one boolean the promote gate also requires: the SUBMITTED
          # revision is STAGED and carries exactly the expected version at 100%.
          # Every branch yields exactly one boolean: an expression that yielded
          # nothing for an odd shape would drop the whole record, not the field.
          expected_version_staged_at_full: (
            if $want == null then false
            else ($o.submittedItemRevisionStatus
              | if type == "object" and .state == "STAGED"
                then ((.distributionChannels | if type == "array" then . else [] end)
                      | any(type == "object" and .crxVersion == $want and .deployPercentage == 100))
                else false end)
            end),
          summary: (env.CWS_RECORD_SUMMARY | if . == "" then null else text(.; 4 * $value_limit * 8) end),
          error: (env.CWS_RECORD_ERROR | if . == "" then null else text(.; $error_limit) end)
        }
    ' >"$CWS_STATUS_OUTPUT.partial" 2>/dev/null \
    && mv -f "$CWS_STATUS_OUTPUT.partial" "$CWS_STATUS_OUTPUT" && return 0
  # Fail closed. The record IS this mode's deliverable: a poll that read the
  # store but left no record (or a truncated one) would let a monitor treat the
  # missing artifact as "nothing new" while the job shows green. Writing to a
  # sibling then renaming means a failed write never leaves a half-written
  # file at PATH for the upload step to publish as if it were the answer.
  rm -f "$CWS_STATUS_OUTPUT.partial" 2>/dev/null || true
  printf 'Chrome Web Store status read failed: could not write the status record to %s\n' \
    "$CWS_STATUS_OUTPUT" >&2
  return 1
}

status_record_written=""
on_exit() {
  local rc=$?
  # A failed status read still leaves a record, so a monitor never has to guess
  # from a missing artifact. report_api_error writes its own (it has the body);
  # this covers transport failures, non-JSON answers and the itemId check.
  if [[ "$MODE" == "status" && -n "${CWS_STATUS_OUTPUT:-}" && -z "$status_record_written" ]]; then
    # rc is already non-zero on this path; a write failure here is reported by
    # the writer, and the shell's exit status stays the original failure.
    write_status_record "" "" "${failure_message:-status read exited with code $rc}" || true
  fi
  rm -rf "$tmp_dir"
}
trap on_exit EXIT

# Print the store's own explanation of a rejected request, then fail.
#
# REDACTION — deliberately narrow, do not widen. Only the RESPONSE BODY is
# echoed. GitHub Actions masks secrets it injected into the job environment, but
# it does NOT mask arbitrary text a script prints, so anything echoed here is
# public in the run log. Request headers therefore never appear (they carry the
# bearer token), $CWS_ACCESS_TOKEN is never interpolated into a message, and any
# exact occurrence of the token inside the body is replaced before printing so
# that an API which echoed request context back cannot turn this diagnostic into
# a credential leak. The substitution uses bash parameter expansion rather than
# sed so the secret never reaches another process's argv, where `ps` would show
# it. Do not extend this to dump headers, the curl command line, or the
# environment.
# A broken gateway or a hostile intermediary must not be able to flood the run
# log: an unbounded echo works against the readability this script exists to
# deliver (a single 400 was measured at 3,000,626 bytes of step output). The
# sibling verify-release-environment.sh bounds its body the same way.
BODY_PRINT_LIMIT=4000

# A store-supplied SCALAR that lands in a refusal message, redacted and bounded in
# that order -- which is report_api_error's order above, and the order is the
# whole point rather than a detail. Both halves have been wrong here before: the
# values went into the message verbatim, so an API that echoed the request context
# could put the bearer token in the run log in full (measured: a synthetic
# 200-character token, emitted whole on all three refusals), and a single oversized
# field produced a 200,066-byte line. Substitute FIRST: a cut applied to the token
# leaves a prefix that the substitution can no longer match, which is exactly how
# summarize_status used to leak (see its redact()).
#
# Control characters are escaped for the same reason the summary escapes them: a
# value carrying newlines would otherwise turn one refusal into many lines, and a
# multi-line refusal is one nobody greps.
VALUE_PRINT_LIMIT=200

store_value() {
  local text=${1-}
  local truncated=""
  text=${text//"$CWS_ACCESS_TOKEN"/<redacted CWS_ACCESS_TOKEN>}
  text=${text//$'\n'/\\n}
  text=${text//$'\r'/\\r}
  text=${text//$'\t'/\\t}
  if [[ ${#text} -gt $VALUE_PRINT_LIMIT ]]; then
    truncated="..."
    text=${text:0:$VALUE_PRINT_LIMIT}
  fi
  printf '%s%s' "$text" "$truncated"
}

report_api_error() {
  local label=$1
  local status=$2
  local output=$3
  local body
  # Pretty-print when the payload is JSON; a proxy or gateway can answer with
  # HTML, which is still worth showing verbatim rather than discarding.
  body=$(jq . "$output" 2>/dev/null) || body=$(cat "$output" 2>/dev/null) || body=""
  body=${body//"$CWS_ACCESS_TOKEN"/<redacted CWS_ACCESS_TOKEN>}
  # Truncate AFTER redacting, so a token near the cut cannot survive by landing
  # on the boundary. Bash parameter expansion rather than `head -c`: this body
  # is a variable, not a file, and `printf ... | head -c` makes head exit early
  # and SIGPIPE the producer, which under `set -o pipefail` aborts this function
  # with 141 before the remedy below is ever printed (measured). The sibling can
  # use `head -c` safely only because it reads a file.
  local truncated=""
  if [[ ${#body} -gt $BODY_PRINT_LIMIT ]]; then
    truncated=" (truncated to $BODY_PRINT_LIMIT of ${#body} characters)"
    body=${body:0:$BODY_PRINT_LIMIT}
  fi
  if [[ "$MODE" == "status" && -n "${CWS_STATUS_OUTPUT:-}" ]]; then
    # $body is already redacted and bounded above; the writer bounds it again.
    # Marked written either way: a failed write is reported by the writer, and
    # retrying it from the EXIT trap would only print the same error twice.
    status_record_written=1
    write_status_record "" "" "$label call returned HTTP $status: $body" || true
  fi
  printf 'Chrome Web Store %s call returned HTTP %s for extension %s v%s.\n' \
    "$label" "$status" "$CWS_EXTENSION_ID" "${EXPECTED_VERSION:-<unknown>}" >&2
  if [[ -n "$body" ]]; then
    # The truncation is announced in the same line that introduces the body, so
    # a cut-off payload cannot be misread as the store's complete answer.
    printf 'Response body (verbatim, from the store)%s:\n%s\n' "$truncated" "$body" >&2
    # No error-reason translation table on purpose. The v2 discovery document
    # (revision 20260906) publishes response schemas but no enum of error
    # reasons, so any mapping from a reason string to a remedy would be invented
    # rather than sourced -- and a confident wrong diagnosis is worse than the
    # body above, which is Google's own words. The pointers below are limited to
    # operations the REST reference documents. This paragraph interprets the
    # body, so it stays inside this branch: with no body there is nothing for it
    # to refer to.
    # The withdraw remedy concerns a refused SUBMISSION, which a read-only poll
    # never makes; printing it in an unattended poll's log would advertise
    # cancelling a pending review, which forfeits the queue position (AGENTS.md).
    # The non-read-only text stays byte-identical so existing log greps match.
    if [[ "${CWS_READ_ONLY:-}" == 1 ]]; then
      printf 'That body is the store speaking, not this script.\n' >&2
    else
      printf 'That body is the store speaking, not this script. If it reports that a\n' >&2
      printf 'submission is already under review, the documented remedies are to wait for\n' >&2
      printf 'that review to finish, or to withdraw it (the cancelSubmission method, or\n' >&2
      printf 'Cancel in the Developer Dashboard) before submitting again.\n' >&2
    fi
  else
    printf 'The store returned no response body to explain this.\n' >&2
  fi
  # Always useful, body or not: where to read the item's current state.
  printf 'Current state: GET %s/v2/%s:fetchStatus\n' "$API_ROOT" "$item" >&2
  exit 1
}

request() {
  local method=$1
  local url=$2
  local output=$3
  local label=$4
  shift 4
  local status
  local rc=0
  # The read-only guard sits at the one place every store call passes through,
  # so no future caller of request() can route around it.
  [[ "${CWS_READ_ONLY:-}" != 1 || "$method" == GET ]] \
    || fail "CWS_READ_ONLY=1 refuses $method $label; only GET is permitted"
  # `--write-out %{http_code}` decides the outcome here instead of
  # `--fail-with-body`. That flag downloads the body and STILL exits 22, so
  # `set -e` killed the script before anything read the file: run 34178951112
  # reported nothing but `curl: (22) The requested URL returned error: 400`
  # while the explanation sat unread in $tmp_dir/upload.json. Judging the status
  # in the shell keeps the body in hand long enough to print it.
  status=$(curl --silent --show-error \
    --write-out '%{http_code}' \
    -X "$method" \
    -H "Authorization: Bearer $CWS_ACCESS_TOKEN" \
    "$@" \
    -o "$output" \
    "$url") || rc=$?
  last_http_status=$status
  [[ $rc -eq 0 ]] \
    || fail "$label call could not reach the Chrome Web Store (curl exit $rc)"
  [[ "$status" == 2[0-9][0-9] ]] || report_api_error "$label" "$status" "$output"
  jq -e . "$output" >/dev/null || fail "$label call returned a non-JSON response"
}

fetch_status() {
  request GET "$status_url" "$1" fetchStatus
  [[ $(jq -r '.itemId // empty' "$1") == "$CWS_EXTENSION_ID" ]] \
    || fail "status response identified a different extension"
}

# Bounds on the compact status summary. distributionChannels[] is a list whose
# length and whose fields' sizes come from the store, so every rendered value is
# bounded individually and the whole line is bounded by arithmetic rather than by
# trusting the payload: 2 revisions x 4 channels x 4 fields x (limit + "..."),
# plus fixed text. The reasoning is BODY_PRINT_LIMIT's -- the summary exists to
# make a failure readable, and an unbounded echo works against exactly that -- and
# a value-level bound is what stops one absurd field from defeating it (a 3 MiB
# `state` rendered as 3,145,904 bytes of run log when only deployInfos was cut).
STATUS_CHANNEL_LIMIT=4
STATUS_VALUE_LIMIT=160

# Print the store's own status fields for one fetchStatus response, compactly
# enough to sit inside a refusal message or to be the whole output of `status`.
#
# Every shape is handled where it is read, never by one guard over the whole
# summary: this runs on a path that is already failing, and a refusal that blanks
# itself is worse than no refusal at all. A revision that is not an object, a
# distributionChannels that is not an array, or a single entry of the wrong type
# must cost only its own field -- never the `state` the first gate reads, and
# never a good neighbour in the same list.
#
# Absent, null and wrong-type are three different answers, and the markers say
# which: `<absent>` for a key the store did not send, `<null>` for one it sent as
# null, `<not a string: object value={...}>` and its siblings for a value of a type
# the field cannot hold, and `<unrecognised shape: ...>` for an object carrying
# none of the fields the gate reads. Collapsing any two of those is how a
# store-side shape change stays invisible, which is the failure this whole
# summary exists to end; `state` in particular is read by the first gate, so a
# missing state and a null one must not look the same.
summarize_status() {
  local file=$1
  local summary
  # Scalars are rendered as `key=value` and every channel entry keeps the field
  # names the STORE uses, so a shape change is legible as itself (a `version=`
  # where crxVersion was expected) instead of looking like an absent field.
  summary=$(jq -r \
    --argjson channel_limit "$STATUS_CHANNEL_LIMIT" \
    --argjson value_limit "$STATUS_VALUE_LIMIT" \
    '
    (env.CWS_ACCESS_TOKEN // "") as $token
    | def bounded($text; $limit):
        if ($text | length) > $limit then $text[0:$limit] + "..." else $text end;
      # Redaction runs HERE, before the cut, and that order is load-bearing. Doing
      # it afterwards on the assembled line -- as this script used to -- cannot
      # see a token the bound has already shortened: the surviving prefix no
      # longer matches the substitution, so a 185-character token put its first
      # 160 characters into a public run log, and a token straddling the cut
      # leaked everything before it. `split`/`join` rather than `gsub`, because
      # the gsub pattern is a regex and a credential is not one. The token arrives
      # through the environment (jq reads env) and never as an argument, so it
      # stays out of the jq process argv, where `ps` would show it.
      def redact($text):
        if $token == "" then $text
        else ($text | split($token) | join("<redacted CWS_ACCESS_TOKEN>"))
        end;
      # A store-supplied value may carry newlines, and a value that does turns one
      # refusal into as many lines as it likes: 3 million newlines in `state`
      # rendered 161 lines (measured), which is a refusal nobody greps. Escaped
      # AFTER the redaction, so the escape cannot split a credential in half and
      # leave a prefix the substitution has already looked past.
      def escape($text):
        $text | split("\n") | join("\\n")
              | split("\r") | join("\\r")
              | split("\t") | join("\\t");
      # `tostring` is total in jq (it renders any value, objects and arrays
      # included) but `length` is not, so the cut is taken on the string form and
      # anything that still fails is replaced rather than allowed to propagate.
      def scalar($value):
        try bounded(escape(redact($value | tostring)); $value_limit) catch "<unrenderable>";
      # A field of a status object, with the three answers that used to collapse
      # into one: absent, present-and-null, or a type the field cannot hold. The
      # offending type and its (redacted, bounded) value are named so the marker
      # says which of the three this is.
      def field($obj; $key; $want):
        if ($obj | has($key) | not) then "<absent>"
        elif $obj[$key] == null then "<null>"
        elif ($obj[$key] | type) != $want
          then "<not a \($want): \($obj[$key] | type) value=\(scalar($obj[$key] | tojson))>"
        else scalar($obj[$key])
        end;
      def entry($channel):
        if ($channel | type) != "object"
          then "<not an object: \($channel | type) value=\(scalar($channel | tojson))>"
        else
          # The whole body is parenthesised because jq 1.8 changed binding
          # precedence (jq 1.8.0 NEWS, jqlang/jq#3053): an unparenthesised
          # `try E as $x | body catch ...` compiled on 1.7 but is a syntax error
          # on 1.8, which blanked every status line once ubuntu-latest shipped
          # jq 1.8.1. On 1.7 the `try` already guarded the binding AND the body,
          # so the parentheses keep that scope and the output on both versions.
          try (
            ([ (["crxVersion", "version", "deployPercentage"][]) as $key
               | select($channel[$key] != null)
               | "\($key)=\(scalar($channel[$key]))" ]
             + [ select($channel.deployInfos != null)
                 | "deployInfos=\(scalar($channel.deployInfos | tojson))" ]) as $fields
            | if ($fields | length) == 0
              then "<unrecognised shape: \(scalar($channel | tojson))>"
              else $fields | join(" ")
              end
          )
          catch "<unrenderable entry: \(scalar($channel | tojson))>"
        end;
      def channels($list):
        if ($list | length) == 0 then "[]"
        else "[" + ($list[0:$channel_limit] | map(entry(.)) | join(" | "))
             + (if ($list | length) > $channel_limit
                then " | +\(($list | length) - $channel_limit) more"
                else "" end) + "]"
        end;
      def channels_field($status):
        if ($status | has("distributionChannels") | not) then "<absent>"
        elif $status.distributionChannels == null then "<null>"
        elif ($status.distributionChannels | type) != "array"
          then "<not an array: \($status.distributionChannels | type) value=\(scalar($status.distributionChannels | tojson))>"
        else channels($status.distributionChannels)
        end;
      # Takes the root and the key rather than the bare value, because the whole
      # point of the vocabulary above is that a key the store never sent and one it
      # sent as null are different answers -- including for the revision object
      # itself, which used to collapse them into one `<absent>`.
      def revision($label; $root; $key):
        if ($root | has($key) | not) then $label + " <absent>"
        elif $root[$key] == null then $label + " <null>"
        elif ($root[$key] | type) != "object"
          then $label + " <not an object: \($root[$key] | type) value=\(scalar($root[$key] | tojson))>"
        else $label + " state=" + field($root[$key]; "state"; "string")
          + " distributionChannels=" + channels_field($root[$key])
        end;
      . as $root
      | if ($root | type) != "object"
        then "<unreadable status response: not an object: \($root | type) value=\(scalar($root | tojson))>"
        else (try
            ([ revision("submitted"; $root; "submittedItemRevisionStatus"),
               revision("published"; $root; "publishedItemRevisionStatus"),
               # Optional field: omitted when the store never sent the key, but
               # `<null>` when it sent one that is null, since those differ too.
               (if ($root | has("lastAsyncUploadState"))
                then "lastAsyncUploadState=" + (if $root.lastAsyncUploadState == null
                     then "<null>" else scalar($root.lastAsyncUploadState) end)
                else empty end)
             ] | join("; "))
          catch "<unreadable status response: \(scalar($root | tojson))>")
        end
    ' "$file" 2>/dev/null) || summary=""
  # Reachable only when jq itself fails -- an unparseable response. Every shape
  # INSIDE the payload is handled per field above, because falling back to one
  # sentence here would discard the readable fields along with the bad one.
  [[ -n "$summary" ]] || summary="<status response carried no readable fields>"
  # The substitution inside the renderer is the load-bearing one: every
  # store-supplied value reaches the summary through scalar(), so by the time this
  # line runs there is nothing left for it to catch that the inner pass has not
  # already handled. It stays as the outer net for text this script composes around
  # those values -- which carries no response text today, so it is defence in depth
  # rather than coverage -- and it cannot be the only one, because it cannot see a
  # token the bound has already shortened. Nor does it stand in for the inner pass
  # on an unsupported jq: without `env` the program fails to compile (measured:
  # rc 3, no output), the fallback below prints instead of a summary, and there is
  # no leak to catch because there is no rendered response text at all.
  printf '%s' "${summary//"$CWS_ACCESS_TOKEN"/<redacted CWS_ACCESS_TOKEN>}"
}

revision_has_full_deploy() {
  local response=$1
  local revision=$2
  local version=$3
  jq -e --arg revision "$revision" --arg version "$version" '
    .[$revision].distributionChannels // []
    | any(.crxVersion == $version and .deployPercentage == 100)
  ' "$response" >/dev/null
}

publish_staged() {
  local output=$1
  # Reusing STAGED_PUBLISH is intentional: the first call requests review with
  # deferred release; on an approved STAGED revision the same API operation is
  # Google's explicit promotion path and does not create another submission.
  request POST "$publish_url" "$output" publish \
    -H 'Content-Type: application/json' \
    --data '{"publishType":"STAGED_PUBLISH","deployInfos":[{"deployPercentage":100}],"blockOnWarnings":true}'
}

if [[ "$MODE" == "stage" ]]; then
  [[ -n "$ZIP_PATH" && -n "$EXPECTED_VERSION" ]] \
    || fail "stage requires ZIP and VERSION arguments"
  [[ -f "$ZIP_PATH" ]] || fail "archive not found: $ZIP_PATH"
  "$(dirname "$0")/validate-store-zip.sh" "$ZIP_PATH" "$EXPECTED_VERSION"

  upload_url="$API_ROOT/upload/v2/$item:upload"
  request POST "$upload_url" "$tmp_dir/upload.json" upload \
    -H 'Content-Type: application/zip' \
    --upload-file "$ZIP_PATH"

  upload_state=$(jq -r '.uploadState // empty' "$tmp_dir/upload.json")
  if [[ "$upload_state" == "SUCCEEDED" ]]; then
    uploaded_version=$(jq -r '.crxVersion // empty' "$tmp_dir/upload.json")
    [[ "$uploaded_version" == "$EXPECTED_VERSION" ]] \
      || fail "store accepted version $(store_value "$uploaded_version"), expected $EXPECTED_VERSION"
  elif [[ "$upload_state" == "IN_PROGRESS" || "$upload_state" == "UPLOAD_IN_PROGRESS" ]]; then
    # fetchStatus exposes only a global lastAsyncUploadState and no operation ID
    # or draft version. It cannot prove that a later SUCCEEDED belongs to this
    # zip, so publishing after polling could submit somebody else's concurrent
    # upload. Leave the validated draft unsubmitted for a deliberate retry.
    fail "asynchronous upload cannot be bound to version $EXPECTED_VERSION; retry after processing finishes"
  else
    fail "upload ended in unexpected state $(store_value "${upload_state:-<missing>}")"
  fi

  publish_staged "$tmp_dir/publish.json"
  publish_state=$(jq -r '.state // empty' "$tmp_dir/publish.json")
  [[ "$publish_state" == "PENDING_REVIEW" || "$publish_state" == "STAGED" ]] \
    || fail "staged submission returned unexpected state $(store_value "${publish_state:-<missing>}")"
  printf 'submitted Chrome Web Store extension %s v%s with STAGED_PUBLISH (%s)\n' \
    "$CWS_EXTENSION_ID" "$EXPECTED_VERSION" "$publish_state"
elif [[ "$MODE" == "promote" ]]; then
  EXPECTED_VERSION=$ZIP_PATH
  [[ -n "$EXPECTED_VERSION" ]] || fail "promote requires VERSION"
  fetch_status "$tmp_dir/before.json"
  [[ $(jq -r '.submittedItemRevisionStatus.state // empty' "$tmp_dir/before.json") == "STAGED" ]] \
    || fail "only an approved STAGED revision can be promoted (store said: $(summarize_status "$tmp_dir/before.json"))"
  revision_has_full_deploy "$tmp_dir/before.json" submittedItemRevisionStatus "$EXPECTED_VERSION" \
    || fail "staged revision must contain version $EXPECTED_VERSION at 100% deployment (store said: $(summarize_status "$tmp_dir/before.json"))"

  publish_staged "$tmp_dir/publish.json"
  for _ in $(seq 1 12); do
    fetch_status "$tmp_dir/after.json"
    state=$(jq -r '.publishedItemRevisionStatus.state // empty' "$tmp_dir/after.json")
    if [[ "$state" == "PUBLISHED" ]] \
      && revision_has_full_deploy "$tmp_dir/after.json" publishedItemRevisionStatus "$EXPECTED_VERSION"; then
      printf 'promoted Chrome Web Store extension %s v%s to PUBLISHED\n' \
        "$CWS_EXTENSION_ID" "$EXPECTED_VERSION"
      exit 0
    fi
    sleep "${CWS_POLL_INTERVAL_SECONDS:-10}"
  done
  # Same diagnosis as the two gates above: the deadline is the one place where
  # the store's own answer is the only way to tell a slow rollout from a
  # publish call that was accepted but never took effect.
  fail "version $EXPECTED_VERSION was not PUBLISHED at 100% before the polling deadline (last response said: $(summarize_status "$tmp_dir/after.json"))"
else
  # `status`: read the queue, mutate nothing. MODE was validated above, so this
  # is the only branch left; `request` handles a non-2xx by way of
  # report_api_error, which is what makes the read fail closed.
  # Refused rather than ignored: an argument this mode cannot act on would
  # otherwise look honoured, and `status VERSION` reads like a check of that
  # version.
  [[ -z "$ZIP_PATH" && -z "$EXPECTED_VERSION" ]] \
    || fail "status takes no arguments (usage: chrome-web-store.sh status)"
  fetch_status "$tmp_dir/status.json"
  status_summary=$(summarize_status "$tmp_dir/status.json")
  printf 'Chrome Web Store status for extension %s: %s\n' \
    "$CWS_EXTENSION_ID" "$status_summary"
  if [[ -n "${CWS_STATUS_OUTPUT:-}" ]]; then
    status_record_written=1
    # The read succeeded but its record could not be kept: exit non-zero so the
    # job is red, never green with no artifact (exit, not fail(), so the error
    # is printed once, by the writer).
    write_status_record "$tmp_dir/status.json" "$status_summary" "" || exit 1
  fi
fi

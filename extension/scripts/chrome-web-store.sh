#!/usr/bin/env bash
set -euo pipefail

# Chrome Web Store release client for the Local Operator extension. Three modes,
# dispatched by the two workflows that hold the store credentials:
#
#   chrome-web-store.sh stage ZIP VERSION
#       Uploads the validated zip, then requests review with deferred release
#       (STAGED_PUBLISH). Run by .github/workflows/chrome-web-store.yml.
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
#       non-2xx response like every other mode. No workflow dispatches it yet:
#       the store credentials exist only inside the protected environments, so
#       exposing it needs a dispatch path of its own, which is a release-process
#       decision rather than a diagnostic one (docs/store/release-record.md
#       states the same: there is no status-only workflow to dispatch).
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
  printf 'Chrome Web Store publish failed: %s\n' "$*" >&2
  exit 1
}

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

item="publishers/$CWS_PUBLISHER_ID/items/$CWS_EXTENSION_ID"
status_url="$API_ROOT/v2/$item:fetchStatus"
publish_url="$API_ROOT/v2/$item:publish"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

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
    printf 'That body is the store speaking, not this script. If it reports that a\n' >&2
    printf 'submission is already under review, the documented remedies are to wait for\n' >&2
    printf 'that review to finish, or to withdraw it (the cancelSubmission method, or\n' >&2
    printf 'Cancel in the Developer Dashboard) before submitting again.\n' >&2
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
      # `tostring` is total in jq (it renders any value, objects and arrays
      # included) but `length` is not, so the cut is taken on the string form and
      # anything that still fails is replaced rather than allowed to propagate.
      def scalar($value):
        try bounded(redact($value | tostring); $value_limit) catch "<unrenderable>";
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
          try
            ([ (["crxVersion", "version", "deployPercentage"][]) as $key
               | select($channel[$key] != null)
               | "\($key)=\(scalar($channel[$key]))" ]
             + [ select($channel.deployInfos != null)
                 | "deployInfos=\(scalar($channel.deployInfos | tojson))" ]) as $fields
          | if ($fields | length) == 0
            then "<unrecognised shape: \(scalar($channel | tojson))>"
            else $fields | join(" ")
            end
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
      def revision($label; $status):
        if $status == null then $label + " <absent>"
        elif ($status | type) != "object"
          then $label + " <not an object: \($status | type) value=\(scalar($status | tojson))>"
        else $label + " state=" + field($status; "state"; "string")
          + " distributionChannels=" + channels_field($status)
        end;
      . as $root
      | if ($root | type) != "object"
        then "<unreadable status response: not an object: \($root | type) value=\(scalar($root | tojson))>"
        else (try
            ([ revision("submitted"; $root.submittedItemRevisionStatus),
               revision("published"; $root.publishedItemRevisionStatus),
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
  # The substitution above is the load-bearing one; this is the outer net, kept
  # for the text that never passed through scalar() -- the markers this script
  # composes itself -- and for a jq too old to expose `env`, where the inner
  # redaction silently does nothing and this is all there is. It cannot see a
  # token the bound has already shortened, which is why it cannot be the only one.
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
      || fail "store accepted version $uploaded_version, expected $EXPECTED_VERSION"
  elif [[ "$upload_state" == "IN_PROGRESS" || "$upload_state" == "UPLOAD_IN_PROGRESS" ]]; then
    # fetchStatus exposes only a global lastAsyncUploadState and no operation ID
    # or draft version. It cannot prove that a later SUCCEEDED belongs to this
    # zip, so publishing after polling could submit somebody else's concurrent
    # upload. Leave the validated draft unsubmitted for a deliberate retry.
    fail "asynchronous upload cannot be bound to version $EXPECTED_VERSION; retry after processing finishes"
  else
    fail "upload ended in unexpected state ${upload_state:-<missing>}"
  fi

  publish_staged "$tmp_dir/publish.json"
  publish_state=$(jq -r '.state // empty' "$tmp_dir/publish.json")
  [[ "$publish_state" == "PENDING_REVIEW" || "$publish_state" == "STAGED" ]] \
    || fail "staged submission returned unexpected state ${publish_state:-<missing>}"
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
  printf 'Chrome Web Store status for extension %s: %s\n' \
    "$CWS_EXTENSION_ID" "$(summarize_status "$tmp_dir/status.json")"
fi

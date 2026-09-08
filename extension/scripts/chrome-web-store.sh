#!/usr/bin/env bash
set -euo pipefail

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
[[ "$MODE" == "stage" || "$MODE" == "promote" ]] \
  || fail "usage: chrome-web-store.sh stage ZIP VERSION | promote VERSION"

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
report_api_error() {
  local label=$1
  local status=$2
  local output=$3
  local body
  # Pretty-print when the payload is JSON; a proxy or gateway can answer with
  # HTML, which is still worth showing verbatim rather than discarding.
  body=$(jq . "$output" 2>/dev/null) || body=$(cat "$output" 2>/dev/null) || body=""
  body=${body//"$CWS_ACCESS_TOKEN"/<redacted CWS_ACCESS_TOKEN>}
  printf 'Chrome Web Store %s call returned HTTP %s for extension %s v%s.\n' \
    "$label" "$status" "$CWS_EXTENSION_ID" "${EXPECTED_VERSION:-<unknown>}" >&2
  if [[ -n "$body" ]]; then
    printf 'Response body (verbatim, from the store):\n%s\n' "$body" >&2
  else
    printf 'Response body was empty.\n' >&2
  fi
  # No error-reason translation table on purpose. The v2 discovery document
  # (revision 20260906) publishes response schemas but no enum of error reasons,
  # so any mapping from a reason string to a remedy would be invented rather
  # than sourced -- and a confident wrong diagnosis is worse than the body
  # above, which is Google's own words. The pointers below are limited to
  # operations the REST reference documents.
  printf 'That body is the store speaking, not this script. If it reports that a\n' >&2
  printf 'submission is already under review, the documented remedies are to wait for\n' >&2
  printf 'that review to finish, or to withdraw it (the cancelSubmission method, or\n' >&2
  printf 'Cancel in the Developer Dashboard) before submitting again.\n' >&2
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
else
  EXPECTED_VERSION=$ZIP_PATH
  [[ -n "$EXPECTED_VERSION" ]] || fail "promote requires VERSION"
  fetch_status "$tmp_dir/before.json"
  [[ $(jq -r '.submittedItemRevisionStatus.state // empty' "$tmp_dir/before.json") == "STAGED" ]] \
    || fail "only an approved STAGED revision can be promoted"
  revision_has_full_deploy "$tmp_dir/before.json" submittedItemRevisionStatus "$EXPECTED_VERSION" \
    || fail "staged revision must contain version $EXPECTED_VERSION at 100% deployment"

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
  fail "version $EXPECTED_VERSION was not PUBLISHED at 100% before the polling deadline"
fi

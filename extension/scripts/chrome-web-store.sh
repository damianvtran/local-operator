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

request() {
  local method=$1
  local url=$2
  local output=$3
  shift 3
  curl --fail-with-body --silent --show-error \
    -X "$method" \
    -H "Authorization: Bearer $CWS_ACCESS_TOKEN" \
    "$@" \
    -o "$output" \
    "$url"
  jq -e . "$output" >/dev/null || fail "API returned a non-JSON response from $url"
}

fetch_status() {
  request GET "$status_url" "$1"
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
  request POST "$publish_url" "$output" \
    -H 'Content-Type: application/json' \
    --data '{"publishType":"STAGED_PUBLISH","deployInfos":[{"deployPercentage":100}],"blockOnWarnings":true}'
}

if [[ "$MODE" == "stage" ]]; then
  [[ -n "$ZIP_PATH" && -n "$EXPECTED_VERSION" ]] \
    || fail "stage requires ZIP and VERSION arguments"
  [[ -f "$ZIP_PATH" ]] || fail "archive not found: $ZIP_PATH"
  "$(dirname "$0")/validate-store-zip.sh" "$ZIP_PATH" "$EXPECTED_VERSION"

  upload_url="$API_ROOT/upload/v2/$item:upload"
  request POST "$upload_url" "$tmp_dir/upload.json" \
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

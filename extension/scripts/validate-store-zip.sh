#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH=${1:-local-operator-extension.zip}
EXPECTED_VERSION=${2:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTENTS_FILE="$ROOT/store-package-files.txt"

fail() {
  printf 'store package validation failed: %s\n' "$*" >&2
  exit 1
}

for command in jq unzip; do
  command -v "$command" >/dev/null || fail "$command is required"
done
[[ -f "$ZIP_PATH" ]] || fail "archive not found: $ZIP_PATH"
[[ -f "$CONTENTS_FILE" ]] || fail "content allowlist not found: $CONTENTS_FILE"

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

# The allowlist makes package growth an explicit review decision. Merely excluding
# source maps would still allow fixtures, credentials, or developer-only files to
# hitch a ride when the build layout changes.
LC_ALL=C sort "$CONTENTS_FILE" > "$tmp_dir/expected"
unzip -Z1 "$ZIP_PATH" \
  | sed '/\/$/d' \
  | LC_ALL=C sort > "$tmp_dir/archive"

[[ -s "$tmp_dir/archive" ]] || fail "archive contains no files"
if [[ $(uniq -d "$tmp_dir/archive" | wc -l | tr -d ' ') != 0 ]]; then
  fail "archive contains duplicate paths"
fi
if grep -Eq '(^/|(^|/)\.\.(/|$)|\\)' "$tmp_dir/archive"; then
  fail "archive contains an unsafe path"
fi
if grep -Eq '\.map$' "$tmp_dir/archive"; then
  fail "source maps are forbidden"
fi
if ! diff -u "$tmp_dir/expected" "$tmp_dir/archive"; then
  fail "archive contents differ from the reviewed allowlist"
fi

# A local build also checks dist so a stale allowlist cannot validate a zip that
# silently omitted a newly generated runtime file.
if [[ -d "$ROOT/dist" ]]; then
  (
    cd "$ROOT/dist"
    find . -type f -print | sed 's#^\./##' | LC_ALL=C sort
  ) > "$tmp_dir/dist"
  if ! diff -u "$tmp_dir/expected" "$tmp_dir/dist"; then
    fail "dist contents differ from the reviewed allowlist"
  fi
fi

unzip -p "$ZIP_PATH" manifest.json > "$tmp_dir/manifest.json"
jq -e . "$tmp_dir/manifest.json" >/dev/null || fail "manifest.json in archive is invalid JSON"
manifest_version=$(jq -r '.version // empty' "$tmp_dir/manifest.json")
source_version=$(jq -r '.version // empty' "$ROOT/manifest.json")
package_version=$(jq -r '.version // empty' "$ROOT/package.json")

[[ -n "$manifest_version" ]] || fail "archive manifest has no version"
[[ "$manifest_version" == "$source_version" ]] \
  || fail "archive manifest version $manifest_version does not match source manifest $source_version"
[[ "$manifest_version" == "$package_version" ]] \
  || fail "manifest version $manifest_version does not match package version $package_version"
if [[ -n "$EXPECTED_VERSION" && "$manifest_version" != "$EXPECTED_VERSION" ]]; then
  fail "archive version $manifest_version does not match expected version $EXPECTED_VERSION"
fi

printf 'validated Chrome Web Store package v%s (%s files, no source maps)\n' \
  "$manifest_version" "$(wc -l < "$tmp_dir/archive" | tr -d ' ')"

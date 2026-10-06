#!/usr/bin/env bash
# Served-asset checks for #2017: what the fixture actually serves for the built SPA.
#
# WHY THIS IS A SCRIPT AND NOT A PASTED COMMAND: the round-1 scan was captured
# against an earlier build and silently passed as the reviewed head's (round-2
# review). The file named no build of its own, so nothing in it could contradict
# the claim. This one prints the SERVED ASSET CONTENT HASHES in its header — read
# from the SPA's own <link>/<script> — so the scan proves which bundle it read.
#
# Usage:
#   LOP_MOBILE_FIXTURE_PASSWORD=<pw> ./served_assets_check.sh <port> <outfile> [label] [source-dir]
#
# The password comes from the environment, never argv, so it stays out of `ps`.
set -euo pipefail

PORT=${1:?usage: served_assets_check.sh <port> <outfile> [label] [source-dir]}
OUT=${2:?usage: served_assets_check.sh <port> <outfile> [label] [source-dir]}
LABEL=${3:-}
SRC=${4:-}
: "${LOP_MOBILE_FIXTURE_PASSWORD:?set LOP_MOBILE_FIXTURE_PASSWORD}"

BASE="http://127.0.0.1:${PORT}"
JAR=$(mktemp)
trap 'rm -f "$JAR"' EXIT

curl -s -c "$JAR" --data-urlencode "password=$LOP_MOBILE_FIXTURE_PASSWORD" -o /dev/null "$BASE/login"
HTML=$(curl -s -b "$JAR" "$BASE/")
CSS_NAME=$(printf '%s' "$HTML" | grep -o 'assets/index-[A-Za-z0-9_-]*\.css' | head -1 || true)
JS_NAME=$(printf '%s' "$HTML" | grep -o 'assets/index-[A-Za-z0-9_-]*\.js' | head -1 || true)
CSS=$(curl -s -b "$JAR" "$BASE/$CSS_NAME")
JS_BODY=$(curl -s -b "$JAR" "$BASE/$JS_NAME")
LOGIN=$(curl -s "$BASE/login")

# Count occurrences of an ERE in a string; 0 when absent.
count() { printf '%s' "$2" | grep -Eo "$1" | wc -l | tr -d ' '; }

{
  echo "# Served-asset checks for #2017 ${LABEL}"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "# served build (content hashes read from the SPA's own <link>/<script>):"
  echo "#   css: ${CSS_NAME:-<none>}"
  echo "#   js:  ${JS_NAME:-<none>}"
  echo
  echo "## SPA meta (dist/index.html, served at /)"
  printf '%s' "$HTML" | grep -A2 'name="viewport"' || echo "(no viewport meta found)"
  echo
  echo "## Tokens in served HTML (login page + authed SPA)  [maximum-scale|user-scalable|text-size-adjust]"
  echo "login page:   $(count 'maximum-scale|user-scalable|text-size-adjust' "$LOGIN")"
  echo "authed SPA /: $(count 'maximum-scale|user-scalable|text-size-adjust' "$HTML")"
  echo
  echo "## Tokens in served CSS (${CSS_NAME:-<none>})"
  echo "maximum-scale:            $(count 'maximum-scale' "$CSS")"
  echo "user-scalable:            $(count 'user-scalable' "$CSS")"
  echo "-webkit-text-size-adjust: $(count '\-webkit-text-size-adjust' "$CSS")  (100% is preflight's inert default, not a suppression)"
  echo "touch-action:             $(count 'touch-action' "$CSS")"
  echo
  echo "## Focus-zoom rules present in served CSS (exact built text)"
  echo "default floor:  $(count 'font-size:max\(16px,1em\)' "$CSS") x  font-size:max(16px,1em)"
  echo "wide rule:      $(count 'font-size:max\(calc\(16px / var\(--lo-fit-scale\)\),1em\)' "$CSS") x  font-size:max(calc(16px / var(--lo-fit-scale)),1em)"
  echo "bare wide rule (pre-guard, must be 0): $(count 'font-size:calc\(16px / var\(--lo-fit-scale\)\)' "$CSS")"
  echo "var published by the bundle: $(count 'lo-fit-scale' "$JS_BODY") reference(s) in ${JS_NAME:-<none>}"
  echo
  echo "## Tokens in served JS bundle (${JS_NAME:-<none>})"
  echo "maximum-scale|user-scalable: $(count 'maximum-scale|user-scalable' "$JS_BODY")"
  echo
  echo "## App source tokens ${SRC:+($SRC)}"
  if [ -n "$SRC" ]; then
    grep -rn "maximum-scale\|user-scalable" "$SRC" || echo "(none — the fix adds neither)"
  else
    echo "(source dir not given)"
  fi
} > "$OUT"

echo "wrote $OUT (css=${CSS_NAME:-none}, js=${JS_NAME:-none})"

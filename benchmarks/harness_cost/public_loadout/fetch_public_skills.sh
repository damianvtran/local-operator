#!/bin/bash
# fetch_public_skills.sh -- fetch the public skill set for the harness-cost
# benchmark's full-loadout arm, at a pinned commit, and verify it.
#
# WHY
#   The full-loadout arm puts a realistic skill catalog in every harness's
#   skill roots, because each harness pays for skills differently (some inline
#   every description in the system prompt, some list names only, some select).
#   The catalog is github.com/anthropics/skills at a fixed commit. Most of those
#   skills are Apache-2.0, but docx/pptx/xlsx/pdf are source-available under
#   Anthropic's proprietary terms (see each skill's LICENSE.txt), so this kit
#   does NOT redistribute any skill file: it fetches them from upstream instead.
#   One fetch path for every skill keeps the setup simple and the licence
#   question out of the kit.
#
# WHAT
#   Fetches exactly commit $PIN (shallow) into $DEST, then verifies every
#   skills/*/SKILL.md against public_skills.sha256 and that upstream has no
#   SKILL.md the manifest does not list. Any mismatch exits non-zero.
#   Idempotent: an existing clone at $PIN is only re-verified.
#
# CONSTRAINTS
#   * Anonymous HTTPS only. git's system config (which may carry a keychain
#     credential helper) is bypassed and prompts are disabled, so a scratch or
#     CI environment never raises a credential dialog.
#   * Never modifies an existing clone that is at another commit or dirty; it
#     refuses instead, so nobody's working copy is overwritten.
#   * Only the SKILL.md files are hash-pinned (that is what harnesses put in
#     the prompt). Bundled scripts/assets ride on the commit SHA.
#
# USAGE
#   ./fetch_public_skills.sh                 # -> kit-src/anthropic-skills
#   SKILLS_DEST=/some/dir ./fetch_public_skills.sh
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
PIN=ca45fc496a3436251f191354091d557da31c15b5
REPO=https://github.com/anthropics/skills.git
DEST=${SKILLS_DEST:-$KIT/kit-src/anthropic-skills}
MANIFEST=$KIT/public_skills.sha256
export GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false

if [ -d "$DEST/.git" ]; then
  head=$(git -C "$DEST" rev-parse HEAD)
  [ "$head" = "$PIN" ] || { echo "refusing: $DEST is at $head, not $PIN" >&2; exit 2; }
  [ -z "$(git -C "$DEST" status --porcelain)" ] || { echo "refusing: $DEST has local changes" >&2; exit 2; }
else
  [ ! -e "$DEST" ] || { echo "refusing: $DEST exists and is not a git clone" >&2; exit 2; }
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$REPO"
  git -C "$DEST" -c credential.helper= fetch -q --depth 1 origin "$PIN"
  git -C "$DEST" -c advice.detachedHead=false checkout -q FETCH_HEAD
  [ "$(git -C "$DEST" rev-parse HEAD)" = "$PIN" ] || { echo "fetched wrong commit" >&2; exit 3; }
fi

cd "$DEST"
grep -v '^#' "$MANIFEST" | shasum -a 256 -c --quiet || { echo "SKILL.md manifest mismatch" >&2; exit 4; }
listed=$(grep -v '^#' "$MANIFEST" | awk '{print $2}' | sort)
present=$(git ls-files 'skills/*/SKILL.md' | sort)
if [ "$listed" != "$present" ]; then
  echo "upstream SKILL.md set differs from manifest:" >&2
  diff <(echo "$listed") <(echo "$present") >&2 || true
  exit 4
fi
echo "ok: $(echo "$present" | wc -l | tr -d ' ') skills verified at $PIN in $DEST"

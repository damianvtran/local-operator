#!/bin/sh
# REPOSITORY-OWNED SOURCE for the stock shim; installation is separately reviewed.
# Read the COMMITTED seed, never dirty checkout Python, and bootstrap a positively
# published capable controller before invoking the immutable destination route.
# This is not a drain/force wrapper: old daemons may remain serving/pending.
set -eu
REF=${1:-main}
case "$REF" in
  -h|--help) printf '%s\n' 'usage: lop-update [committed-ref]' 'Prepare a verified controller and immutable core; serving services may remain pending.'; exit 0 ;;
  -*) printf '%s\n' 'lop-update: unsupported option; no force/downgrade bypass' >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || shift
[ "$#" -eq 0 ] || { printf '%s\n' 'lop-update: unexpected arguments' >&2; exit 2; }
REPO=${LOCAL_OPERATOR_REPO:-"$HOME/local-operator"}
ROOT="$HOME/.local/share/lop"
PYTHON="$ROOT/current/tools/local-operator/bin/python"
[ -d "$REPO/.git" ] && [ -x "$PYTHON" ] || {
  printf '%s\n' 'lop-update: a committed repository and canonical immutable core are required' >&2; exit 1;
}
BASELINE=$(cd "$ROOT/current" && pwd -P)
SEED=$(mktemp "${TMPDIR:-/tmp}/lop-update-seed.XXXXXXXX")
# Only our creator-owned seed file is removed; no installed/candidate tree is
# reclaimed here. The existing bounded worker runner reaps bootstrap children.
trap 'rm -f "$SEED"' EXIT HUP INT TERM
GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false \
  git -C "$REPO" show "$REF:local_operator/update_bootstrap.py" > "$SEED" || {
    printf '%s\n' 'lop-update: this committed source does not carry the capable controller seed' >&2; exit 1;
}
cd "$REPO"
env -u XPC_FLAGS PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B -P "$SEED" \
  --root "$ROOT" --baseline "$BASELINE" --caller rolling-no-kill-v1 --snapshot "$REF"

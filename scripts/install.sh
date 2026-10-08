#!/bin/sh
# One-line installer for Local Operator (macOS and Linux).
#
#   curl -LsSf https://raw.githubusercontent.com/damianvtran/local-operator/main/scripts/install.sh | sh
#
# WHY THIS EXISTS (first-run onboarding audit Q6/U8/D9). The README used to
# lead with `pip install local-operator`, which fails on the two most common
# first-run machines: a system Python older than 3.12 ("from versions: none")
# and an externally managed Python (Debian/Ubuntu/Homebrew, PEP 668). uv
# fetches its own Python and installs the CLI into an isolated tool
# environment, so this script works on a machine with no Python at all, and it
# is FAST: measured on an empty uv cache in the PR that added it (see the PR
# body for the number on that run).
#
# WHAT IT DOES, in three announced steps, each with an ETA and the elapsed
# total, so a person watching knows it is moving and roughly how long is left:
#   1. Getting ready         — install uv into ~/.local/bin if it is missing
#   2. Installing Local Operator — `uv tool install local-operator` (uv brings Python 3.12+)
#   3. Checking it works     — run `lop --version` (a cold first start)
#
# CONSTRAINTS:
# - POSIX sh, no bashisms: `curl … | sh` runs under dash on Debian/Ubuntu.
# - Never sudo, never touches system Python, never edits shell rc files
#   itself: uv's own installer and `uv tool update-shell` are the documented
#   PATH writers, and this script prints the one command when it is needed.
# - Idempotent: re-running upgrades in place (`--upgrade`).
# - Overridable for tests and pinned installs:
#     LOCAL_OPERATOR_PACKAGE   the requirement to install (default: local-operator)
#     LOCAL_OPERATOR_PYTHON    the Python version uv should use (default: 3.12)
#     NO_COLOR                 plain output

set -eu

PACKAGE="${LOCAL_OPERATOR_PACKAGE:-local-operator}"
PYTHON_VERSION="${LOCAL_OPERATOR_PYTHON:-3.12}"
TOTAL_STEPS=3

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD="$(printf '\033[1m')"; DIM="$(printf '\033[2m')"; GREEN="$(printf '\033[32m')"
    RED="$(printf '\033[31m')"; RESET="$(printf '\033[0m')"
else
    BOLD=""; DIM=""; GREEN=""; RED=""; RESET=""
fi

START="$(date +%s)"

elapsed() {
    now="$(date +%s)"
    printf '%ss' "$((now - START))"
}

# step <n> <label> <eta>: one line per phase. The ETA is a typical figure for a
# fresh machine on a home connection, stated as "about", never as a promise.
step() {
    printf '%sStep %s/%s%s · %s %s(about %s · %s elapsed)%s\n' \
        "$BOLD" "$1" "$TOTAL_STEPS" "$RESET" "$2" "$DIM" "$3" "$(elapsed)" "$RESET"
}

fail() {
    printf '%sInstall failed:%s %s\n' "$RED" "$RESET" "$1" >&2
    printf 'Elapsed %s. Re-run the same command to retry; it picks up where it stopped.\n' "$(elapsed)" >&2
    exit 1
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

printf '%sInstalling Local Operator%s\n' "$BOLD" "$RESET"

# -- 1. uv -----------------------------------------------------------------
if UV="$(find_uv)"; then
    step 1 "Getting ready — uv is already installed" "0 s"
else
    step 1 "Getting ready — installing uv" "5 s"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null \
            || fail "could not install uv (https://docs.astral.sh/uv/getting-started/installation/)"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null \
            || fail "could not install uv (https://docs.astral.sh/uv/getting-started/installation/)"
    else
        fail "neither curl nor wget is available to download uv"
    fi
    UV="$(find_uv)" || fail "uv was installed but cannot be found in ~/.local/bin"
fi

# -- 2. the CLI ----------------------------------------------------------------
step 2 "Installing Local Operator (uv brings Python ${PYTHON_VERSION}+)" "15 s"
"$UV" tool install --upgrade --quiet --python "$PYTHON_VERSION" "$PACKAGE" \
    || fail "uv could not install ${PACKAGE}"

# -- 3. check ------------------------------------------------------------------
step 3 "Checking it works" "5 s"
BIN_DIR="$("$UV" tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
LOP="$BIN_DIR/lop"
[ -x "$LOP" ] || fail "the install finished but $LOP is missing"
VERSION="$("$LOP" --version 2>/dev/null | tail -n 1)" || fail "lop is installed but did not start"

printf '\n%s✓ Local Operator %s installed in %s%s\n' "$GREEN" "$VERSION" "$(elapsed)" "$RESET"

case ":$PATH:" in
    *":$BIN_DIR:"*) ON_PATH=1 ;;
    *) ON_PATH=0 ;;
esac
if [ "$ON_PATH" -eq 0 ]; then
    printf '\n%s is not on your PATH yet. Add it with:\n  %s tool update-shell\nthen open a new terminal.\n' "$BIN_DIR" "$UV"
fi

printf '\nNext:\n  lop                 start it\n  /login radient      inside it: one browser sign-in (recommended),\n                      or /login for any other provider or an API key\n'

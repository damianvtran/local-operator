#!/bin/sh
# term_guard.sh <unique-title> — assert the frontmost WINDOW is OURS.
#
# Exits 0 only when the frontmost process is Terminal AND its front window's
# accessibility title CONTAINS our generated unique token. Any other outcome is
# a hard failure and the caller must NOT send keys.
#
# Why a token match and not equality: Terminal decorates the window title
# ("damian — <custom title> — python3 ◂ keylog_app.py — 120×30"), so equality
# can never hold. The token is generated per run (LOPCHK-<epoch>) and is not a
# substring of any other window's title, so this stays unambiguous.
#
# Why this exists at all: `activate` raises an APP, not a window. An earlier
# revision asserted only the process name, passed while ANOTHER session's
# window was frontmost, and typed a gesture into a live session, clearing its
# draft. Never send a keystroke unless this has just exited 0.
TOKEN="$1"
FRONT=$(osascript <<EOF 2>/dev/null
tell application "System Events"
  set p to first process whose frontmost is true
  if name of p is not "Terminal" then return "PROC:" & (name of p)
  try
    return "WIN:" & (value of attribute "AXTitle" of front window of p)
  on error
    return "WIN:<none>"
  end try
end tell
EOF
)
case "$FRONT" in
  WIN:*"$TOKEN"*) echo "OK $FRONT"; exit 0 ;;
  *) echo "ABORT (want a window containing '$TOKEN', got '$FRONT')"; exit 9 ;;
esac

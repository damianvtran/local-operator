#!/bin/sh
# drive_tui.sh <ghostty|terminal> <logfile> <gesture-script.applescript> [shot]
#
# SAFETY: keystrokes go to whatever is frontmost, so this ASSERTS the frontmost
# process is the terminal we launched and aborts otherwise. Without that check a
# gesture once landed in an unrelated window.
#
# The gesture is a FILE, not an inline string: `tell application "System Events"
# to <multi-line body>` parses but silently runs only its first statement, which
# looked exactly like "the keys were delivered and the app ignored them".
TERMAPP="$1"; LOG="$2"; GESTURE="$3"; SHOT="$4"
rm -f "$LOG"
GB=/Applications/Ghostty.app/Contents/MacOS/ghostty
CMD="cd /tmp/lop-cmd && LO_KEYLOG=$LOG ./.venv/bin/python /tmp/kprobe2/keylog_app.py"
if [ "$TERMAPP" = ghostty ]; then
  nohup "$GB" -e /bin/sh -c "$CMD" >/dev/null 2>&1 &
  APP="Ghostty"
else
  osascript -e "tell application \"Terminal\" to do script \"$CMD\"" >/dev/null 2>&1
  APP="Terminal"
fi
sleep 15
osascript -e "tell application \"$APP\" to activate" >/dev/null 2>&1
sleep 2.5
FRONT=$(osascript -e 'tell application "System Events" to return name of (first process whose frontmost is true)' 2>/dev/null)
case "$FRONT" in
  *[Gg]hostty*|*Terminal*) : ;;
  *) echo "ABORT: frontmost is '$FRONT', not $APP"; pkill -f keylog_app.py; exit 3 ;;
esac
echo "frontmost=$FRONT (ok)"
osascript "$GESTURE" >/dev/null 2>&1 || echo "gesture script FAILED"
sleep 3
[ -n "$SHOT" ] && screencapture -x -o "$SHOT" 2>/dev/null
sleep 1

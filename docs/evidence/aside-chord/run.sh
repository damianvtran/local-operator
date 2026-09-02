#!/bin/sh
# run.sh <terminal> <outfile> <applescript-keystroke-line> <label> [kitty_flags]
#
# Launches probe.py in a NEW window of <terminal>, activates that app (the only
# focus method that actually delivers System Events keystrokes here — see
# docs/evidence/cmd-chords/MEASURED.md:15-17), sends ONE keystroke, waits for
# the probe to finish, prints the captured bytes.
#
# <terminal> is one of: ghostty iterm terminal alacritty kitty wezterm
# Each needs its own launch incantation and its own AppleScript application
# name; they are not interchangeable and that is why this is a case statement
# rather than a single parameterised command.
TERM_NAME="$1"; OUT="$2"; KEY="$3"; LABEL="$4"; FLAGS="${5:-25}"
rm -f "$OUT"
DIR=$(cd "$(dirname "$0")" && pwd)
PROBE="$DIR/probe.py.txt"
PY=/usr/bin/python3
CMD="$PY $PROBE $OUT 14 $FLAGS"

case "$TERM_NAME" in
  ghostty)
    APP="Ghostty"
    nohup /Applications/Ghostty.app/Contents/MacOS/ghostty -e /bin/sh -c "$CMD" >/dev/null 2>&1 &
    ;;
  iterm)
    APP="iTerm"
    osascript -e "tell application \"iTerm\"
      create window with default profile
      tell current session of current window to write text \"$CMD; exit\"
    end tell" >/dev/null 2>&1
    ;;
  terminal)
    APP="Terminal"
    osascript -e "tell application \"Terminal\" to do script \"$CMD; exit\"" >/dev/null 2>&1
    ;;
  alacritty)
    APP="Alacritty"
    nohup /opt/homebrew/bin/alacritty -e /bin/sh -c "$CMD" >/dev/null 2>&1 &
    ;;
  kitty)
    APP="kitty"
    nohup /Applications/kitty.app/Contents/MacOS/kitty /bin/sh -c "$CMD" >/dev/null 2>&1 &
    ;;
  wezterm)
    APP="WezTerm"
    nohup /opt/homebrew/bin/wezterm start -- /bin/sh -c "$CMD" >/dev/null 2>&1 &
    ;;
  *) echo "unknown terminal: $TERM_NAME"; exit 2 ;;
esac

sleep 6
osascript -e "tell application \"$APP\" to activate" >/dev/null 2>&1
sleep 1.5
osascript -e "tell application \"System Events\" to $KEY" >/dev/null 2>&1
sleep 11
/usr/bin/python3 -c "
import os
p='$OUT'
d=open(p,'rb').read() if os.path.exists(p) else b'<MISSING>'
print('%-14s %-26s %3d bytes  %r' % ('$TERM_NAME', '$LABEL', len(d), d[:90]))"

#!/bin/sh
# run.sh <outfile> <applescript-keystroke-line> <label>
# Launches probe2 in a NEW Terminal.app window, activates Terminal (the only
# focus method that actually delivers System Events keystrokes here), sends the
# keystroke, waits for the probe to finish, prints the captured bytes.
OUT="$1"; KEY="$2"; LABEL="$3"
rm -f "$OUT"
WID=$(osascript -e 'tell application "Terminal"
  do script "python3 /tmp/kprobe2/probe2.py '"$OUT"' 13"
  return id of window 1
end tell')
sleep 5
osascript -e 'tell application "Terminal" to activate' >/dev/null 2>&1
sleep 1
osascript -e "tell application \"System Events\" to $KEY" >/dev/null 2>&1
sleep 11
python3 -c "
import os
p='$OUT'
d=open(p,'rb').read() if os.path.exists(p) else b'<MISSING>'
print('%-34s %3d bytes  %r' % ('$LABEL', len(d), d[:90]))"
osascript -e "tell application \"Terminal\" to close window id $WID saving no" >/dev/null 2>&1

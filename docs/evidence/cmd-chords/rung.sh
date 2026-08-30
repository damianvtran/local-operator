#!/bin/sh
# rung.sh <outfile> <applescript-keystroke-line> <label> [extra ghostty args...]
# Same as run.sh but drives Ghostty. Launches a dedicated Ghostty instance
# running the probe, activates it, sends one keystroke, reports captured bytes.
OUT="$1"; KEY="$2"; LABEL="$3"; shift 3
GB=/Applications/Ghostty.app/Contents/MacOS/ghostty
rm -f "$OUT"
nohup "$GB" "$@" -e /bin/sh -c "python3 /tmp/kprobe2/probe2.py $OUT 14" >/dev/null 2>&1 &
sleep 6
osascript -e 'tell application "Ghostty" to activate' >/dev/null 2>&1
sleep 1.5
osascript -e "tell application \"System Events\" to $KEY" >/dev/null 2>&1
sleep 11
python3 -c "
import os
p='$OUT'
d=open(p,'rb').read() if os.path.exists(p) else b'<MISSING>'
print('%-40s %3d bytes  %r' % ('$LABEL', len(d), d[:90]))"

#!/bin/sh
# Real Linux clipboard backends, executed against real clipboard tooling.
# X11: xclip under Xvfb. Wayland: wl-paste under a headless sway compositor.
#
# Reproduce with:
#   docker run --rm \
#     -v "$PWD/local_operator:/work/local_operator:ro" \
#     -v "$PWD/docs/evidence/clipboard-paste/linux_evidence.sh:/work/run.sh:ro" \
#     debian:bookworm-slim sh /work/run.sh
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends \
    python3 xclip xvfb wl-clipboard sway >/dev/null 2>&1 || \
apt-get install -y -qq --no-install-recommends python3 xclip xvfb wl-clipboard >/dev/null

# A real PNG, built with the stdlib so the container needs no imaging library.
python3 - <<'PY'
import zlib, struct
def chunk(kind, payload):
    body = kind + payload
    return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)
w, h = 320, 200
rows = b''
for y in range(h):
    rows += b'\x00' + bytes([v for x in range(w) for v in ((x * 7) % 256, (y * 5) % 256, 90)])
png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(rows))
       + chunk(b'IEND', b''))
open('/tmp/evidence.png', 'wb').write(png)
print('made /tmp/evidence.png', len(png), 'bytes', w, 'x', h)
PY

# Only clipboard.py and its one intra-package import (media.py) are needed, so
# the backends run without installing the whole project in the container.
mkdir -p /tmp/pkg/local_operator
cp /work/local_operator/clipboard.py /work/local_operator/media.py /tmp/pkg/local_operator/
touch /tmp/pkg/local_operator/__init__.py

echo
echo "=================== X11 (xclip under Xvfb) ==================="
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &
sleep 2
echo "\$DISPLAY=$DISPLAY  xclip=$(command -v xclip)"
xclip -selection clipboard -t image/png -i /tmp/evidence.png
sleep 1
echo "--- xclip TARGETS on the clipboard ---"
xclip -selection clipboard -t TARGETS -o
echo "--- backend run ---"
cd /tmp/pkg && python3 -c "
from local_operator import clipboard
import hashlib
src = open('/tmp/evidence.png','rb').read()
img = clipboard.read_clipboard(4*1024*1024, platform='linux', env={'DISPLAY': ':99'}).image
print('result:', None if img is None else (img.mime_type, len(img.data)))
print('bytes identical to source:', img is not None and img.data == src)
print('source sha256:', hashlib.sha256(src).hexdigest()[:16])
print('clip   sha256:', 'n/a' if img is None else hashlib.sha256(img.data).hexdigest()[:16])
"
echo "--- negative: text-only clipboard (xclip answers image/png with the TEXT) ---"
printf 'just text, no image' | xclip -selection clipboard -i
sleep 1
printf 'raw xclip -t image/png -o gives: '
xclip -selection clipboard -t image/png -o; echo " (exit $?)"
cd /tmp/pkg && python3 -c "
from local_operator import clipboard
print('backend result:', clipboard.read_clipboard(4*1024*1024, platform='linux', env={'DISPLAY': ':99'}).image)
"
echo "--- negative: size cap (bound below the payload) ---"
xclip -selection clipboard -t image/png -i /tmp/evidence.png
sleep 1
cd /tmp/pkg && python3 -c "
from local_operator import clipboard
print('with max_bytes=1024:', clipboard.read_clipboard(1024, platform='linux', env={'DISPLAY': ':99'}).image)
"
echo "--- negative: SSH skip (clipboard is present and full) ---"
cd /tmp/pkg && python3 -c "
from local_operator import clipboard
env = {'DISPLAY': ':99', 'SSH_CONNECTION': '10.0.0.1 22 10.0.0.2 22'}
print('reads are local:', clipboard.clipboard_reads_are_local(env))
print('image over ssh:', clipboard.read_clipboard(4*1024*1024, platform='linux', env=env).image)
"
echo "--- negative: no DISPLAY, no WAYLAND_DISPLAY (headless) ---"
cd /tmp/pkg && python3 -c "
from local_operator import clipboard
print('headless:', clipboard.read_clipboard(4*1024*1024, platform='linux', env={}).image)
"

echo
echo "=================== Wayland (wl-paste) ==================="
if ! command -v sway >/dev/null 2>&1; then
    echo "sway unavailable in this image; skipping compositor run"
    exit 0
fi
export XDG_RUNTIME_DIR=/tmp/xdg
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
# sway refuses to run as root (it cannot drop privileges irreversibly), so the
# compositor and everything talking to it run as an unprivileged user.
id -u wl >/dev/null 2>&1 || useradd -m wl
chown -R wl "$XDG_RUNTIME_DIR" /tmp/evidence.png /tmp/pkg
su wl -c "env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 WLR_RENDERER=pixman sway --config /dev/null" >/tmp/sway.log 2>&1 &
sleep 5
WAYLAND_DISPLAY=$(ls "$XDG_RUNTIME_DIR" | grep -E '^wayland-[0-9]+$' | head -1 || true)
if [ -z "$WAYLAND_DISPLAY" ]; then
    echo "no wayland socket appeared; sway log:"; tail -20 /tmp/sway.log; exit 0
fi
export WAYLAND_DISPLAY
echo "\$WAYLAND_DISPLAY=$WAYLAND_DISPLAY  wl-paste=$(command -v wl-paste)"
WL="su wl -c"
$WL "env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY wl-copy --type image/png < /tmp/evidence.png"
sleep 1
echo "--- wl-paste --list-types ---"
$WL "env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY wl-paste --list-types"
echo "--- backend run ---"
$WL "cd /tmp/pkg && env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY python3 -c \"
from local_operator import clipboard
import os
src = open('/tmp/evidence.png','rb').read()
env = {'WAYLAND_DISPLAY': os.environ['WAYLAND_DISPLAY'], 'XDG_RUNTIME_DIR': os.environ['XDG_RUNTIME_DIR']}
img = clipboard.read_clipboard(4*1024*1024, platform='linux', env=env).image
print('result:', None if img is None else (img.mime_type, len(img.data)))
print('bytes identical to source:', img is not None and img.data == src)
\""
echo "--- negative: text-only wayland clipboard ---"
$WL "env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY sh -c 'printf \"just text\" | wl-copy'"
sleep 1
$WL "env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY wl-paste --list-types"
$WL "cd /tmp/pkg && env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY python3 -c \"
from local_operator import clipboard
import os
env = {'WAYLAND_DISPLAY': os.environ['WAYLAND_DISPLAY'], 'XDG_RUNTIME_DIR': os.environ['XDG_RUNTIME_DIR']}
print('backend result:', clipboard.read_clipboard(4*1024*1024, platform='linux', env=env).image)
\""
echo "--- wayland is preferred over X11 when both are set ---"
echo "(wayland clipboard holds TEXT; the X11 clipboard still holds the PNG,"
echo " so a None here proves wl-paste was chosen and xclip was not)"
$WL "cd /tmp/pkg && env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY python3 -c \"
from local_operator import clipboard
import os
env = {'WAYLAND_DISPLAY': os.environ['WAYLAND_DISPLAY'], 'XDG_RUNTIME_DIR': os.environ['XDG_RUNTIME_DIR'], 'DISPLAY': ':99'}
print('backend result:', clipboard.read_clipboard(4*1024*1024, platform='linux', env=env).image)
\""
echo "--- missing tooling: wl-paste and xclip not on PATH ---"
mkdir -p /tmp/emptybin && ln -sf "$(command -v python3)" /tmp/emptybin/python3
chown -R wl /tmp/emptybin
$WL "cd /tmp/pkg && env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY DISPLAY=:99 PATH=/tmp/emptybin python3 -c \"
from local_operator import clipboard
import os, shutil
print('wl-paste on PATH:', bool(shutil.which('wl-paste')))
print('xclip on PATH:   ', bool(shutil.which('xclip')))
env = {'WAYLAND_DISPLAY': os.environ['WAYLAND_DISPLAY'], 'XDG_RUNTIME_DIR': os.environ['XDG_RUNTIME_DIR']}
print('wayland backend result:', clipboard.read_clipboard(4*1024*1024, platform='linux', env=env).image)
print('x11 backend result:    ', clipboard.read_clipboard(4*1024*1024, platform='linux', env={'DISPLAY': ':99'}).image)
\""

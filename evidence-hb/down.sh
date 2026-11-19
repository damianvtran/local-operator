#!/bin/bash
S=/Users/damian/.local-operator/sessions/3d299fb1b279/scratchpad/hb
for f in serve hub; do p=$(cat $S/$f.pid 2>/dev/null) || continue; kill -- -$p 2>/dev/null; done
sleep 3
for f in serve hub; do p=$(cat $S/$f.pid 2>/dev/null) || continue; kill -0 $p 2>/dev/null && echo "$f $p ALIVE" || echo "$f $p reaped"; done

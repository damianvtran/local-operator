"""Per-session voice mode (docs/design/qwen-voice.md).

The operator talks hands-free to ONE session at a time through the desktop
app's Qwen rig; that session's own agent answers with its own context and
tools, and the server returns the text of its reply for the rig to speak word
for word (Q1-Q4). No audio is handled here (Q10). This package holds the
machine-wide fact of WHICH session that is (:mod:`local_operator.voice.slot`,
V3) and the rule for turning agent text into something worth speaking
(:mod:`local_operator.voice.speakable`, V8).

Import-light on purpose: the server routes and the machine-wide feed read the
slot on their poll cadence, so nothing heavier than the standard library and
the wake lock's fd discipline lives here.

V7 IS A HARD RULE AND THIS PACKAGE IS WHERE IT STARTS: nothing here answers a
gate, and nothing here may grow a path that does. Voice ANNOUNCES an approval;
the operator approves it on screen through the existing presence-gated card.
"""

from __future__ import annotations

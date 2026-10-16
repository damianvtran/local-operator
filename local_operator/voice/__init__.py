"""Per-session voice mode (docs/design/session-voice.md).

The operator talks to ONE session at a time through push-to-talk; that
session's own agent answers with its own context and tools, and the server
synthesises the agent's final message for the renderer to play (V1). This
package holds the machine-wide fact of WHICH session that is
(:mod:`local_operator.voice.slot`, V3) and the rule for turning agent text
into something worth speaking (:mod:`local_operator.voice.speakable`, V8).

Import-light on purpose: the server routes and the machine-wide feed read the
slot on their poll cadence, so nothing heavier than the standard library and
the wake lock's fd discipline lives here.

V7 IS A HARD RULE AND THIS PACKAGE IS WHERE IT STARTS: nothing here answers a
gate, and nothing here may grow a path that does. Voice ANNOUNCES an approval;
the operator approves it on screen through the existing presence-gated card.
"""

from __future__ import annotations

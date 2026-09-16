"""Discovery and transport for the Local Operator desktop app's browser host.

The app is a THIRD browsable host beside cmux and the Chromium extension. It
speaks the extension bridge's existing session leg — same `Request`/`Response`
envelopes, same method names, same typed errors — so Python needs no new protocol
and no new dependency: only a second discovery file, a second client, and one
place (``local_operator/tools/builtin.py``) that decides which host serves a
given action.

Two modules, deliberately split the way the bridge splits its own:

* :mod:`.state` — the 0600/0700 discovery record and the file-only predicate.
* :mod:`.backend` — the session-side client and the four availability answers
  (available / advertisable / reachable / liveness) every host must supply, or
  the tool's semantics break for that host.
"""

from __future__ import annotations

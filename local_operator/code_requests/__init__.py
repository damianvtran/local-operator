"""Per-session code-request (PR/MR) tracking: detection, ledger, fetch, tool.

The package's job is to answer, for one conversation, *which pull requests and merge
requests did this session touch, and where is each one up to?* It is built in slices.
The first added DETECTION and the durable record of what was detected, plus the pure
parser for the review-round convention. The second (this one) adds the FETCH half: the
forge adapters, the credential resolver, the cache with its TTLs and throttles, the
refresh service that ties them together (with the session's own dirty marks as its
trigger), the ``code_requests`` model tool, and the routes' merge of fetched state.

Modules, and what each owns:

* :mod:`local_operator.code_requests.refs` — parse a URL or qualified ref into a typed
  ``Ref`` and identify its host from PATH SHAPE plus local configuration, never from the
  hostname alone.
* :mod:`local_operator.code_requests.detect` — deterministically classify ONE tool call
  as having opened, acted on, or merely hinted at a code request. Pure; no I/O.
* :mod:`local_operator.code_requests.scan` — derive a session's rows from transcript
  rows (events plus text mentions). Pure; no I/O.
* :mod:`local_operator.code_requests.ledger` — the ``code_request_event.v1`` transcript
  row, the derived index, and the incremental scan cache. The only pre-fetch module here
  that writes.
* :mod:`local_operator.code_requests.rounds` — parse ``### Agent review — round N`` and
  its siblings out of comment bodies into lanes, rounds, verdicts and freshness. Pure.
* :mod:`local_operator.code_requests.adapters` — one module per forge family: parse
  pieces, fetch conditionally, map to ``{open, draft, merged, closed}``; GitHub and
  GitLab are full, everything else is detect-and-link.
* :mod:`local_operator.code_requests.credentials` — the operator's own CLI logins
  (``gh auth token`` / ``glab config get token``), held in memory, never printed.
* :mod:`local_operator.code_requests.cache` — the per-host memory LRU, the disk tier
  with its validators, the TTL/backoff/cooling clocks and the dirty marks. Stdlib-only,
  because the live hook path imports it.
* :mod:`local_operator.code_requests.service` — eligibility, the refresh pass, the
  read-side merge, and the completion signal the feed frame carries.
* :mod:`local_operator.code_requests.tool` — the ``code_requests`` model tool.

The pre-fetch five are stdlib-only by design: the ledger is read from cold paths that
must not pull the session runtime in, and the detectors run inside a post-tool hook
where an import cost is paid on a hot path. ``cache`` extends that rule to itself for
the same reason (the acted-event dirty mark); the network lives in ``adapters`` and
``service``, which only the routes and the tool import.
"""

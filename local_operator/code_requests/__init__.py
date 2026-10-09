"""Per-session code-request (PR/MR) tracking: detection, ledger, and the round parser.

The package's job is to answer, for one conversation, *which pull requests and merge
requests did this session touch, and where is each one up to?* It is built in slices,
and this is the first: DETECTION and the durable record of what was detected, plus the
pure parser for the review-round convention. Nothing here fetches from a forge yet, so
every row this slice produces is link-only; the adapters, cache and refresh service are
the next slice.

Modules, and what each owns:

* :mod:`local_operator.code_requests.refs` — parse a URL or qualified ref into a typed
  ``Ref`` and identify its host from PATH SHAPE plus local configuration, never from the
  hostname alone.
* :mod:`local_operator.code_requests.detect` — deterministically classify ONE tool call
  as having opened, acted on, or merely hinted at a code request. Pure; no I/O.
* :mod:`local_operator.code_requests.scan` — derive a session's rows from transcript
  rows (events plus text mentions). Pure; no I/O.
* :mod:`local_operator.code_requests.ledger` — the ``code_request_event.v1`` transcript
  row, the derived index, and the incremental scan cache. The only module here that
  writes.
* :mod:`local_operator.code_requests.rounds` — parse ``### Agent review — round N`` and
  its siblings out of comment bodies into lanes, rounds, verdicts and freshness. Pure.

The five modules are stdlib-only by design: the ledger is read from cold paths that must
not pull the session runtime in, and the detectors run inside a post-tool hook where an
import cost is paid on a hot path.
"""

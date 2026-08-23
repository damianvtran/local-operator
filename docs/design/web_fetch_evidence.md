# web_fetch — live validation evidence

Real execution proof for the `web_fetch` capability (design `web_fetch.md`).
Commands run from the worktree `/tmp/lop-webfetch` with the shared editable venv,
`LOCAL_OPERATOR_CONFIG_DIR` pointed at a scratch dir per run. The `[fetch]` extra
(markdownify + beautifulsoup4 + soupsieve) was installed into the shared venv so
BOTH render backends could be exercised.

Invocation shape (the console-script wrapper is a stale global install, so the
module entry point is driven directly):

```sh
./.venv/bin/python -c "import sys; sys.argv=['lop']+sys.argv[1:]; \
  from local_operator.cli import main; sys.exit(main())" fetch test <url>
```

## 1. HTML → markdown (markdownify), inline (small page)

```
$ ... fetch test https://example.com
[200] https://example.com
markdownify · text/html · cache miss · sparse/JS-gated (try `browser`)

# Example Domain

This domain is for use in documentation examples without needing permission. Avoid use in operations.

[Learn more](https://iana.org/domains/example)

--- status=200 method=markdownify bytes=559 lines=7 cache=miss spill=(inline)
```

(example.com is legitimately sparse, so the low-quality gate flags it — correct.)

## 2. HTML → markdown + SPILL + chunked read (large docs page)

```
$ ... fetch test https://peps.python.org/pep-0008/
[200] https://peps.python.org/pep-0008/
markdownify · text/html; charset=utf-8 · cache miss
# PEP 8 – Style Guide for Python Code
...
--- status=200 method=markdownify bytes=49590 lines=1492 cache=hit spill=spill://29855223d47035e7c6d46933093582aa
```

Expanding the spill handle through the SAME `read` path any oversized output
uses — a range and a `?q=` search:

```
read spill://29855223d47035e7c6d46933093582aa range=1-3
  spill://29855223d47035e7c6d46933093582aa — lines 1-3 of 1492
  1| PEP 8 – Style Guide for Python Code | peps.python.org
  2|
  3| PEP 8 – Style Guide for Python Code

read spill://29855223d47035e7c6d46933093582aa?q=indentation
  11 of 11 match(es) for 'indentation' in spill://…29855223… (1492 lines):
  122| ### [Indentation](#indentation)
  124| Use 4 spaces per indentation …
```

This proves the central requirement end to end: bounded preview in context, full
page behind a `spill://` handle, chunked/searchable via `read`.

## 3. JSON pretty-print

```
$ ... fetch test https://api.github.com
[200] https://api.github.com
json · application/json; charset=utf-8 · cache miss

{
  "current_user_url": "https://api.github.com/user",
  "current_user_authorizations_html_url": "https://github.com/settings/...",
  ...
```

## 4. Redirect followed + re-validated

```
$ ... fetch test "https://httpbingo.org/redirect-to?url=https://example.com"
[200] https://example.com          # final_url reflects the 302 destination
markdownify · text/html · cache miss · sparse/JS-gated (try `browser`)
```

## 5. Non-2xx (404) leads with the status, not a silent error page

```
$ ... fetch test https://example.com/does-not-exist
[HTTP 404] https://example.com/does-not-exist
markdownify · text/html · cache miss · sparse/JS-gated (try `browser`)
```

## 6. SSRF refusals (the top risk)

Cloud metadata endpoint — refused before any connection:

```
$ ... fetch test http://169.254.169.254/latest/meta-data/
error: refusing 'http://169.254.169.254/latest/meta-data/': host resolves to a
private/loopback/reserved address (169.254.169.254). Set web_fetch.allow_private
to fetch local targets.
```

localhost — refused (default-deny):

```
$ ... fetch test http://localhost:8080/
error: refusing 'http://localhost:8080/': host resolves to a private/loopback/
reserved address (127.0.0.1). Set web_fetch.allow_private to fetch local targets.
```

SSRF-via-redirect (a public URL that 302s to the metadata IP) is refused AT THE
HOP — covered by unit test `test_ssrf_via_redirect_is_refused_at_the_hop` with an
httpx MockTransport, since it cannot be reproduced against a live public host.

## 7. Cache hit — zero network calls

Second fetch of the same URL within TTL returns `cache=hit`; the unit test
`test_cache_hit_makes_no_network_call` asserts the MockTransport call count does
not increase on the hit. `test_cache_miss_when_spill_evicted` proves a hit whose
spill entry was pruned degrades to a fresh fetch rather than returning a dead
handle.

## 8. `read <url>` sugar (same engine, same shape)

```
read https://example.com
  [200] https://example.com
  markdownify · text/html · cache hit · sparse/JS-gated (try `browser`)
  # Example Domain
  ...
  is_error=False
```

Regression guards (unit tests) confirm `read <file>`, `read spill://…`, and
`read skill://…` are unaffected by the new branch.

## 9. Render backend: markdownify vs stdlib fallback

```
$ ... fetch status
Render backend: markdownify

$ ... fetch set backend stdlib ; ... fetch test https://example.com
[200] https://example.com
stdlib · text/html · cache miss · ...
--- status=200 method=stdlib ...
```

Bare install simulated by forcing the markdownify/bs4 import to fail:

```
backend_available: False
method: stdlib
# Hi

Bare install body text here.
```

The stdlib `html.parser` renderer keeps a bare install fully functional; the
extra only upgrades render quality. Degradation is SILENT (no error).

## Gates

- `flake8 .` — clean
- `black --check .` (black==26.1.0) — clean (486 files)
- `isort --check .` (repo config) — clean
- `pyright --pythonpath .venv/bin/python .` — 0 errors
- `pytest tests/unit/web_fetch tests/unit/tui` — 2463 passed, 4 skipped
- `pytest tests/unit` — 6393 passed; the only failure
  (`test_eval_tool::test_background_streams_output_while_it_runs`) is a
  pre-existing timing flake, confirmed failing on a clean `origin/main` checkout
  independent of this change.

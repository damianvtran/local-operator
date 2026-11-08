<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Built for Token and Cache Efficiency

Long-running agents spend most of their tokens re-sending context, so the
harness treats the provider prompt cache as a resource to be managed:

- **A stable prefix.** The tools array is built in a deterministic order and
  rides in the same cache prefix as the system prompt, so the provider can
  reuse it turn after turn. Most extra capabilities live behind gated tools,
  skills, or MCP instead of sitting in every prompt.
- **Cache-aware pruning.** Superseded tool outputs (an older read of a file
  that was read again, a zero-match search) are blanked in place without
  forcing a cache rewrite. Once a session has idled past the cache TTL, the
  cache is provably cold and everything eligible flushes at once.
- **Automatic 1-hour cache TTL.** At or above 150k context tokens, Anthropic
  requests switch from the 5-minute to the 1-hour cache window; tunable via
  `providers.anthropic.cache_ttl_1h_min_context_tokens` (0 disables it).
- **Compaction before overflow.** Context compacts itself before the window
  fills; `/compact` runs it on demand and `/context` reports what is occupying
  the prefix right now.
- **Lazy everything else.** Skills are indexed semantically and their bodies
  load only when read; MCP servers advertise a bounded summary and individual
  tool schemas enter the context only when enabled; `wait` returns on job
  settle or message arrival so a long job costs one round trip, not twelve.

An exploratory harness-cost kit, with its methodology and known confounds, lives in
[benchmarks/harness_cost](../../benchmarks/harness_cost/README.md).

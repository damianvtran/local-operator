# Hindi context-only compatibility addendum

Actual auditor model: **openai/gpt-6.1-sol**, no fallback. Context-only reattestation, no new native coverage or tests.

Read the actual worktree notes for `relay.shell.usage.unit_credits`, `unit_tokens`, `unit_requests`. Context SHA-256 changed from `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1` to independently measured `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`.

**Compatible; no findings.** Notes require the same finite numeric measurement selected by `remaining ?? used`, not formatted text or fraction fallback; unit noun only, no #, outer amount prints the number once. The previously approved Hindi branches deliberately use invariant direct counted nouns क्रेडिट, टोकन, अनुरोध in one/other. That native rule remains correct for those composed amount phrases; it does not require French/Spanish de or Russian fractional-case rules. The French numeric-million NO de/OQLF, Spanish numeric-million de/RAE and Russian fractional-case statements in these shared notes are language-specific context, not independently revalidated by this Hindi-only addendum. English singular correction is documented as intentional; no byte-identical English extraction claim is made. USD remains invariant with no count.

Independently remeasured unchanged English shell source SHA-256: `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`.

Exact Hindi target retention, independently remeasured:
- shell `feab9e023be86d7841bd1e08cb273a48bbaaa118c41ade6040d7a29a701305eb`
- session `0fb4a5abfa0fb1ca26c2683a16a6db2f84f0f44fbf55522a7a21ec33deacb43e`
- login `589c538128272c5cd7c54468f9e126dda52a004ee21b9eab1949590b5156c4ff`

These equal the preceding audit's recorded hashes. Retain its narrow3 native PASS plus prior62 unchanged native coverage; no reset or expansion. No claim about other locales' target hashes. Source still described as working bytes, not committed HEAD content. Rendered/fit HOLD and absent pseudo evidence remain; no ledger/distribution/activation approval. No repository, candidate or ledger edits; no runtime retest.

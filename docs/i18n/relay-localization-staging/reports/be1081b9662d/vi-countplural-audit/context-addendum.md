# Vietnamese audit — context-only compatibility addendum

**Compatibility: PASS. No findings.** Prior three-key count-noun native PASS and retained62 approval stand unchanged. This is a context clarification reattestation, **not a new native coverage claim or test rerun**. Rendered **HOLD** remains.

Auditor: independent lopdev task, session `be1081b9662d`; actual model `openai/gpt-6.1-sol`; no fallback.

Read the exact working-tree `relay.shell.context.json` notes for `unit_credits`, `unit_tokens`, and `unit_requests`. Compared with the prior audit's captured context, only those three **note fields** changed; role/tone and other context rows remain unchanged.

- Old context SHA-256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`.
- New context SHA-256: `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`.
- English shell source remains `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`.

## Exact Vietnamese retention

Rehashed against the prior independent audit; all source and target fingerprints match:

| Namespace | Target SHA-256 |
|---|---|
| relay.shell | `f475dbefccd75d713fdea93fd8c5a12b541f94d00ec00c23b00fe72e2e942319` |
| relay.session | `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a` |
| relay.login | `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020` |

Full source fingerprints and retention assertions are in `context-addendum.json`. This auditor verified these three Vietnamese namespace targets, **not all 15 cross-locale files**.

## Native compatibility reasoning

The notes now explicitly require the **same finite numeric measurement selected by `remaining ?? used`**, not formatted text or `used_fraction`; the unit renders **noun only**, with no `#` or repeated number. That matches the previously audited Vietnamese target and directly exercised composition: `{count, plural, other {tín dụng}}`, `{count, plural, other {token}}`, `{count, plural, other {yêu cầu}}`, with the outer message printing the number exactly once.

The actual Vietnamese native rule is **CLDR cardinal `other` only with invariant unit nouns**. Vietnamese requires no singular/million noun suffix, no `de`, and no extra classifier here. It is therefore compatible with zero/one/two, fractional/negative/million amounts already exercised in the prior delta audit. The clarified USD fixed label and intentional English singular correction also introduce no Vietnamese target change.

The exact notes separately say French numeric million takes plain plural **without de** (citing OQLF numeric-versus-spelled-million guidance), Spanish numeric million takes **de + plural** (citing RAE cardinales §6), and Russian fractional `other` uses its native fractional case. These are locale-specific instructions; they neither override Vietnamese's invariant nouns nor constitute independent French/Spanish/Russian approval from this addendum. No foreign-language morphology rule is silently imposed on Vietnamese.

No candidate/source/ledger edits; no tests, full native reset, rendered checks or browser work. Existing audit evidence remains attached to unchanged source/target bytes. No shipped claim.

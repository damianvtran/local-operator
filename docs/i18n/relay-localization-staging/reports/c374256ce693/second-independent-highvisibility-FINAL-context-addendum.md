# PR2329 — final highvisibility context-only reattestation

**COMPATIBLE.** Existing five-locale final sample CLEAN verdict retained. No new findings.

**Actual model:** openai/gpt-6.1-sol; no fallback. No tests/native-audit reset, no target or source edits.

Old shell.context SHA-256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`
New shell.context SHA-256: `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`
Unchanged EN shell SHA-256: `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`

Read exact unit_credits, unit_tokens and unit_requests notes in the working source; unit_usd note also inspected.

- The exact three new notes clarify SAME finite remaining ?? used numeric count, never used_fraction or preformatted text. Noun-only branches do not print # or count; outer message prints the number once.
- FR bare-plural many matches OQLF Nombres exacts: fully numeric millions omit de.
- ES many de matches RAE DPD cardinales §6 explicit 1 000 000 de personas versus 1 200 000 personas; the previous overbroad no-de finding remains withdrawn.
- RU fractional other uses кредита/токена/запроса rather than integer-many forms; VI invariant other and HI identical direct noun forms remain unchanged.
- USD fixed no-count label and English singular intentional defect correction are compatible with source/targets already audited.

Primary rules:
- https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard
- https://www.rae.es/dpd/cardinales

## Exact retained target SHA-256 pins
### FR
- relay.shell: `7c33c08b1632baec788b8e84f018ae3c861a2d3ccfb180c5029de214341a1e15` — matches final audit
- relay.session: `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018` — matches final audit
- relay.login: `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24` — matches final audit
### ES
- relay.shell: `9e6691cdd0fab8dfbf09c314d8acd353a2d109a135d487493c652cb02a712438` — matches final audit
- relay.session: `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2` — matches final audit
- relay.login: `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658` — matches final audit
### RU
- relay.shell: `b96e8923fe82824db2434720f722d2cf323d5aa733d78ad1c75285391c16245b` — matches final audit
- relay.session: `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec` — matches final audit
- relay.login: `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1` — matches final audit
### VI
- relay.shell: `f475dbefccd75d713fdea93fd8c5a12b541f94d00ec00c23b00fe72e2e942319` — matches final audit
- relay.session: `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a` — matches final audit
- relay.login: `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020` — matches final audit
### HI
- relay.shell: `feab9e023be86d7841bd1e08cb273a48bbaaa118c41ade6040d7a29a701305eb` — matches final audit
- relay.session: `0fb4a5abfa0fb1ca26c2683a16a6db2f84f0f44fbf55522a7a21ec33deacb43e` — matches final audit
- relay.login: `589c538128272c5cd7c54468f9e126dda52a004ee21b9eab1949590b5156c4ff` — matches final audit

HOLD — customer resource restriction; no new pixels, full native audit, activation/shipped/release claim

Prior final audit: `reports/c374256ce693/second-independent-highvisibility-FINAL-five-locales.json`
Addendum JSON: `reports/c374256ce693/second-independent-highvisibility-FINAL-context-addendum.json`

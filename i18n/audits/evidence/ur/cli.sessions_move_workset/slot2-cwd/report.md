# #4403 delta (hi/ur needs_confirmation_cwd_unknown) — slot 2, recorded before reading slot 1's delta output

Translator index 21c33d0ea2334a4a6d184a9d6d3de710682998c70477637fe504dd44a74c25f7 (revision 1), all 8 per-file sha256 verified; es/fr/ru/vi/zh-CN files byte-identical to revision 0 (same sha256), merged targets unchanged.
hi/ur: only needs_confirmation_cwd_unknown changed vs prior merged target; value == slot-2 proposal byte-for-byte.
Order {digest},{source} == en. check.py --catalogues-only on final merged targets (scratch archive of 3030a5e5c): 0 problems; negative control (old ur value): 1 problem (reordered).
Meaning: --confirm-working-set {digest} re-run "so that they stay on {source}" (ताकि ये {source} पर ही रहें / تاکہ یہ {source} پر ہی رہیں) — confirming = LEAVING; never sending. PASS.
INDEPENDENCE CAVEAT: the wording of these 2 values was authored by slot 2 (my T5-HI-1/T5-UR-1 proposals, adopted verbatim). Slot 2 does not claim independence for those 2 values; slot 1's delta re-audit is the independent audit for them.
Targets: hi 33a03aa14454cd1e85facf49abb96cef9eeeefe78ce41e59f8010835743e0f0f · ur 693aa70f1a1b5a40ad5cdcde3eae5036a6e71411ae9623e6314fd277edc2cfbc.

## After reading slot 1 delta (7066915839, index 9ca860a4…)
Slot 1 merged hi/ur targets byte-identical to slot 2's; slot 1's receipts verified and chained.

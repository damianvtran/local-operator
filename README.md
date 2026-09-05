# Evidence: scoped Allow + dangerous allow-all (extension 0.1.8)

Rendered frames for PR #672. **Round 5** — re-captured at head `2213b44d5`
after the round 4 remediation.

Captured from the BUILT extension (`extension/dist`) loaded unpacked into a
throwaway Google Chrome 152 profile via CDP `Extensions.loadUnpacked`,
headless and never focused, using the corrected harness from
`~/workspace/qa-harness-notes/cdp-chrome-safe.py` (mock keychain, one profile
dir per launch, process-group teardown). Chrome count verified back to 0.

New this round: `popup-repeat-ask-deny*.png` — the re-ask after a DENY. The
line no longer says the answer "has already been used", which was untrue
after a refusal (U11). The scope select correctly shows the fail-closed
default, since a denial carries no scope to repeat.

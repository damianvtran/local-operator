# Evidence: scoped Allow + dangerous allow-all (extension 0.1.8)

Rendered frames for PR #672. **Round 2** — re-captured at head `03128edfd`
after the round 1 remediation.

Captured from the BUILT extension (`extension/dist`) loaded unpacked into a
throwaway Google Chrome 152 profile via the CDP `Extensions.loadUnpacked`
command, off-screen and never focused, the same route
`docs/design/browser-extension-evidence.md` records for the 0.1.4 frames.
Light and dark variants are the same scene under
`Emulation.setEmulatedMedia` `prefers-color-scheme`.

The acknowledgement frames are produced by **clicking Allow on a real prompt**
(round 1's were synthesised by assigning `textContent`, which is why they could
not have caught D1). With a live worker the queue drains in ~350 ms, so the
capture freezes the session echo — the exact condition the ack latch exists to
survive — rather than the ack itself.

`before-*` frames are the same scenes on `origin/main` (extension 0.1.7).

This branch exists only to host these images so the PR body can render them.
It is deliberately NOT merged: AGENTS.md keeps PR evidence out of the repo tree.

# Urdu core — independent audit round 3

- Audited revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Delta: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a..5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Scope: exactly 19 changed keys in four namespaces; unrelated values were not re-audited.
- Verdict: **CLEAN / terminal** — 0 blockers, 0 majors, 0 minors, 0 nits. Core `auditor.findings`: **0**.
- All seven round-2 finding groups are resolved. The 13 dispositions comprise 11 changed affected keys plus two intentionally unchanged literal-choice identifier values. Preview uses the accepted idiomatic alternate تجرباتی.
- No new findings. Native rendered RTL and high-visibility second audit remain separate; `wire.slash.hub` is flagged for rendered /help fit review (57 cells vs English 56), not a hard maxCells failure.

## Namespace verdicts

- `wire.errors.settings` — **clean**; 1 scoped key read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.remote_compute` — **clean**; 1 scoped key read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.settings` — **clean**; 16 scoped key read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.slash` — **clean**; 1 scoped key read; findings=0 blocker/major, 0 minor, 0 nit.

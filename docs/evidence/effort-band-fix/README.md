# Effort Band Fix: Visual Evidence

This directory captures visual validation that reasoning effort displays on the TUI status band for OpenRouter reasoning models (specifically `openrouter/google/gemini-3.8-flash`) and persists across frontend state updates, cycles through effort levels via `shift+tab`, and responds to `/effort auto`.

## Captured Frames

1. `1_boot_auto.png`: Initial boot state showing `▴ auto` in the status band beside `openrouter/google/gemini-3.8-flash`.
2. `2_after_frontend_update.png`: Demonstrates that `_apply_frontend_state` now derives `effort=_effort_label(state)` so `▴ auto` is preserved rather than wiped to `""`.
3. `3_cycle_medium.png`: First `shift+tab` cycles effort to `▴ medium`.
4. `4_cycle_high.png`: Second `shift+tab` cycles effort to `▴ high`.
5. `5_cycle_low.png`: Third `shift+tab` cycles effort to `▴ low`.
6. `6_effort_auto.png`: Executing `/effort auto` returns the effort level back to `▴ auto`.

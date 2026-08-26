# Brand fonts (vendored)

These are the exact self-hosted variable-font builds the local-operator.com
website ships (copied from the site repo's `public/fonts/`, which is synced
from the `@fontsource-variable` packages by the site's `scripts/sync-fonts.mjs`).
They are vendored here so `build_assets.py` can regenerate the Chrome Web
Store creative deterministically on any machine, without a checkout of the
website repo and without depending on OS-installed fonts.

| File | Family | Axes | Role in the assets |
|---|---|---|---|
| `fraunces-5.3.0-latin.woff2` | Fraunces Variable (`opsz` build) | opsz 9–144, wght 100–900 | Editorial display headlines |
| `figtree-5.3.0-latin.woff2` | Figtree Variable | wght 300–900 | Body / caption copy |
| `jetbrains-mono-5.3.0-latin.woff2` | JetBrains Mono Variable | wght 100–800 | Uppercase mono eyebrows, URLs |

The Fraunces file is deliberately the `opsz` (optical size) build — the same
choice the website makes — because the display sizes here (40–56 px) rely on
the high-optical-size drawing; the `wght`-only build is a body face scaled up
and reads visibly wrong next to the live site.

All three families are licensed under the SIL Open Font License 1.1, which
permits redistribution; license texts ship inside the upstream
`@fontsource-variable/*` packages.

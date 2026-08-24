# Subagent transcript history visual evidence

Rendered with the real `OperatorApp` and its production stylesheet.

## Before

`before/120x40.svg`, `before/90x28.svg`, and `before/40x20.svg` show the retained trajectory-only page. The page opens at its available tail, but there is no durable-history state and no route to activity older than the trajectory cap.

## After

`after/120x40.svg`, `after/90x28.svg`, and `after/40x20.svg` show the durable transcript tail at the same terminal sizes. The matching `.geometry.txt` files record screen and body sizes, virtual heights, scroll offsets, loaded IDs, and history state. In all three sizes the app screen virtual size equals its actual size and the screen-level scrollbar remains absent.

`after/120x40-prepend-loading.svg` and `after/120x40-prepend-settled.svg` are consecutive paging frames. The first visibly reports the asynchronous read. The settled frame records 143 stable IDs and `scroll_y=43` after 43 newly prepended rows, preserving the previously visible anchor rather than jumping to the new start. It also shows the exhausted `start` marker.

The rendered SVGs were loaded through the cmux browser and inspected as pixels, not as SVG markup. The settled 120x40 frame keeps the title, transcript, footer, dock, and screen geometry stable while history is prepended.

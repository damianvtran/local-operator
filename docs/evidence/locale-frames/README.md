# Per-locale rendered frames — relay SPA (#3165)

Rendered evidence that the relay SPA paints `ru`, `vi` and `fr` from their own
owned copy, on the **first content frame**, from the real built bundle served by
the real relay daemon. Measurement only: the QA pass edited no source file.

- **Commit under test:** `0393a16f2a9647d37e8fbb87dfc4a253b93dd641`
  (`feat/lazy-locale-bundles`, read from `gh pr view 3165 --json headRefOid`)
- **Bundle built from that commit** in a worktree at it, `vite build` →
  `local_operator/mobile/web/dist/assets`
- Host macOS 27.0.1 arm64 · Chrome 156.0.8078.48 (`--headless=new`) ·
  Node 24.22.1 · Python 3.12.13
- Viewport 390×844, DPR 2, default theme, one headless Chrome for the whole run

## Why these files exist

The earlier round proved one locale (Urdu) end to end, including the RTL
metadata and the pre-fix regression. The other three SHIPPED relay locales had no
rendered frame at all: a config that names `fr` and a generated chunk file are
both configuration, and neither says the screen says "Conversations". These
stills are that missing half.

## What each still is

`<locale>-1-first-content-session-deeplink.png` — the FIRST frame that had any
content on a cold deep link (`#/s/locale-a`), photographed the moment
`#root` stopped being empty. This is the frame the "no English flash" claim is
about.

`<locale>-2/3-*` — settled populated list (`#/`) and the settings surface
(`#/settings/account-routing`).

`-4-empty-list` — the list's own empty state (fixture with no sessions).

`-5-list-unavailable` / `-6/7-settings-load-error` — the relay answering 503 for
every API route EXCEPT `/api/config/events` (which is what names the locale). The
list keeps its loading skeleton; the settings screen renders its own
`relay.account_policy.load_error` sentence, so this is the state where a locale's
error copy is visible.

`control-hi-*` — the falsifiability control: a config naming `hi`, the locale the
backend ledger still lists but the relay SPA ships no chunk for. It MUST be
English; if this were localized too, "no English frames" for `ru`/`vi`/`fr` would
mean nothing.

`control-slowchunk-*` — the budget control: the `fr` chunk's HTTP response is
delayed 2 500 ms server-side (real bytes, real 200) so it lands past
`LOCALE_BOOTSTRAP_BUDGET_MS` (1 500 ms). The page must paint the documented
English fallback and then flip when the chunk arrives.

## Method

1. The PR head's tree is built (`vite build`); the five per-locale chunks and the
   main bundle are in `dist/assets`.
2. A scratch fixture (`locale_fixture.py`) serves that `dist` through the REAL
   `MobileDaemon` / `build_app`, with `language: <locale>` written through
   `ConfigManager` into an isolated `LOCAL_OPERATOR_CONFIG_DIR`
   (`scripts/probe_isolation` re-homes `HOME` and the config root on import) and
   `dial_registrants=False`.
3. The capture rig drives one headless Chrome over CDP, one fresh TAB per
   capture, and records for EVERY animation frame from document start: the
   visible text of `#root`, `document.documentElement.lang`, `dir`, and
   `performance.now()`. `innerText` is only read on frames whose text actually
   changed — computing it every frame forces a layout and starved the list's SSE
   frame in an earlier version of this rig.
3. It also records each capture's chunk requests from two independent channels:
   CDP `Network.*` events (status + encoded byte length) and the page's own
   Resource Timing entries.
4. A still is taken at a settled state — "no root-text change for 900 ms" after a
   floor of 4.5–9.5 s, not a fixed sleep: the list takes its first `sessions`
   frame up to ~2.9 s after paint, and a fixed sleep photographed the empty state
   on one run and the list on the next.

## Measured numbers

`frames.json` carries the full table (every count, both chunk channels, the
settle result). Every row below is spliced in from that file by
`rig/render_table.py`, not typed by hand.

<!-- measurements:begin -->
| locale | state | `lang` at first content | first content frame | English content frames | locale chunk (status · decoded bytes · response end) |
|---|---|---|---|---|---|
| ru | session deep link (cold deep link `#/s/<id>`) | `ru` | 635 ms — `‹ подключение к сессии… ↑` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 336 ms |
| ru | populated list `#/` | `ru` | 704 ms — `Беседы Все Ждут вас · 0 Непрочитано · 0 ＋ Новая` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 232 ms |
| ru | settings surface | `ru` | 602 ms — `Назад Настройки компьютера Изменения применяются только к этому компьютеру. Загрузка маршрутизации аккаунтов…` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 176 ms |
| ru | empty list | `ru` | 1,057 ms — `Беседы Все Ждут вас · 0 Непрочитано · 0 ＋ Новая` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 315 ms |
| ru | list, session API answering 503 | `ru` | 444 ms — `Беседы Все Ждут вас · 0 Непрочитано · 0 ＋ Новая` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 166 ms |
| ru | settings, API 503 | `ru` | 234 ms — `Назад Настройки компьютера Изменения применяются только к этому компьютеру. Загрузка маршрутизации аккаунтов…` | 0 | `bundled.ru.gen-CAXLVLTN.js` · 200 · 40,689 B · ends 145 ms |
| vi | session deep link (cold) | `vi` | 1,140 ms — `‹ đang kết nối tới phiên… ↑` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 785 ms |
| vi | populated list `#/` | `vi` | 5,754 ms — `Cuộc trò chuyện Tất cả Cần bạn · 0 Chưa đọc · 0 ＋ Mới` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 5,319 ms |
| vi | settings surface | `vi` | 387 ms — `Quay lại Cài đặt máy tính Thay đổi chỉ áp dụng cho máy tính này. Đang tải định tuyến tài khoản…` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 96 ms |
| vi | empty list | `vi` | 782 ms — `Cuộc trò chuyện Tất cả Cần bạn · 0 Chưa đọc · 0 ＋ Mới` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 434 ms |
| vi | list, session API answering 503 | `vi` | 681 ms — `Cuộc trò chuyện Tất cả Cần bạn · 0 Chưa đọc · 0 ＋ Mới` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 212 ms |
| vi | settings, API 503 | `vi` | 564 ms — `Quay lại Cài đặt máy tính Thay đổi chỉ áp dụng cho máy tính này. Không thể tải định tuyến tài khoản. Hãy kiểm tra kết nối rồi thử lại. Thử lại` | 0 | `bundled.vi.gen-D0sZsHop.js` · 200 · 30,396 B · ends 486 ms |
| fr | session deep link (cold) | `fr` | 775 ms — `‹ connexion à la session… ↑` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 614 ms |
| fr | populated list `#/` | `fr` | 887 ms — `Conversations Toutes À traiter · 0 Non lues · 0 ＋ Nouvelle` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 286 ms |
| fr | settings surface | `fr` | 253 ms — `Retour Paramètres de l’ordinateur Les modifications ne s’appliquent qu’à cet ordinateur. Chargement du routage des comptes…` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 146 ms |
| fr | empty list | `fr` | 452 ms — `Conversations Toutes À traiter · 0 Non lues · 0 ＋ Nouvelle` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 142 ms |
| fr | list, session API answering 503 | `fr` | 479 ms — `Conversations Toutes À traiter · 0 Non lues · 0 ＋ Nouvelle` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 204 ms |
| fr | settings, API 503 | `fr` | 344 ms — `Retour Paramètres de l’ordinateur Les modifications ne s’appliquent qu’à cet ordinateur. Chargement du routage des comptes…` | 0 | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 154 ms |
| hi | CONTROL — deep link, `hi` named in config | `en` | 809 ms — `‹ connecting to session… ↑` | 23 | none requested (no chunk for this locale) |
| hi | CONTROL — list, `hi` named in config | `en` | 580 ms — `Conversations All Needs you · 0 Unread · 0 ＋ New` | 77 | none requested (no chunk for this locale) |
| fr | CONTROL — deep link, `fr` chunk held past 1 500 ms | `en` then `fr` | 1,855 ms — ENGLISH `‹ connecting to session… ↑` | 7, last one at 2,350 ms; French takes over when the chunk lands | `bundled.fr.gen-BSzuhkb2.js` · 200 · 31,226 B · ends 3,726 ms |
<!-- measurements:end -->

<!-- evidence scope: the per-locale rows; the two control rows are deliberately
     NOT part of the "no English frame" claim. -->

## Honest limits## Honest limits

- One viewport, one theme (390×844, default dark). Not a theme or breakpoint
  matrix.
- The "relay unreachable" state here is a relay that ANSWERS the config stream
  and refuses the session API — with no relay at all there is no config frame, so
  the page is English by construction and proves nothing about a locale.
- The chunk hold is an injected server-side delay on one asset path, not a real
  throttled link.
- `dir` is `ltr` for these three locales by design; the RTL half of the contract
  belongs to `ur` and was evidenced in the earlier round.
- These frames say nothing about TUI copy, only the relay SPA.

## Reproducing

`rig/` holds the exact scripts of this pass: the fixture (`locale_fixture.py`),
the capture rig (`locale_frames_capture.py`), the plan that enumerates every case
(`plan_final.json`), the classifier (`analyse.py`), and the two assembly steps
(`assemble.py` writes this directory, `render_table.py` renders the table above).
They carry absolute paths for the worktree and the session scratchpad they ran
from; point `REPO` and the scratch root elsewhere to re-run them on another
checkout. The run itself is:

```
TMPDIR=<scratch> <worktree>/.venv/bin/python rig/locale_frames_capture.py \
  --outdir <scratch>/evidence_run --password-file <scratch>/pw --plan rig/plan_final.json
```

The password file is per-run and never printed; the daemon is spawned by the rig
with `start_new_session=True` and reaped by process group, and the rig fails the
run if a fixture port is still listening afterwards.

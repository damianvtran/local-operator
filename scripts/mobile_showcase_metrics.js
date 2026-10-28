/** Geometry probe shared by before/after evidence. Runs only from the
 * isolated showcase's Audit button, in the actual phone iframe document. */
window.measureRelay = function(win) {
 const {document, innerWidth, innerHeight} = win;
 const getComputedStyle = win.getComputedStyle.bind(win);
  const vw = innerWidth, vh = innerHeight;
  const inView = (r) => r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < vh &&
      r.right > 0 && r.left < vw;
  const visible = (el) => {
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none' ||
        parseFloat(s.opacity) === 0) return false;
    return inView(el.getBoundingClientRect());
  };
  const transparent = (c) => c === 'transparent' || /rgba\(.*,\s*0\)$/.test(c);
  let bordered = 0; const borderedSamples = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (!visible(el)) continue;
    const s = getComputedStyle(el);
    const sides = ['Top','Right','Bottom','Left'].filter((k) =>
      parseFloat(s['border'+k+'Width']) > 0 && s['border'+k+'Style'] !== 'none' &&
          !transparent(s['border'+k+'Color']));
    if (sides.length) {
      bordered++;
      if (borderedSamples.length < 40) borderedSamples.push({
        tag: el.tagName.toLowerCase(),
            sides: sides.length === 4 ? 'all' : sides.join('').toLowerCase(),
        text: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 28)});
    }
  }
  const ctrlSel = 'button, a[href], input:not([type=hidden]), textarea, select, '
      + '[role=button], [role=link]';
  const controls = [...document.querySelectorAll(ctrlSel)].filter(visible)
      .filter((el) => !el.closest('[inert]'));
  /* THE HIT AREA IS THE INTERSECTION, NEVER THE PSEUDO BOX (QA round 1, Q1-1).
     A filter chip draws a 32 px pill and a ::before that claims 44 px, but the
     strip it lives in is `overflow-x: auto`, and an auto-scrolling ancestor
     clips BOTH axes by definition — so the claimed band was 44 px while the real
     pointer band was 29 px (a 40 px strip minus Chromium's 6 px classic
     scrollbar). The audit reported zero target-size failures on that screen,
     which is exactly the false negative this function must never produce: the
     reported band is the pseudo box intersected with every clipping ancestor's
     CLIENT box (clientWidth/clientHeight exclude the scrollbar the rect
     includes), and then MEASURED by the engine's own hit testing. */
  const clipBox = (el) => {
    /* The VIEWPORT is not a clipper here: a row scrolled half off the screen is a
       scroll position, not a small target, and folding the viewport in reported
       exactly that as a 4 px one. Reachability of off-screen controls belongs to
       the QA matrix; this check is about the band a visible target offers. */
    let box = { top: -Infinity, left: -Infinity, right: Infinity, bottom: Infinity };
    for (let a = el.parentElement; a; a = a.parentElement) {
      const st = getComputedStyle(a);
      /* A SCROLLING ancestor MOVES a target, it does not shrink it: intersecting
         against an `auto`/`scroll` viewport reported every half-scrolled row as a
         4 px target. Only `hidden`/`clip` ancestors actually cut the band. */
      if (!/hidden|clip/.test(`${st.overflow} ${st.overflowX} ${st.overflowY}`)) continue;
      const r = a.getBoundingClientRect();
      const left = r.left + a.clientLeft, top = r.top + a.clientTop;
      box = {
        top: Math.max(box.top, top), left: Math.max(box.left, left),
        right: Math.min(box.right, left + a.clientWidth),
        bottom: Math.min(box.bottom, top + a.clientHeight),
      };
    }
    return box;
  };
  const geometricBand = (el, r) => {
    const before = getComputedStyle(el, '::before');
    const grows = before.content && before.content !== 'none';
    const expandT = grows ? Math.max(0, -parseFloat(before.top || 0)) : 0;
    const expandB = grows ? Math.max(0, -parseFloat(before.bottom || 0)) : 0;
    const clip = clipBox(el);
    const top = Math.max(r.top - expandT, clip.top), bottom = Math.min(r.bottom + expandB, clip.bottom);
    const left = Math.max(r.left, clip.left), right = Math.min(r.right, clip.right);
    return { w: Math.max(0, right - left), h: Math.max(0, bottom - top), expanded: expandT > 0 || expandB > 0 };
  };
  /* THE BAND THE POINTER ACTUALLY GETS, measured along the control's own centre
     rays. Three reasons it is rays and not a rectangle sweep:
       - a `rounded-full` control's CORNER is outside the circle, so a corner tap
         really lands on the ancestor (a 9-point sweep reported eight false
         "overlaps" for exactly this);
       - a neighbour's expanded `::before` band steals pixels from the edge of a
         real target, which only a hit test can see (QA round 2, Q2-1);
       - the reported number must be the band a finger owns, not the box.
     Ownership includes a descendant and the associated label, because those ARE
     the control's pointer target. */
  const ownsFactory = (el) => (node) => Boolean(node) && (node === el || el.contains(node)
    || (el.labels && el.labels[0] && (node === el.labels[0] || el.labels[0].contains(node))));
  const rayRun = (start, step, limit, hit) => {
    let last = null;
    for (let i = 0; i <= limit; i += 1) {
      const point = start + step * i;
      if (point < 0 || point >= limit0(step)) break;
      if (!hit(point)) break;
      last = point;
    }
    return last;
  };
  const limit0 = (step) => (step === 0 ? 0 : 1e9);
  const engineBox = (el, r) => {
    const owns = ownsFactory(el);
    const cx = Math.min(Math.max(Math.round(r.left + r.width / 2), 0), innerWidth - 1);
    const cy = Math.min(Math.max(Math.round(r.top + r.height / 2), 0), innerHeight - 1);
    const top = rayRun(cy, -1, 48, (y) => owns(document.elementFromPoint(cx, y)));
    const bottom = rayRun(cy, 1, 48, (y) => owns(document.elementFromPoint(cx, y)));
    if (top === null || bottom === null) return null;
    const left = rayRun(cx, -1, 96, (x) => owns(document.elementFromPoint(x, cy)));
    const right = rayRun(cx, 1, 96, (x) => owns(document.elementFromPoint(x, cy)));
    if (left === null || right === null) return null;
    return { left, right, top, bottom, w: right - left + 1, h: bottom - top + 1 };
  };
  /* A control not WHOLLY inside its scrollers' client boxes (and the viewport)
     is scrolled partly or fully away: that is a scroll position, not a small
     target. A tool row scrolled above the transcript measured 0 px here. Q1-1's
     case is unaffected: those chips sit wholly inside their strip; only their
     ::before band was cut, which the band measurement below still catches. */
  const scrolledAway = (el, r) => {
    if (r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) return true;
    for (let a = el.parentElement; a; a = a.parentElement) {
      const st = getComputedStyle(a);
      if (!/auto|scroll/.test(`${st.overflowX} ${st.overflowY}`)) continue;
      const box = a.getBoundingClientRect();
      const top = box.top + a.clientTop, left = box.left + a.clientLeft;
      if (r.top < top - 0.5 || r.bottom > top + a.clientHeight + 0.5
          || r.left < left - 0.5 || r.right > left + a.clientWidth + 0.5) return true;
    }
    return false;
  };
  const small = controls.filter(el => el.getAttribute("aria-hidden") !== "true").map((el) => ({ el, r: el.getBoundingClientRect() }))
    .filter(({ el, r }) => !scrolledAway(el, r))
    .map(({ el, r }) => {
      // A checkbox's associated label is part of its pointer target, not an
      // invisible exception to a size floor.
      const proxy = el.labels?.[0]?.getBoundingClientRect();
      const geo = geometricBand(el, r);
      const widest = Math.max(geo.w, proxy?.width || 0);
      const tallest = Math.max(geo.h, proxy?.height || 0);
      /* EVERY control is hit-tested now (QA round 2, Q2-1; design D7). The old
         gate scanned only controls that were themselves expanded or under-floor,
         so `attach image` — 44 px of its own box, no `::before` — scored 0 while a
         NEIGHBOUR's expanded band took its bottom 6 px. Ownership is the question,
         so ownership is measured; `engine_h` and `band` carry the answer. */
      const inView = r.top >= 0 && r.bottom <= innerHeight;
      const engine = inView && visible(el) ? engineBox(el, r) : null;
      /* The ENGINE's box is the reported size where it measured one: a control
         whose own rect says 44 but whose band a neighbour takes is a 38 px target,
         not a 44 px one. */
      /* WHAT TOOK THE MISSING BAND (evidence, not a guess): the element that
         answers just outside the engine box tells a reader whether a neighbour
         stole it, an ancestor clipped it, or nothing is there at all. */
      /* A CONTROL UNDER AN OVERLAY IS NOT A SMALL TARGET. The list's FAB floats
         over the rows by design, so a row action that has scrolled beneath it keeps
         only the part the FAB does not cover — a scroll position, exactly like the
         half-scrolled case above. These are counted separately (`under_overlay`)
         instead of failing the floor; a neighbour or an ancestor stealing the band
         still fails. */
      const thiefOf = (y) => document.elementFromPoint(Math.round(r.left + r.width / 2), y);
      const outOfFlow = (node) => {
        if (!node || node === el || el.contains(node)) return false;
        const position = getComputedStyle(node).position;
        return position === 'fixed' || position === 'absolute';
      };
      const partiallyCovered = engine && (engine.h < Math.round(r.height) - 1 || engine.w < Math.round(r.width) - 1);
      const overlayCovered = partiallyCovered && (
        outOfFlow(thiefOf(Math.max(0, engine.top - 2))) || outOfFlow(thiefOf(Math.min(innerHeight - 1, engine.bottom + 2))));
      const stolen = partiallyCovered && !overlayCovered
        ? {
            above: (() => { const n = document.elementFromPoint(Math.round(r.left + r.width / 2), Math.max(0, engine.top - 2));
              return n ? (n.getAttribute('aria-label') || n.tagName) + '.' + String(n.className).slice(0, 24) : null; })(),
            below: (() => { const n = document.elementFromPoint(Math.round(r.left + r.width / 2), Math.min(innerHeight - 1, engine.bottom + 2));
              return n ? (n.getAttribute('aria-label') || n.tagName) + '.' + String(n.className).slice(0, 24) : null; })(),
            parent: (() => { const box = el.parentElement?.getBoundingClientRect();
              return box ? [Math.round(box.top), Math.round(box.height)] : null; })(),
            parent_overflow: el.parentElement ? getComputedStyle(el.parentElement).overflow : null,
            box: [engine.left, engine.top, engine.w, engine.h],
          }
        : null;
      return { el, r, proxy, geo, engine, stolen, overlayCovered,
        w: engine ? Math.max(engine.w, proxy?.width || 0) : widest,
        h: engine ? Math.max(engine.h, proxy?.height || 0) : tallest };
    })
    .filter(({ w, h, overlayCovered }) => (w < 44 || h < 44) && !overlayCovered)
    .map(({ el, r, geo, engine, w, h, stolen }) => ({ label: (el.getAttribute('aria-label') || el.innerText ||
        el.tagName).trim().slice(0, 24),
      w: Math.round(w), h: Math.round(h),
      // Both numbers travel so a reader can tell a small control from a clipped one.
      rect_h: Math.round(r.height), clipped_h: +geo.h.toFixed(1), engine_h: engine ? engine.h : null,
      stolen }));
  const under_overlay = controls
    .filter((el) => el.getAttribute('aria-hidden') !== 'true')
    .map((el) => ({ el, r: el.getBoundingClientRect() }))
    .filter(({ el, r }) => r.width > 0 && r.height > 0 && visible(el) && !scrolledAway(el, r))
    .map(({ el, r }) => {
      const box = engineBox(el, r);
      if (!box || (box.w >= 44 && box.h >= 44)) return null;
      const above = document.elementFromPoint(Math.round(r.left + r.width / 2), Math.max(0, box.top - 2));
      const below = document.elementFromPoint(Math.round(r.left + r.width / 2), Math.min(innerHeight - 1, box.bottom + 2));
      const outOfFlow = (node) => Boolean(node) && !(node === el || el.contains(node))
        && ['fixed', 'absolute'].includes(getComputedStyle(node).position);
      if (!outOfFlow(above) && !outOfFlow(below)) return null;
      return { label: (el.getAttribute('aria-label') || el.innerText || el.tagName).trim().slice(0, 24),
        band: [box.w, box.h], rect: [Math.round(r.width), Math.round(r.height)],
        under: (outOfFlow(above) ? above : below)?.className ? String((outOfFlow(above) ? above : below).className).slice(0, 40) : null };
    })
    .filter(Boolean);
  const scrollers = [...document.querySelectorAll('*')].filter((el) => {
    const s = getComputedStyle(el);
    return /(auto|scroll)/.test(s.overflowY) && visible(el) && el.clientHeight > 80;
  }).map((el) => ({ el, r: el.getBoundingClientRect() }))
    .sort((a, b) => b.r.height - a.r.height);
  /* CONTENT = the viewport minus the chrome. The largest vertical scroller is the
     conversation — EXCEPT on an EMPTY one, which has none (a centred "No messages
     yet" block flexes into the space), so the old rule reported content 0 and
     chrome = the whole viewport: a pair of numbers describing no screen
     (round-1b send-state rows). The fallback is the header-to-composer band. */
  const main = scrollers[0];
  const bandGap = (() => {
    const header = document.querySelector('header'), foot = document.querySelector('.lo-composer');
    if (!header || !foot) return null;
    const top = header.getBoundingClientRect().bottom, bottom = foot.getBoundingClientRect().top;
    return bottom > top ? { top, bottom } : null;
  })();
  const contentPx = main
    ? Math.round(Math.min(main.r.bottom, vh) - Math.max(main.r.top, 0))
    : bandGap ? Math.round(Math.min(bandGap.bottom, vh) - Math.max(bandGap.top, 0)) : 0;
  const content_source = main ? 'scroller' : bandGap ? 'header-to-composer band (empty conversation)' : 'none';
  const fonts = new Set(), colors = new Set();
  for (const el of document.body.querySelectorAll('*')) {
    if (!el.childNodes.length || ![...el.childNodes].some((n) => n.nodeType === 3 &&
        n.textContent.trim())) continue;
    if (!visible(el)) continue;
    const s = getComputedStyle(el); fonts.add(s.fontSize); colors.add(s.color);
  }
  const rows = [...document.querySelectorAll('button')]
      .filter((b) => b.className.includes('flex-col') && b.className.includes('w-full') &&
      visible(b))
    .map((b) => {
      const r = b.getBoundingClientRect();
      const lines = [...b.children].filter((c) => c.getBoundingClientRect().height > 0).length;
      const text = b.innerText.replace(/\s+/g, ' ').trim().slice(0, 48);
      return { h: Math.round(r.height), lines, text };
    });
  const failures = [];
  for (const el of controls) {
    if (!['INPUT','TEXTAREA','SELECT'].includes(el.tagName) || el.type === 'checkbox' || el.type === 'radio') continue;
    const named = el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || el.labels?.length;
    if (!named) failures.push({check:'input-label', text:el.placeholder || el.tagName});
  }
  const rgb = value => {
    const match = value.match(/^rgba?\(([^)]+)\)/);
    if (!match) return null;
    const values = match[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    return values.length > 3 && values[3] === 0 ? null : values.slice(0,3);
  };
  const lum = color => color.map(v => { v /= 255; return v <= .04045 ? v/12.92 : ((v+.055)/1.055)**2.4; }).reduce((y,v,i)=>y+v*[.2126,.7152,.0722][i],0);
  for (const el of document.querySelectorAll('*')) {
    if (!visible(el) || el.closest('[aria-hidden="true"], [inert], :disabled, [aria-disabled="true"]') || ![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue;
    const style = getComputedStyle(el), fg = rgb(style.color);
    let node = el, bg = null;
    while (node && !bg) { bg=rgb(getComputedStyle(node).backgroundColor); node=node.parentElement; }
    if (!fg || !bg) continue;
    const [a,b]=[lum(fg),lum(bg)].sort((a,b)=>b-a), ratio=(a+.05)/(b+.05);
    const large=parseFloat(style.fontSize)>=24 || (parseFloat(style.fontSize)>=18.66 && Number(style.fontWeight)>=700);
    if (ratio < (large ? 3 : 4.5)) failures.push({check:'low-contrast',ratio:+ratio.toFixed(2),text:el.textContent.trim().slice(0,50)});
  }
  for (const target of small) failures.push({check:'target-size', ...target});
  /* EVERY CHIP-SHAPED TARGET'S MEASURED BAND, pass or fail (QA round 1, Q1-1).
     A green audit is only worth something if the number behind it is the band a
     finger actually gets; reporting only the failures let a clipped 29 px band
     read as a pass. */
  const hit_bands = controls
    .filter((el) => /lo-hit|rounded-full/.test(String(el.className)))
    .map((el) => {
      const r = el.getBoundingClientRect();
      const geo = geometricBand(el, r);
      const box = visible(el) ? engineBox(el, r) : null;
      const engine = box ? { w: box.w, h: box.h } : { w: 0, h: 0 };
      return {
        label: (el.getAttribute('aria-label') || el.innerText || el.tagName).trim().slice(0, 24),
        // The BOX, in real CSS px: separation and clearance are distances between
        // boxes, and a screenshot scale is not a measurement (operator finding A).
        rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
        rect_h: Math.round(r.height), clipped_h: +geo.h.toFixed(1), engine_h: engine.h,
        // The engine's own owned box, so a reader can see WHERE the band is.
        band: box ? [box.left, box.top, box.w, box.h] : null,
      };
    });
  /* CLIPPED TEXT (design round 1, D1): a control whose own copy needs more width
    than it has is dropping a fact. `scrollWidth` is where the layout engine says
     so, which is why this is measured rather than eyeballed in a screenshot. */
  const clipped = [];
  for (const el of document.querySelectorAll("button, span, div")) {
    if (!visible(el) || !(el.textContent || "").trim()) continue;
    const style = getComputedStyle(el);
    if (!/hidden|clip/.test(`${style.overflow} ${style.overflowX}`)) continue;
    if (el.clientWidth > 8 && el.scrollWidth > el.clientWidth + 1) {
      clipped.push({ text: (el.innerText || el.textContent).trim().slice(0, 30),
        need: el.scrollWidth, have: el.clientWidth });
    }
  }
  /* ROOT/PAGE HORIZONTAL OVERFLOW (operator finding B). A phone view must never
     scroll sideways as a whole: local chip/tray rails are deliberate, the ROOT is
     not. Reported as actual px so a wide row cannot hide behind a screenshot. */
  const doc = document.documentElement;
  const rootEl = document.body.firstElementChild;
  const overflowOf = (el) => el ? { scroll: el.scrollWidth, client: el.clientWidth,
    over: Math.max(0, el.scrollWidth - el.clientWidth) } : null;
  /* The composer's own box (padding edge), so its controls' clearance below is a
     measured number rather than a class name: `top/left/width/height` of the
     `lo-composer` element plus the padding actually resolved. */
  const composerEl = document.querySelector('.lo-composer');
  const cs = composerEl ? getComputedStyle(composerEl) : null;
  /* Every direct box in the composer's meta row, so a squeezed fact is a
     measured allocation (who took the width) and not a guess from a frame. */
  const metaRow = composerEl?.querySelector('button[aria-label^="Model"]')?.parentElement;
  const meta_row = metaRow ? [...metaRow.children].map((child) => {
    const r = child.getBoundingClientRect();
    return { label: (child.getAttribute('aria-label') || child.innerText || child.tagName).trim().slice(0, 28),
      x: Math.round(r.left), w: Math.round(r.width), need: child.scrollWidth, have: child.clientWidth };
  }) : null;
  /* WHERE THE SPEND READING ACTUALLY ENDS, and whether the model control exists
     at all (QA round 3 Q3-2 and C3-3). `meta_row` reports boxes; a box says nothing
     about the ink inside it, and at a short viewport the model box can be absent
     without anyone noticing. Both are distances from the composer's own box. */
  const composerEdges = (() => {
    if (!composerEl) return null;
    const box = composerEl.getBoundingClientRect();
    const spend = composerEl.querySelector('button[aria-label^="Spend"]');
    const model = composerEl.querySelector('button[aria-label^="Model"]');
    /* A Range belongs to the document whose nodes it selects. This whole audit
       runs in the PHONE frame's context, so it is `document`; `doc` above is the
       host page's documentElement and has no `createRange` — using it threw and
       killed the entire audit silently. */
    const rightOfInk = (el) => {
      if (!el) return null;
      const range = document.createRange();
      range.selectNodeContents(el);
      const rect = range.getBoundingClientRect();
      return rect.width ? +(box.right - rect.right).toFixed(1) : null;
    };
    const spendText = spend?.innerText?.trim() || null;
    return {
      spend_text: spendText,
      spend_text_right: rightOfInk(spend),
      spend_box_right: spend && spend.getBoundingClientRect().width
        ? +(box.right - spend.getBoundingClientRect().right).toFixed(1) : null,
      /* PRESENT means RENDERED. A short-layout `hidden` control is still in
         the DOM with a zero box; DOM presence alone would claim it is tappable. */
      model_present: Boolean(model && model.getBoundingClientRect().width > 0),
      model_text: model && model.getBoundingClientRect().width > 0
        ? (model.innerText || "").trim() || null : null,
      model_text_right: rightOfInk(model),
    };
  })();
  const composer = composerEl ? {
    /* THE FOUR OPTICAL INSETS, from the composer's own box (operator finding,
       round 2): the placeholder is painted at the textarea's content origin, while
       the meta line and the leading glyphs are measured from their real glyph
       boxes — a 44 px control's BOX says nothing about where its ink starts, and
       ink is what the operator was reading. */
    insets: (() => {
      const box = composerEl.getBoundingClientRect();
      const field = composerEl.querySelector('textarea');
      const fieldStyle = field ? getComputedStyle(field) : null;
      const fieldRect = field ? field.getBoundingClientRect() : null;
      const model = composerEl.querySelector('button[aria-label^="Model"]');
      const modelSpan = model ? (model.querySelector('span') || model) : null;
      const range = document.createRange();
      if (modelSpan) range.selectNodeContents(modelSpan);
      const modelRect = modelSpan && range.getClientRects().length ? range.getBoundingClientRect() : null;
      const paperclip = composerEl.querySelector('button[aria-label^="attach image"] svg');
      const folder = composerEl.querySelector('button[aria-label^="Working folder"] svg');
      return {
        box: [Math.round(box.left), Math.round(box.top), Math.round(box.width), Math.round(box.height)],
        placeholder: fieldRect && fieldStyle ? {
          left: +(fieldRect.left - box.left + (parseFloat(fieldStyle.paddingLeft) || 0)).toFixed(1),
          top: +(fieldRect.top - box.top + (parseFloat(fieldStyle.paddingTop) || 0)).toFixed(1),
        } : null,
        lead_paperclip: paperclip ? +(paperclip.getBoundingClientRect().left - box.left).toFixed(1) : null,
        lead_folder: folder ? +(folder.getBoundingClientRect().left - box.left).toFixed(1) : null,
        meta_text: modelRect ? {
          bottom: +(box.bottom - modelRect.bottom).toFixed(1),
          left: +(modelRect.left - box.left).toFixed(1),
        } : null,
      };
    })(),
    meta_row,
    edges: composerEdges,
    rect: (() => { const r = composerEl.getBoundingClientRect();
      return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; })(),
    pad: { top: cs.paddingTop, bottom: cs.paddingBottom, left: cs.paddingLeft, right: cs.paddingRight },
    overflow_x: cs.overflowX,
  } : null;
  const root_overflow = {
    document: overflowOf(doc), body: overflowOf(document.body), app: overflowOf(rootEl),
    widest: (() => {
      let worst = null;
      for (const el of document.querySelectorAll('*')) {
        if (!visible(el)) continue;
        const over = el.scrollWidth - el.clientWidth;
        if (over > 1 && (!worst || over > worst.over)) {
          worst = { over, tag: el.tagName, cls: String(el.className).slice(0, 60) };
        }
      }
      return worst;
    })(),
  };
  if ((root_overflow.document?.over || 0) > 0 || (root_overflow.body?.over || 0) > 0)
    failures.push({ check: 'root-overflow-x', ...root_overflow.document });
  /* EVERY HORIZONTAL SCROLLER, not only the root (operator finding B): an
     element whose computed overflow-x lets it scroll AND whose content is wider
     than its box is a sideways gesture somewhere on the screen. The list must be
     empty; each entry is a failure. `inTranscript` names the HARD case — the
     conversation itself must have zero horizontal play. */
  /* The conversation scroller, found STRUCTURALLY (the nearest vertical
     scroller that holds the last message) so the same audit works on the
     BEFORE bundle, whose transcript classes differ. */
  const transcript = (() => {
    const message = [...document.querySelectorAll('pre, table, p')].pop();
    for (let a = message?.parentElement; a; a = a.parentElement) {
      if (/auto|scroll/.test(getComputedStyle(a).overflowY) && a.scrollHeight > a.clientHeight) return a;
    }
    return null;
  })();
  const horizontal_scrollers = [];
  /* NOT limited to what is in view: a code block or table scrolled above the
     fold still scrolls sideways when the reader gets to it, and the transcript
     is exactly where the operator ruled that out. Rendered + displayed is
     enough; position is not. `transcript_overflow` below is the same rule for
     every descendant of the conversation scroller, scroller or not. */
  const rendered = (el) => {
    const st = getComputedStyle(el);
    return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
  };
  for (const el of document.querySelectorAll('*')) {
    if (!rendered(el)) continue;
    const ox = getComputedStyle(el).overflowX;
    if (!/auto|scroll/.test(ox) || el.scrollWidth <= el.clientWidth + 1) continue;
    horizontal_scrollers.push({ tag: el.tagName, cls: String(el.className).slice(0, 70),
      scroll: el.scrollWidth, client: el.clientWidth,
      inTranscript: Boolean(transcript && transcript !== el && transcript.contains(el)) });
  }
  for (const scroller of horizontal_scrollers) failures.push({ check: 'horizontal-scroller', ...scroller });
  /* EVERY CONTROL OWNS ITS OWN BAND (QA round 2, Q2-1; design D7): a tap
     anywhere in a control's measured band must land on THAT control. Two engine
     boxes that intersect are two controls claiming one pixel — the defect that
     made `attach image` a 38 px target while its own box said 44, so a tap on its
     lower edge opened the working-folder sheet. */
  const labelOf = (node) => !node ? 'none'
    : (node.getAttribute?.('aria-label') || node.innerText || node.tagName || '').trim().slice(0, 26);
  const measured = controls
    .filter((el) => el.getAttribute('aria-hidden') !== 'true')
    .map((el) => ({ el, r: el.getBoundingClientRect() }))
    .filter(({ el, r }) => r.width > 0 && r.height > 0 && visible(el) && !scrolledAway(el, r))
    .map(({ el, r }) => ({ el, box: engineBox(el, r) }))
    .filter((entry) => entry.box);
  const band_overlaps = [];
  for (let i = 0; i < measured.length; i += 1) {
    for (let j = i + 1; j < measured.length; j += 1) {
      const a = measured[i].box, b = measured[j].box;
      const across = Math.min(a.right, b.right) - Math.max(a.left, b.left) + 1;
      const down = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) + 1;
      if (across <= 0 || down <= 0) continue;
      /* AN OVERLAY COVERING A CONTROL IS NORMAL (a FAB over the list, a sheet over
         the page): those two boxes intersect in the layout, and the covered control
         simply is not tappable there — that is the overlay doing its job. The
         defect is two controls in the SAME FLOW stealing each other's band, which
         is what a shared offsetParent identifies: the composer's own controls,
         a chip strip's chips, a row and its ⋮. */
      if (measured[i].el.offsetParent !== measured[j].el.offsetParent) continue;
      band_overlaps.push({ a: labelOf(measured[i].el), b: labelOf(measured[j].el),
        overlap: [across, down] });
    }
  }
  for (const overlap of band_overlaps) failures.push({ check: 'band-overlap', ...overlap });
  /* ZERO HORIZONTAL PLAY IN THE CONVERSATION: the transcript scroller's own
     content may not be wider than it (it carries `overflow-x: hidden`, which
     would CLIP a too-wide block silently rather than scroll it), and the
     widest descendant is named. `pre`/`table` counts let a reader confirm the
     worst-case content was actually on the measured screen. */
  const transcript_overflow = transcript ? (() => {
    /* HORIZONTAL PLAY = something that can move sideways or pokes past the
       conversation's edge (and is silently clipped by its `overflow-x:
       hidden`). An ELLIPSIS truncation is neither: the element clips its own
       text with a visible `…` and keeps the full string as its text content
       (so as its accessible name). Those are counted separately, so a reader
       sees how much copy is truncated rather than having it hidden in a pass. */
    const box = transcript.getBoundingClientRect();
    let widest = null, ellipsis = 0;
    for (const el of transcript.querySelectorAll('*')) {
      if (!rendered(el)) continue;
      const st = getComputedStyle(el);
      if (st.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1) ellipsis += 1;
      const r = el.getBoundingClientRect();
      const over = Math.max(r.right - box.right - 0.5, box.left - r.left - 0.5,
        /auto|scroll/.test(st.overflowX) ? el.scrollWidth - el.clientWidth : 0);
      if (over > 1 && (!widest || over > widest.over))
        widest = { over: Math.round(over), tag: el.tagName, cls: String(el.className).slice(0, 50) };
    }
    return { scroll: transcript.scrollWidth, client: transcript.clientWidth, widest, ellipsis,
      pre: transcript.querySelectorAll('pre').length, table: transcript.querySelectorAll('table').length };
  })() : null;
  if (transcript_overflow && (transcript_overflow.scroll > transcript_overflow.client + 1 || transcript_overflow.widest))
    failures.push({ check: 'transcript-overflow-x', ...transcript_overflow });

  return JSON.stringify({
    viewport: [vw, vh], band_overlaps, under_overlay, content_source, root_overflow, horizontal_scrollers, transcript_overflow, composer, failures, bordered, borderedSamples, controls: controls.length,
    small_targets: small, hit_bands, clipped_text: clipped,
    chrome_px: vh - contentPx, content_px: contentPx,
    main_scroller: main ? { top: Math.round(main.r.top), bottom: Math.round(main.r.bottom),
        cls: String(main.el.className).slice(0, 80) } : null,
    font_sizes: [...fonts].sort(), text_colors: colors.size, rows,
  });

};

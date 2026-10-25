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
  const small = controls.filter(el => el.getAttribute("aria-hidden") !== "true").map((el) => ({ el, r: el.getBoundingClientRect() }))
    .filter(({ el, r }) => {
      // A checkbox's associated label and the pill pseudo-hit area are part of
      // its actual pointer target, not invisible exceptions to a size floor.
      const label = el.labels?.[0];
      const proxy = label?.getBoundingClientRect();
      const before = getComputedStyle(el, '::before');
      const hitHeight = before.content && before.content !== 'none'
        ? r.height + Math.max(0, -parseFloat(before.top || 0)) + Math.max(0, -parseFloat(before.bottom || 0)) : r.height;
      return Math.max(r.width, proxy?.width || 0) < 44 || Math.max(hitHeight, proxy?.height || 0) < 44;
    })
    .map(({ el, r }) => ({ label: (el.getAttribute('aria-label') || el.innerText ||
        el.tagName).trim().slice(0, 24),
      w: Math.round(r.width), h: Math.round(r.height) }));
  const scrollers = [...document.querySelectorAll('*')].filter((el) => {
    const s = getComputedStyle(el);
    return /(auto|scroll)/.test(s.overflowY) && visible(el) && el.clientHeight > 80;
  }).map((el) => ({ el, r: el.getBoundingClientRect() }))
    .sort((a, b) => b.r.height - a.r.height);
  const main = scrollers[0];
  const contentPx = main ? Math.round(Math.min(main.r.bottom, vh) - Math.max(main.r.top, 0)) : 0;
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
  return JSON.stringify({
    viewport: [vw, vh], failures, bordered, borderedSamples, controls: controls.length,
    small_targets: small, chrome_px: vh - contentPx, content_px: contentPx,
    main_scroller: main ? { top: Math.round(main.r.top), bottom: Math.round(main.r.bottom),
        cls: String(main.el.className).slice(0, 80) } : null,
    font_sizes: [...fonts].sort(), text_colors: colors.size, rows,
  });

};

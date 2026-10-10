"use strict";(()=>{
let N,Q;const D=document,R=D.documentElement,S="http://www.w3.org/2000/svg",H=[],V=[],B=[],// §4.1 S-R4: every frame->host message that can move host state (resize/error/pong) echoes the
// per-frame nonce the host minted and sent in its FIRST theme push; `ready` precedes that push, so
// it carries none. WHAT THE NONCE PROVES: that a post comes from the browsing context the host
// mounted (a navigated successor never received the push). It is NOT a secret from code running
// in this document: a component script can add its own `message` listener and read the push, so
// the host still clamps and validates every value. It also stays on `LO.theme` (the push as the
// host sent it): stripping it measured 4,611 B gzip against the 4,608 B cap, and would hide
// nothing from a script that adds its own listener. Navigation after mount is the HOST's to
// stop (one-shot guard / parent frame-src / second-load teardown, memo §4.1).
// N is fixed by the FIRST theme push (a later push cannot rebind it); a nonce that is missing or
// longer than the contract's NONCE_MAX_CHARS (64) binds "" -- refused, never truncated into a
// value the host did not mint -- so posts go out unmarked and the host drops them by rule.
// Q (Q-2): a component that throws during parse does so BEFORE the host's theme push at frame
// `load`, so its `error` would go out unmarked and be dropped. Errors raised before the nonce is
// bound are held -- the FIRST only: it is the cause, later ones are its fallout and a host shows one
// line either way -- and flushed with the nonce the moment the first theme push binds it.
P=m=>{try{parent.postMessage({lo:"supplement",v:1,...m,...(N&&{n:N})},"*")}catch(e){}},
// One spelling of the frame-level error post: three call sites, and the bytes it saves are what
// pays for the pre-nonce queue above inside the memo's gzip cap.
E=x=>{const m={t:"error",msg:String(x).slice(0,300)};N==null?Q??=m:P(m)};
const _ttl=(o,d)=>o.title||d.title,_us=o=>o.unit?(o.unit==="%"?"%":" "+o.unit):"";
// D2-1: the unit rides the top-most tick label, which is end-anchored inside the left margin;
// a margin that does not fit the composed string paints the leading characters outside the
// frame's viewport ("150 req/s" read as "50 req/s"). Widths must therefore be measured, not
// guessed: canvas metrics are the only synchronous source of a label's real width, and the
// font string is read once from the same custom property the stylesheet gives `svg text`.
let _mf,_mc;
const _tw=s=>{if(!_mf){_mf=`12px ${getComputedStyle(R).getPropertyValue("--font-sans").trim()}`;_mc=D.createElement("canvas").getContext("2d");_mc.font=_mf}return _mc.measureText(String(s)).width},
// D2-4: fit a label to the pixels it actually has — full text when it fits, truncated at the
// last character that still fits, "" when the slot is too small to be worth drawing. The full
// text stays in the <title>; on relay/native (no hover) a shortened label is the accepted loss.
_fit=(s,w)=>{s=String(s);if(_tw(s)<=w)return s;let n=s.length;while(n>1&&_tw(s.slice(0,n-1)+"…")>w)n--;return n>1?s.slice(0,n-1)+"…":""},
// A middle-anchored label (the horizontal bar's x ticks) is clamped into the plot's own span, so
// the tick that lands on the axis maximum cannot paint its tail past the frame edge.
_cl=(x,w,iw)=>Math.max(w/2,Math.min(x,iw-w/2));
const LO=window.LO={data:Object.freeze(JSON.parse(D.getElementById("lo-data").textContent||"{}")),theme:null,
onTheme(f){H.push(f);if(LO.theme)f(LO.theme)},
onSize(f){B.push(f)},
el(t,a,...c){const svg=a&&a.svg,e=svg?D.createElementNS(S,t):D.createElement(t);for(const k in a||{})if(k!=="svg"){if(k==="text")e.textContent=a[k];else e.setAttribute(k,a[k])}for(const x of c.flat())e.append(x);return e},
fmt(n,o={}){if(n==null||n==="")return"–";const v=+n;if(!isFinite(v))return String(n);const d=(String(v).split(".")[1]||"").length,s=o.compact&&Math.abs(v)>=1e4?v.toLocaleString(void 0,{notation:"compact",maximumFractionDigits:1}):v.toLocaleString(void 0,{maximumFractionDigits:o.digits??d});return o.unit?(o.unit==="%"?s+"%":s+" "+o.unit):s},
ds(id){const d=LO.data[id];if(!d)throw Error("unknown dataset "+id);return d},
col(id,c){const d=LO.ds(id),i=d.columns.indexOf(c);if(i<0)throw Error("unknown column "+c);return d.rows.map(r=>r[i])},
color(i){return"var(--s"+(i%6+1)+")"},
table(t,id,o={}){if(!o._k)V.push({k:"table",t,id,o});const d=LO.ds(id),ttl=_ttl(o,d),cols=o.columns||d.columns,ix=cols.map(c=>d.columns.indexOf(c)),num=ix.map(i=>d.rows.every(r=>r[i]===null||typeof r[i]==="number"));
const tb=LO.el("table",{},LO.el("thead",{},LO.el("tr",{},cols.map((c,j)=>LO.el("th",{class:num[j]?"n":"",scope:"col",text:c})))),LO.el("tbody",{},d.rows.map(r=>LO.el("tr",{},ix.map((i,j)=>LO.el("td",{class:num[j]?"n":"",text:num[j]?LO.fmt(r[i],{digits:o.digits}):String(r[i]??"")}))))));
const fig=LO.el("figure",{});if(ttl)fig.append(LO.el("div",{class:"ttl",text:ttl}));fig.append(LO.el("div",{class:"wrap"},tb));t.append(fig);LO.size();return tb},
// `need` is the width in px of the widest left-anchored axis label (the top tick carries the
// unit, so it is usually the widest): the left margin reserves it plus a gap. When that
// reservation would eat too much of a narrow frame, `ins` tells the caller to put the top tick
// inside the plot instead — a clipped label is never an option (D2-1).
_frame(t,o,need){const W=Math.max(200,Math.min(880,t.clientWidth||600)),Hh=o.height||Math.round(W*.45),m={t:12,r:o.right??64,b:28,l:o.left??48},ins=need!=null&&need+10>W*.4;if(need!=null&&!ins)m.l=Math.max(m.l,Math.ceil(need+10));return{W,H:Hh,m,iw:W-m.l-m.r,ih:Hh-m.t-m.b,ins}},
_scale(a,b,lo,hi){return v=>lo+(hi-lo)*((v-a)/((b-a)||1))},
_ticks(a,b,n=4){const st=Math.pow(10,Math.floor(Math.log10((b-a)/n||1))),e=[1,2,5,10].map(k=>k*st).find(s=>(b-a)/s<=n)||st*10,o=[];for(let v=Math.ceil(a/e)*e;v<=b+1e-9;v+=e)o.push(+v.toFixed(10));return o},
_svg(f,lab){return LO.el("svg",{svg:1,viewBox:`0 0 ${f.W} ${f.H}`,role:"img","aria-label":lab})},
_g(f,sv){const g=LO.el("g",{svg:1,transform:`translate(${f.m.l},${f.m.t})`});sv.append(g);return g},
bar(t,id,o){if(!o._k)V.push({k:"bar",t,id,o});const x=LO.col(id,o.x),ys=[].concat(o.y),vals=ys.flatMap(c=>LO.col(id,c)).filter(v=>v!=null),mx=Math.max(0,...vals),mn=Math.min(0,...vals),hz=!!o.horizontal,tk=LO._ticks(mn,mx),tl=tk.map(v=>LO.fmt(v,{compact:1})+(v===tk[tk.length-1]?_us(o):"")),vm=hz&&o.values!==false&&ys.length===1?Math.max(...vals.map(v=>_tw(LO.fmt(v)))):0,f=LO._frame(t,{...o,right:hz?Math.max(16,vm+8):16,left:hz?120:48},hz?null:Math.max(...tl.map(_tw))),sv=LO._svg(f,o.title||LO.ds(id).title||""),g=LO._g(f,sv);
const band=(hz?f.ih:f.iw)/x.length,bw=band*.72/ys.length,sc=hz?LO._scale(mn,mx,0,f.iw):LO._scale(mn,mx,f.ih,0);
for(let i=0;i<tk.length;i++){const v=tk[i],p=sc(v);g.append(LO.el("line",{svg:1,class:"grid",...(hz?{x1:p,x2:p,y1:0,y2:f.ih}:{x1:0,x2:f.iw,y1:p,y2:p})}),LO.el("text",{svg:1,text:tl[i],...(hz?{x:_cl(p,_tw(tl[i]),f.iw),y:f.ih+16,"text-anchor":"middle"}:f.ins&&i===tk.length-1?{x:0,y:p-4,"text-anchor":"start"}:{x:-6,y:p+4,"text-anchor":"end"})}))}
const st=Math.ceil(x.length/Math.max(2,Math.floor(f.iw/70))),thin=hz?Math.max(1,Math.ceil(13/band)):st;let pr=-1e9;
x.forEach((lab,i)=>{ys.forEach((c,j)=>{const v=LO.col(id,c)[i];if(v==null)return;const a=sc(Math.max(0,v)),z=sc(Math.min(0,v)),off=band*.14+j*bw+i*band;const r=hz?{x:Math.min(a,z),y:off,width:Math.abs(a-z),height:bw-1}:{x:off,y:Math.min(a,z),width:bw-1,height:Math.abs(a-z)};g.append(LO.el("rect",{svg:1,...r,fill:LO.color(j),rx:2},LO.el("title",{svg:1,text:`${lab} · ${c}: ${LO.fmt(v,{unit:o.unit})}`})));if(o.values!==false&&ys.length===1)g.append(LO.el("text",{svg:1,class:"lbl",text:LO.fmt(v),...(hz?{x:Math.max(a,z)+4,y:off+bw/2+4}:{x:off+bw/2,y:Math.min(a,z)-4,"text-anchor":"middle"})}))});
if(i%thin===0||i===x.length-1){const c=i*band+band/2,w=hz?f.m.l-12:Math.min(thin*band-6,2*(c-pr)-8),s=w>=24?_fit(lab,w):"";if(s){g.append(LO.el("text",{svg:1,text:s,...(hz?{x:-8,y:c+4,"text-anchor":"end"}:{x:c,y:f.ih+16,"text-anchor":"middle"})},LO.el("title",{svg:1,text:String(lab)})));pr=c+_tw(s)/2}}});
g.append(LO.el("line",{svg:1,class:"axis",...(hz?{x1:sc(0),x2:sc(0),y1:0,y2:f.ih}:{x1:0,x2:f.iw,y1:sc(0),y2:sc(0)})}));LO._put(t,sv,ys,o,_ttl(o,LO.ds(id)));return sv},
line(t,id,o){if(!o._k)V.push({k:"line",t,id,o});const x=LO.col(id,o.x),ys=[].concat(o.y),vals=ys.flatMap(c=>LO.col(id,c)).filter(v=>v!=null),num=x.every(v=>typeof v==="number"),lo=o.zero?Math.min(0,...vals):Math.min(...vals),hi=Math.max(...vals),tk=LO._ticks(lo,hi),tl=tk.map(v=>LO.fmt(v,{compact:1})+(v===tk[tk.length-1]?_us(o):"")),f=LO._frame(t,o,Math.max(...tl.map(_tw))),xs=num?LO._scale(Math.min(...x),Math.max(...x),0,f.iw):(v=>x.indexOf(v)*f.iw/Math.max(1,x.length-1)),sc=LO._scale(lo,hi,f.ih,0),sv=LO._svg(f,o.title||LO.ds(id).title||""),g=LO._g(f,sv);
for(let i=0;i<tk.length;i++){const v=tk[i];g.append(LO.el("line",{svg:1,class:"grid",x1:0,x2:f.iw,y1:sc(v),y2:sc(v)}),LO.el("text",{svg:1,text:f.ins&&i===tk.length-1?_fit(tl[i],f.iw):tl[i],...(f.ins&&i===tk.length-1?{x:0,y:sc(v)-4,"text-anchor":"start"}:{x:-6,y:sc(v)+4,"text-anchor":"end"})}))}
const step=Math.ceil(x.length/Math.max(2,Math.floor(f.iw/70))),dx=x.length>1?Math.max(24,(xs(x[x.length-1])-xs(x[0]))/(x.length-1)*step-6):f.iw;let lpr=-1e9;
x.forEach((v,i)=>{if(i%step===0||i===x.length-1){const c=xs(v),w=Math.min(dx,2*(c-lpr)-8),s=w>=24?_fit(v,w):"";if(s){g.append(LO.el("text",{svg:1,text:s,x:c,y:f.ih+16,"text-anchor":"middle"},LO.el("title",{svg:1,text:String(v)})));lpr=c+_tw(s)/2}}});
ys.forEach((c,j)=>{const yv=LO.col(id,c),pts=x.map((v,i)=>yv[i]==null?null:[xs(v),sc(yv[i])]).filter(Boolean);g.append(LO.el("polyline",{svg:1,points:pts.map(p=>p.join(",")).join(" "),fill:"none",stroke:LO.color(j),"stroke-width":2,"stroke-dasharray":["","6 3","2 3","8 3 2 3"][j%4]}));pts.forEach((p,i)=>g.append(LO.el("circle",{svg:1,cx:p[0],cy:p[1],r:2.5,fill:LO.color(j)},LO.el("title",{svg:1,text:`${x[i]} · ${c}: ${LO.fmt(yv[i],{unit:o.unit})}`}))));const last=pts[pts.length-1];if(last&&ys.length>1)g.append(LO.el("text",{svg:1,class:"lbl",text:c,x:last[0]+6,y:last[1]+4}))});
LO._put(t,sv,ys.length>1?[]:ys,o,_ttl(o,LO.ds(id)));return sv},
_put(t,sv,keys,o,ttl){const fig=LO.el("figure",{});if(ttl)fig.append(LO.el("div",{class:"ttl",text:ttl}));if(keys.length>1)fig.append(LO.el("div",{},keys.map((k,j)=>LO.el("span",{class:"k"},LO.el("i",{style:`background:${LO.color(j)}`}),k))));fig.append(sv);if(o.caption)fig.append(LO.el("figcaption",{text:o.caption}));t.append(fig);LO.size()},
size(){cancelAnimationFrame(LO._r);LO._r=requestAnimationFrame(()=>P({t:"resize",h:Math.ceil(R.getBoundingClientRect().height)}))},
_rd(){cancelAnimationFrame(LO._x);LO._x=requestAnimationFrame(()=>{const c=V.slice();V.length=0;for(const v of c)if(v.t.isConnected){v.t.replaceChildren();try{LO[v.k](v.t,v.id,{...v.o,_k:1})}catch(e){E(e)}V.push(v)}for(const f of B)try{f()}catch(e){}LO.size()})}};
addEventListener("message",({source:o,data:d})=>{if(o!==parent||!d||d.lo!=="supplement-host")return;if(d.t==="ping"){P({t:"pong"});return}if(d.t!=="theme")return;const th=d,k=d.nonce;if(N==null){N=typeof k=="string"&&k.length<65?k:"";Q&&P(Q)}R.dataset.mode=th.mode==="dark"?"dark":"light";for(const k in th.vars||{}){const v=String(th.vars[k]).slice(0,120);if(/^--(lo|font)-[a-z0-9-]+$/.test(k)&&CSS.supports(k,v))R.style.setProperty(k,v)}LO.theme=th;R.setAttribute("data-ready","");H.forEach(f=>{try{f(th)}catch(x){E(x)}});LO.size()});
addEventListener("error",e=>E(e.message));
let LW=0;new ResizeObserver(()=>{const w=R.clientWidth;if(LW&&w!==LW)LO._rd();LW=w;LO.size()}).observe(R);
setTimeout(()=>{if(!LO.theme){R.dataset.mode=matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light";R.setAttribute("data-ready","")}},400);
P({t:"ready"});
})();

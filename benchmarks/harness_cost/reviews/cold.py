"""Cold-cache counterfactual: published definition vs an independent one.

WHY: v1's "cold" column claims to reprice cross-run cache hits as first-
session writes. Its producer is not in the rig, so the review reverse-
engineered it (cost + first MAIN-call cache_read x (2.50-0.10)/M; matches
180/180 rows) and checks it against a stricter definition.
Stricter definition: per run and per served model, any call's cache_read
beyond the largest prefix this run had itself already put in cache (max of
read+write over that model's earlier calls) is a cross-run read; reprice it
at that model's (5m write - hit) rate. This also catches side calls and later
main calls, which the published definition ignores.
Constraints: read-only on source data; run from the kit root after
regenerate.py. Writes generated/reviews/cold_alt.json. No private artifacts
or model calls; the generated observed costs are checked against frozen v1.
"""
import json,collections as C,statistics as st
from pathlib import Path
Path('generated/reviews').mkdir(parents=True,exist_ok=True)
P={'sonnet':(2.5,0.10),'haiku':(0.125,0.01)}
req=[json.loads(l) for l in open('v1-snapshot/requests.jsonl')]
msgs=[q for q in req if q['path'].rstrip('/').endswith('v1/messages') and q.get('usage')]
by=C.defaultdict(list)
for q in msgs: by[q['run_id']].append(q)
for v in by.values(): v.sort(key=lambda q:q['ts'])
R=[r for r in json.load(open('generated/results.json'))['rows']]
out={}
for r in R:
    est=C.Counter(); extra=0; xt=0
    for q in by[r['run_id']]:
        m='haiku' if 'haiku' in (q.get('served_model') or '') else 'sonnet'
        u=q['usage'];cr=u.get('cache_read_input_tokens') or 0;cc=u.get('cache_creation_input_tokens') or 0
        x=max(0,cr-est[m]); extra+=x*(P[m][0]-P[m][1])/1e6; xt+=x
        est[m]=max(est[m],cr+cc)
    out[r['run_id']]=dict(warm=r['cost_usd'],v1cold=r.get('cold_cost'),altcold=r['cost_usd']+extra,xtok=xt)
json.dump(out,open('generated/reviews/cold_alt.json','w'),indent=0)
for arm in ['full']:
    for h in ['local-operator','omp','opencode','claude-code','codex']:
        rs=[v for k,v in out.items() if k.startswith(f'{arm}__{h}__') and k.endswith('r0')]
        print(f"{h:15} n={len(rs):2} warm={st.mean(x['warm'] for x in rs)*100:6.2f}c v1cold={st.mean(x['v1cold'] for x in rs)*100 if rs[0]['v1cold'] is not None else float('nan'):6.2f}c altcold={st.mean(x['altcold'] for x in rs)*100:6.2f}c  cross-run tok/run={st.mean(x['xtok'] for x in rs):,.0f}")

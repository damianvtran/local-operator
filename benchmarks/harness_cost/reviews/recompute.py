"""Independent per-run recompute of v1 cost from the frozen snapshot.

WHY: the rigor review must re-derive every headline number from raw wire
logs (v1-snapshot/requests.jsonl) rather than trust results.json or
figure_data.json, whose producing step is not in the rig.
HOW: prices each /v1/messages row with regenerate.price_row (the frozen rate
card), sums per run, and also sums OpenRouter's billed usage.cost so the two
can be reconciled. Writes generated/reviews/per_run.json.
Constraints: read-only on v1-snapshot/; run from the kit root with the
minimal analysis Python. Shares only the rate function, not aggregation.
"""
import json, collections as C, statistics as st, sys
from pathlib import Path
Path('generated/reviews').mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regenerate import price_row as _price
def price_row(model, usage):
    cost, parts, _, _ = _price(model, usage)
    return cost, parts
S='v1-snapshot/'
runs={json.loads(l)['run_id']:json.loads(l) for l in open(S+'runs.jsonl')}
tasks={t['id']:t for t in json.load(open(S+'tasks.json'))}
req=[json.loads(l) for l in open(S+'requests.jsonl')]
msgs=[q for q in req if q['path'].rstrip('/').endswith('v1/messages') and q.get('usage')]
per=C.defaultdict(lambda: dict(cost=0,orcost=0,calls=0,first=None,firstcr=0,firstcc=0,cr=0,cc=0,inp=0,out=0,haiku=0,haikucost=0,ts=[]))
for q in msgs:
    m=q.get('served_model') or q.get('requested_model') or ''
    c,p=price_row(m,q['usage'])
    a=per[q['run_id']]
    a['cost']+=c; a['orcost']+=q['usage'].get('cost') or 0; a['calls']+=1
    a['cr']+=p['cache_read']; a['cc']+=p['cache_write_5m']+p['cache_write_1h']; a['inp']+=p['input']; a['out']+=p['output']
    a['ts'].append(q['ts'])
    if 'haiku' in m: a['haiku']+=1; a['haikucost']+=c
    if a['first'] is None:
        a['first']=q; a['firstcr']=p['cache_read']
json.dump({k:{kk:vv for kk,vv in v.items() if kk!='first'} for k,v in per.items()}, open('generated/reviews/per_run.json','w'))

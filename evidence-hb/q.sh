#!/bin/bash
# q.sh <entry> [extra curl args...]  -> status line, cache-control, compact items
S=/Users/damian/.local-operator/sessions/3d299fb1b279/scratchpad/hb
SP=$(cat $S/serveport); e=$1; shift
curl -s -D $S/h.txt -H "Authorization: Bearer $(cat $S/token)" "$@" http://127.0.0.1:$SP/v1/desktop/sessions/abcdef012345/bridge-cards/$e/body > $S/b.txt
echo "## $e  $(head -1 $S/h.txt | tr -d '\r' | sed 's/^HTTP[^ ]* //')  $(grep -i '^cache-control' $S/h.txt | tr -d '\r')"
/usr/bin/python3 - "$S/b.txt" <<'EOF'
import json,sys
raw=open(sys.argv[1]).read()
try: d=json.loads(raw)
except Exception: print("  raw:", raw[:200]); sys.exit()
r=d.get('result',d) if isinstance(d,dict) else d
if not isinstance(r,dict) or 'items' not in r: print("  body:", json.dumps(d)[:240]); sys.exit()
its=r['items']; print(f"  items={len(its)} more={r.get('more')}")
for i in its[:2]+(its[-1:] if len(its)>2 else []): print("  ", json.dumps({k:(v[:70] if isinstance(v,str) else v) for k,v in i.items()}, ensure_ascii=False))
EOF

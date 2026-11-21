import sys,json,re
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scripts.i18n import check
from local_operator.i18n import catalogues
catalogues._CATALOGUES=Path(sys.argv[2])
problems=check.check_catalogues()
print('Repository check.check_catalogues() scoped to exact en/context/ru bytes:',json.dumps(problems))
assert not problems
en=catalogues.load_catalogue('en','wire.teams.effective');ru=catalogues.load_catalogue('ru','wire.teams.effective')
for key in en:
 a=check.msg_runtime.message_arguments(en[key]);b=check.msg_runtime.message_arguments(ru[key]);assert a==b
 print(key, 'arguments/types=',repr(a),'PARITY PASS')
 for token in ['subagents.models.','provider/model']:
  assert en[key].count(token)==ru[key].count(token),(key,token)
 assert re.findall(r'`[^`]*`',en[key])==re.findall(r'`[^`]*`',ru[key])
 assert not re.search(r'\{[^{}]*,\s*(plural|select|selectordinal)\b',en[key]+ru[key])
 assert not any(0x202a<=ord(c)<=0x202e or 0x2066<=ord(c)<=0x2069 for c in ru[key])
print('9 keys; token/backtick parity; no plural/select; no embedded bidi controls: PASS')

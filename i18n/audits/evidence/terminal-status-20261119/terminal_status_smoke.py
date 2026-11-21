"""Exercise the production status composer, isolated synthetic rows only."""
import json,os,pathlib
from local_operator.resume import SessionRow
from local_operator.session import catalog
from local_operator.incidents import render_cut_off_reason
from local_operator.i18n.resolve import resolve_language
root=pathlib.Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
assert pathlib.Path(catalog.__file__).resolve()==root/'local_operator/session/catalog.py'
results=[]
for locale in ('en','fr','es','zh-CN','ru','vi','hi','ur'):
 os.environ['LOP_LANG']=locale
 catalog._label_locale_memo=None
 assert resolve_language()==locale,(locale,resolve_language())
 for kind,reason in [('retired',''),('error',render_cut_off_reason('runtime-overdue')),('error','external diagnostic unchanged')]:
  row=SessionRow('synthetic-status-smoke',1.0,'Synthetic status smoke')
  attention={'kind':kind,'unseen':True,'reason':reason,'completion_token':'synthetic-token'}
  before=json.dumps(attention)
  item=catalog.entry_for(row,attention)
  assert json.dumps(attention)==before
  assert item.completion_reason==reason
  assert item.status_code==('interrupted' if kind=='retired' else 'error')
  assert item.status
  results.append({'locale':locale,'kind':kind,'reason':reason,'code':item.status_code,'label':item.status})
print(json.dumps({'production_path':str(catalog.__file__),'rows':results,'passed':len(results),'stored_reasons_unchanged':True},ensure_ascii=False,indent=2))

"""AST/data-only source controls; no producer/Textual/App imports or execution."""
import ast
from pathlib import Path
p=Path(__file__).with_name('producer.py')
t=ast.parse(p.read_text())
f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='validate_quality')
ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),'<quality-only>','exec'),ns)
valid=dict(qualified=True,unknown=0,overflow=0,unsupported=0,eager=0)
ns['validate_quality'](valid,1)
for name,value,count in [('unarmed',dict(qualified=False,unarmed=True),1),('empty',valid,0),('unknown',dict(valid,qualified=False,unknown=1),1),('overflow',dict(valid,qualified=False,overflow=1),1)]:
    try: ns['validate_quality'](value,count)
    except ValueError: print(name,'REJECT')
    else: raise AssertionError(name+' falsely accepted')
cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='ProbeApp')
assert len([n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and len(n.decorator_list)==1])==2
print('quality valid accepted; unarmed/empty/unknown/overflow rejected; two decorated methods AST present')
print('NO runtime producer/App/Textual/native/thread/loop execution')

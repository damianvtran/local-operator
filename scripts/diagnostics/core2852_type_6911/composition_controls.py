from pathlib import Path
import os,ast,json,copy,hashlib,types,tomllib,subprocess
b=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])/'core-type-offhost';m=json.loads((b/'manifest.json').read_text());repo=str(Path.home()/'local-operator');gitbytes={}
for kind,ref in (('BASELINE',m['baseline']),('FIXED',m['fixed'])):
 for p in m['snapshots'][kind]:gitbytes[ref+':'+p]=subprocess.check_output(['git','-C',repo,'show',ref+':'+p],timeout=8)
gitbytes[m['fixed']+':pyproject.toml']=subprocess.check_output(['git','-C',repo,'show',m['fixed']+':pyproject.toml'],timeout=8)
sourcebytes={p:subprocess.check_output(['git','-C',repo,'show',m['fixed']+':'+p],timeout=8) for p in m['execution_sources'] if not p.endswith('/runner.py')};sourcebytes['scripts/diagnostics/core2852_type_6911/runner.py']=(b/'runner.py').read_bytes();m['execution_sources']={p:hashlib.sha256(v).hexdigest() for p,v in sourcebytes.items()};tree=ast.parse((b/'runner.py').read_text());results=[]
def run(mode):
 fs={'REPO/'+p:v for p,v in sourcebytes.items()};fs['REPO/scripts/diagnostics/core2852_type_6911/manifest.json']=json.dumps(m).encode();fs['TOOL/package.json']=b'{"version":"1.1.414"}';fs['TOOL/index.js']=b'';calls=[];removed=[]
 class File:
  def __init__(self,path):self.path=path;fs[path]=b''
  def __enter__(self):return self
  def __exit__(self,*a):pass
  def write(self,data):fs[self.path]+=data
 class P:
  def __init__(self,path):self.p=str(path)
  def __truediv__(self,p):return P(self.p+'/'+p)
  def __str__(self):return self.p
  @property
  def parent(self):return P(self.p.rsplit('/',1)[0])
  def __eq__(self,other):return isinstance(other,P) and self.p==other.p
  def relative_to(self,root):return self.p.removeprefix(root.p+'/')
  def is_symlink(self):return False
  def is_file(self):return self.p in fs
  def read_bytes(self):return fs[self.p]
  def write_bytes(self,data):fs[self.p]=data
  def write_text(self,text):fs[self.p]=text.encode()
  def mkdir(self,**kw):pass
  def stat(self):return types.SimpleNamespace(st_size=262145 if mode=='cap' and self.p.endswith('BASELINE.json') else len(fs[self.p]))
  def open(self,mode):return File(self.p)
 def output(cmd,**kw):
  if cmd[0]=='git':
   if cmd[1]=='cat-file':return b''
   return gitbytes[cmd[2]]
  if cmd[-1]=='--version':return 'v24.0.0\n' if str(cmd[-2])=='--max-semi-space-size=16' else 'pyright 1.1.414\n'
  if cmd[-1]=='--help':return '--outputjson --project'
  return str(280*1048576)
 def run_tool(cmd,**kw):
  if '-e' in cmd:
   assert cmd[1:3]==['--max-old-space-size=256','--max-semi-space-size=16']
   value=400*1048576 if mode=='heap_range' else 304*1048576
   data={'heap_size_limit_bytes':value,'execArgv':cmd[1:]}
   if mode=='heap_type':data['heap_size_limit_bytes']=True
   if mode=='heap_flags':data['execArgv']=['PRIVATE']
   if mode=='heap_nonfinite':data['heap_size_limit_bytes']=float('nan')
   if mode=='heap_timeout':raise TimeoutError('synthetic timeout')
   kw['stdout'].write(b'{INVALID' if mode=='heap_json' else json.dumps(data).encode())
   return types.SimpleNamespace(returncode=3 if mode=='heap_rc' else True if mode=='heap_bool' else None if mode=='heap_none' else 0)
  root=kw['cwd'];kind=root.p.rsplit('/',1)[-1];calls.append(kind);assert cmd[-2:]==['tests/worker_code_identity.py','tests/unit/test_worker_code_identity.py'] and '--max-old-space-size=256' in cmd
  count=2 if kind=='BASELINE' else 0
  if mode=='three_errors':count=3
  report=dict(version='1.1.414',summary=dict(filesAnalyzed=2,errorCount=count,warningCount=0,informationCount=0),generalDiagnostics=[dict(file=str(root/'tests/worker_code_identity.py'),severity='error',rule='reportMissingTypeArgument',range=dict(start=dict(line=line))) for line in (67,71,80)[:count]])
  rc=1 if count else 0
  if mode=='nonzero':rc=2
  if mode=='none':rc=None
  if mode=='bool':rc=True
  kw['stdout'].write(b'{INVALID' if mode=='parse' else json.dumps(report).encode());return types.SimpleNamespace(returncode=rc)
 ns=dict(Path=P,ROOT=P('REPO'),HERE=P('REPO/scripts/diagnostics/core2852_type_6911'),FILES=tuple(m['snapshots']['BASELINE'])[:2],INITS=('tests/__init__.py','tests/unit/__init__.py'),CAP=262144,NODE_FLAGS=('--max-old-space-size=256','--max-semi-space-size=16'),HEAP_QUERY=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HEAP_QUERY' for t in n.targets)),json=json,hashlib=hashlib,tomllib=tomllib,os=types.SimpleNamespace(environ={'RUNNER_TEMP':'/owned','TYPE_PREFLIGHT_TOOL':'TOOL'}),sys=types.SimpleNamespace(argv=['runner','/out'],executable='python'),shutil=types.SimpleNamespace(which=lambda n:'node',rmtree=lambda p:removed.append(str(p))),tempfile=types.SimpleNamespace(mkdtemp=lambda **kw:'/owned/check'),resource=None,subprocess=types.SimpleNamespace(check_output=output,run=run_tool,TimeoutExpired=TimeoutError))
 # Use EXACT fixed target ordering, not manifest JSON sorting assumptions.
 ns['FILES']=('tests/worker_code_identity.py','tests/unit/test_worker_code_identity.py')
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'<actual-runner-main-fake-tools>','exec'),ns)
 try:ns['main']()
 except SystemExit as e:rc=e.code
 result=json.loads(fs['/out/typing.json']);assert removed==['/owned/check'] and result['root_removed'] and 'INVALID' not in json.dumps(result)
 if mode.startswith('heap_'):
  assert rc==1 and not result['qualified'] and calls==[] and result['cases']==[]
  assert 'PRIVATE' not in json.dumps(result) and 'INVALID' not in json.dumps(result)
  if mode=='heap_range':assert result['v8_heap_limit_bytes']==400*1048576 and result['failure_reason']=='NODE_HEAP_RANGE'
  if mode=='heap_rc':assert result['node_heap_query_rc']==3 and result['v8_heap_limit_bytes']==304*1048576 and result['failure_reason']=='NODE_HEAP_QUERY_RC'
  if mode in ('heap_bool','heap_none'):assert result['node_heap_query_rc'] is None
  if mode in ('heap_type','heap_nonfinite'):assert result['v8_heap_limit_bytes'] is None
 elif mode=='healthy':assert rc==0 and result['qualified'] and calls==['BASELINE','FIXED']
 else:
  assert rc==1 and not result['qualified'] and calls==['BASELINE'] and len(result['cases'])==1
  case=result['cases'][0];assert case['source_sha256']==m['snapshots']['BASELINE'] and case['qualified'] is False and case['failure_reason'] in ('VALIDATE_REPORT','OUTPUT_CAP','JSON_PARSE')
  if mode=='three_errors':assert case['rc']==1 and case['summary']['errorCount']==3
  if mode=='nonzero':assert case['rc']==2
  if mode in ('none','bool'):assert case['rc'] is None
 results.append(dict(mode=mode,exit=rc,cases=len(result['cases']),failed_observation_retained=mode!='healthy',root_removed=True))
for mode in ('healthy','three_errors','nonzero','cap','none','bool','parse','heap_range','heap_rc','heap_json','heap_flags','heap_type','heap_nonfinite','heap_timeout','heap_bool','heap_none'):run(mode)
(b/'heap-composition-results.json').write_text(json.dumps(dict(cases=results,observations='SIMULATED tool/FS/OS; ACTUAL verify_sources/main/run->projection->validator->finally',actual_analyzer=False),indent=2)+'\n');print('ACTUAL_MAIN_16_CASES_SIMULATED_HEAP_OBSERVATIONS_NO_NODE')

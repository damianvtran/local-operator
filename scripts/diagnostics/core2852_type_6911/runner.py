"""Two explicit files only; no Python subject imports or wrapper auto-install.

Each analyzer runs through the existing bounded PG owner. File-size limits cap
JSON before writing, not after buffering it; setup and code copies are ephemeral.
The isolated tests package prevents resolving a helper from the checkout.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
FILES=('tests/worker_code_identity.py','tests/unit/test_worker_code_identity.py')
INITS=('tests/__init__.py','tests/unit/__init__.py')
CAP=262144


def validate_report(report,rc,root,kind):
    if kind not in ('BASELINE','FIXED'):raise ValueError('kind STOP')
    if type(rc) is not int or rc!=(1 if kind=='BASELINE' else 0):raise ValueError('rc STOP')
    if type(report) is not dict or report.get('version')!='1.1.414':raise ValueError('version STOP')
    summary=report.get('summary');rows=report.get('generalDiagnostics')
    if type(summary) is not dict or type(rows) is not list or len(rows)>20:raise ValueError('report STOP')
    for k in ('filesAnalyzed','errorCount','warningCount','informationCount'):
        if type(summary.get(k)) is not int or summary[k]<0:raise ValueError('summary STOP')
    # CLI targets two files; imported helper is one of those same targets.
    if summary['filesAnalyzed']!=2:raise ValueError('scope STOP')
    safe=[]
    for row in rows:
        if type(row) is not dict or row.get('severity') not in ('error','warning','information'):raise ValueError('diagnostic STOP')
        file=row.get('file');path=Path(file) if type(file) is str else None
        if path not in tuple(root/p for p in FILES):raise ValueError('import root STOP')
        if row['severity']=='error':
            if path!=root/FILES[0] or row.get('rule')!='reportMissingTypeArgument':raise ValueError('unexpected error STOP')
            line=row.get('range',{}).get('start',{}).get('line')
            if type(line) is not int:raise ValueError('position STOP')
            safe.append(dict(file=FILES[0],rule='reportMissingTypeArgument',line=line+1))
    wanted=2 if kind=='BASELINE' else 0
    if summary['errorCount']!=wanted or len(safe)!=wanted or kind=='BASELINE' and sorted(r['line'] for r in safe)!=[68,72]:raise ValueError('baseline discriminator STOP')
    return dict(kind=kind,rc=rc,error_count=wanted,warnings=summary['warningCount'],files_analyzed=2,errors=safe,qualified=True)


def verify_sources(m):
    for p,sha in m['execution_sources'].items():
        path=ROOT/p
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('execution source STOP')
    for label,ref in (('BASELINE',m['baseline']),('FIXED',m['fixed'])):
        subprocess.check_output(['git','cat-file','-e',ref+'^{commit}'],cwd=ROOT,timeout=10)
        for p,sha in m['snapshots'][label].items():
            data=subprocess.check_output(['git','show',ref+':'+p],cwd=ROOT,timeout=10)
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('snapshot STOP')
    data=subprocess.check_output(['git','show',m['fixed']+':pyproject.toml'],cwd=ROOT,timeout=10)
    if hashlib.sha256(data).hexdigest()!=m['config_sha256']:raise ValueError('config source STOP')
    return data


def available_case(kind,bindings,rc=None,report=None,root=None):
    out=dict(kind=kind,stage='EXECUTION',failure_reason=None,qualified=False,rc=rc if type(rc) is int else None,source_sha256=dict(bindings),summary=None,errors=None)
    if type(report) is dict:
        summary=report.get('summary')
        if type(summary) is dict:
            out['summary']={k:v if type(v:=summary.get(k)) is int and 0<=v<=100000 else None for k in ('filesAnalyzed','errorCount','warningCount','informationCount')}
        rows=report.get('generalDiagnostics')
        if type(rows) is list and len(rows)<=20:
            out['errors']=[]
            for row in rows:
                if type(row) is not dict or row.get('severity')!='error':continue
                file=row.get('file');rule=row.get('rule');line=row.get('range',{}).get('start',{}).get('line') if type(row.get('range')) is dict and type(row['range'].get('start')) is dict else None
                if type(file) is str and root is not None and Path(file) in tuple(root/p for p in FILES) and rule=='reportMissingTypeArgument':
                    out['errors'].append(dict(file=str(Path(file).relative_to(root)),rule=rule,line=line+1 if type(line) is int and 0<=line<=100000 else None))
    return out


def capped_file():resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);receipt=dict(qualified=False,status='SETUP',cases=[],root_removed=False)
    work=Path(tempfile.mkdtemp(prefix='core2852-type-',dir=os.environ['RUNNER_TEMP']))
    try:
        m=json.loads((HERE/'manifest.json').read_bytes())
        config_source=verify_sources(m)
        settings=tomllib.loads(config_source.decode())['tool']['pyright']
        if settings.get('reportMissingTypeArgument') is not True:raise ValueError('strictness STOP')
        node=shutil.which('node');tool=Path(os.environ['TYPE_PREFLIGHT_TOOL'])
        package=json.loads((tool/'package.json').read_bytes())
        if package.get('version')!='1.1.414' or not (tool/'index.js').is_file():raise ValueError('tool STOP')
        receipt['node_version']=subprocess.check_output([node,'--version'],text=True,timeout=5).strip()
        # Direct CLI old-space flag wins over bounded runner's NODE_OPTIONS floor.
        heap=subprocess.check_output([node,'--max-old-space-size=256','-e',"process.stdout.write(String(require('v8').getHeapStatistics().heap_size_limit))"],text=True,timeout=5)
        if not heap.isdecimal() or not 256*1048576<=int(heap)<=320*1048576:raise ValueError('heap STOP')
        receipt['v8_heap_limit_bytes']=int(heap)
        version=subprocess.check_output([node,'--max-old-space-size=256',str(tool/'index.js'),'--version'],text=True,timeout=5).strip()
        help_text=subprocess.check_output([node,'--max-old-space-size=256',str(tool/'index.js'),'--help'],text=True,timeout=5)
        if version!='pyright 1.1.414' or '--outputjson' not in help_text or '--project' not in help_text:raise ValueError('CLI STOP')
        configs=[]
        for kind,ref in (('BASELINE',m['baseline']),('FIXED',m['fixed'])):
            receipt['status']=kind;root=work/kind;root.mkdir();case=available_case(kind,m['snapshots'][kind]);receipt['cases'].append(case)
            for p in FILES+INITS:
                data=subprocess.check_output(['git','show',ref+':'+p],cwd=ROOT,timeout=10)
                if hashlib.sha256(data).hexdigest()!=m['snapshots'][kind][p]:raise ValueError('snapshot STOP')
                target=root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            cfg=dict(settings)
            # Preserve relative project extras within THIS isolated source root;
            # never add the repository or its site-packages as a fallback.
            cfg['extraPaths']=[str(root/p) for p in settings.get('extraPaths',[])]
            for p in cfg['extraPaths']:Path(p).mkdir(parents=True,exist_ok=True)
            cfg['pythonVersion']='3.12';cfg['pythonPlatform']='Linux';cfg['include']=list(FILES)
            config=root/'pyrightconfig.json';config.write_text(json.dumps(cfg));configs.append(cfg|dict(extraPaths=settings.get('extraPaths',[])))
            target=work/(kind+'.json');errors=work/(kind+'.stderr')
            with target.open('wb') as stdout,errors.open('wb') as stderr:
                command=[sys.executable,'-B',str(ROOT/'scripts/run_bounded.py'),'--timeout','60','--grace','5','--memory-mb','512','--',node,'--max-old-space-size=256',str(tool/'index.js'),'--outputjson','--project',str(config),*FILES]
                proc=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr,preexec_fn=capped_file)
            case['rc']=proc.returncode if type(proc.returncode) is int else None;case['stage']='OUTPUT_CAP'
            if target.stat().st_size>CAP or errors.stat().st_size>CAP:raise ValueError('cap STOP')
            case['stage']='JSON_PARSE';parsed=json.loads(target.read_bytes())
            case.update(available_case(kind,m['snapshots'][kind],proc.returncode,parsed,root));case['stage']='VALIDATE_REPORT'
            row=validate_report(parsed,proc.returncode,root,kind);case.update(row,stage='COMPLETE')
        if configs[0]!=configs[1]:raise ValueError('config mismatch STOP')
        receipt['qualified']=True;receipt['status']='COMPLETE'
    except Exception:
        if receipt['cases']:
            case=receipt['cases'][-1]
            if case['qualified'] is not True:case['failure_reason']=case['stage'] if case['stage'] in ('EXECUTION','OUTPUT_CAP','JSON_PARSE','VALIDATE_REPORT') else 'UNKNOWN'
        receipt['failure_reason']='CASE_FAILED' if receipt['cases'] else 'SETUP_FAILED'
    finally:
        shutil.rmtree(work);receipt['root_removed']=True;(out/'typing.json').write_text(json.dumps(receipt,allow_nan=False)+'\n')
    raise SystemExit(0 if receipt['qualified'] else 1)


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:verify_sources(json.loads((HERE/'manifest.json').read_bytes()))
    else:main()

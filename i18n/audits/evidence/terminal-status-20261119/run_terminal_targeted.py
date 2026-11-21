import os,pathlib,subprocess
root=pathlib.Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch=pathlib.Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
home=scratch/'terminal-tests-home';home.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('CMUX_','LOP_'))}
env.pop('XPC_FLAGS',None)
env.update(HOME=str(home),LOCAL_OPERATOR_CONFIG_DIR=str(home/'.local-operator'),PYTHONPATH=str(root),GIT_CONFIG_SYSTEM='/dev/null',GIT_TERMINAL_PROMPT='0',GIT_ASKPASS='/usr/bin/false')
py='/Users/damian/local-operator/.venv/bin/python'
args=[py,'scripts/run_bounded.py','--contract','command-status','--timeout','600','--run-id','terminal-status-targeted','--',py,'-m','pytest','tests/unit/session/test_catalog_terminal_i18n.py','tests/unit/session/test_catalog_status_i18n.py','tests/unit/session/test_attention.py','tests/unit/test_incidents.py','-q','-n0','--no-cov']
with (scratch/'terminal-status-targeted-command-status.log').open('w') as out:
    result=subprocess.run(args,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
print('initial exit',result.returncode,flush=True)
print((scratch/'terminal-status-targeted-command-status.log').read_text()[-5000:],flush=True)
raise SystemExit(result.returncode)

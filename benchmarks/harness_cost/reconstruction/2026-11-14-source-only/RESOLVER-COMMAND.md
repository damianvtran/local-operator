# EXACT proposed resolver command — DO NOT EXECUTE without separate admission

This is a proposal only. It needs a LOCAL Docker Linux VM; it is NOT a macOS
cross-platform lock. No Docker/resolver/install/build has run in this phase.

## Prerequisites and disk-desk packet

* Parent/disk desk 00e6a1cd8f62 must register the proposed durable root
  `~/benchmark-runs/reconstruction-20261114-source-v1/` and retention: preserve
  attempts and receipts until an explicit parent/disk-owner disposition. The root
  is NOT created now; a HOLD/ignore marker is not deletion protection.
* Fresh `df` free >=30 GiB, memory GO including Docker VM overhead, no conflicting
  lease, approved exact daemon socket and compatible LOCAL linux/amd64 execution.
  Read-only filesystem metadata currently shows `/var/run/docker.sock` is a
  dangling symlink (target absent). This is a prerequisite BLOCKER, not daemon
  readiness proof. Do not start Docker or switch to a macOS resolver silently.
* The exact carrier in resolver-pins.json must ALREADY be locally cached and its
  Linux/amd64 identity checked. `--pull=never` prevents implicit acquisition.
  Missing carrier requires a separately admitted digest-pinned tool-carrier pull;
  no task image is pulled. No surviving frozen dependency tree is assumed.
* The exact uv wheel must ALREADY be staged at
  `~/benchmark-runs/reconstruction-20261114-source-v1/artifacts/<pinned filename>`.
  Its URL, SHA256 and size are in resolver-pins.json. Staging is a separate
  explicitly authorized acquisition, not performed by this command or now.
* Intended resolver HTTP endpoints: `https://pypi.org/simple/` and wheel/METADATA
  GETs at `https://files.pythonhosted.org/`. Carrier prefetch, if separately
  authorized, additionally uses Docker registry/auth/CDN endpoints. This command
  uses Docker bridge networking, NOT a claimed domain-level egress fence. If an
  enforced allowlist is an admission condition, an approved mirror/proxy is an
  additional prerequisite; no proxy/service is started here. No keys, user Docker
  credential config, model endpoint or task/media payload is mounted or exported.

### Resource envelope — proposed lease, not measured peak

One attempt, 900 s hard supervisor lease, 1 CPU, container memory+swap ceiling
2 GiB, pids 128. Tool/cache/tmp tmpfs limits are 128/256/64 MiB (448 MiB total,
charged within container memory); fresh output/inputs/log growth is stopped at
64 MiB observational growth guard (0.5 s polling, possible overshoot, NOT a host
filesystem quota). A 64 MiB per-file container limit and 16 MiB Docker-client log
limit bound individual writes; global Docker VM growth is still outside these
controls. These are proposed safety budgets with n=0 measurements, not inferred
latency/size distributions or promises of successful resolution. Exhaustion stops
and reports; no automatic retry or install/build fallback.

Metadata reports carrier compressed size 44,588,823 bytes and uv wheel 17,389,450
bytes. If prefetch were separately admitted, an illustrative 4x carrier expansion
plus compressed carrier, wheel and 64 MiB output is about 0.31 decimal GB;
reserve 1 GiB owned-file planning space excluding UNKNOWN VM allocation/build
cache. Current command never pulls and writes the carrier root filesystem
read-only. Docker VM/global disk overhead is NOT capped by the container flags;
the desk must account for it separately. The >=30 GiB floor alone proves none of
these fit. No task writable layer, media, build tree or installed environment is
created. A cache-limit failure is a setup result, not permission to expand it.

## LoP extras and closure identity

Use exact released v0.94.22 pyproject bytes, including core dependencies and
explicit functional extras `server,mcp,images,tokenizer,fetch,audio,lsp`; exclude
`dev`. This is the source-declared `all` capability set expanded explicitly to
avoid its self-referential root requirement. It preserves the potential historical
functional/tool surface without adding test tooling, but the lost installed
extras are UNKNOWN: this is an accepted new-environment policy, not historical
extra-equivalence evidence. Resolve one Linux/amd64 CPython 3.14.0 foundation
closure comprising those requirements, Harbor==0.25.0, shared service constraints,
and the pyproject build-system requirements. Other arms' source locks remain
separate qualification inputs; no parallel environment or install tree is made.

uv 0.13.8 CLI source confirms `--only-binary :all:` and `--no-build` are aliases
that CONFLICT when combined. Use ONLY `--only-binary :all:` below. Metadata may
require downloading dependency wheels to inspect METADATA; no wheel is installed
or executed and no sdist/backend is run. Managed Python downloads are disabled.

## Proposed invocation (copy only AFTER all gates)

Run from the reviewed reconstruction directory. The following Python is presented
as documentation, not a dispatched script. Attempt creation is exclusive; never
reuse the path. Existing macOS environments remain unmounted and unchanged.

```python
# PROPOSED ONLY — no permission to run is conveyed by this code block.
import hashlib, json, os, pathlib, resource, signal, stat, subprocess, time, tomllib
R = pathlib.Path.cwd().resolve()
pins = json.loads((R / 'resolver-pins.json').read_text())
source_head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
assert pins['platform'] == 'linux/amd64'
S = pathlib.Path('/var/run/docker.sock')
assert S.exists() and stat.S_ISSOCK(S.stat().st_mode), 'STOP: approved LOCAL daemon socket absent'
ROOT = pathlib.Path.home() / 'benchmark-runs/reconstruction-20261114-source-v1'
W = ROOT / 'artifacts' / pins['uv_wheel']['filename']
assert W.is_file(), 'STOP: separately admitted uv wheel staging required'
assert hashlib.sha256(W.read_bytes()).hexdigest() == pins['uv_wheel']['sha256']
A = ROOT / 'resolver/linux-amd64-01'
A.parent.mkdir(parents=True, exist_ok=True)
A.mkdir(mode=0o700)  # exclusive: an earlier attempt is never overwritten
for name in ('input', 'output', 'docker-config'):
    (A / name).mkdir()
D = ['docker', '--config', str(A / 'docker-config'), '--host', 'unix:///var/run/docker.sock']
# Empty owned config avoids the operator's credential store/context helpers.
try:
    image = json.loads(subprocess.check_output(D + ['image', 'inspect', pins['base_image']], timeout=15))[0]
    assert image['Architecture'] == 'amd64' and image['Os'] == 'linux'
except BaseException as error:
    (A/'preflight-failure.json').write_text(json.dumps({'status':'carrier-preflight-failed','error_type':type(error).__name__,'source_head':source_head}))
    raise
project = next((R / 'metadata').glob('lop-pyproject-*.toml'))
assert hashlib.sha256(project.read_bytes()).hexdigest() == '0bdee29cfba79826df5ab37b246e45dc11215179782c517cfed57e8c33bd5f61'
p = tomllib.loads(project.read_text())
requirements = list(p['project']['dependencies'])
for extra in pins['lop_extras']:
    requirements += p['project']['optional-dependencies'][extra]
requirements += ['harbor==0.25.0'] + p['build-system']['requires']
assert all(not any(t in line for t in ('://', ' @ ', 'git+', 'file:', '--')) for line in requirements)
(A / 'input/requirements.in').write_text('\n'.join(dict.fromkeys(requirements)) + '\n')
(A / 'input/services.lock').write_bytes((R / 'host-services.requirements.lock').read_bytes())
(A / 'input/pins.json').write_text(json.dumps(pins, sort_keys=True))
INNER = r'''
import hashlib,json,os,pathlib,platform,subprocess,sys,zipfile
assert sys.version_info[:3] == (3,14,0) and platform.machine() == 'x86_64'
pins=json.loads(pathlib.Path('/in/pins.json').read_text())
with zipfile.ZipFile('/uv.whl') as z:
    candidates=[i for i in z.infolist() if pathlib.PurePosixPath(i.filename).name=='uv']
    assert len(candidates)==1
    pathlib.Path('/tool/uv').write_bytes(z.read(candidates[0]))
os.chmod('/tool/uv',0o755)
version=subprocess.check_output(['/tool/uv','--version'],text=True).strip()
assert version.split()[:2]==['uv',pins['uv_version']]
pathlib.Path('/out/identity.json').write_text(json.dumps({'uv':version,'python':sys.version,'platform':platform.platform(),'python_executable_sha256':hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),'libc':platform.libc_ver(),'uv_binary_sha256':hashlib.sha256(pathlib.Path('/tool/uv').read_bytes()).hexdigest()},sort_keys=True))
subprocess.run(['/tool/uv','--no-config','pip','compile','/in/requirements.in','--constraint','/in/services.lock','--python','/usr/local/bin/python3.14','--python-version','3.14.0','--python-platform','x86_64-manylinux_2_28','--only-binary',':all:','--generate-hashes','--index-url','https://pypi.org/simple','--exclude-newer','2026-11-14T00:00:00Z','--output-file','/out/requirements.lock'],check=True)
lock=pathlib.Path('/out/requirements.lock').read_text()
assert '--hash=sha256:' in lock and 'git+' not in lock and 'file://' not in lock
assert ' @ ' not in lock, 'STOP: unreviewed direct reference'
'''
cmd = D + ['run','--rm','--pull=never','--platform=linux/amd64','--read-only','--network=bridge','--cpus=1','--memory=2g','--memory-swap=2g','--pids-limit=128','--ulimit','fsize=67108864:67108864','--cap-drop=ALL','--security-opt=no-new-privileges','--log-driver=none','--user',f'{os.getuid()}:{os.getgid()}','--cidfile',str(A/'container.cid'),'--tmpfs','/tool:rw,exec,nosuid,size=134217728,mode=1777','--tmpfs','/cache:rw,noexec,nosuid,size=268435456,mode=1777','--tmpfs','/tmp:rw,noexec,nosuid,size=67108864,mode=1777','-e','HOME=/tmp','-e','UV_CACHE_DIR=/cache','-e','UV_PYTHON_DOWNLOADS=never','-e','UV_CONCURRENT_DOWNLOADS=1','-v',f'{A}/input:/in:ro','-v',f'{A}/output:/out:rw','-v',f'{W}:/uv.whl:ro',pins['base_image'],'python3','-c',INNER]
(A / 'command.json').write_text(json.dumps(cmd))
started=time.monotonic(); status='failed'; proc=None
with (A / 'stdout.log').open('xb') as log:
    try:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                              preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_FSIZE,(16777216,16777216)))
        while proc.poll() is None:
            size=sum(x.stat().st_size for x in A.rglob('*') if x.is_file())
            if time.monotonic()-started>900 or size>67108864:
                raise RuntimeError('STOP: resolver lease/output-growth budget exhausted')
            time.sleep(0.5)
        if proc.returncode: raise RuntimeError(f'resolver exited {proc.returncode}')
        status='resolved-not-installed-not-qualified'
    finally:
        if proc is not None:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=5)
            try: os.kill(proc.pid,0)
            except ProcessLookupError: pass
            else: raise RuntimeError('STOP: owned process PID still exists')
        cid=A/'container.cid'
        if cid.exists():
            owned=cid.read_text().strip()
            subprocess.run(D+['rm','--force',owned],capture_output=True,timeout=15,check=False)
            check=subprocess.run(D+['container','inspect',owned],capture_output=True,text=True,timeout=15,check=False)
            absent=check.returncode != 0 and ('No such container' in check.stderr or 'No such object' in check.stderr)
            if not absent:
                (A/'cleanup-failure.json').write_text(json.dumps({'status':'cleanup-unverified','returncode':check.returncode,'stderr':check.stderr}))
                raise RuntimeError('STOP: owned container survived or daemon unavailable; cleanup not proven')
        files=[{'path':str(x.relative_to(A)),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted(A.rglob('*')) if x.is_file()]
        (A/'receipt.json').write_text(json.dumps({'status':status,'runtime_qualified':False,'source_head':source_head,'files':files},sort_keys=True,indent=2)+'\n')
```

Validation after the attempt: read actual identity.json/requirements.lock and
stdout/receipt, require zero resolver exit, expected platform/Python/uv identities,
all-package version/hash pins and no unreviewed URL/VCS references; compare root
requirements/constraints and source hashes. Run a separately reviewed static lock
validator before considering installation. Resolver success alone is NOT an
installed closure, build proof, wire proof or freeze. A missing carrier/socket,
unsatisfiable graph, absent wheel, denied network, timeout or memory/storage kill
is a failed prerequisite: keep receipts, stop, report; do not retry automatically.

Only the recorded process group and CID are stopped. No image/cache pruning,
other container deletion, daemon changes, host environment edits or receipt
cleanup is authorized. The `receipt.json` is final/new-attempt output; old attempt
files are never rewritten. Source and manager artifact hashes remain in git.

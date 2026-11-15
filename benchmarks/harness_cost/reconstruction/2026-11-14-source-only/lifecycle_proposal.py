"""UNEXECUTED full-scope lifecycle proposal; imports and default CLI have no effects.

Only a new independent FULL-scope clean review plus a fresh explicit desk slot
can enable --execute. Tests inject stub operations; no Desktop/Docker is used.
"""
import sys
if sys.flags.optimize:
    raise SystemExit('STOP: optimized interpreter prohibited')
import os
if os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('STOP: optimized interpreter prohibited')
import argparse
import datetime
import hashlib
import json
import re
import pwd
import plistlib
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ACK = 'a-7072'
ACK_QUOTE = "Find a way around this that doesn't require my manual involvement, validate in other ways or skip that validation with acknowledgement but make sure the code analysis and review is thorough"
CAP = 4_300_000_000
HELPERS = {'com.apple.Virtualization.VirtualMachine', 'com.docker.virtualization', 'qemu-system-aarch64', 'qemu-system-x86_64'}


def require(condition, message):
    if not condition:
        raise RuntimeError('STOP: ' + message)


DOCKER_BUNDLE=Path('/Applications/Docker.app')
DESKTOP_LAYOUTS=('Contents/MacOS/Docker','Contents/MacOS/Docker Desktop',
                 'Contents/MacOS/Docker Desktop.app/Contents/MacOS/Docker Desktop')
DEFAULT_DESKTOP=DOCKER_BUNDLE/DESKTOP_LAYOUTS[2]


def validate_desktop_layout(executable,run):
    """Only explicit Docker-signed layouts; never discover/launch arbitrary paths.

    Docker 4.43.1's outer bundle runs com.docker.backend and contains a separately
    signed Desktop UI bundle. Bind BOTH plists/signatures, not just a familiar
    filename. Flat legacy layouts remain explicit, not automatic fallbacks.
    """
    executable=Path(executable)
    require(executable in {DOCKER_BUNDLE/p for p in DESKTOP_LAYOUTS},'unapproved Desktop executable')
    nested=executable==DOCKER_BUNDLE/DESKTOP_LAYOUTS[2]
    bundles=[(DOCKER_BUNDLE,'com.docker.docker','com.docker.backend' if nested else executable.name)]
    if nested:bundles.append((DOCKER_BUNDLE/'Contents/MacOS/Docker Desktop.app','com.electron.dockerdesktop','Docker Desktop'))
    evidence=[]
    for bundle,identifier,entry in bundles:
        require(bundle.resolve(strict=True)==bundle,'bundle symlink/escape forbidden')
        info=bundle/'Contents/Info.plist'
        require(info.resolve(strict=True)==info,'plist symlink/escape forbidden')
        data=plistlib.loads(info.read_bytes())
        require(data.get('CFBundleIdentifier')==identifier and data.get('CFBundleExecutable')==entry,'bundle identifier/entrypoint mismatch or escape')
        binary=bundle/'Contents/MacOS'/entry
        require(binary.is_file() and binary.resolve(strict=True)==binary and os.access(binary,os.X_OK),'bundle executable missing/untrusted alias')
        code,_,_=run(['/usr/bin/codesign','--verify','--strict','-R','=anchor apple generic and certificate leaf[subject.OU] = "9BNSXJN65R"',str(bundle)])
        require(code==0,'Docker signature verification failed')
        _,out,err=run(['/usr/bin/codesign','-d','--verbose=2',str(bundle)])
        fields=dict(line.split('=',1) for line in (out+'\n'+err).splitlines() if line.startswith(('TeamIdentifier=','Identifier=')))
        require(fields.get('TeamIdentifier')=='9BNSXJN65R' and fields.get('Identifier')==identifier,'untrusted Docker signing identity')
        evidence.append({'bundle':str(bundle),'identifier':identifier,'entrypoint':entry,'version':data.get('CFBundleShortVersionString'),'plist_sha256':hashlib.sha256(info.read_bytes()).hexdigest(),'team':'9BNSXJN65R','local_signature_verified':True})
    require(executable.is_file() and executable.resolve(strict=True)==executable,'Desktop executable symlink/escape')
    if nested:require(evidence[0]['version']==evidence[1]['version'],'mixed outer/inner bundle versions')
    return {'executable':str(executable),'bundles':evidence,'launch_performed':False}


SWAP_LIMIT_BYTES=12_000_000_000  # New desk admission, NOT a rounded GiB limit.


def swap_measurement(text):
    match=re.search(r'\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMG])\b',text)
    require(match is not None,'swap usage unavailable/unrecognized; fail closed')
    value,unit=match.groups();scale={'K':1024,'M':1024**2,'G':1024**3}[unit]
    nominal=Decimal(value)*scale
    # sysctl prints rounded text. Include half its last displayed unit before
    # ceiling, so formatting/conversion cannot round the admission guard down.
    places=len(value.split('.')[1]) if '.' in value else 0
    upper=(Decimal(value)+Decimal(5).scaleb(-places-1))*scale
    return {'reported_used_bytes_ceiling':int(nominal.to_integral_value(rounding=ROUND_CEILING)),
            'conservative_used_bytes_upper':int(upper.to_integral_value(rounding=ROUND_CEILING)),
            'limit_bytes':SWAP_LIMIT_BYTES,'raw':text,'unit_interpretation':'K/M/G are binary units; formatted precision conservatively bounded'}


def parse_swap_used_bytes(text):
    return swap_measurement(text)['reported_used_bytes_ceiling']


def validated_registered_root(raw):
    """Bind campaign storage independently of Docker's deliberately isolated HOME.

    The native account home is an independent anchor, not a second untrusted
    environment variable. Refuse aliases/traversal rather than silently reading
    another campaign's admission or following a symlink outside registered HOLD.
    """
    require(isinstance(raw,str) and bool(raw),'registered root must be explicit')
    root=Path(raw)
    home=Path(pwd.getpwuid(os.getuid()).pw_dir)
    expected=home/'benchmark-runs/reconstruction-20261114-source-v1'
    require(root.is_absolute() and '..' not in root.parts and root==expected,'unregistered campaign root/escape')
    require(root.is_dir() and root.resolve(strict=True)==root,'registered root absent or symlink escape')
    require(root.stat().st_uid==os.getuid(),'registered root not owned by operator')
    return root


def write_anonymous_cli_config(directory):
    """A non-empty AuthConfigs map suppresses Docker's helper auto-detection.

    Docker CLI fdd4b8e5da335e1d0d83d29af5045f57280cd694:
    LoadDefaultConfigFile calls DetectDefaultStore only if !ContainsAuth();
    ContainsAuth counts map entries, not passwords. With no configured store or
    helper, GetCredentialsStore selects file storage and this empty record is
    anonymous. Empty auths={} would NOT suppress macOS osxkeychain detection.
    """
    config={'auths':{'https://index.docker.io/v1/':{}},'credsStore':'','credHelpers':{}}
    with (directory/'config.json').open('x') as out:
        out.write(json.dumps(config,sort_keys=True)+'\n')
    return config


def anonymous_cli_environment(directory):
    # Whitelist instead of copying DOCKER_AUTH_CONFIG or private proxy/token
    # variables. Never retrieve/read the operator's credential configuration.
    home=directory/'docker-home' if directory is not None else Path.home()
    return {'HOME':str(home),'PATH':'/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin',
            'LANG':'C','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1',
            'GIT_CONFIG_SYSTEM':'/dev/null','GIT_CONFIG_GLOBAL':'/dev/null',
            'GIT_TERMINAL_PROMPT':'0','GIT_ASKPASS':'/usr/bin/false'}


def verify_caps(info):
    memory, cpus = info.get('MemTotal'), info.get('NCPU')
    require(type(memory) is int and type(cpus) is int and 0 < memory <= CAP and 0 < cpus <= 2, 'observed docker info exceeds 4.3 GB/2 CPUs or is invalid')
    return {'phase':'post-start-readiness-probe', 'MemTotal':memory, 'NCPU':cpus}


def verify_carrier(image, pins):
    require(image is not None, 'carrier missing; no implicit pull')
    digest = pins['base_image'].split('@', 1)[1]
    require(image.get('Architecture') == 'amd64' and image.get('Os') == 'linux', 'carrier platform mismatch')
    require(any(item.endswith('@' + digest) for item in image.get('RepoDigests', [])), 'carrier digest mismatch')


def choose_helper(before, current, owner, raw_holders):
    """Never sum unrelated VMs or equate backend RSS with the guest footprint.

    An orphaned XPC helper needs BOTH new process identity and an open descriptor
    for the exact approved Docker.raw, not merely a familiar process name.
    """
    by_pid = {p['pid']:p for p in current}
    old = {(p['pid'],p['start'],p['exe']) for p in before}
    candidates = []
    for p in current:
        if Path(p['exe']).name not in HELPERS or (p['pid'],p['start'],p['exe']) in old:
            continue
        pid=p['ppid']; seen=set(); descendant=False
        while pid in by_pid and pid not in seen:
            seen.add(pid)
            ancestor=by_pid[pid]
            if pid == owner['pid'] and (ancestor['start'],ancestor['exe']) == (owner['start'],owner['exe']):
                descendant=True; break
            pid=ancestor['ppid']
        if descendant or p['pid'] in raw_holders:
            candidates.append(p)
    require(len(candidates) == 1, 'actual VM helper unidentifiable or ambiguous')
    return dict(candidates[0], rss_bytes=candidates[0]['rss_kib']*1024, measurement='one-attributed-helper-RSS-not-footprint')


def run_lifecycle(ops, admission, pins, require_runtime_scope=True):
    review=admission.get('full_scope_review', {})
    require(admission.get('go') is True and review.get('kind') == 'full-scope' and review.get('reviewer'), 'fresh GO and independent full-scope review required')
    if require_runtime_scope:
        require(admission.get('scope') == 'resolver-only', 'resolver scope not granted')
    owner=None; started=False; status='failed'; cleanup_errors=[];failure=None;work_started=ops.now()
    try:
        ops.preflight()
        require(not ops.existing_desktop(), 'Desktop/backend already running; cannot own it')
        owner=ops.start_owned()
        evidence=verify_caps(ops.info())  # FIRST daemon readiness evidence; no container yet.
        helper=ops.vm_helper(owner)
        require(helper['rss_bytes'] <= CAP, 'attributed helper RSS exceeds 4.3 GB')
        ops.check_disk()
        ops.snapshot_assets()  # Before ANY pull: only a later new exact ID is ownable.
        image=ops.cached_carrier()
        if image is None:
            require(review.get('allow_digest_pull') is True and admission.get('authorize_digest_pull') is True, 'carrier missing; optional digest pull not explicitly reviewed/authorized')
            ops.pull()
            image=ops.cached_carrier()
        verify_carrier(image,pins)
        require(ops.wheel_hash() == pins['uv_wheel']['sha256'], 'staged wheel missing/hash mismatch')
        started=True  # Cleanup also covers a signal during startup/assignment.
        ops.begin_resolver()
        last_sample=ops.now()
        while not ops.resolver_done():
            require(ops.now()-work_started<840,'whole-lifecycle working deadline; reserve owned cleanup')
            ops.check_disk()
            if ops.now()-last_sample >= 10:
                helper=ops.vm_helper(owner)
                require(helper['rss_bytes'] <= CAP, 'attributed helper RSS exceeds 4.3 GB')
                last_sample=ops.now()
            ops.sleep(0.5)
        status='resolved-not-installed-not-qualified'
        return status
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        raise
    finally:
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        for s in previous:signal.signal(s,signal.SIG_IGN)
        try:
            # A pending signal after startup ownership assignment must still quit
            # that exact instance. Never interrupt cleanup with a second signal.
            owner=owner or getattr(ops,'owner',None)
            if hasattr(ops,'begin_cleanup'):ops.begin_cleanup()
            for action in ([ops.stop_owned_resolver] if started else []):
                try: action()
                except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
            if owner is not None:
                try: ops.cleanup_owned_image()
                except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
                try: ops.quit_owned(owner)
                except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
            ops.record({'status':'cleanup-unverified' if cleanup_errors else status,'runtime_qualified':False,'cleanup_errors':cleanup_errors,'failure_reason':failure,'prestart_cap_validation':'SKIPPED-with-operator-ack','operator_ack_id':ACK,'measurement_caveat':'VM helper RSS is not guest/process footprint or total host memory'})
        finally:
            for s,handler in previous.items():signal.signal(s,handler)
        require(not cleanup_errors, 'owned lifecycle cleanup could not be proven')


class NativeOps:
    """Future LOCAL adapter. No settings reads/writes or broad process kills."""
    def __init__(self, admission, pins):
        self.a=admission;self.p=pins;self.owner=None;self.proc=None;self.resolver=None;self.directory=None
        self.root=Path.home()/'benchmark-runs/reconstruction-20261114-source-v1'
        self.desktop=Path(admission.get('desktop_executable',str(DEFAULT_DESKTOP)))
        self.app_root='/Applications/Docker.app/'
        self.raw=Path(admission.get('docker_raw_path','/INVALID'))
        self.deadline=0;self.next_rss=0;self.samples=[];self.helpers={};self.before=[]
        self.clients=[];self.monitor_enabled=False;self.monitoring=False;self.startup_uncertain=False;self.cleanup_until=None
        self.assets_before=None;self.asset_receipt={'policy':'exact new carrier ID only; never force/prune/layer/cache removal'}
        self.pull_attempted=False;self.pull_complete=False;self.owned_image_id=None;self.foreign_asset_activity=False
    def now(self): return time.monotonic()
    def sleep(self,seconds): time.sleep(seconds)
    def command(self,args,timeout=15,check=True):
        # Only own CLI child groups are reaped; never select a PID by a name.
        if self.cleanup_until is not None:
            timeout=min(timeout,self.cleanup_until-self.now())
            require(timeout>0,'owned cleanup budget exhausted; do not claim clean shutdown')
        proc=None
        try:
            blocked={signal.SIGTERM,signal.SIGINT};mask=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
            try:
                proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,
                                      env=anonymous_cli_environment(self.directory),
                                      preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,blocked))
                self.clients.append(proc)
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            limit=self.now()+timeout
            while True:
                try:
                    out,err=proc.communicate(timeout=min(0.5,max(0.001,limit-self.now())))
                    break
                except subprocess.TimeoutExpired:
                    require(self.now()<limit,'owned command deadline exhausted')
                    if self.cleanup_until is None and self.monitor_enabled and not self.monitoring:
                        self.check_disk()
                        if self.now()>=self.next_rss:
                            self.monitoring=True
                            try:
                                helper=self.vm_helper(self.owner)
                                require(helper['rss_bytes']<=CAP,'attributed helper RSS exceeds 4.3 GB')
                            finally:self.monitoring=False
            if check and proc.returncode: raise RuntimeError('command failed: '+args[0]+'; '+err[:300])
            return proc.returncode,out,err
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL);proc.communicate(timeout=5)
    def snapshot(self):
        _,out,_=self.command(['/bin/ps','-axo','pid=,ppid=,rss=,lstart=,comm='])
        result=[]
        for line in out.splitlines():
            fields=line.split(None,8)
            if len(fields)!=9: continue
            result.append({'pid':int(fields[0]),'ppid':int(fields[1]),'rss_kib':int(fields[2]),'start':' '.join(fields[3:8]),'exe':fields[8]})
        return result
    def desktop_records(self, rows):
        return [p for p in rows if p['exe'].startswith(self.app_root) or Path(p['exe']).name in {'Docker Desktop','com.docker.backend'}]
    def preflight(self):
        validated_registered_root(str(self.root))  # Reject bad roots before any startup/work.
        self.sample_swap('PRE-startup')  # No Desktop starts when swap evidence fails.
        require(self.a.get('operator_ack_id')==ACK and self.a.get('operator_ack_quote')==ACK_QUOTE,'operator acknowledgement missing')
        require(self.a.get('prestart_cap_validation')=='skipped-with-operator-ack' and 'vm_caps_verified_before_start' not in self.a,'must not fabricate pre-start verification')
        now=datetime.datetime.now(datetime.UTC)
        begin=datetime.datetime.fromisoformat(self.a['slot_start_utc']);end=datetime.datetime.fromisoformat(self.a['slot_end_utc'])
        require(begin <= now and now+datetime.timedelta(seconds=960) <= end and now <= datetime.datetime.fromisoformat(self.a['valid_until_utc']),'no fresh explicit slot/lease')
        require(self.a.get('fresh_memory_go') is True and 0 <= self.a['fleet_mb'] <= 26000,'fleet/memory HOLD')
        require(self.a.get('exclusive_desktop_lease') is True,'exclusive Desktop lease not granted')
        review=self.a['full_scope_review'];head=self.command(['git','-C',str(HERE),'rev-parse','HEAD'])[1].strip()
        require(not self.command(['git','-C',str(HERE),'status','--porcelain'])[1].strip(),'dirty source cannot be admitted under an old reviewed head')
        require(review.get('clean') is True and review.get('source_commit')==head and review['reviewer']!='70ba7562e541','full-scope clean independent current-head review missing')
        for name in ('lifecycle_proposal.py','RESOLVER-COMMAND.md','resolver-pins.json'):
            require(review.get('file_sha256',{}).get(name)==hashlib.sha256((HERE/name).read_bytes()).hexdigest(),'reviewed file hash mismatch')
        self.desktop_layout=validate_desktop_layout(self.desktop,self.command)
        require(self.root.is_dir() and self.a.get('docker_raw_path') and self.raw.is_absolute() and self.raw.name=='Docker.raw','registered root/approved raw path missing')
        fs=os.statvfs(self.root);require(fs.f_bavail*fs.f_frsize >= 30*1024**3,'fresh free disk below 30 GiB')
        initial=self.raw.stat() if self.raw.exists() else None
        self.baseline=initial.st_blocks*512 if initial else 0
        self.raw_identity=(initial.st_dev,initial.st_ino) if initial else None
        require(not Path('/var/run/docker.sock').exists(),'daemon socket target already exists; no foreign/stale engine takeover')
        self.before=self.snapshot()
        for process in self.before:
            if Path(process['exe']).name in HELPERS:
                code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(process['pid']),'-Fn'],check=False,timeout=3)
                require(code==0,'pre-start VM association unavailable; no startup')
                require(('n'+str(self.raw)) not in out.splitlines(),'existing VM already owns approved Docker.raw')
        self.directory=self.root/'lifecycle'/self.a['attempt_id']
        require(bool(self.a['attempt_id']) and all(c.isalnum() or c in '-_' for c in self.a['attempt_id']),'unsafe attempt id')
        require(self.a['attempt_id'] not in {'','.', '..'},'empty attempt id')
        self.directory.parent.mkdir(exist_ok=True);self.directory.mkdir(mode=0o700)
        (self.directory/'docker-config').mkdir()
        (self.directory/'docker-home').mkdir()
        write_anonymous_cli_config(self.directory/'docker-config')
        self.d=['docker','--config',str(self.directory/'docker-config'),'--host','unix:///var/run/docker.sock']
        self.deadline=self.now()+840
        self.samples.append({'phase':'PRE-startup','allocated_bytes':self.baseline,'free_bytes':fs.f_bavail*fs.f_frsize})
        require(not (self.root/'resolver/linux-amd64-01').exists(),'resolver attempt already exists; no reuse or deletion')
    def existing_desktop(self): return bool(self.desktop_records(self.before))
    def start_owned(self):
        blocked={signal.SIGTERM,signal.SIGINT};mask=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
        try:
            self.proc=subprocess.Popen([str(self.desktop)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,
                                       preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,blocked))
            rows=self.snapshot();row=next((p for p in rows if p['pid']==self.proc.pid),None)
            require(self.proc.poll() is None and row is not None and row['exe']==str(self.desktop),'Desktop startup identity unprovable')
            self.owner=row
            with (self.directory/'startup-owner.json').open('x') as out:out.write(json.dumps(row,sort_keys=True))
            return row
        except BaseException:
            self.startup_uncertain=True
            # Popen itself proves ownership of this root child, even if process
            # metadata cannot qualify a Desktop/helper identity. No name kill.
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
            raise
        finally: signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    def info(self):
        # No polling loop/start/config fallback: one bounded readiness request.
        info=json.loads(self.command(self.d+['info','--format','{{json .}}'],timeout=10)[1])
        with (self.directory/'readiness.json').open('x') as out:out.write(json.dumps({'MemTotal':info.get('MemTotal'),'NCPU':info.get('NCPU'),'phase':'post-start-readiness-probe'},sort_keys=True))
        return info
    def vm_helper(self,owner):
        rows=self.snapshot();holders=set()
        for p in rows:
            if Path(p['exe']).name in HELPERS:
                code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(p['pid']),'-Fn'],check=False)
                if code==0 and ('n'+str(self.raw)) in out.splitlines():holders.add(p['pid'])
        require(any(p['pid']==owner['pid'] and (p['start'],p['exe'])==(owner['start'],owner['exe']) for p in rows),'Desktop owner identity no longer live')
        helper=choose_helper(self.before,rows,owner,holders)
        self.helpers[(helper['pid'],helper['start'],helper['exe'])]=helper
        self.samples.append(dict(helper,phase='VM-RSS',measurement_caveat='RSS is NOT footprint; no unrelated VM sum'))
        self.monitor_enabled=True;self.next_rss=self.now()+10
        return helper
    def sample_swap(self,phase):
        _,out,_=self.command(['/usr/sbin/sysctl','vm.swapusage'])
        measurement=dict(swap_measurement(out),phase=phase)
        self.samples.append({'swap':measurement})
        require(measurement['conservative_used_bytes_upper']<=SWAP_LIMIT_BYTES,'swap exceeds 12,000,000,000 bytes or precision bound; abort owned work')
        return measurement
    def check_disk(self):
        if not self.monitoring:
            self.monitoring=True
            try:self.sample_swap('during')
            finally:self.monitoring=False
        require(self.raw.is_file(),'approved Docker.raw absent after startup')
        current=self.raw.stat();identity=(current.st_dev,current.st_ino)
        require(self.raw_identity is None or self.raw_identity==identity,'Docker.raw identity changed; growth comparison invalid')
        self.raw_identity=identity
        fs=os.statvfs(self.root);allocated=current.st_blocks*512
        self.samples.append({'phase':'during','allocated_bytes':allocated,'free_bytes':fs.f_bavail*fs.f_frsize})
        require(allocated-self.baseline <= 5*1024**3 and fs.f_bavail*fs.f_frsize >= 25*1024**3,'raw allocation/free disk fence')
        require(self.now() < self.deadline,'whole-lifecycle working lease exhausted; reserve cleanup')
    def check_cleanup_resources(self):
        # Same physical fences, but NEVER re-enter the already-expired work
        # deadline. Cleanup has its own budget and must retire owned work.
        require(self.cleanup_until is not None and self.now()<self.cleanup_until,'owned cleanup budget exhausted')
        self.sample_swap('cleanup')
        current=self.raw.stat();fs=os.statvfs(self.root)
        require(self.raw_identity is None or self.raw_identity==(current.st_dev,current.st_ino),'cleanup raw-file identity changed; still retire owned container')
        sample={'phase':'cleanup','allocated_bytes':current.st_blocks*512,'free_bytes':fs.f_bavail*fs.f_frsize}
        self.samples.append(sample)
        require(sample['allocated_bytes']-self.baseline<=5*1024**3 and sample['free_bytes']>=25*1024**3,'cleanup resource fence crossed; still retire owned container')
    def asset_inventory(self):
        # CLI output selects IDs/layer metadata only, never Config.Env or secrets.
        ids=set(self.command(self.d+['image','ls','--all','--no-trunc','--quiet'])[1].split())
        require(all(re.fullmatch(r'sha256:[0-9a-f]{64}',x) for x in ids),'unrecognized full image ID')
        images={}
        fmt='{"Id":{{json .Id}},"Layers":{{json .RootFS.Layers}},"RepoTags":{{json .RepoTags}},"RepoDigests":{{json .RepoDigests}}}'
        if ids:
            lines=self.command(self.d+['image','inspect','--format',fmt,*sorted(ids)])[1].splitlines()
            for line in lines:
                row=json.loads(line);require(row['Id'] in ids,'image snapshot identity mismatch');images[row['Id']]=row
            require(set(images)==ids,'incomplete image/content snapshot')
        # Docker CLI pinned source formatter/disk_usage.go: non-table verbose
        # templates expose untruncated BuildCache IDs. Unsupported CLI => STOP.
        cache=[];cache_in_use=False
        cache_format='{{range .BuildCache}}{"ID":{{json .ID}},"InUse":{{json .InUse}}}{{println}}{{end}}'
        for line in self.command(self.d+['system','df','--verbose','--format',cache_format])[1].splitlines():
            row=json.loads(line);require(row['InUse'] in ('true','false'),'build-cache activity unavailable')
            active=row['InUse']=='true';raw_id=row['ID']
            require(bool(raw_id) and raw_id.endswith('*')==active,'cache activity/ID marker inconsistent')
            cache_id=raw_id[:-1] if active else raw_id
            require(re.fullmatch(r'[A-Za-z0-9_:-]+',cache_id) is not None,'unrecognized build-cache ID')
            cache.append(cache_id);cache_in_use=cache_in_use or active
        cids=self.command(self.d+['ps','--all','--no-trunc','--quiet'])[1].split();containers={}
        require(all(re.fullmatch(r'[0-9a-f]{64}',x) for x in cids),'unrecognized container ID')
        for cid in cids:
            row=json.loads(self.command(self.d+['container','inspect','--format','{"Image":{{json .Image}},"Running":{{json .State.Running}}}',cid])[1])
            require(isinstance(row['Running'],bool),'container running state unavailable');containers[cid]=row
        return {'images':images,'cache_ids':sorted(set(cache)),'cache_in_use':cache_in_use,'containers':containers}
    def snapshot_assets(self):
        require(self.assets_before is None and not self.pull_attempted,'asset baseline cannot be replaced')
        try:self.assets_before=self.asset_inventory()
        except BaseException as error:
            self.foreign_asset_activity=True
            self.asset_receipt['snapshot_error']=type(error).__name__+': '+str(error)
            raise
        self.asset_receipt['before']=self.assets_before
        conflict=self.assets_before['cache_in_use'] or any(x['Running'] for x in self.assets_before['containers'].values())
        if conflict:self.foreign_asset_activity=True
        require(not conflict,'foreign running/build work before pull')
    def cleanup_owned_image(self):
        # Always invoked before quit; failed ownership/budget never authorizes a
        # broad prune, force removal, deleting layers or touching shared caches.
        if self.assets_before is None:
            require(not self.pull_attempted,'pull without complete ownership baseline')
            self.asset_receipt['cleanup']='no pull/baseline; no image ownership claimed'
            return
        if self.cleanup_until is None or self.now()>=self.cleanup_until:
            self.foreign_asset_activity=True
            raise RuntimeError('asset cleanup budget absent/exhausted; no commands or ownership claims')
        try:
            try:current=self.asset_inventory()
            except BaseException:
                self.foreign_asset_activity=True  # Cannot prove that a concurrent builder/user is absent.
                raise
            self.asset_receipt['cleanup_before']=current
            before=self.assets_before;new_ids=set(current['images'])-set(before['images'])
            conflict=(current['cache_in_use'] or not set(before['images'])<=set(current['images']) or
                      set(current['cache_ids'])!=set(before['cache_ids']) or
                      set(current['containers'])-set(before['containers']) or
                      any(x['Running'] for x in current['containers'].values()))
            if conflict:
                self.foreign_asset_activity=True
                raise RuntimeError('foreign container/cache activity; preserve all unowned work')
            if not self.pull_attempted:
                if new_ids:self.foreign_asset_activity=True
                require(not new_ids,'unexpected new image without owned pull')
                self.asset_receipt.update(after=current,cleanup='preexisting images preserved',no_new_build_cache=True)
                return
            code,out,err=self.command(self.d+['image','inspect',self.p['base_image']],check=False)
            if code:
                require('No such image' in err or 'No such object' in err,'carrier ownership lookup unavailable')
                if new_ids:self.foreign_asset_activity=True
                raise RuntimeError('partial pull: no exact carrier identity; partial blob retirement UNVERIFIED; no prune')
            target=json.loads(out)[0];verify_carrier(target,self.p);image_id=target['Id']
            require(re.fullmatch(r'sha256:[0-9a-f]{64}',image_id) is not None,'invalid carrier content ID')
            if image_id in before['images']:
                if new_ids:self.foreign_asset_activity=True
                require(not new_ids,'unexpected image alongside preexisting carrier')
                self.asset_receipt.update(after=current,cleanup='carrier content ID preexisted; preserved',no_new_build_cache=True)
                return
            if new_ids!={image_id}:self.foreign_asset_activity=True
            require(new_ids=={image_id},'new image ownership ambiguous; preserve unowned images')
            if target.get('RepoTags'):self.foreign_asset_activity=True
            require(not target.get('RepoTags'),'new carrier gained tags; foreign image use possible')
            expected=self.p['base_image'].split('@',1)[1]
            allowed={prefix+'@'+expected for prefix in ('python','library/python','docker.io/library/python')}
            if not set(target.get('RepoDigests',[]) or [])<=allowed:self.foreign_asset_activity=True
            require(set(target.get('RepoDigests',[]) or [])<=allowed,'foreign reference to new carrier; no removal')
            if any(x['Image']==image_id for x in current['containers'].values()):self.foreign_asset_activity=True
            require(not any(x['Image']==image_id for x in current['containers'].values()),'carrier used by foreign/stopped container; preserve it')
            self.owned_image_id=image_id;self.asset_receipt['owned_image_id']=image_id
            self.asset_receipt['ownership_basis']={'exact_pinned_ref':self.p['base_image'],'platform':'linux/amd64','new_since_before_pull':True,'exclusive_desktop_lease_required':True}
            self.asset_receipt['shared_layers_preserved']=sorted(set(target.get('RootFS',{}).get('Layers',[])) & {x for row in before['images'].values() for x in (row.get('Layers') or [])})
            removal_error=None
            try:self.command(self.d+['image','rm',image_id])  # Exact ID; NEVER --force.
            except BaseException as error:removal_error=type(error).__name__+': '+str(error)
            self.asset_receipt['removal_error']=removal_error
            try:after=self.asset_inventory()
            except BaseException:
                self.foreign_asset_activity=True
                raise
            self.asset_receipt['after']=after
            # Classify post-removal foreign use BEFORE failing owned absence.
            # A failed rm can be the first evidence that another user acquired it.
            if (after['cache_in_use'] or set(after['cache_ids'])!=set(before['cache_ids']) or
                not set(before['images'])<=set(after['images']) or set(after['images'])-(set(before['images'])|{image_id}) or
                set(after['containers'])-set(before['containers']) or any(x['Running'] for x in after['containers'].values())):
                self.foreign_asset_activity=True
                raise RuntimeError('asset namespace changed during retirement; do not quit shared work')
            require(image_id not in after['images'],'owned image absence not proven')
            require(removal_error is None,'image removal command failed; absence observation retained but mandatory cleanup NOT successful')
            self.asset_receipt.update(cleanup='new exact carrier ID removed; baseline images/cache IDs preserved',no_new_build_cache=True,
                                      storage_caveat='Logical image removal only; shared layers retained; no physical sparse-disk reclamation claim')
        except BaseException as error:
            self.asset_receipt['cleanup_error']=type(error).__name__+': '+str(error)
            raise
    def cached_carrier(self):
        require(not self.command(self.d+['ps','-q'])[1].strip(),'foreign auto-started workloads; no pull/resolver')
        code,out,err=self.command(self.d+['image','inspect',self.p['base_image']],check=False)
        if code:
            require('No such image' in err or 'No such object' in err,'daemon unavailable, not a cache miss')
            return None
        return json.loads(out)[0]
    def pull(self):
        # Caller already verified explicit FULL-scope optional-pull signoff.
        require(self.assets_before is not None and not self.pull_attempted,'pull requires immutable asset baseline; no retry')
        self.pull_attempted=True;self.asset_receipt.update(pull_attempted=True,pull_command_completed=False,exact_pull_ref=self.p['base_image'])
        self.command(self.d+['pull','--platform=linux/amd64',self.p['base_image']],timeout=60)
        self.pull_complete=True;self.asset_receipt['pull_command_completed']=True
        self.check_disk()
    def wheel_hash(self):
        path=self.root/'artifacts'/self.p['uv_wheel']['filename']
        require(path.is_file(),'staged uv wheel missing; no implicit download')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def begin_resolver(self):
        registered=validated_registered_root(str(self.root))
        # Refuse foreign auto-restarted workloads rather than stopping their work.
        require(not self.command(self.d+['ps','-q'])[1].strip(),'foreign running containers; resolver forbidden')
        code=(HERE/'RESOLVER-COMMAND.md').read_text().split('```python\n',1)[1].split('\n```',1)[0]
        path=self.directory/'resolver-supervisor.py';path.write_text(code)
        env=dict(anonymous_cli_environment(self.directory),BENCH_REGISTERED_ROOT=str(registered),BENCH_LIFECYCLE_DIR=str(self.directory),PYTHONPATH=str(HERE))
        blocked={signal.SIGTERM,signal.SIGINT};mask=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
        try:
            self.resolver=subprocess.Popen([sys.executable,'-B',str(path)],cwd=HERE,env=env,stdin=subprocess.DEVNULL,start_new_session=True,
                                           preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,blocked))
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    def resolver_done(self):
        code=self.resolver.poll()
        if code is None:return False
        require(code==0,'resolver failed');return True
    def begin_cleanup(self):
        signal.alarm(0)  # Work alarm must not re-enter termination during cleanup.
        self.cleanup_until=self.now()+60
    def recover_owned_container(self):
        # Recovery is attempted even after monitor/client failures, but it may
        # never exceed the independent cleanup budget or touch an unknown CID.
        cid=self.root/'resolver/linux-amd64-01/container.cid'
        if self.resolver is None or not cid.exists():return
        require(self.cleanup_until is not None,'cleanup budget not established; no unbounded recovery commands')
        owned=cid.read_text().strip()
        require(re.fullmatch(r'[0-9a-f]{64}',owned) is not None,'invalid owned CID; no broad cleanup')
        removal_error=None
        try:self.command(self.d+['rm','--force',owned],check=False)
        except BaseException as error:removal_error=type(error).__name__+': '+str(error)
        # A timed-out remove is not proof either way. Always try authoritative
        # absence verification; command() still enforces remaining cleanup time.
        code,_,err=self.command(self.d+['container','inspect',owned],check=False)
        absent=code!=0 and ('No such object' in err or 'No such container' in err)
        self.samples.append({'phase':'owned-CID-recovery','absence_verified':absent,'remove_error':removal_error})
        require(absent,'owned container absence not proven')
    def stop_owned_resolver(self):
        failures=[]
        try:
            if self.resolver is not None and self.resolver.poll() is None:
                try:os.killpg(self.resolver.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                # Existing 60 s cleanup envelope: reserve 5 s for reap, two
                # 15 s CID calls and 15 s for Desktop retirement BEFORE granting
                # client grace. These are budget allocations, not latency claims.
                require(self.cleanup_until is not None,'cleanup budget not established')
                limit=min(self.now()+45,self.cleanup_until-50)
                try:
                    while self.resolver.poll() is None and self.now()<limit:
                        self.check_cleanup_resources()
                        if self.now()>=self.next_rss:
                            require(self.vm_helper(self.owner)['rss_bytes']<=CAP,'VM helper RSS over cap during cleanup')
                        self.sleep(0.5)
                finally:
                    if self.resolver.poll() is None:
                        try:os.killpg(self.resolver.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    self.resolver.wait(timeout=max(0.001,min(5,self.cleanup_until-self.now())))
        except BaseException as error:failures.append(type(error).__name__+': '+str(error))
        finally:
            # A failed monitor or expired WORK lease must not skip CID recovery.
            self.monitor_enabled=False
            try:self.recover_owned_container()
            except BaseException as error:failures.append(type(error).__name__+': '+str(error))
        require(not failures,'owned cleanup incomplete/monitor failed: '+'; '.join(failures))
    def quit_owned(self,owner):
        # No bundle-wide quit, pkill, settings route or unknown process takeover.
        self.monitor_enabled=False
        require(not self.foreign_asset_activity,'foreign asset activity; Desktop quit not safe')
        rows=self.snapshot();current={p['pid']:p for p in rows}
        require(self.owner==owner and not self.existing_desktop(),'ownership was not established before startup')
        pid=owner['pid'];row=current.get(pid)
        if row is not None:
            require(self.proc is not None and self.proc.poll() is None,'original Desktop startup handle is no longer live; no PID takeover')
            require((row['start'],row['exe'])==(owner['start'],owner['exe']),'Desktop PID identity changed')
        # Unknown/foreign Docker workloads make global daemon shutdown unsafe.
        code,out,_=self.command(self.d+['ps','-q'],check=False)
        require(code==0 and not out.strip(),'foreign workloads or daemon unavailable; automatic Desktop quit not proven safe')
        owned=[]
        for p in rows:
            key=(p['pid'],p['start'],p['exe']);ancestor=p['pid'];seen=set()
            while ancestor in current and ancestor not in seen:
                seen.add(ancestor)
                if ancestor==pid:
                    owned.append(p);break
                ancestor=current[ancestor]['ppid']
            else:
                if key in self.helpers:owned.append(p)
        unknown=[p for p in self.desktop_records(rows) if p not in owned]
        require(not unknown,'unattributed Desktop processes; never kill other work')
        for p in reversed(owned):
            fresh=next((x for x in self.snapshot() if x['pid']==p['pid']),None)
            if fresh is None:continue
            require((fresh['start'],fresh['exe'])==(p['start'],p['exe']),'owned process identity changed')
            os.kill(p['pid'],signal.SIGTERM)
        if self.proc is not None:self.proc.wait(timeout=max(0.001,min(15,self.cleanup_until-self.now())))
        final=self.snapshot();live={(p['pid'],p['start'],p['exe']) for p in final}
        require(not any((p['pid'],p['start'],p['exe']) in live for p in owned),'owned Desktop/helper survived cleanup')
        require(not self.desktop_records(final),'new/unattributed Desktop process survived; never kill it by name')
        for process in final:
            if Path(process['exe']).name in HELPERS:
                code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(process['pid']),'-Fn'],check=False,timeout=3)
                require(code==0 and ('n'+str(self.raw)) not in out.splitlines(),'VM shutdown association unverified; do not claim cleanup')
    def record(self,value):
        if self.directory is None:return
        final_swap_error=None
        try:self.sample_swap('final')
        except BaseException as error:final_swap_error=type(error).__name__+': '+str(error);value['status']='swap-final-unverified-or-over-limit'
        value.update(asset_ownership=self.asset_receipt,final_swap_error=final_swap_error,desktop_layout=getattr(self,'desktop_layout',None))
        fs=os.statvfs(self.root)
        if self.startup_uncertain:value['status']='cleanup-unverified-startup-identity'
        value.update(operator_ack_quote=ACK_QUOTE,source_head=self.a['full_scope_review']['source_commit'],samples=self.samples,after={'allocated_bytes':self.raw.stat().st_blocks*512 if self.raw.exists() else 0,'free_bytes':fs.f_bavail*fs.f_frsize})
        breached=value['after']['allocated_bytes']-self.baseline>5*1024**3 or value['after']['free_bytes']<25*1024**3
        if breached:value['status']='disk-fence-failed-after-cleanup'
        with (self.directory/'receipt.json').open('x') as out:out.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
        require(not breached,'final disk fence crossed; retained failed receipt')
        require(final_swap_error is None,'final swap gate failed; retained failed receipt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--admission',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print('PROPOSAL ONLY: no fresh slot/full-scope approval; no execution');return
    require(args.admission is not None,'explicit desk admission path required')
    require(args.admission.resolve()==(Path.home()/'benchmark-runs/reconstruction-20261114-source-v1/desk-admission.json').resolve(),'only the registered desk admission path is accepted')
    admission=json.loads(args.admission.read_text());pins=json.loads((HERE/'resolver-pins.json').read_text())
    def terminate(signum,frame):raise KeyboardInterrupt('termination requested')
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    for s in previous:signal.signal(s,terminate)
    signal.alarm(840)  # Work deadline also interrupts blocked host-side queries.
    try:run_lifecycle(NativeOps(admission,pins),admission,pins)
    finally:
        signal.alarm(0)
        for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':main()

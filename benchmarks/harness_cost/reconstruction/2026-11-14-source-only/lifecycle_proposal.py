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
        self.desktop=Path(admission.get('desktop_executable','/Applications/Docker.app/Contents/MacOS/Docker Desktop'))
        self.app_root='/Applications/Docker.app/'
        self.raw=Path(admission.get('docker_raw_path','/INVALID'))
        self.deadline=0;self.next_rss=0;self.samples=[];self.helpers={};self.before=[]
        self.clients=[];self.monitor_enabled=False;self.monitoring=False;self.startup_uncertain=False;self.cleanup_until=None
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
        require(str(self.desktop) in {'/Applications/Docker.app/Contents/MacOS/Docker Desktop','/Applications/Docker.app/Contents/MacOS/Docker'},'unapproved Desktop executable')
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
    def check_disk(self):
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
        current=self.raw.stat();fs=os.statvfs(self.root)
        require(self.raw_identity is None or self.raw_identity==(current.st_dev,current.st_ino),'cleanup raw-file identity changed; still retire owned container')
        sample={'phase':'cleanup','allocated_bytes':current.st_blocks*512,'free_bytes':fs.f_bavail*fs.f_frsize}
        self.samples.append(sample)
        require(sample['allocated_bytes']-self.baseline<=5*1024**3 and sample['free_bytes']>=25*1024**3,'cleanup resource fence crossed; still retire owned container')
    def cached_carrier(self):
        require(not self.command(self.d+['ps','-q'])[1].strip(),'foreign auto-started workloads; no pull/resolver')
        code,out,err=self.command(self.d+['image','inspect',self.p['base_image']],check=False)
        if code:
            require('No such image' in err or 'No such object' in err,'daemon unavailable, not a cache miss')
            return None
        return json.loads(out)[0]
    def pull(self):
        # Caller already verified explicit FULL-scope optional-pull signoff.
        self.command(self.d+['pull','--platform=linux/amd64',self.p['base_image']],timeout=60)
        self.check_disk()
    def wheel_hash(self):
        path=self.root/'artifacts'/self.p['uv_wheel']['filename']
        require(path.is_file(),'staged uv wheel missing; no implicit download')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def begin_resolver(self):
        # Refuse foreign auto-restarted workloads rather than stopping their work.
        require(not self.command(self.d+['ps','-q'])[1].strip(),'foreign running containers; resolver forbidden')
        code=(HERE/'RESOLVER-COMMAND.md').read_text().split('```python\n',1)[1].split('\n```',1)[0]
        path=self.directory/'resolver-supervisor.py';path.write_text(code)
        env=dict(anonymous_cli_environment(self.directory),BENCH_LIFECYCLE_DIR=str(self.directory),PYTHONPATH=str(HERE))
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
        fs=os.statvfs(self.root)
        if self.startup_uncertain:value['status']='cleanup-unverified-startup-identity'
        value.update(operator_ack_quote=ACK_QUOTE,source_head=self.a['full_scope_review']['source_commit'],samples=self.samples,after={'allocated_bytes':self.raw.stat().st_blocks*512 if self.raw.exists() else 0,'free_bytes':fs.f_bavail*fs.f_frsize})
        breached=value['after']['allocated_bytes']-self.baseline>5*1024**3 or value['after']['free_bytes']<25*1024**3
        if breached:value['status']='disk-fence-failed-after-cleanup'
        with (self.directory/'receipt.json').open('x') as out:out.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
        require(not breached,'final disk fence crossed; retained failed receipt')


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

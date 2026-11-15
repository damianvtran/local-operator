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

from launch_services_broker import CausalLaunch
from owned_session_launch import OwnedSessionLaunch, KIND as SESSION_KIND
from darwin_process_identity import DarwinIdentity
import final_attempt_admission as final_admission

HERE = Path(__file__).resolve().parent
ACK = 'a-7072'
ACK_QUOTE = "Find a way around this that doesn't require my manual involvement, validate in other ways or skip that validation with acknowledgement but make sure the code analysis and review is thorough"
# Configured VM capacity is NOT resident usage: the operator accepted 8.2GB/14CPU,
# while the desk retained the original4.3GB attributed-RSS abort threshold.
CAP = 4_300_000_000
CONFIGURED_MEMORY_CAP = 8_600_000_000
CONFIGURED_CPU_CAP = 14
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


def validated_admission_path(raw,root,attempt):
    require(re.fullmatch(r'linux-amd64-[0-9]{2}',attempt) is not None,'invalid admission namespace')
    path=Path(raw);expected=root/'admissions'/(attempt+'.json')
    require(path==expected and path.is_file() and path.resolve(strict=True)==path,'unregistered/escaping admission path')
    require(path.stat().st_uid==os.getuid(),'admission not owned by operator')
    return path


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
    require(type(info) is dict, 'malformed configured docker info')
    memory, cpus = info.get('MemTotal'), info.get('NCPU')
    require(type(memory) is int and type(cpus) is int and 0 < memory <= CONFIGURED_MEMORY_CAP and 0 < cpus <= CONFIGURED_CPU_CAP, 'observed docker info exceeds configured8.6 GB/14 CPUs or is invalid')
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
            if hasattr(ops,'cleanup_pending_launch'):
                try:ops.cleanup_pending_launch()
                except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
            owner=owner or getattr(ops,'owner',None)
            for action in ([ops.stop_owned_resolver] if started else []):
                try: action()
                except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
            cleanup_authority=owner is not None or (hasattr(ops,'has_cleanup_authority') and ops.has_cleanup_authority())
            if cleanup_authority:
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
    def __init__(self, admission, pins, final_permit=None):
        self.final_permit=final_permit
        self.a=admission;self.p=pins;self.owner=None;self.proc=None;self.resolver=None;self.directory=None
        self.root=Path.home()/'benchmark-runs/reconstruction-20261114-source-v1'
        self.desktop=Path(admission.get('desktop_executable',str(DEFAULT_DESKTOP)))
        self.app_root='/Applications/Docker.app/'
        self.raw=Path(admission.get('docker_raw_path','/INVALID'))
        self.deadline=0;self.next_rss=0;self.samples=[];self.helpers={};self.before=[]
        self.clients=[];self.monitor_enabled=False;self.monitoring=False;self.startup_uncertain=False;self.cleanup_until=None
        self.assets_before=None;self.asset_receipt={'policy':'exact new carrier ID only; never force/prune/layer/cache removal'}
        self.pull_attempted=False;self.pull_complete=False;self.owned_image_id=None;self.foreign_asset_activity=False
        self.broker=None;self.owned_processes={};self.ownership_edges=[];self.verified_process_paths=set();self.startup_deadline=0
        self.startup_active=False;self.startup_guard_running=False
        self.identity_proof_depth=0;self.ownership_scan_depth=0;self.resource_guard_running=False
        self.kernel_identity=None;self.identity_proof_errors=[]
        self.daemon_observed_available=False;self.engine_history_unknown=False;self.engine_query_inflight=False;self.cleanup_engine_evidence={};self.ownership_journal_errors=[]
    def now(self): return time.monotonic()
    def sleep(self,seconds): time.sleep(seconds)
    def command(self,args,timeout=15,check=True):
        # Only own CLI child groups are reaped; never select a PID by a name.
        if self.cleanup_until is not None:
            timeout=min(timeout,self.cleanup_until-self.now())
            require(timeout>0,'owned cleanup budget exhausted; do not claim clean shutdown')
        if getattr(self,'startup_active',False):
            timeout=min(timeout,self.startup_deadline-self.now(),self.deadline-self.now())
            require(timeout>0,'startup/work command deadline exhausted')
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
                    if getattr(self,'startup_active',False):
                        if self.identity_proof_depth or self.ownership_scan_depth or self.startup_guard_running or self.resource_guard_running:self.startup_resource_guard()
                        else:self.startup_guard()
                    if self.cleanup_until is None and not getattr(self,'startup_active',False) and self.monitor_enabled and not self.monitoring:
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
        direct=self.p.get('launch_authority')==SESSION_KIND
        if direct:
            if self.kernel_identity is None:self.kernel_identity=DarwinIdentity()
            _,out,_=self.command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,state=,lstart=,comm='])
            result=[]
            creator=self.broker.creator.pid if self.broker and getattr(self.broker,'kind',None)==SESSION_KIND and self.broker.creator and not self.broker.released else None
            for line in out.splitlines():
                fields=line.split(None,10)
                if len(fields)!=11:continue
                row={'pid':int(fields[0]),'ppid':int(fields[1]),'pgid':int(fields[2]),'rss_kib':int(fields[3]),'state':fields[4],'start':' '.join(fields[5:10]),'exe':fields[10]}
                if row['pid']<=0:continue
                if row['state'].startswith('Z') and row['pid']!=creator:continue  # No live authority; exact absence is checked after creator reap.
                if row['pid']==creator and row['state'].startswith('Z'):
                    row.update(sid=creator,exe=self.broker.executable)  # Held unreaped creator, never a discovered group.
                    if self.broker.ticket.get('birth'):row['start']=self.broker.ticket['birth']
                    result.append(row);continue
                try:row['sid']=os.getsid(row['pid'])
                except ProcessLookupError:
                    if row['pid']==creator:row['sid']=creator  # Reserved creator may have just become zombie.
                    else:continue
                except PermissionError:row['sid']=None
                relevant=row['pid']==creator or row['exe'].startswith(self.app_root) or (creator and row['sid']==creator) or row['pid'] in self.owned_processes
                if relevant:
                    if row['pid']==creator:identity=self.kernel_identity.record(row['pid'],creator_sid=creator,creator_executable=self.broker.executable)
                    else:identity=self.kernel_identity.record(row['pid'])
                    if identity is None:continue
                    row.update(identity)
                result.append(row)
            return result
        _,out,_=self.command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,lstart=,comm='])
        result=[]
        for line in out.splitlines():
            fields=line.split(None,9)
            if len(fields)!=10: continue
            result.append({'pid':int(fields[0]),'ppid':int(fields[1]),'pgid':int(fields[2]),'rss_kib':int(fields[3]),'start':' '.join(fields[4:9]),'exe':fields[9]})
        return result
    def desktop_records(self, rows):
        return [p for p in rows if p['exe'].startswith(self.app_root) or Path(p['exe']).name in {'Docker Desktop','com.docker.backend'}]
    def preflight(self):
        import runpy
        gate=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))
        self.source_generation=gate['validate_selection'](self.a.get('_source_generation'),self.a.get('_source_generation_sha256'))
        final_admission.validate_controller(getattr(self,'final_permit',None),starting=True)
        validated_registered_root(str(self.root))  # Reject bad roots before any startup/work.
        self.admission_path=validated_admission_path(self.a['_admission_path'],self.root,self.a['attempt_id'])
        require(self.p.get('r15_topology_admission',{}).get('ready') is True and self.p.get('launch_authority')==SESSION_KIND,'R15 actual ancestor/sibling topology remains unproven without reviewed created-session authority')
        require(self.a.get('direct_backend_ui_effects')=='acknowledged-unqualified-default-backend','direct backend UI effects must be explicitly acknowledged; no headless/prompt claim')
        self.sample_swap('PRE-startup')  # No Desktop starts when swap evidence fails.
        require(self.a.get('operator_ack_id')==ACK and self.a.get('operator_ack_quote')==ACK_QUOTE,'operator acknowledgement missing')
        require(self.a.get('prestart_cap_validation')=='skipped-with-operator-ack' and 'vm_caps_verified_before_start' not in self.a,'must not fabricate pre-start verification')
        now=datetime.datetime.now(datetime.UTC)
        begin=datetime.datetime.fromisoformat(self.a['slot_start_utc']);end=datetime.datetime.fromisoformat(self.a['slot_end_utc'])
        require(begin <= now and now+datetime.timedelta(seconds=960) <= end and now <= datetime.datetime.fromisoformat(self.a['valid_until_utc']),'no fresh explicit slot/lease')
        require(self.a.get('fresh_memory_go') is True and 0 <= self.a['fleet_mb'] <= 26000,'fleet/memory HOLD')
        require(self.a.get('exclusive_desktop_lease') is True,'exclusive Desktop lease not granted')
        review=self.a['full_scope_review']
        require(review.get('clean') is True and review.get('source_generation_sha256')==self.source_generation['overlay_sha256'] and review['reviewer']!='70ba7562e541','full-scope clean independent source-generation review missing')
        for name in ('lifecycle_proposal.py','RESOLVER-COMMAND.md','resolver-pins.json','launch_services_broker.py','owned_session_launch.py','darwin_process_identity.py'):
            require(review.get('file_sha256',{}).get(name)==hashlib.sha256((HERE/name).read_bytes()).hexdigest(),'reviewed file hash mismatch')
        self.desktop_layout=validate_desktop_layout(self.desktop,self.command)
        require(self.root.is_dir() and self.a.get('docker_raw_path') and self.raw.is_absolute() and self.raw.name=='Docker.raw','registered root/approved raw path missing')
        fs=os.statvfs(self.root);require(fs.f_bavail*fs.f_frsize >= 30*1024**3,'fresh free disk below 30 GiB')
        initial=self.raw.stat() if self.raw.exists() else None
        self.baseline=initial.st_blocks*512 if initial else 0
        self.raw_identity=(initial.st_dev,initial.st_ino) if initial else None
        require(not Path('/var/run/docker.sock').exists(),'daemon socket target already exists; no foreign/stale engine takeover')
        if self.p.get('launch_authority')==SESSION_KIND:
            self.kernel_identity=DarwinIdentity()
            require(self.kernel_identity.record(os.getpid()) is not None,'read-only kernel identity binding unsupported BEFORE startup')
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
        with (self.directory/'pre-start-processes.json').open('x') as out:out.write(json.dumps(self.before,sort_keys=True))
        (self.directory/'docker-config').mkdir()
        (self.directory/'docker-home').mkdir()
        write_anonymous_cli_config(self.directory/'docker-config')
        self.d=['docker','--config',str(self.directory/'docker-config'),'--host','unix:///var/run/docker.sock']
        self.deadline=self.now()+840
        self.samples.append({'phase':'PRE-startup','allocated_bytes':self.baseline,'free_bytes':fs.f_bavail*fs.f_frsize})
        require(re.fullmatch(r'linux-amd64-[0-9]{2}',self.a['attempt_id']) is not None,'invalid exclusive resolver attempt namespace')
        require(not (self.root/'resolver'/self.a['attempt_id']).exists(),'resolver attempt already exists; no reuse or deletion')
    def existing_desktop(self): return bool(self.desktop_records(self.before))
    def startup_event(self,event):
        row=dict(event,recorded_at_utc=datetime.datetime.now(datetime.UTC).isoformat())
        with (self.directory/'startup-events.jsonl').open('a') as out:
            out.write(json.dumps(row,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
    def has_cleanup_authority(self):
        # This is retained causal proof, not a substitute timestamp/singleton.
        if self.broker is None or self.broker.ticket is None:return False
        if getattr(self.broker,'kind',None)==SESSION_KIND:return self.broker.matches(self.broker.ticket)
        ticket=self.broker.ticket;known=self.owned_processes.get(ticket['pid'])
        return bool(known and known['exe']==ticket['executable'] and known['pid'] not in {p['pid'] for p in self.before})
    def ownership_event(self,event):
        try:self.startup_event(event)
        except BaseException as error:
            self.ownership_journal_errors.append({'event':event,'error':type(error).__name__+': '+str(error)})
            # Work must stop on failed durability, but teardown must continue
            # from already validated in-memory authority. Never call it clean.
            if self.cleanup_until is None:raise
    def verify_owned_path(self,path):
        p=Path(path);require(p.is_file() and p.resolve(strict=True)==p and path.startswith(self.app_root),'untrusted/escaping owned process path')
        if path not in self.verified_process_paths:
            self.identity_proof_depth+=1
            try:self.command(['/usr/bin/codesign','--verify','--strict','-R','=anchor apple generic and certificate leaf[subject.OU] = "9BNSXJN65R"',path])
            finally:self.identity_proof_depth-=1
            self.verified_process_paths.add(path)
    def observe_owned(self):
        require(self.ownership_scan_depth==0,'recursive ownership proof refused')
        self.ownership_scan_depth+=1
        try:return self._observe_owned()
        finally:self.ownership_scan_depth-=1
    def verify_session_path(self,path):
        if path.startswith(self.app_root):return self.verify_owned_path(path)
        # A directly launched backend may run transient Apple platform helpers.
        # Session ownership is already kernel-proven; qualification is still
        # bounded/canonical and never admits a foreign-session Apple process.
        roots=('/usr/bin/','/usr/sbin/','/bin/','/sbin/','/System/Library/','/usr/libexec/')
        p=Path(path);require(path.startswith(roots) and p.is_file() and p.resolve(strict=True)==p,'unexpected executable in owned session')
        if path not in self.verified_process_paths:
            self.identity_proof_depth+=1
            try:self.command(['/usr/bin/codesign','--verify','--strict','-R','=anchor apple',path])
            finally:self.identity_proof_depth-=1
            self.verified_process_paths.add(path)
    def _observe_owned_session(self):
        # One process closure supplies SID, microsecond birth, path and parent
        # edges. Membership is inherited from OUR setsid, never an LS ancestor.
        rows=self.snapshot();root=self.broker.bind_birth(rows);ticket=self.broker.ticket
        old={p['pid'] for p in self.before};by_pid={p['pid']:p for p in rows}
        session=[p for p in rows if p.get('sid')==ticket['sid']]
        # setsid starts with only OUR child; other sessions cannot join it.
        # Creator is held unreaped, so its SID/PID cannot be recycled.
        for pid,known in self.owned_processes.items():
            current=by_pid.get(pid)
            if current is None:continue
            require(current['start']==known['start'],'owned PID birth reused')
            if not current.get('state','').startswith('Z'):require(current['exe']==known['exe'],'owned PID executable changed')
            if current.get('sid')!=ticket['sid']:
                # Escapes have no blanket session authority. Retain only a
                # presently observed parent edge to a still-owned session member.
                parent=by_pid.get(current['ppid'])
                require(parent is not None and parent.get('sid')==ticket['sid'] and parent['pid'] in self.owned_processes,'unobserved session escape; no adoption')
        events=[]
        for row in session:
            known=dict(row)
            if row['pid'] not in self.owned_processes:
                known['authority']=SESSION_KIND;self.owned_processes[row['pid']]=known
                events.append({'event':'created-session-member','process':known})
        # The created PID/session is unconditional OWNERSHIP, not executable
        # qualification. A bad signature must abort work, not leak our child.
        if self.owner is None:self.owner=self.owned_processes[ticket['pid']]
        for row in session:
            if row.get('state','').startswith('Z'):continue
            if self.cleanup_until is None:
                try:self.verify_session_path(row['exe'])
                except BaseException as error:
                    self.identity_proof_errors.append(type(error).__name__+': '+str(error));raise
        changed=True
        while changed:
            changed=False
            for row in rows:
                parent=by_pid.get(row['ppid'])
                if parent is None or parent['pid'] not in self.owned_processes:continue
                if parent['start']!=self.owned_processes[parent['pid']]['start']:continue
                if row['pid'] not in self.owned_processes:
                    if row['pid'] in old or not row['exe'].startswith(self.app_root):continue
                    self.verify_owned_path(row['exe'])  # Escaped/other-session edge needs signed exact path.
                    self.owned_processes[row['pid']]=dict(row,authority='observed-parent-edge');changed=True
                if not any(e['child']['pid']==row['pid'] and e['child']['start']==row['start'] for e in self.ownership_edges):
                    self.ownership_edges.append({'parent':parent,'child':row})
        clients={p.pid for p in self.clients if p.poll() is None}
        unknown=[p for p in self.desktop_records(rows) if p['pid'] not in self.owned_processes and p['pid'] not in clients]
        require(not unknown,'foreign/unobserved Docker process outside created session')
        for event in events:self.ownership_event(event)
        live=[p for p in rows if p['pid'] in self.owned_processes and not p.get('state','').startswith('Z')]
        return rows,live
    def _observe_owned(self):
        if getattr(self.broker,'kind',None)==SESSION_KIND:return self._observe_owned_session()
        require(self.broker is not None and self.broker.ticket is not None,'no causal launch ticket')
        rows=self.snapshot();by_pid={p['pid']:p for p in rows};old={p['pid'] for p in self.before}
        ticket=self.broker.ticket;root=by_pid.get(ticket['pid'])
        if root is not None:
            require(self.broker.driver.matches(ticket),'retained application instance no longer matches root')
            require(root['pid'] not in old and root['exe']==ticket['executable'],'old/foreign launch-ticket process')
            if root['pid'] not in self.owned_processes:
                self.verify_owned_path(root['exe']);self.owned_processes[root['pid']]=root
                if self.owner is None:self.owner=root  # Before ANY fallible journal I/O.
                self.ownership_event({'event':'causal-root-observed','process':root,'ticket':ticket})
        for pid,known in self.owned_processes.items():
            current=by_pid.get(pid)
            require(current is None or (current['start'],current['exe'])==(known['start'],known['exe']),'known owned PID reused; no adoption')
        changed=True
        while changed:
            changed=False
            for row in rows:
                if row['pid'] in self.owned_processes or row['pid'] in old or not row['exe'].startswith(self.app_root):continue
                parent=by_pid.get(row['ppid'])
                if parent is not None and parent['pid'] in self.owned_processes:
                    self.verify_owned_path(row['exe']);self.owned_processes[row['pid']]=row;changed=True
                    self.ownership_edges.append({'parent':parent,'child':row})
                    self.ownership_event({'event':'observed-owned-edge','parent':parent,'child':row})
        clients={p.pid for p in self.clients if p.poll() is None}
        unknown=[p for p in self.desktop_records(rows) if p['pid'] not in self.owned_processes and p['pid'] not in clients]
        require(not unknown,'foreign/unobserved Docker process; uniqueness/time is NOT ownership')
        live=[by_pid[pid] for pid in self.owned_processes if pid in by_pid]
        return rows,live
    def effective_owner(self,live):
        if getattr(self.broker,'kind',None)==SESSION_KIND:
            require(bool(live),'created session has no live backend/helper')
            original=next((p for p in live if p['pid']==self.broker.ticket['pid']),None)
            return original or min(live,key=lambda p:p['pid'])  # Role only; authority remains the created SID.
        original=next((p for p in live if p['pid']==self.broker.ticket['pid']),None)
        if original:return original
        # Only an ACTUALLY observed successor may survive reparenting. A new
        # singleton group without a recorded edge is never adopted.
        candidates=[p for p in live if p['pid']==p.get('pgid') and p['exe']==self.broker.ticket['executable']]
        require(len(candidates)==1,'no unique causally observed live handoff root')
        return candidates[0]
    def startup_guard(self):
        previous=self.monitoring;self.monitoring=True;self.startup_guard_running=True
        try:self._startup_guard()
        finally:self.monitoring=previous;self.startup_guard_running=False
    def startup_time_guard(self):
        now=datetime.datetime.now(datetime.UTC)
        require(self.now()<self.startup_deadline and self.now()<self.deadline,'startup/work deadline exhausted')
        require(now+datetime.timedelta(seconds=60)<datetime.datetime.fromisoformat(self.a['slot_end_utc']),'slot must retain cleanup reserve')
        require(now<=datetime.datetime.fromisoformat(self.a.get('valid_until_utc',self.a['slot_end_utc'])),'admission expired during startup')
    def startup_resource_guard(self):
        self.startup_time_guard()
        # An already in-flight resource query keeps command's independent
        # deadline. Never recursively fork another sysctl/ps query from it.
        if self.resource_guard_running:return
        self.resource_guard_running=True
        try:
            self.sample_swap('startup');self.check_disk()
            if self.identity_proof_depth or self.ownership_scan_depth:
                rows=self.snapshot();old={p['pid'] for p in self.before}
                emerging=[p for p in rows if Path(p['exe']).name in HELPERS and p['pid'] not in old]
                # Abort-only ceiling, NOT ownership or a qualified VM peak.
                # No provisional owner goes into vm_helper or cleanup.
                require(all(p['rss_kib']*1024<=CAP for p in emerging),'new VM RSS exceeds cap while identity proof is pending')
        finally:self.resource_guard_running=False
    def _startup_guard(self):
        self.startup_resource_guard()
        if self.broker and self.broker.ticket and not self.identity_proof_depth and not self.ownership_scan_depth:
            rows,live=self.observe_owned();owner=self.effective_owner(live)
            helpers=[p for p in rows if Path(p['exe']).name in HELPERS and p['pid'] not in {x['pid'] for x in self.before}]
            require(all(p['pid'] in self.owned_processes for p in helpers),'unowned/unknown new VM helper during startup')
            require(all(p['rss_kib']*1024<=CAP for p in helpers),'new startup VM RSS over cap')
            if helpers and (not self.monitor_enabled or self.now()>=self.next_rss):
                helper=self.vm_helper(owner);require(helper['rss_bytes']<=CAP,'attributed startup VM RSS over cap')
    def start_owned(self):
        # Time/claim/GO are rechecked immediately before any owned app launch.
        final_admission.begin_native_start(self.final_permit)
        self.startup_active=True
        now=datetime.datetime.now(datetime.UTC)
        require(now<=datetime.datetime.fromisoformat(self.a.get('start_by_utc',self.a['slot_end_utc'])),'hard startup start-by expired')
        slot_remaining=(datetime.datetime.fromisoformat(self.a['slot_end_utc'])-now).total_seconds()-60
        valid_remaining=(datetime.datetime.fromisoformat(self.a.get('valid_until_utc',self.a['slot_end_utc']))-now).total_seconds()
        self.startup_deadline=min(self.now()+120,self.deadline,self.now()+slot_remaining,self.now()+valid_remaining)
        expected=DOCKER_BUNDLE/'Contents/MacOS'/self.desktop_layout['bundles'][0]['entrypoint']
        if self.p.get('launch_authority')==SESSION_KIND:
            require(expected==DOCKER_BUNDLE/'Contents/MacOS/com.docker.backend','only signed OUTER backend launch is admitted')
            self.broker=OwnedSessionLaunch(DOCKER_BUNDLE,expected,{p['pid'] for p in self.before},self.startup_event,log_path=self.directory/'backend-launch.log')
        else:self.broker=CausalLaunch(DOCKER_BUNDLE,expected,{p['pid'] for p in self.before},self.startup_event)  # Historical adapter fixtures only; preflight refuses this mode.
        try:
            self.startup_guard();self.broker.begin()
            while not self.broker.done:
                self.startup_guard();self.broker.pump(min(0.5,max(0,self.startup_deadline-self.now())))
            require(not self.broker.error,self.broker.error or 'launch failed')
            rows,live=self.observe_owned();owner=self.effective_owner(live)
            with (self.directory/'startup-owner.json').open('x') as out:out.write(json.dumps(owner,sort_keys=True))
            return self.owner
        except BaseException:
            self.startup_uncertain=True
            raise
    def query_engine_info(self,timeout=None):
        # Dispatch itself can time out or be cancelled. Pending/unknown history
        # must precede it, and response classification must precede journal I/O.
        # A later known-unavailable reply NEVER clears earlier uncertainty.
        if self.engine_query_inflight:self.engine_history_unknown=True
        try:
            self.engine_query_inflight=True
            kwargs={'check':False}
            if timeout is not None:kwargs['timeout']=timeout
            code,out,err=self.command(self.d+['info','--format','{{json .}}'],**kwargs)
            if code==0:
                self.daemon_observed_available=True
                info=json.loads(out)
                require(isinstance(info,dict) and isinstance(info.get('MemTotal'),int) and info['MemTotal']>0 and isinstance(info.get('NCPU'),int) and info['NCPU']>0,'malformed engine info object')
            elif 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock.' not in err:
                self.engine_history_unknown=True
            return code,out,err
        except BaseException:
            self.engine_history_unknown=True
            raise
        finally:
            self.engine_query_inflight=False
    def info(self):
        # FIRST Docker API stays info. Only the actual transient daemon-not-ready
        # response is retried, within ONE shared 120 s startup/work/slot envelope.
        while True:
            self.startup_guard();remaining=self.startup_deadline-self.now()
            code,out,err=self.query_engine_info(timeout=min(10,remaining))
            caps=json.loads(out) if code==0 else {}
            self.startup_event({'event':'first-api-readiness-info','exit':code,'stderr':err[:300],
                                'observed_caps':{'MemTotal':caps.get('MemTotal'),'NCPU':caps.get('NCPU')} if code==0 else None})
            if code==0:
                info=json.loads(out);verify_caps(info)  # Abort over-cap BEFORE other APIs/work.
                self.startup_guard()
                with (self.directory/'readiness.json').open('x') as output:output.write(json.dumps({'MemTotal':info.get('MemTotal'),'NCPU':info.get('NCPU'),'phase':'post-start-readiness-probe'},sort_keys=True))
                self.startup_active=False
                return info
            require('Cannot connect to the Docker daemon at unix:///var/run/docker.sock.' in err,'non-transient info failure; no readiness retry')
            self.startup_guard();self.broker.pump(min(0.5,max(0,self.startup_deadline-self.now())))
    def cleanup_pending_launch(self):
        if self.broker is None:return
        while self.broker.dispatched and not self.broker.done and self.now()<self.cleanup_until:
            self.broker.pump(min(0.5,self.cleanup_until-self.now()))
        require(not self.broker.dispatched or self.broker.ticket is not None,'launch outcome UNRESOLVED after bounded capture; do not guess a PID')
        if self.broker.ticket:
            self.observe_owned()  # Capture a late authoritative callback before teardown.
    def quit_broker_owned(self,owner):
        require(self.cleanup_until is not None and self.now()<self.cleanup_until,'owned cleanup budget exhausted before termination')
        require(not self.foreign_asset_activity,'foreign asset activity; no Desktop quit')
        rows,live=self.observe_owned()
        # Every RESPONDING engine must pass the foreign-work fence, including
        # startup/over-cap aborts: it may have auto-restarted someone else's work.
        # Only a positively recognized never-ready engine has a no-work exception.
        code,out,err=self.query_engine_info()
        self.cleanup_engine_evidence={'info_exit':code,'info_stderr':err[:300],'state':'unknown'}
        if code==0:
            info=json.loads(out)
            require(isinstance(info,dict) and isinstance(info.get('MemTotal'),int) and info['MemTotal']>0 and isinstance(info.get('NCPU'),int) and info['NCPU']>0,'unknown/malformed cleanup engine response')
            self.daemon_observed_available=True;self.cleanup_engine_evidence['state']='available'
            ps_code,ps_out,ps_err=self.command(self.d+['ps','-q'],check=False)
            self.cleanup_engine_evidence.update(ps_exit=ps_code,foreign_container_ids=ps_out.split(),ps_stderr=ps_err[:300])
            require(ps_code==0 and not ps_out.strip(),'foreign workloads/failed foreign-work probe; no automatic quit')
        else:
            unavailable='Cannot connect to the Docker daemon at unix:///var/run/docker.sock.' in err
            require(unavailable and not self.daemon_observed_available and not self.engine_history_unknown and not self.pull_attempted and self.resolver is None,'unknown/previously available engine or admitted work; no daemon-unavailable quit')
            self.cleanup_engine_evidence['state']='recognized-never-ready-unavailable'
        graceful_error=None
        try:self.broker.terminate()
        except BaseException as error:
            graceful_error=type(error).__name__+': '+str(error)
            self.ownership_journal_errors.append({'event':'causal-terminate-refused','error':graceful_error})
        for known in reversed(live):
            require(self.now()<self.cleanup_until,'owned cleanup budget exhausted before next signal')
            current=next((p for p in self.snapshot() if p['pid']==known['pid']),None)
            if current is not None:
                require((current['start'],current['exe'])==(known['start'],known['exe']),'owned cleanup PID changed')
                if getattr(self.broker,'kind',None)==SESSION_KIND:
                    require(self.broker.matches(self.broker.ticket),'session creator reservation lost before cleanup')
                    if current.get('state','').startswith('Z'):continue
                    if current.get('sid')!=self.broker.ticket['sid']:
                        _,proven=self.observe_owned()
                        require(any(p['pid']==current['pid'] and p['start']==current['start'] for p in proven),'session escape lost observed ancestry before signal')
                os.kill(known['pid'],signal.SIGTERM)
        while self.now()<self.cleanup_until:
            rows,live=self.observe_owned()
            if not live:
                if getattr(self.broker,'kind',None)==SESSION_KIND:
                    self.broker.release_creator(min(5,self.cleanup_until-self.now()))
                    # Reservation now ends. Only read absence; never infer/adopt
                    # a reused SID or signal another process after this point.
                    while True:
                        present=[]
                        for pid in self.owned_processes:
                            try:os.kill(pid,0);present.append(pid)
                            except ProcessLookupError:continue
                        if not present:break
                        require(self.now()<self.cleanup_until,'owned PIDs/zombies remain after creator reap; no late adoption')
                        self.broker.pump(min(0.5,self.cleanup_until-self.now()))
                    final=self.snapshot()
                    require(not self.desktop_records(final),'unattributed Desktop survived own-session shutdown; no late kill')
                    for process in final:
                        if Path(process['exe']).name in HELPERS:
                            code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(process['pid']),'-Fn'],check=False,timeout=3)
                            require(code==0 and ('n'+str(self.raw)) not in out.splitlines(),'VM/raw shutdown association unverified; never kill unknown helper')
                require(graceful_error is None,'graceful termination failed; exact-PID recovery verified but original cleanup remains unverified')
                return
            self.broker.pump(min(0.5,self.cleanup_until-self.now()))
        raise RuntimeError('proven owned processes remain; cleanup budget exhausted')
    def vm_helper(self,owner):
        if getattr(self.broker,'kind',None)==SESSION_KIND:
            rows,live=self.observe_owned();candidates=[]
            for process in live:
                if Path(process['exe']).name not in HELPERS:continue
                code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(process['pid']),'-Fn'],check=False,timeout=3)
                if code==0 and ('n'+str(self.raw)) in out.splitlines():candidates.append(process)
            require(len(candidates)==1,'created-session VM/raw ownership unidentifiable or ambiguous')
            helper=dict(candidates[0],rss_bytes=candidates[0]['rss_kib']*1024,measurement='one-created-session-helper-RSS-not-footprint')
            self.helpers[(helper['pid'],helper['start'],helper['exe'])]=helper
            self.monitor_enabled=True;self.next_rss=self.now()+10
            return helper
        rows=self.snapshot();holders=set()
        if self.broker is not None:
            rows,live=self.observe_owned();owner=self.effective_owner(live)
        for p in rows:
            if Path(p['exe']).name in HELPERS:
                code,out,_=self.command(['/usr/sbin/lsof','-a','-p',str(p['pid']),'-Fn'],check=False)
                if code==0 and ('n'+str(self.raw)) in out.splitlines():holders.add(p['pid'])
        require(any(p['pid']==owner['pid'] and (p['start'],p['exe'])==(owner['start'],owner['exe']) for p in rows),'Desktop owner identity no longer live')
        helper=choose_helper(self.before,rows,owner,holders)
        if self.broker is not None:require(helper['pid'] in self.owned_processes,'VM helper lacks causal owned lineage')
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
        env=dict(anonymous_cli_environment(self.directory),BENCH_REGISTERED_ROOT=str(registered),BENCH_ADMISSION_PATH=str(self.admission_path),BENCH_LIFECYCLE_DIR=str(self.directory),PYTHONPATH=str(HERE),BENCH_SOURCE_GENERATION=str(self.a['_source_generation']),BENCH_SOURCE_GENERATION_SHA256=self.a['_source_generation_sha256'])
        permit_fd=final_admission.issue_resolver_handoff(self.final_permit)
        env['BENCH_FINAL_PERMIT_FD']=str(permit_fd)  # locator only; capability bytes stay in one-use pipe
        blocked={signal.SIGTERM,signal.SIGINT};mask=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
        try:
            self.resolver=subprocess.Popen([sys.executable,'-B',str(path)],cwd=HERE,env=env,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(permit_fd,),
                                           preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_UNBLOCK,blocked))
        finally:
            os.close(permit_fd)
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    def resolver_done(self):
        code=self.resolver.poll()
        if code is None:return False
        require(code==0,'resolver failed');return True
    def begin_cleanup(self):
        self.startup_active=False
        signal.alarm(0)  # Work alarm must not re-enter termination during cleanup.
        self.cleanup_until=self.now()+60
    def recover_owned_container(self):
        # Recovery is attempted even after monitor/client failures, but it may
        # never exceed the independent cleanup budget or touch an unknown CID.
        if self.resolver is None:return
        cid=self.root/'resolver'/self.a['attempt_id']/'container.cid'
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
        if getattr(self,'broker',None) is not None:return self.quit_broker_owned(owner)
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
        value.update(asset_ownership=self.asset_receipt,final_swap_error=final_swap_error,desktop_layout=getattr(self,'desktop_layout',None),
                     causal_launch={'authority':getattr(self.broker,'kind','historical-nsworkspace'),'dispatched':self.broker.dispatched,'ticket':self.broker.ticket,'error':self.broker.error,'done':self.broker.done,'observed_edges':self.ownership_edges} if self.broker else None)
        value['owned_process_identities']=list(self.owned_processes.values())
        value.update(cleanup_engine_evidence=self.cleanup_engine_evidence,ownership_journal_errors=self.ownership_journal_errors,identity_proof_errors=self.identity_proof_errors)
        if self.identity_proof_errors:value['status']='identity-proof-unverified'
        if self.ownership_journal_errors:value['status']='cleanup-unverified'
        fs=os.statvfs(self.root)
        if self.startup_uncertain:value['status']='cleanup-unverified-startup-identity'
        # Standalone source identity is the validated overlay, not a Git label.
        source=getattr(self,'source_generation',None)
        value.update(operator_ack_quote=ACK_QUOTE,source_head='generation:'+source['overlay_sha256'] if source else 'UNVERIFIED-source-generation',source_generation=source,review_source_commit=self.a['full_scope_review'].get('source_commit'),samples=self.samples,after={'allocated_bytes':self.raw.stat().st_blocks*512 if self.raw.exists() else 0,'free_bytes':fs.f_bavail*fs.f_frsize})
        breached=value['after']['allocated_bytes']-self.baseline>5*1024**3 or value['after']['free_bytes']<25*1024**3
        if breached:value['status']='disk-fence-failed-after-cleanup'
        with (self.directory/'receipt.json').open('x') as out:out.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
        require(not breached,'final disk fence crossed; retained failed receipt')
        require(final_swap_error is None,'final swap gate failed; retained failed receipt')
        require(not self.ownership_journal_errors,'ownership journal/termination error; teardown attempted, receipt remains unverified')
        require(not self.identity_proof_errors,'executable qualification failed; own-session cleanup attempted without claiming qualification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--admission',type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--reviewed-source-head')
    parser.add_argument('--verify-source',action='store_true')
    parser.add_argument('--source-generation',type=Path)
    parser.add_argument('--source-generation-sha256')
    args=parser.parse_args()
    if args.verify_source or args.execute:
        import runpy
        gate=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))
        verified=gate['validate_selection'](args.source_generation,args.source_generation_sha256)
        require(not (args.verify_source and args.execute),'source-only verification cannot execute')
        if args.verify_source:
            print(json.dumps(verified,sort_keys=True));return
    if not args.execute:
        print('PROPOSAL ONLY: no fresh slot/full-scope approval; no execution');return
    require(args.admission is not None,'explicit desk admission path required')
    root=validated_registered_root(str(Path.home()/'benchmark-runs/reconstruction-20261114-source-v1'))
    path=validated_admission_path(str(args.admission),root,args.admission.stem)
    # Exclusive persistent consumption occurs before NativeOps construction.
    # No error path deletes/releases this last-attempt claim.
    admission,final_permit=final_admission.claim_final_attempt(path,args.admission_sha256,args.source_generation,args.source_generation_sha256,args.reviewed_source_head)
    try:
        final_admission.begin_controller(final_permit)  # consume before constructing the effect adapter
        admission['_source_generation']=str(args.source_generation);admission['_source_generation_sha256']=args.source_generation_sha256
        admission['_admission_path']=str(path);pins=json.loads((HERE/'resolver-pins.json').read_text())
        def terminate(signum,frame):raise KeyboardInterrupt('termination requested')
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        for s in previous:signal.signal(s,terminate)
        signal.alarm(840)  # Work deadline also interrupts blocked host-side queries.
        try:run_lifecycle(NativeOps(admission,pins,final_permit),admission,pins)
        finally:
            signal.alarm(0)
            for s,handler in previous.items():signal.signal(s,handler)
    finally:
        final_admission.retire_permit(final_permit)  # NEVER remove/release the persistent claim.


if __name__=='__main__':main()

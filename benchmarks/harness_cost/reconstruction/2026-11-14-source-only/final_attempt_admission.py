"""ONE final resolver-only capability, not paid/build/account authorization.

Operator decisions are source-bound facts supplied by the parent, not runtime
'approved' booleans. A review/desk-supplied admission digest is an external trust
anchor, like the source-overlay digest. The permanent exclusive claim consumes
03 even on exception/restart; nothing here deletes/releases/retries it. Only an
active issuer can hand one pipe capability to its direct resolver child.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import runpy
import secrets
import stat

HERE = Path(__file__).resolve().parent
ATTEMPT = 'linux-amd64-03'
CARRIER = 'python@sha256:9bbb8720ae0a24a6ca8dd678bfdf57818fe70caf54c52a53ec01d8db43405056'
SKIP_ACK_QUOTE = "Find a way around this that doesn't require my manual involvement, validate in other ways or skip that validation with acknowledgement but make sure the code analysis and review is thorough"
DECISIONS = {
    'a-d2d2': {'answer':'Authorize one final bounded native attempt'},
    'a-9509': {'answer':'Accept the 8.2 GB / 14 CPU setting',
               'origin':'00e6a1cd8f62','relay':'a-1ca7'},
}
LIMITS = {'configured_memtotal_max_bytes':8_600_000_000,'configured_ncpu_max':14,
          'vm_rss_max_bytes':4_300_000_000,'container_cpus':1,
          'container_memory_bytes':2*1024**3,'raw_growth_max_bytes':5*1024**3,
          'minimum_free_bytes':25*1024**3,'minimum_start_free_bytes':30*1024**3,
          'swap_max_bytes':12_000_000_000,'fleet_max_mb':26000,
          'readiness_seconds':120,'work_seconds':840,'outer_seconds':900,
          'cleanup_seconds':60,'slot_seconds':75*60}
_ACTIVE = {}


def require(value, message):
    if not value:
        raise ValueError('final03 admission: '+message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.datetime.now(datetime.UTC)


def timestamp(value):
    require(type(value) is str,'UTC timestamp required')
    result=datetime.datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset()==datetime.timedelta(0),'UTC timezone required')
    return result


def registered_root():
    root=Path(pwd.getpwuid(os.getuid()).pw_dir)/'benchmark-runs/reconstruction-20261114-source-v1'
    require(root.is_dir() and root.resolve(strict=True)==root and root.stat().st_uid==os.getuid()
            and not root.stat().st_mode&0o022,'protected registered root missing/unsafe')
    return root


def protected_bytes(path):
    require(path.is_absolute() and path.resolve(strict=True)==path,'escaping/symlink file')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and
                stat.S_IMODE(info.st_mode)==0o600,'private owned0600 file required')
        with os.fdopen(fd,'rb',closefd=False) as source:
            raw=source.read(65537)
        require(len(raw)<=65536,'admission/claim too large')
        return raw
    finally:
        os.close(fd)


def checked_go(path, trusted_sha, source, reviewed_head, *, now=None, starting=True):
    """Re-read exact external desk bytes; a forged verified flag has no effect."""
    now=utc() if now is None else now
    root=registered_root();path=Path(path)
    require(path==root/'admissions'/(ATTEMPT+'.json'),'wrong final attempt path')
    require(type(trusted_sha) is str and re.fullmatch('[0-9a-f]{64}',trusted_sha),'review/desk-supplied admission SHA256 required')
    raw=protected_bytes(path);require(digest(raw)==trusted_sha,'desk admission digest mismatch')
    gate=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))
    data=gate['load'](raw)
    require(data.get('attempt_id')==ATTEMPT and data.get('scope')=='resolver-only' and data.get('go') is True,'only final03 resolver GO permitted')
    require(data.get('operator_decisions')==DECISIONS,'exact operator decision bindings missing/wrong')
    require(data.get('operator_ack_id')=='a-7072' and data.get('operator_ack_quote')==SKIP_ACK_QUOTE and data.get('prestart_cap_validation')=='skipped-with-operator-ack' and 'vm_caps_verified_before_start' not in data,'original truthful prestart-skip acknowledgement missing')
    require(data.get('exclusive_desktop_lease') is True and data.get('direct_backend_ui_effects')=='acknowledged-unqualified-default-backend','exclusive Desktop/UI effects acknowledgement missing')
    raw_path=data.get('docker_raw_path')
    require(type(raw_path) is str and Path(raw_path).is_absolute() and Path(raw_path).name=='Docker.raw','explicit approved raw allocation path required')
    require(data.get('limits')==LIMITS and all(type(data['limits'][k]) is int for k in LIMITS),'resource policy may not be weakened/coerced')
    require(data.get('carrier')==CARRIER and data.get('platform')=='linux/amd64','only exact reviewed carrier/platform')
    require(data.get('runtime_qualified',False) is False,'no runtime/paid qualification grant')
    review=data.get('full_scope_review',{});desk=data.get('desk_go',{})
    require(review.get('kind')=='full-scope' and review.get('clean') is True and
            type(review.get('reviewer')) is str and bool(review['reviewer']) and
            review['reviewer']!='70ba7562e541','independent full-scope review required')
    for name in ('lifecycle_proposal.py','RESOLVER-COMMAND.md','resolver-pins.json','launch_services_broker.py','owned_session_launch.py','darwin_process_identity.py'):
        require(review.get('file_sha256',{}).get(name)==digest((HERE/name).read_bytes()),'reviewed native source hash mismatch')
    head=desk.get('source_commit')
    require(type(reviewed_head) is str and re.fullmatch('[0-9a-f]{40}',reviewed_head) and head==reviewed_head and review.get('source_commit')==head,'review/GO head mismatch')
    require(desk.get('source_generation_sha256')==source['overlay_sha256'] and
            review.get('source_generation_sha256')==source['overlay_sha256'],'review/GO source overlay mismatch')
    require(type(desk.get('receipt_id')) is str and re.fullmatch('[a-z0-9-]{8,80}',desk['receipt_id']),'explicit fresh desk receipt ID required')
    issued=timestamp(desk.get('issued_utc'));start_by=timestamp(desk.get('start_by_utc'));expires=timestamp(desk.get('expires_utc'))
    begin=timestamp(data.get('slot_start_utc'));end=timestamp(data.get('slot_end_utc'))
    require(issued<=now<=expires and now-issued<=datetime.timedelta(seconds=LIMITS['slot_seconds']) and issued<=start_by<=min(expires,issued+datetime.timedelta(seconds=LIMITS['slot_seconds'])) and begin<=now and
            end==expires and data.get('valid_until_utc')==desk.get('expires_utc'),'expired/incoherent desk GO')
    if starting:
        require(now<=start_by and now+datetime.timedelta(seconds=960)<=end and
                end<=now+datetime.timedelta(seconds=LIMITS['slot_seconds']),'start-by/75min lease violated')
    require(type(data.get('fleet_mb')) is int and 0<=data['fleet_mb']<=26000 and data.get('fresh_memory_go') is True,'fresh fleet/memory GO missing')
    require(type(data.get('authorize_digest_pull',False)) is bool,'invalid pull permission')
    if data.get('authorize_digest_pull'):
        require(review.get('allow_digest_pull') is True,'exact pull requires explicit fresh review signoff')
    return data,root,raw


class _Permit:
    pass


def claim_final_attempt(admission_path, admission_sha, manifest, source_sha, reviewed_head):
    source=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))['validate_selection'](manifest,source_sha)
    data,root,raw=checked_go(admission_path,admission_sha,source,reviewed_head)
    directory=root/'claims';directory.mkdir(mode=0o700,exist_ok=True)
    require(directory.resolve(strict=True)==directory and directory.stat().st_uid==os.getuid()
            and stat.S_IMODE(directory.stat().st_mode)==0o700,'private claim directory required')
    permit=_Permit();permit.path=directory/(ATTEMPT+'.claim.json');permit.admission_path=Path(admission_path)
    permit.admission_sha=admission_sha;permit.manifest=Path(manifest);permit.source=source;permit.reviewed_head=reviewed_head
    permit.nonce=secrets.token_hex(32);permit.pid=os.getpid();permit.handoff_issued=False
    permit.controller_entered=False;permit.native_started=False
    permit.claim={'attempt_id':ATTEMPT,'controller_pid':permit.pid,'nonce':permit.nonce,
                  'admission_sha256':admission_sha,'source_generation_sha256':source_sha,
                  'source_commit':data['desk_go']['source_commit'],'desk_receipt_id':data['desk_go']['receipt_id'],
                  'claimed_utc':utc().isoformat(),'slot_end_utc':data['slot_end_utc'],
                  'operator_decisions':DECISIONS,'consumed_even_on_error_or_restart':True}
    permit.raw=(json.dumps(permit.claim,sort_keys=True)+'\n').encode()
    # Fixed path, O_EXCL and NOFOLLOW: even a partial/failed claim is terminal.
    fd=os.open(permit.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False) as output:
            output.write(permit.raw);output.flush();os.fsync(fd)
        parent=os.open(directory,os.O_RDONLY)
        try:os.fsync(parent)
        finally:os.close(parent)
    finally:os.close(fd)
    _ACTIVE[permit.nonce]=permit
    return data,permit


def validate_live_permit(permit, *, starting=False):
    require(type(permit) is _Permit and _ACTIVE.get(getattr(permit,'nonce',None)) is permit
            and permit.pid==os.getpid(),'active issuer permit required; receipt booleans cannot authorize')
    require(protected_bytes(permit.path)==permit.raw,'persistent claim changed')
    current=runpy.run_path(str(HERE.parents[1]/'source_generation.py'))['validate_selection'](permit.manifest,permit.source['overlay_sha256'])
    data,root,raw=checked_go(permit.admission_path,permit.admission_sha,current,permit.reviewed_head,starting=starting)
    require(timestamp(permit.claim['slot_end_utc'])<=timestamp(permit.claim['claimed_utc'])+
            datetime.timedelta(seconds=LIMITS['slot_seconds']),'claimed lease exceeds75min')
    return data


def retire_permit(permit):
    # End authority, not consumption: the exclusive filesystem claim stays.
    if type(permit) is _Permit and _ACTIVE.get(permit.nonce) is permit:
        del _ACTIVE[permit.nonce]


def begin_controller(permit):
    validate_live_permit(permit,starting=True)
    require(not permit.controller_entered,'controller capability already consumed')
    permit.controller_entered=True


def validate_controller(permit, *, starting=False):
    data=validate_live_permit(permit,starting=starting)
    require(permit.controller_entered,'controller capability not consumed')
    return data


def begin_native_start(permit):
    validate_controller(permit,starting=True)
    require(not permit.native_started,'native start capability already consumed')
    permit.native_started=True  # Exception after this point NEVER restores it.


def issue_resolver_handoff(permit):
    validate_controller(permit)
    require(permit.native_started,'resolver handoff before owned native start')
    require(not permit.handoff_issued,'resolver handoff already consumed')
    permit.handoff_issued=True
    readfd,writefd=os.pipe()
    try:
        packet={'nonce':permit.nonce,'admission_sha256':permit.admission_sha,
                'source_sha256':permit.source['overlay_sha256'],'source_commit':permit.reviewed_head}
        raw=json.dumps(packet,sort_keys=True).encode()
        # One prefilled frame <=POSIX minimum PIPE_BUF cannot wait for a reader.
        # Paths/full claim are re-read from protected fixed storage in the child.
        require(len(raw)<=min(512,os.fpathconf(writefd,'PC_PIPE_BUF')),'handoff frame exceeds proven atomic bound')
        require(os.write(writefd,raw)==len(raw),'partial resolver handoff')
    except BaseException:
        os.close(readfd);raise
    finally:os.close(writefd)
    return readfd


def consume_resolver_handoff(fd, manifest, source_sha, admission_path):
    require(type(fd) is int and fd>=3,'one-use inherited pipe required')
    try:
        require(stat.S_ISFIFO(os.fstat(fd).st_mode),'handoff must be a pipe, not a forged file/flag')
        # Producer completed and closed the write end BEFORE Popen. No polling
        # or unbounded wait is legitimate here; an empty/partial fake FIFO fails.
        os.set_blocking(fd,False)
        raw=os.read(fd,513)
        require(0<len(raw)<=512,'empty/oversized handoff frame')
    finally:os.close(fd)
    gate=runpy.run_path(str(HERE.parents[1]/'source_generation.py'));packet=gate['load'](raw)
    require(type(packet) is dict and set(packet)=={'nonce','admission_sha256','source_sha256','source_commit'} and type(packet['nonce']) is str and re.fullmatch('[0-9a-f]{64}',packet['nonce']),'malformed one-use handoff frame')
    source=gate['validate_selection'](manifest,source_sha)
    require(packet.get('source_sha256')==source_sha,'handoff source mismatch')
    root=registered_root();claim=gate['load'](protected_bytes(root/'claims'/(ATTEMPT+'.claim.json')))
    require(claim.get('attempt_id')==ATTEMPT and claim.get('controller_pid')==os.getppid(),'handoff not from active claimed controller')
    require(claim.get('nonce')==packet.get('nonce') and claim.get('source_generation_sha256')==source_sha and claim.get('operator_decisions')==DECISIONS,'persistent claim/handoff mismatch')
    data,_,_=checked_go(admission_path,packet.get('admission_sha256'),source,packet.get('source_commit'),starting=False)
    require(claim.get('admission_sha256')==packet['admission_sha256'] and claim.get('source_commit')==data['desk_go']['source_commit'],'handoff GO binding changed')
    return data
